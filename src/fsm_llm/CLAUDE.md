# fsm_llm -- Core Framework

FSM-LLM core package. 2-pass architecture: Pass 1 extracts data + evaluates transitions, Pass 2 generates response from the final state.

- **Version**: 0.6.0
- **Python**: 3.10, 3.11, 3.12
- **Deps**: loguru, litellm (>=1.82,<2.0, !=1.82.7, !=1.82.8), pydantic (>=2.0), python-dotenv

## File Map

```
fsm_llm/
├── api.py                  # API class -- primary entry point (from_file, from_definition, converse, push/pop_fsm)
├── fsm.py                  # FSMManager -- orchestration with per-conversation RLocks, LRU FSM cache
├── pipeline.py             # MessagePipeline -- 2-pass processing (extraction → transition → response)
├── classification.py       # Classifier, HierarchicalClassifier, IntentRouter, HandlerFn type alias
├── definitions.py          # Pydantic models + exception hierarchy (State, Transition, FSMDefinition, FSMContext, FSMInstance, Conversation, all classification/extraction models)
├── handlers.py             # HandlerSystem, HandlerBuilder, BaseHandler, LambdaHandler, HandlerTiming enum (8 points)
├── prompts.py              # Prompt builders: DataExtraction, ResponseGeneration, FieldExtraction, Classification
├── llm.py                  # LLMInterface ABC + LiteLLMInterface (generate_response, extract_field)
├── ollama.py               # Ollama-specific helpers (thinking disable, json_schema format)
├── transition_evaluator.py # TransitionEvaluator + TransitionEvaluatorConfig -- rule-based with confidence scoring
├── expressions.py          # evaluate_logic() -- JsonLogic evaluator (var, and, or, ==, in, has_context, context_length)
├── context.py              # clean_context_keys() + ContextCompactor (transient key clearing, pruning, summarization)
├── memory.py               # WorkingMemory -- 4 named buffers (core, scratch, environment, reasoning)
├── runner.py               # Interactive CLI conversation runner
├── validator.py            # FSMValidator.validate() + validate_fsm_from_file()
├── visualizer.py           # visualize_fsm_ascii() + visualize_fsm_from_file() (full/compact/minimal styles)
├── utilities.py            # extract_json_from_text() (dict | None; non-object JSON -> None), load_fsm_definition(), load_fsm_from_file()
├── constants.py            # DEFAULT_LLM_MODEL, security patterns, INTERNAL_KEY_PREFIXES, ALLOWED_JSONLOGIC_OPERATIONS
├── session.py              # SessionStore ABC + FileSessionStore -- file-based session persistence with atomic writes
├── logging.py              # Loguru setup: setup_logging(), setup_file_logging()
├── __main__.py             # CLI entry point (run, validate, visualize modes)
├── __version__.py          # "0.6.0"
└── __init__.py             # 90+ exports in single __all__ list; enable_debug_logging(), disable_warnings()
```

## Key Classes

- **API** (`api.py`) -- User-facing entry point
  - Factory: `from_file(path, **kwargs)`, `from_definition(fsm_def, **kwargs)`
  - Conversation: `start_conversation(initial_context)` → `(conv_id, greeting)`, `converse(msg, conv_id)` → str, `converse_stream(msg, conv_id)` → `Iterator[str]`, `end_conversation(conv_id)`, `has_conversation_ended(conv_id)`
  - Queries: `get_data(conv_id)`, `get_current_state(conv_id)`, `get_conversation_history(conv_id)`, `list_active_conversations()`
  - FSM stacking: `push_fsm(conv_id, new_fsm)`, `pop_fsm(conv_id, merge_strategy)`, `get_stack_depth(conv_id)`, `get_sub_conversation_id(conv_id)`
  - Handlers: `register_handler(handler)`, `register_handlers(handlers)`, `create_handler(name)` → HandlerBuilder
  - Sessions: `save_session(conv_id)`, `load_session(session_id)` → `SessionState | None`, `restore_session(session_id)` → `(conv_id, SessionState) | None`
  - Management: `update_context(conv_id, data)`, `cleanup_stale_conversations()`, `get_llm_interface()`, `close()`
- **FSMManager** (`fsm.py`) -- Orchestration with per-conversation thread locks, LRU FSM cache (max 64)
  - `start_conversation(fsm_id, initial_context)`, `process_message(conv_id, msg)`, `resolve_state_definition(instance)`
  - ERROR-timing handlers fire on `FSMError` (e.g. `LLMResponseError`) and on the streaming path via `_fire_error_handlers`; the `FSMError` is still re-raised unwrapped. KeyboardInterrupt/SystemExit/GeneratorExit run no handlers
  - Re-entrancy guard (`_active_turns`, `_enter_turn`): a same-conversation `converse`/`converse_stream` from a handler (or between `next()` calls of an open stream) while a turn is in flight raises `FSMError`; `update_context` stays allowed
  - `save_session` on a stacked conversation saves the ROOT frame's state, data, history and working memory (not the sub-FSM's)
- **MessagePipeline** (`pipeline.py`) -- 2-pass engine
  - Pass 1: data extraction → field extractions → classification extractions → transition evaluation → state transition
  - Pass 2: response generation from new state -- skipped entirely when the state's `response_instructions` is empty (no response LLM call; used for intermediate agent states in tool-use loops)
  - `process_message(instance, conv_id, msg)`, `generate_initial_response(instance, conv_id)`
  - Streaming (`process_message_stream`) uses a plain-text Pass-2 prompt (`build_response_prompt(..., plain_text_response=True)`) unless the state carries `_output_response_format`, so yielded tokens and stored history have no `{"message","reasoning"}` envelope
  - `context_scope.read_keys` scopes `<current_context>` and `<rejected_corrections>` in the Pass-2 prompt (turn, stream and greeting), not only `request.context` (D-005, D-032); `<extracted_data>` (keys extracted this turn from the user's own message) is NOT scoped, a named limitation (D-054)
  - Bulk extraction pass provenance: `context.metadata["_pipeline_extracted"]` holds a digest per key the pipeline extracted; the bulk pass overwrites a stored key only if it is config-covered, the FSM is not agent-managed and the stored value still matches the digest (handler-set and `update_context` values are never overwritten, including a value a same-timing CONTEXT_UPDATE handler edited; the digests are persisted in `SessionState.metadata["pipeline_extracted"]` by `save_session` and re-seeded by `restore_session`, so a correction lands after a restart, and an old session file restores an empty map). A correction the rule refuses and the user's message states is carried on `DataExtractionResponse.rejected_corrections` (default `{}`) and shown to Pass 2 as a `<rejected_corrections>` block (`build_response_prompt`'s optional argument), so the reply says the change was not applied; the grounding test is a whole-token match of at least 3 characters (`bored` does not ground `red`, `1500 items` does not ground `500`, and a 1 or 2 character value such as `US` or `42` never produces the block: the named cost, D-049); an instruction-only key (no config) that already holds a different value is reported the same way on non-agent FSMs, never overwritten (D-052); after a back-edge re-extraction an entry whose value is now the stored value is dropped, so an applied correction is never listed as rejected (LV6-01). If the bulk extraction call itself raises, the helper returns a private empty `_BulkFailed` dict, `DataExtractionResponse.extraction_failed` is set (default `False`) and `build_response_prompt`'s last optional argument `extraction_failed` adds one plain line saying a restated value may not have been stored (D-050); a turn with no rejection and no failure builds a byte-identical prompt. Bulk values for config-covered keys are coerced/validated like per-field values. The bulk prompt sanitizes user text; the bulk result drops `agent_trace` and forbidden-name keys; it never fills a classification-owned key on a non-agent FSM
  - `execute_handlers` returns immediately when no handler subscribes to the timing (`HandlerSystem.handlers_at`), skipping the context deep-copies (a zero-handler advance turn: 14 -> 4 copies); the pre-turn rollback snapshots are unaffected. `handlers_at` is an optional fast-path hook read with a guarded `getattr`: a duck-typed `handler_system` with only `execute_handlers` still works
  - `FSMDefinition.handler_only_keys` (opt-in, default `[]`): a listed gate key is dropped from the bulk return, the per-field configs and the post-transition configs, so user text cannot write it; handler writes, `update_context` and `initial_context` still work; a stacked child uses its own list. Only listed keys are covered (an unlisted gate key stays writable, a classification-owned key is not covered). `fsm-llm-validate` emits a WARNING for a listed key no state references (a likely typo) and one for a key that is a `classification_extractions` field name; both are warnings only, silent for an empty list (D-051)
  - Back-edge re-extraction: a transition into a DIFFERENT state whose own config-covered key is already set with provenance re-runs the target state's Pass-1 extraction, so a same-message correction lands; a self-loop, an agent-managed FSM, a handler-seeded key, a forward hop into an empty state and a state that owns `classification_extractions` do not (+1 bulk call on a back edge into a filled state, +1 retry per still-null required key)
  - Skip-if-set filter (`_execute_data_extraction`): a config-covered key that already holds a value is not re-asked; for an agent-managed FSM (context carries `agent_trace`) an EMPTY list or dict counts as unset, so `plan_execute`'s `plan_steps: []` seed is still extracted; a non-agent FSM that seeds `[]` for a config-covered key makes no ask for it (D-046)
  - On Ollama (`ollama/` or `ollama_chat/` prefix, temperature 0) an identical null per-field extraction is memoised within one extraction call (keyed on field name plus the built prompt and message); successes, exceptions and other providers are never memoised
  - A classification result discarded for being below its `confidence_threshold` is a WARNING naming field, intent, confidence and threshold; behaviour is unchanged (the gated key stays unset)
  - An ERROR-timing handler's returned dict is NOT merged (the turn was rolled back); `update_context` is the supported write path
  - A classifier error or fallback intent in `_resolve_ambiguous_transition` returns `None` (a stay, not a transition); the `Classifier` inherits `api_key`/`api_base`/`timeout` from the `LiteLLMInterface`
- **HandlerSystem** (`handlers.py`) -- Event-driven hook execution; `handlers_at(timing)` is the optional subscription probe the pipeline uses to skip empty timings
  - `register_handler(handler)`, `execute_handlers(timing, current_state, target_state, context, updated_keys)` → dict
  - Error modes: "continue" (skip failed) | "raise"
- **HandlerBuilder** (`handlers.py`) -- Fluent API: `.at(timing)` → `.on_state(id)` → `.when(lambda)`/`.when_context_has()`/`.when_keys_updated()` (+ shorthands `.on_state_entry()`, `.on_state_exit()`, `.on_context_update()`, `.with_priority()`) → `.do(lambda)` → `BaseHandler`
- **HandlerTiming** enum -- 8 points: START_CONVERSATION, PRE_PROCESSING, POST_PROCESSING, PRE_TRANSITION, POST_TRANSITION, CONTEXT_UPDATE, END_CONVERSATION, ERROR
- **Classifier** (`classification.py`) -- `classify(msg)` → ClassificationResult, `classify_multi(msg)` → MultiClassificationResult
- **HierarchicalClassifier** -- Two-stage domain → intent for >15 intents
- **IntentRouter** -- `route(msg)` → dispatches to handler functions by intent
- **TransitionEvaluator** (`transition_evaluator.py`) -- Returns DETERMINISTIC | AMBIGUOUS | BLOCKED with confidence scores
- **LiteLLMInterface** (`llm.py`) -- `generate_response(request)`, `extract_field(request)`, `generate_response_stream(request)` → `Iterator[str]` via litellm (100+ providers). Supports `response_format` for schema-enforced JSON output
  - Reply parsing: an uncoercible `confidence` (`"high"`, null, object) keeps the returned value at confidence 0.5; a structured reply with no `message` but a `reasoning` key reaches the user as the JSON text; the plain-text rung strips `<think>` blocks first and replaces brace-shaped text only when it parses as JSON; `strip_think_and_fences` strips a fence only at the start of the reply and the closing fence only when a leading fence was stripped, so a reply that ends in a code block keeps its closing fence (D-048), and `extract_json_from_text` skips a fenced non-object by blanking its span (so JSON before a fenced example is found)
- **WorkingMemory** (`memory.py`) -- `get/set/delete(buffer, key)`, `get_all_data()`, `search(query)`, `get_buffer()`, `clear_buffer()`, `list_buffers()`, `has_buffer()`, `create_buffer()`, `to_scoped_view()`, `update_buffer()`, `import_flat_data()`, `to_dict()`, `from_dict()`
- **SessionStore** (`session.py`) -- ABC for session persistence: `save(id, state)`, `load(id)`, `delete(id) -> bool`, `list_sessions()`, `exists(id)`
- **FileSessionStore** (`session.py`) -- File-based implementation with JSON files and atomic writes (temp file + rename). Path-traversal protection via session ID validation
- **SessionState** (`session.py`) -- Pydantic model: conversation_id, fsm_id, current_state, context_data, conversation_history, stack_depth, saved_at, metadata
- **ContextCompactor** (`context.py`) -- `compact(ctx)` (clear transient), `prune(ctx)` (on transition), `summarize(conversation)`

## Core Models (definitions.py)

- **FSMDefinition**: name, description, states dict, initial_state, version="4.1", persona, handler_only_keys (list, default `[]`; `model_dump` emits it). Validates reachability + terminal states
- **State**: id, description, purpose, extraction_instructions, response_instructions, transitions, required_context_keys, field_extractions, classification_extractions, context_scope
- **Transition**: target_state, description, conditions list, priority (0-1000)
- **TransitionCondition**: description, requires_context_keys, logic (JsonLogic dict), evaluation_priority
- **FSMContext**: data dict, conversation (Conversation), metadata, working_memory
- **FSMInstance**: fsm_id, current_state, context (FSMContext), persona, last_extraction/transition/response debug fields
- **Conversation**: exchanges list, max_history_size, max_message_length, summary. Methods: add_user_message, add_system_message, get_recent, search
- **ClassificationSchema**: intents list (IntentDefinition), fallback_intent, confidence_threshold
- **ClassificationResult**: reasoning, intent, confidence, entities. Property: is_low_confidence
- **FieldExtractionConfig**: field_name, field_type, extraction_instructions, validation_rules, required, confidence_threshold
- **ClassificationExtractionConfig**: field_name, intents list, fallback_intent, confidence_threshold, model override

## JsonLogic Operators (expressions.py)

Comparison: `==`, `!=`, `===`, `!==`, `>`, `>=`, `<`, `<=` | Logical: `and`, `or`, `!` | Arithmetic: `+`, `-`, `*`, `/`, `%` | Functions: `var`, `in`, `contains`, `cat`, `if`, `min`, `max`, `missing`, `missing_some` | Custom: `has_context`, `context_length`

## Constants (constants.py)

- `DEFAULT_LLM_MODEL = "ollama_chat/qwen3.5:4b"`
- `DEFAULT_TEMPERATURE = 0.5`, `DEFAULT_MAX_HISTORY_SIZE = 5`, `DEFAULT_MAX_MESSAGE_LENGTH = 1000`
- `DEFAULT_MAX_STACK_DEPTH = 10`, `FSM_ID_HASH_LENGTH = 8`
- `INTERNAL_KEY_PREFIXES = ["_", "system_", "internal_", "__"]`
- `FORBIDDEN_CONTEXT_PATTERNS`: Regex for passwords, secrets, API keys, tokens
- `DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE = 0.6`

## Testing

```bash
pytest tests/test_fsm_llm/  # 1,925 tests
```

- Mock LLMs: `Mock(spec=LLMInterface)` (simple) and `MockLLM2Interface` (2-pass) in `conftest.py`
- Fixtures: `sample_fsm_definition` (v3.0), `sample_fsm_definition_v2` (v4.1), `mock_llm_interface`, `mock_llm2_interface`
- Test files: `test_<module>.py` + `test_<module>_elaborate.py` for extended scenarios
- Helper functions: `_make_state()`, `_minimal_fsm_dict()` etc.

## Exceptions

```
FSMError (base for all core exceptions)
├── StateNotFoundError(state_id)
├── InvalidTransitionError(source_state, target_state)
├── LLMResponseError
├── TransitionEvaluationError(state_id)
├── ClassificationError
│   ├── SchemaValidationError
│   └── ClassificationResponseError
└── HandlerSystemError
    └── HandlerExecutionError(handler_name, original_error)
```

## Code Conventions

- Logging: `from fsm_llm.logging import logger`
- Models: Pydantic v2 BaseModel with model_validator for complex validation
- Exports: Single `__all__` list in `__init__.py` -- no dynamic extend/append
- Security: Internal key prefixes stripped by clean_context_keys(). XML tag sanitization in prompts (`_TAG_PATTERN` in `prompts.py`: an opener/closer tail is `[^>]{0,256}/?>` or, as a zero-width lookahead, a 257-character overflow, so a closing tag with a nested `<` or a padded tail is escaped, only its `<` and name, without a quadratic scan and without touching the prose after it; `latency < threshold` stays raw only when no `>` follows it anywhere in the text, a padded opener `< name` + 257 characters + `>` is escaped, D-047, D-054)
- Thread safety: Per-conversation RLocks in FSMManager
