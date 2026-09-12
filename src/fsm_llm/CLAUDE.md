# fsm_llm -- Core Framework

FSM-LLM core package. 2-pass architecture: Pass 1 extracts data + evaluates transitions, Pass 2 generates response from the final state.

- **Version**: 0.5.0
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
├── utilities.py            # extract_json_from_text(), load_fsm_definition(), load_fsm_from_file()
├── constants.py            # DEFAULT_LLM_MODEL, security patterns, INTERNAL_KEY_PREFIXES, ALLOWED_JSONLOGIC_OPERATIONS
├── session.py              # SessionStore ABC + FileSessionStore -- file-based session persistence with atomic writes
├── logging.py              # Loguru setup, enable_debug_logging(), disable_warnings()
├── __main__.py             # CLI entry point (run, validate, visualize modes)
├── __version__.py          # "0.5.0"
└── __init__.py             # 90+ exports in single __all__ list
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
- **MessagePipeline** (`pipeline.py`) -- 2-pass engine
  - Pass 1: data extraction → field extractions → classification extractions → transition evaluation → state transition
  - Pass 2: response generation from new state -- skipped entirely when the state's `response_instructions` is empty (no response LLM call; used for intermediate agent states in tool-use loops)
  - `process_message(instance, conv_id, msg)`, `generate_initial_response(instance, conv_id)`
- **HandlerSystem** (`handlers.py`) -- Event-driven hook execution
  - `register_handler(handler)`, `execute_handlers(timing, current_state, target_state, context, updated_keys)` → dict
  - Error modes: "continue" (skip failed) | "raise"
- **HandlerBuilder** (`handlers.py`) -- Fluent API: `.at(timing)` → `.on_state(id)` → `.when(lambda)`/`.when_context_has()`/`.when_keys_updated()` (+ shorthands `.on_state_entry()`, `.on_state_exit()`, `.on_context_update()`, `.with_priority()`) → `.do(lambda)` → `BaseHandler`
- **HandlerTiming** enum -- 8 points: START_CONVERSATION, PRE_PROCESSING, POST_PROCESSING, PRE_TRANSITION, POST_TRANSITION, CONTEXT_UPDATE, END_CONVERSATION, ERROR
- **Classifier** (`classification.py`) -- `classify(msg)` → ClassificationResult, `classify_multi(msg)` → MultiClassificationResult
- **HierarchicalClassifier** -- Two-stage domain → intent for >15 intents
- **IntentRouter** -- `route(msg)` → dispatches to handler functions by intent
- **TransitionEvaluator** (`transition_evaluator.py`) -- Returns DETERMINISTIC | AMBIGUOUS | BLOCKED with confidence scores
- **LiteLLMInterface** (`llm.py`) -- `generate_response(request)`, `extract_field(request)`, `generate_response_stream(request)` → `Iterator[str]` via litellm (100+ providers). Supports `response_format` for schema-enforced JSON output
- **WorkingMemory** (`memory.py`) -- `get/set/delete(buffer, key)`, `get_all_data()`, `search(query)`, `get_buffer()`, `clear_buffer()`, `list_buffers()`, `has_buffer()`, `create_buffer()`, `to_scoped_view()`, `update_buffer()`, `import_flat_data()`, `to_dict()`, `from_dict()`
- **SessionStore** (`session.py`) -- ABC for session persistence: `save(id, state)`, `load(id)`, `delete(id) -> bool`, `list_sessions()`, `exists(id)`
- **FileSessionStore** (`session.py`) -- File-based implementation with JSON files and atomic writes (temp file + rename). Path-traversal protection via session ID validation
- **SessionState** (`session.py`) -- Pydantic model: conversation_id, fsm_id, current_state, context_data, conversation_history, stack_depth, saved_at, metadata
- **ContextCompactor** (`context.py`) -- `compact(ctx)` (clear transient), `prune(ctx)` (on transition), `summarize(conversation)`

## Core Models (definitions.py)

- **FSMDefinition**: name, description, states dict, initial_state, version="4.1", persona. Validates reachability + terminal states
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
pytest tests/test_fsm_llm/  # 643 tests
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
- Security: Internal key prefixes stripped by clean_context_keys(). XML tag sanitization in prompts
- Thread safety: Per-conversation RLocks in FSMManager
