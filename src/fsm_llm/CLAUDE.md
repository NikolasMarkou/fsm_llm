# fsm_llm — Core Package

The fsm_llm core package: **typed λ-calculus kernel** (`lam/`) + **standard library** of named λ-term factories (`stdlib/`) + the **FSM dialog surface** (top-level modules). All execution paths share a single Executor.

- **Version**: 0.3.0
- **Python**: 3.10, 3.11, 3.12
- **Deps**: `loguru`, `litellm` (>=1.82,<2.0), `pydantic` (>=2.0), `python-dotenv`

See `docs/lambda.md` for the architectural thesis. Per §11, the kernel + stdlib live here; the legacy `fsm_llm_*` siblings are sys.modules shims.

## File Map

```
fsm_llm/
├── lam/                    # M1 — typed λ-AST + Executor + Planner + Oracle + FSM compiler
│   ├── ast.py              #   Var, Abs, App, Let, Case, Combinator, Fix, Leaf, Term, is_term
│   ├── dsl.py              #   var, abs_, app, let_, case_, fix, leaf, split, peek, fmap, ffilter, reduce_, concat, cross
│   ├── combinators.py      #   ReduceOp enum + BUILTIN_OPS dict (closed registry)
│   ├── executor.py         #   Executor — β-reduction, depth-bounded, per-Leaf cost
│   ├── planner.py          #   plan(), Plan, PlanInputs — closed-form (k*, τ*, d, predicted_calls)
│   ├── oracle.py           #   Oracle ABC, LiteLLMOracle (adapter over LiteLLMInterface)
│   ├── cost.py             #   CostAccumulator, LeafCall — per-leaf cost telemetry
│   ├── fsm_compile.py      #   M2 — compile_fsm() : FSMDefinition → Term
│   ├── errors.py           #   LambdaError → ASTConstructionError, TerminationError, PlanningError, OracleError
│   └── constants.py        #   K_DEFAULT, TAU_DEFAULT, DEPTH_LIMIT, …
│   (See src/fsm_llm/lam/CLAUDE.md for the kernel-detail file map.)
│
├── stdlib/                 # M3 — named λ-term factories
│   ├── agents/             #   slice 1: react_term, rewoo_term, reflexion_term, memory_term + 12 class agents
│   ├── reasoning/          #   slice 2: 11 strategy factories + classifier_term + solve_term + ReasoningEngine
│   ├── workflows/          #   slice 3: linear/branch/switch/parallel/retry term factories + WorkflowEngine
│   └── long_context/       #   M5: niah, aggregate, pairwise, multi_hop, niah_padded + helpers
│   (See src/fsm_llm/stdlib/CLAUDE.md for the stdlib index.)
│
├── api.py                  # API class — primary entry point (from_file, from_definition, converse, push/pop_fsm)
├── fsm.py                  # FSMManager — orchestration with per-conversation RLocks, LRU compiled-term cache
├── pipeline.py             # MessagePipeline — compiled-path 2-pass body. Public process/process_stream RETIRED in M2 S11
├── classification.py       # Classifier, HierarchicalClassifier, IntentRouter, HandlerFn type alias
├── definitions.py          # Pydantic models (State, Transition, FSMDefinition, FSMContext, FSMInstance, Conversation, classification/extraction models)
├── handlers.py             # HandlerSystem, HandlerBuilder, BaseHandler, LambdaHandler, HandlerTiming enum (8 points)
├── prompts.py              # Prompt builders: DataExtraction, ResponseGeneration, FieldExtraction, Classification
├── llm.py                  # LLMInterface ABC + LiteLLMInterface (generate_response, extract_field, generate_response_stream)
├── ollama.py               # Ollama-specific helpers (thinking disable, json_schema format)
├── transition_evaluator.py # TransitionEvaluator + TransitionEvaluatorConfig — DETERMINISTIC | AMBIGUOUS | BLOCKED
├── expressions.py          # evaluate_logic() — JsonLogic evaluator (var, and, or, ==, in, has_context, context_length)
├── context.py              # clean_context_keys() + ContextCompactor (transient-key clearing, pruning, summarisation)
├── memory.py               # WorkingMemory — 4 named buffers (core, scratch, environment, reasoning)
├── runner.py               # Interactive CLI conversation runner
├── validator.py            # FSMValidator.validate() + validate_fsm_from_file()
├── visualizer.py           # visualize_fsm_ascii() + visualize_fsm_from_file() (full/compact/minimal)
├── utilities.py            # extract_json_from_text(), load_fsm_definition(), load_fsm_from_file()
├── constants.py            # DEFAULT_LLM_MODEL, security patterns, INTERNAL_KEY_PREFIXES, ALLOWED_JSONLOGIC_OPERATIONS
├── session.py              # SessionStore ABC + FileSessionStore — atomic JSON writes (temp + rename)
├── logging.py              # Loguru setup, enable_debug_logging(), disable_warnings()
├── __main__.py             # CLI entry point (run, validate, visualize modes)
├── __version__.py          # "0.3.0"
└── __init__.py             # 90+ exports in single __all__ list
```

## Where Execution Actually Happens

There is one runtime: **`fsm_llm.lam.Executor`**. Both surfaces compile to the same AST.

```
       FSM JSON  →  fsm_compile()  →┐
                                    │
                                    ▼
                                  Term  →  Executor.run(env)  →  result
                                    ▲
       λ-DSL    →  dsl builders   →┘
```

`API.converse()` is one β-reduction step on a cached compiled term. `MessagePipeline` is the body of that term for Category-A FSM dialogs; its public `process` / `process_stream` entry points were retired in M2 S11 — there is no "legacy path" to maintain.

## Key Classes

- **`API`** (`api.py`) — User-facing entry point.
  - Factory: `from_file(path, **kwargs)`, `from_definition(fsm_def, **kwargs)`. Both compile the FSM via `lam.fsm_compile.compile_fsm()` and cache the resulting `Term`.
  - Conversation: `start_conversation(initial_context)` → `(conv_id, greeting)`, `converse(msg, conv_id)` → str, `end_conversation(conv_id)`, `has_conversation_ended(conv_id)`
  - Queries: `get_data(conv_id)`, `get_current_state(conv_id)`, `get_conversation_history(conv_id)`, `list_active_conversations()`
  - FSM stacking: `push_fsm(conv_id, new_fsm)`, `pop_fsm(conv_id, merge_strategy)`, `get_stack_depth(conv_id)`
  - Handlers: `register_handler(handler)`, `create_handler(name)` → HandlerBuilder
  - Management: `update_context(conv_id, data)`, `close()`
- **`FSMManager`** (`fsm.py`) — Orchestration with per-conversation thread locks, LRU compiled-term cache (max 64). Thin adapter over `lam.Executor`.
  - `start_conversation(fsm_id, initial_context)`, `process_message(conv_id, msg)`, `get_current_state(instance)`
- **`MessagePipeline`** (`pipeline.py`) — Compiled-path 2-pass body. Internal-only post-M2 S11.
  - Pass 1: data extraction → field extractions → classification extractions → transition evaluation → state transition
  - Pass 2: response generation from new state
- **`HandlerSystem`** (`handlers.py`) — Event-driven hook execution. Hooks compose into the compiled λ-term per `docs/lambda.md` §6.3.
  - `register_handler(handler)`, `execute_handlers(timing, current_state, target_state, context, updated_keys)` → dict
  - Error modes: `"continue"` (skip failed) | `"raise"`
- **`HandlerBuilder`** (`handlers.py`) — Fluent API: `.at(timing)` → `.on_state(id)` → `.when(lambda)` → `.do(lambda)` → FSMHandler
- **`HandlerTiming`** enum — 8 points: `START_CONVERSATION`, `PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `END_CONVERSATION`, `ERROR`
- **`Classifier`** (`classification.py`) — `classify(msg)` → `ClassificationResult`, `classify_multi(msg)` → `MultiClassificationResult`
- **`HierarchicalClassifier`** — Two-stage domain → intent for >15 intents
- **`IntentRouter`** — `route(msg)` → dispatches to handler functions by intent
- **`TransitionEvaluator`** (`transition_evaluator.py`) — Returns `DETERMINISTIC` | `AMBIGUOUS` | `BLOCKED` with confidence scores
- **`LiteLLMInterface`** (`llm.py`) — `generate_response(request)`, `extract_field(request)`, `generate_response_stream(request)` → `Iterator[str]` via litellm (100+ providers). Supports `response_format` for schema-enforced JSON output. **For small Ollama models (qwen3.5:4b), use `generate_response` + Pydantic `response_format` instead of `extract_field`** — see plans/LESSONS.md "LiteLLMInterface Reuse Pattern" caveat.
- **`WorkingMemory`** (`memory.py`) — `get/set/delete(buffer, key)`, `get_all_data()`, `search(query)`, `get_buffer()`, `clear_buffer()`, `list_buffers()`, `has_buffer()`, `create_buffer()`, `to_scoped_view()`, `update_buffer()`, `import_flat_data()`, `to_dict()`, `from_dict()`
- **`SessionStore`** (`session.py`) — ABC for session persistence: `save(id, state)`, `load(id)`, `delete(id)`, `list_sessions()`, `exists(id)`
- **`FileSessionStore`** (`session.py`) — File-based with JSON files + atomic writes (temp + rename). Path-traversal protection via session ID validation.
- **`SessionState`** (`session.py`) — Pydantic model: `conversation_id`, `fsm_id`, `current_state`, `context_data`, `conversation_history`, `stack_depth`, `saved_at`, `metadata`
- **`ContextCompactor`** (`context.py`) — `compact(ctx)` (clear transient), `prune(ctx)` (on transition), `summarize(conversation)`

## Core Models (`definitions.py`)

- **`FSMDefinition`**: `name`, `description`, `states` dict, `initial_state`, `version="4.1"`, `persona`. Validates reachability + terminal states.
- **`State`**: `id`, `description`, `purpose`, `extraction_instructions`, `response_instructions`, `transitions`, `required_context_keys`, `field_extractions`, `classification_extractions`, `context_scope`
- **`Transition`**: `target_state`, `description`, `conditions` list, `priority` (0-1000)
- **`TransitionCondition`**: `description`, `requires_context_keys`, `logic` (JsonLogic dict), `evaluation_priority`
- **`FSMContext`**: `data` dict, `conversation` (Conversation), `metadata`, `working_memory`
- **`FSMInstance`**: `fsm_id`, `current_state`, `context` (FSMContext), `persona`, `last_extraction/transition/response` debug fields
- **`Conversation`**: `exchanges` list, `max_history_size`, `max_message_length`, `summary`. Methods: `add_user_message`, `add_system_message`, `get_recent`, `search`
- **`ClassificationSchema`**: `intents` list (`IntentDefinition`), `fallback_intent`, `confidence_threshold`
- **`ClassificationResult`**: `reasoning`, `intent`, `confidence`, `entities`. Property: `is_low_confidence`
- **`FieldExtractionConfig`**: `field_name`, `field_type`, `extraction_instructions`, `validation_rules`, `required`, `confidence_threshold`
- **`ClassificationExtractionConfig`**: `field_name`, `intents` list, `fallback_intent`, `confidence_threshold`, `model` override

## JsonLogic Operators (`expressions.py`)

Comparison: `==`, `!=`, `===`, `!==`, `>`, `>=`, `<`, `<=` | Logical: `and`, `or`, `!` | Arithmetic: `+`, `-`, `*`, `/`, `%` | Functions: `var`, `in`, `contains`, `cat`, `if`, `min`, `max`, `missing`, `missing_some` | Custom: `has_context`, `context_length`

## Constants (`constants.py`)

- `DEFAULT_LLM_MODEL = "ollama_chat/qwen3.5:4b"`
- `DEFAULT_TEMPERATURE = 0.5`, `DEFAULT_MAX_HISTORY_SIZE = 5`, `DEFAULT_MAX_MESSAGE_LENGTH = 1000`
- `DEFAULT_MAX_STACK_DEPTH = 10`, `FSM_ID_HASH_LENGTH = 8`
- `INTERNAL_KEY_PREFIXES = ["_", "system_", "internal_", "__"]`
- `FORBIDDEN_CONTEXT_PATTERNS`: regex for passwords, secrets, API keys, tokens
- `DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE = 0.6`

## Testing

```bash
pytest tests/test_fsm_llm/         # 688 tests (core)
pytest tests/test_fsm_llm_lam/     # λ-kernel tests (Executor, Planner, DSL, FSM compiler)
```

- **Mock LLMs**: `Mock(spec=LLMInterface)` (simple) and `MockLLM2Interface` (2-pass) in `conftest.py`
- **Fixtures**: `sample_fsm_definition_v2` (v4.1), `mock_llm_interface`, `mock_llm2_interface`
- **Test files**: `test_<module>.py` + `test_<module>_elaborate.py` for extended scenarios
- **Helpers**: `_make_state()`, `_minimal_fsm_dict()`, etc.
- **λ-kernel contracts**: `Executor.run` ⇒ `ex.oracle_calls == plan(...).predicted_calls` (Theorem 2) on τ·k^d-aligned input.

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

LambdaError (kernel)
├── ASTConstructionError
├── TerminationError
├── PlanningError
└── OracleError
```

## Code Conventions

- **Logging**: `from fsm_llm.logging import logger`
- **Models**: Pydantic v2 `BaseModel` with `model_validator` for complex validation. Recursive AST models work with `ConfigDict(frozen=True)` + `model_rebuild()`.
- **Exports**: Single `__all__` list in `__init__.py` — no dynamic extend/append
- **Security**: Internal key prefixes stripped by `clean_context_keys()`; XML tag sanitisation in prompts.
- **Thread safety**: Per-conversation `RLock`s in `FSMManager`.
- **Stdlib purity invariant**: Modules under `stdlib/<pkg>/lam_factories.py` import **only from `fsm_llm.lam`**. Enforced by AST-walk unit tests per subpackage.
