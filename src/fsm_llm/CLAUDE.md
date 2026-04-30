# fsm_llm — Core Package

The fsm_llm core package: **typed λ-calculus kernel** (`runtime/`, was `lam/` pre-R4) + **FSM dialog surface** (`dialog/`, was top-level pre-R4) + **standard library** of named λ-term factories (`stdlib/`). All execution paths share a single Executor.

- **Version**: 0.8.0
- **Python**: 3.10, 3.11, 3.12
- **Deps**: `loguru`, `litellm` (>=1.82,<2.0), `pydantic` (>=2.0), `python-dotenv`

See `docs/lambda_fsm_merge.md` for the **merge contract (canonical, 2026-04-29)** — invariants I1-I6, falsification gates G1-G5, the unified-API specification, and the deprecation calendar. See `docs/lambda.md` for the architectural thesis (Theorems 1-5). `docs/lambda_integration.md` (v2.0 audit) is **superseded** but preserved for historical context. Per `lambda.md` §11, the kernel + dialog + stdlib live here; the legacy `fsm_llm_*` siblings are sys.modules shims.

## File Map

```
fsm_llm/
├── runtime/                # M1 — typed λ-AST + Executor + Planner + Oracle. (Renamed from lam/ in plan v3 R4 — lam/ shim removed in 0.6.0.)
│   ├── ast.py              #   Var, Abs, App, Let, Case, Combinator, Fix, Leaf, Term, is_term
│   ├── dsl.py              #   var, abs_, app, let_, case_, fix, leaf, split, peek, fmap, ffilter, reduce_, concat, cross
│   ├── combinators.py      #   ReduceOp enum + BUILTIN_OPS dict (closed registry)
│   ├── executor.py         #   Executor — β-reduction, depth-bounded, per-Leaf cost
│   ├── planner.py          #   plan(), Plan, PlanInputs — closed-form (k*, τ*, d, predicted_calls)
│   ├── oracle.py           #   Oracle Protocol, LiteLLMOracle (adapter; R3 env branch — D-005)
│   ├── _litellm.py         #   LLMInterface ABC + LiteLLMInterface (was top-level llm.py — R4 step 23). Private-by-convention adapter.
│   ├── _handlers_ast.py    #   private — compose() + AST splicers + handler_runner_var helpers (moved from handlers.py at 0.8.0; re-exported from there for back-compat).
│   ├── cost.py             #   CostAccumulator, LeafCall — per-leaf cost telemetry
│   ├── errors.py           #   LambdaError → ASTConstructionError, TerminationError, PlanningError, OracleError
│   └── constants.py        #   K_DEFAULT, TAU_DEFAULT, DEPTH_LIMIT, …
│   (Kernel is closed against `dialog/` as of D-001 / plan_2026-04-27_5d8a038b. `compile_fsm` / `compile_fsm_cached` are sourced from `fsm_llm.dialog.compile_fsm` directly — runtime/__init__.py never imports from dialog/.)
│   (See src/fsm_llm/runtime/CLAUDE.md for the kernel-detail file map.)
│
├── dialog/                 # FSM dialog surface (R4) — was top-level fsm_llm/{api,fsm,pipeline,prompts,classification,transition_evaluator,definitions,session}.py.
│   ├── api.py              #   API class — primary entry point (from_file, from_definition, converse, push/pop_fsm)
│   ├── fsm.py              #   FSMManager — per-conversation RLocks; compiled-term cache lives in compile_fsm.py (R2)
│   ├── prompts.py          #   *PromptBuilder (BasePromptBuilder dedup since 0.8.0) + to_template_and_schema producers + free classification_template
│   ├── classification.py   #   Classifier, HierarchicalClassifier, IntentRouter, HandlerFn type alias
│   ├── transition_evaluator.py  #   TransitionEvaluator + TransitionEvaluatorConfig — DETERMINISTIC | AMBIGUOUS | BLOCKED
│   ├── definitions.py      #   Pydantic models (State, Transition, FSMDefinition, FSMContext, FSMInstance, Conversation, classification/extraction models). Type re-exports from fsm_llm.types removed in 0.8.0; import from fsm_llm.types directly.
│   ├── session.py          #   SessionStore ABC + FileSessionStore — atomic JSON writes (temp + rename)
│   ├── compile_fsm.py      #   M2 — compile_fsm() : FSMDefinition → Term + R2 compile_fsm_cached(fsm, fsm_id) — lru_cache(64). Was lam/fsm_compile.py.
│   ├── extraction.py       #   private — ExtractionEngine (Pass-1 extraction cluster: data + field + classification). Extracted from turn.py at 0.8.0; ~976 LOC.
│   └── turn.py             #   MessagePipeline — compiled-path 2-pass body. ~1,747 LOC (post-extraction.py extraction). Renamed from pipeline.py in R13.
│   (See src/fsm_llm/dialog/CLAUDE.md for the dialog-detail file map.)
│
├── stdlib/                 # M3 — named λ-term factories
│   ├── agents/             #   slice 1: react_term, rewoo_term, reflexion_term, memory_term + 12 class agents
│   ├── reasoning/          #   slice 2: 11 strategy factories + classifier_term + solve_term + ReasoningEngine
│   ├── workflows/          #   slice 3: linear/branch/switch/parallel/retry term factories + WorkflowEngine
│   └── long_context/       #   M5: niah, aggregate, pairwise, multi_hop, niah_padded + helpers
│   (See src/fsm_llm/stdlib/CLAUDE.md for the stdlib index.)
│
├── program.py              # Program facade (R1) — unified entry over (term, oracle, session, handlers); from_fsm/from_term/from_factory + .invoke/.explain. ExplainOutput value object. Public __init__ is term-mode only since 0.8.0; from_fsm takes explicit kwargs + **llm_kwargs (no api_kwargs catch-all). (Pre-0.7.0 .run/.converse/.register_handler aliases removed at I5 closure.)
├── types.py                # Neutral types layer (since 0.7.0) — FSMError hierarchy + runtime-touching Pydantic models
├── handlers.py             # HandlerSystem, HandlerBuilder, BaseHandler, LambdaHandler, HandlerTiming enum (8 points). Top-level. compose + AST splicers re-exported from runtime/_handlers_ast.py since 0.8.0 (transparent for callers).
├── ollama.py               # Ollama-specific helpers (thinking disable, json_schema format)
├── expressions.py          # evaluate_logic() — JsonLogic evaluator (var, and, or, ==, in, has_context, context_length)
├── context.py              # clean_context_keys() + ContextCompactor (transient-key clearing, pruning, summarisation)
├── memory.py               # WorkingMemory — 4 named buffers (core, scratch, environment, reasoning)
├── runner.py               # Interactive CLI conversation runner
├── validator.py            # FSMValidator.validate() + validate_fsm_from_file()
├── visualizer.py           # visualize_fsm_ascii() + visualize_fsm_from_file() (full/compact/minimal)
├── utilities.py            # extract_json_from_text(), load_fsm_definition(), load_fsm_from_file()
├── constants.py            # DEFAULT_LLM_MODEL, security patterns, INTERNAL_KEY_PREFIXES, ALLOWED_JSONLOGIC_OPERATIONS
├── logging.py              # Loguru setup, enable_debug_logging(), disable_warnings()
├── __main__.py             # CLI entry point (run, validate, visualize modes)
├── __version__.py          # "{__version__}"
└── __init__.py             # Layered __all__ — L1/L2/L3/L4 partitions enforced by tests/test_fsm_llm/test_layering.py
```

## Where Execution Actually Happens

There is one runtime: **`fsm_llm.runtime.Executor`** (still importable as `fsm_llm.lam.Executor` via the R4 shim). Both surfaces compile to the same AST.

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

- **`Program`** (`program.py`) — **Unified facade** over `(term, oracle, optional_session, optional_handlers)`. `.invoke(...)` is the single verb and returns `Result` uniformly in every mode. `Program.invoke(message=...)` on a term-mode Program raises `ProgramModeError`; vice versa.
  - Constructors: `Program.from_fsm(defn, *, oracle=None, session=None, handlers=None, profile=None, model=None, api_key=None, temperature=None, max_tokens=None, max_history_size=5, max_message_length=1000, handler_error_mode="continue", transition_config=None, **llm_kwargs)` (explicit LLM/API kwargs since 0.8.0; constructs internal `API`); `Program.from_term(term, *, oracle=None, handlers=None, ...)` (wraps a pre-authored λ-term); `Program.from_factory(factory, factory_args=(), factory_kwargs=None, *, oracle=None, handlers=None, ...)` (calls factory immediately, wraps result).
  - Public `Program(...)` constructor: term-mode only since 0.8.0 (FSM-mode is `from_fsm` exclusively). The pre-0.8.0 `_api` / `_profile` private kwargs were dropped from the public ctor.
  - Surface: `.invoke(message=..., conversation_id=...)` (FSM mode) or `.invoke(inputs={...})` (term/factory mode), returning `Result` uniformly. `.explain()` → `ExplainOutput(plans, leaf_schemas, ast_shape)`; `leaf_schemas` is keyed by synthesised `leaf_NNN_<template-prefix>` ids; `ast_shape` is an indented multi-line rendering of the term skeleton. (Pre-0.7.0 `.run(**env)` / `.converse(msg, conv_id)` / `.register_handler(h)` aliases were removed at I5 closure — use `.invoke(...)` and the `handlers=[...]` constructor kwarg.)
  - Oracle handling in `from_fsm`: when `oracle=` is supplied, must be a `LiteLLMOracle`; we unwrap to `oracle._llm` and pass it to `API` as `llm_interface`. Non-`LiteLLMOracle` instances raise `TypeError`. The default oracle is constructed lazily — building a Program without an oracle never touches the network or LLM credentials.
- **`ExplainOutput`** (`program.py`) — frozen dataclass: `plans: list[Plan]`, `leaf_schemas: dict[str, type | None]`, `ast_shape: str`. Returned by `Program.explain()`.
- **`API`** (`api.py`) — User-facing entry point.
  - Factory: `from_file(path, **kwargs)`, `from_definition(fsm_def, **kwargs)`. Both compile the FSM via `lam.fsm_compile.compile_fsm()` and cache the resulting `Term`.
  - Conversation: `start_conversation(initial_context)` → `(conv_id, greeting)`, `converse(msg, conv_id)` → str, `end_conversation(conv_id)`, `has_conversation_ended(conv_id)`
  - Queries: `get_data(conv_id)`, `get_current_state(conv_id)`, `get_conversation_history(conv_id)`, `list_active_conversations()`
  - FSM stacking: `push_fsm(conv_id, new_fsm)`, `pop_fsm(conv_id, merge_strategy)`, `get_stack_depth(conv_id)`
  - Handlers: `register_handler(handler)`, `create_handler(name)` → HandlerBuilder
  - Management: `update_context(conv_id, data)`, `close()`
- **`FSMManager`** (`fsm.py`) — Orchestration with per-conversation thread locks. As of R2 (plan v3 step 8/9), the compiled-term cache lives in the kernel via `fsm_llm.lam.compile_fsm_cached` (lru_cache(maxsize=64) keyed on `(fsm_id, fsm.model_dump_json())`); `FSMManager.get_compiled_term` is a 3-line shim. Thin adapter over `lam.Executor`.
  - `start_conversation(fsm_id, initial_context)`, `process_message(conv_id, msg)`, `get_current_state(instance)`
- **`MessagePipeline`** (`dialog/turn.py`; file renamed from `pipeline.py` in R13) — Compiled-path 2-pass body. Internal-only post-M2 S11. Post-M4 (`da7d03c`), the constructor takes a required `oracle: Oracle` argument and all call sites read from `self._oracle` (enforced by `tests/test_fsm_llm/test_oracle_ownership.py`). Since 0.8.0, the Pass-1 extraction cluster lives in `dialog/extraction.py` as `ExtractionEngine`, owned by `MessagePipeline` as `self._extraction`.
  - Pass 1: data extraction → field extractions → classification extractions → transition evaluation → state transition (delegated to `ExtractionEngine`)
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
pytest tests/test_fsm_llm/         # core
pytest tests/test_fsm_llm_lam/     # λ-kernel (Executor, Planner, DSL, FSM compiler)
```

(Verify per-package counts via `.venv/bin/python -m pytest --collect-only -q | tail -3`. Total at HEAD: ~3,194.)

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
- **Stdlib purity invariant**: Modules under `stdlib/<pkg>/lam_factories.py` import **only from `fsm_llm.runtime`**. Enforced by AST-walk unit tests per subpackage.
