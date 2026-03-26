# fsm_llm -- Core Package

## What This Package Does

Core FSM-LLM framework implementing the 2-pass architecture for stateful conversational AI. Pass 1 extracts data from user input and evaluates transitions using JsonLogic rules. Pass 2 generates the response from the resolved state with full context awareness. Supports FSM stacking (push/pop), a handler system with 8 lifecycle timing points, 100+ LLM providers via LiteLLM, and thread-safe per-conversation processing.

## File Map

| File | Purpose |
|------|---------|
| `api.py` | `API` class -- primary user-facing entry point. Factory methods (`from_file`), conversation management (`start_conversation`, `converse`), FSM stacking (`push_fsm`, `pop_fsm`), handler registration. Also defines `ContextMergeStrategy` enum and `FSMStackFrame` model |
| `fsm.py` | `FSMManager` -- conversation lifecycle orchestrator. Manages `FSMInstance` objects, per-conversation thread locks, FSM definition caching (LRU, max 64). Delegates all message processing to `MessagePipeline` |
| `pipeline.py` | `MessagePipeline` -- the 2-pass message processing engine. Stateless with respect to conversation instances. Handles data extraction, transition evaluation, state transitions, response generation, and handler execution bridging |
| `definitions.py` | Pydantic models: `State`, `Transition`, `TransitionCondition`, `FSMDefinition`, `FSMInstance`, `FSMContext`, `Conversation`. Request/response models: `DataExtractionRequest/Response`, `ResponseGenerationRequest/Response`, `TransitionDecisionRequest/Response`, `TransitionOption`, `TransitionEvaluation`. Enums: `LLMRequestType`, `TransitionEvaluationResult`. All exception classes |
| `handlers.py` | `HandlerSystem` (central orchestrator), `HandlerBuilder` (fluent API), `BaseHandler`, `LambdaHandler`, `FSMHandler` (Protocol), `HandlerTiming` enum (8 values). Type aliases: `ExecutionLambda`, `ConditionLambda` |
| `prompts.py` | Prompt builders: `DataExtractionPromptBuilder` + `DataExtractionPromptConfig`, `ResponseGenerationPromptBuilder` + `ResponsePromptConfig`, `TransitionPromptBuilder` + `TransitionPromptConfig` |
| `llm.py` | `LLMInterface` ABC (three methods: `extract_data`, `generate_response`, `decide_transition`) + `LiteLLMInterface` concrete implementation |
| `ollama.py` | Ollama-specific helpers: `is_ollama_model()`, `apply_ollama_params()`, `build_ollama_response_format()`, JSON schema format constants |
| `transition_evaluator.py` | `TransitionEvaluator` + `TransitionEvaluatorConfig` -- evaluates transitions against context using JsonLogic, produces `DETERMINISTIC`, `AMBIGUOUS`, or `BLOCKED` results |
| `expressions.py` | JsonLogic evaluator (`evaluate_logic`). Operators: `var`, `==`, `!=`, `===`, `!==`, `>`, `>=`, `<`, `<=`, `and`, `or`, `!`, `!!`, `if`, `in`, `contains`, `has_context`, `context_length`, `missing`, `missing_some`, arithmetic (`+`, `-`, `*`, `/`, `%`), `min`, `max`, `cat` |
| `context.py` | `clean_context_keys()` -- stateless context cleaning (strips None values, internal key prefixes, forbidden security patterns). `ContextCompactor` -- configurable context compaction with transient key removal and state-entry pruning |
| `runner.py` | Interactive CLI conversation runner, used by `__main__.py` |
| `validator.py` | `FSMValidator` + `FSMValidationResult` -- structural validation of FSM definitions. Also `validate_fsm_from_file()` |
| `visualizer.py` | ASCII FSM diagram generation: `visualize_fsm_ascii()`, `visualize_fsm_from_file()` |
| `utilities.py` | `extract_json_from_text()` (4 fallback strategies), `load_fsm_definition()`, `load_fsm_from_file()`, `validate_json_structure()`, `get_fsm_summary()` |
| `constants.py` | `DEFAULT_LLM_MODEL` (`"ollama_chat/qwen3.5:4b"`), `DEFAULT_TEMPERATURE` (0.5), `INTERNAL_KEY_PREFIXES`, `FORBIDDEN_CONTEXT_PATTERNS` (pre-compiled), `ALLOWED_JSONLOGIC_OPERATIONS`, `DEFAULT_MAX_HISTORY_SIZE` (5), `DEFAULT_HANDLER_TIMEOUT` (30.0), environment variable keys |
| `logging.py` | Loguru setup with conversation context binding. Import: `from fsm_llm.logging import logger`. Also `setup_logging()`, `handle_conversation_errors()`, `with_conversation_context()` |
| `__main__.py` | CLI entry point: run, validate, visualize modes |
| `__version__.py` | Package version string |
| `__init__.py` | Public API exports -- single `__all__` list (69 entries). Extension check functions (`has_workflows`, `has_agents`, etc.), `get_version_info()`, `quick_start()`, `enable_debug_logging()`, `disable_warnings()` |

## Key Patterns

### HandlerBuilder Fluent API

```python
from fsm_llm import create_handler, HandlerTiming

handler = (
    create_handler("name")
    .at(HandlerTiming.POST_TRANSITION)
    .on_state("state_name")
    .when_context_has("some_key")
    .do(lambda ctx: {"result": process(ctx)})
)
```

Builder methods: `.at()`, `.on_state()`, `.not_on_state()`, `.on_target_state()`, `.when_context_has()`, `.when_keys_updated()`, `.on_state_entry()`, `.on_state_exit()`, `.on_context_update()`, `.when()` (custom condition), `.do()` (execution function, required, returns the built handler).

### HandlerTiming -- 8 Values

`START_CONVERSATION`, `PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `END_CONVERSATION`, `ERROR`

### State Fields (definitions.py)

- `extraction_instructions` -- what data to extract from user input (Pass 1). NOT `instructions`.
- `response_instructions` -- how to respond to the user (Pass 2). NOT `instructions`.
- `purpose` -- what the state should accomplish
- `required_context_keys` -- keys that must exist in context before entering this state
- `description` -- brief description of the state

### Security

- Internal key prefixes: `_`, `system_`, `internal_`, `__` -- stripped by `clean_context_keys()`
- Forbidden context patterns: password, secret, auth/access/refresh/bearer token, api_key -- warned or stripped depending on `strip_forbidden_keys` flag
- Allowed JsonLogic operations are whitelisted in `ALLOWED_JSONLOGIC_OPERATIONS`
- XML tag sanitization in prompts

### ContextMergeStrategy Enum

Controls context merging during `pop_fsm()`: `UPDATE` (merge returned context into parent, default) or `PRESERVE` (keep parent context unchanged).

### MessagePipeline 2-Pass Flow

1. `PRE_PROCESSING` handlers fire
2. Data extraction via LLM (`DataExtractionRequest` / `DataExtractionResponse`)
3. Context update with extracted data
4. `CONTEXT_UPDATE` handlers fire (if keys changed)
5. Transition evaluation via `TransitionEvaluator` (DETERMINISTIC / AMBIGUOUS / BLOCKED)
6. If AMBIGUOUS, LLM decides (`TransitionDecisionRequest` / `TransitionDecisionResponse`)
7. `PRE_TRANSITION` handlers fire
8. State transition
9. `POST_TRANSITION` handlers fire
10. `POST_PROCESSING` handlers fire
11. Response generation via LLM (`ResponseGenerationRequest` / `ResponseGenerationResponse`)

### FSM Stacking (push/pop)

`API.push_fsm()` saves the current conversation state onto a stack and starts a new FSM. `API.pop_fsm()` restores the parent FSM and merges context according to `ContextMergeStrategy`. Stack depth is tracked per conversation ID.

### ContextCompactor

```python
from fsm_llm.context import ContextCompactor

compactor = ContextCompactor(
    transient_keys={"action_result", "action_errors"},
    prune_on_entry={"review": {"structure_done", "connections_done"}},
)
```

Register `compactor.compact` at `PRE_PROCESSING` and `compactor.prune` at `POST_TRANSITION`.

## Exception Hierarchy

```
FSMError
    StateNotFoundError
    InvalidTransitionError
    LLMResponseError
    TransitionEvaluationError
    HandlerSystemError
        HandlerExecutionError
```

FSM errors defined in `fsm_llm.definitions`, handler errors in `fsm_llm.handlers` (both under `FSMError`).

## Dependencies

- `loguru` -- structured logging
- `litellm` (>=1.82, <2.0) -- LLM provider abstraction (100+ providers)
- `pydantic` (>=2.0) -- data validation and models
- `python-dotenv` -- environment variable loading

## Testing

506 tests across 20 test files in `tests/test_fsm_llm/`.

```bash
pytest tests/test_fsm_llm/            # Core package tests
pytest tests/test_fsm_llm_regression/ # Regression tests
```

Key fixtures (in `tests/conftest.py`):

- `sample_fsm_definition` -- v3.0 format FSM definition
- `sample_fsm_definition_v2` -- v4.1 format FSM definition
- `mock_llm_interface` -- single-pass mock LLM (`Mock(spec=LLMInterface)`)
- `mock_llm2_interface` -- 2-pass architecture mock LLM (`MockLLM2Interface`)

Test files cover: `test_api.py`, `test_api_elaborate.py`, `test_fsm.py`, `test_fsm_elaborate.py`, `test_pipeline.py`, `test_handlers_unit.py`, `test_handler_timeout.py`, `test_transition_evaluator.py`, `test_expressions.py`, `test_prompts_unit.py`, `test_llm_unit.py`, `test_ollama.py`, `test_context_unit.py`, `test_utilities_unit.py`, `test_validator_unit.py`, `test_visualizer_unit.py`, `test_runner_unit.py`, `test_logging_unit.py`, `test_logging_structured.py`, `test_classification_transitions.py`.

## Gotchas

- `converse()` returns `str`, not a tuple -- do not unpack it
- The bare `instructions` field on State is silently ignored by Pydantic -- always use `extraction_instructions` and `response_instructions`
- `handler_error_mode` defaults to `"continue"` -- handler failures are logged but swallowed unless set to `"raise"`
- `FSMManager` uses per-conversation thread locks -- concurrent calls to `converse()` for the same conversation ID are serialized
- `error_mode` on `HandlerSystem` defaults to `"continue"` -- set to `"raise"` to surface handler errors
- `DEFAULT_LLM_MODEL` is `"ollama_chat/qwen3.5:4b"` -- override with `model=` parameter or `LLM_MODEL` environment variable
- FSM definitions use v4.1 format -- older formats may work but v4.1 fields (`purpose`, `extraction_instructions`, `response_instructions`) are preferred
- `clean_context_keys()` strips keys starting with `_`, `system_`, `internal_`, `__` -- do not store user data under these prefixes
