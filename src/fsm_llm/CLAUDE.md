# fsm_llm — Core Package

## What This Package Does

Core FSM-LLM framework. Implements the 2-pass architecture for stateful conversational AI: Pass 1 extracts data and evaluates transitions, Pass 2 generates responses from the final state.

## File Map

| File | Purpose |
|------|---------|
| `api.py` | **API** class — primary user-facing entry point. Factory methods, conversation management, FSM stacking |
| `fsm.py` | **FSMManager** — core orchestration. 2-pass processing, per-conversation thread locks |
| `definitions.py` | Pydantic models: State, Transition, TransitionCondition, FSMDefinition, FSMInstance, FSMContext, Conversation |
| `handlers.py` | **HandlerSystem**, **HandlerBuilder** (fluent API), BaseHandler, LambdaHandler, **HandlerTiming** enum (8 values) |
| `prompts.py` | Prompt builders: DataExtractionPromptBuilder, ResponseGenerationPromptBuilder, TransitionPromptBuilder |
| `llm.py` | **LLMInterface** ABC + **LiteLLMInterface** implementation. Three methods: extract_data, generate_response, decide_transition |
| `transition_evaluator.py` | **TransitionEvaluator** — rule-based evaluation → DETERMINISTIC, AMBIGUOUS, or BLOCKED |
| `expressions.py` | JsonLogic evaluator. Operators: var, ==, !=, <, >, and, or, in, has_context, context_length |
| `validator.py` | **FSMValidator** — structural validation of FSM definitions |
| `visualizer.py` | ASCII FSM diagram generation |
| `utilities.py` | JSON extraction (4 fallback strategies), FSM loading |
| `context.py` | Context cleaning utilities: `clean_context_keys()` — extracted from FSMManager |
| `constants.py` | Defaults (DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE), security patterns (pre-compiled), internal key prefixes |
| `logging.py` | loguru setup. Use `from fsm_llm.logging import logger` |
| `__main__.py` | CLI entry point: run, validate, visualize modes |
| `__init__.py` | Public API exports — single `__all__` list, extension check functions |

## Key Patterns

### HandlerBuilder Fluent API
```python
create_handler("name").at(HandlerTiming.POST_TRANSITION).on_state("state_name").do(lambda ctx: {...})
```
Methods: `.at()`, `.on_state()`, `.not_on_state()`, `.on_target_state()`, `.when_context_has()`, `.when_keys_updated()`, `.on_state_entry()`, `.on_state_exit()`, `.on_context_update()`, `.do()`

### HandlerTiming Values
`START_CONVERSATION`, `PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `END_CONVERSATION`, `ERROR`

### State Fields (definitions.py)
- `extraction_instructions` — what data to extract (NOT `instructions`)
- `response_instructions` — how to respond
- `required_context_keys` — keys that must exist before entering state
- `purpose` — what the state should accomplish

### Security
- Internal key prefixes: `_`, `system_`, `internal_`, `__`
- Forbidden context patterns: password, secret, token, api_key
- XML tag sanitization in prompts

### Exception Hierarchy
`FSMError` → `StateNotFoundError`, `InvalidTransitionError`, `LLMResponseError`, `TransitionEvaluationError`
`HandlerSystemError` → `HandlerExecutionError`

## Testing
```bash
pytest tests/test_fsm_llm/           # Core package tests
pytest tests/test_fsm_llm_regression/ # Regression tests
```
Fixtures in `tests/conftest.py`: `mock_llm_interface`, `mock_llm2_interface`, `sample_fsm_definition`, `sample_fsm_definition_v2`

## Gotchas
- `converse()` returns `str`, NOT a tuple — don't unpack
- State `instructions` field is silently ignored — use `extraction_instructions` / `response_instructions`
- Handler `error_mode` defaults to `"continue"` — handler failures are swallowed unless set to `"raise"`
- FSM definitions use version `"4.1"` format
