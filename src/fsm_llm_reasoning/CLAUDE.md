# fsm_llm_reasoning -- Claude Code Instructions

## What This Package Does

Structured reasoning engine that orchestrates 9 reasoning strategies via hierarchical FSMs. Uses a 3-level FSM architecture: orchestrator (controls flow) -> classifier (picks strategy) -> specialized reasoning FSM (executes strategy). The main entry point is `ReasoningEngine.solve_problem()` which returns a `(solution_str, trace_dict)` tuple.

## File Map

| File | Purpose |
|------|---------|
| `engine.py` | `ReasoningEngine` -- loads FSM dicts, registers 6 handlers, runs `solve_problem()` loop with sub-FSM push/pop |
| `reasoning_modes.py` | All FSM definitions as Python dicts (NOT JSON files). `ALL_REASONING_FSMS` dict maps names to definitions. 11 FSMs: orchestrator, classifier, + 9 reasoning types |
| `handlers.py` | `ReasoningHandlers` (static: `validate_solution`, `update_reasoning_trace`, `prune_context`), `ContextManager` (static: `extract_relevant_context`, `merge_reasoning_results`), `OutputFormatter` (static: `extract_final_solution`, `format_reasoning_summary`) |
| `definitions.py` | Pydantic v2 models: `ReasoningStep`, `ReasoningTrace`, `ValidationResult`, `SolutionResult`, `ProblemContext`, `ReasoningClassificationResult`. Base classes: `TimestampedModel`, `ValidatedModel`. Enums: `ConfidenceLevel`, `ReasoningStepType`, `ProblemDomain` |
| `constants.py` | `ReasoningType` (str enum, 9 values), `OrchestratorStates` (6 states), `ClassifierStates` (4 states), `ContextKeys` (class-level string constants, ~80 keys), `HandlerNames`, `Defaults`, `ErrorMessages`, `LogMessages` |
| `utilities.py` | `load_fsm_definition(name)` -- deep-copies from `ALL_REASONING_FSMS`. `map_reasoning_type(str)` -- normalizes aliases to enum values. `get_available_reasoning_types()` -- returns type->description dict |
| `exceptions.py` | `ReasoningEngineError(FSMError)` -> `ReasoningExecutionError` (has `reasoning_type` attr), `ReasoningClassificationError` |
| `__main__.py` | CLI: `python -m fsm_llm_reasoning "problem"`. Flags: `--type`, `--context`, `--model`, `--output` (text/json/detailed), `--save`, `--verbose`, `--quiet`, `--list-types` |
| `__init__.py` | 12 exports in `__all__`: ReasoningEngine, ReasoningType, ReasoningStep, ReasoningTrace, ValidationResult, ReasoningClassificationResult, ProblemContext, SolutionResult, 3 exceptions, get_available_reasoning_types, __version__ |
| `__version__.py` | Imports `__version__` from `fsm_llm.__version__` |

## Key Patterns

### Hierarchical FSM (3-level)

1. **Orchestrator FSM** -- 6 states: `PROBLEM_ANALYSIS` -> `STRATEGY_SELECTION` -> `EXECUTE_REASONING` -> `SYNTHESIZE_SOLUTION` -> `VALIDATE_REFINE` -> `FINAL_ANSWER`. VALIDATE_REFINE loops back to EXECUTE_REASONING on failure.
2. **Classifier FSM** -- 4 states: `ANALYZE_DOMAIN` -> `ANALYZE_STRUCTURE` -> `IDENTIFY_REASONING_NEEDS` -> `RECOMMEND_STRATEGY`. Invoked by handler during STRATEGY_SELECTION.
3. **Specialized FSMs** -- one per ReasoningType, pushed via `API.push_fsm()` during EXECUTE_REASONING, popped with `ContextMergeStrategy.UPDATE`.

### ReasoningType Enum

`str` enum in `constants.py`: `SIMPLE_CALCULATOR`, `ANALYTICAL`, `DEDUCTIVE`, `INDUCTIVE`, `ABDUCTIVE`, `ANALOGICAL`, `CREATIVE`, `CRITICAL`, `HYBRID`.

### ContextKeys Class Pattern

All context keys are class-level string constants on `ContextKeys` (not an enum, not a dataclass). Organized by section: problem analysis, domain analysis, strategy selection, solution synthesis, validation, execution control, and per-reasoning-type keys. ~80 constants total.

### Handler Registration

Engine registers 6 handlers on the orchestrator API:
- `ORCHESTRATOR_CLASSIFIER` -- CONTEXT_UPDATE timing, triggers `_classify_problem()` when `problem_type` key updates
- `ORCHESTRATOR_EXECUTOR` -- on_state_entry(EXECUTE_REASONING), triggers `_prepare_reasoning_execution()`
- `ORCHESTRATOR_VALIDATOR` -- CONTEXT_UPDATE timing, triggers `validate_solution()` when `proposed_solution` updates
- `CONTEXT_PRUNER` -- PRE_TRANSITION timing, runs `prune_context()`
- `REASONING_TRACER` -- POST_TRANSITION timing, runs `update_reasoning_trace()` (registered on both orchestrator and classifier)
- `RETRY_LIMITER` -- on_state_entry(VALIDATE_REFINE), runs `_check_retry_limit()`

### Loop Prevention

- `Defaults.MAX_RETRIES = 3` -- validation retry cap
- `Defaults.MAX_SUB_FSM_ITERATIONS = 30` -- sub-FSM execution cap
- `Defaults.MAX_CLASSIFICATION_ITERATIONS = 10` -- classifier convergence cap
- `Defaults.MAX_TOTAL_ITERATIONS = 50` -- hard ceiling on solve_problem() main loop

### Context Pruning

- `Defaults.MAX_CONTEXT_SIZE = 10000` characters
- `Defaults.CONTEXT_PRUNE_THRESHOLD = 8000` -- pruning starts at 80% of max
- Lists truncated to last `PRUNE_LIST_MAX_LENGTH = 10` items
- Strings truncated to `PRUNE_STRING_MAX_LENGTH = 1000` chars
- Preserve keys: problem_statement, problem_type, reasoning_strategy, proposed_solution, solution_valid, retry_count

### FSM Definitions Are Python Dicts

All FSM definitions live in `reasoning_modes.py` as Python dictionaries using `OrchestratorStates`, `ClassifierStates`, and `ContextKeys` constants. They are NOT loaded from JSON files. `load_fsm_definition()` returns a deep copy from the `ALL_REASONING_FSMS` dict.

### Result Merging

`ContextManager.merge_reasoning_results()` maps sub-FSM context keys to orchestrator-level keys per reasoning type. Each type has its own mapping (e.g., deductive maps `CONCLUSION` -> `DEDUCTIVE_CONCLUSION`). None values are filtered out.

## Dependencies on Core

- `fsm_llm.API` -- FSM execution, `from_definition()`, `push_fsm()`, `pop_fsm()`, `converse()`, `get_data()`, `update_context()`
- `fsm_llm.ContextMergeStrategy` -- used with `pop_fsm()` for result merging
- `fsm_llm.handlers.HandlerTiming` -- CONTEXT_UPDATE, PRE_TRANSITION, POST_TRANSITION for handler registration
- `fsm_llm.definitions.FSMError` -- base class for `ReasoningEngineError`
- `fsm_llm.constants.DEFAULT_LLM_MODEL` -- default model for `Defaults.MODEL`
- `fsm_llm.logging.logger` -- loguru logger

## Exception Hierarchy

```
FSMError (from fsm_llm.definitions)
  -> ReasoningEngineError (base, accepts message + details dict)
       -> ReasoningExecutionError (has reasoning_type attribute)
       -> ReasoningClassificationError
```

## Testing

112 tests across 6 files:

```bash
pytest tests/test_fsm_llm_reasoning/ -v
```

| Test File | Focus |
|-----------|-------|
| `test_engine.py` | Engine init, solve_problem flow, classification, execution, sub-FSM push/pop |
| `test_handlers.py` | Validation logic, trace updates, context pruning, result merging, output formatting |
| `test_definitions.py` | Pydantic model validation, computed fields, serialization, edge cases |
| `test_constants.py` | Enum values, state strings, ContextKeys presence, Defaults values |
| `test_exceptions.py` | Exception hierarchy, attributes, details dict |
| `test_audit_fixes.py` | Regression tests for previously identified issues |

## Gotchas

- **FSM definitions are Python dicts, not JSON files**. They live in `reasoning_modes.py` and use constant references from `constants.py`.
- **`solve_problem()` returns a tuple** `(str, dict)` -- not a `SolutionResult` model. The dict contains `reasoning_trace`, `summary`, `final_context`, and `all_responses`.
- **Adding a new reasoning type requires updates in 4+ files**: `constants.py` (enum + context keys), `reasoning_modes.py` (FSM dict + ALL_REASONING_FSMS), `handlers.py` (merge_reasoning_results mapping), `utilities.py` (aliases + description).
- **ContextKeys uses class-level string constants**, not an enum or dataclass. Access as `ContextKeys.PROBLEM_STATEMENT` (returns `"problem_statement"`).
- **`solve_problem()` is thread-safe** per engine instance via `_solve_lock`, but only one solve runs at a time on a given engine.
- **Unknown reasoning types default to analytical**. `map_reasoning_type()` logs a warning and falls back to `ReasoningType.ANALYTICAL`.
- **`__version__` is imported from core** (`fsm_llm.__version__`), not independently maintained.
- **OutputFormatter.extract_final_solution() has a priority chain** of ~11 context keys it checks in order; the first non-empty value wins.
