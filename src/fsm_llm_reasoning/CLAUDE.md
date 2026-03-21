# fsm_llm_reasoning — Reasoning Extension

## What This Package Does

Structured reasoning engine that orchestrates 9 reasoning strategies via FSMs. Uses a hierarchical FSM approach: orchestrator → classifier → specialized reasoning FSM.

## File Map

| File | Purpose |
|------|---------|
| `engine.py` | **ReasoningEngine** — main entry point. Loads FSMs, registers handlers, orchestrates solve_problem() |
| `reasoning_modes.py` | FSM definitions as Python dicts for all 9 strategies + orchestrator + classifier |
| `handlers.py` | ReasoningHandlers: validation, tracing, context pruning, retry limiting |
| `definitions.py` | Pydantic models: ReasoningStep, ReasoningTrace, ValidationResult, SolutionResult, ProblemContext, ContextSnapshot |
| `constants.py` | **ReasoningType** enum (9 values), **ContextKeys** class with class-level string constants, OrchestratorStates, ClassifierStates, Defaults |
| `utilities.py` | `map_reasoning_type()`, `get_available_reasoning_types()`, `load_fsm_definition()` |
| `__main__.py` | CLI: `python -m fsm_llm_reasoning "problem"` with --type, --context, --output, --save flags |
| `__init__.py` | Public API exports |
| `__version__.py` | Package version string |
| `exceptions.py` | ReasoningEngineError → ReasoningExecutionError, ReasoningClassificationError |

## Key Patterns

### Hierarchical FSM Architecture
1. **Orchestrator FSM** — states: PROBLEM_ANALYSIS → STRATEGY_SELECTION → EXECUTE_REASONING → SYNTHESIZE_SOLUTION → VALIDATE_REFINE → FINAL_ANSWER
2. **Classifier FSM** — invoked during STRATEGY_SELECTION to pick best ReasoningType
3. **Specialized FSMs** — one per ReasoningType, pushed onto FSM stack during EXECUTE_REASONING

### ReasoningType Enum (9 strategies)
`SIMPLE_CALCULATOR`, `ANALYTICAL`, `DEDUCTIVE`, `INDUCTIVE`, `CREATIVE`, `CRITICAL`, `HYBRID`, `ABDUCTIVE`, `ANALOGICAL`

### ContextKeys Pattern
All context keys are centralized in `ContextKeys` class with class-level string constants — prevents silent None on key mismatch. Sub-FSM result keys: `DEDUCTIVE_CONCLUSION`, `INDUCTIVE_HYPOTHESIS`, etc.

### Loop Prevention
- `Defaults.MAX_RETRIES` limits validation retries
- VALIDATE_REFINE state can loop back to EXECUTE_REASONING
- RetryLimiter handler manages count

### Context Pruning
- `Defaults.MAX_CONTEXT_SIZE` and `CONTEXT_PRUNE_THRESHOLD` control pruning
- ContextPruner handler automatically prunes when threshold exceeded

## Dependencies on Core
- `fsm_llm.API` — FSM execution
- `fsm_llm.handlers.HandlerTiming` — handler timing
- `fsm_llm.api.ContextMergeStrategy` — context merging for FSM stacking
- `fsm_llm.constants.DEFAULT_LLM_MODEL` — default model

## Testing
```bash
pytest tests/test_fsm_llm_reasoning/
```

## Gotchas
- FSM definitions are Python dicts in `reasoning_modes.py`, NOT JSON files
- `solve_problem()` returns `(solution_str, trace_dict)` tuple
- Adding new reasoning type requires updates in: constants.py (enum), reasoning_modes.py (FSM def), handlers.py (result merging), utilities.py (aliases + descriptions)
