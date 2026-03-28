# fsm_llm_reasoning -- Structured Reasoning Engine

Multi-strategy reasoning engine that orchestrates 9 reasoning approaches via FSMs. Classifies problems, routes to appropriate strategy FSM, validates solutions with retry logic.

- **Version**: 0.3.0 (synced from fsm_llm)
- **Extra deps**: None beyond core fsm_llm
- **Install**: `pip install fsm-llm[reasoning]`

## File Map

```
fsm_llm_reasoning/
├── engine.py           # ReasoningEngine -- main class, thread-safe solve_problem()
├── reasoning_modes.py  # ALL_REASONING_FSMS dict: orchestrator + classifier + 9 strategy FSMs
├── handlers.py         # ReasoningHandlers (validate, trace, prune), ContextManager, OutputFormatter
├── definitions.py      # Pydantic models: ReasoningStep, ReasoningTrace, SolutionResult, ProblemContext, ValidationResult
├── constants.py        # ReasoningType enum, ContextKeys (220+), OrchestratorStates, ClassifierStates, Defaults
├── utilities.py        # load_fsm_definition(), map_reasoning_type(), get_available_reasoning_types()
├── exceptions.py       # ReasoningEngineError -> ReasoningExecutionError, ReasoningClassificationError
├── __main__.py         # CLI: python -m fsm_llm_reasoning "problem" [--type, --model, --output, --save]
├── __version__.py      # Imports from fsm_llm.__version__
└── __init__.py         # Public exports: ReasoningEngine, ReasoningType, models, exceptions, get_available_reasoning_types
```

## Key Classes

- **ReasoningEngine** (`engine.py`) -- Main entry point, thread-safe via _solve_lock
  - Constructor: `__init__(model=DEFAULT_LLM_MODEL, **kwargs)`
  - `solve_problem(problem, initial_context=None)` → `(solution_text, trace_info_dict)`
  - Internally: loads FSMs → creates API instances → registers handlers → classifies → executes → validates → synthesizes
  - Retry logic: up to MAX_RETRIES (3) on validation failure
  - Loop prevention: MAX_TOTAL_ITERATIONS (50), MAX_SUB_FSM_ITERATIONS (30)

## FSM Architecture (reasoning_modes.py)

Three layers stored in `ALL_REASONING_FSMS` dict:

| FSM | States | Purpose |
|-----|--------|---------|
| `orchestrator_fsm` | 6: PROBLEM_ANALYSIS → STRATEGY_SELECTION → EXECUTE_REASONING → SYNTHESIZE_SOLUTION → VALIDATE_REFINE → FINAL_ANSWER | Overall flow control |
| `classifier_fsm` | 4: ANALYZE_DOMAIN → ANALYZE_STRUCTURE → IDENTIFY_REASONING_NEEDS → RECOMMEND_STRATEGY | Problem classification |
| `simple_calculator_fsm` | 3 | Extract → Calculate → Verify |
| `analytical_fsm` | 3 | Identify components → Analyze → Integrate |
| `deductive_fsm` | 3 | Extract premises → Derive → Validate |
| `inductive_fsm` | 3 | Observe → Identify trends → Generalize |
| `creative_fsm` | 3 | Explore → Generate → Evaluate |
| `critical_fsm` | 3 | Extract claims → Analyze → Assess |
| `abductive_fsm` | 3 | Identify surprises → Hypothesize → Evaluate |
| `analogical_fsm` | 3 | Identify target → Find analogs → Map/adapt |
| `hybrid_fsm` | 3 | Multiple paths → Integrate results |

## ReasoningType Enum (constants.py)

`SIMPLE_CALCULATOR`, `ANALYTICAL`, `DEDUCTIVE`, `INDUCTIVE`, `CREATIVE`, `CRITICAL`, `HYBRID`, `ABDUCTIVE`, `ANALOGICAL`

## Handlers (handlers.py)

**ReasoningHandlers** (static methods):
- `validate_solution(context)` -- Multi-check validation: has_solution, has_insights, sufficient_detail, addresses_problem. Enforces MAX_RETRIES
- `update_reasoning_trace(context)` -- Records state transitions with context snapshots. Prunes at MAX_TRACE_STEPS (50)
- `prune_context(context)` -- Prevents explosion at 8000 char threshold. Truncates lists (10 items), strings (1000 chars)

**ContextManager** (static methods):
- `extract_relevant_context(source, target_keys, max_size)` -- Filter context by key list
- `merge_reasoning_results(orchestrator_ctx, sub_fsm_ctx, reasoning_type)` -- Type-specific result mapping back to orchestrator

**OutputFormatter** (static methods):
- `extract_final_solution(context)` -- Priority: FINAL_SOLUTION > PROPOSED_SOLUTION > CALCULATION_RESULT > INTEGRATED_ANALYSIS
- `format_reasoning_summary(trace_info)` -- Human-readable trace summary

## Data Models (definitions.py)

All models use `TimestampedModel` base (auto-timestamps with computed age_seconds).

- **ReasoningStep**: step_type (ReasoningStepType enum), content, confidence (0-1), evidence list, context_keys_used
- **ReasoningTrace**: steps list, reasoning_types_used set, final_confidence, execution_time_seconds. Computed: total_steps, reasoning_complexity
- **ValidationResult**: is_valid, confidence, checks dict, issues list, recommendations. Computed: pass_rate, validation_summary
- **ProblemContext**: problem_statement, domain (ProblemDomain enum), constraints, initial_context, priority. Computed: has_constraints, is_high_priority
- **SolutionResult**: solution, confidence, reasoning_summary, trace, validation_result, alternative_solutions, key_insights. Computed: confidence_level, is_high_confidence, solution_quality_summary

**Enums**: ConfidenceLevel (LOW/MEDIUM/HIGH/VERY_HIGH), ReasoningStepType (10 types), ProblemDomain (8 domains)

## Constants (constants.py)

- **ContextKeys**: 220+ string constants organized by category (problem, domain, strategy, solution, validation, per-reasoning-type)
- **OrchestratorStates**: PROBLEM_ANALYSIS, STRATEGY_SELECTION, EXECUTE_REASONING, SYNTHESIZE_SOLUTION, VALIDATE_REFINE, FINAL_ANSWER
- **ClassifierStates**: ANALYZE_DOMAIN, ANALYZE_STRUCTURE, IDENTIFY_REASONING_NEEDS, RECOMMEND_STRATEGY
- **Defaults**: MODEL (from fsm_llm), MAX_RETRIES=3, MAX_TOTAL_ITERATIONS=50, MAX_SUB_FSM_ITERATIONS=30, MAX_CLASSIFICATION_ITERATIONS=10, MAX_TRACE_STEPS=50, MIN_SOLUTION_LENGTH=20, CONTEXT_PRUNE_THRESHOLD=8000

## Utilities (utilities.py)

- `load_fsm_definition(fsm_name)` -- Deep copies from ALL_REASONING_FSMS. Raises KeyError
- `map_reasoning_type(type_str)` -- 100+ aliases to canonical type (e.g., "math" → "simple_calculator"). Defaults to ANALYTICAL
- `get_available_reasoning_types()` -- Returns `{type_value: description}` dict

## Testing

```bash
pytest tests/test_fsm_llm_reasoning/  # 112 tests, 7 files
```

## Exceptions

```
FSMError
└── ReasoningEngineError(message, details=None)
    ├── ReasoningExecutionError(message, reasoning_type=None)
    └── ReasoningClassificationError
```
