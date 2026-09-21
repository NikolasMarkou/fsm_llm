# fsm_llm_reasoning

Path: `src/fsm_llm_reasoning`
Purpose: Multi-strategy reasoning engine: an orchestrator FSM classifies a problem, pushes one of 9 strategy FSMs, validates the solution with retries, and returns `(solution, trace_info)`.

## Scope

Pure FSM-LLM application code; no third-party deps beyond core `fsm_llm` (extra `reasoning` installs nothing). Version comes from `fsm_llm.__version__` (0.8.0). All FSM definitions are Python dicts in `reasoning_modes.py`, not JSON files. Consumed by `fsm_llm_agents.reasoning_react` (ReasoningReactAgent) and by examples under `examples/reasoning/`.

## Architecture

```mermaid
sequenceDiagram
    participant C as caller
    participant E as ReasoningEngine
    participant O as orchestrator API
    participant K as classifier API
    C->>E: solve_problem(problem, ctx)
    E->>O: start_conversation({problem_statement, reasoning_trace: [], retry_count: 0, ...ctx})
    loop until ended or 50 iterations
        E->>O: get_data
        alt reasoning_fsm_to_push set
            E->>O: update_context(flag=None); push_fsm(fsm, inherit_context=False, context_to_pass=subset)
            loop up to 30
                E->>O: converse("Continue reasoning.")
            end
            E->>O: pop_fsm(context_to_return=merge_reasoning_results(...), UPDATE)
        else
            E->>O: converse("Continue reasoning: <json context>")
        end
    end
    Note over O,K: CONTEXT_UPDATE on problem_type runs classifier FSM (<=10 turns)
    E->>O: get_data, end_conversation
    E-->>C: (solution, {reasoning_trace, summary, final_context, all_responses})
```

Handlers registered on the orchestrator (`_register_handlers`):

| Name | Timing | Trigger | Action |
| --- | --- | --- | --- |
| `OrchestratorProblemClassifier` | CONTEXT_UPDATE | `problem_type` updated | `_classify_problem` runs classifier FSM, returns `classified_problem_type`, justification, domain, alternatives |
| `OrchestratorStrategyExecutor` | state entry | `execute_reasoning` | `_prepare_reasoning_execution` sets `reasoning_fsm_to_push`, `reasoning_type_selected` |
| `OrchestratorSolutionValidator` | CONTEXT_UPDATE | `proposed_solution` updated | `ReasoningHandlers.validate_solution` |
| `ContextPruner` | PRE_TRANSITION | always | `ReasoningHandlers.prune_context` |
| `ReasoningTracer` | POST_TRANSITION | always (also on classifier) | `ReasoningHandlers.update_reasoning_trace` |
| `RetryLimiter` | state entry | `validate_refine` | `_check_retry_limit` sets `max_retries_reached` |

## Key files

| File | Role | Notes |
| --- | --- | --- |
| `engine.py` | `ReasoningEngine` | `_solve_lock` serializes `solve_problem` |
| `reasoning_modes.py` | `ALL_REASONING_FSMS` | keys: `orchestrator`, `classifier`, 9 strategy values |
| `handlers.py` | `ReasoningHandlers`, `ContextManager`, `OutputFormatter` | all static methods |
| `definitions.py` | Pydantic models on `TimestampedModel` | computed fields |
| `constants.py` | `ReasoningType`, `OrchestratorStates`, `ClassifierStates`, `ContextKeys`, `HandlerNames`, `Defaults`, `ErrorMessages`, `LogMessages` | state names double as dict keys in `reasoning_modes.py` |
| `utilities.py` | `load_fsm_definition`, `map_reasoning_type`, `get_available_reasoning_types` | |
| `__main__.py` | CLI | `python -m fsm_llm_reasoning` (no console script) |

## Public interface

Exports: `ReasoningEngine`, `ReasoningType`, `ReasoningStep`, `ReasoningTrace`, `ValidationResult`, `ReasoningClassificationResult`, `ProblemContext`, `SolutionResult`, `ReasoningEngineError`, `ReasoningExecutionError`, `ReasoningClassificationError`, `get_available_reasoning_types`, `__version__`.

- `ReasoningEngine(model=Defaults.MODEL (fsm_llm DEFAULT_LLM_MODEL), **api_kwargs)`: builds `orchestrator` and `classifier` via `API.from_definition(..., model=model, **api_kwargs)`.
- `solve_problem(problem: str, initial_context: dict | None = None) -> tuple[str, dict]`; dict keys `reasoning_trace` (ReasoningTrace dump), `summary` (str), `final_context`, `all_responses`. Raises `ReasoningExecutionError` (wrapping any loop failure, after ending the conversation).
- `load_fsm_definition(name) -> dict` (deep copy; `KeyError` if unknown). `map_reasoning_type(s) -> str` (aliases like `math`, `logic`, `explain`, `analogy`; unknown -> `analytical` with WARNING). `get_available_reasoning_types() -> {value: description}`.
- CLI: `python -m fsm_llm_reasoning PROBLEM [--type {9 values}] [--context JSON] [--model M] [--output text|json|detailed] [--save FILE] [--verbose|-v] [--quiet] [--list-types] [--version]`. `--type` sets `ContextKeys.PREFERRED_REASONING_TYPE`. Returns exit code 0/1.

## Data shapes

- `ReasoningType` values: `simple_calculator`, `analytical`, `deductive`, `inductive`, `abductive`, `analogical`, `creative`, `critical`, `hybrid`.
- Orchestrator states: `problem_analysis`, `strategy_selection`, `execute_reasoning`, `synthesize_solution`, `validate_refine`, `final_answer`. Classifier states: `analyze_domain`, `analyze_structure`, `identify_reasoning_needs`, `recommend_strategy`. Each strategy FSM has 3 states.
- Models: `ReasoningStep{step_type: ReasoningStepType, content, confidence 0-1, evidence, context_keys_used}`, `ReasoningTrace{steps, reasoning_types_used: set, final_confidence, execution_time_seconds}` (computed `total_steps`, `unique_states_visited`, `reasoning_complexity`, `average_step_time`), `ValidationResult{is_valid, confidence, checks, issues, recommendations}` (computed `pass_rate` etc.), `ReasoningClassificationResult{recommended_type, justification, domain, alternatives}`, `ProblemContext`, `SolutionResult`. Enums `ConfidenceLevel`, `ReasoningStepType` (10), `ProblemDomain` (8).
- `Defaults`: `TEMPERATURE 0.7`, `MAX_TOKENS 2000`, `MAX_RETRIES 3`, `MAX_CONTEXT_SIZE 10000`, `MAX_TRACE_STEPS 50`, `CONTEXT_PRUNE_THRESHOLD 8000`, `MIN_SOLUTION_LENGTH 20`, `PRUNE_LIST_MAX_LENGTH 10`, `PRUNE_STRING_MAX_LENGTH 1000`, `MAX_SUB_FSM_ITERATIONS 30`, `MAX_CLASSIFICATION_ITERATIONS 10`, `MAX_TOTAL_ITERATIONS 50`. `TEMPERATURE`/`MAX_TOKENS` are not passed to `API` by the engine.

## Invariants and constraints

- Strategy choice order in `_prepare_reasoning_execution`: `reasoning_strategy == "direct computation"` -> `simple_calculator`; else mapped `classified_problem_type`; else mapped `reasoning_strategy`; else `analytical`. A valid `preferred_reasoning_type` then overrides. A missing strategy FSM falls back ONLY to `analytical` (never another type); none -> `ReasoningExecutionError`.
- `validate_solution`: checks `has_solution`, `has_insights` (auto-true for simple problems), `sufficient_detail` (>20 chars unless simple), `addresses_problem` (content-word overlap after stop-word removal; skipped for simple). On failure increments `retry_count` while `< MAX_RETRIES`; `max_retries_reached = retry_count >= 3`.
- Sub-FSM loop: stops on stack depth <= 1 or conversation end; on 30 iterations the sub-FSM context is captured, then force-popped once (`force_popped` guards against a double pop), and results are applied via `update_context`.
- `extract_final_solution` priority: `final_solution`, `proposed_solution`, `calculation_result`, `integrated_analysis`, then per-strategy result keys (deductive conclusion, inductive hypothesis, best creative solution, critical assessment, final hybrid solution, best explanation, analogical solution).
- State name constants and `reasoning_modes.py` dict keys must stay in sync.
- The orchestrator conversation is always ended in a `finally`, including after a successful solve.

## Dependencies

- `fsm_llm`: `API`, `HandlerTiming`, `ContextMergeStrategy`, `FSMError`, `logging.logger`, `constants.DEFAULT_LLM_MODEL`. Relies on `get_data` returning cached data for ended conversations.
- pydantic v2 for models.

## Failure modes

- `ReasoningClassificationError`: classifier FSM did not end within 10 turns or failed; raised from a CONTEXT_UPDATE handler, so it surfaces through `converse` and is wrapped as `ReasoningExecutionError` by `solve_problem`.
- Exceeding 50 orchestrator iterations logs an error and returns whatever solution is in context (no exception).
- Exception hierarchy: `FSMError` -> `ReasoningEngineError(message, details)` -> `ReasoningExecutionError(message, reasoning_type)`, `ReasoningClassificationError`.

## Working here

- New strategy: add an FSM dict in `reasoning_modes.py`, register it in `ALL_REASONING_FSMS`, add a `ReasoningType` value, aliases in `map_reasoning_type`, a description in `get_available_reasoning_types`, result mapping in `ContextManager.merge_reasoning_results`, and its result key in `OutputFormatter.extract_final_solution`.
- Use `ContextKeys` constants, never raw strings.
- Tests: `pytest tests/test_fsm_llm_reasoning/` (files `test_audit_fixes.py`, `test_constants.py`, `test_definitions.py`, `test_engine.py`, `test_exceptions.py`, `test_handlers.py`). Example: `examples/reasoning/math_tutor`.
