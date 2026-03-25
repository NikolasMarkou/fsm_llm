# fsm_llm_reasoning

Structured reasoning engine for FSM-LLM that orchestrates 9 reasoning strategies implemented as hierarchical finite state machines. Part of the FSM-LLM framework (v0.3.0).

## Features

- **9 reasoning strategies**: Simple Calculator, Analytical, Deductive, Inductive, Creative, Critical, Hybrid, Abductive, and Analogical -- each implemented as its own FSM
- **Hierarchical FSM architecture**: 3-tier system with orchestrator, classifier, and specialized reasoning FSMs
- **Automatic strategy selection**: Classifier FSM analyzes the problem domain and structure, then recommends the best reasoning approach
- **Context pruning**: Automated context size management prevents explosion during long reasoning chains (threshold at `CONTEXT_PRUNE_THRESHOLD`)
- **Loop prevention**: Configurable retry limits on the VALIDATE_REFINE -> EXECUTE_REASONING cycle via `RetryLimiter` handler
- **Retry limiting**: `MAX_RETRIES`, `MAX_SUB_FSM_ITERATIONS`, `MAX_CLASSIFICATION_ITERATIONS`, and `MAX_TOTAL_ITERATIONS` caps at multiple levels
- **Thread-safe**: `solve_problem()` serialized via internal lock per engine instance

## Installation

```bash
pip install fsm-llm[reasoning]
```

## Quick Start

```python
from fsm_llm_reasoning import ReasoningEngine

# Initialize engine (requires LLM provider key, e.g. OPENAI_API_KEY)
engine = ReasoningEngine(model="openai/gpt-4o-mini")

# Solve a problem -- auto-selects reasoning strategy
solution, trace_info = engine.solve_problem("What is the sum of 42 and 58?")
print(f"Solution: {solution}")
print(f"Steps: {trace_info['reasoning_trace']['total_steps']}")
```

The engine automatically classifies the problem, selects the appropriate reasoning strategy, executes it through a specialized FSM, validates the result, and returns the solution with a full reasoning trace.

### With Structured Context

```python
from fsm_llm_reasoning import ReasoningEngine, ProblemContext

engine = ReasoningEngine(model="openai/gpt-4o-mini")

problem = ProblemContext(
    problem_statement="Design a scalable caching system for a social media platform.",
    domain="technical",
    constraints=["Cost-effective", "Low-latency", "Highly available"]
)

solution, trace_info = engine.solve_problem(
    problem=problem.problem_statement,
    initial_context=problem.model_dump()
)
```

## Reasoning Strategies

| Type | Enum Value | Description |
|------|-----------|-------------|
| Simple Calculator | `simple_calculator` | Direct arithmetic calculations |
| Analytical | `analytical` | Breaking down complex problems into components |
| Deductive | `deductive` | Deriving specific conclusions from general principles |
| Inductive | `inductive` | Finding patterns from specific observations |
| Creative | `creative` | Generating novel solutions through divergent thinking |
| Critical | `critical` | Evaluating arguments and evidence |
| Hybrid | `hybrid` | Combining multiple reasoning approaches |
| Abductive | `abductive` | Finding the best explanation for observations |
| Analogical | `analogical` | Transferring insights through analogies |

## Architecture

The engine uses a 3-tier hierarchical FSM architecture:

### Tier 1: Orchestrator FSM

Controls the overall reasoning flow through 6 states:

```
PROBLEM_ANALYSIS -> STRATEGY_SELECTION -> EXECUTE_REASONING -> SYNTHESIZE_SOLUTION -> VALIDATE_REFINE -> FINAL_ANSWER
```

VALIDATE_REFINE can loop back to EXECUTE_REASONING if the solution fails validation (up to `MAX_RETRIES` times).

### Tier 2: Classifier FSM

Invoked during STRATEGY_SELECTION to analyze the problem and recommend the best `ReasoningType`. Flows through 4 states: ANALYZE_DOMAIN -> ANALYZE_STRUCTURE -> IDENTIFY_REASONING_NEEDS -> RECOMMEND_STRATEGY.

### Tier 3: Specialized Reasoning FSMs

One FSM per reasoning type, pushed onto the FSM stack during EXECUTE_REASONING via `push_fsm()`. Each specialized FSM guides the LLM through strategy-specific reasoning steps. When complete, results are merged back into the orchestrator context via `ContextManager.merge_reasoning_results()`.

### Full Flow

```
User Problem
  -> Orchestrator: PROBLEM_ANALYSIS (extract type, components, constraints)
  -> Orchestrator: STRATEGY_SELECTION (triggers classifier)
     -> Classifier FSM: analyze domain -> structure -> needs -> recommend
  -> Orchestrator: EXECUTE_REASONING (pushes specialized FSM)
     -> Specialized FSM: strategy-specific reasoning steps
  -> Orchestrator: SYNTHESIZE_SOLUTION (combine results)
  -> Orchestrator: VALIDATE_REFINE (check solution quality, retry if needed)
  -> Orchestrator: FINAL_ANSWER (extract and return solution)
```

## CLI Usage

```bash
# Solve a problem
python -m fsm_llm_reasoning "What is 2 + 2?"

# Force a specific reasoning type
python -m fsm_llm_reasoning "Design a recommendation system" --type analytical

# Provide initial context as JSON
python -m fsm_llm_reasoning "Explain why the sky is blue" --context '{"audience": "child"}'

# List available reasoning types
python -m fsm_llm_reasoning --list-types

# Detailed output with reasoning trace
python -m fsm_llm_reasoning "Solve this logic puzzle" --output detailed

# Save results to file
python -m fsm_llm_reasoning "Analyze this argument" --save results.json

# Specify model
python -m fsm_llm_reasoning "Complex problem" --model openai/gpt-4o

# Quiet mode (solution only, no formatting)
python -m fsm_llm_reasoning "Quick question" --quiet
```

### CLI Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--type TYPE` | `-t` | Force a specific reasoning type |
| `--context JSON` | `-c` | Initial context as JSON string |
| `--model MODEL` | `-m` | LLM model to use |
| `--output FORMAT` | `-o` | Output format: `text`, `json`, `detailed` |
| `--save FILE` | `-s` | Save results to file |
| `--verbose` | `-v` | Enable verbose logging |
| `--quiet` | `-q` | Minimal output (solution only) |
| `--list-types` | | List all reasoning types with descriptions |

## API Reference

### ReasoningEngine

Main entry point for structured reasoning.

```python
engine = ReasoningEngine(model="openai/gpt-4o-mini", **kwargs)
solution, trace_info = engine.solve_problem(problem, initial_context=None)
```

- `__init__(model, **kwargs)` -- Initialize with an LLM model name and optional kwargs passed to `fsm_llm.API`
- `solve_problem(problem, initial_context=None)` -- Returns `tuple[str, dict[str, Any]]` with the solution string and a trace info dictionary containing `reasoning_trace`, `summary`, `final_context`, and `all_responses`

### ReasoningType

`str` enum with 9 values: `SIMPLE_CALCULATOR`, `ANALYTICAL`, `DEDUCTIVE`, `INDUCTIVE`, `CREATIVE`, `CRITICAL`, `HYBRID`, `ABDUCTIVE`, `ANALOGICAL`.

### ReasoningStep

Pydantic model for a single reasoning step. Fields: `step_type` (ReasoningStepType enum), `content`, `confidence` (0.0-1.0), `evidence`, `context_keys_used`, `execution_time_ms`. Computed: `confidence_level`, `has_evidence`.

### ReasoningTrace

Pydantic model for the complete reasoning trace. Fields: `steps`, `reasoning_types_used`, `final_confidence`, `execution_time_seconds`, `context_evolution`, `decision_points`. Computed: `total_steps`, `unique_states_visited`, `reasoning_complexity` (simple/moderate/complex/highly_complex), `average_step_time`.

### SolutionResult

Pydantic model for structured solution output. Fields: `solution`, `confidence`, `reasoning_summary`, `trace` (ReasoningTrace), `execution_time_seconds`, `validation_result`, `alternative_solutions`, `key_insights`, `used_context_keys`. Computed: `confidence_level`, `is_high_confidence`, `reasoning_depth`, `has_alternatives`, `is_validated`, `solution_quality_summary`.

### ProblemContext

Pydantic model for structured problem input. Fields: `problem_statement`, `domain`, `constraints`, `initial_context`, `priority`, `expected_solution_type`, `user_preferences`. Computed: `has_constraints`, `context_size`, `is_high_priority`.

### ValidationResult

Pydantic model for solution validation. Fields: `is_valid`, `confidence`, `checks` (dict of check name to bool), `issues`, `recommendations`, `validation_criteria`. Computed: `passed_checks`, `total_checks`, `pass_rate`, `has_issues`, `validation_summary`.

### ReasoningClassificationResult

Pydantic model for problem classification output. Fields: `recommended_type`, `justification`, `domain`, `alternatives`, `confidence`, `complexity_assessment`, `domain_indicators`. Computed: `has_alternatives`, `classification_summary`.

### get_available_reasoning_types()

Returns `dict[str, str]` mapping reasoning type names to human-readable descriptions.

## File Map

| File | Purpose |
|------|---------|
| `engine.py` | `ReasoningEngine` -- main entry point, FSM loading, handler registration, `solve_problem()` orchestration |
| `reasoning_modes.py` | FSM definitions as Python dicts for all 9 strategies plus orchestrator and classifier (`ALL_REASONING_FSMS` dict) |
| `handlers.py` | `ReasoningHandlers` (validation, tracing, context pruning), `ContextManager` (context extraction, result merging), `OutputFormatter` (solution extraction) |
| `definitions.py` | Pydantic models: `ReasoningStep`, `ReasoningTrace`, `ValidationResult`, `SolutionResult`, `ProblemContext`, `ReasoningClassificationResult` |
| `constants.py` | `ReasoningType` enum, `ContextKeys` class, `OrchestratorStates`, `ClassifierStates`, `Defaults`, `HandlerNames`, `ErrorMessages`, `LogMessages` |
| `utilities.py` | `load_fsm_definition()`, `map_reasoning_type()` (with alias support), `get_available_reasoning_types()` |
| `exceptions.py` | `ReasoningEngineError` -> `ReasoningExecutionError`, `ReasoningClassificationError` |
| `__main__.py` | CLI entry point with argument parsing, output formatting (text/json/detailed), and file saving |
| `__init__.py` | Public API exports (12 symbols in `__all__`) |
| `__version__.py` | Package version (imported from `fsm_llm.__version__`) |

## Examples

The `examples/reasoning/math_tutor/` directory contains a working example using the reasoning engine as a math tutor:

```bash
python examples/reasoning/math_tutor/run.py
```

## Integration

The reasoning engine integrates with the agents package via `ReasoningReactAgent` (`fsm_llm_agents.reasoning_react`), which combines ReAct-style tool use with structured reasoning through FSM stacking. Requires both extras:

```bash
pip install fsm-llm[reasoning,agents]
```

See also `examples/agents/reasoning_stacking/` and `examples/agents/reasoning_tool/` for integration examples.

## Development

### Testing

112 tests across 6 test files:

```bash
pytest tests/test_fsm_llm_reasoning/ -v
```

| Test File | Coverage |
|-----------|----------|
| `test_engine.py` | ReasoningEngine initialization, solve_problem, classification, execution |
| `test_handlers.py` | Validation, tracing, context pruning, result merging, output formatting |
| `test_definitions.py` | Pydantic model validation, computed fields, serialization |
| `test_constants.py` | Enum values, ContextKeys, defaults |
| `test_exceptions.py` | Exception hierarchy, error details |
| `test_audit_fixes.py` | Regression tests for audit-identified issues |

### Error Handling

```python
from fsm_llm_reasoning import (
    ReasoningEngineError,
    ReasoningExecutionError,
    ReasoningClassificationError,
)

try:
    solution, trace = engine.solve_problem("...")
except ReasoningExecutionError as e:
    print(f"Execution failed: {e}, type: {e.reasoning_type}")
except ReasoningClassificationError as e:
    print(f"Classification failed: {e}")
except ReasoningEngineError as e:
    print(f"Engine error: {e}")
```

### Adding a New Reasoning Type

1. Add the type to `ReasoningType` enum in `constants.py`
2. Add relevant `ContextKeys` constants in `constants.py`
3. Create FSM definition dict in `reasoning_modes.py` and add to `ALL_REASONING_FSMS`
4. Add result merging logic in `ContextManager.merge_reasoning_results()` in `handlers.py`
5. Add aliases in `map_reasoning_type()` and description in `get_available_reasoning_types()` in `utilities.py`
6. Write tests

## License

GNU General Public License v3.0. See [LICENSE](../../LICENSE).
