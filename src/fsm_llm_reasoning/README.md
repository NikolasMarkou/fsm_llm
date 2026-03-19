# FSM-LLM Reasoning Engine

A structured reasoning engine that enables Large Language Models to perform complex problem-solving through 9 specialized FSM-defined reasoning strategies. Part of the `fsm-llm` package.

## Features

- **9 Reasoning Strategies**: Analytical, Deductive, Inductive, Creative, Critical, Abductive, Analogical, Hybrid, and Simple Calculator — each implemented as its own FSM
- **Smart Classification**: FSM-based classifier automatically selects the most suitable reasoning strategy
- **Loop Prevention**: Built-in retry limits (`Defaults.MAX_RETRIES`) and circuit breakers in the orchestrator's `VALIDATE_REFINE` state
- **Context Management**: Automated context pruning (`Defaults.CONTEXT_PRUNE_THRESHOLD`, `Defaults.MAX_CONTEXT_SIZE`) via ContextManager and dedicated handlers
- **Full Traceability**: Complete reasoning trace with state transitions and context snapshots via `ReasoningTrace` and `ReasoningStep` models
- **CLI Interface**: Solve problems directly from the command line with `python -m fsm_llm_reasoning`

## Installation

```bash
pip install fsm-llm[reasoning]
```

## Quick Start

### Basic Usage

```python
from fsm_llm_reasoning import ReasoningEngine, ProblemContext

# Initialize engine (ensure OPENAI_API_KEY or relevant LLM provider key is set)
engine = ReasoningEngine(model="gpt-4o-mini")

# Solve a simple problem (auto-selects strategy)
solution, trace = engine.solve_problem("What is 15 + 27?")
print(f"Solution: {solution}")

# Solve a complex problem with structured context
problem = ProblemContext(
    problem_statement="Design a scalable caching system for a social media platform.",
    domain="technical",
    constraints=["Cost-effective", "Low-latency", "Highly available"]
)
solution, trace = engine.solve_problem(
    problem.problem_statement,
    initial_context=problem.model_dump()
)
print(f"Solution: {solution}")
print(f"Reasoning steps: {trace['reasoning_trace']['total_steps']}")
```

### Command Line Usage

```bash
# Simple calculation
python -m fsm_llm_reasoning "What is 15 * 24?"

# Specify a reasoning strategy
python -m fsm_llm_reasoning "Explain photosynthesis" --type analytical

# With initial context
python -m fsm_llm_reasoning "Plan a trip to Italy" --context '{"budget": 2000}'

# Save detailed results
python -m fsm_llm_reasoning "Design a database schema" --output detailed --save results.json

# List available reasoning types
python -m fsm_llm_reasoning --list-types
```

## Reasoning Types

| Type | Description | Best For |
|:-----|:-----------|:---------|
| `simple_calculator` | Direct arithmetic calculations | Math problems, basic calculations |
| `analytical` | Breaking down complex systems into components | System design, detailed analysis |
| `deductive` | Applying general principles to specific cases | Logical proofs, rule-based inference |
| `inductive` | Identifying patterns from specific observations | Data analysis, trend finding |
| `abductive` | Inferring the most plausible explanation | Diagnostics, troubleshooting |
| `analogical` | Transferring insights using analogies | Understanding new concepts |
| `creative` | Generating novel and innovative solutions | Brainstorming, innovation |
| `critical` | Evaluating arguments and evidence systematically | Review, validation, critique |
| `hybrid` | Combining multiple reasoning approaches | Multi-faceted problem-solving |

## Architecture

The engine uses a hierarchical FSM approach:

1. **Orchestrator FSM** — Controls the overall flow through states: `PROBLEM_ANALYSIS` → `STRATEGY_SELECTION` → `EXECUTE_REASONING` → `SYNTHESIZE_SOLUTION` → `VALIDATE_REFINE` → `FINAL_ANSWER`
2. **Classifier FSM** — Invoked during `STRATEGY_SELECTION` to analyze the problem and recommend the best `ReasoningType`
3. **Specialized FSMs** — One per reasoning type, pushed onto the FSM stack during `EXECUTE_REASONING`

### Key Design Principles

- **Constants Consolidation**: All key strings, defaults, and state names centralized in `constants.py`
- **Context Pruning**: Automated context size management via `ContextPruner` handler
- **Retry Limits**: `RetryLimiter` handler prevents infinite validation loops
- **Type Safety**: Pydantic models for all data contracts (`ProblemContext`, `SolutionResult`, `ReasoningTrace`)

## File Map

| File | Purpose |
|------|---------|
| `engine.py` | **ReasoningEngine** — main entry point, FSM loading, handler registration, `solve_problem()` |
| `reasoning_modes.py` | FSM definitions as Python dicts for all 9 strategies + orchestrator + classifier |
| `handlers.py` | ReasoningHandlers: validation, tracing (`ReasoningTracer`), context pruning (`ContextPruner`), retry limiting (`RetryLimiter`) |
| `definitions.py` | Pydantic models: ReasoningStep, ReasoningTrace, ValidationResult, SolutionResult, ProblemContext, ContextSnapshot |
| `constants.py` | **ReasoningType** enum, **ContextKeys** dataclass, OrchestratorStates, ClassifierStates, Defaults, ErrorMessages |
| `utilities.py` | `load_fsm_definition()`, `map_reasoning_type()`, `get_available_reasoning_types()` |
| `exceptions.py` | Exception hierarchy: ReasoningEngineError → ReasoningExecutionError, ReasoningClassificationError, ReasoningValidationError |
| `__main__.py` | CLI: `python -m fsm_llm_reasoning "problem"` with --type, --context, --output, --save, --list-types flags |
| `__version__.py` | Package version string |
| `__init__.py` | Public API exports — single `__all__` list |

## Configuration

```python
from fsm_llm_reasoning import ReasoningEngine

engine = ReasoningEngine(
    model="gpt-4-turbo",  # Any LiteLLM-supported model
    temperature=0.5,
    max_tokens=3000,
)
```

Default values are sourced from `constants.Defaults`.

## Error Handling

```python
from fsm_llm_reasoning import ReasoningEngineError, ReasoningExecutionError

try:
    solution, trace = engine.solve_problem("...")
except ReasoningExecutionError as e:
    print(f"Execution failed: {e}, reasoning_type: {e.reasoning_type}")
except ReasoningEngineError as e:
    print(f"Engine error: {e}")
```

### Exception Hierarchy

- `ReasoningEngineError` (base) → `ReasoningExecutionError`, `ReasoningClassificationError`, `ReasoningValidationError`

## API Reference

### ReasoningEngine

| Method | Description |
|--------|-------------|
| `ReasoningEngine(model=..., **kwargs)` | Initialize engine |
| `engine.solve_problem(statement, initial_context=None)` | Solve problem → `(solution_str, trace_dict)` |

### Key Models

| Class | Description |
|-------|-------------|
| `ProblemContext` | Structured problem input (statement, domain, constraints, initial_context) |
| `SolutionResult` | Final solution with confidence, reasoning summary, trace, validation |
| `ReasoningTrace` | Complete trace of reasoning steps |
| `ReasoningStep` | Single step with type, content, confidence, evidence |
| `ValidationResult` | Solution validation with checks, issues, recommendations |
| `ReasoningType` | Enum of 9 reasoning strategies |

### Utilities

| Function | Description |
|----------|-------------|
| `get_available_reasoning_types()` | List all reasoning types with descriptions |
| `map_reasoning_type(name)` | Map string/alias to ReasoningType enum |

## Adding New Reasoning Types

1. Add the type to `ReasoningType` enum in `constants.py`
2. Create FSM definition as Python dict in `reasoning_modes.py`, add to `ALL_REASONING_FSMS`
3. Update `ContextManager.merge_reasoning_results` in `handlers.py` for result mapping
4. Add aliases to `map_reasoning_type()` in `utilities.py`
5. Add description to `get_available_reasoning_types()` in `utilities.py`
6. Write tests

## Development

```bash
pytest tests/test_fsm_llm_reasoning/  # 101 tests across 6 test files
```

## License

GNU General Public License v3.0. See [LICENSE](../../LICENSE).
