# FSM-LLM Reasoning Engine

A sophisticated reasoning framework that enables Large Language Models to perform complex problem-solving through structured Finite State Machines (FSMs). This engine is an integral part of the `fsm-llm` package.

## Features

- **Multiple Reasoning Strategies**: 9 distinct FSM-defined reasoning types: Analytical, Deductive, Inductive, Creative, Critical, Abductive, Analogical, Hybrid, and Simple Calculator
- **Loop Prevention**: Built-in retry limits (`Defaults.MAX_RETRIES`) and circuit breakers in the orchestrator's `VALIDATE_REFINE` state
- **Context Management**: Automated context pruning (`Defaults.CONTEXT_PRUNE_THRESHOLD`, `Defaults.MAX_CONTEXT_SIZE`) via ContextManager and dedicated handlers
- **Smart Classification**: FSM-based problem classifier selects the most suitable reasoning strategy automatically
- **Full Traceability**: Complete reasoning trace with state transitions and context snapshots via `ReasoningTrace` and `ReasoningStep` models
- **Performance Optimized**: Efficient context handling and selective data passing between FSM layers

## Installation

The FSM-LLM Reasoning Engine is included with the main `fsm-llm` package.

```bash
pip install fsm-llm
```
(Development dependencies, if needed, are in `requirements.txt` at the project root.)

## Quick Start

### Basic Usage

```python
from fsm_llm_reasoning import ReasoningEngine, ProblemContext

# Initialize engine (ensure OPENAI_API_KEY or relevant LLM provider key is set)
# By default, uses model defined in fsm_llm_reasoning.constants.Defaults.MODEL
engine = ReasoningEngine(model="gpt-4o-mini") # Or your preferred model

# Solve a simple problem
solution_simple, trace_simple = engine.solve_problem("What is 15 + 27?")
print(f"Simple Solution: {solution_simple}")
# Expected output: Simple Solution: 42

# Solve a complex problem using ProblemContext
problem_complex = ProblemContext(
    problem_statement="Design a scalable caching system for a social media platform with high read traffic.",
    domain="technical",
    constraints=["Cost-effective", "Low-latency", "Highly available"]
)
solution_complex, trace_complex = engine.solve_problem(
    problem_complex.problem_statement,
    initial_context=problem_complex.model_dump() # Pass context as a dictionary
)
print(f"\nComplex Solution: {solution_complex}")
print(f"Reasoning steps: {trace_complex['reasoning_trace']['total_steps']}")
```

### Command Line Usage

The reasoning engine can also be invoked via the command line:

```bash
# Simple calculation
python -m fsm_llm_reasoning "What is 15 * 24?"

# Complex problem with a specific reasoning type
python -m fsm_llm_reasoning "Explain the process of photosynthesis to a high school student" --type analytical

# Using initial context
python -m fsm_llm_reasoning "Plan a 7-day trip to Italy" --context '{"budget": 2000, "interests": ["history", "food"]}'

# Save detailed results to a JSON file
python -m fsm_llm_reasoning "Design a database schema for an e-commerce store" --output detailed --save results.json

# List available reasoning types
python -m fsm_llm_reasoning --list-types
```
*(Ensure your LLM API keys are set as environment variables, e.g., `OPENAI_API_KEY`)*

## Reasoning Types

The engine supports various FSM-defined reasoning strategies, each tailored for different problem types:

| Type                | Description                                                | Best For                                  |
| :------------------ | :--------------------------------------------------------- | :---------------------------------------- |
| `simple_calculator` | Direct arithmetic calculations                             | Math problems, basic calculations         |
| `analytical`        | Breaking down complex systems into components              | System design, detailed analysis          |
| `deductive`         | Applying general principles to specific cases              | Logical proofs, rule-based inference      |
| `inductive`         | Identifying patterns from specific observations            | Data analysis, trend finding, hypothesis  |
| `abductive`         | Inferring the most plausible explanation for observations  | Diagnostics, troubleshooting, explanation |
| `analogical`        | Transferring insights and solutions using analogies        | Understanding new concepts, creative links|
| `creative`          | Generating novel and innovative solutions                  | Brainstorming, innovation challenges      |
| `critical`          | Evaluating arguments, claims, and evidence systematically  | Review, validation, critique of ideas   |
| `hybrid`            | Combining multiple reasoning approaches for complex tasks  | Multi-faceted, complex problem-solving    |

These reasoning strategies are implemented as individual FSMs, defined as Python dictionaries in `fsm_llm_reasoning.reasoning_modes.py`.

## Advanced Features

### Context Management
The engine automatically manages the size of the context passed between FSM states and to the LLM.
-   Uses `Defaults.MAX_CONTEXT_SIZE` and `Defaults.CONTEXT_PRUNE_THRESHOLD` from `constants.py`.
-   A dedicated context pruning handler (`ContextPruner`) and `ContextManager` help in selectively reducing context size while preserving critical information.

```python
from fsm_llm_reasoning import ReasoningEngine, ProblemContext

engine = ReasoningEngine()
# Large context example
context = ProblemContext(
    problem_statement="Integrate a new payment gateway into an existing e-commerce platform.",
    constraints=["PCI DSS compliant", "Supports multiple currencies", "Low transaction fees"],
    initial_context={
        "existing_platform_tech_stack": ["Python/Django", "PostgreSQL", "React"],
        "current_payment_gateways": ["Stripe (to be replaced)", "PayPal"],
        "user_base_size": 100000,
        "transaction_volume_per_day": 5000,
        "project_timeline": "3 months",
        "team_expertise": {"Python": "High", "React": "Medium", "Payment_Integration": "Low"}
    }
)
# The engine will automatically prune 'initial_context' if it exceeds thresholds.
solution, trace = engine.solve_problem(
    context.problem_statement,
    initial_context=context.model_dump()
)
```

### Loop Prevention
The main orchestrator FSM includes a `VALIDATE_REFINE` state. If a solution fails validation, the engine can retry the reasoning process.
-   This loop is limited by `Defaults.MAX_RETRIES` (from `constants.py`) to prevent infinite iterations.
-   The `RetryLimiter` handler manages this count.

```python
# Example of a problem that might require retries or hit the limit:
# solution, trace = engine.solve_problem("Solve this seemingly paradoxical logic puzzle: ...")
# if trace['reasoning_trace']['final_confidence'] < 0.5:
#     print("Low confidence solution - may need human review or hit retry limit.")
```

### Custom Problem Context
Use the `ProblemContext` Pydantic model (from `definitions.py`) for structured problem input:

```python
from fsm_llm_reasoning import ReasoningEngine, ProblemContext

engine = ReasoningEngine()
problem = ProblemContext(
    problem_statement="Optimize our company's supply chain for perishable goods.",
    domain="business",
    constraints=["Reduce spoilage by 20%", "Maintain delivery times", "Budget under $50k"],
    initial_context={
        "current_spoilage_rate": "15%",
        "average_delivery_time": "48 hours",
        "relevant_data_sources": ["inventory_levels.csv", "shipment_logs.xlsx", "weather_api_feed"]
    }
)
solution, trace = engine.solve_problem(
    problem.problem_statement,
    initial_context=problem.model_dump()
)
```

## Architecture

The reasoning engine employs a hierarchical FSM approach:

1.  **Main Orchestrator (`orchestrator_fsm`)**: This FSM controls the overall problem-solving flow. Its states (defined in `OrchestratorStates` from `constants.py`) include:
    *   `PROBLEM_ANALYSIS`: Understands the problem statement.
    *   `STRATEGY_SELECTION`: Chooses the best reasoning strategy. This step internally uses the `Classifier FSM`.
    *   `EXECUTE_REASONING`: Pushes a specialized reasoning FSM (e.g., `analytical_fsm`, `creative_fsm`) onto the FSM stack to perform the core reasoning.
    *   `SYNTHESIZE_SOLUTION`: Gathers results from the specialized FSM.
    *   `VALIDATE_REFINE`: Validates the solution. If invalid and retries are available, it can loop back to `EXECUTE_REASONING` with refined context.
    *   `FINAL_ANSWER`: Presents the final solution.

2.  **Classifier FSM (`classifier_fsm`)**: Invoked by the orchestrator during `STRATEGY_SELECTION`. This FSM analyzes the problem's domain, structure, and requirements to recommend the most suitable `ReasoningType`. Its states are defined in `ClassifierStates` from `constants.py`.

3.  **Specialized Reasoning FSMs**: Individual FSMs for each `ReasoningType` (e.g., `analytical_fsm`, `deductive_fsm`). These are defined as Python dictionaries in `fsm_llm_reasoning.reasoning_modes.py` and perform the detailed, step-by-step reasoning for their specific strategy.

This layered architecture allows for complex, adaptable, and robust problem-solving.

### Key Design Principles & Improvements

-   **Constants Consolidation**: All key strings, default values (e.g., `Defaults.MODEL`, `Defaults.MAX_RETRIES`), and FSM state names are centralized in `fsm_llm_reasoning.constants`.
-   **Standardized Handlers**: `fsm_llm_reasoning.handlers` provides consistent mechanisms for common tasks like trace updates (`ReasoningTracer`), context pruning (`ContextPruner`), retry logic (`RetryLimiter`), and result merging.
-   **Context Pruning**: Automated management of context size to ensure stability and performance.
-   **Retry Limits**: Built-in limits for validation loops prevent infinite processing and ensure termination.
-   **Type Safety**: Extensive use of Pydantic models (`fsm_llm_reasoning.definitions`) for data validation and clear data contracts (e.g., `ProblemContext`, `SolutionResult`, `ReasoningTrace`).

## Configuration

The `ReasoningEngine` can be configured at initialization. Default values are sourced from `fsm_llm_reasoning.constants.Defaults`.

```python
from fsm_llm_reasoning import ReasoningEngine, constants

# Custom configuration
engine = ReasoningEngine(
    model="gpt-4-turbo", # Specify a different LLM
    temperature=0.5,
    max_tokens=3000
)

# You can also modify defaults globally (use with caution)
# constants.Defaults.MAX_RETRIES = 5
# constants.Defaults.MAX_CONTEXT_SIZE = 15000
# constants.Defaults.CONTEXT_PRUNE_THRESHOLD = 12000
```

## Error Handling

The engine is designed to handle various error conditions gracefully. Standardized error messages are defined in `ErrorMessages` within `constants.py`.

```python
try:
    solution, trace = engine.solve_problem("An ill-defined or impossible problem statement...")
except Exception as e:
    print(f"Reasoning Engine Error: {e}")
    # Engine methods will raise exceptions with meaningful messages for common issues.
```

Common errors include:
-   `Maximum retry attempts exceeded` (if validation fails repeatedly).
-   `Context size exceeds maximum allowed` (if pruning fails to reduce context sufficiently).
-   `FSM definition not found` (if a specified reasoning type's FSM is missing).

## Performance Tips

1.  **Use `simple_calculator` for Math**: For straightforward arithmetic, this dedicated FSM is faster than general-purpose reasoning like `analytical`.
2.  **Provide Clear Problems**: Well-defined problem statements and relevant initial context lead to more efficient reasoning and fewer steps.
3.  **Set Appropriate Context**: Include only necessary information in the `initial_context` to reduce processing load and token usage.
4.  **Monitor Trace Length**: The `ReasoningTrace` can become large. Very long traces might indicate inefficient reasoning or a problem too complex for the current configuration.

## API Reference

### ReasoningEngine

| Method | Description |
|--------|-------------|
| `ReasoningEngine(model=..., temperature=..., max_tokens=...)` | Initialize engine |
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

## Development

### Running Tests

```bash
pytest tests/test_fsm_llm_reasoning/
```

### Adding New Reasoning Types

1. Add the type to `ReasoningType` enum in `constants.py`
2. Create FSM definition as Python dict in `reasoning_modes.py`, add to `ALL_REASONING_FSMS`
3. Update `ContextManager.merge_reasoning_results` in `handlers.py` for result mapping
4. Add aliases to `map_reasoning_type()` in `utilities.py`
5. Add description to `get_available_reasoning_types()` in `utilities.py`
6. Write tests

## License

GNU General Public License v3.0. See [LICENSE](../../LICENSE).

---
