# LLM-FSM Reasoning Engine

A sophisticated reasoning framework that enables Large Language Models to perform complex problem-solving through structured Finite State Machines.

## Features

- **🧠 Multiple Reasoning Strategies**: 9 different reasoning types including analytical, deductive, creative, and more
- **🔄 Loop Prevention**: Built-in retry limits and circuit breakers prevent infinite loops
- **📊 Context Management**: Automatic context pruning to prevent memory explosion
- **🎯 Smart Classification**: Intelligent problem analysis to select optimal reasoning approach
- **📝 Full Traceability**: Complete reasoning trace for transparency and debugging
- **⚡ Performance Optimized**: Efficient context handling and selective data passing

## Installation

```bash
pip install llm-fsm
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from llm_fsm_reasoning import ReasoningEngine

# Initialize engine
engine = ReasoningEngine(model="gpt-4o-mini")

# Solve a simple problem
solution, trace = engine.solve_problem("What is 15 + 27?")
print(solution)  # Output: 42

# Solve a complex problem
problem = "Design a scalable caching system for a social media platform"
solution, trace = engine.solve_problem(problem)
print(f"Solution: {solution}")
print(f"Reasoning steps: {trace['reasoning_trace']['total_steps']}")
```

### Command Line Usage

```bash
# Simple calculation
python -m llm_fsm_reasoning "What is 15 * 24?"

# Complex problem with specific reasoning type
python -m llm_fsm_reasoning "Explain photosynthesis" --type analytical

# With context
python -m llm_fsm_reasoning "Plan a trip" --context '{"budget": 2000, "days": 7}'

# Save results
python -m llm_fsm_reasoning "Design a database schema" --save results.json --output json

# List available reasoning types
python -m llm_fsm_reasoning --list-types
```

## Reasoning Types

| Type | Description | Best For |
|------|-------------|----------|
| `simple_calculator` | Direct arithmetic calculations | Math problems, calculations |
| `analytical` | Breaking down complex systems | System design, analysis |
| `deductive` | General to specific reasoning | Logical proofs, rule application |
| `inductive` | Pattern recognition | Data analysis, trend finding |
| `creative` | Novel solution generation | Innovation, brainstorming |
| `critical` | Argument evaluation | Review, validation, critique |
| `hybrid` | Combined approaches | Complex multi-faceted problems |
| `abductive` | Best explanation finding | Diagnostics, troubleshooting |
| `analogical` | Learning through comparison | Understanding via examples |

## Advanced Features

### Context Management

The engine automatically manages context size to prevent explosion:

```python
# Large context is automatically pruned
context = {
    "requirements": ["scalable", "secure", "fast"],
    "constraints": {"budget": 100000, "timeline": "6 months"},
    "existing_systems": [...],  # Large data
}

solution, trace = engine.solve_problem(
    "Integrate new payment system",
    initial_context=context
)
```

### Loop Prevention

Built-in retry limits prevent infinite validation loops:

```python
# Maximum 3 retries for validation
# Automatically stops if solution can't be improved
solution, trace = engine.solve_problem(
    "Solve this paradox: ..."
)

if trace['reasoning_trace']['final_confidence'] < 0.5:
    print("Low confidence solution - may need human review")
```

### Custom Problem Context

```python
from llm_fsm_reasoning import ProblemContext

problem = ProblemContext(
    problem_statement="Optimize our recommendation algorithm",
    domain="e-commerce",
    constraints=["real-time", "privacy-compliant"],
    initial_context={
        "current_performance": "200ms latency",
        "target_performance": "50ms latency"
    }
)

solution, trace = engine.solve_problem(
    problem.problem_statement,
    problem.initial_context
)
```

## Architecture

### Structured Flow

1. **Problem Analysis** - Understand the problem type and components
2. **Strategy Selection** - Choose optimal reasoning approach
3. **Reasoning Execution** - Apply selected strategy with sub-FSM
4. **Solution Synthesis** - Combine insights into solution
5. **Validation** - Check solution quality (with retry limit)
6. **Final Answer** - Present solution with confidence

### Key Improvements

- **Constants Consolidation**: All strings and magic values in `constants.py`
- **Standardized Handlers**: Consistent result extraction and merging
- **Context Pruning**: Automatic size management with configurable limits
- **Retry Limits**: Maximum 3 validation attempts before accepting result
- **Type Safety**: Pydantic models with validation

## Configuration

```python
from llm_fsm_reasoning import ReasoningEngine, Defaults

# Custom configuration
engine = ReasoningEngine(
    model="gpt-4",
    temperature=0.7,
    max_tokens=2000
)

# Modify defaults
Defaults.MAX_RETRIES = 5
Defaults.MAX_CONTEXT_SIZE = 15000
Defaults.CONTEXT_PRUNE_THRESHOLD = 12000
```

## Error Handling

The engine gracefully handles various error conditions:

```python
try:
    solution, trace = engine.solve_problem("Your problem")
except Exception as e:
    print(f"Error: {e}")
    # Engine provides meaningful error messages
```

Common errors:
- `Maximum retry attempts exceeded` - Validation loop limit hit
- `Context size exceeds maximum allowed` - Need context pruning
- `FSM definition not found` - Reasoning type unavailable

## Performance Tips

1. **Use Simple Calculator for Math**: Faster than analytical reasoning
2. **Provide Clear Problems**: Better problem statements → fewer steps
3. **Set Appropriate Context**: Only include relevant information
4. **Monitor Trace Length**: Long traces may indicate issues

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Reasoning Types

1. Add to `ReasoningType` enum in `constants.py`
2. Create FSM definition in `reasoning_modes.py`
3. Add mapping in `utilities.py`
4. Update context merging in `handlers.py`

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Support

- Documentation: [Full Docs](#)
- Issues: [GitHub Issues](#)
- Discussions: [GitHub Discussions](#)