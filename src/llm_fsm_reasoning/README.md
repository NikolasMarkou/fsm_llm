# Reasoning Engine for LLM-FSM

A structured reasoning framework that enables language models to perform complex reasoning by breaking down thought processes into manageable states.

## Features

- **Multiple Reasoning Strategies**: Analytical, Deductive, Inductive, Creative, Critical, and Hybrid reasoning
- **Intelligent Classification**: FSM-based problem analysis to select optimal reasoning approach
- **Modular Design**: Easy to extend with new reasoning patterns and domain-specific classifiers
- **Full Traceability**: Complete reasoning trace for transparency and debugging
- **Type-Safe**: Built with Pydantic models for robust data validation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from llm_fsm_reasoning import ReasoningEngine

# Initialize engine
engine = ReasoningEngine(model="gpt-4o-mini")

# Solve a problem
problem = "Design a sustainable water purification system for remote villages"
solution, trace = engine.solve_problem(problem)

print(f"Solution: {solution}")
print(f"Reasoning steps: {trace['reasoning_trace']['total_steps']}")
```

## Project Structure

```
llm_fsm_reasoning/
├── engine.py       # Main engine implementation
├── constants.py    # Constants and enumerations
├── models.py       # Pydantic data models
├── handlers.py     # Handler implementations
├── utilities.py        # Utility functions
└── fsms/          # FSM definition JSON files
    ├── orchestrator.json
    ├── classifier.json
    ├── analytical.json
    ├── deductive.json
    ├── inductive.json
    ├── creative.json
    ├── critical.json
    └── hybrid.json
```

## Reasoning Types

### 1. Analytical Reasoning
Breaks down complex problems into components for systematic analysis.

### 2. Deductive Reasoning
Derives specific conclusions from general principles (top-down).

### 3. Inductive Reasoning
Forms general principles from specific observations (bottom-up).

### 4. Creative Reasoning
Generates novel solutions through divergent thinking.

### 5. Critical Reasoning
Evaluates arguments and evidence for validity.

### 6. Hybrid Reasoning
Combines multiple reasoning approaches for complex problems.

## Extending the Engine

### Add a Custom Reasoning Pattern

1. Create a new FSM JSON file in `fsms/`:
```json
{
  "name": "custom_reasoning",
  "initial_state": "start",
  "states": {
    "start": {
      "id": "start",
      "purpose": "Begin custom reasoning",
      "transitions": [...]
    }
  }
}
```

2. Add the reasoning type to constants:
```python
class ReasoningType(Enum):
    CUSTOM = "custom"
```

3. Load the FSM in the engine:
```python
ReasoningType.CUSTOM: load_fsm_definition("custom")
```

### Add Domain-Specific Classification

```python
# Create a domain-specific classifier FSM
medical_classifier = {
    "name": "medical_classifier",
    "states": {...}
}

# Register it with the engine
engine.add_custom_classification_strategy("medical", medical_classifier)
```

## Advanced Usage

### With Initial Context

```python
context = {
    "domain": "healthcare",
    "constraints": ["HIPAA compliant", "cost < $1M"],
    "existing_systems": ["EHR", "billing"]
}

solution, trace = engine.solve_problem(
    "Integrate AI diagnostics into our hospital system",
    initial_context=context
)
```

### Analyzing Reasoning Traces

```python
# Get detailed trace information
trace_info = trace['reasoning_trace']

# View state transitions
for step in trace_info['steps']:
    print(f"{step['from']} → {step['to']}")
    
# Check reasoning confidence
print(f"Confidence: {trace_info['final_confidence']}")
```

## Performance Considerations

- Each reasoning step involves an LLM call
- Complex problems may require 10-20+ API calls
- Use caching for repeated problem types
- Consider batch processing for multiple problems
