# FSM-LLM Reasoning

> Structured multi-strategy reasoning engine powered by FSM-orchestrated LLM pipelines.

---

## Overview

`fsm_llm_reasoning` is an extension package that adds structured reasoning capabilities to FSM-LLM. It implements **9 distinct reasoning strategies**, each modeled as its own FSM, orchestrated by a meta-FSM that classifies problems and routes them to the appropriate strategy.

The engine automatically:
1. **Classifies** the problem to determine the best reasoning approach
2. **Executes** the selected strategy through a dedicated reasoning FSM
3. **Validates** the solution and retries if needed
4. **Synthesizes** the final answer with confidence scoring

## Installation

```bash
pip install fsm-llm[reasoning]
```

**Requirements**: Python 3.10+ | No additional dependencies beyond core `fsm-llm`.

## Quick Start

**1. Solve a problem**:

```python
from fsm_llm_reasoning import ReasoningEngine

engine = ReasoningEngine(model="gpt-4o-mini")
solution, trace = engine.solve_problem("What is 15% of 240?")
print(solution)
```

**2. With initial context**:

```python
solution, trace = engine.solve_problem(
    "Should we expand into the European market?",
    initial_context={
        "company_size": "mid-market",
        "current_revenue": "50M",
        "growth_target": "2x in 3 years",
    },
)
print(solution)
print(f"Reasoning types used: {trace['reasoning_types_used']}")
print(f"Confidence: {trace['final_confidence']}")
```

**3. Force a specific strategy**:

```python
from fsm_llm_reasoning import ReasoningType

solution, trace = engine.solve_problem(
    "All mammals are warm-blooded. Whales are mammals. What can we conclude?",
    initial_context={"reasoning_type_selected": ReasoningType.DEDUCTIVE},
)
```

**4. CLI usage**:

```bash
# Simple query
python -m fsm_llm_reasoning "What is the square root of 144?"

# With options
python -m fsm_llm_reasoning "Analyze this market trend" \
    --model gpt-4o \
    --output detailed \
    --save results.json

# List available reasoning types
python -m fsm_llm_reasoning --list-types
```

## Architecture

```
Problem Input
  │
  ▼
┌──────────────────────────────────────────┐
│  Orchestrator FSM (6 states)             │
│                                          │
│  PROBLEM_ANALYSIS                        │
│       │                                  │
│       ▼                                  │
│  STRATEGY_SELECTION ←── Classifier FSM   │
│       │                  (4 states)      │
│       ▼                                  │
│  EXECUTE_REASONING ←── Strategy FSM      │
│       │                 (1 of 9)         │
│       ▼                                  │
│  SYNTHESIZE_SOLUTION                     │
│       │                                  │
│       ▼                                  │
│  VALIDATE_REFINE ──→ retry? ──→ back     │
│       │                                  │
│       ▼                                  │
│  FINAL_ANSWER                            │
└──────────────────────────────────────────┘
  │
  ▼
Solution + Trace
```

### Three Layers of FSMs

| Layer | FSM | States | Purpose |
|-------|-----|--------|---------|
| Orchestrator | `orchestrator_fsm` | 6 | Overall flow control, retry logic, solution synthesis |
| Classifier | `classifier_fsm` | 4 | Problem analysis, domain detection, strategy recommendation |
| Strategy | 9 strategy FSMs | 3-4 each | Execute specific reasoning approach |

## Reasoning Strategies

| Type | Description | Best For |
|------|-------------|----------|
| `simple_calculator` | Direct arithmetic and calculations | Math problems, unit conversions |
| `analytical` | Breaking down complex systems | System analysis, root cause investigation |
| `deductive` | Deriving conclusions from premises | Logic puzzles, formal arguments |
| `inductive` | Finding patterns from examples | Trend analysis, generalization |
| `creative` | Generating novel solutions | Brainstorming, design problems |
| `critical` | Evaluating arguments and claims | Decision analysis, argument assessment |
| `hybrid` | Combining multiple approaches | Complex multi-faceted problems |
| `abductive` | Finding best explanations | Diagnosis, hypothesis generation |
| `analogical` | Learning through analogies | Novel domains, knowledge transfer |

## Key API Reference

### ReasoningEngine

The main entry point. Thread-safe — `solve_problem()` is serialized internally.

```python
from fsm_llm_reasoning import ReasoningEngine

engine = ReasoningEngine(model="gpt-4o-mini")

# Returns (solution_text, trace_info_dict)
solution, trace = engine.solve_problem(
    problem="Your problem here",
    initial_context={"key": "value"},  # optional
)
```

**Trace dict contains**:
- `steps`: List of FSM state transitions
- `reasoning_types_used`: Set of strategies applied
- `final_confidence`: Float 0.0-1.0
- `execution_time_seconds`: Total solve time

### Data Models

```python
from fsm_llm_reasoning import (
    ReasoningStep,      # Individual reasoning step with confidence
    ReasoningTrace,     # Full execution trace with metrics
    SolutionResult,     # Solution with validation and alternatives
    ProblemContext,      # Problem specification with constraints
    ValidationResult,   # Multi-check validation result
)
```

**SolutionResult** computed properties:
- `confidence_level` — LOW / MEDIUM / HIGH / VERY_HIGH
- `is_high_confidence` — bool (>= 0.7)
- `reasoning_depth` — simple / moderate / complex / highly_complex
- `is_validated` — whether validation was run
- `solution_quality_summary` — human-readable quality string

### Utility Functions

```python
from fsm_llm_reasoning import (
    ReasoningType,               # Enum of 9 reasoning types
    get_available_reasoning_types,  # Returns {type: description} dict
)

# List all strategies
for rtype, desc in get_available_reasoning_types().items():
    print(f"{rtype}: {desc}")
```

## Configuration

The engine respects these defaults (from `constants.py`):

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_RETRIES` | 3 | Max validation retry attempts |
| `MAX_TOTAL_ITERATIONS` | 50 | Max FSM iterations across all phases |
| `MAX_TRACE_STEPS` | 50 | Max recorded trace steps |
| `MIN_SOLUTION_LENGTH` | 20 | Minimum chars for valid solution |
| `CONTEXT_PRUNE_THRESHOLD` | 8000 | Context size before pruning (chars) |

## Handlers

Six handlers manage cross-cutting concerns:

| Handler | Timing | Purpose |
|---------|--------|---------|
| `OrchestratorProblemClassifier` | CONTEXT_UPDATE | Triggers classification when problem_type updates |
| `OrchestratorStrategyExecutor` | POST_TRANSITION | Prepares execution when entering EXECUTE_REASONING |
| `OrchestratorSolutionValidator` | CONTEXT_UPDATE | Validates proposed solutions |
| `ReasoningTracer` | POST_TRANSITION | Records state transitions for trace |
| `ContextPruner` | POST_PROCESSING | Prevents context explosion (>8000 chars) |
| `RetryLimiter` | CONTEXT_UPDATE | Enforces max retry attempts |

## Exception Hierarchy

```
FSMError
└── ReasoningEngineError
    ├── ReasoningExecutionError      # Strategy FSM execution failures
    └── ReasoningClassificationError # Problem classification failures
```

## License

GPL-3.0-or-later. See [LICENSE](../../LICENSE) for details.
