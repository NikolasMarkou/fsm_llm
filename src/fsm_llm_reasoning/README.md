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

```python
from fsm_llm_reasoning import ReasoningEngine

engine = ReasoningEngine(model="gpt-4o-mini")
solution, trace = engine.solve_problem("What is 15% of 240?")
print(solution)
```

With initial context:

```python
solution, trace = engine.solve_problem(
    "Should we expand into the European market?",
    initial_context={
        "company_size": "mid-market",
        "current_revenue": "50M",
    },
)
print(f"Reasoning types used: {trace['reasoning_types_used']}")
```

Force a specific strategy:

```python
from fsm_llm_reasoning import ReasoningType

solution, trace = engine.solve_problem(
    "All mammals are warm-blooded. Whales are mammals. What can we conclude?",
    initial_context={"reasoning_type_selected": ReasoningType.DEDUCTIVE},
)
```

CLI usage:

```bash
python -m fsm_llm_reasoning "What is the square root of 144?"
python -m fsm_llm_reasoning "Analyze this trend" --model gpt-4o --save results.json
python -m fsm_llm_reasoning --list-types
```

## Architecture

```
Problem Input → Orchestrator FSM (6 states)
                  ├── PROBLEM_ANALYSIS
                  ├── STRATEGY_SELECTION ← Classifier FSM (4 states)
                  ├── EXECUTE_REASONING ← Strategy FSM (1 of 9, 3 states each)
                  ├── SYNTHESIZE_SOLUTION
                  ├── VALIDATE_REFINE → retry if needed
                  └── FINAL_ANSWER
              → Solution + Trace
```

## Reasoning Strategies

| Type | Description | Best For |
|------|-------------|----------|
| `simple_calculator` | Direct arithmetic and calculations | Math, unit conversions |
| `analytical` | Breaking down complex systems | System analysis, root cause |
| `deductive` | Deriving conclusions from premises | Logic puzzles, formal arguments |
| `inductive` | Finding patterns from examples | Trend analysis, generalization |
| `creative` | Generating novel solutions | Brainstorming, design |
| `critical` | Evaluating arguments and claims | Decision analysis, assessment |
| `hybrid` | Combining multiple approaches | Complex multi-faceted problems |
| `abductive` | Finding best explanations | Diagnosis, hypothesis generation |
| `analogical` | Learning through analogies | Novel domains, knowledge transfer |

## Key API Reference

### ReasoningEngine

Thread-safe — `solve_problem()` is serialized internally.

```python
engine = ReasoningEngine(model="gpt-4o-mini")
solution, trace = engine.solve_problem(problem="Your problem here", initial_context={})
```

**Trace dict contains**: `steps`, `reasoning_types_used`, `final_confidence`, `execution_time_seconds`.

### Data Models

```python
from fsm_llm_reasoning import (
    ReasoningStep,      # Individual step with confidence
    ReasoningTrace,     # Full execution trace
    SolutionResult,     # Solution with validation and alternatives
    ProblemContext,      # Problem specification with constraints
    ValidationResult,   # Multi-check validation result
)
```

**SolutionResult** computed properties: `confidence_level`, `is_high_confidence`, `reasoning_depth`, `is_validated`, `solution_quality_summary`.

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_RETRIES` | 3 | Max validation retry attempts |
| `MAX_TOTAL_ITERATIONS` | 50 | Max FSM iterations across all phases |
| `MAX_TRACE_STEPS` | 50 | Max recorded trace steps |
| `MIN_SOLUTION_LENGTH` | 20 | Minimum chars for valid solution |
| `CONTEXT_PRUNE_THRESHOLD` | 8000 | Context size before pruning (chars) |

## Handlers

| Handler | Timing | Purpose |
|---------|--------|---------|
| `OrchestratorProblemClassifier` | CONTEXT_UPDATE | Triggers classification |
| `OrchestratorStrategyExecutor` | POST_TRANSITION | Prepares execution |
| `OrchestratorSolutionValidator` | CONTEXT_UPDATE | Validates solutions |
| `ReasoningTracer` | POST_TRANSITION | Records state transitions |
| `ContextPruner` | POST_PROCESSING | Prevents context explosion |
| `RetryLimiter` | CONTEXT_UPDATE | Enforces max retries |

## Exception Hierarchy

```
FSMError
└── ReasoningEngineError
    ├── ReasoningExecutionError      # Strategy FSM execution failures
    └── ReasoningClassificationError # Problem classification failures
```

## License

GPL-3.0-or-later. See [LICENSE](../../LICENSE) for details.
