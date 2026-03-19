# FSM-LLM Workflow Engine

A workflow orchestration system built on top of FSM-LLM that enables automated state transitions, event-driven workflows, external system integration, parallel execution, and monitoring with error recovery. Part of the `fsm-llm` package.

## Features

- **Automated State Transitions**: Steps that automatically advance the workflow without user input
- **Event-Driven Workflows**: Wait for and respond to external events
- **External API Integration**: Built-in step type for calling external APIs with input/output mapping
- **LLM Processing Steps**: Integrate LLM-powered processing within workflows
- **Timers**: Schedule delayed transitions and timeouts
- **Parallel Execution**: Run multiple workflow steps concurrently
- **Python DSL**: Fluent API and factory functions for workflow definition
- **Validation**: Structural validation including reachability analysis and cycle detection

## Installation

```bash
pip install fsm-llm[workflows]
```

This installs additional dependencies: `networkx` (graph operations), `graphviz` (visualization), `aiofiles` (async file I/O).

## Quick Start

### Define a Workflow with DSL

```python
from fsm_llm_workflows import (
    create_workflow, auto_step, llm_step, condition_step, api_step
)

# Create a simple processing workflow
workflow = create_workflow("order_processor", initial_state="receive")

workflow.with_step(auto_step(
    "receive", "validate",
    action=lambda ctx: {"validated": True}
))

workflow.with_step(condition_step(
    "validate", "process", "reject",
    condition=lambda ctx: ctx.get("validated", False)
))

workflow.with_step(llm_step(
    "process", "complete",
    prompt_template="Summarize this order: {order_details}"
))

workflow.with_step(auto_step(
    "complete", None,  # Terminal state
    action=lambda ctx: {"status": "done"}
))

# Validate the workflow
workflow.validate()
```

### Use the WorkflowBuilder

```python
from fsm_llm_workflows import workflow_builder, auto_step

wf = (
    workflow_builder("my_workflow")
    .add_step(auto_step("start", "middle", action=do_first))
    .add_step(auto_step("middle", "end", action=do_second))
    .build()
)
```

### Factory Functions for Common Patterns

```python
from fsm_llm_workflows import linear_workflow, conditional_workflow, event_driven_workflow

# Sequential steps
wf = linear_workflow("pipeline", steps=[
    ("fetch", fetch_data),
    ("transform", transform_data),
    ("load", load_data),
])

# Branching workflow
wf = conditional_workflow("router", ...)

# Event-driven workflow
wf = event_driven_workflow("listener", ...)
```

### Run with WorkflowEngine

```python
from fsm_llm_workflows import WorkflowEngine

engine = WorkflowEngine()
engine.register_workflow(workflow)
instance = await engine.start_workflow("order_processor", initial_context={"order_id": "123"})
```

## Step Types

| Step | Purpose |
|------|---------|
| `AutoTransitionStep` | Automatically transitions to next state after executing an action |
| `APICallStep` | Calls external APIs with configurable input/output mapping |
| `ConditionStep` | Evaluates a condition to choose between two target states |
| `LLMProcessingStep` | Processes data using an LLM with a prompt template |
| `WaitForEventStep` | Pauses workflow until a specific external event is received |
| `TimerStep` | Waits for a specified duration before transitioning |
| `ParallelStep` | Executes multiple sub-steps concurrently |

## Workflow Status Lifecycle

```
PENDING → RUNNING → WAITING (for events/timers) → COMPLETED
                  ↘ FAILED
                  ↘ CANCELLED
```

## Handler Integration

Three handlers integrate workflows with the FSM-LLM handler system:

- **AutoTransitionHandler**: Detects and executes automatic transitions
- **EventHandler**: Processes external events and routes to waiting workflows
- **TimerHandler**: Manages workflow timers and triggers transitions on expiry

## WorkflowDefinition Validation

```python
workflow.validate()           # Full structural validation
workflow.has_cycles()          # Detect cycles in workflow graph
workflow.serialize()           # Serialize to dictionary
```

Validation checks: steps exist, state transitions reference valid states, all states reachable from initial, terminal states exist.

## Development

```bash
pytest tests/test_fsm_llm_workflows/  # Workflow tests (auto-skips if not installed)
```

## License

GNU General Public License v3.0. See [LICENSE](../../LICENSE).

---
