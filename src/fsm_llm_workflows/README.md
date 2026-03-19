# FSM-LLM Workflow Engine

An event-driven workflow orchestration engine built on FSM-LLM that enables automated state transitions, external API integration, LLM processing, timers, parallel execution, and embedded FSM conversations. Part of the `fsm-llm` package.

## Features

- **9 Step Types**: AutoTransition, APICall, Condition, LLMProcessing, WaitForEvent, Timer, Parallel, and ConversationStep
- **ConversationStep**: Embed a full FSM-LLM conversation within a workflow step with context mapping
- **Python DSL**: Fluent API and factory functions for workflow definition
- **Event-Driven**: Wait for and respond to external events
- **Parallel Execution**: Run multiple workflow steps concurrently
- **Timers**: Schedule delayed transitions and timeouts
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

| Step | DSL Function | Purpose |
|------|-------------|---------|
| `AutoTransitionStep` | `auto_step()` | Automatically transitions to next state after executing an action |
| `APICallStep` | `api_step()` | Calls external APIs with configurable input/output mapping |
| `ConditionStep` | `condition_step()` | Evaluates a condition to choose between two target states |
| `LLMProcessingStep` | `llm_step()` | Processes data using an LLM with a prompt template |
| `WaitForEventStep` | `wait_event_step()` | Pauses workflow until a specific external event is received |
| `TimerStep` | `timer_step()` | Waits for a specified duration before transitioning |
| `ParallelStep` | `parallel_step()` | Executes multiple sub-steps concurrently |
| `ConversationStep` | `conversation_step()` | Embeds a full FSM-LLM conversation within a workflow step |

### ConversationStep

The `ConversationStep` embeds a complete FSM-LLM conversation inside a workflow step. It runs an FSM to completion, maps context between the workflow and conversation, and supports auto-messages for non-interactive flows.

```python
from fsm_llm_workflows import conversation_step

step = conversation_step(
    "collect_data", "Data Collection",
    fsm_file="form.json",
    model="gpt-4o-mini",
    initial_context={"user_name": "name"},       # workflow_key → conversation_key
    context_mapping={"collected_name": "name"},   # conversation_key → workflow_key
    auto_messages=["My name is Alice"],
    success_state="process_data",
)
```

## File Map

| File | Purpose |
|------|---------|
| `engine.py` | **WorkflowEngine** — async execution engine with timer management and workflow lifecycle |
| `dsl.py` | Python DSL: `create_workflow()`, `auto_step()`, `api_step()`, `llm_step()`, `conversation_step()`, factory functions, **WorkflowBuilder** |
| `steps.py` | 9 step type implementations: WorkflowStep (base ABC) + 8 concrete types including ConversationStep |
| `definitions.py` | **WorkflowDefinition** with validation (reachability, cycles, state transitions), **WorkflowValidator** |
| `models.py` | **WorkflowStatus** enum, WorkflowEvent, WorkflowStepResult, WorkflowInstance, EventListener, WaitEventConfig |
| `handlers.py` | **AutoTransitionHandler**, **EventHandler**, **TimerHandler** — integrate with fsm_llm handler system |
| `exceptions.py` | **WorkflowError** hierarchy: Definition, Step, Instance, Timeout, Validation, State, Event, Resource errors |
| `__version__.py` | Package version string |
| `__init__.py` | Public API exports — single `__all__` list |

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

## Exception Hierarchy

- `WorkflowError` (extends `FSMError`) → `WorkflowDefinitionError`, `WorkflowStepError`, `WorkflowInstanceError`, `WorkflowTimeoutError`, `WorkflowValidationError`, `WorkflowStateError`, `WorkflowEventError`, `WorkflowResourceError`

## Development

```bash
pytest tests/test_fsm_llm_workflows/  # 83 tests across 4 test files (auto-skips if not installed)
```

## License

GNU General Public License v3.0. See [LICENSE](../../LICENSE).
