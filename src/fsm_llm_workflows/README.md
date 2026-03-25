# FSM-LLM Workflow Engine

An event-driven workflow orchestration engine built on FSM-LLM that enables automated state transitions, external API integration, LLM processing, timers, parallel execution, and embedded FSM conversations. Part of the `fsm-llm` package.

## Features

- **8 Step Types**: AutoTransition, APICall, Condition, LLMProcessing, WaitForEvent, Timer, Parallel, and ConversationStep
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
    create_workflow, auto_step, llm_step, condition_step
)

# Create a simple processing workflow
workflow = create_workflow("order_processor", "Order Processor")

workflow.with_initial_step(auto_step(
    "receive", "Receive Order", next_state="validate",
    action=lambda ctx: {"validated": True}
))

workflow.with_step(condition_step(
    "validate", "Validate Order",
    condition=lambda ctx: ctx.get("validated", False),
    true_state="process", false_state="reject"
))

workflow.with_step(llm_step(
    "process", "Process Order",
    llm_interface=my_llm,
    prompt_template="Summarize this order: {order_details}",
    context_mapping={"order_details": "order_details"},
    output_mapping={"summary": "order_summary"},
    next_state="complete"
))

workflow.with_step(auto_step(
    "complete", "Complete Order", next_state="done",
    action=lambda ctx: {"status": "done"}
))

workflow.with_step(auto_step(
    "done", "Done", next_state="",  # Terminal state
))

workflow.with_step(auto_step(
    "reject", "Reject Order", next_state="done",
))

# Validate the workflow
workflow.validate()
```

### Use the WorkflowBuilder

```python
from fsm_llm_workflows import workflow_builder, auto_step

wf = (
    workflow_builder("my_workflow", "My Workflow")
    .set_initial_step(auto_step("start", "Start", next_state="middle", action=do_first))
    .add_step(auto_step("middle", "Middle", next_state="end", action=do_second))
    .add_step(auto_step("end", "End", next_state=""))
    .build()
)
```

### Factory Functions for Common Patterns

```python
from fsm_llm_workflows import linear_workflow, auto_step

# Sequential steps — pass a list of WorkflowStep objects
wf = linear_workflow("pipeline", "Data Pipeline", steps=[
    auto_step("fetch", "Fetch Data", next_state="transform", action=fetch_data),
    auto_step("transform", "Transform Data", next_state="load", action=transform_data),
    auto_step("load", "Load Data", next_state="done", action=load_data),
    auto_step("done", "Done", next_state=""),
])
```

### Run with WorkflowEngine

```python
from fsm_llm_workflows import WorkflowEngine

engine = WorkflowEngine()
engine.register_workflow(workflow)

# start_workflow returns an instance_id string
instance_id = await engine.start_workflow(
    "order_processor", initial_context={"order_id": "123"}
)
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
    initial_context={"user_name": "name"},       # workflow_key -> conversation_key
    context_mapping={"collected_name": "name"},   # conversation_key -> workflow_key
    auto_messages=["My name is Alice"],
    success_state="process_data",
)
```

## File Map

| File | Purpose |
|------|---------|
| `engine.py` | **WorkflowEngine** -- async execution engine with timer management and workflow lifecycle |
| `dsl.py` | Python DSL: `create_workflow()`, `auto_step()`, `api_step()`, `llm_step()`, `conversation_step()`, factory functions, **WorkflowBuilder** |
| `steps.py` | 8 concrete step types + WorkflowStep base ABC |
| `definitions.py` | **WorkflowDefinition** with validation (reachability, cycles, state transitions), **WorkflowValidator** |
| `models.py` | **WorkflowStatus** enum, WorkflowEvent, WorkflowStepResult, WorkflowInstance, EventListener, WaitEventConfig |
| `handlers.py` | Handler integration module (handlers removed -- engine manages operations directly) |
| `exceptions.py` | **WorkflowError** hierarchy: Definition, Step, Instance, Timeout, Validation, State, Event, Resource errors |
| `__version__.py` | Package version string |
| `__init__.py` | Public API exports -- single `__all__` list |

## Workflow Status Lifecycle

```
PENDING -> RUNNING -> WAITING (for events/timers) -> COMPLETED
                  \-> FAILED
                  \-> CANCELLED
```

## WorkflowDefinition Validation

```python
workflow.validate()           # Full structural validation
workflow.has_cycles()          # Detect cycles in workflow graph
workflow.serialize()           # Serialize to dictionary
```

Validation checks: steps exist, state transitions reference valid states, all states reachable from initial, terminal states exist.

## API Reference

### WorkflowEngine

| Method | Description |
|--------|-------------|
| `WorkflowEngine()` | Create engine instance |
| `register_workflow(definition)` | Register a workflow definition |
| `await start_workflow(workflow_id, initial_context=None)` | Start instance → `instance_id` |
| `await advance_workflow(instance_id, event=None)` | Advance to next step |
| `await send_event(instance_id, event)` | Send event to waiting workflow |
| `await cancel_workflow(instance_id)` | Cancel a running workflow |
| `get_instance(instance_id)` | Get workflow instance |
| `get_status(instance_id)` | Get `WorkflowStatus` |

### WorkflowBuilder

| Method | Description |
|--------|-------------|
| `workflow_builder(workflow_id, name)` | Create builder instance |
| `.set_initial_step(step)` | Set the initial step (chainable) |
| `.add_step(step)` | Add a step (chainable) |
| `.validate()` | Validate definition (chainable) |
| `.build()` | Build and return `WorkflowDefinition` |

### Factory Functions

| Function | Description |
|----------|-------------|
| `linear_workflow(workflow_id, name, steps)` | Sequential step execution |
| `conditional_workflow(workflow_id, name, ...)` | Branching based on conditions |
| `event_driven_workflow(workflow_id, name, ...)` | Event-based triggers |

## Integration with Agents

Workflows can be combined with agents for orchestrated tool-use. See [`examples/agents/workflow_agent/`](../../examples/agents/workflow_agent/) for an example of agent + workflow integration.

## Exception Hierarchy

- `WorkflowError` (extends `FSMError`) -> `WorkflowDefinitionError`, `WorkflowStepError`, `WorkflowInstanceError`, `WorkflowTimeoutError`, `WorkflowValidationError`, `WorkflowStateError`, `WorkflowEventError`, `WorkflowResourceError`

## Development

```bash
pytest tests/test_fsm_llm_workflows/  # 116 tests across 5 test files (auto-skips if not installed)
```

## License

GNU General Public License v3.0. See [LICENSE](../../LICENSE).
