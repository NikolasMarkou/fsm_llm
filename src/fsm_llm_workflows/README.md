# fsm_llm_workflows

Async event-driven workflow orchestration engine for FSM-LLM. Provides 8 step types, a Python DSL for defining workflows declaratively, factory functions for common patterns, and a fluent builder API. Workflows execute asynchronously and can embed full FSM conversations via ConversationStep.

Part of the [FSM-LLM](https://github.com/NikolasMarkou/fsm_llm) framework (v0.3.0).

## Features

- **Async execution** -- workflows run with `asyncio`, supporting concurrent instances with configurable limits
- **8 step types** -- AutoTransition, APICall, Condition, LLMProcessing, WaitForEvent, Timer, Parallel, Conversation
- **Python DSL** -- `create_workflow()`, `auto_step()`, `llm_step()`, and other helper functions for concise definitions
- **Factory functions** -- `linear_workflow()`, `conditional_workflow()`, `event_driven_workflow()` for common patterns
- **ConversationStep** -- embed full FSM-LLM conversations (including push/pop stacking) inside workflow steps
- **Validation** -- reachability analysis, cycle detection, state transition validation, terminal state warnings
- **WorkflowBuilder** -- fluent API for chaining step additions

## Installation

```bash
pip install fsm-llm[workflows]
```

This adds the optional dependencies: `networkx`, `graphviz`, `aiofiles`.

## Quick Start

```python
import asyncio
from fsm_llm_workflows import (
    WorkflowEngine,
    create_workflow,
    auto_step,
)

# Define a workflow
wf = create_workflow("my_workflow", "My Workflow", description="A simple example")
wf.with_initial_step(
    auto_step("start", "Start", next_state="process", action=lambda ctx: {"ready": True})
)
wf.with_step(auto_step("process", "Process", next_state="done"))
wf.with_step(auto_step("done", "Done", next_state=""))  # terminal step

# Run the workflow
async def main():
    engine = WorkflowEngine()
    engine.register_workflow(wf)
    instance_id = await engine.start_workflow("my_workflow", initial_context={"input": "hello"})
    print(f"Workflow completed: {instance_id}")

asyncio.run(main())
```

## Step Types

| Step Type | DSL Function | Description |
|-----------|-------------|-------------|
| `AutoTransitionStep` | `auto_step()` | Executes an optional action, then transitions to `next_state` |
| `APICallStep` | `api_step()` | Calls an external API with input/output mapping, routes to `success_state` or `failure_state` |
| `ConditionStep` | `condition_step()` | Evaluates a callable, branches to `true_state` or `false_state` |
| `LLMProcessingStep` | `llm_step()` | Sends a prompt template to an LLM, extracts output via regex patterns |
| `WaitForEventStep` | `wait_event_step()` | Pauses execution until an external event arrives (with optional timeout) |
| `TimerStep` | `timer_step()` | Waits for a specified delay before transitioning |
| `ParallelStep` | `parallel_step()` | Executes multiple sub-steps concurrently, aggregates results |
| `ConversationStep` | `conversation_step()` | Runs a full FSM-LLM conversation as a workflow step |

All steps inherit from `WorkflowStep` (abstract base) and implement `async execute(context) -> WorkflowStepResult`. Each step supports an optional `timeout` field (seconds) that wraps execution with `asyncio.wait_for`.

## DSL Reference

### Workflow Creation

- **`create_workflow(workflow_id, name, description="")`** -- returns a `WorkflowDefinition`

### Step Functions

- **`auto_step(step_id, name, next_state, action=None, description="")`** -- auto-transition with optional sync/async action
- **`api_step(step_id, name, api_function, success_state, failure_state, input_mapping=None, output_mapping=None, description="")`** -- external API call
- **`condition_step(step_id, name, condition, true_state, false_state, description="")`** -- boolean branching
- **`llm_step(step_id, name, llm_interface, prompt_template, context_mapping, output_mapping, next_state, error_state=None, description="")`** -- LLM processing
- **`wait_event_step(step_id, name, event_type, success_state, timeout_seconds=None, timeout_state=None, event_mapping=None, description="")`** -- event waiting
- **`timer_step(step_id, name, delay_seconds, next_state, description="")`** -- timed delay
- **`parallel_step(step_id, name, steps, next_state, error_state=None, aggregation_function=None, description="")`** -- parallel execution
- **`conversation_step(step_id, name, success_state="", fsm_file=None, fsm_definition=None, model=None, initial_context=None, context_mapping=None, auto_messages=None, max_turns=20, error_state=None, description="")`** -- FSM conversation

## Factory Functions

Factory functions create pre-wired `WorkflowDefinition` objects for common patterns:

- **`linear_workflow(workflow_id, name, steps, description="")`** -- executes a list of steps in sequence, sets the first step as initial
- **`conditional_workflow(workflow_id, name, initial_step, condition_step, true_branch, false_branch, description="")`** -- branches based on a condition step
- **`event_driven_workflow(workflow_id, name, setup_steps, event_step, processing_steps, description="")`** -- runs setup, waits for an event, then processes

## WorkflowBuilder

The `WorkflowBuilder` provides a fluent API for constructing workflows:

```python
from fsm_llm_workflows import workflow_builder, auto_step

wf = (
    workflow_builder("wf_id", "My Workflow")
    .set_initial_step(auto_step("start", "Start", next_state="end"))
    .add_step(auto_step("end", "End", next_state=""))
    .add_metadata("version", "1.0")
    .build()
)
```

Methods: `.add_step(step)`, `.set_initial_step(step)`, `.add_metadata(key, value)`, `.build()`.

## ConversationStep

`ConversationStep` bridges workflows with FSM-LLM core. A workflow step can invoke a full FSM conversation, pass context in, and extract results back.

```python
from fsm_llm_workflows import conversation_step

step = conversation_step(
    "collect_info", "Collect User Info",
    fsm_file="form_filling.json",
    model="gpt-4o-mini",
    initial_context={"user_name": "name"},      # workflow ctx key -> conversation ctx key
    context_mapping={"collected_name": "name"},  # workflow ctx key <- conversation ctx key
    auto_messages=["My name is Alice", "I live in Berlin"],
    max_turns=20,
    success_state="process_data",
    error_state="handle_error",
)
```

- **`initial_context`** -- maps workflow context keys to conversation initial context keys
- **`context_mapping`** -- maps workflow context keys from conversation collected data keys
- **`auto_messages`** -- messages sent to drive the conversation forward
- **`fsm_file`** or **`fsm_definition`** -- provide the FSM definition (one is required)
- Raw conversation data is also stored under `_conversation_{step_id}_data`

## Workflow Lifecycle

```
PENDING --> RUNNING --> WAITING (events/timers) --> COMPLETED
                  |                              --> FAILED
                  |                              --> CANCELLED
                  +--> COMPLETED
                  +--> FAILED
                  +--> CANCELLED
```

| Status | Description |
|--------|-------------|
| `PENDING` | Instance created, not yet started |
| `RUNNING` | Actively executing steps |
| `WAITING` | Paused for an event or timer |
| `COMPLETED` | All steps finished successfully |
| `FAILED` | An unrecoverable error occurred |
| `CANCELLED` | Externally cancelled |

Terminal states (`COMPLETED`, `FAILED`, `CANCELLED`) allow no further transitions. The `WorkflowInstance.update_status()` method enforces valid transitions.

## File Map

| File | Description |
|------|-------------|
| `engine.py` | `WorkflowEngine` -- async execution engine, timer management, event delivery, instance lifecycle |
| `dsl.py` | Python DSL functions, `WorkflowBuilder` class, factory functions (`linear_workflow`, etc.) |
| `steps.py` | 8 step types: `WorkflowStep` (ABC), `AutoTransitionStep`, `APICallStep`, `ConditionStep`, `LLMProcessingStep`, `WaitForEventStep`, `TimerStep`, `ParallelStep`, `ConversationStep` |
| `definitions.py` | `WorkflowDefinition` (Pydantic model with validation), `WorkflowValidator` utility |
| `models.py` | `WorkflowStatus`, `WorkflowEvent`, `WorkflowStepResult`, `WorkflowInstance`, `EventListener`, `WaitEventConfig` |
| `handlers.py` | Handler integration module (intentionally empty -- engine manages operations directly) |
| `exceptions.py` | `WorkflowError` hierarchy (8 exception classes) |
| `__init__.py` | Public API exports (36 symbols in `__all__`) |
| `__version__.py` | Version string (synced from `fsm_llm.__version__`) |

## Examples

- **`examples/workflows/order_processing/`** -- order processing workflow demonstrating step types and event handling
- **`examples/agents/workflow_agent/`** -- agent pattern that integrates with workflows

## Development

### Testing

```bash
pytest tests/test_fsm_llm_workflows/  # 116 tests across 5 files
```

Tests auto-skip if the workflows extension is not installed.

| Test File | Focus |
|-----------|-------|
| `test_workflows.py` | Engine lifecycle, registration, start/complete/cancel, event delivery |
| `test_dsl.py` | DSL functions, factory functions, WorkflowBuilder |
| `test_steps.py` | Step execution, input/output mapping, error handling |
| `test_step_timeouts.py` | Per-step timeout enforcement |
| `test_audit_fixes.py` | Regression tests for audit findings |
