# FSM-LLM Workflows

> Async event-driven workflow orchestration with 11 step types and a Python DSL.

---

## Overview

`fsm_llm_workflows` is an extension package that adds workflow orchestration to FSM-LLM. It enables building complex, multi-step processes that combine API calls, LLM processing, FSM conversations, agent execution, timers, events, and conditional logic — all in an async, event-driven engine.

Key capabilities:
- **11 step types** covering common workflow patterns
- **Python DSL** for declarative workflow construction
- **Event-driven execution** with external event listeners and timers
- **Parallel execution** of independent workflow branches
- **Validation** with reachability analysis and cycle detection
- **Dependency resolution** -- topological sort for parallel execution waves
- **Agent and conversation integration** — run FSM-LLM agents and conversations as workflow steps

## Installation

```bash
pip install fsm-llm[workflows]
```

**Requirements**: Python 3.10+ | No additional dependencies beyond core `fsm-llm`.

## Quick Start

**1. Build a workflow with the DSL**:

```python
from fsm_llm_workflows import WorkflowEngine, create_workflow, auto_step, condition_step

workflow = create_workflow("order_flow", "Order Processing")

workflow.with_initial_step(
    auto_step("receive", "Receive Order", next_state="validate",
              action=lambda ctx: {"order_id": ctx.get("order_id", "ORD-001")})
)
workflow.with_step(
    condition_step("validate", "Validate Order",
                   condition=lambda ctx: ctx.get("order_id") is not None,
                   true_state="process", false_state="reject")
)
workflow.with_step(auto_step("process", "Process Order", next_state="complete",
                             action=lambda ctx: {"status": "processed"}))
workflow.with_step(auto_step("reject", "Reject Order", next_state=""))
workflow.with_step(auto_step("complete", "Complete", next_state=""))
```

**2. Run the workflow**:

```python
import asyncio

async def main():
    engine = WorkflowEngine()
    engine.register_workflow(workflow)
    instance_id = await engine.start_workflow("order_flow", initial_context={"order_id": "ORD-123"})

    while await engine.advance_workflow(instance_id):
        status = engine.get_workflow_status(instance_id)
        print(f"Status: {status}")

    await engine.shutdown()

asyncio.run(main())
```

**3. Use pattern helpers**:

```python
from fsm_llm_workflows import linear_workflow, auto_step

workflow = linear_workflow("pipeline", "Data Pipeline", steps=[
    auto_step("fetch", "Fetch Data", next_state="", action=lambda ctx: {"data": "raw"}),
    auto_step("transform", "Transform", next_state="", action=lambda ctx: {"data": "cleaned"}),
    auto_step("load", "Load", next_state="", action=lambda ctx: {"loaded": True}),
])
```

## Workflow Status Lifecycle

```
PENDING → RUNNING → COMPLETED | FAILED | CANCELLED
                  → WAITING → RUNNING (event received) | FAILED | CANCELLED
```

## Step Types

| Step | DSL Function | Description |
|------|-------------|-------------|
| `AutoTransitionStep` | `auto_step()` | Auto-transition with optional action |
| `APICallStep` | `api_step()` | Call external API with input/output mapping |
| `ConditionStep` | `condition_step()` | Binary branching on condition |
| `LLMProcessingStep` | `llm_step()` | LLM processing with template and regex extraction |
| `WaitForEventStep` | `wait_event_step()` | Pause until external event or timeout |
| `TimerStep` | `timer_step()` | Wait N seconds before proceeding |
| `ParallelStep` | `parallel_step()` | Execute multiple steps concurrently |
| `ConversationStep` | `conversation_step()` | Run FSM conversation as a step |
| `AgentStep` | `agent_step()` | Run FSM-LLM agent as a step |
| `RetryStep` | `retry_step()` | Wrap any step with retry logic + backoff |
| `SwitchStep` | `switch_step()` | N-way branching on context key value |

### Integration Steps

**ConversationStep** — run an FSM conversation:
```python
conversation_step("chat", "Customer Chat", fsm_file="support.json",
                  model="gpt-4o-mini", auto_messages=["I need help"],
                  context_mapping={"resolution": "chat_result"}, max_turns=20)
```

**AgentStep** — run any FSM-LLM agent:
```python
agent_step("research", "Research Task", agent=agent,
           task_template="Research: {topic}", success_state="summarize")
```

**ParallelStep** — concurrent execution:
```python
parallel_step("parallel_fetch", "Fetch All Sources",
              steps=[api_step(...), api_step(...)], next_state="merge")
```

## Key API Reference

### WorkflowEngine

```python
engine = WorkflowEngine(max_concurrent_workflows=100)
engine.register_workflow(workflow_definition)
instance_id = await engine.start_workflow("workflow_id", initial_context={})
advanced = await engine.advance_workflow(instance_id)
await engine.process_event(WorkflowEvent(event_type="payment_received", payload={}))
await engine.schedule_timer(instance_id, delay_seconds=300, next_state="timeout")
status = engine.get_workflow_status(instance_id)
await engine.cancel_workflow(instance_id, reason="User cancelled")
await engine.shutdown()
```

### WorkflowDefinition

```python
workflow = WorkflowDefinition(workflow_id="my_workflow", name="My Workflow")
workflow.with_initial_step(step1).with_step(step2).with_step(step3)
workflow.validate()  # raises WorkflowValidationError on failure
terminal = workflow.get_terminal_states()
has_loops = workflow.has_cycles()
```

### Pattern Helpers

`linear_workflow()` (sequential), `conditional_workflow()` (binary branch), `event_driven_workflow()` (setup → wait → process), `workflow_builder()` (fluent builder).

## Dependency Resolution

```python
from fsm_llm_workflows import DependencyResolver

resolver = DependencyResolver()
resolver.add_step("fetch_data")
resolver.add_step("fetch_config")
resolver.add_step("process", depends_on=["fetch_data", "fetch_config"])
resolver.add_step("notify", depends_on=["process"])

waves = resolver.resolve()
# [["fetch_config", "fetch_data"], ["process"], ["notify"]]
# Steps within each wave can execute in parallel
```

Or from a dictionary:

```python
resolver = DependencyResolver.from_dict({
    "fetch": [],
    "process": ["fetch"],
    "notify": ["process"],
})
```

## Exception Hierarchy

```
FSMError
└── WorkflowError
    ├── WorkflowDefinitionError    # Invalid workflow structure
    ├── WorkflowStepError          # Step execution failure
    ├── WorkflowInstanceError      # Instance state issues
    ├── WorkflowTimeoutError       # Operation timed out
    ├── WorkflowValidationError    # Validation failures
    ├── WorkflowStateError         # Invalid state transition
    ├── WorkflowEventError         # Event processing failure
    └── WorkflowResourceError      # Resource limit reached
```

## License

GPL-3.0-or-later. See [LICENSE](../../LICENSE) for details.
