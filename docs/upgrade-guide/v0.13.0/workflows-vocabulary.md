# Workflows vocabulary

Covers FSM-LLM 0.12.0 to 0.13.0. The workflows package (`fsm_llm.workflows`) now uses the same words as core: a time limit carries its unit, a budget of steps is `max_steps`, a method does not repeat its class name, an id is `<thing>_id`, and a step names a step, not a state. Every rename is a clean break with no alias. The old spelling fails at once, with a `TypeError`, a `ValidationError`, an `AttributeError` or an `ImportError`, with two exceptions that are called out below (an old-shaped internal marker, and a custom step whose routing field keeps its old name).

The sections follow the order of the CHANGELOG entries. A search for the old name in your code finds every place to edit.

## Time limits carry their unit

Core calls a run budget `max_seconds`, so `WorkflowEngine.start(workflow_timeout=)` and the `WorkflowInstance.workflow_timeout` field are now `max_seconds`. The base `WorkflowStep.timeout` field and the `timeout=` parameter of every step factory in `fsm_llm.workflows.dsl` (`auto_step`, `api_step`, `condition_step`, `llm_step`, `parallel_step`, `conversation_step`, `agent_step`, `retry_step`) are now `timeout_seconds`, the name `wait_event_step` and `WaitEventConfig` already used. `serialize()` writes `timeout_seconds` instead of `timeout` for a step. `WorkflowTimeoutError.timeout_seconds` and `ConversationStep.conversation_timeout` keep their names (the latter is a second limit: the smaller of it and `timeout_seconds` applies).

A factory called with `workflow_timeout=` or `timeout=` raises a `TypeError`. A step class, `WorkflowInstance` or `WaitEventConfig` built directly with either raises a `ValidationError`.

**Before**

```python
step = auto_step("greet", "Greet", next_state="", timeout=5.0)
instance_id = await engine.start_workflow("hello", workflow_timeout=30.0)
```

**After**

```python
import asyncio

from fsm_llm.workflows import WorkflowEngine, auto_step, create_workflow


async def main():
    engine = WorkflowEngine()
    workflow = create_workflow("hello", "Hello").with_initial_step(
        auto_step("greet", "Greet", next_step_id="", timeout_seconds=5.0)
    )
    engine.register_workflow(workflow)
    instance_id = await engine.start("hello", max_seconds=30.0)
    assert engine.get_instance(instance_id).max_seconds == 30.0
    step = workflow.serialize()["steps"]["greet"]
    assert step["timeout_seconds"] == 5.0 and "timeout" not in step
    await engine.close()


asyncio.run(main())
```

## The step budget is max_steps

`WorkflowEngine(max_steps_per_run=)` is renamed `max_steps` (also the engine attribute and the `ValueError` text), and the constant `MAX_STEPS_PER_RUN` (in `fsm_llm.workflows`, `fsm_llm.workflows.constants` and the engine module) is renamed `DEFAULT_MAX_STEPS`. The default (1000) and the behaviour are unchanged. The name now matches the core `run_until_terminal(max_steps=)` budget it resembles.

**Before**

```python
from fsm_llm.workflows import MAX_STEPS_PER_RUN, WorkflowEngine

engine = WorkflowEngine(max_steps_per_run=MAX_STEPS_PER_RUN)
```

**After**

```python
from fsm_llm.workflows import DEFAULT_MAX_STEPS, WorkflowEngine

engine = WorkflowEngine(max_steps=DEFAULT_MAX_STEPS)
assert DEFAULT_MAX_STEPS == 1000
```

## Context key constants are renamed CONTEXT_KEY

The engine context-key constants of `fsm_llm.workflows.constants` are renamed from `KEY_*` to `CONTEXT_KEY_*`, the spelling of core (`CONTEXT_KEY_CUT_OFF`). The values (`"_waiting_info"`, `"_timeout"`, and so on) are unchanged, so no stored context changes. None of them was in `fsm_llm.workflows.__all__`; only code that imported one from `fsm_llm.workflows.constants` is affected.

| Before | After |
| --- | --- |
| `KEY_WAITING_INFO` | `CONTEXT_KEY_WAITING_INFO` |
| `KEY_TIMER_INFO` | `CONTEXT_KEY_TIMER_INFO` |
| `KEY_WORKFLOW_INFO` | `CONTEXT_KEY_WORKFLOW_INFO` |
| `KEY_TIMER_EXPIRED` | `CONTEXT_KEY_TIMER_EXPIRED` |
| `KEY_LAST_EVENT` | `CONTEXT_KEY_LAST_EVENT` |
| `KEY_USER_INPUT` | `CONTEXT_KEY_USER_INPUT` |
| `KEY_CANCELLATION_REASON` | `CONTEXT_KEY_CANCELLATION_REASON` |
| `KEY_TIMEOUT` | `CONTEXT_KEY_EVENT_TIMEOUT` (it holds `{event_type, timeout_at}`, not a limit) |

**Before**

```python
from fsm_llm.workflows.constants import KEY_TIMEOUT, KEY_WAITING_INFO

waiting = context.get(KEY_WAITING_INFO)
```

**After**

```python
from fsm_llm.workflows.constants import (
    CONTEXT_KEY_EVENT_TIMEOUT,
    CONTEXT_KEY_WAITING_INFO,
)

assert CONTEXT_KEY_WAITING_INFO == "_waiting_info"
assert CONTEXT_KEY_EVENT_TIMEOUT == "_timeout"
```

## Engine methods no longer repeat the class name

The `WorkflowEngine` methods drop the word `workflow`, like `engine.start` in core: `start_workflow` is `start`, `advance_workflow` is `advance`, `cancel_workflow` is `cancel`, `get_workflow_instance` is `get_instance`, `get_workflow_definition` is `get_definition`, `get_workflow_status` is `get_status` and `get_workflow_context` is `get_context`. `register_workflow`, `remove_instance`, `add_hook`, `process_event` and `get_statistics` keep their names. The monitor's own `InstanceManager.advance_workflow`, `cancel_workflow`, `get_workflow_status`, `start_workflow_instance` and `send_workflow_event` are different methods and are unchanged.

**Before**

```python
instance_id = await engine.start_workflow("order")
status = engine.get_workflow_status(instance_id)
context = engine.get_workflow_context(instance_id)
await engine.cancel_workflow(instance_id, "no longer needed")
```

**After**

```python
import asyncio

from fsm_llm.workflows import (
    WorkflowEngine,
    WorkflowStatus,
    create_workflow,
    wait_event_step,
)


async def main():
    engine = WorkflowEngine()
    workflow = create_workflow("order", "Order").with_initial_step(
        wait_event_step("wait", "Wait", event_type="paid", success_step_id="")
    )
    engine.register_workflow(workflow)
    instance_id = await engine.start("order")
    assert engine.get_definition("order").workflow_id == "order"
    assert engine.get_status(instance_id) == WorkflowStatus.WAITING
    assert engine.get_instance(instance_id).workflow_id == "order"
    assert isinstance(engine.get_context(instance_id), dict)
    await engine.cancel(instance_id, "no longer needed")
    assert engine.get_status(instance_id) == WorkflowStatus.CANCELLED
    await engine.close()


asyncio.run(main())
```

## list_active_instances and close

`WorkflowEngine.get_active_workflows` is renamed `list_active_instances` (it returns instance ids, like the core `list_*` readers) and `WorkflowEngine.shutdown` is renamed `close` (the verb of core `API.close`). The `WorkflowResourceError` that `start` raises on a closed engine now has `resource_id="closed"` and the message `Engine has been closed` (it was `"shutdown"` and `Engine has been shut down`); a caller that matched the old text must change it.

**Before**

```python
active = engine.get_active_workflows()
await engine.shutdown()
```

**After**

```python
import asyncio

from fsm_llm.workflows import (
    WorkflowEngine,
    WorkflowResourceError,
    create_workflow,
    wait_event_step,
)


async def main():
    engine = WorkflowEngine()
    workflow = create_workflow("order", "Order").with_initial_step(
        wait_event_step("wait", "Wait", event_type="paid", success_step_id="")
    )
    engine.register_workflow(workflow)
    instance_id = await engine.start("order")
    assert engine.list_active_instances() == [instance_id]
    await engine.close()
    try:
        await engine.start("order")
    except WorkflowResourceError as error:
        assert error.resource_id == "closed"


asyncio.run(main())
```

## Step classes match their factories

Four step classes are renamed to match their factory functions: `AutoTransitionStep` is `AutoStep` (`auto_step`), `APICallStep` is `APIStep` (`api_step`), `LLMProcessingStep` is `LLMStep` (`llm_step`) and `WaitForEventStep` is `WaitEventStep` (`wait_event_step`). The factories keep their names, and the old class names no longer import from `fsm_llm.workflows`.

`WorkflowDefinition.serialize()` writes the class name as each step's `"type"`, so the tag changes with the class (`"AutoStep"`, `"APIStep"`, `"LLMStep"`, `"WaitEventStep"`). The output of `serialize()` is the only place the tag appears and no loader reads it, but stored output that you compare with a new run differs. The error texts of `APIStep` and `LLMStep` name the new class. `steps.STEP_EXECUTOR` is now the internal `_STEP_EXECUTOR`.

**Before**

```python
from fsm_llm.workflows import AutoTransitionStep, WaitForEventStep

assert isinstance(step, AutoTransitionStep)
```

**After**

```python
from fsm_llm.workflows import AutoStep, auto_step

step = auto_step("a", "A", next_step_id="")
assert isinstance(step, AutoStep)
assert type(step).__name__ == "AutoStep"
```

## Pattern helpers take the create prefix

The four workflow pattern helpers take the `create_` prefix of core's factories: `workflow_builder` is `create_workflow_builder`, `linear_workflow` is `create_linear_workflow`, `conditional_workflow` is `create_conditional_workflow` and `event_driven_workflow` is `create_event_driven_workflow` (also in `fsm_llm.workflows.__all__`). The `<kind>_step` factories, `create_workflow` (which returns a `WorkflowDefinition`), `with_step`, `with_initial_step` and `serialize()` keep their names.

**Before**

```python
from fsm_llm.workflows import linear_workflow

definition = linear_workflow("flow", "Flow", [step_a, step_b])
```

**After**

```python
from fsm_llm.workflows import auto_step, create_linear_workflow

step_a = auto_step("a", "A", next_step_id="b")
step_b = auto_step("b", "B", next_step_id="")
definition = create_linear_workflow("flow", "Flow", [step_a, step_b])
assert definition.initial_step_id == "a"
```

## validate is validate_structure

`WorkflowDefinition.validate()` is renamed `validate_structure()`. The old name shadowed pydantic's own `BaseModel.validate` (it needed a `# type: ignore[override]`); core names its checks `validate_<what>`. It still raises `WorkflowValidationError` with every error in `validation_errors`. `WorkflowEngine.register_workflow` and `WorkflowBuilder.build()` call it, so most code never calls it directly.

**Before**

```python
definition.validate()
```

**After**

```python
from fsm_llm.workflows import WorkflowValidationError, auto_step, create_workflow

definition = create_workflow("flow", "Flow").with_initial_step(
    auto_step("a", "A", next_step_id="missing")
)
try:
    definition.validate_structure()
except WorkflowValidationError as error:
    assert error.validation_errors
```

## Workflow errors take the message first

The eight `Workflow*Error` constructors of `fsm_llm.workflows.exceptions` take the message first and the identifying data as keyword-only arguments, like core (`StateNotFoundError(message, state_id=None)`), with no forwarding of the old positional shape:

| Class | Constructor |
| --- | --- |
| `WorkflowDefinitionError` | `(message, *, workflow_id=None, details=None)` |
| `WorkflowInstanceError` | `(message, *, instance_id=None, details=None)` |
| `WorkflowStepError` | `(message, *, step_id=None, cause=None, details=None)` |
| `WorkflowTimeoutError` | `(message, *, operation=None, timeout_seconds=None, instance_id=None, details=None)` |
| `WorkflowValidationError` | `(message, *, validation_errors=None, details=None)` |
| `WorkflowStateError` | `(message, *, current_step_id=None, operation=None, details=None)` |
| `WorkflowEventError` | `(message, *, event_type=None, details=None)` |
| `WorkflowResourceError` | `(message, *, resource_type=None, resource_id=None, details=None)` |

The attributes keep their names and the engine's messages read as before (an id still prefixes the text, `"Step 'a': ..."`; an id left out adds no prefix). `WorkflowValidationError` no longer builds its own `Validation failed with N error(s)` headline: you pass the message and the per-error lines are appended to it. `WorkflowTimeoutError` no longer builds its text from `operation` and `timeout_seconds`: you pass the message. Code that raises these classes, or a subclass that calls `super().__init__` positionally, must change.

**Before**

```python
raise WorkflowStepError("a", "the action failed")
```

**After**

```python
from fsm_llm.workflows import WorkflowStepError

error = WorkflowStepError("the action failed", step_id="a")
assert error.step_id == "a"
assert "the action failed" in str(error)
try:
    WorkflowStepError("a", "the action failed")
except TypeError:
    pass
```

## WorkflowStateError carries current_step_id

`WorkflowStateError.current_state` is renamed `current_step_id`: the attribute, the constructor argument and the `details["current_step_id"]` key. It always held a step id, never a run status. The class name is unchanged. A caller reading `err.current_state` gets an `AttributeError`.

**Before**

```python
try:
    engine.check(instance_id)
except WorkflowStateError as err:
    print(err.current_state)
```

**After**

```python
from fsm_llm.workflows import WorkflowStateError

err = WorkflowStateError("cannot advance", current_step_id="review")
assert err.current_step_id == "review"
assert err.details["current_step_id"] == "review"
```

## Routing fields end in step_id

The workflow routing fields end in `_step_id`, with no second suffix: `next_state` is `next_step_id`, `success_state` is `success_step_id`, `error_state` is `error_step_id`, `failure_state` is `failure_step_id`, `timeout_state` is `timeout_step_id`, `default_state` is `default_step_id`, `true_state` is `true_step_id` and `false_state` is `false_step_id`. The rename covers the step classes and their factory keywords (`auto_step(next_step_id=)`, `condition_step(true_step_id=, false_step_id=)`, `api_step`, `timer_step`, `wait_event_step`, `switch_step`, and the others), `WaitEventConfig`, `EventListener`, `Timer`, and the `register_event_listener(success_step_id=, timeout_step_id=)` and `schedule_timer(next_step_id)` parameters. A state is an FSM concept; a workflow step names a step.

An old keyword raises a `TypeError` (the factories) or a pydantic `ValidationError` (the step classes, `WaitEventConfig`, `EventListener` and `WorkflowStepResult`, which refuse any unknown keyword), so a step that still passes `next_state=` fails when it is built.

**Before**

```python
check = condition_step(
    "check", "Check", condition=ok, true_state="done", false_state="retry"
)
wait = wait_event_step("wait", "Wait", event_type="paid", success_state="ship")
```

**After**

```python
from fsm_llm.workflows import condition_step, wait_event_step

check = condition_step(
    "check",
    "Check",
    condition=lambda context: bool(context.get("ok")),
    true_step_id="done",
    false_step_id="retry",
)
wait = wait_event_step(
    "wait",
    "Wait",
    event_type="paid",
    success_step_id="ship",
    timeout_seconds=60.0,
    timeout_step_id="late",
)
assert check.referenced_step_ids() == {"done", "retry"}
assert wait.referenced_step_ids() == {"ship", "late"}
```

## WorkflowStepResult carries next_step_id

`WorkflowStepResult.next_state` is renamed `next_step_id`, and so is the `next_state` keyword of `WorkflowStepResult.success_result` and `failure_result`, and the `next_state` key of the `step_completed` hook data, which is now `{step_id, success, next_step_id, error}`. A custom step that returns `success_result(next_state=...)`, or a hook that reads `data["next_state"]`, must change. Terminal semantics are unchanged: an empty `next_step_id` on any successful result ends the workflow.

**Before**

```python
return WorkflowStepResult.success_result({"n": 2}, next_state="done")

def hook(event, instance, data):
    if event == "step_completed":
        print(data["next_state"])
```

**After**

```python
import asyncio
from typing import Any

from fsm_llm.workflows import (
    WorkflowEngine,
    WorkflowStep,
    WorkflowStepResult,
    auto_step,
    create_workflow,
)


class Double(WorkflowStep):
    next_step_id: str

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        return WorkflowStepResult.success_result(
            {"n": context.get("n", 1) * 2}, next_step_id=self.next_step_id
        )


seen = []


def hook(event, instance, data):
    if event == "step_completed":
        seen.append((data["step_id"], data["next_step_id"]))


async def main():
    engine = WorkflowEngine()
    engine.add_hook(hook)
    workflow = (
        create_workflow("double", "Double")
        .with_initial_step(Double(step_id="d", name="D", next_step_id="end"))
        .with_step(auto_step("end", "End", next_step_id=""))
    )
    engine.register_workflow(workflow)
    instance_id = await engine.start("double", initial_context={"n": 4})
    assert engine.get_context(instance_id)["n"] == 8
    assert ("d", "end") in seen
    await engine.close()


asyncio.run(main())
```

## Workflow models refuse unknown keywords

`WorkflowStep` (so every step class), `WorkflowStepResult`, `WorkflowInstance`, `EventListener` and `WaitEventConfig` now refuse unknown keywords (`extra="forbid"`, like `AgentConfig`). Pydantic used to drop them, so a removed spelling such as `error_state=`, `timeout=`, `default_state=` or `workflow_timeout=` was silently ignored, and a step with `error_state="recover"` ran with no error route. Every unknown keyword, a typo included, now raises a `ValidationError`. The same holds for `ToolDefinition` (see the agents guide).

Nothing needs to change in code that already used the new names. Code that passed a keyword that did nothing now fails at construction, which is the point: remove the keyword or use its new name.

**Before**

```python
step = AutoStep(step_id="a", name="A", next_step_id="", error_state="recover")
```

**After**

```python
from pydantic import ValidationError

from fsm_llm.workflows import AutoStep

step = AutoStep(step_id="a", name="A", next_step_id="", error_step_id="recover")
assert step.error_step_id == "recover"
try:
    AutoStep(step_id="a", name="A", next_step_id="", error_state="recover")
except ValidationError as error:
    assert "error_state" in str(error)
```

## The waiting and timer marker keys follow the rename

The dict keys inside the `_waiting_info` and `_timer_info` context values follow the same rename: `success_state` is `success_step_id` and `timeout_state` is `timeout_step_id` in `_waiting_info` (written by `WaitEventStep`, read by the engine), and `next_state` is `next_step_id` in `_timer_info` (written by `TimerStep`). Both values are internal context keys that a custom step may write itself (they are in `STEP_INTERNAL_WHITELIST`). Such a step must use the new keys.

An old-shaped marker is not refused, and each failure is quiet, so check your own steps by searching for the old key names:

- a `_waiting_info` whose `success_state` key was not renamed reads `success_step_id` as empty, so the workflow COMPLETES when the event arrives instead of routing to the old target;
- an old `timeout_state` key is ignored, so a timed-out wait FAILS instead of routing to its timeout step;
- an old-shaped `_timer_info` makes the instance FAILED with the error text `'next_step_id'` (a `KeyError` that the engine records) when the timer fires.

**Before**

```python
return WorkflowStepResult.success_result(
    data={
        "_waiting_info": {
            "waiting_for_event": True,
            "event_type": "approved",
            "success_state": "ship",
            "timeout_state": "escalate",
        }
    }
)
```

**After**

```python
import asyncio
from typing import Any

from fsm_llm.workflows import (
    WorkflowEngine,
    WorkflowEvent,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepResult,
    auto_step,
    create_workflow,
)
from fsm_llm.workflows.constants import CONTEXT_KEY_WAITING_INFO


class WaitForApproval(WorkflowStep):
    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        return WorkflowStepResult.success_result(
            data={
                CONTEXT_KEY_WAITING_INFO: {
                    "waiting_for_event": True,
                    "event_type": "approved",
                    "success_step_id": "ship",
                    "timeout_step_id": "escalate",
                }
            }
        )

    def referenced_step_ids(self) -> set[str]:
        return {"ship", "escalate"}


async def main():
    engine = WorkflowEngine()
    workflow = (
        create_workflow("approval", "Approval")
        .with_initial_step(WaitForApproval(step_id="wait", name="Wait"))
        .with_step(auto_step("ship", "Ship", next_step_id=""))
        .with_step(auto_step("escalate", "Escalate", next_step_id=""))
    )
    engine.register_workflow(workflow)
    instance_id = await engine.start("approval")
    assert engine.get_status(instance_id) == WorkflowStatus.WAITING
    await engine.process_event(
        WorkflowEvent(event_type="approved", instance_id=instance_id)
    )
    assert engine.get_status(instance_id) == WorkflowStatus.COMPLETED
    await engine.close()


asyncio.run(main())
```

## referenced_states is referenced_step_ids

`WorkflowStep.referenced_states()` is renamed `referenced_step_ids()`, and its single discovery rule is now the field-name suffix `_step_id` (it was `_state`). A custom `WorkflowStep` subclass must name its routing fields with that suffix (`next_step_id`, `error_step_id`, and so on) or override `referenced_step_ids()`. A field still named `next_state` is no longer found, so validation, reachability and terminal detection no longer see it.

**Before**

```python
class Route(WorkflowStep):
    next_state: str

    def referenced_states(self):
        return {self.next_state}
```

**After**

```python
from typing import Any

from fsm_llm.workflows import WorkflowStep, WorkflowStepResult


class Route(WorkflowStep):
    next_step_id: str
    fallback_step_id: str = ""

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        return WorkflowStepResult.success_result(next_step_id=self.next_step_id)


route = Route(step_id="r", name="R", next_step_id="done", fallback_step_id="oops")
assert route.referenced_step_ids() == {"done", "oops"}
```
