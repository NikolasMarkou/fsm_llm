# fsm_llm_workflows

An async workflow engine for FSM-LLM. It runs multi-step processes where each step can be plain Python, an API call, an LLM prompt, a full FSM chat, an agent, a timer, or a wait for an outside event.

## What it is for

Some jobs are not a single conversation but a pipeline: take an order, check stock, ask a payment service, wait for confirmation, then send an email. This package lets you describe such a pipeline as named steps in Python, each step saying which step comes next. The engine runs the steps, keeps a shared context dictionary between them, pauses when a step waits for a timer or event, and records a history of every step. Steps can wrap FSM-LLM conversations (the core `fsm_llm` package, which runs chatbots as finite state machines) and agents from `fsm_llm_agents`.

## How it works

```mermaid
flowchart TD
    D[WorkflowDefinition: steps + initial step] --> R[engine.register_workflow validates it]
    R --> S[start_workflow creates an instance with a context dict]
    S --> X[run current step]
    X -- result.next_state --> X
    X -- waits for event --> W[WAITING until process_event]
    X -- waits for timer --> T[WAITING until timer fires]
    W --> X
    T --> X
    X -- terminal step or next_state == '' --> C[COMPLETED]
    X -- failure with no error route --> F[FAILED]
```

Each step returns a result with `success`, some `data` (merged into the context), and `next_state`. The engine follows `next_state` until a step has nowhere to go. An instance moves through the statuses `pending`, `running`, `waiting`, and ends as `completed`, `failed`, or `cancelled`.

Step types: automatic (run a function, move on), API call, condition (yes/no branch), switch (branch on a context value), LLM processing, wait for event, timer, FSM conversation, agent, parallel (run several steps at once), and retry (re-run a step with backoff).

## Files

- `engine.py` - `WorkflowEngine`: runs instances, handles events and timers, cancels, cleans up.
- `steps.py` - the 11 step classes.
- `dsl.py` - short builder functions (`create_workflow`, `auto_step`, `condition_step`, ...) and three helpers that register a set of steps (they do not wire the steps together).
- `definitions.py` - `WorkflowDefinition` with validation (targets exist, all steps reachable), and `WorkflowValidator`.
- `models.py` - statuses, events, step results, instances, event listeners, wait settings.
- `dependency_resolver.py` - `DependencyResolver`: orders steps into "waves" that can run in parallel. It is a standalone helper; the engine does not use it.
- `exceptions.py` - error classes.
- `constants.py` - engine context keys, limits and defaults.
- `__init__.py`, `__version__.py`, `py.typed` - public exports, version, type marker.

## How to use it

```python
import asyncio
from fsm_llm_workflows import WorkflowEngine, auto_step, condition_step, create_workflow

wf = create_workflow("orders", "Order check")
wf.with_initial_step(auto_step("load", "Load order", next_state="route",
                               action=lambda ctx: {"amount": 1500}))
wf.with_step(condition_step("route", "Big order?",
                            condition=lambda ctx: ctx["amount"] >= 1000,
                            true_state="review", false_state="done"))
wf.with_step(auto_step("review", "Manual review", next_state="done",
                       action=lambda ctx: {"reviewed": True}))
wf.with_step(auto_step("done", "Finish", next_state=""))

async def main():
    engine = WorkflowEngine()
    engine.register_workflow(wf)
    instance_id = await engine.start_workflow("orders")
    print(engine.get_workflow_status(instance_id), engine.get_workflow_context(instance_id))

asyncio.run(main())
```

## Things to know

- Everything is `async`. Plain functions (actions, conditions, agents) run in a thread pool (pass `WorkflowEngine(executor=...)` to choose it). A function that returns a coroutine is awaited. A timeout cannot stop a plain function that is still running in its thread.
- A step returning `next_state=""` ends the workflow there.
- A failed step goes to its `error_state` (or `failure_state` for API steps). With no error route the workflow FAILS; it never continues down the success route.
- Loops are allowed when they pass through a timer or event wait (polling, retry-until). A loop made only of steps that run back to back is rejected when the workflow is registered. One engine call runs at most `max_steps_per_run` steps (default 1000).
- Events: `process_event` wakes every instance waiting for that event type, or only one instance when `WorkflowEvent.instance_id` is set (such a targeted event is kept until that instance waits for it). `wait_event_step(..., correlation_key="order_id")` only accepts events whose payload `order_id` equals the instance's own. A wait with `timeout_seconds` and no `timeout_state` fails the instance when it times out.
- Keys a step returns that look internal (starting with `_`, `system_`, `internal_`, `__`) are dropped from the context, except the engine's own wait and timer markers.
- Nothing is saved to disk. Instances live in memory; by default the 1000 most recent finished ones are kept (`max_completed_instances`), each with at most 1000 history entries.
- `workflow_timeout` on `start_workflow` limits the whole run, waits included.
- `engine.add_hook(fn)` calls `fn(event_name, instance, data)` on step start, step end and status change.
- Log output is off until `fsm_llm.setup_logging()` or `fsm_llm.enable_debug_logging()` is called, like the core package.
