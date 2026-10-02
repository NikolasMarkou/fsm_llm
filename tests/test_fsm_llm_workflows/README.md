# test_fsm_llm_workflows

The pytest suite for `fsm_llm.workflows`, the async workflow engine that lives in `src/fsm_llm/workflows/` of the FSM-LLM repository. It holds 269 tests in 7 files, and none of them call a real LLM or the network.

## What it is for

`fsm_llm.workflows` runs workflows: named graphs of steps (automatic transitions, conditions, API calls, LLM calls, timers, waits for events, parallel groups, retries, switches, agents and FSM conversations). An engine keeps workflow instances in memory and moves them from step to step. This suite checks that each step type behaves correctly on its own, that the engine moves instances through their statuses correctly, and that bugs found in past audits stay fixed.

## How it works

Most tests build a step or a small workflow directly in Python, run it, and check the result:

- Step tests call `await step.execute(context)` on one step and check the returned `WorkflowStepResult` (`success`, `next_state`, `data`, `error`).
- Engine tests build a workflow with `create_workflow(...)`, `with_initial_step(...)` and `with_step(...)`, register it on a fresh `WorkflowEngine()`, call `start_workflow(...)`, then check the instance status (`PENDING`, `RUNNING`, `WAITING`, `COMPLETED`, `FAILED`, `CANCELLED`) and its context.
- LLMs, agents and the core `fsm_llm.API` are replaced by small fake classes or `unittest.mock.MagicMock`. `ConversationStep` tests patch `fsm_llm.API`.
- Timing tests use real sleeps and tiny timeouts (0.05 s to 0.5 s). Most sleeps are short (0.01 s to 1 s); the 8 `slow` tests in `test_step_timeouts.py` sleep 5.0 s and rely on a 0.1 s step timeout to cut them off.

Async tests need no decorator: the repository's `pyproject.toml` sets `asyncio_mode = "auto"`.

## Files

- `test_workflows.py` - exceptions, models (`WorkflowStatus`, `WorkflowEvent`, `WorkflowStepResult`, `WorkflowInstance`, `EventListener`, `WaitEventConfig`), definition validation, basic DSL, internal-key filtering, terminal routing on `""`, nested-step serialization, package exports. 41 tests.
- `test_dsl.py` - every DSL factory (`auto_step`, `api_step`, `llm_step`, `timer_step`, ...), the `WorkflowBuilder` fluent builder, and the `linear_workflow`, `conditional_workflow`, `event_driven_workflow` helpers. The builder classes pin that `build()` always validates (an empty builder is refused), that step call order is kept (the last `set_initial_step` is initial but the step keeps its position), that every failure inside `build()` is a `BuildError` chained from its cause, and that each build is isolated (mutating the builder after a build, or the product before the next build, touches neither; steps stay shared by reference). 41 tests.
- `test_steps.py` - execution of `AutoTransitionStep`, `ConditionStep`, `APICallStep`, `WaitForEventStep`, `TimerStep`, `ParallelStep`, `ConversationStep`. 25 tests.
- `test_new_steps.py` - `SwitchStep`, `RetryStep`, `AgentStep`, their DSL factories, and engine instance removal and purging. 46 tests.
- `test_step_timeouts.py` - the per-step `timeout` field and the `_with_timeout` helper. 21 tests, 8 marked `slow`.
- `test_audit_fixes.py` - regressions for earlier audit findings: parallel failure reporting, event listener cleanup, conversation cleanup, template errors, per-instance locking, terminal-status guards, float timeouts. 21 tests.
- `test_audit_2026_09_27.py` - regressions for the 2026-09-27 audit (findings labelled H, M, L), grouped one class per finding. 74 tests.
- `__init__.py` - empty package marker.

## How to use it

From the repository root, with the project virtualenv:

```bash
.venv/bin/python -m pytest tests/test_fsm_llm_workflows/
.venv/bin/python -m pytest tests/test_fsm_llm_workflows/ -m "not slow"
.venv/bin/python -m pytest tests/test_fsm_llm_workflows/test_steps.py -k ParallelStep
```

The full run takes about 10 to 15 seconds. `-m "not slow"` skips the 8 slow timeout tests (261 run).

## Things to know

- Tests import from `fsm_llm.workflows` (a subpackage of `fsm_llm`), not from a separate `fsm_llm_workflows` package.
- The 8 `slow` tests each wait for a 0.1 s timeout on a 5 s sleep.
- `test_audit_2026_09_27.py::TestLoggingOffByDefault` starts a Python subprocess to check that workflow logs stay off until `fsm_llm.logging.enable_library_logging()` is called.
- Several concurrency tests repeat a race 5 or 20 times and accept either outcome of the race, but always check that a step body ran at most once and that nothing raised.
