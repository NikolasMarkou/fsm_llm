# fsm_llm_workflows -- Claude Code Instructions

## What This Package Does

Async workflow orchestration engine built on FSM-LLM. Enables step-based workflow execution with 11 step types (auto-transition, API call, condition, LLM processing, wait-for-event, timer, parallel, conversation, agent, retry, switch). Workflows are defined via a Python DSL or fluent builder API and executed asynchronously by `WorkflowEngine`. The `ConversationStep` bridges workflows with FSM-LLM core, allowing full FSM conversations to run as workflow steps.

## File Map

| File | Purpose |
|------|---------|
| `engine.py` | `WorkflowEngine` -- core async execution engine. Manages workflow instances, timer scheduling, event listener registration, step execution with recursion depth limit (`MAX_STEP_DEPTH=20`). `Timer` class for scheduled transitions. `remove_instance()` for manual cleanup, `max_completed_instances` for auto-purge. |
| `dsl.py` | Python DSL: `create_workflow()`, `auto_step()`, `api_step()`, `condition_step()`, `llm_step()`, `wait_event_step()`, `timer_step()`, `parallel_step()`, `conversation_step()`, `agent_step()`, `retry_step()`, `switch_step()`. `WorkflowBuilder` fluent class. Factory functions: `linear_workflow()`, `conditional_workflow()`, `event_driven_workflow()`. |
| `steps.py` | 11 step types inheriting from `WorkflowStep` (Pydantic BaseModel + ABC). Each implements `async execute(context) -> WorkflowStepResult`. Base class provides `_with_timeout()` wrapper. `ConversationStep` imports `fsm_llm.API` at runtime. `ParallelStep` deep-copies context for each sub-step. New: `AgentStep` (run agents as steps), `RetryStep` (retry with backoff), `SwitchStep` (n-way branching). |
| `definitions.py` | `WorkflowDefinition` -- Pydantic model with `with_step()`, `with_initial_step()`, `validate()` (checks reachability, cycles, state transitions, terminal states), `has_cycles()`, `get_terminal_states()`, `serialize()`. `WorkflowValidator` static utility. |
| `models.py` | `WorkflowStatus` enum (PENDING, RUNNING, WAITING, COMPLETED, FAILED, CANCELLED) with valid transition map. `WorkflowEvent`, `WorkflowStepResult` (with `success_result()`/`failure_result()` class methods), `WorkflowInstance` (with `update_status()`, `is_active()`, `is_terminal()`), `EventListener`, `WaitEventConfig`. |
| `exceptions.py` | `WorkflowError(FSMError)` base, plus 8 specific exceptions: `WorkflowDefinitionError`, `WorkflowStepError`, `WorkflowInstanceError`, `WorkflowTimeoutError`, `WorkflowValidationError`, `WorkflowStateError`, `WorkflowEventError`, `WorkflowResourceError`. |
| `__init__.py` | Public API -- single `__all__` list with 36 exports. |
| `__version__.py` | Version synced from `fsm_llm.__version__`. |

## Key Patterns

### DSL Functions
```python
from fsm_llm_workflows import create_workflow, auto_step

wf = create_workflow("my_workflow", "My Workflow")
wf.with_initial_step(auto_step("start", "Start", next_state="process", action=my_func))
wf.with_step(auto_step("process", "Process", next_state="done"))
wf.with_step(auto_step("done", "Done", next_state=""))  # empty string = terminal
```

### Factory Functions
- `linear_workflow(workflow_id, name, steps)` -- sequential execution, first step is initial
- `conditional_workflow(workflow_id, name, initial_step, condition_step, true_branch, false_branch)` -- branching
- `event_driven_workflow(workflow_id, name, setup_steps, event_step, processing_steps)` -- event-based

### WorkflowBuilder Fluent API
```python
workflow_builder("wf_id", "Name").set_initial_step(step).add_step(step2).add_metadata("k", "v").build()
```

### WorkflowStatus Lifecycle
`PENDING` -> `RUNNING` -> `WAITING` (events/timers) -> `COMPLETED` | `FAILED` | `CANCELLED`

Valid transitions are enforced by `_VALID_STATUS_TRANSITIONS` dict in `models.py`. Terminal states allow no outgoing transitions. `WorkflowInstance.update_status()` raises `WorkflowStateError` on illegal transitions.

### WorkflowDefinition Validation
`validate()` checks: workflow has steps, initial step exists, all step IDs match keys, all referenced states exist, all states reachable from initial, no cycles in auto-transition chains. Warns (via logger) about states that cannot reach terminal states.

### ConversationStep Integration
Uses `fsm_llm.API` (imported at runtime in `_run_conversation()`). Maps workflow context to conversation initial context via `initial_context` dict, maps results back via `context_mapping` dict. Drives conversation with `auto_messages` list. Raw data stored under `_conversation_{step_id}_data`.

### Async Execution Model
All step `execute()` methods are async. `WorkflowEngine` uses asyncio for concurrent workflow instances, timer scheduling (`asyncio.Task`), and parallel step execution (`asyncio.gather`). Sync callables in steps (actions, conditions, API functions) are wrapped with `loop.run_in_executor()`.

## Dependencies on Core

- `fsm_llm.handlers.HandlerSystem` -- handler infrastructure (engine creates one if not provided)
- `fsm_llm.logging.logger` -- loguru logging
- `fsm_llm.definitions.FSMError` -- base exception class
- `fsm_llm.API` -- used by `ConversationStep` (runtime import) for FSM conversations

## Exception Hierarchy

```
FSMError
  WorkflowError
    WorkflowDefinitionError(workflow_id, message)
    WorkflowStepError(step_id, message, cause)
    WorkflowInstanceError(instance_id, message)
    WorkflowTimeoutError(operation, timeout_seconds)
    WorkflowValidationError(validation_errors: list)
    WorkflowStateError(current_state, operation, message)
    WorkflowEventError(event_type, message)
    WorkflowResourceError(resource_type, resource_id, message)
```

## Testing

```bash
pytest tests/test_fsm_llm_workflows/  # 136 tests in 7 files
```

Tests auto-skip if workflows extension is not installed (conftest.py hook).

| Test File | Count | Focus |
|-----------|-------|-------|
| `test_workflows.py` | 32 | Engine lifecycle, registration, execution, events, cancellation |
| `test_dsl.py` | 27 | DSL functions, factory functions, WorkflowBuilder |
| `test_steps.py` | 24 | Step execution, mapping, error handling, parallel aggregation |
| `test_step_timeouts.py` | 24 | Per-step timeout enforcement for all step types |
| `test_audit_fixes.py` | 9 | Regression tests for audit findings |

## Gotchas

- **Async execution** -- all workflow operations are async. Use `asyncio.run()` to call from sync code. Step `execute()` methods must be awaited.
- **No extra deps** -- `pip install fsm-llm[workflows]` has no additional dependencies beyond core. Tests skip if extension not installed.
- **Version synced from core** -- `__version__.py` imports from `fsm_llm.__version__`, not independently versioned.
- **Terminal steps use empty string** -- a step with `next_state=""` is terminal (empty strings are discarded by `_get_referenced_states()`).
- **Recursion depth limit** -- `MAX_STEP_DEPTH=20` in engine.py prevents infinite auto-transition chains.
- **ParallelStep deep-copies context** -- each sub-step gets an isolated context copy. Warns if >10 parallel steps due to memory.
- **ConversationStep runtime import** -- `fsm_llm.API` is imported inside `_run_conversation()` to avoid circular imports.
- **Step internal whitelist** -- `_waiting_info` and `_timer_info` are whitelisted context keys that bypass the underscore filter.
