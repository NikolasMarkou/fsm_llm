# fsm_llm_workflows -- Workflow Orchestration Engine

Async event-driven workflow engine with 11 step types and a Python DSL. Supports API calls, LLM processing, FSM conversations, agent execution, timers, events, parallel execution, retry logic, and conditional routing.

- **Version**: 0.3.0 (synced from fsm_llm)
- **Extra deps**: None beyond core fsm_llm
- **Install**: `pip install fsm-llm[workflows]`

## File Map

```
fsm_llm_workflows/
├── engine.py       # WorkflowEngine + Timer -- async execution, event listeners, timers, instance lifecycle
├── dsl.py          # Python DSL: create_workflow(), WorkflowBuilder, 11 step factories, 3 pattern helpers
├── steps.py        # 11 WorkflowStep subclasses (all async execute(context) -> WorkflowStepResult)
├── definitions.py  # WorkflowDefinition with validation (reachability, cycles), WorkflowValidator
├── models.py       # WorkflowStatus enum, WorkflowEvent, WorkflowStepResult, WorkflowInstance, EventListener, WaitEventConfig
├── exceptions.py   # WorkflowError -> 8 subtypes (Definition, Step, Instance, Timeout, Validation, State, Event, Resource)
├── __version__.py  # Imports from fsm_llm.__version__
└── __init__.py     # 47 public exports
```

## Key Classes

### WorkflowEngine (`engine.py`)

Main orchestrator. All step execution is async.

- Constructor: `__init__(handler_system=None, max_concurrent_workflows=100, max_completed_instances=None)`
- **Lifecycle**: `register_workflow(definition)`, `start_workflow(workflow_id, initial_context, instance_id, workflow_timeout)` → instance_id, `advance_workflow(instance_id, user_input)` → bool, `cancel_workflow(instance_id, reason)`, `shutdown()`
- **Events**: `process_event(WorkflowEvent)` → list[affected_ids], `register_event_listener(instance_id, event_type, ...)`, `schedule_timer(instance_id, delay, next_state)`
- **Queries**: `get_workflow_instance(id)`, `get_workflow_status(id)`, `get_workflow_context(id)`, `get_active_workflows()`, `get_statistics()`
- **Limits**: MAX_STEP_DEPTH=20 (prevents infinite loops), max_concurrent_workflows enforced at start
- **Internal context keys**: `_waiting_info`, `_timer_info`, `_workflow_info`, `_timeout`, `_last_event`, `_user_input`
- Sync functions wrapped via `loop.run_in_executor()`. Async detection via `inspect.iscoroutinefunction()`

### WorkflowDefinition (`definitions.py`)

- Fields: workflow_id, name, description, steps dict, initial_step_id, metadata
- Builder: `with_step(step, is_initial)`, `with_initial_step(step)`
- Validation: `validate()` checks structure, transition targets, reachability (DFS), cycles. Raises WorkflowValidationError
- Analysis: `get_terminal_states()`, `has_cycles()`, `_find_reachable_states(start)`

### WorkflowStatus (`models.py`) -- State Machine

```
PENDING → RUNNING → COMPLETED | FAILED | CANCELLED
                  → WAITING → RUNNING | FAILED | CANCELLED
```

Terminal states: COMPLETED, FAILED, CANCELLED (no outgoing transitions).

### WorkflowInstance (`models.py`)

- Fields: instance_id, workflow_id, current_step_id, context, status, created_at, updated_at, deadline, history
- `update_status(status, error)` -- Validates transitions, updates timestamps, records history
- `is_active()` → RUNNING | WAITING, `is_terminal()` → COMPLETED | FAILED | CANCELLED

## Step Types (`steps.py`)

All inherit `WorkflowStep` ABC. Async `execute(context)` → `WorkflowStepResult`.

| Step | Key Fields | Behavior |
|------|-----------|----------|
| `AutoTransitionStep` | next_state, action (optional callable) | Execute action, go to next_state |
| `APICallStep` | api_function, success/failure_state, input/output_mapping | Map context → params, call API, map results |
| `ConditionStep` | condition (callable), true/false_state | Binary branching |
| `LLMProcessingStep` | llm_interface, prompt_template, context/output_mapping, next/error_state | Format prompt, call LLM, regex-extract results |
| `WaitForEventStep` | config (WaitEventConfig) | Pause until event or timeout |
| `TimerStep` | delay_seconds, next_state | Wait N seconds |
| `ConversationStep` | fsm_file/fsm_definition, model, auto_messages, context_mapping, max_turns | Run FSM conversation |
| `AgentStep` | agent, task_template, context_mapping, success/error_state | Run BaseAgent via executor |
| `ParallelStep` | steps list, next/error_state, aggregation_function | Concurrent execution, merge results |
| `RetryStep` | step (inner), max_retries, backoff_factor | Retry with exponential backoff |
| `SwitchStep` | key, cases dict, default_state | N-way routing on context value |

## DSL Functions (`dsl.py`)

**Workflow creation**: `create_workflow(id, name, desc)` → WorkflowDefinition, `workflow_builder(id, name, desc)` → WorkflowBuilder

**Step factories**: `auto_step()`, `api_step()`, `condition_step()`, `llm_step()`, `wait_event_step()`, `timer_step()`, `parallel_step()`, `conversation_step()`, `agent_step()`, `retry_step()`, `switch_step()`

**Pattern helpers**: `linear_workflow()` (sequential), `conditional_workflow()` (binary branch), `event_driven_workflow()` (setup → wait → process)

## Testing

```bash
pytest tests/test_fsm_llm_workflows/  # 136 tests, 7 files
```

Tests auto-skip if workflows extension not installed.

## Exceptions

```
FSMError
└── WorkflowError(message, details=None)
    ├── WorkflowDefinitionError(workflow_id, message)
    ├── WorkflowStepError(step_id, message, cause=None)
    ├── WorkflowInstanceError(instance_id, message)
    ├── WorkflowTimeoutError(operation, timeout_seconds)
    ├── WorkflowValidationError(validation_errors: list)
    ├── WorkflowStateError(current_state, operation, message)
    ├── WorkflowEventError(event_type, message)
    └── WorkflowResourceError(resource_type, resource_id, message)
```

## Async Patterns

- `asyncio.create_task()` for timers and event timeouts
- `asyncio.gather(return_exceptions=True)` for parallel steps
- `asyncio.wait_for()` for step timeouts
- `asyncio.Lock()` for thread-safe event listener management
- `loop.run_in_executor()` for sync functions (agents, actions, conditions)
