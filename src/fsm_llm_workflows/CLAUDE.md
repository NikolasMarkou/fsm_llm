# fsm_llm_workflows — Workflow Extension

## What This Package Does

Workflow orchestration engine built on FSM-LLM. Enables automated state transitions, event-driven workflows, external API integration, timers, and parallel execution. Provides a Python DSL for defining workflows.

## File Map

| File | Purpose |
|------|---------|
| `engine.py` | **WorkflowEngine** — core execution engine. Timer management, workflow instance lifecycle |
| `dsl.py` | Python DSL: `create_workflow()`, `auto_step()`, `api_step()`, `llm_step()`, **WorkflowBuilder** class, factory functions |
| `steps.py` | Step types: **WorkflowStep** (base), AutoTransitionStep, APICallStep, ConditionStep, LLMProcessingStep, WaitForEventStep, TimerStep, ParallelStep, **ConversationStep** |
| `definitions.py` | **WorkflowDefinition** with validation (reachability, cycles, state transitions) |
| `models.py` | **WorkflowStatus** enum, WorkflowEvent, WorkflowStepResult, WorkflowInstance, EventListener |
| `handlers.py` | Handler integration module (handlers removed — engine manages operations directly) |
| `exceptions.py` | **WorkflowError** hierarchy: Definition, Step, Instance, Timeout, Validation, State, Event, Resource |
| `__init__.py` | Public exports — single `__all__` list |

## Key Patterns

### DSL Functions
```python
from fsm_llm_workflows import create_workflow, auto_step, llm_step, condition_step

wf = create_workflow("my_workflow", initial_state="start")
wf.with_step(auto_step("start", "process", action=my_func))
wf.with_step(llm_step("process", "done", prompt_template="..."))
```

### Factory Functions (for common patterns)
- `linear_workflow(name, steps)` — sequential steps
- `conditional_workflow(name, ...)` — branching based on conditions
- `event_driven_workflow(name, ...)` — event-based triggers

### WorkflowBuilder (Fluent API)
```python
workflow_builder("name").add_step(auto_step(...)).add_step(...).build()
```

### WorkflowStatus Lifecycle
`PENDING` → `RUNNING` → `WAITING` (for events/timers) → `COMPLETED` | `FAILED` | `CANCELLED`

### WorkflowDefinition Validation
- `validate()` checks: steps exist, state transitions valid, all states reachable, terminal states exist
- `has_cycles()` detects cycles in workflow graph

### ConversationStep (FSM integration)
```python
from fsm_llm_workflows import conversation_step

step = conversation_step(
    "collect_data", "Data Collection",
    fsm_file="form.json",
    model="gpt-4o-mini",
    initial_context={"user_name": "name"},  # workflow_key → conversation_key
    context_mapping={"collected_name": "name"},  # conversation_key → workflow_key
    auto_messages=["My name is Alice"],
    success_state="process_data"
)
```

## Dependencies on Core
- `fsm_llm.handlers.HandlerSystem`, `BaseHandler`, `HandlerTiming` — handler infrastructure
- `fsm_llm.logging.logger` — logging
- `fsm_llm.API` — used by ConversationStep for FSM conversation integration
- Optional deps: `networkx` (graph ops), `graphviz` (visualization), `aiofiles` (async file I/O)

## Testing
```bash
pytest tests/test_fsm_llm_workflows/  # Auto-skips if extension not installed
```

## Gotchas
- Steps use `async execute()` — workflows are async
- Install with `pip install fsm-llm[workflows]` for optional deps (networkx, graphviz, aiofiles)
- Tests auto-skip if workflows extension not installed (conftest.py hook)
- Version synced from `fsm_llm.__version__` — not independent
