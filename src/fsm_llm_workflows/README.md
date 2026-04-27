# fsm_llm_workflows — Back-Compat Shim

> This package is a `sys.modules` shim. The canonical home is **`fsm_llm.stdlib.workflows`**.

Per `docs/lambda.md` §11 (M3 reorganisation), the workflows subsystem moved into `fsm_llm/stdlib/workflows/`. The top-level `fsm_llm_workflows` package is preserved as a silent back-compat shim — every legacy import keeps working, with the same module identity.

## Quick Start

```python
# Either of these works — identical objects.
from fsm_llm_workflows import create_workflow, auto_step, llm_step, conversation_step

# (Preferred for new code)
from fsm_llm.stdlib.workflows import create_workflow, auto_step, llm_step, conversation_step

workflow = (
    create_workflow("order_pipeline")
    .add(auto_step("validate", action=validate_order))
    .add(llm_step("summarize", prompt="Summarize: {order}"))
    .add(conversation_step("support", fsm_file="support.json"))
    .build()
)
```

For λ-term factories (M3 slice 3 — `linear_term`, `branch_term`, `switch_term`, `parallel_term`, `retry_term`), see the canonical README.

## Installation

```bash
pip install fsm-llm[workflows]   # Same install path
```

## Where Everything Lives Now

| What | New canonical path |
|------|--------------------|
| `WorkflowEngine`, `Timer` | `fsm_llm.stdlib.workflows.engine` |
| `WorkflowBuilder`, `create_workflow`, `auto_step`, `api_step`, `llm_step`, `condition_step`, `parallel_step`, `retry_step`, `switch_step`, `conversation_step`, `agent_step`, `wait_event_step`, `timer_step` | `fsm_llm.stdlib.workflows.dsl` |
| 11 `WorkflowStep` subclasses | `fsm_llm.stdlib.workflows.steps` |
| `WorkflowDefinition`, `WorkflowValidator` | `fsm_llm.stdlib.workflows.definitions` |
| `WorkflowInstance`, `WorkflowStatus`, `WorkflowEvent`, `WorkflowStepResult`, `EventListener`, `WaitEventConfig` | `fsm_llm.stdlib.workflows.models` |
| `DependencyResolver` | `fsm_llm.stdlib.workflows.dependency_resolver` |
| Exceptions (`WorkflowError` + 8 subclasses) | `fsm_llm.stdlib.workflows.exceptions` |
| **λ-term factories** (M3 slice 3) | `fsm_llm.stdlib.workflows.lam_factories` — `linear_term`, `branch_term`, `switch_term`, `parallel_term`, `retry_term` |

## Documentation

See **`src/fsm_llm/stdlib/workflows/`** for:
- `README.md` — public-facing walkthrough (lifecycle, steps, DSL)
- `CLAUDE.md` — file map, factories, Theorem-2 forms, exceptions, testing

For the architectural thesis behind the move, read `docs/lambda.md` §11 and §13 (M3 milestone status).

## License

GPL-3.0-or-later. See [LICENSE](../../LICENSE).
