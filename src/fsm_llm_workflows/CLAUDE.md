# fsm_llm_workflows — Back-Compat Shim

> This package is a `sys.modules` shim. The real implementation lives at **`fsm_llm.stdlib.workflows`**.

Per `docs/lambda.md` §11: the post-unification home of `fsm_llm_workflows` is `fsm_llm/stdlib/workflows/`. This top-level path is preserved for back-compat — every existing `from fsm_llm_workflows import X` and `from fsm_llm_workflows.engine import Y` resolves to the same module object via `sys.modules` aliasing. No deprecation warning; no behavioural difference.

## How the Shim Works

`__init__.py` re-exports the full public surface from `fsm_llm.stdlib.workflows` and registers `sys.modules["fsm_llm_workflows.<submodule>"] = fsm_llm.stdlib.workflows.<submodule>` for the internal submodules. Module identity is preserved — mock-patches on either path take effect on the other.

## Where to Look for Documentation

- **Canonical docs**: `src/fsm_llm/stdlib/workflows/CLAUDE.md` — file map, 5 λ-factories (`linear_term`, `branch_term`, `switch_term`, `parallel_term`, `retry_term`), 11 `WorkflowStep` subclasses, DSL functions, exceptions, testing.
- **Public README**: `src/fsm_llm/stdlib/workflows/README.md`.
- **Architectural rationale**: `docs/lambda.md` §11.

## What's Here in This Directory

```
src/fsm_llm_workflows/
├── __init__.py        # The shim itself (re-exports + sys.modules aliases)
├── __version__.py     # Imports __version__ from fsm_llm.stdlib.workflows
├── py.typed           # PEP 561 marker
└── CLAUDE.md          # This file (pointer to canonical docs)
```

Nothing else. There is no source code in this directory.

## Migration Note for New Code

```python
# Old, still works
from fsm_llm_workflows import create_workflow, auto_step, linear_term

# New (preferred)
from fsm_llm.stdlib.workflows import create_workflow, auto_step, linear_term
```

The shim will stay in place indefinitely — no removal milestone is planned.
