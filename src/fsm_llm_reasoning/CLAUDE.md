# fsm_llm_reasoning — Back-Compat Shim

> This package is a `sys.modules` shim. The real implementation lives at **`fsm_llm.stdlib.reasoning`**.

Per `docs/lambda.md` §11: the post-unification home of `fsm_llm_reasoning` is `fsm_llm/stdlib/reasoning/`. This top-level path is preserved for back-compat — every existing `from fsm_llm_reasoning import X` and `from fsm_llm_reasoning.engine import Y` resolves to the same module object via `sys.modules` aliasing. No deprecation warning; no behavioural difference.

## How the Shim Works

`__init__.py` (89 LOC) does three things:

1. Re-exports every public symbol from `fsm_llm.stdlib.reasoning`.
2. Registers `sys.modules["fsm_llm_reasoning.<submodule>"] = fsm_llm.stdlib.reasoning.<submodule>` for all 7 internal submodules.
3. Binds each submodule as a package attribute so `from fsm_llm_reasoning import engine` works as well as `from fsm_llm_reasoning.engine import X`.

Module identity is preserved — a test that mock-patches `fsm_llm_reasoning.engine.SomeClass` patches the same object that `fsm_llm.stdlib.reasoning.engine.SomeClass` refers to. Patches landed on either path take effect on the other.

## Where to Look for Documentation

- **Canonical docs**: `src/fsm_llm/stdlib/reasoning/CLAUDE.md` — file map, λ-factories, ReasoningEngine, exceptions, testing.
- **Public README**: `src/fsm_llm/stdlib/reasoning/README.md` (if you need a usage walkthrough).
- **Architectural rationale**: `docs/lambda.md` §11 (sub-package reorganisation table).

## What's Here in This Directory

```
src/fsm_llm_reasoning/
├── __init__.py        # The shim itself (re-exports + sys.modules aliases)
├── __version__.py     # Imports __version__ from fsm_llm.stdlib.reasoning
└── CLAUDE.md          # This file (pointer to the canonical docs)
```

Nothing else. There is no source code in this directory.

## Migration Note for New Code

Prefer importing from the canonical home for new code:

```python
# Old, still works
from fsm_llm_reasoning import ReasoningEngine, analytical_term

# New (preferred)
from fsm_llm.stdlib.reasoning import ReasoningEngine, analytical_term
```

The shim will stay in place indefinitely — no removal milestone is planned.
