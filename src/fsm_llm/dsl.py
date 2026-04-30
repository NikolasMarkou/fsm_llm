from __future__ import annotations

"""fsm_llm.dsl — λ-calculus builder DSL.

Thin re-export of the term builders from ``fsm_llm.runtime.dsl``. Canonical
import path for term-mode authoring since 0.9.0:

    from fsm_llm.dsl import leaf, var, abs_, app, let, case_, fix

Naming convention:

- ``abs_`` and ``case_`` keep the trailing underscore (they collide with the
  ``abs()`` builtin and with ``case`` as a soft keyword in match statements)
- ``let`` and ``reduce`` lose the trailing underscore (no collision; renamed
  at 0.9.0 from ``let_`` / ``reduce_``)
- ``fmap`` and ``ffilter`` keep the ``f``-prefix (Haskell-style; ``map`` and
  ``filter`` are true builtins)
"""

from .runtime.dsl import (
    abs_,
    app,
    case_,
    concat,
    cross,
    ffilter,
    fix,
    fmap,
    host_call,
    leaf,
    let,
    peek,
    reduce,
    split,
    var,
)

__all__ = [
    "var",
    "abs_",
    "app",
    "let",
    "case_",
    "fix",
    "leaf",
    "split",
    "peek",
    "fmap",
    "ffilter",
    "reduce",
    "concat",
    "cross",
    "host_call",
]
