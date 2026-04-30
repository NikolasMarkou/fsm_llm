from __future__ import annotations

"""fsm_llm.combinators — closed-set combinator builders.

Thin re-export of the closed combinator builders. The DSL builders for
combinators (``split``, ``fmap``, ``ffilter``, ``reduce``, ``concat``,
``cross``, ``peek``, ``host_call``) live alongside the rest of the DSL in
``fsm_llm.runtime.dsl``; this module surfaces them under the
``fsm_llm.combinators`` namespace for callers that prefer to separate the
λ-form imports from the combinator imports.

    from fsm_llm.combinators import split, fmap, ffilter, reduce, concat

The closed registry itself (``ReduceOp``, ``BUILTIN_OPS``) is here too.
``BUILTIN_OPS`` is architecturally closed — new ops bind through env at the
call site (see ``stdlib/long_context/pairwise.py`` for the canonical
"new op via env" pattern).
"""

from .runtime.combinators import BUILTIN_OPS, ReduceOp
from .runtime.dsl import (
    concat,
    cross,
    ffilter,
    fmap,
    host_call,
    peek,
    reduce,
    split,
)

__all__ = [
    "ReduceOp",
    "BUILTIN_OPS",
    "split",
    "fmap",
    "ffilter",
    "reduce",
    "concat",
    "cross",
    "peek",
    "host_call",
]
