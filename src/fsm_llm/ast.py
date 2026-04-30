from __future__ import annotations

"""fsm_llm.ast — typed λ-AST node types.

Thin re-export of the AST node classes from ``fsm_llm.runtime.ast``. This is
the canonical import path for AST-shape consumers (validators, splicers,
test fixtures) since 0.9.0:

    from fsm_llm.ast import Term, Var, Abs, App, Let, Case, Leaf, Fix

The legacy top-level imports (``from fsm_llm import Term``) were retired at
0.9.0; use this module or ``fsm_llm.runtime`` directly.
"""

from .runtime.ast import (
    Abs,
    App,
    Case,
    Combinator,
    CombinatorOp,
    Fix,
    Leaf,
    Let,
    Term,
    Var,
    is_term,
)
from .runtime.combinators import ReduceOp
from .runtime.cost import LeafCall

__all__ = [
    "Term",
    "Var",
    "Abs",
    "App",
    "Let",
    "Case",
    "Combinator",
    "CombinatorOp",
    "ReduceOp",
    "Leaf",
    "Fix",
    "LeafCall",
    "is_term",
]
