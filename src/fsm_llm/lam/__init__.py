from __future__ import annotations

"""
fsm_llm.lam — λ-calculus substrate (M1 kernel).

Additive subpackage introducing a typed λ-AST, Python builder DSL, pure
combinator library, planner, β-reduction executor, oracle adapter over
``LiteLLMInterface``, and per-leaf cost accumulator.

See ``docs/lambda.md`` for the full design and theorems. M1 delivers the
kernel only — the FSM-JSON → λ compiler (M2), stdlib reorg (M3), example
migration (M4), and Category-C benchmarks (M5) land in subsequent plans.

Users import via ``from fsm_llm.lam import X``; the top-level ``fsm_llm``
package is intentionally not modified during M1 (D-004).
"""

from .ast import (
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
from .combinators import BUILTIN_OPS, ReduceOp
from .cost import CostAccumulator, LeafCall
from .dsl import (
    abs_,
    app,
    case_,
    concat,
    cross,
    ffilter,
    fix,
    fmap,
    leaf,
    let_,
    peek,
    reduce_,
    split,
    var,
)
from .errors import (
    ASTConstructionError,
    LambdaError,
    OracleError,
    PlanningError,
    TerminationError,
)
from .executor import Executor
from .fsm_compile import compile_fsm
from .oracle import LiteLLMOracle, Oracle
from .planner import Plan, PlanInputs, plan

__all__ = [
    # AST
    "Var",
    "Abs",
    "App",
    "Let",
    "Case",
    "Combinator",
    "CombinatorOp",
    "Fix",
    "Leaf",
    "Term",
    "is_term",
    # DSL
    "var",
    "abs_",
    "app",
    "let_",
    "case_",
    "fix",
    "leaf",
    "split",
    "peek",
    "fmap",
    "ffilter",
    "reduce_",
    "concat",
    "cross",
    # Combinators
    "ReduceOp",
    "BUILTIN_OPS",
    # Planner
    "PlanInputs",
    "Plan",
    "plan",
    # Oracle
    "Oracle",
    "LiteLLMOracle",
    # Cost
    "LeafCall",
    "CostAccumulator",
    # Executor
    "Executor",
    # FSM compiler (M2)
    "compile_fsm",
    # Errors
    "LambdaError",
    "ASTConstructionError",
    "TerminationError",
    "PlanningError",
    "OracleError",
]
