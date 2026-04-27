from __future__ import annotations

"""
fsm_llm.runtime — λ-calculus substrate (M1 kernel, post-R4 home).

Renamed from ``fsm_llm.lam`` per plan v3 step 19 (D-PLAN-08, D-PLAN-10).
The previous import path ``fsm_llm.lam`` continues to work via a sys.modules
shim defined in ``fsm_llm/lam/__init__.py``; new code should import from
``fsm_llm.runtime``.

See ``docs/lambda.md`` for the full design and theorems.
"""

# Explicit submodule imports so that the lam → runtime sys.modules shim can
# re-export each submodule (Scenario 9 from plan v3 — submodules need
# individual sys.modules registration). The trailing `as <name>` re-bindings
# are intentional: they make `fsm_llm.runtime.ast` resolve to the module
# object via attribute access, which the lam shim relies on.
# fsm_compile lives in fsm_llm.dialog.compile_fsm post-R4 step 21; re-exported
# from runtime for back-compat with `from fsm_llm.lam import compile_fsm`.
# The module alias `fsm_compile` (= fsm_llm.dialog.compile_fsm) is also bound
# below so that the lam shim's `getattr(_runtime, "fsm_compile")` keeps
# working for `from fsm_llm.lam.fsm_compile import compile_fsm_cached`.
import fsm_llm.dialog.compile_fsm as fsm_compile  # noqa: F401  module alias for lam shim
from fsm_llm.dialog.compile_fsm import compile_fsm, compile_fsm_cached
from fsm_llm.runtime import ast as ast
from fsm_llm.runtime import combinators as combinators
from fsm_llm.runtime import constants as constants
from fsm_llm.runtime import cost as cost
from fsm_llm.runtime import dsl as dsl
from fsm_llm.runtime import errors as errors
from fsm_llm.runtime import executor as executor
from fsm_llm.runtime import oracle as oracle
from fsm_llm.runtime import planner as planner

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
    host_call,
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
from .oracle import LiteLLMOracle, Oracle, StreamingOracle
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
    "host_call",
    # Combinators
    "ReduceOp",
    "BUILTIN_OPS",
    # Planner
    "PlanInputs",
    "Plan",
    "plan",
    # Oracle
    "Oracle",
    "StreamingOracle",
    "LiteLLMOracle",
    # Cost
    "LeafCall",
    "CostAccumulator",
    # Executor
    "Executor",
    # FSM compiler (M2) — kernel + cached front-door (R2)
    "compile_fsm",
    "compile_fsm_cached",
    # Errors
    "LambdaError",
    "ASTConstructionError",
    "TerminationError",
    "PlanningError",
    "OracleError",
]
