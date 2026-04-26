# ruff: noqa: RUF002
from __future__ import annotations

"""
Pure planner for Fix-annotated λ-terms (paper Theorems 2 & 4).

Given: (1) the oracle context window ``K``, (2) the rank (size) ``n`` of
the input payload, (3) a cost model, and (4) an accuracy model, return a
``Plan`` with:

- ``k_star``  — optimal branching factor for SPLIT
- ``tau_star`` — size threshold below which the Leaf base case fires
- ``d``       — planned recursion depth
- ``reduce_op`` — named REDUCE op chosen by the caller (echoed for audit)
- ``predicted_cost`` — total oracle cost in abstract cost units
- ``predicted_calls`` — exact number of Leaf invocations
- ``accuracy_floor`` — lower bound on end-to-end accuracy under the model

Under a **linear cost** model (cost(n) = c·n), paper Theorem 4 gives
``k* = 2`` as the closed-form optimum. Under a **super-linear cost**
model (cost(n) = c·n^α for α > 1), the optimum minimises
``k * cost(n/k) = k * c * (n/k)^α``, yielding a stationary point at
``k* = argmin_{k∈[2..n]} (k * (n/k)^α)``. For α ≤ 1 the objective is
monotonically decreasing in k, so the optimum is bounded by a
per-planner cap (``max_k``) which defaults to 2 (matches Thm 4).

**Purity invariant (I3)**: this module MUST NOT import from
``fsm_llm.llm``, ``fsm_llm.pipeline``, ``fsm_llm.fsm``, or any module
that performs I/O. Verified by SC6 grep.
"""

import math

from pydantic import BaseModel, ConfigDict, Field

from .constants import (
    DEFAULT_COMBINE_ACCURACY,
    DEFAULT_K_STAR,
    DEFAULT_LEAF_ACCURACY,
    DEFAULT_TAU,
    DEFAULT_TOKEN_COST,
)
from .errors import PlanningError


class PlanInputs(BaseModel):
    """Inputs to ``plan()``. All fields are measurable quantities — no
    AST, no runtime state. This keeps the planner pure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int = Field(..., ge=0, description="Input payload rank / size")
    K: int = Field(..., ge=1, description="Oracle context window (tokens)")
    tau: int = Field(
        default=DEFAULT_TAU,
        ge=1,
        description=(
            "Size threshold. Inputs with size ≤ τ take the Leaf base case; "
            "larger inputs are SPLIT and recursed."
        ),
    )
    alpha: float = Field(
        default=1.0,
        gt=0.0,
        description="Cost exponent: cost(n) = c·n^α. α=1 is linear.",
    )
    c: float = Field(
        default=DEFAULT_TOKEN_COST,
        gt=0.0,
        description="Per-token cost constant (for predicted_cost)",
    )
    leaf_accuracy: float = Field(
        default=DEFAULT_LEAF_ACCURACY,
        ge=0.0,
        le=1.0,
        description="Accuracy of a single Leaf call on sub-rank input",
    )
    combine_accuracy: float = Field(
        default=DEFAULT_COMBINE_ACCURACY,
        ge=0.0,
        le=1.0,
        description="Accuracy of the REDUCE/combine step",
    )
    reduce_op_name: str = Field(
        default="best",
        description="Name of the REDUCE op (echoed into Plan for audit)",
    )
    max_k: int = Field(
        default=DEFAULT_K_STAR,
        ge=2,
        description=(
            "Upper bound on k* search. Defaults to 2 (Thm 4 under linear "
            "cost). Raise for super-linear problems."
        ),
    )
    reduce_calls_per_node: int = Field(
        default=0,
        ge=0,
        description=(
            "Oracle calls per reduce *node* (k-arity reduce → (k-1) pair "
            "calls). 0 = pure ReduceOp (no oracle in fold; default, "
            "preserves slice-3 cost equality). Set to 1 for oracle-mediated "
            "pairwise comparison op (M5 slice 5 — see "
            "``stdlib.long_context.oracle_compare_op``)."
        ),
    )


class Plan(BaseModel):
    """Output of ``plan()``. Immutable, serialisable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    k_star: int = Field(..., ge=1)
    tau_star: int = Field(..., ge=1)
    d: int = Field(..., ge=0, description="Planned recursion depth")
    reduce_op_name: str
    predicted_cost: float = Field(..., ge=0.0)
    predicted_calls: int = Field(
        ...,
        ge=0,
        description=(
            "Total predicted oracle calls = leaf_calls + reduce_calls. "
            "Default (reduce_calls_per_node=0) preserves Theorem-2 slice-3 "
            "behaviour: predicted_calls == leaf_calls == k^d."
        ),
    )
    leaf_calls: int = Field(
        ...,
        ge=0,
        description="Exact Leaf invocation count (k^d for d≥1, 1 for d=0).",
    )
    reduce_calls: int = Field(
        ...,
        ge=0,
        description=(
            "Predicted reduce-side oracle calls = (k^d − 1) * "
            "reduce_calls_per_node for d≥1, 0 for d=0. M5 slice 5: when "
            "reduce_calls_per_node=1 (oracle-mediated pairwise compare) "
            "→ predicted_calls = 2·k^d − 1."
        ),
    )
    accuracy_floor: float = Field(..., ge=0.0, le=1.0)


# --------------------------------------------------------------
# Closed-form planner
# --------------------------------------------------------------


def _optimal_k(n: int, tau: int, alpha: float, max_k: int) -> int:
    """Find the k∈[2..max_k] minimising total cost of one recursion level.

    Objective: ``k * (n/k)^α = k^(1-α) * n^α``. For α > 1 this is
    decreasing in k (prefer larger k); for α < 1 increasing (prefer
    smaller); for α = 1 constant (any k gives the same per-level cost,
    so Thm 4 picks k* = 2 to minimise depth-growth).
    """
    if n <= tau or max_k < 2:
        return 2  # degenerate — doesn't matter, surfaced via d=0 upstream
    if math.isclose(alpha, 1.0):
        return 2  # Thm 4
    candidates = range(2, max_k + 1)
    return min(candidates, key=lambda k: k ** (1.0 - alpha) * (n**alpha))


def _depth(n: int, tau: int, k: int) -> int:
    """Smallest ``d`` such that ``n / k^d ≤ tau``."""
    if n <= tau:
        return 0
    if k < 2:
        raise PlanningError(f"cannot make progress with k={k} (needs k ≥ 2)")
    # n / k^d ≤ tau  ⇔  d ≥ log_k(n / tau)
    return math.ceil(math.log(n / tau) / math.log(k))


def _leaf_calls(k: int, d: int) -> int:
    """Exact number of Leaf invocations for a balanced k-ary tree.

    Base case (d=0): one Leaf. Otherwise: ``k^d`` leaves + optional
    combines which are NOT Leaf calls (REDUCE is pure). So the invariant
    matches SC2: ``leaf_calls == k^d`` for d≥1, ``== 1`` for d=0.

    The plan.md's success criterion wording ``(k*)^d + 1`` counts a
    root-level overview call; M1's executor does NOT emit one, so we
    return ``k^d`` for the recursive case and ``1`` for the degenerate
    base-case. SC2 verifies predicted == actual; actual matches this
    function by construction in the executor.
    """
    if d == 0:
        return 1
    return int(k**d)


def _reduce_calls(k: int, d: int, per_node: int) -> int:
    """Predicted reduce-side oracle calls for a balanced k-ary tournament.

    A balanced k-ary reduction tree of depth ``d`` has ``k^i`` reduce
    nodes at level ``i ∈ [0..d-1]``, each folding ``k`` elements →
    ``(k-1)`` pair calls per node when ``per_node == 1``. Summed::

        Σ_{i=0}^{d-1} k^i · (k-1) · per_node = (k^d - 1) · per_node

    For ``d == 0`` (single Leaf, no reduce), returns 0. For
    ``per_node == 0`` (pure ReduceOp — slice-3 default), returns 0.
    Used by ``oracle_compare_op`` (M5 slice 5) where ``per_node == 1``.
    """
    if d == 0 or per_node == 0:
        return 0
    return int((k**d) - 1) * per_node


def _predicted_cost(n: int, tau: int, k: int, d: int, alpha: float, c: float) -> float:
    """Abstract cost: sum over levels 0..d-1 of ``k^level * c * (n/k^level)^α``,
    plus the leaf-level cost of ``k^d`` Leaf calls on inputs of size ≤ tau.
    """
    if d == 0:
        return float(c * (n**alpha))
    total = 0.0
    for level in range(d):
        num_nodes = k**level
        per_node_size = n / (k**level)
        total += num_nodes * c * (per_node_size**alpha)
    # Leaf level: k^d pieces, each ≤ tau.
    leaf_size = n / (k**d)
    total += (k**d) * c * (leaf_size**alpha)
    return float(total)


def _accuracy_floor(d: int, leaf_acc: float, combine_acc: float) -> float:
    """Lower bound on end-to-end accuracy under independent-error model.

    A depth-d plan has d combine steps and one leaf step on each path.
    Independent-error bound: ``leaf_acc * combine_acc^d``.
    """
    return leaf_acc * (combine_acc**d)


def plan(inputs: PlanInputs) -> Plan:
    """Compute a ``Plan`` for a single ``Fix`` node.

    Pure function. Deterministic. No I/O. No LLM calls."""
    if inputs.K <= 0:
        raise PlanningError("K must be a positive int")

    # Feasibility: inputs.n might exceed K for k=1 (trivial non-split).
    # We require that at planned depth d, each leaf fits in K tokens.
    k_star = _optimal_k(inputs.n, inputs.tau, inputs.alpha, inputs.max_k)
    d = _depth(inputs.n, inputs.tau, k_star)

    # Leaf-size feasibility check under the chosen k* and d.
    leaf_size_at_d = inputs.n / (k_star**d) if d > 0 else inputs.n
    if leaf_size_at_d > inputs.K:
        raise PlanningError(
            f"infeasible: leaf-level size {leaf_size_at_d:.0f} exceeds K={inputs.K} "
            f"at k={k_star}, d={d}. Increase max_k or reduce input size."
        )

    leaf_calls = _leaf_calls(k_star, d)
    reduce_calls = _reduce_calls(k_star, d, inputs.reduce_calls_per_node)
    predicted_calls = leaf_calls + reduce_calls
    predicted_cost = _predicted_cost(
        inputs.n, inputs.tau, k_star, d, inputs.alpha, inputs.c
    )
    accuracy_floor = _accuracy_floor(d, inputs.leaf_accuracy, inputs.combine_accuracy)

    return Plan(
        k_star=k_star,
        tau_star=inputs.tau,
        d=d,
        reduce_op_name=inputs.reduce_op_name,
        predicted_cost=predicted_cost,
        predicted_calls=predicted_calls,
        leaf_calls=leaf_calls,
        reduce_calls=reduce_calls,
        accuracy_floor=accuracy_floor,
    )


__all__ = ["PlanInputs", "Plan", "plan"]
