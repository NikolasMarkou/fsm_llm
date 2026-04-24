# ruff: noqa: RUF002, RUF003
from __future__ import annotations

"""Tests for fsm_llm.lam.planner — purity + Theorem 2 / Theorem 4 numerics."""

import pathlib
import re

import pytest

from fsm_llm.lam.errors import PlanningError
from fsm_llm.lam.planner import Plan, PlanInputs, plan


class TestPurity:
    """SC6: planner imports no LLM/pipeline/fsm module (structural)."""

    def test_no_llm_import(self) -> None:
        src = pathlib.Path("src/fsm_llm/lam/planner.py").read_text()
        # Reject any `from fsm_llm.llm`, `from fsm_llm.pipeline`,
        # `from fsm_llm.fsm`, `import litellm`, etc.
        for pattern in [
            r"^from fsm_llm\.llm\b",
            r"^from fsm_llm\.pipeline\b",
            r"^from fsm_llm\.fsm\b",
            r"^import litellm\b",
            r"^from litellm\b",
        ]:
            assert not re.search(pattern, src, re.MULTILINE), (
                f"planner.py must be pure; found forbidden import matching {pattern!r}"
            )


class TestDefaultK:
    """Thm 4: under linear cost (α=1), k* = 2."""

    def test_k_star_linear_cost(self) -> None:
        p = plan(PlanInputs(n=10_000, K=8192, tau=512, alpha=1.0))
        assert p.k_star == 2

    def test_k_star_sublinear_cost(self) -> None:
        # α < 1 → objective increasing in k; minimum at k=2 (Thm 4 still holds).
        p = plan(PlanInputs(n=10_000, K=8192, tau=512, alpha=0.8, max_k=8))
        assert p.k_star == 2

    def test_k_star_superlinear_prefers_larger_k(self) -> None:
        # α > 1 → objective decreasing in k; optimum at max_k.
        p = plan(PlanInputs(n=10_000, K=8192, tau=512, alpha=1.5, max_k=8))
        assert p.k_star == 8


class TestDepth:
    def test_depth_zero_when_n_leq_tau(self) -> None:
        p = plan(PlanInputs(n=100, K=8192, tau=512))
        assert p.d == 0
        assert p.predicted_calls == 1

    def test_depth_grows_logarithmically(self) -> None:
        # n=4096, tau=512, k=2 → d = ceil(log2(4096/512)) = ceil(log2(8)) = 3.
        p = plan(PlanInputs(n=4096, K=8192, tau=512, alpha=1.0))
        assert p.d == 3
        assert p.predicted_calls == 2**3  # = 8

    def test_depth_k4(self) -> None:
        # α=2 → k* = max_k; with max_k=4, n=4096, tau=512:
        # d = ceil(log4(8)) = ceil(1.5) = 2
        p = plan(PlanInputs(n=4096, K=8192, tau=512, alpha=2.0, max_k=4))
        assert p.k_star == 4
        assert p.d == 2
        assert p.predicted_calls == 4**2  # = 16


class TestPredictedCalls:
    """SC2 corresponds to predicted_calls == actual in the executor."""

    def test_predicted_calls_is_k_pow_d(self) -> None:
        for n, tau, alpha, max_k in [
            (1000, 100, 1.0, 2),
            (10_000, 512, 1.0, 2),
            (1024, 128, 2.0, 4),
        ]:
            p = plan(PlanInputs(n=n, K=999_999, tau=tau, alpha=alpha, max_k=max_k))
            expected = 1 if p.d == 0 else p.k_star**p.d
            assert p.predicted_calls == expected


class TestPredictedCost:
    def test_cost_is_finite_and_positive(self) -> None:
        p = plan(PlanInputs(n=10_000, K=8192, tau=512, alpha=1.0, c=1.0))
        assert p.predicted_cost > 0
        assert p.predicted_cost < float("inf")

    def test_cost_zero_for_zero_input(self) -> None:
        p = plan(PlanInputs(n=0, K=8192, tau=512, alpha=1.0, c=1.0))
        assert p.d == 0
        assert p.predicted_cost == 0.0

    def test_cost_linear_scales_with_n(self) -> None:
        # Under linear cost, depth-0 case: cost == c * n.
        p1 = plan(PlanInputs(n=100, K=1_000_000, tau=1000, alpha=1.0, c=1.0))
        p2 = plan(PlanInputs(n=200, K=1_000_000, tau=1000, alpha=1.0, c=1.0))
        assert p1.d == 0 and p2.d == 0
        assert p2.predicted_cost == pytest.approx(2 * p1.predicted_cost)


class TestAccuracyFloor:
    def test_floor_monotone_decreasing_in_d(self) -> None:
        p1 = plan(
            PlanInputs(
                n=1000, K=8192, tau=500, leaf_accuracy=0.9, combine_accuracy=0.95
            )
        )
        p2 = plan(
            PlanInputs(
                n=10_000, K=8192, tau=500, leaf_accuracy=0.9, combine_accuracy=0.95
            )
        )
        assert p1.d < p2.d
        assert p1.accuracy_floor > p2.accuracy_floor


class TestFeasibility:
    def test_infeasible_when_tau_exceeds_K(self) -> None:
        # If leaf-level pieces don't fit in K, planner must raise.
        # n=10000, K=100, tau=50, k=2: leaf_size = 10000/2^d; need d s.t. leaf_size ≤ 100.
        # But tau=50 forces d = ceil(log2(10000/50)) = 8 → leaf_size = 10000/256 ≈ 39 ≤ 100. OK.
        # To force infeasibility: tau > K.
        with pytest.raises(PlanningError):
            plan(PlanInputs(n=10_000, K=50, tau=200, alpha=1.0, max_k=2))


class TestPlanSerialisable:
    def test_plan_is_frozen(self) -> None:
        p = plan(PlanInputs(n=100, K=8192, tau=512))
        with pytest.raises(Exception):  # pydantic ValidationError
            p.k_star = 999  # type: ignore[misc]

    def test_plan_roundtrip(self) -> None:
        p = plan(PlanInputs(n=4096, K=8192, tau=512))
        restored = Plan.model_validate(p.model_dump())
        assert restored == p
