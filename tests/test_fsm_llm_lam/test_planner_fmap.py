"""
R6.3 — `PlanInputs.fmap_leaf_count` planner extension.

The new field is an additive contribution to ``leaf_calls`` (and therefore
``predicted_calls``). Default 0 preserves byte-equivalent Theorem-2 strict
equality for all existing shapes.
"""

from __future__ import annotations

import pytest

from fsm_llm.runtime.planner import Plan, PlanInputs, plan


def _baseline_inputs(**overrides) -> PlanInputs:
    """Construct a minimal PlanInputs with sensible defaults for these tests."""
    base = {"n": 8, "tau": 4, "K": 8192}
    base.update(overrides)
    return PlanInputs(**base)


class TestFmapLeafCountAdditive:
    """Non-zero ``fmap_leaf_count`` adds to ``leaf_calls`` and ``predicted_calls``."""

    def test_default_zero_preserves_baseline(self):
        inputs = _baseline_inputs()
        baseline: Plan = plan(inputs)
        # Default value is documented as 0 — verify so a future change to the
        # default is caught by this test rather than silently shifting baselines.
        assert inputs.fmap_leaf_count == 0
        assert baseline.predicted_calls == baseline.leaf_calls + baseline.reduce_calls

    def test_fmap_count_adds_to_leaf_calls(self):
        baseline = plan(_baseline_inputs())
        bumped = plan(_baseline_inputs(fmap_leaf_count=3))
        assert bumped.leaf_calls == baseline.leaf_calls + 3
        assert bumped.predicted_calls == baseline.predicted_calls + 3

    def test_fmap_count_does_not_change_reduce_calls(self):
        baseline = plan(_baseline_inputs())
        bumped = plan(_baseline_inputs(fmap_leaf_count=5))
        assert bumped.reduce_calls == baseline.reduce_calls

    def test_fmap_count_does_not_change_other_plan_attrs(self):
        baseline = plan(_baseline_inputs())
        bumped = plan(_baseline_inputs(fmap_leaf_count=2))
        assert bumped.k_star == baseline.k_star
        assert bumped.tau_star == baseline.tau_star
        assert bumped.d == baseline.d
        assert bumped.predicted_cost == baseline.predicted_cost
        assert bumped.accuracy_floor == baseline.accuracy_floor

    @pytest.mark.parametrize("count", [0, 1, 7, 100])
    def test_fmap_count_arithmetic_holds_at_various_scales(self, count):
        baseline = plan(_baseline_inputs())
        bumped = plan(_baseline_inputs(fmap_leaf_count=count))
        assert bumped.predicted_calls == baseline.predicted_calls + count


class TestFmapLeafCountValidation:
    """``fmap_leaf_count`` must be a non-negative int."""

    def test_negative_value_rejected_by_pydantic(self):
        with pytest.raises(Exception):
            _baseline_inputs(fmap_leaf_count=-1)

    def test_zero_accepted_explicitly(self):
        inputs = _baseline_inputs(fmap_leaf_count=0)
        assert inputs.fmap_leaf_count == 0


class TestFmapLeafCountBackwardCompat:
    """Existing PlanInputs constructions without ``fmap_leaf_count`` still work."""

    def test_existing_callers_unaffected(self):
        # No fmap_leaf_count kwarg — old callers transparently get 0.
        result = plan(PlanInputs(n=16, tau=4, K=8192, reduce_calls_per_node=1))
        assert result.predicted_calls > 0
        # leaf_calls = k^d (no fmap contribution).
        assert result.leaf_calls + result.reduce_calls == result.predicted_calls

    def test_combined_with_reduce_calls_per_node(self):
        """Both extension fields compose additively."""
        baseline = plan(_baseline_inputs())
        with_reduce = plan(_baseline_inputs(reduce_calls_per_node=1))
        with_fmap = plan(_baseline_inputs(fmap_leaf_count=2))
        with_both = plan(_baseline_inputs(reduce_calls_per_node=1, fmap_leaf_count=2))

        # Both extensions contribute additively.
        assert with_both.predicted_calls == baseline.predicted_calls + (
            with_reduce.predicted_calls - baseline.predicted_calls
        ) + (with_fmap.predicted_calls - baseline.predicted_calls)
