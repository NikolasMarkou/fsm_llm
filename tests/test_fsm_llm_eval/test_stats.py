"""Tests for fsm_llm.eval.stats: Wilson interval, Fisher exact, pass_rate."""

from __future__ import annotations

import pytest

from fsm_llm.eval import fisher_exact_two_sided, pass_rate, wilson_ci
from fsm_llm.eval.stats import below_percent


class TestWilsonCI:
    def test_known_value_33_of_40(self):
        lo, hi = wilson_ci(33, 40)
        assert lo == pytest.approx(0.6805, abs=1e-3)
        assert hi == pytest.approx(0.9125, abs=1e-3)

    def test_bounds_stay_inside_unit_interval_at_the_edges(self):
        lo0, hi0 = wilson_ci(0, 40)
        lon, hin = wilson_ci(40, 40)
        assert lo0 == 0.0 and hi0 < 0.15
        assert lon > 0.85 and hin == 1.0

    def test_n_zero_returns_vacuous_interval(self):
        assert wilson_ci(0, 0) == (0.0, 1.0)

    @pytest.mark.parametrize(("k", "n"), [(5, 4), (-1, 4), (0, -1)])
    def test_impossible_counts_raise(self, k: int, n: int):
        with pytest.raises(ValueError):
            wilson_ci(k, n)

    def test_wider_z_gives_wider_interval(self):
        lo95, hi95 = wilson_ci(10, 20)
        lo99, hi99 = wilson_ci(10, 20, z=2.576)
        assert lo99 < lo95 and hi99 > hi95


class TestFisherExact:
    def test_known_value_2_5_vs_0_5(self):
        assert fisher_exact_two_sided(2, 5, 0, 5) == pytest.approx(0.4444, abs=1e-3)

    def test_known_value_5_10_vs_0_10(self):
        assert fisher_exact_two_sided(5, 10, 0, 10) == pytest.approx(
            504 / 15504, abs=1e-4
        )

    def test_symmetric_in_arm_order(self):
        a = fisher_exact_two_sided(32, 40, 20, 40)
        b = fisher_exact_two_sided(20, 40, 32, 40)
        assert a == pytest.approx(b)

    def test_identical_arms_give_p_one(self):
        assert fisher_exact_two_sided(20, 40, 20, 40) == pytest.approx(1.0)

    def test_extreme_split_is_small(self):
        assert fisher_exact_two_sided(10, 10, 0, 10) < 1e-4

    @pytest.mark.parametrize(
        "args", [(1, 2, 0, 0), (3, 2, 1, 2), (-1, 2, 1, 2), (1, 2, 3, 2)]
    )
    def test_impossible_arms_raise(self, args):
        with pytest.raises(ValueError):
            fisher_exact_two_sided(*args)


class TestPassRate:
    def test_shape_and_values(self):
        result = pass_rate(33, 40)
        assert set(result) == {"k", "n", "rate", "wilson_ci"}
        assert result["k"] == 33 and result["n"] == 40
        assert result["rate"] == pytest.approx(0.825)
        assert result["wilson_ci"] == list(wilson_ci(33, 40))

    def test_zero_trials_is_vacuous_not_a_measured_zero(self):
        assert pass_rate(0, 0) == {"k": 0, "n": 0, "rate": 0.0, "wilson_ci": [0.0, 1.0]}

    def test_impossible_counts_raise(self):
        with pytest.raises(ValueError):
            pass_rate(3, 2)


class TestBelowPercent:
    """``--fail-under`` must compare exactly: float division misreads these."""

    @pytest.mark.parametrize(
        ("k", "n", "percent"),
        [(57, 100, 57), (116, 200, 58), (29, 50, 58), (573, 1000, 57.3), (1, 1, 100)],
    )
    def test_exact_threshold_is_not_below(self, k, n, percent):
        assert below_percent(k, n, percent) is False

    def test_float_division_would_misfire_here(self):
        assert 57 / 100 * 100 < 57 and 116 / 200 * 100 < 58  # why ints are used

    @pytest.mark.parametrize(
        ("k", "n", "percent"),
        [(56, 100, 57), (115, 200, 58), (572, 1000, 57.3), (99, 100, 100)],
    )
    def test_one_short_is_below(self, k, n, percent):
        assert below_percent(k, n, percent) is True

    def test_empty_counts_as_zero_percent(self):
        assert below_percent(0, 0, 0) is False
        assert below_percent(0, 0, 0.5) is True
