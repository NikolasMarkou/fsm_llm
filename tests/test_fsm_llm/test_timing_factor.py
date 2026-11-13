"""The ``timing_factor`` fixture (``tests/conftest.py``, plan d454fd62 step 30).

Every condition is simulated: the tests patch ``sys.gettrace``, the environment
and the coverage probe, so they pass the same way plain and under ``--cov``.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

from tests import conftest
from tests.conftest import (
    TIMING_FACTOR_ENV,
    TRACED_TIMING_FACTOR,
    current_timing_factor,
)


@pytest.fixture
def plain(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """No tracer, no pytest-cov, no coverage, no override: the plain state."""
    monkeypatch.setattr(sys, "gettrace", lambda: None)
    monkeypatch.delenv("COV_CORE_SOURCE", raising=False)
    monkeypatch.delenv(TIMING_FACTOR_ENV, raising=False)
    monkeypatch.setattr(conftest, "_coverage_running", lambda: False)
    return monkeypatch


class TestTimingFactor:
    def test_it_is_one_when_nothing_is_tracing(self, plain):
        assert current_timing_factor() == 1.0

    def test_a_tracer_makes_it_five(self, plain):
        plain.setattr(sys, "gettrace", lambda: object())
        assert current_timing_factor() == TRACED_TIMING_FACTOR == 5.0

    def test_pytest_cov_environment_makes_it_five(self, plain):
        plain.setenv("COV_CORE_SOURCE", "fsm_llm")
        assert current_timing_factor() == 5.0

    def test_a_running_coverage_without_a_tracer_makes_it_five(self, plain):
        """Python 3.12+ measures through ``sys.monitoring``: no ``gettrace``."""
        plain.setattr(conftest, "_coverage_running", lambda: True)
        assert sys.gettrace() is None
        assert current_timing_factor() == 5.0

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("8", 8.0), ("2.5", 2.5), (" 12 ", 12.0), ("1", 1.0)],
    )
    def test_the_override_raises_the_plain_factor(self, plain, raw, expected):
        plain.setenv(TIMING_FACTOR_ENV, raw)
        assert current_timing_factor() == expected

    def test_the_override_raises_the_traced_factor_only_above_it(self, plain):
        plain.setattr(sys, "gettrace", lambda: object())
        plain.setenv(TIMING_FACTOR_ENV, "20")
        assert current_timing_factor() == 20.0
        plain.setenv(TIMING_FACTOR_ENV, "2")
        assert current_timing_factor() == 5.0

    @pytest.mark.parametrize("raw", ["0.5", "0.01"])
    def test_an_override_below_one_cannot_lower_the_floor(self, plain, raw):
        plain.setenv(TIMING_FACTOR_ENV, raw)
        assert current_timing_factor() == 1.0

    @pytest.mark.parametrize(
        "raw", ["", "  ", "fast", "0", "-3", "nan", "inf", "-inf", "1e999"]
    )
    def test_an_invalid_override_is_ignored(self, plain, raw):
        plain.setenv(TIMING_FACTOR_ENV, raw)
        assert current_timing_factor() == 1.0
        plain.setattr(sys, "gettrace", lambda: object())
        assert current_timing_factor() == 5.0

    def test_a_probe_that_raises_means_not_running(self, monkeypatch):
        class _Boom:
            class Coverage:
                @staticmethod
                def current():
                    raise RuntimeError("coverage is broken")

        monkeypatch.setattr(sys, "gettrace", lambda: None)
        monkeypatch.delenv("COV_CORE_SOURCE", raising=False)
        monkeypatch.delenv(TIMING_FACTOR_ENV, raising=False)
        monkeypatch.setitem(sys.modules, "coverage", _Boom)
        assert conftest._coverage_running() is False
        assert current_timing_factor() == 1.0

    def test_the_fixture_returns_the_current_value(self, timing_factor):
        assert isinstance(timing_factor, float)
        assert timing_factor == current_timing_factor()
        assert timing_factor >= 1.0

    def test_the_fixture_is_five_under_the_real_coverage_run(self, timing_factor):
        """On a ``--cov`` run the real conditions, unpatched, must read as traced."""
        if (
            conftest._coverage_running()
            or sys.gettrace() is not None
            or "COV_CORE_SOURCE" in os.environ
        ):
            assert timing_factor >= TRACED_TIMING_FACTOR
        else:
            assert timing_factor >= 1.0


class TestNoRatioBoundIsScaled:
    """Review gate 1 (plan d454fd62 step 58.1, concern 6): a ratio bound is never
    multiplied by ``timing_factor``.

    A tracer or a loaded machine slows every size of a sweep by the same factor,
    so the ratio of two sizes does not move; scaling its bound only loosens it
    (``_LINEAR_SPREAD = 2.5`` became 12.5 under coverage, and
    ``FSM_LLM_TEST_TIMING_FACTOR=20`` made a quadratic crypto-gap regex pass).
    Only an absolute time budget scales. A noisy ratio is retried, never widened.
    """

    _RATIO_WORDS = ("spread", "ratio", "growth", "slope")

    @staticmethod
    def _names(node: ast.AST) -> set[str]:
        return {
            n.id if isinstance(n, ast.Name) else n.attr
            for n in ast.walk(node)
            if isinstance(n, ast.Name | ast.Attribute)
        }

    def _scaled_ratios(self, source: str) -> list[int]:
        """Line numbers where ``timing_factor`` meets a ratio name: in a product
        (``bound = _LINEAR_SPREAD * timing_factor``) or in one comparison
        (``assert spread <= 2.0 * timing_factor``)."""
        hits: set[int] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                parts = [self._names(node.left), self._names(node.right)]
            elif isinstance(node, ast.Compare):
                parts = [self._names(n) for n in (node.left, *node.comparators)]
            else:
                continue
            names = set().union(*parts)
            if "timing_factor" in names and any(
                word in name.lower() for name in names for word in self._RATIO_WORDS
            ):
                hits.add(node.lineno)
        return sorted(hits)

    def test_the_detector_sees_a_scaled_ratio(self):
        assert self._scaled_ratios("bound = _LINEAR_SPREAD * timing_factor") == [1]
        assert self._scaled_ratios("assert ratio <= 2.0 * timing_factor") == [1]
        assert self._scaled_ratios("limit = timing_factor * growth_cap") == [1]

    def test_the_detector_passes_an_absolute_budget(self):
        assert self._scaled_ratios("assert elapsed < 0.5 * timing_factor") == []
        assert self._scaled_ratios("w = _LINEAR_BUDGET_S * timing_factor") == []

    def test_no_test_file_scales_a_ratio_bound(self):
        offenders = {}
        for path in sorted(Path(conftest.__file__).parent.rglob("*.py")):
            if path == Path(__file__):
                continue
            lines = self._scaled_ratios(path.read_text(encoding="utf-8"))
            if lines:
                offenders[str(path.relative_to(Path(conftest.__file__).parent))] = lines
        assert offenders == {}, (
            "a ratio bound must stay fixed (a uniform slowdown cancels in a "
            "ratio): scale only absolute time budgets"
        )
