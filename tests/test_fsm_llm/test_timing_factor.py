"""The ``timing_factor`` fixture (``tests/conftest.py``, plan d454fd62 step 30).

Every condition is simulated: the tests patch ``sys.gettrace``, the environment
and the coverage probe, so they pass the same way plain and under ``--cov``.
"""

from __future__ import annotations

import os
import sys

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
