"""The ``timing_factor`` fixture (``tests/conftest.py``, plan d454fd62 step 30).

Every condition is simulated: the tests patch ``sys.gettrace``, the environment
and the coverage probe, so they pass the same way plain and under ``--cov``.
"""

from __future__ import annotations

import ast
import os
import re
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


_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
_FACTOR_NAME = "timing_factor"
_FACTOR_CALL = "current_timing_factor"
# A name that holds a ratio, unless it also says it is an absolute time budget.
_RATIO_WORD = re.compile(r"ratio|spread|slope|growth|linear|quadratic", re.I)
_BUDGET_WORD = re.compile(r"budget|seconds|_s$|_ms$|timeout|deadline|elapsed", re.I)


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """Every node of *scope*'s own body, not those of a nested def or class (they
    are scopes of their own and are yielded themselves, undescended)."""
    found: list[ast.AST] = []
    stack: list[ast.AST] = list(getattr(scope, "body", []))
    while stack:
        node = stack.pop()
        found.append(node)
        if not isinstance(node, _DEF_NODES):
            stack.extend(ast.iter_child_nodes(node))
    return found


def _called_name(call: ast.Call) -> str | None:
    f = call.func
    return (
        f.id
        if isinstance(f, ast.Name)
        else f.attr
        if isinstance(f, ast.Attribute)
        else None
    )


class _ScaleScan:
    """Finds a ratio bound scaled by ``timing_factor`` in one module's source.

    Interface contract: ``_ScaleScan(source).hits() -> sorted line numbers``;
    a source that does not parse raises ``SyntaxError``. It follows the factor
    through ``timing_factor`` itself, ``current_timing_factor()`` (also under an
    import alias), any name assigned from an expression holding either (an alias
    ``tf = timing_factor``, a bound ``bound = 2.5 * timing_factor``, ``lim *=
    timing_factor``) and a parameter of a same-module helper that a call passes
    such a value to. A *ratio* is a division of two non-constant expressions (not
    by a constant, not by ``len(..)``, not by the factor itself), a name that
    says ``ratio|spread|slope|growth|linear|quadratic`` without also saying it is
    a time budget, and any name assigned from one. A line is a hit when one
    comparison or one arithmetic expression holds a scaled value and a ratio.
    An absolute budget (``elapsed < 2.0 * timing_factor``) is never one.
    """

    def __init__(self, source: str) -> None:
        self._tree = ast.parse(source)
        self._factor_calls = {_FACTOR_CALL}
        for node in ast.walk(self._tree):
            if isinstance(node, ast.ImportFrom | ast.Import):
                for alias in node.names:
                    if alias.name.rsplit(".", 1)[-1] == _FACTOR_CALL and alias.asname:
                        self._factor_calls.add(alias.asname)
        self._funcs: dict[str, list[ast.AST]] = {}
        for node in ast.walk(self._tree):
            if isinstance(node, _FUNC_NODES):
                self._funcs.setdefault(node.name, []).append(node)
        self._seeds: dict[int, tuple[set[str], set[str]]] = {}
        self._lines: set[int] = set()

    def hits(self) -> list[int]:
        for _ in range(5):  # parameter seeds feed the helpers, until stable
            before = {k: (set(a), set(b)) for k, (a, b) in self._seeds.items()}
            self._lines.clear()
            self._visit(self._tree, {_FACTOR_NAME}, set())
            if before == self._seeds:
                break
        return sorted(self._lines)

    # -- expression classes ------------------------------------------------

    def _scaled(self, expr: ast.AST, scaled: set[str]) -> bool:
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Name) and sub.id in scaled:
                return True
            if isinstance(sub, ast.Attribute) and sub.attr == _FACTOR_NAME:
                return True
            if isinstance(sub, ast.Call) and _called_name(sub) in self._factor_calls:
                return True
        return False

    def _ratio(self, expr: ast.AST, scaled: set[str], ratios: set[str]) -> bool:
        for sub in ast.walk(expr):
            text = (
                sub.id
                if isinstance(sub, ast.Name)
                else (sub.attr if isinstance(sub, ast.Attribute) else None)
            )
            if text is not None and (
                text in ratios
                or (_RATIO_WORD.search(text) and not _BUDGET_WORD.search(text))
            ):
                return True
            if (
                isinstance(sub, ast.BinOp)
                and isinstance(sub.op, ast.Div)
                and not isinstance(sub.left, ast.Constant)
                and not isinstance(sub.right, ast.Constant)
                and not (
                    isinstance(sub.right, ast.Call) and _called_name(sub.right) == "len"
                )
                and not self._scaled(sub.right, scaled)
                and not self._scaled(sub.left, scaled)
            ):
                return True
        return False

    # -- one scope ---------------------------------------------------------

    def _visit(self, scope: ast.AST, scaled_in: set[str], ratios_in: set[str]) -> None:
        scaled, ratios = set(scaled_in), set(ratios_in)
        if isinstance(scope, _FUNC_NODES):
            seeded = self._seeds.get(id(scope), (set(), set()))
            scaled |= seeded[0]
            ratios |= seeded[1]
        own = _own_nodes(scope)
        for _ in range(10):  # assignments in any order, until nothing new
            size = (len(scaled), len(ratios))
            for node in own:
                targets, value = self._assignment(node)
                if value is None:
                    continue
                if self._scaled(value, scaled):
                    scaled.update(targets)
                if self._ratio(value, scaled, ratios):
                    ratios.update(targets)
            if size == (len(scaled), len(ratios)):
                break
        for node in own:
            if isinstance(node, ast.Compare):
                parts = [node.left, *node.comparators]
                if any(self._scaled(p, scaled) for p in parts) and any(
                    self._ratio(p, scaled, ratios) for p in parts
                ):
                    self._lines.add(node.lineno)
            elif isinstance(node, ast.BinOp | ast.AugAssign):
                if self._scaled(node, scaled) and self._ratio(node, scaled, ratios):
                    self._lines.add(node.lineno)
            elif isinstance(node, ast.Call):
                self._seed_helpers(node, scaled, ratios)
        for node in own:
            if isinstance(node, _DEF_NODES):
                self._visit(node, scaled, ratios)

    @staticmethod
    def _assignment(node: ast.AST) -> tuple[set[str], ast.AST | None]:
        if isinstance(node, ast.Assign):
            names = {
                n.id
                for t in node.targets
                for n in ast.walk(t)
                if isinstance(n, ast.Name)
            }
            return names, node.value
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            return {node.target.id}, node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return {node.target.id}, node.value
        if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            return {node.target.id}, node.value
        return set(), None

    def _seed_helpers(self, call: ast.Call, scaled: set[str], ratios: set[str]) -> None:
        """A call into a same-module function taints the parameters it receives."""
        for func in self._funcs.get(_called_name(call) or "", []):
            args = func.args  # type: ignore[attr-defined]
            params = [a.arg for a in (*args.posonlyargs, *args.args)]
            shift = int(
                isinstance(call.func, ast.Attribute)
                and params[:1] in (["self"], ["cls"])
            )
            pairs = [
                (params[i + shift], a)
                for i, a in enumerate(call.args)
                if i + shift < len(params)
            ]
            pairs += [(k.arg, k.value) for k in call.keywords if k.arg in params]
            seed_scaled, seed_ratios = self._seeds.setdefault(id(func), (set(), set()))
            for name, value in pairs:
                if self._scaled(value, scaled):
                    seed_scaled.add(name)
                if self._ratio(value, scaled, ratios):
                    seed_ratios.add(name)


def scaled_ratio_lines(source: str) -> list[int]:
    """Line numbers where *source* scales a ratio bound by ``timing_factor``."""
    return _ScaleScan(source).hits()


# (source, flagged lines): every way found to scale a ratio bound
_SCALED_SHAPES = [
    ("assert max(t) / min(t) <= 2.5 * timing_factor", [1]),
    ("assert 2.5 * timing_factor >= max(t) / min(t)", [1]),
    ("bound = 2.5 * timing_factor\nassert max(t) / min(t) <= bound", [2]),
    ("assert max(t) / min(t) <= 2.5 * current_timing_factor()", [1]),
    (
        "from tests.conftest import current_timing_factor as ctf\n"
        "assert max(t) / min(t) <= 2.5 * ctf()",
        [2],
    ),
    ("tf = timing_factor\nassert max(t) / min(t) <= 2.5 * tf", [2]),
    ("f = current_timing_factor()\nassert max(t) / min(t) <= 2.5 * f", [2]),
    ("lim = 2.5\nlim *= timing_factor\nassert max(t) / min(t) <= lim", [3]),
    ("bound = _LINEAR_SPREAD * timing_factor", [1]),
    ("assert ratio <= 2.0 * timing_factor", [1]),
    ("limit = timing_factor * growth_cap", [1]),
    ("spread = a / b\nassert spread <= 3 * timing_factor", [2]),
    (
        "def _le(a, b):\n    assert a <= b\n"
        "def test_x(timing_factor):\n"
        "    _le(max(t) / min(t), 2.5 * timing_factor)",
        [2],
    ),
    (
        "class T:\n    def _le(self, a, b):\n        assert a <= b\n"
        "    def test_x(self, timing_factor):\n"
        "        self._le(max(t) / min(t), b=2.5 * timing_factor)",
        [3],
    ),
    (
        "def test_x(timing_factor):\n    def inner():\n"
        "        assert max(t) / min(t) <= 2.5 * timing_factor\n    inner()",
        [3],
    ),
]
# absolute budgets, the legitimate uses: none may be flagged
_ABSOLUTE_SHAPES = [
    "assert elapsed < 0.5 * timing_factor",
    "w = _LINEAR_BUDGET_S * timing_factor",
    "assert elapsed < 2.0 * timing_factor",
    "assert growth_seconds < 2 * timing_factor",
    "bound = 2.5 * timing_factor\nassert elapsed <= bound",
    "tf = timing_factor\nassert elapsed < 3 * tf",
    "f = current_timing_factor()\nassert elapsed < 3 * f",
    "assert sum(times) / len(times) < 0.1 * timing_factor",
    "assert elapsed / timing_factor < 2.0",
    "assert max(t) / min(t) <= 2.5",
    "assert max(t) / min(t) <= _LINEAR_SPREAD",
    "assert elapsed < budget * timing_factor",
    "def _le(a, b):\n    assert a <= b\n"
    "def test_x(timing_factor):\n    _le(elapsed, 2.5 * timing_factor)",
]


class TestNoRatioBoundIsScaled:
    """Review gate 1 (plan d454fd62 step 58.1, concern 6; broadened at step 58.2,
    re-review concern 4): a ratio bound is never multiplied by ``timing_factor``.

    A tracer or a loaded machine slows every size of a sweep by the same factor,
    so the ratio of two sizes does not move; scaling its bound only loosens it
    (``_LINEAR_SPREAD = 2.5`` became 12.5 under coverage, and
    ``FSM_LLM_TEST_TIMING_FACTOR=20`` made a quadratic crypto-gap regex pass).
    Only an absolute time budget scales. A noisy ratio is retried, never widened.

    The scan is :class:`_ScaleScan` (by value, not by one spelling): the five
    shapes the review used to defeat the first, name-based guard are cases below.
    """

    @pytest.mark.parametrize(("source", "lines"), _SCALED_SHAPES)
    def test_the_scan_flags_a_scaled_ratio(self, source, lines):
        assert scaled_ratio_lines(source) == lines

    @pytest.mark.parametrize("source", _ABSOLUTE_SHAPES)
    def test_the_scan_passes_an_absolute_budget(self, source):
        assert scaled_ratio_lines(source) == []

    def test_a_ratio_bound_held_in_a_variable_is_followed_across_scopes(self):
        source = (
            "BOUND = 2.5 * timing_factor\n"
            "def test_x():\n    assert max(t) / min(t) <= BOUND\n"
        )
        assert scaled_ratio_lines(source) == [3]

    def test_no_test_file_scales_a_ratio_bound(self):
        offenders = {}
        root = Path(conftest.__file__).parent
        for path in sorted(root.rglob("*.py")):
            if path == Path(__file__):
                continue
            lines = scaled_ratio_lines(path.read_text(encoding="utf-8"))
            if lines:
                offenders[str(path.relative_to(root))] = lines
        assert offenders == {}, (
            "a ratio bound must stay fixed (a uniform slowdown cancels in a "
            "ratio): scale only absolute time budgets"
        )
