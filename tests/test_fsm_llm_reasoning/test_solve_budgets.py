"""The budgets of a solve (plan baa37f40, step 16, D-032, roadmap E3.3).

``solve_problem(problem, initial_context=None, *, max_steps=None, max_seconds=None)``: the default
step budget is derived from the ``call.max_steps`` of the orchestrator file (no second copy of the
30), an explicit limit replaces or bounds the one run, a bad limit is refused before any model call,
and ``reasoning.solve`` maps the limits of its run context instead of refusing them. Every row is
scripted: no network, the clock of the seconds budget is a stand-in.
"""

from __future__ import annotations

import ast
import copy
import inspect
import itertools
import time
from pathlib import Path
from typing import Any, ClassVar

import pytest

import fsm_llm
from fsm_llm import RunBudgetExceededError
from fsm_llm.core.codes import Reason, Status
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.reasoning import ReasoningEngine, ReasoningExecutionError
from fsm_llm.reasoning import constants as constants_module
from fsm_llm.reasoning import engine as engine_module
from fsm_llm.reasoning.constants import ContextKeys, Defaults
from fsm_llm.reasoning.executables import SOLVE_ID
from fsm_llm.reasoning.strategy_files import load_fsm
from tests.conftest import bind_api_clock

from .test_engine_scripted import (
    _NEVER_ENDING_SCRIPT,
    _PROBLEM,
    _VALID_SCRIPT,
    _engine,
    _ScriptedLLM,
)

REPO = Path(__file__).resolve().parents[2]
CALL_STATE = "execute_reasoning"


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(REPO / "src")
    assert Path(engine_module.__file__).resolve().is_relative_to(REPO / "src")


class _TickingClock:
    """The ``time`` binding of the run loop: every read of ``monotonic`` is one second later."""

    def __init__(self) -> None:
        self._ticks = itertools.count(1000.0, 1.0)

    def monotonic(self) -> float:
        return next(self._ticks)

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def _with_call_steps(monkeypatch: pytest.MonkeyPatch, steps: int | None) -> None:
    """Make the engine read an orchestrator file whose call state has ``max_steps = steps``."""
    original = engine_module.get_reasoning_fsm

    def patched(name: str) -> dict[str, Any]:
        fsm = copy.deepcopy(original(name))
        if name == "orchestrator":
            call = fsm["states"][CALL_STATE]["call"]
            if steps is None:
                call.pop("max_steps", None)
            else:
                call["max_steps"] = steps
        return fsm

    monkeypatch.setattr(engine_module, "get_reasoning_fsm", patched)


class TestTheDerivedDefault:
    def test_the_formula_gives_the_former_170_from_the_file(self) -> None:
        file_steps = load_fsm("orchestrator")["states"][CALL_STATE]["call"]["max_steps"]
        assert file_steps == 30
        assert constants_module.solve_step_budget(file_steps) == 170
        assert constants_module.solve_step_budget(file_steps) == (
            Defaults.MAX_TOTAL_STEPS + (Defaults.MAX_RETRIES + 1) * file_steps
        )

    def test_the_literal_is_gone(self) -> None:
        assert not hasattr(Defaults, "MAX_SOLVE_STEPS")
        assert not hasattr(constants_module, "MAX_SOLVE_STEPS")

    def test_the_engine_default_is_the_derivation_over_its_file(self) -> None:
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        assert engine._solve_steps == 170

    def test_a_file_with_another_call_budget_moves_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _with_call_steps(monkeypatch, 10)
        engine = _engine(_ScriptedLLM(_NEVER_ENDING_SCRIPT))
        assert engine._solve_steps == Defaults.MAX_TOTAL_STEPS + 4 * 10 == 90
        with pytest.raises(ReasoningExecutionError) as info:
            engine.solve_problem(_PROBLEM)
        assert isinstance(info.value.__cause__, RunBudgetExceededError)
        assert info.value.__cause__.steps_done == 90

    def test_a_call_without_a_step_budget_is_a_packaging_bug_at_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _with_call_steps(monkeypatch, None)
        with pytest.raises(ReasoningExecutionError, match="max_steps"):
            _engine(_ScriptedLLM(_VALID_SCRIPT))

    def test_an_unbounded_solve_stops_at_the_derived_default(self) -> None:
        engine = _engine(_ScriptedLLM(_NEVER_ENDING_SCRIPT))
        with pytest.raises(ReasoningExecutionError) as info:
            engine.solve_problem(_PROBLEM)
        cause = info.value.__cause__
        assert isinstance(cause, RunBudgetExceededError)
        assert (cause.budget, cause.limit, cause.steps_done) == ("steps", 170, 170)


class TestExplicitBudgets:
    def test_max_steps_bounds_the_one_run(self) -> None:
        engine = _engine(_ScriptedLLM(_NEVER_ENDING_SCRIPT))
        with pytest.raises(ReasoningExecutionError, match=r"within 7 steps") as info:
            engine.solve_problem(_PROBLEM, max_steps=7)
        cause = info.value.__cause__
        assert isinstance(cause, RunBudgetExceededError)
        assert (cause.budget, cause.limit, cause.steps_done) == ("steps", 7, 7)
        partial = info.value.details["partial_context"]
        assert partial[ContextKeys.PROBLEM_STATEMENT] == _PROBLEM
        assert engine.orchestrator.list_active_conversations() == []

    def test_max_steps_may_exceed_the_default(self) -> None:
        engine = _engine(_ScriptedLLM(_NEVER_ENDING_SCRIPT))
        with pytest.raises(ReasoningExecutionError) as info:
            engine.solve_problem(_PROBLEM, max_steps=200)
        assert info.value.__cause__.steps_done == 200

    def test_max_seconds_bounds_the_one_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bind_api_clock(monkeypatch, _TickingClock())
        engine = _engine(_ScriptedLLM(_NEVER_ENDING_SCRIPT))
        with pytest.raises(ReasoningExecutionError) as info:
            engine.solve_problem(_PROBLEM, max_seconds=3.5)
        cause = info.value.__cause__
        assert isinstance(cause, RunBudgetExceededError)
        assert (cause.budget, cause.limit) == ("seconds", 3.5)
        assert cause.steps_done < 170
        assert engine.orchestrator.list_active_conversations() == []

    def test_generous_limits_change_nothing_about_a_finished_solve(self) -> None:
        plain = _engine(_ScriptedLLM(_VALID_SCRIPT)).solve_problem(_PROBLEM)
        bounded_llm = _ScriptedLLM(_VALID_SCRIPT)
        bounded = _engine(bounded_llm).solve_problem(
            _PROBLEM, max_steps=170, max_seconds=600.0
        )
        assert bounded[0] == plain[0] == _VALID_SCRIPT["final_solution"]
        assert bounded[1]["all_responses"] == plain[1]["all_responses"]

    def test_the_two_limits_are_keyword_only_and_the_old_shape_is_kept(self) -> None:
        parameters = inspect.signature(ReasoningEngine.solve_problem).parameters
        assert list(parameters) == [
            "self",
            "problem",
            "initial_context",
            "max_steps",
            "max_seconds",
        ]
        for name in ("max_steps", "max_seconds"):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default is None
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        with pytest.raises(TypeError):
            engine.solve_problem(_PROBLEM, None, 5)  # type: ignore[misc]


class TestABadLimitIsRefusedBeforeAnyCall:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("max_steps", 0),
            ("max_steps", -1),
            ("max_steps", True),
            ("max_steps", 1.5),
            ("max_steps", "5"),
            ("max_seconds", 0),
            ("max_seconds", -1.0),
            ("max_seconds", True),
            ("max_seconds", float("nan")),
            ("max_seconds", "5"),
        ],
    )
    def test_value_error_no_conversation_no_model_call_lock_free(
        self, name: str, value: Any
    ) -> None:
        llm = _ScriptedLLM(_VALID_SCRIPT)
        engine = _engine(llm)
        with pytest.raises(ValueError, match=name):
            engine.solve_problem(_PROBLEM, **{name: value})
        assert llm.requests == [] and llm.completions == []
        assert engine.orchestrator.list_active_conversations() == []
        assert engine._solve_lock.acquire(blocking=False)
        engine._solve_lock.release()


class _SpyEngine:
    """Stands in for ``ReasoningEngine`` and keeps the limits it was asked to solve with."""

    solved: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        pass

    def solve_problem(self, problem: str, initial_context: Any = None, **limits: Any):
        type(self).solved.append(limits)
        return "the solution", {"reasoning_trace": {"steps": []}, "summary": "s"}

    def close(self) -> None:
        pass


def _registry() -> ExecutableRegistry:
    registry = ExecutableRegistry()
    assert not registry.load_sources().failed
    return registry


class TestTheSolveExecutableMapsTheBudgets:
    def test_it_names_no_refused_budget(self) -> None:
        assert _registry().get(SOLVE_ID).refused_budgets == ()

    @pytest.mark.parametrize(
        ("limits", "expected"),
        [
            ({}, {"max_steps": None, "max_seconds": None}),
            ({"max_steps": 5}, {"max_steps": 5, "max_seconds": None}),
            (
                {"max_steps": 5, "max_seconds": 2.5},
                {"max_steps": 5, "max_seconds": 2.5},
            ),
        ],
    )
    def test_the_run_context_limits_reach_the_engine(
        self,
        monkeypatch: pytest.MonkeyPatch,
        limits: dict[str, Any],
        expected: dict[str, Any],
    ) -> None:
        from fsm_llm.reasoning import executables as module

        _SpyEngine.solved = []
        monkeypatch.setattr(module, "ReasoningEngine", _SpyEngine)
        result = (
            _registry()
            .get(SOLVE_ID)
            .run(
                {"problem": "p"},
                context=RunContext(llm_interface=_ScriptedLLM(_VALID_SCRIPT), **limits),
            )
        )
        assert result.status is Status.OK
        assert _SpyEngine.solved == [expected]

    def test_a_spent_step_budget_ends_aborted_budget_spent(self) -> None:
        result = (
            _registry()
            .get(SOLVE_ID)
            .run(
                {"problem": _PROBLEM},
                context=RunContext(
                    llm_interface=_ScriptedLLM(_NEVER_ENDING_SCRIPT), max_steps=5
                ),
            )
        )
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert result.detail["steps_done"] == 5 and result.detail["limit"] == 5


class TestTheCallSurfaceOfTheReasoningAgentIsUnchanged:
    def test_it_still_calls_solve_problem_with_the_problem_alone(self) -> None:
        source = (REPO / "src/fsm_llm/agents/reasoning_react.py").read_text(
            encoding="utf-8"
        )
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "solve_problem"
        ]
        assert calls
        for call in calls:
            assert len(call.args) == 1 and not call.keywords
