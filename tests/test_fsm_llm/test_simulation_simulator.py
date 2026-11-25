"""The generic ``fsm_llm.simulation.Simulator`` and ``Budget`` (offline, scripted doubles).

The class moved out of ``synthesis.simulation`` (plan c720601d step 5): it holds the
interface rule (D-006 there, D-047 of plan 2cefeb61 before) and the bound rule (D-060)
and nothing of the G6b verdict. The G6b rows live in ``tests/test_fsm_llm_synthesis``.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import pytest

from fsm_llm.engine.api.conversation import API
from fsm_llm.simulation import Budget, Simulator, Trial
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm.test_simulation_driver import _CHAT

_FSM = _CHAT  # chat stays open until the user says stop


class Clock:
    """A clock that steps ``step`` seconds on every read."""

    def __init__(self, step: float = 0.0) -> None:
        self.t, self.step, self.reads = 0.0, step, 0

    def __call__(self) -> float:
        self.reads += 1
        t, self.t = self.t, self.t + self.step
        return t


def _llm() -> ScriptedLLM:
    return ScriptedLLM(responses={"chat": "r"})


def _play(sim: Simulator, turns: list[str], **bounds: Any) -> Trial | None:
    budget = sim.budget(bounds.get("max_seconds", 300), bounds.get("max_calls", 400))
    return sim.trial(copy.deepcopy(_FSM), {"seed": 1}, turns, budget=budget)


class TestTheInterface:
    def test_an_executor_without_usage_is_refused(self):
        executor = ScriptedLLM([])
        executor.usage = None  # type: ignore[assignment,method-assign]
        with pytest.raises(ValueError, match="usage"):
            Simulator(executor, allow_same_model=True)

    def test_a_judge_without_complete_is_refused(self):
        class Bare:
            def usage(self) -> Any: ...

        with pytest.raises(ValueError, match="complete"):
            Simulator(_llm(), Bare())  # type: ignore[arg-type]

    def test_the_judge_is_the_executor_unless_one_is_given(self):
        executor, other = _llm(), _llm()
        assert Simulator(executor, allow_same_model=True).judge is executor
        sim = Simulator(executor, other, allow_same_model=True)
        assert sim.executor is executor and sim.judge is other

    def test_the_interface_is_never_closed_and_never_replaced(self):
        class Closable(ScriptedLLM):
            closed = 0

            def close(self) -> None:
                self.closed += 1

        executor = Closable(responses={"chat": "r"})
        sim = Simulator(executor, allow_same_model=True)
        trial = _play(sim, ["a", "b"])
        assert trial is not None and trial.error is None and trial.turns_sent == 2
        assert executor.closed == 0 and sim.executor is executor

    def test_the_trial_uses_the_executor_as_given(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        seen: list[Any] = []
        real = API.from_definition

        def spy(definition: Any, **kwargs: Any) -> API:
            seen.append(kwargs.get("llm_interface"))
            return real(definition, **kwargs)

        monkeypatch.setattr(API, "from_definition", spy)
        executor = _llm()
        assert _play(Simulator(executor, allow_same_model=True), ["a"]) is not None
        assert seen == [executor]

    def test_the_simulator_decides_nothing(self):
        assert not hasattr(Simulator, "run") and not hasattr(Simulator, "replay")
        assert "passed" not in {f.name for f in dataclasses.fields(Trial)}


class TestTheBound:
    def test_a_spent_bound_before_the_first_trial_drops_it(self):
        sim = Simulator(_llm(), allow_same_model=True, clock=Clock(1.0))
        assert _play(sim, ["a"], max_seconds=0) is None

    def test_the_seconds_bound_stops_a_trial_between_turns_and_drops_it(self):
        # reads: budget start 0, trial check 1, then one read per turn after the first
        sim = Simulator(_llm(), allow_same_model=True, clock=Clock(1.0))
        assert _play(sim, ["a", "b", "c", "d"], max_seconds=3) is None
        sim = Simulator(_llm(), allow_same_model=True, clock=Clock(1.0))
        done = _play(sim, ["a", "b", "c", "d"], max_seconds=100)
        assert done is not None and done.turns_sent == 4

    def test_the_calls_bound_reads_the_executor_meter(self):
        executor = _llm()
        sim = Simulator(executor, allow_same_model=True)
        assert _play(sim, ["a", "b", "c"], max_calls=1) is None
        assert executor.usage().calls >= 1  # it spent one turn past the bound

    def test_a_bound_equal_to_the_spend_counts_as_spent(self):
        class Meter:
            def __init__(self) -> None:
                self.calls = 0

            def read(self) -> int:
                return self.calls

        meter, clock = Meter(), Clock(0.0)
        budget = Budget(meter.read, clock, 10.0, 3, clock(), meter.read())
        assert not budget.spent()
        meter.calls = 3
        assert budget.spent() and budget.calls() == 3
        clock.t = 10.0
        meter.calls = 0
        assert budget.spent()  # seconds: equal to the bound is spent

    def test_budget_reads_the_clock_once_per_check_and_the_meter_after_it(self):
        clock = Clock(0.0)
        sim = Simulator(_llm(), allow_same_model=True, clock=clock)
        budget = sim.budget(300, 400)
        reads = clock.reads
        assert reads == 1  # the start mark
        assert not budget.spent() and clock.reads == reads + 1
