"""The generic simulator: the interface that runs the procedure, a judge, two bounds."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fsm_llm.core.definitions import FSMDefinition
from fsm_llm.engine.api.conversation import API
from fsm_llm.llm import LLMInterface, implements_complete

from .driver import Trial, play


@dataclass(frozen=True)
class Budget:
    """The two bounds of one simulation run, read between turns.

    ``spent()`` is true once ``max_seconds`` passed on ``clock`` since ``start`` or the
    executor's meter counted ``max_calls`` calls since the run began. ``calls()`` is
    the meter's delta since then.
    """

    meter: Callable[[], int]
    clock: Callable[[], float]
    max_seconds: float
    max_calls: int
    start: float
    calls0: int

    def calls(self) -> int:
        return self.meter() - self.calls0

    def spent(self) -> bool:
        return (
            self.clock() - self.start >= self.max_seconds
            or self.calls() >= self.max_calls
        )


# DECISION plan-2026-11-25T074601-c720601d/D-006
# The package contract and the interface rule (D-047 of plan 2cefeb61, carried to the
# class). ``simulation`` imports no consumer (eval, synthesis, agents, testing). The
# executor is THE interface the host gave: one object, metered once, never closed,
# never replaced, refused without ``usage()`` (a bound it cannot read is no bound);
# never build an interface from a model name here. The class returns a plain ``Trial``
# and decides nothing: a verdict is the caller's code (G6b's ``_passes``), never a
# field of the trial. Do NOT add ``run`` or ``replay`` here (they need
# ``SynthesisPolicy``) and do NOT read a model-written field of a scenario.
class Simulator:
    """Plays scripted users against a definition on one executor interface.

    Interface contract (callers: ``synthesis.simulation.run_g6b`` and
    ``replay_g6b``, tests): ``Simulator(executor, judge=None, *, clock=time.monotonic)``.
    ``executor`` runs the procedure under test and needs a callable ``usage()``, else
    ``ValueError``. ``judge`` (default ``executor``) writes scenarios and must
    implement ``complete``, else ``ValueError``; it is read as ``.judge``. Neither is
    ever closed or replaced. ``budget(max_seconds, max_calls)`` starts a run's bounds.
    ``trial(fsm, initial_context, turns, *, budget)`` plays one conversation on a
    fresh ``API`` over the executor and returns the ``Trial`` (an error is data), or
    ``None`` when ``budget`` was spent before the trial or before a turn after its
    first: a stopped trial is dropped, it says nothing about the procedure.
    """

    def __init__(
        self,
        executor: LLMInterface,
        judge: LLMInterface | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(getattr(executor, "usage", None)):
            raise ValueError("executor must be metered: it needs a usage() method")
        writer = executor if judge is None else judge
        if not implements_complete(writer):
            raise ValueError("the judge must implement complete")
        self.executor, self.judge, self._clock = executor, writer, clock

    def _calls(self) -> int:
        return int(self.executor.usage().calls)  # type: ignore[attr-defined]

    def budget(self, max_seconds: float, max_calls: int) -> Budget:
        """The bounds of a run that begins now."""
        return Budget(
            self._calls,
            self._clock,
            max_seconds,
            max_calls,
            self._clock(),
            self._calls(),
        )

    # DECISION plan-2026-11-10T194139-2cefeb61/D-060
    # The bound is read between turns through ``play(stop=...)`` and a stopped trial is
    # dropped. Do NOT wrap the interface to refuse calls (the pipeline swallows an
    # extraction error, so a turn would run on) and do NOT count a stopped trial as a
    # failed procedure.
    def trial(
        self,
        fsm: FSMDefinition | dict[str, Any],
        initial_context: Mapping[str, Any],
        turns: Sequence[str],
        *,
        budget: Budget,
    ) -> Trial | None:
        """One trial, or ``None`` when the bound stopped it. The bound is read before
        the trial and before every turn after the first, so a trial spends at most one
        turn past it."""
        if budget.spent():
            return None
        cut, first = False, True

        def stop() -> bool:
            nonlocal cut, first
            if first:  # the check above stands for the first turn
                first = False
                return False
            cut = cut or budget.spent()
            return cut

        definition = (
            fsm.model_dump(exclude_defaults=True)
            if isinstance(fsm, FSMDefinition)
            else fsm
        )
        trial = play(
            lambda: API.from_definition(definition, llm_interface=self.executor),
            initial_context,
            turns,
            stop=stop,
        )
        return None if cut else trial
