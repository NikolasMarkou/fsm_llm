"""The lifetime turn budget of one manager conversation and the one limit validator.

Layer ``engine`` (plan 5a6a1f80, step 7, roadmap row 1.7). ``FSMManager`` keeps one
``TurnBudget`` per manager conversation id that was given a limit, so the root
conversation and every pushed frame each have their own. ``TurnsMixin._enter_turn`` charges it
(``engine/fsm/turns.py``); nothing else reads or writes it.

Interface contract:
    - ``TurnBudget(max_steps, max_seconds)``: the private record. ``started`` is the
      ``_monotonic()`` reading at creation, ``steps_taken`` the turns charged so far.
      ``spent() -> "seconds" | "steps" | None`` names the budget that forbids the NEXT turn
      (seconds first when both are spent, as ``run_until_terminal`` reports them); ``charge()``
      raises ``RunBudgetExceededError(budget, limit, steps_taken)`` when ``spent()`` names one,
      and otherwise counts one turn. No lock of its own: the caller of ``_enter_turn`` holds the
      manager ``_lock``.
    - ``open_budget(max_steps, max_seconds) -> TurnBudget | None``: ``None`` when both limits are
      ``None``, so a conversation with no limit has no record and nothing changes for it.
    - ``check_turn_limits(max_steps, max_seconds)``: the validator of a lifetime limit, each
      ``None`` or valid (``ValueError`` otherwise).
    - ``_check_run_budgets(max_steps, max_seconds)``: the validator of a per-call run budget
      (moved here from ``engine/api/run.py``; the one definition of "a valid limit").
    - ``_monotonic``: the clock of the seconds budget; a test binds a stand-in on this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from fsm_llm.core.definitions import RunBudgetExceededError

#: The clock of the seconds budget. Read through the module attribute at every use, so a test
#: that binds a stand-in here moves the budget; it is never the order of anything.
_monotonic = time.monotonic


def _check_run_budgets(max_steps: int, max_seconds: float | None) -> None:
    """Validate the budgets of a bounded run; raise ``ValueError`` if invalid.

    Contract: ``max_steps`` must be an ``int`` (not a ``bool``) of at least 1;
    ``max_seconds`` must be ``None`` or an ``int`` or ``float`` (not a
    ``bool``) above 0 (NaN is refused).
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError(f"max_steps must be an integer >= 1, got {max_steps!r}")
    if max_seconds is not None and (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, (int, float))
        or not max_seconds > 0
    ):
        raise ValueError(
            f"max_seconds must be a number > 0 or None, got {max_seconds!r}"
        )


def check_turn_limits(max_steps: int | None, max_seconds: float | None) -> None:
    """Validate the lifetime limits of a conversation; each is ``None`` or a valid limit.

    Contract: the same rules as ``_check_run_budgets`` (it is called for each given limit, so
    the messages are the same), ``ValueError`` for the first invalid one; ``None`` passes.
    """
    if max_steps is not None:
        _check_run_budgets(max_steps, None)
    if max_seconds is not None:
        _check_run_budgets(1, max_seconds)


# DECISION plan-2026-11-18T071726-5a6a1f80/D-010
# The lifetime budget is ONE private record per manager conversation id, owned by
# `FSMManager._turn_budgets` and charged in `_enter_turn` only. Do NOT keep the count in
# `context.data`, `metadata`, a handler or `_TurnSnapshot`: a model-extractable key that a driver
# trusts as a gate is forgeable, a handler counter misses a null-extraction turn, and a snapshot
# is rolled back (a failed turn would be refunded). Do NOT count in `run.py` (its counters are
# locals of one call and restart, the reason the synthesis runner kept its own). Do NOT share one
# record between a parent and a pushed frame (each frame is its own manager conversation and its
# own budget), do NOT make a spent frame "ended" (`has_conversation_ended` stays as it is and the
# engine pops no frame of its own accord, with ONE exception: `settle` pops the frame a call state
# pushed when its child ended or its budget is spent, plan baa37f40 decision D-023) and do NOT
# put the record or its count in a session snapshot (a restore, a fork and a resume start a
# FRESH budget from the API defaults).
@dataclass
class TurnBudget:
    """The limits and the count of one conversation (see the module docstring)."""

    max_steps: int | None
    max_seconds: float | None
    started: float = field(default_factory=lambda: _monotonic())
    steps_taken: int = 0

    def spent(self) -> Literal["steps", "seconds"] | None:
        """Name the budget that forbids the next turn, or ``None`` (seconds first)."""
        if (
            self.max_seconds is not None
            and _monotonic() - self.started >= self.max_seconds
        ):
            return "seconds"
        if self.max_steps is not None and self.steps_taken >= self.max_steps:
            return "steps"
        return None

    def charge(self) -> None:
        """Count one turn, or raise ``RunBudgetExceededError`` when a budget forbids it."""
        spent = self.spent()
        if spent == "seconds":
            raise RunBudgetExceededError(
                "seconds", self.max_seconds or 0.0, self.steps_taken
            )
        if spent == "steps":
            raise RunBudgetExceededError("steps", self.max_steps or 0, self.steps_taken)
        self.steps_taken += 1


def open_budget(max_steps: int | None, max_seconds: float | None) -> TurnBudget | None:
    """Return a new record for the limits, or ``None`` when both are ``None``."""
    if max_steps is None and max_seconds is None:
        return None
    return TurnBudget(max_steps, max_seconds)
