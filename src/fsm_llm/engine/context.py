"""The run context (roadmap 1.14, contract C12): ``RunContext`` and ``CancelToken``.

One frozen value that a runner takes for a run: which model to call, where to log, how long
to run, how to stop it, which host objects a code-backed pattern may use, and which run asked.
The first runner that takes it is the FSM-body executable of ``fsm_llm.engine.executables``;
the others adopt it later (roadmap 1.14 proper). No ``API`` constructor reads it today.

Interface contract:

- ``RunContext(llm_interface=None, llm_factory=None, event_log=None, max_steps=None,
  max_seconds=None, cancel=<a fresh CancelToken>, callables=None, parent=None,
  registry=None, budget=None)``, all keyword or positional in that order, frozen, compared by identity. ``llm_interface`` and
  ``llm_factory`` (a zero-argument callable returning an ``LLMInterface``) are refused
  together, as ``API(llm_interface=...)`` refuses every other LLM setting beside it.
  ``event_log`` is what ``API(event_log=)`` takes (an ``EventLog`` or a path). ``max_steps``
  and ``max_seconds`` are the limits of ONE run (``None`` means no limit), validated by the
  engine's own rule (``check_turn_limits``): an ``int`` of at least 1 and a number above 0,
  never a ``bool``. ``callables`` is an optional ``NameRegistry`` holding the host objects of
  code-backed patterns and workflow callables. ``parent`` is the reference (a conversation
  or instance id) of the run that asked, or ``None``. ``registry`` is the ``TargetResolver``
  (``engine.kinds.targets``: an ``ExecutableRegistry`` is one) that the ``API`` of a run is
  given as ``targets=``, so a body holding a ``call`` state resolves its target the way a
  bare ``API(targets=)`` does (plan baa37f40 D-021); ``None`` gives that ``API`` no resolver.
  ``budget`` is a ``RunBudget`` (below) or ``None``. A violation raises ``ValueError`` (or
  ``TypeError`` for a wrong type of ``cancel``, ``callables``, ``registry`` and ``budget``)
  at construction.
- ``resolve_llm()`` returns the interface, or calls the factory ONCE and returns its result
  (one call per run, nothing cached), or returns ``None`` (the run then resolves a model the
  way ``API`` does, from ``LLM_MODEL``).
- ``derive(*, parent, max_steps=None, max_seconds=None, cancel=None)`` returns a child context
  that shares the interface, factory, log, callables and registry, names ``parent`` (the
  handle of the run that is asking, not blank) and narrows each limit that is given: a value
  above the parent's own limit raises ``ValueError`` (a parent with no limit accepts any
  valid one; ``None`` keeps the parent's, it never removes a limit). ``cancel=None`` shares the
  parent's token (one cancel reaches every child); a ``CancelToken`` gives the child its OWN
  token, so cancelling it leaves the parent's untouched (the token of a delegated run, plan
  baa37f40 D-049; the parent's cancel does not reach the child by itself: the host that made
  the token sets it). A value that is not a ``CancelToken`` raises ``TypeError``. ``max_steps`` and
  ``max_seconds`` are per run: a child's steps are not subtracted from the parent's. ``derive``
  shares ``budget`` by reference (one ledger for a whole tree of runs).
- ``RunBudget(max_steps=None, max_seconds=None)``: ONE ledger shared by every runner of a
  context (plan 7efd5cb4). ``charge()`` counts one step or raises ``RunBudgetExceededError``
  (the next after ``max_steps``, or once ``max_seconds`` passed since the FIRST charge, never
  since construction), ``spent()`` names the budget that forbids the next charge or ``None``,
  ``steps_taken`` counts. The unit is the loop step of the runner that charges it. A leaf lock
  guards the arithmetic only. It is not the engine's per-conversation ``TurnBudget``.
- ``step_hook()`` returns a fresh ``before_step`` callable for ONE run: it raises
  ``RunCancelledError`` when the token is set, BEFORE it charges the ledger, and charges once per
  distinct step number (the engine offers an ended pushed frame the same number again).
  ``budget_spent()`` is ``budget.spent()`` or ``None`` (a caller reads it after a catch);
  ``ledger_detail()`` is ``{"shared_budget": "steps" | "seconds"}`` when it is set, else ``{}``
  (what an adapter adds to the ``detail`` of a budget stop, so a ledger stop reads as the shared
  budget by a fact, never by error text).
- ``CancelToken``: one thread-safe, set-once request. ``cancel()``, ``cancelled``,
  ``raise_if_cancelled()`` (raises ``RunCancelledError``, an ``FSMError``). A token is checked
  BETWEEN steps (the executable's ``before_step``); it never interrupts a model call in
  flight (that reply is discarded at the next check, roadmap 4.8), and it is not an
  ``asyncio`` cancel: a cancelled task still raises ``CancelledError`` and the engine rolls
  its turn back whole (plan f150ec28 D-002).

Imported by name (``from fsm_llm.engine.context import RunContext``); ``fsm_llm/__init__.py``
and ``engine/__init__.py`` import nothing of it (the laziness table pins that). Plan ace7a8c8
D-058.
"""

from __future__ import annotations

import dataclasses
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from fsm_llm.core.definitions.errors import FSMError
from fsm_llm.core.names import NameRegistry
from fsm_llm.engine.fsm.budget import TurnBudget, check_turn_limits
from fsm_llm.llm.interface import LLMInterface

if TYPE_CHECKING:
    from fsm_llm.engine.kinds.targets import TargetResolver
    from fsm_llm.runtime import EventLog

__all__ = ["CancelToken", "RunBudget", "RunCancelledError", "RunContext"]


class RunCancelledError(FSMError):
    """A run's ``CancelToken`` was set. Not an ``asyncio.CancelledError``."""

    def __init__(
        self,
        message: str = "the run was cancelled",
        *,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)


class CancelToken:
    """A thread-safe, set-once cancel request, checked between the steps of a run."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request the cancel. Idempotent; a set token is never cleared."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """True once ``cancel`` was called."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise ``RunCancelledError`` when the token is set; return ``None`` otherwise."""
        if self._event.is_set():
            raise RunCancelledError()

    def __repr__(self) -> str:
        return f"CancelToken(cancelled={self.cancelled})"


# DECISION plan-2026-11-26T022222-7efd5cb4/D-004
# The shared run budget is ONE run-level ledger, a DECLARED field of `RunContext` carried by
# reference through `dataclasses.replace` (`derive` and `ExecutableStep` rebuild the context), charged
# once per distinct step number by the hook of `step_hook`, under ONE leaf lock. Do NOT set it in
# `__post_init__` (the second `replace` loses it), keep it in context data, a handler, a snapshot
# or the engine's `_turn_budgets` (5a6a1f80 D-010), read it from the log, start its clock at
# construction, charge per hook CALL (the ended-frame offer repeats a number), or hold the lock
# across a hook, a raise or a log write.
class RunBudget:
    """The ledger of the steps and seconds of every run under one context (see above)."""

    def __init__(
        self, max_steps: int | None = None, max_seconds: float | None = None
    ) -> None:
        check_turn_limits(max_steps, max_seconds)
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self._lock = threading.Lock()
        self._record: TurnBudget | None = None

    @property
    def steps_taken(self) -> int:
        with self._lock:
            return self._record.steps_taken if self._record else 0

    def spent(self) -> Literal["steps", "seconds"] | None:
        """The budget that forbids the next charge (seconds first), or ``None``."""
        with self._lock:
            return self._record.spent() if self._record else None

    def charge(self) -> None:
        """Count one step, or raise ``RunBudgetExceededError`` (the count is unchanged)."""
        with self._lock:
            if self._record is None:
                self._record = TurnBudget(self.max_steps, self.max_seconds)
            self._record.charge()


@dataclass(frozen=True, eq=False)
class RunContext:
    """The frozen value a runner takes for one run. See the module docstring."""

    llm_interface: LLMInterface | None = None
    llm_factory: Callable[[], LLMInterface] | None = None
    event_log: EventLog | str | os.PathLike[str] | None = None
    max_steps: int | None = None
    max_seconds: float | None = None
    cancel: CancelToken = field(default_factory=CancelToken)
    callables: NameRegistry[Any] | None = None
    parent: str | None = None
    registry: TargetResolver | None = None
    budget: RunBudget | None = None

    def __post_init__(self) -> None:
        if self.llm_interface is not None and self.llm_factory is not None:
            raise ValueError(
                "RunContext takes llm_interface or llm_factory, not both: an injected "
                "interface refuses every other LLM setting beside it"
            )
        if self.llm_interface is not None and not isinstance(
            self.llm_interface, LLMInterface
        ):
            raise ValueError("llm_interface must be an instance of LLMInterface")
        if self.llm_factory is not None and not callable(self.llm_factory):
            raise ValueError("llm_factory must be a callable returning an LLMInterface")
        if not isinstance(self.cancel, CancelToken):
            raise TypeError(
                f"cancel must be a CancelToken, got {type(self.cancel).__name__}"
            )
        if self.callables is not None and not isinstance(self.callables, NameRegistry):
            raise TypeError(
                f"callables must be a NameRegistry, got {type(self.callables).__name__}"
            )
        _check_parent(self.parent)
        if self.registry is not None and not callable(
            getattr(self.registry, "resolve_target", None)
        ):
            raise TypeError(
                "registry must provide resolve_target(target_id, *, conversation_id=None) "
                f"(a TargetResolver), got {type(self.registry).__name__}"
            )
        if self.budget is not None and not isinstance(self.budget, RunBudget):
            raise TypeError(
                f"budget must be a RunBudget, got {type(self.budget).__name__}"
            )
        check_turn_limits(self.max_steps, self.max_seconds)

    def resolve_llm(self) -> LLMInterface | None:
        """The interface, or one fresh ``llm_factory()`` call, or ``None`` (neither is set)."""
        if self.llm_interface is not None:
            return self.llm_interface
        if self.llm_factory is not None:
            return self.llm_factory()
        return None

    def step_hook(self) -> Callable[[int], None]:
        """A fresh ``before_step`` for ONE run: the cancel check, then one charge per step."""
        last = 0

        def hook(step: int) -> None:
            nonlocal last
            self.cancel.raise_if_cancelled()
            if self.budget is not None and step > last:
                self.budget.charge()
                last = step

        return hook

    def budget_spent(self) -> Literal["steps", "seconds"] | None:
        """The shared budget that is spent (a fact for a caller after a catch), or ``None``."""
        return self.budget.spent() if self.budget is not None else None

    def ledger_detail(self) -> dict[str, str]:
        """The ``detail`` a stopped run adds when the shared ledger is spent, else ``{}``."""
        spent = self.budget_spent()
        return {"shared_budget": spent} if spent else {}

    def derive(
        self,
        *,
        parent: str,
        max_steps: int | None = None,
        max_seconds: float | None = None,
        cancel: CancelToken | None = None,
    ) -> RunContext:
        """A child context: same members, ``parent`` set, each given limit narrowed.

        Raises ``ValueError`` for a blank ``parent``, an invalid limit, or a limit above the
        parent's own, ``TypeError`` for a ``cancel`` that is not a ``CancelToken``. ``None``
        keeps the parent's limit and its token.
        """
        if parent is None:
            raise ValueError("parent must name the run that is asking, not None")
        _check_parent(parent)
        check_turn_limits(max_steps, max_seconds)
        for name, mine, new in (
            ("max_steps", self.max_steps, max_steps),
            ("max_seconds", self.max_seconds, max_seconds),
        ):
            if new is not None and mine is not None and new > mine:
                raise ValueError(
                    f"derive cannot widen {name}: {new!r} is above the parent's {mine!r}"
                )
        return dataclasses.replace(
            self,
            parent=parent,
            max_steps=self.max_steps if max_steps is None else max_steps,
            max_seconds=self.max_seconds if max_seconds is None else max_seconds,
            cancel=self.cancel if cancel is None else cancel,
        )


def _check_parent(parent: str | None) -> None:
    if parent is not None and (not isinstance(parent, str) or not parent.strip()):
        raise ValueError(f"parent must be a non-blank str or None, got {parent!r}")
