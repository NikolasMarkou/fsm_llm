"""The run context (roadmap 1.14, contract C12): ``RunContext`` and ``CancelToken``.

One frozen value that a runner takes for a run: which model to call, where to log, how long
to run, how to stop it, which host objects a code-backed pattern may use, and which run asked.
The first runner that takes it is the FSM-body executable of ``fsm_llm.engine.executables``;
the others adopt it later (roadmap 1.14 proper). No ``API`` constructor reads it today.

Interface contract:

- ``RunContext(llm_interface=None, llm_factory=None, event_log=None, max_steps=None,
  max_seconds=None, cancel=<a fresh CancelToken>, callables=None, parent=None)``, all keyword
  or positional in that order, frozen, compared by identity. ``llm_interface`` and
  ``llm_factory`` (a zero-argument callable returning an ``LLMInterface``) are refused
  together, as ``API(llm_interface=...)`` refuses every other LLM setting beside it.
  ``event_log`` is what ``API(event_log=)`` takes (an ``EventLog`` or a path). ``max_steps``
  and ``max_seconds`` are the limits of ONE run (``None`` means no limit), validated by the
  engine's own rule (``check_turn_limits``): an ``int`` of at least 1 and a number above 0,
  never a ``bool``. ``callables`` is an optional ``NameRegistry`` holding the host objects of
  code-backed patterns and workflow callables. ``parent`` is the reference (a conversation
  or instance id) of the run that asked, or ``None``. A violation raises ``ValueError`` (or
  ``TypeError`` for a wrong type of ``cancel`` and ``callables``) at construction.
- ``resolve_llm()`` returns the interface, or calls the factory ONCE and returns its result
  (one call per run, nothing cached), or returns ``None`` (the run then resolves a model the
  way ``API`` does, from ``LLM_MODEL``).
- ``derive(*, parent, max_steps=None, max_seconds=None)`` returns a child context that
  shares the interface, factory, log, callables and cancel token, names ``parent`` (the
  handle of the run that is asking, not blank) and narrows each limit that is given: a value
  above the parent's own limit raises ``ValueError`` (a parent with no limit accepts any
  valid one; ``None`` keeps the parent's, it never removes a limit). The limits are per run,
  not a shared ledger: the parent's remaining steps are not decremented by a child (shared
  budgets across runs wait for phase S).
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
from typing import TYPE_CHECKING, Any

from fsm_llm.core.definitions.errors import FSMError
from fsm_llm.core.names import NameRegistry
from fsm_llm.engine.fsm.budget import check_turn_limits
from fsm_llm.llm.interface import LLMInterface

if TYPE_CHECKING:
    from fsm_llm.runtime import EventLog

__all__ = ["CancelToken", "RunCancelledError", "RunContext"]


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
        check_turn_limits(self.max_steps, self.max_seconds)

    def resolve_llm(self) -> LLMInterface | None:
        """The interface, or one fresh ``llm_factory()`` call, or ``None`` (neither is set)."""
        if self.llm_interface is not None:
            return self.llm_interface
        if self.llm_factory is not None:
            return self.llm_factory()
        return None

    def derive(
        self,
        *,
        parent: str,
        max_steps: int | None = None,
        max_seconds: float | None = None,
    ) -> RunContext:
        """A child context: same members, ``parent`` set, each given limit narrowed.

        Raises ``ValueError`` for a blank ``parent``, an invalid limit, or a limit above the
        parent's own. ``None`` keeps the parent's limit.
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
        )


def _check_parent(parent: str | None) -> None:
    if parent is not None and (not isinstance(parent, str) or not parent.strip()):
        raise ValueError(f"parent must be a non-blank str or None, got {parent!r}")
