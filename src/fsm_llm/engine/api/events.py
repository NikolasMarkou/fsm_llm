"""The ``event_log`` opt-in of ``API``: setup, the turn scope, the stream scope, the start
record, the release of a finished conversation and the close.

Layer ``engine`` (plan 21e6f00d, step 7). These are MODULE-LEVEL functions taking the API, not
a mixin called through ``self``, on purpose: the D-021 anchor of plan 07ad3f8c
in ``conversation.py`` records that a ``self._helper()`` call resolves to a ``Mock`` under the
``Mock(spec=API)`` unbound-self tests, which would then bind a scope or fail. The log is read with
``getattr(api, "_event_log", None)`` for the same reason (``_event_log`` is an instance attribute,
so such a double does not have it and counts as "log off"). With no log every function returns
at once and binds nothing.

Interface contract (callers: ``conversation.py`` and ``lifecycle.py``; *api* is an ``API``):
    - ``open_event_log(event_log) -> (log, owned)``: ``None`` keeps the log off; an ``EventLog`` is
      shared and left open by ``close``; a ``str`` or ``os.PathLike`` makes an ``EventLog`` the API
      OWNS and closes; anything else is ``TypeError``. ``API.__init__`` assigns the pair to
      ``self._event_log`` and ``self._owns_event_log`` (a test reads these assignments).
    - ``event_scope(api, conversation_id)``: a context manager that binds the turn scope (root id
      = *conversation_id*, frame = stack depth minus one) around one manager call; a shared no-op
      context with the log off or for an id with no stack (the caller's own lookup raises the
      public error).
    - ``event_stream(api, conversation_id, stream)``: *stream* itself with the log off, else
      ``runtime.scoped_iter`` over it (the scope is bound only while the stream runs, so a stream
      consumed on another thread never resets a token it did not set).
    - ``record_started(api, conversation_id)``: appends the ``context/record`` (``source:
      "start"``) with the state after the greeting; when the log cannot be written the
      conversation is ended and the error raised, so the caller never holds an id whose every turn
      would refuse.
    - ``release_event_writer(api, conversation_id)``: closes the writer of a finished root
      conversation (no file descriptor or lock stays with it). ``close_event_log(api)``: closes an
      owned log only.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext, suppress
from typing import Any

from fsm_llm.engine.events import record_state
from fsm_llm.runtime import EventLog, TurnScope, activate, scoped_iter

from .surface import _ApiSurface

#: One shared, reusable no-op context for the log-off path.
_NO_SCOPE: AbstractContextManager[Any] = nullcontext()


def open_event_log(
    event_log: EventLog | str | os.PathLike[str] | None,
) -> tuple[EventLog | None, bool]:
    """Return ``(log, owned)`` for the ``event_log`` argument (see the module docstring)."""
    if event_log is None or isinstance(event_log, EventLog):
        return event_log, False
    if isinstance(event_log, (str, os.PathLike)):
        return EventLog(event_log), True
    raise TypeError(
        f"event_log must be an EventLog, a str or a path, got {type(event_log).__name__}"
    )


def _scope_of(api: _ApiSurface, conversation_id: str) -> TurnScope | None:
    """The turn scope of the top frame of *conversation_id*, or ``None`` (log off, no stack)."""
    log = getattr(api, "_event_log", None)
    if log is None:
        return None
    with api._stack_lock:
        stack = api.conversation_stacks.get(conversation_id)
        frame = len(stack) - 1 if stack else None
    return None if frame is None else TurnScope(log, conversation_id, frame)


def event_scope(api: _ApiSurface, conversation_id: str) -> AbstractContextManager[Any]:
    """Return the context manager that binds the turn scope (see the module docstring)."""
    scope = _scope_of(api, conversation_id)
    return _NO_SCOPE if scope is None else activate(scope)


def event_stream(
    api: _ApiSurface, conversation_id: str, stream: Iterator[str]
) -> Iterator[str]:
    """Return *stream*, scoped while it runs when the log is on."""
    scope = _scope_of(api, conversation_id)
    return stream if scope is None else scoped_iter(scope, stream)


def record_started(api: _ApiSurface, conversation_id: str) -> None:
    """Append the ``start`` record of a new conversation (see the module docstring)."""
    log = getattr(api, "_event_log", None)
    if log is None:
        return
    try:
        record_state(api.fsm_manager, log, conversation_id, source="start")
    except Exception:
        with suppress(Exception):
            api.end_conversation(conversation_id)
        raise


def release_event_writer(api: _ApiSurface, conversation_id: str) -> None:
    """Close the writer of a finished root conversation (a no-op with the log off)."""
    log = getattr(api, "_event_log", None)
    if log is not None:
        log.release(conversation_id)


def close_event_log(api: _ApiSurface) -> None:
    """Close the log when the API owns it (a log passed in stays open)."""
    log = getattr(api, "_event_log", None)
    if log is not None and getattr(api, "_owns_event_log", False):
        log.close()
