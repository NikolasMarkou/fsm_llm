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
    - ``child_event_scope(api, conversation_id)``: the scope of the frame ``push_fsm`` is about to
      add (frame = the current depth), so the pushed FSM's greeting is logged under its own
      frame; the shared no-op context with the log off.
    - ``record_push(api, conversation_id, child_id, fsm_id, definition, depth)`` and
      ``record_pop(api, conversation_id, child_id, fsm_id, merge_strategy, merged_keys, depth)``:
      append ``stack/push`` (``child_conversation_id``, ``fsm_id``, ``depth``, ``definition_ref``
      to the full definition blob) and ``stack/pop`` (``child_conversation_id``, ``fsm_id``,
      ``merge_strategy``, ``merged_key_count``, ``depth``) under the PARENT frame, after the stack
      changed and outside ``_stack_lock``. A failed write is a WARNING, not an error: the stack
      already changed and cannot be undone, and the poisoned writer refuses the next checkpoint.
    - ``record_context_update(api, conversation_id, child_id, delta)``: appends ``context/record``
      (``source: "update_context"``, ``delta_ref`` to ``engine.events.delta_blob``, ``keys`` the
      written key names; ``engine.events.record_delta`` does the write) after ``API.update_context`` wrote the delta. A failed write RAISES
      (``EventLogError``): the memory changed and the log did not, and the caller must know.
    - ``release_event_writer(api, conversation_id)``: closes the writer of a finished root
      conversation (no file descriptor or lock stays with it). ``close_event_log(api)``: closes an
      owned log only.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, nullcontext, suppress
from typing import Any

from fsm_llm.core.definitions import FSMDefinition
from fsm_llm.engine.events import record_delta, record_state
from fsm_llm.logging import logger
from fsm_llm.runtime import EventLog, EventType, TurnScope, activate, scoped_iter

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


def child_event_scope(
    api: _ApiSurface, conversation_id: str
) -> AbstractContextManager[Any]:
    """Return the context manager that binds the scope of the frame a push is adding."""
    log = getattr(api, "_event_log", None)
    if log is None:
        return _NO_SCOPE
    with api._stack_lock:
        stack = api.conversation_stacks.get(conversation_id)
        frame = len(stack) if stack else None
    return (
        _NO_SCOPE if frame is None else activate(TurnScope(log, conversation_id, frame))
    )


def _record_stack_event(
    api: _ApiSurface, kind: EventType, conversation_id: str, depth: int, **fields: Any
) -> None:
    """Append one stack event under the parent frame (``depth - 2`` after a push, ``depth - 1``
    after a pop); a failure is a WARNING (see the module docstring)."""
    log = getattr(api, "_event_log", None)
    if log is None:
        return
    parent = depth - 2 if kind is EventType.STACK_PUSH else depth - 1
    try:
        log.append(conversation_id, kind, frame=max(parent, 0), depth=depth, **fields)
    except Exception as exc:
        logger.warning(f"event log: {kind.value} not written ({type(exc).__name__})")


def record_push(
    api: _ApiSurface,
    conversation_id: str,
    child_id: str,
    fsm_id: str,
    definition: FSMDefinition,
    depth: int,
) -> None:
    """Append ``stack/push`` for a frame that was just added (a no-op with the log off)."""
    log = getattr(api, "_event_log", None)
    if log is None:
        return
    fields: dict[str, Any] = {"child_conversation_id": child_id, "fsm_id": fsm_id}
    try:
        fields["definition_ref"] = log.put_json(definition.model_dump(mode="json"))
    except Exception as exc:
        logger.warning(f"event log: definition not recorded ({type(exc).__name__})")
        fields["definition_unrecorded"] = True
    _record_stack_event(api, EventType.STACK_PUSH, conversation_id, depth, **fields)


def record_pop(
    api: _ApiSurface,
    conversation_id: str,
    child_id: str,
    fsm_id: str | None,
    merge_strategy: str,
    merged_keys: int,
    depth: int,
) -> None:
    """Append ``stack/pop`` for a frame that was just removed (a no-op with the log off)."""
    _record_stack_event(
        api,
        EventType.STACK_POP,
        conversation_id,
        depth,
        child_conversation_id=child_id,
        fsm_id=fsm_id,
        merge_strategy=merge_strategy,
        merged_key_count=merged_keys,
    )


def record_context_update(
    api: _ApiSurface, conversation_id: str, child_id: str, delta: Mapping[str, Any]
) -> None:
    """Append the ``context/record`` of an ``update_context`` write (a no-op with the log off)."""
    scope = _scope_of(api, conversation_id)
    if scope is None:
        return
    record_delta(api.fsm_manager, scope, child_id, delta, source="update_context")


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
