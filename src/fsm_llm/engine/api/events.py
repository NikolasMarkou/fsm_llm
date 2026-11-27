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
      ``runtime.scoped_iter`` over it, or ``aio.asteps`` for an async generator (the scope is
      bound only while the stream runs a step, so a stream consumed on another thread or task
      never resets a token it did not set).
    - ``arecord_started(api, conversation_id)``: appends the ``context/record`` (``source:
      "start"``) with the state after the greeting; when the log cannot be written the
      conversation is ended and the error raised, so the caller never holds an id whose every turn
      would refuse.
    - ``start_event_id(api, suppress_start, given_id) -> str | None`` and
      ``start_event_scope(api, conversation_id)``: the greeting of a logged ``start_conversation``
      (a ``restore_session`` and a ``resume`` keep their own paths). With the log on, no
      ``_suppress_start`` and no id given, the id is minted here (``uuid4``) and passed to the
      manager, and the scope of frame 0 binds around the manager call, so the greeting is an
      ``llm/request`` and ``llm/response`` under the same rules as a turn (the request fsynced
      before the call). Any failure of the start (an error or a cancel) releases the writer of
      the id, so no descriptor or lock stays with a conversation that never existed; the log keeps
      the request and its ``provider/error``. ``None`` and the shared no-op context otherwise.
    - ``child_event_scope(api, conversation_id)``: the scope of the frame ``push_fsm`` is about to
      add (frame = the current depth), so the pushed FSM's greeting is logged under its own
      frame; the shared no-op context with the log off.
    - ``record_push(api, conversation_id, child_id, fsm_id, definition, depth, *, return_context,
      shared_context_keys, preserve_history)`` and ``record_pop(api, conversation_id, child_id,
      fsm_id, merge_strategy, merged_keys, depth, *, parent_id)``: append ``stack/push``
      (``child_conversation_id``, ``fsm_id``, ``depth``, ``definition_ref`` to the full definition
      blob, ``state_ref`` of the child's post-greeting state, ``return_context_ref`` (through
      ``filter_context``), ``shared_context_keys``, ``preserve_history`` and the frame's lifetime
      limits ``max_steps`` and ``max_seconds`` when it has them, from ``api.budget.logged_limits``;
      a restore rebuilds the frame with the same limits and a fresh count) and ``stack/pop``
      (``child_conversation_id``, ``fsm_id``, ``merge_strategy``, ``merged_key_count``, ``depth``,
      ``state_ref`` of the parent *parent_id* after the merge) under the PARENT frame, after the
      stack changed and outside ``_stack_lock``: ``runtime.stores.stack.fold_stack`` rebuilds the
      stack from them. A failed write is a WARNING, not an error: the stack already changed and
      cannot be undone, and the poisoned writer refuses the next checkpoint. An optional field
      that cannot be built (a blob or a state read that fails) is left out with a WARNING and the
      event is still written; the fold refuses a frame it cannot restore from what is missing
      (D-005 of plan 5a6a1f80: the writes stay best effort, the fold detects the gap).
    - ``record_context_update(api, conversation_id, child_id, delta)``: appends ``context/record``
      (``source: "update_context"``, ``delta_ref`` to ``engine.events.delta_blob``, ``keys`` the
      written key names; ``engine.events.record_delta`` does the write) after ``API.update_context`` wrote the delta. A failed write RAISES
      (``EventLogError``): the memory changed and the log did not, and the caller must know.
    - ``record_restored(api, conversation_id, restored_from, last_seq)``: appends the
      ``context/record`` (``source: "restore"``, ``restored_from: {conversation_id, seq?}``) with the
      state after every seed of ``restore_session``; the first event of the new log
      (``start_conversation`` writes no start record for a restore). A failed write RAISES, and the
      restore's own teardown ends the half-restored conversation.
    - ``record_forked(api, conversation_id, forked_from)``: appends the ``context/record``
      (``source: "fork"``, ``forked_from: {conversation_id, seq, turn}``) with the state of the new
      root after every seed of ``fork``: the first event of the fork's log. A failed write RAISES
      and the restore's teardown ends the half-built fork.
    - ``record_restored_frame(api, conversation_id, child_id, fsm_id, definition, depth, *,
      return_context, shared_context_keys, preserve_history)``: appends a ``stack/push`` marked
      ``restored: true`` for a frame ``fork`` rebuilt, with the SAME fields a live push carries and
      the frame's current state, so the fork's own log folds to its stack with no help from the
      source (a fork of a fork). Unlike ``record_push`` it is strict: any field that cannot be
      written RAISES, because a fork log that lost a frame would restore a shorter stack.
    - ``record_resumed(api, conversation_id, top_id, top_frame, resumed_from, unknown_calls)``:
      appends the ``context/record`` (``source: "resume"``, ``resumed_from: {seq}``) with the state of
      the TOP frame (manager id *top_id*, frame *top_frame*, always the top: an event at a lower
      frame would read as a lost pop to the stack fold), then ONE batch of ``tool/result`` events
      (``status: "unknown"``, ``call_frame`` the frame of the call) for each tool call the log
      could not resolve, also at the top frame, so the next fold reports none. Results come after
      the record, so a resume that fails leaves the calls unresolved to be reported again. A failed
      write RAISES and the restore's own teardown ends the half-resumed conversation.
    - Level ``shape`` (``log.stores_values`` False, plan 5a6a1f80): ``record_push``, ``record_pop``
      and ``record_restored_frame`` build no definition, return context or state blob and write
      the names, ids, counts and flags only; a ``definition`` blob is stored with its
      ``RedactionKind`` for the user's hook at level ``values``.
    - ``release_event_writer(api, conversation_id)``: closes the writer of a finished root
      conversation (no file descriptor or lock stays with it). ``close_event_log(api)``: closes an
      owned log only.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import uuid
from collections.abc import AsyncGenerator, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from functools import partial
from typing import Any, TypeVar, cast

from fsm_llm.aio import SYNC, asteps
from fsm_llm.core.definitions import FSMDefinition
from fsm_llm.engine.events import (
    arecord_state,
    record_delta,
    record_state,
    state_ref_later,
    state_ref_of,
)
from fsm_llm.logging import logger
from fsm_llm.runtime import (
    EventLog,
    EventType,
    RedactionKind,
    TurnScope,
    activate,
    filter_context,
    scoped_iter,
    warn_on_failure,
)

from .budget import logged_limits
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


_Stream = TypeVar("_Stream", AsyncGenerator[str, None], Iterator[str])


def event_stream(api: _ApiSurface, conversation_id: str, stream: _Stream) -> _Stream:
    """Return *stream*, scoped while it runs when the log is on (a sync or an async stream)."""
    scope = _scope_of(api, conversation_id)
    if scope is None:
        return stream
    if isinstance(stream, AsyncGenerator):
        return cast(_Stream, asteps(lambda: activate(scope), stream))
    return cast(_Stream, scoped_iter(scope, stream))


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


def start_event_id(
    api: _ApiSurface, suppress_start: bool, given_id: str | None
) -> str | None:
    """A minted conversation id when a start must log its greeting, else ``None``."""
    if getattr(api, "_event_log", None) is None or suppress_start or given_id:
        return None
    return str(uuid.uuid4())


@contextmanager
def start_event_scope(api: _ApiSurface, conversation_id: str | None) -> Iterator[None]:
    """Bind frame 0 of *conversation_id* around a start (a no-op for ``None``); a failure of
    the start releases the writer, and the failure is re-raised."""
    log = getattr(api, "_event_log", None)
    if log is None or conversation_id is None:
        yield
        return
    try:
        with activate(TurnScope(log, conversation_id, 0)):
            yield
    except BaseException:
        log.release(conversation_id)
        raise


def _record_stack_event(
    api: _ApiSurface, kind: EventType, conversation_id: str, depth: int, **fields: Any
) -> None:
    """Append one stack event under the parent frame (``depth - 2`` after a push, ``depth - 1``
    after a pop); a failure is a WARNING (see the module docstring)."""
    log = getattr(api, "_event_log", None)
    if log is None:
        return
    parent = depth - 2 if kind is EventType.STACK_PUSH else depth - 1
    with warn_on_failure(kind):
        log.append(conversation_id, kind, frame=max(parent, 0), depth=depth, **fields)


def _optional_field(
    fields: dict[str, Any], name: str, build: Callable[[], Any]
) -> None:
    """Set ``fields[name]`` to ``build()``; a failure leaves it out with a WARNING (type name only)."""
    try:
        fields[name] = build()
    except Exception as exc:
        logger.warning(
            f"event log: stack field {name} not recorded ({type(exc).__name__})"
        )


def _later(log: EventLog, kind: RedactionKind, body: Any) -> Callable[[], str]:
    """The call that stores *body*, read now, as a blob (it runs in the job)."""
    return lambda: log.put_json(body, kind=kind)


# DECISION plan-2026-11-26T213359-77e5fef9/D-008
# [PARTLY SUPERSEDES clause 6 of D-017 of plan 3f4530d3: the push is no longer written on the
# loop.] In async mode the three blobs and the `stack/push` line are ONE job queued behind the
# tree's jobs (`lane_after`: after the child's greeting, before the caller's next write); every
# body is read HERE, on the loop (a pop or an update right after changes them). An inline
# `stack/pop`, `run/end` or `update_context` cannot land before the push: `EventLog.append`
# waits for the tree's deferred job (else the fold reads a phantom frame, or raises
# LogCorruptError for a child record no push covers). Do NOT await a lane call after the frame
# append (a cancel point, D-020 of plan baa37f40), read state, context or definition inside the
# job, queue the pop instead of waiting (`update_context` raises: it cannot be deferred), run
# the job inline outside SYNC mode or add a public name. See decisions.md D-008.
def record_push(
    api: _ApiSurface,
    conversation_id: str,
    child_id: str,
    fsm_id: str,
    definition: FSMDefinition,
    depth: int,
    *,
    return_context: Mapping[str, Any],
    shared_context_keys: Sequence[str],
    preserve_history: bool,
    limits: Mapping[str, Any] | None = None,
) -> None:
    """Append ``stack/push`` for a frame just added (a no-op with the log off; see the anchor).

    In async mode the write is one lane job (inline when no job ever started the lane of the
    tree, and in sync mode).
    """
    log = getattr(api, "_event_log", None)
    if log is None:
        return
    fields: dict[str, Any] = {
        "child_conversation_id": child_id,
        "fsm_id": fsm_id,
        "shared_context_keys": list(shared_context_keys),
        "preserve_history": bool(preserve_history),
        **logged_limits(limits),
    }
    stores: dict[str, Any] = {}  # field name -> the call that stores its blob (the job)
    if log.stores_values:
        try:
            stores["definition_ref"] = _later(
                log, RedactionKind.DEFINITION, definition.model_dump(mode="json")
            )
        except Exception as exc:
            logger.warning(f"event log: definition not recorded ({type(exc).__name__})")
            fields["definition_unrecorded"] = True
        _optional_field(
            stores,
            "return_context_ref",
            lambda: _later(log, RedactionKind.DELTA, filter_context(return_context)),
        )
        _optional_field(
            stores,
            "state_ref",
            lambda: state_ref_later(api.fsm_manager, log, child_id),
        )

    def write() -> None:
        for name, store in stores.items():
            _optional_field(fields, name, store)
        if "definition_ref" in stores and "definition_ref" not in fields:
            fields["definition_unrecorded"] = True
        _record_stack_event(api, EventType.STACK_PUSH, conversation_id, depth, **fields)

    if SYNC.get():
        write()
    else:
        log.lane_after(conversation_id, partial(contextvars.copy_context().run, write))


def record_pop(
    api: _ApiSurface,
    conversation_id: str,
    child_id: str,
    fsm_id: str | None,
    merge_strategy: str,
    merged_keys: int,
    depth: int,
    *,
    parent_id: str,
) -> None:
    """Append ``stack/pop`` for a frame that was just removed (a no-op with the log off)."""
    log = getattr(api, "_event_log", None)
    if log is None:
        return
    fields: dict[str, Any] = {
        "child_conversation_id": child_id,
        "fsm_id": fsm_id,
        "merge_strategy": merge_strategy,
        "merged_key_count": merged_keys,
    }
    if log.stores_values:
        _optional_field(
            fields, "state_ref", lambda: state_ref_of(api.fsm_manager, log, parent_id)
        )
    _record_stack_event(api, EventType.STACK_POP, conversation_id, depth, **fields)


def record_context_update(
    api: _ApiSurface, conversation_id: str, child_id: str, delta: Mapping[str, Any]
) -> None:
    """Append the ``context/record`` of an ``update_context`` write (a no-op with the log off)."""
    scope = _scope_of(api, conversation_id)
    if scope is None:
        return
    record_delta(api.fsm_manager, scope, child_id, delta, source="update_context")


async def arecord_started(api: _ApiSurface, conversation_id: str) -> None:
    """Append the ``start`` record of a new conversation (the first write of a silent one)."""
    log = getattr(api, "_event_log", None)
    if log is None:
        return
    try:
        await arecord_state(api.fsm_manager, log, conversation_id, source="start")
    except (Exception, asyncio.CancelledError):
        # a cancel on the record hop is a cancelled start: nothing may stay registered
        with suppress(Exception):
            api.end_conversation(conversation_id)
        raise


def record_restored(
    api: _ApiSurface, conversation_id: str, restored_from: str, last_seq: Any
) -> None:
    """Append the ``restore`` record of a restored conversation: its state after every seed and
    ``restored_from`` (the old id, and the last event the fold read when the session came from a
    log). The first event of the new log. A no-op with the log off; a failing write raises."""
    log = getattr(api, "_event_log", None)
    if log is None:
        return
    lineage: dict[str, Any] = {"conversation_id": restored_from}
    if isinstance(last_seq, int) and not isinstance(last_seq, bool):
        lineage["seq"] = last_seq
    record_state(
        api.fsm_manager, log, conversation_id, source="restore", restored_from=lineage
    )


def record_forked(
    api: _ApiSurface, conversation_id: str, forked_from: Mapping[str, Any]
) -> None:
    """Append the ``fork`` record of a new fork root (a no-op with the log off; a failing write raises)."""
    log = getattr(api, "_event_log", None)
    if log is None:
        return
    record_state(
        api.fsm_manager,
        log,
        conversation_id,
        source="fork",
        forked_from=dict(forked_from),
    )


def record_restored_frame(
    api: _ApiSurface,
    conversation_id: str,
    child_id: str,
    fsm_id: str,
    definition: FSMDefinition,
    depth: int,
    *,
    return_context: Mapping[str, Any],
    shared_context_keys: Sequence[str],
    preserve_history: bool,
    limits: Mapping[str, Any] | None = None,
) -> None:
    """Append the strict, ``restored`` ``stack/push`` of a frame ``fork`` rebuilt."""
    log = getattr(api, "_event_log", None)
    if log is None:
        return
    refs: dict[str, Any] = {}
    if log.stores_values:
        refs = {
            "definition_ref": log.put_json(
                definition.model_dump(mode="json"), kind=RedactionKind.DEFINITION
            ),
            "return_context_ref": log.put_json(
                filter_context(return_context), kind=RedactionKind.DELTA
            ),
            "state_ref": state_ref_of(api.fsm_manager, log, child_id),
        }
    log.append(
        conversation_id,
        EventType.STACK_PUSH,
        frame=depth - 2,
        depth=depth,
        restored=True,
        child_conversation_id=child_id,
        fsm_id=fsm_id,
        shared_context_keys=list(shared_context_keys),
        preserve_history=bool(preserve_history),
        **logged_limits(limits),
        **refs,
    )


def record_resumed(
    api: _ApiSurface,
    conversation_id: str,
    top_id: str,
    top_frame: int,
    resumed_from: Mapping[str, Any],
    unknown_calls: Sequence[Mapping[str, Any]],
) -> None:
    """Append the ``resume`` record and the ``unknown`` results of a resumed tree (strict)."""
    log = getattr(api, "_event_log", None)
    if log is None:
        return
    # DECISION plan-2026-11-18T071726-5a6a1f80/D-007
    # Both writes go to the TOP frame of the resumed stack (`top_frame`), never to frame 0 and
    # never to the frame a tool call was made at. The stack fold infers a lost `stack/pop` from
    # any event at a frame below the top (D-004 clause c), so a record at a lower frame would
    # make the next fold drop every frame above it: a runner's tool call at frame 0 stays
    # unresolved while its child frames are open. Do NOT call `record_state(frame=0)` here, do NOT
    # write a closing `turn/end` for a killed turn (it fakes an outcome the engine never decided;
    # the next `turn/start` supersedes the open one), and do NOT write the `unknown` results
    # before the resume record: a resume that fails must leave the calls unresolved so the next
    # resume reports them again. See decisions.md D-007.
    record_state(
        api.fsm_manager,
        log,
        top_id,
        source="resume",
        frame=top_frame,
        root_id=conversation_id,
        resumed_from=dict(resumed_from),
    )
    if unknown_calls:
        log.append_batch(
            conversation_id,
            [
                {
                    "type": EventType.TOOL_RESULT,
                    "name": call.get("name"),
                    "call_id": call.get("call_id"),
                    "status": "unknown",
                    "call_frame": call.get("frame"),
                }
                for call in unknown_calls
            ],
            frame=top_frame,
        )


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
