"""The fold of a log into a whole conversation STACK, and the turn cut a fork reads (plan
5a6a1f80, roadmap row 1.4).

Layer ``runtime``: plain dicts and this module's own pydantic models, never an engine type.
``session.fold_session`` stays root-only (a snapshot restore and ``EventLogSessionStore`` rely
on it); this module is the one reader of frames above the root, for ``fork`` and ``resume``.

Interface contract:
    - ``fold_stack(events, load_blob) -> StackProjection | None``: *events* are decoded
      envelopes in ``seq`` order (``EventLog.read`` or ``query``), *load_blob* maps a ``sha256:``
      reference to its JSON value (``EventLog.get_json``). ``None`` when no frame 0 event carries
      a state (as ``fold_session``). Otherwise ``StackProjection``: ``root`` (a
      ``SessionProjection`` of frame 0, built by the one ``session.build_projection``, whose
      ``unknown_tool_calls`` cover EVERY frame and whose ``last_seq`` is the last event read) and
      ``frames``, the frames above the root in stack order, each a ``FrameProjection``:
      ``depth`` (the stack length with this frame on top, 2 for the first child),
      ``child_conversation_id``, ``fsm_id``, ``definition_ref`` (a blob that was read and is a
      JSON object, NOT yet loaded as a definition: the restore parses it and compares the
      ``fsm_id``), ``state`` (a ``SessionProjection``; ``completion_context`` is what the log
      carried and is NOT checked against a definition here), ``return_context`` (public keys,
      values already redacted when written), ``shared_context_keys``, ``preserve_history`` and
      ``max_steps`` and ``max_seconds`` (the frame's own lifetime turn limits as pushed, ``None``
      for none; a bool, NaN, negative or non-number is ``LogCorruptError``).
      Internal-prefixed keys of ``context_data`` are dropped in every frame, whatever a blob says,
      and so are internal-prefixed keys of ``return_context`` and internal-prefixed NAMES of
      ``shared_context_keys`` (the writer never logs one, so a log that has one is forged).
    - ``turn_cut(events, at_turn) -> int``: the ``seq`` after which a fork of turn *at_turn* cuts
      the log (``EventLog.query(until_seq=...)``). A turn is one ``turn/start`` .. ``turn/end``
      pair at ANY frame (one ``converse`` or ``advance`` call), a rolled-back turn counts, the
      greeting is not a turn. ``at_turn=0`` is the genesis: the last event before the first
      ``turn/start`` (every event when there is none; 0 when the first event is a turn start);
      ``at_turn=N`` is the ``seq`` of the Nth ``turn/end``, so a delta or a push written between
      turn N and turn N+1 is not in it. ``ValueError`` for a bool, a non-int or a negative
      *at_turn*, for one past the last finished turn and for a turn that has not finished.

Failure: ``LogPrivacyError`` for an event that carries the ``lvl`` marker (a ``shape`` log, or a
``values`` log a ``shape`` writer continued; ``turn_cut`` reads no state and does not refuse);
``LogCorruptError`` for a frame the log cannot give back, see the recovery rule. Reads
only what survives to the final stack: the blobs of a frame that was popped are never read.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from fsm_llm.constants import DEFAULT_MAX_STACK_DEPTH, has_internal_prefix
from fsm_llm.runtime.events import EventType, LogCorruptError, TurnOutcome
from fsm_llm.runtime.privacy import refuse_shape
from fsm_llm.runtime.stores.session import (
    SessionProjection,
    build_projection,
    public_context,
)

_TOOL_TYPES = (EventType.TOOL_CALL.value, EventType.TOOL_RESULT.value)


class FrameProjection(BaseModel):
    """One frame above the root, as the log says it stood (see the module docstring)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    depth: int = Field(ge=2)
    child_conversation_id: str
    fsm_id: str
    definition_ref: str
    state: SessionProjection
    return_context: dict[str, Any]
    shared_context_keys: list[str]
    preserve_history: StrictBool
    max_steps: int | None = None
    max_seconds: float | None = None


class StackProjection(BaseModel):
    """The root projection and the frames above it, bottom to top."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: SessionProjection
    frames: list[FrameProjection] = Field(default_factory=list)

    @property
    def depth(self) -> int:
        """The stack length: 1 for the root alone."""
        return 1 + len(self.frames)


@dataclass
class _Frame:
    """What the replay tracks of one frame: block references only, read at the end."""

    push: Mapping[str, Any] | None  # the stack/push that made it; None for the root
    ref: str | None = None  # the state blob in force
    deltas: list[str] = field(default_factory=list)  # the delta blobs written after it
    turn_start: str | None = None  # the state blob the open turn began with

    @property
    def child_id(self) -> Any:
        return None if self.push is None else self.push.get("child_conversation_id")


def _ref(event: Mapping[str, Any], key: str) -> str | None:
    value = event.get(key)
    return value if isinstance(value, str) and value else None


def _carries_state(event: Mapping[str, Any]) -> bool:
    """True for the events that move a frame's state: both turn boundaries and a record that
    brings a state or a delta."""
    kind = event.get("type")
    if kind in (EventType.TURN_START.value, EventType.TURN_END.value):
        return True
    return kind == EventType.CONTEXT_RECORD.value and bool(
        _ref(event, "state_ref") or _ref(event, "delta_ref")
    )


def _int_field(event: Mapping[str, Any], key: str) -> int:
    value = event.get(key)
    if type(value) is not int or value < 0:
        raise LogCorruptError(
            f"event {event.get('seq')} has a bad {key} (an int of 0 or more is required)"
        )
    return value


def _apply_state(frame: _Frame, event: Mapping[str, Any]) -> None:
    """Rule (a) for one state-carrying event of the frame (the body of ``fold_session``)."""
    kind = event.get("type")
    state_ref = _ref(event, "state_ref")
    if kind == EventType.TURN_START.value and state_ref:
        frame.ref, frame.deltas = state_ref, []
        frame.turn_start = state_ref
    elif kind == EventType.TURN_END.value:
        if event.get("outcome") == TurnOutcome.ROLLED_BACK.value:
            state_ref = frame.turn_start or state_ref
        if state_ref:
            frame.ref, frame.deltas = state_ref, []
        frame.turn_start = None
    elif kind == EventType.CONTEXT_RECORD.value:
        if state_ref:
            frame.ref, frame.deltas = state_ref, []
        elif frame.ref and (delta_ref := _ref(event, "delta_ref")):
            frame.deltas.append(delta_ref)


def _at_frame(
    frames: list[_Frame], index: int, event: Mapping[str, Any]
) -> _Frame | None:
    """The frame an event of frame *index* belongs to, after rules (b) and (c).

    An event at a frame below the top means the frames above it were popped (events occur only
    at the top frame): they are dropped. An event at a frame the stack does not hold is
    ignored unless it carries a state: the greeting of a child comes before its ``stack/push``,
    and a frame no push covers has no state to give (the root, which has no push, is made by
    its own first state-carrying event).
    """
    if index < len(frames) - 1:
        del frames[index + 1 :]
    if index < len(frames):
        return frames[index]
    if not _carries_state(event):
        return None
    if index == 0:
        frames.append(_Frame(push=None))
        return frames[0]
    raise LogCorruptError(
        f"event {event.get('seq')} carries the state of frame {index}, which no "
        "stack/push of this log covers"
    )


def _push(frames: list[_Frame], event: Mapping[str, Any]) -> None:
    """Rule (b): a ``stack/push`` of depth d makes frame d-1 exist."""
    depth = _int_field(event, "depth")
    if (
        not 2 <= depth <= DEFAULT_MAX_STACK_DEPTH
        or _int_field(event, "frame") != depth - 2
    ):
        raise LogCorruptError(
            f"stack/push event {event.get('seq')} has an impossible depth"
        )
    child_id = event.get("child_conversation_id")
    if (
        isinstance(child_id, str)
        and depth - 1 < len(frames)
        and frames[depth - 1].child_id == child_id
    ):
        return  # the same push again: it is the same push, whatever the frame did since
    parent = _at_frame(frames, depth - 2, event)
    if parent is None or len(frames) != depth - 1:
        raise LogCorruptError(
            f"stack/push event {event.get('seq')} at depth {depth} follows no frame "
            f"{depth - 2} (a push or a pop record was lost)"
        )
    frame = _Frame(push=event, ref=_ref(event, "state_ref"))
    frames.append(frame)


def _pop(frames: list[_Frame], event: Mapping[str, Any]) -> None:
    """Rule (c): a ``stack/pop`` of depth d truncates the stack to d frames."""
    depth = _int_field(event, "depth")
    if depth < 1 or _int_field(event, "frame") != depth - 1:
        raise LogCorruptError(
            f"stack/pop event {event.get('seq')} has an impossible depth"
        )
    if len(frames) < depth:
        raise LogCorruptError(
            f"stack/pop event {event.get('seq')} returns to depth {depth} but the log "
            f"holds {len(frames)} frames (a push record was lost)"
        )
    if len(frames) == depth:
        return  # the pop was already inferred, or this is a replay of it
    del frames[depth:]
    state_ref = _ref(event, "state_ref")
    if state_ref:  # the parent's state after the merge
        frames[-1].ref, frames[-1].deltas = state_ref, []


def _lifetime_limit(push: Mapping[str, Any], key: str, where: str) -> Any:
    """The ``max_steps`` or ``max_seconds`` a push logged: absent or null is no limit."""
    value = push.get(key)
    if value is None:
        return None
    # `type(...) is int` is False for a bool; an int of any size compares as an int
    if type(value) is int:
        ok = value >= 1
    else:
        ok = key == "max_seconds" and type(value) is float and 0 < value < math.inf
    if not ok:
        raise LogCorruptError(
            f"{where} has a bad {key} (a number above 0, or null, is required)"
        )
    return value


def _frame_projection(
    index: int, frame: _Frame, load_blob: Callable[[str], Any], last_seq: int
) -> FrameProjection:
    """Rule (f) and the per-frame build of a frame that survived to the final stack."""
    push = frame.push or {}
    depth = index + 1
    where = f"the frame of stack/push event {push.get('seq')} (depth {depth})"
    if frame.ref is None:
        raise LogCorruptError(f"{where} has no recorded state")
    definition_ref = _ref(push, "definition_ref")
    if push.get("definition_unrecorded") or definition_ref is None:
        raise LogCorruptError(f"{where} has no recorded definition")
    if not isinstance(load_blob(definition_ref), dict):
        raise LogCorruptError(
            f"{where} has a definition blob that is not a JSON object"
        )
    return_ref = _ref(push, "return_context_ref")
    if return_ref is None:
        raise LogCorruptError(f"{where} has no recorded return context")
    # DECISION plan-2026-11-18T071726-5a6a1f80/D-009
    # The return context and the shared key NAMES of a frame come out of the log, which is a
    # file, and `pop_fsm` merges them into the parent through a writer that DOES write internal
    # keys. The writer filters them at push time (`filter_context`), so a log that has one is
    # forged: drop every internal-prefixed key here, at the one reader both `fork` and `resume`
    # use, and never merge a log-sourced `_approval_granted` or `_run_output_keys` into a parent.
    # Do NOT move this to the two callers (a third would forget), and do NOT test only the
    # context blob (the shared key names are the second route). See decisions.md D-009.
    return_context = public_context(load_blob(return_ref))
    shared = push.get("shared_context_keys")
    if isinstance(shared, list):
        shared = [
            k for k in shared if not (isinstance(k, str) and has_internal_prefix(k))
        ]
    child_id = push.get("child_conversation_id")
    state = build_projection(frame.ref, frame.deltas, load_blob, child_id, last_seq, [])
    try:
        return FrameProjection.model_validate(
            {
                "depth": depth,
                "child_conversation_id": child_id,
                "fsm_id": push.get("fsm_id"),
                "definition_ref": definition_ref,
                "state": state,
                "return_context": return_context,
                "shared_context_keys": shared,
                "preserve_history": push.get("preserve_history"),
                "max_steps": _lifetime_limit(push, "max_steps", where),
                "max_seconds": _lifetime_limit(push, "max_seconds", where),
            }
        )
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
        raise LogCorruptError(
            f"{where} is malformed (fields: {', '.join(fields)})"
        ) from None


def fold_stack(
    events: Iterable[Mapping[str, Any]], load_blob: Callable[[str], Any]
) -> StackProjection | None:
    """Return the stack a log holds (see the module docstring)."""
    # DECISION plan-2026-11-18T071726-5a6a1f80/D-004
    # The fold is DEPTH-INDEXED and replays events in `seq` order. A frame's state is the last
    # state-carrying event of that frame (`turn/start`, `turn/end`, `context/record`, the
    # creating `stack/push`, and for the parent the `stack/pop`). `stack/push` of depth d
    # replaces frame d-1 and drops deeper ones and is the ONLY commit of a frame (a state event
    # of an uncovered frame is `LogCorruptError`; its greeting events are ignored); `stack/pop`
    # of depth d truncates to d; an event at a lower frame infers a lost pop (events occur only
    # at the top frame); an open turn contributes its pre-turn state; unresolved tool calls of
    # any frame are `unknown`; an unrecorded or unreadable definition is `LogCorruptError`.
    # A duplicate or replayed push or pop is a no-op. Do NOT count pushes and pops (one lost or
    # doubled record would shift every later frame silently), do NOT guess a lost push from a
    # child's own data (its definition is gone with it), and do NOT make this read frames for
    # `fold_session` (it stays root-only). The crash sweep of step 6 asserts this rule against
    # a golden view observed from the live API. See decisions.md D-004.
    frames: list[_Frame] = []
    tools: list[Mapping[str, Any]] = []
    conversation_id: Any = None
    last_seq = 0
    for event in events:
        refuse_shape(event)  # a log without values holds no stack: never None
        last_seq = event["seq"]
        conversation_id = conversation_id or event.get("conversation_id")
        kind = event.get("type")
        if kind in _TOOL_TYPES:
            tools.append(event)
        if kind == EventType.STACK_PUSH.value:
            _push(frames, event)
        elif kind == EventType.STACK_POP.value:
            _pop(frames, event)
        else:
            target = _at_frame(frames, _int_field(event, "frame"), event)
            if target is not None and _carries_state(event):
                _apply_state(target, event)
    if not frames or frames[0].ref is None:
        return None
    root = build_projection(
        frames[0].ref, frames[0].deltas, load_blob, conversation_id, last_seq, tools
    )
    return StackProjection(
        root=root,
        frames=[
            _frame_projection(index, frame, load_blob, last_seq)
            for index, frame in enumerate(frames)
            if index > 0
        ],
    )


def turn_cut(events: Iterable[Mapping[str, Any]], at_turn: int) -> int:
    """Return the ``seq`` a fork of turn *at_turn* cuts at (see the module docstring)."""
    if type(at_turn) is not int or at_turn < 0:
        raise ValueError(f"at_turn must be an int of 0 or more, got {at_turn!r:.40}")
    previous = starts = ends = 0
    for event in events:
        kind = event.get("type")
        if kind == EventType.TURN_START.value:
            if at_turn == 0:
                return previous
            starts += 1
        elif kind == EventType.TURN_END.value:
            ends += 1
            if ends == at_turn:
                return int(event["seq"])
        previous = event["seq"]
    if at_turn == 0:
        return int(previous)
    if starts >= at_turn:
        raise ValueError(f"turn {at_turn} has not finished ({ends} turns have)")
    raise ValueError(f"at_turn {at_turn} is past the last turn ({ends} finished)")
