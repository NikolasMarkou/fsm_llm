"""The stack in the event log and its fold (plan 5a6a1f80 step 2, roadmap row 1.4): the push and pop
records carry what a stack restore needs, ``fold_stack`` replays them under the recovery rule
(D-004), ``turn_cut`` finds the cut of a fork.

Three layers: the fold on hand-built tapes (one test per clause of the recovery rule, so a gap, a
duplicate or a forged record is exact), the turn cut, and a real stacked conversation
(``ScriptedLLM``) whose log is folded after every state-carrying append and compared with what the
LIVE ``API`` showed at that moment (the golden view of the crash sweep of step 6: it is observed
from the engine, never computed by a second fold). Bytes, counts and sequences, never timing.
Unverified until the Windows run: the byte-level tampering helper opens a blob in binary mode.
"""

from __future__ import annotations

import asyncio
import copy
import threading
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from fsm_llm import API
from fsm_llm.engine.events import record_state
from fsm_llm.runtime import EventLog, EventType, LogCorruptError, LogWriteError
from fsm_llm.runtime import lanes as lanes_module
from fsm_llm.runtime.lanes import Lanes
from fsm_llm.runtime.stores import (
    StackProjection,
    fold_session,
    fold_stack,
    turn_cut,
)
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm.event_log_child import chat_fsm
from tests.test_fsm_llm.event_log_child import stack_view as _view
from tests.test_fsm_llm.support import _state

_START, _END, _RECORD = (
    EventType.TURN_START,
    EventType.TURN_END,
    EventType.CONTEXT_RECORD,
)
_FOLD_TYPES = (_START, _END, _RECORD, EventType.STACK_PUSH, EventType.STACK_POP)


def _blob(state: str = "a", **context: Any) -> dict[str, Any]:
    return {
        "fsm_id": "fsm-1",
        "current_state": state,
        "context_data": context,
        "conversation_history": [],
        "conversation_summary": None,
        "working_memory": None,
        "hidden_buffers": [],
        "provenance": {},
        "internal": {},
    }


_DROP: Any = object()  # a field a test removes from the default push


class _Tape:
    """Hand-built events of one root conversation ``r1``, written through a real ``EventLog``."""

    def __init__(self, log: EventLog) -> None:
        self.log = log

    def _ref(self, state: str, **context: Any) -> str:
        return self.log.put_json(_blob(state, **context))

    def _event(self, kind: EventType, frame: int, state: str, **fields: Any) -> int:
        context = fields.pop("context", {})
        ref = self._ref(state, **context)
        return self.log.append("r1", kind, frame=frame, state_ref=ref, **fields)

    def start(self, frame: int, state: str, **context: Any) -> int:
        return self._event(_START, frame, state, context=context)

    def end(
        self, frame: int, state: str, outcome: str = "committed", **context: Any
    ) -> int:
        return self._event(_END, frame, state, outcome=outcome, context=context)

    def turn(self, frame: int, state: str, **context: Any) -> None:
        self.start(frame, state, **context)
        self.end(frame, state, **context)

    def record(self, frame: int, state: str, source: str = "start", **context: Any):
        return self._event(_RECORD, frame, state, source=source, context=context)

    def delta(self, frame: int, context: dict[str, Any]) -> int:
        ref = self.log.put_json({"context": context, "internal": {}})
        return self.log.append("r1", _RECORD, frame=frame, source="u", delta_ref=ref)

    def push(
        self, depth: int, child: str, state: str | None = "hello", **over: Any
    ) -> int:
        fields: dict[str, Any] = {
            "depth": depth,
            "child_conversation_id": child,
            "fsm_id": f"fsm_{child}",
            "definition_ref": self.log.put_json({"name": child}),
            "return_context_ref": self.log.put_json({}),
            "shared_context_keys": [],
            "preserve_history": False,
        }
        if state is not None:
            fields["state_ref"] = self._ref(state)
        fields.update(over)
        fields = {k: v for k, v in fields.items() if v is not _DROP}
        return self.log.append(
            "r1", EventType.STACK_PUSH, frame=max(depth - 2, 0), **fields
        )

    def pop(self, depth: int, state: str | None = None, **context: Any) -> int:
        fields: dict[str, Any] = {"depth": depth, "child_conversation_id": "kid"}
        if state is not None:
            fields["state_ref"] = self._ref(state, **context)
        return self.log.append("r1", EventType.STACK_POP, frame=depth - 1, **fields)

    def fold(self, until: int | None = None) -> StackProjection | None:
        return fold_stack(self.log.query("r1", until_seq=until), self.log.get_json)

    def states(self) -> list[str]:
        folded = self.fold()
        assert folded is not None
        return [folded.root.current_state] + [
            f.state.current_state for f in folded.frames
        ]


def _tape(log: EventLog) -> _Tape:
    tape = _Tape(log)
    tape.record(0, "root")
    return tape


class TestTheRecoveryRule:
    def test_a_push_makes_the_frame_with_its_state_and_options(self, log: EventLog):
        tape = _tape(log)
        tape.push(
            2,
            "kid",
            "hello",
            shared_context_keys=["a", "b"],
            preserve_history=True,
            return_context_ref=log.put_json({"answer": 42}),
        )
        folded = tape.fold()
        assert folded is not None and folded.depth == 2
        (frame,) = folded.frames
        assert frame.depth == 2 and frame.child_conversation_id == "kid"
        assert frame.fsm_id == "fsm_kid"
        assert frame.state.current_state == "hello"
        assert frame.state.conversation_id == "kid"
        assert frame.return_context == {"answer": 42}
        assert frame.shared_context_keys == ["a", "b"] and frame.preserve_history
        assert log.get_json(frame.definition_ref) == {"name": "kid"}
        assert folded.root.current_state == "root"

    def test_each_frame_keeps_the_last_state_event_of_its_own(self, log: EventLog):
        tape = _tape(log)
        tape.turn(0, "root2")
        tape.push(2, "kid", "k0")
        tape.turn(1, "k1")
        tape.push(3, "grandkid", "g0")
        tape.turn(2, "g1")
        assert tape.states() == ["root2", "k1", "g1"]

    def test_a_rolled_back_turn_returns_the_frame_to_its_own_turn_start(
        self, log: EventLog
    ):
        tape = _tape(log)
        tape.push(2, "kid", "k0")
        tape.start(1, "k0")
        tape.end(1, "liar", outcome="rolled_back")
        assert tape.states() == ["root", "k0"]

    def test_an_open_turn_contributes_its_pre_turn_state(self, log: EventLog):
        tape = _tape(log)
        tape.push(2, "kid", "k0")
        tape.turn(1, "k1")
        tape.start(1, "k1")  # killed here: no turn/end
        assert tape.states() == ["root", "k1"]
        tape.start(1, "k1b")  # a later turn start supersedes it
        assert tape.states() == ["root", "k1b"]

    def test_a_delta_applies_to_the_frame_it_names(self, log: EventLog):
        tape = _tape(log)
        tape.push(2, "kid", "k0")
        tape.delta(1, {"note": "n"})
        folded = tape.fold()
        assert folded is not None
        assert folded.frames[0].state.context_data == {"note": "n"}
        assert folded.root.context_data == {}

    def test_a_duplicate_push_is_the_same_push_even_after_the_child_moved(
        self, log: EventLog
    ):
        tape = _tape(log)
        tape.push(2, "kid", "k0")
        assert tape.states() == ["root", "k0"]
        tape.push(2, "kid", "k0")  # an immediate duplicate
        tape.turn(1, "k1")
        tape.push(2, "kid", "k0")  # a replay long after
        assert tape.states() == ["root", "k1"]

    def test_a_push_at_an_occupied_depth_replaces_the_frame(self, log: EventLog):
        tape = _tape(log)
        tape.push(2, "kid", "k0")
        tape.push(3, "grandkid", "g0")
        tape.push(2, "other", "o0")  # the pops of both were lost
        folded = tape.fold()
        assert folded is not None and folded.depth == 2
        assert folded.frames[0].child_conversation_id == "other"
        assert tape.states() == ["root", "o0"]

    def test_a_pop_truncates_and_gives_the_parent_its_merged_state(self, log: EventLog):
        tape = _tape(log)
        tape.push(2, "kid", "k0")
        tape.turn(1, "k1")
        tape.pop(1, "root", result=1)
        folded = tape.fold()
        assert folded is not None and folded.depth == 1
        assert folded.root.context_data == {"result": 1}

    def test_a_replayed_pop_changes_nothing(self, log: EventLog):
        tape = _tape(log)
        tape.push(2, "kid", "k0")
        tape.pop(1, "root", result=1)
        tape.turn(0, "root2", result=1, more=2)
        tape.pop(1, "root", result=1)  # the same record again
        folded = tape.fold()
        assert folded is not None and folded.depth == 1
        assert folded.root.current_state == "root2"
        assert folded.root.context_data == {"result": 1, "more": 2}

    def test_a_missing_pop_is_inferred_from_an_event_at_a_lower_frame(
        self, log: EventLog
    ):
        tape = _tape(log)
        tape.push(2, "kid", "k0")
        tape.push(3, "grandkid", "g0")
        tape.start(1, "k0")  # only the grandchild's pop was lost
        assert tape.fold().depth == 2  # type: ignore[union-attr]
        tape.turn(0, "root2")  # the child's pop was lost as well
        assert tape.fold().depth == 1  # type: ignore[union-attr]
        assert tape.states() == ["root2"]

    def test_the_greeting_events_before_a_push_are_ignored(self, log: EventLog):
        tape = _tape(log)
        log.append("r1", EventType.LLM_REQUEST, frame=1, source="engine")
        log.append("r1", EventType.LLM_RESPONSE, frame=1, source="engine")
        assert tape.fold().depth == 1  # type: ignore[union-attr]
        tape.push(2, "kid", "k0")
        assert tape.states() == ["root", "k0"]

    @pytest.mark.parametrize("frame", [1, 2, 5])
    @pytest.mark.parametrize("kind", ["start", "end", "record"])
    def test_a_state_event_of_a_frame_no_push_covers_is_corrupt(
        self, log: EventLog, kind: str, frame: int
    ):
        tape = _tape(log)
        getattr(tape, kind)(frame, "x")
        with pytest.raises(LogCorruptError, match="no stack/push"):
            tape.fold()

    def test_a_delta_of_a_frame_no_push_covers_is_corrupt(self, log: EventLog):
        tape = _tape(log)
        tape.delta(1, {"k": 1})
        with pytest.raises(LogCorruptError, match="no stack/push"):
            tape.fold()

    @pytest.mark.parametrize(
        "script",
        [
            lambda t: t.push(3, "grandkid", "g0"),  # the push of the child was lost
            lambda t: (t.push(2, "kid", "k0"), t.pop(3, "root")),
        ],
    )
    def test_a_record_that_needs_a_lost_frame_is_corrupt(
        self, log: EventLog, script: Any
    ):
        tape = _tape(log)
        script(tape)
        with pytest.raises(LogCorruptError, match="lost"):
            tape.fold()

    def test_a_push_before_the_root_has_a_state_is_corrupt(self, log: EventLog):
        tape = _Tape(log)
        tape.push(2, "kid", "k0")
        with pytest.raises(LogCorruptError, match="lost"):
            tape.fold()

    @pytest.mark.parametrize(
        "over",
        [
            {"depth": 1},
            {"depth": 11},
            {"depth": True},
            {"depth": "2"},
            {"depth": None},
            {"depth": 3},  # the frame field says the parent of depth 2
        ],
    )
    def test_an_impossible_push_is_corrupt(self, log: EventLog, over: dict[str, Any]):
        tape = _tape(log)
        fields = {"depth": 2, "child_conversation_id": "kid", **over}
        log.append("r1", EventType.STACK_PUSH, frame=0, **fields)
        with pytest.raises(LogCorruptError, match=r"depth|lost"):
            tape.fold()

    @pytest.mark.parametrize("frame", [True, "0", -1, None, 1.5])
    def test_a_frame_that_is_not_an_index_is_corrupt(self, log: EventLog, frame: Any):
        tape = _tape(log)
        bad = {**next(log.read("r1")), "seq": 2, "frame": frame, "type": "llm/request"}
        with pytest.raises(LogCorruptError, match="bad frame"):
            fold_stack(iter([next(log.read("r1")), bad]), log.get_json)
        assert tape.fold() is not None

    def test_the_unresolved_tool_calls_of_every_frame_are_unknown(self, log: EventLog):
        tape = _tape(log)
        tape.push(2, "kid", "k0")
        log.append(
            "r1", EventType.TOOL_CALL, frame=1, call_id="c1", name="send", arguments={}
        )
        log.append("r1", EventType.TOOL_CALL, frame=1, call_id="c2", name="read")
        log.append("r1", EventType.TOOL_RESULT, frame=1, call_id="c2", status="ok")
        folded = tape.fold()
        assert folded is not None
        assert [(c["call_id"], c["frame"]) for c in folded.root.unknown_tool_calls] == [
            ("c1", 1)
        ]
        tape.pop(1, "root")  # the frame is gone, the call is still unresolved
        folded = tape.fold()
        assert folded is not None and len(folded.root.unknown_tool_calls) == 1

    def test_internal_keys_are_dropped_in_every_frame(self, log: EventLog):
        tape = _Tape(log)
        tape.record(0, "root", _approval_granted={"tool": "x"}, kept=1)
        tape.push(2, "kid", "k0")
        tape.start(1, "k1", kept=2, _approval_granted={"tool": "x"})
        tape.delta(1, {"later": 3, "system_x": 9})
        folded = tape.fold()
        assert folded is not None
        assert folded.root.context_data == {"kept": 1}
        assert folded.frames[0].state.context_data == {"kept": 2, "later": 3}

    def test_a_push_with_no_state_waits_for_the_childs_first_turn(self, log: EventLog):
        tape = _tape(log)
        tape.push(2, "kid", None)
        with pytest.raises(LogCorruptError, match="no recorded state"):
            tape.fold()
        tape.start(1, "k0")
        assert tape.states() == ["root", "k0"]

    def test_no_state_at_all_folds_to_none(self, log: EventLog):
        assert fold_stack(iter(()), log.get_json) is None
        log.append("r1", EventType.LLM_REQUEST, frame=0, source="engine")
        assert fold_stack(log.read("r1"), log.get_json) is None
        log.append("r1", _RECORD, frame=0, source="x", delta_ref=log.put_json({}))
        assert fold_stack(log.read("r1"), log.get_json) is None


def _blob_path(log: EventLog, ref: str) -> Any:
    digest = ref.removeprefix("sha256:")
    return log.root / "blobs" / digest[:2] / digest[2:]


class TestADefinitionTheLogCannotGiveBack:
    def _frame(self, log: EventLog, **over: Any) -> _Tape:
        tape = _tape(log)
        tape.push(2, "kid", "k0", **over)
        return tape

    def test_an_unrecorded_definition_is_corrupt(self, log: EventLog):
        tape = self._frame(log, definition_ref=_DROP, definition_unrecorded=True)
        with pytest.raises(LogCorruptError, match="no recorded definition"):
            tape.fold()

    def test_a_push_with_no_definition_reference_is_corrupt(self, log: EventLog):
        tape = self._frame(log, definition_ref=_DROP)
        with pytest.raises(LogCorruptError, match="no recorded definition"):
            tape.fold()

    def test_a_deleted_definition_blob_is_corrupt(self, log: EventLog):
        ref = log.put_json({"name": "gone"})
        tape = self._frame(log, definition_ref=ref)
        _blob_path(log, ref).unlink()
        with pytest.raises(LogCorruptError):
            tape.fold()

    def test_a_tampered_definition_blob_is_corrupt(self, log: EventLog):
        ref = log.put_json({"name": "forged"})
        tape = self._frame(log, definition_ref=ref)
        path = _blob_path(log, ref)
        path.write_bytes(path.read_bytes().replace(b"forged", b"FORGED"))
        with pytest.raises(LogCorruptError):
            tape.fold()

    def test_a_definition_that_is_not_an_object_is_corrupt(self, log: EventLog):
        tape = self._frame(log, definition_ref=log.put_json(["states"]))
        with pytest.raises(LogCorruptError, match="not a JSON object"):
            tape.fold()

    @pytest.mark.parametrize(
        ("over", "message"),
        [
            ({"return_context_ref": _DROP}, "return context"),
            ({"shared_context_keys": _DROP}, "malformed"),
            ({"shared_context_keys": "ab"}, "malformed"),
            ({"shared_context_keys": [1]}, "malformed"),
            ({"preserve_history": _DROP}, "malformed"),
            ({"preserve_history": "yes"}, "malformed"),
            ({"fsm_id": _DROP}, "malformed"),
            ({"child_conversation_id": _DROP}, "malformed|conversation_id"),
        ],
    )
    def test_a_push_missing_or_misshapen_a_restore_field_is_corrupt(
        self, log: EventLog, over: dict[str, Any], message: str
    ):
        tape = self._frame(log, **over)
        with pytest.raises(LogCorruptError, match=message):
            tape.fold()

    def test_a_return_context_that_is_not_an_object_is_corrupt(self, log: EventLog):
        tape = self._frame(log, return_context_ref=log.put_json([1]))
        with pytest.raises(LogCorruptError, match="malformed"):
            tape.fold()

    def test_a_popped_frame_is_never_read(self, log: EventLog):
        ref = log.put_json({"name": "popped"})
        tape = self._frame(log, definition_ref=ref)
        tape.pop(1, "root")
        _blob_path(log, ref).unlink()
        folded = tape.fold()
        assert folded is not None and folded.depth == 1


class TestTheTurnCut:
    def _tape(self, log: EventLog) -> _Tape:
        tape = _tape(log)  # seq 1: start record
        tape.turn(0, "t1")  # 2, 3
        tape.push(2, "kid", "k0")  # 4
        tape.turn(1, "k1")  # 5, 6
        tape.start(1, "k2")  # 7
        tape.end(1, "liar", outcome="rolled_back")  # 8: a rolled-back turn counts
        tape.pop(1, "root")  # 9
        tape.start(0, "t3")  # 10: open
        return tape

    def _cut(self, tape: _Tape, at_turn: Any) -> int:
        return turn_cut(tape.log.read("r1"), at_turn)

    def test_the_genesis_and_the_nth_turn_end_of_any_frame(self, log: EventLog):
        tape = self._tape(log)  # genesis: before turn 1; turn 3 is the rolled-back one
        assert [self._cut(tape, n) for n in (0, 1, 2, 3)] == [1, 3, 6, 8]

    def test_a_push_between_two_turns_is_not_in_the_earlier_cut(self, log: EventLog):
        tape = self._tape(log)
        cut = self._cut(tape, 1)
        folded = tape.fold(until=cut)
        assert folded is not None and folded.depth == 1
        assert tape.states() == ["t3"]  # the whole log: the pop is in, turn 3 is open
        folded = tape.fold(until=self._cut(tape, 2))
        assert folded is not None and folded.depth == 2

    @pytest.mark.parametrize(
        ("at_turn", "message"), [(4, "turn 4 has not finished"), (5, "past the last")]
    )
    def test_an_unfinished_or_missing_turn_is_refused(
        self, log: EventLog, at_turn: int, message: str
    ):
        with pytest.raises(ValueError, match=message):
            self._cut(self._tape(log), at_turn)

    @pytest.mark.parametrize("bad", [True, False, -1, 1.0, "1", None, [1], 2**70 + 0.5])
    def test_an_at_turn_that_is_not_a_plain_count_is_refused(
        self, log: EventLog, bad: Any
    ):
        with pytest.raises(ValueError, match="at_turn"):
            self._cut(self._tape(log), bad)

    def test_genesis_edges(self, log: EventLog):
        assert turn_cut(iter(()), 0) == 0
        assert turn_cut(iter([{"seq": 1, "type": "turn/start"}]), 0) == 0
        assert turn_cut(iter([{"seq": 7, "type": "llm/request"}]), 0) == 7

    def test_the_scan_stops_at_the_cut(self, log: EventLog):
        tape = self._tape(log)
        seen: list[int] = []

        def lazy() -> Iterator[dict[str, Any]]:
            for event in tape.log.read("r1"):
                seen.append(event["seq"])
                yield event

        assert turn_cut(lazy(), 1) == 3
        assert seen == [1, 2, 3]


# --------------------------------------------------------------
# A real stacked conversation
# --------------------------------------------------------------

_ROOT: dict[str, Any] = chat_fsm()
_KID: dict[str, Any] = {
    "name": "Kid",
    "description": "A child that says hello and bye",
    "initial_state": "hello",
    "states": {
        "hello": _state(
            "hello",
            response_instructions="Kid hello",
            transitions=[{"target_state": "bye", "description": "go"}],
        ),
        "bye": _state("bye", response_instructions="Kid bye"),
    },
}


def _llm() -> ScriptedLLM:
    return ScriptedLLM(
        [],
        extraction_data={"name": "Ann"},
        responses={"chat": "root reply", "hello": "kid hello", "bye": "kid bye"},
    )


def _folded_view(
    folded: StackProjection,
) -> tuple[int, list[tuple[str, dict[str, Any]]]]:
    states = [folded.root, *(f.state for f in folded.frames)]
    return folded.depth, [(s.current_state, s.context_data) for s in states]


class _Watched(EventLog):
    """A log that notes what the live API shows right after each state-carrying append."""

    api: API | None = None
    seen: list[tuple[int, tuple[int, list[tuple[str, dict[str, Any]]]]]]

    def __init__(self, root: Any) -> None:
        super().__init__(root)
        self.seen = []

    def append(self, conversation_id: str, type: Any, **kw: Any) -> int:  # type: ignore[override]
        seq = super().append(conversation_id, type, **kw)
        if self.api is not None and EventType.from_string(type) in _FOLD_TYPES:
            self.seen.append((seq, _view(self.api, conversation_id)))
        return seq


def _scenario(api: API, push_options: dict[str, Any] | None = None) -> str:
    """Start, a root turn, a push (with options), a child turn and delta, a pop with a merge,
    a root turn, a second push left open."""
    cid, _ = api.start_conversation()
    assert api.converse("Ann", cid) == "root reply"
    api.push_fsm(
        cid,
        copy.deepcopy(_KID),
        context_to_pass={"topic": "x"},
        **(push_options or {}),
    )
    assert api.converse("hi", cid) == "kid bye"
    api.update_context(cid, {"note": "n"})
    api.pop_fsm(cid, context_to_return={"result": 1})
    assert api.converse("again", cid) == "root reply"
    api.push_fsm(cid, copy.deepcopy(_KID))
    return cid


def _of(log: EventLog, cid: str, *types: EventType) -> list[dict[str, Any]]:
    return [e for e in log.read(cid) if EventType(e["type"]) in types]


class TestTheLiveStackRecords:
    def test_the_push_carries_what_a_restore_needs(self, log: EventLog):
        api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
        cid = _scenario(
            api,
            {
                "return_context": {"answer": 42, "password": "hunter2"},
                "shared_context_keys": ["name"],
                "preserve_history": True,
            },
        )
        first, second = _of(log, cid, EventType.STACK_PUSH)
        state = log.get_json(first["state_ref"])
        assert (
            state["current_state"] == "hello" and state["context_data"]["topic"] == "x"
        )
        assert first["shared_context_keys"] == ["name"]
        assert first["preserve_history"] is True
        returned = log.get_json(first["return_context_ref"])
        assert returned["answer"] == 42 and "hunter2" not in str(returned)
        assert second["shared_context_keys"] == [] and not second["preserve_history"]
        assert log.get_json(second["return_context_ref"]) == {}
        assert log.get_json(second["definition_ref"])["name"] == "Kid"

    def test_the_pop_carries_the_parents_state_after_the_merge(self, log: EventLog):
        api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
        cid, _ = api.start_conversation()
        api.converse("Ann", cid)
        api.push_fsm(cid, copy.deepcopy(_KID))
        api.pop_fsm(cid, context_to_return={"result": 1})
        (pop,) = _of(log, cid, EventType.STACK_POP)
        blob = log.get_json(pop["state_ref"])
        assert blob["context_data"] == api.get_data(cid)
        assert blob["context_data"]["result"] == 1
        # the merge reaches the log through this record only, until the parent's next turn
        folded = fold_stack(log.query(cid, until_seq=pop["seq"]), log.get_json)
        assert folded is not None and folded.depth == 1
        assert folded.root.context_data["result"] == 1
        root = fold_session(log.query(cid, until_seq=pop["seq"]), log.get_json)
        assert root is not None and "result" not in root.context_data

    def test_the_fold_of_every_state_carrying_prefix_is_what_the_live_api_showed(
        self, tmp_path: Any
    ):
        with _Watched(tmp_path / "events") as log:
            api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
            log.api = api
            cid = _scenario(api, {"shared_context_keys": ["name"]})
            log.api = None
            assert len(log.seen) >= 10
            depths = {view[0] for _, view in log.seen}
            assert depths == {1, 2}
            for seq, observed in log.seen:
                folded = fold_stack(log.query(cid, until_seq=seq), log.get_json)
                assert folded is not None
                assert _folded_view(folded) == observed, f"after event {seq}"

    def test_fold_session_still_reads_the_root_frame_only(self, log: EventLog):
        api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
        cid = _scenario(api)
        root = fold_session(log.read(cid), log.get_json)
        folded = fold_stack(log.read(cid), log.get_json)
        assert root is not None and folded is not None and folded.depth == 2
        assert root.current_state == "chat"
        assert folded.frames[0].state.current_state == "hello"
        assert root.context_data == folded.root.context_data

    def test_a_push_whose_state_cannot_be_read_stands_and_the_childs_turn_supplies_it(
        self, log: EventLog
    ):
        api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
        cid, _ = api.start_conversation()
        with (
            patch(
                "fsm_llm.engine.api.events.state_ref_later", side_effect=OSError("x")
            ),
            patch("fsm_llm.engine.api.events.logger") as warned,
        ):
            api.push_fsm(cid, copy.deepcopy(_KID))
        assert api.get_stack_depth(cid) == 2 and warned.warning.called
        (push,) = _of(log, cid, EventType.STACK_PUSH)
        assert "state_ref" not in push and "definition_ref" in push
        with pytest.raises(LogCorruptError, match="no recorded state"):
            fold_stack(log.read(cid), log.get_json)
        api.converse("hi", cid)
        folded = fold_stack(log.read(cid), log.get_json)
        assert folded is not None and folded.frames[0].state.current_state == "bye"

    def test_a_pop_whose_state_cannot_be_read_leaves_the_pre_merge_parent(
        self, log: EventLog
    ):
        api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
        cid, _ = api.start_conversation()
        api.push_fsm(cid, copy.deepcopy(_KID))
        with patch("fsm_llm.engine.api.events.state_ref_of", side_effect=OSError("x")):
            api.pop_fsm(cid, context_to_return={"result": 1})
        (pop,) = _of(log, cid, EventType.STACK_POP)
        assert "state_ref" not in pop
        folded = fold_stack(log.read(cid), log.get_json)
        assert folded is not None and folded.depth == 1
        assert "result" not in folded.root.context_data

    def test_a_pop_inside_a_deeper_stack_records_the_parent_frame(self, log: EventLog):
        api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
        cid, _ = api.start_conversation()
        api.push_fsm(cid, copy.deepcopy(_KID))
        api.push_fsm(cid, copy.deepcopy(_KID))
        api.update_context(cid, {"deep": 1})
        api.pop_fsm(cid, context_to_return={"merged": 2})
        (pop,) = _of(log, cid, EventType.STACK_POP)
        assert (pop["depth"], pop["frame"]) == (2, 1)
        parent = log.get_json(pop["state_ref"])  # the child frame, not the root
        assert parent["context_data"]["merged"] == 2
        assert fold_stack(log.read(cid), log.get_json).depth == 2  # type: ignore[union-attr]


class TestTheLogOffAndTheLoggedStateRead:
    def test_with_no_log_the_stack_makes_no_extra_call(self):
        api = API(copy.deepcopy(_ROOT), llm_interface=_llm())
        with (
            patch("fsm_llm.engine.api.events.state_ref_of") as read,
            patch("fsm_llm.engine.api.events.state_ref_later") as read_later,
            patch("fsm_llm.engine.api.events.filter_context") as filtered,
        ):
            _scenario(api)
        assert not read.called and not read_later.called and not filtered.called

    def test_record_state_writes_a_pushed_frame_under_the_root_id(self, log: EventLog):
        api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
        cid, _ = api.start_conversation()
        api.push_fsm(cid, copy.deepcopy(_KID))
        child = api.get_sub_conversation_id(cid)
        seq = record_state(
            api.fsm_manager, log, child, source="probe", frame=1, root_id=cid
        )
        event = next(e for e in log.read(cid) if e["seq"] == seq)
        assert (event["conversation_id"], event["frame"]) == (cid, 1)
        assert log.get_json(event["state_ref"])["current_state"] == "hello"
        assert child not in log.conversation_ids()
        # the default stays the one id of a root conversation
        seq = record_state(api.fsm_manager, log, cid, source="probe")
        assert next(e for e in log.read(cid) if e["seq"] == seq)["frame"] == 0


async def _ascenario(api: API) -> str:
    """The async twin of ``_scenario`` (same calls, the push and the turns awaited)."""
    cid, _ = await api.astart_conversation()
    assert await api.aconverse("Ann", cid) == "root reply"
    await api.apush_fsm(cid, copy.deepcopy(_KID), context_to_pass={"topic": "x"})
    assert await api.aconverse("hi", cid) == "kid bye"
    api.update_context(cid, {"note": "n"})
    api.pop_fsm(cid, context_to_return={"result": 1})
    assert await api.aconverse("again", cid) == "root reply"
    await api.apush_fsm(cid, copy.deepcopy(_KID))
    return cid


def _shape(log: EventLog, cid: str) -> list[tuple[Any, ...]]:
    """What a log says without the bytes that differ between two runs (ids, timestamps)."""
    return [
        (e["type"], e["frame"], e.get("depth"), e.get("source")) for e in log.read(cid)
    ]


class TestAnAsyncPush:
    """Plan 77e5fef9 step 6 (D-008): in async mode ``stack/push`` is one job queued behind the
    tree's lane jobs and an inline write of the tree waits for it. After a drain the log holds
    what the sync run holds, in the same order, and it folds to the stack the API had. Every row
    asserts the log AFTER ``release``/``close`` (the drains) and the NEXT call of the caller."""

    async def test_after_a_drain_the_async_run_logs_what_the_sync_run_logs(
        self, tmp_path
    ):
        """Kills (M29, M30): a push written on the loop is in the log at once too, so the equality
        is the order and the content; the fold is compared with what the live API shows."""
        sync_log, async_log = EventLog(tmp_path / "s"), EventLog(tmp_path / "a")
        sync_api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=sync_log)
        sync_cid = _scenario(sync_api)
        async_api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=async_log)
        async_cid = await _ascenario(async_api)
        live = _view(async_api, async_cid)
        sync_log.release(sync_cid)
        async_log.release(async_cid)
        assert _shape(async_log, async_cid) == _shape(sync_log, sync_cid)
        folded = fold_stack(async_log.read(async_cid), async_log.get_json)
        assert folded is not None and _folded_view(folded) == live
        want = fold_stack(sync_log.read(sync_cid), sync_log.get_json)
        assert want is not None and _folded_view(folded) == _folded_view(want)
        for log in (sync_log, async_log):
            log.close()

    async def test_a_pop_right_after_the_push_cannot_precede_it(self, log: EventLog):
        """The sequence the guard exists for: no awaited lane write between the push and the pop.
        Without the wait the pop is written FIRST and the fold reads a phantom child frame (3 of
        3 on the probe). Kills (M31): an ``append`` that does not wait for the deferred push."""
        for n in range(10):
            api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
            cid, _ = await api.astart_conversation()
            await api.aconverse("Ann", cid)
            await api.apush_fsm(cid, copy.deepcopy(_KID))
            api.pop_fsm(cid, context_to_return={"result": 1})
            live = _view(api, cid)
            log.release(cid)
            kinds = [e["type"] for e in log.read(cid) if e["type"].startswith("stack/")]
            assert kinds == ["stack/push", "stack/pop"], (n, kinds)
            folded = fold_stack(log.read(cid), log.get_json)
            assert folded is not None and folded.depth == 1
            assert _folded_view(folded) == live, n

    async def test_an_update_right_after_the_push_cannot_precede_it(
        self, log: EventLog
    ):
        """``update_context`` writes at the CHILD frame, inline, and its write raises on failure
        (it cannot be deferred): before the push the fold raised ``LogCorruptError`` (3 of 3 on
        the probe). Kills (M31): an ``append`` that does not wait."""
        for n in range(10):
            api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
            cid, _ = await api.astart_conversation()
            await api.apush_fsm(cid, copy.deepcopy(_KID))
            api.update_context(cid, {"note": "n"})
            live = _view(api, cid)
            log.release(cid)
            frames = [
                (e["type"], e["frame"])
                for e in log.read(cid)
                if e["type"] in ("stack/push", "context/record") and e["frame"] >= 0
            ]
            assert frames[-2:] == [("stack/push", 0), ("context/record", 1)], (
                n,
                frames,
            )
            folded = fold_stack(log.read(cid), log.get_json)
            assert folded is not None and _folded_view(folded) == live, n

    async def test_the_childs_greeting_comes_before_the_push_as_before(
        self, log: EventLog
    ):
        """The greeting of the child is an awaited lane job written before the push line; FIFO
        keeps it there. Kills (M30 order): a push queued ahead of the greeting, or a push that
        is written before the child exists."""
        api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
        cid, _ = await api.astart_conversation()
        await api.apush_fsm(cid, copy.deepcopy(_KID))
        log.release(cid)
        events = list(log.read(cid))
        (push,) = [e for e in events if e["type"] == "stack/push"]
        greeting = [e for e in events if e["frame"] == 1 and e["type"] == "llm/request"]
        assert greeting and all(e["seq"] < push["seq"] for e in greeting)
        assert [e["type"] for e in events if e["seq"] > push["seq"]] == []

    async def test_the_blobs_are_read_on_the_loop_before_the_job_runs(
        self, log: EventLog
    ):
        """The job is held back (captured, run after the caller changed the child and its
        context): the push blobs are the state, the return context and the definition AT the
        push. Kills (M30): bodies read inside the job (the state blob would hold ``note``)."""
        api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
        cid, _ = await api.astart_conversation()
        jobs: list[Any] = []
        with patch.object(
            Lanes, "after", lambda self, root_id, fn: jobs.append(fn) or None
        ):
            await api.apush_fsm(
                cid,
                copy.deepcopy(_KID),
                return_context={"answer": 1},
                inherit_context=False,
            )
            api.update_context(cid, {"note": "later"})
            assert len(jobs) == 1  # the push job, queued and not yet run
            jobs[0]()
        log.release(cid)
        (push,) = _of(log, cid, EventType.STACK_PUSH)
        assert "note" not in log.get_json(push["state_ref"])["context_data"]
        assert log.get_json(push["return_context_ref"]) == {"answer": 1}
        assert log.get_json(push["definition_ref"])["name"] == "Kid"

    async def test_a_sync_push_writes_inline_even_with_a_started_lane(
        self, log: EventLog
    ):
        """``SYNC`` (a ``push_fsm`` from a worker thread) never queues: the lane is parked, so a
        queued job could not have written the line. Kills (M29 sync): a deferral that ignores
        ``SYNC``."""
        api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
        cid, _ = await api.astart_conversation()  # a lane thread now serves the tree
        entered, gate = threading.Event(), threading.Event()

        def hold() -> None:
            entered.set()
            gate.wait(20.0)

        parked = asyncio.ensure_future(log.lane_call(cid, hold))
        try:
            assert await asyncio.to_thread(entered.wait, 20.0), "the lane never parked"
            await asyncio.to_thread(api.push_fsm, cid, copy.deepcopy(_KID))
            assert len(_of(log, cid, EventType.STACK_PUSH)) == 1
        finally:
            gate.set()
            await parked

    @pytest.mark.parametrize("how", ["end_conversation", "log_close"])
    async def test_the_next_call_sees_the_push_the_drain_wrote(self, tmp_path, how):
        """Push, then end (or close), then resume in the same or a new process: the frame is
        there, 30 times (a ``release`` that does not queue behind the job lost it 30 of 30 for
        the stream reply). Kills (M28): ``release`` or ``close`` that does not wait for the job."""
        for n in range(30):
            log = EventLog(tmp_path / f"{how}{n}")
            api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
            cid, _ = await api.astart_conversation()
            await api.apush_fsm(cid, copy.deepcopy(_KID))
            if how == "end_conversation":
                api.end_conversation(cid)
                again, again_api = log, api
            else:
                log.close()
                again = EventLog(tmp_path / f"{how}{n}")
                again_api = API(
                    copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=again
                )
            again_api.resume(cid)
            assert again_api.get_stack_depth(cid) == 2, n
            assert again_api.get_current_state(cid) == "hello", n
            again.close()

    @pytest.mark.parametrize("after", ["pop", "update"])
    async def test_past_the_bound_the_inline_write_is_refused_and_the_log_still_folds(
        self, log: EventLog, after: str
    ):
        """D-009 (review probes p7_giveup, p13_lostpop): the lane is parked just ahead of the
        deferred push and the wait bound is tiny. A write that went on past the bound landed
        BEFORE the push: a ``stack/pop`` made a phantom child frame, an ``update_context`` record
        made the fold raise ``LogCorruptError`` for good. Now the write is refused: the pop is
        dropped (its WARNING), ``update_context`` raises ``LogWriteError``, the log folds, and
        once the lane drains the conversation is usable (a parent turn heals the dropped pop).
        Kills: a guard that goes on after the bound."""
        api = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
        cid, _ = await api.astart_conversation()
        gate = threading.Event()
        real_after = log.lane_after

        def parked_after(root: str, fn: Any, **kw: Any) -> None:
            log._lanes._submit(
                root, True, gate.wait, 20.0
            )  # parked just ahead of the job
            real_after(root, fn, **kw)

        try:
            with patch.object(log, "lane_after", parked_after):
                await api.apush_fsm(cid, copy.deepcopy(_KID))
            with patch.object(lanes_module, "LANE_WAIT_SECONDS", 0.2):
                if after == "pop":
                    api.pop_fsm(cid, context_to_return={"result": 1})
                else:
                    with pytest.raises(LogWriteError):
                        api.update_context(cid, {"note": "n"})
            live = _view(api, cid)
            assert not [e for e in log.read(cid) if e["type"] == "stack/push"]
            kinds = [(e["type"], e["frame"]) for e in log.read(cid)]
            assert ("stack/pop", 0) not in kinds and ("context/record", 1) not in kinds
        finally:
            gate.set()
        if after == "pop":
            await api.aconverse(
                "again", cid
            )  # the next parent turn heals the dropped pop
        log.release(cid)  # the drain: the push job runs behind the park
        folded = fold_stack(log.read(cid), log.get_json)  # never LogCorruptError
        assert folded is not None and folded.depth == live[0]
        api2 = API(copy.deepcopy(_ROOT), llm_interface=_llm(), event_log=log)
        cid2, _ = await api2.astart_conversation()
        await api2.apush_fsm(cid2, copy.deepcopy(_KID))
        api2.update_context(cid2, {"note": "n"})  # the tree is usable again
        api2.pop_fsm(cid2, context_to_return={"result": 1})
        log.release(cid2)
        again = fold_stack(log.read(cid2), log.get_json)
        assert again is not None and again.depth == 1
        assert live[0] == (2 if after == "update" else 1)
