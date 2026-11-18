"""Session projections of the event log (plan 21e6f00d step 9, roadmap row 1.2): the versioned
``SessionProjection`` and its one adjacent migration, the reader of old ``FileSessionStore`` files,
the fold of a log, ``EventLogSessionStore`` and the restore seeding (the D-009 allow-list and the
lineage record).

Three layers: the migration and the legacy reader on files, the fold on hand-built logs (so a
rolled-back turn that lies about its state, a pushed child and a future event type are exact),
and real conversations (``ScriptedLLM``) written by the engine, stopped between turns or
inside one, and restored. Bytes, counts and sequences, never timing. Unverified until the
Windows run: nothing here opens a file itself except the byte-level tampering helpers, which use
binary mode.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from fsm_llm import API, FileSessionStore
from fsm_llm.core.definitions import FSMError, LLMResponseError
from fsm_llm.engine.session import (
    METADATA_COMPLETION_CONTEXT,
    EventLogSessionStore,
    SessionState,
)
from fsm_llm.runtime import (
    EventLog,
    EventLogError,
    EventPayloadError,
    EventType,
    LogCorruptError,
    LogWriteError,
)
from fsm_llm.runtime.stores import (
    SessionProjection,
    SessionVersionError,
    fold_session,
    jsonl,
    migrate_session,
    projection_from_legacy_file,
)
from fsm_llm.runtime.stores import session as session_module
from fsm_llm.testing import ScriptedLLM, final_reply, tool_call_reply
from tests.test_fsm_llm.support import _state
from tests.test_fsm_llm.test_completion_state import (
    _MESSAGES,
    _ScriptedLLM,
    _start,
    _tool_fsm,
    _tool_runner,
)

_GRANT = {"tool": "add", "arguments": {"a": 2, "b": 3}}
_CHAT: dict[str, Any] = {
    "name": "Sessions",
    "description": "Two speaking states",
    "initial_state": "intake",
    "states": {
        "intake": _state(
            "intake",
            required_context_keys=["name"],
            response_instructions="Greet",
            transitions=[
                {
                    "target_state": "talk",
                    "description": "named",
                    "conditions": [
                        {
                            "description": "name known",
                            "requires_context_keys": ["name"],
                            "logic": {"!=": [{"var": "name"}, None]},
                        }
                    ],
                }
            ],
        ),
        "talk": _state(
            "talk",
            response_instructions="Talk",
            transitions=[
                {
                    "target_state": "end",
                    "description": "never taken here",
                    "conditions": [{"description": "never", "logic": {"==": [1, 2]}}],
                }
            ],
        ),
        "end": _state("end"),
    },
}


def _script() -> ScriptedLLM:
    return ScriptedLLM(
        [],
        extraction_data={"name": "Ann"},
        responses={"intake": "hi Ann", "talk": "talking"},
    )


def _reopen(log: EventLog) -> EventLog:
    """Simulate the process dying between turns: release every writer, open the directory anew."""
    root = log._logs.parent
    log.close()
    return EventLog(root)


def _blob(**over: Any) -> dict[str, Any]:
    blob: dict[str, Any] = {
        "fsm_id": "fsm-1",
        "current_state": "a",
        "context_data": {},
        "conversation_history": [],
        "conversation_summary": None,
        "working_memory": None,
        "hidden_buffers": [],
        "provenance": {},
        "internal": {},
    }
    blob.update(over)
    return blob


class _Script:
    """Append hand-built events of one root conversation ``c1``."""

    def __init__(self, log: EventLog) -> None:
        self.log = log

    def state(
        self, kind: EventType, frame: int = 0, outcome: str | None = None, **over: Any
    ) -> int:
        fields: dict[str, Any] = {"state_ref": self.log.put_json(_blob(**over))}
        if outcome:
            fields["outcome"] = outcome
        return self.log.append("c1", kind, frame=frame, **fields)

    def delta(
        self, context: dict[str, Any], internal: dict[str, Any] | None = None
    ) -> int:
        ref = self.log.put_json({"context": context, "internal": internal or {}})
        return self.log.append(
            "c1",
            EventType.CONTEXT_RECORD,
            source="update_context",
            keys=sorted(context),
            delta_ref=ref,
        )

    def fold(self) -> SessionProjection | None:
        return fold_session(self.log.read("c1"), self.log.get_json)


_START, _END, _RECORD = (
    EventType.TURN_START,
    EventType.TURN_END,
    EventType.CONTEXT_RECORD,
)


# --------------------------------------------------------------
# The migration and the legacy reader
# --------------------------------------------------------------


def _saved_session(directory: Path, session_id: str = "Old_1") -> SessionState:
    state = SessionState(
        conversation_id=session_id,
        fsm_id="fsm-old",
        current_state="talk",
        context_data={"name": "Ann", "nested": {"k": [1, 2]}},
        conversation_history=[{"user": "hi"}, {"system": "hello"}],
        stack_depth=3,
        working_memory={"buffers": {"core": {"a": 1}}, "hidden_buffers": ["metadata"]},
        conversation_summary="earlier",
        metadata={"pipeline_extracted": {"name": "abc"}, "other": 1},
    )
    FileSessionStore(directory).save(session_id, state)
    return state


class TestTheMigration:
    @pytest.mark.parametrize("version", [None, 2])
    def test_internal_keys_of_context_data_are_dropped_whatever_the_version(
        self, version: int | None
    ) -> None:
        planted = {
            "keep": 1,
            "_approval_granted": _GRANT,
            "system_x": 1,
            "INTERNAL_y": 2,
        }
        payload: dict[str, Any] = {
            "conversation_id": "c",
            "fsm_id": "f",
            "current_state": "s",
            "context_data": {**planted, "nested": {"_k": 1}},
        }
        if version:
            payload["v"] = version
        assert migrate_session(payload).context_data == {"keep": 1, "nested": {"_k": 1}}

    def test_a_v1_file_of_the_real_store_equals_what_the_store_loads(
        self, tmp_path: Path, log: EventLog
    ) -> None:
        _saved_session(tmp_path)
        loaded = FileSessionStore(tmp_path).load("Old_1")
        assert loaded is not None
        projection = projection_from_legacy_file(tmp_path / "Old_1.json")
        via_store = EventLogSessionStore(log, legacy_dir=tmp_path).load("Old_1")
        assert via_store is not None
        for state in (via_store,):
            for name in (
                "conversation_id",
                "fsm_id",
                "current_state",
                "context_data",
                "conversation_history",
                "working_memory",
                "conversation_summary",
            ):
                assert getattr(state, name) == getattr(loaded, name), name
            assert state.metadata == {"pipeline_extracted": {"name": "abc"}}
        assert projection.v == 2
        assert projection.provenance == {"name": "abc"}
        assert (projection.completion_context, projection.last_seq) == ({}, 0)

    @pytest.mark.parametrize("version", [3, 99])
    def test_a_newer_version_is_refused(self, version: int) -> None:
        with pytest.raises(SessionVersionError, match="understands up to 2"):
            migrate_session({"v": version, "conversation_id": "x"})

    @pytest.mark.parametrize("version", [True, "2", 2.0, None, 0, -1])
    def test_a_bad_version_is_corrupt_not_a_crash(self, version: Any) -> None:
        with pytest.raises(LogCorruptError):
            migrate_session({"v": version})

    def test_a_v2_payload_is_validated_as_it_is_and_forbids_extras(self) -> None:
        good = {"v": 2, "conversation_id": "c", "fsm_id": "f", "current_state": "s"}
        assert migrate_session(good).last_seq == 0
        with pytest.raises(LogCorruptError, match="stack_depth"):
            migrate_session({**good, "stack_depth": 1})

    def test_a_malformed_v1_names_fields_and_never_echoes_a_value(self) -> None:
        with pytest.raises(LogCorruptError) as caught:
            migrate_session({"conversation_id": "SECRET-VALUE", "context_data": 5})
        assert "fsm_id" in str(caught.value) and "SECRET-VALUE" not in str(caught.value)

    def test_the_depth_bound_applies_to_a_payload(self) -> None:
        deep: Any = "x"
        for _ in range(40):
            deep = [deep]
        with pytest.raises(LogCorruptError, match="deeper"):
            migrate_session({"conversation_id": "c", "context_data": {"k": deep}})

    def test_a_non_mapping_payload_is_corrupt(self) -> None:
        with pytest.raises(LogCorruptError):
            migrate_session([1, 2])  # type: ignore[arg-type]


class TestTheLegacyReader:
    @pytest.mark.parametrize(
        "content",
        [b"not json", b'["a"]', b"\xff\xfe", b"[" * 30_000, b"{" * 30_000],
        ids=["text", "list", "not-utf8", "deep-list", "deep-object"],
    )
    def test_a_hostile_file_is_corrupt_never_a_crash(
        self, tmp_path: Path, content: bytes
    ) -> None:
        (tmp_path / "bad.json").write_bytes(content)
        with pytest.raises(LogCorruptError):
            projection_from_legacy_file(tmp_path / "bad.json")

    def test_an_oversized_file_is_refused_unread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _saved_session(tmp_path)
        monkeypatch.setattr(session_module, "MAX_SESSION_FILE_BYTES", 100)
        with pytest.raises(LogCorruptError, match="larger than"):
            projection_from_legacy_file(tmp_path / "Old_1.json")

    def test_a_missing_file_is_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            projection_from_legacy_file(tmp_path / "none.json")


# --------------------------------------------------------------
# The fold on hand-built logs
# --------------------------------------------------------------


class TestTheFold:
    def test_an_empty_log_and_a_log_without_state_fold_to_nothing(
        self, log: EventLog
    ) -> None:
        assert _Script(log).fold() is None
        log.append("c1", EventType.TOOL_RESULT, call_id="x", status="ok")
        assert _Script(log).fold() is None

    @pytest.mark.parametrize(
        ("steps", "winner"),
        [
            ([(_START, "a", None), (_END, "b", "committed")], "b"),
            ([(_START, "a", None), (_END, "b", "committed"), (_START, "b", None)], "b"),
            # a rolled-back turn is ignored even when its end lies about the state
            ([(_START, "a", None), (_END, "zzz", "rolled_back")], "a"),
            (
                [
                    (_START, "a", None),
                    (_END, "b", "committed"),
                    (_START, "b", None),
                    (_END, "zzz", "rolled_back"),
                ],
                "b",
            ),
            ([(_START, "a", None), (_END, "b", "abandoned")], "b"),
            ([(_START, "a", None), (_END, "b", "interrupted")], "b"),
            # a crash: no turn/end, the pre-turn state of the open turn stands
            ([(_START, "a", None), (_END, "b", "committed"), (_START, "b", None)], "b"),
            ([(_START, "a", None)], "a"),
            # a record carrying a state wins over what came before
            ([(_START, "a", None), (_RECORD, "r", "ignored")], "r"),
        ],
        ids=[
            "committed",
            "open-after-commit",
            "rolled-back-lies",
            "rolled-back-after-commit",
            "abandoned",
            "interrupted",
            "crash-open-turn",
            "first-turn-open",
            "record-state",
        ],
    )
    def test_the_last_state_of_a_committed_turn_wins(
        self, log: EventLog, steps: list[tuple[EventType, str, str | None]], winner: str
    ) -> None:
        script = _Script(log)
        for kind, state, outcome in steps:
            script.state(kind, outcome=outcome, current_state=state)
        projection = script.fold()
        assert projection is not None and projection.current_state == winner

    def test_only_the_root_frame_is_read(self, log: EventLog) -> None:
        script = _Script(log)
        script.state(_START, current_state="root")
        script.state(_START, frame=1, current_state="child")
        script.state(_END, frame=1, outcome="committed", current_state="child2")
        script.state(_RECORD, frame=2, current_state="grandchild")
        projection = script.fold()
        assert projection is not None and projection.current_state == "root"

    def test_an_unknown_future_event_is_skipped(self, log: EventLog) -> None:
        script = _Script(log)
        script.state(_START, current_state="a")
        events = [
            *log.read("c1"),
            {
                "v": 1,
                "seq": 2,
                "ts": "t",
                "conversation_id": "c1",
                "frame": 0,
                "type": "future/kind",
                "state_ref": "sha256:" + "0" * 64,
            },
        ]
        projection = fold_session(iter(events), log.get_json)
        assert projection is not None and (
            projection.current_state,
            projection.last_seq,
        ) == ("a", 2)

    def test_deltas_apply_in_order_and_none_is_a_value_as_in_the_write(
        self, log: EventLog
    ) -> None:
        script = _Script(log)
        script.state(
            _START,
            context_data={"keep": 1, "drop": 2},
            internal={"_completion_messages": ["x"]},
        )
        script.delta(
            {"new": 3, "drop": None}, {"_completion_messages": ["y"], "other": None}
        )
        script.delta({"new": 4})
        projection = script.fold()
        assert projection is not None
        # API.update_context keeps a key written as None (FSMContext.update), so the fold does
        assert projection.context_data == {"keep": 1, "drop": None, "new": 4}
        assert projection.completion_context == {
            "_completion_messages": ["y"],
            "other": None,
        }

    def test_a_turn_start_after_a_delta_carries_it_and_a_rollback_loses_the_delta(
        self, log: EventLog
    ) -> None:
        script = _Script(log)
        script.state(_START, context_data={"a": 1})
        script.state(_END, outcome="committed", context_data={"a": 1})
        script.delta({"a": 2})
        script.state(_START, context_data={"a": 2})
        script.state(_END, outcome="rolled_back", context_data={"a": 99})
        projection = script.fold()
        assert projection is not None and projection.context_data == {"a": 2}

    def test_an_internal_key_in_a_blob_context_is_dropped_here(
        self, log: EventLog
    ) -> None:
        script = _Script(log)
        script.state(_START, context_data={"ok": 1, "_approval_granted": _GRANT})
        script.delta({"_approval_granted": _GRANT, "also": 2})
        projection = script.fold()
        assert projection is not None
        assert projection.context_data == {"ok": 1, "also": 2}

    def test_unresolved_tool_calls_of_any_frame_become_unknown(
        self, log: EventLog
    ) -> None:
        script = _Script(log)
        script.state(_START, current_state="a")
        log.append("c1", EventType.TOOL_CALL, call_id="t1", name="send", arguments={})
        log.append(
            "c1", EventType.TOOL_CALL, frame=1, call_id="t2", name="pay", arguments={}
        )
        log.append("c1", EventType.TOOL_RESULT, call_id="t1", status="ok")
        projection = script.fold()
        assert projection is not None
        assert [
            (c["call_id"], c["status"], c["frame"])
            for c in projection.unknown_tool_calls
        ] == [("t2", "unknown", 1)]

    def test_the_working_memory_keeps_the_session_state_shape(
        self, log: EventLog
    ) -> None:
        script = _Script(log)
        script.state(
            _START, working_memory={"core": {"a": 1}}, hidden_buffers=["metadata"]
        )
        projection = script.fold()
        assert projection is not None
        assert projection.working_memory == {
            "buffers": {"core": {"a": 1}},
            "hidden_buffers": ["metadata"],
        }

    @pytest.mark.parametrize(
        "bad",
        [
            {"conversation_history": [{"user": 5}]},
            {"conversation_history": "no"},
            {"context_data": ["no"]},
            {"internal": 3},
            {"current_state": None},
        ],
        ids=[
            "history-value",
            "history-type",
            "context-type",
            "internal-type",
            "state-none",
        ],
    )
    def test_a_blob_of_the_wrong_shape_is_corrupt(
        self, log: EventLog, bad: dict[str, Any]
    ) -> None:
        script = _Script(log)
        script.state(_START, **bad)
        with pytest.raises(LogCorruptError):
            script.fold()

    def test_a_blob_that_is_not_an_object_and_a_missing_blob_are_corrupt(
        self, log: EventLog
    ) -> None:
        log.append("c1", _START, state_ref=log.put_json(["x"]))
        with pytest.raises(LogCorruptError, match="not a JSON object"):
            _Script(log).fold()
        other = EventLog(log._logs.parent.parent / "other")
        try:
            other.append("c1", _START, state_ref="sha256:" + "a" * 64)
            with pytest.raises(LogCorruptError):
                fold_session(other.read("c1"), other.get_json)
        finally:
            other.close()

    def test_a_delta_blob_that_is_not_an_object_is_corrupt(self, log: EventLog) -> None:
        script = _Script(log)
        script.state(_START)
        log.append(
            "c1", _RECORD, source="update_context", keys=[], delta_ref=log.put_json([1])
        )
        with pytest.raises(LogCorruptError, match="delta"):
            script.fold()

    def test_a_torn_tail_changes_nothing_and_a_corrupt_middle_line_raises(
        self, log: EventLog
    ) -> None:
        script = _Script(log)
        script.state(_START, current_state="a")
        script.state(_END, outcome="committed", current_state="b")
        path = log._logs / "c1.jsonl"
        good = path.read_bytes()
        log.release("c1")
        with open(path, "ab") as handle:
            handle.write(b'{"v":1,"seq":3,"ts":"t","conversation_id":"c1","fra')
        projection = _Script(log).fold()
        assert projection is not None and projection.current_state == "b"
        first, rest = good.split(b"\n", 1)
        path.write_bytes(first + b"\n" + b"not json\n" + rest)
        with pytest.raises(LogCorruptError):
            _Script(log).fold()


# --------------------------------------------------------------
# Real conversations
# --------------------------------------------------------------


def _chat(log: EventLog | None, **kwargs: Any) -> tuple[API, str]:
    api = API(copy.deepcopy(_CHAT), llm_interface=_script(), event_log=log, **kwargs)
    cid, _ = api.start_conversation()
    return api, cid


def _resume(log: EventLog, **kwargs: Any) -> API:
    return API(
        copy.deepcopy(_CHAT),
        llm_interface=_script(),
        session_store=EventLogSessionStore(log),
        **kwargs,
    )


def _view(api: API, cid: str) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    snapshot = api.fsm_manager.get_conversation_snapshot(cid)
    return (
        snapshot["current_state"],
        api.get_data(cid),
        snapshot["conversation_history"],
    )


class TestARealConversation:
    def test_the_fold_of_a_multi_turn_conversation_equals_the_live_state(
        self, log: EventLog
    ) -> None:
        api, cid = _chat(log)
        assert api.converse("I am Ann", cid) == "talking"
        assert api.converse("more", cid) == "talking"
        projection = fold_session(log.read(cid), log.get_json)
        assert projection is not None
        state, data, history = _view(api, cid)
        assert (projection.current_state, projection.context_data) == (state, data)
        assert projection.conversation_history == history
        assert projection.conversation_id == cid
        assert projection.last_seq == max(e["seq"] for e in log.read(cid))

    def test_a_restore_after_a_stop_between_turns_gives_the_same_conversation(
        self, log: EventLog
    ) -> None:
        api, cid = _chat(log)
        api.converse("I am Ann", cid)
        before = _view(api, cid)
        last = max(e["seq"] for e in log.read(cid))
        reopened = _reopen(log)
        revived = _resume(reopened, event_log=reopened)
        new_id, state = revived.restore_session(cid)  # type: ignore[misc]
        assert new_id != cid
        assert _view(revived, new_id) == before
        assert state.metadata["last_seq"] == last
        first = next(iter(reopened.read(new_id)))
        assert (first["seq"], first["type"], first["source"]) == (
            1,
            "context/record",
            "restore",
        )
        assert first["restored_from"] == {"conversation_id": cid, "seq": last}
        assert revived.converse("next", new_id) == "talking"
        reopened.close()

    def test_a_turn_that_never_ended_restores_to_the_pre_turn_state(
        self, log: EventLog
    ) -> None:
        api, cid = _chat(log)
        api.converse("I am Ann", cid)
        api.converse("more", cid)
        before = _view(api, cid)
        api.converse("third", cid)
        path = log._logs / f"{cid}.jsonl"
        lines = path.read_bytes().split(b"\n")
        starts = [i for i, ln in enumerate(lines) if b'"type":"turn/start"' in ln]
        reopened = _reopen(log)
        # the process died right after the last turn/start: cut the file there
        path.write_bytes(b"\n".join(lines[: starts[-1] + 1]) + b"\n")
        revived = _resume(reopened)
        new_id, _ = revived.restore_session(cid)  # type: ignore[misc]
        assert _view(revived, new_id) == before
        reopened.close()

    def test_a_rolled_back_turn_restores_to_the_state_before_it(
        self, log: EventLog
    ) -> None:
        class _Down(ScriptedLLM):
            down = False

            def generate_response(self, request: Any) -> Any:
                if self.down:
                    raise RuntimeError("provider down")
                return super().generate_response(request)

        llm = _Down(
            [], extraction_data={"name": "Ann"}, responses={"intake": "hi", "talk": "t"}
        )
        api = API(copy.deepcopy(_CHAT), llm_interface=llm, event_log=log)
        cid, _ = api.start_conversation()
        before = _view(api, cid)
        llm.down = True
        with pytest.raises(FSMError):
            api.converse("I am Ann", cid)
        assert _view(api, cid) == before
        outcomes = [e["outcome"] for e in log.read(cid) if e["type"] == "turn/end"]
        assert outcomes == ["rolled_back"]
        revived = _resume(_reopen(log))
        new_id, _ = revived.restore_session(cid)  # type: ignore[misc]
        assert _view(revived, new_id) == before

    def test_a_restore_with_the_log_off_writes_no_log(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        api, cid = _chat(log)
        api.converse("I am Ann", cid)
        reopened = _reopen(log)
        revived = _resume(reopened)  # log OFF on the resuming API
        new_id, _ = revived.restore_session(cid)  # type: ignore[misc]
        assert reopened.conversation_ids() == [cid]
        assert revived.get_current_state(new_id) == "talk"
        reopened.close()

    def test_a_snapshot_restore_with_the_log_on_records_its_lineage(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        store = FileSessionStore(tmp_path / "sessions")
        api, cid = _chat(None, session_store=store)
        api.converse("I am Ann", cid)
        api.save_session(cid)
        revived = API(
            copy.deepcopy(_CHAT),
            llm_interface=_script(),
            session_store=store,
            event_log=log,
        )
        new_id, _ = revived.restore_session(cid)  # type: ignore[misc]
        first = next(iter(log.read(new_id)))
        assert (first["source"], first["restored_from"]) == (
            "restore",
            {"conversation_id": cid},
        )
        projection = fold_session(log.read(new_id), log.get_json)
        assert projection is not None and projection.current_state == "talk"

    def test_a_failing_restore_record_ends_the_half_restored_conversation(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api, cid = _chat(log)
        api.converse("I am Ann", cid)
        reopened = _reopen(log)
        revived = _resume(reopened, event_log=reopened)

        def boom(fd: int, data: bytes) -> int:
            raise OSError("disk full")

        monkeypatch.setattr(jsonl, "_write", boom)
        with pytest.raises(LogWriteError):
            revived.restore_session(cid)
        assert revived.list_active_conversations() == []
        reopened.close()

    def test_the_log_store_works_as_the_auto_save_store_of_a_live_api(
        self, log: EventLog
    ) -> None:
        api = API(
            copy.deepcopy(_CHAT),
            llm_interface=_script(),
            event_log=log,
            session_store=EventLogSessionStore(log),
        )
        cid, _ = api.start_conversation()
        assert api.converse("I am Ann", cid) == "talking"
        api.save_session(cid)
        assert log.conversation_ids() == [cid]

    def test_a_corrupt_middle_line_stops_the_restore_and_leaks_nothing(
        self, log: EventLog
    ) -> None:
        api, cid = _chat(log)
        api.converse("I am Ann", cid)
        path = log._logs / f"{cid}.jsonl"
        data = path.read_bytes().split(b"\n")
        reopened = _reopen(log)
        data[1] = b"{broken"
        path.write_bytes(b"\n".join(data))
        revived = _resume(reopened)
        with pytest.raises(LogCorruptError):
            revived.restore_session(cid)
        assert revived.list_active_conversations() == []
        reopened.close()

    def test_a_turn_with_the_log_on_and_off_gives_identical_replies_and_snapshots(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        runs: list[Any] = []
        for index, opened in enumerate((None, log)):
            store = FileSessionStore(tmp_path / f"s{index}")
            api, cid = _chat(opened, session_store=store)
            replies = [api.converse("I am Ann", cid), api.converse("more", cid)]
            api.save_session(cid)
            saved = json.loads(
                (tmp_path / f"s{index}" / f"{cid}.json").read_text("utf-8")
            )
            for volatile in ("saved_at", "conversation_id"):
                saved.pop(volatile)
            runs.append((replies, _view(api, cid), saved))
        assert runs[0] == runs[1]


# --------------------------------------------------------------
# The store itself
# --------------------------------------------------------------


class TestTheStore:
    def test_it_is_read_only_and_append_only(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        store = EventLogSessionStore(log)
        assert (
            store.save(
                "c1", SessionState(conversation_id="c1", fsm_id="f", current_state="s")
            )
            is None
        )
        assert log.conversation_ids() == []
        with pytest.raises(EventLogError, match="append-only"):
            store.delete("c1")
        assert store.load("nope") is None and not store.exists("nope")

    def test_exists_is_membership_and_never_decodes_the_log(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        """C5: ``exists`` was the inherited ``load(...) is not None``, so it folded
        the whole log and RAISED for a corrupt log or a shape-level log, though
        ``list_sessions`` names both ids."""
        store = EventLogSessionStore(log)
        _Script(log).state(_START)
        assert store.exists("c1") and not store.exists("nope")
        path = log._logs / "c1.jsonl"
        path.write_bytes(b"{broken\n" + path.read_bytes())
        with pytest.raises(LogCorruptError):
            store.load("c1")
        assert store.exists("c1") is True
        with EventLog(tmp_path / "shape", privacy="shape") as shape:
            shape.append("s1", _START)
            sstore = EventLogSessionStore(shape)
            with pytest.raises(FSMError):
                sstore.load("s1")
            assert sstore.exists("s1") is True
            assert sstore.exists("s1") == ("s1" in sstore.list_sessions())
        (log._logs / "empty1.jsonl").write_bytes(b"")
        assert store.exists("empty1") == ("empty1" in store.list_sessions())

    def test_exists_follows_the_id_rule_and_the_legacy_dir(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        with pytest.raises(EventPayloadError):
            EventLogSessionStore(log).exists("Upper")
        _saved_session(tmp_path)
        legacy = EventLogSessionStore(log, legacy_dir=tmp_path)
        assert legacy.exists("Old_1") is True and legacy.exists("Missing") is False

    def test_list_sessions_is_the_log_ids_and_the_legacy_ids_sorted(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        _Script(log).state(_START)
        log.append("b2", _START, state_ref=log.put_json(_blob()))
        _saved_session(tmp_path / "old", "Legacy")
        assert EventLogSessionStore(log).list_sessions() == ["b2", "c1"]
        both = EventLogSessionStore(log, legacy_dir=tmp_path / "old")
        assert both.list_sessions() == ["Legacy", "b2", "c1"]

    def test_an_id_outside_the_log_rule_raises_unless_a_legacy_dir_can_hold_it(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        with pytest.raises(EventPayloadError):
            EventLogSessionStore(log).load("Upper")
        _saved_session(tmp_path)
        legacy = EventLogSessionStore(log, legacy_dir=tmp_path)
        assert legacy.load("Old_1") is not None and legacy.load("Missing") is None

    def test_the_log_wins_over_a_legacy_file_of_the_same_id(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        _saved_session(tmp_path, "c1")
        _Script(log).state(_START, current_state="from-log")
        state = EventLogSessionStore(log, legacy_dir=tmp_path).load("c1")
        assert state is not None and state.current_state == "from-log"

    def test_a_legacy_snapshot_restores_through_the_log_store(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        store = FileSessionStore(tmp_path / "sessions")
        api, cid = _chat(None, session_store=store)
        api.converse("I am Ann", cid)
        api.save_session(cid)
        before = _view(api, cid)
        revived = API(
            copy.deepcopy(_CHAT),
            llm_interface=_script(),
            session_store=EventLogSessionStore(log, legacy_dir=tmp_path / "sessions"),
        )
        new_id, _ = revived.restore_session(cid)  # type: ignore[misc]
        assert _view(revived, new_id) == before


# --------------------------------------------------------------
# A completion state: the transcript, the unknown tool, the forged grant
# --------------------------------------------------------------


def _tool_world(log: EventLog) -> tuple[API, str]:
    """A tool conversation advanced to ``call_model`` with one exchange in the transcript."""
    llm = _ScriptedLLM(tool_call_reply(("add", {"a": 2, "b": 3})), final_reply("5"))
    api = API.from_definition(_tool_fsm(), llm_interface=llm, event_log=log)
    api.register_handler(_tool_runner())
    cid = _start(api)
    for _ in range(3):  # intake -> call_model -> run_tools -> call_model
        api.advance(cid)
    assert api.get_current_state(cid) == "call_model"
    return api, cid


def _tool_resume(store: Any, llm: _ScriptedLLM) -> API:
    api = API.from_definition(_tool_fsm(), llm_interface=llm, session_store=store)
    api.register_handler(_tool_runner())
    return api


class TestACompletionState:
    def test_a_restore_from_the_log_mid_exchange_continues_the_transcript(
        self, log: EventLog
    ) -> None:
        api, cid = _tool_world(log)
        transcript = copy.deepcopy(
            api.fsm_manager.copy_raw_context_data(cid)[_MESSAGES]
        )
        reopened = _reopen(log)
        llm = _ScriptedLLM(final_reply("5"))
        revived = _tool_resume(EventLogSessionStore(reopened), llm)
        new_id, _ = revived.restore_session(cid)  # type: ignore[misc]
        assert (
            revived.fsm_manager.copy_raw_context_data(new_id)[_MESSAGES] == transcript
        )
        step = revived.advance(new_id)
        assert step.state_after == "done"
        assert llm.requests[0].messages[1:] == transcript  # exactly what was stored
        assert len(llm.requests) == 1
        reopened.close()

    def test_the_same_restore_from_a_snapshot_still_refuses(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        store = FileSessionStore(tmp_path / "snap")
        llm = _ScriptedLLM(tool_call_reply(("add", {"a": 2, "b": 3})), final_reply("x"))
        api = API.from_definition(_tool_fsm(), llm_interface=llm, session_store=store)
        api.register_handler(_tool_runner())
        cid = _start(api)
        for _ in range(3):
            api.advance(cid)
        api.save_session(cid)
        sent = _ScriptedLLM(final_reply("never sent"))
        revived = _tool_resume(store, sent)
        new_id, _ = revived.restore_session(cid)  # type: ignore[misc]
        with pytest.raises(LLMResponseError, match="no user message"):
            revived.advance(new_id)
        assert sent.requests == []

    def test_a_tool_in_flight_is_unknown_and_never_retried(self, log: EventLog) -> None:
        _, cid = _tool_world(log)
        with pytest.raises(KeyboardInterrupt):
            with log.tool_call(cid, name="add", call_id="c9", arguments={"a": 1}):
                raise KeyboardInterrupt
        reopened = _reopen(log)
        ran: list[str] = []
        llm = _ScriptedLLM(final_reply("5"))
        revived = API.from_definition(
            _tool_fsm(), llm_interface=llm, session_store=EventLogSessionStore(reopened)
        )
        revived.register_handler(_tool_runner(ran))
        _, state = revived.restore_session(cid)  # type: ignore[misc]
        unknown = state.metadata["unknown_tool_calls"]
        assert [(c["call_id"], c["status"]) for c in unknown] == [("c9", "unknown")]
        assert ran == [] and llm.requests == []
        reopened.close()

    @pytest.mark.parametrize(
        "forge",
        ["blob", "delta", "snapshot"],
    )
    def test_a_forged_grant_or_undeclared_internal_key_is_refused_on_restore(
        self, log: EventLog, tmp_path: Path, forge: str
    ) -> None:
        planted = {
            _MESSAGES: [{"role": "user", "content": "task"}],
            "_approval_granted": _GRANT,
            "_run_output_keys": ["x"],
            "_counter": 7,
        }
        script = _Script(log)
        if forge == "blob":
            script.state(_START, current_state="call_model", internal=planted)
            store: Any = EventLogSessionStore(log)
        elif forge == "delta":
            script.state(_START, current_state="call_model")
            script.delta({}, planted)
            store = EventLogSessionStore(log)
        else:
            store = FileSessionStore(tmp_path / "snap")
            store.save(
                "c1",
                SessionState(
                    conversation_id="c1",
                    fsm_id="f",
                    current_state="call_model",
                    metadata={METADATA_COMPLETION_CONTEXT: planted},
                ),
            )
        revived = _tool_resume(store, _ScriptedLLM(final_reply("x")))
        new_id, _ = revived.restore_session("c1")  # type: ignore[misc]
        data = revived.fsm_manager.copy_raw_context_data(new_id)
        assert data[_MESSAGES] == [{"role": "user", "content": "task"}]
        assert not {"_approval_granted", "_run_output_keys", "_counter"} & set(data)

    def test_a_forged_grant_in_a_legacy_file_is_not_restored(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        planted = {"keep": "v", "_approval_granted": _GRANT, "_counter": 7}
        FileSessionStore(tmp_path / "snap").save(
            "c1",
            SessionState(
                conversation_id="c1",
                fsm_id="f",
                current_state="call_model",
                context_data=planted,
            ),
        )
        legacy = tmp_path / "snap" / "c1.json"
        assert projection_from_legacy_file(legacy).context_data == {"keep": "v"}
        store = EventLogSessionStore(log, legacy_dir=tmp_path / "snap")
        revived = _tool_resume(store, _ScriptedLLM(final_reply("x")))
        new_id, _ = revived.restore_session("c1")  # type: ignore[misc]
        data = revived.fsm_manager.copy_raw_context_data(new_id)
        assert data["keep"] == "v" and not {"_approval_granted", "_counter"} & set(data)

    def test_the_log_carries_only_declared_keys_to_begin_with(
        self, log: EventLog
    ) -> None:
        api, cid = _tool_world(log)
        api.update_context(cid, {"_approval_granted": _GRANT})
        projection = fold_session(log.read(cid), log.get_json)
        assert projection is not None
        assert set(projection.completion_context) == {_MESSAGES}
        assert projection.context_data == api.get_data(cid)
