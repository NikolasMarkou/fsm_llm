"""The EventLog of the runtime layer (plan 21e6f00d step 5): append, read, query, checkpoints.

Bytes, counts and orderings, never timing: ``ts`` comes from an injected clock and is never
asserted on for order (``seq`` is the order); fsyncs are counted through the module-level
``_write`` and ``_fsync`` of ``runtime.stores.jsonl``. Unverified until the Windows run: the
behaviour of the files under a real Windows file system (the lock and ``O_BINARY`` are
``stores.jsonl``'s, proven there).
"""

from __future__ import annotations

import ast
import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from fsm_llm.runtime import log as log_module
from fsm_llm.runtime.checkpoint import is_checkpoint, unresolved_tool_calls
from fsm_llm.runtime.constants import MAX_INLINE_BYTES
from fsm_llm.runtime.events import (
    EventPayloadError,
    EventType,
    LogCorruptError,
    LogLockedError,
    LogWriteError,
    check_depth,
)
from fsm_llm.runtime.log import EventLog, filter_context, warn_on_failure
from fsm_llm.runtime.stores import jsonl
from tests.conftest import EveryReadLater
from tests.test_fsm_llm.event_log_support import (
    _Dead,
    tree_bytes,
    tree_files,
)
from tests.test_fsm_llm.test_event_log_format import _deep_list

SECRET_KEY = "sk-live-0123456789abcdefghijklmnopqrstuv"
SECRET_PASSWORD = "hunter2-correct-horse-battery"
SECRET_TOKEN = "tok_live_9876543210zyxwvutsrqponmlkj"


def _types(events: Any) -> list[str]:
    return [event["type"] for event in events]


@pytest.fixture
def trace(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record every write (with its event type) and every fsync of the JSONL writer."""
    steps: list[tuple[str, str]] = []
    real_write = jsonl._write

    def write(fd: int, data: Any) -> int:
        for line in bytes(data).split(b"\n")[:-1]:
            steps.append(("write", json.loads(line)["type"]))
        return real_write(fd, data)

    def fsync(fd: int) -> None:
        steps.append(("fsync", ""))

    monkeypatch.setattr(jsonl, "_write", write)
    monkeypatch.setattr(jsonl, "_fsync", fsync)
    return steps


class TestAppend:
    def test_nothing_is_created_until_something_is_written(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "root"
        with EventLog(root) as fresh:
            assert fresh.conversation_ids() == []
            assert list(fresh.read("c1")) == []
        assert not root.exists()

    def test_seq_is_one_based_and_contiguous_across_append_and_batch(
        self, log: EventLog
    ) -> None:
        seqs = [log.append("c1", "turn/start")]
        seqs += log.append_batch(
            "c1", [{"type": "executor/start", "kind": "a"}, {"type": "transition"}]
        )
        seqs.append(log.append("c1", EventType.TURN_END))
        seqs.append(log.append("c2", "turn/start"))
        assert seqs == [1, 2, 3, 4, 1]
        assert (log.root / "logs" / "c1.jsonl").is_file()
        events = list(log.read("c1"))
        assert [e["seq"] for e in events] == [1, 2, 3, 4]
        assert events[1]["kind"] == "a"

    def test_the_envelope_carries_the_roadmap_keys_and_the_fields(
        self, log: EventLog
    ) -> None:
        log.append("c1", "llm/request", frame=2, model="m", stream=False)
        (event,) = list(log.read("c1"))
        assert event["v"] == 1 and event["frame"] == 2
        assert event["conversation_id"] == "c1" and event["type"] == "llm/request"
        assert event["model"] == "m" and event["stream"] is False
        assert len(event["ts"]) == 24 and event["ts"].endswith("Z")

    def test_a_non_json_leaf_becomes_a_typed_marker_not_a_repr(
        self, log: EventLog
    ) -> None:
        log.append("c1", "transition", odd={1, 2}, blob=b"abc")
        (event,) = list(log.read("c1"))
        assert event["odd"] == "<set>" and event["blob"] == "<bytes:3>"

    @pytest.mark.parametrize(
        ("kind", "frame", "fields", "error"),
        [
            ("llm_request", 0, {}, ValueError),
            (None, 0, {}, ValueError),
            ("turn/start", -1, {}, EventPayloadError),
            ("turn/start", True, {}, EventPayloadError),
            ("turn/start", "0", {}, EventPayloadError),
            ("turn/start", 0, {"ts": "x"}, EventPayloadError),
            ("turn/start", 0, {"seq": 9}, EventPayloadError),
            ("turn/start", 0, {"v": 2}, EventPayloadError),
        ],
    )
    def test_a_bad_event_is_refused_before_any_file_exists(
        self,
        log: EventLog,
        kind: Any,
        frame: Any,
        fields: dict[str, Any],
        error: type[Exception],
    ) -> None:
        with pytest.raises(error):
            log.append("c1", kind, frame=frame, **fields)
        with pytest.raises(error):
            log.append_batch("c1", [{"type": kind, "frame": frame, **fields}])
        assert not (log.root / "logs").exists()

    @pytest.mark.parametrize("bad", ["Abc", "CON", "../x", "a" * 65, ""])
    def test_a_bad_conversation_id_is_refused_for_every_verb(
        self, log: EventLog, bad: str
    ) -> None:
        for call in (
            lambda: log.append(bad, "turn/start"),
            lambda: log.append_batch(bad, []),
            lambda: list(log.read(bad)),
            lambda: list(log.query(bad)),
        ):
            with pytest.raises(EventPayloadError):
                call()

    def test_a_batch_without_a_type_writes_nothing(self, log: EventLog) -> None:
        with pytest.raises(EventPayloadError):
            log.append_batch("c1", [{"type": "turn/start"}, {"kind": "x"}])
        assert list(log.read("c1")) == []
        assert log.append_batch("c1", []) == []

    def test_a_batch_is_one_write_and_a_conversation_is_one_file(
        self, log: EventLog, trace: list[tuple[str, str]]
    ) -> None:
        items = [{"type": "executor/start"}, {"type": "transition", "frame": 1}]
        assert log.append_batch("c1", items) == [1, 2]
        assert trace == [("write", "executor/start"), ("write", "transition")]
        assert [e["frame"] for e in log.read("c1")] == [0, 1]

    def test_a_closed_log_refuses_and_release_lets_an_id_reopen(
        self, log: EventLog
    ) -> None:
        log.append("c1", "turn/start")
        log.release("c1")
        assert (
            log.append("c1", "turn/end") == 2
        )  # reopened: seq continues from the file
        log.close()
        log.close()
        with pytest.raises(LogWriteError):
            log.append("c1", "turn/start")

    def test_a_poisoned_writer_recovers_through_release(self, log: EventLog) -> None:
        log.append("c1", "turn/start")
        with patch.object(jsonl, "_write", side_effect=OSError("disk full")):
            with pytest.raises(LogWriteError):
                log.append("c1", "turn/end")
        with pytest.raises(LogWriteError):  # poisoned, though the disk is back
            log.append("c1", "turn/end")
        log.release("c1")
        assert log.append("c1", "turn/end") == 2

    def test_a_second_log_on_one_directory_refuses_the_second_writer(
        self, log: EventLog
    ) -> None:
        log.append("c1", "turn/start")
        with EventLog(log.root) as other:
            with pytest.raises(LogLockedError):
                other.append("c1", "turn/start")
            assert other.append("c2", "turn/start") == 1  # another tree is free

    def test_close_releases_every_lock_so_another_log_can_take_the_id(
        self, log: EventLog
    ) -> None:
        log.append("c1", "turn/start")
        log.append("c2", "turn/start")
        log.close()
        with EventLog(log.root) as other:
            assert (other.append("c1", "turn/end"), other.append("c2", "turn/end")) == (
                2,
                2,
            )

    def test_threads_opening_one_id_together_share_one_writer(
        self, log: EventLog
    ) -> None:
        barrier = threading.Barrier(8)
        errors: list[BaseException] = []

        def work(thread: int) -> None:
            try:
                barrier.wait()
                for n in range(60):
                    log.append("c1", "transition", thread=thread, n=n)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        assert [e["seq"] for e in log.read("c1")] == list(range(1, 481))
        assert len(log._writers) == 1


class TestClock:
    def test_ts_comes_from_the_module_level_clock(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(log_module, "time", EveryReadLater(start=0.0, step=1.0))
        for _ in range(3):
            log.append("c1", "turn/start")
        stamps = [e["ts"] for e in log.read("c1")]
        assert stamps == [
            "1970-01-01T00:00:01.000Z",
            "1970-01-01T00:00:02.000Z",
            "1970-01-01T00:00:03.000Z",
        ]

    @pytest.mark.parametrize(
        "readings", [[5.0] * 10, [10.0 - n for n in range(10)], [float("nan")] * 10]
    )
    def test_a_frozen_backwards_or_broken_clock_changes_no_order(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch, readings: list[float]
    ) -> None:
        feed = iter(readings)
        monkeypatch.setattr(
            log_module, "time", SimpleNamespace(time=lambda: next(feed))
        )
        for n in range(10):
            log.append("c1", "transition", n=n)
        events = list(log.read("c1"))
        assert [e["n"] for e in events] == list(range(10))
        assert [e["seq"] for e in events] == list(range(1, 11))
        assert [e["n"] for e in log.query("c1", since_seq=3, until_seq=6)] == [3, 4, 5]


class TestReadAndQuery:
    @pytest.fixture
    def filled(self, log: EventLog) -> EventLog:
        plan = [
            ("turn/start", 0),
            ("llm/request", 0),
            ("llm/response", 0),
            ("stack/push", 0),
            ("llm/request", 1),
            ("llm/response", 1),
            ("stack/pop", 0),
            ("turn/end", 0),
        ]
        for n, (kind, frame) in enumerate(plan, start=1):
            log.append("c1", kind, frame=frame, n=n)
        return log

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({}, [1, 2, 3, 4, 5, 6, 7, 8]),
            ({"types": ["llm/request"]}, [2, 5]),
            ({"types": [EventType.LLM_REQUEST, "llm/response"]}, [2, 3, 5, 6]),
            ({"frame": 1}, [5, 6]),
            ({"frame": 1, "types": ["llm/response"]}, [6]),
            ({"since_seq": 5}, [6, 7, 8]),
            ({"until_seq": 3}, [1, 2, 3]),
            ({"since_seq": 2, "until_seq": 5}, [3, 4, 5]),
            ({"types": ["llm/request"], "limit": 1}, [2]),
            ({"limit": 0}, []),
            ({"types": []}, []),
        ],
    )
    def test_query_filters_by_type_frame_and_seq_range(
        self, filled: EventLog, kwargs: dict[str, Any], expected: list[int]
    ) -> None:
        assert [e["seq"] for e in filled.query("c1", **kwargs)] == expected

    def test_read_since_seq_and_an_unknown_type_and_an_unknown_id(
        self, filled: EventLog
    ) -> None:
        assert [e["seq"] for e in filled.read("c1", since_seq=6)] == [7, 8]
        with pytest.raises(ValueError, match="event type"):
            list(filled.query("c1", types=["llm_request"]))
        assert list(filled.read("nobody")) == []

    def test_conversation_ids_are_the_valid_log_names_sorted(
        self, log: EventLog
    ) -> None:
        for name in ("b2", "a1"):
            log.append(name, "turn/start")
        (log.root / "logs" / "Foreign.jsonl").write_bytes(b"")
        (log.root / "logs" / "notes.txt").write_bytes(b"")
        assert log.conversation_ids() == ["a1", "b2"]

    def test_a_damaged_middle_line_raises_and_the_reader_repairs_nothing(
        self, log: EventLog
    ) -> None:
        log.append("c1", "turn/start")
        log.append("c1", "turn/end")
        path = log.root / "logs" / "c1.jsonl"
        lines = path.read_bytes().split(b"\n")
        damaged = b"\n".join([b"{not json", lines[1], b""])
        path.write_bytes(damaged)
        with pytest.raises(LogCorruptError):
            list(log.read("c1"))
        assert path.read_bytes() == damaged


class TestFsyncPolicy:
    def test_only_checkpoint_events_are_fsynced_each_right_after_its_write(
        self, log: EventLog, trace: list[tuple[str, str]]
    ) -> None:
        log.append("c1", "llm/request")
        log.append("c1", "llm/response")
        log.append("c1", "tool/call", name="w", call_id="1", side_effecting=True)
        log.append("c1", "tool/call", name="r", call_id="2", side_effecting=False)
        log.append("c1", "turn/start")
        log.append("c1", "executor/start")
        log.append("c1", "transition")
        log.append("c1", "llm/request")
        log.append("c1", "turn/end")
        fsyncs = [i for i, step in enumerate(trace) if step[0] == "fsync"]
        assert len(fsyncs) == 4
        assert [trace[i - 1] for i in fsyncs] == [
            ("write", "llm/request"),
            ("write", "tool/call"),
            ("write", "llm/request"),
            ("write", "turn/end"),
        ]
        assert [trace[i + 1][0] for i in fsyncs[:-1]] == ["write"] * 3

    def test_an_explicit_checkpoint_flag_overrides_the_rule(
        self, log: EventLog, trace: list[tuple[str, str]]
    ) -> None:
        log.append("c1", "llm/request", checkpoint=False)
        log.append("c1", "transition", checkpoint=True)
        assert trace == [
            ("write", "llm/request"),
            ("write", "transition"),
            ("fsync", ""),
        ]

    def test_a_batch_makes_one_fsync_when_any_event_is_a_checkpoint(
        self, log: EventLog, trace: list[tuple[str, str]]
    ) -> None:
        log.append_batch("c1", [{"type": "transition"}, {"type": "executor/start"}])
        assert [s[0] for s in trace].count("fsync") == 0
        log.append_batch("c1", [{"type": "transition"}, {"type": "turn/end"}])
        assert [s[0] for s in trace].count("fsync") == 1

    @pytest.mark.parametrize(
        ("kind", "fields", "expected"),
        [
            ("llm/request", {}, True),
            ("turn/end", {}, True),
            ("tool/call", {"side_effecting": True}, True),
            ("tool/call", {}, True),  # unknown means side-effecting
            ("tool/call", {"side_effecting": False}, False),
            ("tool/result", {}, False),
            ("llm/response", {}, False),
            ("turn/start", {}, False),
            (EventType.TURN_END, {}, True),
        ],
    )
    def test_is_checkpoint_table(
        self, kind: Any, fields: dict[str, Any], expected: bool
    ) -> None:
        assert is_checkpoint(kind, fields) is expected

    def test_is_checkpoint_refuses_a_type_outside_the_vocabulary(self) -> None:
        with pytest.raises(ValueError):
            is_checkpoint("llm_request", {})


class TestFailClosed:
    def test_a_failing_checkpoint_raises_and_every_later_append_refuses(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(fd: int) -> None:
            raise OSError(5, "disk gone")

        monkeypatch.setattr(jsonl, "_fsync", broken)
        called: list[str] = []
        with pytest.raises(LogWriteError):
            log.append("c1", "llm/request")
            called.append(
                "the model call"
            )  # never reached: this is what fail closed means
        assert called == []
        with pytest.raises(LogWriteError):
            log.append("c1", "transition")  # poisoned: even a non-checkpoint refuses
        assert [e["type"] for e in log.read("c1")] == [
            "llm/request"
        ]  # written, never acknowledged

    def test_a_failing_tool_checkpoint_never_runs_the_tool(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(fd: int, data: Any) -> int:
            raise OSError(28, "no space")

        monkeypatch.setattr(jsonl, "_write", broken)
        ran: list[int] = []
        with pytest.raises(LogWriteError):
            with log.tool_call("c1", name="w", call_id="1", arguments={}):
                ran.append(1)
        assert ran == []


class TestToolCall:
    def test_an_ok_call_writes_call_then_result_and_fsyncs_only_the_call(
        self, log: EventLog, trace: list[tuple[str, str]]
    ) -> None:
        with log.tool_call(
            "c1", name="write_file", call_id="k1", arguments={"path": "a"}
        ):
            trace.append(("body", ""))
        assert trace == [
            ("write", "tool/call"),
            ("fsync", ""),
            ("body", ""),
            ("write", "tool/result"),
        ]
        call, result = list(log.read("c1"))
        assert call["arguments"] == {"path": "a"} and call["side_effecting"] is True
        assert (result["call_id"], result["status"]) == ("k1", "ok")
        assert log.unknown_tool_calls("c1") == []

    def test_a_read_only_call_is_not_fsynced_but_is_still_tracked(
        self, log: EventLog, trace: list[tuple[str, str]]
    ) -> None:
        with pytest.raises(_Dead):
            with log.tool_call(
                "c1", name="read", call_id="k1", arguments={}, side_effecting=False
            ):
                raise _Dead
        assert [s[0] for s in trace].count("fsync") == 0
        (unknown,) = log.unknown_tool_calls("c1")
        assert unknown["call_id"] == "k1" and unknown["side_effecting"] is False

    def test_a_failing_body_records_the_type_name_only_and_reraises(
        self, log: EventLog
    ) -> None:
        with pytest.raises(ValueError, match="private-detail"):
            with log.tool_call("c1", name="w", call_id="k1", arguments={}):
                raise ValueError("private-detail")
        result = list(log.read("c1"))[-1]
        assert (result["status"], result["error_type"]) == ("failed", "ValueError")
        assert b"private-detail" not in tree_bytes(log.root)
        assert log.unknown_tool_calls("c1") == []

    @pytest.mark.parametrize("dying", [KeyboardInterrupt, SystemExit, _Dead])
    def test_a_body_that_never_returns_leaves_the_call_unknown_and_nothing_retries_it(
        self, log: EventLog, dying: type[BaseException]
    ) -> None:
        ran: list[int] = []
        with pytest.raises(dying):
            with log.tool_call(
                "c1", name="deploy", call_id="k9", arguments={"env": "p"}
            ):
                ran.append(1)
                raise dying
        assert ran == [1]
        assert _types(log.read("c1")) == ["tool/call"]
        (unknown,) = log.unknown_tool_calls("c1")
        assert unknown == {
            "status": "unknown",
            "seq": 1,
            "frame": 0,
            "call_id": "k9",
            "name": "deploy",
            "side_effecting": True,
            "arguments": {"env": "p"},
        }

    def test_secret_arguments_are_redacted_and_large_ones_become_a_blob(
        self, log: EventLog
    ) -> None:
        with log.tool_call(
            "c1", name="w", call_id="a", arguments={"api_key": SECRET_KEY, "q": "x"}
        ):
            pass
        big = {"text": "y" * (MAX_INLINE_BYTES + 1), "password": SECRET_PASSWORD}
        with log.tool_call("c1", name="w", call_id="b", arguments=big):
            pass
        first, _, second, _ = list(log.read("c1"))
        assert first["arguments"] == {"api_key": "<redacted>", "q": "x"}
        assert "arguments" not in second
        stored = log.get_json(second["arguments_ref"])
        assert (
            stored["password"] == "<redacted>"
            and len(stored["text"]) > MAX_INLINE_BYTES
        )
        data = tree_bytes(log.root)
        assert SECRET_KEY.encode() not in data and SECRET_PASSWORD.encode() not in data

    @pytest.mark.parametrize(
        "bad",
        [
            {"name": "", "call_id": "1", "arguments": {}},
            {"name": "w", "call_id": "", "arguments": {}},
            {"name": 3, "call_id": "1", "arguments": {}},
            {"name": "w", "call_id": "1", "arguments": "x"},
        ],
    )
    def test_a_malformed_call_is_refused_before_anything_is_written(
        self, log: EventLog, bad: dict[str, Any]
    ) -> None:
        with pytest.raises(EventPayloadError):
            with log.tool_call("c1", **bad):
                pytest.fail("the body must not run")
        assert not (log.root / "logs").exists()

    def test_unresolved_calls_match_one_result_per_call_oldest_first(self) -> None:
        def call(seq: int, call_id: str) -> dict[str, Any]:
            return {"type": "tool/call", "seq": seq, "frame": 0, "call_id": call_id}

        def result(seq: int, call_id: str) -> dict[str, Any]:
            return {"type": "tool/result", "seq": seq, "frame": 0, "call_id": call_id}

        events = [
            call(1, "a"),
            call(2, "a"),  # a reused id: one result resolves the oldest only
            result(3, "a"),
            result(4, "ghost"),  # a result with no call is ignored
            call(5, "b"),
            {"type": "turn/end", "seq": 6, "frame": 0},
        ]
        assert [(r["seq"], r["call_id"]) for r in unresolved_tool_calls(events)] == [
            (2, "a"),
            (5, "b"),
        ]
        assert unresolved_tool_calls([]) == []


class TestAppendHelpers:
    def test_warn_on_failure_swallows_an_exception_and_names_its_type_only(
        self,
    ) -> None:
        with patch.object(log_module, "logger") as logger:
            with warn_on_failure("turn/end"):
                raise RuntimeError("PRIVATE detail")
        text = logger.warning.call_args[0][0]
        assert "turn/end" in text and "RuntimeError" in text and "PRIVATE" not in text
        with pytest.raises(_Dead), warn_on_failure("turn/end"):
            raise _Dead  # an interrupt or a vanished process is never swallowed

    def test_the_id_is_checked_once_per_append_and_per_batch(
        self, log: EventLog
    ) -> None:
        with patch.object(
            log_module, "check_conversation_id", wraps=log_module.check_conversation_id
        ) as check:
            log.append("c1", "turn/start")
            assert check.call_count == 1
            log.append_batch("c1", [{"type": "turn/end"}])
        assert check.call_count == 2
        with pytest.raises(EventPayloadError):
            log.append("Bad Id", "turn/start")


class TestSecretFilter:
    CONTEXT: ClassVar[dict[str, Any]] = {
        "api_key": SECRET_KEY,
        "password": SECRET_PASSWORD,
        "user": {"token": SECRET_TOKEN, "name": "ada"},
        "history": [{"role": "user", "secret": SECRET_PASSWORD}],
        "city": "Paris",
        "_approval_granted": "grant-1",
        "system_note": "n",
        "__dunder": 1,
        "nested": {"_hidden": "h", "ok": 1},
    }

    def test_no_secret_reaches_a_log_or_a_blob_and_public_values_do(
        self, log: EventLog
    ) -> None:
        safe = filter_context(self.CONTEXT)
        log.append("c1", "context/record", delta=safe)
        log.put_json(safe)
        data = tree_bytes(log.root)
        for secret in (SECRET_KEY, SECRET_PASSWORD, SECRET_TOKEN):
            assert secret.encode() not in data
        assert b"Paris" in data and b"<redacted>" in data  # not vacuous: it did write
        assert safe["api_key"] == "<redacted>"
        assert safe["user"] == {"token": "<redacted>", "name": "ada"}

    def test_internal_keys_are_dropped_at_every_depth_unless_allow_listed(self) -> None:
        transcript = [{"role": "user", "content": "hi"}]
        context = {**self.CONTEXT, "_completion_messages": transcript}
        safe = filter_context(context, keep_internal=["_completion_messages"])
        assert safe["_completion_messages"] == transcript
        assert safe["nested"] == {"ok": 1}
        assert {"_approval_granted", "system_note", "__dunder"}.isdisjoint(safe)
        assert "_completion_messages" not in filter_context(context)
        # an allow-list names TOP-LEVEL keys only, and never un-drops a grant by prefix
        assert "_approval_granted" not in filter_context(
            context, keep_internal=["_completion_messages"]
        )

    def test_the_input_is_never_mutated_and_a_non_mapping_is_refused(self) -> None:
        context = copy.deepcopy(self.CONTEXT)
        filter_context(context, keep_internal=["_approval_granted"])
        assert context == self.CONTEXT
        with pytest.raises(TypeError):
            filter_context([("a", 1)])  # type: ignore[arg-type]

    def test_a_transcripts_tool_call_arguments_text_is_redacted_when_it_holds_a_secret(
        self,
    ) -> None:
        def calls(arguments: str) -> list[dict[str, Any]]:
            function = {"name": "login", "arguments": arguments}
            call = {"id": "1", "type": "function", "function": function}
            return [{"role": "assistant", "tool_calls": [call]}]

        secret = json.dumps(
            {"user": "a", "password": SECRET_PASSWORD, "opts": {"api_key": SECRET_KEY}}
        )
        plain, broken = json.dumps({"user": "a", "n": 1}, indent=1), "not json {"
        transcripts = {"_s": calls(secret), "_p": calls(plain), "_b": calls(broken)}
        safe = filter_context(transcripts, keep_internal=list(transcripts))
        text = safe["_s"][0]["tool_calls"][0]["function"]["arguments"]
        redacted = {
            "user": "a",
            "password": "<redacted>",
            "opts": {"api_key": "<redacted>"},
        }
        assert json.loads(text) == redacted
        assert SECRET_PASSWORD not in json.dumps(safe) and SECRET_KEY not in json.dumps(
            safe
        )
        assert safe["_p"] == calls(plain) and safe["_b"] == calls(broken)  # untouched

    def test_a_100000_deep_value_is_cut_on_the_filter_and_on_append(
        self, log: EventLog
    ) -> None:
        deep = _deep_list(100_000)
        safe = filter_context({"v": deep, "ok": 1})
        assert check_depth(safe, 20)
        log.append("c1", "context/record", via_filter=safe, direct=deep)
        (event,) = list(log.read("c1"))
        assert check_depth(event, 32) and event["via_filter"]["ok"] == 1


class TestBlobs:
    def test_bytes_and_json_round_trip_through_the_log_root(
        self, log: EventLog
    ) -> None:
        ref = log.put_blob(b"a\r\nb\x00")
        assert log.get_blob(ref) == b"a\r\nb\x00"
        jref = log.put_json({"b": 1, "a": [1, 2]})
        assert log.get_json(jref) == {"a": [1, 2], "b": 1}
        assert log.put_json({"a": [1, 2], "b": 1}) == jref
        assert (log.root / "blobs").is_dir()

    def test_a_blob_that_is_not_json_or_is_too_deep_is_corruption(
        self, log: EventLog
    ) -> None:
        with pytest.raises(LogCorruptError):
            log.get_json(log.put_blob(b"\xff not json"))
        for depth in (
            40,
            100_000,
        ):  # valid JSON: past the cap, and past any parser's recursion
            deep = b"[" * depth + b"]" * depth
            with pytest.raises(LogCorruptError):
                log.get_json(log.put_blob(deep))


def _reads_every_file(source: str) -> list[int]:
    """Line numbers of comprehensions that read file content over ``rglob("*")``."""

    def calls(node: ast.AST, attrs: set[str]) -> list[ast.Call]:
        return [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in attrs
        ]

    found: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            continue
        every = [
            c
            for gen in node.generators
            for c in calls(gen.iter, {"rglob"})
            if c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "*"
        ]
        if every and calls(node.elt, {"read_bytes", "read_text"}):
            found.append(node.lineno)
    return found


class TestTheTestsNeverReadALockFile:
    """Windows refuses a read of a byte-range-locked file, even in the process that holds the
    lock (CI run 44121497599: nine tests failed on it), so a test reads a log tree through
    ``tree_bytes`` / ``tree_files`` only, which skip the ``*.lock`` files."""

    def test_tree_bytes_skips_a_lock_file_that_cannot_be_read(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "c1.jsonl").write_bytes(b"line\n")
        (tmp_path / "logs" / "c1.lock").write_bytes(b"\x00")
        (tmp_path / "blobs").mkdir()
        (tmp_path / "blobs" / "ab").write_bytes(b"blob")
        real = Path.read_bytes

        def read_bytes(self: Path) -> bytes:
            if self.name.endswith(".lock"):
                raise PermissionError(13, "simulated: a byte range is locked")
            return real(self)

        monkeypatch.setattr(Path, "read_bytes", read_bytes)
        assert tree_bytes(tmp_path) == b"blob" + b"line\n"  # sorted by path
        assert [p.name for p in tree_files(tmp_path)] == ["ab", "c1.jsonl"]
        with pytest.raises(PermissionError):  # the stand-in is a real trap
            (tmp_path / "logs" / "c1.lock").read_bytes()

    def test_an_empty_or_missing_tree_is_empty_bytes(self, tmp_path: Path) -> None:
        assert tree_bytes(tmp_path) == b"" == tree_bytes(tmp_path / "missing")

    def test_no_event_log_test_reads_every_file_of_a_tree_itself(self) -> None:
        folder = Path(__file__).parent
        files = sorted(folder.glob("test_event_log_*.py"))
        assert len(files) >= 10
        bad = {
            f.name: lines
            for f in files
            if (lines := _reads_every_file(f.read_text(encoding="utf-8")))
        }
        assert bad == {}, "use tree_bytes or tree_files from event_log_support"

    def test_the_scan_catches_the_pattern_and_passes_the_helper(self) -> None:
        trap = 'b"".join(p.read_bytes() for p in root.rglob("*") if p.is_file())'
        assert _reads_every_file(trap) == [1]
        assert _reads_every_file("[p.read_text() for p in root.rglob('*')]") == [1]
        assert (
            _reads_every_file('[p.read_bytes() for p in root.rglob("*.jsonl")]') == []
        )
        assert _reads_every_file('[p for p in root.rglob("*") if p.is_file()]') == []
        assert _reads_every_file("tree_bytes(root)") == []


def _nest(depth: int) -> dict[str, Any]:
    value: Any = 1
    for _ in range(depth):
        value = {"a": value}
    return value  # type: ignore[no-any-return]


def _depth_and_leaf(value: Any) -> tuple[int, Any]:
    depth = 0
    while isinstance(value, dict):
        depth, value = depth + 1, value["a"]
    return depth, value


class TestAHookThatDeepensToolArguments:
    """D4 (runtime audit 2.1): a hook output of depth 31 or 32 was cut to ``<truncated:depth>``
    when it fit inline, and kept whole when it was big enough for a blob. The inline form is
    only for what an event field can hold whole; the rest takes the blob path, intact."""

    @staticmethod
    def _call(tmp_path: Path, depth: int) -> tuple[EventLog, dict[str, Any]]:
        log = EventLog(tmp_path / "deep", redact=lambda kind, value: _nest(depth))
        with log.tool_call("c1", name="t", call_id="k1", arguments={"q": 1}):
            pass
        (event,) = [e for e in log.read("c1") if e["type"] == "tool/call"]
        return log, event

    def test_depth_30_stays_inline_and_intact(self, tmp_path: Path) -> None:
        log, event = self._call(tmp_path, 30)
        with log:
            assert _depth_and_leaf(event["arguments"]) == (30, 1)

    @pytest.mark.parametrize("depth", [31, 32])
    def test_depth_31_and_32_are_stored_whole_through_the_blob_path(
        self, tmp_path: Path, depth: int
    ) -> None:
        log, event = self._call(tmp_path, depth)
        with log:
            assert "arguments" not in event
            assert _depth_and_leaf(log.get_json(event["arguments_ref"])) == (depth, 1)

    def test_depth_33_is_refused_and_the_body_never_runs(self, tmp_path: Path) -> None:
        with EventLog(tmp_path / "deep", redact=lambda kind, value: _nest(33)) as log:
            with pytest.raises(EventPayloadError):
                with log.tool_call("c1", name="t", call_id="k1", arguments={"q": 1}):
                    pytest.fail("the body must not run")


class TestAnEmptyBatchCreatesNothing:
    """D4 (runtime audit 1.5): ``nothing is created until something is written``."""

    def test_an_empty_batch_makes_no_file_and_no_writer(self, log: EventLog) -> None:
        assert log.append_batch("c1", []) == []
        assert not log.has_writer("c1")
        assert not (log.root / "logs").exists()
        assert log.conversation_ids() == []

    def test_an_empty_batch_does_not_take_the_lock_of_a_second_log(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        log.append_batch("c1", [])
        with EventLog(
            log.root
        ) as second:  # would be LogLockedError if the first took it
            assert second.append("c1", "turn/start") == 1

    def test_an_empty_batch_still_refuses_a_bad_id_and_checks_it_once(
        self, log: EventLog
    ) -> None:
        with pytest.raises(EventPayloadError):
            log.append_batch("Bad Id", [])
        with patch.object(
            log_module, "check_conversation_id", wraps=log_module.check_conversation_id
        ) as check:
            log.append_batch("c1", [])
        assert check.call_count == 1
