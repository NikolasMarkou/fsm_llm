"""The EventLog of the runtime layer (plan 21e6f00d step 5): append, read, query, checkpoints.

Bytes, counts and orderings, never timing: ``ts`` comes from an injected clock and is never
asserted on for order (``seq`` is the order); fsyncs are counted through the module-level
``_write`` and ``_fsync`` of ``runtime.stores.jsonl``. Unverified until the Windows run: the
behaviour of the files under a real Windows file system (the lock and ``O_BINARY`` are
``stores.jsonl``'s, proven there).
"""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

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
from fsm_llm.runtime.log import EventLog, filter_context
from fsm_llm.runtime.stores import jsonl
from tests.conftest import EveryReadLater
from tests.test_fsm_llm.test_event_log_format import _deep_list

SECRET_KEY = "sk-live-0123456789abcdefghijklmnopqrstuv"
SECRET_PASSWORD = "hunter2-correct-horse-battery"
SECRET_TOKEN = "tok_live_9876543210zyxwvutsrqponmlkj"


class _Dead(BaseException):
    """A stand-in for a process that vanished: not an ``Exception``, not an ``OSError``."""


@pytest.fixture
def log(tmp_path: Path):
    """An EventLog under ``tmp_path``, closed at teardown (an open handle blocks Windows)."""
    opened = EventLog(tmp_path)
    yield opened
    opened.close()


def _types(events: Any) -> list[str]:
    return [event["type"] for event in events]


def _all_bytes(root: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file())


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
        self, log: EventLog, tmp_path: Path
    ) -> None:
        seqs = [log.append("c1", "turn/start")]
        seqs += log.append_batch(
            "c1", [{"type": "executor/start", "kind": "a"}, {"type": "transition"}]
        )
        seqs.append(log.append("c1", EventType.TURN_END))
        seqs.append(log.append("c2", "turn/start"))
        assert seqs == [1, 2, 3, 4, 1]
        assert (tmp_path / "logs" / "c1.jsonl").is_file()
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
        tmp_path: Path,
        kind: Any,
        frame: Any,
        fields: dict[str, Any],
        error: type[Exception],
    ) -> None:
        with pytest.raises(error):
            log.append("c1", kind, frame=frame, **fields)
        with pytest.raises(error):
            log.append_batch("c1", [{"type": kind, "frame": frame, **fields}])
        assert not (tmp_path / "logs").exists()

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

    def test_a_second_log_on_one_directory_refuses_the_second_writer(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        log.append("c1", "turn/start")
        with EventLog(tmp_path) as other:
            with pytest.raises(LogLockedError):
                other.append("c1", "turn/start")
            assert other.append("c2", "turn/start") == 1  # another tree is free

    def test_close_releases_every_lock_so_another_log_can_take_the_id(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        log.append("c1", "turn/start")
        log.append("c2", "turn/start")
        log.close()
        with EventLog(tmp_path) as other:
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
        self, log: EventLog, tmp_path: Path
    ) -> None:
        for name in ("b2", "a1"):
            log.append(name, "turn/start")
        (tmp_path / "logs" / "Foreign.jsonl").write_bytes(b"")
        (tmp_path / "logs" / "notes.txt").write_bytes(b"")
        assert log.conversation_ids() == ["a1", "b2"]

    def test_a_damaged_middle_line_raises_and_the_reader_repairs_nothing(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        log.append("c1", "turn/start")
        log.append("c1", "turn/end")
        path = tmp_path / "logs" / "c1.jsonl"
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
        self, log: EventLog, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="private-detail"):
            with log.tool_call("c1", name="w", call_id="k1", arguments={}):
                raise ValueError("private-detail")
        result = list(log.read("c1"))[-1]
        assert (result["status"], result["error_type"]) == ("failed", "ValueError")
        assert b"private-detail" not in _all_bytes(tmp_path)
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
        self, log: EventLog, tmp_path: Path
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
        data = _all_bytes(tmp_path)
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
        self, log: EventLog, tmp_path: Path, bad: dict[str, Any]
    ) -> None:
        with pytest.raises(EventPayloadError):
            with log.tool_call("c1", **bad):
                pytest.fail("the body must not run")
        assert not (tmp_path / "logs").exists()

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
        self, log: EventLog, tmp_path: Path
    ) -> None:
        safe = filter_context(self.CONTEXT)
        log.append("c1", "context/record", delta=safe)
        log.put_json(safe)
        data = _all_bytes(tmp_path)
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
        self, log: EventLog, tmp_path: Path
    ) -> None:
        ref = log.put_blob(b"a\r\nb\x00")
        assert log.get_blob(ref) == b"a\r\nb\x00"
        jref = log.put_json({"b": 1, "a": [1, 2]})
        assert log.get_json(jref) == {"a": [1, 2], "b": 1}
        assert log.put_json({"a": [1, 2], "b": 1}) == jref
        assert (tmp_path / "blobs").is_dir()

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
