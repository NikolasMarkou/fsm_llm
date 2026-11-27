"""The JSONL store of the event log: unbuffered append, torn tail, lock, bounded reader.

Plan 21e6f00d step 2. Everything here is about bytes on disk: one event is one unbuffered
write ending in a newline, the newline is the only commit marker, a torn last line is cut by
the next lock holder and ignored by every reader, a bad middle line raises. No test reads a
clock or asserts on ``ts``; ``seq`` is the order. The crash tests are deterministic
(injected through the module-level ``_write``, ``_fsync`` and ``_truncate``) except one
child process that is killed with ``Popen.kill()``.

Unverified until the Windows run: ``msvcrt.locking`` semantics on a real Windows file, the
effect of ``O_BINARY`` on a real fd (the sentinel test proves the flag is passed, not what
Windows does with it) and killing a Windows child. Each has a test that runs on every
platform and states what only the Windows leg can prove.
"""

from __future__ import annotations

import ast
import gc
import inspect
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.runtime import EventLog
from fsm_llm.runtime.constants import (
    LOG_FORMAT_VERSION,
    MAX_EVENT_BYTES,
    WINDOWS_RESERVED_NAMES,
)
from fsm_llm.runtime.events import (
    EventPayloadError,
    LogCorruptError,
    LogLockedError,
    LogWriteError,
    encode_event,
)
from fsm_llm.runtime.stores import jsonl
from fsm_llm.runtime.stores.jsonl import (
    JsonlWriter,
    check_conversation_id,
    read_events,
)
from tests.conftest import current_timing_factor, symlink_or_skip
from tests.test_fsm_llm.event_log_support import (
    child_env,
    count_reads,
    reopen_after_kill,
)

JSONL_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "fsm_llm"
    / "runtime"
    / "stores"
    / "jsonl.py"
)


def _event(n: int = 0, **extra: Any) -> dict[str, Any]:
    """The fields a caller gives the writer: every envelope key but ``v`` and ``seq``."""
    base: dict[str, Any] = {
        "ts": "2026-11-17T10:00:00.000Z",
        "conversation_id": "c1",
        "frame": 0,
        "type": "turn/start",
        "n": n,
    }
    base.update(extra)
    return base


def _line(seq: int, **extra: Any) -> bytes:
    """A valid encoded log line, built without the writer (to craft files by hand)."""
    return encode_event({"v": 1, "seq": seq, **_event(seq, **extra)})


@pytest.fixture
def open_writer(tmp_path: Path):
    """``make(name)`` opens a writer under ``tmp_path/logs``; every one is closed at teardown
    (an open handle blocks the removal of its directory on Windows)."""
    writers: list[JsonlWriter] = []

    def make(name: str = "c1", directory: Path | None = None) -> JsonlWriter:
        folder = directory if directory is not None else tmp_path / "logs"
        writer = JsonlWriter(folder / f"{name}.jsonl")
        writers.append(writer)
        return writer

    yield make
    for writer in writers:
        writer.close()


class TestAppend:
    def test_seq_is_one_based_and_contiguous_and_the_file_is_one_line_per_event(
        self, tmp_path: Path, open_writer: Any
    ) -> None:
        writer = open_writer()
        assert writer.last_seq == 0
        assert [writer.append(_event(i)) for i in range(5)] == [1, 2, 3, 4, 5]
        assert writer.last_seq == 5
        raw = writer.path.read_bytes()
        assert raw.endswith(b"\n") and raw.count(b"\n") == 5
        events = list(read_events(writer.path))
        assert [e["seq"] for e in events] == [1, 2, 3, 4, 5]
        assert {e["v"] for e in events} == {LOG_FORMAT_VERSION}

    def test_the_writer_assigns_v_and_seq_and_refuses_a_caller_that_does(
        self, open_writer: Any
    ) -> None:
        writer = open_writer()
        for key in ("seq", "v"):
            with pytest.raises(EventPayloadError, match="assigns"):
                writer.append(_event(**{key: 9}))
        assert writer.last_seq == 0

    def test_text_separators_and_crlf_round_trip_on_one_line_each(
        self, open_writer: Any
    ) -> None:
        text = "a\r\nb c\u0085d\x1ce \ud800 café \U0001f600 €"
        writer = open_writer()
        writer.append(_event(text=text))
        writer.append(_event(text=text))
        raw = writer.path.read_bytes()
        assert raw.count(b"\n") == 2 and b"\r" not in raw  # ensure_ascii escapes them
        assert raw.isascii()
        assert [e["text"] for e in read_events(writer.path)] == [text, text]

    def test_a_refused_payload_writes_nothing_and_does_not_poison(
        self, open_writer: Any
    ) -> None:
        writer = open_writer()
        writer.append(_event(0))
        deep: Any = None
        for _ in range(100_000):
            deep = [deep]
        with pytest.raises(EventPayloadError):
            writer.append(_event(deep=deep))
        with pytest.raises(EventPayloadError, match="blob"):
            writer.append(_event(text="x" * MAX_EVENT_BYTES))
        assert writer.last_seq == 1
        assert writer.append(_event(1)) == 2  # seq not consumed, writer still usable
        assert [e["seq"] for e in read_events(writer.path)] == [1, 2]

    def test_eight_threads_of_200_appends_keep_seq_contiguous_and_lines_whole(
        self, open_writer: Any
    ) -> None:
        writer = open_writer()
        failures: list[BaseException] = []

        def work(thread: int) -> None:
            try:
                for i in range(200):
                    writer.append(_event(i, thread=thread, pad="é" * (i % 7)))
            except BaseException as exc:  # a thread must report, not die silently
                failures.append(exc)

        threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert failures == []
        events = list(read_events(writer.path))
        assert [e["seq"] for e in events] == list(range(1, 1601))
        assert writer.last_seq == 1600
        for thread in range(8):  # per-thread order is kept
            mine = [e["n"] for e in events if e["thread"] == thread]
            assert mine == list(range(200))

    def test_a_one_byte_short_write_still_writes_whole_lines(
        self, monkeypatch: pytest.MonkeyPatch, open_writer: Any
    ) -> None:
        calls: list[int] = []

        def one_byte(fd: int, data: Any) -> int:
            calls.append(len(data))
            return os.write(fd, bytes(data[:1]))

        writer = open_writer()
        monkeypatch.setattr(jsonl, "_write", one_byte)
        writer.append(_event(text="short"))
        assert len(calls) > 50 and calls[0] > calls[-1] == 1  # a real loop, shrinking
        assert [e["text"] for e in read_events(writer.path)] == ["short"]

    def test_a_zero_byte_write_is_an_error_not_an_endless_loop(
        self, monkeypatch: pytest.MonkeyPatch, open_writer: Any
    ) -> None:
        writer = open_writer()
        monkeypatch.setattr(jsonl, "_write", lambda fd, data: 0)
        with pytest.raises(LogWriteError):
            writer.append(_event())

    def test_the_whole_event_is_one_write_when_the_os_takes_it(
        self, monkeypatch: pytest.MonkeyPatch, open_writer: Any
    ) -> None:
        calls: list[int] = []

        def spy(fd: int, data: Any) -> int:
            calls.append(len(data))
            return os.write(fd, data)

        writer = open_writer()
        monkeypatch.setattr(jsonl, "_write", spy)
        writer.append(_event(text="é" * 500))
        assert len(calls) == 1 and calls[0] == len(writer.path.read_bytes())


class TestAppendMany:
    def test_a_batch_is_one_write_with_contiguous_seq_and_one_fsync(
        self, monkeypatch: pytest.MonkeyPatch, open_writer: Any
    ) -> None:
        writer = open_writer()
        writer.append(_event(0))
        writes: list[int] = []
        syncs: list[int] = []
        real = jsonl._write
        monkeypatch.setattr(
            jsonl, "_write", lambda fd, d: (writes.append(1), real(fd, d))[1]
        )
        monkeypatch.setattr(jsonl, "_fsync", lambda fd: syncs.append(fd))
        assert writer.append_many(
            [_event(i) for i in range(1, 4)], checkpoint=True
        ) == [2, 3, 4]
        assert (len(writes), len(syncs), writer.last_seq) == (1, 1, 4)
        assert [e["seq"] for e in read_events(writer.path)] == [1, 2, 3, 4]

    def test_an_empty_batch_writes_nothing_and_a_refused_event_refuses_the_whole_batch(
        self, open_writer: Any
    ) -> None:
        writer = open_writer()
        assert writer.append_many([]) == []
        too_big = _event(1, text="x" * MAX_EVENT_BYTES)
        with pytest.raises(EventPayloadError):
            writer.append_many([_event(0), too_big])
        with pytest.raises(EventPayloadError):
            writer.append_many([_event(0), {**_event(1), "seq": 7}])
        assert writer.path.read_bytes() == b"" and writer.last_seq == 0
        assert writer.append(_event(2)) == 1  # not poisoned

    def test_a_failing_batch_write_poisons_and_a_one_byte_short_write_still_lands_whole(
        self, monkeypatch: pytest.MonkeyPatch, open_writer: Any
    ) -> None:
        writer = open_writer()
        real = jsonl._write
        monkeypatch.setattr(jsonl, "_write", lambda fd, d: real(fd, bytes(d)[:1]))
        assert writer.append_many([_event(0), _event(1)]) == [1, 2]
        monkeypatch.setattr(jsonl, "_write", lambda fd, d: 0)
        with pytest.raises(LogWriteError):
            writer.append_many([_event(2), _event(3)])
        monkeypatch.undo()
        with pytest.raises(LogWriteError, match="poisoned"):
            writer.append_many([_event(4)])
        assert [e["seq"] for e in read_events(writer.path)] == [1, 2]


class TestFailClosed:
    def test_a_failing_write_raises_log_write_error_and_poisons_the_writer(
        self, monkeypatch: pytest.MonkeyPatch, open_writer: Any
    ) -> None:
        writer = open_writer()
        writer.append(_event(0))

        def broken(fd: int, data: Any) -> int:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(jsonl, "_write", broken)
        with pytest.raises(LogWriteError):
            writer.append(_event(1))
        monkeypatch.undo()
        with pytest.raises(LogWriteError, match="poisoned"):
            writer.append(_event(2))  # even though the disk is fine again
        with pytest.raises(LogWriteError, match="poisoned"):
            writer.append(_event(3), checkpoint=True)

    def test_a_failing_fsync_on_a_checkpoint_raises_and_poisons(
        self, monkeypatch: pytest.MonkeyPatch, open_writer: Any
    ) -> None:
        writer = open_writer()

        def broken(fd: int) -> None:
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(jsonl, "_fsync", broken)
        assert writer.append(_event(0)) == 1  # no fsync, no failure
        with pytest.raises(LogWriteError):
            writer.append(_event(1), checkpoint=True)
        with pytest.raises(LogWriteError, match="poisoned"):
            writer.append(_event(2))

    def test_an_interrupt_mid_write_poisons_so_nothing_lands_after_a_partial_line(
        self, monkeypatch: pytest.MonkeyPatch, open_writer: Any
    ) -> None:
        writer = open_writer()
        writer.append(_event(0))

        def interrupted(fd: int, data: Any) -> int:
            os.write(fd, bytes(data[:10]))  # a partial line reached the file
            raise KeyboardInterrupt

        monkeypatch.setattr(jsonl, "_write", interrupted)
        with pytest.raises(KeyboardInterrupt):
            writer.append(_event(1))
        monkeypatch.undo()
        with pytest.raises(LogWriteError, match="poisoned"):
            writer.append(_event(2))
        # the reader ignores the partial tail
        assert next(iter(read_events(writer.path)))["seq"] == 1

    def test_a_closed_writer_refuses(self, open_writer: Any) -> None:
        writer = open_writer()
        writer.close()
        writer.close()  # idempotent
        with pytest.raises(LogWriteError, match="closed"):
            writer.append(_event())


class TestFsyncPolicy:
    def test_only_a_checkpoint_append_is_followed_by_one_fsync_of_its_fd(
        self, monkeypatch: pytest.MonkeyPatch, open_writer: Any
    ) -> None:
        order: list[str] = []
        monkeypatch.setattr(
            jsonl, "_fsync", lambda fd: order.append(f"fsync:{fd == writer._fd}")
        )
        real_write = jsonl._write

        def traced(fd: int, data: Any) -> int:
            order.append("write")
            return real_write(fd, data)

        monkeypatch.setattr(jsonl, "_write", traced)
        writer = open_writer()
        writer.append(_event(0))
        writer.append(_event(1), checkpoint=True)
        writer.append(_event(2))
        writer.append(_event(3), checkpoint=False)
        writer.append(_event(4), checkpoint=True)
        assert order == [
            "write",
            "write",
            "fsync:True",
            "write",
            "write",
            "write",
            "fsync:True",
        ]

    def test_the_directory_is_fsynced_once_when_the_file_is_created_never_after(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        synced: list[Path] = []
        monkeypatch.setattr(jsonl, "fsync_directory", synced.append)
        path = tmp_path / "logs" / "dirsync.jsonl"
        JsonlWriter(path).close()
        JsonlWriter(path).close()  # the file exists now
        assert synced == [path.parent]


class TestOpenFlags:
    def test_every_raw_open_carries_o_binary_when_the_platform_has_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Linux has no O_BINARY: a sentinel bit stands in for it, and is masked off before the
        # real open. What Windows does with the flag on a real fd is unverified until the
        # Windows run; this proves the flag is read at call time and reaches every os.open.
        sentinel = 1 << 29
        monkeypatch.setattr(os, "O_BINARY", sentinel, raising=False)
        real_open = os.open
        seen: dict[str, int] = {}

        def spy(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            seen[Path(path).name] = flags
            return real_open(path, flags & ~sentinel, *args, **kwargs)

        monkeypatch.setattr(os, "open", spy)
        writer = JsonlWriter(tmp_path / "logs" / "bin.jsonl")
        writer.append(_event())
        list(read_events(writer.path))
        writer.close()
        assert seen["bin.jsonl"] & sentinel
        assert seen["bin.lock"] & sentinel

    def test_without_o_binary_the_flag_is_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delattr(os, "O_BINARY", raising=False)
        assert jsonl._binary_flag() == 0

    def test_the_log_is_opened_append_only_create_and_private(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        real_open = os.open
        seen: dict[str, tuple[int, int]] = {}

        def spy(path: Any, flags: int, mode: int = 0o777, **kwargs: Any) -> int:
            seen.setdefault(
                Path(path).name, (flags, mode)
            )  # the writer's open is first
            return real_open(path, flags, mode, **kwargs)

        monkeypatch.setattr(os, "open", spy)
        JsonlWriter(tmp_path / "logs" / "flags.jsonl").close()
        flags, mode = seen["flags.jsonl"]
        assert flags & os.O_APPEND and flags & os.O_CREAT and flags & os.O_WRONLY
        assert mode == 0o600


class TestTornTail:
    def _three(self, tmp_path: Path) -> tuple[bytes, int]:
        writer = JsonlWriter(tmp_path / "src" / "sweep.jsonl")
        for i in range(3):
            writer.append(_event(i, text="café   \U0001f600 \r\n tail"))
        writer.close()
        whole = writer.path.read_bytes()
        return whole, whole.rfind(b"\n", 0, len(whole) - 1) + 1

    def test_every_strict_prefix_of_the_last_line_recovers_byte_identically(
        self, tmp_path: Path
    ) -> None:
        whole, last = self._three(tmp_path)
        tail = whole[last:]
        pre_tear = [e["seq"] for e in read_events(tmp_path / "src" / "sweep.jsonl")][:2]
        assert pre_tear == [1, 2] and len(tail) > 60
        tried = 0
        for k in range(
            len(tail)
        ):  # k == 0 and the whole line minus its newline included
            path = tmp_path / "torn" / f"t{k}.jsonl"
            path.parent.mkdir(exist_ok=True)
            torn = whole[:last] + tail[:k]
            path.write_bytes(torn)
            # a reader sees the committed lines and never repairs
            assert [e["seq"] for e in read_events(path)] == [1, 2], k
            assert path.read_bytes() == torn, k
            writer = JsonlWriter(path)
            try:
                assert path.read_bytes() == whole[:last], k  # the writer cut the tail
                assert writer.last_seq == 2, k
                assert writer.append(_event(2)) == 3, k
            finally:
                writer.close()
            assert [e["seq"] for e in read_events(path)] == [1, 2, 3], k
            tried += 1
        assert tried == len(tail)  # non-vacuity: the sweep ran every prefix

    def test_the_complete_last_line_is_kept(self, tmp_path: Path) -> None:
        whole, _ = self._three(tmp_path)
        path = tmp_path / "whole" / "whole.jsonl"
        path.parent.mkdir()
        path.write_bytes(whole)
        writer = JsonlWriter(path)
        writer.close()
        assert path.read_bytes() == whole and writer.last_seq == 3

    def test_the_tail_is_cut_through_the_truncate_indirection_only_when_torn(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        whole, last = self._three(tmp_path)
        cuts: list[tuple[bool, int]] = []
        real = os.ftruncate

        def spy(fd: int, length: int) -> None:
            cuts.append((fd >= 0, length))
            real(fd, length)

        monkeypatch.setattr(jsonl, "_truncate", spy)
        clean = tmp_path / "x" / "clean.jsonl"
        torn = tmp_path / "x" / "torn.jsonl"
        clean.parent.mkdir()
        clean.write_bytes(whole)
        torn.write_bytes(whole[:-5])
        JsonlWriter(clean).close()
        assert cuts == []
        JsonlWriter(torn).close()
        assert cuts == [(True, last)]

    def test_a_failing_truncate_fails_the_open_and_releases_the_lock(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        whole, _ = self._three(tmp_path)
        path = tmp_path / "x" / "cut.jsonl"
        path.parent.mkdir()
        path.write_bytes(whole[:-5])

        def broken(fd: int, length: int) -> None:
            raise OSError(1, "Operation not permitted")

        monkeypatch.setattr(jsonl, "_truncate", broken)
        with pytest.raises(LogWriteError):
            JsonlWriter(path)
        monkeypatch.undo()
        JsonlWriter(path).close()  # the lock was released by the failed open

    def test_a_reader_never_mutates_a_torn_file(self, tmp_path: Path) -> None:
        whole, last = self._three(tmp_path)
        path = tmp_path / "x" / "ro.jsonl"
        path.parent.mkdir()
        torn = whole[: last + 7]
        path.write_bytes(torn)
        assert len(list(read_events(path))) == 2
        assert path.read_bytes() == torn


class TestHostileInput:
    def _file(self, tmp_path: Path, body: bytes, name: str = "bad") -> Path:
        path = tmp_path / "hostile" / f"{name}.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(body)
        return path

    def test_a_bad_middle_line_raises_for_reader_and_writer(
        self, tmp_path: Path
    ) -> None:
        path = self._file(tmp_path, _line(1) + b"{not json}\n" + _line(3))
        with pytest.raises(LogCorruptError):
            list(read_events(path))
        before = path.read_bytes()
        with pytest.raises(LogCorruptError):
            JsonlWriter(path)
        assert path.read_bytes() == before  # a corrupt log is never "repaired"

    def test_an_empty_line_is_corruption_not_a_skip(self, tmp_path: Path) -> None:
        path = self._file(tmp_path, _line(1) + b"\n" + _line(2))
        with pytest.raises(LogCorruptError):
            list(read_events(path))

    @pytest.mark.parametrize(
        "seqs", [[1, 3], [2], [1, 1], [0, 1], [1, 2, 2], [1, 2, 4]], ids=str
    )
    def test_a_gapped_repeated_or_misnumbered_seq_raises(
        self, tmp_path: Path, seqs: list[int]
    ) -> None:
        path = self._file(tmp_path, b"".join(_line(s) for s in seqs))
        with pytest.raises(LogCorruptError, match="seq"):
            list(read_events(path))
        with pytest.raises(LogCorruptError, match="seq"):
            JsonlWriter(path)

    def test_a_string_or_bool_seq_is_corruption(self, tmp_path: Path) -> None:
        for index, bad in enumerate(("1", True, 1.0)):
            line = encode_event({"v": 1, "seq": bad, **_event()})
            with pytest.raises(LogCorruptError, match="seq"):
                list(read_events(self._file(tmp_path, line, name=f"seq{index}")))

    def test_an_over_cap_line_is_refused_before_json_loads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        big = b'{"v":1,"seq":1,"pad":"' + b"x" * MAX_EVENT_BYTES + b'"}\n'
        path = self._file(tmp_path, big)

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("json.loads ran on an over-cap line")

        monkeypatch.setattr(json, "loads", boom)
        with pytest.raises(LogCorruptError, match="cap"):
            list(read_events(path))
        with pytest.raises(LogCorruptError, match="cap"):
            JsonlWriter(path)

    def test_a_10_mb_newline_free_file_is_refused_by_size_without_reading_it_all(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = self._file(tmp_path, b"x" * (10 * 1024 * 1024))
        read_total = [0]
        real = os.read

        def counting(fd: int, n: int) -> bytes:
            data = real(fd, n)
            read_total[0] += len(data)
            return data

        monkeypatch.setattr(os, "read", counting)
        with pytest.raises(LogCorruptError, match="newline"):
            list(read_events(path))
        assert read_total[0] < 1024 * 1024
        read_total[0] = 0
        with pytest.raises(LogCorruptError, match="newline"):
            JsonlWriter(path)
        assert read_total[0] < 1024 * 1024
        assert (
            path.stat().st_size == 10 * 1024 * 1024
        )  # never truncated: not a torn tail

    def test_a_tail_just_under_the_cap_is_a_torn_tail_not_corruption(
        self, tmp_path: Path
    ) -> None:
        good = _line(1)
        path = self._file(tmp_path, good + b"y" * (MAX_EVENT_BYTES - 1))
        assert len(list(read_events(path))) == 1
        JsonlWriter(path).close()
        assert path.read_bytes() == good

    def test_a_30000_deep_line_within_the_byte_cap_is_refused(
        self, tmp_path: Path
    ) -> None:
        depth = 30_000
        body = (
            b'{"conversation_id":"c","frame":0,"seq":1,"ts":"t","type":"turn/start",'
            b'"v":1,"x":' + b"[" * depth + b"]" * depth + b"}\n"
        )
        assert len(body) < MAX_EVENT_BYTES
        path = self._file(tmp_path, body, name="deep")
        with pytest.raises(
            LogCorruptError
        ):  # RecursionError on 3.11, depth check after
            list(read_events(path))
        with pytest.raises(LogCorruptError):
            JsonlWriter(path)

    def test_a_non_ascii_or_non_object_line_is_corruption(self, tmp_path: Path) -> None:
        for name, body in (("u", "é\n".encode()), ("a", b"[1]\n"), ("n", b"7\n")):
            with pytest.raises(LogCorruptError):
                list(read_events(self._file(tmp_path, body, name=name)))

    def test_a_missing_file_is_a_file_not_found_for_the_reader(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            list(read_events(tmp_path / "nope.jsonl"))


class TestIds:
    @pytest.mark.parametrize(
        "name",
        [
            *("Abc", "ABC", "CON", "con", "nul", "NUL.txt", "com1", "lpt9", "aux.x"),
            "x" * 65,
            "",
            "../x",
            "a/b",
            "a\\b",
            "a.b",
            "a b",
            "é",
            "a\n",
            "c:",
            "x\x00",
        ],
    )
    def test_a_refused_id(self, name: str) -> None:
        with pytest.raises(EventPayloadError):
            check_conversation_id(name)

    @pytest.mark.parametrize(
        "name", ["abc_1-2", "c1", "x" * 64, "0", "a-b_c", "com", "com10", "lpt0"]
    )
    def test_an_accepted_id(self, name: str) -> None:
        assert check_conversation_id(name) == name

    def test_a_uuid4_is_accepted(self) -> None:
        import uuid

        value = str(uuid.uuid4())
        assert check_conversation_id(value) == value

    def test_every_reserved_device_name_is_refused(self) -> None:
        for name in sorted(WINDOWS_RESERVED_NAMES):
            with pytest.raises(EventPayloadError):
                check_conversation_id(name)

    def test_the_writer_refuses_a_bad_name_before_touching_the_disk(
        self, tmp_path: Path
    ) -> None:
        for name in (
            "Abc.jsonl",
            "CON.jsonl",
            "nul.txt.jsonl",
            "plain.json",
            "x.jsonl.bak",
        ):
            with pytest.raises(EventPayloadError):
                JsonlWriter(tmp_path / "logs" / name)
        assert not (tmp_path / "logs").exists()


class TestLock:
    def test_a_second_writer_on_one_log_in_one_process_is_refused(
        self, tmp_path: Path, open_writer: Any
    ) -> None:
        first = open_writer("held")
        with pytest.raises(LogLockedError):
            JsonlWriter(first.path)
        first.append(_event())  # the refused attempt did not disturb the holder
        first.close()
        open_writer("held").append(_event())  # released on close

    def test_two_different_logs_do_not_conflict(self, open_writer: Any) -> None:
        open_writer("one").append(_event())
        open_writer("two").append(_event())

    def test_a_refused_second_open_leaves_the_log_untouched(
        self, open_writer: Any
    ) -> None:
        first = open_writer("held")
        first.append(_event())
        first.append(_event())
        before = first.path.read_bytes()
        with pytest.raises(LogLockedError):
            JsonlWriter(first.path)
        assert first.path.read_bytes() == before

    def test_a_failed_open_releases_the_lock(self, tmp_path: Path) -> None:
        path = tmp_path / "logs" / "bad.jsonl"
        path.parent.mkdir()
        path.write_bytes(b"{nope}\n")
        for _ in range(2):  # the second try would be LogLockedError if the first leaked
            with pytest.raises(LogCorruptError):
                JsonlWriter(path)

    def test_the_log_and_its_directory_are_private_on_posix(
        self, tmp_path: Path, open_writer: Any
    ) -> None:
        if sys.platform == "win32":
            pytest.skip("POSIX file modes are not enforced on Windows")
        writer = open_writer("private")
        assert (writer.path.stat().st_mode & 0o777) == 0o600 & ~_umask()
        assert (writer.path.parent.stat().st_mode & 0o777) == 0o700 & ~_umask()

    def test_a_lock_error_that_is_not_contention_is_a_write_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def broken(fd: int) -> bool:
            raise OSError(37, "No locks available")

        monkeypatch.setattr(jsonl, "_try_lock", broken)
        with pytest.raises(LogWriteError):
            JsonlWriter(tmp_path / "logs" / "nolock.jsonl")


class TestASymlinkNeverRedirectsTheWriter:
    """D1 (plan 52ce5464): a link at ``<id>.jsonl`` or ``<id>.lock`` must not send a write
    (or the torn-tail cut) to a file outside the log root. Writing under ``logs/`` needs the
    same trust as the session directory, so this is defense in depth with a data-destroying
    worst case: a victim file with no trailing newline used to be cut to zero bytes."""

    @staticmethod
    def _refusal(path: Path, create: bool) -> BaseException | None:
        """The error a writer open raises (captured by type, never a bare except), or None."""
        try:
            JsonlWriter(path, create=create).close()
        except (LogCorruptError, LogWriteError) as exc:
            return exc
        return None

    @pytest.mark.parametrize("victim", [b"precious=1 no newline", b"a\nb\nprecious"])
    def test_a_link_to_a_file_is_refused_and_the_victim_keeps_every_byte(
        self, tmp_path: Path, victim: bytes
    ) -> None:
        target = tmp_path / "victim.txt"
        target.write_bytes(victim)
        path = tmp_path / "logs" / "r1.jsonl"
        path.parent.mkdir()
        symlink_or_skip(path, target)
        refusal = self._refusal(path, create=True)
        assert target.read_bytes() == victim  # first: the data, whatever was raised
        assert isinstance(refusal, LogCorruptError), refusal
        assert "not a regular file" in str(refusal)
        assert path.is_symlink()

    def test_a_dangling_link_is_refused_and_the_target_is_never_created(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "outside" / "made.txt"
        target.parent.mkdir()
        path = tmp_path / "logs" / "r1.jsonl"
        path.parent.mkdir()
        symlink_or_skip(path, target)
        refusal = self._refusal(path, create=True)
        assert not target.exists()
        assert isinstance(refusal, LogCorruptError), refusal

    def test_a_directory_at_the_log_path_is_refused_in_create_mode(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "logs" / "r1.jsonl"
        path.mkdir(parents=True)
        assert isinstance(self._refusal(path, create=True), LogCorruptError)

    @pytest.mark.parametrize("create", [True, False])
    def test_a_dangling_lock_link_is_a_write_error_and_creates_nothing(
        self, tmp_path: Path, create: bool
    ) -> None:
        if not hasattr(os, "O_NOFOLLOW"):
            pytest.skip(
                "no O_NOFOLLOW here (Windows): only the Windows legs can show it"
            )
        path = tmp_path / "logs" / "r1.jsonl"
        JsonlWriter(path).close()  # a real log, so create=False reaches the lock
        lock = path.with_name("r1.lock")
        lock.unlink()
        target = tmp_path / "outside" / "lock-made.txt"
        target.parent.mkdir()
        symlink_or_skip(lock, target)
        refusal = self._refusal(path, create=create)
        assert not target.exists()
        assert isinstance(refusal, LogWriteError), refusal

    def test_the_log_and_the_lock_are_opened_with_no_follow_where_there_is_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        if not hasattr(os, "O_NOFOLLOW"):
            pytest.skip("no O_NOFOLLOW here (Windows)")
        real_open = os.open
        seen: dict[str, int] = {}

        def spy(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            seen[Path(path).name] = seen.get(Path(path).name, 0) | flags
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", spy)
        JsonlWriter(tmp_path / "logs" / "nf.jsonl").close()
        assert seen["nf.jsonl"] & os.O_NOFOLLOW
        assert seen["nf.lock"] & os.O_NOFOLLOW

    def test_a_log_swapped_between_the_two_opens_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        if os.name == "nt":
            pytest.skip("Windows refuses to rename a file held open")
        path = tmp_path / "logs" / "r1.jsonl"

        def swap(directory: Path) -> None:  # between the append and the scan open
            os.replace(path, path.with_name("moved"))
            path.write_bytes(b"abc")

        monkeypatch.setattr(jsonl, "fsync_directory", swap)
        assert isinstance(self._refusal(path, create=True), LogCorruptError)

    def test_without_o_nofollow_the_flag_is_zero_and_a_plain_log_still_opens(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
        assert jsonl._nofollow() == 0
        writer = JsonlWriter(tmp_path / "logs" / "plain.jsonl")
        assert writer.append(_event()) == 1
        writer.close()

    def test_the_lstat_guard_alone_protects_the_victim_where_there_is_no_o_nofollow(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # the Windows case: no O_NOFOLLOW, so the lstat guard is the whole defense
        monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
        target = tmp_path / "victim.txt"
        target.write_bytes(b"precious=1 no newline")
        path = tmp_path / "logs" / "r1.jsonl"
        path.parent.mkdir()
        symlink_or_skip(path, target)
        refusal = self._refusal(path, create=True)
        assert target.read_bytes() == b"precious=1 no newline"
        assert isinstance(refusal, LogCorruptError), refusal


class TestADroppedHolder:
    @pytest.mark.parametrize("holder", ["writer", "log"])
    def test_a_writer_or_log_dropped_without_close_releases_its_lock(
        self, tmp_path: Path, holder: str
    ) -> None:
        if holder == "writer":
            JsonlWriter(tmp_path / "logs" / "gone.jsonl").append(_event())
        else:
            EventLog(tmp_path).append("gone", "turn/start")
        gc.collect()  # the finalizer, not an explicit close, frees the fds and the lock
        JsonlWriter(
            tmp_path / "logs" / "gone.jsonl"
        ).close()  # LogLockedError if leaked


class TestReopenAfterKill:
    def test_it_retries_only_when_told_to_and_the_default_is_windows_only(self) -> None:
        calls: list[int] = []

        def opener() -> str:
            calls.append(1)
            if len(calls) < 3:
                raise LogLockedError("held")
            return "writer"

        assert reopen_after_kill(opener, retry=True) == "writer" and len(calls) == 3
        calls.clear()
        with pytest.raises(LogLockedError):
            reopen_after_kill(opener, retry=False)
        assert len(calls) == 1
        default = inspect.signature(reopen_after_kill).parameters["retry"].default
        assert default is (sys.platform == "win32")


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def can_lock_in_child() -> bool:
    """Capability probe: can this platform start a Python child and does it have a lock module?

    Interface contract: no parameters; returns True when a child interpreter starts and
    imports the platform's lock module (``msvcrt`` on Windows, ``fcntl`` elsewhere); False on
    a failure to start or import (never raises). A test skips only when this says False; the
    non-vacuity test below pins that it says True on every platform CI runs.
    """
    module = "msvcrt" if sys.platform == "win32" else "fcntl"
    try:
        done = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            env=child_env(),
            capture_output=True,
            timeout=120 * current_timing_factor(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


CHILD = """
import sys
from fsm_llm.runtime.stores.jsonl import JsonlWriter
writer = JsonlWriter(sys.argv[1])
pad = "x" * 4000
def append():
    seq = writer.append({"ts": "t", "conversation_id": "child", "frame": 0,
                         "type": "turn/start", "pad": pad})
    print(seq, flush=True)
for _ in range(20):
    append()
print("ready", flush=True)
sys.stdin.readline()
for _ in range(3000):
    append()
"""


class TestLockAndAcknowledgementAcrossProcesses:
    def test_the_capability_probe_says_run_on_every_ci_platform(self) -> None:
        # Non-vacuity: a probe that always said False would skip the child test everywhere.
        if sys.platform not in {"linux", "win32", "darwin"}:
            pytest.skip("not a platform CI runs")
        assert can_lock_in_child() is True

    def test_a_killed_holder_loses_no_acknowledged_event_and_frees_the_lock(
        self, tmp_path: Path
    ) -> None:
        if not can_lock_in_child():
            pytest.skip("this platform cannot start a child that locks a file")
        path = tmp_path / "logs" / "child.jsonl"
        proc = subprocess.Popen(
            [sys.executable, "-c", CHILD, str(path)],
            env=child_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        watchdog = threading.Timer(180 * current_timing_factor(), proc.kill)
        watchdog.start()
        acked: list[int] = []
        try:
            assert proc.stdout is not None and proc.stdin is not None
            for raw in iter(proc.stdout.readline, b""):
                token = raw.strip()
                if token == b"ready":
                    break
                acked.append(int(token))
            else:
                pytest.fail("the child died before it was ready: " + _stderr(proc))
            assert acked == list(range(1, 21))
            with pytest.raises(
                LogLockedError
            ):  # refused across processes while it lives
                JsonlWriter(path)
            proc.stdin.write(b"go\n")
            proc.stdin.flush()
            for _ in range(5):  # let it append more, then kill it mid-run
                line = proc.stdout.readline().strip()
                if line:
                    acked.append(int(line))
            proc.kill()  # Popen.kill(): SIGKILL on POSIX, TerminateProcess on Windows
            proc.wait()
            acked.extend(int(x) for x in proc.stdout.read().split())
        finally:
            watchdog.cancel()
            proc.kill()
            proc.wait()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
        events = list(
            read_events(path)
        )  # a torn tail, if the kill made one, is ignored
        assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
        assert len(events) >= max(acked) >= 20  # no acknowledged event is missing
        writer = reopen_after_kill(lambda: JsonlWriter(path))  # the dead holder's lock
        try:
            assert writer.last_seq == len(events)
            assert writer.append(_event()) == len(events) + 1
        finally:
            writer.close()
        # Unverified until the Windows run: msvcrt.locking release on TerminateProcess, and
        # that a Windows child's log is readable by the parent right after the kill.


def _stderr(proc: subprocess.Popen[bytes]) -> str:
    assert proc.stderr is not None
    proc.kill()
    return proc.stderr.read().decode("utf-8", "replace")[-2000:]


# ---------------------------------------------------------------------------------------------
# Plan 77e5fef9 step 3b (D-005): the re-open of an evicted writer reads the tail, not the log.
# ---------------------------------------------------------------------------------------------

WINDOW_BYTES = (
    2 * MAX_EVENT_BYTES + 1
)  # the literal of the rule; the module constant may not move it


def _exact_line(seq: int, size: int) -> bytes:
    """A valid log line of exactly *size* bytes, newline included (``size`` up to the cap)."""
    line = _line(seq, pad="x" * (size - len(_line(seq, pad=""))))
    assert len(line) == size
    return line


def _small_lines(count: int) -> list[bytes]:
    return [_line(i + 1) for i in range(count)]


def _tail_shapes() -> dict[str, list[bytes]]:
    """Committed contents, by name: empty, one line, a few, over the window, a last line at the
    cap, three lines at the cap, a blob-bearing last line."""
    return {
        "empty": [],
        "one": _small_lines(1),
        "few": _small_lines(3),
        "over-window": [_exact_line(i + 1, 8_000) for i in range(20)],
        "edge-last": [*_small_lines(3), _exact_line(4, MAX_EVENT_BYTES)],
        "huge-lines": [_exact_line(i + 1, MAX_EVENT_BYTES) for i in range(3)],
        "blob-last": [
            *_small_lines(2),
            _line(3, response_ref="sha256:" + "a" * 64, finish_reason="stop"),
        ],
    }


def _torn_lengths() -> list[int]:
    """0 to 300, the edges of the cap, and a stride up to the cap."""
    edges = {MAX_EVENT_BYTES - 2, MAX_EVENT_BYTES - 1, MAX_EVENT_BYTES}
    edges |= {MAX_EVENT_BYTES + 1, MAX_EVENT_BYTES + 5}
    return sorted(set(range(301)) | edges | set(range(301, MAX_EVENT_BYTES, 4_099)))


def _torn_bytes(seq: int, length: int) -> bytes:
    """The first *length* bytes of the next line (never its newline); over the cap, filler."""
    if length >= MAX_EVENT_BYTES:
        return b"y" * length
    return _exact_line(seq, MAX_EVENT_BYTES)[:length]


def _point(path: Path, seq: int) -> tuple[int, int, int]:
    held = os.stat(path)
    return (seq, held.st_dev, held.st_ino)


def _reopen(
    base: Path, name: str, lines: list[bytes], torn: bytes, *, tail: bool
) -> tuple[str, int, bytes, bytes]:
    """Open the same bytes by full scan (``tail`` False) or by tail read; return the outcome.

    ``(outcome, last_seq, file bytes after the open, file bytes after one more append)``; the
    outcome is ``"ok"`` or the error type name.
    """
    path = base / ("tail" if tail else "scan") / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(lines))
    expect = _point(path, len(lines)) if tail else None
    with path.open("ab") as fh:
        fh.write(torn)
    try:
        writer = JsonlWriter(path, create=False, expect=expect)
    except LogCorruptError as exc:
        return type(exc).__name__, -1, path.read_bytes(), b""
    try:
        opened = path.read_bytes()
        writer.append(_event(99))
        return "ok", writer.last_seq, opened, path.read_bytes()
    finally:
        writer.close()


class TestTailReopen:
    """G2: a tail-read re-open gives what the full-scan open gives, for every shape of log and
    every torn tail, at a cost the size of the log does not move."""

    def test_the_tail_read_equals_the_full_scan_for_every_shape_and_torn_tail(
        self, tmp_path: Path
    ) -> None:
        """Kills: the tail ``seq`` off by one (M17), a torn tail not cut (M18), a window too small
        with no fallback (M19), a torn last line read as committed."""
        cases = 0
        for name, lines in _tail_shapes().items():
            for k in _torn_lengths():
                torn = _torn_bytes(len(lines) + 1, k)
                scan = _reopen(tmp_path, f"{name}-{k}", lines, torn, tail=False)
                tail = _reopen(tmp_path, f"{name}-{k}", lines, torn, tail=True)
                assert tail == scan, (name, k)
                if k < MAX_EVENT_BYTES:
                    assert scan[0] == "ok" and scan[1] == len(lines) + 1, (name, k)
                    assert scan[2] == b"".join(lines), (
                        name,
                        k,
                    )  # the torn tail was cut
                else:
                    assert scan[0] == "LogCorruptError", (name, k)
                cases += 1
        assert cases > 1_000  # every shape against every length

    def test_the_comparer_can_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The differential above is not vacuous: a tail read that cuts one byte early or reports
        another ``seq`` makes the two outcomes differ."""
        lines = _tail_shapes()["few"]
        real = jsonl._tail

        def early(fd: int, size: int) -> tuple[int, int] | None:
            found = real(fd, size)
            return None if found is None else (found[0] - 1, found[1])

        monkeypatch.setattr(jsonl, "_tail", early)
        with pytest.raises(AssertionError):
            torn = _torn_bytes(4, 7)
            scan = _reopen(tmp_path, "cmp", lines, torn, tail=False)
            tail = _reopen(tmp_path, "cmp", lines, torn, tail=True)
            assert tail == scan

    @pytest.mark.parametrize("shape", ["clean", "worst"])
    def test_a_reopen_of_a_big_log_reads_at_most_the_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
    ) -> None:
        """Counts, never a clock: the bytes a re-open reads on this thread stay under the window
        of ``2 * MAX_EVENT_BYTES + 1`` however large the log. Kills: a full scan (the equal
        result, the cost of the log), a window that falls back to the scan on the worst shape (a
        torn tail of the cap minus one after a last line of the cap)."""
        lines = [_exact_line(i + 1, 30_000) for i in range(100)]
        torn = b""
        if shape == "worst":
            lines[-1] = _exact_line(100, MAX_EVENT_BYTES)
            torn = _exact_line(101, MAX_EVENT_BYTES)[: MAX_EVENT_BYTES - 1]
        path = tmp_path / "big.jsonl"
        path.write_bytes(b"".join(lines))
        assert path.stat().st_size > 20 * WINDOW_BYTES
        point = _point(path, 100)
        with path.open("ab") as fh:
            fh.write(torn)
        read = count_reads(monkeypatch)
        writer = JsonlWriter(path, create=False, expect=point)
        monkeypatch.undo()
        try:
            assert writer.last_seq == 100
            assert path.read_bytes() == b"".join(lines)  # a torn tail was cut
        finally:
            writer.close()
        assert 0 < sum(read) <= WINDOW_BYTES, sum(read)

    def test_a_first_open_and_open_existing_keep_the_full_scan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A first open and ``resume`` never read the tail: only a re-open with a resume point
        does (D-005 e). Kills: the tail read used for a first open (a gapped middle would pass)."""
        calls: list[int] = []
        real = jsonl._tail
        monkeypatch.setattr(
            jsonl, "_tail", lambda fd, size: calls.append(size) or real(fd, size)
        )
        path = tmp_path / "logs" / "c1.jsonl"
        path.parent.mkdir()
        gapped = [_line(1), _line(5), _line(3)]  # a bad middle, a good last seq
        path.write_bytes(b"".join(gapped))
        with pytest.raises(LogCorruptError, match="line 2"):
            JsonlWriter(path)
        with pytest.raises(LogCorruptError, match="line 2"):
            JsonlWriter(path, create=False)
        with EventLog(tmp_path) as log, pytest.raises(LogCorruptError):
            log.open_existing("c1")
        assert calls == []
        path.write_bytes(b"".join(_small_lines(3)))
        JsonlWriter(path).close()
        assert calls == []

    def test_the_middle_of_the_log_is_trusted_after_a_reopen(
        self, tmp_path: Path
    ) -> None:
        """The documented limit (D-005 f): a re-open checks the tail only, so a gapped middle is
        found by the next full scan and not by this one. Kills: a re-open that scans."""
        path = tmp_path / "logs" / "c1.jsonl"
        path.parent.mkdir()
        path.write_bytes(b"".join(_small_lines(3)))
        point = _point(path, 3)
        path.write_bytes(
            b"".join([_line(1), _line(5), _line(3)])
        )  # same inode, bad middle
        writer = JsonlWriter(path, create=False, expect=point)
        try:
            assert writer.append(_event(0)) == 4
        finally:
            writer.close()
        with pytest.raises(LogCorruptError, match="line 2"):
            JsonlWriter(path)

    def test_a_reopen_refuses_a_log_another_writer_appended_to_while_idle(
        self, tmp_path: Path
    ) -> None:
        """Kills: the recorded ``seq`` trusted without looking at the file (a fork of the log)
        and a changed ``seq`` accepted (M21)."""
        path = tmp_path / "logs" / "c1.jsonl"
        first = JsonlWriter(path)
        first.append(_event(1))
        first.append(_event(2))
        point = first.resume_point
        first.close()
        other = JsonlWriter(path)  # the lock is free: another process could do this
        other.append(_event(3))
        other.close()
        before = path.read_bytes()
        with pytest.raises(LogCorruptError, match="changed while idle"):
            JsonlWriter(path, create=False, expect=point)
        assert path.read_bytes() == before  # the refusal wrote and cut nothing
        JsonlWriter(path).close()  # and it released the lock again

    def test_a_reopen_refuses_a_log_cut_back_while_idle(self, tmp_path: Path) -> None:
        path = tmp_path / "logs" / "c1.jsonl"
        first = JsonlWriter(path)
        for n in range(3):
            first.append(_event(n))
        point = first.resume_point
        first.close()
        data = path.read_bytes()
        cut = data[: data.rfind(b"\n", 0, len(data) - 1) + 1]
        path.write_bytes(cut)  # the same file, one line fewer
        with pytest.raises(LogCorruptError, match="changed while idle"):
            JsonlWriter(path, create=False, expect=point)
        assert path.read_bytes() == cut
        path.write_bytes(b"")  # and a file emptied
        with pytest.raises(LogCorruptError, match="changed while idle"):
            JsonlWriter(path, create=False, expect=point)

    def test_a_reopen_refuses_a_different_file_with_the_same_last_seq(
        self, tmp_path: Path
    ) -> None:
        """A swapped file: a copy of the log put over it has the same bytes and the same last
        ``seq`` and not the same inode. Kills: a swapped file accepted (M16)."""
        path = tmp_path / "logs" / "c1.jsonl"
        first = JsonlWriter(path)
        first.append(_event(1))
        point = first.resume_point
        first.close()
        if point[2] == 0:
            pytest.skip("this file system gives no inode: nothing to compare")
        copy = path.with_name("copy.tmp")
        copy.write_bytes(path.read_bytes())
        os.replace(copy, path)
        assert _point(path, 1)[1:] != point[1:]
        with pytest.raises(LogCorruptError, match="replaced while idle"):
            JsonlWriter(path, create=False, expect=point)
        assert [e["seq"] for e in read_events(path)] == [1]

    def test_a_reopen_of_a_vanished_log_creates_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "logs" / "c1.jsonl"
        first = JsonlWriter(path)
        first.append(_event(1))
        point = first.resume_point
        first.close()
        path.unlink()
        with pytest.raises(FileNotFoundError):
            JsonlWriter(path, create=False, expect=point)
        assert not path.exists()

    def test_a_reopen_refuses_a_last_line_that_does_not_decode(
        self, tmp_path: Path
    ) -> None:
        """Kills: an unparsable last line skipped or read as a torn tail."""
        path = tmp_path / "logs" / "c1.jsonl"
        first = JsonlWriter(path)
        first.append(_event(1))
        first.append(_event(2))
        point = first.resume_point
        first.close()
        data = path.read_bytes()
        bad = data[: data.rfind(b"\n", 0, len(data) - 1) + 1] + b"not json at all\n"
        path.write_bytes(bad)
        with pytest.raises(LogCorruptError, match="not valid JSON"):
            JsonlWriter(path, create=False, expect=point)
        assert path.read_bytes() == bad  # nothing was cut

    def test_a_window_that_cannot_decide_falls_back_to_the_scan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A short read, or a tail with no newline in a window that does not start at 0, is not
        guessed at: the scan decides (and raises for a shape no valid log has). Kills: a window
        with no fallback (M19) and a guess from a short buffer."""
        lines = [_exact_line(i + 1, 8_000) for i in range(30)]
        path = tmp_path / "logs" / "c1.jsonl"
        path.parent.mkdir()
        path.write_bytes(b"".join(lines))
        point = _point(path, 30)
        real = jsonl._read_at
        monkeypatch.setattr(jsonl, "_read_at", lambda fd, n, at: real(fd, n, at)[:-5])
        writer = JsonlWriter(path, create=False, expect=point)  # short read: the scan
        assert writer.last_seq == 30
        writer.close()
        monkeypatch.undo()
        path.write_bytes(b"".join(lines[:-1]) + b"z" * (2 * WINDOW_BYTES))
        with pytest.raises(LogCorruptError, match="no newline within"):
            JsonlWriter(path, create=False, expect=(29, 0, 0))
        # one newline in a window that does not start at 0: a last line over the cap, which the
        # scan refuses by its own rule (a decision from the window would be "line is N bytes")
        huge = b"z" * (WINDOW_BYTES + 5_000) + b"\n"
        path.write_bytes(b"".join(lines[:-1]) + huge)
        with pytest.raises(LogCorruptError, match="no newline within"):
            JsonlWriter(path, create=False, expect=(29, 0, 0))

    def test_a_reopen_keeps_the_open_flags_and_creates_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kills: the re-open creating (``mkdir`` or a directory fsync, M22) and dropping
        ``O_NOFOLLOW`` (M15): every ``os.open`` of the re-open carries the no-follow flag, none
        creates the log, and no directory is made or synced."""
        path = tmp_path / "logs" / "c1.jsonl"
        first = JsonlWriter(path)
        first.append(_event(1))
        point = first.resume_point
        first.close()
        opened: list[tuple[str, int]] = []
        real = os.open

        def spy(file: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            opened.append((os.fspath(file), flags))
            return real(file, flags, *args, **kwargs)

        synced: list[int] = []
        monkeypatch.setattr(os, "open", spy)
        monkeypatch.setattr(jsonl, "fsync_directory", lambda d: synced.append(1))
        monkeypatch.setattr(Path, "mkdir", lambda *a, **k: synced.append(2))
        JsonlWriter(path, create=False, expect=point).close()
        monkeypatch.undo()
        assert synced == []
        assert len(opened) == 3  # the lock, the log, the read handle
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        for name, flags in opened:
            assert flags & nofollow == nofollow, name
            assert not flags & os.O_CREAT or name.endswith(".lock"), name


class TestSource:
    def test_the_indirections_exist_and_default_to_the_os_calls(self) -> None:
        assert jsonl._write is os.write
        assert jsonl._fsync is os.fsync
        assert jsonl._truncate is os.ftruncate

    def test_the_d011_anchor_sits_once_at_the_writer_and_names_the_forbidden_shapes(
        self,
    ) -> None:
        text = JSONL_SOURCE.read_text(encoding="utf-8")
        token = "plan-2026-11-17T100529-21e6f00d/D-011"
        assert text.count(token) == 1
        block = text[text.index(f"# DECISION {token}") :].split("\n\n")[0].lower()
        for phrase in ("shared", "default=str", "text-mode", "buffered", "splitlines"):
            assert phrase in block, phrase

    def test_every_raw_os_open_in_the_module_adds_the_binary_flag(self) -> None:
        tree = ast.parse(JSONL_SOURCE.read_text(encoding="utf-8"))
        opens = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "open"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
        ]
        assert len(opens) >= 4  # lock, log, reader, directory
        for call in opens:
            flags = ast.unparse(call.args[1])
            directory = "O_RDONLY" in flags and "_binary_flag" not in flags
            assert "_binary_flag()" in flags or directory, ast.unparse(call)
