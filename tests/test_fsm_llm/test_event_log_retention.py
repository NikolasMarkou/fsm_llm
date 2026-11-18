"""Retention of the event log (plan 5a6a1f80 step 11, roadmap row 1.11b): ``EventLog.prune``.

Bytes, counts and mtimes set with ``os.utime``: no sleeps for ordering, no ``ts``. Every walk of
a tree skips ``.lock`` files (``tree_files``: Windows refuses to read a locked one). The model
differential is the safety proof: random trees of logs, forks, blobs that name blobs, held
writers and old and young files, judged against a plain model. Unverified until the Windows run:
a real refused unlink (simulated through ``retention._unlink``) and the lock of a killed child
(a bounded poll).
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import os
import random
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.runtime import EventLog, EventType, PruneReport, blobs, retention
from fsm_llm.runtime import log as log_module
from fsm_llm.runtime.blobs import BlobStore
from fsm_llm.runtime.constants import BLOB_REF_RE
from fsm_llm.runtime.stores.jsonl import LOCK_SUFFIX
from tests.test_fsm_llm.event_log_support import (
    child_env,
    tree_files,
    wait_ready,
)
from tests.test_fsm_llm.test_event_log_forks import _api, _chain, _shown
from tests.test_fsm_llm.test_event_log_jsonl import can_lock_in_child

HOUR = 3600.0
OLD = 10 * HOUR


def _age(path: Path, seconds: float = OLD) -> None:
    moment = time.time() - seconds
    os.utime(path, (moment, moment))


def _logfile(log: EventLog, cid: str) -> Path:
    return log.root / "logs" / f"{cid}.jsonl"


def _blobfile(log: EventLog, ref: str) -> Path:
    return BlobStore(log.root).path_for(ref)


def _mklog(
    log: EventLog, cid: str, *refs: str, age: float | None = OLD, hold: bool = False
) -> Path:
    """Write one event naming *refs*, release the writer (unless *hold*), age the file."""
    log.append(cid, EventType.TURN_START, refs=list(refs))
    if not hold:
        log.release(cid)
    path = _logfile(log, cid)
    if age is not None:
        _age(path, age)
    return path


def _mkblob(log: EventLog, name: str, *, age: float | None = OLD) -> str:
    ref = log.put_json({"blob": name})
    if age is not None:
        _age(_blobfile(log, ref), age)
    return ref


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in tree_files(root)
    }


def _exists(log: EventLog, ref: str) -> bool:
    return _blobfile(log, ref).is_file()


class TestNothingHappensByDefault:
    def test_a_default_prune_reads_and_deletes_nothing(self, log: EventLog) -> None:
        orphan = _mkblob(log, "orphan")
        _mklog(log, "old", _mkblob(log, "kept"))
        before = _snapshot(log.root)
        assert log.prune() == PruneReport()
        assert log.prune(blob_grace_seconds=0.0) == PruneReport()
        assert log.prune(dry_run=True) == PruneReport()
        assert _snapshot(log.root) == before and _exists(log, orphan)

    def test_the_defaults_are_pinned(self) -> None:
        for function in (EventLog.prune, retention.prune_root):
            params = inspect.signature(function).parameters
            assert params["max_age_seconds"].default is None
            assert params["max_logs"].default is None
            assert params["blob_grace_seconds"].default == 3600.0
            assert params["dry_run"].default is False

    def test_nothing_calls_prune_by_itself(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("prune was called")

        monkeypatch.setattr(log_module, "prune_root", boom)
        _mklog(log, "c1", age=None)
        log.append("c2", EventType.TURN_START)
        log.release("c2")
        log.close()
        users = [
            node.name
            for node in ast.walk(ast.parse(inspect.getsource(log_module)))
            if isinstance(node, ast.FunctionDef)
            and any(
                isinstance(n, ast.Name) and n.id == "prune_root" for n in ast.walk(node)
            )
        ]
        assert users == ["prune"]  # close, release and the writer table never call it

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("max_age_seconds", -1),
            ("max_age_seconds", "9"),
            ("max_age_seconds", True),
            ("max_age_seconds", float("nan")),
            ("max_logs", -1),
            ("max_logs", 1.5),
            ("max_logs", True),
            ("blob_grace_seconds", -0.5),
            ("blob_grace_seconds", None),
        ],
    )
    def test_a_bad_limit_is_refused_before_anything_is_touched(
        self, log: EventLog, name: str, value: Any
    ) -> None:
        path = _mklog(log, "old")
        with pytest.raises(ValueError, match=name):
            log.prune(**{"max_age_seconds": 1, name: value})
        assert path.is_file()

    def test_the_report_is_frozen_with_the_five_counts(self) -> None:
        report = PruneReport(1, 2, 3, 4, 5)
        assert [f.name for f in dataclasses.fields(report)] == [
            "logs_removed",
            "logs_busy",
            "blobs_removed",
            "blobs_young",
            "bytes_freed",
        ]
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.logs_removed = 9  # type: ignore[misc]


class TestLogsByAge:
    def test_age_is_the_file_mtime_and_a_young_log_stays(self, log: EventLog) -> None:
        old = _mklog(log, "old", age=5000)
        older = _mklog(log, "older", age=9000)
        young = _mklog(log, "young", age=10)
        size = old.stat().st_size + older.stat().st_size
        report = log.prune(max_age_seconds=1000)
        assert report == PruneReport(logs_removed=2, bytes_freed=size)
        assert not old.exists() and not older.exists() and young.is_file()

    def test_age_never_comes_from_ts(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            log_module, "_timestamp", lambda: "1970-01-01T00:00:00.000Z"
        )
        ancient_ts = _mklog(log, "ancient-ts", age=10)  # old ts, fresh file
        monkeypatch.undo()
        fresh_ts = _mklog(log, "fresh-ts", age=OLD)  # fresh ts, old file
        assert next(log.read("ancient-ts"))["ts"].startswith("1970")
        assert log.prune(max_age_seconds=HOUR).logs_removed == 1
        assert ancient_ts.is_file() and not fresh_ts.exists()

    def test_max_logs_keeps_the_newest_by_mtime_and_zero_keeps_none(
        self, log: EventLog
    ) -> None:
        paths = [_mklog(log, f"c{n}", age=100 * (n + 1)) for n in range(5)]
        report = log.prune(max_logs=2)
        assert report.logs_removed == 3
        assert [p.is_file() for p in paths] == [True, True, False, False, False]
        assert log.prune(max_logs=0).logs_removed == 2
        assert not any(p.exists() for p in paths)

    def test_both_limits_remove_the_union(self, log: EventLog) -> None:
        paths = [_mklog(log, f"c{n}", age=age) for n, age in enumerate((10, 20, OLD))]
        log.prune(max_age_seconds=HOUR, max_logs=1)
        assert [p.is_file() for p in paths] == [True, False, False]

    def test_a_file_that_is_no_conversation_log_is_never_touched(
        self, log: EventLog
    ) -> None:
        blob = _mkblob(log, "named-by-a-foreign-file")
        foreign = log.root / "logs" / "Not An Id.jsonl"
        foreign.parent.mkdir(parents=True, exist_ok=True)
        foreign.write_text(f'{{"refs": ["{blob}"]}}\n', encoding="utf-8")
        note = log.root / "logs" / "notes.txt"
        note.write_text("keep me", encoding="utf-8")
        for path in (foreign, note):
            _age(path)
        assert log.prune(max_age_seconds=1, max_logs=0) == PruneReport()
        assert foreign.is_file() and note.is_file() and _exists(log, blob)

    def test_a_log_appended_since_the_listing_is_kept(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _mklog(log, "busy-later")
        real = retention.exclusive_lock

        @contextmanager
        def touching(logs_dir: Path, stem: str, *, create: bool) -> Iterator[bool]:
            os.utime(path)  # a writer appended between the listing and the lock
            with real(logs_dir, stem, create=create) as held:
                yield held

        monkeypatch.setattr(retention, "exclusive_lock", touching)
        assert log.prune(max_age_seconds=HOUR) == PruneReport()
        assert path.is_file()

    def test_a_torn_tail_is_read_as_bytes_and_never_repaired(
        self, log: EventLog
    ) -> None:
        kept = _mkblob(log, "kept")
        partial = "sha256:" + "ab" * 15  # a ref cut by the crash: names nothing
        path = log.root / "logs" / "torn.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = f'{{"seq": 1, "refs": ["{kept}"]}}\n{{"seq": 2, "refs": ["{partial}'
        path.write_bytes(raw.encode())
        _age(path, 10)
        before = path.read_bytes()
        assert log.prune(max_logs=5, blob_grace_seconds=0.0).blobs_removed == 0
        assert path.read_bytes() == before and _exists(log, kept)

    def test_a_dry_run_changes_nothing_and_reports_what_a_run_does(
        self, log: EventLog
    ) -> None:
        live = _mkblob(log, "live")
        orphan = _mkblob(log, "orphan")
        _mklog(log, "old", live)
        _mklog(log, "young", age=10)
        (
            log.root / "logs" / f"old{LOCK_SUFFIX}"
        ).unlink()  # no writer ever: no lock file
        before = _snapshot(log.root)
        entries = sorted(p.name for p in (log.root / "logs").iterdir())
        dry = log.prune(max_age_seconds=HOUR, dry_run=True)
        assert dry == PruneReport(1, 0, 2, 0, dry.bytes_freed) and dry.bytes_freed > 0
        assert _snapshot(log.root) == before and _exists(log, orphan)
        assert sorted(p.name for p in (log.root / "logs").iterdir()) == entries
        assert log.prune(max_age_seconds=HOUR) == dry
        assert not _exists(log, orphan) and not _exists(log, live)


class TestTheWriterLock:
    def test_a_held_log_is_skipped_and_counted_then_taken_once_released(
        self, log: EventLog
    ) -> None:
        path = _mklog(log, "live", hold=True)
        assert log.has_writer("live")
        assert log.prune(max_age_seconds=1) == PruneReport(logs_busy=1)
        assert path.is_file() and log.append("live", EventType.TURN_START) == 2
        _age(path)
        log.release("live")
        assert log.prune(max_age_seconds=1).logs_removed == 1 and not path.exists()

    def test_a_second_event_log_cannot_take_a_held_log_either(
        self, log: EventLog
    ) -> None:
        path = _mklog(log, "live", hold=True)
        with EventLog(log.root) as other:
            assert other.prune(max_age_seconds=1).logs_busy == 1
        assert path.is_file()

    def test_a_blob_named_only_by_a_held_log_survives_and_an_orphan_goes(
        self, log: EventLog
    ) -> None:
        named = _mkblob(log, "named-by-held")
        orphan = _mkblob(log, "orphan")
        _mklog(log, "live", named, hold=True)
        report = log.prune(max_age_seconds=1, blob_grace_seconds=0.0)
        assert (report.logs_busy, report.blobs_removed) == (1, 1)
        assert _exists(log, named) and not _exists(log, orphan)

    def test_lock_files_survive_and_the_next_writer_opens(self, log: EventLog) -> None:
        _mklog(log, "old")
        lock = log.root / "logs" / f"old{LOCK_SUFFIX}"
        assert lock.is_file()
        assert log.prune(max_age_seconds=1).logs_removed == 1
        assert lock.is_file() and not _logfile(log, "old").exists()
        assert log.append("old", EventType.TURN_START) == 1  # a new log, a free lock
        assert _logfile(log, "old").is_file()

    def test_a_lock_file_is_never_read_by_a_scan(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mklog(log, "old", _mkblob(log, "x"))
        _mklog(log, "other", _mkblob(log, "y"), age=10)
        seen: list[str] = []
        real = retention._refs_in
        monkeypatch.setattr(
            retention, "_refs_in", lambda p: (seen.append(p.name), real(p))[1]
        )
        log.prune(max_logs=1)
        assert "other.jsonl" in seen and not any(
            name.endswith(LOCK_SUFFIX) for name in seen
        )

    def test_a_dry_run_creates_no_lock_file(self, log: EventLog) -> None:
        path = _mklog(log, "old")
        lock = log.root / "logs" / f"old{LOCK_SUFFIX}"
        lock.unlink()
        assert log.prune(max_age_seconds=1, dry_run=True).logs_removed == 1
        assert path.is_file() and not lock.exists()


CHILD = """
import pathlib, sys
from fsm_llm.runtime import EventLog
log = EventLog(sys.argv[1])
log.append(sys.argv[3], "turn/start", refs=[sys.argv[4]])
pathlib.Path(sys.argv[2]).write_text("ready")
sys.stdin.readline()
"""


class TestAnotherProcessHoldsTheLock:
    def test_the_capability_probe_says_run_on_every_ci_platform(self) -> None:
        if sys.platform not in {"linux", "win32", "darwin"}:
            pytest.skip("not a platform CI runs")
        assert (
            can_lock_in_child() is True
        )  # a probe that said False would skip all below

    def test_a_log_held_by_a_child_is_skipped_and_its_blob_survives(
        self, log: EventLog, tmp_path: Path
    ) -> None:
        if not can_lock_in_child():
            pytest.skip("this platform cannot start a child that locks a file")
        named = _mkblob(log, "named-by-child")
        orphan = _mkblob(log, "orphan")
        ready, err = tmp_path / "ready", tmp_path / "err"
        argv = [str(log.root), str(ready), "child", named]
        with err.open("wb") as errfile:
            proc = subprocess.Popen(
                [sys.executable, "-c", CHILD, *argv],
                env=child_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=errfile,
            )
        try:
            wait_ready(proc, ready, err)
            path = _logfile(log, "child")
            _age(path)
            report = log.prune(max_age_seconds=1, blob_grace_seconds=0.0)
            assert (report.logs_removed, report.logs_busy) == (0, 1)
            assert path.is_file() and _exists(log, named) and not _exists(log, orphan)
        finally:
            proc.kill()
            proc.wait()
            if proc.stdin is not None:
                proc.stdin.close()
        for _ in range(1200):  # Windows frees a killed process's lock asynchronously
            report = log.prune(max_age_seconds=1, blob_grace_seconds=0.0)
            if report.logs_busy == 0 or sys.platform != "win32":
                break
            time.sleep(0.05)
        assert (
            report.logs_removed == 1 and not path.exists() and not _exists(log, named)
        )


class TestBlobs:
    def test_a_referenced_blob_survives_and_an_old_orphan_goes(
        self, log: EventLog
    ) -> None:
        kept = _mkblob(log, "kept")
        gone = _mkblob(log, "gone")
        size = _blobfile(log, gone).stat().st_size
        _mklog(log, "young", kept, age=10)
        assert log.prune(max_logs=5) == PruneReport(0, 0, 1, 0, size)
        assert _exists(log, kept) and not _exists(log, gone)

    def test_a_blob_inside_the_grace_stays_and_is_counted_young(
        self, log: EventLog
    ) -> None:
        fresh = _mkblob(log, "fresh", age=None)
        inside = _mkblob(log, "inside", age=HOUR - 60)
        outside = _mkblob(log, "outside", age=HOUR + 60)
        report = log.prune(max_logs=5)
        assert (report.blobs_removed, report.blobs_young) == (1, 2)
        assert _exists(log, fresh) and _exists(log, inside)
        assert not _exists(log, outside)
        assert log.prune(max_logs=5, blob_grace_seconds=0.0).blobs_removed == 2

    def test_a_blob_that_names_blobs_keeps_them_all_alive(self, log: EventLog) -> None:
        third = log.put_blob(b"the end of the chain")
        second = log.put_blob(json.dumps({"next": third}).encode())
        first = log.put_blob(json.dumps({"next": second, "more": [1]}).encode())
        loose = _mkblob(log, "loose")
        for ref in (first, second, third):
            _age(_blobfile(log, ref))
        path = _mklog(log, "young", first, age=10)
        assert log.prune(max_logs=5).blobs_removed == 1
        assert all(_exists(log, r) for r in (first, second, third))
        assert not _exists(log, loose)
        _age(path)
        assert log.prune(max_age_seconds=1).blobs_removed == 3

    @pytest.mark.parametrize("chunk", [1, 7, 16, 70, 71, 72, 140, 1 << 20])
    @pytest.mark.parametrize("offset", [0, 1, 35, 69, 70, 71])
    def test_a_reference_cut_by_a_chunk_edge_is_still_found(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        chunk: int,
        offset: int,
    ) -> None:
        ref = "sha256:" + "0123456789abcdef" * 4
        partial = "sha256:" + "ab" * 20
        path = tmp_path / "scan.bin"
        path.write_bytes(b"x" * offset + ref.encode() + b"y" + partial.encode())
        monkeypatch.setattr(retention, "_SCAN_CHUNK_BYTES", chunk)
        assert retention._refs_in(path) == {ref}
        assert retention._refs_in(tmp_path / "missing") == set()

    def test_a_stray_temp_file_and_a_foreign_name_are_never_removed(
        self, log: EventLog
    ) -> None:
        ref = _mkblob(log, "real")
        folder = _blobfile(log, ref).parent
        stray = folder / "blob-123.tmp"
        foreign = log.root / "blobs" / "zz" / ("f" * 62)
        short = log.root / "blobs" / "ab" / "short"
        for path in (stray, foreign, short):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
            _age(path)
        log.prune(max_logs=5, blob_grace_seconds=0.0)
        assert stray.is_file() and foreign.is_file() and short.is_file()
        assert not _exists(log, ref) and folder.is_dir()  # no directory is removed

    def test_an_unreadable_log_means_no_sweep_at_all(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        orphan = _mkblob(log, "orphan")
        old = _mklog(log, "old")
        _mklog(log, "kept", age=10)
        real = retention._refs_in

        def unreadable(path: Path) -> set[str]:
            if path.name == "kept.jsonl":
                raise PermissionError(13, "denied")
            return real(path)

        monkeypatch.setattr(retention, "_refs_in", unreadable)
        report = log.prune(max_age_seconds=HOUR, blob_grace_seconds=0.0)
        assert (report.logs_removed, report.blobs_removed) == (1, 0)
        assert not old.exists() and _exists(log, orphan)

    def test_a_blob_written_while_the_pass_runs_is_inside_the_grace(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mklog(log, "kept", age=10)
        real = retention._mark
        made: list[str] = []

        def mark_then_a_writer_acts(*args: Any) -> Any:
            live = real(*args)
            made.append(log.put_json({"fresh": "blob, event not yet written"}))
            log.append("kept", EventType.TURN_START, refs=made[:1])
            return live

        monkeypatch.setattr(retention, "_mark", mark_then_a_writer_acts)
        assert log.prune(max_logs=5) == PruneReport(blobs_young=1)
        assert _exists(log, made[0])
        assert len(list(log.read("kept"))) == 2  # the appended line is whole

    def test_bytes_freed_adds_logs_and_blobs(self, log: EventLog) -> None:
        orphan = _mkblob(log, "orphan")
        path = _mklog(log, "old")
        sizes = path.stat().st_size + _blobfile(log, orphan).stat().st_size
        assert log.prune(max_age_seconds=1).bytes_freed == sizes


class TestAReputCountsAsYoung:
    def test_a_put_of_an_old_blob_refreshes_its_mtime_and_a_prune_keeps_it(
        self, log: EventLog
    ) -> None:
        data = b"some bytes a conversation reuses"
        ref = log.put_blob(data)
        path = _blobfile(log, ref)
        _age(path)
        assert log.put_blob(data) == ref
        assert time.time() - path.stat().st_mtime < 600
        assert log.prune(max_logs=5) == PruneReport(blobs_young=1) and _exists(log, ref)

    def test_a_reput_between_the_mark_and_the_sweep_survives(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = b"an old orphan a writer needs again"
        ref = log.put_blob(data)
        _age(_blobfile(log, ref))
        real = retention._mark

        def mark_then_reput(*args: Any) -> Any:
            live = real(*args)
            log.put_blob(data)
            return live

        monkeypatch.setattr(retention, "_mark", mark_then_reput)
        assert log.prune(max_logs=5) == PruneReport(blobs_young=1)
        assert log.get_blob(ref) == data

    def test_the_touch_changes_nothing_about_the_size_only_trust(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ref = log.put_blob(b"abcdef")
        path = _blobfile(log, ref)
        path.write_bytes(b"ABCDEF")  # tampered, same size
        monkeypatch.setattr(
            BlobStore,
            "_write_new",
            lambda *a: pytest.fail("a present blob is rewritten"),
        )
        assert log.put_blob(b"abcdef") == ref and path.read_bytes() == b"ABCDEF"

    def test_a_touch_that_fails_is_ignored_and_a_vanished_file_is_written_again(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = b"bytes of a blob"
        ref = log.put_blob(data)
        path = _blobfile(log, ref)

        def refused(target: Any) -> None:
            raise PermissionError(13, "held open")

        monkeypatch.setattr(blobs, "_utime", refused)
        assert log.put_blob(data) == ref and path.read_bytes() == data

        def swept(target: Any) -> None:
            os.unlink(target)  # a prune removed it between the size check and the touch
            raise FileNotFoundError(2, "gone")

        monkeypatch.setattr(blobs, "_utime", swept)
        assert log.put_blob(data) == ref and path.read_bytes() == data


class TestAFailedUnlinkIsSkippedAndCounted:
    def test_a_refused_unlink_of_a_log_and_of_a_blob_is_counted_never_retried(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        orphan = _mkblob(log, "orphan")
        path = _mklog(log, "old")
        refused: list[str] = []
        real = os.unlink

        def windows_says_no(target: Any) -> None:
            refused.append(Path(target).name)
            raise PermissionError(13, "in use")

        monkeypatch.setattr(retention, "_unlink", windows_says_no)
        report = log.prune(max_age_seconds=1)
        assert (report.logs_removed, report.logs_busy) == (0, 1)
        assert (report.blobs_removed, report.blobs_young) == (0, 1)
        assert sorted(refused) == sorted([path.name, _blobfile(log, orphan).name])
        assert path.is_file() and _exists(log, orphan) and report.bytes_freed == 0
        monkeypatch.setattr(retention, "_unlink", real)
        second = log.prune(max_age_seconds=1)
        assert (second.logs_removed, second.blobs_removed) == (1, 1)
        assert not path.exists() and not _exists(log, orphan)


class TestForksKeepSharedBlobsAlive:
    def test_pruning_the_source_of_a_fork_keeps_what_the_fork_names(
        self, log: EventLog
    ) -> None:
        api = _api(log, _chain())
        source, _ = api.start_conversation()
        for text in ("Ann", "b", "c"):
            api.converse(text, source)
        forked, _ = api.fork(source, at_turn=2)
        api.converse("d", forked)
        shown = _shown(api, forked)
        log.close()  # every writer released; the files stay
        root = log.root
        for path in (root / "blobs").rglob("*"):
            if path.is_file():
                _age(path)
        source_log = _logfile(log, source)
        _age(source_log)
        before = {p.name for p in (root / "blobs").rglob("*") if p.is_file()}
        with EventLog(root) as fresh:
            report = fresh.prune(max_age_seconds=HOUR)
            assert report.logs_removed == 1 and not source_log.exists()
            assert report.blobs_removed > 0  # the source's own blobs went: not vacuous
            after = {p.name for p in (root / "blobs").rglob("*") if p.is_file()}
            assert after < before
            named = set(
                BLOB_REF_RE.findall(_logfile(fresh, forked).read_text(encoding="utf-8"))
            )
            assert named and all(_exists(fresh, ref) for ref in named)
            other = _api(fresh, _chain())
            new_id, _ = other.resume(forked)
            assert new_id == forked and _shown(other, forked) == shown
            other.converse("e", forked)  # and it goes on
            assert other.get_current_state(forked) != shown[0]

    def test_without_the_mark_of_the_fork_log_the_shared_blobs_would_go(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Non-vacuity: the scenario above fails when a remaining log marks nothing.
        api = _api(log, _chain())
        source, _ = api.start_conversation()
        api.converse("Ann", source)
        forked, _ = api.fork(source, at_turn=1)
        log.close()
        for path in (log.root / "blobs").rglob("*"):
            if path.is_file():
                _age(path)
        _age(_logfile(log, source))
        monkeypatch.setattr(retention, "_refs_in", lambda path: set())
        with EventLog(log.root) as fresh:
            fresh.prune(max_age_seconds=HOUR)
            named = set(
                BLOB_REF_RE.findall(_logfile(fresh, forked).read_text(encoding="utf-8"))
            )
            assert named and not all(_exists(fresh, ref) for ref in named)


_SEEDS = range(40)


def _scenario(rnd: random.Random, log: EventLog) -> dict[str, Any]:
    """Build a random tree; return the model of what is named, old, young and held."""
    refs: list[str] = []
    named_by_blob: dict[str, list[str]] = {}
    blob_old: dict[str, bool] = {}
    for i in range(rnd.randint(3, 10)):
        inner = [r for r in refs if rnd.random() < 0.25]
        ref = log.put_blob(json.dumps({"i": i, "named": inner}).encode())
        refs.append(ref)
        named_by_blob[ref] = inner
        blob_old[ref] = rnd.random() < 0.6
        if blob_old[ref]:
            _age(_blobfile(log, ref))
    logs: dict[str, dict[str, Any]] = {}
    shared = [r for r in refs if rnd.random() < 0.3]  # what a fork and its source share
    for n in range(rnd.randint(1, 5)):
        named = shared + [r for r in refs if rnd.random() < 0.3]
        entry = {"named": named, "old": rnd.random() < 0.5, "held": rnd.random() < 0.25}
        cid = f"c{n}"
        _mklog(log, cid, *named, age=OLD if entry["old"] else 10, hold=entry["held"])
        logs[cid] = entry
    return {"named_by_blob": named_by_blob, "blob_old": blob_old, "logs": logs}


def _model(log: EventLog, scenario: dict[str, Any]) -> tuple[PruneReport, set[str]]:
    logs, named_by_blob = scenario["logs"], scenario["named_by_blob"]
    gone = {c for c, e in logs.items() if e["old"] and not e["held"]}
    busy = {c for c, e in logs.items() if e["old"] and e["held"]}
    live: set[str] = set()
    pending = [r for c, e in logs.items() if c not in gone for r in e["named"]]
    while pending:
        ref = pending.pop()
        if ref not in live:
            live.add(ref)
            pending.extend(named_by_blob[ref])
    dead = [r for r in named_by_blob if r not in live]
    swept = {r for r in dead if scenario["blob_old"][r]}
    young = [r for r in dead if not scenario["blob_old"][r]]
    freed = sum(_logfile(log, c).stat().st_size for c in gone)
    freed += sum(_blobfile(log, r).stat().st_size for r in swept)
    return PruneReport(len(gone), len(busy), len(swept), len(young), freed), swept


class TestTheModelDifferential:
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_every_blob_a_remaining_log_names_survives_and_every_old_orphan_goes(
        self, tmp_path: Path, seed: int
    ) -> None:
        with EventLog(tmp_path / f"events-{seed}") as log:
            scenario = _scenario(random.Random(seed), log)
            expected, swept = _model(log, scenario)
            before = _snapshot(log.root)
            assert (
                log.prune(
                    max_age_seconds=5 * HOUR, blob_grace_seconds=5 * HOUR, dry_run=True
                )
                == expected
            )
            assert _snapshot(log.root) == before
            report = log.prune(max_age_seconds=5 * HOUR, blob_grace_seconds=5 * HOUR)
            assert report == expected
            on_disk = {r for r in scenario["named_by_blob"] if _exists(log, r)}
            assert on_disk == set(scenario["named_by_blob"]) - swept
            for cid, entry in scenario["logs"].items():
                if _logfile(log, cid).exists():
                    assert all(_exists(log, r) for r in entry["named"]), cid
            assert log.prune(max_age_seconds=5 * HOUR, blob_grace_seconds=5 * HOUR) == (
                PruneReport(
                    logs_busy=expected.logs_busy, blobs_young=expected.blobs_young
                )
            )

    def test_the_scenarios_cover_every_branch_they_are_meant_to(
        self, tmp_path: Path
    ) -> None:
        seen = {"busy": 0, "gone": 0, "nested": 0, "young": 0, "swept": 0}
        for seed in _SEEDS:
            with EventLog(tmp_path / f"cover-{seed}") as log:
                scenario = _scenario(random.Random(seed), log)
                report, swept = _model(log, scenario)
                seen["busy"] += report.logs_busy
                seen["gone"] += report.logs_removed
                seen["young"] += report.blobs_young
                seen["swept"] += len(swept)
                seen["nested"] += any(scenario["named_by_blob"].values())
        assert all(count > 0 for count in seen.values()), seen
