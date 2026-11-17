"""The blob store of the event log (plan 21e6f00d step 3): content addressed, write-if-absent.

Bytes and races, no clock. Unverified until the Windows run: real ``os.replace`` contention
on a target another process holds open (the simulated ``PermissionError`` below proves the
branch, not that Windows takes it), path length on a runner, and killing a Windows child.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.runtime import blobs
from fsm_llm.runtime.blobs import BlobStore, blob_ref
from fsm_llm.runtime.events import EventPayloadError, LogCorruptError, LogWriteError
from tests.conftest import current_timing_factor
from tests.test_fsm_llm.test_event_log_jsonl import _child_env


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _files(root: Path, suffix: str | None = None) -> list[Path]:
    found = [p for p in (root / "blobs").rglob("*") if p.is_file()]
    if suffix is None:
        return [p for p in found if p.suffix != ".tmp"]
    return [p for p in found if p.suffix == suffix]


@pytest.fixture
def store(tmp_path: Path) -> BlobStore:
    return BlobStore(tmp_path)


class TestContentAddress:
    @pytest.mark.parametrize(
        "data",
        [b"", b"\n", b"\r\n", b"a\x00b", bytes(range(256)), b"\xff" * 5, b"\r\n" * 3],
        ids=["empty", "lf", "crlf", "nul", "all-bytes", "high", "crlfs"],
    )
    def test_any_bytes_round_trip_under_their_hash(
        self, store: BlobStore, tmp_path: Path, data: bytes
    ) -> None:
        ref = store.put(data)
        assert ref == "sha256:" + _digest(data) == blob_ref(data)
        assert store.exists(ref)
        assert store.get(ref) == data
        (path,) = _files(tmp_path)
        assert path == tmp_path / "blobs" / _digest(data)[:2] / _digest(data)[2:]
        assert path.read_bytes() == data  # no newline translation on disk (O_BINARY)

    def test_put_json_is_canonical_and_text_safe(self, store: BlobStore) -> None:
        value = {"b": [1, {"y": 2, "x": 1}], "t": " \u0085 \r\n \ud800 é"}
        ref = store.put_json(value)
        assert ref == store.put_json(json.loads(json.dumps(value)))
        assert store.put_json({"b": 1, "a": 2}) == store.put_json({"a": 2, "b": 1})
        raw = store.get(ref)
        assert raw == (
            json.dumps(
                value, separators=(",", ":"), sort_keys=True, ensure_ascii=True
            ).encode("ascii")
        )
        assert json.loads(raw) == value

    def test_put_json_refuses_what_the_event_encoder_refuses(
        self, store: BlobStore, tmp_path: Path
    ) -> None:
        deep: Any = []
        for _ in range(100_000):
            deep = [deep]
        for bad in (deep, float("nan"), {1, 2}, object()):
            with pytest.raises(EventPayloadError):
                store.put_json({"x": bad})
        assert not (tmp_path / "blobs").exists()

    def test_the_size_cap_is_refused_before_any_write(
        self, store: BlobStore, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(blobs, "MAX_BLOB_BYTES", 10)
        assert store.put(b"x" * 10)
        with pytest.raises(EventPayloadError, match="blob"):
            store.put(b"x" * 11)
        with pytest.raises(EventPayloadError, match="bytes"):
            store.put("text")  # type: ignore[arg-type]
        assert len(_files(tmp_path)) == 1

    def test_the_relative_path_is_short_enough_for_a_long_root(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path.joinpath(*(["d" * 15] * 4))  # 60 characters of directory names
        ref = BlobStore(root).put(b"x")
        relative = BlobStore(root).path_for(ref).relative_to(root).as_posix()
        assert len(relative) == len("blobs/aa/") + 62 == 71
        assert len(str(root)) - len(str(tmp_path)) + 71 + 20 < 260  # with a temp name


class TestRefsAndTampering:
    @pytest.mark.parametrize(
        "ref",
        [
            "sha256:../..",
            "sha256:" + "A" * 64,
            "sha256:" + "a" * 63,
            "sha256:" + "a" * 64 + "\n",
            "md5:" + "a" * 64,
            "../" + "a" * 64,
            "",
            None,
            7,
        ],
    )
    def test_a_malformed_ref_is_refused_before_any_io(
        self, tmp_path: Path, ref: Any
    ) -> None:
        store = BlobStore(tmp_path / "never")
        for call in (store.get, store.exists, store.path_for):
            with pytest.raises(EventPayloadError):
                call(ref)
        assert not (tmp_path / "never").exists()

    def test_a_tampered_or_missing_blob_is_refused_on_read(
        self, store: BlobStore
    ) -> None:
        ref = store.put(b"abcdef")
        store.path_for(ref).write_bytes(
            b"abcdeX"
        )  # same size: only the re-hash sees it
        with pytest.raises(LogCorruptError, match="hash"):
            store.get(ref)
        store.path_for(ref).unlink()
        with pytest.raises(LogCorruptError, match="missing"):
            store.get(ref)
        assert not store.exists(ref)

    def test_a_blob_over_the_cap_on_disk_is_refused_without_reading_it_all(
        self, store: BlobStore, monkeypatch
    ) -> None:
        ref = store.put(b"x" * 4)
        monkeypatch.setattr(blobs, "MAX_BLOB_BYTES", 3)
        asked: list[int] = []
        real_open = open

        class Spy:
            def __init__(self, handle: Any) -> None:
                self.handle = handle

            def __enter__(self) -> Spy:
                return self

            def __exit__(self, *exc: object) -> None:
                self.handle.close()

            def read(self, size: int = -1) -> bytes:
                asked.append(size)
                return self.handle.read(size)  # type: ignore[no-any-return]

        monkeypatch.setattr(
            blobs, "open", lambda p, m: Spy(real_open(p, m)), raising=False
        )
        with pytest.raises(LogCorruptError, match="cap"):
            store.get(ref)
        assert asked == [4]  # the cap plus one, never the whole file


class TestWriteIfAbsent:
    def test_a_second_put_writes_nothing_and_a_wrong_size_file_is_repaired(
        self, store: BlobStore, tmp_path: Path, monkeypatch
    ) -> None:
        calls: list[Any] = []
        real = blobs._replace
        monkeypatch.setattr(
            blobs, "_replace", lambda a, b: (calls.append(b), real(a, b))[1]
        )
        ref = store.put(b"payload")
        assert store.put(b"payload") == ref
        assert len(calls) == 1 and len(_files(tmp_path)) == 1
        store.path_for(ref).write_bytes(
            b"pay"
        )  # a torn or foreign file of the wrong size
        assert store.put(b"payload") == ref
        assert len(calls) == 2 and store.get(ref) == b"payload"

    @pytest.mark.parametrize(
        ("winner", "outcome"),
        [(b"payload", "success"), (None, "refused"), (b"payloaX", "refused")],
        ids=["final-file-present-and-right", "no-final-file", "final-file-wrong-hash"],
    )
    @pytest.mark.parametrize("error", [PermissionError, FileExistsError])
    def test_a_lost_replace_race_is_success_only_when_the_final_file_is_right(
        self,
        store: BlobStore,
        tmp_path: Path,
        monkeypatch,
        winner: bytes | None,
        outcome: str,
        error: type[OSError],
    ) -> None:
        final = store.path_for(blob_ref(b"payload"))

        def lose(src: str, dst: str) -> None:
            if (
                winner is not None
            ):  # another writer's file lands, ours cannot replace it
                Path(dst).write_bytes(winner)
            raise error(13, "simulated: target held open")

        monkeypatch.setattr(blobs, "_replace", lose)
        if outcome == "success":
            assert store.put(b"payload") == blob_ref(b"payload")
            assert final.read_bytes() == b"payload"
        else:
            with pytest.raises(LogWriteError):
                store.put(b"payload")
        assert _files(tmp_path, ".tmp") == []  # our temp file never outlives the call

    def test_another_oserror_is_a_write_error_and_leaves_no_temp(
        self, store: BlobStore, tmp_path: Path, monkeypatch
    ) -> None:
        def broken(src: str, dst: str) -> None:
            raise OSError(28, "no space")

        monkeypatch.setattr(blobs, "_replace", broken)
        with pytest.raises(LogWriteError):
            store.put(b"payload")
        assert _files(tmp_path, ".tmp") == [] and _files(tmp_path) == []

    def test_the_source_uses_no_symlink(self) -> None:
        tree = ast.parse(Path(blobs.__file__).read_text(encoding="utf-8"))
        assert not [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr in {"symlink", "link"}
        ]


PAYLOADS = 100
CHILD = """
import sys
from fsm_llm.runtime.blobs import BlobStore
store = BlobStore(sys.argv[1])
print("ready", flush=True)
sys.stdin.readline()
for i in range(100):
    store.put(i.to_bytes(4, "big") * 50)
print("done", flush=True)
"""


class TestRaces:
    def test_threads_putting_the_same_payloads_leave_one_file_each(
        self, store: BlobStore, tmp_path: Path
    ) -> None:
        start = threading.Barrier(8)
        errors: list[BaseException] = []

        def work() -> None:
            try:
                start.wait()
                for i in range(PAYLOADS):
                    store.put(i.to_bytes(4, "big") * 50)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(_files(tmp_path)) == PAYLOADS
        assert len(_files(tmp_path, ".tmp")) <= 8  # a bound, not a promise: Windows AV
        for i in range(PAYLOADS):
            data = i.to_bytes(4, "big") * 50
            assert store.get(blob_ref(data)) == data

    def test_two_processes_racing_on_the_same_payloads_leave_one_file_each(
        self, store: BlobStore, tmp_path: Path
    ) -> None:
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", CHILD, str(tmp_path)],
                env=_child_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(2)
        ]
        watchdog = threading.Timer(
            180 * current_timing_factor(), lambda: [p.kill() for p in procs]
        )
        watchdog.start()
        try:
            for p in procs:  # both imported and ready: start them together
                assert p.stdout is not None
                assert p.stdout.readline().strip() == b"ready", (
                    p.stderr and p.stderr.read()
                )
            for p in procs:
                assert p.stdin is not None
                p.stdin.write(b"go\n")
                p.stdin.flush()
            results = [p.communicate() for p in procs]
        finally:
            watchdog.cancel()
            for p in procs:
                p.kill()
                p.wait()
        assert [p.returncode for p in procs] == [0, 0], results
        assert len(_files(tmp_path)) == PAYLOADS
        for i in range(PAYLOADS):
            data = i.to_bytes(4, "big") * 50
            assert store.get(blob_ref(data)) == data
