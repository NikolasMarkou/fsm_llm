"""The JSONL store of the event log: one append-only file per conversation tree.

One event is one ``os.write`` of ``compact_ascii_json + b"\\n"`` on an ``O_APPEND`` fd, never
buffered, so a process killed between two events loses nothing it acknowledged. The newline
is the only commit marker: a last line without one was never committed. Only the writer that
holds the log's advisory lock repairs a torn tail (cuts the file back to the last newline); a
reader ignores a newline-less tail and never writes. A bad or gapped line in the middle is
corruption and raises, it is never skipped.

``JsonlWriter`` is the writer, ``read_events`` the reader, ``check_conversation_id`` the file
name rule. Imports ``runtime.events``, ``runtime.constants`` and the standard library only
(``fcntl`` and ``msvcrt`` are imported inside the lock functions, so the module imports on
every platform).

Unverified until the Windows run: ``msvcrt.locking`` on a real file (contention and release
when the process dies), the effect of ``O_BINARY`` on the raw fd, ``os.ftruncate`` on an
``O_APPEND`` handle, and killing a Windows child. Power-loss durability is argued, not tested:
a checkpoint append is followed by ``os.fsync`` (see ``append``); nothing else is. The
directory fsync covers a new file's entry in its own directory only: the entries of ``logs/`` and
``blobs/`` in the log root are not fsynced either (a power loss on the first write into a fresh
root can lose the directory).
"""

from __future__ import annotations

import errno
import os
import stat
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fsm_llm.runtime.constants import (
    CONVERSATION_ID_RE,
    LOG_FORMAT_VERSION,
    MAX_EVENT_BYTES,
    WINDOWS_RESERVED_NAMES,
)
from fsm_llm.runtime.events import (
    EventPayloadError,
    LogCorruptError,
    LogLockedError,
    LogWriteError,
    decode_event,
    encode_event,
)

LOG_SUFFIX = ".jsonl"
LOCK_SUFFIX = ".lock"

#: Read size of the scan. A line (or a torn tail) never needs more than ``MAX_EVENT_BYTES``.
_CHUNK_BYTES = 65_536

#: ``errno`` values that mean "another holder has the lock" (``flock`` and ``msvcrt.locking``);
#: any other ``OSError`` of the lock call is a real failure, not contention.
_LOCK_HELD = frozenset({errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK})

# Indirections tests patch to count fsyncs, inject a short write and simulate death. The
# methods call these module globals at call time, so a patch takes effect at once.
_write = os.write
_fsync = os.fsync
_truncate = os.ftruncate


def _binary_flag() -> int:
    """``os.O_BINARY`` on Windows, else 0. Read at call time (a test substitutes a sentinel)."""
    return getattr(os, "O_BINARY", 0)


def _nofollow() -> int:
    """``os.O_NOFOLLOW`` where the platform has it, else 0 (Windows: the ``lstat`` guard of
    ``JsonlWriter._open`` is then the whole defense). Read at call time, like ``_binary_flag``."""
    return getattr(os, "O_NOFOLLOW", 0)


def check_conversation_id(conversation_id: str) -> str:
    """Return *conversation_id* when it is a safe file name on every platform.

    Interface contract (callers: ``JsonlWriter`` now, ``EventLog`` in a later step):
        - ``conversation_id`` must fully match ``CONVERSATION_ID_RE`` (lowercase letters,
          digits, ``_`` and ``-``, 1 to 64 characters: two ids never differ only by case, and
          nothing can reach another directory) and must not be a Windows device name
          (``con``, ``nul``, ``com1`` ...; reserved with or without an extension).
        - Returns the id unchanged; raises ``EventPayloadError`` otherwise. Never touches
          the disk.
    """
    if not isinstance(conversation_id, str) or not CONVERSATION_ID_RE.fullmatch(
        conversation_id
    ):
        raise EventPayloadError(
            "conversation id must be 1 to 64 characters of [a-z0-9_-], got "
            f"{conversation_id!r:.80}"
        )
    if conversation_id.split(".", 1)[0] in WINDOWS_RESERVED_NAMES:
        raise EventPayloadError(
            f"conversation id {conversation_id!r} is a Windows device name"
        )
    return conversation_id


def _try_lock(fd: int) -> bool:
    """Take the exclusive advisory lock on *fd*: True if taken, False if another holds it.

    POSIX ``flock(LOCK_EX | LOCK_NB)``, Windows ``msvcrt.locking(LK_NBLCK)`` on byte 0 (the fd
    is at position 0). Refused, never waited for. The OS releases the lock when the process
    dies, so there is no stale lock and no pid to probe. Any ``OSError`` that is not
    contention propagates.
    """
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in _LOCK_HELD:
            return False
        raise
    return True


def _release_lock(fd: int) -> None:
    """Release the lock of *fd* and close it (closing alone releases a POSIX flock)."""
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass  # closing the handle below releases it anyway
    finally:
        os.close(fd)


@contextmanager
def exclusive_lock(
    log_dir: Path, conversation_id: str, *, create: bool = True
) -> Iterator[bool]:
    """Yield True while the caller holds the writer lock ``<id>.lock`` (caller: retention).

    Refused, never waited for: a live writer of this process or another, or a lock file that
    cannot be opened, yields False. The file is never read or unlinked. With *create* False a
    missing lock file yields True and creates nothing (no writer ever existed).
    """
    path = log_dir / (check_conversation_id(conversation_id) + LOCK_SUFFIX)
    try:
        fd = os.open(
            path,
            os.O_RDWR | _binary_flag() | _nofollow() | (os.O_CREAT if create else 0),
            0o600,
        )
    except FileNotFoundError:
        yield True
        return
    except OSError:
        yield False
        return
    try:
        try:
            held = _try_lock(fd)
        except OSError:
            held = False
        yield held
    finally:
        _release_lock(fd)


def fsync_directory(directory: Path) -> None:
    """Make a new file's directory entry durable (callers: ``JsonlWriter`` and ``BlobStore``).

    Interface contract: *directory* is an existing directory; returns None and never raises (a
    file system that cannot fsync a directory is skipped). POSIX only: ``os.open`` of a
    directory raises on Windows, so it does nothing there.
    """
    if sys.platform == "win32":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _scan(fd: int) -> Iterator[tuple[dict[str, Any], int]]:
    """Yield ``(event, end_offset)`` of every committed line of *fd*, from offset 0.

    A committed line ends in ``b"\\n"``. Splits on that byte only (never ``splitlines``).
    Raises ``LogCorruptError`` for a committed line that fails ``decode_event`` (over the cap,
    not JSON, too deep ...), whose ``seq`` is not its 1-based line number, or a tail with no
    newline that is too long to be the torn prefix of any valid line. A shorter newline-less
    tail is not committed: the scan ends before it. Memory is bounded by ``_CHUNK_BYTES`` plus
    ``MAX_EVENT_BYTES`` however large the file is.
    """
    pending = b""
    offset = 0
    index = 0
    while True:
        chunk = os.read(fd, _CHUNK_BYTES)
        if not chunk:
            return
        pending += chunk
        start = 0
        while (end := pending.find(b"\n", start)) != -1:
            line = pending[start : end + 1]
            start = end + 1
            event = decode_event(line)
            index += 1
            seq = event["seq"]
            if type(seq) is not int or seq != index:
                raise LogCorruptError(
                    f"log line {index} has seq {seq!r}, expected {index}"
                )
            offset += len(line)
            yield event, offset
        pending = pending[start:]
        if len(pending) >= MAX_EVENT_BYTES:
            raise LogCorruptError(
                f"no newline within the {MAX_EVENT_BYTES} byte cap after line {index}"
            )


#: What a re-open reads from the end: a torn tail (under ``MAX_EVENT_BYTES``), the last line (at
#: most ``MAX_EVENT_BYTES`` with its newline) and the newline before it.
_TAIL_BYTES = 2 * MAX_EVENT_BYTES + 1


def _read_at(fd: int, size: int, offset: int) -> bytes:
    """Up to *size* bytes of *fd* at *offset* (``os.pread``; a seek and read where it is absent)."""
    if hasattr(os, "pread"):
        return os.pread(fd, size, offset)
    os.lseek(fd, offset, os.SEEK_SET)
    return os.read(fd, size)


# DECISION plan-2026-11-26T213359-77e5fef9/D-005
# A re-open of an evicted writer trusts the middle of the log and checks the tail: ONE bounded
# window from the end (never the whole file) gives the `seq` of the last committed line, which the
# writer compares, with the file identity, to what it recorded at the eviction. Do NOT continue
# from the recorded `seq` without reading the file (the lock was free while the writer idled:
# another process may have appended, cut or replaced the log, and the next line would fork it),
# do NOT guess when the window cannot decide (None: the caller scans), do NOT skip a last line
# that fails `decode_event`, and do NOT use this for a first open or `resume` (full scan, which
# also finds a gapped middle). See decisions.md D-005.
def _tail(fd: int, size: int) -> tuple[int, int] | None:
    """``(end offset of the last committed line, its seq)`` from the last ``_TAIL_BYTES`` of *fd*.

    Interface contract (caller: ``JsonlWriter._open``): *size* is the ``fstat`` size; no committed
    line gives ``(0, 0)``; None when the window cannot decide (a short read, fewer than two
    newlines in a window not at offset 0), so the caller scans; ``LogCorruptError`` when the last
    committed line fails ``decode_event`` or has no integer ``seq``. Never writes.
    """
    start = max(0, size - _TAIL_BYTES)
    buf = _read_at(fd, size - start, start)
    end = buf.rfind(b"\n")
    if len(buf) != size - start or len(buf) - end - 1 >= MAX_EVENT_BYTES:
        return None
    if end < 0:
        return (0, 0) if start == 0 else None
    before = buf.rfind(b"\n", 0, end)
    if before < 0 and start > 0:
        return None
    seq = decode_event(buf[before + 1 : end + 1])["seq"]
    if type(seq) is not int:
        raise LogCorruptError(f"the last log line has seq {seq!r}")
    return start + end + 1, seq


def read_events(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Yield the committed events of the log at *path*, in file order. Never writes.

    Interface contract (callers: tests now, ``EventLog.read`` later):
        - A generator: the file is opened at the first ``next()`` and closed when the
          generator ends or is closed. A missing file raises ``FileNotFoundError`` then.
        - Ignores a last line without a newline (it was never committed) and never repairs
          it; only a lock-holding ``JsonlWriter`` truncates.
        - Raises ``LogCorruptError`` on a bad, over-cap, too-deep or misnumbered line and on
          a newline-free tail of ``MAX_EVENT_BYTES`` or more; a refused line is never parsed.
        - Each event is the decoded envelope dict; ``seq`` is the order, ``ts`` is not.
    """
    fd = os.open(path, os.O_RDONLY | _binary_flag())
    try:
        for event, _ in _scan(fd):
            yield event
    finally:
        os.close(fd)


class JsonlWriter:
    """The single writer of one log file; holds the file's advisory lock until ``close``.

    Interface contract (callers: tests now, ``EventLog`` in a later step):
        - ``JsonlWriter(path)``: *path* must be ``<id>.jsonl`` with ``id`` accepted by
          ``check_conversation_id`` (else ``EventPayloadError``, nothing touched). Creates the
          parent directory (``0o700``), takes the lock ``<id>.lock`` beside the log
          (``LogLockedError`` if another writer holds it, in this process or another),
          opens the log (``0o600``, append-only), repairs a torn tail and learns the last
          ``seq``. ``LogCorruptError`` for a log with a bad or gapped middle line (the lock is
          released again); ``LogWriteError`` for any other ``OSError``.
        - ``append(event, *, checkpoint=False) -> int``: *event* holds every envelope key
          except ``v`` and ``seq``; the writer assigns both (``seq`` is the previous plus 1,
          never a clock) and returns the ``seq``. Thread safe. ``EventPayloadError`` (bad or
          oversized payload) writes nothing and changes nothing. A failure while writing or
          fsyncing poisons the writer: that append raises ``LogWriteError`` (``OSError``) or
          the interrupt itself, and every later append raises ``LogWriteError``, so nothing
          can land after a partial line and a failed checkpoint can never be followed by the
          call it protects.
        - ``append_many(events, *, checkpoint=False) -> list[int]``: *events* are appended as
          consecutive lines in ONE write (contiguous ``seq``, one fsync when *checkpoint*);
          every event is encoded first, so a refused payload writes nothing. Not crash
          atomic: a kill inside the write leaves a whole-line prefix of the batch (never a
          half line once the next writer cut the tail). An empty list writes nothing.
        - ``JsonlWriter(path, create=False)`` (``EventLog.open_existing``, plan 5a6a1f80): the log must
          already exist. A missing log raises ``FileNotFoundError`` BEFORE anything is created (no
          directory, no ``.lock``, no log) and so does a log that is not a regular file (a symbolic
          link, ``LogCorruptError``: the appends would land outside the log root); the lock is
          taken before the log is scanned, and the
          torn-tail repair and the ``seq`` continuation are the same code as for a new writer.
          A log path that is not a regular file is refused in BOTH modes (before the lock is
          opened), and the log and the lock are opened with ``O_NOFOLLOW`` where the platform has
          it, so a link at ``<id>.lock`` is ``LogWriteError`` and never followed. Windows has no
          ``O_NOFOLLOW``: there the ``lstat`` check is the whole defense (unverified until the
          Windows run).
        - ``JsonlWriter(path, create=False, expect=writer.resume_point)`` (a RE-OPEN of an
          evicted idle writer): every check above is kept, the scan becomes a read of the last
          ``2 * MAX_EVENT_BYTES + 1`` bytes, so the cost does not grow with the log. A file that
          is not the one that was closed (another inode) or whose last committed ``seq`` is not
          the recorded one (appended or cut while the lock was free) is ``LogCorruptError``; a
          torn tail is cut as for a new writer. The middle is trusted; a first open and ``resume``
          scan the whole log.
        - ``close()``: idempotent; releases the fd and the lock. The ``.lock`` file stays. A
          writer that is dropped without ``close`` is closed by its finalizer, so a forgotten
          one frees its two descriptors and the lock when the object is collected (do not rely
          on it: a long-running host closes, see ``EventLog``).
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        create: bool = True,
        expect: tuple[int, int, int] | None = None,
    ) -> None:
        self.path = Path(path)
        self._create = create
        self._expect = expect
        name = self.path.name
        if not name.endswith(LOG_SUFFIX):
            raise EventPayloadError(
                f"a log file is named <id>{LOG_SUFFIX}, got {name!r:.80}"
            )
        self.conversation_id = check_conversation_id(name[: -len(LOG_SUFFIX)])
        self._guard = threading.Lock()
        self._sync_guard = (
            threading.Lock()
        )  # held across an fsync; order: _sync_guard, _guard
        self._fd = -1
        self._lock_fd = -1
        self._seq = 0
        self._failure: str | None = None
        opened = False
        try:
            self._open()
            opened = True
        except OSError as exc:
            if not create and isinstance(exc, FileNotFoundError):
                raise  # an unknown id: the caller says so, nothing was created
            raise LogWriteError(
                f"cannot open log {self.path.name}: {type(exc).__name__} "
                f"{exc.strerror or exc.errno}"
            ) from exc
        finally:
            if not opened:
                self.close()

    @property
    def last_seq(self) -> int:
        """The ``seq`` of the last committed event (0 for an empty log)."""
        return self._seq

    @property
    def resume_point(self) -> tuple[int, int, int]:
        """``(last seq, st_dev, st_ino)`` of the open file: what ``expect=`` of a re-open checks.

        The device and inode are 0 where the platform gives none (no identity check then).
        """
        try:
            held = os.fstat(self._fd)
        except OSError:
            return (self._seq, 0, 0)
        return (self._seq, held.st_dev, held.st_ino)

    @property
    def poisoned(self) -> bool:
        """True after a failed write or fsync: every append raises until the writer is replaced."""
        return self._failure is not None

    def _open(self) -> None:
        if self._create:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # A log is a regular file under the log root: a symbolic link (dangling or not) would
        # send the appends, and the torn-tail cut, to a file elsewhere. Both modes look before
        # anything is opened; an unknown id raises FileNotFoundError here, before any file is
        # made, and only a creating writer may meet a missing file.
        try:
            mode = os.lstat(self.path).st_mode
        except FileNotFoundError:
            if not self._create:
                raise
        else:
            if not stat.S_ISREG(mode):
                raise LogCorruptError(
                    f"log {self.path.name} is not a regular file (a link or a directory)"
                )
        lock_path = self.path.with_name(self.conversation_id + LOCK_SUFFIX)
        self._lock_fd = os.open(
            lock_path, os.O_RDWR | os.O_CREAT | _binary_flag() | _nofollow(), 0o600
        )
        if not _try_lock(self._lock_fd):
            raise LogLockedError(
                f"log {self.path.name} is held by another writer; a lock is refused, "
                "never waited for"
            )
        existed = self.path.exists()
        self._fd = os.open(
            self.path,
            os.O_WRONLY
            | os.O_APPEND
            | (os.O_CREAT if self._create else 0)
            | _binary_flag()
            | _nofollow(),
            0o600,
        )
        if not existed:
            fsync_directory(self.path.parent)
        read_fd = os.open(self.path, os.O_RDONLY | _binary_flag() | _nofollow())
        try:
            held, scanned = os.fstat(self._fd), os.fstat(read_fd)
            # a path swapped since the append fd opened: scan and cut must judge the file
            # that fd holds (st_ino is 0 on some Windows filesystems: no comparison there)
            same = (held.st_dev, held.st_ino) == (scanned.st_dev, scanned.st_ino)
            if held.st_ino and not same:
                raise LogCorruptError(f"log {self.path.name} changed while opened")
            expect = self._expect
            if expect and expect[2] and (held.st_dev, held.st_ino) != expect[1:]:
                raise LogCorruptError(f"log {self.path.name} was replaced while idle")
            good = 0
            size = scanned.st_size
            tail = _tail(read_fd, size) if expect else None
            if (
                tail is not None
            ):  # a re-open: the tail only, the middle is trusted (D-005)
                good, self._seq = tail
            else:
                for event, end in _scan(read_fd):
                    self._seq = event["seq"]
                    good = end
            if expect and self._seq != expect[0]:
                raise LogCorruptError(
                    f"log {self.path.name} changed while idle: last seq {self._seq}, "
                    f"expected {expect[0]}"
                )
        finally:
            os.close(read_fd)
        if size > good:  # a torn tail: we hold the lock, so cutting it is ours to do
            _truncate(self._fd, good)

    # DECISION plan-2026-11-17T100529-21e6f00d/D-011
    # This is a LOCAL writer: do NOT replace it with, or share it as, an atomic-save helper
    # (four anchors forbid a shared helper without a maintainer decision, and an append is
    # not a replace). Do NOT use a buffered open("ab") (a SIGKILL lost 10 to 22 acknowledged
    # events), a text-mode open (newline translation, the locale codec), str.splitlines()
    # (U+2028 and U+0085 would cut one event into three), default=str or indent= in the
    # encoder (hidden reprs, gigabytes on 3.14), or a second write per event (a kill between
    # the two leaves a torn line). One event is one _write of the whole line (a batch is one
    # _write of its concatenated lines), looped only over a short write; a partial line may
    # exist only as a torn tail, cut by the next lock holder, so any interrupted write
    # poisons this writer. See D-011 and D-012.
    # DECISION plan-2026-11-24T193347-3f4530d3/D-014
    # The line and its seq are written under `_guard`; the fsync of a checkpoint runs AFTER it,
    # under `_sync_guard`, so an inline append on the event loop never waits for a disk that a
    # lane thread is flushing. Do NOT fsync before the write or return before the fsync (a
    # checkpoint is acknowledged only when durable), do NOT swallow a failed fsync (it poisons
    # the writer and the append raises: the call the checkpoint protects must not run), do NOT
    # take `_sync_guard` while holding `_guard` (order: `_sync_guard`, then `_guard`), and do NOT
    # close a descriptor outside both locks. See decisions.md D-014.
    def append_many(
        self, events: Sequence[Mapping[str, Any]], *, checkpoint: bool = False
    ) -> list[int]:
        """Append *events* as consecutive lines in one write; return their ``seq`` numbers."""
        for event in events:
            if "seq" in event or "v" in event:
                raise EventPayloadError(
                    "the writer assigns v and seq; the caller must not"
                )
        with self._guard:
            if self._fd < 0:
                raise LogWriteError(f"log {self.path.name} is closed")
            if self._failure is not None:
                raise LogWriteError(
                    f"log {self.path.name} is poisoned by an earlier failure "
                    f"({self._failure}); reopen the log to recover"
                )
            if not events:
                return []
            first = self._seq + 1
            seqs = list(range(first, first + len(events)))
            data = b"".join(
                encode_event({**event, "v": LOG_FORMAT_VERSION, "seq": seq})
                for event, seq in zip(events, seqs, strict=True)
            )
            self._failure = "interrupted write"
            try:
                view = memoryview(data)
                while view:
                    written = _write(self._fd, view)
                    if written <= 0:
                        raise OSError(errno.EIO, "write made no progress")
                    view = view[written:]
                self._seq = seqs[-1]
                self._failure = None
            except OSError as exc:
                self._failure = f"{type(exc).__name__} {exc.strerror or exc.errno}"
                raise LogWriteError(
                    f"append to {self.path.name} failed: {self._failure}"
                ) from exc
        if checkpoint:
            self._flush()
        return seqs

    def _flush(self) -> None:
        """fsync the log file (outside ``_guard``); a failure poisons the writer and raises."""
        with self._sync_guard:
            fd = self._fd
            if fd < 0:
                raise LogWriteError(f"log {self.path.name} is closed")
            try:
                _fsync(fd)
            except BaseException as exc:
                detail = getattr(exc, "strerror", None) or getattr(exc, "errno", None)
                with self._guard:
                    self._failure = f"{type(exc).__name__} {detail}"
                if isinstance(exc, OSError):
                    raise LogWriteError(
                        f"append to {self.path.name} failed: {self._failure}"
                    ) from exc
                raise

    def append(self, event: Mapping[str, Any], *, checkpoint: bool = False) -> int:
        """Append *event* as the next line and return its ``seq``; fsync when *checkpoint*."""
        return self.append_many([event], checkpoint=checkpoint)[0]

    def close(self) -> None:
        """Release the log fd and the lock. Safe to call twice and after a failed open."""
        with self._sync_guard, self._guard:
            fd, lock_fd = self._fd, self._lock_fd
            self._fd = self._lock_fd = -1
        if fd >= 0:
            os.close(fd)
        if lock_fd >= 0:
            _release_lock(lock_fd)

    def __del__(self) -> None:
        try:
            self.close()
        except (
            Exception
        ):  # a half-built writer or interpreter shutdown: nothing to report
            pass

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
