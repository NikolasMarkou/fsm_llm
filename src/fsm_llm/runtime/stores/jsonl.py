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
a checkpoint append is followed by ``os.fsync`` (see ``append``); nothing else is.
"""

from __future__ import annotations

import errno
import os
import sys
import threading
from collections.abc import Iterator, Mapping
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


def _fsync_directory(directory: Path) -> None:
    """Make a new file's directory entry durable. POSIX only: ``os.open`` of a directory
    raises on Windows. Best effort: a file system that cannot fsync a directory is skipped."""
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
        - ``close()``: idempotent; releases the fd and the lock. The ``.lock`` file stays.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        name = self.path.name
        if not name.endswith(LOG_SUFFIX):
            raise EventPayloadError(
                f"a log file is named <id>{LOG_SUFFIX}, got {name!r:.80}"
            )
        self.conversation_id = check_conversation_id(name[: -len(LOG_SUFFIX)])
        self._guard = threading.Lock()
        self._fd = -1
        self._lock_fd = -1
        self._seq = 0
        self._failure: str | None = None
        opened = False
        try:
            self._open()
            opened = True
        except OSError as exc:
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

    def _open(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.conversation_id + LOCK_SUFFIX)
        self._lock_fd = os.open(
            lock_path, os.O_RDWR | os.O_CREAT | _binary_flag(), 0o600
        )
        if not _try_lock(self._lock_fd):
            raise LogLockedError(
                f"log {self.path.name} is held by another writer; a lock is refused, "
                "never waited for"
            )
        existed = self.path.exists()
        self._fd = os.open(
            self.path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | _binary_flag(),
            0o600,
        )
        if not existed:
            _fsync_directory(self.path.parent)
        read_fd = os.open(self.path, os.O_RDONLY | _binary_flag())
        try:
            good = 0
            for event, end in _scan(read_fd):
                self._seq = event["seq"]
                good = end
            size = os.fstat(read_fd).st_size
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
    # the two leaves a torn line). One event is one _write of the whole line, looped only
    # over a short write; a partial line may exist only as a torn tail, cut by the next
    # lock holder, so any interrupted write poisons this writer. See D-011 and D-012.
    def append(self, event: Mapping[str, Any], *, checkpoint: bool = False) -> int:
        """Append *event* as the next line and return its ``seq``; fsync when *checkpoint*."""
        if "seq" in event or "v" in event:
            raise EventPayloadError("the writer assigns v and seq; the caller must not")
        with self._guard:
            if self._fd < 0:
                raise LogWriteError(f"log {self.path.name} is closed")
            if self._failure is not None:
                raise LogWriteError(
                    f"log {self.path.name} is poisoned by an earlier failure "
                    f"({self._failure}); reopen the log to recover"
                )
            seq = self._seq + 1
            data = encode_event({**event, "v": LOG_FORMAT_VERSION, "seq": seq})
            self._failure = "interrupted write"
            try:
                view = memoryview(data)
                while view:
                    written = _write(self._fd, view)
                    if written <= 0:
                        raise OSError(errno.EIO, "write made no progress")
                    view = view[written:]
                self._seq = seq
                if checkpoint:
                    _fsync(self._fd)
                self._failure = None
            except OSError as exc:
                self._failure = f"{type(exc).__name__} {exc.strerror or exc.errno}"
                raise LogWriteError(
                    f"append to {self.path.name} failed: {self._failure}"
                ) from exc
            return seq

    def close(self) -> None:
        """Release the log fd and the lock. Safe to call twice and after a failed open."""
        with self._guard:
            fd, lock_fd = self._fd, self._lock_fd
            self._fd = self._lock_fd = -1
        if fd >= 0:
            os.close(fd)
        if lock_fd >= 0:
            _release_lock(lock_fd)

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
