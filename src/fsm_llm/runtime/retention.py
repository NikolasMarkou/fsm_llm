"""Retention of the event log: delete old logs on request, then the blobs nobody names.

``EventLog.prune`` calls ``prune_root``. Nothing here runs by itself and with no criterion given
nothing is read or deleted. A log's age is its file mtime, never ``ts``. Windows behavior is
unverified until the Windows run: a refused unlink (``PermissionError``, simulated through
``_unlink``) is skipped, counted, never retried.
"""

from __future__ import annotations

import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path

from fsm_llm.logging import logger
from fsm_llm.runtime.blobs import blob_path
from fsm_llm.runtime.constants import BLOB_DIR, BLOB_REF_RE, LOG_DIR
from fsm_llm.runtime.events import EventPayloadError
from fsm_llm.runtime.stores.jsonl import (
    LOG_SUFFIX,
    check_conversation_id,
    exclusive_lock,
)

# Indirection a test patches to simulate a refused unlink, read at call time.
_unlink = os.unlink

_REF_BYTES = re.compile(BLOB_REF_RE.pattern.encode("ascii"))
_SCAN_CHUNK_BYTES = 1 << 20
_REF_OVERLAP = len("sha256:") + 64 - 1  # a reference cut by a chunk edge is found whole


@dataclass(frozen=True)
class PruneReport:
    """What one prune did (or, with ``dry_run``, would do). ``logs_busy``: held logs and logs
    whose unlink was refused; ``blobs_young``: unreferenced blobs inside the grace and blobs
    whose unlink was refused; ``bytes_freed``: the size of every file removed."""

    logs_removed: int = 0
    logs_busy: int = 0
    blobs_removed: int = 0
    blobs_young: int = 0
    bytes_freed: int = 0


# DECISION plan-2026-11-18T071726-5a6a1f80/D-018
# Retention is an explicit call and its default deletes NOTHING. Do NOT add a thread or a call
# from close or release (a resumable crashed log is old by mtime), a sidecar reference count
# (a crash desyncs it: mark and sweep is stateless, logs go first, orphans next pass), an age
# from ``ts``, an unlink or a read of a ``.lock`` file (it races a new writer on POSIX and is
# refused on Windows), a delete of a log without the non-blocking writer lock, or a sweep that
# marks only the logs and not the live blobs (a blob may name blobs), ignores the grace (a
# writer puts a blob BEFORE the event naming it) or ignores a file it cannot read (fail
# closed). Empty ``<aa>`` directories stay (a writer may sit between mkdir and temp file).
def prune_root(
    root: str | os.PathLike[str],
    *,
    max_age_seconds: float | None = None,
    max_logs: int | None = None,
    blob_grace_seconds: float = 3600.0,
    dry_run: bool = False,
) -> PruneReport:
    """Prune the log root *root* (``EventLog.prune`` has the contract). ``ValueError`` for a bad
    limit, before anything is touched; with no criterion the report is empty, nothing is read."""
    _non_negative("max_age_seconds", max_age_seconds, allow_none=True)
    _non_negative("max_logs", max_logs, allow_none=True, whole=True)
    _non_negative("blob_grace_seconds", blob_grace_seconds)
    if max_age_seconds is None and max_logs is None:
        return PruneReport()
    now = time.time()
    root = Path(root)
    logs_dir = root / LOG_DIR
    picked = _candidates(logs_dir, now, max_age_seconds, max_logs)
    removed, busy, freed, gone = _prune_logs(
        logs_dir, picked, now, max_age_seconds, dry_run
    )
    live = _mark(root, logs_dir, gone)
    if live is None:
        return PruneReport(removed, busy, 0, 0, freed)
    swept, young, blob_bytes = _sweep(
        root / BLOB_DIR, live, now, blob_grace_seconds, dry_run
    )
    return PruneReport(removed, busy, swept, young, freed + blob_bytes)


def _non_negative(
    name: str, value: float | None, *, allow_none: bool = False, whole: bool = False
) -> None:
    if value is None and allow_none:
        return
    kinds = int if whole else (int, float)
    if isinstance(value, bool) or not isinstance(value, kinds) or not value >= 0:
        raise ValueError(f"{name} is a number of at least 0, got {value!r:.40}")


def _is_log_id(stem: str) -> bool:
    try:
        check_conversation_id(stem)
    except EventPayloadError:
        return False
    return True


def _aged(now: float, mtime: float, max_age: float | None) -> bool:
    return max_age is not None and now - mtime >= max_age


def _candidates(
    logs_dir: Path, now: float, max_age: float | None, max_logs: int | None
) -> list[tuple[str, bool]]:
    """The logs to try as ``(conversation id, by_rank)``: older than *max_age*, or ranked
    (newest first by mtime) beyond *max_logs*. A file that is no conversation id's log is
    never ranked and never a candidate."""
    try:
        names = os.listdir(logs_dir)
    except OSError:
        return []
    ours: list[tuple[float, str]] = []
    for name in names:
        stem = name[: -len(LOG_SUFFIX)]
        if name.endswith(LOG_SUFFIX) and _is_log_id(stem):
            try:
                info = os.stat(logs_dir / name)
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                ours.append((info.st_mtime, stem))
    ours.sort(reverse=True)
    picked = []
    for rank, (mtime, stem) in enumerate(ours):
        by_rank = max_logs is not None and rank >= max_logs
        if by_rank or _aged(now, mtime, max_age):
            picked.append((stem, by_rank))
    return picked


def _prune_logs(
    logs_dir: Path,
    candidates: list[tuple[str, bool]],
    now: float,
    max_age: float | None,
    dry_run: bool,
) -> tuple[int, int, int, set[str]]:
    removed = busy = freed = 0
    gone: set[str] = set()
    for stem, by_rank in candidates:
        path = logs_dir / (stem + LOG_SUFFIX)
        with exclusive_lock(logs_dir, stem, create=not dry_run) as held:
            if not held:
                busy += 1
                continue
            try:  # read again under the lock: a writer may have appended since the listing
                info = path.stat()
            except OSError:
                continue
            if not by_rank and not _aged(now, info.st_mtime, max_age):
                continue
            if not dry_run:
                try:
                    _unlink(path)
                except FileNotFoundError:
                    continue
                except OSError:
                    busy += 1
                    continue
            removed += 1
            freed += info.st_size
            gone.add(path.name)
    return removed, busy, freed, gone


def _refs_in(path: Path) -> set[str]:
    """Every blob reference named in the bytes of *path* (a missing file names none)."""
    found: set[str] = set()
    tail = b""
    try:
        with open(path, "rb") as handle:
            while chunk := handle.read(_SCAN_CHUNK_BYTES):
                data = tail + chunk
                found.update(m.decode("ascii") for m in _REF_BYTES.findall(data))
                tail = data[-_REF_OVERLAP:]
    except FileNotFoundError:
        pass
    return found


def _mark(root: Path, logs_dir: Path, gone: set[str]) -> set[str] | None:
    """The references named by a remaining log (a held one included), and by a live blob in
    turn. None when a file cannot be read: the caller then sweeps nothing."""
    live: set[str] = set()
    try:
        for name in os.listdir(logs_dir) if logs_dir.is_dir() else ():
            if name.endswith(LOG_SUFFIX) and name not in gone:
                live |= _refs_in(logs_dir / name)
        pending = list(live)
        while pending:
            for ref in _refs_in(blob_path(root / BLOB_DIR, pending.pop())) - live:
                live.add(ref)
                pending.append(ref)
    except OSError as exc:
        logger.warning(
            f"event log prune: blobs kept, a file was unreadable ({exc!r:.80})"
        )
        return None
    return live


def _sweep(
    blobs_dir: Path, live: set[str], now: float, grace: float, dry_run: bool
) -> tuple[int, int, int]:
    removed = young = freed = 0
    try:
        prefixes = os.listdir(blobs_dir)
    except OSError:
        return 0, 0, 0
    for prefix in prefixes:
        try:
            names = os.listdir(blobs_dir / prefix)
        except OSError:
            continue
        for name in names:
            ref = f"sha256:{prefix}{name}"
            if len(prefix) != 2 or len(name) != 62 or not BLOB_REF_RE.fullmatch(ref):
                continue  # a stray temp file or a foreign name: never ours to remove
            if ref in live:
                continue
            path = blobs_dir / prefix / name
            try:
                info = path.stat()
            except OSError:
                continue
            if now - info.st_mtime < grace:  # also a future mtime: kept
                young += 1
                continue
            if not dry_run:
                try:
                    _unlink(path)
                except FileNotFoundError:
                    continue
                except OSError:
                    young += 1
                    continue
            removed += 1
            freed += info.st_size
    return removed, young, freed
