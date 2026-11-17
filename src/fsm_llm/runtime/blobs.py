"""The blob store of the event log: content-addressed files, written once, checked on read.

A blob is bytes named by their SHA-256: ``<root>/blobs/<aa>/<62 hex>`` and the reference
``sha256:<64 hex>`` that an event carries (``BLOB_REF_RE``). A large or sensitive-shaped payload
goes here and the event keeps only the reference, so an event line stays under
``MAX_EVENT_BYTES``. The name is the hash, so a write is idempotent: the same bytes from two
threads or two processes land on the same file, and the loser of a race has nothing to retry.
A read re-hashes, so a torn, tampered or substituted file is refused, never returned.

``BlobStore`` is the one class, ``blob_ref`` the one hash rule. Imports ``runtime.constants``,
``runtime.events``, ``runtime.stores.jsonl`` (its directory fsync) and the standard library.

Unverified until the Windows run: ``os.replace`` onto a target another process holds open
(``PermissionError`` there, handled below and simulated in the tests), antivirus holding a
fresh file, and path length on a runner (the relative path is 71 characters). Power-loss
durability is argued, not tested: a new blob is fsynced and so is its directory (POSIX).
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from fsm_llm.runtime.constants import BLOB_DIR, BLOB_REF_RE, MAX_BLOB_BYTES
from fsm_llm.runtime.events import (
    EventPayloadError,
    LogCorruptError,
    LogWriteError,
    canonical_json,
)
from fsm_llm.runtime.stores.jsonl import _fsync_directory

_REF_PREFIX = "sha256:"

# Indirection a test patches to simulate a lost race or a failing disk, read at call time.
_replace = os.replace


def blob_ref(data: bytes) -> str:
    """Return the content address of *data*: ``sha256:`` and 64 lowercase hex digits."""
    return _REF_PREFIX + hashlib.sha256(data).hexdigest()


class BlobStore:
    """Content-addressed blobs under ``<root>/blobs``.

    Interface contract (callers: ``EventLog`` in step 5, the engine through it later):
        - ``BlobStore(root)``: *root* is the log root (the directory holding ``logs/`` and
          ``blobs/``). Does no I/O; nothing is created until the first ``put``.
        - ``put(data: bytes) -> ref``: stores *data* and returns its reference; the same
          bytes always give the same reference and one file. ``EventPayloadError`` for a
          non-bytes value or one over ``MAX_BLOB_BYTES`` (nothing written), ``LogWriteError``
          for an ``OSError`` (no temp file is left behind).
        - ``put_json(value) -> ref``: ``put`` of ``canonical_json(value)``, so the same value
          gives the same reference whatever its key order; ``EventPayloadError`` for a value
          that is too deep (``MAX_PAYLOAD_DEPTH``), cyclic or not JSON.
        - ``get(ref) -> bytes``: the bytes, after re-hashing them. ``LogCorruptError`` when
          the blob is missing, over the cap, or does not hash to its name.
        - ``exists(ref) -> bool`` and ``path_for(ref) -> Path``: a reference is checked
          against ``BLOB_REF_RE`` (``fullmatch``) before any path is built; a malformed one
          raises ``EventPayloadError`` and touches no disk.
        - Thread and process safe without a lock: see ``put``. A stray ``*.tmp`` (a writer
          killed between the temp file and the replace) is ignored by every method.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self._dir = self.root / BLOB_DIR

    def path_for(self, ref: str) -> Path:
        """Return the file path of *ref* (``blobs/<aa>/<62 hex>``) without touching the disk."""
        if not isinstance(ref, str) or not BLOB_REF_RE.fullmatch(ref):
            raise EventPayloadError(f"not a blob reference: {ref!r:.80}")
        digest = ref[len(_REF_PREFIX) :]
        return self._dir / digest[:2] / digest[2:]

    def exists(self, ref: str) -> bool:
        """Return True when a file is stored under *ref* (its bytes are not checked)."""
        return self.path_for(ref).is_file()

    def put_json(self, value: Any) -> str:
        """Store the canonical JSON of *value* and return its reference."""
        return self.put(canonical_json(value, what="blob payload"))

    # DECISION plan-2026-11-17T100529-21e6f00d/D-011
    # This is a LOCAL write-if-absent writer: do NOT replace it with, or share it as, an
    # atomic-save helper (four anchors forbid a shared helper without a maintainer decision,
    # and this is not a replace: the target may legitimately exist). Do NOT rewrite a target
    # that already has the right size, retry in a loop (the loser of a content-addressed
    # race has nothing to retry), treat a PermissionError or FileExistsError as success
    # without checking that the final file now hashes right, open a path in text mode or
    # with a bare os.open (no O_BINARY: "\n" would become "\r\n" on Windows and fail the
    # re-hash), or write the final path directly (a reader would see a half file). The write
    # is a temp file in the SAME directory, fsync, then os.replace. See D-011 and D-012.
    def put(self, data: bytes) -> str:
        """Store *data* if absent and return its reference (see the class contract)."""
        if not isinstance(data, bytes):
            raise EventPayloadError(f"a blob is bytes, got {type(data).__name__}")
        if len(data) > MAX_BLOB_BYTES:
            raise EventPayloadError(
                f"blob is {len(data)} bytes, over the {MAX_BLOB_BYTES} byte cap"
            )
        ref = blob_ref(data)
        path = self.path_for(ref)
        if _has_size(path, len(data)):
            return ref  # present: the name is the hash, so the content is the caller's again
        try:
            self._write_new(path, data, ref)
        except OSError as exc:
            raise LogWriteError(
                f"cannot store blob {ref[:19]}: {type(exc).__name__} "
                f"{exc.strerror or exc.errno}"
            ) from exc
        return ref

    def _write_new(self, path: Path, data: bytes, ref: str) -> None:
        fresh = not path.parent.exists()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if fresh:
            _fsync_directory(self._dir)
        fd, temp = tempfile.mkstemp(dir=path.parent, prefix="blob-", suffix=".tmp")
        try:
            with os.fdopen(
                fd, "wb"
            ) as handle:  # mkstemp opens with O_BINARY on Windows
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                _replace(temp, path)
            except (PermissionError, FileExistsError):
                # Windows refuses to replace a target another process holds open, and a
                # racing writer may have created it first: success only if it is the blob.
                if not _is_blob(path, ref):
                    raise
            _fsync_directory(path.parent)
        finally:
            try:
                os.unlink(temp)  # gone after a successful replace; ours alone otherwise
            except OSError:
                pass

    def get(self, ref: str) -> bytes:
        """Return the bytes stored under *ref*, re-hashed (see the class contract)."""
        path = self.path_for(ref)
        try:
            data = _read(path)
        except FileNotFoundError:
            raise LogCorruptError(f"blob {ref[:19]} is missing") from None
        if blob_ref(data) != ref:
            raise LogCorruptError(f"blob {ref[:19]} does not hash to its name")
        return data


def _has_size(path: Path, size: int) -> bool:
    """True when *path* is a regular file of exactly *size* bytes."""
    try:
        info = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_size == size


def _read(path: Path) -> bytes:
    """Read a blob file in binary, at most ``MAX_BLOB_BYTES``; ``LogCorruptError`` over it."""
    with open(path, "rb") as handle:
        data = handle.read(MAX_BLOB_BYTES + 1)
    if len(data) > MAX_BLOB_BYTES:
        raise LogCorruptError(f"blob file is over the {MAX_BLOB_BYTES} byte cap")
    return data


def _is_blob(path: Path, ref: str) -> bool:
    """True when *path* holds bytes that hash to *ref* (a missing or foreign file: False)."""
    try:
        return blob_ref(_read(path)) == ref
    except (OSError, LogCorruptError):
        return False
