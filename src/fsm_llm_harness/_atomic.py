"""The one atomic-write primitive shared by ``storage.py`` and ``tools.py``.

This module exists solely to break an import cycle: ``storage.py`` imports
``PlanMemory`` from ``tools.py`` (for confinement + ownership), and
``tools.py`` needs the same atomic-write primitive ``storage.py`` already
defined for ``PlanDirectory``. Importing ``storage`` from ``tools`` would
close a cycle (``tools -> storage -> tools``), so the primitive lives here,
one level below both.

Do NOT add anything else to this module beyond the one primitive and its
direct support. It is intentionally a leaf: it must never import from
``storage`` or ``tools``, or the cycle this module exists to break comes back.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fsm_llm.logging import logger

from .exceptions import HarnessArtifactError

__all__ = ["atomic_write_text"]


def atomic_write_text(target: Path, content: str, *, artifact: str) -> Path:
    """Write *content* to *target* atomically, or raise and change nothing.

    Interface contract (call sites: :func:`storage._atomic_write_text`,
    re-exported for ``PlanDirectory.write_text``/``append_text``, and
    :class:`tools.PlanMemory`'s own ``write_text``/``append_text``):
        - ``target`` must already be an AUTHORISED absolute path -- this
          function performs no confinement or ownership check of its own.
        - On success the file's content is exactly ``content``. On failure the
          file is untouched: a reader concurrent with either outcome sees the
          old bytes or the new bytes, never a truncated blend.
        - Leaves no temp file behind on either path.
        - Raises :class:`HarnessArtifactError` (tagged ``artifact``) for any
          ``OSError`` -- a full disk, a read-only mount, a vanished parent.
    """
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-019
    # The temp file MUST be created in `target.parent`, not in the system temp
    # directory. `os.replace` is only atomic within one filesystem; across a
    # mount boundary it degrades to copy-then-unlink, which reintroduces exactly
    # the torn-write window this function exists to close -- and on many systems
    # `/tmp` is a different filesystem (tmpfs) from a repository checkout.
    # Do NOT "tidy" the `dir=` argument away, and do NOT reach for
    # `tempfile.NamedTemporaryFile()` without it.
    # The `finally` shape is `FileSessionStore.save`'s (session.py:151-173),
    # copied deliberately: an `except OSError: raise` shape leaks the temp file
    # on every non-OSError exit, and the existence check is what makes the
    # cleanup a no-op after a successful `os.replace` consumed the temp name.
    # See decisions.md D-019.
    directory = target.parent
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle_fd, tmp_name = tempfile.mkstemp(
            dir=str(directory), prefix=f".{target.name}.", suffix=".tmp"
        )
    except OSError as exc:
        raise HarnessArtifactError(
            artifact, f"could not open a temp file beside '{target}'", cause=exc
        ) from exc
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, str(target))
    except OSError as exc:
        raise HarnessArtifactError(
            artifact, f"could not be written to '{target}'", cause=exc
        ) from exc
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:  # pragma: no cover - cleanup is best-effort
                logger.debug(f"could not remove temp file {tmp_name}")
    logger.debug(f"atomically wrote {target} ({len(content)} chars)")
    return target
