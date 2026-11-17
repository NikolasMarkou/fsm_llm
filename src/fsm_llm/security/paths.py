"""Path confinement: `resolve_within`, the one shared containment check."""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

# --------------------------------------------------------------
# Path Confinement
# --------------------------------------------------------------

#: ``OSError.winerror`` values of a Windows symlink loop (no ``ELOOP`` there).
_WIN_LOOP_ERRORS = (1920, 1921)


def resolve_within(root: Path, relative: str | os.PathLike[str]) -> Path | None:
    """Resolve *relative* under *root* and return it only if it stays inside.

    This is the shared comparison used by the OKF bundle loader and writer,
    the Agent Skills reads and the monitor's preset ids; other sites keep
    their own checks. Never re-inline it in any of them.

    Interface contract:
        - ``root``: a directory path that is ALREADY resolved (absolute, no
          symlinks). An unresolved root fails closed: the resolved candidate
          can then never equal it or have it as a parent, so ``None`` comes
          back for every input.
        - ``relative``: a ``str`` or path-like, joined under ``root``. An
          absolute value replaces ``root`` in the join and is therefore
          refused unless it already lands inside ``root``.
        - Returns the fully resolved absolute path when it equals ``root`` or
          lies below it, else ``None``. The target need not exist.
        - Performs no read, write, create or delete of the target.
        - Failure mode: ``ValueError`` for an embedded NUL byte in *relative*
          (raised here on every platform: Windows with Python 3.13 or later no
          longer raises it from ``resolve()``), ``RuntimeError`` or ``OSError``
          for a symlink loop (on Windows, ``OSError`` with ``winerror`` 1920 or
          1921 counts as one). Other ``Path.resolve()`` errors propagate
          unchanged. The caller decides how to refuse.

    The order is load-bearing: RESOLVE FIRST, COMPARE SECOND, and compare path
    COMPONENTS. A lexical check before resolving (``startswith``, a ``..``
    scan, ``os.path.normpath``) passes a symlink created inside the root that
    points outside it, and ``/tmp/ws-evil`` is a string-prefix of ``/tmp/ws``.
    Only ``Path.resolve()`` collapses links, and only ``resolved == root or
    root in resolved.parents`` is correct for siblings. This is the reasoning
    recorded as D-032 of plan 191b2eb2.

    Residual hole (TOCTOU): a symlink swapped in between this call and the
    caller's open still wins. Closing that needs ``openat2`` / ``O_NOFOLLOW``;
    this guard stops path arguments chosen by a model or found in a bundle, not
    a hostile local process.
    """
    if "\0" in str(relative):
        raise ValueError("embedded null byte")
    resolved = (root / relative).resolve()
    if sys.version_info >= (3, 13):
        # Python 3.13 made the non-strict resolve() above stop quietly at a
        # symlink loop and return the looped path; 3.11 and 3.12 raise
        # RuntimeError. A strict resolve of the result raises ELOOP again, so the
        # failure mode in the contract holds on every version. A missing target
        # is fine and stays allowed. Windows has no ELOOP: its loop is
        # ERROR_CANT_ACCESS_FILE (1920) or ERROR_CANT_RESOLVE_FILENAME (1921),
        # and the non-strict resolve() swallows both.
        try:
            resolved.resolve(strict=True)
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            if exc.errno == errno.ELOOP or winerror in _WIN_LOOP_ERRORS:
                raise
    if resolved == root or root in resolved.parents:
        return resolved
    return None
