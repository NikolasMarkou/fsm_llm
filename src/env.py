"""Explicit ``.env`` loading: the one place the library reads a dotenv file.

Importing ``fsm_llm`` reads no ``.env`` file (``llm.py`` stops litellm doing
it). A host or a CLI that wants one calls :func:`load_env` on purpose, or the
CLIs take ``--env-file PATH``, which calls it.
"""

from __future__ import annotations

import os
from pathlib import Path

import dotenv

__all__ = ["load_env"]


def load_env(
    path: str | os.PathLike[str] | None = None, *, override: bool = False
) -> Path | None:
    """Read a dotenv file into ``os.environ`` and return the file used.

    Interface contract (callers: the ``fsm-llm`` runner and, from plan d454fd62
    step 18, the other CLIs through ``cli_support.apply_env_file``; hosts):
        - *path*: a file to read. ``None`` searches for a file named ``.env``
          from the CURRENT WORKING DIRECTORY upward (never from the package
          location, so a checkout's own ``.env`` is not found from elsewhere).
        - *override*: ``False`` (default) sets only the variables the process
          does not have, so the process environment always wins over the file;
          ``True`` lets the file replace a variable that is already set. A
          variable that is set to an empty string counts as set.
        - Returns the ``Path`` read, or ``None`` when *path* is ``None`` and no
          file was found (nothing is set then).
        - Raises ``FileNotFoundError`` when an explicit *path* is not an
          existing file (a directory included), so a typo is never silent. It
          is an ``OSError``, which the CLI runner turns into one stderr line.
        - Never logs or returns a value from the file.
    """
    # DECISION plan-2026-11-12T073150-d454fd62/D-005
    # Opt-in only: nothing may call this at import time, and the search starts
    # at the working directory (find_dotenv's default starts at the CALLER's
    # file, which made the result depend on where the package is installed).
    # Do NOT add an implicit load, an env-var switch for one, or a second
    # dotenv import elsewhere in src.
    if path is None:
        found = dotenv.find_dotenv(usecwd=True)
        if not found:
            return None
        target = Path(found)
    else:
        target = Path(os.fspath(path))
        if not target.is_file():
            raise FileNotFoundError(f"No .env file at {target}")
    dotenv.load_dotenv(dotenv_path=target, override=override, encoding="utf-8")
    return target
