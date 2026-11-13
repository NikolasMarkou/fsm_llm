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
          variable the process holds as an empty or whitespace-only string
          counts as NOT set, the rule ``llm.resolve_model`` applies to
          ``LLM_MODEL`` (a blank value carries nothing): the file fills it
          when the file gives it a non-blank value (a ``${VAR}`` in that value
          resolves as for an unset variable) and leaves it as it is otherwise.
          To keep a variable deliberately empty (``HTTP_PROXY=`` to switch a
          proxy off for one run), do not name it in the file: the file cannot
          tell a deliberate empty value from an unset one.
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
    # DECISION plan-2026-11-12T073150-d454fd62/D-024
    # A blank process variable does not block the file. python-dotenv's
    # ``override=False`` keeps an empty value, so ``LLM_MODEL=""`` in the shell
    # plus ``--env-file`` naming a model failed with "No LLM model is set"
    # while ``resolve_model`` treats that blank as unset (review gate 1,
    # concern 8). Do NOT special-case ``LLM_MODEL`` (one rule for every
    # variable), and do NOT drop a blank variable the file gives no value.
    # Re-review of the fixes, concern 1: the blank variables the file names are
    # removed from the environment BEFORE ``load_dotenv`` runs, so a ``${VAR}``
    # in the file resolves with the process values first, exactly as for an
    # unset variable. Do NOT fill them from a separate ``dotenv_values`` read:
    # that read resolves ``${VAR}`` with the file's own values first and gave
    # a blanked variable a value the process had overridden.
    held_blank: dict[str, str] = {}
    if not override:
        for name in dotenv.dotenv_values(target, encoding="utf-8"):
            held = os.environ.get(name)
            if held is not None and not held.strip():
                held_blank[name] = held
                del os.environ[name]
    try:
        dotenv.load_dotenv(dotenv_path=target, override=override, encoding="utf-8")
    finally:
        # Only a non-blank file value replaces a blank variable: a blank,
        # valueless or blank-resolving entry leaves the variable as it was.
        for name, held in held_blank.items():
            filled = os.environ.get(name)
            if filled is None or not filled.strip():
                os.environ[name] = held
    return target
