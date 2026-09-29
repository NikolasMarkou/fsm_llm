"""
Append-only result records, JSON output, and run-directory naming.

``append_row``/``read_rows`` and the bodies of ``write_json``, ``utc_now`` and
``git_commit`` were moved from ``scripts/harness_bench.py`` (which delegates
here); ``git_short_hash``, ``model_slug`` and the run-directory name keep the
historical ``scripts/eval.py`` rules so new run directories sort and read
like the old ones.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import (
    GIT_UNKNOWN,
    MAX_RUN_DIR_SUFFIX,
    RUN_DIR_TIMESTAMP_FORMAT,
    UTC_TIMESTAMP_FORMAT,
)
from .exceptions import EvalError

PathLike = str | Path


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append ONE jsonl row and flush -- rows survive an aborted block."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read every non-blank jsonl row; a missing file yields ``[]``."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as JSON: indent 2, sorted keys, trailing newline."""
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime(UTC_TIMESTAMP_FORMAT)


def git_commit(cwd: PathLike | None = None) -> str:
    """Full ``HEAD`` hash of the repo at ``cwd``.

    Raises ``subprocess.CalledProcessError`` (not a repo) or ``OSError`` (no
    ``git``): bench manifests must not record a commit they cannot prove.
    """
    cmd = ("git", "rev-parse", "HEAD")
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def git_short_hash(cwd: PathLike | None = None) -> str:
    """Short ``HEAD`` hash of the repo at ``cwd``, or ``"unknown"`` on any failure."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=None if cwd is None else str(cwd),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # git missing, not a repo, detached oddities: metadata only
        return GIT_UNKNOWN
    return out or GIT_UNKNOWN


def model_slug(model: str) -> str:
    """Filesystem-safe model label: last ``/`` segment with ``:`` turned into ``-``."""
    return model.split("/")[-1].replace(":", "-")


def make_run_dir(
    root: PathLike,
    model: str,
    cwd: PathLike | None = None,
    now: datetime | None = None,
) -> Path:
    """Create and return a fresh run directory under ``root``.

    Interface contract (callers: the examples runner and the conversation-case
    runner):
        - Name: ``<YYYY-MM-DD_HH-MM>_<git-short-hash of cwd>_<model_slug>``
          (local time, ``now`` defaults to the current time). When that name
          exists, ``_2``, ``_3``, ... is appended; an existing directory is
          never reused or overwritten (created with ``exist_ok=False``).
        - Missing parents of ``root`` are created.
        - Raises ``EvalError`` when the directory cannot be created (unwritable
          root, or ``MAX_RUN_DIR_SUFFIX`` names already taken).
    """
    stamp = (now or datetime.now()).strftime(RUN_DIR_TIMESTAMP_FORMAT)
    base = f"{stamp}_{git_short_hash(cwd)}_{model_slug(model)}"
    root_path = Path(root)
    for index in range(1, MAX_RUN_DIR_SUFFIX + 1):
        candidate = root_path / (base if index == 1 else f"{base}_{index}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        except OSError as exc:
            raise EvalError(
                f"cannot create run directory {candidate}: {exc}",
                details={"path": str(candidate)},
            ) from exc
        return candidate
    raise EvalError(
        f"no free run directory name for {base!r} under {root_path}",
        details={"root": str(root_path), "base": base},
    )
