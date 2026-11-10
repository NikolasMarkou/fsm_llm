"""Run mypy over the `fsm_llm` package, whose source directory is `src/`.

The distribution maps the import name `fsm_llm` to the directory `src`
(`package-dir` in pyproject.toml). mypy cannot follow setuptools' editable
finder, and `mypy src/` would name every module `src.*`, so each
`from fsm_llm.x import y` would silently become `Any` under
`ignore_missing_imports`. This script exposes `src` under its import name
through a temporary symlink and checks the package by name instead.

Usage: python scripts/type_check.py [extra mypy args]
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    with tempfile.TemporaryDirectory(prefix="fsm_llm_mypy_") as tmp:
        (Path(tmp) / "fsm_llm").symlink_to(ROOT / "src", target_is_directory=True)
        env = {**os.environ, "MYPYPATH": tmp}
        cmd = [
            sys.executable,
            "-m",
            "mypy",
            "-p",
            "fsm_llm",
            "--ignore-missing-imports",
        ]
        return subprocess.call([*cmd, *argv], cwd=ROOT, env=env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
