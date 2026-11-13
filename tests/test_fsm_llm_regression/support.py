"""Shared by the layout safety nets: where the package lives and its modules.

Used by `test_module_path_map.py`, `test_module_size_guard.py` and
`tests/test_examples/test_example_import_paths.py`. Not collected (no `test_`
prefix).

Interface:
- `REPO`: the repository root.
- `package_dir(repo=REPO) -> Path`: the directory of the `fsm_llm` package, which
  is `src/fsm_llm/` once the package sits in a standard layout and `src/` itself
  while it is flat (the import package `fsm_llm` is the directory `src/`). Decided
  by `__init__.py`, never by a leftover `__pycache__` directory, so every net
  holds before and after that move.
- `source_modules(repo=REPO) -> dict[str, Path]`: every `.py` file of the package
  as `{dotted module: path}`, in sorted order; a package `__init__.py` is named
  by its package (`fsm_llm.agents`) and the root `__init__.py` by `fsm_llm`. Reads the
  file system only, imports nothing. `__pycache__` and dot directories are
  skipped. Never raises for a missing package: returns `{}`.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = "fsm_llm"


def package_dir(repo: Path = REPO) -> Path:
    nested = repo / "src" / PACKAGE
    return nested if (nested / "__init__.py").is_file() else repo / "src"


def source_modules(repo: Path = REPO) -> dict[str, Path]:
    root = package_dir(repo)
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(p == "__pycache__" or p.startswith(".") for p in rel.parts):
            continue
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        found[".".join([PACKAGE, *parts])] = path
    return found
