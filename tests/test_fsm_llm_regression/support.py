"""Shared by the layout safety nets: where the package lives and its modules.

Imported by the regression tests that read the source tree (`test_module_path_map.py`,
`test_module_size_guard.py`, `test_patch_targets.py`, `test_decision_index.py`,
`test_decision_index_gate.py`, `test_decision_anchor_multiset.py`,
`test_decision_scan_scope.py`, `test_docs_layout_paths.py`), by
`tests/test_examples/test_example_import_paths.py` and by `tests/test_packaging.py`. Not
collected (no `test_` prefix).

Interface:
- `REPO`: the repository root.
- `package_dir(repo=REPO) -> Path`: the directory of the `fsm_llm` package, always
  `<repo>/src/fsm_llm`. It does not look at the file system: the package has sat in the
  standard `src/` layout since the 0.3 move (plan d454fd62, step 62), so there is no
  second layout to detect. A tree that has no such directory gives a path that does not
  exist, not another directory.
- `load_script_module(source, name, script) -> types.ModuleType`: run the *source* text of a
  `scripts/*.py` file as a module called *name* (registered in `sys.modules` while it runs and
  removed again when it raises), with `__file__` set to *script*; used to load a script as it
  is on disk and, with an edited text, a mutant of it. Not an import: the script is never put
  on `sys.path`.
- `source_modules(repo=REPO) -> dict[str, Path]`: every `.py` file of the package
  as `{dotted module: path}`, in sorted order; a package `__init__.py` is named
  by its package (`fsm_llm.agents`) and the root `__init__.py` by `fsm_llm`. Reads the
  file system only, imports nothing. `__pycache__` and dot directories are
  skipped. Never raises for a missing package: returns `{}`.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = "fsm_llm"


def package_dir(repo: Path = REPO) -> Path:
    return repo / "src" / PACKAGE


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


def load_script_module(source: str, name: str, script: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_loader(name, loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(script)
    sys.modules[name] = module
    try:
        exec(compile(source, str(script), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
