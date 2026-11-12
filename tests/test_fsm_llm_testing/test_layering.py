"""`fsm_llm.testing` is a leaf of the package: wired in, never loaded by the library.

The module is imported by name (`from fsm_llm.testing import ...`), like
`fsm_llm.okf`: `import fsm_llm` never loads it, no library module imports it,
and it imports the core only, never `pytest`, `litellm` or another subpackage.
The checks read the source with `ast` and run one child process per import
probe, so a test session that already holds the module cannot hide a leak.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import fsm_llm

#: The directory of the import package (`src/` today); derived, so it survives a layout move.
_PKG_ROOT = Path(fsm_llm.__file__).resolve().parent
_TESTING_DIR = _PKG_ROOT / "testing"
_OTHER_SUBPACKAGES = (
    "agents",
    "reasoning",
    "workflows",
    "monitor",
    "eval",
    "synthesis",
)
_FORBIDDEN_THIRD_PARTY = ("pytest", "litellm")


def _imported_modules(tree: ast.AST) -> list[tuple[str, int, int]]:
    """`(module, level, line)` of every import; `module` is dotted, `level` the dots.

    `from . import x` yields `("x", 1, line)` and `from fsm_llm import x`
    yields `("fsm_llm.x", 0, line)`, so a bare package import names the module
    it binds.
    """
    found: list[tuple[str, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, 0, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if base == "" or base == "fsm_llm":
                found += [
                    (f"{base}.{a.name}".strip("."), node.level, node.lineno)
                    for a in node.names
                ]
            else:
                found.append((base, node.level, node.lineno))
    return found


def _names_testing(module: str, level: int) -> bool:
    """True when an import target is `fsm_llm.testing` (or inside it)."""
    parts = module.split(".")
    if level == 0:
        return parts[:2] == ["fsm_llm", "testing"] or parts == ["testing"]
    return parts[0] == "testing"


def _library_files() -> list[Path]:
    return sorted(
        p
        for p in _PKG_ROOT.rglob("*.py")
        if _TESTING_DIR not in p.parents and "__pycache__" not in p.parts
    )


def _run_probe(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=_PKG_ROOT.parent,
        env={
            "PATH": "",
            "OLLAMA_API_BASE": "http://127.0.0.1:9",
            "OLLAMA_HOST": "http://127.0.0.1:9",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        },
    )


class TestImportProbes:
    def test_importing_testing_loads_no_other_subpackage(self) -> None:
        subs = repr(_OTHER_SUBPACKAGES)
        probe = (
            "import sys\n"
            "import fsm_llm\n"
            "assert 'fsm_llm.testing' not in sys.modules, 'core imports testing'\n"
            "import fsm_llm.testing\n"
            f"bad = [m for m in sys.modules if m.split('.')[:2] in "
            f"[['fsm_llm', s] for s in {subs}]]\n"
            "assert not bad, bad\n"
            "assert 'pytest' not in sys.modules, 'testing imports pytest'\n"
            "print('ok')\n"
        )
        result = _run_probe(probe)

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"

    def test_import_fsm_llm_does_not_load_testing(self) -> None:
        result = _run_probe(
            "import sys, fsm_llm\n"
            "assert 'fsm_llm.testing' not in sys.modules\n"
            "assert 'testing' not in fsm_llm.__all__\n"
            "print('ok')\n"
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"


class TestSourceLayering:
    def test_the_library_scan_sees_files(self) -> None:
        # Guards the derivation: an empty list would make the scan below vacuous.
        names = {p.name for p in _library_files()}
        assert {"api.py", "pipeline.py", "__init__.py"} <= names

    def test_no_library_module_imports_testing(self) -> None:
        offenders = [
            f"{path.relative_to(_PKG_ROOT)}:{line}"
            for path in _library_files()
            for module, level, line in _imported_modules(
                ast.parse(path.read_text(encoding="utf-8"))
            )
            if _names_testing(module, level)
        ]

        assert offenders == []

    def test_the_package_init_does_not_mention_testing_as_an_import(self) -> None:
        tree = ast.parse((_PKG_ROOT / "__init__.py").read_text(encoding="utf-8"))
        hits = [
            line
            for module, level, line in _imported_modules(tree)
            if _names_testing(module, level)
        ]

        assert hits == []

    @pytest.mark.parametrize("third_party", _FORBIDDEN_THIRD_PARTY)
    def test_testing_imports_no_pytest_and_no_litellm(self, third_party: str) -> None:
        offenders = [
            f"{path.relative_to(_PKG_ROOT)}:{line}"
            for path in sorted(_TESTING_DIR.rglob("*.py"))
            for module, _level, line in _imported_modules(
                ast.parse(path.read_text(encoding="utf-8"))
            )
            if module.split(".")[0] == third_party
        ]

        assert offenders == []

    def test_testing_imports_no_other_subpackage(self) -> None:
        offenders = []
        for path in sorted(_TESTING_DIR.rglob("*.py")):
            for module, level, line in _imported_modules(
                ast.parse(path.read_text(encoding="utf-8"))
            ):
                parts = module.split(".")
                if level == 0:
                    sub = parts[1] if parts[0] == "fsm_llm" and len(parts) > 1 else ""
                else:
                    # `..agents` from `testing/x.py` reaches a sibling subpackage.
                    sub = parts[0] if level >= 2 else ""
                if sub in _OTHER_SUBPACKAGES:
                    offenders.append(f"{path.relative_to(_PKG_ROOT)}:{line}")

        assert offenders == []


class TestSkeleton:
    def test_the_module_has_a_docstring_and_a_static_all(self) -> None:
        tree = ast.parse((_TESTING_DIR / "__init__.py").read_text(encoding="utf-8"))
        assert ast.get_docstring(tree)
        assigns = [
            n
            for n in tree.body
            if isinstance(n, (ast.Assign, ast.AnnAssign))
            and "__all__"
            in [getattr(t, "id", None) for t in getattr(n, "targets", [n.target])]
        ]
        assert len(assigns) == 1, "exactly one static __all__"
        assert isinstance(assigns[0].value, (ast.List, ast.Tuple))
