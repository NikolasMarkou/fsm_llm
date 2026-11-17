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
from tests.test_fsm_llm_regression.workflow_support import passthrough_env

#: The directory of the import package (`src/fsm_llm/`); derived, so it survives a layout move.
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


def _package_of(path: Path) -> str:
    """The dotted package a file of the package belongs to (`fsm_llm.testing` for
    `testing/user.py`; a package's own `__init__.py` is that package)."""
    parts = path.resolve().relative_to(_PKG_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("fsm_llm", *parts[:-1]))


def _private_imports(source: str, package: str) -> set[tuple[str, str]]:
    """`(module, name)` of every `from <module> import _name` in *source* that
    reaches into the package `fsm_llm` from outside `package` itself.

    Interface contract: *package* is the dotted package of the file the source
    belongs to (`fsm_llm.testing`). A relative import (`level` above 0) is
    resolved against it, so `from ..llm import _x` is `("fsm_llm.llm", "_x")`;
    one that climbs above `fsm_llm` is ignored. Dunder names are not private,
    names imported from inside *package* are its own business, and a plain
    `import x` binds no name of its own. Pure, never raises on a parseable source.
    """
    found: set[tuple[str, str]] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            base = package.split(".")
            if node.level - 1 >= len(base):
                continue
            base = base[: len(base) - (node.level - 1)]
            module = ".".join([*base, *([node.module] if node.module else [])])
        else:
            module = node.module or ""
        if module.split(".")[0] != "fsm_llm":
            continue
        if module == package or module.startswith(package + "."):
            continue
        found |= {
            (module, alias.name)
            for alias in node.names
            if alias.name.startswith("_") and not alias.name.startswith("__")
        }
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
            **passthrough_env(),  # Windows: `import asyncio` needs SYSTEMROOT
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
        assert {"conversation.py", "message_pipeline.py", "__init__.py"} <= names

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

    def test_the_private_imports_of_testing_are_exactly_the_recorded_debt(
        self,
    ) -> None:
        """`doubles.py` reuses the usage meter of `fsm_llm.llm.usage` (`_Metered`,
        `_meter_of`) so `ScriptedLLM.usage()` is the real meter, not a copy. That
        is recorded debt (review gate 1 concern 12, plan d454fd62 step 58.1). Step 73
        split `llm.py` and retargeted the two names to their defining module,
        `fsm_llm.llm.usage`; it did not give the seam a public name. This pins it: no
        third private name may be imported from the package into `testing/`,
        absolute or relative (re-review concern 9)."""
        found = set()
        for path in sorted(_TESTING_DIR.rglob("*.py")):
            package = _package_of(path)
            found |= _private_imports(path.read_text(encoding="utf-8"), package)

        assert found == {
            ("fsm_llm.llm.usage", "_Metered"),
            ("fsm_llm.llm.usage", "_meter_of"),
        }

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("from fsm_llm.llm import _x", {("fsm_llm.llm", "_x")}),
            ("from ..llm import _x", {("fsm_llm.llm", "_x")}),
            ("from ..llm import _x, y, __z__", {("fsm_llm.llm", "_x")}),
            ("from .. import _x", {("fsm_llm", "_x")}),
            ("from ...fsm_llm.llm import _x", set()),  # beyond the top: not ours
            ("from . import llm", set()),
            ("from .llm import _x", set()),  # inside `testing` itself
            ("from .doubles import _x", set()),
            ("from fsm_llm.testing.doubles import _x", set()),
            ("from ..agents.base import _x", {("fsm_llm.agents.base", "_x")}),
            ("from json import _x", set()),
            ("import fsm_llm.llm", set()),
        ],
    )
    def test_the_scan_resolves_relative_and_absolute_private_imports(
        self, source: str, expected: set[tuple[str, str]]
    ) -> None:
        assert _private_imports(source, "fsm_llm.testing") == expected

    def test_a_relative_private_import_added_to_testing_is_caught(self) -> None:
        """The scratch mutant of the re-review: `from ..llm import _UsageMeter`
        appended to `testing/doubles.py` adds a third private name."""
        doubles = _TESTING_DIR / "doubles.py"
        mutant = (
            doubles.read_text(encoding="utf-8") + "\nfrom ..llm import _UsageMeter\n"
        )
        found = _private_imports(mutant, _package_of(doubles))
        assert ("fsm_llm.llm", "_UsageMeter") in found

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
            if (
                isinstance(n, ast.Assign)
                and "__all__" in [getattr(t, "id", None) for t in n.targets]
            )
            or (
                isinstance(n, ast.AnnAssign)
                and getattr(n.target, "id", None) == "__all__"
            )
        ]
        assert len(assigns) == 1, "exactly one static __all__"
        assert isinstance(assigns[0].value, (ast.List, ast.Tuple))
