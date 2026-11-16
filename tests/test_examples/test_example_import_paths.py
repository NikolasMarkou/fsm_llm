"""Every `fsm_llm` import in `examples/` resolves (decision D-003c of the layout plan).

`examples/` is never edited by the layout steps (65 to 94): the eval baselines
and `test_example_calls_bind.py` rest on them. So whichever module a split moves,
every import path an example uses must keep resolving at that same path: the real
module or a package `__init__` that defines or re-exports the name (the packages
of `fsm_llm` re-export through a static `__all__`, not a shim). This test reads
every `.py` under `examples/` with `ast` (nothing is executed, nothing edited) and
checks each statement:

- `import fsm_llm.x.y [as z]`: the module imports;
- `from fsm_llm.x.y import name`: the module imports and `name` is an attribute
  of it or a submodule of it.

How a name is decided: the package is imported for real, because that also
catches a name a package builds at import time. When an import fails only for a
third-party module that is not installed (`ModuleNotFoundError` whose `name` is
outside `fsm_llm`, such as `mcp` or `opentelemetry`), the statement is resolved
statically from the file system instead: the module path must exist as a file or
a package, and the name must be bound at the top level of that file (a `def`,
`class`, assignment, or an import alias, the way a re-export is written) or be a
submodule file or folder. In the project virtualenv every fsm_llm module imports
(the optional extras are imported lazily or guarded), so the static path is the
fallback and has its own unit tests; the count of statements resolved that way
is asserted to be what the environment gives, never a skip.

Relative imports and `import *` do not occur in `examples/` and are refused so
that a new one is noticed rather than ignored. The scanner takes a directory, so
the mutation tests run it over scratch copies and never touch `examples/`.
"""

from __future__ import annotations

import ast
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from tests.test_fsm_llm_regression.support import PACKAGE, REPO, package_dir

EXAMPLES = REPO / "examples"


@dataclass(frozen=True)
class FsmLlmImport:
    file: str
    line: int
    module: str
    name: str | None  # None for `import fsm_llm.x`


def scan_imports(root: Path = EXAMPLES) -> tuple[list[FsmLlmImport], list[str]]:
    """Every fsm_llm import under `root` (all `.py`, any depth) and the unsupported ones."""
    found: list[FsmLlmImport] = []
    refused: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found += [
                    FsmLlmImport(rel, node.lineno, a.name, None)
                    for a in node.names
                    if a.name == PACKAGE or a.name.startswith(f"{PACKAGE}.")
                ]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    refused.append(f"{rel}:{node.lineno}: relative import")
                elif node.module and (
                    node.module == PACKAGE or node.module.startswith(f"{PACKAGE}.")
                ):
                    for a in node.names:
                        if a.name == "*":
                            refused.append(
                                f"{rel}:{node.lineno}: `from {node.module} import *`"
                            )
                        else:
                            found.append(
                                FsmLlmImport(rel, node.lineno, node.module, a.name)
                            )
    return found, refused


class _ThirdPartyMissing(Exception):
    """An import failed only because an optional third-party package is not installed."""


def _import(importer: Callable[[str], ModuleType], name: str) -> ModuleType | None:
    """The module, `None` when `name` is a missing fsm_llm path; third-party gaps raise."""
    try:
        return importer(name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == PACKAGE or missing.startswith(f"{PACKAGE}."):
            return None
        raise _ThirdPartyMissing(missing) from exc


def _module_file(module: str, repo: Path) -> Path | None:
    parts = module.split(".")[1:]
    base = package_dir(repo).joinpath(*parts)
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _top_level_names(tree: ast.Module) -> set[str]:
    """Names bound at module level, also inside `if`, `try`, `with` and `for` blocks."""
    names: set[str] = set()

    def visit(body: list[ast.stmt]) -> None:
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(stmt.name)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    names.update(
                        n.id for n in ast.walk(target) if isinstance(n, ast.Name)
                    )
            elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)) and isinstance(
                stmt.target, ast.Name
            ):
                names.add(stmt.target.id)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in stmt.names)
            else:
                for field in ("body", "orelse", "finalbody"):
                    visit(getattr(stmt, field, []))
                for handler in getattr(stmt, "handlers", []):
                    visit(handler.body)

    visit(tree.body)
    return names


def static_problem(imp: FsmLlmImport, repo: Path = REPO) -> str | None:
    """Resolve one import from the files alone; a readable problem, or `None`."""
    path = _module_file(imp.module, repo)
    if path is None:
        return f"module `{imp.module}` is not a file or package of {PACKAGE}"
    if imp.name is None:
        return None
    if imp.name in _top_level_names(ast.parse(path.read_text(encoding="utf-8"))):
        return None
    if path.name == "__init__.py" and _module_file(f"{imp.module}.{imp.name}", repo):
        return None
    return f"`{imp.name}` is neither defined, re-exported nor a submodule in `{imp.module}`"


def dynamic_problem(
    imp: FsmLlmImport, importer: Callable[[str], ModuleType]
) -> str | None:
    module = _import(importer, imp.module)
    if module is None:
        return f"module `{imp.module}` does not import"
    if imp.name is None or hasattr(module, imp.name):
        return None
    if _import(importer, f"{imp.module}.{imp.name}") is not None:
        return None
    return f"`{imp.name}` is not an attribute or a submodule of `{imp.module}`"


def import_problems(
    root: Path = EXAMPLES,
    *,
    importer: Callable[[str], ModuleType] = importlib.import_module,
    repo: Path = REPO,
) -> tuple[list[str], int]:
    """Problems of every import under `root`, and how many were decided statically."""
    imports, refused = scan_imports(root)
    problems = [f"{r} (not supported by this test)" for r in refused]
    static = 0
    for imp in imports:
        try:
            problem = dynamic_problem(imp, importer)
        except _ThirdPartyMissing:
            static += 1
            problem = static_problem(imp, repo)
        if problem:
            problems.append(f"{imp.file}:{imp.line}: {problem}")
    return problems, static


class TestTheRealExamples:
    def test_every_fsm_llm_import_resolves(self) -> None:
        problems, _ = import_problems()
        assert problems == [], "\n" + "\n".join(problems)

    def test_the_project_environment_needs_no_static_fallback(self) -> None:
        # `fsm_llm` imports lazily: no optional extra is needed to import any path
        # the examples use. A non-zero count means an example now needs one.
        _, static = import_problems()
        assert static == 0

    def test_the_scan_is_not_vacuous(self) -> None:
        imports, refused = scan_imports()
        files = {i.file for i in imports}
        assert refused == []
        assert len(list(EXAMPLES.rglob("*.py"))) > 100 and len(files) > 80
        assert len(imports) > 200
        assert {
            ("fsm_llm.handlers", "HandlerTiming"),
            ("fsm_llm.memory", "WorkingMemory"),
        } <= {(i.module, i.name) for i in imports}

    def test_the_static_resolver_agrees_with_the_real_import_on_every_example(
        self,
    ) -> None:
        imports, _ = scan_imports()
        disagree = [
            f"{i.file}:{i.line} {i.module}.{i.name}: {static_problem(i)}"
            for i in imports
            if static_problem(i) is not None
        ]
        assert disagree == []


def _write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestMutations:
    """The mutation proofs: bogus imports in scratch copies, `examples/` untouched."""

    GOOD = "import fsm_llm.handlers\nfrom fsm_llm.handlers import HandlerTiming\nfrom fsm_llm import agents\n"

    def _copy(self, tmp_path: Path) -> Path:
        copy = tmp_path / "examples"
        for path in EXAMPLES.rglob("*.py"):
            _write(
                copy,
                path.relative_to(EXAMPLES).as_posix(),
                path.read_text(encoding="utf-8"),
            )
        return copy

    def test_a_copy_of_the_real_examples_is_clean(self, tmp_path: Path) -> None:
        assert import_problems(self._copy(tmp_path))[0] == []

    @pytest.mark.parametrize(
        "line, needle",
        [
            (
                "from fsm_llm.handlers import NoSuchName\n",
                "`NoSuchName` is not an attribute",
            ),
            (
                "from fsm_llm.no_such_module import X\n",
                "module `fsm_llm.no_such_module` does not import",
            ),
            (
                "import fsm_llm.agents.no_such\n",
                "module `fsm_llm.agents.no_such` does not import",
            ),
            (
                "from fsm_llm.agents.tools import NoSuchTool\n",
                "`NoSuchTool` is not an attribute",
            ),
        ],
    )
    def test_a_bogus_import_in_a_copy_fails(
        self, tmp_path: Path, line: str, needle: str
    ) -> None:
        copy = self._copy(tmp_path)
        _write(copy, "basic/zz_new/run.py", line)
        problems, _ = import_problems(copy)
        assert (
            len(problems) == 1
            and "basic/zz_new/run.py:1" in problems[0]
            and needle in problems[0]
        ), problems

    def test_the_good_shapes_resolve(self, tmp_path: Path) -> None:
        _write(tmp_path, "a/run.py", self.GOOD)
        assert import_problems(tmp_path) == ([], 0)

    def test_relative_and_star_imports_are_noticed(self, tmp_path: Path) -> None:
        _write(tmp_path, "a/run.py", "from fsm_llm.handlers import *\n")
        _write(tmp_path, "b/run.py", "from . import sibling\nimport os\n")
        problems, _ = import_problems(tmp_path)
        assert [p.split(" (")[0] for p in problems] == [
            "a/run.py:1: `from fsm_llm.handlers import *`",
            "b/run.py:1: relative import",
        ]

    def test_an_import_inside_a_function_is_read(self, tmp_path: Path) -> None:
        _write(
            tmp_path, "a/run.py", "def f():\n    from fsm_llm import NoSuchTopLevel\n"
        )
        problems, _ = import_problems(tmp_path)
        assert len(problems) == 1 and "a/run.py:2" in problems[0]

    def test_non_fsm_llm_imports_are_ignored(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "a/run.py",
            "import os\nfrom pathlib import Path\nimport fsm_llm_other\nfrom fsm_llm_other import x\n",
        )
        assert scan_imports(tmp_path) == ([], [])


class TestStaticFallback:
    """What the fallback decides when a third-party extra is missing."""

    def _repo(self, tmp_path: Path) -> Path:
        pkg = tmp_path / "src" / "fsm_llm"
        _write(pkg, "__init__.py", "from .a import exported\n")
        _write(
            pkg,
            "a.py",
            "def exported(): ...\nclass Klass: ...\nCONST = 1\nannotated: int = 2\nif True:\n    guarded = 1\n",
        )
        _write(pkg, "sub/__init__.py", "from .inner import thing as shown\n")
        _write(pkg, "sub/inner.py", "def thing(): ...\n")
        return tmp_path

    @pytest.mark.parametrize(
        "module, name, ok",
        [
            ("fsm_llm.a", "exported", True),
            ("fsm_llm.a", "Klass", True),
            ("fsm_llm.a", "CONST", True),
            ("fsm_llm.a", "annotated", True),
            ("fsm_llm.a", "guarded", True),
            ("fsm_llm", "exported", True),
            ("fsm_llm.sub", "shown", True),
            ("fsm_llm.sub", "inner", True),
            ("fsm_llm.sub.inner", "thing", True),
            ("fsm_llm.sub", "missing", False),
            ("fsm_llm.a", "inner", False),
            ("fsm_llm.gone", "x", False),
            ("fsm_llm.sub.gone", "x", False),
        ],
    )
    def test_the_file_system_decides(
        self, tmp_path: Path, module: str, name: str, ok: bool
    ) -> None:
        repo = self._repo(tmp_path)
        assert (
            static_problem(FsmLlmImport("f.py", 1, module, name), repo) is None
        ) is ok

    def test_a_module_import_with_no_name_needs_only_the_module(
        self, tmp_path: Path
    ) -> None:
        repo = self._repo(tmp_path)
        assert (
            static_problem(FsmLlmImport("f.py", 1, "fsm_llm.sub.inner", None), repo)
            is None
        )
        assert (
            static_problem(FsmLlmImport("f.py", 1, "fsm_llm.nope", None), repo)
            is not None
        )

    def test_a_missing_third_party_package_falls_back_and_counts(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path / "ex", "a/run.py", "from fsm_llm.a import Klass, Missing\n")
        repo = self._repo(tmp_path)

        def importer(name: str) -> ModuleType:
            raise ModuleNotFoundError("No module named 'mcp'", name="mcp")

        problems, static = import_problems(
            tmp_path / "ex", importer=importer, repo=repo
        )
        assert static == 2
        assert problems == [
            "a/run.py:1: `Missing` is neither defined, re-exported nor a submodule in `fsm_llm.a`"
        ]

    def test_a_missing_fsm_llm_module_is_a_problem_not_a_fallback(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path / "ex", "a/run.py", "from fsm_llm.a import Klass\n")

        def importer(name: str) -> ModuleType:
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)

        problems, static = import_problems(
            tmp_path / "ex", importer=importer, repo=self._repo(tmp_path)
        )
        assert static == 0 and problems == [
            "a/run.py:1: module `fsm_llm.a` does not import"
        ]
