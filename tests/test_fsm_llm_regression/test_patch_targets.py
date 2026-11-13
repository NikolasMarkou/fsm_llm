"""No test patches a name through a package that merely re-exports it.

A package `__init__` that re-exports a name binds it a second time: patching
`pkg.NAME` replaces that binding only, while the code that performs the lookup
(`pkg.sub.NAME`) keeps the real object, so the patch silently does nothing (a
patched `llm.completion` that is no-op reaches the network). The rule of the
layout plan (D-014, P1): a patch string aims at the module whose code performs
the lookup, never at a package `__init__` that imports the name.

The scanner reads the tests as text and the package as files, imports nothing:

- a patch string is the first string argument of `patch(...)`, `mock.patch(...)`,
  `patch.multiple(...)`, `patch.dict(...)`, or the first argument of
  `<x>.setattr(...)` / `<x>.delattr(...)` when it is a dotted string (the
  `monkeypatch.setattr("a.b.c", value)` and `monkeypatch.delattr("a.b.c")` forms;
  the `(obj, "name", value)` form has an object first and is not a string target);
- the target `a.b.c.d` is cut at its longest prefix that names a module of the
  package (`modules`, as `{dotted: path}` like `support.source_modules`); the
  next segment is the attribute;
- a problem is reported when that module is a package `__init__.py` and the
  attribute is bound there by an `import` or `from ... import` statement and
  not defined there (no `def`, `class`, or assignment of that name).

f-string and computed targets are not read (none exist today; a new form would
need its own rule). A name an `__init__` defines itself is a legitimate target.

`scan_patch_targets(root, modules)` is a pure function over a directory of test
files, so the unit tests below drive it with scratch trees.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from tests.test_fsm_llm_regression.support import REPO, source_modules

_PATCH_ATTRS = {"patch", "multiple", "dict"}
_SETATTR = {"setattr", "delattr"}


def _is_patch_call(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id == "patch"
    if isinstance(func, ast.Attribute):
        if func.attr == "patch":
            return True
        # patch.multiple("mod", ...), patch.dict("mod.NAME", ...)
        return (
            func.attr in _PATCH_ATTRS
            and isinstance(func.value, (ast.Name, ast.Attribute))
            and getattr(func.value, "id", getattr(func.value, "attr", None)) == "patch"
        )
    return False


def patch_strings(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """Every `(line, dotted string)` that a call uses as a patch target."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        first = node.args[0] if node.args else None
        if first is None:
            for kw in node.keywords:
                if kw.arg == "target" and _is_patch_call(node.func):
                    first = kw.value
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        if _is_patch_call(node.func):
            yield first.lineno, first.value
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in _SETATTR
            and "." in first.value
            and len(node.args) >= (2 if node.func.attr == "setattr" else 1)
        ):
            yield first.lineno, first.value


def init_bindings(path: Path) -> tuple[set[str], set[str]]:
    """`(imported, defined)` top-level names of a package `__init__` (also under `if`/`try`)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    defined: set[str] = set()

    def visit(body: list[ast.stmt]) -> None:
        for stmt in body:
            if isinstance(stmt, ast.Import):
                imported.update((a.asname or a.name.split(".")[0]) for a in stmt.names)
            elif isinstance(stmt, ast.ImportFrom):
                imported.update((a.asname or a.name) for a in stmt.names)
            elif isinstance(
                stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                defined.add(stmt.name)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                )
                for target in targets:
                    defined.update(
                        n.id for n in ast.walk(target) if isinstance(n, ast.Name)
                    )
            elif isinstance(stmt, (ast.If, ast.Try, ast.With)):
                visit(stmt.body)
                visit(stmt.orelse if not isinstance(stmt, ast.With) else [])
                if isinstance(stmt, ast.Try):
                    visit(stmt.finalbody)
                    for handler in stmt.handlers:
                        visit(handler.body)

    visit(tree.body)
    return imported, defined


def split_target(target: str, modules: Mapping[str, Path]) -> tuple[str, str] | None:
    """`(module, attribute)` for the longest module prefix of `target`, else `None`."""
    parts = target.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        name = ".".join(parts[:cut])
        if name in modules:
            return name, parts[cut]
    return None


def scan_patch_targets(root: Path, modules: Mapping[str, Path]) -> list[str]:
    """Problems for every patch string under `root` that aims at a re-exported name."""
    cache: dict[str, tuple[set[str], set[str]]] = {}
    bad: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for line, target in patch_strings(tree):
            hit = split_target(target, modules)
            if hit is None:
                continue
            module, attr = hit
            if modules[module].name != "__init__.py":
                continue
            if module not in cache:
                cache[module] = init_bindings(modules[module])
            imported, defined = cache[module]
            if attr in imported and attr not in defined:
                bad.append(
                    f"{path.relative_to(root)}:{line}: `{target}` aims at the package "
                    f"`{module}`, which only re-exports `{attr}`: patch the module "
                    f"that performs the lookup"
                )
    return bad


def test_no_patch_string_aims_at_a_package_reexport() -> None:
    problems = scan_patch_targets(REPO / "tests", source_modules())
    assert problems == [], "\n" + "\n".join(problems)


def _tree(root: Path, files: dict[str, str]) -> dict[str, Path]:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    pkg = root / "pkg"
    return {
        "pkg": pkg / "__init__.py",
        "pkg.mod": pkg / "mod.py",
        "pkg.sub": pkg / "sub" / "__init__.py",
        "pkg.sub.deep": pkg / "sub" / "deep.py",
    }


PKG = {
    "pkg/__init__.py": "from pkg.mod import X\nfrom .mod import Z as Zed\nY = 1\ndef f(): ...\n",
    "pkg/mod.py": "X = 1\nZ = 2\n",
    "pkg/sub/__init__.py": (
        "try:\n    from pkg.sub.deep import D\nexcept ImportError:\n    pass\n"
        "try:\n    from pkg.sub.deep import E\nexcept ImportError:\n    E = None\n"
    ),
    "pkg/sub/deep.py": "D = 1\n",
}


class TestScanPatchTargets:
    """Mutation proofs of the guard: a synthetic patch string aimed at a re-export."""

    def _scan(self, tmp_path: Path, test_source: str) -> list[str]:
        modules = _tree(tmp_path / "src", PKG)
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_x.py").write_text(test_source, encoding="utf-8")
        return scan_patch_targets(tests, modules)

    @pytest.mark.parametrize(
        "source",
        [
            'from unittest.mock import patch\n\ndef t():\n    with patch("pkg.X"):\n        pass\n',
            'import unittest.mock as m\n\ndef t():\n    m.patch("pkg.Zed")\n',
            'def t(monkeypatch):\n    monkeypatch.setattr("pkg.X", 1)\n',
            'def t(monkeypatch):\n    monkeypatch.delattr("pkg.X")\n',
            'from unittest.mock import patch\n\n@patch(\n    "pkg.X.attr",\n    new=1,\n)\ndef t(): ...\n',
            'from unittest.mock import patch\n\nwith patch(target="pkg.X"):\n    pass\n',
            'from unittest.mock import patch\n\nwith patch.dict("pkg.X", {"a": 1}):\n    pass\n',
        ],
    )
    def test_a_patch_aimed_at_a_reexport_fails(
        self, tmp_path: Path, source: str
    ) -> None:
        problems = self._scan(tmp_path, source)
        assert len(problems) == 1 and "only re-exports" in problems[0], problems

    def test_the_lookup_module_is_clean(self, tmp_path: Path) -> None:
        source = (
            'from unittest.mock import patch\n\nwith patch("pkg.mod.X"):\n    pass\n'
        )
        assert self._scan(tmp_path, source) == []

    def test_a_name_the_init_defines_itself_is_clean(self, tmp_path: Path) -> None:
        source = (
            'from unittest.mock import patch\n\nwith patch("pkg.Y"), patch("pkg.f"):\n'
            "    pass\n"
        )
        assert self._scan(tmp_path, source) == []

    def test_a_reexport_inside_a_try_block_is_found(self, tmp_path: Path) -> None:
        source = (
            'from unittest.mock import patch\n\nwith patch("pkg.sub.D"):\n    pass\n'
        )
        assert len(self._scan(tmp_path, source)) == 1

    def test_a_name_imported_and_also_assigned_in_the_init_counts_as_defined(
        self, tmp_path: Path
    ) -> None:
        source = (
            'from unittest.mock import patch\n\nwith patch("pkg.sub.E"):\n    pass\n'
        )
        assert self._scan(tmp_path, source) == []

    def test_an_object_patch_and_the_three_argument_setattr_are_not_strings(
        self, tmp_path: Path
    ) -> None:
        source = (
            "from unittest.mock import patch\n\n"
            'def t(monkeypatch, obj):\n    patch.object(obj, "X")\n'
            '    monkeypatch.setattr(obj, "X", 1)\n'
            '    monkeypatch.setenv("pkg.X", "1")\n'
        )
        assert self._scan(tmp_path, source) == []

    def test_a_target_outside_the_package_is_ignored(self, tmp_path: Path) -> None:
        source = (
            'from unittest.mock import patch\n\nwith patch("os.getcwd"):\n    pass\n'
        )
        assert self._scan(tmp_path, source) == []

    def test_the_line_and_file_are_reported(self, tmp_path: Path) -> None:
        source = 'from unittest.mock import patch\n\nwith patch("pkg.X"):\n    pass\n'
        assert self._scan(tmp_path, source)[0].startswith("test_x.py:3: `pkg.X`")
