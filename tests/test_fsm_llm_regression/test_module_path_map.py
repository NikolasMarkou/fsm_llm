"""The old-to-new module path map stays true while the layout changes.

`tests/fixtures/module_path_map.json` is the one record of where every module
of the `fsm_llm` package lives and where it moved from. Shape:

    {"moved":   {"<old dotted module>": "<new dotted module>"
                                       | {"<symbol>": "<new dotted module>", ...}},
     "unmoved": ["<dotted module>", ...]}

`moved` names a module that left its path (a string: the whole module went
to one place; a table: each public symbol of a split module went to the module
that defines it). `unmoved` is every other module, packages (`__init__`)
included. A table key may stay on disk as the PACKAGE that replaced the module
(P1: its `__init__` is the one import surface): the table entry is then that
package's decision and it is NOT listed in `unmoved`. A table key may also stay
on disk as a PLAIN module that keeps its other definitions (a partial move, step
85: `fsm_llm.agents.tools` kept `register_agent` and gave the registry to
`fsm_llm.tools`): the entry lists only what left, it is that module's decision
too (NOT listed in `unmoved`), and no listed symbol may still be defined there. The map is written for the
layout of steps 65 to 94 (each of those steps adds its entries in the commit
that does the move).

What the tests pin, in the project's no-shim rule (a moved path is gone, never
re-exported):

- every old path of `moved` is not importable (a table entry may leave the old
  path importable as a package, the P3 `__init__` re-export surface, and then
  each symbol must be DEFINED in its new module: `__module__` says where; or as
  a plain module that keeps other definitions, and then no listed symbol is
  still DEFINED in its file: `still_defined` reads the top-level statements);
- every new path imports, and every symbol resolves in it;
- every module found by walking the package is a value of `moved` or listed in
  `unmoved` (a new module without a decision fails), no module is in both, no
  key of `moved` equals its value, no `unmoved` entry names a missing module;
- the file is deterministic JSON (sorted keys, indent 2, `\\n`, UTF-8) and
  `unmoved` is sorted without duplicates.

The check that `docs/upgrade-guide/v0.13.0/module-paths.md` names every old
key is owned by step 118 (it needs that file); it is not part of this module.

The scanners are pure functions over data and a module set, so the unit tests
below drive them with synthetic maps and scratch directories. To refresh
`unmoved` from the tree after a split (keeping `moved` as written):

    .venv/bin/python -m tests.test_fsm_llm_regression.test_module_path_map --write
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.test_fsm_llm_regression.support import (
    PACKAGE,
    REPO,
    package_dir,
    source_modules,
)

FIXTURE = REPO / "tests" / "fixtures" / "module_path_map.json"


def render(data: dict[str, Any]) -> str:
    """The one byte form of the map file: sorted keys, indent 2, final newline."""
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def load(path: Path = FIXTURE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def new_paths(data: dict[str, Any]) -> set[str]:
    """Every dotted module a `moved` entry names as a destination."""
    out: set[str] = set()
    for target in data.get("moved", {}).values():
        out.update(target.values() if isinstance(target, dict) else [target])
    return out


def shape_problems(data: Any) -> list[str]:
    """Problems with the structure of the map itself; `[]` for a sound one."""
    if not isinstance(data, dict) or set(data) != {"moved", "unmoved"}:
        return ["top level must be an object with exactly `moved` and `unmoved`"]
    moved, unmoved = data["moved"], data["unmoved"]
    if not isinstance(moved, dict):
        return ["`moved` must be an object"]
    if not isinstance(unmoved, list) or not all(isinstance(m, str) for m in unmoved):
        return ["`unmoved` must be a list of strings"]
    bad: list[str] = []
    for old, target in moved.items():
        if not old.startswith(PACKAGE):
            bad.append(f"`{old}` is not a dotted module of {PACKAGE}")
        if isinstance(target, str):
            targets = [target]
        elif isinstance(target, dict) and target:
            targets = list(target.values())
            bad += [
                f"`{old}`: `{s}` is not a symbol name"
                for s in target
                if not s.isidentifier()
            ]
        else:
            bad.append(f"`{old}` must map to a module or a non-empty symbol table")
            continue
        for new in targets:
            if not isinstance(new, str) or not new.startswith(PACKAGE):
                bad.append(f"`{old}` -> `{new}`: not a dotted module of {PACKAGE}")
            elif new == old:
                bad.append(f"`{old}` maps to itself: that is not a move")
    if unmoved != sorted(unmoved):
        bad.append("`unmoved` is not sorted")
    if len(set(unmoved)) != len(unmoved):
        bad.append("`unmoved` holds a duplicate")
    both = sorted((set(moved) | new_paths(data)) & set(unmoved))
    bad += [f"`{m}` is in `unmoved` and also named by `moved`" for m in both]
    return bad


def decided_modules(data: dict[str, Any]) -> set[str]:
    """Modules the map decides: `unmoved`, every `moved` target, and every table key
    (a split module whose path survives as the package that replaced it)."""
    tables = {old for old, target in data["moved"].items() if isinstance(target, dict)}
    return set(data["unmoved"]) | new_paths(data) | tables


def coverage_problems(data: dict[str, Any], modules: set[str]) -> list[str]:
    """Every module on disk has a decision; nothing named is missing from disk."""
    decided = decided_modules(data)
    bad = [
        f"`{m}` has no decision: add it to `unmoved` or as a `moved` target"
        for m in sorted(modules - decided)
    ]
    bad += [
        f"`{m}` is in `unmoved` but is not a module on disk"
        for m in sorted(set(data["unmoved"]) - modules)
    ]
    bad += [
        f"`{m}` is a `moved` target but is not a module on disk"
        for m in sorted(new_paths(data) - modules)
    ]
    return bad


def still_defined(source: Path, symbols: Any) -> list[str]:
    """The listed symbols that the file at *source* still defines at top level.

    A definition is a `def`, an `async def`, a `class` or an assignment at module
    level; a name that is only imported (a binding for a pydantic annotation, say)
    is not one. Reads the text, never imports the module.
    """
    defined: set[str] = set()
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                defined.update(
                    n.id for n in ast.walk(target) if isinstance(n, ast.Name)
                )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
    return sorted(set(symbols) & defined)


def _spec(name: str) -> Any:
    try:
        return importlib.util.find_spec(name)
    except ModuleNotFoundError:  # a missing parent counts as missing
        return None


def import_problems(data: dict[str, Any]) -> list[str]:
    """No shim at an old path, and every new path and symbol really resolves."""
    bad: list[str] = []
    for old, target in data["moved"].items():
        spec = _spec(old)
        if isinstance(target, str):
            if spec is not None:
                bad.append(f"old path `{old}` is still importable (no shim allowed)")
            targets: dict[str | None, str] = {None: target}
        else:
            if spec is not None and spec.submodule_search_locations is None:
                origin = spec.origin
                for symbol in (
                    still_defined(Path(origin), target) if origin else sorted(target)
                ):
                    bad.append(
                        f"`{symbol}` moved from `{old}` but is still defined there"
                    )
            targets = dict(target)
        for symbol, new in targets.items():
            try:
                module = importlib.import_module(new)
            except ImportError as exc:
                bad.append(f"new path `{new}` for `{old}` does not import: {exc}")
                continue
            if symbol is None:
                continue
            if not hasattr(module, symbol):
                bad.append(f"`{new}` has no `{symbol}` (moved from `{old}`)")
                continue
            obj = getattr(module, symbol)
            home = getattr(obj, "__module__", None)
            if (inspect.isclass(obj) or inspect.isfunction(obj)) and home != new:
                bad.append(
                    f"`{symbol}` in `{new}` is defined in `{home}`: a re-export is not a move"
                )
    return bad


def _refresh(path: Path = FIXTURE) -> None:
    data = load(path) if path.exists() else {"moved": {}}
    decided = decided_modules({"moved": data["moved"], "unmoved": []})
    data["unmoved"] = sorted(set(source_modules()) - decided)
    path.write_text(render(data), encoding="utf-8", newline="\n")


class TestTheRealMap:
    def test_the_file_is_deterministic_json(self) -> None:
        text = FIXTURE.read_bytes().decode("utf-8")
        assert text == render(json.loads(text)), (
            "rewrite with sorted keys, indent 2, final newline"
        )

    def test_the_shape_is_sound(self) -> None:
        problems = shape_problems(load())
        assert problems == [], "\n" + "\n".join(problems)

    def test_every_module_has_exactly_one_decision(self) -> None:
        problems = coverage_problems(load(), set(source_modules()))
        assert problems == [], "\n" + "\n".join(problems)

    def test_every_move_is_clean(self) -> None:
        problems = import_problems(load())
        assert problems == [], "\n" + "\n".join(problems)

    def test_the_walk_finds_the_package(self) -> None:
        modules = source_modules()
        assert {
            "fsm_llm",
            "fsm_llm.engine.api.conversation",
            "fsm_llm.agents",
            "fsm_llm.agents.base",
        } <= set(modules)
        assert all(p.suffix == ".py" for p in modules.values())


def _data(
    moved: dict[str, Any] | None = None, unmoved: list[str] | None = None
) -> dict[str, Any]:
    return {"moved": moved or {}, "unmoved": unmoved or []}


class TestShapeProblems:
    @pytest.mark.parametrize(
        "data, needle",
        [
            ([], "exactly"),
            ({"moved": {}}, "exactly"),
            (_data({"fsm_llm.a": "fsm_llm.a"}), "maps to itself"),
            (_data({"fsm_llm.a": {"X": "fsm_llm.a"}}), "maps to itself"),
            (
                _data({"fsm_llm.a": "fsm_llm.b"}, ["fsm_llm.a"]),
                "unmoved` and also named",
            ),
            (
                _data({"fsm_llm.a": "fsm_llm.b"}, ["fsm_llm.b"]),
                "unmoved` and also named",
            ),
            (_data(unmoved=["fsm_llm.b", "fsm_llm.a"]), "not sorted"),
            (_data(unmoved=["fsm_llm.a", "fsm_llm.a"]), "duplicate"),
            (_data({"fsm_llm.a": {}}), "non-empty symbol table"),
            (_data({"fsm_llm.a": {"not a name": "fsm_llm.b"}}), "not a symbol name"),
            (_data({"other.a": "fsm_llm.b"}), "not a dotted module"),
            (_data({"fsm_llm.a": "other.b"}), "not a dotted module"),
        ],
    )
    def test_a_bad_map_is_refused(self, data: Any, needle: str) -> None:
        assert any(needle in p for p in shape_problems(data)), shape_problems(data)

    def test_a_sound_map_is_accepted(self) -> None:
        data = _data(
            {"fsm_llm.a": "fsm_llm.x.a", "fsm_llm.b": {"B": "fsm_llm.x.b"}},
            ["fsm_llm.c"],
        )
        assert shape_problems(data) == []


def _scratch(root: Path, files: list[str]) -> Path:
    base = root / "src" / PACKAGE
    for name in files:
        target = base / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("X = 1\n", encoding="utf-8")
    return root


FILES = ["__init__.py", "a.py", "pkg/__init__.py", "pkg/b.py"]
MODULES = {"fsm_llm", "fsm_llm.a", "fsm_llm.pkg", "fsm_llm.pkg.b"}


class TestScratchTrees:
    """The mutation proofs of the coverage net, on scratch directories."""

    def test_the_package_directory_is_src_fsm_llm_and_nothing_else(
        self, tmp_path: Path
    ) -> None:
        """`package_dir` has no second layout: the real tree, a scratch tree, an empty
        directory and a tree left in the old flat `src/` layout all give `src/fsm_llm`
        (the flat tree has no modules there: it is not read as a package)."""
        assert package_dir() == REPO / "src" / PACKAGE
        assert package_dir().is_dir()
        repo = _scratch(tmp_path / "scratch", FILES)
        assert package_dir(repo) == repo / "src" / PACKAGE
        assert set(source_modules(repo)) == MODULES
        assert package_dir(tmp_path / "empty") == tmp_path / "empty" / "src" / PACKAGE
        assert source_modules(tmp_path / "empty") == {}
        flat = tmp_path / "flat"
        (flat / "src").mkdir(parents=True)
        (flat / "src" / "__init__.py").write_text("X = 1\n", encoding="utf-8")
        (flat / "src" / "a.py").write_text("X = 1\n", encoding="utf-8")
        assert package_dir(flat) == flat / "src" / PACKAGE
        assert source_modules(flat) == {}

    def test_a_leftover_pycache_directory_adds_no_module(self, tmp_path: Path) -> None:
        repo = _scratch(tmp_path, FILES)
        (repo / "src" / PACKAGE / "__pycache__").mkdir(parents=True)
        (repo / "src" / PACKAGE / "__pycache__" / "x.cpython-311.pyc.py").write_text(
            "", encoding="utf-8"
        )
        assert set(source_modules(repo)) == MODULES

    def test_a_decided_tree_is_clean(self, tmp_path: Path) -> None:
        repo = _scratch(tmp_path, FILES)
        assert (
            coverage_problems(_data(unmoved=sorted(MODULES)), set(source_modules(repo)))
            == []
        )

    def test_a_stray_module_without_a_decision_fails(self, tmp_path: Path) -> None:
        repo = _scratch(tmp_path, [*FILES, "stray.py"])
        problems = coverage_problems(
            _data(unmoved=sorted(MODULES)), set(source_modules(repo))
        )
        assert problems == [
            "`fsm_llm.stray` has no decision: add it to `unmoved` or as a `moved` target"
        ]

    def test_a_renamed_module_fails_twice(self, tmp_path: Path) -> None:
        repo = _scratch(
            tmp_path, ["__init__.py", "renamed.py", "pkg/__init__.py", "pkg/b.py"]
        )
        problems = coverage_problems(
            _data(unmoved=sorted(MODULES)), set(source_modules(repo))
        )
        assert any("`fsm_llm.renamed` has no decision" in p for p in problems)
        assert any(
            "`fsm_llm.a` is in `unmoved` but is not a module on disk" in p
            for p in problems
        )

    def test_a_moved_entry_covers_its_new_path(self, tmp_path: Path) -> None:
        repo = _scratch(tmp_path, ["__init__.py", "x/a.py", "x/__init__.py"])
        data = _data({"fsm_llm.a": "fsm_llm.x.a"}, ["fsm_llm", "fsm_llm.x"])
        assert coverage_problems(data, set(source_modules(repo))) == []

    def test_a_table_key_that_became_a_package_is_decided_by_its_table(
        self, tmp_path: Path
    ) -> None:
        repo = _scratch(tmp_path, ["__init__.py", "m/__init__.py", "m/a.py"])
        data = _data({"fsm_llm.m": {"X": "fsm_llm.m.a"}}, ["fsm_llm"])
        assert shape_problems(data) == []
        assert coverage_problems(data, set(source_modules(repo))) == []

    def test_a_string_key_that_is_still_on_disk_has_no_decision(
        self, tmp_path: Path
    ) -> None:
        repo = _scratch(tmp_path, ["__init__.py", "m/__init__.py", "m/a.py"])
        data = _data({"fsm_llm.m": "fsm_llm.m.a"}, ["fsm_llm"])
        assert coverage_problems(data, set(source_modules(repo))) == [
            "`fsm_llm.m` has no decision: add it to `unmoved` or as a `moved` target"
        ]

    def test_a_moved_target_that_does_not_exist_fails(self, tmp_path: Path) -> None:
        repo = _scratch(tmp_path, ["__init__.py"])
        data = _data({"fsm_llm.a": "fsm_llm.x.a"}, ["fsm_llm"])
        assert coverage_problems(data, set(source_modules(repo))) == [
            "`fsm_llm.x.a` is a `moved` target but is not a module on disk"
        ]


class TestImportProblems:
    """The no-shim and resolves-for-real checks, over synthetic moves of today's tree."""

    def test_an_old_path_that_still_imports_is_a_shim(self) -> None:
        bad = import_problems(
            _data({"fsm_llm.engine.api.conversation": "fsm_llm.engine.fsm.manager"})
        )
        assert bad == [
            "old path `fsm_llm.engine.api.conversation` is still importable (no shim allowed)"
        ]

    def test_an_old_path_that_is_gone_and_a_new_one_that_imports_is_clean(self) -> None:
        assert (
            import_problems(
                _data({"fsm_llm.gone_module": "fsm_llm.engine.fsm.manager"})
            )
            == []
        )

    def test_a_new_path_that_does_not_import_fails(self) -> None:
        bad = import_problems(_data({"fsm_llm.gone_module": "fsm_llm.not_there"}))
        assert (
            len(bad) == 1
            and "`fsm_llm.not_there` for `fsm_llm.gone_module` does not import"
            in bad[0]
        )

    def test_a_symbol_must_exist_in_the_new_module(self) -> None:
        bad = import_problems(
            _data({"fsm_llm.gone": {"NoSuchName": "fsm_llm.engine.fsm.manager"}})
        )
        assert bad == [
            "`fsm_llm.engine.fsm.manager` has no `NoSuchName` (moved from `fsm_llm.gone`)"
        ]

    def test_a_reexport_is_not_a_move(self) -> None:
        bad = import_problems(_data({"fsm_llm.gone": {"API": "fsm_llm"}}))
        assert (
            len(bad) == 1
            and "`API` in `fsm_llm` is defined in `fsm_llm.engine.api.conversation`"
            in bad[0]
        )

    def test_a_symbol_defined_in_the_new_module_is_clean(self) -> None:
        assert (
            import_problems(
                _data({"fsm_llm.gone": {"API": "fsm_llm.engine.api.conversation"}})
            )
            == []
        )

    def test_a_split_module_may_stay_importable_only_as_a_package(self) -> None:
        ok = import_problems(
            _data({"fsm_llm.agents": {"BaseAgent": "fsm_llm.agents.base.agent"}})
        )
        assert ok == []
        bad = import_problems(
            _data(
                {
                    "fsm_llm.engine.api.conversation": {
                        "API": "fsm_llm.engine.api.conversation"
                    }
                }
            )
        )
        assert bad == [
            "`API` moved from `fsm_llm.engine.api.conversation` but is still defined there"
        ]

    def test_a_plain_module_that_kept_other_definitions_may_stay(self) -> None:
        assert (
            import_problems(
                _data(
                    {"fsm_llm.agents.tools": {"ToolRegistry": "fsm_llm.tools.registry"}}
                )
            )
            == []
        )

    def test_a_package_whose_parent_is_missing_counts_as_gone(self) -> None:
        assert (
            import_problems(
                _data({"fsm_llm.nope.deeper": "fsm_llm.engine.fsm.manager"})
            )
            == []
        )


class TestStillDefined:
    """`still_defined`: what a plain module that gave symbols away may not keep."""

    SOURCE = (
        "import os\n"
        "from elsewhere import Imported\n"
        "class Klass: ...\n"
        "def func(): ...\n"
        "async def coro(): ...\n"
        "CONST = 1\n"
        "ann: int = 2\n"
        "left, right = 1, 2\n"
        "if True:\n"
        "    nested = 3\n"
    )

    def test_each_kind_of_top_level_definition_is_found(self, tmp_path: Path) -> None:
        path = tmp_path / "m.py"
        path.write_text(self.SOURCE, encoding="utf-8")
        names = ["Klass", "func", "coro", "CONST", "ann", "left", "right"]
        assert still_defined(path, names) == sorted(names)

    def test_an_import_or_a_nested_name_is_not_a_definition(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "m.py"
        path.write_text(self.SOURCE, encoding="utf-8")
        assert still_defined(path, ["os", "Imported", "nested", "absent"]) == []

    def test_only_the_listed_symbols_are_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "m.py"
        path.write_text(self.SOURCE, encoding="utf-8")
        assert still_defined(path, ["func", "Imported"]) == ["func"]


class TestRefresh:
    def test_refresh_keeps_moved_and_lists_the_rest(self, tmp_path: Path) -> None:
        path = tmp_path / "map.json"
        path.write_text(
            render(_data({"fsm_llm.zz_old": "fsm_llm.engine.api.conversation"})),
            encoding="utf-8",
        )
        _refresh(path)
        data = load(path)
        assert data["moved"] == {"fsm_llm.zz_old": "fsm_llm.engine.api.conversation"}
        assert (
            "fsm_llm.engine.api.conversation" not in data["unmoved"]
            and "fsm_llm.engine.fsm.manager" in data["unmoved"]
        )
        assert path.read_text(encoding="utf-8") == render(data)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit(
            "usage: python -m tests.test_fsm_llm_regression.test_module_path_map --write"
        )
    _refresh()
    print(f"wrote {FIXTURE.relative_to(REPO)}")
