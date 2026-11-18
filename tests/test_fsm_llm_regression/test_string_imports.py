"""No module of the library imports another module by a string, outside two files.

Plan d454fd62, review gate 2 blind spot. The import contracts of ``.importlinter`` read
``import`` statements; ``importlib.import_module("fsm_llm.engine")`` (or ``__import__``) is
invisible to them (a plant of such a call in ``llm/text.py`` left every contract kept). Today
``src/fsm_llm`` has exactly two places that load by a name computed at run time, both on purpose:

* ``fsm_llm/__init__.py``: the ``has_*`` / ``get_*`` probes of the subpackages
  (``find_spec("fsm_llm.agents")``, ``import_module(package)``), which must not import the
  subpackage eagerly;
* ``fsm_llm/agents/python_tools.py``: it loads the Python files of a user's tool folder
  (``spec_from_file_location``, ``module_from_spec``, ``exec_module``).

This test reads every module of the package with ``ast`` and fails on a call of one of the
string-import functions anywhere else, and on an allowlist entry that no longer holds such a
call (so the list cannot go stale). The scanner is itself tested on synthetic text.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "fsm_llm"

#: the callable names that load a module by a string or a path, as a name or an attribute
STRING_IMPORTS = frozenset(
    {
        "__import__",
        "import_module",
        "find_spec",
        "spec_from_file_location",
        "module_from_spec",
        "exec_module",
    }
)

#: module path (relative to ``src/fsm_llm``) -> the string-import names it may call, and why
ALLOWED: dict[str, tuple[frozenset[str], str]] = {
    "__init__.py": (
        frozenset({"find_spec", "import_module"}),
        "subpackage probes: has_workflows, has_reasoning, has_agents and their get_*",
    ),
    "agents/python_tools.py": (
        frozenset({"spec_from_file_location", "module_from_spec", "exec_module"}),
        "PythonToolLoader executes the Python files of a trusted tool folder",
    ),
}


def string_import_calls(source: str) -> set[str]:
    """The names of the string-import functions that *source* calls.

    A call counts when its function is a bare name (``__import__(...)``,
    ``import_module(...)``) or an attribute (``importlib.import_module(...)``,
    ``importlib.util.find_spec(...)``, ``spec.loader.exec_module(...)``).
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name in STRING_IMPORTS:
            found.add(name)
    return found


def scan(root: Path) -> dict[str, set[str]]:
    """Relative module path -> string-import names called there, for every module with any."""
    hits: dict[str, set[str]] = {}
    for path in sorted(root.rglob("*.py")):
        names = string_import_calls(path.read_text(encoding="utf-8"))
        if names:
            hits[path.relative_to(root).as_posix()] = names
    return hits


class TestTheRealTree:
    def test_only_the_two_allowed_modules_import_by_a_string(self) -> None:
        extra = {
            module: sorted(names - ALLOWED.get(module, (frozenset(), ""))[0])
            for module, names in scan(SRC).items()
            if names - ALLOWED.get(module, (frozenset(), ""))[0]
        }
        assert extra == {}, (
            "a module imports by a string: the import contracts cannot see it. Use an "
            f"`import` statement, or extend ALLOWED here with a reason: {extra}"
        )

    def test_every_allowance_is_still_used(self) -> None:
        hits = scan(SRC)
        stale = {
            module: sorted(names - hits.get(module, set()))
            for module, (names, _why) in ALLOWED.items()
            if names - hits.get(module, set())
        }
        assert stale == {}, f"an ALLOWED name is no longer called, drop it: {stale}"

    def test_the_allowed_files_exist(self) -> None:
        for module in ALLOWED:
            assert (SRC / module).is_file(), module


class TestTheScanner:
    @pytest.mark.parametrize(
        "source, expected",
        [
            (
                "import importlib\nimportlib.import_module('fsm_llm.engine')",
                {"import_module"},
            ),
            (
                "from importlib import import_module\nimport_module(name)",
                {"import_module"},
            ),
            ("__import__('fsm_llm.engine')", {"__import__"}),
            ("import importlib.util\nimportlib.util.find_spec('x')", {"find_spec"}),
            (
                "spec = u.spec_from_file_location(n, p)\nm = u.module_from_spec(spec)\n"
                "spec.loader.exec_module(m)",
                {"spec_from_file_location", "module_from_spec", "exec_module"},
            ),
            (
                "def f():\n    import importlib\n    return importlib.import_module(p)",
                {"import_module"},
            ),
            ("import importlib  # no call", set()),
            ("x = 'import_module'", set()),
            ("def import_module(): ...", set()),
            ("from fsm_llm.engine import API", set()),
        ],
    )
    def test_what_it_reports(self, source: str, expected: set[str]) -> None:
        assert string_import_calls(source) == expected

    def test_a_planted_call_in_a_copy_of_a_module_is_reported(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "fsm_llm"
        (root / "llm").mkdir(parents=True)
        original = (SRC / "llm" / "text.py").read_text(encoding="utf-8")
        (root / "llm" / "text.py").write_text(
            original
            + '\nimport importlib\nimportlib.import_module("fsm_llm.engine")\n',
            encoding="utf-8",
        )
        assert scan(root) == {"llm/text.py": {"import_module"}}
        assert scan(root).keys() - ALLOWED.keys() == {"llm/text.py"}
