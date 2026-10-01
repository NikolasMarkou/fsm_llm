"""Every call an example makes into ``fsm_llm`` must bind to the live signature.

``examples/`` cannot be edited (they are the eval baselines), so a removed or
renamed parameter breaks them silently: the FSM JSON still loads and no other
suite runs an example. This test parses every ``examples/**/*.py`` with
``ast``, resolves each call on a name imported from ``fsm_llm`` (constructors,
classmethods, functions) and binds its positional count and keyword names
against ``inspect.signature``. Offline, no LLM call, no example is executed.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.examples

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
_PACKAGE = "fsm_llm"


def _imported_names(tree: ast.AST, problems: list[str]) -> dict[str, Any]:
    """Local name -> object, for every import from the package in ``tree``."""
    bound: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] != _PACKAGE:
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                if hasattr(module, alias.name):
                    bound[alias.asname or alias.name] = getattr(module, alias.name)
                    continue
                try:
                    bound[alias.asname or alias.name] = importlib.import_module(
                        f"{node.module}.{alias.name}"
                    )
                except ImportError:
                    problems.append(
                        f"line {node.lineno}: {node.module}.{alias.name} is missing"
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != _PACKAGE:
                    continue
                module = importlib.import_module(alias.name)
                if alias.asname:
                    bound[alias.asname] = module
                else:
                    bound[_PACKAGE] = importlib.import_module(_PACKAGE)
    # A local definition with the same name shadows the import.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.pop(node.name, None)
    return bound


def _resolve(expr: ast.expr, bound: dict[str, Any]) -> Any | None:
    """The package object a ``Name`` / ``Attribute`` chain names, else None."""
    if isinstance(expr, ast.Name):
        return bound.get(expr.id)
    if isinstance(expr, ast.Attribute):
        base = _resolve(expr.value, bound)
        if inspect.ismodule(base) or inspect.isclass(base):
            return getattr(base, expr.attr, None)
    return None


def _bind_problem(target: Any, call: ast.Call) -> str | None:
    """Why ``call`` cannot bind to ``target``, or None when it binds."""
    keywords = {keyword.arg for keyword in call.keywords}
    fields = getattr(target, "model_fields", None)
    if inspect.isclass(target) and fields is not None:
        known = set(fields) | {f.alias for f in fields.values() if f.alias}
        unknown = sorted(str(name) for name in keywords - known)
        return f"unknown fields {unknown}" if unknown else None
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return None
    try:
        signature.bind(
            *[object()] * len(call.args), **{str(name): object() for name in keywords}
        )
    except TypeError as error:
        return str(error)
    return None


def check_source(source: str) -> tuple[int, list[str]]:
    """Bind every resolvable package call in ``source``.

    Returns:
        ``(calls_checked, problems)``. A call with ``*args`` or ``**kwargs``
        cannot be bound statically and is skipped, as is a call on anything
        that is not a name imported from the package (an instance method).
    """
    problems: list[str] = []
    tree = ast.parse(source)
    bound = _imported_names(tree, problems)
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _resolve(node.func, bound)
        if target is None or not callable(target):
            continue
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            keyword.arg is None for keyword in node.keywords
        ):
            continue
        checked += 1
        problem = _bind_problem(target, node)
        if problem:
            name = getattr(target, "__qualname__", repr(target))
            problems.append(f"line {node.lineno}: {name}: {problem}")
    return checked, problems


class TestExampleCallsBind:
    def test_every_example_call_binds_to_the_live_signature(self):
        files = sorted(EXAMPLES_DIR.rglob("*.py"))
        checked = 0
        problems: list[str] = []
        for path in files:
            count, found = check_source(path.read_text())
            checked += count
            problems += [f"{path.relative_to(EXAMPLES_DIR)} {line}" for line in found]

        assert len(files) >= 100
        assert checked >= 300  # the check is not vacuous
        assert problems == []

    def test_from_definition_keyword_the_examples_use_binds(self):
        source = (
            "from fsm_llm import API\n"
            "api = API.from_definition(\n    definition={}, model='m'\n)\n"
        )
        assert check_source(source) == (1, [])

    @pytest.mark.parametrize(
        ("source", "fragment"),
        [
            (
                "from fsm_llm import API\nAPI.from_definition({}, {}, model='m')\n",
                "too many positional arguments",
            ),
            (
                "from fsm_llm import API\nAPI.from_file()\n",
                "missing a required argument: 'path'",
            ),
            (
                "from fsm_llm.definitions import State\nState(id='a', bogus=1)\n",
                "unknown fields ['bogus']",
            ),
            (
                "from fsm_llm import no_such_name\n",
                "fsm_llm.no_such_name is missing",
            ),
        ],
    )
    def test_the_checker_reports_a_call_that_cannot_bind(self, source, fragment):
        _, problems = check_source(source)
        assert len(problems) == 1
        assert fragment in problems[0]
