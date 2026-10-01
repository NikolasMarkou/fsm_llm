"""Example calls into ``fsm_llm`` must bind to the live signatures.

``examples/`` cannot be edited (they are the eval baselines), so a removed or
renamed parameter breaks them silently: the FSM JSON still loads and no other
suite runs an example. This test parses every ``examples/**/*.py`` with
``ast`` and binds, against ``inspect.signature`` (offline, no LLM call, no
example is executed):

- calls on a name imported from ``fsm_llm`` (constructors, classmethods,
  functions);
- method calls on a name whose class is known within its scope: assigned once
  from a known constructor or from a call whose return annotation is a package
  class (``api = API.from_file(...)``, ``agent = ReactAgent(...)``,
  ``result = agent.run(...)``), a ``create_agent("<pattern>", ...)`` call with
  a literal pattern, a ``with ... as name`` over such a value, or an annotated
  parameter or variable; chained calls (``API.from_file(...).converse(...)``)
  too;
- keywords forwarded through ``**kwargs``: ``API.from_file`` and
  ``API.from_definition`` forward to ``API.__init__``; ``create_agent`` to the
  pattern's constructor; an agent constructor's leftover keywords go through
  ``BaseAgent``'s own denylist (``_reject_misplaced_kwargs``); a keyword that
  lands in ``API.__init__``'s ``**llm_kwargs`` may not be one the LLM layer
  ignores (``RESERVED_LLM_CALL_KWARGS``), so renaming ``model`` is caught.

Not checked (see ``_SKIPS``): calls with ``*args``/``**kwargs``, receivers
whose class is unknown (attributes such as ``self.api``, subscripts, loop
variables, names assigned more than one type in a scope, function results
without a package return annotation), keywords a ``**kwargs`` parameter
forwards anywhere else, and the litellm passthrough itself.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import typing
from pathlib import Path
from typing import Any

import pytest

from fsm_llm import API
from fsm_llm.agents import _PATTERNS, create_agent
from fsm_llm.agents.base import BaseAgent, _reject_misplaced_kwargs
from fsm_llm.constants import RESERVED_LLM_CALL_KWARGS

pytestmark = pytest.mark.examples

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
_PACKAGE = "fsm_llm"
_API_FACTORIES = ("from_file", "from_definition")
# A name assigned values of two different types in one scope.
_AMBIGUOUS = object()
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef)


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


def _package_class(value: Any) -> type | None:
    """``value`` when it is a class defined in the package, else None."""
    if inspect.isclass(value) and value.__module__.split(".")[0] == _PACKAGE:
        return value
    return None


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """The nodes of ``scope`` outside its nested functions and classes."""
    found: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if isinstance(node, (*_SCOPES, ast.ClassDef)):
            continue
        found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return found


def _is_api_factory(target: Any) -> bool:
    return any(
        getattr(target, "__func__", None) is getattr(API, name).__func__
        for name in _API_FACTORIES
    )


def _literal_pattern(call: ast.Call) -> str | None:
    """The pattern name of a ``create_agent`` call, when it is a literal."""
    node: ast.expr | None = call.args[0] if call.args else None
    for keyword in call.keywords:
        if keyword.arg == "pattern":
            node = keyword.value
    if node is None:
        return "react"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip().lower()
    return None


class _Checker:
    """Binds the resolvable package calls of one source file (see module doc)."""

    def __init__(self, source: str) -> None:
        self.tree = ast.parse(source)
        self.problems: list[str] = []
        self.bound = _imported_names(self.tree, self.problems)
        self.checked = 0
        # Module-level function name -> the package class it returns.
        self.returns: dict[str, type] = {}

    # -- types ------------------------------------------------------------

    def _annotation(self, node: ast.expr | None) -> type | None:
        if node is None:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                node = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                return None
        return _package_class(_resolve(node, self.bound))

    def _type_of(self, expr: ast.expr, types: dict[str, Any]) -> type | None:
        """The package class of the value ``expr`` evaluates to, else None."""
        if isinstance(expr, ast.Name):
            found = types.get(expr.id)
            return None if found is _AMBIGUOUS else found
        if not isinstance(expr, ast.Call):
            return None
        if isinstance(expr.func, ast.Name) and expr.func.id in self.returns:
            return self.returns[expr.func.id]
        target, _ = self._callee(expr.func, types, report=False)
        if target is create_agent:
            return _PATTERNS.get(_literal_pattern(expr) or "")
        if inspect.isclass(target):
            return _package_class(target)
        try:
            hints = typing.get_type_hints(target)
        except Exception:
            return None
        return _package_class(hints.get("return"))

    def _scope_types(self, scope: ast.AST, outer: dict[str, Any]) -> dict[str, Any]:
        """Name -> class for the names ``scope`` assigns exactly one class."""
        own = _own_nodes(scope)
        assigned: list[tuple[int, str, Any]] = []
        local: set[str] = set()
        for node in own:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            local.add(name.id)
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    assigned.append((node.lineno, node.targets[0].id, node.value))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                local.add(node.target.id)
                assigned.append((node.lineno, node.target.id, node.annotation))
            elif isinstance(node, ast.withitem) and isinstance(
                node.optional_vars, ast.Name
            ):
                local.add(node.optional_vars.id)
                assigned.append(
                    (node.context_expr.lineno, node.optional_vars.id, node.context_expr)
                )
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                for name in ast.walk(node.target):
                    if isinstance(name, ast.Name):
                        local.add(name.id)
        types = {name: kind for name, kind in outer.items() if name not in local}
        if isinstance(scope, _SCOPES):
            arguments = scope.args
            for arg in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            ):
                types[arg.arg] = self._annotation(arg.annotation) or _AMBIGUOUS
        for _, name, value in sorted(assigned, key=lambda item: item[0]):
            kind = (
                self._annotation(value)
                if value in {a.annotation for a in own if isinstance(a, ast.AnnAssign)}
                else self._type_of(value, types)
            )
            previous = types.get(name)
            if name in types and previous is not kind:
                types[name] = _AMBIGUOUS
            else:
                types[name] = kind if kind is not None else _AMBIGUOUS
        return types

    # -- callees ----------------------------------------------------------

    def _callee(
        self, func: ast.expr, types: dict[str, Any], *, report: bool, line: int = 0
    ) -> tuple[Any | None, bool]:
        """``(callable, needs a self slot)`` for a call's ``func``, else None."""
        direct = _resolve(func, self.bound)
        if direct is not None:
            return direct, False
        if not isinstance(func, ast.Attribute):
            return None, False
        owner = self._type_of(func.value, types)
        if owner is None:
            return None, False
        static = inspect.getattr_static(owner, func.attr, None)
        if static is None:
            if self._dynamic_attributes(owner, func.attr):
                return None, False
            if report:
                self.problems.append(
                    f"line {line}: {owner.__qualname__} has no attribute {func.attr!r}"
                )
            return None, False
        if isinstance(static, (staticmethod, classmethod)):
            return getattr(owner, func.attr), False
        if inspect.isfunction(static):
            return static, True
        return None, False

    @staticmethod
    def _dynamic_attributes(owner: type, attr: str) -> bool:
        """True when ``attr`` may exist on an instance though not on the class:
        a pydantic field or extra, or a class with its own ``__getattr__``."""
        fields = getattr(owner, "model_fields", None)
        if fields is not None:
            extra = getattr(owner, "model_config", {}).get("extra")
            return attr in fields or extra == "allow"
        return hasattr(owner, "__getattr__")

    # -- binding ----------------------------------------------------------

    def _bind(
        self, target: Any, positional: int, keywords: set[str]
    ) -> tuple[str | None, set[str]]:
        """``(problem, keywords that land in **kwargs)`` for one call shape."""
        fields = getattr(target, "model_fields", None)
        if inspect.isclass(target) and fields is not None:
            known = set(fields) | {f.alias for f in fields.values() if f.alias}
            unknown = sorted(keywords - known)
            return (f"unknown fields {unknown}" if unknown else None), set()
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return None, set()
        try:
            signature.bind(*[object()] * positional, **dict.fromkeys(keywords))
        except TypeError as error:
            return str(error), set()
        params = signature.parameters.values()
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
            return None, set()
        named = {
            p.name
            for p in params
            if p.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return None, keywords - named

    def _forward(self, target: Any, leftover: set[str], call: ast.Call) -> list[str]:
        """Problems of the keywords ``target`` forwards through ``**kwargs``."""
        if not leftover:
            return []
        if target is API:
            reserved = sorted(leftover & RESERVED_LLM_CALL_KWARGS)
            if reserved:
                return [
                    f"API keywords {reserved} land in **llm_kwargs, where the LLM "
                    "layer ignores them"
                ]
            return []
        if _is_api_factory(target):
            return self._bind_and_forward(API, 0, leftover | {"fsm_definition"}, call)
        if target is create_agent:
            pattern = _PATTERNS.get(_literal_pattern(call) or "")
            if pattern is None:
                return []
            return self._bind_and_forward(pattern, 0, leftover, call)
        if inspect.isclass(target) and issubclass(target, BaseAgent):
            named = {
                p
                for cls in target.__mro__
                if "__init__" in vars(cls)
                for p in inspect.signature(vars(cls)["__init__"]).parameters
            }
            try:
                _reject_misplaced_kwargs(
                    target.__name__, dict.fromkeys(leftover - named)
                )
            except TypeError as error:
                return [str(error)]
            return self._forward(API, leftover - named, call)
        return []

    def _bind_and_forward(
        self, target: Any, positional: int, keywords: set[str], call: ast.Call
    ) -> list[str]:
        problem, leftover = self._bind(target, positional, keywords)
        if problem:
            return [problem]
        return self._forward(target, leftover, call)

    # -- driver -----------------------------------------------------------

    def _return_type(self, function: ast.AST, outer: dict[str, Any]) -> type | None:
        """The package class a module-level helper returns: its annotation,
        else the one class all its ``return`` values have."""
        if not isinstance(function, _SCOPES):
            return None
        annotated = self._annotation(function.returns)
        if annotated is not None:
            return annotated
        types = self._scope_types(function, outer)
        kinds = {
            self._type_of(node.value, types)
            for node in _own_nodes(function)
            if isinstance(node, ast.Return) and node.value is not None
        }
        return kinds.pop() if len(kinds) == 1 else None

    def run(self) -> tuple[int, list[str]]:
        helpers = [node for node in self.tree.body if isinstance(node, _SCOPES)]
        module_types = self._scope_types(self.tree, {})
        for _ in range(2):  # a helper may return another helper's value
            for helper in helpers:
                kind = self._return_type(helper, module_types)
                if kind is not None:
                    self.returns[helper.name] = kind
            module_types = self._scope_types(self.tree, {})
        scopes: list[tuple[ast.AST, dict[str, Any]]] = [(self.tree, module_types)]
        for node in ast.walk(self.tree):
            if isinstance(node, _SCOPES):
                scopes.append((node, self._scope_types(node, module_types)))
        for scope, types in scopes:
            for node in _own_nodes(scope):
                if isinstance(node, ast.Call):
                    self._check_call(node, types)
        return self.checked, self.problems

    def _check_call(self, call: ast.Call, types: dict[str, Any]) -> None:
        target, self_slot = self._callee(
            call.func, types, report=True, line=call.lineno
        )
        if target is None or not callable(target):
            return
        if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
            keyword.arg is None for keyword in call.keywords
        ):
            return
        self.checked += 1
        keywords = {str(keyword.arg) for keyword in call.keywords}
        found = self._bind_and_forward(
            target, len(call.args) + int(self_slot), keywords, call
        )
        name = getattr(target, "__qualname__", repr(target))
        self.problems += [f"line {call.lineno}: {name}: {p}" for p in found]


def check_source(source: str) -> tuple[int, list[str]]:
    """``(calls_checked, problems)`` for one example source (see module doc)."""
    return _Checker(source).run()


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
        assert checked >= 900  # instance-method calls are checked too
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
            (
                "from fsm_llm import API\napi = API.from_file('f.json')\n"
                "api.converse('hi', 'x', conv_id='x')\n",
                "unexpected keyword argument 'conv_id'",
            ),
            (
                "from fsm_llm import API\napi = API('f.json')\napi.no_such()\n",
                "API has no attribute 'no_such'",
            ),
            (
                "from fsm_llm import API\nAPI.from_file('f.json').converse()\n",
                "missing a required argument: 'user_message'",
            ),
            (
                "from fsm_llm import API\n"
                "def main(api: API):\n    api.start_conversation(bogus=1)\n",
                "unexpected keyword argument 'bogus'",
            ),
            (
                "from fsm_llm import API\nwith API('f.json') as api:\n"
                "    api.get_data()\n",
                "missing a required argument: 'conversation_id'",
            ),
            (
                "from fsm_llm import API\nAPI.from_file('f.json', stream=True)\n",
                "land in **llm_kwargs",
            ),
            (
                "from fsm_llm import API\nAPI.from_definition({}, fsm_definition={})\n",
                "multiple values for argument 'fsm_definition'",
            ),
            (
                "from fsm_llm.agents import ReactAgent\n"
                "agent = ReactAgent(tools=None)\nagent.run('t', query='q')\n",
                "unexpected keyword argument 'query'",
            ),
            (
                "from fsm_llm.agents import ReactAgent\n"
                "ReactAgent(tools=None, model='m')\n",
                "set them on AgentConfig",
            ),
            (
                "from fsm_llm.agents import create_agent\n"
                "agent = create_agent('debate', model='m')\n",
                "DebateAgent does not accept ['model']",
            ),
            (
                "from fsm_llm.workflows import WorkflowEngine\n"
                "def build():\n    engine = WorkflowEngine()\n    return engine\n"
                "def main():\n    engine = build()\n    engine.no_such()\n",
                "WorkflowEngine has no attribute 'no_such'",
            ),
            (
                "from fsm_llm.agents import create_agent\n"
                "agent = create_agent('react')\nresult = agent.run('t')\n"
                "result.no_such()\n",
                "AgentResult has no attribute 'no_such'",
            ),
        ],
    )
    def test_the_checker_reports_a_call_that_cannot_bind(self, source, fragment):
        _, problems = check_source(source)
        assert len(problems) == 1, problems
        assert fragment in problems[0]

    @pytest.mark.parametrize(
        "source",
        [
            # One name, two types in one scope: not bound.
            "from fsm_llm import API\napi = API('f')\napi = object()\napi.no_such()\n",
            # A local class shadows nothing from the package.
            "class API:\n    pass\napi = API()\napi.no_such()\n",
            # A function-local name hides the module-level one.
            "from fsm_llm import API\napi = API('f')\n"
            "def main():\n    api = object()\n    api.no_such()\n",
        ],
    )
    def test_an_unknown_receiver_is_not_checked(self, source):
        assert check_source(source)[1] == []


class TestRenamesTheExamplesWouldMiss:
    """Review round 2 (``findings/review-iter-1-pass4.md`` concern 3): the
    guard reported nothing for these renames; each must now be reported.
    A rename of a parameter every example passes positionally does not break
    the examples, so a keyword use is what is checked."""

    @staticmethod
    def _rename(monkeypatch: pytest.MonkeyPatch, fn: Any, old: str, new: str):
        signature = inspect.signature(fn)
        params = [
            p.replace(name=new) if p.name == old else p
            for p in signature.parameters.values()
        ]
        monkeypatch.setattr(
            fn, "__signature__", signature.replace(parameters=params), raising=False
        )

    def test_api_model_renamed(self, monkeypatch: pytest.MonkeyPatch):
        self._rename(monkeypatch, API.__init__, "model", "llm_model")
        source = (
            "from fsm_llm import API\n"
            "api = API.from_definition(definition={}, model='m')\n"
        )
        (problem,) = check_source(source)[1]
        assert "['model'] land in **llm_kwargs" in problem

    def test_converse_conversation_id_renamed(self, monkeypatch: pytest.MonkeyPatch):
        self._rename(monkeypatch, API.converse, "conversation_id", "conv")
        source = (
            "from fsm_llm import API\napi = API.from_file('f.json')\n"
            "cid, _ = api.start_conversation()\n"
            "api.converse('hi', conversation_id=cid)\n"
        )
        (problem,) = check_source(source)[1]
        assert problem.startswith("line 4: API.converse: ")
        assert "'conv'" in problem

    def test_react_run_task_renamed(self, monkeypatch: pytest.MonkeyPatch):
        from fsm_llm.agents import ReactAgent

        self._rename(monkeypatch, ReactAgent.run, "task", "query")
        source = (
            "from fsm_llm.agents import ReactAgent, ToolRegistry\n"
            "agent = ReactAgent(tools=ToolRegistry())\nagent.run(task='t')\n"
        )
        (problem,) = check_source(source)[1]
        assert problem.startswith("line 3: ReactAgent.run: ")
        assert "'query'" in problem
