"""A flagged (``requires_approval``) tool never runs with no approval path (plan a555dcfe, step 28.1, D-035).

The Phase P1 review found that ``PatternAgent`` decided "this file runs tools" from the CLAIM
``requires.tools == "some"`` or from the binding ``agents.tool_executor`` alone, so a file (or a
model-origin preset on the allow-list) that lists tools by NAME and binds ``agents.rewoo_executor`` or
``agents.native_fc`` ran a flagged tool with no grant. Three layers now hold it: the library declares which
factories run tools (``TOOL_RUNNING``) and the agent refuses from the bindings, the two ungated factories
fail closed themselves, and V7 refuses a model preset that rewrites its base's tool claim to ``"none"``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.agents import create_agent
from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents.base import accepts_tools
from fsm_llm.agents.handlers import library as lib
from fsm_llm.agents.handlers import library_react, library_workers
from fsm_llm.agents.handlers.context import HandlerContext
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.verify.executable import load_executable
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.engine.executables.namespaces import (
    GENERATORS,
    PATTERN_AGENTS,
    namespace_resolver,
)
from fsm_llm.tools import ToolRegistry
from tests import pattern_runs as runs

_HANDLERS_DIR = Path(lib.__file__).parent
_WORDING = "no human-in-the-loop approval"
_ALLOW = {"tools": ["danger"], "executables": ["rewoo"]}


def _file(pattern: str, tools: Any, doc_id: str = "user.p") -> dict[str, Any]:
    """A shipped pattern file under a new id that LISTS its tools by name (the review's probe)."""
    doc = pf.load_document(pattern)
    doc["id"] = doc_id
    doc["requires"]["tools"] = tools
    return doc


def _danger_registry(name: str = "danger") -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(
        lambda params: "ran", name=name, description="d", requires_approval=True
    )
    return registry


#: the tool the file lists by name (``native_fc.json`` carries a placeholder tool called ``sample``)
_TOOL = {"rewoo": "danger", "native_fc": "sample"}


def _make(doc: dict[str, Any], tools: ToolRegistry | None, **kw: Any) -> Any:
    registry = ExecutableRegistry()
    registry.add_document(doc)
    kw.setdefault("config", runs._config())
    kw.setdefault("llm_interface", runs.ScriptedModel())
    return create_agent(doc["id"], tools, registry=registry, **kw)


class TestAFileThatListsToolsByName:
    @pytest.mark.parametrize("pattern", ["rewoo", "native_fc"])
    def test_a_flagged_tool_is_refused_at_creation_whatever_the_claim_says(
        self, pattern: str
    ):
        model = runs.ScriptedModel()
        tool = _TOOL[pattern]
        for claim in ([tool], "some"):
            with pytest.raises(AgentError, match=_WORDING):
                _make(
                    _file(pattern, claim),
                    _danger_registry(tool),
                    llm_interface=model,
                )
        assert model.log == []

    @pytest.mark.parametrize("pattern", ["rewoo", "native_fc"])
    def test_and_at_run_start_when_the_registry_gains_one_later(self, pattern: str):
        model = runs.ScriptedModel()
        tool = _TOOL[pattern]
        tools = runs._tools()
        tools.register_function(
            lambda p: "x", name=tool, description="d", requires_approval=False
        )
        agent = _make(_file(pattern, [tool]), tools, llm_interface=model)
        tools.register_function(
            lambda p: "x", name="later", description="d", requires_approval=True
        )

        with pytest.raises(AgentError, match=_WORDING):
            agent.run("task")
        assert model.log == []

    @pytest.mark.parametrize("pattern", ["rewoo", "native_fc"])
    def test_an_unflagged_tool_still_builds(self, pattern: str):
        tool = _TOOL[pattern]
        tools = runs._tools()
        tools.register_function(lambda p: "x", name=tool, description="d")

        agent = _make(_file(pattern, [tool]), tools)

        assert type(agent) is PatternAgent
        assert agent._executes_tools is True

    def test_a_list_claim_alone_refuses_a_flagged_tool_even_with_no_runner_bound(self):
        """``requires.tools`` that is not ``none`` says the file uses tools: the claim is enough to refuse."""
        model = runs.ScriptedModel()
        with pytest.raises(AgentError, match=_WORDING):
            _make(
                _file("orchestrator", ["danger"]),
                _danger_registry(),
                llm_interface=model,
            )
        assert model.log == []

    def test_a_gate_beside_an_ungated_runner_is_still_refused(self):
        """The gate is consulted by ``agents.tool_executor`` only: ``rewoo_executor`` never asks."""
        doc = pf.load_document("react")
        doc["id"] = "user.mixed"
        swap = {"agents.tool_executor": "agents.rewoo_executor"}
        for binding in doc["body"]["handlers"]:
            binding["handler"] = swap.get(binding["handler"], binding["handler"])
        doc["requires"]["handlers"] = [
            swap.get(n, n) for n in doc["requires"]["handlers"]
        ]
        with pytest.raises(AgentError, match=_WORDING):
            _make(doc, _danger_registry(), hitl=runs._hitl())


class TestTheFactoriesFailClosedThemselves:
    def _ctx(self, tools: ToolRegistry | None, *, hitl: bool = False) -> HandlerContext:
        if not hitl:
            return HandlerContext(tools=tools, config=runs._config())
        return HandlerContext(
            tools=tools,
            config=runs._config(),
            hitl=runs._hitl(),
            hitl_active=True,
            approval_predicate=lambda call, context: True,
        )

    @pytest.mark.parametrize("hitl", [False, True])
    def test_rewoo_executor_refuses_a_flagged_tool(self, hitl: bool):
        with pytest.raises(AgentError, match="danger"):
            library_workers.rewoo_executor(self._ctx(_danger_registry(), hitl=hitl))

    @pytest.mark.parametrize("hitl", [False, True])
    def test_native_fc_refuses_a_flagged_tool(self, hitl: bool):
        with pytest.raises(AgentError, match="danger"):
            library_react.native_fc(self._ctx(_danger_registry(), hitl=hitl))

    def test_an_unflagged_registry_still_builds_both(self):
        ctx = self._ctx(runs._tools())

        assert library_workers.rewoo_executor(ctx)
        assert library_react.native_fc(ctx)


class TestTheLibraryDeclaresWhoRunsTools:
    def test_the_set_names_exactly_the_three_factories_that_call_execute(self):
        assert lib.TOOL_RUNNING == {
            "agents.tool_executor",
            "agents.rewoo_executor",
            "agents.native_fc",
        }
        assert lib.UNGATED_TOOL_RUNNING == lib.TOOL_RUNNING - {"agents.tool_executor"}
        assert lib.TOOL_RUNNING <= {name for name, _ in lib.LIBRARY}

    def test_no_other_handler_module_calls_a_registry_execute(self):
        """A module under ``handlers/`` that runs a tool must be one of the three above."""
        callers = set()
        for path in sorted(_HANDLERS_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute"
                for node in ast.walk(tree)
            ):
                callers.add(path.stem)
        assert callers == {"executor", "native_fc", "rewoo"}


class TestAModelOriginPreset:
    def _load(self, requires: dict[str, Any]) -> Any:
        held = {e.id: e for e in _builtin().values()}
        doc = {
            "schema": "fsm-llm/executable",
            "kind": "pattern",
            "id": "m.rw",
            "name": "m.rw",
            "description": "d",
            "requires": requires,
            "body": {"extends": "rewoo"},
        }
        return load_executable(
            doc,
            origin="model",
            resolver=namespace_resolver(),
            generators=GENERATORS.get,
            allow=_ALLOW,
            bases=lambda name: getattr(held.get(name), "envelope", None),
        )

    def test_naming_an_allow_listed_flagged_tool_loads_and_the_agent_refuses_it(self):
        envelope, report = self._load({"tools": ["danger"]})

        assert envelope is not None, report.errors
        with pytest.raises(AgentError, match=_WORDING):
            PatternAgent(
                envelope,
                tools=_danger_registry(),
                config=runs._config(),
                llm_interface=runs.ScriptedModel(),
            )

    def test_a_preset_may_not_claim_no_tools_over_its_base(self):
        envelope, report = self._load({"tools": "none"})

        assert envelope is None
        assert any("requires.tools" in e and "none" in e for e in report.errors)

    def test_a_preset_that_writes_no_tool_claim_is_unchanged(self):
        envelope, report = self._load({})

        assert envelope is not None, report.errors
        assert envelope.requires.tools == "some"


def _builtin() -> dict[str, Any]:
    import fsm_llm.agents.executables as ae

    return {e.id: e for e in ae.builtin_executables()}


class TestEveryShippedFileStillBuilds:
    @pytest.mark.parametrize("pattern", pf.PATTERN_ORDER)
    def test_the_shipped_pattern_builds_with_unflagged_tools(self, pattern: str):
        cls = PATTERN_AGENTS.get(pattern)
        tools = runs._tools() if cls is None or accepts_tools(cls) else None
        kw: dict[str, Any] = {}
        if pattern == "evaluator_optimizer":
            kw["evaluation_fn"] = lambda *a: None
        if pattern == "prompt_chain":
            kw["chain"] = runs._chain()
        if pattern in ("maker_checker",):
            kw.update(maker_instructions="m", checker_instructions="c")
        registry = ExecutableRegistry()
        registry.load_sources()
        agent = create_agent(
            pattern,
            tools,
            registry=registry,
            config=runs._config(),
            llm_interface=runs.ScriptedModel(),
            **kw,
        )
        assert agent is not None
