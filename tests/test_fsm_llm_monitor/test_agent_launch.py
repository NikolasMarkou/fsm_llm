"""Plan 659c07c3 step 4: the monitor launches an agent by executable id (roadmap E4.2).

``agent_type`` of the wire is an id (``react``), matched EXACTLY against the launchable rows; the class
table and the tool-based table are gone. ``create_agent`` gets only ``config``, ``handlers``,
``registry`` and, when due, ``tools``: no approver, no flagged stub tool, no grant (HITL census D1).
Doubles only: no model, no network.
"""

from __future__ import annotations

import ast
import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import fsm_llm.monitor.instance_manager.agent as agent_src
from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents.executables import builtin_executables, preset_executables
from fsm_llm.core.definitions import ModelNotConfiguredError
from fsm_llm.engine.executables.directory import load_directory_registry
from fsm_llm.monitor.definitions import LaunchAgentRequest, StubToolConfig
from fsm_llm.monitor.instance_manager import InstanceManager
from fsm_llm.monitor.instance_manager import agent as agent_mod
from fsm_llm.monitor.server import app, configure
from tests.test_fsm_llm_agents import agent_preview_cells as cells
from tests.test_fsm_llm_monitor.support import _manager
from tests.test_fsm_llm_monitor.test_agent_types import LAUNCHABLE, NOT_LAUNCHABLE

_STUB = StubToolConfig(name="lookup", description="A stub.", stub_response="R")
_ALLOWED = {"config", "handlers", "registry", "tools"}
_GOLDEN = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "agent_preview_golden.json").read_text(
        encoding="utf-8"
    )
)


@pytest.fixture(autouse=True)
def _model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "stub/double")
    yield
    configure(manager=InstanceManager())


class _Quick:
    """The agent a spy hands back: runs at once, no model."""

    def run(self, task):
        return SimpleNamespace(
            success=True, answer="done", trace=None, final_context={}
        )


class _Spy:
    """Stands in for ``create_agent`` of the agent module and records every call."""

    def __init__(self):
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, pattern, **kwargs):
        self.calls.append((pattern, kwargs))
        return _Quick()


def _spy(monkeypatch) -> _Spy:
    spy = _Spy()
    monkeypatch.setattr(agent_mod, "create_agent", spy)
    return spy


def _launch(mgr, agent_type, **kwargs):
    kwargs.setdefault("tools_config", [_STUB])
    managed = mgr.launch_agent(agent_type=agent_type, task="t", **kwargs)
    managed.thread.join(10)
    assert not managed.thread.is_alive(), "agent thread did not finish"
    return managed


class TestParityWithTheOldTable:
    """G2: launching by id builds the class and the FSM the class table built."""

    @pytest.mark.parametrize("old_name", sorted(cells.OLD_TYPES))
    def test_launch_by_id_builds_the_class_and_fsm_of_the_old_name(
        self, old_name, monkeypatch
    ):
        real = agent_mod.create_agent
        built: list[Any] = []

        def make(pattern, **kwargs):
            agent = real(pattern, llm_interface=cells._llm(), **kwargs)
            built.append(agent)
            return _Quick()

        monkeypatch.setattr(agent_mod, "create_agent", make)
        mgr = _manager()
        managed = _launch(
            mgr,
            cells.OLD_TYPES[old_name],
            tools_config=[StubToolConfig(name=cells.STUB_TOOL, description="A stub.")],
            max_iterations=4,
        )
        assert managed.error is None
        assert len(built) == 1
        assert cells._g2_cell(built[0]) == {
            k: v for k, v in _GOLDEN["g2"][old_name].items() if k != "id"
        }

    def test_every_launchable_id_constructs_with_the_real_create_agent(
        self, monkeypatch
    ):
        real = agent_mod.create_agent
        done: dict[str, threading.Event] = {}

        def make(pattern, **kwargs):
            agent = real(pattern, **kwargs)
            event = done[pattern]

            def run(task):
                event.set()
                raise RuntimeError("stop after construction")

            agent.run = run
            return agent

        monkeypatch.setattr(agent_mod, "create_agent", make)
        mgr = _manager(max_running_agents=64)
        done.update({name: threading.Event() for name in LAUNCHABLE})
        launched = {
            name: mgr.launch_agent(
                agent_type=name, task="t", tools_config=[_STUB], model="stub/double"
            )
            for name in LAUNCHABLE
        }
        for name, event in done.items():
            assert event.wait(10), f"{name} did not construct: {launched[name].error}"
            assert launched[name].error is None or "stop after" in launched[name].error


class TestTheKwargsOfCreateAgent:
    """H1: exactly ``config``, ``handlers``, ``registry`` and, when due, ``tools``."""

    @pytest.mark.parametrize("agent_type", LAUNCHABLE)
    def test_the_key_set_is_the_allowed_one_for_every_id(self, agent_type, monkeypatch):
        spy = _spy(monkeypatch)
        mgr = _manager()
        _launch(mgr, agent_type)
        ((pattern, kwargs),) = spy.calls
        assert pattern == agent_type
        assert {"config", "handlers", "registry"} <= set(kwargs) <= _ALLOWED
        for forbidden in ("hitl", "initial_context", "llm_interface", "system_prompt"):
            assert forbidden not in kwargs
        assert kwargs["registry"] is mgr.executables

    def test_tools_go_only_to_a_pattern_that_takes_them(self, monkeypatch):
        spy = _spy(monkeypatch)
        mgr = _manager(max_running_agents=64)
        rows = {r["id"]: r for r in mgr.agent_types()}
        for name in LAUNCHABLE:
            _launch(mgr, name)
        for pattern, kwargs in spy.calls:
            assert ("tools" in kwargs) is rows[pattern]["takes_tools"], pattern

    def test_a_pattern_that_takes_no_tools_gets_no_registry_when_tools_come(
        self, monkeypatch
    ):
        spy = _spy(monkeypatch)
        _launch(_manager(), "debate")
        assert "tools" not in spy.calls[0][1]

    @pytest.mark.parametrize("agent_type", ["plan_execute", "adapt"])
    def test_a_pattern_with_optional_tools_launches_with_none(
        self, agent_type, monkeypatch
    ):
        spy = _spy(monkeypatch)
        managed = _launch(_manager(), agent_type, tools_config=[])
        assert managed.status == "completed"
        assert "tools" not in spy.calls[0][1]

    def test_a_pattern_with_optional_tools_gets_the_ones_given(self, monkeypatch):
        spy = _spy(monkeypatch)
        _launch(_manager(), "plan_execute")
        assert len(spy.calls[0][1]["tools"].list_tools()) == 1

    def test_no_stub_tool_is_flagged_for_approval(self, monkeypatch):
        spy = _spy(monkeypatch)
        many = [StubToolConfig(name=f"t{i}", description="d") for i in range(3)]
        _launch(_manager(), "react", tools_config=many)
        tools = spy.calls[0][1]["tools"].list_tools()
        assert len(tools) == 3
        assert [t.requires_approval for t in tools] == [False] * 3

    def test_the_monitor_handlers_reach_the_agent(self, monkeypatch):
        spy = _spy(monkeypatch)
        _launch(_manager(), "react")
        handlers = spy.calls[0][1]["handlers"]
        assert len(handlers) >= 3
        assert all(hasattr(h, "name") for h in handlers)

    def test_the_source_builds_no_approver_and_names_no_grant(self):
        """No identifier, keyword or dict key of the agent module names an approver or a grant."""
        tree = ast.parse(Path(agent_src.__file__).read_text(encoding="utf-8"))
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, ast.keyword) and node.arg:
                used.add(node.arg)
            elif isinstance(node, ast.Dict):
                used |= {k.value for k in node.keys if isinstance(k, ast.Constant)}
        assert used.isdisjoint(
            {"HumanInTheLoop", "requires_approval", "hitl", "initial_context"}
        )


class TestTheIdIsMatchedExactly:
    """H2 and H9: no alias, no raw forward, no registry read for an id that is not listed."""

    @pytest.mark.parametrize(
        "bad",
        [
            " react",
            "react ",
            "React",
            "REACT",
            "ReactAgent",
            "../x.json",
            "x" * 65,
            "react\n",
            "re\x1bact",
            "",
        ],
    )
    def test_an_id_that_is_not_listed_is_refused_before_anything_is_built(
        self, bad, monkeypatch
    ):
        spy = _spy(monkeypatch)
        mgr = _manager()
        mgr.agent_types()
        reads: list[str] = []
        real_get = mgr.executables.get
        monkeypatch.setattr(
            mgr.executables, "get", lambda *a, **k: reads.append(a) or real_get(*a, **k)
        )
        with pytest.raises(ValueError, match="Unknown agent type") as info:
            mgr.launch_agent(agent_type=bad, task="t", tools_config=[_STUB])
        assert spy.calls == [] and reads == []
        text = str(info.value)
        assert "\n" not in text and "\x1b" not in text
        assert "x" * 65 not in text
        assert "react" in text.split("Available:")[1]
        assert mgr.list_instances() == []

    @pytest.mark.parametrize("name", NOT_LAUNCHABLE)
    def test_a_listed_id_that_is_not_launchable_is_refused(self, name, monkeypatch):
        spy = _spy(monkeypatch)
        mgr = _manager()
        with pytest.raises(ValueError, match="Unknown agent type"):
            mgr.launch_agent(agent_type=name, task="t", tools_config=[_STUB])
        assert spy.calls == []

    def test_the_refusal_lists_the_launchable_ids(self):
        with pytest.raises(ValueError) as info:
            _manager().launch_agent(agent_type="nope", task="t")
        assert str(info.value).split("Available: ")[1] == ", ".join(LAUNCHABLE)

    def test_an_agent_that_needs_tools_is_refused_without_any(self, monkeypatch):
        spy = _spy(monkeypatch)
        with pytest.raises(ValueError, match="requires at least one tool"):
            _manager().launch_agent(agent_type="react", task="t")
        assert spy.calls == []

    def test_the_old_tables_are_gone(self):
        assert not hasattr(agent_mod, "_AGENT_CLASSES")
        assert not hasattr(agent_mod, "_TOOL_BASED_AGENTS")
        tree = ast.parse(Path(agent_src.__file__).read_text(encoding="utf-8"))
        named = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "fsm_llm.agents"
            for alias in node.names
        }
        assert named == {"AgentConfig", "ToolRegistry", "create_agent"}


class TestTheModelIsResolvedFirst:
    """H3: no model is an error before anything is built (D-048 of plan ace7a8c8)."""

    def test_no_model_raises_before_construction(self, monkeypatch):
        monkeypatch.delenv("LLM_MODEL", raising=False)
        spy = _spy(monkeypatch)
        mgr = _manager()
        with pytest.raises(ModelNotConfiguredError):
            mgr.launch_agent(task="t", tools_config=[_STUB])
        assert spy.calls == []
        assert mgr.list_instances() == []


class TestUserFiles:
    """A user directory file launches by its id when it meets the rule, and only then."""

    def _registry(self, tmp_path, doc_id="my_react", base="react"):
        doc = pf.load_document(base)
        doc["id"] = doc_id
        (tmp_path / f"{doc_id}.json").write_text(json.dumps(doc), encoding="utf-8")
        builtins = [*builtin_executables(), *preset_executables()]
        return load_directory_registry(tmp_path, builtins, prog="t")  # type: ignore[arg-type]

    def test_a_user_file_id_launches_through_the_registry(self, tmp_path, monkeypatch):
        registry = self._registry(tmp_path)
        real = agent_mod.create_agent
        built: list[Any] = []

        def make(pattern, **kwargs):
            built.append(real(pattern, **kwargs))
            return _Quick()

        monkeypatch.setattr(agent_mod, "create_agent", make)
        mgr = _manager()
        mgr.attach_executables(registry)
        managed = _launch(mgr, "my_react", model="stub/double")
        assert managed.error is None and managed.status == "completed"
        assert type(built[0]).__name__ == "PatternAgent"

    def test_a_user_file_that_is_not_listed_launchable_is_not_built(
        self, tmp_path, monkeypatch
    ):
        registry = self._registry(tmp_path, "my_eval", "evaluator_optimizer")
        spy = _spy(monkeypatch)
        mgr = _manager()
        mgr.attach_executables(registry)
        with pytest.raises(ValueError, match="Unknown agent type"):
            mgr.launch_agent(agent_type="my_eval", task="t", tools_config=[_STUB])
        assert spy.calls == []


class TestTheWire:
    """``agent_type`` is an id in the request, the instance info, the status and the lists."""

    def _client(self, monkeypatch):
        _spy(monkeypatch)
        mgr = _manager(refresh_interval=0.5)
        configure(manager=mgr)
        return TestClient(app), mgr

    def test_the_request_defaults_to_react_and_bounds_the_id(self):
        assert LaunchAgentRequest(task="x").agent_type == "react"
        with pytest.raises(ValueError):
            LaunchAgentRequest(task="x", agent_type="x" * 65)
        assert LaunchAgentRequest(task="x", agent_type="x" * 64)

    def test_launch_answers_the_id_and_every_list_carries_it(self, monkeypatch):
        client, mgr = self._client(monkeypatch)
        resp = client.post(
            "/api/agent/launch",
            json={
                "agent_type": "react",
                "task": "t",
                "tools": [_STUB.model_dump()],
            },
        )
        assert resp.status_code == 200
        info = resp.json()
        assert info["agent_type"] == "react"
        instance_id = info["instance_id"]
        mgr.get_instance(instance_id).thread.join(10)
        listed = client.get("/api/instances").json()
        assert [i["agent_type"] for i in listed] == ["react"]
        assert client.get(f"/api/agent/{instance_id}/status").json()["agent_type"] == (
            "react"
        )
        with client.websocket_connect("/ws") as ws:
            frame = ws.receive_json()
        assert [i["agent_type"] for i in frame["instances"]] == ["react"]

    @pytest.mark.parametrize("bad", ["ReactAgent", "React", " react", "swarm", "nope"])
    def test_an_id_that_is_not_launchable_is_http_400_with_the_ids(
        self, bad, monkeypatch
    ):
        client, _ = self._client(monkeypatch)
        resp = client.post(
            "/api/agent/launch",
            json={"agent_type": bad, "task": "t", "tools": [_STUB.model_dump()]},
        )
        assert resp.status_code == 400
        assert "Unknown agent type" in resp.json()["detail"]
        assert "react" in resp.json()["detail"].split("Available:")[1]

    def test_an_id_over_64_characters_is_http_422(self, monkeypatch):
        client, _ = self._client(monkeypatch)
        resp = client.post(
            "/api/agent/launch", json={"agent_type": "x" * 65, "task": "t"}
        )
        assert resp.status_code == 422

    def test_no_model_is_http_400_and_builds_nothing(self, monkeypatch):
        monkeypatch.delenv("LLM_MODEL", raising=False)
        client, _ = self._client(monkeypatch)
        resp = client.post(
            "/api/agent/launch",
            json={"task": "t", "tools": [_STUB.model_dump()]},
        )
        assert resp.status_code == 400
        assert "model" in resp.json()["detail"].lower()
        assert agent_mod.create_agent.calls == []


_RETIRED = (
    "_AGENT_CLASSES",
    "_TOOL_BASED_AGENTS",
    "TOOL_BASED_AGENTS",
    "AGENT_TYPE_MAP",
)
_CODE_SUFFIXES = {".py", ".js", ".mjs", ".html", ".json", ".css"}


class TestTheRetiredTablesAreGoneFromSource:
    """Plan 659c07c3 step 7: no file under ``src/fsm_llm`` names a retired launch table.

    A word-bound scan of every code and data file (a hit in a comment counts) plus an AST scan
    of every module (a name, an attribute, an import alias or a defined function or class), so
    a table that comes back under its old name, in any language of the package, is red.
    """

    _SRC = Path(__file__).parent.parent.parent / "src" / "fsm_llm"

    def _files(self):
        files = [
            p
            for p in self._SRC.rglob("*")
            if p.suffix in _CODE_SUFFIXES and "__pycache__" not in p.parts
        ]
        assert len(files) > 100
        return files

    def test_no_code_or_data_file_names_a_retired_table(self):
        pattern = re.compile(r"\b(?:" + "|".join(_RETIRED) + r")\b")
        hits = [
            f"{p.relative_to(self._SRC).as_posix()}: {m.group(0)}"
            for p in self._files()
            for m in pattern.finditer(p.read_text(encoding="utf-8"))
        ]
        assert hits == []

    def test_no_module_binds_a_retired_name(self):
        bound: list[str] = []
        for path in self._files():
            if path.suffix != ".py":
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names: list[str] = []
                if isinstance(node, ast.Name):
                    names = [node.id]
                elif isinstance(node, ast.Attribute):
                    names = [node.attr]
                elif isinstance(node, ast.alias):
                    names = [node.name, node.asname or ""]
                elif isinstance(
                    node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
                ):
                    names = [node.name]
                bound += [f"{path.name}: {n}" for n in names if n in _RETIRED]
        assert bound == []

    def test_the_agent_half_of_flows_json_is_gone(self):
        flows = json.loads(
            (self._SRC / "monitor" / "static" / "flows.json").read_text("utf-8")
        )
        assert list(flows) == ["workflows"]
