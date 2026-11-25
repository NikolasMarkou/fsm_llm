"""Plan 659c07c3 step 5: the monitor draws an agent by executable id (roadmap E2.4).

``GET /api/agent/visualize?agent_type=<id>`` answers ``_fsm_viz_or_400`` of the preview FSM of a LISTED
registry row (anchor 07ad3f8c D-018: one graph builder); the hand-drawn agents half of ``flows.json`` is
gone and its workflows half stays. G1: every drawable id draws the graph of the parent golden
(``tests/fixtures/agent_preview_golden.json``, written from the parent archive). Hostile ids are 404 with a
fixed text (never echoed, never a 422 that echoes its input). Doubles only: no model, no network.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import threading
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.monitor.instance_manager import InstanceManager
from fsm_llm.monitor.server import app, application, configure, routes_visualize
from tests.test_fsm_llm_monitor.support import _manager
from tests.test_fsm_llm_monitor.test_agent_types import _doc, _user_registry, _write

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads(
    (ROOT / "tests" / "fixtures" / "agent_preview_golden.json").read_text("utf-8")
)["g1"]
DRAWABLE = sorted(i for i, cell in GOLDEN.items() if cell["graph"] is not None)
URL = "/api/agent/visualize"


@pytest.fixture(autouse=True)
def _reset_app():
    yield
    configure(manager=InstanceManager())


def _client() -> TestClient:
    configure(manager=_manager())
    return TestClient(app)


def _graph(body: dict[str, Any]) -> dict[str, Any]:
    """The wire payload in the shape of the golden: sorted nodes and edges."""
    return {
        "nodes": sorted(
            [n["id"], n["is_initial"], n["is_terminal"]] for n in body["nodes"]
        ),
        "edges": sorted([e["from"], e["to"], e["priority"]] for e in body["edges"]),
    }


class TestEveryDrawableIdDrawsTheParentGolden:
    def test_the_golden_has_twenty_drawable_ids(self):
        assert len(DRAWABLE) == 20
        assert "swarm" not in DRAWABLE and "reasoning.solve" not in DRAWABLE

    @pytest.mark.parametrize("agent_id", DRAWABLE)
    def test_the_route_draws_the_g1_graph(self, agent_id):
        resp = _client().get(URL, params={"agent_type": agent_id})
        assert resp.status_code == 200, resp.text
        assert _graph(resp.json()) == GOLDEN[agent_id]["graph"]

    @pytest.mark.parametrize(
        "agent_id", [i for i in DRAWABLE if GOLDEN[i]["real"] and i != "native_fc"]
    )
    def test_the_drawing_equals_the_real_agents_fsm(self, agent_id):
        body = _client().get(URL, params={"agent_type": agent_id}).json()
        assert _graph(body) == GOLDEN[agent_id]["real"]["graph"]
        assert body["fsm"]["state_count"] == GOLDEN[agent_id]["real"]["states"]

    def test_native_fc_draws_the_five_state_file_machine(self):
        body = _client().get(URL, params={"agent_type": "native_fc"}).json()
        assert body["fsm"]["state_count"] == 5
        assert GOLDEN["native_fc"]["real"]["states"] == 3

    def test_the_default_id_is_react(self):
        client = _client()
        assert (
            client.get(URL).json()
            == client.get(URL, params={"agent_type": "react"}).json()
        )

    def test_the_payload_has_the_keys_the_page_reads(self):
        body = _client().get(URL, params={"agent_type": "react"}).json()
        assert set(body) == {"fsm", "nodes", "edges"}
        assert body["fsm"]["name"] and body["nodes"] and body["edges"]
        assert all("x" not in n and "y" not in n for n in body["nodes"])


class TestNothingToDrawIsAFixed404:
    @pytest.mark.parametrize("agent_id", ["swarm", "reasoning.solve"])
    def test_a_listed_id_with_no_preview_is_404_not_an_empty_graph(self, agent_id):
        resp = _client().get(URL, params={"agent_type": agent_id})
        assert resp.status_code == 404
        assert resp.json() == {"detail": "no preview for this agent type"}

    @pytest.mark.parametrize(
        "agent_id",
        [
            "nope",
            "ReactAgent",
            "React",
            " react",
            "react ",
            "../x.json",
            "a/b",
            "x" * 64,
            "x" * 65,
            "x" * 5000,
            "re\nact",
            "re\x00act",
            "re\x1b[31mact",
            "constructor",
            "__proto__",
            "",
        ],
    )
    def test_an_unlisted_or_hostile_id_is_404_and_is_never_echoed(self, agent_id):
        resp = _client().get(URL, params={"agent_type": agent_id})
        assert resp.status_code == 404
        assert resp.json() == {"detail": "unknown agent type"}
        assert agent_id.strip() == "" or agent_id.strip() not in resp.text

    def test_a_registry_id_that_is_no_agent_row_is_not_drawn(self):
        # a non-pattern executable of the registry (a workflow preset) is not a row
        registry = ExecutableRegistry()
        registry.load_sources()
        other = [e.id for e in registry.list() if e.kind != "pattern"]
        assert other
        mgr = _manager()
        mgr.attach_executables(registry)
        configure(manager=mgr)
        resp = TestClient(app).get(URL, params={"agent_type": other[0]})
        assert resp.status_code == 404


class TestTheDrawingPath:
    def test_the_route_answers_through_the_one_graph_builder(self):
        tree = ast.parse(inspect.getsource(routes_visualize))
        route = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "api_agent_visualize"
        )
        called = {
            c.func.id
            for c in ast.walk(route)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        }
        assert "_fsm_viz_or_400" in called
        assert "build_fsm_graph" not in called
        assert "get_flows" not in called

    def test_build_fsm_graph_is_called_in_one_function_only(self):
        tree = ast.parse(inspect.getsource(routes_visualize))
        callers = {
            fn.name
            for fn in ast.walk(tree)
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            for c in ast.walk(fn)
            if isinstance(c, ast.Call)
            and getattr(c.func, "id", None) == "build_fsm_graph"
        }
        assert callers == {"_fsm_viz_or_400"}

    def test_a_draw_builds_no_agent_and_calls_no_model(self):
        client = _client()
        boom = RuntimeError("must not be called")
        with (
            mock.patch("fsm_llm.agents.create_agent", side_effect=boom),
            mock.patch("fsm_llm.llm.providers.litellm.completion", side_effect=boom),
            mock.patch("fsm_llm.policy.hitl.HumanInTheLoop.__init__", side_effect=boom),
        ):
            for agent_id in DRAWABLE:
                assert (
                    client.get(URL, params={"agent_type": agent_id}).status_code == 200
                )

    def test_the_preview_is_built_off_the_event_loop(self):
        seen: list[bool] = []
        real = InstanceManager.agent_preview

        def probe(self: InstanceManager, agent_type: str) -> Any:
            try:
                asyncio.get_running_loop()
                seen.append(True)
            except RuntimeError:
                seen.append(False)
            return real(self, agent_type)

        client = _client()
        with mock.patch.object(InstanceManager, "agent_preview", probe):
            assert client.get(URL, params={"agent_type": "react"}).status_code == 200
        assert seen == [False]

    def test_two_draws_are_independent_dicts(self):
        mgr = _manager()
        first = mgr.agent_preview("react")
        second = mgr.agent_preview("react")
        assert first == second and first is not second
        first["states"].clear()
        first["mutated"] = True
        assert mgr.agent_preview("react") == second

    def test_concurrent_first_draws_all_answer_the_same_graph(self):
        mgr = _manager()
        out: list[Any] = []
        threads = [
            threading.Thread(target=lambda: out.append(mgr.agent_preview("react")))
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert len(out) == 4 and all(o == out[0] for o in out)


class TestUserFilesDrawTheirOwnFile:
    def test_an_id_of_the_executables_directory_draws_its_file(self, tmp_path):
        _write(tmp_path, "user_loop.json", _doc("user.loop", "react"))
        mgr = _manager()
        mgr.attach_executables(_user_registry(tmp_path))
        configure(manager=mgr)
        body = TestClient(app).get(URL, params={"agent_type": "user.loop"}).json()
        assert _graph(body) == GOLDEN["react"]["graph"]

    def test_a_file_whose_preview_cannot_be_built_is_no_preview(self, tmp_path):
        mgr = _manager()
        mgr.attach_executables(_user_registry(tmp_path))
        with mock.patch(
            "fsm_llm.agents.executables.PatternExecutable.preview_definition",
            side_effect=ValueError("needs options"),
        ):
            configure(manager=mgr)
            resp = TestClient(app).get(URL, params={"agent_type": "react"})
        assert resp.status_code == 404
        assert resp.json() == {"detail": "no preview for this agent type"}


class TestFlowsHoldOnlyTheWorkflows:
    def test_the_shipped_file_has_only_the_workflows_half(self):
        path = Path(application.STATIC_DIR) / "flows.json"
        assert list(json.loads(path.read_text("utf-8"))) == ["workflows"]

    def test_get_flows_and_the_static_route_list_no_agent_graph(self):
        client = _client()
        assert list(application.get_flows()) == ["workflows"]
        served = client.get("/static/flows.json").json()
        assert list(served) == ["workflows"] and served["workflows"]

    def test_a_missing_file_falls_back_to_empty_workflows_only(self, tmp_path):
        with mock.patch.object(application, "STATIC_DIR", tmp_path):
            assert application._load_flows() == {"workflows": {}}

    def test_the_workflow_route_still_serves_its_graphs(self):
        client = _client()
        resp = client.get("/api/workflow/visualize?workflow_id=order_processing")
        assert resp.status_code == 200 and resp.json()["nodes"]
        assert client.get("/api/workflow/visualize?workflow_id=nope").status_code == 404
