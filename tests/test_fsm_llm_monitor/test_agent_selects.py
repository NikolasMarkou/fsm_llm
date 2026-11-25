"""Plan 659c07c3 step 6: the page holds no list of agent types.

The two agent selects are empty in the template and filled by ``loadAgentTypes`` from
``GET /api/agent/types`` (the node probes of ``test_static_js.py`` run the fill); these rows
read the shipped files and the real app: no class name or option survives in the page, every id
the visualizer select offers draws through the real route, and every id the Builder can emit is
a registry id.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fsm_llm.agents.meta_builders import AgentArtifactBuilder
from fsm_llm.monitor.server import app, configure
from tests.test_fsm_llm_monitor.support import _manager

_MONITOR = Path(__file__).parent.parent.parent / "src" / "fsm_llm" / "monitor"
_OLD_NAMES = re.compile(
    r"\b(?:React|Reflexion|PlanExecute|Debate|SelfConsistency|REWOO|PromptChain"
    r"|EvaluatorOptimizer|MakerChecker|Orchestrator|ADaPT|ReasoningReact)Agent\b"
    r"|TOOL_BASED_AGENTS|AGENT_TYPE_MAP|_AGENT_CLASSES|_agentClass"
)
_BUILDER_NOT_LAUNCHABLE = {
    "evaluator_optimizer",
    "maker_checker",
    "orchestrator",
    "prompt_chain",
}


def _scripts() -> list[Path]:
    return sorted((_MONITOR / "static").rglob("*.js"))


def _select(html: str, element_id: str) -> str:
    match = re.search(rf'<select id="{element_id}"[^>]*>(.*?)</select>', html, re.S)
    assert match is not None, element_id
    return match.group(1)


class TestThePageHoldsNoAgentList:
    @pytest.mark.parametrize("element_id", ["viz-agent-type", "launch-agent-type"])
    def test_the_agent_selects_are_empty_in_the_template(self, element_id):
        html = (_MONITOR / "templates" / "index.html").read_text(encoding="utf-8")
        assert _select(html, element_id).strip() == ""

    def test_no_class_name_or_table_survives_in_the_template_or_a_script(self):
        files = [_MONITOR / "templates" / "index.html", *_scripts()]
        assert len(files) > 10
        hits = [
            f"{f.name}: {m.group(0)}"
            for f in files
            for m in _OLD_NAMES.finditer(f.read_text(encoding="utf-8"))
        ]
        assert hits == []

    def test_the_scripts_set_option_text_with_textcontent_only(self):
        source = (_MONITOR / "static" / "utils" / "dom.js").read_text(encoding="utf-8")
        body = source[source.index("export function fillSelect") :]
        body = body[: body.index("\n}\n")]
        assert "createElement('option')" in body
        assert "textContent" in body
        assert "innerHTML" not in body and "insertAdjacentHTML" not in body

    def test_the_two_pages_fill_through_the_one_helper_and_the_one_loader(self):
        pages = _MONITOR / "static" / "pages"
        for name in ("launch.js", "visualizer.js"):
            text = (pages / name).read_text(encoding="utf-8")
            assert "fillSelect(" in text and "loadAgentTypes()" in text, name

    def test_the_graph_is_drawn_on_entering_the_agents_tab_and_the_visualizer_page(
        self,
    ):
        """The select starts empty and defaults to ``react``: the tab and the page draw it."""
        app_js = (_MONITOR / "static" / "app.js").read_text(encoding="utf-8")
        tab = app_js[app_js.index("function switchTab") :]
        tab = tab[: tab.index("\n}\n")]
        assert "tab-agents" in tab and "visualizer.showAgentGraph()" in tab
        page = app_js[app_js.index("function showPage") :]
        page = page[: page.index("\n}\n")]
        assert "visualizer.showAgentGraph()" in page

    def test_the_default_id_is_the_servers_default(self):
        state = (_MONITOR / "static" / "services" / "state.js").read_text(
            encoding="utf-8"
        )
        assert "DEFAULT_AGENT_TYPE = 'react'" in state
        configure(manager=_manager())
        route = TestClient(app).get("/api/agent/visualize")
        assert route.status_code == 200  # no agent_type: the route default is react


class TestEveryOfferedIdAnswers:
    def _rows(self):
        configure(manager=_manager())
        client = TestClient(app)
        return client, client.get("/api/agent/types").json()["agent_types"]

    def test_every_drawable_id_the_select_offers_draws_a_graph_the_page_can_lay_out(
        self,
    ):
        client, rows = self._rows()
        drawable = [r["id"] for r in rows if r["drawable"]]
        assert "react" in drawable and len(drawable) >= 17
        for agent_id in drawable:
            got = client.get("/api/agent/visualize", params={"agent_type": agent_id})
            assert got.status_code == 200, agent_id
            body = got.json()
            assert body["nodes"] and isinstance(body["edges"], list), agent_id
            assert body["fsm"]["name"], agent_id

    def test_an_id_that_is_not_drawable_is_not_offered_and_answers_404(self):
        client, rows = self._rows()
        silent = [r["id"] for r in rows if not r["drawable"]]
        assert "swarm" in silent
        for agent_id in silent:
            got = client.get("/api/agent/visualize", params={"agent_type": agent_id})
            assert got.status_code == 404, agent_id

    def test_every_needs_tools_row_takes_tools_so_the_tools_form_is_shown(self):
        _, rows = self._rows()
        assert all(r["takes_tools"] for r in rows if r["needs_tools"])

    def test_every_launchable_id_is_one_the_launch_select_can_group(self):
        _, rows = self._rows()
        launchable = [r for r in rows if r["launchable"]]
        assert len(launchable) == 15
        assert all(isinstance(r["needs_tools"], bool) for r in launchable)


class TestTheBuilderEmitsRegistryIds:
    def test_every_builder_id_launches_or_is_one_of_the_four_it_names(self):
        """Every id of the Builder enum is a registry id; the ones the monitor cannot
        launch are the four the Builder note (gate probe) names."""
        configure(manager=_manager())
        rows = TestClient(app).get("/api/agent/types").json()["agent_types"]
        ids = {r["id"] for r in rows}
        launchable = {r["id"] for r in rows if r["launchable"]}
        builder = AgentArtifactBuilder.VALID_AGENT_TYPES
        assert builder <= ids
        assert builder - launchable == _BUILDER_NOT_LAUNCHABLE
