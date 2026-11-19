"""Plan 51fde95b step 13: the Builder improvements (D-038), one class per sub-step."""

from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path
from typing import ClassVar

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import fsm_llm.agents as agents_pkg
import fsm_llm.agents.meta_builder as meta_builder
from fsm_llm.agents.meta_builders import AgentArtifactBuilder
from fsm_llm.monitor.constants import (
    DEFAULT_BUILDER_MAX_TOKENS,
    MAX_AGENT_ITERATIONS,
    MAX_AGENT_TIMEOUT_SECONDS,
    MAX_BUILDER_TOKENS,
)
from fsm_llm.monitor.instance_manager import agent as agent_mod
from fsm_llm.monitor.server import app, configure, routes_builder
from tests.test_fsm_llm_monitor.support import _manager

_MONITOR = Path(__file__).parent.parent.parent / "src" / "fsm_llm" / "monitor"


def _html() -> str:
    return (_MONITOR / "templates" / "index.html").read_text(encoding="utf-8")


def _input_attrs(element_id: str) -> str:
    match = re.search(rf'<input[^>]*id="{element_id}"[^>]*>', _html())
    assert match is not None, element_id
    return match.group(0)


class TestTheLaunchFormLimitsMatchTheServer:
    """13.1: the editable limits of the built-agent launch form mirror the request bounds."""

    def test_the_iterations_and_timeout_inputs_carry_the_server_maximums(self):
        iters = re.search(r'\bmax="(\d+)"', _input_attrs("builder-agent-iters"))
        timeout = re.search(r'\bmax="(\d+)"', _input_attrs("builder-agent-timeout"))
        assert iters and int(iters.group(1)) == MAX_AGENT_ITERATIONS
        assert timeout and int(timeout.group(1)) == int(MAX_AGENT_TIMEOUT_SECONDS)


def _js_map(source: str, name: str) -> dict[str, str]:
    body = re.search(rf"const {name} = \{{(.*?)\}};", source, re.S)
    assert body is not None, name
    return dict(re.findall(r"(\w+):\s*'(\w+)'", body.group(1)))


class TestTheLaunchableAgentTypesAreOneList:
    """13.2: ``/api/capabilities`` lists ``_AGENT_CLASSES``; the JS type map, the launch
    form and the builder's type enum agree with it."""

    def _builder_js(self) -> str:
        return (_MONITOR / "static" / "pages" / "builder.js").read_text(
            encoding="utf-8"
        )

    def test_capabilities_lists_the_launchable_classes(self):
        configure(manager=_manager())
        data = TestClient(app).get("/api/capabilities").json()
        assert data["agent_types"] == sorted(agent_mod._AGENT_CLASSES)
        assert (data["fsm"], data["workflows"], data["agents"]) == (True,) * 3

    def test_the_list_follows_the_classes_the_manager_runs(self, monkeypatch):
        monkeypatch.delitem(agent_mod._AGENT_CLASSES, "DebateAgent")
        configure(manager=_manager())
        data = TestClient(app).get("/api/capabilities").json()
        assert "DebateAgent" not in data["agent_types"]

    def test_the_js_type_map_is_the_builder_enum_of_real_classes(self):
        mapping = _js_map(self._builder_js(), "AGENT_TYPE_MAP")
        assert set(mapping) == AgentArtifactBuilder.VALID_AGENT_TYPES
        assert all(hasattr(agents_pkg, cls) for cls in mapping.values())

    def test_every_launchable_class_is_a_type_the_builder_can_emit(self):
        mapping = _js_map(self._builder_js(), "AGENT_TYPE_MAP")
        assert set(agent_mod._AGENT_CLASSES) <= set(mapping.values())

    def test_the_launch_form_offers_exactly_the_launchable_classes(self):
        select = re.search(
            r'<select id="launch-agent-type"[^>]*>(.*?)</select>', _html(), re.S
        )
        assert select is not None
        options = re.findall(r'<option value="(\w+)"', select.group(1))
        assert sorted(options) == sorted(agent_mod._AGENT_CLASSES)

    def test_the_js_tool_based_list_mirrors_the_server_set(self):
        state_js = (_MONITOR / "static" / "services" / "state.js").read_text(
            encoding="utf-8"
        )
        listed = re.search(r"TOOL_BASED_AGENTS = \[(.*?)\]", state_js).group(1)
        assert set(re.findall(r"'(\w+)'", listed)) == agent_mod._TOOL_BASED_AGENTS


class _FakeBuilderAgent:
    """Stands in for ``MetaBuilderAgent``: records its config, answers with no model call."""

    configs: ClassVar[list] = []

    def __init__(self, config):
        self.config = config
        type(self).configs.append(config)

    def start(self, message):
        return "hello"

    def is_complete(self):
        return False

    def get_internal_state(self):
        return {"phase": "collect", "turn_count": 0}


@pytest.fixture
def builder_client(monkeypatch):
    """A client whose Builder routes build a `_FakeBuilderAgent` (a real route, no model call)."""
    _FakeBuilderAgent.configs = []
    monkeypatch.setattr(meta_builder, "MetaBuilderAgent", _FakeBuilderAgent)
    configure(manager=_manager())
    yield TestClient(app)
    routes_builder._builder_sessions.clear()
    routes_builder._builder_busy.clear()


class TestTheBuilderMaxTokensField:
    """13.4: the default is the library's 4096 (was 2000) and a request value reaches the config."""

    def test_the_default_is_the_meta_builder_default(self, builder_client):
        assert builder_client.post("/api/builder/start", json={}).status_code == 200
        assert (
            _FakeBuilderAgent.configs[-1].max_tokens
            == DEFAULT_BUILDER_MAX_TOKENS
            == 4096
        )

    def test_a_request_value_reaches_the_config_and_the_bounds_hold(
        self, builder_client
    ):
        post = builder_client.post
        assert post("/api/builder/start", json={"max_tokens": 777}).status_code == 200
        assert _FakeBuilderAgent.configs[-1].max_tokens == 777
        for bad in (0, MAX_BUILDER_TOKENS + 1):
            assert (
                post("/api/builder/start", json={"max_tokens": bad}).status_code == 422
            )

    def test_the_form_field_mirrors_the_default_and_the_bound(self):
        attrs = _input_attrs("builder-max-tokens")
        assert f'value="{DEFAULT_BUILDER_MAX_TOKENS}"' in attrs
        assert f'max="{MAX_BUILDER_TOKENS}"' in attrs


class _SlowBuilderAgent(_FakeBuilderAgent):
    """``send`` blocks until ``gate`` is set, as a model call that is still running."""

    def __init__(self):
        self.started, self.gate = threading.Event(), threading.Event()

    def send(self, message):
        self.started.set()
        self.gate.wait(10)
        return "late reply"


class TestForceDeleteOfABusyBuilderSession:
    """13.5: ``DELETE ?force=true`` drops a session with a send in flight at once; the
    worker thread runs on, the id stays busy until it ends and the session is never put back."""

    def _scenario(self, *steps):
        from fsm_llm.monitor.definitions import BuilderSendRequest

        agent = _SlowBuilderAgent()
        routes_builder._builder_sessions["b1"] = (agent, 0.0)

        async def run():
            send = asyncio.ensure_future(
                routes_builder.api_builder_send(
                    BuilderSendRequest(session_id="b1", message="m")
                )
            )
            while not agent.started.is_set():
                await asyncio.sleep(0.01)
            seen = [await step() for step in steps]
            agent.gate.set()
            await send  # the abandoned reply is computed and dropped
            for _ in range(200):
                if "b1" not in routes_builder._builder_busy:
                    break
                await asyncio.sleep(0.01)
            return [
                *seen,
                (
                    "b1" in routes_builder._builder_sessions,
                    "b1" in routes_builder._builder_busy,
                ),
            ]

        try:
            return asyncio.run(run())
        finally:
            agent.gate.set()
            routes_builder._builder_busy.discard("b1")
            routes_builder._builder_sessions.pop("b1", None)

    def test_a_plain_delete_is_refused_and_a_forced_one_drops_the_session(self):
        async def plain():
            with pytest.raises(HTTPException) as refused:
                await routes_builder.api_builder_delete("b1")
            return refused.value.status_code, "b1" in routes_builder._builder_sessions

        async def forced():
            result = await routes_builder.api_builder_delete("b1", force=True)
            return (
                result,
                "b1" in routes_builder._builder_sessions,
                "b1" in routes_builder._builder_busy,
            )

        expected = [
            (409, True),
            (
                {"deleted": True},
                False,
                True,
            ),  # gone at once, still busy: the thread runs on
            (False, False),  # the thread ended: released, the session is not put back
        ]
        assert self._scenario(plain, forced) == expected

    def test_force_on_an_idle_or_unknown_session_deletes_normally(self, builder_client):
        assert builder_client.delete("/api/builder/nope?force=true").json() == {
            "deleted": False
        }
        routes_builder._builder_sessions["b2"] = (_FakeBuilderAgent(None), 0.0)
        assert builder_client.delete("/api/builder/b2?force=true").json() == {
            "deleted": True
        }

    def test_a_forced_delete_is_behind_the_api_key(self):
        configure(manager=_manager(), api_key="k")
        try:
            client = TestClient(app)
            assert client.delete("/api/builder/b1?force=true").status_code == 401
            keyed = client.delete(
                "/api/builder/b1?force=true", headers={"X-API-Key": "k"}
            )
            assert keyed.status_code == 200
        finally:
            configure(manager=_manager())  # no key: the env key is unset by conftest
