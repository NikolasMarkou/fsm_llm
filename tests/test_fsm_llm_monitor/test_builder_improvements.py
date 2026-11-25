"""Plan 51fde95b step 13: the Builder improvements (D-038), one class per sub-step."""

from __future__ import annotations

import asyncio
import re
import threading
import time
from pathlib import Path
from typing import ClassVar

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import fsm_llm.agents.meta_builder as meta_builder
from fsm_llm.agents.meta_builders import AgentArtifactBuilder
from fsm_llm.monitor.constants import (
    DEFAULT_BUILDER_MAX_TOKENS,
    MAX_AGENT_ITERATIONS,
    MAX_AGENT_TIMEOUT_SECONDS,
    MAX_BUILDER_TOKENS,
)
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


class TestTheLaunchableAgentTypesAreOneList:
    """13.2: ``/api/capabilities`` lists the launchable ids; the builder's type enum
    names registry ids only."""

    def test_capabilities_lists_the_launchable_ids(self):
        mgr = _manager()
        configure(manager=mgr)
        data = TestClient(app).get("/api/capabilities").json()
        assert data["agent_types"] == [
            r["id"] for r in mgr.agent_types() if r["launchable"]
        ]
        assert (data["fsm"], data["workflows"], data["agents"]) == (True,) * 3

    def test_every_type_the_builder_can_emit_is_a_registry_id(self):
        """The Builder only offers a type that names an executable of the registry."""
        ids = {r["id"] for r in _manager().agent_types()}
        assert AgentArtifactBuilder.VALID_AGENT_TYPES <= ids

    def test_an_orchestrator_artifact_is_buildable_but_not_listed_as_launchable(self):
        """D-058: the Builder can emit ``orchestrator``; ``agent_types`` does not offer it."""
        assert "orchestrator" in AgentArtifactBuilder.VALID_AGENT_TYPES
        configure(manager=_manager())
        listed = TestClient(app).get("/api/capabilities").json()["agent_types"]
        assert "orchestrator" not in listed


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


class TestAbandonedBuildsStayCounted:
    """D-054: a force-deleted build whose worker thread still runs counts against the
    session cap and against ``MAX_ABANDONED_BUILDERS`` until that thread ends."""

    @staticmethod
    def _abandon(sid: str, agent: _SlowBuilderAgent):
        """Start a send on ``sid`` and force-delete the session while it runs."""
        from fsm_llm.monitor.definitions import BuilderSendRequest

        async def go():
            routes_builder._builder_sessions[sid] = (agent, 0.0)
            send = asyncio.ensure_future(
                routes_builder.api_builder_send(
                    BuilderSendRequest(session_id=sid, message="m")
                )
            )
            while not agent.started.is_set():
                await asyncio.sleep(0.01)
            deleted = await routes_builder.api_builder_delete(sid, force=True)
            assert deleted == {"deleted": True}
            return send

        return go()

    @staticmethod
    async def _released(sid: str):
        for _ in range(300):
            if sid not in routes_builder._builder_busy:
                return True
            await asyncio.sleep(0.01)
        return False

    def test_start_is_refused_at_the_bound_and_allowed_when_a_worker_ends(
        self, builder_client
    ):
        from fsm_llm.monitor.constants import MAX_ABANDONED_BUILDERS
        from fsm_llm.monitor.definitions import BuilderStartRequest

        agents = [_SlowBuilderAgent() for _ in range(MAX_ABANDONED_BUILDERS)]

        async def run():
            sends = [await self._abandon(f"x{i}", a) for i, a in enumerate(agents)]
            try:
                assert routes_builder._builder_sessions == {}
                with pytest.raises(HTTPException) as refused:
                    await routes_builder.api_builder_start(BuilderStartRequest())
                assert refused.value.status_code == 429
                assert "still running" in refused.value.detail
                agents[0].gate.set()
                await sends[0]
                assert await self._released("x0")
                allowed = await routes_builder.api_builder_start(BuilderStartRequest())
                assert allowed["response"] == "hello"
            finally:
                for agent in agents:
                    agent.gate.set()
                await asyncio.gather(*sends)
            assert all([await self._released(f"x{i}") for i in range(len(agents))])
            assert routes_builder._builder_busy == set()

        asyncio.run(run())

    def test_an_abandoned_worker_counts_against_the_session_cap(self, builder_client):
        from fsm_llm.monitor.constants import MAX_BUILDER_SESSIONS
        from fsm_llm.monitor.definitions import BuilderStartRequest

        agent = _SlowBuilderAgent()

        async def run():
            send = await self._abandon("gone", agent)
            for i in range(MAX_BUILDER_SESSIONS - 1):
                routes_builder._builder_sessions[f"s{i}"] = (
                    _FakeBuilderAgent(None),
                    time.time(),
                )
            try:
                with pytest.raises(HTTPException) as refused:
                    await routes_builder.api_builder_start(BuilderStartRequest())
                assert refused.value.status_code == 429
                assert "too many builder sessions" in refused.value.detail
            finally:
                agent.gate.set()
                await send
            assert await self._released("gone")
            ok = await routes_builder.api_builder_start(BuilderStartRequest())
            assert ok["response"] == "hello"

        asyncio.run(run())
