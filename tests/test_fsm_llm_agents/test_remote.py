"""AgentServer opt-in API key and input size limit; RemoteAgentTool auth header.

Plan plan-2026-09-24T091842-c1d5bfbc, D-006.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from fsm_llm_agents import remote
from fsm_llm_agents.remote import AgentServer, RemoteAgentTool

KEY = "s3cret-key"
ROUTES = ["/invoke", "/stream"]


class _StubAgent:
    """Records every run call and returns a fixed result."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def run(self, task: str, initial_context: Any = None) -> Any:
        self.calls.append((task, initial_context))
        return SimpleNamespace(
            answer=f"done: {task[:10]}",
            success=True,
            trace=SimpleNamespace(total_iterations=1, tools_used=[]),
        )


def _client(**kwargs: Any) -> tuple[TestClient, _StubAgent]:
    agent = _StubAgent()
    server = AgentServer(agent=agent, **kwargs)
    return TestClient(server.app), agent


def _post(client: TestClient, route: str, body: dict, headers: Any = None):
    return client.post(route, json=body, headers=headers)


class TestApiKey:
    @pytest.mark.parametrize("route", ROUTES)
    def test_no_key_is_401(self, route):
        client, agent = _client(api_key=KEY)
        resp = _post(client, route, {"task": "hi"})
        assert resp.status_code == 401
        assert agent.calls == []

    @pytest.mark.parametrize("route", ROUTES)
    def test_wrong_key_is_401(self, route):
        client, agent = _client(api_key=KEY)
        resp = _post(client, route, {"task": "hi"}, {"x-api-key": "nope"})
        assert resp.status_code == 401
        resp = _post(client, route, {"task": "hi"}, {"authorization": "Bearer nope"})
        assert resp.status_code == 401
        assert agent.calls == []

    @pytest.mark.parametrize("route", ROUTES)
    @pytest.mark.parametrize("header", ["x-api-key", "authorization"])
    def test_non_ascii_key_is_401_not_500(self, route, header):
        client, agent = _client(api_key=KEY)
        # Header values travel as latin-1 bytes; the server decodes them to a
        # non-ASCII str, which str-overload compare_digest would reject with
        # TypeError (a 500).
        raw = "\xe9t\xe9".encode("latin-1")
        value = b"Bearer " + raw if header == "authorization" else raw
        resp = _post(client, route, {"task": "hi"}, {header: value})
        assert resp.status_code == 401
        assert agent.calls == []

    @pytest.mark.parametrize("route", ROUTES)
    def test_bearer_key_is_200(self, route):
        client, agent = _client(api_key=KEY)
        resp = _post(client, route, {"task": "hi"}, {"Authorization": f"Bearer {KEY}"})
        assert resp.status_code == 200
        assert len(agent.calls) == 1

    @pytest.mark.parametrize("route", ROUTES)
    def test_x_api_key_is_200(self, route):
        client, agent = _client(api_key=KEY)
        resp = _post(client, route, {"task": "hi"}, {"X-API-Key": KEY})
        assert resp.status_code == 200
        assert len(agent.calls) == 1

    def test_health_and_info_open(self):
        client, _ = _client(api_key=KEY)
        assert client.get("/health").status_code == 200
        assert client.get("/info").status_code == 200

    @pytest.mark.parametrize("route", ROUTES)
    def test_no_api_key_keeps_open_behaviour(self, route):
        client, agent = _client()
        resp = _post(client, route, {"task": "hi"})
        assert resp.status_code == 200
        assert len(agent.calls) == 1

    def test_invoke_body_unchanged_when_authorized(self):
        client, _ = _client(api_key=KEY)
        resp = _post(client, "/invoke", {"task": "hello"}, {"X-API-Key": KEY})
        assert resp.json()["answer"] == "done: hello"
        assert resp.json()["success"] is True


class TestInputLimit:
    LIMIT = 50

    @pytest.mark.parametrize("route", ROUTES)
    def test_oversize_task_is_413_and_agent_not_called(self, route):
        client, agent = _client(max_input_chars=self.LIMIT)
        # context None counts as json.dumps({}) == "{}" (2 chars)
        resp = _post(client, route, {"task": "x" * (self.LIMIT - 1)})
        assert resp.status_code == 413
        assert agent.calls == []

    @pytest.mark.parametrize("route", ROUTES)
    def test_oversize_context_is_413(self, route):
        client, agent = _client(max_input_chars=self.LIMIT)
        resp = _post(client, route, {"task": "t", "context": {"k": "v" * self.LIMIT}})
        assert resp.status_code == 413
        assert agent.calls == []

    @pytest.mark.parametrize("route", ROUTES)
    def test_at_limit_is_200(self, route):
        client, agent = _client(max_input_chars=self.LIMIT)
        resp = _post(client, route, {"task": "x" * (self.LIMIT - 2)})
        assert resp.status_code == 200
        assert len(agent.calls) == 1

    @pytest.mark.parametrize("route", ROUTES)
    def test_at_limit_with_context_is_200(self, route):
        client, agent = _client(max_input_chars=self.LIMIT)
        context = {"k": "v"}
        task = "x" * (self.LIMIT - len(json.dumps(context)))
        resp = _post(client, route, {"task": task, "context": context})
        assert resp.status_code == 200
        assert len(agent.calls) == 1

    @pytest.mark.parametrize("route", ROUTES)
    def test_none_disables_limit(self, route):
        client, agent = _client(max_input_chars=None)
        resp = _post(client, route, {"task": "x" * 200_000})
        assert resp.status_code == 200
        assert len(agent.calls) == 1

    @pytest.mark.parametrize("route", ROUTES)
    def test_default_limit_is_100000(self, route):
        client, agent = _client()
        ok = _post(client, route, {"task": "x" * (100_000 - 2)})
        assert ok.status_code == 200
        too_big = _post(client, route, {"task": "x" * (100_000 - 1)})
        assert too_big.status_code == 413
        assert len(agent.calls) == 1

    @pytest.mark.parametrize("route", ROUTES)
    def test_auth_checked_before_size(self, route):
        client, agent = _client(api_key=KEY, max_input_chars=self.LIMIT)
        resp = _post(client, route, {"task": "x" * 500})
        assert resp.status_code == 401
        assert agent.calls == []


class TestRemoteAgentToolAuth:
    @staticmethod
    def _handler(seen: list[httpx.Request]):
        def handle(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"answer": "ok"})

        return handle

    def _patch(self, monkeypatch, seen: list[httpx.Request]) -> None:
        transport = httpx.MockTransport(self._handler(seen))
        orig_client, orig_async = httpx.Client, httpx.AsyncClient
        monkeypatch.setattr(
            remote.httpx, "Client", lambda **kw: orig_client(transport=transport, **kw)
        )
        monkeypatch.setattr(
            remote.httpx,
            "AsyncClient",
            lambda **kw: orig_async(transport=transport, **kw),
        )

    def test_invoke_sends_bearer(self, monkeypatch):
        seen: list[httpx.Request] = []
        self._patch(monkeypatch, seen)
        tool = RemoteAgentTool(
            url="http://agent.test", name="r", description="d", api_key=KEY
        )
        assert tool.invoke("hi") == "ok"
        assert seen[0].headers["authorization"] == f"Bearer {KEY}"

    def test_ainvoke_sends_bearer(self, monkeypatch):
        seen: list[httpx.Request] = []
        self._patch(monkeypatch, seen)
        tool = RemoteAgentTool(
            url="http://agent.test", name="r", description="d", api_key=KEY
        )
        assert asyncio.run(tool.ainvoke("hi")) == "ok"
        assert seen[0].headers["authorization"] == f"Bearer {KEY}"

    def test_no_key_sends_no_header(self, monkeypatch):
        seen: list[httpx.Request] = []
        self._patch(monkeypatch, seen)
        tool = RemoteAgentTool(url="http://agent.test", name="r", description="d")
        tool.invoke("hi")
        asyncio.run(tool.ainvoke("hi"))
        assert all("authorization" not in r.headers for r in seen)

    def test_round_trip_against_server(self, monkeypatch):
        """The client's header is accepted by a keyed AgentServer."""
        agent = _StubAgent()
        server = AgentServer(agent=agent, api_key=KEY)
        transport = httpx.ASGITransport(app=server.app)
        orig_async = httpx.AsyncClient
        monkeypatch.setattr(
            remote.httpx,
            "AsyncClient",
            lambda **kw: orig_async(transport=transport, **kw),
        )
        good = RemoteAgentTool(
            url="http://agent.test", name="r", description="d", api_key=KEY
        )
        assert asyncio.run(good.ainvoke("hi")) == "done: hi"
        bad = RemoteAgentTool(url="http://agent.test", name="r", description="d")
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(bad.ainvoke("hi"))
        assert len(agent.calls) == 1
