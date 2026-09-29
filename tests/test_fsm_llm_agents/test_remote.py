"""AgentServer opt-in API key and input size limit; RemoteAgentTool auth header.

Plan plan-2026-09-24T091842-c1d5bfbc, D-006.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from fsm_llm.agents import remote
from fsm_llm.agents.definitions import ToolCall
from fsm_llm.agents.exceptions import ToolExecutionError
from fsm_llm.agents.remote import AgentServer, RemoteAgentTool
from fsm_llm.agents.tools import ToolRegistry

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

    @pytest.mark.parametrize("empty", ["", "   ", "\t\n"])
    def test_empty_api_key_rejected_at_construction(self, empty):
        """W6: an empty key would authenticate an empty header (compare b"" to b"")."""
        with pytest.raises(ValueError, match="api_key"):
            AgentServer(agent=_StubAgent(), api_key=empty)

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
    def test_non_ascii_context_counted_as_characters(self, route):
        """A non-ASCII character counts once in context, as it does in task."""
        client, agent = _client(max_input_chars=self.LIMIT)
        context = {"k": "\u6f22" * 30}  # 39 chars; 189 with \uXXXX escapes
        task = "x" * (self.LIMIT - len(json.dumps(context, ensure_ascii=False)))
        resp = _post(client, route, {"task": task, "context": context})
        assert resp.status_code == 200
        over = _post(client, route, {"task": task + "x", "context": context})
        assert over.status_code == 413
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

    @pytest.mark.parametrize("empty", ["", "  "])
    def test_empty_api_key_rejected_at_construction(self, empty):
        """An empty key would send a bare ``Bearer `` header; fail loud instead."""
        with pytest.raises(ValueError, match="api_key"):
            RemoteAgentTool(
                url="http://a.test", name="r", description="d", api_key=empty
            )

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


class TestRemoteAgentToolSuccess:
    """TOOL-14: a server reporting ``success: false`` fails the tool call."""

    @staticmethod
    def _patch(monkeypatch, body: dict) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        orig_client, orig_async = httpx.Client, httpx.AsyncClient
        monkeypatch.setattr(
            remote.httpx, "Client", lambda **kw: orig_client(transport=transport, **kw)
        )
        monkeypatch.setattr(
            remote.httpx,
            "AsyncClient",
            lambda **kw: orig_async(transport=transport, **kw),
        )

    @staticmethod
    def _tool() -> RemoteAgentTool:
        return RemoteAgentTool(url="http://agent.test", name="billing", description="d")

    def test_invoke_raises_on_success_false(self, monkeypatch):
        self._patch(monkeypatch, {"answer": "gave up", "success": False})
        with pytest.raises(ToolExecutionError, match="gave up"):
            self._tool().invoke("hi")

    def test_ainvoke_raises_on_success_false(self, monkeypatch):
        self._patch(monkeypatch, {"answer": "gave up", "success": False})
        with pytest.raises(ToolExecutionError, match="billing"):
            asyncio.run(self._tool().ainvoke("hi"))

    def test_tool_definition_gives_a_failed_tool_result(self, monkeypatch):
        self._patch(monkeypatch, {"answer": "gave up", "success": False})
        registry = ToolRegistry()
        registry.register(self._tool().to_tool_definition())

        result = registry.execute(
            ToolCall(tool_name="billing", parameters={"task": "refund"})
        )

        assert result.success is False
        assert "gave up" in result.error

    @pytest.mark.parametrize(
        "body", [{"answer": "ok", "success": True}, {"answer": "ok"}]
    )
    def test_success_true_or_absent_returns_the_answer(self, monkeypatch, body):
        self._patch(monkeypatch, body)
        tool = self._tool()
        assert tool.invoke("hi") == "ok"
        assert asyncio.run(tool.ainvoke("hi")) == "ok"


class _FailingAgent:
    SECRET = "db password is hunter2 at /srv/internal/path"

    def run(self, task: str, initial_context: Any = None) -> Any:
        raise RuntimeError(self.SECRET)


class TestGenericErrorBody:
    """SEC-09: a raising agent's exception text never reaches the client."""

    @staticmethod
    def _capture_errors():
        from fsm_llm.logging import logger

        logger.enable("fsm_llm")
        captured: list[str] = []
        sink_id = logger.add(lambda m: captured.append(str(m)), level="ERROR")
        return logger, captured, sink_id

    def test_invoke_500_is_generic_and_logged(self):
        logger, captured, sink_id = self._capture_errors()
        try:
            client = TestClient(AgentServer(agent=_FailingAgent()).app)
            resp = client.post("/invoke", json={"task": "t"})
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")

        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "hunter2" not in resp.text
        assert detail.startswith("Agent execution failed (error_id=")
        error_id = detail.split("error_id=")[1].rstrip(")")
        # The exception is logged server-side under the same id.
        assert any(error_id in m and "hunter2" in m for m in captured)

    def test_stream_error_event_is_generic(self):
        client = TestClient(AgentServer(agent=_FailingAgent()).app)
        resp = client.post("/stream", json={"task": "t"})

        assert resp.status_code == 200
        assert "hunter2" not in resp.text
        event = json.loads(resp.text.strip().removeprefix("data: "))
        assert event["error"] == "Agent execution failed"
        assert event["error_id"]


class _BlockingAgent:
    """Holds every run in its worker thread until ``release`` is set."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def run(self, task: str, initial_context: Any = None) -> Any:
        self.calls += 1
        self.started.set()
        if not self.release.wait(10):
            raise RuntimeError("test agent was never released")
        return SimpleNamespace(
            answer=f"done: {task}",
            success=True,
            trace=SimpleNamespace(total_iterations=1, tools_used=[]),
        )


def _async_client(server: AgentServer) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://agent.test"
    )


class TestMaxConcurrent:
    """SEC-09: ``max_concurrent`` bounds agent runs in flight; overflow is 503."""

    def test_default_comes_from_constants(self):
        from fsm_llm.agents.constants import Defaults

        server = AgentServer(agent=_StubAgent())
        assert server._max_concurrent == Defaults.SERVER_MAX_CONCURRENT

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, None])
    def test_invalid_max_concurrent_rejected(self, bad):
        with pytest.raises(ValueError, match="max_concurrent"):
            AgentServer(agent=_StubAgent(), max_concurrent=bad)

    @pytest.mark.parametrize("route", ROUTES)
    def test_request_while_saturated_is_503(self, route):
        agent = _BlockingAgent()
        server = AgentServer(agent=agent, max_concurrent=1)

        async def scenario():
            async with _async_client(server) as client:
                first = asyncio.create_task(client.post("/invoke", json={"task": "a"}))
                assert await asyncio.to_thread(agent.started.wait, 5)
                second = await client.post(route, json={"task": "b"})
                agent.release.set()
                first_resp = await first
                third = await client.post(route, json={"task": "c"})
                return first_resp, second, third

        first_resp, second, third = asyncio.run(scenario())

        assert second.status_code == 503
        assert "busy" in second.json()["detail"]
        assert first_resp.status_code == 200
        # The slot came back, and the rejected request never ran the agent.
        assert third.status_code == 200
        assert agent.calls == 2

    def test_timed_out_run_keeps_its_slot_until_its_thread_ends(self):
        agent = _BlockingAgent()
        server = AgentServer(agent=agent, max_concurrent=1, timeout=0.05)

        async def scenario():
            async with _async_client(server) as client:
                timed_out = await client.post("/invoke", json={"task": "a"})
                while_orphaned = await client.post("/invoke", json={"task": "b"})
                agent.release.set()
                for _ in range(500):
                    if not server._slots.locked():
                        break
                    await asyncio.sleep(0.01)
                server._timeout = 5.0
                after = await client.post("/invoke", json={"task": "c"})
                return timed_out, while_orphaned, after

        timed_out, while_orphaned, after = asyncio.run(scenario())

        assert timed_out.status_code == 504
        assert while_orphaned.status_code == 503
        assert after.status_code == 200
        assert agent.calls == 2
