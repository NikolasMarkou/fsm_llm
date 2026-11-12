"""AgentServer and RemoteAgentTool without fastapi/httpx at import time, plus the invoke timeout.

``test_remote.py`` skips its whole module when fastapi or httpx is missing; these classes keep the
per-test skips they always had (the import checks must run without the extras).
"""

from __future__ import annotations

import importlib.util
from unittest.mock import Mock

import pytest


class TestRemoteAgentTimeout:
    """Verify agent server has configurable execution timeouts."""

    @pytest.mark.skipif(
        importlib.util.find_spec("fastapi") is None,
        reason="fastapi not installed",
    )
    def test_invoke_endpoint_has_timeout(self):
        """AgentServer /invoke uses asyncio.wait_for with timeout."""
        import inspect

        from fsm_llm.agents.remote import AgentServer

        agent = Mock()
        agent.__class__.__name__ = "TestAgent"
        server = AgentServer(agent=agent)

        invoke_routes = [
            r for r in server.app.routes if hasattr(r, "path") and r.path == "/invoke"
        ]
        assert len(invoke_routes) == 1

        source = inspect.getsource(invoke_routes[0].endpoint)
        assert "wait_for" in source
        assert "timeout" in source

    @pytest.mark.skipif(
        importlib.util.find_spec("fastapi") is None,
        reason="fastapi not installed",
    )
    def test_timeout_is_configurable(self):
        """AgentServer accepts custom timeout parameter."""
        from fsm_llm.agents.remote import AgentServer

        agent = Mock()
        agent.__class__.__name__ = "TestAgent"
        server = AgentServer(agent=agent, timeout=600.0)
        assert server._timeout == 600.0

    @pytest.mark.skipif(
        importlib.util.find_spec("fastapi") is None,
        reason="fastapi not installed",
    )
    def test_default_timeout_is_300(self):
        """AgentServer default timeout is 300 seconds."""
        from fsm_llm.agents.remote import AgentServer

        agent = Mock()
        agent.__class__.__name__ = "TestAgent"
        server = AgentServer(agent=agent)
        assert server._timeout == 300.0


def _has_httpx():
    import importlib.util

    return importlib.util.find_spec("httpx") is not None


def _has_fastapi():
    import importlib.util

    return importlib.util.find_spec("fastapi") is not None


class TestRemoteAgentTool:
    """Tests for RemoteAgentTool."""

    def test_import_without_httpx(self):
        """RemoteAgentTool module imports even without httpx."""
        from fsm_llm.agents.remote import RemoteAgentTool

        assert RemoteAgentTool is not None

    @pytest.mark.skipif(
        not _has_httpx(),
        reason="httpx not installed",
    )
    def test_tool_definition_creation(self):
        """to_tool_definition creates valid ToolDefinition."""
        from fsm_llm.agents.remote import RemoteAgentTool

        tool = RemoteAgentTool(
            url="http://localhost:8500",
            name="remote-agent",
            description="A remote agent",
        )
        td = tool.to_tool_definition()
        assert td.name == "remote-agent"
        assert td.description == "A remote agent"
        assert "task" in td.parameter_schema["properties"]

    @pytest.mark.skipif(
        not _has_httpx(),
        reason="httpx not installed",
    )
    def test_health_check_failure(self):
        """health_check returns False when server unreachable."""
        from fsm_llm.agents.remote import RemoteAgentTool

        tool = RemoteAgentTool(
            url="http://localhost:99999",
            name="unreachable",
            description="Unreachable agent",
        )
        assert not tool.health_check()

    @pytest.mark.skipif(
        not _has_httpx(),
        reason="httpx not installed",
    )
    def test_url_property(self):
        """url property returns configured URL."""
        from fsm_llm.agents.remote import RemoteAgentTool

        tool = RemoteAgentTool(url="http://example.com/", name="t", description="t")
        assert tool.url == "http://example.com"


class TestAgentServer:
    """Tests for AgentServer."""

    def test_import_without_fastapi(self):
        """AgentServer module imports even without fastapi."""
        from fsm_llm.agents.remote import AgentServer

        assert AgentServer is not None

    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_server_creates_app(self):
        """AgentServer creates a FastAPI app."""
        from fsm_llm.agents.remote import AgentServer

        agent = Mock()
        agent.__class__.__name__ = "MockAgent"
        server = AgentServer(agent=agent, port=9999)
        assert server.app is not None

    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_server_has_endpoints(self):
        """AgentServer app has /invoke, /stream, /health, /info routes."""
        from fsm_llm.agents.remote import AgentServer

        agent = Mock()
        agent.__class__.__name__ = "TestAgent"
        server = AgentServer(agent=agent)

        routes = [r.path for r in server.app.routes]
        assert "/invoke" in routes
        assert "/stream" in routes
        assert "/health" in routes
        assert "/info" in routes
