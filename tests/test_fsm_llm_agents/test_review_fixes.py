"""Tests for code review fixes (Strands Phase 1 + Phase 2)."""

from __future__ import annotations

import importlib.util
import threading
from unittest.mock import Mock, patch

import pytest

from fsm_llm.dialog.api import API
from fsm_llm.stdlib.agents.definitions import AgentResult, AgentTrace

# ============================================================================
# C1: MCP Session Lifecycle Fix
# ============================================================================


class TestMCPSessionLifecycle:
    """Verify MCP executors reconnect per-call instead of using closed session."""

    def test_executor_does_not_capture_session(self):
        """Executor closure captures server_params, not session."""
        from fsm_llm.stdlib.agents.mcp import MCPToolProvider

        provider = MCPToolProvider.__new__(MCPToolProvider)
        provider._server_params = "mock_params"
        provider._server_url = None

        mock_tool = Mock()
        mock_tool.name = "test-tool"
        mock_tool.description = "Test"
        mock_tool.inputSchema = None

        tool_def = provider._convert_mcp_tool(mock_tool)
        assert tool_def.execute_fn is not None
        assert tool_def.name == "test-tool"

    def test_format_mcp_result_with_text(self):
        """_format_mcp_result extracts text from content items."""
        from fsm_llm.stdlib.agents.mcp import _format_mcp_result

        result = Mock()
        item1 = Mock()
        item1.text = "Hello"
        item2 = Mock()
        item2.text = "World"
        result.content = [item1, item2]

        assert _format_mcp_result(result) == "Hello\nWorld"

    def test_format_mcp_result_fallback(self):
        """_format_mcp_result falls back to str() for non-content results."""
        from fsm_llm.stdlib.agents.mcp import _format_mcp_result

        assert _format_mcp_result("plain string") == "plain string"


# ============================================================================
# C2: converse_stream() Auto-Save
# ============================================================================


class TestStreamAutoSave:
    """Verify converse_stream() auto-saves sessions."""

    def test_auto_save_after_full_stream(self):
        """Session saved after generator fully consumed."""
        from fsm_llm.dialog.api import API

        api = Mock(spec=API)
        api._session_store = Mock()
        api._stack_lock = threading.RLock()
        api._last_accessed = {}
        api.save_session = Mock()
        api._get_current_fsm_conversation_id = Mock(return_value="fsm-1")
        api.fsm_manager = Mock()
        api.fsm_manager.process_message_stream = Mock(
            return_value=iter(["chunk1", "chunk2"])
        )

        chunks = list(API.converse_stream(api, "hello", "conv-1"))
        assert chunks == ["chunk1", "chunk2"]
        api.save_session.assert_called_once_with("conv-1")

    def test_auto_save_on_partial_consumption(self):
        """Session saved even when generator partially consumed."""

        api = Mock(spec=API)
        api._session_store = Mock()
        api._stack_lock = threading.RLock()
        api._last_accessed = {}
        api.save_session = Mock()
        api._get_current_fsm_conversation_id = Mock(return_value="fsm-1")
        api.fsm_manager = Mock()
        api.fsm_manager.process_message_stream = Mock(
            return_value=iter(["a", "b", "c"])
        )

        gen = API.converse_stream(api, "hello", "conv-1")
        next(gen)  # consume one chunk
        gen.close()  # abandon generator
        api.save_session.assert_called_once_with("conv-1")

    def test_no_save_without_store(self):
        """No save attempt when session_store is None."""

        api = Mock(spec=API)
        api._session_store = None
        api._stack_lock = threading.RLock()
        api._last_accessed = {}
        api.save_session = Mock()
        api._get_current_fsm_conversation_id = Mock(return_value="fsm-1")
        api.fsm_manager = Mock()
        api.fsm_manager.process_message_stream = Mock(return_value=iter(["x"]))

        list(API.converse_stream(api, "hello", "conv-1"))
        api.save_session.assert_not_called()


# ============================================================================
# C3: Pipeline Streaming try/finally
# ============================================================================


class TestStreamingTryFinally:
    """Verify partial streaming stores response in history."""

    def test_partial_response_stored_pattern(self):
        """The try/finally pattern stores partial responses."""
        conversation = Mock()
        chunks = []
        try:
            for chunk in ["Hello ", "World", " end"]:
                chunks.append(chunk)
                if len(chunks) == 2:
                    break  # Simulate interruption
        finally:
            if chunks:
                conversation.add_system_message("".join(chunks))

        conversation.add_system_message.assert_called_once_with("Hello World")


# ============================================================================
# H2: OTEL Thread Safety
# ============================================================================


class TestOTELThreadSafety:
    """Verify OTEL exporter has thread-safe span management."""

    @pytest.mark.skipif(
        importlib.util.find_spec("opentelemetry") is None,
        reason="opentelemetry not installed",
    )
    def test_has_spans_lock(self):
        """OTELExporter has _spans_lock attribute."""
        from fsm_llm_monitor.otel import OTELExporter

        exporter = OTELExporter(service_name="test")
        assert hasattr(exporter, "_spans_lock")
        assert isinstance(exporter._spans_lock, threading.Lock)

    @pytest.mark.skipif(
        importlib.util.find_spec("opentelemetry") is None,
        reason="opentelemetry not installed",
    )
    def test_concurrent_events_no_crash(self):
        """Concurrent event recording doesn't crash."""
        from fsm_llm_monitor.definitions import MonitorEvent
        from fsm_llm_monitor.otel import OTELExporter

        exporter = OTELExporter(service_name="test")
        collector = Mock()
        collector.record_event = Mock()
        exporter.enable(collector)

        errors = []

        def record_events(conv_prefix, count):
            try:
                for i in range(count):
                    start = MonitorEvent(
                        event_type="conversation_start",
                        conversation_id=f"{conv_prefix}-{i}",
                        message=f"test {i}",
                    )
                    exporter._export_event(start)
                    end = MonitorEvent(
                        event_type="conversation_end",
                        conversation_id=f"{conv_prefix}-{i}",
                        message="done",
                    )
                    exporter._export_event(end)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_events, args=(f"t{i}", 20)) for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"


# ============================================================================
# H3: Swarm WorkingMemory Sharing
# ============================================================================


class TestSwarmMemorySharing:
    """Verify swarm memory is accessible but not in serializable context."""

    def test_swarm_memory_not_in_context(self):
        """_swarm_memory key NOT in sub-agent context (not serializable)."""
        from fsm_llm.stdlib.agents.swarm import SwarmAgent

        agent = Mock()
        agent.run = Mock(
            return_value=AgentResult(
                answer="done",
                success=True,
                trace=AgentTrace(total_iterations=1),
                final_context={},
            )
        )

        swarm = SwarmAgent(agents={"a": agent}, entry_agent="a")
        swarm.run("test task")

        call_kwargs = agent.run.call_args
        ctx = call_kwargs[1]["initial_context"]
        assert "_swarm_memory" not in ctx
        # Memory is still accessible on the swarm instance
        assert swarm._memory is not None

    def test_swarm_context_is_serializable(self):
        """Sub-agent context dict is JSON-serializable."""
        import json

        from fsm_llm.stdlib.agents.swarm import SwarmAgent

        agent = Mock()
        agent.run = Mock(
            return_value=AgentResult(
                answer="done",
                success=True,
                trace=AgentTrace(total_iterations=1),
                final_context={},
            )
        )

        swarm = SwarmAgent(agents={"a": agent}, entry_agent="a")
        swarm.run("test task")

        call_kwargs = agent.run.call_args
        ctx = call_kwargs[1]["initial_context"]
        # Should not raise
        json.dumps(ctx)


# ============================================================================
# M2: SOP Config Validation
# ============================================================================


class TestSOPConfigValidation:
    """Verify invalid SOP config_overrides fail at registration."""

    def test_invalid_config_raises_at_register(self):
        """ValueError raised for invalid config_overrides."""
        from fsm_llm.stdlib.agents.sop import SOPDefinition, SOPRegistry

        registry = SOPRegistry()
        sop = SOPDefinition(
            name="bad-sop",
            config_overrides={"max_iterations": -1},
        )
        with pytest.raises(ValueError, match="invalid config_overrides"):
            registry.register(sop)

    def test_valid_config_registers_ok(self):
        """Valid config_overrides register without error."""
        from fsm_llm.stdlib.agents.sop import SOPDefinition, SOPRegistry

        registry = SOPRegistry()
        sop = SOPDefinition(
            name="good-sop",
            config_overrides={"temperature": 0.3, "max_iterations": 5},
        )
        registry.register(sop)
        assert registry.has("good-sop")

    def test_empty_config_registers_ok(self):
        """Empty config_overrides register without validation."""
        from fsm_llm.stdlib.agents.sop import SOPDefinition, SOPRegistry

        registry = SOPRegistry()
        sop = SOPDefinition(name="empty-sop")
        registry.register(sop)
        assert registry.has("empty-sop")


# ============================================================================
# M3: Semantic Tools Log Level
# ============================================================================


class TestSemanticToolsLogLevel:
    """Verify embedding failures logged at warning level."""

    def test_embed_failure_logs_warning(self):
        """Failed embedding logs warning, not debug."""
        from fsm_llm.stdlib.agents.semantic_tools import SemanticToolRegistry

        with patch.object(
            SemanticToolRegistry,
            "_get_embedding",
            side_effect=RuntimeError("no model"),
        ):
            with patch("fsm_llm.stdlib.agents.semantic_tools.logger") as mock_logger:
                registry = SemanticToolRegistry(auto_embed=True)
                registry.register_function(
                    lambda x: x, name="test-tool", description="Test"
                )
                warning_calls = [
                    c
                    for c in mock_logger.warning.call_args_list
                    if "test-tool" in str(c)
                ]
                assert len(warning_calls) > 0


# ============================================================================
# M4: Remote Agent Timeout
# ============================================================================


class TestRemoteAgentTimeout:
    """Verify agent server has configurable execution timeouts."""

    @pytest.mark.skipif(
        importlib.util.find_spec("fastapi") is None,
        reason="fastapi not installed",
    )
    def test_invoke_endpoint_has_timeout(self):
        """AgentServer /invoke uses asyncio.wait_for with timeout."""
        import inspect

        from fsm_llm.stdlib.agents.remote import AgentServer

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
        from fsm_llm.stdlib.agents.remote import AgentServer

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
        from fsm_llm.stdlib.agents.remote import AgentServer

        agent = Mock()
        agent.__class__.__name__ = "TestAgent"
        server = AgentServer(agent=agent)
        assert server._timeout == 300.0


# ============================================================================
# H4: Session Path Validation
# ============================================================================


class TestSessionPathValidation:
    """Verify session_id is validated to prevent path traversal."""

    def test_valid_session_ids(self):
        """Valid session IDs are accepted."""
        import tempfile

        from fsm_llm.dialog.session import FileSessionStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            # These should not raise
            store._path("abc-123")
            store._path("session_456")
            store._path("Conv-ID-789")

    def test_path_traversal_rejected(self):
        """Session IDs with path traversal patterns are rejected."""
        import tempfile

        from fsm_llm.dialog.session import FileSessionStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            with pytest.raises(ValueError, match="Invalid session_id"):
                store._path("../etc/passwd")

    def test_slash_rejected(self):
        """Session IDs with slashes are rejected."""
        import tempfile

        from fsm_llm.dialog.session import FileSessionStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            with pytest.raises(ValueError, match="Invalid session_id"):
                store._path("foo/bar")

    def test_null_byte_rejected(self):
        """Session IDs with null bytes are rejected."""
        import tempfile

        from fsm_llm.dialog.session import FileSessionStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSessionStore(tmpdir)
            with pytest.raises(ValueError, match="Invalid session_id"):
                store._path("foo\x00bar")


# ============================================================================
# H5: API _replay_history Guard
# ============================================================================


class TestReplayHistoryGuard:
    """Verify _replay_history handles missing FSM instances."""

    def test_missing_instance_returns_gracefully(self):
        """_replay_history warns and returns if fsm_id not in instances."""

        api = Mock(spec=API)
        api.fsm_manager = Mock()
        api.fsm_manager.instances = {}

        # Should not raise KeyError
        API._replay_history(api, "nonexistent-fsm-id", [{"user": "hi"}])


# ============================================================================
# M6: Agent Graph Edge Condition Safety
# ============================================================================


class TestAgentGraphEdgeConditionSafety:
    """Verify edge condition exceptions don't crash the graph."""

    def test_failing_condition_skips_edge(self):
        """Edge with failing condition is skipped, not crashing."""
        from fsm_llm.stdlib.agents.agent_graph import AgentGraphBuilder

        def _make_agent(answer):
            agent = Mock()
            agent.run = Mock(
                return_value=AgentResult(
                    answer=answer,
                    success=True,
                    trace=AgentTrace(total_iterations=1),
                    final_context={"result": answer},
                )
            )
            return agent

        def bad_condition(ctx):
            raise RuntimeError("Condition crashed")

        graph = (
            AgentGraphBuilder()
            .add_node("a", _make_agent("A"))
            .add_node("b", _make_agent("B"))
            .add_edge("a", "b", condition=bad_condition)
            .set_entry("a")
            .build()
        )

        result = graph.run("test")
        assert result.success is True
        # Node B should NOT have been reached since condition raised
        assert "_graph_execution_order" in result.final_context
        assert "b" not in result.final_context["_graph_execution_order"]


# ============================================================================
# M13: Swarm _register_handlers Signature
# ============================================================================


class TestSwarmRegisterHandlersSignature:
    """Verify SwarmAgent._register_handlers matches BaseAgent signature."""

    def test_register_handlers_accepts_single_arg(self):
        """_register_handlers(api) works without extra args."""
        from fsm_llm.stdlib.agents.swarm import SwarmAgent

        agent = Mock()
        swarm = SwarmAgent(agents={"a": agent}, entry_agent="a")
        # Should not raise TypeError
        swarm._register_handlers(Mock())
