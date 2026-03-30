"""Tests for Strands Feature Adaptation Phase 2."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

from fsm_llm_agents.definitions import (
    AgentResult,
    AgentTrace,
)

# ============================================================================
# 1. MCP Tool Integration Tests
# ============================================================================


class TestMCPToolProvider:
    """Tests for MCPToolProvider."""

    def test_import_without_mcp(self):
        """MCPToolProvider module imports even without mcp SDK."""
        from fsm_llm_agents.mcp import MCPToolProvider

        assert MCPToolProvider is not None

    def test_create_mock_tool(self):
        """create_mock_tool creates valid ToolDefinitions."""
        from fsm_llm_agents.mcp import MCPToolProvider

        tool = MCPToolProvider.create_mock_tool(
            name="test-tool",
            description="A test tool",
            parameter_schema={"properties": {"query": {"type": "string"}}},
        )
        assert tool.name == "test-tool"
        assert tool.description == "A test tool"
        assert tool.execute_fn is not None

    def test_create_mock_tool_execution(self):
        """Mock tool executes and returns JSON."""
        from fsm_llm_agents.mcp import MCPToolProvider

        tool = MCPToolProvider.create_mock_tool(
            name="echo",
            description="Echo input",
            execute_fn=lambda **kwargs: f"echoed: {kwargs}",
        )
        result = tool.execute_fn(query="hello")
        assert "hello" in result

    def test_register_mock_tools_with_registry(self):
        """Mock tools can be registered with ToolRegistry."""
        from fsm_llm_agents.mcp import MCPToolProvider
        from fsm_llm_agents.tools import ToolRegistry

        provider = MCPToolProvider.__new__(MCPToolProvider)
        provider._tools = [
            MCPToolProvider.create_mock_tool("tool-a", "Tool A"),
            MCPToolProvider.create_mock_tool("tool-b", "Tool B"),
        ]

        registry = ToolRegistry()
        count = provider.register_tools(registry)
        assert count == 2
        assert "tool-a" in registry
        assert "tool-b" in registry

    def test_schema_conversion(self):
        """MCP input schema converts to ToolDefinition parameter_schema."""
        from fsm_llm_agents.mcp import _mcp_schema_to_parameter_schema

        schema = {
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        result = _mcp_schema_to_parameter_schema(schema)
        assert result["properties"]["query"]["type"] == "string"
        assert "query" in result["required"]

    def test_empty_schema_conversion(self):
        """Empty schema converts to empty dict."""
        from fsm_llm_agents.mcp import _mcp_schema_to_parameter_schema

        assert _mcp_schema_to_parameter_schema({}) == {}
        assert _mcp_schema_to_parameter_schema(None) == {}

    def test_get_tool_names(self):
        """get_tool_names returns names of discovered tools."""
        from fsm_llm_agents.mcp import MCPToolProvider

        provider = MCPToolProvider.__new__(MCPToolProvider)
        provider._tools = [
            MCPToolProvider.create_mock_tool("alpha", "Alpha tool"),
            MCPToolProvider.create_mock_tool("beta", "Beta tool"),
        ]
        assert provider.get_tool_names() == ["alpha", "beta"]


# ============================================================================
# 2. Swarm Pattern Tests
# ============================================================================


class TestSwarmAgent:
    """Tests for SwarmAgent."""

    def _make_mock_agent(self, answer: str, context: dict | None = None):
        """Create a mock agent that returns a fixed answer."""
        agent = Mock()
        agent.run = Mock(
            return_value=AgentResult(
                answer=answer,
                success=True,
                trace=AgentTrace(total_iterations=1),
                final_context=context or {},
            )
        )
        return agent

    def test_basic_swarm_no_handoff(self):
        """Swarm with single agent, no handoff."""
        from fsm_llm_agents.swarm import SwarmAgent

        agent_a = self._make_mock_agent("Done!", {})
        swarm = SwarmAgent(
            agents={"a": agent_a},
            entry_agent="a",
        )
        result = swarm.run("test task")
        assert result.success
        assert result.answer == "Done!"
        assert result.final_context["_swarm_handoff_count"] == 0

    def test_swarm_handoff(self):
        """Swarm with handoff from agent A to agent B."""
        from fsm_llm_agents.swarm import SwarmAgent

        agent_a = self._make_mock_agent(
            "Routing to billing",
            {"next_agent": "b", "handoff_message": "Handle billing"},
        )
        agent_b = self._make_mock_agent("Billing resolved", {})

        swarm = SwarmAgent(
            agents={"a": agent_a, "b": agent_b},
            entry_agent="a",
        )
        result = swarm.run("billing issue")
        assert result.success
        assert result.answer == "Billing resolved"
        assert result.final_context["_swarm_handoff_count"] == 1
        assert result.final_context["_swarm_handoff_chain"] == ["a", "b"]

    def test_swarm_max_handoffs(self):
        """Swarm stops at max_handoffs."""
        from fsm_llm_agents.swarm import SwarmAgent

        # Agent always hands off to itself
        agent_a = self._make_mock_agent(
            "Loop",
            {"next_agent": "a"},
        )

        swarm = SwarmAgent(
            agents={"a": agent_a},
            entry_agent="a",
            max_handoffs=3,
        )
        result = swarm.run("loop task")
        assert result.final_context["_swarm_handoff_count"] >= 3

    def test_swarm_invalid_handoff_target(self):
        """Swarm stops when handoff target doesn't exist."""
        from fsm_llm_agents.swarm import SwarmAgent

        agent_a = self._make_mock_agent(
            "Go to unknown",
            {"next_agent": "nonexistent"},
        )
        swarm = SwarmAgent(
            agents={"a": agent_a},
            entry_agent="a",
        )
        result = swarm.run("test")
        assert result.answer == "Go to unknown"

    def test_swarm_empty_agents_raises(self):
        """Empty agents dict raises ValueError."""
        from fsm_llm_agents.swarm import SwarmAgent

        with pytest.raises(ValueError, match="at least one agent"):
            SwarmAgent(agents={}, entry_agent="a")

    def test_swarm_invalid_entry_raises(self):
        """Invalid entry agent raises ValueError."""
        from fsm_llm_agents.swarm import SwarmAgent

        with pytest.raises(ValueError, match="not found"):
            SwarmAgent(
                agents={"a": Mock()},
                entry_agent="nonexistent",
            )

    def test_swarm_agent_failure(self):
        """Swarm handles agent failure gracefully."""
        from fsm_llm_agents.swarm import SwarmAgent

        agent_a = Mock()
        agent_a.run = Mock(side_effect=RuntimeError("Agent crashed"))

        swarm = SwarmAgent(agents={"a": agent_a}, entry_agent="a")
        result = swarm.run("test")
        assert not result.success
        assert "crashed" in result.answer

    def test_swarm_add_agent(self):
        """add_agent() adds agents to swarm."""
        from fsm_llm_agents.swarm import SwarmAgent

        agent_a = self._make_mock_agent("A", {})
        agent_b = self._make_mock_agent("B", {})

        swarm = SwarmAgent(agents={"a": agent_a}, entry_agent="a")
        swarm.add_agent("b", agent_b)
        assert "b" in swarm.agents


# ============================================================================
# 3. Agent Graph Tests
# ============================================================================


class TestAgentGraph:
    """Tests for AgentGraphBuilder and AgentGraph."""

    def _make_mock_agent(self, answer: str, context: dict | None = None):
        agent = Mock()
        agent.run = Mock(
            return_value=AgentResult(
                answer=answer,
                success=True,
                trace=AgentTrace(total_iterations=1),
                final_context=context or {},
            )
        )
        return agent

    def test_linear_graph(self):
        """Linear graph: A -> B -> C."""
        from fsm_llm_agents.agent_graph import AgentGraphBuilder

        a = self._make_mock_agent("A done", {"step": "a"})
        b = self._make_mock_agent("B done", {"step": "b"})
        c = self._make_mock_agent("C done", {"step": "c"})

        graph = (
            AgentGraphBuilder()
            .add_node("a", a)
            .add_node("b", b)
            .add_node("c", c)
            .add_edge("a", "b")
            .add_edge("b", "c")
            .set_entry("a")
            .build()
        )
        result = graph.run("test")
        assert result.success
        assert result.final_context["_graph_execution_order"] == ["a", "b", "c"]

    def test_branching_graph(self):
        """Branching graph with conditions."""
        from fsm_llm_agents.agent_graph import AgentGraphBuilder

        classifier = self._make_mock_agent("classified", {"intent": "billing"})
        billing = self._make_mock_agent("billing handled", {})
        support = self._make_mock_agent("support handled", {})

        graph = (
            AgentGraphBuilder()
            .add_node("classifier", classifier)
            .add_node("billing", billing)
            .add_node("support", support)
            .add_edge(
                "classifier",
                "billing",
                condition=lambda ctx: ctx.get("intent") == "billing",
            )
            .add_edge(
                "classifier",
                "support",
                condition=lambda ctx: ctx.get("intent") == "support",
            )
            .set_entry("classifier")
            .build()
        )
        result = graph.run("invoice help")
        assert result.success
        order = result.final_context["_graph_execution_order"]
        assert "classifier" in order
        assert "billing" in order
        assert "support" not in order

    def test_cycle_detection(self):
        """Cycles are detected and rejected."""
        from fsm_llm_agents.agent_graph import AgentGraphBuilder

        a = self._make_mock_agent("A", {})
        b = self._make_mock_agent("B", {})

        with pytest.raises(ValueError, match="cycle"):
            (
                AgentGraphBuilder()
                .add_node("a", a)
                .add_node("b", b)
                .add_edge("a", "b")
                .add_edge("b", "a")
                .set_entry("a")
                .build()
            )

    def test_missing_entry(self):
        """Missing entry node raises ValueError."""
        from fsm_llm_agents.agent_graph import AgentGraphBuilder

        with pytest.raises(ValueError, match="Entry node must be set"):
            AgentGraphBuilder().add_node("a", Mock()).build()

    def test_invalid_edge_source(self):
        """Edge referencing non-existent source raises ValueError."""
        from fsm_llm_agents.agent_graph import AgentGraphBuilder

        with pytest.raises(ValueError, match="not in nodes"):
            (
                AgentGraphBuilder()
                .add_node("a", Mock())
                .add_edge("nonexistent", "a")
                .set_entry("a")
                .build()
            )

    def test_terminal_nodes(self):
        """get_terminal_nodes returns nodes with no outgoing edges."""
        from fsm_llm_agents.agent_graph import AgentGraphBuilder

        a = self._make_mock_agent("A", {})
        b = self._make_mock_agent("B", {})

        graph = (
            AgentGraphBuilder()
            .add_node("a", a)
            .add_node("b", b)
            .add_edge("a", "b")
            .set_entry("a")
            .build()
        )
        assert graph.get_terminal_nodes() == ["b"]

    def test_node_failure_propagation(self):
        """Failed node still produces result."""
        from fsm_llm_agents.agent_graph import AgentGraphBuilder

        a = Mock()
        a.run = Mock(side_effect=RuntimeError("boom"))

        graph = AgentGraphBuilder().add_node("a", a).set_entry("a").build()
        result = graph.run("test")
        assert not result.success


# ============================================================================
# 4. OTEL Observability Tests
# ============================================================================


def _has_otel():
    import importlib.util

    return importlib.util.find_spec("opentelemetry") is not None


def _has_httpx():
    import importlib.util

    return importlib.util.find_spec("httpx") is not None


def _has_fastapi():
    import importlib.util

    return importlib.util.find_spec("fastapi") is not None


class TestOTELExporter:
    """Tests for OTELExporter."""

    def test_import_without_otel(self):
        """OTELExporter module imports even without opentelemetry."""
        from fsm_llm_monitor.otel import OTELExporter

        assert OTELExporter is not None

    @pytest.mark.skipif(
        not _has_otel(),
        reason="opentelemetry not installed",
    )
    def test_exporter_initialization(self):
        """OTELExporter initializes with service name."""
        from fsm_llm_monitor.otel import OTELExporter

        exporter = OTELExporter(service_name="test-service")
        assert not exporter.is_enabled
        assert exporter.active_conversations == []

    @pytest.mark.skipif(
        not _has_otel(),
        reason="opentelemetry not installed",
    )
    def test_enable_disable(self):
        """Enable and disable OTEL export."""
        from fsm_llm_monitor.otel import OTELExporter

        collector = Mock()
        collector.record_event = Mock()

        exporter = OTELExporter(service_name="test")
        exporter.enable(collector)
        assert exporter.is_enabled

        exporter.disable()
        assert not exporter.is_enabled

    def test_generate_trace_id(self):
        """generate_trace_id returns a hex string."""
        from fsm_llm_monitor.otel import OTELExporter

        trace_id = OTELExporter.generate_trace_id()
        assert isinstance(trace_id, str)
        assert len(trace_id) == 32

    def test_generate_span_id(self):
        """generate_span_id returns a hex string."""
        from fsm_llm_monitor.otel import OTELExporter

        span_id = OTELExporter.generate_span_id()
        assert isinstance(span_id, str)
        assert len(span_id) == 16


# ============================================================================
# 5. Dependency Resolver Tests
# ============================================================================


class TestDependencyResolver:
    """Tests for DependencyResolver."""

    def test_simple_linear_deps(self):
        """Linear A -> B -> C produces 3 waves."""
        from fsm_llm_workflows.dependency_resolver import DependencyResolver

        resolver = DependencyResolver()
        resolver.add_step("a")
        resolver.add_step("b", depends_on=["a"])
        resolver.add_step("c", depends_on=["b"])

        waves = resolver.resolve()
        assert waves == [["a"], ["b"], ["c"]]

    def test_parallel_steps(self):
        """Independent steps are in the same wave."""
        from fsm_llm_workflows.dependency_resolver import DependencyResolver

        resolver = DependencyResolver()
        resolver.add_step("a")
        resolver.add_step("b")
        resolver.add_step("c", depends_on=["a", "b"])

        waves = resolver.resolve()
        assert waves[0] == ["a", "b"]  # Parallel
        assert waves[1] == ["c"]

    def test_diamond_dependency(self):
        """Diamond: A -> B,C -> D."""
        from fsm_llm_workflows.dependency_resolver import DependencyResolver

        resolver = DependencyResolver.from_dict(
            {
                "a": [],
                "b": ["a"],
                "c": ["a"],
                "d": ["b", "c"],
            }
        )
        waves = resolver.resolve()
        assert waves[0] == ["a"]
        assert waves[1] == ["b", "c"]
        assert waves[2] == ["d"]

    def test_cycle_detection(self):
        """Cycles raise WorkflowValidationError."""
        from fsm_llm_workflows.dependency_resolver import DependencyResolver
        from fsm_llm_workflows.exceptions import WorkflowValidationError

        resolver = DependencyResolver()
        resolver.add_step("a", depends_on=["b"])
        resolver.add_step("b", depends_on=["a"])

        with pytest.raises(WorkflowValidationError):
            resolver.resolve()

    def test_has_cycles(self):
        """has_cycles returns True for cyclic graph."""
        from fsm_llm_workflows.dependency_resolver import DependencyResolver

        resolver = DependencyResolver()
        resolver.add_step("a", depends_on=["b"])
        resolver.add_step("b", depends_on=["a"])
        assert resolver.has_cycles()

    def test_unknown_dependency_raises(self):
        """Dependency on non-existent step raises error."""
        from fsm_llm_workflows.dependency_resolver import DependencyResolver
        from fsm_llm_workflows.exceptions import WorkflowValidationError

        resolver = DependencyResolver()
        resolver.add_step("a", depends_on=["nonexistent"])

        with pytest.raises(WorkflowValidationError, match="unknown"):
            resolver.resolve()

    def test_from_dict(self):
        """from_dict creates resolver correctly."""
        from fsm_llm_workflows.dependency_resolver import DependencyResolver

        resolver = DependencyResolver.from_dict(
            {
                "fetch": [],
                "process": ["fetch"],
            }
        )
        assert resolver.step_count == 2
        assert resolver.dependency_count == 1

    def test_clear(self):
        """clear() removes all state."""
        from fsm_llm_workflows.dependency_resolver import DependencyResolver

        resolver = DependencyResolver()
        resolver.add_step("a")
        resolver.clear()
        assert resolver.step_count == 0

    def test_single_step(self):
        """Single step with no deps produces one wave."""
        from fsm_llm_workflows.dependency_resolver import DependencyResolver

        resolver = DependencyResolver()
        resolver.add_step("only")
        waves = resolver.resolve()
        assert waves == [["only"]]


# ============================================================================
# 6. Agent SOP Tests
# ============================================================================


class TestSOPRegistry:
    """Tests for SOPDefinition and SOPRegistry."""

    def test_sop_definition_creation(self):
        """SOPDefinition stores all fields."""
        from fsm_llm_agents.sop import SOPDefinition

        sop = SOPDefinition(
            name="test-sop",
            description="A test SOP",
            agent_pattern="react",
            task_template="Review: {code}",
        )
        assert sop.name == "test-sop"
        assert sop.agent_pattern == "react"

    def test_render_task(self):
        """render_task substitutes template variables."""
        from fsm_llm_agents.sop import SOPDefinition

        sop = SOPDefinition(
            name="review",
            task_template="Review this {language} code: {code}",
        )
        result = sop.render_task(language="Python", code="x=1")
        assert "Python" in result
        assert "x=1" in result

    def test_render_task_missing_var(self):
        """render_task raises on missing variable."""
        from fsm_llm_agents.sop import SOPDefinition

        sop = SOPDefinition(name="test", task_template="{missing}")
        with pytest.raises(ValueError, match="missing"):
            sop.render_task()

    def test_to_agent_config(self):
        """to_agent_config creates AgentConfig from SOP."""
        from fsm_llm_agents.sop import SOPDefinition

        sop = SOPDefinition(
            name="test",
            config_overrides={"temperature": 0.1, "max_iterations": 3},
        )
        config = sop.to_agent_config()
        assert config.temperature == 0.1
        assert config.max_iterations == 3

    def test_sop_serialization(self):
        """SOPDefinition roundtrips through dict."""
        from fsm_llm_agents.sop import SOPDefinition

        sop = SOPDefinition(name="test", description="Test SOP")
        data = sop.to_dict()
        restored = SOPDefinition.from_dict(data)
        assert restored.name == "test"
        assert restored.description == "Test SOP"

    def test_registry_register_and_get(self):
        """Registry stores and retrieves SOPs."""
        from fsm_llm_agents.sop import SOPDefinition, SOPRegistry

        registry = SOPRegistry()
        sop = SOPDefinition(name="my-sop", description="Test")
        registry.register(sop)

        assert registry.has("my-sop")
        assert registry.get("my-sop").description == "Test"

    def test_registry_list(self):
        """list_names returns sorted SOP names."""
        from fsm_llm_agents.sop import SOPDefinition, SOPRegistry

        registry = SOPRegistry()
        registry.register(SOPDefinition(name="beta"))
        registry.register(SOPDefinition(name="alpha"))
        assert registry.list_names() == ["alpha", "beta"]

    def test_registry_missing_raises(self):
        """get() raises KeyError for missing SOP."""
        from fsm_llm_agents.sop import SOPRegistry

        registry = SOPRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_load_from_json_file(self, tmp_path):
        """register_from_file loads JSON SOP."""
        from fsm_llm_agents.sop import SOPRegistry

        sop_file = tmp_path / "test.json"
        sop_file.write_text(
            json.dumps(
                {
                    "name": "json-sop",
                    "description": "From JSON",
                    "agent_pattern": "react",
                }
            )
        )

        registry = SOPRegistry()
        registry.register_from_file(sop_file)
        assert registry.has("json-sop")

    def test_load_builtin_sops(self):
        """load_builtin_sops returns registry with 3 SOPs."""
        from fsm_llm_agents.sop import load_builtin_sops

        registry = load_builtin_sops()
        assert len(registry) == 3
        assert registry.has("code-review")
        assert registry.has("summarize")
        assert registry.has("data-extraction")

    def test_registry_remove(self):
        """remove() deletes SOP and returns True."""
        from fsm_llm_agents.sop import SOPDefinition, SOPRegistry

        registry = SOPRegistry()
        registry.register(SOPDefinition(name="temp"))
        assert registry.remove("temp")
        assert not registry.has("temp")
        assert not registry.remove("temp")  # Already removed


# ============================================================================
# 7. Semantic Tool Retrieval Tests
# ============================================================================


class TestSemanticToolRegistry:
    """Tests for SemanticToolRegistry."""

    def test_inherits_tool_registry(self):
        """SemanticToolRegistry is a ToolRegistry subclass."""
        from fsm_llm_agents.semantic_tools import SemanticToolRegistry
        from fsm_llm_agents.tools import ToolRegistry

        assert issubclass(SemanticToolRegistry, ToolRegistry)

    def test_register_without_embedding(self):
        """Tools register even when embedding fails."""
        from fsm_llm_agents.semantic_tools import SemanticToolRegistry

        registry = SemanticToolRegistry(auto_embed=False)
        registry.register_function(lambda x: x, name="test", description="Test tool")
        assert "test" in registry
        assert registry.embedded_tool_count == 0

    def test_fallback_for_small_registry(self):
        """Small registries return all tools without embedding."""
        from fsm_llm_agents.semantic_tools import SemanticToolRegistry

        registry = SemanticToolRegistry(auto_embed=False)
        for i in range(5):
            registry.register_function(
                lambda x: x, name=f"tool-{i}", description=f"Tool {i}"
            )
        result = registry.retrieve("query")
        assert len(result) == 5  # All tools returned

    def test_cosine_similarity(self):
        """Cosine similarity computation is correct."""
        from fsm_llm_agents.semantic_tools import _cosine_similarity

        assert _cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
        assert _cosine_similarity([0, 0], [1, 1]) == pytest.approx(0.0)

    @patch("fsm_llm_agents.semantic_tools.SemanticToolRegistry._get_embedding")
    def test_retrieve_with_mock_embeddings(self, mock_embed):
        """Semantic retrieval returns top-K tools by similarity."""
        from fsm_llm_agents.semantic_tools import SemanticToolRegistry

        # Set up mock embeddings
        embeddings = {
            "search": [1.0, 0.0, 0.0],
            "calculate": [0.0, 1.0, 0.0],
            "translate": [0.0, 0.0, 1.0],
        }
        call_count = [0]

        def side_effect(text):
            call_count[0] += 1
            for name, emb in embeddings.items():
                if name in text.lower():
                    return emb
            # Query embedding - assume it's about search
            return [0.9, 0.1, 0.0]

        mock_embed.side_effect = side_effect

        registry = SemanticToolRegistry(auto_embed=True)
        registry.FALLBACK_THRESHOLD = 0  # Force embedding path

        for name in ["search", "calculate", "translate"]:
            registry.register_function(
                lambda x: x, name=name, description=f"{name.title()} tool"
            )

        # Register enough tools to bypass fallback
        result = registry.retrieve("find information", top_k=2)
        assert len(result) <= 2
        assert result[0].name == "search"  # Most similar

    def test_prompt_description_without_query(self):
        """to_prompt_description without query returns all tools."""
        from fsm_llm_agents.semantic_tools import SemanticToolRegistry

        registry = SemanticToolRegistry(auto_embed=False)
        registry.register_function(lambda: None, name="tool-a", description="Tool A")
        desc = registry.to_prompt_description()
        assert "tool-a" in desc


# ============================================================================
# 8. A2A Protocol Tests
# ============================================================================


class TestRemoteAgentTool:
    """Tests for RemoteAgentTool."""

    def test_import_without_httpx(self):
        """RemoteAgentTool module imports even without httpx."""
        from fsm_llm_agents.remote import RemoteAgentTool

        assert RemoteAgentTool is not None

    @pytest.mark.skipif(
        not _has_httpx(),
        reason="httpx not installed",
    )
    def test_tool_definition_creation(self):
        """to_tool_definition creates valid ToolDefinition."""
        from fsm_llm_agents.remote import RemoteAgentTool

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
        from fsm_llm_agents.remote import RemoteAgentTool

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
        from fsm_llm_agents.remote import RemoteAgentTool

        tool = RemoteAgentTool(url="http://example.com/", name="t", description="t")
        assert tool.url == "http://example.com"


class TestAgentServer:
    """Tests for AgentServer."""

    def test_import_without_fastapi(self):
        """AgentServer module imports even without fastapi."""
        from fsm_llm_agents.remote import AgentServer

        assert AgentServer is not None

    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_server_creates_app(self):
        """AgentServer creates a FastAPI app."""
        from fsm_llm_agents.remote import AgentServer

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
        from fsm_llm_agents.remote import AgentServer

        agent = Mock()
        agent.__class__.__name__ = "TestAgent"
        server = AgentServer(agent=agent)

        routes = [r.path for r in server.app.routes]
        assert "/invoke" in routes
        assert "/stream" in routes
        assert "/health" in routes
        assert "/info" in routes


# ============================================================================
# Integration: Exports
# ============================================================================


class TestPhase2Exports:
    """Test that all Phase 2 classes are exported correctly."""

    def test_agents_exports(self):
        """All Phase 2 agent classes are importable from fsm_llm_agents."""
        import fsm_llm_agents as m

        assert hasattr(m, "SwarmAgent")
        assert hasattr(m, "AgentGraph")
        assert hasattr(m, "AgentGraphBuilder")
        assert hasattr(m, "MCPToolProvider")
        assert hasattr(m, "SOPDefinition")
        assert hasattr(m, "SOPRegistry")
        assert hasattr(m, "load_builtin_sops")
        assert hasattr(m, "SemanticToolRegistry")
        assert hasattr(m, "AgentServer")
        assert hasattr(m, "RemoteAgentTool")

    def test_workflows_exports(self):
        """DependencyResolver is importable from fsm_llm_workflows."""
        from fsm_llm_workflows import DependencyResolver

        assert DependencyResolver is not None

    def test_monitor_exports(self):
        """OTELExporter is importable from fsm_llm_monitor."""
        from fsm_llm_monitor import OTELExporter

        assert OTELExporter is not None

    def test_swarm_in_create_agent_patterns(self):
        """SwarmAgent is available via create_agent pattern list."""
        from fsm_llm_agents import create_agent

        with pytest.raises(TypeError):
            # SwarmAgent needs 'agents' kwarg, so this will fail,
            # but it proves the pattern is recognized
            create_agent(pattern="swarm")
