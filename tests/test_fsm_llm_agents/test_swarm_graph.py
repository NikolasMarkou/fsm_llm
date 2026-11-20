"""SwarmAgent and AgentGraph: hand-offs, shared memory, iteration counts, edge conditions.

``TestSwarmAgent`` and ``TestAgentGraph`` are the base coverage of both patterns; the other classes
pin review findings (F-AI-01 empty caller containers, F-AI-07 iteration counts, memory sharing,
edge-condition safety, the ``_register_handlers`` signature).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from fsm_llm.agents import PythonToolLoader, ToolRegistry
from fsm_llm.agents.definitions import AgentResult, AgentTrace
from fsm_llm.agents.swarm import SwarmAgent
from fsm_llm.core.codes import Status
from fsm_llm.memory import WorkingMemory
from fsm_llm.tools import ToolCall


def _agent(
    answer: str = "done",
    *,
    iterations: int = 1,
    calls: int = 0,
    context: dict[str, Any] | None = None,
) -> Mock:
    agent = Mock()
    agent.run = Mock(
        return_value=AgentResult(
            answer=answer,
            status=Status.OK,
            trace=AgentTrace(
                total_iterations=iterations,
                tool_calls=[ToolCall(tool_name="t", parameters={})] * calls,
            ),
            final_context=context or {},
        )
    )
    return agent


class TestEmptyCallerContainerIsKept:
    """F-AI-01: ``x or Default()`` replaced an empty (falsy) caller-owned one."""

    def test_swarm_writes_to_an_empty_caller_memory(self):
        memory = WorkingMemory()
        swarm = SwarmAgent(agents={"a": _agent()}, entry_agent="a", memory=memory)
        swarm.run("task")
        assert swarm._memory is memory
        assert memory.get("metadata", "current_agent") == "a"

    def test_to_tool_registry_fills_an_empty_caller_registry(self):
        def echo(text: str) -> str:
            """Echo."""
            return text

        from fsm_llm.agents import tool

        tools = PythonToolLoader.from_functions(tool(echo))
        registry = ToolRegistry()
        returned = PythonToolLoader.to_tool_registry(tools, registry=registry)
        assert returned is registry
        assert len(registry) == 1


class TestSwarmIterationsAgreeWithGraph:
    """F-AI-07: Swarm summed agents run, AgentGraph sums member iterations."""

    def test_total_iterations_sums_member_iterations(self):
        a = _agent("one", iterations=5, context={"next_agent": "b"})
        b = _agent("two", iterations=5)
        result = SwarmAgent(agents={"a": a, "b": b}, entry_agent="a").run("task")
        assert result.trace.total_iterations == 10

    def test_failure_keeps_what_already_ran(self):
        a = _agent("one", iterations=4, calls=1, context={"next_agent": "b"})
        b = Mock()
        b.run = Mock(side_effect=RuntimeError("boom"))
        result = SwarmAgent(agents={"a": a, "b": b}, entry_agent="a").run("task")
        assert result.success is False
        assert result.trace.total_iterations == 4
        assert len(result.trace.tool_calls) == 1
        assert [t["agent"] for t in result.final_context["_swarm_traces"]] == ["a"]
        assert result.final_context["_swarm_handoff_count"] == 1


class TestSwarmMemorySharing:
    """Verify swarm memory is accessible but not in serializable context."""

    def test_swarm_memory_not_in_context(self):
        """_swarm_memory key NOT in sub-agent context (not serializable)."""
        from fsm_llm.agents.swarm import SwarmAgent

        agent = Mock()
        agent.run = Mock(
            return_value=AgentResult(
                answer="done",
                status=Status.OK,
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

        from fsm_llm.agents.swarm import SwarmAgent

        agent = Mock()
        agent.run = Mock(
            return_value=AgentResult(
                answer="done",
                status=Status.OK,
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


class TestAgentGraphEdgeConditionSafety:
    """Verify edge condition exceptions don't crash the graph."""

    def test_failing_condition_skips_edge(self):
        """Edge with failing condition is skipped, not crashing."""
        from fsm_llm.agents.agent_graph import AgentGraphBuilder

        def _make_agent(answer):
            agent = Mock()
            agent.run = Mock(
                return_value=AgentResult(
                    answer=answer,
                    status=Status.OK,
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


class TestSwarmRegisterHandlersSignature:
    """Verify SwarmAgent._register_handlers matches BaseAgent signature."""

    def test_register_handlers_accepts_single_arg(self):
        """_register_handlers(api) works without extra args."""
        from fsm_llm.agents.swarm import SwarmAgent

        agent = Mock()
        swarm = SwarmAgent(agents={"a": agent}, entry_agent="a")
        # Should not raise TypeError
        swarm._register_handlers(Mock())


class TestSwarmAgent:
    """Tests for SwarmAgent."""

    def _make_mock_agent(self, answer: str, context: dict | None = None):
        """Create a mock agent that returns a fixed answer."""
        agent = Mock()
        agent.run = Mock(
            return_value=AgentResult(
                answer=answer,
                status=Status.OK,
                trace=AgentTrace(total_iterations=1),
                final_context=context or {},
            )
        )
        return agent

    def test_basic_swarm_no_handoff(self):
        """Swarm with single agent, no handoff."""
        from fsm_llm.agents.swarm import SwarmAgent

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
        from fsm_llm.agents.swarm import SwarmAgent

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
        from fsm_llm.agents.swarm import SwarmAgent

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
        # max_handoffs=N allows exactly N handoffs (PAT-08).
        assert result.final_context["_swarm_handoff_count"] == 3
        assert agent_a.run.call_count == 4

    def test_swarm_invalid_handoff_target(self):
        """Swarm stops when handoff target doesn't exist."""
        from fsm_llm.agents.swarm import SwarmAgent

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
        from fsm_llm.agents.swarm import SwarmAgent

        with pytest.raises(ValueError, match="at least one agent"):
            SwarmAgent(agents={}, entry_agent="a")

    def test_swarm_invalid_entry_raises(self):
        """Invalid entry agent raises ValueError."""
        from fsm_llm.agents.swarm import SwarmAgent

        with pytest.raises(ValueError, match="not found"):
            SwarmAgent(
                agents={"a": Mock()},
                entry_agent="nonexistent",
            )

    def test_swarm_agent_failure(self):
        """Swarm handles agent failure gracefully."""
        from fsm_llm.agents.swarm import SwarmAgent

        agent_a = Mock()
        agent_a.run = Mock(side_effect=RuntimeError("Agent crashed"))

        swarm = SwarmAgent(agents={"a": agent_a}, entry_agent="a")
        result = swarm.run("test")
        assert not result.success
        assert "crashed" in result.answer

    def test_swarm_add_agent(self):
        """add_agent() adds agents to swarm."""
        from fsm_llm.agents.swarm import SwarmAgent

        agent_a = self._make_mock_agent("A", {})
        agent_b = self._make_mock_agent("B", {})

        swarm = SwarmAgent(agents={"a": agent_a}, entry_agent="a")
        swarm.add_agent("b", agent_b)
        assert "b" in swarm.agents

    def test_swarm_wall_clock_timeout(self):
        """Swarm raises AgentTimeoutError if wall-clock budget is exceeded,
        even though each individual sub-agent hand-off completes quickly
        enough to never trip its own per-call timeout and handoff_count
        never reaches max_handoffs."""
        import time

        from fsm_llm.agents.definitions import AgentConfig
        from fsm_llm.agents.exceptions import AgentTimeoutError
        from fsm_llm.agents.swarm import SwarmAgent

        def _slow_handoff(*args, **kwargs):
            time.sleep(0.05)
            return AgentResult(
                answer="Handing off",
                status=Status.OK,
                trace=AgentTrace(total_iterations=1),
                final_context={"next_agent": "a"},
            )

        agent_a = Mock()
        agent_a.run = Mock(side_effect=_slow_handoff)

        swarm = SwarmAgent(
            agents={"a": agent_a},
            entry_agent="a",
            max_handoffs=10_000,
            config=AgentConfig(timeout_seconds=0.01),
        )
        with pytest.raises(AgentTimeoutError):
            swarm.run("loop forever")


class TestAgentGraph:
    """Tests for AgentGraphBuilder and AgentGraph."""

    def _make_mock_agent(self, answer: str, context: dict | None = None):
        agent = Mock()
        agent.run = Mock(
            return_value=AgentResult(
                answer=answer,
                status=Status.OK,
                trace=AgentTrace(total_iterations=1),
                final_context=context or {},
            )
        )
        return agent

    def test_linear_graph(self):
        """Linear graph: A -> B -> C."""
        from fsm_llm.agents.agent_graph import AgentGraphBuilder

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
        from fsm_llm.agents.agent_graph import AgentGraphBuilder

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
        from fsm_llm.agents.agent_graph import AgentGraphBuilder

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

    def test_cycle_is_build_error(self):
        from fsm_llm import BuildError
        from fsm_llm.agents.agent_graph import AgentGraphBuilder

        a = self._make_mock_agent("A", {})
        b = self._make_mock_agent("B", {})
        with pytest.raises(BuildError, match="cycle") as exc:
            (
                AgentGraphBuilder()
                .add_node("a", a)
                .add_node("b", b)
                .add_edge("a", "b")
                .add_edge("b", "a")
                .set_entry("a")
                .build()
            )
        assert isinstance(exc.value, ValueError)

    def test_duplicate_node_reported_at_build(self):
        from fsm_llm import BuildError
        from fsm_llm.agents.agent_graph import AgentGraphBuilder

        a = self._make_mock_agent("A", {})
        builder = AgentGraphBuilder().add_node("a", a).add_node("a", a)
        builder.set_entry("a")
        with pytest.raises(BuildError, match="Duplicate node name 'a'") as exc:
            builder.build()
        assert isinstance(exc.value, ValueError)

    def test_graph_errors_are_build_errors(self):
        from fsm_llm import BuildError
        from fsm_llm.agents.agent_graph import AgentGraphBuilder

        a = self._make_mock_agent("A", {})
        with pytest.raises(BuildError, match="Entry node must be set"):
            AgentGraphBuilder().add_node("a", a).build()
        with pytest.raises(BuildError, match="Edge target 'z' not in nodes"):
            AgentGraphBuilder().add_node("a", a).add_edge("a", "z").set_entry(
                "a"
            ).build()

    def test_missing_entry(self):
        """Missing entry node raises ValueError."""
        from fsm_llm.agents.agent_graph import AgentGraphBuilder

        with pytest.raises(ValueError, match="Entry node must be set"):
            AgentGraphBuilder().add_node("a", Mock()).build()

    def test_invalid_edge_source(self):
        """Edge referencing non-existent source raises ValueError."""
        from fsm_llm.agents.agent_graph import AgentGraphBuilder

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
        from fsm_llm.agents.agent_graph import AgentGraphBuilder

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
        from fsm_llm.agents.agent_graph import AgentGraphBuilder

        a = Mock()
        a.run = Mock(side_effect=RuntimeError("boom"))

        graph = AgentGraphBuilder().add_node("a", a).set_entry("a").build()
        result = graph.run("test")
        assert not result.success


class TestSwarmNeedsNoModel:
    """Review gate 1 sweep: a swarm builds no API, so it owns no model.

    The root autouse fixture sets ``LLM_MODEL``; this test deletes it. The
    swarm reads its own config for ``timeout_seconds`` only, and its members
    hold their own interfaces.
    """

    def test_a_swarm_of_mock_members_builds_and_runs_without_llm_model(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("LLM_MODEL", raising=False)
        swarm = SwarmAgent(agents={"a": _agent("hello")}, entry_agent="a")
        assert swarm.config.timeout_seconds > 0
        assert swarm.run("task").answer == "hello"

    def test_a_given_config_is_kept_as_is(self, monkeypatch: pytest.MonkeyPatch):
        from fsm_llm.agents import AgentConfig

        monkeypatch.delenv("LLM_MODEL", raising=False)
        config = AgentConfig(model="provider/own", timeout_seconds=7.0)
        swarm = SwarmAgent(agents={"a": _agent()}, entry_agent="a", config=config)
        assert swarm.config is config
