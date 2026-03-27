from __future__ import annotations

"""Tests for meta-agent builder tools."""

import pytest

from fsm_llm_agents.definitions import ArtifactType
from fsm_llm_agents.meta_builders import AgentBuilder, FSMBuilder, WorkflowBuilder
from fsm_llm_agents.meta_tools import (
    create_agent_tools,
    create_builder_tools,
    create_fsm_tools,
    create_workflow_tools,
)


class TestFSMTools:
    def test_creates_registry(self, fsm_builder: FSMBuilder):
        registry = create_fsm_tools(fsm_builder)
        names = [t.name for t in registry.list_tools()]
        assert "set_overview" in names
        assert "add_state" in names
        assert "add_transition" in names
        assert "remove_state" in names
        assert "remove_transition" in names
        assert "set_initial_state" in names
        assert "validate" in names
        assert "get_summary" in names

    def test_set_overview(self, fsm_builder: FSMBuilder):
        registry = create_fsm_tools(fsm_builder)
        result = registry.execute(
            _make_call("set_overview", name="Bot", description="A bot", persona="Nice")
        )
        assert result.success
        assert fsm_builder.name == "Bot"
        assert fsm_builder.description == "A bot"
        assert fsm_builder.persona == "Nice"

    def test_add_state(self, fsm_builder: FSMBuilder):
        registry = create_fsm_tools(fsm_builder)
        result = registry.execute(
            _make_call("add_state", state_id="s1", description="S1", purpose="P1")
        )
        assert result.success
        assert "s1" in fsm_builder.states
        assert fsm_builder.initial_state == "s1"  # Auto-set

    def test_add_multiple_states(self, fsm_builder: FSMBuilder):
        registry = create_fsm_tools(fsm_builder)
        for sid in ["a", "b", "c"]:
            result = registry.execute(
                _make_call(
                    "add_state",
                    state_id=sid,
                    description=f"State {sid}",
                    purpose=f"P-{sid}",
                )
            )
            assert result.success
        assert len(fsm_builder.states) == 3
        assert fsm_builder.initial_state == "a"

    def test_add_transition(self, fsm_builder: FSMBuilder):
        registry = create_fsm_tools(fsm_builder)
        registry.execute(
            _make_call("add_state", state_id="a", description="A", purpose="PA")
        )
        registry.execute(
            _make_call("add_state", state_id="b", description="B", purpose="PB")
        )
        result = registry.execute(
            _make_call(
                "add_transition", from_state="a", target_state="b", description="A to B"
            )
        )
        assert result.success
        assert len(fsm_builder.states["a"]["transitions"]) == 1

    def test_add_transition_missing_state_returns_error(self, fsm_builder: FSMBuilder):
        registry = create_fsm_tools(fsm_builder)
        result = registry.execute(
            _make_call(
                "add_transition", from_state="x", target_state="y", description="bad"
            )
        )
        assert result.success  # Tool doesn't crash
        assert "Error" in result.result

    def test_remove_state(self, fsm_builder: FSMBuilder):
        registry = create_fsm_tools(fsm_builder)
        registry.execute(
            _make_call("add_state", state_id="s1", description="S", purpose="P")
        )
        result = registry.execute(_make_call("remove_state", state_id="s1"))
        assert result.success
        assert "s1" not in fsm_builder.states

    def test_validate_empty_returns_errors(self, fsm_builder: FSMBuilder):
        registry = create_fsm_tools(fsm_builder)
        result = registry.execute(_make_call("validate"))
        assert result.success
        assert "ERRORS" in result.result

    def test_validate_complete_returns_valid(self, populated_fsm_builder: FSMBuilder):
        registry = create_fsm_tools(populated_fsm_builder)
        result = registry.execute(_make_call("validate"))
        assert result.success
        assert "ERRORS" not in result.result

    def test_get_summary(self, populated_fsm_builder: FSMBuilder):
        registry = create_fsm_tools(populated_fsm_builder)
        result = registry.execute(_make_call("get_summary"))
        assert result.success
        assert "GreetingBot" in result.result

    def test_update_state(self, fsm_builder: FSMBuilder):
        registry = create_fsm_tools(fsm_builder)
        registry.execute(
            _make_call("add_state", state_id="s1", description="Old", purpose="P")
        )
        result = registry.execute(
            _make_call("update_state", state_id="s1", description="New")
        )
        assert result.success
        assert fsm_builder.states["s1"]["description"] == "New"

    def test_set_initial_state(self, fsm_builder: FSMBuilder):
        registry = create_fsm_tools(fsm_builder)
        registry.execute(
            _make_call("add_state", state_id="a", description="A", purpose="PA")
        )
        registry.execute(
            _make_call("add_state", state_id="b", description="B", purpose="PB")
        )
        result = registry.execute(_make_call("set_initial_state", state_id="b"))
        assert result.success
        assert fsm_builder.initial_state == "b"


class TestWorkflowTools:
    def test_creates_registry(self, workflow_builder: WorkflowBuilder):
        registry = create_workflow_tools(workflow_builder)
        names = [t.name for t in registry.list_tools()]
        assert "set_overview" in names
        assert "add_step" in names
        assert "set_step_transition" in names
        assert "validate" in names

    def test_set_overview(self, workflow_builder: WorkflowBuilder):
        registry = create_workflow_tools(workflow_builder)
        result = registry.execute(
            _make_call(
                "set_overview", workflow_id="wf1", name="Flow", description="A flow"
            )
        )
        assert result.success
        assert workflow_builder.name == "Flow"

    def test_add_step(self, workflow_builder: WorkflowBuilder):
        registry = create_workflow_tools(workflow_builder)
        result = registry.execute(
            _make_call(
                "add_step", step_id="start", step_type="auto_transition", name="Start"
            )
        )
        assert result.success
        assert "start" in workflow_builder.steps

    def test_set_step_transition(self, workflow_builder: WorkflowBuilder):
        registry = create_workflow_tools(workflow_builder)
        registry.execute(
            _make_call("add_step", step_id="a", step_type="auto_transition", name="A")
        )
        registry.execute(
            _make_call("add_step", step_id="b", step_type="auto_transition", name="B")
        )
        result = registry.execute(
            _make_call("set_step_transition", from_step="a", to_step="b")
        )
        assert result.success


class TestAgentTools:
    def test_creates_registry(self, agent_builder: AgentBuilder):
        registry = create_agent_tools(agent_builder)
        names = [t.name for t in registry.list_tools()]
        assert "set_overview" in names
        assert "set_agent_type" in names
        assert "add_tool" in names
        assert "validate" in names

    def test_set_overview(self, agent_builder: AgentBuilder):
        registry = create_agent_tools(agent_builder)
        result = registry.execute(
            _make_call("set_overview", name="MyAgent", description="An agent")
        )
        assert result.success
        assert agent_builder.name == "MyAgent"

    def test_set_agent_type(self, agent_builder: AgentBuilder):
        registry = create_agent_tools(agent_builder)
        result = registry.execute(_make_call("set_agent_type", agent_type="react"))
        assert result.success
        assert agent_builder.agent_type == "react"

    def test_set_agent_type_invalid_returns_error(self, agent_builder: AgentBuilder):
        registry = create_agent_tools(agent_builder)
        result = registry.execute(_make_call("set_agent_type", agent_type="invalid"))
        assert result.success
        assert "Error" in result.result

    def test_add_tool(self, agent_builder: AgentBuilder):
        registry = create_agent_tools(agent_builder)
        result = registry.execute(
            _make_call("add_tool", name="search", description="Search the web")
        )
        assert result.success
        assert len(agent_builder.tools) == 1

    def test_remove_tool(self, agent_builder: AgentBuilder):
        registry = create_agent_tools(agent_builder)
        registry.execute(_make_call("add_tool", name="search", description="Search"))
        result = registry.execute(_make_call("remove_tool", name="search"))
        assert result.success
        assert len(agent_builder.tools) == 0


class TestCreateBuilderTools:
    def test_dispatch_fsm(self, fsm_builder: FSMBuilder):
        registry = create_builder_tools(fsm_builder, ArtifactType.FSM)
        assert any(t.name == "add_state" for t in registry.list_tools())

    def test_dispatch_workflow(self, workflow_builder: WorkflowBuilder):
        registry = create_builder_tools(workflow_builder, ArtifactType.WORKFLOW)
        assert any(t.name == "add_step" for t in registry.list_tools())

    def test_dispatch_agent(self, agent_builder: AgentBuilder):
        registry = create_builder_tools(agent_builder, ArtifactType.AGENT)
        assert any(t.name == "set_agent_type" for t in registry.list_tools())

    def test_type_mismatch_raises(self, fsm_builder: FSMBuilder):
        with pytest.raises(TypeError):
            create_builder_tools(fsm_builder, ArtifactType.WORKFLOW)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_call(tool_name: str, **kwargs):
    """Create a ToolCall for testing."""
    from fsm_llm_agents.definitions import ToolCall

    return ToolCall(tool_name=tool_name, parameters=kwargs)
