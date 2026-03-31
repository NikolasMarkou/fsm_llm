"""
Integration tests for the agentic meta builder.

Tests reachability validation, builder tools,
type detection, and result building.
"""

from __future__ import annotations

from fsm_llm_agents.definitions import ArtifactType
from fsm_llm_agents.meta_builder import MetaBuilderAgent
from fsm_llm_agents.meta_builders import (
    FSMBuilder,
    WorkflowBuilder,
)
from fsm_llm_agents.meta_prompts import build_welcome_message
from fsm_llm_agents.meta_tools import create_builder_tools

# ---------------------------------------------------------------------------
# Type alias matching
# ---------------------------------------------------------------------------


class TestTypeAliasMatching:
    def test_longest_match_first(self):
        aliases = MetaBuilderAgent._build_type_aliases()
        keys = list(aliases.keys())
        assert keys.index("finite state machine") < keys.index("state machine")
        assert keys.index("data pipeline") < keys.index("pipeline")


# ---------------------------------------------------------------------------
# Reachability validation
# ---------------------------------------------------------------------------


class TestReachabilityValidation:
    def test_unreachable_state_detected(self):
        b = FSMBuilder()
        b.set_overview("Test", "Test FSM")
        b.add_state("start", "Start", "Start")
        b.add_state("mid", "Mid", "Mid")
        b.add_state("orphan", "Orphan", "Orphan")
        b.add_transition("start", "mid", "go")
        b.initial_state = "start"
        errors = b.validate_complete()
        assert any("orphan" in e.lower() for e in errors)

    def test_all_reachable_no_errors(self):
        b = FSMBuilder()
        b.set_overview("Test", "Test FSM")
        b.add_state("start", "Start", "Start")
        b.add_state("end", "End", "End")
        b.add_transition("start", "end", "finish")
        b.initial_state = "start"
        errors = b.validate_complete()
        unreachable = [e for e in errors if "unreachable" in e.lower()]
        assert len(unreachable) == 0

    def test_transition_to_nonexistent_state(self):
        b = FSMBuilder()
        b.set_overview("Test", "Test FSM")
        b.add_state("start", "Start", "Start")
        b.add_state("end", "End", "End")
        b.add_transition("start", "end", "finish")
        b.initial_state = "start"
        b.states["start"]["transitions"].append(
            {
                "target_state": "ghost",
                "description": "bad",
                "priority": 100,
                "conditions": [],
            }
        )
        errors = b.validate_complete()
        assert any("ghost" in e for e in errors)


# ---------------------------------------------------------------------------
# Workflow builder ordering (by construction via tools)
# ---------------------------------------------------------------------------


class TestWorkflowBuilderOrdering:
    def test_sequential_transitions(self):
        """WorkflowBuilder handles add_step + set_step_transition correctly."""
        b = WorkflowBuilder()
        b.set_overview(workflow_id="wf", name="Test", description="Test")
        b.add_step("s1", "auto_transition", "Step 1")
        b.add_step("s2", "auto_transition", "Step 2")
        b.add_step("s3", "auto_transition", "Step 3")
        b.set_step_transition("s1", "s2")
        b.set_step_transition("s2", "s3")
        assert len(b.steps["s1"]["transitions"]) == 1
        assert b.steps["s1"]["transitions"][0]["target"] == "s2"
        assert len(b.steps["s3"]["transitions"]) == 0


# ---------------------------------------------------------------------------
# Welcome message
# ---------------------------------------------------------------------------


class TestWelcomeMessage:
    def test_mentions_artifact_types(self):
        msg = build_welcome_message()
        assert "FSM" in msg
        assert "Workflow" in msg
        assert "Agent" in msg


# ---------------------------------------------------------------------------
# Builder result has final_context
# ---------------------------------------------------------------------------


class TestBuildResultFinalContext:
    def test_has_final_context(self):
        agent = MetaBuilderAgent()
        agent._artifact_type = ArtifactType.FSM
        agent._builder = FSMBuilder()
        agent._builder.set_overview("Test", "A test FSM")
        agent._builder.add_state("start", "Start", "Start")
        agent._builder.initial_state = "start"
        agent._build_result()
        fc = agent._result.final_context
        assert "artifact_json" in fc
        assert isinstance(fc["artifact_json"], dict)
        assert fc["artifact_type"] == "fsm"

    def test_no_builder_empty_context(self):
        agent = MetaBuilderAgent()
        agent._build_result()
        assert agent._result.final_context == {}


# ---------------------------------------------------------------------------
# Agentic builder creates correct tool registries
# ---------------------------------------------------------------------------


class TestAgenticToolSelection:
    def test_fsm_tools_created(self):
        b = FSMBuilder()
        tools = create_builder_tools(b, ArtifactType.FSM)
        names = {t.name for t in tools.list_tools()}
        assert "set_overview" in names
        assert "add_state" in names
        assert "add_transition" in names
        assert "validate" in names

    def test_workflow_tools_created(self):
        b = WorkflowBuilder()
        tools = create_builder_tools(b, ArtifactType.WORKFLOW)
        names = {t.name for t in tools.list_tools()}
        assert "add_step" in names
        assert "set_step_transition" in names

    def test_builder_tools_mutate_builder(self):
        """Tools modify the builder in place — confirming closure pattern works."""
        from fsm_llm_agents.definitions import ToolCall

        b = FSMBuilder()
        tools = create_builder_tools(b, ArtifactType.FSM)
        tools.execute(
            ToolCall(
                tool_name="set_overview",
                parameters={"name": "Test", "description": "A test"},
            )
        )
        assert b.name == "Test"
        tools.execute(
            ToolCall(
                tool_name="add_state",
                parameters={
                    "state_id": "s1",
                    "description": "State 1",
                    "purpose": "P1",
                },
            )
        )
        assert "s1" in b.states
