from __future__ import annotations

"""Tests for artifact builders."""

import pytest

from fsm_llm_meta.builders import AgentBuilder, FSMBuilder, WorkflowBuilder
from fsm_llm_meta.definitions import ArtifactType
from fsm_llm_meta.exceptions import BuilderError

# ===================================================================
# FSMBuilder Tests
# ===================================================================


class TestFSMBuilderOverview:
    def test_set_overview(self, fsm_builder: FSMBuilder):
        warnings = fsm_builder.set_overview("MyBot", "A test bot")
        assert warnings == []
        assert fsm_builder.name == "MyBot"
        assert fsm_builder.description == "A test bot"

    def test_set_overview_with_persona(self, fsm_builder: FSMBuilder):
        fsm_builder.set_overview("MyBot", "A test bot", persona="Friendly helper")
        assert fsm_builder.persona == "Friendly helper"

    def test_empty_name_warns(self, fsm_builder: FSMBuilder):
        warnings = fsm_builder.set_overview("", "desc")
        assert any("empty" in w.lower() for w in warnings)


class TestFSMBuilderStates:
    def test_add_state(self, fsm_builder: FSMBuilder):
        warnings = fsm_builder.add_state("greeting", "Greet user", "Welcome them")
        assert "greeting" in fsm_builder.states
        assert fsm_builder.states["greeting"]["id"] == "greeting"
        assert fsm_builder.states["greeting"]["description"] == "Greet user"
        # First state auto-sets initial
        assert fsm_builder.initial_state == "greeting"
        assert any("Auto-set" in w for w in warnings)

    def test_add_second_state_no_auto_initial(self, fsm_builder: FSMBuilder):
        fsm_builder.add_state("first", "First state", "Purpose")
        warnings = fsm_builder.add_state("second", "Second state", "Purpose")
        assert fsm_builder.initial_state == "first"
        assert not any("Auto-set" in w for w in warnings)

    def test_add_state_with_instructions(self, fsm_builder: FSMBuilder):
        fsm_builder.add_state(
            "s1",
            "State 1",
            "Purpose",
            extraction_instructions="Extract name",
            response_instructions="Greet by name",
        )
        assert fsm_builder.states["s1"]["extraction_instructions"] == "Extract name"
        assert fsm_builder.states["s1"]["response_instructions"] == "Greet by name"

    def test_add_duplicate_warns(self, fsm_builder: FSMBuilder):
        fsm_builder.add_state("s1", "First", "Purpose")
        warnings = fsm_builder.add_state("s1", "Overwritten", "New purpose")
        assert any("already exists" in w for w in warnings)
        assert fsm_builder.states["s1"]["description"] == "Overwritten"

    def test_add_state_empty_id_raises(self, fsm_builder: FSMBuilder):
        with pytest.raises(BuilderError, match="empty"):
            fsm_builder.add_state("", "desc", "purpose")

    def test_remove_state(self, fsm_builder: FSMBuilder):
        fsm_builder.add_state("s1", "State 1", "Purpose")
        fsm_builder.add_state("s2", "State 2", "Purpose")
        assert fsm_builder.remove_state("s1")
        assert "s1" not in fsm_builder.states
        assert fsm_builder.initial_state is None  # was s1

    def test_remove_nonexistent(self, fsm_builder: FSMBuilder):
        assert not fsm_builder.remove_state("nope")

    def test_remove_cleans_transitions(self, fsm_builder: FSMBuilder):
        fsm_builder.add_state("a", "A", "Purpose")
        fsm_builder.add_state("b", "B", "Purpose")
        fsm_builder.add_transition("a", "b", "go to b")
        fsm_builder.remove_state("b")
        assert fsm_builder.states["a"]["transitions"] == []

    def test_update_state(self, fsm_builder: FSMBuilder):
        fsm_builder.add_state("s1", "Old desc", "Old purpose")
        warnings = fsm_builder.update_state("s1", description="New desc")
        assert warnings == []
        assert fsm_builder.states["s1"]["description"] == "New desc"

    def test_update_nonexistent_raises(self, fsm_builder: FSMBuilder):
        with pytest.raises(BuilderError, match="not found"):
            fsm_builder.update_state("nope", description="x")

    def test_update_ignores_unknown_fields(self, fsm_builder: FSMBuilder):
        fsm_builder.add_state("s1", "desc", "purpose")
        warnings = fsm_builder.update_state("s1", unknown_field="value")
        assert any("Ignoring" in w for w in warnings)


class TestFSMBuilderTransitions:
    def test_add_transition(self, fsm_builder: FSMBuilder):
        fsm_builder.add_state("a", "A", "Purpose")
        fsm_builder.add_state("b", "B", "Purpose")
        warnings = fsm_builder.add_transition("a", "b", "Move to B")
        assert warnings == []
        assert len(fsm_builder.states["a"]["transitions"]) == 1
        assert fsm_builder.states["a"]["transitions"][0]["target_state"] == "b"

    def test_add_transition_with_conditions(self, fsm_builder: FSMBuilder):
        fsm_builder.add_state("a", "A", "Purpose")
        fsm_builder.add_state("b", "B", "Purpose")
        fsm_builder.add_transition(
            "a",
            "b",
            "When ready",
            conditions=[{"description": "Name set", "logic": {"has_context": "name"}}],
        )
        t = fsm_builder.states["a"]["transitions"][0]
        assert len(t["conditions"]) == 1

    def test_add_transition_missing_source_raises(self, fsm_builder: FSMBuilder):
        fsm_builder.add_state("b", "B", "Purpose")
        with pytest.raises(BuilderError, match="not found"):
            fsm_builder.add_transition("a", "b", "desc")

    def test_add_transition_missing_target_raises(self, fsm_builder: FSMBuilder):
        fsm_builder.add_state("a", "A", "Purpose")
        with pytest.raises(BuilderError, match="not found"):
            fsm_builder.add_transition("a", "b", "desc")

    def test_remove_transition(self, fsm_builder: FSMBuilder):
        fsm_builder.add_state("a", "A", "Purpose")
        fsm_builder.add_state("b", "B", "Purpose")
        fsm_builder.add_transition("a", "b", "desc")
        assert fsm_builder.remove_transition("a", "b")
        assert fsm_builder.states["a"]["transitions"] == []

    def test_remove_nonexistent_transition(self, fsm_builder: FSMBuilder):
        fsm_builder.add_state("a", "A", "Purpose")
        assert not fsm_builder.remove_transition("a", "b")

    def test_set_initial_state(self, fsm_builder: FSMBuilder):
        fsm_builder.add_state("a", "A", "Purpose")
        fsm_builder.add_state("b", "B", "Purpose")
        fsm_builder.set_initial_state("b")
        assert fsm_builder.initial_state == "b"

    def test_set_initial_state_nonexistent_raises(self, fsm_builder: FSMBuilder):
        with pytest.raises(BuilderError, match="not found"):
            fsm_builder.set_initial_state("nope")


class TestFSMBuilderSerialization:
    def test_to_dict_minimal(self, fsm_builder: FSMBuilder):
        fsm_builder.set_overview("Bot", "Desc")
        fsm_builder.add_state("s1", "State 1", "Purpose")
        d = fsm_builder.to_dict()
        assert d["name"] == "Bot"
        assert d["description"] == "Desc"
        assert d["initial_state"] == "s1"
        assert d["version"] == "4.1"
        assert "s1" in d["states"]

    def test_to_dict_with_persona(self, fsm_builder: FSMBuilder):
        fsm_builder.set_overview("Bot", "Desc", persona="Helper")
        d = fsm_builder.to_dict()
        assert d["persona"] == "Helper"

    def test_to_dict_produces_valid_fsm(self, populated_fsm_builder: FSMBuilder):
        """Verify the populated builder produces a valid FSMDefinition."""
        from fsm_llm.definitions import FSMDefinition

        d = populated_fsm_builder.to_dict()
        # Should not raise
        fsm = FSMDefinition(**d)
        assert fsm.name == "GreetingBot"
        assert len(fsm.states) == 3


class TestFSMBuilderValidation:
    def test_validate_complete_empty(self, fsm_builder: FSMBuilder):
        errors = fsm_builder.validate_complete()
        assert len(errors) > 0
        assert any("name" in e.lower() for e in errors)

    def test_validate_complete_valid(self, populated_fsm_builder: FSMBuilder):
        errors = populated_fsm_builder.validate_complete()
        assert errors == []

    def test_validate_partial(self, fsm_builder: FSMBuilder):
        warnings = fsm_builder.validate_partial()
        assert any("name" in w.lower() for w in warnings)
        assert any("No states" in w for w in warnings)

    def test_get_missing_fields_empty(self, fsm_builder: FSMBuilder):
        missing = fsm_builder.get_missing_fields()
        assert "FSM name" in missing
        assert "FSM description" in missing

    def test_get_missing_fields_partial(self, fsm_builder: FSMBuilder):
        fsm_builder.set_overview("Bot", "Desc")
        missing = fsm_builder.get_missing_fields()
        assert "At least one state" in missing

    def test_get_progress(self, populated_fsm_builder: FSMBuilder):
        progress = populated_fsm_builder.get_progress()
        assert progress.percentage > 0
        assert progress.total_required > 0


class TestFSMBuilderSummary:
    def test_summary_empty(self, fsm_builder: FSMBuilder):
        summary = fsm_builder.get_summary()
        assert "FSM Builder Status" in summary
        assert "(not set)" in summary

    def test_summary_populated(self, populated_fsm_builder: FSMBuilder):
        summary = populated_fsm_builder.get_summary()
        assert "GreetingBot" in summary
        assert "greeting" in summary
        assert "[INITIAL]" in summary
        assert "[TERMINAL]" in summary
        assert "farewell" in summary


# ===================================================================
# WorkflowBuilder Tests
# ===================================================================


class TestWorkflowBuilder:
    def test_artifact_type(self, workflow_builder: WorkflowBuilder):
        assert workflow_builder.artifact_type == ArtifactType.WORKFLOW

    def test_set_overview(self, workflow_builder: WorkflowBuilder):
        workflow_builder.set_overview(
            "order_flow", "Order Processing", "Handles orders"
        )
        assert workflow_builder.workflow_id == "order_flow"
        assert workflow_builder.name == "Order Processing"

    def test_add_step(self, workflow_builder: WorkflowBuilder):
        workflow_builder.add_step(
            "validate", "condition", "Validate Order", "Check order validity"
        )
        assert "validate" in workflow_builder.steps
        # Auto-set initial step
        assert workflow_builder.initial_step_id == "validate"

    def test_add_step_invalid_type_warns(self, workflow_builder: WorkflowBuilder):
        warnings = workflow_builder.add_step("s1", "invalid_type", "Step 1")
        assert any("Unknown step type" in w for w in warnings)

    def test_remove_step(self, workflow_builder: WorkflowBuilder):
        workflow_builder.add_step("s1", "auto_transition", "S1")
        assert workflow_builder.remove_step("s1")
        assert "s1" not in workflow_builder.steps

    def test_set_step_transition(self, workflow_builder: WorkflowBuilder):
        workflow_builder.add_step("s1", "auto_transition", "S1")
        workflow_builder.add_step("s2", "auto_transition", "S2")
        workflow_builder.set_step_transition("s1", "s2", condition="order.valid")
        assert len(workflow_builder.steps["s1"]["transitions"]) == 1

    def test_set_step_transition_missing_raises(
        self, workflow_builder: WorkflowBuilder
    ):
        workflow_builder.add_step("s1", "auto_transition", "S1")
        with pytest.raises(BuilderError):
            workflow_builder.set_step_transition("s1", "missing", None)

    def test_to_dict(self, workflow_builder: WorkflowBuilder):
        workflow_builder.set_overview("wf1", "My Workflow", "Does stuff")
        workflow_builder.add_step("s1", "auto_transition", "Step 1")
        d = workflow_builder.to_dict()
        assert d["workflow_id"] == "wf1"
        assert d["name"] == "My Workflow"
        assert "s1" in d["steps"]

    def test_validate_complete_empty(self, workflow_builder: WorkflowBuilder):
        errors = workflow_builder.validate_complete()
        assert len(errors) > 0

    def test_validate_complete_valid(self, workflow_builder: WorkflowBuilder):
        workflow_builder.set_overview("wf1", "Flow", "Description")
        workflow_builder.add_step("s1", "auto_transition", "Step 1")
        errors = workflow_builder.validate_complete()
        assert errors == []

    def test_get_summary(self, workflow_builder: WorkflowBuilder):
        workflow_builder.set_overview("wf1", "Flow", "Description")
        workflow_builder.add_step("s1", "auto_transition", "Step 1")
        summary = workflow_builder.get_summary()
        assert "Workflow Builder Status" in summary
        assert "wf1" in summary


# ===================================================================
# AgentBuilder Tests
# ===================================================================


class TestAgentBuilder:
    def test_artifact_type(self, agent_builder: AgentBuilder):
        assert agent_builder.artifact_type == ArtifactType.AGENT

    def test_set_agent_type(self, agent_builder: AgentBuilder):
        warnings = agent_builder.set_agent_type("react")
        assert warnings == []
        assert agent_builder.agent_type == "react"

    def test_set_agent_type_invalid_warns(self, agent_builder: AgentBuilder):
        warnings = agent_builder.set_agent_type("unknown_type")
        assert any("Unknown agent type" in w for w in warnings)

    def test_set_overview(self, agent_builder: AgentBuilder):
        agent_builder.set_overview("SearchAgent", "Searches the web")
        assert agent_builder.name == "SearchAgent"

    def test_add_tool(self, agent_builder: AgentBuilder):
        warnings = agent_builder.add_tool("search", "Search the web")
        assert warnings == []
        assert len(agent_builder.tools) == 1
        assert agent_builder.tools[0]["name"] == "search"

    def test_add_tool_duplicate_warns(self, agent_builder: AgentBuilder):
        agent_builder.add_tool("search", "V1")
        warnings = agent_builder.add_tool("search", "V2")
        assert any("already exists" in w for w in warnings)
        assert len(agent_builder.tools) == 1
        assert agent_builder.tools[0]["description"] == "V2"

    def test_add_tool_empty_name_raises(self, agent_builder: AgentBuilder):
        with pytest.raises(BuilderError, match="empty"):
            agent_builder.add_tool("", "desc")

    def test_add_tool_with_schema(self, agent_builder: AgentBuilder):
        schema = {"type": "object", "properties": {"query": {"type": "string"}}}
        agent_builder.add_tool("search", "Search", parameter_schema=schema)
        assert agent_builder.tools[0]["parameter_schema"] == schema

    def test_remove_tool(self, agent_builder: AgentBuilder):
        agent_builder.add_tool("search", "desc")
        assert agent_builder.remove_tool("search")
        assert len(agent_builder.tools) == 0

    def test_remove_nonexistent_tool(self, agent_builder: AgentBuilder):
        assert not agent_builder.remove_tool("nope")

    def test_set_config(self, agent_builder: AgentBuilder):
        agent_builder.set_config(model="gpt-4o", max_iterations=20)
        assert agent_builder.config["model"] == "gpt-4o"
        assert agent_builder.config["max_iterations"] == 20

    def test_set_config_unknown_warns(self, agent_builder: AgentBuilder):
        warnings = agent_builder.set_config(unknown_field="value")
        assert any("Ignoring" in w for w in warnings)

    def test_to_dict(self, agent_builder: AgentBuilder):
        agent_builder.set_agent_type("react")
        agent_builder.set_overview("Bot", "My bot")
        agent_builder.add_tool("search", "Search")
        d = agent_builder.to_dict()
        assert d["agent_type"] == "react"
        assert d["name"] == "Bot"
        assert len(d["tools"]) == 1

    def test_validate_complete_empty(self, agent_builder: AgentBuilder):
        errors = agent_builder.validate_complete()
        assert len(errors) > 0

    def test_validate_complete_valid(self, agent_builder: AgentBuilder):
        agent_builder.set_agent_type("react")
        agent_builder.set_overview("Bot", "My bot")
        agent_builder.add_tool("search", "Search the web")
        errors = agent_builder.validate_complete()
        assert errors == []

    def test_get_summary(self, agent_builder: AgentBuilder):
        agent_builder.set_agent_type("react")
        agent_builder.add_tool("search", "Search")
        summary = agent_builder.get_summary()
        assert "Agent Builder Status" in summary
        assert "react" in summary
        assert "search" in summary


# ===================================================================
# Tiered Summary Tests
# ===================================================================


class TestFSMBuilderTieredSummary:
    """Tests for FSMBuilder tiered summary levels."""

    def test_minimal_summary_compact(self, populated_fsm_builder: FSMBuilder):
        minimal = populated_fsm_builder.get_summary("minimal")
        full = populated_fsm_builder.get_summary("full")
        assert len(minimal) < len(full)
        assert "FSM Builder Status" in minimal
        assert "greeting" in minimal
        assert "[INITIAL]" in minimal

    def test_minimal_omits_descriptions(self, populated_fsm_builder: FSMBuilder):
        minimal = populated_fsm_builder.get_summary("minimal")
        # Minimal should not include extraction/response instruction snippets
        assert "extraction:" not in minimal
        assert "response:" not in minimal

    def test_standard_includes_descriptions(self, populated_fsm_builder: FSMBuilder):
        standard = populated_fsm_builder.get_summary("standard")
        assert "FSM Builder Status" in standard
        assert "->" in standard  # transition targets

    def test_standard_is_between_minimal_and_full(
        self, populated_fsm_builder: FSMBuilder
    ):
        minimal = populated_fsm_builder.get_summary("minimal")
        standard = populated_fsm_builder.get_summary("standard")
        full = populated_fsm_builder.get_summary("full")
        assert len(minimal) <= len(standard) <= len(full)

    def test_full_matches_default(self, populated_fsm_builder: FSMBuilder):
        default = populated_fsm_builder.get_summary()
        full = populated_fsm_builder.get_summary("full")
        assert default == full

    def test_minimal_empty_builder(self, fsm_builder: FSMBuilder):
        minimal = fsm_builder.get_summary("minimal")
        assert "none yet" in minimal


class TestWorkflowBuilderTieredSummary:
    def test_minimal_compact(self, workflow_builder: WorkflowBuilder):
        workflow_builder.set_overview("wf1", "Flow", "A flow")
        workflow_builder.add_step("step1", "auto_transition", "First", "First step")
        minimal = workflow_builder.get_summary("minimal")
        full = workflow_builder.get_summary("full")
        assert len(minimal) < len(full)
        assert "step1" in minimal

    def test_full_matches_default(self, workflow_builder: WorkflowBuilder):
        workflow_builder.set_overview("wf1", "Flow", "A flow")
        assert workflow_builder.get_summary() == workflow_builder.get_summary("full")


class TestAgentBuilderTieredSummary:
    def test_minimal_compact(self, agent_builder: AgentBuilder):
        agent_builder.set_agent_type("react")
        agent_builder.set_overview("Bot", "A bot")
        agent_builder.add_tool("search", "Search the web")
        minimal = agent_builder.get_summary("minimal")
        full = agent_builder.get_summary("full")
        assert len(minimal) < len(full)
        assert "search" in minimal

    def test_full_matches_default(self, agent_builder: AgentBuilder):
        agent_builder.set_agent_type("react")
        assert agent_builder.get_summary() == agent_builder.get_summary("full")
