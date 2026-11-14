from __future__ import annotations

"""Tests for artifact builders."""

import pytest

from fsm_llm.agents.definitions import ArtifactType
from fsm_llm.agents.exceptions import BuilderError
from fsm_llm.agents.meta_builders import (
    AgentArtifactBuilder,
    FSMArtifactBuilder,
    WorkflowArtifactBuilder,
)
from fsm_llm.core.definitions.schema import SCHEMA_VERSION_DEFAULT

# ===================================================================
# FSMArtifactBuilder Tests
# ===================================================================


class TestFSMBuilderOverview:
    def test_set_overview(self, fsm_builder: FSMArtifactBuilder):
        warnings = fsm_builder.set_overview("MyBot", "A test bot").take_warnings()
        assert warnings == []
        assert fsm_builder.name == "MyBot"
        assert fsm_builder.description == "A test bot"

    def test_set_overview_with_persona(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.set_overview("MyBot", "A test bot", persona="Friendly helper")
        assert fsm_builder.persona == "Friendly helper"

    def test_empty_name_warns(self, fsm_builder: FSMArtifactBuilder):
        warnings = fsm_builder.set_overview("", "desc").take_warnings()
        assert any("empty" in w.lower() for w in warnings)


class TestFSMBuilderStates:
    def test_add_state(self, fsm_builder: FSMArtifactBuilder):
        warnings = fsm_builder.add_state(
            "greeting", "Greet user", "Welcome them"
        ).take_warnings()
        assert "greeting" in fsm_builder.states
        assert fsm_builder.states["greeting"]["id"] == "greeting"
        assert fsm_builder.states["greeting"]["description"] == "Greet user"
        # First state auto-sets initial
        assert fsm_builder.initial_state == "greeting"
        assert any("Auto-set" in w for w in warnings)

    def test_add_second_state_no_auto_initial(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.add_state("first", "First state", "Purpose").take_warnings()
        warnings = fsm_builder.add_state(
            "second", "Second state", "Purpose"
        ).take_warnings()
        assert fsm_builder.initial_state == "first"
        assert not any("Auto-set" in w for w in warnings)

    def test_add_state_with_instructions(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.add_state(
            "s1",
            "State 1",
            "Purpose",
            extraction_instructions="Extract name",
            response_instructions="Greet by name",
        )
        assert fsm_builder.states["s1"]["extraction_instructions"] == "Extract name"
        assert fsm_builder.states["s1"]["response_instructions"] == "Greet by name"

    def test_add_duplicate_warns(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.add_state("s1", "First", "Purpose")
        warnings = fsm_builder.add_state(
            "s1", "Overwritten", "New purpose"
        ).take_warnings()
        assert any("already exists" in w for w in warnings)
        assert fsm_builder.states["s1"]["description"] == "Overwritten"

    def test_add_state_empty_id_raises(self, fsm_builder: FSMArtifactBuilder):
        with pytest.raises(BuilderError, match="empty"):
            fsm_builder.add_state("", "desc", "purpose")

    def test_remove_state(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.add_state("s1", "State 1", "Purpose")
        fsm_builder.add_state("s2", "State 2", "Purpose")
        fsm_builder.remove_state("s1")
        assert "s1" not in fsm_builder.states
        assert fsm_builder.initial_state is None  # was s1

    def test_remove_nonexistent(self, fsm_builder: FSMArtifactBuilder):
        assert fsm_builder.remove_state("nope") is fsm_builder
        assert fsm_builder.states == {}

    def test_remove_cleans_transitions(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.add_state("a", "A", "Purpose")
        fsm_builder.add_state("b", "B", "Purpose")
        fsm_builder.add_transition("a", "b", "go to b")
        fsm_builder.remove_state("b")
        assert fsm_builder.states["a"]["transitions"] == []

    def test_update_state(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.add_state("s1", "Old desc", "Old purpose").take_warnings()
        warnings = fsm_builder.update_state(
            "s1", description="New desc"
        ).take_warnings()
        assert warnings == []
        assert fsm_builder.states["s1"]["description"] == "New desc"

    def test_update_nonexistent_raises(self, fsm_builder: FSMArtifactBuilder):
        with pytest.raises(BuilderError, match="not found"):
            fsm_builder.update_state("nope", description="x")

    def test_update_ignores_unknown_fields(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.add_state("s1", "desc", "purpose")
        warnings = fsm_builder.update_state("s1", unknown_field="value").take_warnings()
        assert any("Ignoring" in w for w in warnings)


class TestFSMBuilderTransitions:
    def test_add_transition(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.add_state("a", "A", "Purpose")
        fsm_builder.add_state("b", "B", "Purpose")
        fsm_builder.take_warnings()
        warnings = fsm_builder.add_transition("a", "b", "Move to B").take_warnings()
        assert warnings == []
        assert len(fsm_builder.states["a"]["transitions"]) == 1
        assert fsm_builder.states["a"]["transitions"][0]["target_state"] == "b"

    def test_add_transition_with_conditions(self, fsm_builder: FSMArtifactBuilder):
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

    def test_add_transition_missing_source_raises(
        self, fsm_builder: FSMArtifactBuilder
    ):
        fsm_builder.add_state("b", "B", "Purpose")
        with pytest.raises(BuilderError, match="not found"):
            fsm_builder.add_transition("a", "b", "desc")

    def test_add_transition_missing_target_raises(
        self, fsm_builder: FSMArtifactBuilder
    ):
        fsm_builder.add_state("a", "A", "Purpose")
        with pytest.raises(BuilderError, match="not found"):
            fsm_builder.add_transition("a", "b", "desc")

    def test_remove_transition(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.add_state("a", "A", "Purpose")
        fsm_builder.add_state("b", "B", "Purpose")
        fsm_builder.add_transition("a", "b", "desc")
        fsm_builder.remove_transition("a", "b")
        assert fsm_builder.states["a"]["transitions"] == []

    def test_remove_nonexistent_transition(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.add_state("a", "A", "Purpose")
        assert fsm_builder.remove_transition("a", "b") is fsm_builder
        assert fsm_builder.states["a"]["transitions"] == []

    def test_set_initial_state(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.add_state("a", "A", "Purpose")
        fsm_builder.add_state("b", "B", "Purpose")
        fsm_builder.set_initial_state("b")
        assert fsm_builder.initial_state == "b"

    def test_set_initial_state_nonexistent_raises(
        self, fsm_builder: FSMArtifactBuilder
    ):
        with pytest.raises(BuilderError, match="not found"):
            fsm_builder.set_initial_state("nope")


class TestFSMBuilderSerialization:
    def test_to_dict_minimal(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.set_overview("Bot", "Desc")
        fsm_builder.add_state("s1", "State 1", "Purpose")
        d = fsm_builder.to_dict()
        assert d["name"] == "Bot"
        assert d["description"] == "Desc"
        assert d["initial_state"] == "s1"
        assert d["version"] == SCHEMA_VERSION_DEFAULT
        assert "s1" in d["states"]

    def test_to_dict_with_persona(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.set_overview("Bot", "Desc", persona="Helper")
        d = fsm_builder.to_dict()
        assert d["persona"] == "Helper"

    def test_to_dict_produces_valid_fsm(
        self, populated_fsm_builder: FSMArtifactBuilder
    ):
        """Verify the populated builder produces a valid FSMDefinition."""
        from fsm_llm.core.definitions import FSMDefinition

        d = populated_fsm_builder.to_dict()
        # Should not raise
        fsm = FSMDefinition(**d)
        assert fsm.name == "GreetingBot"
        assert len(fsm.states) == 3


class TestFSMBuilderValidation:
    def test_validate_complete_empty(self, fsm_builder: FSMArtifactBuilder):
        errors = fsm_builder.validate_complete()
        assert len(errors) > 0
        assert any("name" in e.lower() for e in errors)

    def test_validate_complete_valid(self, populated_fsm_builder: FSMArtifactBuilder):
        errors = populated_fsm_builder.validate_complete()
        assert errors == []

    def test_validate_partial(self, fsm_builder: FSMArtifactBuilder):
        warnings = fsm_builder.validate_partial()
        assert any("name" in w.lower() for w in warnings)
        assert any("No states" in w for w in warnings)

    def test_get_missing_fields_empty(self, fsm_builder: FSMArtifactBuilder):
        missing = fsm_builder.get_missing_fields()
        assert "FSM name" in missing
        assert "FSM description" in missing

    def test_get_missing_fields_partial(self, fsm_builder: FSMArtifactBuilder):
        fsm_builder.set_overview("Bot", "Desc")
        missing = fsm_builder.get_missing_fields()
        assert "At least one state" in missing

    def test_get_progress(self, populated_fsm_builder: FSMArtifactBuilder):
        progress = populated_fsm_builder.get_progress()
        assert progress.percentage > 0
        assert progress.total_required > 0


class TestFSMBuilderSummary:
    def test_summary_empty(self, fsm_builder: FSMArtifactBuilder):
        summary = fsm_builder.get_summary()
        assert "FSM Builder Status" in summary
        assert "(not set)" in summary

    def test_summary_populated(self, populated_fsm_builder: FSMArtifactBuilder):
        summary = populated_fsm_builder.get_summary()
        assert "GreetingBot" in summary
        assert "greeting" in summary
        assert "[INITIAL]" in summary
        assert "[TERMINAL]" in summary
        assert "farewell" in summary


# ===================================================================
# WorkflowArtifactBuilder Tests
# ===================================================================


class TestWorkflowBuilder:
    def test_artifact_type(self, workflow_builder: WorkflowArtifactBuilder):
        assert workflow_builder.artifact_type == ArtifactType.WORKFLOW

    def test_set_overview(self, workflow_builder: WorkflowArtifactBuilder):
        workflow_builder.set_overview(
            "order_flow", "Order Processing", "Handles orders"
        )
        assert workflow_builder.workflow_id == "order_flow"
        assert workflow_builder.name == "Order Processing"

    def test_add_step(self, workflow_builder: WorkflowArtifactBuilder):
        workflow_builder.add_step(
            "validate", "condition", "Validate Order", "Check order validity"
        )
        assert "validate" in workflow_builder.steps
        # Auto-set initial step
        assert workflow_builder.initial_step_id == "validate"

    def test_add_step_invalid_type_warns(
        self, workflow_builder: WorkflowArtifactBuilder
    ):
        warnings = workflow_builder.add_step(
            "s1", "invalid_type", "Step 1"
        ).take_warnings()
        assert any("Unknown step type" in w for w in warnings)

    def test_remove_step(self, workflow_builder: WorkflowArtifactBuilder):
        workflow_builder.add_step("s1", "auto_transition", "S1")
        workflow_builder.remove_step("s1")
        assert "s1" not in workflow_builder.steps

    def test_set_step_transition(self, workflow_builder: WorkflowArtifactBuilder):
        workflow_builder.add_step("s1", "auto_transition", "S1")
        workflow_builder.add_step("s2", "auto_transition", "S2")
        workflow_builder.set_step_transition("s1", "s2", condition="order.valid")
        assert len(workflow_builder.steps["s1"]["transitions"]) == 1

    def test_set_step_transition_missing_raises(
        self, workflow_builder: WorkflowArtifactBuilder
    ):
        workflow_builder.add_step("s1", "auto_transition", "S1")
        with pytest.raises(BuilderError):
            workflow_builder.set_step_transition("s1", "missing", None)

    def test_to_dict(self, workflow_builder: WorkflowArtifactBuilder):
        workflow_builder.set_overview("wf1", "My Workflow", "Does stuff")
        workflow_builder.add_step("s1", "auto_transition", "Step 1")
        d = workflow_builder.to_dict()
        assert d["workflow_id"] == "wf1"
        assert d["name"] == "My Workflow"
        assert "s1" in d["steps"]

    def test_validate_complete_empty(self, workflow_builder: WorkflowArtifactBuilder):
        errors = workflow_builder.validate_complete()
        assert len(errors) > 0

    def test_validate_complete_valid(self, workflow_builder: WorkflowArtifactBuilder):
        workflow_builder.set_overview("wf1", "Flow", "Description")
        workflow_builder.add_step("s1", "auto_transition", "Step 1")
        errors = workflow_builder.validate_complete()
        assert errors == []

    def test_validate_complete_unknown_step_type_is_error(
        self, workflow_builder: WorkflowArtifactBuilder
    ):
        workflow_builder.set_overview("wf1", "Flow", "Description")
        workflow_builder.add_step("s1", "invalid_type", "Step 1")
        errors = workflow_builder.validate_complete()
        assert any("s1" in e and "invalid_type" in e for e in errors)

    def test_get_summary(self, workflow_builder: WorkflowArtifactBuilder):
        workflow_builder.set_overview("wf1", "Flow", "Description")
        workflow_builder.add_step("s1", "auto_transition", "Step 1")
        summary = workflow_builder.get_summary()
        assert "Workflow Builder Status" in summary
        assert "wf1" in summary


# ===================================================================
# AgentArtifactBuilder Tests
# ===================================================================


class TestAgentBuilder:
    def test_artifact_type(self, agent_builder: AgentArtifactBuilder):
        assert agent_builder.artifact_type == ArtifactType.AGENT

    def test_set_agent_type(self, agent_builder: AgentArtifactBuilder):
        warnings = agent_builder.set_agent_type("react").take_warnings()
        assert warnings == []
        assert agent_builder.agent_type == "react"

    def test_set_agent_type_invalid_raises(self, agent_builder: AgentArtifactBuilder):
        with pytest.raises(BuilderError, match="Unknown agent type"):
            agent_builder.set_agent_type("unknown_type")

    def test_set_overview(self, agent_builder: AgentArtifactBuilder):
        agent_builder.set_overview("SearchAgent", "Searches the web")
        assert agent_builder.name == "SearchAgent"

    def test_add_tool(self, agent_builder: AgentArtifactBuilder):
        warnings = agent_builder.add_tool("search", "Search the web").take_warnings()
        assert warnings == []
        assert len(agent_builder.tools) == 1
        assert agent_builder.tools[0]["name"] == "search"

    def test_add_tool_duplicate_warns(self, agent_builder: AgentArtifactBuilder):
        agent_builder.add_tool("search", "V1")
        warnings = agent_builder.add_tool("search", "V2").take_warnings()
        assert any("already exists" in w for w in warnings)
        assert len(agent_builder.tools) == 1
        assert agent_builder.tools[0]["description"] == "V2"

    def test_add_tool_empty_name_raises(self, agent_builder: AgentArtifactBuilder):
        with pytest.raises(BuilderError, match="empty"):
            agent_builder.add_tool("", "desc")

    def test_add_tool_with_schema(self, agent_builder: AgentArtifactBuilder):
        schema = {"type": "object", "properties": {"query": {"type": "string"}}}
        agent_builder.add_tool("search", "Search", parameter_schema=schema)
        assert agent_builder.tools[0]["parameter_schema"] == schema

    def test_remove_tool(self, agent_builder: AgentArtifactBuilder):
        agent_builder.add_tool("search", "desc")
        agent_builder.remove_tool("search")
        assert len(agent_builder.tools) == 0

    def test_remove_nonexistent_tool(self, agent_builder: AgentArtifactBuilder):
        assert agent_builder.remove_tool("nope") is agent_builder
        assert agent_builder.tools == []

    def test_set_config(self, agent_builder: AgentArtifactBuilder):
        agent_builder.set_config(model="gpt-4o", max_iterations=20)
        assert agent_builder.config["model"] == "gpt-4o"
        assert agent_builder.config["max_iterations"] == 20

    def test_set_config_unknown_warns(self, agent_builder: AgentArtifactBuilder):
        warnings = agent_builder.set_config(unknown_field="value").take_warnings()
        assert any("Ignoring" in w for w in warnings)

    def test_to_dict(self, agent_builder: AgentArtifactBuilder):
        agent_builder.set_agent_type("react")
        agent_builder.set_overview("Bot", "My bot")
        agent_builder.add_tool("search", "Search")
        d = agent_builder.to_dict()
        assert d["agent_type"] == "react"
        assert d["name"] == "Bot"
        assert len(d["tools"]) == 1

    def test_validate_complete_empty(self, agent_builder: AgentArtifactBuilder):
        errors = agent_builder.validate_complete()
        assert len(errors) > 0

    def test_validate_complete_valid(self, agent_builder: AgentArtifactBuilder):
        agent_builder.set_agent_type("react")
        agent_builder.set_overview("Bot", "My bot")
        agent_builder.add_tool("search", "Search the web")
        errors = agent_builder.validate_complete()
        assert errors == []

    def test_get_summary(self, agent_builder: AgentArtifactBuilder):
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
    """Tests for FSMArtifactBuilder tiered summary levels."""

    def test_minimal_summary_compact(self, populated_fsm_builder: FSMArtifactBuilder):
        minimal = populated_fsm_builder.get_summary("minimal")
        full = populated_fsm_builder.get_summary("full")
        assert len(minimal) < len(full)
        assert "FSM Builder Status" in minimal
        assert "greeting" in minimal
        assert "[INITIAL]" in minimal

    def test_minimal_omits_descriptions(
        self, populated_fsm_builder: FSMArtifactBuilder
    ):
        minimal = populated_fsm_builder.get_summary("minimal")
        # Minimal should not include extraction/response instruction snippets
        assert "extraction:" not in minimal
        assert "response:" not in minimal

    def test_standard_includes_descriptions(
        self, populated_fsm_builder: FSMArtifactBuilder
    ):
        standard = populated_fsm_builder.get_summary("standard")
        assert "FSM Builder Status" in standard
        assert "->" in standard  # transition targets

    def test_standard_is_between_minimal_and_full(
        self, populated_fsm_builder: FSMArtifactBuilder
    ):
        minimal = populated_fsm_builder.get_summary("minimal")
        standard = populated_fsm_builder.get_summary("standard")
        full = populated_fsm_builder.get_summary("full")
        assert len(minimal) <= len(standard) <= len(full)

    def test_full_matches_default(self, populated_fsm_builder: FSMArtifactBuilder):
        default = populated_fsm_builder.get_summary()
        full = populated_fsm_builder.get_summary("full")
        assert default == full

    def test_minimal_empty_builder(self, fsm_builder: FSMArtifactBuilder):
        minimal = fsm_builder.get_summary("minimal")
        assert "none yet" in minimal


class TestWorkflowBuilderTieredSummary:
    def test_minimal_compact(self, workflow_builder: WorkflowArtifactBuilder):
        workflow_builder.set_overview("wf1", "Flow", "A flow")
        workflow_builder.add_step("step1", "auto_transition", "First", "First step")
        minimal = workflow_builder.get_summary("minimal")
        full = workflow_builder.get_summary("full")
        assert len(minimal) < len(full)
        assert "step1" in minimal

    def test_full_matches_default(self, workflow_builder: WorkflowArtifactBuilder):
        workflow_builder.set_overview("wf1", "Flow", "A flow")
        assert workflow_builder.get_summary() == workflow_builder.get_summary("full")


class TestAgentBuilderTieredSummary:
    def test_minimal_compact(self, agent_builder: AgentArtifactBuilder):
        agent_builder.set_agent_type("react")
        agent_builder.set_overview("Bot", "A bot")
        agent_builder.add_tool("search", "Search the web")
        minimal = agent_builder.get_summary("minimal")
        full = agent_builder.get_summary("full")
        assert len(minimal) < len(full)
        assert "search" in minimal

    def test_full_matches_default(self, agent_builder: AgentArtifactBuilder):
        agent_builder.set_agent_type("react")
        assert agent_builder.get_summary() == agent_builder.get_summary("full")


# ===================================================================
# Fluent mutators, take_warnings(), strict build()
# ===================================================================


def _complete_fsm() -> FSMArtifactBuilder:
    return (
        FSMArtifactBuilder()
        .set_overview("Bot", "A bot")
        .add_state("a", "A", "Start")
        .add_state("b", "B", "End")
        .add_transition("a", "b", "go")
    )


def _complete_workflow() -> WorkflowArtifactBuilder:
    return (
        WorkflowArtifactBuilder()
        .set_overview("wf", "Flow", "A flow")
        .add_step("s1", "auto_transition", "One")
        .add_step("s2", "auto_transition", "Two")
        .set_step_transition("s1", "s2")
    )


def _complete_agent() -> AgentArtifactBuilder:
    return (
        AgentArtifactBuilder()
        .set_overview("Ag", "An agent")
        .set_agent_type("react")
        .add_tool("search", "Search")
    )


class TestFluentMutators:
    def test_fsm_mutators_return_same_builder(self):
        b = FSMArtifactBuilder()
        assert b.set_overview("n", "d") is b
        assert b.add_state("a", "A", "p") is b
        assert b.add_state("b", "B", "p") is b
        assert b.update_state("a", description="x") is b
        assert b.add_transition("a", "b", "go") is b
        assert b.set_initial_state("b") is b
        assert b.remove_transition("a", "b") is b
        assert b.remove_state("b") is b
        assert b.remove_state("missing") is b

    def test_workflow_mutators_return_same_builder(self):
        b = WorkflowArtifactBuilder()
        assert b.set_overview("w", "n", "d") is b
        assert b.add_step("s1", "timer", "One") is b
        assert b.add_step("s2", "timer", "Two") is b
        assert b.set_step_transition("s1", "s2") is b
        assert b.set_initial_step("s2") is b
        assert b.remove_step("s2") is b
        assert b.remove_step("missing") is b

    def test_agent_mutators_return_same_builder(self):
        b = AgentArtifactBuilder()
        assert b.set_overview("n", "d") is b
        assert b.set_agent_type("react") is b
        assert b.set_config(model="m") is b
        assert b.add_tool("t", "d") is b
        assert b.remove_tool("t") is b
        assert b.remove_tool("missing") is b

    def test_take_warnings_returns_then_clears(self):
        b = FSMArtifactBuilder()
        b.add_state("a", "A", "p")
        b.add_state("a", "A2", "p")
        warnings = b.take_warnings()
        assert any("Auto-set initial state" in w for w in warnings)
        assert any("already exists" in w for w in warnings)
        assert b.take_warnings() == []

    def test_take_warnings_without_mutation_is_empty(self):
        assert AgentArtifactBuilder().take_warnings() == []

    def test_call_time_refusal_still_raises(self):
        with pytest.raises(BuilderError) as exc:
            FSMArtifactBuilder().add_state("", "d", "p")
        assert exc.value.action == "add_state"


class TestStrictBuild:
    @pytest.mark.parametrize(
        "make", [FSMArtifactBuilder, WorkflowArtifactBuilder, AgentArtifactBuilder]
    )
    def test_incomplete_raises_build_error(self, make):
        from fsm_llm import BuildError

        builder = make()
        expected = builder.validate_complete()
        assert expected
        with pytest.raises(BuildError) as exc:
            builder.build()
        assert exc.value.errors == expected

    @pytest.mark.parametrize(
        "make", [_complete_fsm, _complete_workflow, _complete_agent]
    )
    def test_complete_returns_dict_equal_to_to_dict(self, make):
        builder = make()
        assert builder.validate_complete() == []
        assert builder.build() == builder.to_dict()

    @pytest.mark.parametrize(
        "make", [_complete_fsm, _complete_workflow, _complete_agent]
    )
    def test_build_twice_is_isolated(self, make):
        builder = make()
        first = builder.build()
        second = builder.build()
        assert first == second
        assert first is not second
        first["name"] = "mutated"
        for value in first.values():
            if isinstance(value, dict):
                value["poison"] = True
            elif isinstance(value, list):
                value.append("poison")
        assert builder.build() == second

    def test_builder_mutation_after_build_does_not_change_product(self):
        builder = _complete_fsm()
        product = builder.build()
        builder.update_state("a", description="changed")
        builder.add_transition("a", "b", "again")
        assert product["states"]["a"]["description"] == "A"
        assert len(product["states"]["a"]["transitions"]) == 1

    def test_workflow_build_isolated_from_nested_steps(self):
        builder = _complete_workflow()
        product = builder.build()
        builder.set_step_transition("s2", "s1")
        assert product["steps"]["s2"]["transitions"] == []


class TestBuilderErrorHierarchy:
    def test_is_both_build_error_and_meta_builder_error(self):
        from fsm_llm import BuildError
        from fsm_llm.agents.exceptions import MetaBuilderError

        err = BuilderError("boom", action="add_state", details={"k": 1})
        assert isinstance(err, BuildError)
        assert isinstance(err, MetaBuilderError)
        assert isinstance(err, ValueError)
        assert err.action == "add_state"
        assert err.details == {"k": 1}
        assert err.errors == ["boom"]
        assert str(err) == "boom"


class TestFSMBuilderReportsAnOrphanOnce:
    """F-XD-12: the builder's own reachability walk no longer repeats the fact
    FSMDefinition already reported (plan 5980fbb7, step 6)."""

    def test_one_orphan_is_one_error(self):
        b = FSMArtifactBuilder()
        b.set_overview("Test", "Test FSM")
        b.add_state("start", "Start", "Start")
        b.add_state("mid", "Mid", "Mid")
        b.add_state("orphan", "Orphan", "Orphan")
        b.add_transition("start", "mid", "go")
        b.initial_state = "start"
        errors = [e for e in b.validate_complete() if "orphan" in e.lower()]
        assert len(errors) == 1, errors

    def test_an_orphan_is_still_reported_when_a_field_error_hides_the_model_check(
        self,
    ):
        b = FSMArtifactBuilder()
        b.set_overview("Test", "Test FSM")
        b.add_state("start", "Start", "Start")
        b.add_state("mid", "Mid", "Mid")
        b.add_state("orphan", "Orphan", "Orphan")
        b.add_transition("start", "mid", "go")
        b.initial_state = "start"
        b.states["mid"]["description"] = "x" * 5000
        errors = b.validate_complete()
        assert any("orphan" in e.lower() for e in errors), errors


# ===================================================================
# Package layout (plan d454fd62, step 89)
# ===================================================================

_BUILDER_MODULE_OF = {
    "ArtifactBuilder": "base",
    "FSMArtifactBuilder": "fsm",
    "WorkflowArtifactBuilder": "workflow",
    "AgentArtifactBuilder": "agent",
}


class TestMetaBuildersPackageLayout:
    """``agents/meta_builders.py`` is a package at the same dotted path."""

    def test_the_facade_names_exactly_the_four_builder_classes(self):
        import fsm_llm.agents.meta_builders as package

        assert set(package.__all__) == set(_BUILDER_MODULE_OF)
        assert len(package.__all__) == 4
        for name in package.__all__:
            assert hasattr(package, name), name

    def test_each_class_is_defined_in_its_own_module(self):
        import fsm_llm.agents.meta_builders as package

        for name, module in _BUILDER_MODULE_OF.items():
            assert getattr(package, name).__module__ == (
                f"fsm_llm.agents.meta_builders.{module}"
            ), name

    def test_the_three_builders_share_the_one_abstract_base(self):
        from fsm_llm.agents.meta_builders import ArtifactBuilder

        for cls in (FSMArtifactBuilder, WorkflowArtifactBuilder, AgentArtifactBuilder):
            assert cls.__mro__[1] is ArtifactBuilder
        with pytest.raises(TypeError, match="abstract"):
            ArtifactBuilder()  # type: ignore[abstract]

    def test_the_template_model_and_its_anchor_stay_in_the_agent_module(self):
        from pathlib import Path

        import fsm_llm.agents.meta_builders.agent as agent_module
        from fsm_llm.agents.constants import MetaDefaults

        assert AgentArtifactBuilder().config["model"] == MetaDefaults.AGENT_MODEL
        text = Path(agent_module.__file__).read_text(encoding="utf-8")
        assert "MetaDefaults.AGENT_MODEL" in text
        assert "Do NOT switch" in text
        # the anchor of the call-time refusals sits in the FSM module, the lazy
        # warnings anchor in the base, the workflow load-check anchor in the workflow
        for module, token in (
            ("fsm", "89b03f61/D-005"),
            ("base", "042b086f/D-012"),
            ("workflow", "3e4eb3e5/D-004"),
            ("agent", "0068b055/D-011"),
        ):
            source = Path(agent_module.__file__).with_name(f"{module}.py")
            assert token in source.read_text(encoding="utf-8"), (module, token)

    def test_a_log_record_is_named_after_its_module(self):
        from loguru import logger

        names: list[str] = []
        logger.enable("fsm_llm")
        sink = logger.add(lambda m: names.append(m.record["name"]), level="DEBUG")
        try:
            b = FSMArtifactBuilder()
            b.add_state("a", "d", "p")
            b.add_state("b", "d", "p")
            b.add_transition("a", "b", "go")
            b.remove_state("b")
        finally:
            logger.remove(sink)
            logger.disable("fsm_llm")
        assert names == ["fsm_llm.agents.meta_builders.fsm"] * 4

    def test_the_old_module_attributes_are_not_on_the_package(self):
        import fsm_llm.agents.meta_builders as package

        for name in ("logger", "MetaDefaults", "BuildError", "FSMDefinition", "copy"):
            assert not hasattr(package, name), name


class TestSerializationAndProgressEdges:
    """Two edges the step 89 mutation proof found no test for (only a differential saw others)."""

    def test_to_dict_returns_copies_of_the_top_level_containers(self):
        fsm = FSMArtifactBuilder().set_overview("Bot", "d")
        fsm.add_state("a", "d", "p")
        wf = WorkflowArtifactBuilder().set_overview("wf", "W", "d")
        wf.add_step("a", "timer", "A")
        agent = AgentArtifactBuilder().set_overview("A", "d")
        agent.add_tool("t", "d")

        fsm.to_dict()["states"]["zzz"] = {}
        wf.to_dict()["steps"]["zzz"] = {}
        agent.to_dict()["tools"].append({"name": "zzz"})
        agent.to_dict()["config"]["model"] = "zzz"

        assert list(fsm.states) == ["a"]
        assert list(wf.steps) == ["a"]
        assert [t["name"] for t in agent.tools] == ["t"]
        assert agent.config["model"] != "zzz"

    def test_progress_never_reports_a_negative_completed_count(self):
        b = FSMArtifactBuilder()
        for sid in ("a", "b", "c"):
            b.add_state(sid, "", "")
        missing = b.get_missing_fields()
        assert len(missing) > b._TOTAL_REQUIRED
        progress = b.get_progress()
        assert progress.total_required == b._TOTAL_REQUIRED
        assert progress.completed == 0
        assert progress.missing == missing


class TestBuilderEdgesOnlyTheDifferentialSaw:
    """Edges of the builders that no earlier test pinned (step 89 mutation proof).

    Each case was a mutant of the split that only a 400-program differential killed.
    """

    def test_the_minimal_summary_counts_edges_not_nodes(self):
        fsm = FSMArtifactBuilder().set_overview("Bot", "d")
        for sid in ("a", "b", "c"):
            fsm.add_state(sid, "d", "p")
        fsm.add_transition("a", "b", "x").add_transition("a", "c", "y")
        assert "Transitions: 2 defined" in fsm.get_summary("minimal").splitlines()

        wf = WorkflowArtifactBuilder().set_overview("wf", "W", "d")
        for sid in ("a", "b", "c"):
            wf.add_step(sid, "timer", sid)
        wf.set_step_transition("a", "b").set_step_transition("b", "c")
        wf.set_step_transition("a", "c")
        assert "Connections: 3 defined" in wf.get_summary("minimal").splitlines()

    def test_update_state_strips_the_value(self):
        b = FSMArtifactBuilder()
        b.add_state("a", "d", "p")
        b.update_state("a", description="  padded  ", purpose="\tp2\n")
        assert b.states["a"]["description"] == "padded"
        assert b.states["a"]["purpose"] == "p2"

    def test_a_graph_with_no_terminal_state_is_missing_one(self):
        b = FSMArtifactBuilder().set_overview("Bot", "d")
        b.add_state("a", "d", "p").add_state("b", "d", "p")
        b.add_transition("a", "b", "x").add_transition("b", "a", "y")
        assert b.get_missing_fields() == [
            "At least one terminal state (state with no outgoing transitions)"
        ]
        b.add_state("end", "d", "p")
        assert b.get_missing_fields() == []

    def test_remove_transition_removes_only_the_named_target(self):
        b = FSMArtifactBuilder()
        for sid in ("a", "b", "c"):
            b.add_state(sid, "d", "p")
        b.add_transition("a", "b", "x").add_transition("a", "c", "y")
        assert b.remove_transition("a", "zzz") is b
        assert b.remove_transition("nope", "b") is b
        assert [t["target_state"] for t in b.states["a"]["transitions"]] == ["b", "c"]
        b.remove_transition("a", "b")
        assert [t["target_state"] for t in b.states["a"]["transitions"]] == ["c"]

    def test_the_standard_workflow_summary_cuts_a_step_name(self):
        from fsm_llm.agents.constants import MetaDefaults

        width = MetaDefaults.SUMMARY_LINE_WIDTH
        wf = WorkflowArtifactBuilder().set_overview("wf", "W", "d")
        wf.add_step("a", "timer", "n" * (width + 25))
        line = next(ln for ln in wf.get_summary("standard").splitlines() if "- a" in ln)
        assert line.endswith(": " + "n" * width)

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("react_agent", "react"),
            ("reactagent", "react"),
            ("plan_and_execute", "plan_execute"),
            ("plan_execute_agent", "plan_execute"),
            ("planexecute", "plan_execute"),
            ("self_consistent", "self_consistency"),
            ("eval_opt", "evaluator_optimizer"),
            ("eval_optimize", "evaluator_optimizer"),
            ("maker_check", "maker_checker"),
            ("reflexion_agent", "reflexion"),
            ("debate_agent", "debate"),
            ("orchestrate", "orchestrator"),
            ("adapt_agent", "adapt"),
            ("prompt_chaining", "prompt_chain"),
            ("rewoo_agent", "rewoo"),
        ],
    )
    def test_every_agent_type_alias_resolves_to_its_pattern(self, alias, canonical):
        b = AgentArtifactBuilder().set_agent_type(f"  {alias.upper()} ")
        assert b.agent_type == canonical

    def test_a_config_without_a_model_still_reports_the_other_keys(self, monkeypatch):
        monkeypatch.delenv("LLM_MODEL", raising=False)
        problems = AgentArtifactBuilder._config_problems({"max_iterations": 0})
        assert set(problems) == {"model", "max_iterations"}
        assert "No LLM model is set" in problems["model"]
        assert problems["max_iterations"] == "max_iterations must be at least 1"

    def test_a_validator_reason_loses_the_pydantic_prefix(self):
        config = {"model": "m", "temperature": float("nan")}
        reason = AgentArtifactBuilder._config_problems(config)["temperature"]
        assert not reason.startswith("Value error")
