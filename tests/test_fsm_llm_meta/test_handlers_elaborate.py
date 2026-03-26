from __future__ import annotations

"""Elaborate tests for meta-agent handlers: workflow/agent dispatch,
classification integration, normalize_decision edge cases, and
action dispatch edge cases."""

import pytest

from fsm_llm_meta.builders import AgentBuilder, FSMBuilder, WorkflowBuilder
from fsm_llm_meta.constants import Actions, ContextKeys
from fsm_llm_meta.definitions import ArtifactType
from fsm_llm_meta.exceptions import BuilderError
from fsm_llm_meta.handlers import MetaHandlers

# ---- Fixtures --------------------------------------------------------


@pytest.fixture
def handlers() -> MetaHandlers:
    return MetaHandlers()


@pytest.fixture
def handlers_with_workflow(handlers: MetaHandlers) -> MetaHandlers:
    handlers._ensure_builder("workflow")
    assert isinstance(handlers.builder, WorkflowBuilder)
    return handlers


@pytest.fixture
def handlers_with_agent(handlers: MetaHandlers) -> MetaHandlers:
    handlers._ensure_builder("agent")
    assert isinstance(handlers.builder, AgentBuilder)
    return handlers


@pytest.fixture
def handlers_with_fsm(handlers: MetaHandlers) -> MetaHandlers:
    handlers._ensure_builder("fsm")
    assert isinstance(handlers.builder, FSMBuilder)
    return handlers


# ---- Classification / Artifact Type Validation -----------------------


class TestClassifyArtifactType:
    """Tests for the classification-schema-based type validation handler."""

    def test_valid_fsm(self, handlers: MetaHandlers):
        result = handlers.classify_artifact_type({ContextKeys.ARTIFACT_TYPE: "fsm"})
        assert result[ContextKeys.ARTIFACT_TYPE] == "fsm"

    def test_valid_workflow(self, handlers: MetaHandlers):
        result = handlers.classify_artifact_type(
            {ContextKeys.ARTIFACT_TYPE: "workflow"}
        )
        assert result[ContextKeys.ARTIFACT_TYPE] == "workflow"

    def test_valid_agent(self, handlers: MetaHandlers):
        result = handlers.classify_artifact_type({ContextKeys.ARTIFACT_TYPE: "agent"})
        assert result[ContextKeys.ARTIFACT_TYPE] == "agent"

    def test_alias_chatbot_resolves_to_fsm(self, handlers: MetaHandlers):
        result = handlers.classify_artifact_type({ContextKeys.ARTIFACT_TYPE: "chatbot"})
        assert result[ContextKeys.ARTIFACT_TYPE] == "fsm"

    def test_alias_pipeline_resolves_to_workflow(self, handlers: MetaHandlers):
        result = handlers.classify_artifact_type(
            {ContextKeys.ARTIFACT_TYPE: "pipeline"}
        )
        assert result[ContextKeys.ARTIFACT_TYPE] == "workflow"

    def test_alias_tools_resolves_to_agent(self, handlers: MetaHandlers):
        result = handlers.classify_artifact_type({ContextKeys.ARTIFACT_TYPE: "tools"})
        assert result[ContextKeys.ARTIFACT_TYPE] == "agent"

    def test_alias_react_resolves_to_agent(self, handlers: MetaHandlers):
        result = handlers.classify_artifact_type({ContextKeys.ARTIFACT_TYPE: "react"})
        assert result[ContextKeys.ARTIFACT_TYPE] == "agent"

    def test_alias_state_machine_resolves_to_fsm(self, handlers: MetaHandlers):
        result = handlers.classify_artifact_type(
            {ContextKeys.ARTIFACT_TYPE: "state machine"}
        )
        assert result[ContextKeys.ARTIFACT_TYPE] == "fsm"

    def test_invalid_type_clears_context(self, handlers: MetaHandlers):
        result = handlers.classify_artifact_type(
            {ContextKeys.ARTIFACT_TYPE: "spaceship"}
        )
        assert result[ContextKeys.ARTIFACT_TYPE] is None

    def test_case_insensitive(self, handlers: MetaHandlers):
        result = handlers.classify_artifact_type({ContextKeys.ARTIFACT_TYPE: "FSM"})
        assert result[ContextKeys.ARTIFACT_TYPE] == "fsm"

    def test_whitespace_stripped(self, handlers: MetaHandlers):
        result = handlers.classify_artifact_type(
            {ContextKeys.ARTIFACT_TYPE: "  workflow  "}
        )
        assert result[ContextKeys.ARTIFACT_TYPE] == "workflow"

    def test_empty_string_returns_empty(self, handlers: MetaHandlers):
        result = handlers.classify_artifact_type({ContextKeys.ARTIFACT_TYPE: ""})
        assert result == {}

    def test_none_returns_empty(self, handlers: MetaHandlers):
        result = handlers.classify_artifact_type({ContextKeys.ARTIFACT_TYPE: None})
        assert result == {}

    def test_missing_key_returns_empty(self, handlers: MetaHandlers):
        result = handlers.classify_artifact_type({})
        assert result == {}

    def test_non_string_returns_empty(self, handlers: MetaHandlers):
        result = handlers.classify_artifact_type({ContextKeys.ARTIFACT_TYPE: 42})
        assert result == {}

    def test_schema_has_three_intents(self):
        schema = MetaHandlers._ARTIFACT_SCHEMA
        assert len(schema.intents) == 3
        assert set(schema.intent_names) == {"fsm", "workflow", "agent"}

    def test_schema_fallback_is_fsm(self):
        assert MetaHandlers._ARTIFACT_SCHEMA.fallback_intent == "fsm"


# ---- Builder Initialization Edge Cases -------------------------------


class TestEnsureBuilderEdgeCases:
    def test_type_aliases_are_classvar(self):
        """Type aliases should be a ClassVar on the class, not local."""
        assert hasattr(MetaHandlers, "_TYPE_ALIASES")
        assert isinstance(MetaHandlers._TYPE_ALIASES, dict)

    def test_alias_bot(self, handlers: MetaHandlers):
        handlers._ensure_builder("bot")
        assert handlers._artifact_type == ArtifactType.FSM

    def test_alias_automation(self, handlers: MetaHandlers):
        handlers._ensure_builder("automation")
        assert handlers._artifact_type == ArtifactType.WORKFLOW

    def test_alias_agentic(self, handlers: MetaHandlers):
        handlers._ensure_builder("agentic")
        assert handlers._artifact_type == ArtifactType.AGENT

    def test_unknown_raises_builder_error(self, handlers: MetaHandlers):
        with pytest.raises(BuilderError, match="Unknown artifact type"):
            handlers._ensure_builder("spaceship")

    def test_inject_builder_state_tolerates_unknown_type(self, handlers: MetaHandlers):
        """inject_builder_state should not crash on unknown artifact type;
        it should catch the BuilderError and return empty."""
        result = handlers.inject_builder_state({ContextKeys.ARTIFACT_TYPE: "spaceship"})
        assert result == {}
        assert handlers.builder is None


# ---- Workflow Action Dispatch ----------------------------------------


class TestWorkflowActionDispatch:
    def test_add_step(self, handlers_with_workflow: MetaHandlers):
        context = {
            ContextKeys.ACTION: Actions.ADD_STEP,
            ContextKeys.ACTION_PARAMS: {
                "step_id": "start",
                "step_type": "auto_transition",
                "name": "Start Step",
                "description": "Entry point",
            },
        }
        result = handlers_with_workflow.dispatch_action(context)
        assert ContextKeys.ACTION_RESULT in result
        assert "start" in result[ContextKeys.ACTION_RESULT]

        builder = handlers_with_workflow.builder
        assert isinstance(builder, WorkflowBuilder)
        assert "start" in builder.steps

    def test_remove_step(self, handlers_with_workflow: MetaHandlers):
        builder = handlers_with_workflow.builder
        assert isinstance(builder, WorkflowBuilder)
        builder.add_step("s1", "auto_transition", "Step 1")

        context = {
            ContextKeys.ACTION: Actions.REMOVE_STEP,
            ContextKeys.ACTION_PARAMS: {"step_id": "s1"},
        }
        result = handlers_with_workflow.dispatch_action(context)
        assert ContextKeys.ACTION_RESULT in result
        assert "s1" not in builder.steps

    def test_set_step_transition(self, handlers_with_workflow: MetaHandlers):
        builder = handlers_with_workflow.builder
        assert isinstance(builder, WorkflowBuilder)
        builder.add_step("s1", "auto_transition", "Step 1")
        builder.add_step("s2", "auto_transition", "Step 2")

        context = {
            ContextKeys.ACTION: Actions.SET_STEP_TRANSITION,
            ContextKeys.ACTION_PARAMS: {
                "from_step": "s1",
                "to_step": "s2",
            },
        }
        result = handlers_with_workflow.dispatch_action(context)
        assert ContextKeys.ACTION_RESULT in result
        assert len(builder.steps["s1"]["transitions"]) == 1

    def test_set_initial_step(self, handlers_with_workflow: MetaHandlers):
        builder = handlers_with_workflow.builder
        assert isinstance(builder, WorkflowBuilder)
        builder.add_step("s1", "auto_transition", "Step 1")
        builder.add_step("s2", "auto_transition", "Step 2")

        context = {
            ContextKeys.ACTION: Actions.SET_INITIAL_STEP,
            ContextKeys.ACTION_PARAMS: {"step_id": "s2"},
        }
        result = handlers_with_workflow.dispatch_action(context)
        assert ContextKeys.ACTION_RESULT in result
        assert builder.initial_step_id == "s2"

    def test_unknown_workflow_action(self, handlers_with_workflow: MetaHandlers):
        context = {
            ContextKeys.ACTION: "unknown_action",
            ContextKeys.ACTION_PARAMS: {},
        }
        result = handlers_with_workflow.dispatch_action(context)
        assert "Unknown workflow action" in result[ContextKeys.ACTION_RESULT]


# ---- Agent Action Dispatch -------------------------------------------


class TestAgentActionDispatch:
    def test_set_agent_type(self, handlers_with_agent: MetaHandlers):
        context = {
            ContextKeys.ACTION: Actions.SET_AGENT_TYPE,
            ContextKeys.ACTION_PARAMS: {"agent_type": "react"},
        }
        result = handlers_with_agent.dispatch_action(context)
        assert ContextKeys.ACTION_RESULT in result

        builder = handlers_with_agent.builder
        assert isinstance(builder, AgentBuilder)
        assert builder.agent_type == "react"

    def test_set_agent_type_invalid_returns_error(
        self, handlers_with_agent: MetaHandlers
    ):
        context = {
            ContextKeys.ACTION: Actions.SET_AGENT_TYPE,
            ContextKeys.ACTION_PARAMS: {"agent_type": "invalid_type"},
        }
        result = handlers_with_agent.dispatch_action(context)
        assert ContextKeys.ACTION_ERRORS in result

    def test_set_config(self, handlers_with_agent: MetaHandlers):
        context = {
            ContextKeys.ACTION: Actions.SET_CONFIG,
            ContextKeys.ACTION_PARAMS: {
                "model": "gpt-4o",
                "temperature": 0.3,
            },
        }
        result = handlers_with_agent.dispatch_action(context)
        assert ContextKeys.ACTION_RESULT in result

        builder = handlers_with_agent.builder
        assert isinstance(builder, AgentBuilder)
        assert builder.config["model"] == "gpt-4o"
        assert builder.config["temperature"] == 0.3

    def test_add_tool(self, handlers_with_agent: MetaHandlers):
        context = {
            ContextKeys.ACTION: Actions.ADD_TOOL,
            ContextKeys.ACTION_PARAMS: {
                "name": "search",
                "description": "Search the web",
            },
        }
        result = handlers_with_agent.dispatch_action(context)
        assert ContextKeys.ACTION_RESULT in result

        builder = handlers_with_agent.builder
        assert isinstance(builder, AgentBuilder)
        assert len(builder.tools) == 1
        assert builder.tools[0]["name"] == "search"

    def test_remove_tool(self, handlers_with_agent: MetaHandlers):
        builder = handlers_with_agent.builder
        assert isinstance(builder, AgentBuilder)
        builder.add_tool("search", "Search the web")

        context = {
            ContextKeys.ACTION: Actions.REMOVE_TOOL,
            ContextKeys.ACTION_PARAMS: {"name": "search"},
        }
        result = handlers_with_agent.dispatch_action(context)
        assert ContextKeys.ACTION_RESULT in result
        assert len(builder.tools) == 0

    def test_remove_nonexistent_tool(self, handlers_with_agent: MetaHandlers):
        context = {
            ContextKeys.ACTION: Actions.REMOVE_TOOL,
            ContextKeys.ACTION_PARAMS: {"name": "nonexistent"},
        }
        result = handlers_with_agent.dispatch_action(context)
        assert "not found" in result[ContextKeys.ACTION_RESULT]

    def test_unknown_agent_action(self, handlers_with_agent: MetaHandlers):
        context = {
            ContextKeys.ACTION: "unknown_action",
            ContextKeys.ACTION_PARAMS: {},
        }
        result = handlers_with_agent.dispatch_action(context)
        assert "Unknown agent action" in result[ContextKeys.ACTION_RESULT]


# ---- Normalize Decision Edge Cases -----------------------------------


class TestNormalizeDecisionEdgeCases:
    def test_approve_canonical(self, handlers: MetaHandlers):
        result = handlers.normalize_decision({ContextKeys.USER_DECISION: "approve"})
        assert result[ContextKeys.USER_DECISION] == "approve"

    def test_revise_canonical(self, handlers: MetaHandlers):
        result = handlers.normalize_decision({ContextKeys.USER_DECISION: "revise"})
        assert result[ContextKeys.USER_DECISION] == "revise"

    def test_lgtm_normalizes_to_approve(self, handlers: MetaHandlers):
        result = handlers.normalize_decision({ContextKeys.USER_DECISION: "lgtm"})
        assert result[ContextKeys.USER_DECISION] == "approve"

    def test_sounds_good_normalizes_to_approve(self, handlers: MetaHandlers):
        result = handlers.normalize_decision({ContextKeys.USER_DECISION: "sounds good"})
        assert result[ContextKeys.USER_DECISION] == "approve"

    def test_ship_it_normalizes_to_approve(self, handlers: MetaHandlers):
        result = handlers.normalize_decision({ContextKeys.USER_DECISION: "ship it"})
        assert result[ContextKeys.USER_DECISION] == "approve"

    def test_nope_normalizes_to_revise(self, handlers: MetaHandlers):
        result = handlers.normalize_decision({ContextKeys.USER_DECISION: "nope"})
        assert result[ContextKeys.USER_DECISION] == "revise"

    def test_fix_normalizes_to_revise(self, handlers: MetaHandlers):
        result = handlers.normalize_decision({ContextKeys.USER_DECISION: "fix"})
        assert result[ContextKeys.USER_DECISION] == "revise"

    def test_try_again_normalizes_to_revise(self, handlers: MetaHandlers):
        result = handlers.normalize_decision({ContextKeys.USER_DECISION: "try again"})
        assert result[ContextKeys.USER_DECISION] == "revise"

    def test_unknown_decision_returns_empty(self, handlers: MetaHandlers):
        result = handlers.normalize_decision({ContextKeys.USER_DECISION: "maybe later"})
        assert result == {}

    def test_none_returns_empty(self, handlers: MetaHandlers):
        result = handlers.normalize_decision({ContextKeys.USER_DECISION: None})
        assert result == {}

    def test_non_string_returns_empty(self, handlers: MetaHandlers):
        result = handlers.normalize_decision({ContextKeys.USER_DECISION: 42})
        assert result == {}

    def test_missing_key_returns_empty(self, handlers: MetaHandlers):
        result = handlers.normalize_decision({})
        assert result == {}

    def test_case_insensitive(self, handlers: MetaHandlers):
        result = handlers.normalize_decision({ContextKeys.USER_DECISION: "APPROVE"})
        assert result[ContextKeys.USER_DECISION] == "approve"

    def test_whitespace_stripped(self, handlers: MetaHandlers):
        result = handlers.normalize_decision({ContextKeys.USER_DECISION: "  revise  "})
        assert result[ContextKeys.USER_DECISION] == "revise"


# ---- Action Dispatch Edge Cases --------------------------------------


class TestDispatchActionEdgeCases:
    def test_malformed_params_uses_empty_dict(self, handlers_with_fsm: MetaHandlers):
        """If action_params is not a dict, should use empty dict."""
        context = {
            ContextKeys.ACTION: Actions.DONE,
            ContextKeys.ACTION_PARAMS: "not a dict",
        }
        result = handlers_with_fsm.dispatch_action(context)
        assert result.get(ContextKeys.STRUCTURE_DONE) is True

    def test_done_variants_normalized(self, handlers_with_fsm: MetaHandlers):
        for variant in ["finish", "complete", "ready", "move on", "proceed"]:
            handlers_with_fsm.builder = FSMBuilder()
            context = {ContextKeys.ACTION: variant}
            result = handlers_with_fsm.dispatch_action(context)
            assert result.get(ContextKeys.STRUCTURE_DONE) is True, (
                f"Variant '{variant}' should normalize to done"
            )

    def test_builder_not_initialized_returns_message(self, handlers: MetaHandlers):
        """Dispatch with no builder should return a message, not crash."""
        handlers.builder = None
        context = {ContextKeys.ACTION: Actions.ADD_STATE, ContextKeys.ACTION_PARAMS: {}}
        result = handlers.dispatch_action(context)
        assert "not initialized" in result.get(ContextKeys.ACTION_RESULT, "")

    def test_update_state_does_not_mutate_params(self, handlers_with_fsm: MetaHandlers):
        """Bug fix: update_state should not pop from the params dict."""
        builder = handlers_with_fsm.builder
        assert isinstance(builder, FSMBuilder)
        builder.add_state("s1", "State 1", "Purpose")

        params = {"state_id": "s1", "description": "Updated"}
        context = {
            ContextKeys.ACTION: Actions.UPDATE_STATE,
            ContextKeys.ACTION_PARAMS: params,
        }
        handlers_with_fsm.dispatch_action(context)
        # params should NOT have state_id removed
        assert "state_id" in params


# ---- Handle Overview for Workflow/Agent ------------------------------


class TestHandleOverviewMultiType:
    def test_workflow_overview(self, handlers_with_workflow: MetaHandlers):
        context = {
            ContextKeys.ARTIFACT_NAME: "Order Pipeline",
            ContextKeys.ARTIFACT_DESCRIPTION: "Processes orders",
        }
        handlers_with_workflow.handle_overview(context)
        builder = handlers_with_workflow.builder
        assert isinstance(builder, WorkflowBuilder)
        assert builder.name == "Order Pipeline"
        assert builder.workflow_id == "order_pipeline"

    def test_agent_overview(self, handlers_with_agent: MetaHandlers):
        context = {
            ContextKeys.ARTIFACT_NAME: "SearchBot",
            ContextKeys.ARTIFACT_DESCRIPTION: "Searches things",
        }
        handlers_with_agent.handle_overview(context)
        builder = handlers_with_agent.builder
        assert isinstance(builder, AgentBuilder)
        assert builder.name == "SearchBot"
        assert builder.description == "Searches things"
