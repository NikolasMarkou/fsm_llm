from __future__ import annotations

"""Tests for meta-agent handlers."""

import pytest

from fsm_llm_meta.builders import FSMBuilder
from fsm_llm_meta.constants import Actions, ContextKeys
from fsm_llm_meta.definitions import ArtifactType
from fsm_llm_meta.handlers import MetaHandlers


@pytest.fixture
def handlers() -> MetaHandlers:
    """Fresh handlers instance."""
    return MetaHandlers()


@pytest.fixture
def handlers_with_fsm_builder(handlers: MetaHandlers) -> MetaHandlers:
    """Handlers with an FSM builder initialized."""
    handlers._ensure_builder("fsm")
    assert isinstance(handlers.builder, FSMBuilder)
    return handlers


class TestBuilderInitialization:
    def test_ensure_builder_fsm(self, handlers: MetaHandlers):
        handlers._ensure_builder("fsm")
        assert isinstance(handlers.builder, FSMBuilder)
        assert handlers._artifact_type == ArtifactType.FSM

    def test_ensure_builder_workflow(self, handlers: MetaHandlers):
        handlers._ensure_builder("workflow")
        assert handlers._artifact_type == ArtifactType.WORKFLOW

    def test_ensure_builder_agent(self, handlers: MetaHandlers):
        handlers._ensure_builder("agent")
        assert handlers._artifact_type == ArtifactType.AGENT

    def test_ensure_builder_unknown_defaults_to_fsm(self, handlers: MetaHandlers):
        handlers._ensure_builder("unknown")
        assert handlers._artifact_type == ArtifactType.FSM

    def test_ensure_builder_idempotent(self, handlers: MetaHandlers):
        handlers._ensure_builder("fsm")
        builder1 = handlers.builder
        handlers._ensure_builder("workflow")  # should not change
        assert handlers.builder is builder1

    def test_reset(self, handlers_with_fsm_builder: MetaHandlers):
        handlers_with_fsm_builder.reset()
        assert handlers_with_fsm_builder.builder is None
        assert handlers_with_fsm_builder._artifact_type is None


class TestInjectBuilderState:
    def test_no_builder_returns_empty(self, handlers: MetaHandlers):
        result = handlers.inject_builder_state({})
        assert result == {}

    def test_initializes_builder_on_artifact_type(self, handlers: MetaHandlers):
        result = handlers.inject_builder_state({ContextKeys.ARTIFACT_TYPE: "fsm"})
        assert handlers.builder is not None
        assert ContextKeys.BUILDER_SUMMARY in result

    def test_injects_summary_and_progress(
        self, handlers_with_fsm_builder: MetaHandlers
    ):
        result = handlers_with_fsm_builder.inject_builder_state({})
        assert ContextKeys.BUILDER_SUMMARY in result
        assert ContextKeys.BUILDER_PROGRESS in result
        assert ContextKeys.BUILDER_MISSING in result


class TestHandleOverview:
    def test_sets_fsm_overview(self, handlers_with_fsm_builder: MetaHandlers):
        context = {
            ContextKeys.ARTIFACT_NAME: "MyBot",
            ContextKeys.ARTIFACT_DESCRIPTION: "A test bot",
            ContextKeys.ARTIFACT_PERSONA: "Friendly",
        }
        handlers_with_fsm_builder.handle_overview(context)
        builder = handlers_with_fsm_builder.builder
        assert isinstance(builder, FSMBuilder)
        assert builder.name == "MyBot"
        assert builder.description == "A test bot"
        assert builder.persona == "Friendly"

    def test_no_builder_returns_empty(self, handlers: MetaHandlers):
        result = handlers.handle_overview({ContextKeys.ARTIFACT_NAME: "Bot"})
        assert result == {}


class TestDispatchAction:
    def test_no_action_returns_empty(self, handlers_with_fsm_builder: MetaHandlers):
        result = handlers_with_fsm_builder.dispatch_action({})
        assert result == {}

    def test_add_state(self, handlers_with_fsm_builder: MetaHandlers):
        context = {
            ContextKeys.ACTION: Actions.ADD_STATE,
            ContextKeys.ACTION_PARAMS: {
                "state_id": "greeting",
                "description": "Greet user",
                "purpose": "Welcome",
            },
        }
        result = handlers_with_fsm_builder.dispatch_action(context)
        assert ContextKeys.ACTION_RESULT in result
        assert "greeting" in result[ContextKeys.ACTION_RESULT]

        builder = handlers_with_fsm_builder.builder
        assert isinstance(builder, FSMBuilder)
        assert "greeting" in builder.states

    def test_add_transition(self, handlers_with_fsm_builder: MetaHandlers):
        builder = handlers_with_fsm_builder.builder
        assert isinstance(builder, FSMBuilder)
        builder.add_state("a", "State A", "Purpose A")
        builder.add_state("b", "State B", "Purpose B")

        context = {
            ContextKeys.ACTION: Actions.ADD_TRANSITION,
            ContextKeys.ACTION_PARAMS: {
                "from_state": "a",
                "target_state": "b",
                "description": "Go to B",
            },
        }
        result = handlers_with_fsm_builder.dispatch_action(context)
        assert ContextKeys.ACTION_RESULT in result

    def test_done_action(self, handlers_with_fsm_builder: MetaHandlers):
        context = {ContextKeys.ACTION: Actions.DONE}
        result = handlers_with_fsm_builder.dispatch_action(context)
        assert result.get(ContextKeys.STRUCTURE_DONE) is True

    def test_builder_error_returns_error_message(
        self, handlers_with_fsm_builder: MetaHandlers
    ):
        context = {
            ContextKeys.ACTION: Actions.ADD_STATE,
            ContextKeys.ACTION_PARAMS: {
                "state_id": "",
                "description": "x",
                "purpose": "y",
            },
        }
        result = handlers_with_fsm_builder.dispatch_action(context)
        assert ContextKeys.ACTION_ERRORS in result

    def test_clears_action_after_dispatch(
        self, handlers_with_fsm_builder: MetaHandlers
    ):
        context = {
            ContextKeys.ACTION: Actions.ADD_STATE,
            ContextKeys.ACTION_PARAMS: {
                "state_id": "s1",
                "description": "S1",
                "purpose": "P",
            },
        }
        result = handlers_with_fsm_builder.dispatch_action(context)
        assert result.get(ContextKeys.ACTION) is None
        assert result.get(ContextKeys.ACTION_PARAMS) is None


class TestFinalize:
    def test_finalize_valid(self, handlers_with_fsm_builder: MetaHandlers):
        builder = handlers_with_fsm_builder.builder
        assert isinstance(builder, FSMBuilder)
        builder.set_overview("Bot", "Desc")
        builder.add_state("s1", "State", "Purpose")

        result = handlers_with_fsm_builder.finalize({})
        assert ContextKeys.FINAL_ARTIFACT in result
        assert ContextKeys.FINAL_JSON in result
        assert '"Bot"' in result[ContextKeys.FINAL_JSON]

    def test_finalize_invalid(self, handlers_with_fsm_builder: MetaHandlers):
        # Empty builder → validation errors
        result = handlers_with_fsm_builder.finalize({})
        assert ContextKeys.VALIDATION_ERRORS in result

    def test_finalize_no_builder(self, handlers: MetaHandlers):
        result = handlers.finalize({})
        assert ContextKeys.VALIDATION_ERRORS in result


class TestRunValidation:
    def test_run_validation_empty(self, handlers_with_fsm_builder: MetaHandlers):
        result = handlers_with_fsm_builder.run_validation({})
        assert result.get(ContextKeys.VALIDATION_ERRORS) is not None

    def test_run_validation_valid(self, handlers_with_fsm_builder: MetaHandlers):
        builder = handlers_with_fsm_builder.builder
        assert isinstance(builder, FSMBuilder)
        builder.set_overview("Bot", "A bot")
        builder.add_state("s1", "State 1", "Purpose")

        result = handlers_with_fsm_builder.run_validation({})
        # s1 is terminal and initial so should be valid
        assert result.get(ContextKeys.VALIDATION_ERRORS) is None
