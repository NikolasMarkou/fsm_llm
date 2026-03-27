from __future__ import annotations

"""Tests for cross-module integration methods."""

import pytest

from fsm_llm_agents.tools import ToolRegistry


def _noop(params):
    """No-op tool function."""
    return "ok"


class TestToClassificationSchema:
    """Tests for ToolRegistry.to_classification_schema() producing valid schema."""

    def test_schema_has_intents(self):
        """Generated schema should have intents list."""
        registry = ToolRegistry()
        registry.register_function(_noop, name="search", description="Search the web")
        registry.register_function(_noop, name="calculate", description="Do math")

        schema = registry.to_classification_schema()
        assert "intents" in schema
        assert isinstance(schema["intents"], list)
        assert len(schema["intents"]) >= 3  # 2 tools + none fallback

    def test_schema_has_fallback(self):
        """Generated schema should have a 'none' fallback intent."""
        registry = ToolRegistry()
        registry.register_function(_noop, name="search", description="Search")

        schema = registry.to_classification_schema()
        assert schema["fallback_intent"] == "none"
        intent_names = [i["name"] for i in schema["intents"]]
        assert "none" in intent_names

    def test_schema_has_confidence_threshold(self):
        """Generated schema should have a confidence threshold."""
        registry = ToolRegistry()
        registry.register_function(_noop, name="search", description="Search")

        schema = registry.to_classification_schema()
        assert "confidence_threshold" in schema
        assert 0.0 <= schema["confidence_threshold"] <= 1.0

    def test_schema_accepted_by_classification_schema(self):
        """Generated dict should be accepted by ClassificationSchema."""
        classification = pytest.importorskip("fsm_llm_classification")

        registry = ToolRegistry()
        registry.register_function(_noop, name="search", description="Search")
        registry.register_function(_noop, name="calculate", description="Calculate")

        schema_dict = registry.to_classification_schema()
        schema = classification.ClassificationSchema(
            intents=[
                classification.IntentDefinition(**i) for i in schema_dict["intents"]
            ],
            fallback_intent=schema_dict["fallback_intent"],
            confidence_threshold=schema_dict["confidence_threshold"],
        )

        assert len(schema.intents) >= 3
        assert schema.fallback_intent == "none"
        assert "search" in schema.intent_names
        assert "calculate" in schema.intent_names

    def test_schema_no_duplicate_none(self):
        """Registering a tool named 'none' is now blocked (reserved name)."""
        registry = ToolRegistry()
        registry.register_function(_noop, name="search", description="Search")

        import pytest

        with pytest.raises(ValueError, match="reserved"):
            registry.register_function(_noop, name="none", description="No tool")


class TestBuildReactFsmForWorkflow:
    """Tests for using build_react_fsm() output with workflow steps."""

    def test_react_fsm_is_valid_dict(self):
        """build_react_fsm() should produce a valid FSM definition dict."""
        from fsm_llm_agents.fsm_definitions import build_react_fsm

        registry = ToolRegistry()
        registry.register_function(_noop, name="search", description="Search")

        fsm_def = build_react_fsm(registry, task_description="test task")

        assert isinstance(fsm_def, dict)
        assert "name" in fsm_def
        assert "initial_state" in fsm_def
        assert "states" in fsm_def
        assert "think" in fsm_def["states"]
        assert "act" in fsm_def["states"]
        assert "conclude" in fsm_def["states"]

    def test_react_fsm_has_persona(self):
        """Generated FSM should have a persona."""
        from fsm_llm_agents.fsm_definitions import build_react_fsm

        registry = ToolRegistry()
        registry.register_function(_noop, name="search", description="Search")

        fsm_def = build_react_fsm(registry, task_description="test task")
        assert "persona" in fsm_def
        assert isinstance(fsm_def["persona"], str)

    def test_self_consistency_fsm_is_valid_dict(self):
        """build_self_consistency_fsm() should produce a valid FSM definition."""
        from fsm_llm_agents.fsm_definitions import build_self_consistency_fsm

        fsm_def = build_self_consistency_fsm(task_description="test task")

        assert isinstance(fsm_def, dict)
        assert "name" in fsm_def
        assert "initial_state" in fsm_def
        assert "states" in fsm_def

    def test_react_fsm_accepted_by_fsm_definition(self):
        """Generated FSM dict should be accepted by FSMDefinition validator."""
        from fsm_llm.definitions import FSMDefinition
        from fsm_llm_agents.fsm_definitions import build_react_fsm

        registry = ToolRegistry()
        registry.register_function(_noop, name="search", description="Search")

        fsm_dict = build_react_fsm(registry, task_description="test")
        fsm_def = FSMDefinition(**fsm_dict)

        assert fsm_def.initial_state == "think"
        assert "think" in fsm_def.states
        assert "conclude" in fsm_def.states

    def test_react_fsm_accepted_by_conversation_step(self):
        """Generated FSM dict should be accepted as ConversationStep input."""
        workflows = pytest.importorskip("fsm_llm_workflows")

        from fsm_llm_agents.fsm_definitions import build_react_fsm

        registry = ToolRegistry()
        registry.register_function(_noop, name="search", description="Search")

        fsm_def = build_react_fsm(registry, task_description="test")

        step = workflows.conversation_step(
            step_id="agent",
            name="Agent Step",
            fsm_definition=fsm_def,
            success_state="done",
            auto_messages=["Continue."],
            max_turns=5,
        )

        assert step.step_id == "agent"
        assert step.fsm_definition == fsm_def


class TestReasoningIntegrationKeysUniqueness:
    """Tests that ReasoningIntegrationKeys values are globally unique."""

    def test_all_values_unique(self):
        """All key values in ReasoningIntegrationKeys should be unique."""
        from fsm_llm_agents.constants import ReasoningIntegrationKeys

        values = [
            v
            for k, v in vars(ReasoningIntegrationKeys).items()
            if not k.startswith("_") and isinstance(v, str)
        ]
        assert len(values) == len(set(values))

    def test_no_overlap_with_all_agent_constants(self):
        """ReasoningIntegrationKeys should not overlap with any agent ContextKeys."""
        from fsm_llm_agents.constants import ContextKeys, ReasoningIntegrationKeys

        r_values = {
            v
            for k, v in vars(ReasoningIntegrationKeys).items()
            if not k.startswith("_")
            and isinstance(v, str)
            # Exclude REASONING_TOOL_NAME since "reason" is a tool name, not a context key
            and k != "REASONING_TOOL_NAME"
        }
        c_values = {
            v
            for k, v in vars(ContextKeys).items()
            if not k.startswith("_") and isinstance(v, str)
        }
        overlap = r_values & c_values
        assert not overlap, f"Overlapping keys: {overlap}"
