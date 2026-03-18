"""
Functional tests for fsm_llm_2 core modules using mock LLM (2-pass architecture).

Tests the full conversation lifecycle: API, FSMManager, TransitionEvaluator,
data extraction, response generation, and FSM stacking.
"""
import pytest
from unittest.mock import MagicMock

from fsm_llm_2.api import API
from fsm_llm_2.fsm import FSMManager
from fsm_llm_2.definitions import (
    FSMDefinition,
    DataExtractionResponse,
    ResponseGenerationResponse,
    TransitionDecisionResponse,
    TransitionEvaluationResult,
)
from fsm_llm_2.transition_evaluator import TransitionEvaluator, TransitionEvaluatorConfig


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def greeting_fsm_dict():
    """A 3-state FSM: greeting -> collect_name -> farewell."""
    return {
        "name": "test_greeting",
        "description": "Test greeting FSM",
        "version": "4.1",
        "initial_state": "greeting",
        "states": {
            "greeting": {
                "id": "greeting",
                "description": "Initial greeting",
                "purpose": "Greet the user",
                "transitions": [
                    {
                        "target_state": "collect_name",
                        "description": "Move to collect name",
                        "priority": 100,
                    }
                ],
            },
            "collect_name": {
                "id": "collect_name",
                "description": "Collect user name",
                "purpose": "Ask for and collect user name",
                "transitions": [
                    {
                        "target_state": "farewell",
                        "description": "Name collected, say goodbye",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "User name collected",
                                "requires_context_keys": ["user_name"],
                            }
                        ],
                    }
                ],
            },
            "farewell": {
                "id": "farewell",
                "description": "Farewell state",
                "purpose": "Say goodbye",
                "transitions": [],
            },
        },
    }


@pytest.fixture
def sub_fsm_dict():
    """A simple sub-FSM for stacking tests."""
    return {
        "name": "address_collector",
        "description": "Collect address",
        "version": "4.1",
        "initial_state": "ask_address",
        "states": {
            "ask_address": {
                "id": "ask_address",
                "description": "Ask for address",
                "purpose": "Collect address from user",
                "transitions": [
                    {
                        "target_state": "address_done",
                        "description": "Address collected",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "Address provided",
                                "requires_context_keys": ["address"],
                            }
                        ],
                    }
                ],
            },
            "address_done": {
                "id": "address_done",
                "description": "Address collected",
                "purpose": "Confirm address",
                "transitions": [],
            },
        },
    }


# ── API Lifecycle Tests ─────────────────────────────────────────


class TestAPILifecycle:
    """Test full conversation lifecycle via API."""

    def test_start_conversation(self, greeting_fsm_dict, mock_llm2_interface):
        """API.start_conversation returns conversation_id and initial response."""
        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        )
        conv_id, response = api.start_conversation()

        assert conv_id is not None
        assert isinstance(response, str)
        assert len(response) > 0
        # generate_response should have been called for initial message
        assert any(call[0] == "generate_response" for call in mock_llm2_interface.call_history)

    def test_converse_calls_extract_and_generate(self, greeting_fsm_dict, mock_llm2_interface):
        """converse() triggers data extraction then response generation."""
        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        )
        conv_id, _ = api.start_conversation()
        mock_llm2_interface.call_history.clear()

        response = api.converse("Hello!", conv_id)

        call_types = [call[0] for call in mock_llm2_interface.call_history]
        assert "extract_data" in call_types
        assert "generate_response" in call_types
        assert isinstance(response, str)

    def test_get_current_state(self, greeting_fsm_dict, mock_llm2_interface):
        """get_current_state returns the initial state."""
        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        )
        conv_id, _ = api.start_conversation()

        state = api.get_current_state(conv_id)
        assert state == "greeting"

    def test_end_conversation_cleanup(self, greeting_fsm_dict, mock_llm2_interface):
        """end_conversation cleans up all tracking structures."""
        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        )
        conv_id, _ = api.start_conversation()
        api.end_conversation(conv_id)

        assert conv_id not in api.active_conversations
        assert conv_id not in api.conversation_stacks

    def test_context_manager(self, greeting_fsm_dict, mock_llm2_interface):
        """API works as a context manager and cleans up on exit."""
        with API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        ) as api:
            conv_id, _ = api.start_conversation()
            assert conv_id in api.active_conversations

        # After exit, conversations should be cleaned up
        assert len(api.active_conversations) == 0

    def test_has_conversation_ended_initial_state(self, greeting_fsm_dict, mock_llm2_interface):
        """has_conversation_ended returns False for non-terminal state."""
        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        )
        conv_id, _ = api.start_conversation()

        assert api.has_conversation_ended(conv_id) is False


# ── Transition Evaluation Tests ──────────────────────────────────


class TestTransitionEvaluation:
    """Test transition evaluation with real evaluator."""

    def test_deterministic_transition_with_context(self, greeting_fsm_dict, mock_llm2_interface):
        """Transition fires deterministically when conditions are met."""
        # Set the mock to extract user_name
        mock_llm2_interface.extraction_data = {"user_name": "Alice"}

        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        )
        conv_id, _ = api.start_conversation()

        # First message — in greeting state, unconditional transition to collect_name
        api.converse("Hi there!", conv_id)
        state_after_first = api.get_current_state(conv_id)

        # Second message — now in collect_name, with user_name extracted, should transition to farewell
        api.converse("My name is Alice", conv_id)
        state_after_second = api.get_current_state(conv_id)

        # Should have transitioned through greeting -> collect_name -> farewell
        assert state_after_first == "collect_name"
        assert state_after_second == "farewell"

    def test_blocked_transition_stays_in_state(self, greeting_fsm_dict, mock_llm2_interface):
        """When conditions aren't met, FSM stays in current state."""
        # No extraction data — conditions for farewell transition won't be met
        mock_llm2_interface.extraction_data = {}

        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        )
        conv_id, _ = api.start_conversation()

        # First message transitions greeting -> collect_name (unconditional)
        api.converse("Hi", conv_id)
        assert api.get_current_state(conv_id) == "collect_name"

        # Second message — no user_name extracted, should stay in collect_name
        api.converse("I don't want to give my name", conv_id)
        assert api.get_current_state(conv_id) == "collect_name"

    def test_terminal_state_no_processing(self, greeting_fsm_dict, mock_llm2_interface):
        """Terminal state (farewell) has no transitions; conversation should end."""
        mock_llm2_interface.extraction_data = {"user_name": "Bob"}

        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        )
        conv_id, _ = api.start_conversation()

        # Drive to farewell
        api.converse("Hi", conv_id)
        api.converse("I'm Bob", conv_id)

        assert api.get_current_state(conv_id) == "farewell"
        assert api.has_conversation_ended(conv_id) is True


# ── Data Collection Tests ────────────────────────────────────────


class TestDataCollection:
    """Test context/data collection through conversation flow."""

    def test_extracted_data_appears_in_context(self, greeting_fsm_dict, mock_llm2_interface):
        """Data extracted by LLM appears in conversation context."""
        mock_llm2_interface.extraction_data = {"user_name": "Charlie"}

        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        )
        conv_id, _ = api.start_conversation()
        api.converse("My name is Charlie", conv_id)

        data = api.get_data(conv_id)
        assert data["user_name"] == "Charlie"

    def test_initial_context_preserved(self, greeting_fsm_dict, mock_llm2_interface):
        """Initial context passed to start_conversation is preserved."""
        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        )
        conv_id, _ = api.start_conversation(initial_context={"source": "web"})

        data = api.get_data(conv_id)
        assert data["source"] == "web"


# ── FSM Stacking Tests ──────────────────────────────────────────


class TestFSMStacking:
    """Test FSM push/pop stacking operations."""

    def test_push_fsm(self, greeting_fsm_dict, sub_fsm_dict, mock_llm2_interface):
        """Pushing a sub-FSM starts a new conversation on top of the stack."""
        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        )
        conv_id, _ = api.start_conversation()

        response = api.push_fsm(conv_id, sub_fsm_dict)

        assert isinstance(response, str)
        assert len(api.conversation_stacks[conv_id]) == 2
        # Current state should be from the sub-FSM
        assert api.get_current_state(conv_id) == "ask_address"

    def test_pop_fsm(self, greeting_fsm_dict, sub_fsm_dict, mock_llm2_interface):
        """Popping returns to the previous FSM."""
        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        )
        conv_id, _ = api.start_conversation()
        api.push_fsm(conv_id, sub_fsm_dict)

        response = api.pop_fsm(conv_id)

        assert isinstance(response, str)
        assert len(api.conversation_stacks[conv_id]) == 1
        assert api.get_current_state(conv_id) == "greeting"

    def test_pop_last_fsm_raises(self, greeting_fsm_dict, mock_llm2_interface):
        """Cannot pop the last FSM from the stack."""
        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        )
        conv_id, _ = api.start_conversation()

        with pytest.raises(ValueError, match="only one FSM remaining"):
            api.pop_fsm(conv_id)

    def test_context_inheritance(self, greeting_fsm_dict, sub_fsm_dict, mock_llm2_interface):
        """Pushing FSM inherits parent context."""
        mock_llm2_interface.extraction_data = {"user_name": "Diana"}

        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        )
        conv_id, _ = api.start_conversation(initial_context={"user_name": "Diana"})
        api.push_fsm(conv_id, sub_fsm_dict, shared_context_keys=["user_name"])

        # Sub-FSM should have inherited user_name
        data = api.get_data(conv_id)
        assert data.get("user_name") == "Diana"


# ── FSM Definition Validation Tests ──────────────────────────────


class TestFSMDefinitionValidation:
    """Test FSM definition validation."""

    def test_valid_definition_accepted(self, greeting_fsm_dict):
        """Valid FSM definition is accepted without errors."""
        fsm_def = FSMDefinition.model_validate(greeting_fsm_dict)
        assert fsm_def.name == "test_greeting"
        assert fsm_def.initial_state == "greeting"

    def test_missing_initial_state_rejected(self):
        """FSM with non-existent initial state is rejected."""
        bad_fsm = {
            "name": "bad",
            "description": "Bad FSM",
            "initial_state": "nonexistent",
            "states": {
                "greeting": {
                    "id": "greeting",
                    "description": "State",
                    "purpose": "Purpose",
                    "transitions": [],
                }
            },
        }
        with pytest.raises(ValueError, match="Initial state.*not found"):
            FSMDefinition.model_validate(bad_fsm)

    def test_invalid_transition_target_rejected(self):
        """FSM with transition to non-existent state is rejected."""
        bad_fsm = {
            "name": "bad",
            "description": "Bad FSM",
            "initial_state": "start",
            "states": {
                "start": {
                    "id": "start",
                    "description": "Start",
                    "purpose": "Start",
                    "transitions": [
                        {
                            "target_state": "nonexistent",
                            "description": "Bad transition",
                            "priority": 100,
                        }
                    ],
                }
            },
        }
        with pytest.raises(ValueError, match="non-existent state"):
            FSMDefinition.model_validate(bad_fsm)

    def test_no_terminal_state_rejected(self):
        """FSM with no terminal state is rejected."""
        bad_fsm = {
            "name": "bad",
            "description": "Looping FSM",
            "initial_state": "a",
            "states": {
                "a": {
                    "id": "a",
                    "description": "State A",
                    "purpose": "Loop",
                    "transitions": [
                        {"target_state": "a", "description": "Self loop", "priority": 100}
                    ],
                }
            },
        }
        with pytest.raises(ValueError, match="terminal state"):
            FSMDefinition.model_validate(bad_fsm)


# ── TransitionEvaluator Unit Tests ───────────────────────────────


class TestTransitionEvaluatorUnit:
    """Test TransitionEvaluator in isolation."""

    def test_no_transitions_returns_blocked(self, sample_fsm_definition_v2):
        """State with no transitions returns BLOCKED."""
        from fsm_llm_2.definitions import State, FSMContext

        farewell_state = sample_fsm_definition_v2.states["farewell"]
        evaluator = TransitionEvaluator()
        context = FSMContext()

        result = evaluator.evaluate_transitions(farewell_state, context)
        assert result.result_type == TransitionEvaluationResult.BLOCKED

    def test_condition_met_returns_deterministic(self, sample_fsm_definition_v2):
        """When conditions are met, returns DETERMINISTIC."""
        from fsm_llm_2.definitions import FSMContext

        greeting_state = sample_fsm_definition_v2.states["greeting"]
        evaluator = TransitionEvaluator()
        context = FSMContext()
        context.update({"user_name": "Eve"})

        result = evaluator.evaluate_transitions(greeting_state, context)
        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC
        assert result.deterministic_transition == "farewell"

    def test_condition_not_met_returns_blocked(self, sample_fsm_definition_v2):
        """When conditions aren't met, returns BLOCKED."""
        from fsm_llm_2.definitions import FSMContext

        greeting_state = sample_fsm_definition_v2.states["greeting"]
        evaluator = TransitionEvaluator()
        context = FSMContext()  # No user_name in context

        result = evaluator.evaluate_transitions(greeting_state, context)
        assert result.result_type == TransitionEvaluationResult.BLOCKED


# ── Conversation History Tests ───────────────────────────────────


class TestConversationHistory:
    """Test conversation history management."""

    def test_history_tracks_messages(self, greeting_fsm_dict, mock_llm2_interface):
        """Conversation history records user and system messages."""
        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
        )
        conv_id, _ = api.start_conversation()
        api.converse("Hello!", conv_id)

        history = api.get_conversation_history(conv_id)
        assert len(history) >= 2  # At least initial response + user msg + response

    def test_history_respects_max_size(self, greeting_fsm_dict, mock_llm2_interface):
        """History is capped at max_history_size exchanges."""
        api = API(
            fsm_definition=greeting_fsm_dict,
            llm_interface=mock_llm2_interface,
            max_history_size=2,
        )
        conv_id, _ = api.start_conversation()

        for i in range(10):
            api.converse(f"Message {i}", conv_id)

        history = api.get_conversation_history(conv_id)
        # max_history_size=2 means 2 exchanges = 4 messages max
        assert len(history) <= 4
