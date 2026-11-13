from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


def configure_mock_extract_field(mock_llm, mock_data=None):
    """Configure a mock LLM with extract_field support."""
    from fsm_llm.core.definitions import FieldExtractionResponse

    data = mock_data or {}

    def _side_effect(request):
        value = data.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="Mock field extraction",
            is_valid=value is not None,
        )

    mock_llm.extract_field.side_effect = _side_effect
    return mock_llm


import threading
from typing import Any

from fsm_llm.api import API
from fsm_llm.context.layout import (
    DataExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from fsm_llm.core.definitions import (
    ClassificationResult,
    DataExtractionResponse,
    FSMContext,
    FSMDefinition,
    FSMInstance,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
    State,
    Transition,
    TransitionCondition,
)
from fsm_llm.core.loaders import load_fsm_from_file
from fsm_llm.core.transition_evaluator import TransitionEvaluator
from fsm_llm.fsm import FSMManager
from fsm_llm.llm import LLMInterface
from fsm_llm.llm.json_extraction import extract_json_from_text
from fsm_llm.validator import FSMValidator
from fsm_llm.visualizer import visualize_fsm_ascii
from tests.test_fsm_llm.support import (
    _JOIN_TIMEOUT_S,
    _a2026_09_22_api,
    _blocked_fsm,
    _join_all,
    _mock_llm,
)

# Test fixtures


@pytest.fixture
def valid_fsm_data():
    """Fixture for a valid FSM definition."""
    return {
        "name": "Test FSM",
        "description": "A test FSM for unit testing",
        "version": "4.0",
        "initial_state": "welcome",
        "persona": "A friendly AI assistant",
        "states": {
            "welcome": {
                "id": "welcome",
                "description": "Welcome state",
                "purpose": "Welcome the user",
                "response_instructions": "Greet the user warmly",
                "transitions": [
                    {
                        "target_state": "collect_name",
                        "description": "Transition to name collection",
                        "priority": 1,
                    }
                ],
            },
            "collect_name": {
                "id": "collect_name",
                "description": "Name collection state",
                "purpose": "Collect user's name",
                "extraction_instructions": "Extract the user's name from their input",
                "required_context_keys": ["name"],
                "transitions": [
                    {
                        "target_state": "farewell",
                        "description": "Transition to farewell",
                        "conditions": [
                            {
                                "description": "Name has been provided",
                                "requires_context_keys": ["name"],
                            }
                        ],
                        "priority": 1,
                    }
                ],
            },
            "farewell": {
                "id": "farewell",
                "description": "Farewell state",
                "purpose": "Say goodbye to the user",
                "response_instructions": "Say goodbye using the user's name",
                "transitions": [],  # Terminal state
            },
        },
    }


@pytest.fixture
def invalid_fsm_data_missing_initial():
    """Fixture for an invalid FSM definition with missing initial state."""
    return {
        "name": "Invalid FSM",
        "description": "An invalid FSM for unit testing",
        "version": "4.0",
        "initial_state": "non_existent_state",  # This state doesn't exist
        "states": {
            "welcome": {
                "id": "welcome",
                "description": "Welcome state",
                "purpose": "Welcome the user",
                "transitions": [],
            }
        },
    }


@pytest.fixture
def invalid_fsm_data_orphaned_state():
    """Fixture for an invalid FSM definition with an orphaned state."""
    return {
        "name": "Invalid FSM",
        "description": "An invalid FSM for unit testing",
        "version": "4.0",
        "initial_state": "welcome",
        "states": {
            "welcome": {
                "id": "welcome",
                "description": "Welcome state",
                "purpose": "Welcome the user",
                "transitions": [],  # No transitions to other states
            },
            "orphaned": {
                "id": "orphaned",
                "description": "Orphaned state",
                "purpose": "This state is orphaned",
                "transitions": [],
            },
        },
    }


@pytest.fixture
def mock_llm_interface():
    """Fixture for a mocked LLM interface for the 2-pass architecture."""
    mock_interface = MagicMock(spec=LLMInterface)

    # Mock response generation response
    mock_response_generation = ResponseGenerationResponse(
        message="Hello John! Nice to meet you.",
        reasoning="Generated greeting using extracted name",
    )

    # Set up the mock methods
    mock_interface.generate_response.return_value = mock_response_generation

    configure_mock_extract_field(mock_interface, {"name": "John"})

    return mock_interface


# Actual tests


def test_load_valid_fsm_definition(valid_fsm_data, tmp_path):
    """Test loading a valid FSM definition from a file."""
    # Write the valid FSM data to a temporary file
    fsm_file = tmp_path / "valid_fsm.json"
    with open(fsm_file, "w", encoding="utf-8") as f:
        json.dump(valid_fsm_data, f)

    # Load the FSM definition
    fsm_def = load_fsm_from_file(str(fsm_file))

    # Verify the loaded FSM definition
    assert fsm_def.name == "Test FSM"
    assert fsm_def.initial_state == "welcome"
    assert len(fsm_def.states) == 3
    assert "welcome" in fsm_def.states
    assert "collect_name" in fsm_def.states
    assert "farewell" in fsm_def.states


def test_fsm_definition_validation_valid(valid_fsm_data):
    """Test validation of a valid FSM definition."""
    # Create a validator for the valid FSM data
    validator = FSMValidator(valid_fsm_data)

    # Run validation
    result = validator.validate()

    # Verify validation results
    assert result.is_valid
    assert len(result.errors) == 0


def test_fsm_definition_validation_invalid_initial(invalid_fsm_data_missing_initial):
    """Test validation of an invalid FSM definition with missing initial state."""
    # Create a validator for the invalid FSM data
    validator = FSMValidator(invalid_fsm_data_missing_initial)

    # Run validation
    result = validator.validate()

    # Verify validation results
    assert not result.is_valid
    assert len(result.errors) > 0
    assert any("Initial state" in error for error in result.errors)


def test_fsm_definition_validation_orphaned_state(invalid_fsm_data_orphaned_state):
    """Test validation of an invalid FSM definition with orphaned states."""
    # Create a validator for the invalid FSM data
    validator = FSMValidator(invalid_fsm_data_orphaned_state)

    # Run validation
    result = validator.validate()

    # Verify validation results
    assert not result.is_valid
    assert len(result.errors) > 0
    assert any("Orphaned states" in error for error in result.errors)


def test_fsm_context_update():
    """Test that context updates work correctly."""
    # Create a context
    context = FSMContext()

    # Update the context
    context.update({"name": "John", "age": 30})

    # Verify the context data
    assert context.data["name"] == "John"
    assert context.data["age"] == 30

    # Update with additional data
    context.update({"email": "john@example.com"})

    # Verify the context data again
    assert context.data["name"] == "John"  # Original data preserved
    assert context.data["email"] == "john@example.com"  # New data added


def test_conversation_context_missing_keys():
    """Test context missing keys functionality."""
    context = FSMContext()
    context.update({"name": "John", "email": "john@example.com"})

    # Test key presence via data dict
    assert "name" in context.data
    assert "email" in context.data
    assert "age" not in context.data

    # Test missing keys detection via set difference
    required = {"name", "age", "phone"}
    missing = required - set(context.data.keys())
    assert "age" in missing
    assert "phone" in missing
    assert "name" not in missing


def test_transition_evaluator():
    """Test the transition evaluator with deterministic transitions."""
    from fsm_llm.core.transition_evaluator import (
        TransitionEvaluator,
        TransitionEvaluatorConfig,
    )

    # Create a transition evaluator
    config = TransitionEvaluatorConfig(strict_condition_matching=True)
    evaluator = TransitionEvaluator(config)

    # Create a simple state with transitions
    state = State(
        id="test_state",
        description="Test state",
        purpose="Test state purpose",
        transitions=[
            Transition(
                target_state="state2",
                description="Go to state 2",
                priority=1,
                conditions=[
                    TransitionCondition(
                        description="Name is provided", requires_context_keys=["name"]
                    )
                ],
            )
        ],
    )

    # Create context with required data
    context = FSMContext()
    context.update({"name": "John"})

    # Evaluate transitions
    evaluation = evaluator.evaluate_transitions(state, context)

    # Should be deterministic since we have the required context
    assert evaluation.result_type.value == "deterministic"
    assert evaluation.deterministic_transition == "state2"


def test_fsm_manager_initialization(valid_fsm_data, mock_llm_interface):
    """Test FSM manager initialization and basic operations."""

    # Mock the FSM loader to return our valid FSM definition
    def mock_loader(fsm_id):
        return FSMDefinition(**valid_fsm_data)

    # Create prompt builders
    data_extraction_builder = DataExtractionPromptBuilder()
    response_generation_builder = ResponseGenerationPromptBuilder()
    transition_evaluator = TransitionEvaluator()

    # Create an FSM manager with the mock loader and interface
    fsm_manager = FSMManager(
        fsm_loader=mock_loader,
        llm_interface=mock_llm_interface,
        data_extraction_prompt_builder=data_extraction_builder,
        response_generation_prompt_builder=response_generation_builder,
        transition_evaluator=transition_evaluator,
    )

    # Test the FSM manager initialization
    assert fsm_manager.fsm_loader is not None
    assert fsm_manager.llm_interface is not None
    assert fsm_manager.data_extraction_prompt_builder is not None
    assert fsm_manager.response_generation_prompt_builder is not None

    # Test FSM definition retrieval
    fsm_def = fsm_manager.get_fsm_definition("test_fsm")
    assert fsm_def.name == "Test FSM"
    assert fsm_def.initial_state == "welcome"


def test_prompt_builders():
    """Test the prompt builders for the 2-pass architecture."""
    # Create farewell state (terminal state)
    farewell_state = State(
        id="farewell",
        description="Farewell state",
        purpose="Say goodbye to the user",
        response_instructions="Say goodbye using the user's name",
        transitions=[],  # Terminal state
    )

    # Create collect_name state
    collect_name_state = State(
        id="collect_name",
        description="Collect user's name",
        purpose="Ask for and record the user's name",
        extraction_instructions="Extract the user's name from their input",
        response_instructions="Ask for the user's name if not provided",
        required_context_keys=["name"],
        transitions=[
            Transition(
                target_state="farewell",
                description="Transition to farewell",
                conditions=[
                    TransitionCondition(
                        description="Name has been provided",
                        requires_context_keys=["name"],
                    )
                ],
            )
        ],
    )

    fsm_def = FSMDefinition(
        name="Test FSM",
        description="Test FSM",
        initial_state="collect_name",
        persona="A friendly assistant",
        states={"collect_name": collect_name_state, "farewell": farewell_state},
    )

    instance = FSMInstance(
        fsm_id="test_fsm", current_state="collect_name", persona="A friendly assistant"
    )

    # Test response generation prompt builder
    response_builder = ResponseGenerationPromptBuilder()
    response_prompt = response_builder.build_response_prompt(
        instance=instance,
        state=collect_name_state,
        fsm_definition=fsm_def,
        extracted_data={"name": "John"},
        transition_occurred=False,
        previous_state=None,
        user_message="Hello",
    )

    # Verify the response prompt contains key elements
    assert "<response_generation>" in response_prompt
    assert "</response_generation>" in response_prompt
    assert "persona" in response_prompt.lower()


def test_extract_json_from_text():
    """Test the utility function for extracting JSON from text."""
    # Test with JSON in code block
    text1 = 'Here is the JSON:\n```json\n{"name": "John", "age": 30}\n```'
    json1 = extract_json_from_text(text1)
    assert json1 is not None
    assert json1["name"] == "John"
    assert json1["age"] == 30

    # Test with JSON without code block
    text2 = 'Here is the JSON: {"name": "Jane", "age": 25}'
    json2 = extract_json_from_text(text2)
    assert json2 is not None
    assert json2["name"] == "Jane"
    assert json2["age"] == 25

    # Test with invalid JSON
    text3 = "This is not JSON"
    json3 = extract_json_from_text(text3)
    assert json3 is None


def test_visualize_fsm_ascii(valid_fsm_data):
    """Test the FSM visualization function."""
    # Generate ASCII visualization
    ascii_art = visualize_fsm_ascii(valid_fsm_data)

    # Verify the visualization contains key elements
    assert "Test FSM" in ascii_art
    assert "welcome" in ascii_art
    assert "collect_name" in ascii_art
    assert "farewell" in ascii_art
    assert "INITIAL" in ascii_art
    assert "TERMINAL" in ascii_art


def test_conversation_flow_2pass_architecture(mock_llm_interface, valid_fsm_data):
    """Test the complete conversation flow with 2-pass architecture."""

    # Mock the FSM loader
    def mock_loader(fsm_id):
        return FSMDefinition(**valid_fsm_data)

    # Create FSM manager with all required components
    fsm_manager = FSMManager(
        fsm_loader=mock_loader,
        llm_interface=mock_llm_interface,
        data_extraction_prompt_builder=DataExtractionPromptBuilder(),
        response_generation_prompt_builder=ResponseGenerationPromptBuilder(),
        transition_evaluator=TransitionEvaluator(),
    )

    # Start a conversation
    conversation_id, _initial_response = fsm_manager.start_conversation("test_fsm")

    # Verify initial state
    assert fsm_manager.get_conversation_state(conversation_id) == "welcome"
    assert not fsm_manager.has_conversation_ended(conversation_id)

    # Process a user message
    fsm_manager.process_message(conversation_id, "Hello, my name is John")

    # Verify response generation was called
    assert mock_llm_interface.generate_response.called

    # Send second message in collect_name state (which has required_context_keys)
    fsm_manager.process_message(conversation_id, "My name is John")

    # Now extract_field should have been called for the "name" key
    assert mock_llm_interface.extract_field.called

    # Verify conversation data
    conversation_data = fsm_manager.get_conversation_data(conversation_id)
    assert "name" in conversation_data
    assert conversation_data["name"] == "John"


def test_conversation_ended_detection(mock_llm_interface, valid_fsm_data):
    """Test detection of conversation end state."""

    # Mock the FSM loader
    def mock_loader(fsm_id):
        return FSMDefinition(**valid_fsm_data)

    # Create an FSM manager
    fsm_manager = FSMManager(
        fsm_loader=mock_loader,
        llm_interface=mock_llm_interface,
        data_extraction_prompt_builder=DataExtractionPromptBuilder(),
        response_generation_prompt_builder=ResponseGenerationPromptBuilder(),
        transition_evaluator=TransitionEvaluator(),
    )

    # Start a conversation
    conversation_id, _ = fsm_manager.start_conversation("test_fsm")

    # Initially, conversation should not be ended (welcome state has transitions)
    assert not fsm_manager.has_conversation_ended(conversation_id)

    # Manually change the state to a terminal state (farewell has no transitions)
    fsm_manager.instances[conversation_id].current_state = "farewell"

    # Now the conversation should be detected as ended
    assert fsm_manager.has_conversation_ended(conversation_id)


def test_data_extraction_response_model():
    """Test the data extraction response model."""
    response = DataExtractionResponse(
        extracted_data={"name": "John Smith"},
        confidence=0.95,
        reasoning="User clearly stated their name",
    )

    assert response.extracted_data["name"] == "John Smith"
    assert response.confidence == 0.95
    assert response.reasoning == "User clearly stated their name"


def test_response_generation_request_response_models():
    """Test the response generation request and response models."""
    # Create a response generation request
    request = ResponseGenerationRequest(
        system_prompt="Generate a friendly greeting using the user's name",
        user_message="Hello",
        transition_occurred=False,
    )

    assert request.system_prompt == "Generate a friendly greeting using the user's name"
    assert request.user_message == "Hello"

    # Create a response generation response
    response = ResponseGenerationResponse(
        message="Hello John! Welcome to our service.",
        reasoning="Generated personalized greeting using extracted name",
    )

    assert response.message == "Hello John! Welcome to our service."
    assert response.reasoning == "Generated personalized greeting using extracted name"


def test_classification_result_models():
    """Test the ClassificationResult model (replaces TransitionDecisionRequest/Response)."""
    # Create a classification result
    result = ClassificationResult(
        reasoning="User expressed intent to provide their name",
        intent="collect_name",
        confidence=0.9,
        entities={},
    )

    assert result.intent == "collect_name"
    assert result.reasoning == "User expressed intent to provide their name"
    assert result.confidence == 0.9
    assert result.entities == {}


def test_fsm_context_conversation_management():
    """Test conversation management within FSM context."""
    context = FSMContext(max_history_size=3, max_message_length=100)

    # Test adding messages
    context.conversation.add_user_message("Hello")
    context.conversation.add_system_message("Hi there!")
    context.conversation.add_user_message("How are you?")
    context.conversation.add_system_message("I'm doing well, thank you!")

    # The get_recent method multiplies by 2, so get_recent(2) returns last 4 exchanges
    # This appears to be designed for "exchange pairs" concept
    recent = context.conversation.get_recent(2)
    assert len(recent) == 4  # Returns last 4 exchanges for 2 "pairs"

    # Test with get_recent(1) - should return last 2 exchanges
    recent_1 = context.conversation.get_recent(1)
    assert len(recent_1) == 2  # Returns last 2 exchanges for 1 "pair"

    # Test message truncation
    long_message = "x" * 150  # Longer than max_message_length
    context.conversation.add_user_message(long_message)

    # Should be truncated
    exchanges = context.conversation.exchanges
    last_exchange = exchanges[-1]
    assert len(last_exchange["user"]) <= 100 + len("... [truncated]")
    assert "truncated" in last_exchange["user"]


class TestGetConversationDataFiltering:
    """Tests for internal key filtering in get_conversation_data()."""

    def test_internal_keys_not_returned(
        self, sample_fsm_definition_v2, mock_llm2_interface
    ):
        """get_conversation_data() must filter out internal metadata keys."""
        manager = FSMManager(
            fsm_loader=lambda fid: sample_fsm_definition_v2,
            llm_interface=mock_llm2_interface,
            transition_evaluator=TransitionEvaluator(sample_fsm_definition_v2),
        )
        conv_id, _ = manager.start_conversation("test-fsm")

        data = manager.get_conversation_data(conv_id)

        # Internal keys injected at start_conversation() must NOT appear
        for key in data:
            assert not key.startswith("_"), (
                f"Internal key '{key}' leaked through get_data()"
            )
            assert not key.startswith("system_"), (
                f"Internal key '{key}' leaked through get_data()"
            )
            assert not key.startswith("internal_"), (
                f"Internal key '{key}' leaked through get_data()"
            )

    def test_user_data_preserved(self, sample_fsm_definition_v2, mock_llm2_interface):
        """User-facing data must still be returned after filtering."""
        manager = FSMManager(
            fsm_loader=lambda fid: sample_fsm_definition_v2,
            llm_interface=mock_llm2_interface,
            transition_evaluator=TransitionEvaluator(sample_fsm_definition_v2),
        )
        conv_id, _ = manager.start_conversation("test-fsm")

        # Inject user data directly
        manager.instances[conv_id].context.data["user_name"] = "Alice"
        manager.instances[conv_id].context.data["preference"] = "dark mode"

        data = manager.get_conversation_data(conv_id)

        assert data["user_name"] == "Alice"
        assert data["preference"] == "dark mode"


class TestGetConversationDataNestedFiltering:
    """The internal-prefix filter applies at EVERY nesting level (D-010).

    Both clauses matter together: a fix that recursed but dropped the whole
    nested dict would pass the "secret is gone" clause while destroying the
    caller's data, and a fix that kept everything would pass the "sibling
    survives" clause while leaking. Neither clause alone pins the behaviour.
    """

    @staticmethod
    def _manager_with(data):
        import threading
        from unittest.mock import Mock

        from fsm_llm.core.definitions import FSMContext, FSMInstance

        manager = FSMManager(llm_interface=Mock(spec=LLMInterface))
        context = FSMContext()
        context.data.update(data)
        manager.instances["conv-1"] = FSMInstance(
            fsm_id="f", current_state="start", context=context
        )
        # get_conversation_data now snapshots under the per-conversation lock,
        # which production always creates alongside the instance. Register it so
        # this fixture reflects the create-together / remove-together invariant
        # the read path (and L11) now enforce.
        manager._conversation_locks["conv-1"] = threading.RLock()
        return manager

    def test_nested_internal_key_stripped_and_sibling_kept(self):
        """The exact shape probed in findings/core-partial-twins.md item 3."""
        manager = self._manager_with(
            {"user": {"_internal_note": "secret", "name": "bob"}, "system_flag": True}
        )

        data = manager.get_conversation_data("conv-1")

        assert data == {"user": {"name": "bob"}}

    def test_every_internal_prefix_stripped_at_depth_case_insensitively(self):
        """Nesting must not weaken the predicate: same prefixes, same casing."""
        nested = {
            "_lead": "x",
            "__dunder": "x",
            "System_flag": "x",
            "INTERNAL_id": "x",
            "keep": "x",
        }
        manager = self._manager_with({"outer": {"inner": nested}})

        data = manager.get_conversation_data("conv-1")

        assert data == {"outer": {"inner": {"keep": "x"}}}

    def test_dicts_inside_lists_and_tuples_are_filtered(self):
        """``{"users": [{"_note": ...}]}`` is the same leak as the flat form."""
        manager = self._manager_with(
            {
                "users": [{"_note": "secret", "name": "bob"}],
                "pairs": ({"_note": "secret", "name": "ann"},),
            }
        )

        data = manager.get_conversation_data("conv-1")

        assert data["users"] == [{"name": "bob"}]
        assert data["pairs"] == ({"name": "ann"},)

    def test_scalars_and_falsy_values_survive_at_depth(self):
        """Recursion must not become a truthiness filter."""
        manager = self._manager_with(
            {"a": {"b": {"empty_list": [], "zero": 0, "false": False, "blank": ""}}}
        )

        data = manager.get_conversation_data("conv-1")

        assert data == {
            "a": {"b": {"empty_list": [], "zero": 0, "false": False, "blank": ""}}
        }

    def test_non_str_key_is_kept_but_its_value_is_still_filtered(self):
        """``has_internal_prefix`` cannot match a non-str key (constants D-017)."""
        manager = self._manager_with({7: {"_note": "secret", "name": "bob"}})

        data = manager.get_conversation_data("conv-1")

        assert data == {7: {"name": "bob"}}

    def test_depth_bound_is_fail_closed_and_shared_with_the_sibling_filters(self):
        """A container past the bound is DROPPED, never returned unfiltered.

        The bound is pinned against ``clean_context_keys`` rather than against a
        hardcoded number, so re-declaring a local depth limit in ``fsm.py``
        fails this test (D-011: one bound for every context filter).
        """
        from fsm_llm.constants import MAX_CONTEXT_FILTER_DEPTH
        from fsm_llm.context import clean_context_keys

        # Build {"n": {"n": ... {"payload": {"_secret": "x"}}}} with the payload
        # dict sitting exactly one level past the bound.
        payload = {"_secret": "x", "kept": "y"}
        too_deep = payload
        for _ in range(MAX_CONTEXT_FILTER_DEPTH + 1):
            too_deep = {"n": too_deep}

        manager = self._manager_with(dict(too_deep))
        data = manager.get_conversation_data("conv-1")

        # Walk to the deepest surviving level: the payload must be gone whole,
        # not present-and-unfiltered.
        cursor = data
        for _ in range(MAX_CONTEXT_FILTER_DEPTH):
            cursor = cursor["n"]
        assert cursor == {}, f"container past the bound survived: {cursor!r}"

        # Same input, same disposition as the two sibling filters.
        assert data == clean_context_keys(dict(too_deep), "conv-1")

    def test_just_inside_the_bound_is_still_filtered_normally(self):
        """Vacuity guard: the bound must not be swallowing the whole input."""
        from fsm_llm.constants import MAX_CONTEXT_FILTER_DEPTH

        payload = {"_secret": "x", "kept": "y"}
        inside = payload
        for _ in range(MAX_CONTEXT_FILTER_DEPTH):
            inside = {"n": inside}

        manager = self._manager_with(dict(inside))
        data = manager.get_conversation_data("conv-1")

        cursor = data
        for _ in range(MAX_CONTEXT_FILTER_DEPTH):
            cursor = cursor["n"]
        assert cursor == {"kept": "y"}

    def test_self_referential_dict_terminates(self):
        """A cycle terminates: the active-path guard drops it (D-011 of
        plan-2026-09-21T203800-8a03483a); the depth bound stays the backstop."""
        cyclic: dict = {"name": "bob"}
        cyclic["self"] = cyclic

        manager = self._manager_with({"root": cyclic})

        assert manager.get_conversation_data("conv-1")["root"]["name"] == "bob"

    def test_matched_keys_are_dropped_not_redacted(self):
        """This site DROPS; ``context.filtering.redact_context`` is the redactor (D-010)."""
        manager = self._manager_with({"user": {"_internal_note": "secret"}})

        data = manager.get_conversation_data("conv-1")

        assert "_internal_note" not in data["user"]
        assert "<redacted>" not in repr(data)


class TestFsmHandlerContractTwins:
    """A ``critical=True`` handler's raise must not vanish inside ``fsm.py``.

    Two sites, one contract (D-009 of plan-2026-07-20T040150-876e7164):

    * ``_process_message_locked``'s ERROR-timing pass — the handler failure is
      promoted, chained from the failure it was reporting on.
    * ``update_conversation_context``'s CONTEXT_UPDATE pass — the keys the call
      committed are rolled back before the failure propagates.
    """

    def _manager(self, fsm_definition, llm_interface):
        return FSMManager(
            fsm_loader=lambda fid: fsm_definition,
            llm_interface=llm_interface,
            transition_evaluator=TransitionEvaluator(fsm_definition),
        )

    @staticmethod
    def _raising_handler(name, timing, exc):
        from fsm_llm.handlers import HandlerBuilder

        def _boom(context):
            raise exc

        return HandlerBuilder(name).at(timing).critical().do(_boom)

    # -- SC-12: ERROR-timing handler ------------------------------------

    def test_critical_error_handler_failure_reaches_caller_with_cause_and_log(
        self, sample_fsm_definition_v2, mock_llm2_interface
    ):
        """All three clauses at once.

        Surfacing the handler error ALONE would be strictly worse than the old
        swallow: it would destroy the diagnosis of the failure being reported,
        which is exactly what the original deferral feared. ``raise X from Y``
        plus an explicit ``log.error`` keeps both, so all three are asserted
        together and none of them may be dropped.
        """
        from fsm_llm.handlers import HandlerExecutionError, HandlerTiming
        from fsm_llm.logging import logger

        manager = self._manager(sample_fsm_definition_v2, mock_llm2_interface)
        conv_id, _ = manager.start_conversation("test-fsm")

        original = RuntimeError("pipeline exploded")

        def _explode(*args, **kwargs):
            raise original

        manager._pipeline.process = _explode
        manager.register_handler(
            self._raising_handler(
                "BoomOnError",
                HandlerTiming.ERROR,
                ValueError("error handler exploded"),
            )
        )

        records: list[str] = []
        sink_id = logger.add(
            lambda m: records.append(m.record["message"]), level="ERROR"
        )
        logger.enable("fsm_llm")
        try:
            with pytest.raises(HandlerExecutionError) as exc_info:
                manager.process_message(conv_id, "hello")
        finally:
            logger.remove(sink_id)
            # Restore the library default (logging.py disables on import).
            logger.disable("fsm_llm")

        # (1) the critical handler's failure reaches the caller -- it is NOT
        #     relabelled as the generic FSMError("Failed to process message").
        assert "error handler exploded" in str(exc_info.value)

        # (2) the failure being unwound survives as __cause__.
        assert exc_info.value.__cause__ is original

        # (3) a single log.error line names BOTH failures, because a traceback
        #     is not guaranteed to reach the operator.
        both = [
            line
            for line in records
            if "error handler exploded" in line and "pipeline exploded" in line
        ]
        assert both, (
            "no single ERROR log line named both the handler failure and the "
            f"original failure; captured: {records}"
        )

    # -- SC-13: CONTEXT_UPDATE-timing handler ---------------------------

    def test_context_update_is_rolled_back_when_critical_handler_raises(
        self, sample_fsm_definition_v2, mock_llm2_interface
    ):
        """The keys this call committed are restored before the raise escapes."""
        from fsm_llm.handlers import HandlerExecutionError, HandlerTiming

        manager = self._manager(sample_fsm_definition_v2, mock_llm2_interface)
        conv_id, _ = manager.start_conversation("test-fsm")
        manager.instances[conv_id].context.data["existing_key"] = "pre_call_value"

        manager.register_handler(
            self._raising_handler(
                "BoomOnContextUpdate",
                HandlerTiming.CONTEXT_UPDATE,
                ValueError("context handler exploded"),
            )
        )

        with pytest.raises(HandlerExecutionError):
            manager.update_conversation_context(
                conv_id, {"existing_key": "post_call_value", "brand_new_key": "added"}
            )

        data = manager.get_conversation_data(conv_id)
        # Overwritten key restored to its pre-call value...
        assert data["existing_key"] == "pre_call_value"
        # ...and a key the call introduced is removed entirely.
        assert "brand_new_key" not in data

    def test_unrelated_keys_are_untouched_by_the_rollback(
        self, sample_fsm_definition_v2, mock_llm2_interface
    ):
        """The snapshot is SCOPED: only the written keys are restored.

        A full-context deepcopy would also discard deltas that other, already
        successful handlers wrote -- which is why shape (c) was chosen.
        """
        from fsm_llm.handlers import HandlerExecutionError, HandlerTiming

        manager = self._manager(sample_fsm_definition_v2, mock_llm2_interface)
        conv_id, _ = manager.start_conversation("test-fsm")
        manager.instances[conv_id].context.data["untouched"] = "still here"

        manager.register_handler(
            self._raising_handler(
                "BoomOnContextUpdate",
                HandlerTiming.CONTEXT_UPDATE,
                ValueError("context handler exploded"),
            )
        )

        with pytest.raises(HandlerExecutionError):
            manager.update_conversation_context(conv_id, {"written": "value"})

        data = manager.get_conversation_data(conv_id)
        assert data["untouched"] == "still here"
        assert "written" not in data

    def test_shallow_snapshot_boundary_in_place_mutation_is_not_restored(
        self, sample_fsm_definition_v2, mock_llm2_interface
    ):
        """Pins the KNOWN, DELIBERATE limit of the scoped rollback.

        ``pre_commit`` holds the same objects as ``context.data``, so a handler
        that mutates a pre-existing dict/list value IN PLACE before failing is
        NOT restored. This test documents that boundary rather than leaving it
        latent -- it is not a bug report. Do not "fix" it with a deep copy;
        D-005/D-020 rejected that deliberately (handlers are contractually
        supposed to return a delta, not reach into ``context.data``).
        """
        from fsm_llm.handlers import (
            HandlerBuilder,
            HandlerExecutionError,
            HandlerTiming,
        )

        manager = self._manager(sample_fsm_definition_v2, mock_llm2_interface)
        conv_id, _ = manager.start_conversation("test-fsm")

        shared = {"kept": "yes"}
        manager.instances[conv_id].context.data["profile"] = shared

        def _mutate_then_raise(context):
            manager.instances[conv_id].context.data["profile"]["injected"] = "leaked"
            raise ValueError("context handler exploded")

        manager.register_handler(
            HandlerBuilder("MutateThenBoom")
            .at(HandlerTiming.CONTEXT_UPDATE)
            .critical()
            .do(_mutate_then_raise)
        )

        with pytest.raises(HandlerExecutionError):
            manager.update_conversation_context(
                conv_id, {"profile": shared, "brand_new_key": "added"}
            )

        data = manager.get_conversation_data(conv_id)
        # Key-level rollback still works for the key the call introduced.
        assert "brand_new_key" not in data
        # But the in-place mutation of the pre-existing value survives.
        assert data["profile"]["injected"] == "leaked"


class TestStep4CacheSizeValidation:
    """Review concern 4: a cache bound below 1 made every start fail with a
    cryptic popitem error; it is now rejected at construction."""

    @pytest.mark.parametrize("size", [0, -1])
    def test_step_4_1_manager_rejects_cache_size_below_one(self, size):
        from fsm_llm.fsm import FSMManager

        with pytest.raises(ValueError, match="max_fsm_cache_size"):
            FSMManager(llm_interface=_mock_llm(), max_fsm_cache_size=size)

    def test_step_4_1_api_rejects_cache_size_zero_at_construction(self):
        with pytest.raises(ValueError, match="max_fsm_cache_size"):
            API.from_definition(
                _blocked_fsm(), llm_interface=_mock_llm(), max_fsm_cache_size=0
            )

    def test_step_4_1_cache_size_one_is_accepted(self):
        api = API.from_definition(
            _blocked_fsm(), llm_interface=_mock_llm(), max_fsm_cache_size=1
        )
        conv_id, _ = api.start_conversation()
        assert api.get_current_state(conv_id) == "start"


class _Step14Tracker:
    """Per-thread held-lock stacks plus a shared log of every acquire."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._guard = threading.Lock()
        self.acquires: list[tuple[str, tuple[str, ...]]] = []

    def held(self) -> list[str]:
        if not hasattr(self._local, "held"):
            self._local.held = []
        return self._local.held

    def record(self, name: str) -> None:
        with self._guard:
            self.acquires.append((name, tuple(self.held())))

    def under_cache_lock(self) -> list[tuple[str, tuple[str, ...]]]:
        with self._guard:
            return [(n, h) for n, h in self.acquires if "cache" in h]


class _Step14Lock:
    """Recording wrapper; a blocking acquire is bounded so a deadlock fails."""

    def __init__(self, inner: Any, name: str, tracker: _Step14Tracker) -> None:
        self._inner = inner
        self._name = name
        self._tracker = tracker

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        self._tracker.record(self._name)
        if blocking and timeout == -1:
            timeout = _JOIN_TIMEOUT_S
        got = self._inner.acquire(blocking, timeout)
        if blocking and not got:
            raise AssertionError(f"{self._name} not acquired: possible deadlock")
        if got:
            self._tracker.held().append(self._name)
        return got

    def release(self) -> None:
        self._tracker.held().remove(self._name)
        self._inner.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


class _Step14ConvLocks(dict):
    def __init__(self, tracker: _Step14Tracker, existing: dict) -> None:
        super().__init__()
        self._tracker = tracker
        for key, value in existing.items():
            self[key] = value

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, _Step14Lock(value, f"conv:{key}", self._tracker))


class TestStep14CacheLock:
    def test_cache_hit_does_not_take_manager_lock(self):
        api = _a2026_09_22_api()
        conv_id, _ = api.start_conversation()
        manager = api.fsm_manager
        manager.get_fsm_definition(api.fsm_id)
        tracker = _Step14Tracker()
        manager._lock = _Step14Lock(manager._lock, "manager", tracker)

        fsm_def = manager.get_fsm_definition(api.fsm_id)
        state = manager.resolve_state_definition(manager.instances[conv_id], conv_id)

        assert fsm_def is api.fsm_definition
        assert state.id == "start"
        assert [n for n, _ in tracker.acquires if n == "manager"] == []

    def test_lock_order_under_concurrent_start_and_converse(self):
        api = _a2026_09_22_api()
        manager = api.fsm_manager
        convs = [api.start_conversation()[0] for _ in range(3)]
        tracker = _Step14Tracker()
        manager._lock = _Step14Lock(manager._lock, "manager", tracker)
        manager._fsm_cache_lock = _Step14Lock(manager._fsm_cache_lock, "cache", tracker)
        manager._conversation_locks = _Step14ConvLocks(
            tracker, manager._conversation_locks
        )
        inner_loader = manager.fsm_loader
        loader_held: list[tuple[str, ...]] = []

        def _loader(fsm_id: str) -> FSMDefinition:
            loader_held.append(tuple(tracker.held()))
            return inner_loader(fsm_id)

        manager.fsm_loader = _loader
        stop = threading.Event()
        errors: list[BaseException] = []

        def _guarded(fn) -> Any:
            def _run() -> None:
                try:
                    fn()
                except BaseException as exc:  # surfaced below
                    errors.append(exc)

            return threading.Thread(target=_run, daemon=True)

        def _evict() -> None:
            while not stop.is_set():
                with manager._fsm_cache_lock:
                    manager.fsm_cache.clear()

        def _start() -> None:
            for _ in range(10):
                api.start_conversation()

        def _converse(conv_id: str) -> Any:
            def _run() -> None:
                for i in range(10):
                    api.converse(f"turn {i}", conv_id)

            return _run

        evictor = _guarded(_evict)
        workers = [_guarded(_start) for _ in range(2)]
        workers += [_guarded(_converse(c)) for c in convs]
        evictor.start()
        for t in workers:
            t.start()
        _join_all(workers)
        stop.set()
        _join_all([evictor])

        assert errors == []
        assert loader_held, "the evictor never forced a cache miss"
        assert all("cache" not in held for held in loader_held)
        assert tracker.under_cache_lock() == []

    def test_concurrent_miss_keeps_first_inserted_definition(self):
        api = _a2026_09_22_api()
        manager = api.fsm_manager
        manager.fsm_cache.clear()
        first = api.fsm_definition.model_copy()
        second = api.fsm_definition.model_copy()
        calls: list[int] = []

        def _loader(fsm_id: str) -> FSMDefinition:
            calls.append(1)
            if len(calls) == 1:
                other = threading.Thread(
                    target=manager.get_fsm_definition, args=(fsm_id,), daemon=True
                )
                other.start()
                other.join(timeout=1.0)
                return first
            return second

        manager.fsm_loader = _loader
        result = manager.get_fsm_definition(api.fsm_id)
        assert len(calls) == 2
        assert result is second
        assert manager.fsm_cache[api.fsm_id] is second
