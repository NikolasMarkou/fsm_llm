import json
from unittest.mock import MagicMock

import pytest


def configure_mock_extract_field(mock_llm, mock_data=None):
    """Configure a mock LLM with extract_field support."""
    from fsm_llm.types import FieldExtractionResponse

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


from fsm_llm.dialog.definitions import (
    ClassificationResult,
    FSMContext,
    FSMDefinition,
    FSMInstance,
    State,
    Transition,
    TransitionCondition,
)
from fsm_llm.dialog.fsm import FSMManager
from fsm_llm.dialog.prompts import (
    DataExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from fsm_llm.dialog.transition_evaluator import TransitionEvaluator
from fsm_llm.runtime._litellm import LLMInterface
from fsm_llm.types import (
    DataExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.utilities import extract_json_from_text, load_fsm_from_file
from fsm_llm.validator import FSMValidator
from fsm_llm.visualizer import visualize_fsm_ascii

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
    with open(fsm_file, "w") as f:
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
    from fsm_llm.dialog.transition_evaluator import (
        TransitionEvaluator,
        TransitionEvaluatorConfig,
    )

    # Create a transition evaluator
    config = TransitionEvaluatorConfig(minimum_confidence=0.7, ambiguity_threshold=0.2)
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

    # Test data extraction prompt builder
    data_extraction_builder = DataExtractionPromptBuilder()
    extraction_prompt = data_extraction_builder.build_extraction_prompt(
        instance, collect_name_state, fsm_def
    )

    # Verify the extraction prompt contains key elements
    assert "<data_extraction>" in extraction_prompt
    assert "</data_extraction>" in extraction_prompt
    assert "extraction_focus" in extraction_prompt.lower()

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
        extracted_data={"name": "John"},
        context={"current_state": "welcome"},
        transition_occurred=False,
        previous_state=None,
    )

    assert request.system_prompt == "Generate a friendly greeting using the user's name"
    assert request.user_message == "Hello"
    assert request.extracted_data["name"] == "John"

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


# --------------------------------------------------------------
# S7: compile-once-per-FSM cache on FSMManager
# --------------------------------------------------------------


def _s7_greeter_fsm_dict() -> dict:
    """Single-state FSM used across S7 cache tests."""
    return {
        "name": "s7_greeter",
        "description": "one-state greeter",
        "initial_state": "hello",
        "persona": "test",
        "states": {
            "hello": {
                "id": "hello",
                "description": "say hi",
                "purpose": "greet",
                "response_instructions": "Say hello.",
                "transitions": [],
            },
        },
    }


class TestFSMManagerCompiledCache:
    """S7: FSMManager exposes get_compiled_term(fsm_id) with LRU cache."""

    def _make_manager(self, mock_llm_interface, **kwargs) -> FSMManager:
        defn = FSMDefinition.model_validate(_s7_greeter_fsm_dict())
        return FSMManager(
            fsm_loader=lambda fid: defn,
            llm_interface=mock_llm_interface,
            **kwargs,
        )

    def test_get_compiled_term_returns_term(self, mock_llm_interface) -> None:
        from fsm_llm.runtime.ast import Abs

        manager = self._make_manager(mock_llm_interface)
        term = manager.get_compiled_term("fsm-a")
        # compile_fsm returns an Abs chain (outer state_id abstraction)
        assert isinstance(term, Abs)

    def test_double_call_returns_same_object(self, mock_llm_interface) -> None:
        """Cache hit returns the literally-same Term object (identity, not
        just equality). Proves the cache stores and reuses."""
        manager = self._make_manager(mock_llm_interface)
        t1 = manager.get_compiled_term("fsm-a")
        t2 = manager.get_compiled_term("fsm-a")
        assert t1 is t2

    def test_compiled_terms_attr_removed_in_r2(self, mock_llm_interface) -> None:
        """R2 (D-PLAN-07): the per-manager `_compiled_terms` OrderedDict
        was removed. Cache lives in the kernel via `compile_fsm_cached`.
        """
        manager = self._make_manager(mock_llm_interface)
        assert not hasattr(manager, "_compiled_terms"), (
            "FSMManager._compiled_terms must be deleted in R2 — the "
            "compiled-term cache lives in lam.compile_fsm_cached."
        )

    def test_kernel_cache_observes_repeat_call_as_hit(self, mock_llm_interface) -> None:
        """Repeat call on the same (loader-resolved) fsm_id increments
        the kernel cache's hit counter (lru_cache.cache_info().hits)."""
        from fsm_llm.dialog.compile_fsm import _compile_fsm_by_id

        # Clear residual cache state from prior tests so our hits/misses
        # accounting is deterministic.
        _compile_fsm_by_id.cache_clear()

        manager = self._make_manager(mock_llm_interface)
        manager.get_compiled_term("fsm-a")
        info_after_first = _compile_fsm_by_id.cache_info()
        assert info_after_first.misses == 1
        assert info_after_first.hits == 0

        manager.get_compiled_term("fsm-a")
        info_after_second = _compile_fsm_by_id.cache_info()
        assert info_after_second.hits == info_after_first.hits + 1
        assert info_after_second.misses == info_after_first.misses

    def test_distinct_fsm_ids_get_distinct_cache_entries(
        self, mock_llm_interface
    ) -> None:
        """fsm_id is part of the cache key — two distinct ids on the same
        loader produce two cache entries (currsize grows by 2)."""
        from fsm_llm.dialog.compile_fsm import _compile_fsm_by_id

        _compile_fsm_by_id.cache_clear()
        manager = self._make_manager(mock_llm_interface)
        manager.get_compiled_term("fsm-a")
        manager.get_compiled_term("fsm-b")
        info = _compile_fsm_by_id.cache_info()
        # Both ids resolve to the same FSMDefinition via the loader, so
        # the JSON content is byte-equal — but the (fsm_id, json) tuple
        # differs, giving us two cache slots. Telemetry coherence per
        # D-PLAN-07.
        assert info.currsize >= 2

    def test_compile_failure_does_not_pollute_kernel_cache(
        self, mock_llm_interface
    ) -> None:
        """If a compile attempt fails, the kernel lru_cache rejects the
        entry (lru_cache stores function results; raised exceptions skip
        the store). A subsequent call with a valid definition compiles
        successfully.

        Note (R2 behaviour shift): pre-R2 the failure surfaced as
        ASTConstructionError because ``compile_fsm`` was invoked
        directly. Post-R2 ``compile_fsm_cached`` round-trips the
        definition through ``model_dump_json`` → ``model_validate_json``
        — so a hand-mutated FSMDefinition (states={} via
        ``object.__setattr__`` bypass) is caught by Pydantic at re-
        validation time, raising ``pydantic.ValidationError`` instead.
        Either exception path satisfies the "compile failure" contract;
        the behavioural invariant is "no cache pollution + retry works"."""
        from pydantic import ValidationError

        from fsm_llm.dialog.compile_fsm import _compile_fsm_by_id
        from fsm_llm.runtime.errors import ASTConstructionError

        _compile_fsm_by_id.cache_clear()

        bad_defn = FSMDefinition.model_validate(_s7_greeter_fsm_dict())
        object.__setattr__(bad_defn, "states", {})
        good_defn = FSMDefinition.model_validate(_s7_greeter_fsm_dict())

        calls = {"n": 0}

        def flaky_loader(fid: str) -> FSMDefinition:
            calls["n"] += 1
            return bad_defn if calls["n"] == 1 else good_defn

        manager = FSMManager(
            fsm_loader=flaky_loader,
            llm_interface=mock_llm_interface,
        )
        with pytest.raises((ASTConstructionError, ValidationError)):
            manager.get_compiled_term("fsm-flaky")
        # Behavioural assertion: a retry with the good definition succeeds.
        manager.fsm_cache.pop("fsm-flaky", None)
        term = manager.get_compiled_term("fsm-flaky")
        from fsm_llm.runtime.ast import Abs

        assert isinstance(term, Abs)

    def test_fsm_cache_independent_of_kernel_compile_cache(
        self, mock_llm_interface
    ) -> None:
        """The per-manager fsm_cache (FSM-definition cache) is independent
        of the kernel compile cache. Touching only the definition cache
        does not populate the compile cache."""
        from fsm_llm.dialog.compile_fsm import _compile_fsm_by_id

        _compile_fsm_by_id.cache_clear()
        manager = self._make_manager(mock_llm_interface)
        # Touch only the definition cache.
        manager.get_fsm_definition("fsm-a")
        assert "fsm-a" in manager.fsm_cache
        assert _compile_fsm_by_id.cache_info().currsize == 0
        # Touch the compile path — kernel cache populates.
        manager.get_compiled_term("fsm-b")
        assert "fsm-b" in manager.fsm_cache
        assert _compile_fsm_by_id.cache_info().currsize == 1

    def test_concurrent_get_compiled_term_is_thread_safe(
        self, mock_llm_interface
    ) -> None:
        """Two threads calling get_compiled_term(fsm_id) concurrently
        both observe a Term. The kernel lru_cache is thread-safe at the
        Python level (CPython's GIL serialises dict access); both
        threads receive the same cached object once the cache is warm.
        """
        import threading

        from fsm_llm.dialog.compile_fsm import _compile_fsm_by_id

        _compile_fsm_by_id.cache_clear()
        manager = self._make_manager(mock_llm_interface)
        results: list = []
        barrier = threading.Barrier(2)

        def worker() -> None:
            barrier.wait()
            results.append(manager.get_compiled_term("fsm-a"))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive(), "thread hung — possible deadlock in cache lock"

        assert len(results) == 2
        # Once the race resolves, repeat calls return the same object.
        cached = manager.get_compiled_term("fsm-a")
        assert results[0] is cached or results[1] is cached
