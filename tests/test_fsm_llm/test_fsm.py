import json
import pytest
from unittest.mock import MagicMock, AsyncMock

from llm_fsm.definitions import (
    FSMDefinition, FSMInstance, State, Transition,
    TransitionCondition, FSMContext,
    DataExtractionRequest, DataExtractionResponse,
    ResponseGenerationRequest, ResponseGenerationResponse,
    TransitionDecisionRequest, TransitionDecisionResponse
)
from llm_fsm.fsm import FSMManager
from llm_fsm.validator import FSMValidator
from llm_fsm.utilities import extract_json_from_text, load_fsm_from_file
from llm_fsm.llm import LLMInterface
from llm_fsm.visualizer import visualize_fsm_ascii
from llm_fsm.prompts import (
    DataExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
    TransitionPromptBuilder
)
from llm_fsm.transition_evaluator import TransitionEvaluator


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
                        "priority": 1
                    }
                ]
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
                                "requires_context_keys": ["name"]
                            }
                        ],
                        "priority": 1
                    }
                ]
            },
            "farewell": {
                "id": "farewell",
                "description": "Farewell state",
                "purpose": "Say goodbye to the user",
                "response_instructions": "Say goodbye using the user's name",
                "transitions": []  # Terminal state
            }
        }
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
                "transitions": []
            }
        }
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
                "transitions": []  # No transitions to other states
            },
            "orphaned": {
                "id": "orphaned",
                "description": "Orphaned state",
                "purpose": "This state is orphaned",
                "transitions": []
            }
        }
    }


@pytest.fixture
def mock_llm_interface():
    """Fixture for a mocked LLM interface for the 2-pass architecture."""
    mock_interface = MagicMock(spec=LLMInterface)

    # Mock data extraction response
    mock_extraction_response = DataExtractionResponse(
        extracted_data={"name": "John"},
        confidence=0.9,
        reasoning="User provided their name in the message"
    )

    # Mock response generation response
    mock_response_generation = ResponseGenerationResponse(
        message="Hello John! Nice to meet you.",
        reasoning="Generated greeting using extracted name"
    )

    # Mock transition decision response
    mock_transition_decision = TransitionDecisionResponse(
        selected_transition="collect_name",
        reasoning="User wants to provide their name"
    )

    # Set up the mock methods
    mock_interface.extract_data.return_value = mock_extraction_response
    mock_interface.generate_response.return_value = mock_response_generation
    mock_interface.decide_transition.return_value = mock_transition_decision

    return mock_interface


# Actual tests

def test_load_valid_fsm_definition(valid_fsm_data, tmp_path):
    """Test loading a valid FSM definition from a file."""
    # Write the valid FSM data to a temporary file
    fsm_file = tmp_path / "valid_fsm.json"
    with open(fsm_file, 'w') as f:
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

    # Test has_keys method
    assert context.has_keys(["name"])
    assert context.has_keys(["name", "email"])
    assert not context.has_keys(["name", "age"])

    # Test get_missing_keys method
    missing = context.get_missing_keys(["name", "age", "phone"])
    assert "age" in missing
    assert "phone" in missing
    assert "name" not in missing


def test_transition_evaluator():
    """Test the transition evaluator with deterministic transitions."""
    from llm_fsm.transition_evaluator import TransitionEvaluator, TransitionEvaluatorConfig

    # Create a transition evaluator
    config = TransitionEvaluatorConfig(
        minimum_confidence=0.7,
        ambiguity_threshold=0.2
    )
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
                        description="Name is provided",
                        requires_context_keys=["name"]
                    )
                ]
            )
        ]
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
    transition_builder = TransitionPromptBuilder()
    transition_evaluator = TransitionEvaluator()

    # Create an FSM manager with the mock loader and interface
    fsm_manager = FSMManager(
        fsm_loader=mock_loader,
        llm_interface=mock_llm_interface,
        data_extraction_prompt_builder=data_extraction_builder,
        response_generation_prompt_builder=response_generation_builder,
        transition_prompt_builder=transition_builder,
        transition_evaluator=transition_evaluator
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
        transitions=[]  # Terminal state
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
                        requires_context_keys=["name"]
                    )
                ]
            )
        ]
    )

    fsm_def = FSMDefinition(
        name="Test FSM",
        description="Test FSM",
        initial_state="collect_name",
        persona="A friendly assistant",
        states={
            "collect_name": collect_name_state,
            "farewell": farewell_state
        }
    )

    instance = FSMInstance(
        fsm_id="test_fsm",
        current_state="collect_name",
        persona="A friendly assistant"
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
        user_message="Hello"
    )

    # Verify the response prompt contains key elements
    assert "<response_generation>" in response_prompt
    assert "</response_generation>" in response_prompt
    assert "persona" in response_prompt.lower()


def test_extract_json_from_text():
    """Test the utility function for extracting JSON from text."""
    # Test with JSON in code block
    text1 = "Here is the JSON:\n```json\n{\"name\": \"John\", \"age\": 30}\n```"
    json1 = extract_json_from_text(text1)
    assert json1 is not None
    assert json1["name"] == "John"
    assert json1["age"] == 30

    # Test with JSON without code block
    text2 = "Here is the JSON: {\"name\": \"Jane\", \"age\": 25}"
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
        transition_prompt_builder=TransitionPromptBuilder(),
        transition_evaluator=TransitionEvaluator()
    )

    # Start a conversation
    conversation_id, initial_response = fsm_manager.start_conversation("test_fsm")

    # Verify initial state
    assert fsm_manager.get_conversation_state(conversation_id) == "welcome"
    assert not fsm_manager.has_conversation_ended(conversation_id)

    # Process a user message
    response = fsm_manager.process_message(conversation_id, "Hello, my name is John")

    # Verify the LLM interface methods were called
    assert mock_llm_interface.extract_data.called
    assert mock_llm_interface.generate_response.called

    # Verify conversation data
    conversation_data = fsm_manager.get_conversation_data(conversation_id)
    assert "name" in conversation_data  # Should have extracted name
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
        transition_prompt_builder=TransitionPromptBuilder(),
        transition_evaluator=TransitionEvaluator()
    )

    # Start a conversation
    conversation_id, _ = fsm_manager.start_conversation("test_fsm")

    # Initially, conversation should not be ended (welcome state has transitions)
    assert not fsm_manager.has_conversation_ended(conversation_id)

    # Manually change the state to a terminal state (farewell has no transitions)
    fsm_manager.instances[conversation_id].current_state = "farewell"

    # Now the conversation should be detected as ended
    assert fsm_manager.has_conversation_ended(conversation_id)


def test_data_extraction_request_response_models():
    """Test the data extraction request and response models."""
    # Create a data extraction request
    request = DataExtractionRequest(
        system_prompt="Extract the user's name from their message",
        user_message="Hi, I'm John Smith",
        context={"current_state": "collect_name"}
    )

    assert request.system_prompt == "Extract the user's name from their message"
    assert request.user_message == "Hi, I'm John Smith"
    assert request.context["current_state"] == "collect_name"

    # Create a data extraction response
    response = DataExtractionResponse(
        extracted_data={"name": "John Smith"},
        confidence=0.95,
        reasoning="User clearly stated their name"
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
        previous_state=None
    )

    assert request.system_prompt == "Generate a friendly greeting using the user's name"
    assert request.user_message == "Hello"
    assert request.extracted_data["name"] == "John"

    # Create a response generation response
    response = ResponseGenerationResponse(
        message="Hello John! Welcome to our service.",
        reasoning="Generated personalized greeting using extracted name"
    )

    assert response.message == "Hello John! Welcome to our service."
    assert response.reasoning == "Generated personalized greeting using extracted name"


def test_transition_decision_models():
    """Test the transition decision request and response models."""
    from llm_fsm.definitions import TransitionOption

    # Create transition options
    options = [
        TransitionOption(
            target_state="collect_name",
            description="User wants to provide their name",
            priority=1
        ),
        TransitionOption(
            target_state="farewell",
            description="User wants to end the conversation",
            priority=2
        )
    ]

    # Create a transition decision request
    request = TransitionDecisionRequest(
        system_prompt="Choose the appropriate transition based on user input",
        current_state="welcome",
        available_transitions=options,
        context={"greeting_given": True},
        user_message="I'd like to tell you my name",
        extracted_data={"intent": "provide_name"}
    )

    assert request.current_state == "welcome"
    assert len(request.available_transitions) == 2
    assert request.user_message == "I'd like to tell you my name"

    # Create a transition decision response
    response = TransitionDecisionResponse(
        selected_transition="collect_name",
        reasoning="User expressed intent to provide their name"
    )

    assert response.selected_transition == "collect_name"
    assert response.reasoning == "User expressed intent to provide their name"


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