import json
import pytest
from pathlib import Path

from llm_fsm.definitions import (
    FSMDefinition, FSMInstance, State, Transition,
    TransitionCondition, FSMContext, StateTransition, LLMResponse
)
from llm_fsm.fsm import FSMManager
from llm_fsm.validator import FSMValidator
from llm_fsm.utilities import extract_json_from_text, load_fsm_from_file
from llm_fsm.llm import LLMInterface
from llm_fsm.prompts import PromptBuilder
from llm_fsm.visualizer import visualize_fsm_ascii


# Test fixtures

@pytest.fixture
def valid_fsm_data():
    """Fixture for a valid FSM definition."""
    return {
        "name": "Test FSM",
        "description": "A test FSM for unit testing",
        "version": "3.0",
        "initial_state": "welcome",
        "states": {
            "welcome": {
                "id": "welcome",
                "description": "Welcome state",
                "purpose": "Welcome the user",
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
        "version": "3.0",
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
        "version": "3.0",
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
def mock_llm_interface(mocker):
    """Fixture for a mocked LLM interface using pytest-mock."""
    mock_interface = mocker.create_autospec(LLMInterface)

    # Set up the mock to return a predefined response
    mock_response = LLMResponse(
        transition=StateTransition(
            target_state="collect_name",
            context_update={"name": "John"}
        ),
        message="Hello John!",
        reasoning="User provided their name."
    )

    mock_interface.send_request.return_value = mock_response
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


def test_transition_validation(mocker):
    """Test validation of state transitions."""
    # Create an FSM instance with the necessary state and transition definitions
    fsm_def = FSMDefinition(
        name="Test FSM",
        description="Test FSM for transition validation",
        initial_state="state1",
        states={
            "state1": State(
                id="state1",
                description="State 1",
                purpose="Test state 1",
                transitions=[
                    Transition(
                        target_state="state2",
                        description="Go to state 2",
                        conditions=[
                            TransitionCondition(
                                description="Name is provided",
                                requires_context_keys=["name"]
                            )
                        ]
                    )
                ]
            ),
            "state2": State(
                id="state2",
                description="State 2",
                purpose="Test state 2",
                transitions=[]
            )
        }
    )

    instance = FSMInstance(
        fsm_id="test_fsm",
        current_state="state1"
    )

    # Create a mock loader function
    mock_loader = mocker.MagicMock(return_value=fsm_def)

    # Create an FSM manager
    fsm_manager = FSMManager(
        fsm_loader=mock_loader,
        llm_interface=mocker.MagicMock()
    )

    # Test transition validation without required context - should fail
    is_valid, _ = fsm_manager.validate_transition(instance, "state2")
    assert not is_valid

    # Add the required context and test again - should pass
    instance.context.update({"name": "John"})
    is_valid, _ = fsm_manager.validate_transition(instance, "state2")
    assert is_valid

    # Test invalid target state - should fail
    is_valid, _ = fsm_manager.validate_transition(instance, "non_existent_state")
    assert not is_valid


def test_prompt_builder():
    """Test the prompt builder."""
    # Create state and instance objects
    state = State(
        id="collect_name",
        description="Collect user's name",
        purpose="Ask for and record the user's name",
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

    instance = FSMInstance(
        fsm_id="test_fsm",
        current_state="collect_name",
        persona="A friendly assistant"
    )

    # Create a prompt builder
    prompt_builder = PromptBuilder()

    # Build a system prompt
    system_prompt = prompt_builder.build_system_prompt(instance, state)

    # Print the prompt for debugging
    print(f"SYSTEM PROMPT: {system_prompt}")

    # Verify the prompt contains key elements - with XML structure in mind
    assert "<task>" in system_prompt
    assert "</task>" in system_prompt
    assert "<fsm>" in system_prompt
    assert "</fsm>" in system_prompt
    assert "<current_state>" in system_prompt
    assert "</current_state>" in system_prompt
    assert "<available_state_transitions>" in system_prompt
    assert "</available_state_transitions>" in system_prompt
    assert "<current_context>" in system_prompt
    assert "</current_context>" in system_prompt
    assert "<response>" in system_prompt
    assert "</response>" in system_prompt

def test_fsm_manager_initialization(valid_fsm_data, mock_llm_interface, mocker):
    """Test FSM manager initialization and basic operations."""
    # Mock the FSM loader to return our valid FSM definition
    mock_loader = mocker.MagicMock()
    mock_loader.return_value = FSMDefinition(**valid_fsm_data)

    # Create an FSM manager with the mock loader and interface
    fsm_manager = FSMManager(
        fsm_loader=mock_loader,
        llm_interface=mock_llm_interface
    )

    # Test the FSM manager initialization
    assert fsm_manager.fsm_loader is not None
    assert fsm_manager.llm_interface is not None
    assert fsm_manager.prompt_builder is not None

    # Test FSM definition retrieval
    fsm_def = fsm_manager.get_fsm_definition("test_fsm")
    mock_loader.assert_called_once_with("test_fsm")
    assert fsm_def.name == "Test FSM"
    assert fsm_def.initial_state == "welcome"


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


def test_conversation_ended_detection(mock_llm_interface, valid_fsm_data, mocker):
    """Test detection of conversation end state."""
    # Mock the FSM loader to return our valid FSM definition
    mock_loader = mocker.MagicMock()
    mock_loader.return_value = FSMDefinition(**valid_fsm_data)

    # Create an FSM manager
    fsm_manager = FSMManager(
        fsm_loader=mock_loader,
        llm_interface=mock_llm_interface
    )

    # Start a conversation (mocked, no actual LLM call)
    mocker.patch.object(fsm_manager, '_process_user_input')

    # Set up the mock to simulate a conversation
    instance = FSMInstance(
        fsm_id="test_fsm",
        current_state="welcome"
    )
    fsm_manager._process_user_input.return_value = (instance, "Welcome!")

    conversation_id, _ = fsm_manager.start_conversation("test_fsm")

    # Now check if the conversation has ended (it shouldn't have, not in terminal state)
    assert not fsm_manager.is_conversation_ended(conversation_id)

    # Change the state to a terminal state (farewell has no transitions)
    fsm_manager.instances[conversation_id].current_state = "farewell"

    # Now the conversation should be detected as ended
    assert fsm_manager.is_conversation_ended(conversation_id)