"""
Robust test suite for the LLM-FSM API class.

This test file uses proper FSMDefinition objects and handles Pydantic default values
to ensure tests match real-world usage scenarios.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from llm_fsm.api import API
from llm_fsm.definitions import (
    FSMDefinition, State, Transition, TransitionCondition,
    StateTransition, LLMResponse,
)
from llm_fsm.llm import LLMInterface

# ======================================================================
# ROBUST FIXTURES WITH COMPLETE FSM DEFINITIONS
# ======================================================================


@pytest.fixture
def complete_simple_fsm():
    """Fixture for a complete simple FSM with all Pydantic defaults."""
    greeting_state = State(
        id="greeting",
        description="Greeting state",
        purpose="Greet the user and ask for their name",
        required_context_keys=["name"],
        transitions=[
            Transition(
                target_state="farewell",
                description="Move to farewell when name is collected",
                conditions=[
                    TransitionCondition(
                        description="Name has been provided",
                        requires_context_keys=["name"]
                    )
                ],
                priority=1
            )
        ]
    )

    farewell_state = State(
        id="farewell",
        description="Farewell state",
        purpose="Say goodbye to the user",
        transitions=[]  # Terminal state
    )

    return FSMDefinition(
        name="Simple Greeting FSM",
        description="A simple greeting FSM for testing",
        version="3.0",
        initial_state="greeting",
        persona="A friendly assistant",
        states={
            "greeting": greeting_state,
            "farewell": farewell_state
        }
    )

@pytest.fixture
def complete_simple_fsm_dict():
    """Fixture for simple FSM as a complete dictionary with all required fields."""
    return {
        "name": "Simple Greeting FSM",
        "description": "A simple greeting FSM for testing",
        "version": "3.0",
        "initial_state": "greeting",
        "persona": "A friendly assistant",
        "function_handlers": [],  # Include Pydantic default
        "states": {
            "greeting": {
                "id": "greeting",
                "description": "Greeting state",
                "purpose": "Greet the user and ask for their name",
                "required_context_keys": ["name"],
                "instructions": None,  # Include Pydantic default
                "example_dialogue": None,  # Include Pydantic default
                "transitions": [
                    {
                        "target_state": "farewell",
                        "description": "Move to farewell when name is collected",
                        "conditions": [
                            {
                                "description": "Name has been provided",
                                "requires_context_keys": ["name"],
                                "logic": None  # Include Pydantic default
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
                "required_context_keys": None,  # Include Pydantic default
                "instructions": None,
                "example_dialogue": None,
                "transitions": []  # Terminal state
            }
        }
    }

@pytest.fixture
def minimal_fsm_dict():
    """Fixture for FSM with only required fields (tests Pydantic defaults)."""
    return {
        "name": "Minimal FSM",
        "description": "FSM with only required fields",
        "initial_state": "only_state",
        "states": {
            "only_state": {
                "id": "only_state",
                "description": "The only state",
                "purpose": "Single state FSM",
                "transitions": []
            }
        }
    }

@pytest.fixture
def complex_fsm():
    """Fixture for a more complex FSM definition."""
    start_state = State(
        id="start",
        description="Starting state",
        purpose="Initialize the conversation",
        transitions=[
            Transition(
                target_state="collect_info",
                description="Move to information collection",
                priority=1
            )
        ]
    )

    collect_info_state = State(
        id="collect_info",
        description="Information collection state",
        purpose="Collect user information",
        required_context_keys=["user_name", "email"],
        transitions=[
            Transition(
                target_state="process",
                description="Process the collected information",
                conditions=[
                    TransitionCondition(
                        description="All required info collected",
                        requires_context_keys=["user_name", "email"]
                    )
                ],
                priority=1
            ),
            Transition(
                target_state="error_handling",
                description="Handle errors",
                priority=2
            )
        ]
    )

    process_state = State(
        id="process",
        description="Processing state",
        purpose="Process the user information",
        transitions=[
            Transition(
                target_state="complete",
                description="Complete the process",
                priority=1
            )
        ]
    )

    error_state = State(
        id="error_handling",
        description="Error handling state",
        purpose="Handle any errors that occurred",
        transitions=[
            Transition(
                target_state="collect_info",
                description="Return to information collection",
                priority=1
            )
        ]
    )

    complete_state = State(
        id="complete",
        description="Completion state",
        purpose="Process is complete",
        transitions=[]  # Terminal state
    )

    return FSMDefinition(
        name="Complex FSM",
        description="A complex FSM for advanced testing",
        version="3.0",
        initial_state="start",
        persona="A professional consultant",
        states={
            "start": start_state,
            "collect_info": collect_info_state,
            "process": process_state,
            "error_handling": error_state,
            "complete": complete_state
        }
    )

@pytest.fixture
def mock_llm_interface():
    """Fixture for a mocked LLM interface."""
    mock_interface = Mock(spec=LLMInterface)

    # Default response for most test cases
    mock_response = LLMResponse(
        transition=StateTransition(
            target_state="farewell",
            context_update={"name": "TestUser"}
        ),
        message="Hello TestUser! Nice to meet you.",
        reasoning="User provided their name, transitioning to farewell."
    )

    mock_interface.send_request.return_value = mock_response
    return mock_interface

@pytest.fixture
def temp_fsm_file(complete_simple_fsm_dict):
    """Fixture for a temporary FSM file with complete definition."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(complete_simple_fsm_dict, f, indent=2)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)

# ======================================================================
# ROBUST FSM DEFINITION PROCESSING TESTS
# ======================================================================

class TestRobustFSMDefinitionProcessing:
    """Test FSM definition processing with proper handling of Pydantic defaults."""

    def test_process_complete_fsm_dict(self, complete_simple_fsm_dict):
        """Test processing complete FSM definition dictionary."""
        fsm_def, fsm_id = API.process_fsm_definition(complete_simple_fsm_dict)

        # Verify FSMDefinition object was created correctly
        assert isinstance(fsm_def, FSMDefinition)
        assert fsm_def.name == "Simple Greeting FSM"
        assert fsm_def.initial_state == "greeting"
        assert fsm_def.version == "3.0"
        assert fsm_def.persona == "A friendly assistant"
        assert len(fsm_def.states) == 2

        # Verify Pydantic filled in defaults correctly
        assert fsm_def.function_handlers == []  # Should be default empty list

        # Verify states have proper structure
        greeting_state = fsm_def.states["greeting"]
        assert greeting_state.id == "greeting"
        assert greeting_state.required_context_keys == ["name"]
        assert greeting_state.instructions is None  # Pydantic default
        assert len(greeting_state.transitions) == 1

        # Verify transition structure
        transition = greeting_state.transitions[0]
        assert transition.target_state == "farewell"
        assert transition.priority == 1
        assert len(transition.conditions) == 1

        # Verify condition structure
        condition = transition.conditions[0]
        assert condition.requires_context_keys == ["name"]
        assert condition.logic is None  # Pydantic default

        # Verify ID generation
        assert isinstance(fsm_id, str)
        assert "fsm_dict_" in fsm_id
        assert len(fsm_id) > 10

    def test_process_minimal_fsm_dict(self, minimal_fsm_dict):
        """Test processing FSM dict with only required fields (tests Pydantic defaults)."""
        fsm_def, fsm_id = API.process_fsm_definition(minimal_fsm_dict)

        # Verify FSMDefinition was created
        assert isinstance(fsm_def, FSMDefinition)
        assert fsm_def.name == "Minimal FSM"
        assert fsm_def.initial_state == "only_state"

        # Verify Pydantic filled in defaults
        assert fsm_def.version == "3.0"  # Default version
        assert fsm_def.persona is None  # Default None
        assert fsm_def.function_handlers == []  # Default empty list

        # Verify state defaults
        state = fsm_def.states["only_state"]
        assert state.required_context_keys is None  # Default None
        assert state.instructions is None  # Default None
        assert state.example_dialogue is None  # Default None
        assert state.transitions == []  # Empty list

        # Verify ID generation
        assert isinstance(fsm_id, str)
        assert "fsm_dict_" in fsm_id

    def test_process_fsm_definition_object(self, complete_simple_fsm):
        """Test processing existing FSMDefinition object."""
        fsm_def, fsm_id = API.process_fsm_definition(complete_simple_fsm)

        # Should return the same object
        assert fsm_def is complete_simple_fsm
        assert fsm_def.name == "Simple Greeting FSM"

        # Verify ID generation for objects
        assert isinstance(fsm_id, str)
        assert "fsm_def_" in fsm_id

    def test_process_fsm_from_file(self, temp_fsm_file):
        """Test processing FSM from file with proper validation."""
        fsm_def, fsm_id = API.process_fsm_definition(temp_fsm_file)

        # Should load and create valid FSMDefinition
        assert isinstance(fsm_def, FSMDefinition)
        assert fsm_def.name == "Simple Greeting FSM"
        assert fsm_def.initial_state == "greeting"

        # Verify all Pydantic validation occurred
        assert len(fsm_def.states) == 2
        assert "greeting" in fsm_def.states
        assert "farewell" in fsm_def.states

        # Verify file-based ID
        assert isinstance(fsm_id, str)
        assert "fsm_file_" in fsm_id
        assert temp_fsm_file in fsm_id

    def test_invalid_fsm_dict_validation(self):
        """Test that invalid FSM dictionaries are properly rejected."""
        # Missing required fields
        invalid_fsm1 = {
            "name": "Invalid FSM"
            # Missing description, initial_state, states
        }

        with pytest.raises(ValueError, match="Invalid FSM definition dictionary"):
            API.process_fsm_definition(invalid_fsm1)

        # Invalid state structure
        invalid_fsm2 = {
            "name": "Invalid FSM",
            "description": "Invalid state structure",
            "initial_state": "bad_state",
            "states": {
                "bad_state": {
                    "id": "bad_state",
                    # Missing required purpose field
                    "description": "Bad state"
                }
            }
        }

        with pytest.raises(ValueError, match="Invalid FSM definition dictionary"):
            API.process_fsm_definition(invalid_fsm2)

# ======================================================================
# ROBUST INITIALIZATION TESTS
# ======================================================================

class TestRobustInitialization:
    """Test API initialization with proper FSM definition handling."""

    def test_init_with_fsm_definition_object(self, complete_simple_fsm, mock_llm_interface):
        """Test initialization with FSMDefinition object."""
        api = API(
            fsm_definition=complete_simple_fsm,
            llm_interface=mock_llm_interface
        )

        # Verify the FSM definition is stored correctly
        assert api.fsm_definition is complete_simple_fsm
        assert api.fsm_definition.name == "Simple Greeting FSM"

        # Verify FSM ID was generated
        assert isinstance(api.fsm_id, str)
        assert "fsm_def_" in api.fsm_id

    def test_init_with_complete_dict(self, complete_simple_fsm_dict, mock_llm_interface):
        """Test initialization with complete FSM dictionary."""
        api = API(
            fsm_definition=complete_simple_fsm_dict,
            llm_interface=mock_llm_interface
        )

        # Verify FSMDefinition was created from dict
        assert isinstance(api.fsm_definition, FSMDefinition)
        assert api.fsm_definition.name == "Simple Greeting FSM"
        assert api.fsm_definition.persona == "A friendly assistant"

        # Verify all Pydantic defaults were applied
        assert api.fsm_definition.function_handlers == []
        assert api.fsm_definition.version == "3.0"

    def test_init_with_minimal_dict(self, minimal_fsm_dict, mock_llm_interface):
        """Test initialization with minimal FSM dictionary."""
        api = API(
            fsm_definition=minimal_fsm_dict,
            llm_interface=mock_llm_interface
        )

        # Verify FSMDefinition was created with defaults
        assert isinstance(api.fsm_definition, FSMDefinition)
        assert api.fsm_definition.name == "Minimal FSM"
        assert api.fsm_definition.version == "3.0"  # Pydantic default
        assert api.fsm_definition.persona is None   # Pydantic default
        assert api.fsm_definition.function_handlers == []  # Pydantic default

    def test_fsm_validation_during_init(self):
        """Test that FSM validation occurs during initialization."""
        # Invalid FSM should be caught during initialization
        invalid_fsm = {
            "name": "Invalid FSM",
            "description": "This FSM will fail validation",
            "initial_state": "nonexistent_state",  # State doesn't exist
            "states": {
                "real_state": {
                    "id": "real_state",
                    "description": "A real state",
                    "purpose": "Test state",
                    "transitions": []
                }
            }
        }

        # Should raise error during FSM processing
        with pytest.raises(ValueError):
            API(fsm_definition=invalid_fsm, model="gpt-4")

# ======================================================================
# ROBUST CONVERSATION TESTS
# ======================================================================

class TestRobustConversations:
    """Test conversations with properly validated FSM definitions."""

    def test_conversation_with_complete_fsm(self, complete_simple_fsm, mock_llm_interface):
        """Test complete conversation flow with validated FSM."""
        api = API(
            fsm_definition=complete_simple_fsm,
            llm_interface=mock_llm_interface
        )

        # Start conversation
        conv_id, response = api.start_conversation()

        # Verify conversation started correctly
        assert isinstance(conv_id, str)
        assert isinstance(response, str)
        assert conv_id in api.active_conversations

        # Verify initial state matches FSM definition
        current_state = api.get_current_state(conv_id)
        assert current_state == complete_simple_fsm.initial_state

        # Verify stack was created correctly
        assert api.get_stack_depth(conv_id) == 1
        stack_frame = api.conversation_stacks[conv_id][0]
        assert stack_frame.fsm_definition is complete_simple_fsm

    def test_conversation_with_fsm_dict(self, complete_simple_fsm_dict, mock_llm_interface):
        """Test conversation with FSM dictionary (converted to FSMDefinition)."""
        api = API(
            fsm_definition=complete_simple_fsm_dict,
            llm_interface=mock_llm_interface
        )

        conv_id, response = api.start_conversation()

        # Verify FSMDefinition was created and used
        assert isinstance(api.fsm_definition, FSMDefinition)
        assert api.fsm_definition.name == "Simple Greeting FSM"

        # Verify conversation works with converted definition
        current_state = api.get_current_state(conv_id)
        assert current_state == api.fsm_definition.initial_state

    def test_context_handling_with_validated_fsm(self, complete_simple_fsm, mock_llm_interface):
        """Test context handling with properly validated FSM."""
        initial_context = {
            "user_id": "test_user",
            "session_data": {"type": "test"},
            "complex_data": {
                "nested": {"value": 42},
                "list": [1, 2, 3]
            }
        }

        api = API(
            fsm_definition=complete_simple_fsm,
            llm_interface=mock_llm_interface
        )

        conv_id, _ = api.start_conversation(initial_context=initial_context)

        # Verify context was preserved correctly
        data = api.get_data(conv_id)
        assert data["user_id"] == "test_user"
        assert data["session_data"]["type"] == "test"
        assert data["complex_data"]["nested"]["value"] == 42
        assert data["complex_data"]["list"] == [1, 2, 3]

    def test_fsm_loader_functionality(self, complete_simple_fsm, mock_llm_interface):
        """Test that the custom FSM loader works correctly."""
        api = API(
            fsm_definition=complete_simple_fsm,
            llm_interface=mock_llm_interface
        )

        # The FSM manager should be able to load our FSM by ID
        loaded_fsm = api.fsm_manager.fsm_loader(api.fsm_id)
        assert loaded_fsm is complete_simple_fsm
        assert loaded_fsm.name == "Simple Greeting FSM"

# ======================================================================
# COMPARISON TESTS
# ======================================================================

class TestFSMDefinitionEquivalence:
    """Test that different input formats produce equivalent results."""

    def test_dict_vs_object_equivalence(self, complete_simple_fsm_dict, mock_llm_interface):
        """Test that FSM dict and FSMDefinition object produce equivalent APIs."""
        # Create FSMDefinition from dict
        from llm_fsm.definitions import FSMDefinition
        fsm_obj = FSMDefinition(**complete_simple_fsm_dict)

        # Create APIs with both formats
        api_from_dict = API(
            fsm_definition=complete_simple_fsm_dict,
            llm_interface=mock_llm_interface
        )

        api_from_obj = API(
            fsm_definition=fsm_obj,
            llm_interface=mock_llm_interface
        )

        # Both should have equivalent FSM definitions
        assert api_from_dict.fsm_definition.name == api_from_obj.fsm_definition.name
        assert api_from_dict.fsm_definition.initial_state == api_from_obj.fsm_definition.initial_state
        assert len(api_from_dict.fsm_definition.states) == len(api_from_obj.fsm_definition.states)

        # Both should produce equivalent conversations
        conv_id1, response1 = api_from_dict.start_conversation({"test": "data"})
        conv_id2, response2 = api_from_obj.start_conversation({"test": "data"})

        # States should be equivalent
        assert api_from_dict.get_current_state(conv_id1) == api_from_obj.get_current_state(conv_id2)

    def test_minimal_vs_complete_dict_behavior(self, minimal_fsm_dict, complete_simple_fsm_dict, mock_llm_interface):
        """Test behavior differences between minimal and complete FSM dicts."""
        api_minimal = API(
            fsm_definition=minimal_fsm_dict,
            llm_interface=mock_llm_interface
        )

        api_complete = API(
            fsm_definition=complete_simple_fsm_dict,
            llm_interface=mock_llm_interface
        )

        # Both should work but have different characteristics
        assert api_minimal.fsm_definition.persona is None
        assert api_complete.fsm_definition.persona == "A friendly assistant"

        # Both should start conversations successfully
        conv_id1, _ = api_minimal.start_conversation()
        conv_id2, _ = api_complete.start_conversation()

        assert conv_id1 in api_minimal.active_conversations
        assert conv_id2 in api_complete.active_conversations

# ======================================================================
# EDGE CASE TESTS
# ======================================================================

class TestRobustEdgeCases:
    """Test edge cases with proper FSM validation."""

    def test_fsm_with_no_transitions(self, mock_llm_interface):
        """Test FSM with terminal state only."""
        terminal_only_fsm = FSMDefinition(
            name="Terminal Only FSM",
            description="FSM with only terminal state",
            initial_state="terminal",
            states={
                "terminal": State(
                    id="terminal",
                    description="Terminal state",
                    purpose="Immediately terminal",
                    transitions=[]
                )
            }
        )

        api = API(
            fsm_definition=terminal_only_fsm,
            llm_interface=mock_llm_interface
        )

        conv_id, _ = api.start_conversation()

        # Should immediately be in terminal state
        assert api.has_conversation_ended(conv_id)
        assert api.get_current_state(conv_id) == "terminal"

    def test_fsm_with_complex_transitions(self, mock_llm_interface):
        """Test FSM with complex transition conditions."""
        complex_state = State(
            id="complex",
            description="State with complex transitions",
            purpose="Test complex conditions",
            required_context_keys=["key1", "key2"],
            transitions=[
                Transition(
                    target_state="target1",
                    description="Transition with multiple conditions",
                    conditions=[
                        TransitionCondition(
                            description="First condition",
                            requires_context_keys=["key1"]
                        ),
                        TransitionCondition(
                            description="Second condition",
                            requires_context_keys=["key2"]
                        )
                    ],
                    priority=1
                ),
                Transition(
                    target_state="target2",
                    description="Fallback transition",
                    priority=2
                )
            ]
        )

        target1_state = State(
            id="target1",
            description="First target",
            purpose="Target 1",
            transitions=[]
        )

        target2_state = State(
            id="target2",
            description="Second target",
            purpose="Target 2",
            transitions=[]
        )

        complex_fsm = FSMDefinition(
            name="Complex Transition FSM",
            description="FSM with complex transitions",
            initial_state="complex",
            states={
                "complex": complex_state,
                "target1": target1_state,
                "target2": target2_state
            }
        )

        api = API(
            fsm_definition=complex_fsm,
            llm_interface=mock_llm_interface
        )

        conv_id, _ = api.start_conversation()

        # Should start in complex state
        assert api.get_current_state(conv_id) == "complex"

        # Verify FSM structure is properly loaded
        assert len(api.fsm_definition.states) == 3
        complex_loaded = api.fsm_definition.states["complex"]
        assert len(complex_loaded.transitions) == 2
        assert complex_loaded.required_context_keys == ["key1", "key2"]



