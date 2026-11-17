"""
Robust test suite for the FSM-LLM API class with enhanced 2-pass architecture.

This test file uses proper FSMDefinition objects and handles Pydantic default values
to ensure tests match real-world usage scenarios with the new architecture.
"""

from __future__ import annotations

import inspect
import json
import tempfile
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from fsm_llm import APIBuilder
from fsm_llm.core.definitions import (
    ConversationBusyError,
    FSMDefinition,
    FSMError,
    ResponseGenerationResponse,
    State,
    StateNotFoundError,
    Transition,
    TransitionCondition,
)
from fsm_llm.core.definitions.schema import SCHEMA_VERSION_DEFAULT
from fsm_llm.core.transition_evaluator import TransitionEvaluatorConfig
from fsm_llm.engine.api.conversation import API
from fsm_llm.engine.fsm.manager import FSMManager
from fsm_llm.engine.kinds import ConversationExecutor
from fsm_llm.handlers import HandlerErrorMode, HandlerTiming, create_handler
from fsm_llm.llm import LiteLLMInterface, LLMInterface
from fsm_llm.logging import logger
from fsm_llm.testing import MockLLM2Interface
from tests.conftest import EveryReadLater, bind_api_clock, current_timing_factor
from tests.test_fsm_llm.support import (
    _JOIN_TIMEOUT_S,
    _TWO_STATE,
    _a2026_09_22_api,
    _blocked_fsm,
    _c3_fsm,
    _join_all,
    _mock_llm,
    _raw_data,
    _reentry_fsm,
    _Session,
)
from tests.test_fsm_llm.test_completion_state import _tool_fsm
from tests.test_fsm_llm.test_llm_complete import _Provider, _reply

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
        extraction_instructions="Extract the user's name from their input",
        response_instructions="Greet the user warmly and ask for their name if not provided",
        transitions=[
            Transition(
                target_state="farewell",
                description="Move to farewell when name is collected",
                conditions=[
                    TransitionCondition(
                        description="Name has been provided",
                        requires_context_keys=["name"],
                    )
                ],
                priority=1,
                llm_description="User has provided their name",
            )
        ],
    )

    farewell_state = State(
        id="farewell",
        description="Farewell state",
        purpose="Say goodbye to the user",
        response_instructions="Say a personalized goodbye using the user's name",
        transitions=[],  # Terminal state
    )

    return FSMDefinition(
        name="Simple Greeting FSM",
        description="A simple greeting FSM for testing",
        version="4.0",
        initial_state="greeting",
        persona="A friendly assistant who loves meeting new people",
        states={"greeting": greeting_state, "farewell": farewell_state},
    )


@pytest.fixture
def complete_simple_fsm_dict():
    """Fixture for simple FSM as a complete dictionary with all required fields."""
    return {
        "name": "Simple Greeting FSM",
        "description": "A simple greeting FSM for testing",
        "version": "4.0",
        "initial_state": "greeting",
        "persona": "A friendly assistant who loves meeting new people",
        "states": {
            "greeting": {
                "id": "greeting",
                "description": "Greeting state",
                "purpose": "Greet the user and ask for their name",
                "required_context_keys": ["name"],
                "extraction_instructions": "Extract the user's name from their input",
                "response_instructions": "Greet the user warmly and ask for their name if not provided",
                "transitions": [
                    {
                        "target_state": "farewell",
                        "description": "Move to farewell when name is collected",
                        "conditions": [
                            {
                                "description": "Name has been provided",
                                "requires_context_keys": ["name"],
                                "logic": None,
                                "evaluation_priority": 100,
                            }
                        ],
                        "priority": 1,
                        "llm_description": "User has provided their name",
                    }
                ],
            },
            "farewell": {
                "id": "farewell",
                "description": "Farewell state",
                "purpose": "Say goodbye to the user",
                "required_context_keys": None,
                "extraction_instructions": None,
                "response_instructions": "Say a personalized goodbye using the user's name",
                "transitions": [],  # Terminal state
            },
        },
    }


@pytest.fixture
def complex_fsm():
    """Fixture for a more complex FSM definition."""
    start_state = State(
        id="start",
        description="Starting state",
        purpose="Initialize the conversation",
        response_instructions="Welcome the user and explain the process",
        transitions=[
            Transition(
                target_state="collect_info",
                description="Move to information collection",
                priority=1,
            )
        ],
    )

    collect_info_state = State(
        id="collect_info",
        description="Information collection state",
        purpose="Collect user information",
        required_context_keys=["user_name", "email"],
        extraction_instructions="Extract the user's name and email address",
        response_instructions="Ask for missing information politely",
        transitions=[
            Transition(
                target_state="process",
                description="Process the collected information",
                conditions=[
                    TransitionCondition(
                        description="All required info collected",
                        requires_context_keys=["user_name", "email"],
                    )
                ],
                priority=1,
            ),
            Transition(
                target_state="error_handling", description="Handle errors", priority=2
            ),
        ],
    )

    process_state = State(
        id="process",
        description="Processing state",
        purpose="Process the user information",
        response_instructions="Confirm processing and provide status updates",
        transitions=[
            Transition(
                target_state="complete", description="Complete the process", priority=1
            )
        ],
    )

    error_state = State(
        id="error_handling",
        description="Error handling state",
        purpose="Handle any errors that occurred",
        response_instructions="Apologize for the error and offer help",
        transitions=[
            Transition(
                target_state="collect_info",
                description="Return to information collection",
                priority=1,
            )
        ],
    )

    complete_state = State(
        id="complete",
        description="Completion state",
        purpose="Process is complete",
        response_instructions="Thank the user and summarize what was accomplished",
        transitions=[],  # Terminal state
    )

    return FSMDefinition(
        name="Complex FSM",
        description="A complex FSM for advanced testing",
        version="4.0",
        initial_state="start",
        persona="A professional consultant who guides users through complex processes",
        states={
            "start": start_state,
            "collect_info": collect_info_state,
            "process": process_state,
            "error_handling": error_state,
            "complete": complete_state,
        },
    )


@pytest.fixture
def mock_llm_interface():
    """Fixture for a mocked LLM interface using the new 2-pass architecture."""
    from fsm_llm.core.definitions import FieldExtractionResponse

    mock_interface = Mock(spec=LLMInterface)

    # Mock response generation response
    mock_response_generation = ResponseGenerationResponse(
        message="Hello TestUser! Nice to meet you. What can I help you with today?",
        reasoning="Generated greeting using extracted name",
    )

    # Mock field extraction (new primary path)
    mock_data = {"name": "TestUser"}

    def _mock_extract_field(request):
        value = mock_data.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=0.95 if value is not None else 0.0,
            reasoning="Mock field extraction",
            is_valid=value is not None,
        )

    mock_interface.extract_field.side_effect = _mock_extract_field

    # Set up method returns
    mock_interface.generate_response.return_value = mock_response_generation

    return mock_interface


@pytest.fixture
def temp_fsm_file(complete_simple_fsm_dict):
    """Fixture for a temporary FSM file with complete definition."""
    with tempfile.NamedTemporaryFile(
        encoding="utf-8", mode="w", suffix=".json", delete=False
    ) as f:
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
        assert fsm_def.version == "4.0"
        assert fsm_def.persona == "A friendly assistant who loves meeting new people"
        assert len(fsm_def.states) == 2

        # Verify states have proper structure
        greeting_state = fsm_def.states["greeting"]
        assert greeting_state.id == "greeting"
        assert greeting_state.required_context_keys == ["name"]
        assert (
            greeting_state.extraction_instructions
            == "Extract the user's name from their input"
        )
        assert (
            greeting_state.response_instructions
            == "Greet the user warmly and ask for their name if not provided"
        )
        assert len(greeting_state.transitions) == 1

        # Verify transition structure
        transition = greeting_state.transitions[0]
        assert transition.target_state == "farewell"
        assert transition.priority == 1
        assert transition.llm_description == "User has provided their name"
        assert len(transition.conditions) == 1

        # Verify condition structure
        condition = transition.conditions[0]
        assert condition.requires_context_keys == ["name"]
        assert condition.logic is None  # Pydantic default
        assert condition.evaluation_priority == 100  # Pydantic default

        # Verify ID generation -- content-hash derived (D-016), no
        # construction-path-specific prefix.
        assert isinstance(fsm_id, str)
        assert fsm_id.startswith("fsm_")
        assert len(fsm_id) > 10

    def test_process_minimal_fsm_dict(self, minimal_fsm_dict):
        """Test processing FSM dict with only required fields (tests Pydantic defaults)."""
        fsm_def, fsm_id = API.process_fsm_definition(minimal_fsm_dict)

        # Verify FSMDefinition was created
        assert isinstance(fsm_def, FSMDefinition)
        assert fsm_def.name == "Minimal FSM"
        assert fsm_def.initial_state == "only_state"

        # Verify Pydantic filled in defaults
        assert fsm_def.version == SCHEMA_VERSION_DEFAULT  # Default version
        assert fsm_def.persona is None  # Default None

        # Verify state defaults
        state = fsm_def.states["only_state"]
        assert state.required_context_keys is None  # Default None
        assert state.extraction_instructions is None  # Default None
        assert state.response_instructions is None  # Default None
        assert state.transitions == []  # Empty list

        # Verify ID generation -- content-hash derived (D-016), no
        # construction-path-specific prefix.
        assert isinstance(fsm_id, str)
        assert fsm_id.startswith("fsm_")

    def test_process_fsm_definition_object(self, complete_simple_fsm):
        """Test processing existing FSMDefinition object."""
        fsm_def, fsm_id = API.process_fsm_definition(complete_simple_fsm)

        # Should return the same object
        assert fsm_def is complete_simple_fsm
        assert fsm_def.name == "Simple Greeting FSM"

        # Verify ID generation for objects -- content-hash derived (D-016),
        # no construction-path-specific prefix.
        assert isinstance(fsm_id, str)
        assert fsm_id.startswith("fsm_")

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

        # DECISION plan-2026-09-20T114608-a8e47b88/D-016: the file-path ID
        # is content-hash derived, NOT the path string itself -- the path
        # must NOT leak into fsm_id (that was CRITICAL 1 in
        # review-iter-2.md: a file overwritten in place with a different
        # FSM used to keep the SAME id since the id was `fsm_file_{path}`).
        assert isinstance(fsm_id, str)
        assert fsm_id.startswith("fsm_")
        assert temp_fsm_file not in fsm_id

    def test_process_fsm_from_file_content_hash_survives_overwrite(
        self, complete_simple_fsm_dict
    ):
        """D-016 / review-iter-2.md CRITICAL 1 regression: a file loaded,
        then overwritten in place with a semantically different FSM, must
        NOT keep the old fsm_id -- the id has to track the file's CONTENT,
        not its path.
        """
        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(complete_simple_fsm_dict, f, indent=2)
            temp_path = f.name

        try:
            _, fsm_id_alpha = API.process_fsm_definition(temp_path)

            omega_dict = dict(complete_simple_fsm_dict)
            omega_dict["name"] = "Omega Public Chatbot"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(omega_dict, f, indent=2)

            _, fsm_id_omega = API.process_fsm_definition(temp_path)

            assert fsm_id_alpha != fsm_id_omega
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_process_fsm_definition_same_content_different_construction_paths(
        self, complete_simple_fsm_dict, complete_simple_fsm
    ):
        """D-016 / review-iter-2.md WARNING 2 regression: the SAME FSM
        content must yield the SAME fsm_id whether constructed via the raw
        dict, an FSMDefinition object built from that dict, or a file
        written with that dict's JSON -- previously each construction path
        hashed a different input (raw dict / model_dump() / path string)
        under a different prefix, producing spurious mismatch warnings on
        restore for what is actually the identical FSM.
        """
        _, fsm_id_from_dict = API.process_fsm_definition(complete_simple_fsm_dict)

        fsm_def_from_dict = FSMDefinition(**complete_simple_fsm_dict)
        _, fsm_id_from_object = API.process_fsm_definition(fsm_def_from_dict)

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(complete_simple_fsm_dict, f, indent=2)
            temp_path = f.name
        try:
            _, fsm_id_from_file = API.process_fsm_definition(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

        assert fsm_id_from_dict == fsm_id_from_object == fsm_id_from_file

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
                    "description": "Bad state",
                }
            },
        }

        with pytest.raises(ValueError, match="Invalid FSM definition dictionary"):
            API.process_fsm_definition(invalid_fsm2)


# ======================================================================
# ROBUST INITIALIZATION TESTS
# ======================================================================


class TestRobustInitialization:
    """Test API initialization with proper FSM definition handling."""

    def test_init_with_fsm_definition_object(
        self, complete_simple_fsm, mock_llm_interface
    ):
        """Test initialization with FSMDefinition object."""
        api = API(fsm_definition=complete_simple_fsm, llm_interface=mock_llm_interface)

        # Verify the FSM definition is stored correctly
        assert api.fsm_definition is complete_simple_fsm
        assert api.fsm_definition.name == "Simple Greeting FSM"

        # Verify FSM ID was generated -- content-hash derived (D-016)
        assert isinstance(api.fsm_id, str)
        assert api.fsm_id.startswith("fsm_")

    def test_init_with_complete_dict(
        self, complete_simple_fsm_dict, mock_llm_interface
    ):
        """Test initialization with complete FSM dictionary."""
        api = API(
            fsm_definition=complete_simple_fsm_dict, llm_interface=mock_llm_interface
        )

        # Verify FSMDefinition was created from dict
        assert isinstance(api.fsm_definition, FSMDefinition)
        assert api.fsm_definition.name == "Simple Greeting FSM"
        assert (
            api.fsm_definition.persona
            == "A friendly assistant who loves meeting new people"
        )

        # Verify all fields were properly set
        assert api.fsm_definition.version == "4.0"

    def test_init_with_minimal_dict(self, minimal_fsm_dict, mock_llm_interface):
        """Test initialization with minimal FSM dictionary."""
        api = API(fsm_definition=minimal_fsm_dict, llm_interface=mock_llm_interface)

        # Verify FSMDefinition was created with defaults
        assert isinstance(api.fsm_definition, FSMDefinition)
        assert api.fsm_definition.name == "Minimal FSM"
        assert api.fsm_definition.version == SCHEMA_VERSION_DEFAULT  # Pydantic default
        assert api.fsm_definition.persona is None  # Pydantic default

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
                    "transitions": [],
                }
            },
        }

        # Should raise error during FSM processing
        with pytest.raises(ValueError):
            API(fsm_definition=invalid_fsm, model="gpt-4")

    def test_init_with_default_llm_interface(self, complete_simple_fsm):
        """Test initialization with default LiteLLM interface."""
        api = API(
            fsm_definition=complete_simple_fsm,
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1500,
        )

        # Verify LLM interface was created
        assert api.llm_interface is not None
        assert hasattr(api.llm_interface, "generate_response")


# ======================================================================
# ROBUST CONVERSATION TESTS
# ======================================================================


class TestRobustConversations:
    """Test conversations with properly validated FSM definitions."""

    def test_conversation_with_complete_fsm(
        self, complete_simple_fsm, mock_llm_interface
    ):
        """Test complete conversation flow with validated FSM."""
        api = API(fsm_definition=complete_simple_fsm, llm_interface=mock_llm_interface)

        # Start conversation
        conv_id, response = api.start_conversation()

        # Verify conversation started correctly
        assert isinstance(conv_id, str)
        assert isinstance(response, str)
        assert conv_id in api.conversation_stacks

        # Verify initial state matches FSM definition
        current_state = api.get_current_state(conv_id)
        assert current_state == complete_simple_fsm.initial_state

        # Verify stack was created correctly
        assert len(api.conversation_stacks[conv_id]) == 1
        stack_frame = api.conversation_stacks[conv_id][0]
        assert stack_frame.fsm_definition is complete_simple_fsm

    def test_conversation_with_fsm_dict(
        self, complete_simple_fsm_dict, mock_llm_interface
    ):
        """Test conversation with FSM dictionary (converted to FSMDefinition)."""
        api = API(
            fsm_definition=complete_simple_fsm_dict, llm_interface=mock_llm_interface
        )

        conv_id, _response = api.start_conversation()

        # Verify FSMDefinition was created and used
        assert isinstance(api.fsm_definition, FSMDefinition)
        assert api.fsm_definition.name == "Simple Greeting FSM"

        # Verify conversation works with converted definition
        current_state = api.get_current_state(conv_id)
        assert current_state == api.fsm_definition.initial_state

    def test_context_handling_with_validated_fsm(
        self, complete_simple_fsm, mock_llm_interface
    ):
        """Test context handling with properly validated FSM."""
        initial_context = {
            "user_id": "test_user",
            "session_data": {"type": "test"},
            "complex_data": {"nested": {"value": 42}, "list": [1, 2, 3]},
        }

        api = API(fsm_definition=complete_simple_fsm, llm_interface=mock_llm_interface)

        conv_id, _ = api.start_conversation(initial_context=initial_context)

        # Verify context was preserved correctly
        data = api.get_data(conv_id)
        assert data["user_id"] == "test_user"
        assert data["session_data"]["type"] == "test"
        assert data["complex_data"]["nested"]["value"] == 42
        assert data["complex_data"]["list"] == [1, 2, 3]

    def test_fsm_loader_functionality(self, complete_simple_fsm, mock_llm_interface):
        """Test that the custom FSM loader works correctly."""
        api = API(fsm_definition=complete_simple_fsm, llm_interface=mock_llm_interface)

        # The FSM manager should be able to load our FSM by ID
        loaded_fsm = api.fsm_manager.fsm_loader(api.fsm_id)
        assert loaded_fsm is complete_simple_fsm
        assert loaded_fsm.name == "Simple Greeting FSM"

    def test_message_processing_with_2pass_architecture(
        self, complete_simple_fsm, mock_llm_interface
    ):
        """Test message processing with the new 2-pass architecture."""
        api = API(fsm_definition=complete_simple_fsm, llm_interface=mock_llm_interface)

        conv_id, _ = api.start_conversation()

        # Process a user message
        response = api.converse("Hi, my name is Alice", conv_id)

        # Verify LLM interface methods were called
        assert mock_llm_interface.extract_field.called
        assert mock_llm_interface.generate_response.called

        # Verify response is a string
        assert isinstance(response, str)

        # Verify context was updated (name should be extracted)
        data = api.get_data(conv_id)
        assert "name" in data
        assert data["name"] == "TestUser"  # From mock response


# ======================================================================
# FSM STACKING TESTS
# ======================================================================


class TestFSMStacking:
    """Test FSM stacking functionality with new architecture."""

    def test_push_fsm_with_complete_definition(
        self, complete_simple_fsm, complex_fsm, mock_llm_interface
    ):
        """Test pushing a new FSM onto the conversation stack."""
        api = API(fsm_definition=complete_simple_fsm, llm_interface=mock_llm_interface)

        conv_id, _ = api.start_conversation()
        api.get_current_state(conv_id)  # verify state is accessible

        # Push a new FSM
        response = api.push_fsm(
            conversation_id=conv_id,
            new_fsm_definition=complex_fsm,
            context_to_pass={"user_id": "test123"},
            preserve_history=True,
        )

        # Verify FSM was pushed
        assert isinstance(response, str)
        assert len(api.conversation_stacks[conv_id]) == 2

        # Verify new FSM is active
        current_state = api.get_current_state(conv_id)
        assert current_state == complex_fsm.initial_state

    def test_pop_fsm_with_context_merge(
        self, complete_simple_fsm, complex_fsm, mock_llm_interface
    ):
        """Test popping FSM and merging context."""
        api = API(fsm_definition=complete_simple_fsm, llm_interface=mock_llm_interface)

        conv_id, _ = api.start_conversation()

        # Push a new FSM
        api.push_fsm(
            conversation_id=conv_id,
            new_fsm_definition=complex_fsm,
            shared_context_keys=["user_data"],
        )

        # Update context in pushed FSM
        api.fsm_manager.update_conversation_context(
            api._get_current_fsm_conversation_id(conv_id),
            {"user_data": "important_value", "temp_data": "temporary"},
        )

        # Pop the FSM
        response = api.pop_fsm(
            conversation_id=conv_id,
            context_to_return={"result": "success"},
            merge_strategy="update",
        )

        # Verify pop occurred
        assert isinstance(response, str)
        assert len(api.conversation_stacks[conv_id]) == 1

        # Verify context was merged
        data = api.get_data(conv_id)
        assert data["result"] == "success"


# ======================================================================
# COMPARISON TESTS
# ======================================================================


class TestFSMDefinitionEquivalence:
    """Test that different input formats produce equivalent results."""

    def test_dict_vs_object_equivalence(
        self, complete_simple_fsm_dict, mock_llm_interface
    ):
        """Test that FSM dict and FSMDefinition object produce equivalent APIs."""
        # Create FSMDefinition from dict
        from fsm_llm.core.definitions import FSMDefinition

        fsm_obj = FSMDefinition(**complete_simple_fsm_dict)

        # Create APIs with both formats
        api_from_dict = API(
            fsm_definition=complete_simple_fsm_dict, llm_interface=mock_llm_interface
        )

        api_from_obj = API(fsm_definition=fsm_obj, llm_interface=mock_llm_interface)

        # Both should have equivalent FSM definitions
        assert api_from_dict.fsm_definition.name == api_from_obj.fsm_definition.name
        assert (
            api_from_dict.fsm_definition.initial_state
            == api_from_obj.fsm_definition.initial_state
        )
        assert len(api_from_dict.fsm_definition.states) == len(
            api_from_obj.fsm_definition.states
        )

        # Both should produce equivalent conversations
        conv_id1, _response1 = api_from_dict.start_conversation({"test": "data"})
        conv_id2, _response2 = api_from_obj.start_conversation({"test": "data"})

        # States should be equivalent
        assert api_from_dict.get_current_state(
            conv_id1
        ) == api_from_obj.get_current_state(conv_id2)

    def test_minimal_vs_complete_dict_behavior(
        self, minimal_fsm_dict, complete_simple_fsm_dict, mock_llm_interface
    ):
        """Test behavior differences between minimal and complete FSM dicts."""
        api_minimal = API(
            fsm_definition=minimal_fsm_dict, llm_interface=mock_llm_interface
        )

        api_complete = API(
            fsm_definition=complete_simple_fsm_dict, llm_interface=mock_llm_interface
        )

        # Both should work but have different characteristics
        assert api_minimal.fsm_definition.persona is None
        assert (
            api_complete.fsm_definition.persona
            == "A friendly assistant who loves meeting new people"
        )

        # Both should start conversations successfully
        conv_id1, _ = api_minimal.start_conversation()
        conv_id2, _ = api_complete.start_conversation()

        assert conv_id1 in api_minimal.conversation_stacks
        assert conv_id2 in api_complete.conversation_stacks


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
                    response_instructions="Provide final message",
                    transitions=[],
                )
            },
        )

        api = API(fsm_definition=terminal_only_fsm, llm_interface=mock_llm_interface)

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
            extraction_instructions="Extract key1 and key2 from user input",
            response_instructions="Ask for missing keys",
            transitions=[
                Transition(
                    target_state="target1",
                    description="Transition with multiple conditions",
                    conditions=[
                        TransitionCondition(
                            description="First condition",
                            requires_context_keys=["key1"],
                            evaluation_priority=50,
                        ),
                        TransitionCondition(
                            description="Second condition",
                            requires_context_keys=["key2"],
                            evaluation_priority=60,
                        ),
                    ],
                    priority=1,
                ),
                Transition(
                    target_state="target2",
                    description="Fallback transition",
                    priority=2,
                ),
            ],
        )

        target1_state = State(
            id="target1",
            description="First target",
            purpose="Target 1",
            response_instructions="Confirm successful transition to target1",
            transitions=[],
        )

        target2_state = State(
            id="target2",
            description="Second target",
            purpose="Target 2",
            response_instructions="Confirm fallback to target2",
            transitions=[],
        )

        complex_fsm = FSMDefinition(
            name="Complex Transition FSM",
            description="FSM with complex transitions",
            initial_state="complex",
            states={
                "complex": complex_state,
                "target1": target1_state,
                "target2": target2_state,
            },
        )

        api = API(fsm_definition=complex_fsm, llm_interface=mock_llm_interface)

        conv_id, _ = api.start_conversation()

        # Should start in complex state
        assert api.get_current_state(conv_id) == "complex"

        # Verify FSM structure is properly loaded
        assert len(api.fsm_definition.states) == 3
        complex_loaded = api.fsm_definition.states["complex"]
        assert len(complex_loaded.transitions) == 2
        assert complex_loaded.required_context_keys == ["key1", "key2"]

        # Verify transition evaluation mode

    def test_conversation_termination_handling(
        self, complete_simple_fsm, mock_llm_interface
    ):
        """Test proper handling of conversation termination."""
        api = API(fsm_definition=complete_simple_fsm, llm_interface=mock_llm_interface)

        conv_id, _ = api.start_conversation()

        # Verify conversation is not ended initially
        assert not api.has_conversation_ended(conv_id)

        # End conversation
        api.end_conversation(conv_id)

        # Verify conversation was properly cleaned up from active_conversations
        assert conv_id not in api.conversation_stacks

        # Verify conversation_stacks is cleaned up (was a known bug, now fixed)
        assert conv_id not in api.conversation_stacks

    def test_handler_system_integration(self, complete_simple_fsm, mock_llm_interface):
        """Test integration with the handler system."""
        # Create a simple handler
        from fsm_llm.handlers import HandlerTiming, create_handler

        test_handler = (
            create_handler("test_handler")
            .at(HandlerTiming.POST_TRANSITION)
            .on_target_state("farewell")
            .do(lambda ctx: {"handler_executed": True})
        )

        api = API(
            fsm_definition=complete_simple_fsm,
            llm_interface=mock_llm_interface,
            handlers=[test_handler],
        )

        _conv_id, _ = api.start_conversation()

        # Verify handler was registered
        assert len(api.handler_system.handlers) == 1
        assert api.handler_system.handlers[0].name == "test_handler"


# ======================================================================
# PERFORMANCE AND INTEGRATION TESTS
# ======================================================================


class TestPerformanceAndIntegration:
    """Test performance characteristics and integration points."""

    def test_multiple_concurrent_conversations(
        self, complete_simple_fsm, mock_llm_interface
    ):
        """Test handling multiple concurrent conversations."""
        api = API(fsm_definition=complete_simple_fsm, llm_interface=mock_llm_interface)

        # Start multiple conversations
        conversations = []
        for i in range(5):
            conv_id, _ = api.start_conversation({"user_id": f"user_{i}"})
            conversations.append(conv_id)

        # Verify all conversations are active
        assert len(api.conversation_stacks) == 5
        assert len(api.conversation_stacks) == 5

        # Verify each conversation has proper isolation
        for i, conv_id in enumerate(conversations):
            data = api.get_data(conv_id)
            assert data["user_id"] == f"user_{i}"
            assert api.get_current_state(conv_id) == "greeting"

        # Clean up all conversations
        for conv_id in conversations:
            api.end_conversation(conv_id)

        assert len(api.conversation_stacks) == 0

    def test_transition_evaluator_integration(self, complex_fsm, mock_llm_interface):
        """Test integration with the transition evaluator."""
        api = API(fsm_definition=complex_fsm, llm_interface=mock_llm_interface)

        # Verify transition evaluator is configured
        assert api.fsm_manager.transition_evaluator is not None

        conv_id, _ = api.start_conversation()

        # Process a message that should trigger transition evaluation
        response = api.converse("I need help with something", conv_id)

        # Verify the transition evaluator was used
        assert isinstance(response, str)

    def test_context_manager_usage(self, complete_simple_fsm, mock_llm_interface):
        """Test API usage as a context manager."""
        with API(
            fsm_definition=complete_simple_fsm, llm_interface=mock_llm_interface
        ) as api:
            conv_id, _ = api.start_conversation()
            assert conv_id in api.conversation_stacks

            # Use the API normally
            response = api.converse("Hello", conv_id)
            assert isinstance(response, str)

        # After exiting context, conversations should be cleaned up
        # Note: This test verifies the context manager protocol works
        assert hasattr(api, "__enter__")
        assert hasattr(api, "__exit__")


class TestSameConversationReentrancyIsBounded:
    def test_error_handler_reentering_converse_does_not_recurse(self):
        """repro1 port: RED on HEAD, the ERROR handler recursed 99 deep."""
        from fsm_llm.core.definitions import FSMError

        depth = {"n": 0, "max": 0}
        raised: list[BaseException] = []
        with _Session(_reentry_fsm()) as s:

            def on_error(ctx):
                depth["n"] += 1
                depth["max"] = max(depth["max"], depth["n"])
                try:
                    s.api.converse("sorry, retrying", s.cid)
                except FSMError as e:
                    raised.append(e)
                finally:
                    depth["n"] -= 1
                return {}

            s.on(HandlerTiming.ERROR, on_error)
            s.provider.down = True
            with pytest.raises(FSMError):
                s.api.converse("hello", s.cid)
            assert depth["max"] == 1
            # greeting + the ambiguity classifier (same binding) + the turn's
            # own generation call; a recursing handler adds one per level.
            assert s.provider.calls <= 3
            assert len(raised) == 1
            assert "already being processed" in str(raised[0])

    def test_error_handler_reentering_converse_stream_does_not_recurse(self):
        from fsm_llm.core.definitions import FSMError

        depth = {"max": 0, "n": 0}
        with _Session(_reentry_fsm()) as s:

            def on_error(ctx):
                depth["n"] += 1
                depth["max"] = max(depth["max"], depth["n"])
                try:
                    list(s.api.converse_stream("retry", s.cid))
                except FSMError:
                    pass
                finally:
                    depth["n"] -= 1
                return {}

            s.on(HandlerTiming.ERROR, on_error)
            s.provider.fail_stream = True
            with pytest.raises(FSMError):
                list(s.api.converse_stream("hello", s.cid))
            assert depth["max"] == 1
            # greeting + the ambiguity classifier (same binding) + the turn's
            # own generation call; a recursing handler adds one per level.
            assert s.provider.calls <= 3

    def test_pre_processing_handler_calling_converse_raises_not_nests(self):
        from fsm_llm.core.definitions import FSMError

        seen: list[BaseException] = []
        with _Session(_reentry_fsm()) as s:
            armed = {"on": False}

            def pre(ctx):
                if armed["on"]:
                    try:
                        s.api.converse("nested", s.cid)
                    except FSMError as e:
                        seen.append(e)
                return {}

            s.on(HandlerTiming.PRE_PROCESSING, pre)
            armed["on"] = True
            s.api.converse("hello", s.cid)
            assert len(seen) == 1
            assert "already being processed" in str(seen[0])
            # the nested call never ran: its message is nowhere in the history
            users = [
                m["user"] for m in s.api.get_conversation_history(s.cid) if "user" in m
            ]
            assert users == ["hello"]

    def test_error_handler_can_still_write_via_update_context(self):
        from fsm_llm.core.definitions import FSMError

        with _Session(_reentry_fsm()) as s:
            s.on(
                HandlerTiming.ERROR,
                lambda ctx: s.api.update_context(s.cid, {"error_seen": True}) or {},
            )
            s.provider.down = True
            with pytest.raises(FSMError):
                s.api.converse("hello", s.cid)
            assert s.api.get_data(s.cid).get("error_seen") is True

    def test_guard_is_released_after_a_failed_turn(self):
        from fsm_llm.core.definitions import FSMError

        with _Session(_reentry_fsm()) as s:
            s.provider.down = True
            with pytest.raises(FSMError):
                s.api.converse("hello", s.cid)
            s.provider.down = False
            assert s.api.converse("again", s.cid)

    def test_abandoned_stream_releases_the_guard(self):
        with _Session(_reentry_fsm()) as s:
            gen = s.api.converse_stream("hello", s.cid)
            next(gen)
            gen.close()
            assert s.api.converse("after close", s.cid)
            gen2 = s.api.converse_stream("hello", s.cid)
            next(gen2)
            del gen2  # CPython closes an unreferenced generator
            assert s.api.converse("after del", s.cid)

    def test_never_iterated_stream_does_not_hold_the_guard(self):
        with _Session(_reentry_fsm()) as s:
            _unused = s.api.converse_stream("hello", s.cid)
            assert s.api.converse("meanwhile", s.cid)
            del _unused

    def test_converse_between_stream_iterations_raises(self):
        """Documented D-018 cost: an open stream owns its conversation."""
        from fsm_llm.core.definitions import FSMError

        with _Session(_reentry_fsm()) as s:
            gen = s.api.converse_stream("hello", s.cid)
            next(gen)
            with pytest.raises(FSMError, match="already being processed"):
                s.api.converse("interleaved", s.cid)
            gen.close()


@contextmanager
def _c12_turn_holding(api: API, frame_id: str, max_hold: float | None = None):
    """Hold ``frame_id``'s conversation lock on another thread, as a running
    turn does. Released on exit, or after ``max_hold`` seconds (default 3 s
    times ``current_timing_factor()``) so a regression that waits for the lock
    cannot hang the test."""
    if max_hold is None:
        max_hold = 3.0 * current_timing_factor()
    lock = api.fsm_manager._conversation_locks[frame_id]
    held, done = threading.Event(), threading.Event()

    def _hold() -> None:
        with lock:
            held.set()
            done.wait(max_hold)

    worker = threading.Thread(target=_hold, daemon=True)
    worker.start()
    assert held.wait(5.0)
    try:
        yield
    finally:
        done.set()
        worker.join(5.0)


def _c12_api() -> tuple[API, list[str]]:
    """An API with an END_CONVERSATION handler that records each end."""
    ends: list[str] = []
    api = API.from_definition(_c3_fsm(), llm_interface=_mock_llm())
    api.register_handler(
        create_handler("c12_end")
        .at(HandlerTiming.END_CONVERSATION)
        .do(lambda ctx: ends.append(ctx.get("_conversation_id")) or {})
    )
    return api, ends


@pytest.fixture
def _c12_short_lock_timeout():
    with patch(
        "fsm_llm.engine.fsm.snapshots.END_CONVERSATION_LOCK_TIMEOUT_SECONDS", 0.2
    ):
        yield


@pytest.fixture
def _later_every_read(monkeypatch: pytest.MonkeyPatch) -> EveryReadLater:
    """The idle sweep ends a conversation when ``now - last_access >
    max_idle_seconds``, so a sweep with ``max_idle_seconds=0`` needs a clock that has
    moved since each conversation started: ``time.monotonic()`` ticks every 15.6 ms on
    Windows before Python 3.13 and a run of fast calls reads the same value (CI run
    44028115098)."""
    clock = EveryReadLater()
    bind_api_clock(monkeypatch, clock)
    return clock


@pytest.mark.usefixtures("_c12_short_lock_timeout")
class TestStep14_1:
    """Review-iter-1 W2: the C12 refusal on the API path."""

    def test_c12_api_end_conversation_propagates_lock_timeout(self, timing_factor):
        import time

        api, ends = _c12_api()
        conv_id, _ = api.start_conversation(initial_context={"note": "n"})
        frame_id = api._get_current_fsm_conversation_id(conv_id)
        with _c12_turn_holding(api, frame_id):
            started = time.monotonic()
            with pytest.raises(FSMError, match="still"):
                api.end_conversation(conv_id)
            # Bounded by the (patched) lock timeout at every step, never by
            # the holder releasing after 3 s (both scaled by the same factor).
            assert time.monotonic() - started < 2.0 * timing_factor
        assert ends == []
        assert frame_id in api.fsm_manager.instances
        assert conv_id in api.list_active_conversations()

    def test_c12_api_bookkeeping_intact_after_refusal_and_retry_succeeds(self):
        api, ends = _c12_api()
        conv_id, _ = api.start_conversation(initial_context={"note": "n"})
        child_def = _c3_fsm().model_copy(update={"name": "child"})
        api.push_fsm(conv_id, child_def)
        child_id = api._get_current_fsm_conversation_id(conv_id)
        assert child_id != conv_id
        for held in (child_id, conv_id):
            with _c12_turn_holding(api, held):
                with pytest.raises(FSMError, match="still"):
                    api.end_conversation(conv_id)
            # Nothing the API tracks was dropped by the refusal.
            assert conv_id in api.list_active_conversations()
            assert api.get_stack_depth(conv_id) == (2 if held == child_id else 1)
            assert api._get_current_fsm_conversation_id(conv_id) == (
                child_id if held == child_id else conv_id
            )
            assert conv_id in api._last_accessed
            assert conv_id not in api._ended_conversations
            assert api.get_data(conv_id)["note"] == "n"
            assert api.has_conversation_ended(conv_id) is False
        # The child frame ended before the root refused; the root is intact.
        assert child_id not in api.fsm_manager.instances
        assert ends == [child_id]
        # Retry after the turn: ends normally and the ended cache answers.
        api.end_conversation(conv_id)
        assert api.list_active_conversations() == []
        assert conv_id not in api.fsm_manager.instances
        assert ends == [child_id, conv_id]
        assert api.get_data(conv_id)["note"] == "n"
        assert api.has_conversation_ended(conv_id) is True

    def test_c12_cleanup_sweep_continues_past_a_refusal(self, _later_every_read):
        api, ends = _c12_api()
        busy_id, _ = api.start_conversation()
        idle_ids = [api.start_conversation()[0] for _ in range(2)]
        with _c12_turn_holding(api, busy_id):
            cleaned = api.cleanup_stale_conversations(max_idle_seconds=0)
        assert sorted(cleaned) == sorted(idle_ids)
        assert api.list_active_conversations() == [busy_id]
        assert busy_id in api.fsm_manager.instances
        # close() also logs and continues per conversation, then ends the rest.
        other_id, _ = api.start_conversation()
        with _c12_turn_holding(api, busy_id):
            api.close()
        assert api.list_active_conversations() == [busy_id]
        assert other_id not in api.fsm_manager.instances
        assert api.cleanup_stale_conversations(max_idle_seconds=0) == [busy_id]
        assert sorted(ends) == sorted([*idle_ids, other_id, busy_id])


def _c12_pathological_graph() -> list[Any]:
    """Review pass 2 p2_cyc.py: a 3-way aliased 14-level list chain with a
    back-edge (a cycle that is itself heavily aliased)."""
    nodes: list[list[Any]] = [[] for _ in range(14)]
    for i in range(13):
        nodes[i].extend([nodes[i + 1]] * 3)
    nodes[13].append(nodes[0])
    return nodes[0]


class TestStep14_2:
    """Review pass 2 concern 1 (API half): only the lock-timeout refusal
    (``ConversationBusyError``) blocks ``API.end_conversation``; any other
    failure of the ended-cache snapshot is logged and the end proceeds
    (plan-2026-09-21T203800-8a03483a/D-053)."""

    @pytest.mark.usefixtures("_c12_short_lock_timeout")
    def test_c12_refusal_is_conversation_busy_error(self):
        import fsm_llm
        from fsm_llm import ConversationBusyError

        assert issubclass(ConversationBusyError, FSMError)
        assert "ConversationBusyError" in fsm_llm.__all__
        api, _ = _c12_api()
        conv_id, _ = api.start_conversation()
        with _c12_turn_holding(api, conv_id):
            with pytest.raises(ConversationBusyError):
                api.fsm_manager.end_conversation(conv_id)
            with pytest.raises(ConversationBusyError):
                api.end_conversation(conv_id)
        assert conv_id in api.list_active_conversations()

    def test_c12_end_proceeds_when_snapshot_fails(self):
        api, ends = _c12_api()
        conv_id, _ = api.start_conversation()
        with patch.object(
            api.fsm_manager, "get_end_snapshot", side_effect=RuntimeError("boom")
        ):
            api.end_conversation(conv_id)
        assert api.list_active_conversations() == []
        assert conv_id not in api.fsm_manager.instances
        assert ends == [conv_id]
        assert conv_id not in api._ended_conversations

    def test_c12_pathological_cyclic_context_can_be_ended_and_swept(
        self, _later_every_read
    ):
        from fsm_llm.context.filtering import ContextFilterError

        api, ends = _c12_api()
        ended_id, _ = api.start_conversation()
        swept_id, _ = api.start_conversation()
        for cid in (ended_id, swept_id):
            _raw_data(api, cid)["graph"] = _c12_pathological_graph()
        with pytest.raises(ContextFilterError):
            api.get_data(ended_id)
        api.end_conversation(ended_id)
        assert ended_id not in api.fsm_manager.instances
        assert api.cleanup_stale_conversations(max_idle_seconds=0) == [swept_id]
        assert api.list_active_conversations() == []
        assert swept_id not in api.fsm_manager.instances
        assert sorted(ends) == sorted([ended_id, swept_id])


class _ParkOnWrite(dict):
    """Parks the cache write for ``key`` until ``release`` is set."""

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key
        self.entered = threading.Event()
        self.release = threading.Event()

    def __setitem__(self, k, v) -> None:
        if k == self.key:
            self.entered.set()
            self.release.wait(timeout=_JOIN_TIMEOUT_S)
        super().__setitem__(k, v)


class _MeetOnEvict(dict):
    """The first two evictions meet on a barrier before popping, so two
    unsynchronised ends pick the same oldest key."""

    def __init__(self) -> None:
        super().__init__()
        self.barrier = threading.Barrier(2, timeout=0.5)

    def pop(self, *args):
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return super().pop(*args)


class TestStep2EndedCacheRace:
    """P0-1: ``_ended_conversations`` is written, evicted and read only under
    ``API._stack_lock`` (plan-2026-09-22T080837-8b258a25/D-003)."""

    def test_p0_1_reader_never_sees_the_gap_between_drop_and_cache_write(self):
        api = _a2026_09_22_api()
        conv_id, _ = api.start_conversation(initial_context={"note": "n"})
        cache = _ParkOnWrite(conv_id)
        api._ended_conversations = cache

        errors: list[BaseException] = []
        results: dict[str, Any] = {}

        def _end() -> None:
            try:
                api.end_conversation(conv_id)
            except BaseException as e:  # pragma: no cover - failure path
                errors.append(e)

        def _read(name: str) -> None:
            try:
                results[name] = getattr(api, name)(conv_id)
            except BaseException as e:
                results[name] = e

        ender = threading.Thread(target=_end)
        ender.start()
        assert cache.entered.wait(timeout=_JOIN_TIMEOUT_S)
        # The end is parked inside the cache write. Readers started now either
        # wait for it (fixed) or run in the gap and see neither the live
        # conversation nor its cache entry (the P0-1 race).
        getters = (
            "get_data",
            "has_conversation_ended",
            "get_current_state",
            "get_conversation_history",
        )
        readers = [threading.Thread(target=_read, args=(g,)) for g in getters]
        for r in readers:
            r.start()
        for r in readers:
            r.join(timeout=0.5)
        cache.release.set()
        _join_all([ender, *readers])

        assert errors == []
        assert results["get_data"] == {"note": "n"}
        assert results["has_conversation_ended"] is True
        assert results["get_current_state"] == "start"
        assert isinstance(results["get_conversation_history"], list)

    def test_p0_1_concurrent_evictions_never_pop_the_same_key(self):
        api = _a2026_09_22_api()
        ids = [api.start_conversation()[0] for _ in range(2)]
        cache = _MeetOnEvict()
        dict.__setitem__(cache, "oldest", {"data": {}, "state": "s", "history": []})
        api._ended_conversations = cache
        api._MAX_ENDED_CACHE = 1

        errors: list[BaseException] = []

        def _end(cid: str) -> None:
            try:
                api.end_conversation(cid)
            except BaseException as e:
                errors.append(e)

        enders = [threading.Thread(target=_end, args=(cid,)) for cid in ids]
        for t in enders:
            t.start()
        _join_all(enders)

        assert errors == []
        assert len(cache) == 1
        assert next(iter(cache)) in ids
        assert api.list_active_conversations() == []

    def test_the_ended_cache_keeps_exactly_its_bound_and_evicts_the_oldest(self):
        """The bound is inclusive: at ``_MAX_ENDED_CACHE`` entries nothing is evicted, one
        more drops the oldest. Found by a step-81 mutant (``>=``) that no test killed."""
        api = _a2026_09_22_api()
        api._MAX_ENDED_CACHE = 2
        ids = [api.start_conversation()[0] for _ in range(3)]

        api.end_conversation(ids[0])
        api.end_conversation(ids[1])
        assert list(api._ended_conversations) == ids[:2]

        api.end_conversation(ids[2])
        assert list(api._ended_conversations) == ids[1:]


class TestStep2GetterFallbacks:
    """P1-4 / P1-5: all four conversation getters share one fallback rule
    (plan-2026-09-22T080837-8b258a25/D-004)."""

    def test_p1_4_history_after_end_comes_from_the_cache(self):
        api = _a2026_09_22_api()
        conv_id, _ = api.start_conversation()
        api.converse("hello", conv_id)
        live = api.get_conversation_history(conv_id)
        assert live
        api.end_conversation(conv_id)
        assert api.get_conversation_history(conv_id) == live

    @pytest.mark.parametrize(
        "getter", ["get_data", "get_current_state", "get_conversation_history"]
    )
    def test_p1_4_unknown_id_still_raises_value_error(self, getter):
        api = _a2026_09_22_api()
        with pytest.raises(ValueError):
            getattr(api, getter)("never-started")

    def test_p1_4_has_conversation_ended_unknown_id_is_false(self):
        assert _a2026_09_22_api().has_conversation_ended("never-started") is False

    @pytest.mark.parametrize(
        "getter",
        [
            "get_data",
            "has_conversation_ended",
            "get_current_state",
            "get_conversation_history",
        ],
    )
    def test_p1_5_not_found_fsm_error_falls_back_to_the_cache(self, getter):
        """The real "API stack intact, FSM instance torn down" shape: the
        manager's own ``end_conversation`` removes the instance while the API
        keeps the conversation stack, so every getter raises a not-found
        ``FSMError`` internally and must answer from the ended cache."""
        api = _a2026_09_22_api()
        conv_id, _ = api.start_conversation()
        api.converse("hello", conv_id)
        frame_id = api._get_current_fsm_conversation_id(conv_id)
        # Cache the same snapshot API.end_conversation would, then tear only
        # the FSM instance down (no patching of the manager or the getters).
        snapshot = api.fsm_manager.get_end_snapshot(frame_id)
        api._ended_conversations[conv_id] = snapshot
        api.fsm_manager.end_conversation(frame_id)
        assert api.fsm_manager.has_instance(frame_id) is False
        assert api._get_current_fsm_conversation_id(conv_id) == frame_id

        if getter == "has_conversation_ended":
            assert api.has_conversation_ended(conv_id) is True
        else:
            key = {
                "get_data": "data",
                "get_current_state": "state",
                "get_conversation_history": "history",
            }[getter]
            assert getattr(api, getter)(conv_id) == snapshot[key]

    @pytest.mark.parametrize(
        ("getter", "manager_method"),
        [
            ("get_data", "get_conversation_data"),
            ("has_conversation_ended", "has_conversation_ended"),
            ("get_current_state", "get_conversation_state"),
            ("get_conversation_history", "get_conversation_history"),
        ],
    )
    def test_step_2_1_live_conversation_error_propagates_despite_cache(
        self, getter, manager_method
    ):
        """A non-busy FSMError on a LIVE conversation is a real failure, not
        "ended": it re-raises even when a stale cache entry exists."""
        api = _a2026_09_22_api()
        conv_id, _ = api.start_conversation()
        api._ended_conversations[conv_id] = {
            "data": {"note": "stale"},
            "state": "end",
            "history": [],
        }
        err = StateNotFoundError("state vanished")
        with patch.object(api.fsm_manager, manager_method, side_effect=err):
            with pytest.raises(StateNotFoundError) as info:
                getattr(api, getter)(conv_id)
        assert info.value is err

    def test_step_2_1_has_conversation_ended_state_resolution_failure_raises(
        self,
    ):
        """Review concern 2: a live conversation whose state resolution fails
        must raise, not report "not ended"."""
        api = _a2026_09_22_api()
        conv_id, _ = api.start_conversation()

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise StateNotFoundError("gone")

        with patch.object(
            api.fsm_manager, "resolve_state_definition", side_effect=_boom
        ):
            with pytest.raises(StateNotFoundError):
                api.has_conversation_ended(conv_id)

    @pytest.mark.parametrize(
        ("getter", "manager_method"),
        [
            ("get_data", "get_conversation_data"),
            ("has_conversation_ended", "has_conversation_ended"),
            ("get_current_state", "get_conversation_state"),
            ("get_conversation_history", "get_conversation_history"),
        ],
    )
    def test_p1_5_busy_refusal_propagates_despite_a_cache_entry(
        self, getter, manager_method
    ):
        api = _a2026_09_22_api()
        conv_id, _ = api.start_conversation()
        api._ended_conversations[conv_id] = {
            "data": {},
            "state": "end",
            "history": [],
        }
        busy = ConversationBusyError("still running a turn")
        with patch.object(api.fsm_manager, manager_method, side_effect=busy):
            with pytest.raises(ConversationBusyError):
                getattr(api, getter)(conv_id)

    def test_p1_5_not_found_fsm_error_without_cache_reraises_original(self):
        api = _a2026_09_22_api()
        conv_id, _ = api.start_conversation()
        err = FSMError(f"Conversation {conv_id} not found")
        with patch.object(api.fsm_manager, "get_conversation_data", side_effect=err):
            with pytest.raises(FSMError) as info:
                api.get_data(conv_id)
        assert info.value is err


class _LockOrderTracker:
    """Records, per thread, which tracked locks are held, and every time the
    manager ``_lock`` is acquired while a ``conv_lock`` is held."""

    def __init__(self) -> None:
        self._local = threading.local()
        self.violations: list[str] = []

    def held(self) -> list[str]:
        if not hasattr(self._local, "held"):
            self._local.held = []
        return self._local.held


class _RecordingLock:
    """Wraps a Lock/RLock; blocking acquires are bounded so a real deadlock
    fails the test instead of hanging the suite."""

    def __init__(self, inner: Any, name: str, tracker: _LockOrderTracker) -> None:
        self._inner = inner
        self._name = name
        self._tracker = tracker

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        held = self._tracker.held()
        if self._name == "manager" and any(h.startswith("conv:") for h in held):
            self._tracker.violations.append(f"_lock under {held}")
        if blocking and timeout == -1:
            timeout = _JOIN_TIMEOUT_S
        got = self._inner.acquire(blocking, timeout)
        if blocking and not got:
            raise AssertionError(f"{self._name} not acquired: possible deadlock")
        if got:
            held.append(self._name)
        return got

    def release(self) -> None:
        self._tracker.held().remove(self._name)
        self._inner.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


class _WrapNewLocks(dict):
    """``_conversation_locks`` replacement that wraps each new conv_lock."""

    def __init__(self, tracker: _LockOrderTracker) -> None:
        super().__init__()
        self._tracker = tracker

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, _RecordingLock(value, f"conv:{key}", self._tracker))


class _OneSessionStore:
    """In-memory store that hands back one prepared ``SessionState``."""

    def __init__(self, state: Any) -> None:
        self.state = state

    def save(self, session_id: str, state: Any) -> None:
        self.state = state

    def load(self, session_id: str) -> Any:
        return self.state

    def delete(self, session_id: str) -> bool:
        return True


def _saved_state(api: API, **overrides: Any) -> Any:
    from fsm_llm.engine.session import SessionState

    fields: dict[str, Any] = {
        "conversation_id": "saved",
        "fsm_id": api.fsm_id,
        "current_state": "start",
        "context_data": {"name": "Ada"},
        "conversation_history": [{"user": "hi"}, {"system": "hello"}],
        "conversation_summary": "earlier: talked about tea",
        "metadata": {"pipeline_extracted": {"name": {"turn": 1}}},
        "working_memory": {"buffers": {"core": {"k": "v"}}},
    }
    fields.update(overrides)
    return SessionState(**fields)


def _restore_api(**overrides: Any) -> tuple[API, _OneSessionStore]:
    api = _a2026_09_22_api()
    store = _OneSessionStore(None)
    api._session_store = store
    store.state = _saved_state(api, **overrides)
    return api, store


def _restored_instance(api: API, conv_id: str) -> Any:
    fsm_id = api._get_current_fsm_conversation_id(conv_id)
    return api.fsm_manager.instances[fsm_id]


class TestStep3RestoreLockOrder:
    def test_p0_3_restore_never_takes_manager_lock_under_conv_lock(self):
        api, _ = _restore_api()
        tracker = _LockOrderTracker()
        manager = api.fsm_manager
        manager._lock = _RecordingLock(manager._lock, "manager", tracker)
        manager._conversation_locks = _WrapNewLocks(tracker)

        outcome: dict[str, Any] = {}

        def _run() -> None:
            try:
                outcome["result"] = api.restore_session("saved")
            except BaseException as exc:  # surfaced below
                outcome["error"] = exc

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        _join_all([worker])
        assert "error" not in outcome, outcome.get("error")
        assert tracker.violations == []

        conv_id, _ = outcome["result"]
        inst = _restored_instance(api, conv_id)
        assert inst.context.conversation.summary == "earlier: talked about tea"
        assert api.get_conversation_history(conv_id) == [
            {"user": "hi"},
            {"system": "hello"},
        ]
        from fsm_llm.constants import PROVENANCE_METADATA_KEY

        assert inst.context.metadata[PROVENANCE_METADATA_KEY] == {"name": {"turn": 1}}
        assert inst.context.working_memory.get("core", "k") == "v"

    def test_p0_3_api_has_no_manager_lock_reach_in(self):
        import ast
        import importlib
        import inspect
        import pkgutil

        import fsm_llm.engine.api as api_package

        # The API is a package since step 81: read every module, the facade included.
        modules = [
            api_package,
            *(
                importlib.import_module(f"{api_package.__name__}.{info.name}")
                for info in pkgutil.iter_modules(api_package.__path__)
            ),
        ]
        assert len(modules) >= 8
        reach_ins = [
            f"{module.__name__}:{node.lineno}: fsm_manager.{node.attr}"
            for module in modules
            for node in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(node, ast.Attribute)
            and node.attr in {"_lock", "_conversation_locks", "instances"}
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "fsm_manager"
        ]
        assert reach_ins == []

    def test_p0_3_seed_on_unknown_conversation_raises_fsm_error(self):
        api = _a2026_09_22_api()
        with pytest.raises(FSMError, match="not found"):
            api.fsm_manager.seed_restored_conversation(
                "no-such-id",
                summary=None,
                history=[{"user": "hi"}],
                provenance=None,
                working_memory=None,
            )

    def test_p0_3_missing_summary_leaves_fresh_summary(self):
        api, _ = _restore_api(conversation_summary=None)
        conv_id, _ = api.restore_session("saved")
        assert _restored_instance(api, conv_id).context.conversation.summary is None

    def test_p0_3_absent_working_memory_keeps_fresh_default(self):
        api, _ = _restore_api(working_memory=None)
        conv_id, _ = api.restore_session("saved")
        fresh = api.start_conversation()[0]
        restored_wm = _restored_instance(api, conv_id).context.working_memory
        fresh_wm = _restored_instance(api, fresh).context.working_memory
        assert type(restored_wm) is type(fresh_wm)
        if fresh_wm is not None:
            assert restored_wm.list_buffers() == fresh_wm.list_buffers()

    def test_p0_3_null_buffers_mean_default_buffers(self):
        from fsm_llm.memory import DEFAULT_BUFFERS

        api, _ = _restore_api(working_memory={"buffers": None})
        conv_id, _ = api.restore_session("saved")
        wm = _restored_instance(api, conv_id).context.working_memory
        assert set(wm.list_buffers()) == set(DEFAULT_BUFFERS)

    def test_p0_3_explicit_empty_buffers_are_honoured(self):
        api, _ = _restore_api(working_memory={"buffers": {}, "hidden_buffers": []})
        conv_id, _ = api.restore_session("saved")
        wm = _restored_instance(api, conv_id).context.working_memory
        assert wm.list_buffers() == []
        assert wm.hidden_buffers == frozenset()

    def test_p0_3_seed_failure_tears_down_the_new_conversation(self):
        api, _ = _restore_api()
        before = set(api.conversation_stacks)
        with patch.object(
            api.fsm_manager,
            "seed_restored_conversation",
            side_effect=FSMError("seed failed"),
            create=True,
        ):
            with pytest.raises(FSMError, match="seed failed"):
                api.restore_session("saved")
        assert set(api.conversation_stacks) == before


def _sleeping_handler(seconds: float) -> Any:
    import time

    from fsm_llm.handlers import HandlerTiming, create_handler

    def _sleep(ctx: dict[str, Any]) -> dict[str, Any]:
        time.sleep(seconds)
        return {"slept": True}

    return create_handler("sleeper").at(HandlerTiming.PRE_PROCESSING).do(_sleep)


class TestStep4ConfigPassthrough:
    def test_p0_4_handler_timeout_reaches_handler_system(self):
        from fsm_llm.handlers import HandlerExecutionError, HandlerTiming

        api = API.from_definition(
            _blocked_fsm(),
            llm_interface=_mock_llm(),
            handler_error_mode="raise",
            handler_timeout=0.05,
        )
        assert api.handler_system.handler_timeout == 0.05
        api.handler_system.register_handler(_sleeping_handler(0.3))
        with pytest.raises(HandlerExecutionError) as exc_info:
            api.handler_system.execute_handlers(
                HandlerTiming.PRE_PROCESSING, "start", None, {}
            )
        assert isinstance(exc_info.value.original_error, TimeoutError)

    def test_p0_4_default_api_has_no_handler_timeout(self):
        from fsm_llm.handlers import HandlerTiming

        api = API.from_definition(
            _blocked_fsm(), llm_interface=_mock_llm(), handler_error_mode="raise"
        )
        assert api.handler_system.handler_timeout is None
        api.handler_system.register_handler(_sleeping_handler(0.1))
        result = api.handler_system.execute_handlers(
            HandlerTiming.PRE_PROCESSING, "start", None, {}
        )
        assert result == {"slept": True}

    def test_p0_4_max_fsm_cache_size_reaches_manager_and_evicts(self):
        api = API.from_definition(
            _blocked_fsm(), llm_interface=_mock_llm(), max_fsm_cache_size=1
        )
        manager = api.fsm_manager
        api._temp_fsm_definitions["other"] = _blocked_fsm()
        manager.fsm_cache.clear()
        manager.get_fsm_definition(api.fsm_id)
        manager.get_fsm_definition("other")
        assert list(manager.fsm_cache) == ["other"]

    def test_p0_4_default_fsm_cache_size_is_unchanged(self):
        from fsm_llm.constants import DEFAULT_MAX_FSM_CACHE_SIZE
        from fsm_llm.engine.fsm.manager import FSMManager

        api = _a2026_09_22_api()
        assert api.fsm_manager._max_fsm_cache_size == 64
        assert DEFAULT_MAX_FSM_CACHE_SIZE == 64
        assert FSMManager(llm_interface=_mock_llm())._max_fsm_cache_size == 64

    def test_p0_4_inert_default_handler_timeout_is_gone(self):
        from fsm_llm import constants

        assert not hasattr(constants, "DEFAULT_HANDLER_TIMEOUT")


# (class, kwarg) -> (API kwarg, non-default value factory, read-back from API)
_REACHABLE = {
    ("HandlerSystem", "handler_error_mode"): (
        "handler_error_mode",
        lambda: HandlerErrorMode.RAISE,  # a member: the read-back is by identity
        lambda api: api.handler_system.handler_error_mode,
    ),
    ("HandlerSystem", "handler_timeout"): (
        "handler_timeout",
        lambda: 0.25,
        lambda api: api.handler_system.handler_timeout,
    ),
    ("FSMManager", "llm_interface"): (
        "llm_interface",
        None,  # identity: the mock passed to every API below
        lambda api: api.fsm_manager.llm_interface,
    ),
    ("FSMManager", "max_history_size"): (
        "max_history_size",
        lambda: 7,
        lambda api: api.fsm_manager.max_history_size,
    ),
    ("FSMManager", "max_message_length"): (
        "max_message_length",
        lambda: 1234,
        lambda api: api.fsm_manager.max_message_length,
    ),
    ("FSMManager", "max_fsm_cache_size"): (
        "max_fsm_cache_size",
        lambda: 3,
        lambda api: api.fsm_manager._max_fsm_cache_size,
    ),
    ("FSMManager", "transition_evaluator"): (
        "transition_config",
        TransitionEvaluatorConfig,
        lambda api: api.fsm_manager.transition_evaluator.config,
    ),
    # a mapping is checked, then copied read-only: the read-back is equal, not
    # the same object (see the Mapping branch of the round-trip test)
    ("FSMManager", "executors"): (
        "executors",
        lambda: {"conversation": ConversationExecutor()},
        lambda api: api.fsm_manager._pipeline._executors,
    ),
}


_EXEMPT = {
    ("FSMManager", "fsm_loader"): "API builds its own loader over fsm_definition "
    "and the push_fsm temp definitions; a caller loader would bypass both",
    ("FSMManager", "handler_system"): "API builds one HandlerSystem from "
    "handler_error_mode + handler_timeout and shares it (both reachable above)",
    ("FSMManager", "handler_error_mode"): "ignored when handler_system is "
    "given, which API always does; reached via API(handler_error_mode=...)",
    ("FSMManager", "data_extraction_prompt_builder"): "API constructs "
    "default-config prompt builders; custom builders need FSMManager directly",
    ("FSMManager", "response_generation_prompt_builder"): "API constructs "
    "default-config prompt builders; custom builders need FSMManager directly",
    ("FSMManager", "field_extraction_prompt_builder"): "API constructs "
    "default-config prompt builders; custom builders need FSMManager directly",
}


def _init_kwargs(cls: type) -> set[str]:
    return {
        n
        for n, p in inspect.signature(cls.__init__).parameters.items()
        if n != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }


def _sweep_fsm() -> dict[str, Any]:
    return {
        "name": "sweep",
        "description": "d",
        "initial_state": "a",
        "states": {
            "a": {
                "id": "a",
                "description": "a",
                "purpose": "a",
                "response_instructions": "hi",
            }
        },
    }


class TestConfigReachability:
    def test_every_option_is_classified(self):
        from fsm_llm.engine.fsm.manager import FSMManager
        from fsm_llm.handlers import HandlerSystem

        declared = {
            (cls.__name__, kw)
            for cls in (HandlerSystem, FSMManager)
            for kw in _init_kwargs(cls)
        }
        classified = set(_REACHABLE) | set(_EXEMPT)
        assert declared == classified, (
            f"unclassified: {sorted(declared - classified)}; "
            f"stale: {sorted(classified - declared)}"
        )
        assert set(_REACHABLE).isdisjoint(_EXEMPT)
        assert all(reason.strip() for reason in _EXEMPT.values())

    @pytest.mark.parametrize(
        "option", sorted(_REACHABLE), ids=lambda o: f"{o[0]}.{o[1]}"
    )
    def test_reachable_option_round_trips(self, option):
        from fsm_llm.engine.api.conversation import API
        from fsm_llm.llm import LLMInterface

        api_kw, make, read_back = _REACHABLE[option]
        assert api_kw in _init_kwargs(API)
        llm = MagicMock(spec=LLMInterface)
        value = llm if make is None else make()
        kwargs = {"fsm_definition": _sweep_fsm(), "llm_interface": llm, api_kw: value}
        held = read_back(API(**kwargs))
        if isinstance(value, Mapping):
            assert dict(held) == dict(value)
        else:
            assert held is value


class TestHistoryAndLengthLimitsAreValidatedAtConstruction:
    """F-AF-04: a bad limit used to fail every start_conversation instead."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_history_size": -1},
            {"max_history_size": 1001},
            {"max_message_length": 0},
            {"max_message_length": 50001},
        ],
    )
    def test_api_constructor_refuses_the_value(self, kwargs):
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            API.from_definition(_TWO_STATE, llm_interface=MockLLM2Interface(), **kwargs)

    @pytest.mark.parametrize(
        "kwargs", [{"max_history_size": -1}, {"max_message_length": 0}]
    )
    def test_manager_constructor_refuses_the_value(self, kwargs):
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            FSMManager(llm_interface=MockLLM2Interface(), **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_history_size": 0, "max_message_length": 1},
            {"max_history_size": 1000, "max_message_length": 50000},
        ],
    )
    def test_the_bounds_themselves_are_accepted(self, kwargs):
        api = API.from_definition(
            _TWO_STATE, llm_interface=MockLLM2Interface(), **kwargs
        )
        api.start_conversation()


class TestStartConversationErrorContract:
    """F-AF-05 (type) and F-AF-12 (falsy ``initial_context``)."""

    def test_a_value_error_passes_through_like_push_and_pop(self, monkeypatch):
        api = API.from_definition(_TWO_STATE, llm_interface=MockLLM2Interface())

        def boom(*args, **kwargs):
            raise ValueError("bad value")

        monkeypatch.setattr(api.fsm_manager, "start_conversation", boom)
        with pytest.raises(ValueError, match="bad value") as caught:
            api.start_conversation()
        assert not isinstance(caught.value, FSMError)

    def test_an_unexpected_error_is_still_wrapped(self, monkeypatch):
        api = API.from_definition(_TWO_STATE, llm_interface=MockLLM2Interface())

        def boom(*args, **kwargs):
            raise RuntimeError("kaput")

        monkeypatch.setattr(api.fsm_manager, "start_conversation", boom)
        with pytest.raises(FSMError, match="Failed to start conversation: kaput"):
            api.start_conversation()

    @pytest.mark.parametrize("bad", [[], "", 0, [1], "x"])
    def test_a_non_dict_initial_context_is_refused_whatever_its_truthiness(self, bad):
        api = API.from_definition(_TWO_STATE, llm_interface=MockLLM2Interface())
        with pytest.raises(FSMError, match="initial_context must be a dict"):
            api.start_conversation(initial_context=bad)

    @pytest.mark.parametrize("fine", [None, {}, {"k": "v"}])
    def test_none_and_dicts_still_start(self, fine):
        api = API.from_definition(_TWO_STATE, llm_interface=MockLLM2Interface())
        cid, _ = api.start_conversation(initial_context=fine)
        assert cid


class TestStreamFailureIsLoggedLikeTheSyncTurn:
    """F-AF-09: a failing stream turn left no ERROR line, the sync turn did."""

    def _failing_api(self, monkeypatch):
        api = API.from_definition(
            _TWO_STATE, llm_interface=MockLLM2Interface(response_text="x")
        )
        cid, _ = api.start_conversation()

        def boom(*args, **kwargs):
            raise RuntimeError("pipeline exploded")

        monkeypatch.setattr(api.fsm_manager._pipeline, "process", boom)
        monkeypatch.setattr(api.fsm_manager._pipeline, "process_stream", boom)
        return api, cid

    def _error_records(self, run):
        records: list[str] = []
        logger.enable("fsm_llm")
        sink_id = logger.add(lambda msg: records.append(str(msg)), level="ERROR")
        try:
            with pytest.raises(FSMError, match="Failed to process message"):
                run()
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")
        return records

    def test_sync_turn_logs_the_failure(self, monkeypatch):
        api, cid = self._failing_api(monkeypatch)
        records = self._error_records(lambda: api.converse("hi", cid))
        assert any("pipeline exploded" in r for r in records)

    def test_stream_turn_logs_the_failure(self, monkeypatch):
        api, cid = self._failing_api(monkeypatch)
        records = self._error_records(lambda: list(api.converse_stream("hi", cid)))
        assert any("pipeline exploded" in r for r in records)


class TestPathDefinitionIsAcceptedEverywhere:
    """F-AF-19: ``API(Path)`` was refused while ``API.from_file(Path)`` worked."""

    @pytest.fixture
    def fsm_path(self, tmp_path):
        path = tmp_path / "fsm.json"
        path.write_text(json.dumps(_TWO_STATE), encoding="utf-8")
        return path

    def test_constructor_takes_a_path(self, fsm_path):
        api = API(fsm_definition=fsm_path, llm_interface=MockLLM2Interface())
        assert api.fsm_definition.name == "b"

    def test_process_fsm_definition_takes_a_path(self, fsm_path):
        definition, _ = API.process_fsm_definition(fsm_path)
        assert definition.initial_state == "s"

    def test_from_file_and_builder_still_take_a_path(self, fsm_path):
        assert API.from_file(fsm_path, llm_interface=MockLLM2Interface()).fsm_id
        built = (
            APIBuilder().set_definition(fsm_path).set_llm_interface(MockLLM2Interface())
        )
        assert built.build().fsm_definition.name == "b"


class TestApiRefusesSettingsBesideAnInjectedInterface:
    """Review area 7, concern 1: ``API(llm_interface=..., seed=7)`` dropped
    the seed silently (native_fc's ``agent.seed`` still reported it)."""

    @pytest.mark.parametrize(
        "settings", [{"seed": 7}, {"api_key": "k"}, {"timeout": 5, "caching": True}]
    )
    def test_refused(self, settings: dict[str, Any]):
        with pytest.raises(ValueError, match="takes no LLM settings"):
            API.from_definition(
                _tool_fsm(), llm_interface=MockLLM2Interface(), **settings
            )

    def test_native_fc_seed_with_an_injected_interface_is_refused(self):
        from fsm_llm.agents import AgentConfig, ToolRegistry, create_agent

        def echo(x: str) -> str:
            """Echo."""
            return x

        registry = ToolRegistry()
        registry.register_function(echo, name="echo")
        agent = create_agent(
            "native_fc",
            registry,
            config=AgentConfig(model="gpt-4o-mini"),
            seed=7,
            llm_interface=LiteLLMInterface(model="gpt-4o-mini"),
        )
        # The provider binding is patched so a regression never sends.
        with (
            _Provider(_reply("done")) as provider,
            pytest.raises(ValueError, match=r"\['seed'\]"),
        ):
            agent.run("hi")
        assert provider.calls == []


class TestApiRefusesEverySettingBesideAnInterface:
    """Pass 8 warning 4: temperature, max_tokens and model beside an injected
    interface were accepted and silently ignored."""

    @pytest.mark.parametrize(
        ("setting", "value"),
        [("temperature", 0.9), ("max_tokens", 7), ("model", "gpt-4o"), ("seed", 7)],
    )
    def test_each_setting_is_refused(self, setting, value):
        injected = LiteLLMInterface("ollama_chat/qwen3.5:4b", temperature=0.5)
        with pytest.raises(ValueError, match=f"takes no LLM settings.*{setting}"):
            API.from_definition(_tool_fsm(), llm_interface=injected, **{setting: value})

    def test_a_model_on_an_interface_without_one_is_refused(self):
        with pytest.raises(ValueError, match="model"):
            API.from_definition(
                _tool_fsm(), llm_interface=MockLLM2Interface(), model="gpt-4o"
            )

    def test_the_interface_own_model_is_accepted(self):
        injected = LiteLLMInterface("ollama_chat/qwen3.5:4b")
        api = API.from_definition(
            _tool_fsm(), llm_interface=injected, model="ollama_chat/qwen3.5:4b"
        )
        assert api.llm_interface is injected


class TestApiSurfaceIsBacked:
    """``_ApiSurface`` (``engine/api/surface.py``, step 81) declares for mypy only what the
    mixins and helpers of ``API`` share. mypy cannot tell that a declaration is backed by
    nothing (a step-81 mutant that declared a member no class defines passed ``make
    type-check``), so this reads it: every declared attribute or method is defined by a class
    of the ``API`` MRO (class level) or assigned as ``self.<name>`` in one of them."""

    def test_every_declaration_is_defined_by_the_api(self):
        import ast

        from fsm_llm.engine.api import surface

        tree = ast.parse(inspect.getsource(surface))
        surface_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "_ApiSurface"
        )
        declared: set[str] = set()
        for node in surface_class.body:
            if not isinstance(node, ast.If):  # the TYPE_CHECKING block
                continue
            for member in node.body:
                if isinstance(member, ast.AnnAssign) and isinstance(
                    member.target, ast.Name
                ):
                    declared.add(member.target.id)
                elif isinstance(member, ast.FunctionDef):
                    declared.add(member.name)
        assert len(declared) >= 20  # an empty scan would pass vacuously

        defined: set[str] = set()
        for klass in API.__mro__:
            if klass in (surface._ApiSurface, object):
                continue
            defined |= set(vars(klass))
            for node in ast.walk(ast.parse(inspect.getsource(klass))):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, ast.Store)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                ):
                    defined.add(node.attr)
        assert declared - defined == set()
