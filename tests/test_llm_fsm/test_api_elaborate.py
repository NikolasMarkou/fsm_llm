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

from llm_fsm.api import API, ContextMergeStrategy
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
# ADVANCED FIXTURES
# ======================================================================

@pytest.fixture
def multi_step_form_fsm():
    """Complex multi-step form FSM for elaborate testing."""
    states = {
        "welcome": State(
            id="welcome",
            description="Welcome state",
            purpose="Welcome user and start form",
            transitions=[
                Transition(target_state="personal_info", description="Start personal info collection", priority=1)
            ]
        ),
        "personal_info": State(
            id="personal_info",
            description="Personal information collection",
            purpose="Collect name and email",
            required_context_keys=["name", "email"],
            instructions="Ask for name and email separately. Validate email format.",
            transitions=[
                Transition(
                    target_state="preferences",
                    description="Move to preferences after personal info",
                    conditions=[
                        TransitionCondition(
                            description="Personal info complete",
                            requires_context_keys=["name", "email"]
                        )
                    ],
                    priority=1
                ),
                Transition(
                    target_state="validation_error",
                    description="Handle validation errors",
                    priority=2
                )
            ]
        ),
        "preferences": State(
            id="preferences",
            description="Preferences collection",
            purpose="Collect user preferences",
            required_context_keys=["communication_method", "interests"],
            transitions=[
                Transition(
                    target_state="confirmation",
                    description="Move to confirmation",
                    conditions=[
                        TransitionCondition(
                            description="Preferences collected",
                            requires_context_keys=["communication_method", "interests"]
                        )
                    ],
                    priority=1
                ),
                Transition(target_state="personal_info", description="Go back to personal info", priority=2)
            ]
        ),
        "validation_error": State(
            id="validation_error",
            description="Validation error handling",
            purpose="Handle and recover from validation errors",
            transitions=[
                Transition(target_state="personal_info", description="Retry personal info", priority=1),
                Transition(target_state="help", description="Get help", priority=2)
            ]
        ),
        "help": State(
            id="help",
            description="Help state",
            purpose="Provide help to user",
            transitions=[
                Transition(target_state="personal_info", description="Return to form", priority=1),
                Transition(target_state="exit", description="Exit form", priority=2)
            ]
        ),
        "confirmation": State(
            id="confirmation",
            description="Confirmation state",
            purpose="Confirm all collected information",
            transitions=[
                Transition(target_state="complete", description="Confirm and complete", priority=1),
                Transition(target_state="preferences", description="Go back to edit", priority=2)
            ]
        ),
        "complete": State(
            id="complete",
            description="Completion state",
            purpose="Form completed successfully",
            transitions=[]  # Terminal
        ),
        "exit": State(
            id="exit",
            description="Exit state",
            purpose="User exited form",
            transitions=[]  # Terminal
        )
    }

    return FSMDefinition(
        name="Multi-Step Form FSM",
        description="Complex form with validation and error handling",
        initial_state="welcome",
        persona="A helpful form assistant that guides users through data collection",
        states=states
    )

@pytest.fixture
def decision_tree_fsm():
    """Decision tree FSM for testing branching logic."""
    states = {
        "root": State(
            id="root",
            description="Root decision point",
            purpose="Determine user's primary need",
            required_context_keys=["primary_need"],
            transitions=[
                Transition(target_state="technical_support", description="Technical issues", priority=1),
                Transition(target_state="billing_inquiry", description="Billing questions", priority=2),
                Transition(target_state="general_info", description="General information", priority=3)
            ]
        ),
        "technical_support": State(
            id="technical_support",
            description="Technical support branch",
            purpose="Handle technical issues",
            required_context_keys=["issue_type"],
            transitions=[
                Transition(target_state="advanced_tech", description="Complex technical issue", priority=1),
                Transition(target_state="basic_tech", description="Basic technical issue", priority=2),
                Transition(target_state="escalation", description="Escalate issue", priority=3)
            ]
        ),
        "billing_inquiry": State(
            id="billing_inquiry",
            description="Billing inquiry branch",
            purpose="Handle billing questions",
            required_context_keys=["billing_question_type"],
            transitions=[
                Transition(target_state="payment_issue", description="Payment problems", priority=1),
                Transition(target_state="account_inquiry", description="Account questions", priority=2),
                Transition(target_state="refund_request", description="Refund requests", priority=3)
            ]
        ),
        "general_info": State(
            id="general_info",
            description="General information",
            purpose="Provide general information",
            transitions=[
                Transition(target_state="complete", description="Information provided", priority=1)
            ]
        ),
        "advanced_tech": State(id="advanced_tech", description="Advanced tech", purpose="Advanced tech support", transitions=[]),
        "basic_tech": State(id="basic_tech", description="Basic tech", purpose="Basic tech support", transitions=[]),
        "escalation": State(id="escalation", description="Escalation", purpose="Escalate to specialist", transitions=[]),
        "payment_issue": State(id="payment_issue", description="Payment issue", purpose="Handle payment issues", transitions=[]),
        "account_inquiry": State(id="account_inquiry", description="Account inquiry", purpose="Handle account questions", transitions=[]),
        "refund_request": State(id="refund_request", description="Refund request", purpose="Process refund requests", transitions=[]),
        "complete": State(id="complete", description="Complete", purpose="Process complete", transitions=[])
    }

    return FSMDefinition(
        name="Decision Tree FSM",
        description="Complex decision tree for customer service",
        initial_state="root",
        states=states
    )

@pytest.fixture
def sub_form_fsm():
    """Sub-FSM for testing nested form workflows."""
    states = {
        "collect_address": State(
            id="collect_address",
            description="Address collection",
            purpose="Collect detailed address information",
            required_context_keys=["street", "city", "zip_code"],
            transitions=[
                Transition(
                    target_state="validate_address",
                    description="Validate collected address",
                    conditions=[
                        TransitionCondition(
                            description="All address fields present",
                            requires_context_keys=["street", "city", "zip_code"]
                        )
                    ],
                    priority=1
                )
            ]
        ),
        "validate_address": State(
            id="validate_address",
            description="Address validation",
            purpose="Validate address format and existence",
            transitions=[
                Transition(target_state="address_complete", description="Address valid", priority=1),
                Transition(target_state="collect_address", description="Address invalid, retry", priority=2)
            ]
        ),
        "address_complete": State(
            id="address_complete",
            description="Address collection complete",
            purpose="Address successfully collected and validated",
            transitions=[]  # Terminal
        )
    }

    return FSMDefinition(
        name="Address Sub-Form",
        description="Sub-form for collecting and validating addresses",
        initial_state="collect_address",
        states=states
    )

# ======================================================================
# COMPLEX FSM STACKING TESTS
# ======================================================================

class TestAdvancedFSMStacking:
    """Test complex FSM stacking scenarios."""

    def test_deep_nested_stacking(self, multi_step_form_fsm, sub_form_fsm, mock_llm_interface):
        """Test deeply nested FSM stacking with context inheritance."""
        # Set up responses for nested workflow
        responses = [
            LLMResponse(transition=StateTransition(target_state="welcome", context_update={}),
                        message="Welcome! Let's start.", reasoning=""),
            LLMResponse(transition=StateTransition(target_state="collect_address", context_update={}),
                        message="Let's collect your address.", reasoning=""),
            LLMResponse(transition=StateTransition(target_state="validate_address",
                                                   context_update={"street": "123 Main St", "city": "Anytown", "zip_code": "12345"}),
                        message="Validating address...", reasoning=""),
            LLMResponse(transition=StateTransition(target_state="address_complete", context_update={}),
                        message="Address validated!", reasoning=""),
            LLMResponse(transition=StateTransition(target_state="personal_info", context_update={"address_data": "collected"}),
                        message="Address complete, back to main form.", reasoning="")
        ]
        mock_llm_interface.send_request.side_effect = responses

        api = API(fsm_definition=multi_step_form_fsm, llm_interface=mock_llm_interface)

        # Start main conversation
        conv_id, _ = api.start_conversation({"user_id": "test_user"})
        assert api.get_stack_depth(conv_id) == 1

        # Navigate to personal info
        api.converse("Start form", conv_id)

        # Push address sub-form with inheritance
        api.push_fsm(
            conv_id,
            sub_form_fsm,
            context_to_pass={"form_section": "address"},
            shared_context_keys=["user_id"],
            inherit_context=True,
            preserve_history=True
        )
        assert api.get_stack_depth(conv_id) == 2

        # Work through address collection
        api.converse("123 Main St, Anytown, 12345", conv_id)
        api.converse("Validate this address", conv_id)

        # Verify we're in sub-FSM
        current_state = api.get_current_state(conv_id)
        assert current_state in ["collect_address", "validate_address", "address_complete"]

        # Pop back to main FSM
        response = api.pop_fsm(
            conv_id,
            context_to_return={"address_validated": True, "address_data": "complete"},
            merge_strategy=ContextMergeStrategy.UPDATE
        )

        assert api.get_stack_depth(conv_id) == 1
        assert "back to main form" in response.lower() or "address" in response.lower()

        # Verify context was merged
        data = api.get_data(conv_id)
        assert data["user_id"] == "test_user"  # Preserved from original
        assert data["address_validated"] is True  # From sub-FSM
        assert data["address_data"] == "complete"  # From sub-FSM

    def test_multiple_parallel_stacking_scenarios(self, multi_step_form_fsm, sub_form_fsm, decision_tree_fsm, mock_llm_interface):
        """Test multiple independent conversations with stacking."""
        mock_llm_interface.send_request.return_value = LLMResponse(
            transition=StateTransition(target_state="personal_info", context_update={"progress": "ongoing"}),
            message="Processing...",
            reasoning=""
        )

        api = API(fsm_definition=multi_step_form_fsm, llm_interface=mock_llm_interface)

        # Start multiple conversations
        conv_id1, _ = api.start_conversation({"user": "Alice"})
        conv_id2, _ = api.start_conversation({"user": "Bob"})

        # Push different sub-FSMs to each conversation
        api.push_fsm(conv_id1, sub_form_fsm, context_to_pass={"task": "address"})
        api.push_fsm(conv_id2, decision_tree_fsm, context_to_pass={"task": "support"})

        # Verify independent stacking
        assert api.get_stack_depth(conv_id1) == 2
        assert api.get_stack_depth(conv_id2) == 2

        # Interact with both independently
        api.converse("Alice's message", conv_id1)
        api.converse("Bob's message", conv_id2)

        # Verify data isolation
        data1 = api.get_data(conv_id1)
        data2 = api.get_data(conv_id2)

        assert data1["user"] == "Alice"
        assert data2["user"] == "Bob"
        assert data1["task"] == "address"
        assert data2["task"] == "support"

    def test_stack_context_flow_analysis(self, multi_step_form_fsm, sub_form_fsm, mock_llm_interface):
        """Test detailed context flow analysis in stacked FSMs."""
        mock_llm_interface.send_request.return_value = LLMResponse(
            transition=StateTransition(target_state="collect_address", context_update={}),
            message="Collecting address",
            reasoning=""
        )

        api = API(fsm_definition=multi_step_form_fsm, llm_interface=mock_llm_interface)

        conv_id, _ = api.start_conversation({
            "user_id": "user123",
            "session": "session456",
            "metadata": {"source": "web"}
        })

        # Push sub-FSM with specific sharing configuration
        api.push_fsm(
            conv_id,
            sub_form_fsm,
            context_to_pass={"sub_task": "address_collection"},
            shared_context_keys=["user_id", "session"],
            return_context={"expected_return": "address_data"},
            preserve_history=True
        )

        # Analyze context flow
        flow_info = api.get_context_flow(conv_id)

        assert flow_info["stack_depth"] == 2
        assert len(flow_info["frames"]) == 2

        # Main FSM frame
        main_frame = flow_info["frames"][0]
        assert main_frame["level"] == 0
        assert "user_id" in main_frame["context_keys"]
        assert "session" in main_frame["context_keys"]
        assert "metadata" in main_frame["context_keys"]

        # Sub FSM frame
        sub_frame = flow_info["frames"][1]
        assert sub_frame["level"] == 1
        assert sub_frame["shared_context_keys"] == ["user_id", "session"]
        assert sub_frame["return_context_keys"] == ["expected_return"]
        assert sub_frame["preserve_history"] is True

    def test_context_synchronization_across_stack(self, multi_step_form_fsm, sub_form_fsm, mock_llm_interface):
        """Test that shared context keys remain synchronized across the stack."""
        mock_llm_interface.send_request.return_value = LLMResponse(
            transition=StateTransition(
                target_state="collect_address",
                context_update={"user_id": "updated_user123", "new_data": "test"}
            ),
            message="Updated context",
            reasoning=""
        )

        api = API(fsm_definition=multi_step_form_fsm, llm_interface=mock_llm_interface)

        conv_id, _ = api.start_conversation({"user_id": "original_user123"})

        # Push sub-FSM with shared context
        api.push_fsm(
            conv_id,
            sub_form_fsm,
            shared_context_keys=["user_id"]
        )

        # Process message that updates shared context in sub-FSM
        api.converse("Update my user ID", conv_id)

        # Synchronize shared context
        api.sync_shared_context(conv_id)

        # Check that main FSM received the shared context update
        all_stack_data = api.get_all_stack_data(conv_id)

        # Both FSMs should have the updated user_id
        assert all_stack_data[1]["user_id"] == "updated_user123"  # Sub FSM
        assert all_stack_data[0]["user_id"] == "updated_user123"  # Main FSM


        # But only sub-FSM should have the non-shared data
        assert "new_data" not in all_stack_data[0]  # Main FSM shouldn't have this
        assert all_stack_data[1]["new_data"] == "test"  # Sub FSM should have this
