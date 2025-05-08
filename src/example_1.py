"""
Simple example of using the LLM-FSM framework with logging.

This demonstrates creating a very basic FSM for collecting a user's name
and greeting them appropriately.
"""

import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Import modules
from .fsm import (
    FSMDefinition,
    LiteLLMInterface,
    FSMManager
)
from .logging import logger


def load_fsm_definition(fsm_id: str) -> FSMDefinition:
    """
    Load an FSM definition by ID, now with improved security handling.

    Args:
        fsm_id: The ID of the FSM definition

    Returns:
        The FSM definition
    """
    if fsm_id == "simple_greeting":
        # Define a simple FSM for greeting a user
        fsm_def = {
            "name": "Simple Greeting",
            "description": "A simple FSM that greets the user and collects their name",
            "initial_state": "start",
            "states": {
                "start": {
                    "id": "start",
                    "description": "Initial state",
                    "purpose": "Begin the conversation",
                    "transitions": [
                        {
                            "target_state": "greeting",
                            "description": "Always transition to greeting",
                            "priority": 0
                        }
                    ]
                },
                "greeting": {
                    "id": "greeting",
                    "description": "Greet the user",
                    "purpose": "Welcome the user and ask for their name",
                    "instructions": "Greet the user warmly and ask for their name. If the user mentions self-harm, suicide, or anything concerning, transition to the safety_concern state.",
                    "transitions": [
                        {
                            "target_state": "collect_name",
                            "description": "After greeting, collect the user's name",
                            "priority": 1
                        },
                        {
                            "target_state": "safety_concern",
                            "description": "If the user mentions self-harm or suicide, transition to address safety concerns",
                            "priority": 0
                        }
                    ]
                },
                "safety_concern": {
                    "id": "safety_concern",
                    "description": "Address safety concerns",
                    "purpose": "Respond to mentions of self-harm or crisis with appropriate resources",
                    "instructions": "Express concern for the user's wellbeing. Provide appropriate crisis resources. Encourage the user to seek professional help. Do not collect personal information in this state.",
                    "transitions": [
                        {
                            "target_state": "greeting",
                            "description": "Return to greeting after addressing safety concerns",
                            "priority": 0
                        }
                    ]
                },
                "collect_name": {
                    "id": "collect_name",
                    "description": "Collect the user's name",
                    "purpose": "Get the user's name",
                    "required_context_keys": ["name"],
                    "instructions": "Extract the user's name from their message if provided. If they haven't provided it, ask for it politely. If the user mentions self-harm or suicide, transition to the safety_concern state.",
                    "transitions": [
                        {
                            "target_state": "personalized_greeting",
                            "description": "When name is collected, give a personalized greeting",
                            "conditions": [
                                {
                                    "description": "Name has been provided",
                                    "requires_context_keys": ["name"]
                                }
                            ],
                            "priority": 1
                        },
                        {
                            "target_state": "safety_concern",
                            "description": "If the user mentions self-harm or suicide, transition to address safety concerns",
                            "priority": 0
                        }
                    ]
                },
                "personalized_greeting": {
                    "id": "personalized_greeting",
                    "description": "Give a personalized greeting",
                    "purpose": "Greet the user by name",
                    "instructions": "Greet the user warmly, using their name. Then ask how you can help them today.",
                    "transitions": [
                        {
                            "target_state": "conversation",
                            "description": "Move to general conversation",
                            "priority": 1
                        },
                        {
                            "target_state": "safety_concern",
                            "description": "If the user mentions self-harm or suicide, transition to address safety concerns",
                            "priority": 0
                        }
                    ]
                },
                "conversation": {
                    "id": "conversation",
                    "description": "General conversation",
                    "purpose": "Have a general conversation with the user",
                    "instructions": "Engage in friendly conversation with the user, responding to their queries and comments. Use their name occasionally to personalize the interaction.",
                    "transitions": [
                        {
                            "target_state": "end",
                            "description": "If the user says goodbye or thanks, end the conversation",
                            "priority": 1
                        },
                        {
                            "target_state": "safety_concern",
                            "description": "If the user mentions self-harm or suicide, transition to address safety concerns",
                            "priority": 0
                        }
                    ]
                },
                "end": {
                    "id": "end",
                    "description": "End of conversation",
                    "purpose": "End the conversation",
                    "instructions": "Thank the user by name for the conversation and say goodbye.",
                    "transitions": []  # No transitions from the end state
                }
            }
        }

        # Convert the dictionary to an FSMDefinition object
        return FSMDefinition(**fsm_def)

    else:
        logger.error(f"Unknown FSM ID: {fsm_id}")
        raise ValueError(f"Unknown FSM ID: {fsm_id}")




def main():
    """Run the example FSM conversation."""

    # Load environment variables from .env file
    load_dotenv()

    # Check if critical environment variables are set
    required_vars = ["OPENAI_API_KEY", "LLM_MODEL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")


    # Set up your API key (in a real system, use environment variables or a config file)
    api_key = os.environ["OPENAI_API_KEY"]
    llm_model = os.environ["LLM_MODEL"]
    logger.info("Starting Simple Greeting FSM example")

    # Create a LiteLLM interface
    llm_interface = LiteLLMInterface(
        model=llm_model,
        api_key=api_key,
        temperature=0.5,
        max_tokens=1000
    )

    # Create an FSM manager
    fsm_manager = FSMManager(
        fsm_loader=load_fsm_definition,
        llm_interface=llm_interface
    )

    # Create an FSM instance
    instance = fsm_manager.create_instance("simple_greeting")

    logger.info("Starting conversation...")
    logger.info("Type 'exit' to end the conversation.")

    # Initial state transition (from start to greeting)
    # We use an empty input just to trigger the initial greeting
    instance, response = fsm_manager.process_user_input(instance, "")
    logger.info(f"System: {response}")

    # Main conversation loop
    while instance.current_state != "end":
        # Get user input
        user_input = input("You: ")

        # Check for exit command
        if user_input.lower() == "exit":
            logger.info("User requested exit")
            break

        try:
            # Process the user input
            instance, response = fsm_manager.process_user_input(instance, user_input)
            logger.info(f"System: {response}")

            # Log the current state and context
            logger.info(f"Current state: {instance.current_state}")
            logger.debug(f"Context data: {json.dumps(instance.context.data)}")

        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")

    logger.info("Conversation ended")


if __name__ == "__main__":
    main()