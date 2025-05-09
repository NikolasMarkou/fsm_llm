import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Import modules
from .llm import LiteLLMInterface
from .fsm_manager import FSMManager

from .logging import logger
from .utilities import load_fsm_definition

def main(fsm_path: Optional[str] = None):
    """
    Run the example FSM conversation with a JSON definition loaded from a file.

    Args:
        fsm_path: Path to the FSM definition JSON file (optional)
    """
    # Load environment variables from .env file
    load_dotenv()

    # Check if critical environment variables are set
    required_vars = ["OPENAI_API_KEY", "LLM_MODEL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Set up your API key and model from environment variables
    api_key = os.environ["OPENAI_API_KEY"]
    llm_model = os.environ["LLM_MODEL"]
    temperature = os.environ.get("LLM_TEMPERATURE", 0.5)
    max_tokens = os.environ.get("LLM_MAX_TOKENS", 1000)

    # Use FSM path from environment if not provided as argument
    if not fsm_path and os.getenv("FSM_PATH"):
        fsm_path = os.getenv("FSM_PATH")

    # If still no FSM path, use the default example
    if not fsm_path:
        logger.info("No FSM file specified, using built-in 'simple_greeting' example")
        fsm_source = "simple_greeting"
    else:
        logger.info(f"Loading FSM from file: {fsm_path}")
        fsm_source = fsm_path

    logger.info(f"Starting FSM conversation with model: {llm_model}")

    # Create a LiteLLM interface
    llm_interface = LiteLLMInterface(
        model=llm_model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Create an FSM manager with the appropriate loader
    fsm_manager = FSMManager(
        fsm_loader=load_fsm_definition,
        llm_interface=llm_interface
    )

    logger.info(f"Starting conversation with FSM: {fsm_source}")
    logger.info("Type 'exit' to end the conversation.")

    # Start a new conversation
    conversation_id, response = fsm_manager.start_conversation(fsm_source)
    logger.info(f"System: {response}")

    # Main conversation loop
    while not fsm_manager.is_conversation_ended(conversation_id):
        # Get user input
        user_input = input("You: ")

        # Check for exit command
        if user_input.lower() == "exit":
            logger.info("User requested exit")
            break

        try:
            # Process the user input
            response = fsm_manager.process_message(conversation_id, user_input)
            logger.info(f"System: {response}")

            # Log the current state and context
            logger.info(f"Current state: {fsm_manager.get_conversation_state(conversation_id)}")
            logger.debug(f"Context data: {json.dumps(fsm_manager.get_conversation_data(conversation_id))}")

        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            logger.exception(e)

    data = fsm_manager.get_conversation_data(conversation_id)
    logger.info(f"Data: \n{json.dumps(data, indent=3)}")

    # Clean up when done
    fsm_manager.end_conversation(conversation_id)
    logger.info("Conversation ended")

if __name__ == "__main__":
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Run an FSM-based conversation")
    parser.add_argument("--fsm", "-f", type=str, help="Path to FSM definition JSON file")

    args = parser.parse_args()

    # Run with the provided FSM path (if any)
    main(fsm_path=args.fsm)