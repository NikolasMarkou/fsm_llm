"""
Updated main module with configurable conversation history parameters.
"""

import os
import json
import dotenv
import argparse
from typing import Optional

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .fsm import FSMManager
from .logging import logger
from .llm import LiteLLMInterface
from .constants import (
    ENV_OPENAI_API_KEY, ENV_LLM_MODEL, ENV_LLM_TEMPERATURE,
    ENV_LLM_MAX_TOKENS, ENV_FSM_PATH, DEFAULT_MAX_HISTORY_SIZE,
    DEFAULT_MAX_MESSAGE_LENGTH
)

# --------------------------------------------------------------

def main(
    fsm_path: Optional[str] = None,
    max_history_size: int = DEFAULT_MAX_HISTORY_SIZE,
    max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH
):
    """
    Run the example FSM conversation with a JSON definition loaded from a file.

    Args:
        fsm_path: Path to the FSM definition JSON file (optional)
        max_history_size: Maximum number of conversation exchanges to keep in history
        max_message_length: Maximum length of a message in characters
    """
    # Load environment variables from .env file
    dotenv.load_dotenv()

    # Check if critical environment variables are set
    required_vars = [ENV_OPENAI_API_KEY, ENV_LLM_MODEL]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Set up your API key and model from environment variables
    api_key = os.environ[ENV_OPENAI_API_KEY]
    llm_model = os.environ[ENV_LLM_MODEL]
    temperature = float(os.environ.get(ENV_LLM_TEMPERATURE, 0.5))
    max_tokens = int(os.environ.get(ENV_LLM_MAX_TOKENS, 1000))

    logger.info(
        json.dumps({
            "llm_model": llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": api_key[:10] if api_key else "Not set",
        }, indent=3)
    )

    # Use FSM path from environment if not provided as argument
    if not fsm_path and os.getenv(ENV_FSM_PATH):
        fsm_path = os.getenv(ENV_FSM_PATH)

    # If still no FSM path, use the default example
    if not fsm_path:
        logger.info("No FSM file specified, using built-in 'simple_greeting' example")
        fsm_source = "simple_greeting"
    else:
        logger.info(f"Loading FSM from file: {fsm_path}")
        fsm_source = fsm_path

    logger.info(f"Starting FSM conversation with model: {llm_model}")
    logger.info(f"Conversation history parameters: max_history_size={max_history_size}, max_message_length={max_message_length}")

    # Create a LiteLLM interface
    llm_interface = LiteLLMInterface(
        model=llm_model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Create an FSM manager with the appropriate loader and conversation parameters
    fsm_manager = FSMManager(
        llm_interface=llm_interface,
        max_history_size=max_history_size,
        max_message_length=max_message_length
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
            logger.debug(f"Current state: {fsm_manager.get_conversation_state(conversation_id)}")
            logger.debug(f"Context data: {json.dumps(fsm_manager.get_conversation_data(conversation_id))}")

        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            logger.exception(e)

    data = fsm_manager.get_conversation_data(conversation_id)
    logger.info(f"Data: \n{json.dumps(data, indent=3)}")

    # Clean up when done
    fsm_manager.end_conversation(conversation_id)
    logger.info("Conversation ended")

def main_cli():
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Run an FSM-based conversation")
    parser.add_argument("--fsm", "-f", type=str, help="Path to FSM definition JSON file")
    parser.add_argument("--history-size", "-n", type=int, default=DEFAULT_MAX_HISTORY_SIZE,
                       help=f"Maximum number of conversation exchanges to include in history (default: {DEFAULT_MAX_HISTORY_SIZE})")
    parser.add_argument("--message-length", "-l", type=int, default=DEFAULT_MAX_MESSAGE_LENGTH,
                       help=f"Maximum length of messages in characters (default: {DEFAULT_MAX_MESSAGE_LENGTH})")

    args = parser.parse_args()

    # Run with the provided parameters
    main(
        fsm_path=args.fsm,
        max_history_size=args.history_size,
        max_message_length=args.message_length
    )

if __name__ == "__main__":
    main_cli()