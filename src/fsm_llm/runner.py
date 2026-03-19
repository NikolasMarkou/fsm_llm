from __future__ import annotations

import os
import json
import dotenv

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .fsm import FSMManager
from .logging import logger, setup_file_logging
from .llm import LiteLLMInterface

# Enable logging for CLI usage (library disables it by default)
logger.enable("fsm_llm")
from .constants import (
    ENV_LLM_MODEL, ENV_LLM_TEMPERATURE,
    ENV_LLM_MAX_TOKENS, ENV_FSM_PATH
)

# --------------------------------------------------------------

def main(fsm_path, max_history_size, max_message_length):
    """
    Run the example FSM conversation with a JSON definition loaded from a file.
    """

    # Set up file logging now that we're actually running
    setup_file_logging()

    # Load environment variables from .env file
    dotenv.load_dotenv()

    # Check if critical environment variables are set
    if not os.getenv(ENV_LLM_MODEL):
        logger.error(f"Missing required environment variable: {ENV_LLM_MODEL}")
        raise OSError(f"Missing required environment variable: {ENV_LLM_MODEL}")

    # Set up model from environment variables (API key handled by LiteLLM)
    llm_model = os.environ[ENV_LLM_MODEL]
    temperature = float(os.environ.get(ENV_LLM_TEMPERATURE, 0.5))
    max_tokens = int(os.environ.get(ENV_LLM_MAX_TOKENS, 1000))

    logger.info(
        json.dumps({
            "llm_model": llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
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
    logger.info(f"Conversation history parameters: "
                f"max_history_size={max_history_size}, "
                f"max_message_length={max_message_length}")

    # Create a LiteLLM interface (API key discovered automatically by LiteLLM)
    llm_interface = LiteLLMInterface(
        model=llm_model,
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

    try:
        # Main conversation loop
        while not fsm_manager.has_conversation_ended(conversation_id):
            # Get user input
            try:
                user_input = input("You: ")
            except (EOFError, KeyboardInterrupt):
                logger.info("Input stream closed or interrupted")
                break

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
                logger.exception(e)
                return -1

        data = fsm_manager.get_conversation_data(conversation_id)
        logger.info(f"Data: \n{json.dumps(data, indent=3)}")
    finally:
        # Clean up when done — always runs even on exception
        fsm_manager.end_conversation(conversation_id)
        logger.info("Conversation ended")

    return 0

# --------------------------------------------------------------
