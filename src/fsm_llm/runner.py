from __future__ import annotations

import json
import os

import dotenv

from .api import API
from .constants import (
    ENV_FSM_PATH,
    ENV_LLM_MAX_TOKENS,
    ENV_LLM_MODEL,
    ENV_LLM_TEMPERATURE,
)
from .logging import logger, setup_file_logging

# --------------------------------------------------------------


def main(fsm_path, max_history_size, max_message_length):
    """
    Run an interactive FSM conversation from the CLI.

    Uses the public API class (not FSMManager directly) for consistency
    with how extension packages integrate.
    """

    # Load environment variables from .env file
    dotenv.load_dotenv()

    # Enable logging for CLI usage (library disables it by default)
    logger.enable("fsm_llm")

    # Set up file logging now that we're actually running
    setup_file_logging()

    # Check if critical environment variables are set
    if not os.getenv(ENV_LLM_MODEL):
        logger.error(f"Missing required environment variable: {ENV_LLM_MODEL}")
        raise RuntimeError(f"Missing required environment variable: {ENV_LLM_MODEL}")

    # Set up model from environment variables (API key handled by LiteLLM)
    llm_model = os.environ[ENV_LLM_MODEL]
    temperature = float(os.environ.get(ENV_LLM_TEMPERATURE, 0.5))
    max_tokens = int(os.environ.get(ENV_LLM_MAX_TOKENS, 1000))

    logger.info(
        json.dumps(
            {
                "llm_model": llm_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            indent=3,
        )
    )

    # Use FSM path from environment if not provided as argument
    if not fsm_path and os.getenv(ENV_FSM_PATH):
        fsm_path = os.getenv(ENV_FSM_PATH)

    # FSM path is required
    if not fsm_path:
        raise RuntimeError(
            "No FSM file specified. Use --fsm <path> or set FSM_PATH environment variable."
        )
    logger.info(f"Loading FSM from file: {fsm_path}")
    fsm_source = fsm_path

    logger.info(f"Starting FSM conversation with model: {llm_model}")
    logger.info(
        f"Conversation history parameters: "
        f"max_history_size={max_history_size}, "
        f"max_message_length={max_message_length}"
    )

    # Create the API instance using the public interface
    fsm = API.from_file(
        fsm_source,
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_history_size=max_history_size,
        max_message_length=max_message_length,
    )

    logger.info(f"Starting conversation with FSM: {fsm_source}")
    logger.info("Type 'exit' to end the conversation.")

    # Start a new conversation
    conversation_id, response = fsm.start_conversation()
    logger.info(f"System: {response}")

    try:
        # Main conversation loop
        while not fsm.has_conversation_ended(conversation_id):
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
                response = fsm.converse(
                    user_message=user_input, conversation_id=conversation_id
                )
                logger.info(f"System: {response}")

                # Log the current state and context
                data = fsm.get_data(conversation_id)
                logger.debug(f"Context data: {json.dumps(data)}")

            except Exception as e:
                logger.exception(e)
                return -1

        data = fsm.get_data(conversation_id)
        logger.info(f"Data: \n{json.dumps(data, indent=3)}")
    finally:
        # Clean up when done — always runs even on exception
        fsm.end_conversation(conversation_id)
        logger.info("Conversation ended")

    return 0


# --------------------------------------------------------------
