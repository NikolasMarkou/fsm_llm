"""The ``fsm-llm`` chat REPL (``--mode run``)."""

from __future__ import annotations

import json
import os
import sys

import dotenv

from .api import API
from .cli_support import escape_unencodable_output
from .constants import (
    CLI_EXIT_FAILURE,
    CLI_EXIT_INTERRUPTED,
    CLI_EXIT_OK,
    DEFAULT_TEMPERATURE,
    ENV_FSM_PATH,
    ENV_LLM_MAX_TOKENS,
    ENV_LLM_MODEL,
    ENV_LLM_TEMPERATURE,
)
from .llm import DEFAULT_MAX_TOKENS
from .logging import logger, setup_file_logging
from .utilities import redact_context, redacting_json_default

# --------------------------------------------------------------


def _report_failure(e: Exception) -> int:
    """Log *e*, print ``Error: <e>`` on stderr and return the failure exit code.

    Interface contract: the one failure report of the REPL, shared by the first
    model call (``start_conversation``) and every later turn. Returns
    ``CLI_EXIT_FAILURE`` (1; C8: it was -1, which the shell sees as 255, and
    every other failure path of the three CLIs exits 1). Never raises for a
    message the terminal cannot encode (``escape_unencodable_output`` ran).
    """
    logger.exception(e)
    print(f"Error: {e}", file=sys.stderr)
    return CLI_EXIT_FAILURE


def main(fsm_path, max_history_size, max_message_length):
    """
    Run an interactive FSM conversation from the CLI.

    Uses the public API class (not FSMManager directly) for consistency
    with how extension packages integrate.
    """

    # Load environment variables from .env file
    dotenv.load_dotenv()

    # Enable logging for CLI usage (the library disables it by default): file
    # logging enables the library loggers itself.
    setup_file_logging()

    # Check if critical environment variables are set
    if not os.getenv(ENV_LLM_MODEL):
        logger.error(f"Missing required environment variable: {ENV_LLM_MODEL}")
        raise RuntimeError(f"Missing required environment variable: {ENV_LLM_MODEL}")

    # Set up model from environment variables (API key handled by LiteLLM)
    llm_model = os.environ[ENV_LLM_MODEL]
    try:
        temperature = float(os.environ.get(ENV_LLM_TEMPERATURE, DEFAULT_TEMPERATURE))
        max_tokens = int(os.environ.get(ENV_LLM_MAX_TOKENS, DEFAULT_MAX_TOKENS))
    except ValueError as e:
        raise RuntimeError(
            f"Invalid environment variable value for temperature or max_tokens: {e}"
        ) from e

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
    # DECISION plan-2026-11-05T134105-5ed71cf2/D-018
    # The terminal gets the CLI's product (reply, hint, data dump, failure)
    # through print(), NOT the logger: setup_file_logging() swaps loguru's
    # default sink for one that skips fsm_llm records (93d78536/D-011) and
    # every record of this module is an fsm_llm record. Do NOT "fix" this by
    # adding a console sink (it fights D-011's host-record dedup) or by
    # touching replace_default_handler. The logger calls stay for the file log.
    # A reply the terminal cannot encode (lone surrogate, cp1252/ascii stdout)
    # is written as \uXXXX through the shared cli_support helper; do NOT add a
    # second escape routine here, and do NOT let a print end the session.
    escape_unencodable_output()
    print("Type 'exit' to end the conversation.")

    # Start a new conversation. A failure of this first model call is reported
    # like a failure of any later turn (``Error: ...``, exit 1), not as a raw
    # traceback; Ctrl-C still propagates to ``__main__`` (exit 130). No
    # conversation exists yet, so there is nothing to end.
    try:
        conversation_id, response = fsm.start_conversation()
    except Exception as e:
        return _report_failure(e)

    try:
        # A silent state replies "": print no empty "System:" line for it.
        if response:
            logger.info(f"System: {response}")
            print(f"System: {response}")

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
                if response:
                    logger.info(f"System: {response}")
                    print(f"System: {response}")

                # Log the current state and context
                # DECISION plan-2026-09-20T114608-a8e47b88/D-014
                # `default=redacting_json_default`, NEVER `default=str`: a
                # handler-stored non-JSON-native value (datetime, set, ...)
                # must not crash the CLI's debug/dump path, and `str(obj)`
                # falls back to `repr(obj)`, which would write a secret field
                # of a custom object (`Creds(api_key='sk-...')`) verbatim into
                # `logs/`. The shared hook returns only the type name.
                # Lazy (D-016): no redaction or dump runs unless DEBUG is on.
                logger.opt(lazy=True).debug(
                    "Context data: {}",
                    lambda: json.dumps(
                        redact_context(fsm.get_data(conversation_id)),
                        default=redacting_json_default,
                    ),
                )

            except KeyboardInterrupt:
                # C8: Ctrl-C mid-turn ends the session cleanly (the finally
                # below still ends the conversation) instead of a traceback.
                logger.info("Interrupted during a turn")
                return CLI_EXIT_INTERRUPTED
            except Exception as e:
                return _report_failure(e)

        data = fsm.get_data(conversation_id)
        dump = json.dumps(
            redact_context(data), indent=3, default=redacting_json_default
        )
        logger.info("Data: \n{}", dump)
        print(f"Data: \n{dump}")
    finally:
        # Clean up when done — always runs even on exception
        fsm.end_conversation(conversation_id)
        logger.info("Conversation ended")

    return CLI_EXIT_OK


# --------------------------------------------------------------
