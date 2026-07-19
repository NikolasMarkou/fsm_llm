from __future__ import annotations

import json
import os

import dotenv

from .api import API
from .constants import (
    COMPILED_FORBIDDEN_CONTEXT_PATTERNS,
    ENV_FSM_PATH,
    ENV_LLM_MAX_TOKENS,
    ENV_LLM_MODEL,
    ENV_LLM_TEMPERATURE,
    MAX_CONTEXT_FILTER_DEPTH,
)
from .logging import logger, setup_file_logging

# --------------------------------------------------------------

_REDACTED = "<redacted>"


# DECISION plan-2026-07-18T162030-a02151fe/D-015
# This duplicates the key-matching loop in ``prompts.py``
# ``_filter_context_for_security`` ON PURPOSE. Do NOT "clean this up" by
# importing that method or hoisting it into a shared helper: it is a bound
# method of ``BasePromptBuilder`` gated on ``config.filter_internal_context``,
# so reusing it from the CLI would mean either instantiating a prompt builder
# here or refactoring a security path. The REGEX list is the thing that must
# stay single-sourced, and it is — never inline a secret pattern here.
# Values are replaced rather than dropped so an operator debugging the CLI can
# still see WHICH keys exist; a dropped key looks identical to a missing one.
#
# DECISION plan-2026-07-19-4b664252/D-014
# The recursion is NOT optional and the replace-don't-drop rule above is NOT a
# harmonization bug. `clean_context_keys` (D-010) and `prompts.py`'s
# `_filter_context_for_security` (D-011) DROP a matched key; this one keeps the
# key and redacts its VALUE, on purpose — see the paragraph above. Do NOT
# "unify" the three. What IS shared, and must stay shared, is the matching
# (`COMPILED_FORBIDDEN_CONTEXT_PATTERNS`) and the bound
# (`MAX_CONTEXT_FILTER_DEPTH`); never inline a pattern or re-declare the depth.
# Behavior AT the bound is fail-CLOSED for the same reason as D-010: a subtree
# too deep to inspect is redacted wholesale rather than logged verbatim,
# otherwise burying a secret 17 levels down prints it. See decisions.md D-014.
def _redact_value(value: object, depth: int) -> object:
    """Redact one context value; recurses into dicts and into lists/tuples."""
    if isinstance(value, dict):
        if depth > MAX_CONTEXT_FILTER_DEPTH:
            return _REDACTED
        return _redact_mapping(value, depth)
    if isinstance(value, (list, tuple)):
        if depth > MAX_CONTEXT_FILTER_DEPTH:
            return _REDACTED
        items = [_redact_value(item, depth + 1) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


def _redact_mapping(source: dict, depth: int) -> dict:
    """Apply the key match at one level, then recurse into the values kept."""
    return {
        key: (
            _REDACTED
            if isinstance(key, str)
            and any(p.match(key) for p in COMPILED_FORBIDDEN_CONTEXT_PATTERNS)
            else _redact_value(value, depth + 1)
        )
        for key, value in source.items()
    }


def _redact_context(data: dict) -> dict:
    """Replace secret-shaped context values before they are written to a log.

    The match is applied at every nesting level — inside nested dicts and
    inside dicts nested in lists/tuples — so ``{"user": {"password": "x"}}``
    is redacted like its flat equivalent. Matched KEYS stay visible and only
    their values become ``"<redacted>"``. Recursion is bounded at
    ``MAX_CONTEXT_FILTER_DEPTH``; anything deeper is redacted wholesale.
    """
    return _redact_mapping(data, 0)


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
    try:
        temperature = float(os.environ.get(ENV_LLM_TEMPERATURE, 0.5))
        max_tokens = int(os.environ.get(ENV_LLM_MAX_TOKENS, 1000))
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
                logger.debug(f"Context data: {json.dumps(_redact_context(data))}")

            except Exception as e:
                logger.exception(e)
                return -1

        data = fsm.get_data(conversation_id)
        logger.info(f"Data: \n{json.dumps(_redact_context(data), indent=3)}")
    finally:
        # Clean up when done — always runs even on exception
        fsm.end_conversation(conversation_id)
        logger.info("Conversation ended")

    return 0


# --------------------------------------------------------------
