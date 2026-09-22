from __future__ import annotations

import json
import os

import dotenv

from .api import API
from .constants import (
    CLI_EXIT_FAILURE,
    CLI_EXIT_INTERRUPTED,
    CLI_EXIT_OK,
    ENV_FSM_PATH,
    ENV_LLM_MAX_TOKENS,
    ENV_LLM_MODEL,
    ENV_LLM_TEMPERATURE,
    MAX_CONTEXT_FILTER_DEPTH,
    is_forbidden_context_entry,
)
from .logging import logger, setup_file_logging

# --------------------------------------------------------------

_REDACTED = "<redacted>"
_CYCLE_PLACEHOLDER = "<redacted:cycle>"


class _RedactionWalk:
    """Per-call state of one ``_redact_context`` walk (never shared)."""

    def __init__(self) -> None:
        self.active: set[int] = set()  # ids of containers on the current path
        # (id, depth) -> redacted result, only for subtrees that met no cycle.
        self.memo: dict[tuple[int, int], object] = {}
        self.met_cycle = False


# DECISION plan-2026-07-18T162030-a02151fe/D-015 [STALE]
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
# DECISION plan-2026-07-19T191147-4b664252/D-014 [STALE]
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
#
# DECISION plan-2026-09-22T080837-8b258a25/D-016
# A cycle is REPLACED by `_CYCLE_PLACEHOLDER`, never dropped (the rule above).
# The (id, depth) memo caches ONLY a subtree whose walk met no placeholder: a
# placeholder depends on the active path, so sharing it is wrong (D-052). Do
# NOT key the memo by id alone (the depth bound truncates per depth), and do
# NOT route this through `utilities.filter_context_tree` (it drops, D-023).
def _redact_value(value: object, depth: int, walk: _RedactionWalk) -> object:
    """Redact one context value; recurses into dicts and into lists/tuples."""
    if not isinstance(value, (dict, list, tuple)):
        return value
    if depth > MAX_CONTEXT_FILTER_DEPTH:
        return _REDACTED
    node = id(value)
    if node in walk.active:
        walk.met_cycle = True
        return _CYCLE_PLACEHOLDER
    memo_key = (node, depth)
    if memo_key in walk.memo:
        return walk.memo[memo_key]
    # Each subtree reports its OWN cycle flag; it is OR-ed back into the
    # parent's afterwards, so an ancestor of a placeholder is never cached.
    outer_met_cycle, walk.met_cycle = walk.met_cycle, False
    walk.active.add(node)
    if isinstance(value, dict):
        result: object = _redact_mapping(value, depth, walk)
    else:
        items = [_redact_value(item, depth + 1, walk) for item in value]
        result = tuple(items) if isinstance(value, tuple) else items
    walk.active.discard(node)
    if not walk.met_cycle:
        walk.memo[memo_key] = result
    walk.met_cycle = walk.met_cycle or outer_met_cycle
    return result


def _redact_mapping(source: dict, depth: int, walk: _RedactionWalk) -> dict:
    """Apply the key match at one level, then recurse into the values kept."""
    result = {}
    for key, value in source.items():
        # See constants.py D-017: a non-str key cannot be pattern-matched, so it
        # bypasses redaction entirely. Log it rather than skipping silently --
        # a `bytes` key here means a secret is about to be written to a log
        # verbatim. Do NOT downgrade to debug.
        if not isinstance(key, str):
            logger.warning(
                f"Log-redaction skipped for context key {key!r} "
                f"({type(key).__name__}): only str keys can be pattern-matched"
            )
            result[key] = _redact_value(value, depth + 1, walk)
        elif is_forbidden_context_entry(key, value):
            result[key] = _REDACTED
        else:
            result[key] = _redact_value(value, depth + 1, walk)
    return result


def _redact_context(data: dict) -> dict:
    """Replace secret-shaped context values before they are written to a log.

    The match is applied at every nesting level — inside nested dicts and
    inside dicts nested in lists/tuples — so ``{"user": {"password": "x"}}``
    is redacted like its flat equivalent. Matched KEYS stay visible and only
    their values become ``"<redacted>"``. Recursion is bounded at
    ``MAX_CONTEXT_FILTER_DEPTH``; anything deeper is redacted wholesale. A
    container already on the recursion path (a cycle) becomes
    ``"<redacted:cycle>"``. Work is linear in the distinct (container, depth)
    pairs, so aliased input does not blow up. The result may share one
    redacted subtree between the places an input subtree was aliased.
    """
    walk = _RedactionWalk()
    walk.active.add(id(data))
    return _redact_mapping(data, 0, walk)


# DECISION plan-2026-09-20T114608-a8e47b88/D-014
# `json.dumps(..., default=...)` on the two log sites below MUST use this
# callable, never the bare builtin `str`. `default=str` calls `str(obj)`,
# which for an arbitrary object falls back to `repr(obj)` -- so a
# secret-bearing object under a benign key (e.g. `Creds(api_key='sk-...')`)
# would serialize its field values verbatim into `logs/`, on the exact
# path whose entire purpose (`_redact_context`/`_redact_mapping` above) is
# to prevent that. Do NOT "simplify" this back to `default=str` even though
# it is one token shorter -- this function returns only the object's TYPE
# NAME, never its value, mirroring `_redact_mapping`'s own established
# WARNING-on-non-str-key pattern two functions above ("a `bytes` key here
# means a secret is about to be written to a log verbatim. Do NOT downgrade
# to debug."). See decisions.md D-014/D-015.
def _json_default(obj: object) -> str:
    """Safe ``json.dumps(default=...)`` fallback for a non-JSON-native
    context value (e.g. ``datetime``, or a caller's custom object).

    Returns a type-name placeholder and WARNs once per call -- never
    ``str(obj)``/``repr(obj)``, since that would leak the object's own
    field values (this is the CLI's redaction path).
    """
    logger.warning(
        "Log-serialization fallback used for a non-JSON-native context "
        f"value of type {type(obj).__name__!r}: only the type name is "
        "logged, not its str()/repr() (which could leak a secret field)."
    )
    return f"<non-serializable: {type(obj).__name__}>"


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
                # default=_json_default (D-014), matching logging.py:90's own
                # _record_to_json: a handler-stored non-JSON-native value
                # (datetime, set, ...) must not crash the CLI's debug/dump
                # logging path -- but the fallback must never str()/repr()
                # the object (see _json_default's own docstring/D-014).
                # Lazy (D-016): no redaction or dump runs unless DEBUG is on.
                logger.opt(lazy=True).debug(
                    "Context data: {}",
                    lambda data=data: json.dumps(
                        _redact_context(data), default=_json_default
                    ),
                )

            except KeyboardInterrupt:
                # C8: Ctrl-C mid-turn ends the session cleanly (the finally
                # below still ends the conversation) instead of a traceback.
                logger.info("Interrupted during a turn")
                return CLI_EXIT_INTERRUPTED
            except Exception as e:
                logger.exception(e)
                # C8: was -1, which the shell sees as 255; every other
                # failure path of the three CLIs exits 1.
                return CLI_EXIT_FAILURE

        data = fsm.get_data(conversation_id)
        logger.opt(lazy=True).info(
            "Data: \n{}",
            lambda: json.dumps(_redact_context(data), indent=3, default=_json_default),
        )
    finally:
        # Clean up when done — always runs even on exception
        fsm.end_conversation(conversation_id)
        logger.info("Conversation ended")

    return CLI_EXIT_OK


# --------------------------------------------------------------
