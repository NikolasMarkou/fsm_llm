from __future__ import annotations

"""
This module provides utility functions for FSM definition loading,
JSON processing, and other common operations in the enhanced
FSM-LLM framework.

Key Features:
- Enhanced FSM definition loading with validation
- Improved JSON extraction for LLM responses
- Error handling and logging integration
- Support for new FSM definition format
"""

import json
import os
import re
from typing import Any

from .definitions import FSMDefinition

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------
from .logging import logger

# --------------------------------------------------------------
# JSON Processing Utilities
# --------------------------------------------------------------


def _match_brace_partners(text: str, brace_positions: list[int]) -> dict[int, int]:
    """Map each ``{`` position in *text* to the index of the ``}`` balancing it.

    Contract:
        - ``brace_positions`` must be the ascending list of every ``{`` index in
          ``text`` (i.e. ``[m.start() for m in re.finditer(r"\\{", text)]``).
        - Returns ``{start: end}`` for every start whose balanced-brace scan
          terminates. Starts that never balance before end-of-text are ABSENT
          from the mapping (callers must use ``.get()``).
        - Each span is the one a JSON-string/escape-aware scan begun *at that
          start position with a fresh in-string state* would find — not the one
          a single global scan from index 0 would find. The two differ whenever
          an earlier lone ``"`` shifts the global in-string parity, e.g.
          ``'he said " {"a": 1}'``; the per-start reading is the historical
          behavior and is load-bearing (see the D-023/D-002 notes below).
        - Linear rather than quadratic: spans resolve innermost-first
          (descending start order) so an outer scan jumps over an already
          resolved nested span instead of re-walking it.
        - Never raises for any ``str`` input.
    """
    partner: dict[int, int] = {}
    text_len = len(text)

    # Innermost-first: every nested span an outer scan can meet is already known.
    for start_pos in reversed(brace_positions):
        i = start_pos + 1
        in_string = False
        escape_next = False

        while i < text_len:
            char = text[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == "\\":
                escape_next = True
                i += 1
                continue

            if char == '"':
                in_string = not in_string
                i += 1
                continue

            if not in_string:
                if char == "{":
                    nested_end = partner.get(i)
                    if nested_end is None:
                        # The nested span never balances, so neither can this
                        # one — the original scan would have run to end-of-text.
                        break
                    # Resume just past the nested span. Its closing `}` is
                    # outside a string, so the string/escape state is clean.
                    i = nested_end + 1
                    continue
                if char == "}":
                    partner[start_pos] = i
                    break

            i += 1

    return partner


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """
    Enhanced JSON extraction from text with multiple fallback strategies.

    This function handles various formats of JSON that might be returned
    by LLMs, including code blocks, partial JSON, and embedded structures.

    Args:
        text: Text potentially containing JSON data

    Returns:
        Extracted JSON dictionary or None if extraction fails
    """
    if not isinstance(text, str) or not text.strip():
        return None

    logger.debug("Attempting enhanced JSON extraction from text")

    # Strategy 1: Direct JSON parsing
    try:
        parsed: dict[str, Any] = json.loads(text.strip())
        return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from code blocks
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_match:
        try:
            json_str = json_match.group(1).strip()
            logger.debug("Found JSON in code block")
            result: dict[str, Any] = json.loads(json_str)
            return result
        except json.JSONDecodeError:
            logger.debug("Code block JSON parsing failed")

    # Strategy 3: Find balanced JSON objects
    try:
        # Find all potential JSON start positions
        brace_positions = [m.start() for m in re.finditer(r"\{", text)]

        # DECISION plan-2026-07-19T191147-4b664252/D-002
        # The closing partner of every `{` is precomputed ONCE, innermost-first,
        # instead of rescanning text[start_pos:] from every start position. That
        # rescan was O(n^2): 20,000 bare `{` characters took 18.2s of CPU on
        # text that arrives straight from the LLM provider, while the caller
        # holds the per-conversation lock (a real DoS vector).
        #
        # This changed the COMPLEXITY ONLY. Two things below are load-bearing
        # and must NOT be "simplified":
        #   1. The loop stays FIRST-wins — see the D-023 block immediately
        #      below, and decisions.md D-023. A last-wins flip was shipped and
        #      REVERTED once already.
        #   2. `_match_brace_partners` deliberately resolves each span with a
        #      scan begun AT that start position with a fresh in-string state.
        #      Do NOT replace it with one global left-to-right stack pass from
        #      index 0. That looks equivalent and is not: a single earlier lone
        #      `"` flips the global in-string parity, so a real object gets
        #      classified as string content and dropped. Probe that regresses:
        #        'x " {"a":1} " y'  ->  must return {'a': 1}, not None.
        # See decisions.md D-002.
        closing_index = _match_brace_partners(text, brace_positions)

        skip_until = -1
        # DECISION plan-2026-07-18T162030-a02151fe/D-023
        # This strategy is FIRST-wins: it returns on the first successful parse.
        # That disagrees with Strategy 4 below and with
        # llm.py::_extract_content_from_thinking, both of which prefer the LAST
        # object. The divergence is REAL and still OPEN — which helper you hit
        # depends only on whether the provider split the reasoning trace into a
        # separate `thinking` field.
        #
        # Do NOT "fix" it by flipping this loop to last-wins. That was tried
        # (D-021) and REVERTED, because it does not work:
        #   - Strategy 2 (code fence, above) runs FIRST and is also first-match,
        #     so the fenced draft-then-final case that motivated the flip stayed
        #     broken. Only the unfenced variant changed.
        #   - It actively broke the answer-then-example shape:
        #     'The intent is {"intent": "buy"}. Schema: {"intent": "<name>"}'
        #     started returning the EXAMPLE. For classification.py that silently
        #     degrades a correct intent to `fallback_intent`.
        # Resolving this means deciding what these three helpers are FOR (is a
        # trailing object a correction, or a restated schema?), and changing
        # Strategy 2 in step with whatever is chosen. That is a design question,
        # not a one-line tie-break. See decisions.md D-023.
        for start_pos in brace_positions:
            if start_pos <= skip_until:
                continue  # Skip positions inside a previously scanned span

            end_pos = closing_index.get(start_pos)
            if end_pos is None:
                continue  # Never balances before end of text — try next start

            # Found complete JSON object
            json_str = text[start_pos : end_pos + 1]
            try:
                brace_result: dict[str, Any] = json.loads(json_str)
                logger.debug(
                    "Successfully extracted JSON using balanced brace matching"
                )
                return brace_result
            except json.JSONDecodeError:
                # Skip nested positions inside this failed span
                skip_until = end_pos

    except Exception as e:
        logger.debug(f"Error during balanced brace JSON extraction: {e}")

    # Strategy 4: Extract key-value pairs using regex (fallback)
    try:
        # Patterns for simple string values
        string_patterns = [
            r'"message"\s*:\s*"([^"]*)"',
            r'"selected_transition"\s*:\s*"([^"]*)"',
            r'"reasoning"\s*:\s*"([^"]*)"',
        ]

        extracted = {}

        for pattern in string_patterns:
            # Prefer the LAST match: a <think> trace often quotes these keys
            # before the real JSON, and the trailing occurrence is the
            # authoritative value.
            matches = re.findall(pattern, text)
            if matches:
                key = pattern.split('"')[1]
                extracted[key] = matches[-1]

        # For extracted_data, find the key and then use balanced braces
        ed_match = re.search(r'"extracted_data"\s*:\s*\{', text)
        if ed_match:
            # Start balanced brace matching from the opening brace
            brace_start = text.index("{", ed_match.start())
            depth = 0
            in_str = False
            esc = False
            for i, ch in enumerate(text[brace_start:], brace_start):
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                extracted["extracted_data"] = json.loads(
                                    text[brace_start : i + 1]
                                )
                            except json.JSONDecodeError:
                                extracted["extracted_data"] = {}
                            break

        # Only return if we have structurally meaningful keys, not just auxiliary ones.
        # Includes keys used by classification (intent, confidence) and response
        # generation (message, reasoning) callers — not just data extraction.
        meaningful_keys = {
            "selected_transition",
            "extracted_data",
            "message",
            "reasoning",
            "intent",
            "confidence",
        }
        if extracted and (meaningful_keys & extracted.keys()):
            logger.debug(
                f"Extracted JSON using regex fallback: {list(extracted.keys())}"
            )
            return extracted
        elif extracted:
            logger.debug(
                f"Regex fallback found only auxiliary keys {list(extracted.keys())}, treating as failed"
            )

    except Exception as e:
        logger.debug(f"Regex fallback extraction failed: {e}")

    logger.warning("All JSON extraction strategies failed")
    return None


def validate_json_structure(data: dict[str, Any], required_keys: list[str]) -> bool:
    """
    Validate that JSON data contains required keys.

    Args:
        data: JSON data to validate
        required_keys: List of required key names

    Returns:
        True if all required keys are present, False otherwise
    """
    if not isinstance(data, dict):
        return False

    missing_keys = [key for key in required_keys if key not in data]

    if missing_keys:
        logger.debug(f"JSON validation failed: missing keys {missing_keys}")
        return False

    return True


# --------------------------------------------------------------
# FSM Definition Loading
# --------------------------------------------------------------


def load_fsm_from_file(file_path: str) -> FSMDefinition:
    """
    Load FSM definition from JSON file with enhanced validation.

    Args:
        file_path: Path to JSON file containing FSM definition

    Returns:
        Validated FSM definition object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or doesn't conform to FSM structure
    """
    logger.info(f"Loading FSM definition from file: {file_path}")

    try:
        # Check file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FSM definition file not found: {file_path}")

        # Load and parse JSON
        with open(file_path, encoding="utf-8") as f:
            fsm_data = json.load(f)

        # Validate basic structure
        if not isinstance(fsm_data, dict):
            raise ValueError("FSM definition must be a JSON object")

        # Enhance with version info if missing
        if "version" not in fsm_data:
            fsm_data["version"] = "4.1"
            logger.debug("Added default version 4.1 to FSM definition")

        # Create and validate FSM definition
        fsm_definition = FSMDefinition(**fsm_data)

        logger.info(f"Successfully loaded FSM definition: {fsm_definition.name}")
        logger.debug(
            f"FSM contains {len(fsm_definition.states)} states, "
            f"initial state: {fsm_definition.initial_state}"
        )

        return fsm_definition

    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in FSM definition file: {e!s}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Error loading FSM definition from {file_path}: {e!s}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def load_fsm_definition(fsm_id_or_path: str) -> FSMDefinition:
    """
    Load FSM definition by ID or file path with fallback logic.

    Args:
        fsm_id_or_path: Either FSM ID or file path

    Returns:
        Loaded FSM definition

    Raises:
        ValueError: If FSM cannot be loaded
    """
    # Check if input looks like a file path
    if (
        os.path.exists(fsm_id_or_path)
        or "/" in fsm_id_or_path
        or "\\" in fsm_id_or_path
        or fsm_id_or_path.endswith(".json")
    ):
        return load_fsm_from_file(fsm_id_or_path)

    # Otherwise treat as FSM ID - no built-in FSM registry for now
    logger.error(f"Unknown FSM ID: {fsm_id_or_path}")
    raise ValueError(f"Unknown FSM ID: {fsm_id_or_path}")


# --------------------------------------------------------------
# Debug and Development Utilities
# --------------------------------------------------------------


def get_fsm_summary(fsm_definition: FSMDefinition) -> dict[str, Any]:
    """
    Generate summary information about an FSM definition.

    Args:
        fsm_definition: FSM definition to summarize

    Returns:
        Dictionary with summary information
    """
    states = fsm_definition.states

    # Count transitions
    total_transitions = sum(len(state.transitions) for state in states.values())

    # Find terminal states
    terminal_states = [
        state_id for state_id, state in states.items() if not state.transitions
    ]

    # Find states with conditions
    states_with_conditions = [
        state_id
        for state_id, state in states.items()
        if any(transition.conditions for transition in state.transitions)
    ]

    # Find required context keys
    all_required_keys = set()
    for state in states.values():
        if state.required_context_keys:
            all_required_keys.update(state.required_context_keys)

    return {
        "name": fsm_definition.name,
        "version": fsm_definition.version,
        "state_count": len(states),
        "initial_state": fsm_definition.initial_state,
        "terminal_states": terminal_states,
        "terminal_count": len(terminal_states),
        "total_transitions": total_transitions,
        "states_with_conditions": len(states_with_conditions),
        "unique_required_keys": sorted(all_required_keys),
        "has_persona": bool(fsm_definition.persona),
    }
