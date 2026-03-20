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

import os
import re
import json
from typing import Any

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------

from .logging import logger
from .definitions import FSMDefinition


# --------------------------------------------------------------
# JSON Processing Utilities
# --------------------------------------------------------------

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
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            json_str = json_match.group(1).strip()
            logger.debug("Found JSON in code block")
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.debug("Code block JSON parsing failed")

    # Strategy 3: Find balanced JSON objects
    try:
        # Find all potential JSON start positions
        brace_positions = [m.start() for m in re.finditer(r'\{', text)]

        for start_pos in brace_positions:
            brace_count = 0
            in_string = False
            escape_next = False

            for i, char in enumerate(text[start_pos:], start_pos):
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"':
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1

                        if brace_count == 0:
                            # Found complete JSON object
                            json_str = text[start_pos:i + 1]
                            try:
                                result = json.loads(json_str)
                                logger.debug("Successfully extracted JSON using balanced brace matching")
                                return result
                            except json.JSONDecodeError:
                                break  # Try next start position

    except Exception as e:
        logger.debug(f"Error during balanced brace JSON extraction: {e}")

    # Strategy 4: Extract key-value pairs using regex (fallback)
    try:
        # Look for common JSON patterns
        patterns = [
            r'"message"\s*:\s*"([^"]*)"',
            r'"selected_transition"\s*:\s*"([^"]*)"',
            r'"extracted_data"\s*:\s*(\{[^}]*\})',
            r'"reasoning"\s*:\s*"([^"]*)"'
        ]

        extracted = {}

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                key = pattern.split('"')[1]  # Extract key name from pattern
                value = match.group(1)

                # Try to parse nested JSON for extracted_data
                if key == "extracted_data":
                    try:
                        extracted[key] = json.loads(value)
                    except json.JSONDecodeError:
                        extracted[key] = {}
                else:
                    extracted[key] = value

        # Only return if we have structurally meaningful keys, not just auxiliary ones
        meaningful_keys = {"selected_transition", "extracted_data"}
        if extracted and (meaningful_keys & extracted.keys()):
            logger.debug(f"Extracted JSON using regex fallback: {list(extracted.keys())}")
            return extracted
        elif extracted:
            logger.debug(f"Regex fallback found only auxiliary keys {list(extracted.keys())}, treating as failed")

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
        with open(file_path, 'r', encoding='utf-8') as f:
            fsm_data = json.load(f)

        # Validate basic structure
        if not isinstance(fsm_data, dict):
            raise ValueError("FSM definition must be a JSON object")

        # Enhance with version info if missing
        if 'version' not in fsm_data:
            fsm_data['version'] = '4.1'
            logger.debug("Added default version 4.1 to FSM definition")

        # Create and validate FSM definition
        fsm_definition = FSMDefinition(**fsm_data)

        logger.info(f"Successfully loaded FSM definition: {fsm_definition.name}")
        logger.debug(f"FSM contains {len(fsm_definition.states)} states, "
                     f"initial state: {fsm_definition.initial_state}")

        return fsm_definition

    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in FSM definition file: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Error loading FSM definition from {file_path}: {str(e)}"
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
    if (os.path.exists(fsm_id_or_path) or
            '/' in fsm_id_or_path or
            '\\' in fsm_id_or_path or
            fsm_id_or_path.endswith('.json')):
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
        state_id for state_id, state in states.items()
        if not state.transitions
    ]

    # Find states with conditions
    states_with_conditions = [
        state_id for state_id, state in states.items()
        if any(transition.conditions for transition in state.transitions)
    ]

    # Find required context keys
    all_required_keys = set()
    for state in states.values():
        if state.required_context_keys:
            all_required_keys.update(state.required_context_keys)

    return {
        'name': fsm_definition.name,
        'version': fsm_definition.version,
        'state_count': len(states),
        'initial_state': fsm_definition.initial_state,
        'terminal_states': terminal_states,
        'terminal_count': len(terminal_states),
        'total_transitions': total_transitions,
        'states_with_conditions': len(states_with_conditions),
        'unique_required_keys': sorted(all_required_keys),
        'has_persona': bool(fsm_definition.persona),
    }
