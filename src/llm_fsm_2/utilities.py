# /utilities.py

"""
Enhanced utilities for 2-Pass LLM-FSM Architecture.

This module provides utility functions for FSM definition loading,
JSON processing, and other common operations in the enhanced
LLM-FSM framework.

Key Features:
- Enhanced FSM definition loading with validation
- Improved JSON extraction for LLM responses
- Error handling and logging integration
- Support for new FSM definition format
"""

import os
import re
import json
from typing import Dict, Optional, Any, List

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------

from .logging import logger
from .definitions import FSMDefinition


# --------------------------------------------------------------
# JSON Processing Utilities
# --------------------------------------------------------------

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
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

                if char == '"' and not escape_next:
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

        if extracted:
            logger.debug(f"Extracted JSON using regex fallback: {list(extracted.keys())}")
            return extracted

    except Exception as e:
        logger.debug(f"Regex fallback extraction failed: {e}")

    logger.warning("All JSON extraction strategies failed")
    return None


def validate_json_structure(data: Dict[str, Any], required_keys: List[str]) -> bool:
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


def sanitize_json_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize JSON values to prevent issues with serialization.

    Args:
        data: JSON data to sanitize

    Returns:
        Sanitized JSON data
    """
    if not isinstance(data, dict):
        return data

    sanitized = {}

    for key, value in data.items():
        if isinstance(value, str):
            # Remove potential control characters and normalize whitespace
            sanitized[key] = re.sub(r'\s+', ' ', value.strip())
        elif isinstance(value, dict):
            sanitized[key] = sanitize_json_values(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_json_values(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value

    return sanitized


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
            fsm_data['version'] = '4.0'
            logger.debug("Added default version 4.0 to FSM definition")

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
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading FSM definition from {file_path}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


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
# FSM Validation Utilities
# --------------------------------------------------------------

def validate_fsm_states(fsm_data: Dict[str, Any]) -> List[str]:
    """
    Validate FSM state structure and return any issues found.

    Args:
        fsm_data: FSM definition data

    Returns:
        List of validation issues (empty if valid)
    """
    issues = []

    if 'states' not in fsm_data:
        issues.append("Missing 'states' field")
        return issues

    states = fsm_data['states']
    if not isinstance(states, dict):
        issues.append("States field must be a dictionary")
        return issues

    initial_state = fsm_data.get('initial_state')
    if not initial_state:
        issues.append("Missing 'initial_state' field")
    elif initial_state not in states:
        issues.append(f"Initial state '{initial_state}' not found in states")

    # Validate each state
    for state_id, state_data in states.items():
        if not isinstance(state_data, dict):
            issues.append(f"State '{state_id}' must be a dictionary")
            continue

        # Check required fields
        required_fields = ['description', 'purpose']
        for field in required_fields:
            if field not in state_data:
                issues.append(f"State '{state_id}' missing required field '{field}'")

        # Validate transitions
        transitions = state_data.get('transitions', [])
        if not isinstance(transitions, list):
            issues.append(f"State '{state_id}' transitions must be a list")
            continue

        for i, transition in enumerate(transitions):
            if not isinstance(transition, dict):
                issues.append(f"State '{state_id}' transition {i} must be a dictionary")
                continue

            if 'target_state' not in transition:
                issues.append(f"State '{state_id}' transition {i} missing 'target_state'")
            elif transition['target_state'] not in states:
                issues.append(
                    f"State '{state_id}' transition {i} targets non-existent state '{transition['target_state']}'")

    return issues


def enhance_fsm_definition(fsm_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance FSM definition with defaults and improvements.

    Args:
        fsm_data: Original FSM definition data

    Returns:
        Enhanced FSM definition data
    """
    enhanced = fsm_data.copy()

    # Add version if missing
    if 'version' not in enhanced:
        enhanced['version'] = '4.0'

    # Add transition evaluation mode if missing
    if 'transition_evaluation_mode' not in enhanced:
        enhanced['transition_evaluation_mode'] = 'hybrid'

    # Enhance states
    states = enhanced.get('states', {})
    for state_id, state_data in states.items():
        # Ensure state has ID field
        if 'id' not in state_data:
            state_data['id'] = state_id

        # Add default transition properties
        transitions = state_data.get('transitions', [])
        for transition in transitions:
            if 'priority' not in transition:
                transition['priority'] = 100
            if 'is_deterministic' not in transition:
                transition['is_deterministic'] = True

    return enhanced


# --------------------------------------------------------------
# Text Processing Utilities
# --------------------------------------------------------------

def normalize_state_name(name: str) -> str:
    """
    Normalize state name to valid identifier format.

    Args:
        name: Original state name

    Returns:
        Normalized state name
    """
    if not name:
        return "unnamed_state"

    # Convert to lowercase and replace spaces/special chars with underscores
    normalized = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())

    # Ensure starts with letter or underscore
    if normalized and normalized[0].isdigit():
        normalized = f"state_{normalized}"

    # Remove multiple consecutive underscores
    normalized = re.sub(r'_+', '_', normalized)

    # Remove leading/trailing underscores
    normalized = normalized.strip('_')

    return normalized or "unnamed_state"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncating

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    truncate_length = max_length - len(suffix)
    if truncate_length <= 0:
        return suffix[:max_length]

    return text[:truncate_length] + suffix


def extract_quoted_strings(text: str) -> List[str]:
    """
    Extract quoted strings from text.

    Args:
        text: Text to search for quoted strings

    Returns:
        List of quoted strings (without quotes)
    """
    pattern = r'"([^"]*)"'
    matches = re.findall(pattern, text)

    return matches


# --------------------------------------------------------------
# Debug and Development Utilities
# --------------------------------------------------------------

def get_fsm_summary(fsm_definition: FSMDefinition) -> Dict[str, Any]:
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
        'transition_evaluation_mode': fsm_definition.transition_evaluation_mode
    }