"""Loading and summarising FSM definition files.

Layer ``core`` (plan d454fd62, step 72): split out of the old ``fsm_llm.utilities``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fsm_llm.core.definitions import FSMDefinition, FSMDefinitionNotFoundError
from fsm_llm.logging import logger

# --------------------------------------------------------------
# FSM Definition Loading
# --------------------------------------------------------------


def read_json_object(path: str) -> dict[str, Any]:
    """Read a raw FSM definition file: UTF-8 JSON whose top level is an object.

    The one reader of ``load_fsm_from_file``, ``validate_fsm_from_file`` and
    ``visualize_fsm_from_file``; each keeps its own wording for a failure.

    Raises:
        FileNotFoundError, OSError: the file cannot be opened.
        UnicodeDecodeError: the file is not UTF-8 text.
        json.JSONDecodeError: the text is not JSON.
        ValueError: the JSON is not an object ("FSM definition must be a JSON
            object").
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("FSM definition must be a JSON object")
    return data


def load_fsm_from_file(path: str) -> FSMDefinition:
    """
    Load FSM definition from JSON file with enhanced validation.

    Args:
        path: Path to JSON file containing FSM definition

    Returns:
        Validated FSM definition object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or doesn't conform to FSM structure
    """
    logger.info(f"Loading FSM definition from file: {path}")

    try:
        # Check file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"FSM definition file not found: {path}")

        # Load and parse JSON
        fsm_data = read_json_object(path)

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
        error_msg = f"Error loading FSM definition from {path}: {e!s}"
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
        FSMDefinitionNotFoundError: If a non-path id is given (no registry;
            also a ``ValueError``)
        ValueError: If the file cannot be loaded
    """
    # Check if input looks like a file path
    if (
        os.path.exists(fsm_id_or_path)
        or "/" in fsm_id_or_path
        or "\\" in fsm_id_or_path
        or fsm_id_or_path.endswith(".json")
    ):
        return load_fsm_from_file(fsm_id_or_path)

    # Otherwise treat as FSM ID - there is no FSM registry
    error = FSMDefinitionNotFoundError(fsm_id_or_path)
    logger.error(str(error))
    raise error


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
