import os
import re
import json
from typing import Dict, Optional, Any

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .logging import logger
from .definitions import FSMDefinition

# --------------------------------------------------------------

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from text.

    Args:
        text: The text to extract from

    Returns:
        The extracted JSON or None
    """
    logger.debug("Attempting to extract JSON from text")

    # Try to find JSON between code blocks first
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            json_str = json_match.group(1)
            logger.debug("Found JSON in code block, parsing...")
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from code block")
            pass

    # Try to find any JSON object in the text
    json_pattern = r'{[\s\S]*}'
    json_match = re.search(json_pattern, text)
    if json_match:
        try:
            json_str = json_match.group(0)
            logger.debug("Found JSON pattern in text, parsing...")
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from text pattern")
            pass

    logger.warning("Could not extract valid JSON from text")
    return None

def load_fsm_from_file(file_path: str) -> FSMDefinition:
    """
    Load an FSM definition directly from a JSON file.

    Args:
        file_path: Path to the JSON file containing the FSM definition

    Returns:
        The FSM definition object

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the JSON is invalid or doesn't conform to FSM structure
    """
    logger.info(f"Loading FSM definition from file: {file_path}")

    try:
        with open(file_path, 'r') as f:
            fsm_data = json.load(f)

        logger.info(f"Successfully loaded FSM definition: {fsm_data.get('name', 'Unnamed FSM')}")
        return FSMDefinition(**fsm_data)

    except FileNotFoundError:
        error_msg = f"FSM definition file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in FSM definition file: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    except Exception as e:
        error_msg = f"Error loading FSM definition from {file_path}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

# For backward compatibility, keep the original loader but make it also support file paths
def load_fsm_definition(fsm_id_or_path: str) -> FSMDefinition:
    """
    Load an FSM definition either by ID or directly from a file path.

    Args:
        fsm_id_or_path: Either an FSM ID (e.g., "simple_greeting") or a file path

    Returns:
        The FSM definition
    """
    # If the input looks like a file path, try to load it as a file
    if os.path.exists(fsm_id_or_path) or '/' in fsm_id_or_path or '\\' in fsm_id_or_path:
        return load_fsm_from_file(fsm_id_or_path)


    logger.error(f"Unknown FSM ID: {fsm_id_or_path}")
    raise ValueError(f"Unknown FSM ID: {fsm_id_or_path}")