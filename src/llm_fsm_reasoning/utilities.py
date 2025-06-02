"""
Utility functions for the reasoning engine.
"""
import json
from pathlib import Path
from typing import Dict, Any
from llm_fsm.logging import logger


def load_fsm_definition(fsm_name: str) -> Dict[str, Any]:
    """
    Load an FSM definition from the reasoning_fsms directory.

    :param fsm_name: Name of the FSM file (without .json extension)
    :return: FSM definition dictionary
    """
    fsm_path = Path(__file__).parent / "fsms" / f"{fsm_name}.json"

    try:
        with open(fsm_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"FSM definition not found: {fsm_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in FSM definition {fsm_path}: {e}")
        raise


def map_reasoning_type(type_str: str) -> str:
    """
    Map a reasoning type string to a standardized value.

    :param type_str: Input reasoning type string
    :return: Standardized reasoning type
    """
    type_map = {
        "analytical": "analytical",
        "deductive": "deductive",
        "inductive": "inductive",
        "creative": "creative",
        "critical": "critical",
        "hybrid": "hybrid"
    }

    return type_map.get(type_str.lower(), "analytical")