# src/llm_fsm_reasoning/utilities.py
"""
Utility functions for the reasoning engine.
"""
# Remove: import json
# Remove: from pathlib import Path
from typing import Dict, Any
from llm_fsm.logging import logger
# Import the new Python module containing FSM definitions
from .reasoning_modes import ALL_REASONING_FSMS # CHANGED


def load_fsm_definition(fsm_name: str) -> Dict[str, Any]:
    """
    Load an FSM definition from the Python dictionary store.

    :param fsm_name: Name of the FSM (key in ALL_REASONING_FSMS)
    :return: FSM definition dictionary
    :raises KeyError: if the fsm_name is not found
    """
    # CHANGED: Load from Python dictionary instead of file
    try:
        fsm_dict = ALL_REASONING_FSMS[fsm_name]
        logger.info(f"Loaded FSM definition '{fsm_name}' from Python dictionary store.")
        return fsm_dict # Return a copy to prevent modification of the original
    except KeyError:
        logger.error(f"FSM definition for '{fsm_name}' not found in Python dictionary store.")
        raise KeyError(f"FSM definition for '{fsm_name}' not found.")
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error loading FSM definition '{fsm_name}': {e}")
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
        "hybrid": "hybrid",
        "simple_calculator": "simple_calculator", # Added
        "abductive": "abductive",                 # Added
        "analogical": "analogical",               # Added
        "direct computation": "simple_calculator", # Alias
        "direct calculation": "simple_calculator", # Alias
    }
    # Ensure it maps to a value that exists as a key in ALL_REASONING_FSMS or is a valid ReasoningType value
    return type_map.get(type_str.lower(), "analytical") # Default to analytical