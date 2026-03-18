"""
Utility functions for the reasoning engine.
Enhanced with better error handling and type mapping.
"""
from typing import Dict, Any, Optional
from llm_fsm.logging import logger

from .reasoning_modes import ALL_REASONING_FSMS
from .constants import ReasoningType, ErrorMessages, LogMessages


def load_fsm_definition(fsm_name: str) -> Dict[str, Any]:
    """
    Load an FSM definition from the Python dictionary store.

    :param fsm_name: Name of the FSM
    :return: FSM definition dictionary
    :raises KeyError: if the fsm_name is not found
    """
    try:
        fsm_dict = ALL_REASONING_FSMS.get(fsm_name)
        if not fsm_dict:
            raise KeyError(ErrorMessages.FSM_NOT_FOUND.format(name=fsm_name))

        logger.debug(f"Loaded FSM definition: {fsm_name}")
        return fsm_dict.copy()  # Return copy to prevent modification

    except Exception as e:
        logger.error(f"Error loading FSM '{fsm_name}': {e}")
        raise


def map_reasoning_type(type_str: str) -> str:
    """
    Map a reasoning type string to a standardized value.

    :param type_str: Input reasoning type string
    :return: Standardized reasoning type that matches ReasoningType enum values
    """
    # Comprehensive mapping including aliases
    type_map = {
        # Direct mappings
        "analytical": ReasoningType.ANALYTICAL.value,
        "deductive": ReasoningType.DEDUCTIVE.value,
        "inductive": ReasoningType.INDUCTIVE.value,
        "creative": ReasoningType.CREATIVE.value,
        "critical": ReasoningType.CRITICAL.value,
        "hybrid": ReasoningType.HYBRID.value,
        "simple_calculator": ReasoningType.SIMPLE_CALCULATOR.value,
        "abductive": ReasoningType.ABDUCTIVE.value,
        "analogical": ReasoningType.ANALOGICAL.value,

        # Aliases for calculator
        "direct computation": ReasoningType.SIMPLE_CALCULATOR.value,
        "direct calculation": ReasoningType.SIMPLE_CALCULATOR.value,
        "arithmetic": ReasoningType.SIMPLE_CALCULATOR.value,
        "calculation": ReasoningType.SIMPLE_CALCULATOR.value,

        # Aliases for other types
        "analyze": ReasoningType.ANALYTICAL.value,
        "analysis": ReasoningType.ANALYTICAL.value,
        "deduce": ReasoningType.DEDUCTIVE.value,
        "deduction": ReasoningType.DEDUCTIVE.value,
        "induce": ReasoningType.INDUCTIVE.value,
        "induction": ReasoningType.INDUCTIVE.value,
        "create": ReasoningType.CREATIVE.value,
        "innovation": ReasoningType.CREATIVE.value,
        "critique": ReasoningType.CRITICAL.value,
        "evaluation": ReasoningType.CRITICAL.value,
        "mixed": ReasoningType.HYBRID.value,
        "combined": ReasoningType.HYBRID.value,
        "explain": ReasoningType.ABDUCTIVE.value,
        "explanation": ReasoningType.ABDUCTIVE.value,
        "analogy": ReasoningType.ANALOGICAL.value,
        "comparison": ReasoningType.ANALOGICAL.value
    }

    # Normalize input
    normalized = type_str.lower().strip()

    # Return mapped value or default to analytical
    mapped_value = type_map.get(normalized, ReasoningType.ANALYTICAL.value)

    if normalized not in type_map:
        logger.warning(f"Unknown reasoning type '{type_str}', defaulting to analytical")

    return mapped_value


def validate_reasoning_type(type_str: str) -> bool:
    """
    Check if a reasoning type string is valid.

    :param type_str: Reasoning type string to validate
    :return: True if valid, False otherwise
    """
    try:
        mapped = map_reasoning_type(type_str)
        # Check if it's a valid enum value
        ReasoningType(mapped)
        return True
    except ValueError:
        return False


def get_available_reasoning_types() -> Dict[str, str]:
    """
    Get all available reasoning types with descriptions.

    :return: Dictionary of type -> description
    """
    descriptions = {
        ReasoningType.SIMPLE_CALCULATOR: "Direct arithmetic calculations",
        ReasoningType.ANALYTICAL: "Breaking down complex problems into components",
        ReasoningType.DEDUCTIVE: "Deriving specific conclusions from general principles",
        ReasoningType.INDUCTIVE: "Finding patterns from specific observations",
        ReasoningType.CREATIVE: "Generating novel solutions through divergent thinking",
        ReasoningType.CRITICAL: "Evaluating arguments and evidence",
        ReasoningType.HYBRID: "Combining multiple reasoning approaches",
        ReasoningType.ABDUCTIVE: "Finding the best explanation for observations",
        ReasoningType.ANALOGICAL: "Transferring insights through analogies"
    }

    return {rt.value: descriptions[rt] for rt in descriptions}


def estimate_context_size(context: Dict[str, Any]) -> int:
    """
    Estimate the size of a context dictionary in characters.

    :param context: Context dictionary
    :return: Estimated size in characters
    """
    try:
        import json
        return len(json.dumps(context, default=str))
    except Exception:
        # Fallback to rough estimation
        return sum(len(str(k)) + len(str(v)) for k, v in context.items())


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length.

    :param text: Text to truncate
    :param max_length: Maximum length
    :param suffix: Suffix to add if truncated
    :return: Truncated string
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary with type checking.

    :param dictionary: Dictionary to get from
    :param key: Key to retrieve
    :param default: Default value if key not found
    :return: Value or default
    """
    try:
        return dictionary.get(key, default)
    except (AttributeError, TypeError):
        return default


def format_list_items(items: list, max_items: int = 5) -> str:
    """
    Format a list of items for display.

    :param items: List of items
    :param max_items: Maximum items to show
    :return: Formatted string
    """
    if not items:
        return "None"

    if len(items) <= max_items:
        return ", ".join(str(item) for item in items)

    shown = ", ".join(str(item) for item in items[:max_items])
    return f"{shown}, ... ({len(items) - max_items} more)"