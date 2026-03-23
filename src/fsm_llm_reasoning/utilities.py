from __future__ import annotations

"""
Utility functions for the reasoning engine.
Enhanced with better error handling and type mapping.
"""
from typing import Any

from fsm_llm.logging import logger

from .constants import ErrorMessages, ReasoningType
from .reasoning_modes import ALL_REASONING_FSMS


def load_fsm_definition(fsm_name: str) -> dict[str, Any]:
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
        "math": ReasoningType.SIMPLE_CALCULATOR.value,
        "calculator": ReasoningType.SIMPLE_CALCULATOR.value,
        "compute": ReasoningType.SIMPLE_CALCULATOR.value,
        "numeric": ReasoningType.SIMPLE_CALCULATOR.value,
        # Aliases for analytical
        "analyze": ReasoningType.ANALYTICAL.value,
        "analysis": ReasoningType.ANALYTICAL.value,
        # Aliases for deductive
        "deduce": ReasoningType.DEDUCTIVE.value,
        "deduction": ReasoningType.DEDUCTIVE.value,
        "logic": ReasoningType.DEDUCTIVE.value,
        "logical": ReasoningType.DEDUCTIVE.value,
        "syllogism": ReasoningType.DEDUCTIVE.value,
        # Aliases for inductive
        "induce": ReasoningType.INDUCTIVE.value,
        "induction": ReasoningType.INDUCTIVE.value,
        "pattern": ReasoningType.INDUCTIVE.value,
        "observation": ReasoningType.INDUCTIVE.value,
        "generalize": ReasoningType.INDUCTIVE.value,
        # Aliases for creative
        "create": ReasoningType.CREATIVE.value,
        "innovation": ReasoningType.CREATIVE.value,
        "innovative": ReasoningType.CREATIVE.value,
        "brainstorm": ReasoningType.CREATIVE.value,
        "divergent": ReasoningType.CREATIVE.value,
        # Aliases for critical
        "critique": ReasoningType.CRITICAL.value,
        "evaluation": ReasoningType.CRITICAL.value,
        "evaluate": ReasoningType.CRITICAL.value,
        "assessment": ReasoningType.CRITICAL.value,
        "judge": ReasoningType.CRITICAL.value,
        # Aliases for hybrid
        "mixed": ReasoningType.HYBRID.value,
        "combined": ReasoningType.HYBRID.value,
        "multi": ReasoningType.HYBRID.value,
        # Aliases for abductive
        "explain": ReasoningType.ABDUCTIVE.value,
        "explanation": ReasoningType.ABDUCTIVE.value,
        "hypothesis": ReasoningType.ABDUCTIVE.value,
        "abduction": ReasoningType.ABDUCTIVE.value,
        "diagnose": ReasoningType.ABDUCTIVE.value,
        # Aliases for analogical
        "analogy": ReasoningType.ANALOGICAL.value,
        "comparison": ReasoningType.ANALOGICAL.value,
        "compare": ReasoningType.ANALOGICAL.value,
        "similar": ReasoningType.ANALOGICAL.value,
    }

    # Normalize input
    normalized = type_str.lower().strip()

    # Return mapped value or default to analytical
    mapped_value = type_map.get(normalized, ReasoningType.ANALYTICAL.value)

    if normalized not in type_map:
        logger.warning(f"Unknown reasoning type '{type_str}', defaulting to analytical")

    return mapped_value


def get_available_reasoning_types() -> dict[str, str]:
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
        ReasoningType.ANALOGICAL: "Transferring insights through analogies",
    }

    return {rt.value: descriptions[rt] for rt in descriptions}
