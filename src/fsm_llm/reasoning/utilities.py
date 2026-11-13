"""
Utility functions for the reasoning engine.
"""

from __future__ import annotations

import copy
from typing import Any

from fsm_llm.logging import logger

from .constants import STRATEGIES, ErrorMessages, ReasoningType
from .reasoning_modes import ALL_REASONING_FSMS


def get_reasoning_fsm(fsm_name: str) -> dict[str, Any]:
    """
    Load a reasoning FSM definition from the Python dictionary store.

    Not ``fsm_llm.load_fsm_definition`` (an id or path, returning an
    ``FSMDefinition``): this returns a deep copy of the dict in
    ``ALL_REASONING_FSMS``.

    :param fsm_name: Name of the FSM
    :return: FSM definition dictionary
    :raises KeyError: if the fsm_name is not found
    """
    fsm_dict = ALL_REASONING_FSMS.get(fsm_name)
    if not fsm_dict:
        raise KeyError(ErrorMessages.FSM_NOT_FOUND.format(name=fsm_name))

    logger.debug(f"Loaded FSM definition: {fsm_name}")
    return copy.deepcopy(fsm_dict)


# Every spelling ``map_reasoning_type`` accepts: the nine values and the aliases
# of ``STRATEGIES``.
_TYPE_MAP: dict[str, str] = {
    **{value: value for value in STRATEGIES},
    **{alias: value for value, spec in STRATEGIES.items() for alias in spec.aliases},
}


def map_reasoning_type(type_str: str) -> str:
    """
    Map a reasoning type string to a standardized value.

    :param type_str: Input reasoning type string
    :return: Standardized reasoning type that matches ReasoningType enum values
    """
    normalized = type_str.lower().strip()

    if normalized not in _TYPE_MAP:
        logger.warning(f"Unknown reasoning type '{type_str}', defaulting to analytical")
        return ReasoningType.ANALYTICAL.value

    return _TYPE_MAP[normalized]


def get_available_reasoning_types() -> dict[str, str]:
    """
    Get all available reasoning types with descriptions.

    :return: Dictionary of type -> description
    """
    return {value: spec.description for value, spec in STRATEGIES.items()}
