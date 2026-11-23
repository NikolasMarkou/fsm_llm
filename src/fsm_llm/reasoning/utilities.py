"""
Utility functions for the reasoning engine.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.logging import logger

from .constants import ErrorMessages, ReasoningType
from .strategy_files import FSM_NAMES, STRATEGY_ORDER, load_fsm, strategy_facts


def get_reasoning_fsm(fsm_name: str) -> dict[str, Any]:
    """
    Load a reasoning FSM definition from the package data files.

    Not ``fsm_llm.load_fsm_definition`` (an id or path, returning an
    ``FSMDefinition``): this returns a deep copy of the dict in
    ``reasoning/strategies/<name>.json`` (``strategy_files.FSM_NAMES`` lists the names).

    :param fsm_name: Name of the FSM
    :return: FSM definition dictionary
    :raises KeyError: if the fsm_name is not found
    """
    if fsm_name not in FSM_NAMES:
        raise KeyError(ErrorMessages.FSM_NOT_FOUND.format(name=fsm_name))
    fsm_dict = load_fsm(fsm_name)

    logger.debug(f"Loaded FSM definition: {fsm_name}")
    return fsm_dict


def _spellings() -> dict[str, str]:
    """Every spelling ``map_reasoning_type`` accepts: the nine values and the aliases
    each strategy file declares. Built per call, never a module table."""
    table = {value: value for value in STRATEGY_ORDER}
    for value in STRATEGY_ORDER:
        for alias in strategy_facts(value)["aliases"]:
            table[alias] = value
    return table


def lookup_reasoning_type(type_str: str) -> str | None:
    """The strategy a spelling names (a value or an alias, case and padding ignored), or ``None``.

    Contract: never raises for a ``str`` and never logs; ``map_reasoning_type`` is its warning
    wrapper. The one place that tells a known spelling from an unknown one (an unknown spelling
    maps to ``analytical`` there, which is not a signal a caller can read).
    """
    return _spellings().get(type_str.lower().strip())


def map_reasoning_type(type_str: str) -> str:
    """
    Map a reasoning type string to a standardized value.

    :param type_str: Input reasoning type string
    :return: Standardized reasoning type that matches ReasoningType enum values
    """
    mapped = lookup_reasoning_type(type_str)
    if mapped is None:
        logger.warning(f"Unknown reasoning type '{type_str}', defaulting to analytical")
        return ReasoningType.ANALYTICAL.value

    return mapped


def get_available_reasoning_types() -> dict[str, str]:
    """
    Get all available reasoning types with descriptions.

    :return: Dictionary of type -> description
    """
    return {value: strategy_facts(value)["description"] for value in STRATEGY_ORDER}
