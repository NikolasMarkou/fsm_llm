"""
Utility functions for the reasoning engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fsm_llm.logging import logger

from .constants import ErrorMessages, ReasoningType
from .selection import choices_for, find_choice
from .strategy_files import FSM_NAMES, load_fsm

if TYPE_CHECKING:
    from fsm_llm.engine.executables import ExecutableRegistry


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


def lookup_reasoning_type(
    type_str: str, *, registry: ExecutableRegistry | None = None
) -> str | None:
    """The strategy a spelling names (a name or an alias, case and padding ignored), or ``None``.

    Contract: ``registry`` is the ``ExecutableRegistry`` whose eligible strategies
    (``selection.strategy_choices``) are consulted; ``None`` means the nine built-ins. Never
    raises for a ``str`` and never logs a spelling it does not know (``map_reasoning_type`` is
    its warning wrapper); a registry with a skipped strategy logs that WARNING per call, so a
    long-lived caller reads the choices once (``ReasoningEngine`` does).
    """
    choice = find_choice(choices_for(registry), type_str)
    return None if choice is None else choice.name


def map_reasoning_type(
    type_str: str, *, registry: ExecutableRegistry | None = None
) -> str:
    """
    Map a reasoning type string to a standardized value.

    :param type_str: Input reasoning type string
    :param registry: Registry whose strategies are known; ``None`` is the built-in nine
    :return: The strategy name; an unknown spelling is ``analytical`` with a WARNING
    """
    mapped = lookup_reasoning_type(type_str, registry=registry)
    if mapped is None:
        logger.warning(f"Unknown reasoning type '{type_str}', defaulting to analytical")
        return ReasoningType.ANALYTICAL.value

    return mapped


def get_available_reasoning_types(
    *, registry: ExecutableRegistry | None = None
) -> dict[str, str]:
    """
    Get all available reasoning types with descriptions.

    :param registry: Registry whose eligible strategies are listed (the built-ins first, in
        their documented order); ``None`` is the built-in nine
    :return: Dictionary of type name -> one-line description (a new dict on every call)
    """
    return {choice.name: choice.description for choice in choices_for(registry)}
