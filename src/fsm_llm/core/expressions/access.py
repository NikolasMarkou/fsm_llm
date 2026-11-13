"""Variable access: ``get_var``, the one "missing" predicate and the data-access operator handlers."""

from __future__ import annotations

from typing import Any

from fsm_llm.core.definitions import TransitionEvaluationError
from fsm_llm.logging import logger


def get_var(data: dict[str, Any], var_name: str, not_found: Any = None) -> Any:
    """
    Get variable value from data dictionary using dot notation.

    This function supports nested object access using dot notation and
    array indexing using numeric keys.

    Args:
        data: Dictionary containing the data to search
        var_name: Variable name to look up, supports dot notation for nested access
        not_found: Value to return if the variable is not found (default: None)

    Returns:
        Any: The value of the variable or not_found if not present

    Example:
        >>> data = {"user": {"name": "Alice", "scores": [95, 87, 92]}}
        >>> get_var(data, "user.name")
        'Alice'
        >>> get_var(data, "user.scores.1")
        87
        >>> get_var(data, "user.age", "unknown")
        'unknown'

    Note:
        - Supports both dictionary key access and list index access
        - Numeric keys are automatically converted for list indexing
        - Returns not_found value for any access failures
    """
    if var_name == "" or var_name is None:
        return data

    current_data: Any = data
    keys = str(var_name).split(".")

    for _i, key in enumerate(keys):
        try:
            # Try dictionary key access first
            current_data = current_data[key]
        except (TypeError, KeyError):
            try:
                # Try numeric index access for lists/arrays
                current_data = current_data[int(key)]
            except (ValueError, IndexError, TypeError, KeyError):
                return not_found

    return current_data


_ABSENT = object()


def is_missing(data: dict[str, Any], var_name: Any) -> bool:
    """True when ``var_name`` is absent from ``data``, or holds ``None`` or ``""``.

    Interface contract: ``data`` any mapping, ``var_name`` a (dotted) path as
    accepted by ``get_var``. Never raises. ``0``, ``False`` and empty
    containers are present. The single "missing" predicate behind the
    ``missing`` / ``missing_some`` operators and ``TransitionCondition.
    requires_context_keys`` (D-013 of plan-2026-09-21T203800-8a03483a).
    """
    value = get_var(data, var_name, _ABSENT)
    return value is _ABSENT or value is None or (isinstance(value, str) and not value)


def missing(data: dict[str, Any], *args: Any) -> list[str]:
    """
    Implement the 'missing' operator for finding missing variables.

    This function checks if the specified variables exist in the data and
    returns a list of those that don't exist.

    Args:
        data: Dictionary containing the data to check
        *args: Variable names to check for existence. Can be individual strings
               or a single list of strings.

    Returns:
        list[str]: List of variable names that are missing from the data

    Example:
        >>> data = {"a": 1, "c": 3}
        >>> missing(data, "a", "b", "c")
        ['b']
        >>> missing(data, ["a", "b", "c"])
        ['b']

    Note:
        - Supports both individual arguments and a single list argument
        - A variable is missing per ``is_missing`` (absent, None or ``""``),
          with dot notation
        - Empty list means all variables are present
    """
    # Handle case where args is a single list
    var_names: tuple[Any, ...] | list[Any] = args
    if args and isinstance(args[0], list):
        var_names = args[0]

    return [arg for arg in var_names if is_missing(data, arg)]


def missing_some(data: dict[str, Any], min_required: int, args: list[str]) -> list[str]:
    """
    Implement the 'missing_some' operator for conditional missing variable check.

    This function checks if at least min_required variables are present in the data.
    If the minimum requirement is met, it returns an empty list. Otherwise,
    it returns the list of missing variables.

    Args:
        data: Dictionary containing the data to check
        min_required: Minimum number of variables that must be present
        args: List of variable names to check

    Returns:
        list[str]: Empty list if minimum requirement is met, otherwise list of missing variables

    Example:
        >>> data = {"a": 1, "c": 3}
        >>> missing_some(data, 2, ["a", "b", "c"])
        []  # 2 of 3 variables are present, requirement met
        >>> missing_some(data, 3, ["a", "b", "c"])
        ['b']  # Only 2 of 3 present, requirement not met

    Note:
        - Returns empty list as soon as minimum requirement is satisfied
        - If min_required < 1, always returns empty list
        - A variable is missing per ``is_missing`` (absent, None or ``""``)
    """
    if min_required < 1:
        return []

    found = 0
    missing_vars = []

    # Guard: treat string arg as single var name, not iterable of chars
    if isinstance(args, str):
        args = [args]

    for arg in args:
        if is_missing(data, arg):
            missing_vars.append(arg)
        else:
            found += 1
            # Early exit if we've found enough variables
            if found >= min_required:
                return []

    return missing_vars


# These operators need direct access to the data dict (no recursive
# evaluation of their arguments), so they are dispatched separately.


def _op_var(values: list, data: dict[str, Any], _depth: int) -> Any:
    """Handle the 'var' operator — variable access with dot notation."""
    if not values:
        logger.error("var operator requires at least one argument")
        return None
    # DECISION plan-2026-07-20T040150-876e7164/D-012 [STALE]
    # `values[2:]` stays DROPPED. This is observability only: log, then carry on
    # with `values[0]`/`values[1]` exactly as before. Do NOT "finish the job" by
    # returning `None` or raising here — `_op_var` is the most-used operator in
    # every FSM definition in this repo, so a hard failure would turn a typo in
    # a rarely-taken transition condition into a broken conversation, and the
    # three sibling handlers that bound their arity (`_op_missing_some`,
    # `_op_has_context`, `_op_context_length`) all return a NEUTRAL value rather
    # than raising too. What was wrong was the SILENCE, not the leniency.
    # See decisions.md D-012.
    if len(values) > 2:
        logger.error(f"var operator requires 1 or 2 arguments, got {len(values)}")
    var_name = values[0]
    default = values[1] if len(values) > 1 else None
    return get_var(data, var_name, default)


def _op_missing(values: list, data: dict[str, Any], _depth: int) -> list[str]:
    """Handle the 'missing' operator — find missing variables."""
    return missing(data, *values)


def _op_missing_some(values: list, data: dict[str, Any], _depth: int) -> list[str]:
    """Handle the 'missing_some' operator — conditional missing check."""
    if len(values) != 2:
        logger.error(f"missing_some requires exactly 2 arguments, got {len(values)}")
        return []
    minimum = values[0]
    if isinstance(minimum, float) and minimum.is_integer():
        minimum = int(minimum)
    if not isinstance(minimum, int) or isinstance(minimum, bool):
        # A clean, catchable error (the condition fails) instead of a bare
        # TypeError from `min_required < 1` (audit B8).
        raise TransitionEvaluationError(
            f"missing_some minimum must be an integer, got {type(values[0]).__name__}"
        )
    return missing_some(data, minimum, values[1])
