"""The recursive evaluator: ``evaluate_logic`` and the handlers that evaluate their operands."""

from __future__ import annotations

from typing import Any

from fsm_llm.constants import ALLOWED_JSONLOGIC_OPERATIONS, MAX_JSONLOGIC_DEPTH
from fsm_llm.core.definitions import TransitionEvaluationError
from fsm_llm.logging import logger

from .access import _op_missing, _op_missing_some, _op_var, get_var
from .registry import _SHORT_CIRCUIT_OPERATORS, operations
from .undefined import (
    _NONE_IS_FALSE_OPERATORS,
    _UNDEFINED,
    _UNDEFINED_IS_FALSE_OPERATORS,
    _UNDEFINED_PROPAGATING_OPERATORS,
)


def _op_has_context(values: list, data: dict[str, Any], _depth: int) -> bool:
    """Handle the 'has_context' operator — check key existence in context.

    Supports two forms:
    - Single argument: {"has_context": "key"} — checks if key exists in data
    - Two arguments: {"has_context": [context_obj, "key"]} — checks key in a specific dict

    Returns True if the key exists in the dict, regardless of value (including
    None, False, empty string, empty list).
    """
    if len(values) == 1:
        # Single-argument shorthand: check key in the top-level data dict
        key = evaluate_logic(values[0], data, _depth + 1)
        if not isinstance(key, str):
            logger.warning(f"has_context expected string key, got {type(key)}")
            return False
        return key in data
    if len(values) == 2:
        context_obj = evaluate_logic(values[0], data, _depth + 1)
        key = evaluate_logic(values[1], data, _depth + 1)
        if not isinstance(context_obj, dict):
            logger.warning(
                f"has_context expected dict as first argument, got {type(context_obj)}"
            )
            return False
        return key in context_obj
    logger.error(f"has_context requires 1 or 2 arguments, got {len(values)}")
    return False


def _op_context_length(values: list, data: dict[str, Any], _depth: int) -> int:
    """Handle the 'context_length' operator — get length of context value."""
    if len(values) != 2:
        logger.error(f"context_length requires exactly 2 arguments, got {len(values)}")
        return 0
    context_obj = evaluate_logic(values[0], data, _depth + 1)
    path = evaluate_logic(values[1], data, _depth + 1)
    if not isinstance(context_obj, dict):
        logger.warning(
            f"context_length expected dict as first argument, got {type(context_obj)}"
        )
        return 0
    value = get_var(context_obj, path, [])
    if isinstance(value, list | str | dict):
        return len(value)
    return 0


#: Dispatch dict for operators that need direct data access. `var`, `missing`
#: and `missing_some` never evaluate their arguments; that set is
#: `constants.JSONLOGIC_RAW_ARGUMENT_OPERATIONS`, which the load-time walk in
#: core/definitions/transition.py reads. Keep them in step (pinned by the step-10 audit tests).
_data_operators: dict[str, Any] = {
    "var": _op_var,
    "missing": _op_missing,
    "missing_some": _op_missing_some,
    "has_context": _op_has_context,
    "context_length": _op_context_length,
}


def _evaluate_short_circuit(
    operator: str, values: list, data: dict[str, Any], _depth: int
) -> Any:
    """Lazily evaluate the short-circuiting operators ``and``/``or``/``if``.

    # DECISION plan-2026-07-21T045419-9925aa3a/D-006
    # Do NOT revert this to eager evaluation (the old `evaluated_values`
    # list-comp evaluated EVERY operand before dispatch). Guard idioms such as
    # `{"and": [{"!=": [{"var":"x"},0]}, {"<": [{"/":[100,{"var":"x"}]},5]}]}`
    # rely on the untaken branch NEVER being evaluated: with `x=0` the div-by-
    # zero operand must be skipped and the whole condition return a clean
    # `False`, not raise. Eager evaluation silently turned every such guard into
    # an error-swallowed `False`. See decisions.md D-006.
    #
    # Do NOT wrap the sub-evaluations in a swallow-all try/except either: the
    # TAKEN branch keeps the D-012/D-023 total contract — a div/mod-by-zero on
    # the path actually taken still raises `TransitionEvaluationError` exactly
    # as before (Invariant 4). Only the UNTAKEN branch is skipped.

    Contract:
        operator: one of ``"and"``/``"or"``/``"if"`` (caller guarantees membership).
        values: the RAW, un-evaluated operand list from the logic expression.
        data / _depth: threaded unchanged into each operand's ``evaluate_logic``.
    Returns the decisive operand value — first falsy (``and``) / first truthy
    (``or``), else the last operand; ``and: [] -> True``, ``or: [] -> False`` —
    or, for ``if``, the matched branch (else the trailing default, else ``None``).
    Failure mode: propagates any error raised while evaluating a TAKEN operand.
    """
    if operator == "and":
        if not values:
            return True
        result: Any = True
        for val in values:
            result = evaluate_logic(val, data, _depth + 1)
            if not result:
                return result
        return result

    if operator == "or":
        if not values:
            return False
        result = False
        for val in values:
            result = evaluate_logic(val, data, _depth + 1)
            if result:
                return result
        return result

    # operator == "if": condition/result pairs, evaluated lazily left-to-right.
    for i in range(0, len(values) - 1, 2):
        condition = evaluate_logic(values[i], data, _depth + 1)
        if condition:
            return evaluate_logic(values[i + 1], data, _depth + 1)
    # Odd arg count -> trailing default; even -> no default -> None.
    if len(values) % 2:
        return evaluate_logic(values[-1], data, _depth + 1)
    return None


def evaluate_logic(
    logic: Any, data: dict[str, Any] | None = None, _depth: int = 0
) -> Any:
    """
    Evaluate a JsonLogic expression against provided data.

    Arithmetic on a missing operand evaluates to a module-private UNDEFINED
    value inside the expression (see the module docstring); the outermost
    call (``_depth == 0``) returns it as ``None``.

    This is the main entry point for evaluating JsonLogic expressions. It recursively
    processes the expression structure and applies the appropriate operators.

    Args:
        logic: The JsonLogic expression to evaluate. Can be:
            - Primitive values (str, int, float, bool, None) - returned as-is
            - Dict with single key-value pair representing operation and arguments
        data: The data object to evaluate against (default: empty dict)

    Returns:
        Any: The result of evaluating the expression

    Raises:
        Exception: Various exceptions possible during evaluation (logged but not re-raised)

    Example:
        Basic comparisons::

            >>> evaluate_logic({"==": [1, 1]})
            True
            >>> evaluate_logic({"!=": ["hello", "world"]})
            True

        Variable access::

            >>> data = {"user": {"name": "Alice", "age": 30}}
            >>> evaluate_logic({"var": "user.name"}, data)
            'Alice'

        Complex conditions::

            >>> logic = {
            ...     "and": [
            ...         {">=": [{"var": "age"}, 18]},
            ...         {"==": [{"var": "has_id"}, True]}
            ...     ]
            ... }
            >>> evaluate_logic(logic, {"age": 20, "has_id": True})
            True

    Note:
        JsonLogic expressions are structured as:

        - Primitive values are returned as-is
        - Objects with a single key represent operations:
          - Key: operator name (e.g., "and", "==", "var")
          - Value: arguments to that operator (single value or list)

        Special operators with direct data access:

        - "var": Variable access with dot notation
        - "missing": Find missing variables
        - "missing_some": Conditional missing check
        - "has_context": Check key existence
        - "context_length": Get length of context value
    """
    result = _evaluate_logic(logic, data, _depth)
    if _depth == 0 and result is _UNDEFINED:
        return None
    return result


def _evaluate_logic(logic: Any, data: dict[str, Any] | None, _depth: int) -> Any:
    """Body of ``evaluate_logic``; may return ``_UNDEFINED``. Recursion goes
    through ``evaluate_logic`` with ``_depth + 1`` (no conversion there)."""
    # Guard against deeply nested logic (DoS protection)
    if _depth > MAX_JSONLOGIC_DEPTH:
        raise TransitionEvaluationError(
            f"JsonLogic recursion depth exceeded ({MAX_JSONLOGIC_DEPTH}). "
            f"Expression is too deeply nested."
        )

    # Handle primitive values (base case for recursion)
    if logic is None or not isinstance(logic, dict):
        return logic

    # Ensure we have data to work with
    data = data or {}

    # Extract operator and arguments from the logic expression
    if not logic:
        logger.warning("Empty logic expression provided")
        return None

    if len(logic) > 1:
        raise TransitionEvaluationError(
            f"Invalid JsonLogic expression: has multiple keys {list(logic.keys())}. "
            f"JsonLogic requires exactly one key per operation."
        )

    operator = next(iter(logic.keys()))
    values = logic[operator]

    # Enforce allowed operations
    if operator not in ALLOWED_JSONLOGIC_OPERATIONS:
        raise TransitionEvaluationError(
            f"Disallowed JsonLogic operation: '{operator}'. "
            f"Allowed: {sorted(ALLOWED_JSONLOGIC_OPERATIONS)}"
        )

    # Convert single values to list for consistent handling
    if not isinstance(values, list | tuple):
        values = [values]

    # Data-access operators get raw values + data (no recursive eval)
    data_op = _data_operators.get(operator)
    if data_op is not None:
        return data_op(values, data, _depth)

    # Short-circuit operators evaluate operands lazily (only the taken branch).
    # Intercept BEFORE the eager list-comp below so an untaken branch that would
    # raise (e.g. a div-by-zero guard) is never evaluated. See D-006 at
    # `_evaluate_short_circuit`; `operations` has no entry for and/or/if.
    if operator in _SHORT_CIRCUIT_OPERATORS:
        return _evaluate_short_circuit(operator, values, data, _depth)

    # For other operators, recursively evaluate values first
    evaluated_values = [evaluate_logic(val, data, _depth + 1) for val in values]

    # DECISION plan-2026-09-21T203800-8a03483a/D-013: an unset operand makes an
    # ordering operator False. Do NOT extend this to `==`/`!=`
    # (`null == null` must stay True, D-017) and do NOT fall back to treating
    # None as 0 or as "operand omitted" (`{"-": [a, null]}` was `-a`, and
    # `<=`/`>=` were True for two unset vars through `soft_equals(None, None)`).
    # DECISION plan-2026-09-21T203800-8a03483a/D-048: arithmetic on an unset
    # operand does NOT return False: a False result was read as 0 by the
    # enclosing comparison (`{"<": [{"-": [150, null]}, 100]}` was True).
    # DECISION plan-2026-09-21T203800-8a03483a/D-054: nor does it return None:
    # None satisfied `null == null`, `!= 0` and `in [null]`, so an "outstanding
    # balance" gate fired on a missing `paid`. It returns the private
    # `_UNDEFINED`, which every comparison/equality/membership operator treats
    # as unsatisfiable. Do NOT let `_UNDEFINED` reach `soft_equals` & co, and
    # do NOT make a `{"var"}` miss UNDEFINED (real nulls keep D-017). See D-054.
    if any(v is _UNDEFINED for v in evaluated_values):
        if operator in _UNDEFINED_IS_FALSE_OPERATORS:
            return False
        if operator in _UNDEFINED_PROPAGATING_OPERATORS:
            return _UNDEFINED
        evaluated_values = [None if v is _UNDEFINED else v for v in evaluated_values]
    if any(v is None for v in evaluated_values):
        if operator in _NONE_IS_FALSE_OPERATORS:
            return False
        if operator in _UNDEFINED_PROPAGATING_OPERATORS:
            return _UNDEFINED

    # Every allowed operator is in exactly one of the three dispatch tables
    # (pinned by test_expressions.py::TestOperatorSweep), and the first two returned above.
    operation = operations[operator]

    # Apply the operation with error handling
    try:
        return operation(*evaluated_values)
    except TransitionEvaluationError:
        raise
    except Exception as e:
        logger.warning(
            f"Error evaluating operation '{operator}' with args {evaluated_values}: {e}"
        )
        # DECISION plan-2026-11-05T134105-5ed71cf2/D-007: a failed ARITHMETIC
        # operator (non-numeric operand such as an extracted "N/A", wrong
        # arity) is `_UNDEFINED`, the same as the None-operand case above.
        # Do NOT return False for it: the enclosing comparison reads False as
        # 0 (`{"<": [{"+": ["abc", 1]}, 5]}` was True), the D-048 hazard
        # through the error path. Other operators still fail to False.
        if operator in _UNDEFINED_PROPAGATING_OPERATORS:
            return _UNDEFINED
        return False
