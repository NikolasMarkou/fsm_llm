from __future__ import annotations

"""
JsonLogic Expression Evaluator for FSM-LLM.

This module provides a lightweight implementation of JsonLogic for evaluating
transition conditions in Finite State Machines (FSMs) driven by Large Language Models.

JsonLogic is a way to write logical expressions as JSON objects, making them:

1. Easy to read and write
2. Easy to serialize and store with FSM definitions
3. Expressive enough for complex conditions
4. Programming language-agnostic

This implementation is specifically tailored for FSM-LLM and includes:
    * All standard JsonLogic operators (comparison, logical, arithmetic)
    * Special operators for FSM context data access
    * Data validation operators for checking required fields

Example:
    Basic usage example::

        from fsm_llm.expressions import evaluate_logic

        # Sample condition: If customer is VIP and issue is high priority
        logic = {
            "and": [
                {"==": [{"var": "customer.status"}, "vip"]},
                {"==": [{"var": "issue.priority"}, "high"]}
            ]
        }

        # Sample context data
        context = {
            "customer": {"status": "vip"},
            "issue": {"priority": "high"}
        }

        # Evaluate the logic
        result = evaluate_logic(logic, context)
        # result is True
"""

from collections.abc import Callable
from functools import reduce
from typing import Any

from .constants import ALLOWED_JSONLOGIC_OPERATIONS
from .definitions import TransitionEvaluationError

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------
from .logging import logger

# --------------------------------------------------------------
# Type definitions
# --------------------------------------------------------------

#: Type hint for JsonLogic expressions - can be a dict with operations or primitive values
JsonLogicExpression = dict[str, Any] | Any

# --------------------------------------------------------------
# Comparison operators
# --------------------------------------------------------------


def soft_equals(a: Any, b: Any) -> bool:
    """
    Implement the '==' operator with type coercion similar to JavaScript.

    This comparison attempts to match JavaScript-style equality by converting
    values to comparable types when one operand is a string or boolean.

    Args:
        a: First value to compare
        b: Second value to compare

    Returns:
        bool: True if values are equal after coercion, False otherwise

    Example:
        >>> soft_equals("123", 123)
        True
        >>> soft_equals(True, 1)
        True
        >>> soft_equals("hello", "hello")
        True

    Note:
        - If either value is a string, both are converted to strings
        - If either value is a boolean, both are converted to booleans
        - Otherwise, uses standard Python equality
    """
    # String comparison — case-insensitive when one operand is bool
    # (LLMs return JSON "true"/"false" but Python str(True) is "True")
    if isinstance(a, str) or isinstance(b, str):
        return str(a).lower() == str(b).lower() if (isinstance(a, bool) or isinstance(b, bool)) else str(a) == str(b)

    # Boolean comparison
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)

    # Standard equality
    return bool(a == b)


def hard_equals(a: Any, b: Any) -> bool:
    """
    Implement the '===' operator for strict equality.

    This comparison requires both type and value equality, similar to
    JavaScript's strict equality operator.

    Args:
        a: First value to compare
        b: Second value to compare

    Returns:
        bool: True if values are equal and same type, False otherwise

    Example:
        >>> hard_equals("123", 123)
        False
        >>> hard_equals(123, 123)
        True
        >>> hard_equals(True, 1)
        False

    Note:
        Both the type and value must match exactly for this to return True.
    """
    if type(a) is not type(b):
        return False

    return bool(a == b)


def less(a: Any, b: Any, *args: Any) -> bool:
    """
    Implement the '<' operator with type coercion and chaining support.

    This function supports chained comparisons like a < b < c by accepting
    multiple arguments and verifying the entire chain.

    Args:
        a: First value to compare
        b: Second value to compare
        *args: Additional values for chained comparison (a < b < c < ...)

    Returns:
        bool: True if values satisfy the less-than relationship, False otherwise

    Raises:
        TypeError: If values cannot be converted for comparison
        ValueError: If numeric conversion fails

    Example:
        >>> less(1, 2)
        True
        >>> less(1, 2, 3)  # Equivalent to 1 < 2 < 3
        True
        >>> less("10", "2")  # Numeric coercion: float("10") < float("2")
        False

    Note:
        - If either value is numeric (int/float), both are converted to float
        - Chained comparisons evaluate left-to-right
        - Type conversion errors result in False return value
    """
    # Always attempt numeric conversion for comparisons
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        pass  # Fall back to native comparison

    # Perform the primary comparison
    try:
        result = a < b
    except TypeError:
        return False

    # If primary comparison fails or no additional args, return result
    if not result or not args:
        return bool(result)

    # Recursively check the chain: b < args[0] < args[1] < ...
    return bool(result) and less(b, *args)


def less_or_equal(a: Any, b: Any, *args: Any) -> bool:
    """
    Implement the '<=' operator with type coercion and chaining support.

    This function supports chained comparisons like a <= b <= c by accepting
    multiple arguments and verifying the entire chain.

    Args:
        a: First value to compare
        b: Second value to compare
        *args: Additional values for chained comparison (a <= b <= c <= ...)

    Returns:
        bool: True if values satisfy the less-than-or-equal relationship, False otherwise

    Example:
        >>> less_or_equal(1, 2)
        True
        >>> less_or_equal(2, 2)
        True
        >>> less_or_equal(1, 2, 2, 3)  # Equivalent to 1 <= 2 <= 2 <= 3
        True

    Note:
        Uses combination of less() and soft_equals() for evaluation.
        Supports the same type coercion as the less() function.
    """
    # Check if a <= b (either a < b or a == b)
    primary_result = less(a, b) or soft_equals(a, b)

    # If no additional args or primary comparison fails, return result
    if not args or not primary_result:
        return primary_result

    # Recursively check the chain
    return primary_result and less_or_equal(b, *args)

def greater(a: Any, b: Any, *args: Any) -> bool:
    """Implement '>' with chaining support (a > b > c)."""
    result = less(b, a)
    if not result or not args:
        return result
    return result and greater(b, *args)


def greater_or_equal(a: Any, b: Any, *args: Any) -> bool:
    """Implement '>=' with chaining support (a >= b >= c)."""
    result = less(b, a) or soft_equals(a, b)
    if not args or not result:
        return result
    return result and greater_or_equal(b, *args)


# --------------------------------------------------------------
# Logical operators
# --------------------------------------------------------------


def if_condition(*args: Any) -> Any:
    """
    Implement the 'if' operator with support for multiple elseif branches.

    This function processes if/then/else logic with support for multiple
    condition-result pairs, similar to a switch statement.

    Args:
        *args: Variable number of arguments representing if/then/else branches.
               Format: [condition1, result1, condition2, result2, ..., default_result]

    Returns:
        Any: The value of the first matching condition's result, or default if provided

    Example:
        >>> if_condition(True, "first", False, "second", "default")
        'first'
        >>> if_condition(False, "first", False, "second", "default")
        'default'
        >>> if_condition(False, "first", True, "second")
        'second'

    Note:
        - Arguments are processed in pairs: (condition, result)
        - If the number of arguments is odd, the last argument is the default result
        - Returns None if no conditions match and no default is provided
    """
    # Process condition-result pairs
    for i in range(0, len(args) - 1, 2):
        condition = args[i]
        result = args[i + 1]

        if condition:
            return result

    # Check for default value (odd number of arguments)
    if len(args) % 2:
        return args[-1]

    return None

# --------------------------------------------------------------
# Data access operators
# --------------------------------------------------------------


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
    keys = str(var_name).split('.')

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
        - Uses get_var() internally, so supports dot notation
        - Empty list means all variables are present
    """
    # Sentinel object to detect missing values
    not_found = object()

    # Handle case where args is a single list
    var_names: tuple[Any, ...] | list[Any] = args
    if args and isinstance(args[0], list):
        var_names = args[0]

    missing_vars = []
    for arg in var_names:
        if get_var(data, arg, not_found) is not_found:
            missing_vars.append(arg)

    return missing_vars


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
        - Uses get_var() internally for consistent behavior
    """
    if min_required < 1:
        return []

    found = 0
    not_found = object()
    missing_vars = []

    # Guard: treat string arg as single var name, not iterable of chars
    if isinstance(args, str):
        args = [args]

    for arg in args:
        if get_var(data, arg, not_found) is not_found:
            missing_vars.append(arg)
        else:
            found += 1
            # Early exit if we've found enough variables
            if found >= min_required:
                return []

    return missing_vars

# --------------------------------------------------------------
# String operators
# --------------------------------------------------------------


def cat(*args: Any) -> str:
    """
    Concatenate the string representation of all arguments.

    This function converts all arguments to strings and concatenates them
    together without any separators.

    Args:
        *args: Values to concatenate (any type, will be converted to string)

    Returns:
        str: Concatenated string representation of all arguments

    Example:
        >>> cat("Hello", " ", "World")
        'Hello World'
        >>> cat(1, 2, 3)
        '123'
        >>> cat("Value: ", 42, ", Active: ", True)
        'Value: 42, Active: True'

    Note:
        - All arguments are converted using str() function
        - No separators are added between arguments
        - Empty arguments list returns empty string
    """
    return "".join(str(arg) for arg in args)

# --------------------------------------------------------------
# Operation registry
# --------------------------------------------------------------

#: Dictionary mapping operator names to their implementation functions
operations: dict[str, Callable[..., Any]] = {
    # Comparison operators
    "==": soft_equals,
    "===": hard_equals,
    "!=": lambda a, b: not soft_equals(a, b),
    "!==": lambda a, b: not hard_equals(a, b),
    ">": greater,
    ">=": greater_or_equal,
    "<": less,
    "<=": less_or_equal,

    # Logical operators
    "!": lambda *args: not args[0] if args else True,  # Logical NOT
    "!!": bool,  # Double negation (convert to boolean)
    "and": lambda *args: next((a for a in args if not a), args[-1]) if args else True,
    "or": lambda *args: next((a for a in args if a), args[-1]) if args else False,
    "if": if_condition,

    # Note: Access operators handled directly in evaluate_logic()
    # "var", "missing", "missing_some", "has_context", "context_length"

    # Membership operators
    "in": lambda a, b: a in b if hasattr(b, "__contains__") else False,
    "contains": lambda a, b: b in a if hasattr(a, "__contains__") else False,

    # Arithmetic operators
    "+": lambda *args: sum(float(arg) for arg in args),
    "-": lambda a, b=None: -float(a) if b is None else float(a) - float(b),
    "*": lambda *args: reduce(lambda x, y: float(x) * float(y), args, 1.0),
    "/": lambda a, b: 0 if float(b) == 0 else float(a) / float(b),
    "%": lambda a, b: 0 if float(b) == 0 else float(a) % float(b),

    # Min/max operators (with numeric coercion like other arithmetic ops)
    "min": lambda *args: min(float(a) for a in args) if args else None,
    "max": lambda *args: max(float(a) for a in args) if args else None,

    # String operators
    "cat": cat,
}

# --------------------------------------------------------------
# Main evaluation function
# --------------------------------------------------------------


MAX_JSONLOGIC_DEPTH = 50


# --------------------------------------------------------------
# Data-access operator handlers
# --------------------------------------------------------------
# These operators need direct access to the data dict (no recursive
# evaluation of their arguments), so they are dispatched separately.


def _op_var(values: list, data: dict[str, Any], _depth: int) -> Any:
    """Handle the 'var' operator — variable access with dot notation."""
    if not values:
        logger.error("var operator requires at least one argument")
        return None
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
    return missing_some(data, values[0], values[1])


def _op_has_context(values: list, data: dict[str, Any], _depth: int) -> bool:
    """Handle the 'has_context' operator — check key existence in context.

    Supports two forms:
    - Single argument: {"has_context": "key"} — checks if key exists in data with a truthy value
    - Two arguments: {"has_context": [context_obj, "key"]} — checks key in a specific dict
    """
    if len(values) == 1:
        # Single-argument shorthand: check key in the top-level data dict
        key = evaluate_logic(values[0], data, _depth + 1)
        if not isinstance(key, str):
            logger.warning(f"has_context expected string key, got {type(key)}")
            return False
        value = data.get(key)
        # Key must exist and have a truthy value (not None, not empty, not False)
        return value is not None and value != [] and value != "" and value is not False
    if len(values) == 2:
        context_obj = evaluate_logic(values[0], data, _depth + 1)
        key = evaluate_logic(values[1], data, _depth + 1)
        if not isinstance(context_obj, dict):
            logger.warning(f"has_context expected dict as first argument, got {type(context_obj)}")
            return False
        value = context_obj.get(key)
        return value is not None and value != [] and value != "" and value is not False
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
        logger.warning(f"context_length expected dict as first argument, got {type(context_obj)}")
        return 0
    value = get_var(context_obj, path, [])
    if isinstance(value, (list, str, dict)):
        return len(value)
    return 0


#: Dispatch dict for operators that need direct data access (no arg pre-eval)
_data_operators: dict[str, Any] = {
    "var": _op_var,
    "missing": _op_missing,
    "missing_some": _op_missing_some,
    "has_context": _op_has_context,
    "context_length": _op_context_length,
}


def evaluate_logic(
        logic: JsonLogicExpression,
        data: dict[str, Any] | None = None,
        _depth: int = 0) -> Any:
    """
    Evaluate a JsonLogic expression against provided data.

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
    if not isinstance(values, (list, tuple)):
        values = [values]

    # Data-access operators get raw values + data (no recursive eval)
    data_op = _data_operators.get(operator)
    if data_op is not None:
        return data_op(values, data, _depth)

    # For other operators, recursively evaluate values first
    evaluated_values = [evaluate_logic(val, data, _depth + 1) for val in values]

    # Get the operation function from the registry
    operation = operations.get(operator)
    if operation is None:
        logger.warning(f"Unsupported operation in condition: '{operator}'")
        return False

    # Apply the operation with error handling
    try:
        return operation(*evaluated_values)
    except Exception as e:
        logger.warning(f"Error evaluating operation '{operator}' with args {evaluated_values}: {e}")
        return False
