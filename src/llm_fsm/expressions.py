"""
JsonLogic Expression Evaluator for LLM-FSM
==========================================

This module provides a lightweight implementation of JsonLogic for evaluating
transition conditions in Finite State Machines (FSMs) driven by Large Language Models.

JsonLogic is a way to write logical expressions as JSON objects, making them:

1. Easy to read and write
2. Easy to serialize and store with FSM definitions
3. Expressive enough for complex conditions
4. Programming language-agnostic

This implementation is specifically tailored for LLM-FSM and includes:
    * All standard JsonLogic operators (comparison, logical, arithmetic)
    * Special operators for FSM context data access
    * Data validation operators for checking required fields

Example Usage:
-------------
.. code-block:: python

    from llm_fsm.expression import evaluate_logic

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

import json
from typing import Dict, List, Any, Optional, Union, Callable
from functools import reduce

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .logging import logger

# --------------------------------------------------------------

# Type hint for JsonLogic expressions
JsonLogicExpression = Union[Dict[str, Any], Any]


def soft_equals(a: Any, b: Any) -> bool:
    """
    Implements the '==' operator with type coercion similar to JavaScript.

    :param a: First value to compare
    :param b: Second value to compare
    :return: True if values are equal after coercion, False otherwise

    This comparison attempts to match JavaScript-style equality:
    - Strings compared to any type will convert both to strings
    - Booleans compared to any type will convert both to booleans
    - Otherwise, uses standard Python equality
    """
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) is bool(b)
    return a == b


def hard_equals(a: Any, b: Any) -> bool:
    """
    Implements the '===' operator for strict equality.

    :param a: First value to compare
    :param b: Second value to compare
    :return: True if values are equal and same type, False otherwise

    This comparison requires both type and value equality.
    """
    if type(a) != type(b):
        return False
    return a == b


def less(a: Any, b: Any, *args: Any) -> bool:
    """
    Implements the '<' operator with type coercion.

    :param a: First value to compare
    :param b: Second value to compare
    :param args: Additional values for chained comparison (a < b < c < ...)
    :return: True if values satisfy the less-than relationship, False otherwise

    This function can be called with multiple arguments to check if they're
    in ascending order, like: less(1, 2, 3) => 1 < 2 < 3
    """
    types = set([type(a), type(b)])
    if float in types or int in types:
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            return False
    result = a < b
    if not result or not args:
        return result
    return result and less(b, *args)


def less_or_equal(a: Any, b: Any, *args: Any) -> bool:
    """
    Implements the '<=' operator with type coercion.

    :param a: First value to compare
    :param b: Second value to compare
    :param args: Additional values for chained comparison (a <= b <= c <= ...)
    :return: True if values satisfy the less-than-or-equal relationship, False otherwise

    This function can be called with multiple arguments to check if they're
    in non-descending order, like: less_or_equal(1, 2, 2, 3) => 1 <= 2 <= 2 <= 3
    """
    return (
        less(a, b) or soft_equals(a, b)
    ) and (not args or less_or_equal(b, *args))


def if_condition(*args: Any) -> Any:
    """
    Implements the 'if' operator with support for multiple elseif-s.

    :param args: Variable number of arguments representing if/then/else branches
    :return: The value of the selected branch

    The arguments should be structured as:
    [condition1, result1, condition2, result2, ..., default_result]

    If the number of arguments is odd, the last argument is treated as the default result.
    """
    for i in range(0, len(args) - 1, 2):
        if args[i]:
            return args[i + 1]
    if len(args) % 2:
        return args[-1]
    return None


def get_var(data: Dict[str, Any], var_name: str, not_found: Any = None) -> Any:
    """
    Gets variable value from data dictionary using dot notation.

    :param data: Dictionary containing the data
    :param var_name: Variable name to look up, can use dot notation for nested access
    :param not_found: Value to return if the variable is not found
    :return: The value of the variable or not_found if not present

    Examples:
        get_var({"user": {"name": "Alice"}}, "user.name") => "Alice"
        get_var({"items": [1, 2, 3]}, "items.1") => 2
        get_var({"a": 1}, "b", "default") => "default"
    """
    try:
        for key in str(var_name).split('.'):
            try:
                data = data[key]
            except (TypeError, KeyError):
                try:
                    data = data[int(key)]
                except (ValueError, IndexError, TypeError, KeyError):
                    return not_found
    except (KeyError, TypeError, ValueError):
        return not_found
    return data


def missing(data: Dict[str, Any], *args: Any) -> List[str]:
    """
    Implements the 'missing' operator for finding missing variables.

    :param data: Dictionary containing the data
    :param args: Variable names to check for existence
    :return: List of variable names that are missing from the data

    This function checks if the specified variables exist in the data and
    returns a list of those that don't exist.

    Examples:
        missing({"a": 1, "c": 3}, "a", "b", "c") => ["b"]
    """
    not_found = object()
    if args and isinstance(args[0], list):
        args = args[0]
    ret = []
    for arg in args:
        if get_var(data, arg, not_found) is not_found:
            ret.append(arg)
    return ret


def missing_some(data: Dict[str, Any], min_required: int, args: List[str]) -> List[str]:
    """
    Implements the 'missing_some' operator for finding missing variables.

    :param data: Dictionary containing the data
    :param min_required: Minimum number of arguments that must be present
    :param args: List of variable names to check
    :return: List of missing variables, or empty list if minimum requirement is met

    This function checks if at least min_required variables are present in the data.
    If so, it returns an empty list. Otherwise, it returns the list of missing variables.

    Examples:
        missing_some({"a": 1, "c": 3}, 2, ["a", "b", "c"]) => [] (since 2 of 3 are present)
        missing_some({"a": 1}, 2, ["a", "b", "c"]) => ["b", "c"] (since only 1 of 3 is present)
    """
    if min_required < 1:
        return []
    found = 0
    not_found = object()
    ret = []
    for arg in args:
        if get_var(data, arg, not_found) is not_found:
            ret.append(arg)
        else:
            found += 1
            if found >= min_required:
                return []
    return ret


def cat(*args: Any) -> str:
    """
    Concatenates the string representation of all arguments.

    :param args: Values to concatenate
    :return: Concatenated string

    Examples:
        cat("Hello", " ", "World") => "Hello World"
        cat(1, 2, 3) => "123"
    """
    return "".join(str(arg) for arg in args)


# Dictionary of supported operations
operations = {
    # Comparison operators
    "==": soft_equals,
    "===": hard_equals,
    "!=": lambda a, b: not soft_equals(a, b),
    "!==": lambda a, b: not hard_equals(a, b),
    ">": lambda a, b: less(b, a),
    ">=": lambda a, b: less(b, a) or soft_equals(a, b),
    "<": less,
    "<=": less_or_equal,

    # Logical operators
    "!": lambda a: not a,
    "!!": bool,
    "and": lambda *args: all(args),
    "or": lambda *args: any(args),
    "if": if_condition,

    # Access operators - special handling done directly in evaluate_logic
    # "var", "missing", "missing_some", "has_context", "context_length"

    # Membership operators
    "in": lambda a, b: a in b if hasattr(b, "__contains__") else False,
    "contains": lambda a, b: b in a if hasattr(a, "__contains__") else False,

    # Arithmetic operators
    "+": lambda *args: sum(float(arg) for arg in args),
    "-": lambda a, b=None: -float(a) if b is None else float(a) - float(b),
    "*": lambda *args: reduce(lambda x, y: float(x) * float(y), args, 1),
    "/": lambda a, b: float(a) / float(b),
    "%": lambda a, b: float(a) % float(b),

    # Min/max operators
    "min": lambda *args: min(args),
    "max": lambda *args: max(args),

    # String operators
    "cat": cat,
}


def evaluate_logic(
        logic: JsonLogicExpression,
        data: Dict[str, Any] = None) -> Any:
    """
    Evaluates a JsonLogic expression against provided data.

    :param logic: The JsonLogic expression to evaluate
    :param data: The data object to evaluate against
    :return: The result of evaluating the expression
    :raises: Various exceptions possible during evaluation

    This is the main entry point for evaluating JsonLogic expressions. It recursively
    processes the expression structure and applies the appropriate operators.

    JsonLogic expressions are structured as:
    - Primitive values (strings, numbers, booleans) are returned as-is
    - Objects with a single key represent operations, where:
      - The key is the operator name (e.g., "and", "==", "var")
      - The value is the arguments to that operator

    Examples:
        evaluate_logic({"==": [1, 1]}) => True
        evaluate_logic({"var": "user.name"}, {"user": {"name": "Alice"}}) => "Alice"
        evaluate_logic(
            {"and": [{">=": [{"var": "age"}, 18]}, {"==": [{"var": "has_id"}, true]}]},
            {"age": 20, "has_id": true}
        ) => True
    """
    # Handle primitives
    if logic is None or not isinstance(logic, dict):
        return logic

    data = data or {}

    # Get operator and values
    operator = list(logic.keys())[0]
    values = logic[operator]

    # Convert single values to list for consistent handling
    if not isinstance(values, (list, tuple)):
        values = [values]

    # Special handling for operators that need direct access to data
    if operator == "var":
        var_name = values[0]
        default = values[1] if len(values) > 1 else None
        return get_var(data, var_name, default)

    elif operator == "missing":
        return missing(data, *values)

    elif operator == "missing_some":
        return missing_some(data, values[0], values[1])

    elif operator == "has_context":
        # Check if a key exists in the data object
        if len(values) != 2:
            logger.error(f"has_context requires exactly 2 arguments, got {len(values)}")
            return False

        context_obj = values[0]
        key = values[1]
        return key in context_obj

    elif operator == "context_length":
        # Get the length of a value in the context
        if len(values) != 2:
            logger.error(f"context_length requires exactly 2 arguments, got {len(values)}")
            return False

        context_obj = values[0]
        path = values[1]
        value = get_var(context_obj, path, [])
        if isinstance(value, (list, str)):
            return len(value)
        return 0

    # For other operators, recursively evaluate values
    evaluated_values = []
    for val in values:
        evaluated_values.append(evaluate_logic(val, data))

    # Get the operation function
    if operator not in operations:
        logger.warning(f"Unsupported operation in condition: {operator}")
        return False

    operation = operations[operator]

    # Apply the operation
    try:
        return operation(*evaluated_values)
    except Exception as e:
        logger.error(f"Error evaluating condition {operator}: {e}")
        return False