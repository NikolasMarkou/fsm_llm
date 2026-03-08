"""
JsonLogic Expression Evaluator for LLM-FSM.

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

Example:
    Basic usage example::

        from llm_fsm.expressions import evaluate_logic

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

from functools import reduce
from typing import Dict, List, Any, Union

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------

from .logging import logger

# --------------------------------------------------------------
# Type definitions
# --------------------------------------------------------------

#: Type hint for JsonLogic expressions - can be a dict with operations or primitive values
JsonLogicExpression = Union[Dict[str, Any], Any]

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
    logger.debug(f"Comparing soft equality: {a} == {b} (types: {type(a)}, {type(b)})")

    # String comparison takes precedence
    if isinstance(a, str) or isinstance(b, str):
        result = str(a) == str(b)
        logger.debug(f"String comparison result: {result}")
        return result

    # Boolean comparison
    if isinstance(a, bool) or isinstance(b, bool):
        result = bool(a) == bool(b)
        logger.debug(f"Boolean comparison result: {result}")
        return result

    # Standard equality
    result = a == b
    logger.debug(f"Standard equality result: {result}")
    return result


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
    logger.debug(f"Comparing strict equality: {a} === {b} (types: {type(a)}, {type(b)})")

    if type(a) != type(b):
        logger.debug("Types don't match, returning False")
        return False

    result = a == b
    logger.debug(f"Strict equality result: {result}")
    return result


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
        >>> less("10", "2")  # String comparison
        True

    Note:
        - If either value is numeric (int/float), both are converted to float
        - Chained comparisons evaluate left-to-right
        - Type conversion errors result in False return value
    """
    logger.debug(f"Comparing less than: {a} < {b} (with {len(args)} additional args)")

    # Attempt numeric conversion if any numeric types are present
    types = set([type(a), type(b)])
    if float in types or int in types:
        try:
            a, b = float(a), float(b)
            logger.debug(f"Converted to floats: {a} < {b}")
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to convert values to numeric: {e}")
            return False

    # Perform the primary comparison
    result = a < b
    logger.debug(f"Primary comparison result: {result}")

    # If primary comparison fails or no additional args, return result
    if not result or not args:
        return result

    # Recursively check the chain: b < args[0] < args[1] < ...
    logger.debug("Checking chained comparison")
    return result and less(b, *args)


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
    logger.debug(f"Comparing less than or equal: {a} <= {b} (with {len(args)} additional args)")

    # Check if a <= b (either a < b or a == b)
    primary_result = less(a, b) or soft_equals(a, b)
    logger.debug(f"Primary <= comparison result: {primary_result}")

    # If no additional args or primary comparison fails, return result
    if not args or not primary_result:
        return primary_result

    # Recursively check the chain
    logger.debug("Checking chained <= comparison")
    return primary_result and less_or_equal(b, *args)

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
    logger.debug(f"Evaluating if condition with {len(args)} arguments")

    # Process condition-result pairs
    for i in range(0, len(args) - 1, 2):
        condition = args[i]
        result = args[i + 1]
        logger.debug(f"Checking condition {i//2 + 1}: {condition}")

        if condition:
            logger.debug(f"Condition {i//2 + 1} matched, returning: {result}")
            return result

    # Check for default value (odd number of arguments)
    if len(args) % 2:
        default_result = args[-1]
        logger.debug(f"No conditions matched, returning default: {default_result}")
        return default_result

    logger.debug("No conditions matched and no default provided, returning None")
    return None

# --------------------------------------------------------------
# Data access operators
# --------------------------------------------------------------


def get_var(data: Dict[str, Any], var_name: str, not_found: Any = None) -> Any:
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
    logger.debug(f"Getting variable '{var_name}' from data")

    try:
        current_data = data
        keys = str(var_name).split('.')

        for i, key in enumerate(keys):
            logger.debug(f"Accessing key '{key}' at depth {i}")

            try:
                # Try dictionary key access first
                current_data = current_data[key]
                logger.debug(f"Successfully accessed key '{key}' as dict key")
            except (TypeError, KeyError):
                try:
                    # Try numeric index access for lists/arrays
                    current_data = current_data[int(key)]
                    logger.debug(f"Successfully accessed key '{key}' as numeric index")
                except (ValueError, IndexError, TypeError, KeyError) as e:
                    logger.debug(f"Failed to access key '{key}': {e}")
                    return not_found

        logger.debug(f"Successfully retrieved variable '{var_name}': {current_data}")
        return current_data

    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"Error accessing variable '{var_name}': {e}")
        return not_found


def missing(data: Dict[str, Any], *args: Any) -> List[str]:
    """
    Implement the 'missing' operator for finding missing variables.

    This function checks if the specified variables exist in the data and
    returns a list of those that don't exist.

    Args:
        data: Dictionary containing the data to check
        *args: Variable names to check for existence. Can be individual strings
               or a single list of strings.

    Returns:
        List[str]: List of variable names that are missing from the data

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
    logger.debug(f"Checking for missing variables in data with {len(args)} args")

    # Sentinel object to detect missing values
    not_found = object()

    # Handle case where args is a single list
    if args and isinstance(args[0], list):
        args = args[0]
        logger.debug("Using list format for variable names")

    missing_vars = []
    for arg in args:
        logger.debug(f"Checking if variable '{arg}' exists")

        if get_var(data, arg, not_found) is not_found:
            logger.debug(f"Variable '{arg}' is missing")
            missing_vars.append(arg)
        else:
            logger.debug(f"Variable '{arg}' exists")

    logger.debug(f"Found {len(missing_vars)} missing variables: {missing_vars}")
    return missing_vars


def missing_some(data: Dict[str, Any], min_required: int, args: List[str]) -> List[str]:
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
        List[str]: Empty list if minimum requirement is met, otherwise list of missing variables

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
    logger.debug(f"Checking missing_some: need {min_required} of {len(args)} variables")

    if min_required < 1:
        logger.debug("Minimum required is less than 1, returning empty list")
        return []

    found = 0
    not_found = object()
    missing_vars = []

    for arg in args:
        logger.debug(f"Checking variable '{arg}'")

        if get_var(data, arg, not_found) is not_found:
            logger.debug(f"Variable '{arg}' is missing")
            missing_vars.append(arg)
        else:
            found += 1
            logger.debug(f"Variable '{arg}' exists (found: {found}/{min_required})")

            # Early exit if we've found enough variables
            if found >= min_required:
                logger.debug("Minimum requirement satisfied, returning empty list")
                return []

    logger.debug(f"Only found {found}/{min_required} required variables")
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
    logger.debug(f"Concatenating {len(args)} arguments")
    result = "".join(str(arg) for arg in args)
    logger.debug(f"Concatenation result: '{result}'")
    return result

# --------------------------------------------------------------
# Operation registry
# --------------------------------------------------------------

#: Dictionary mapping operator names to their implementation functions
operations = {
    # Comparison operators
    "==": soft_equals,
    "===": hard_equals,
    "!=": lambda a, b: not soft_equals(a, b),
    "!==": lambda a, b: not hard_equals(a, b),
    ">": lambda a, b: less(b, a),  # Reverse arguments for greater-than
    ">=": lambda a, b: less(b, a) or soft_equals(a, b),  # Greater-than-or-equal
    "<": less,
    "<=": less_or_equal,

    # Logical operators
    "!": lambda a: not a,  # Logical NOT
    "!!": bool,  # Double negation (convert to boolean)
    "and": lambda *args: all(args),  # Logical AND
    "or": lambda *args: any(args),   # Logical OR
    "if": if_condition,

    # Note: Access operators handled directly in evaluate_logic()
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

# --------------------------------------------------------------
# Main evaluation function
# --------------------------------------------------------------


def evaluate_logic(
        logic: JsonLogicExpression,
        data: Dict[str, Any] = None) -> Any:
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
    logger.debug(f"Evaluating logic: {logic} with data keys: {list(data.keys()) if data else []}")

    # Handle primitive values (base case for recursion)
    if logic is None or not isinstance(logic, dict):
        logger.debug(f"Returning primitive value: {logic}")
        return logic

    # Ensure we have data to work with
    data = data or {}

    # Extract operator and arguments from the logic expression
    if not logic:
        logger.warning("Empty logic expression provided")
        return None

    operator = list(logic.keys())[0]
    values = logic[operator]

    logger.debug(f"Processing operator '{operator}' with values: {values}")

    # Convert single values to list for consistent handling
    if not isinstance(values, (list, tuple)):
        values = [values]
        logger.debug(f"Converted single value to list: {values}")

    # Special handling for operators that need direct access to data
    if operator == "var":
        if not values:
            logger.error("var operator requires at least one argument")
            return None

        var_name = values[0]
        default = values[1] if len(values) > 1 else None
        result = get_var(data, var_name, default)
        logger.debug(f"Variable access result: '{var_name}' = {result}")
        return result

    elif operator == "missing":
        result = missing(data, *values)
        logger.debug(f"Missing variables check result: {result}")
        return result

    elif operator == "missing_some":
        if len(values) != 2:
            logger.error(f"missing_some requires exactly 2 arguments, got {len(values)}")
            return []

        result = missing_some(data, values[0], values[1])
        logger.debug(f"Missing some check result: {result}")
        return result

    elif operator == "has_context":
        # Check if a key exists in the context object
        if len(values) != 2:
            logger.error(f"has_context requires exactly 2 arguments, got {len(values)}")
            return False

        context_obj = values[0]
        key = values[1]

        if not isinstance(context_obj, dict):
            logger.warning(f"has_context expected dict as first argument, got {type(context_obj)}")
            return False

        result = key in context_obj
        logger.debug(f"Context key existence check: '{key}' in context = {result}")
        return result

    elif operator == "context_length":
        # Get the length of a value in the context
        if len(values) != 2:
            logger.error(f"context_length requires exactly 2 arguments, got {len(values)}")
            return 0

        context_obj = values[0]
        path = values[1]

        if not isinstance(context_obj, dict):
            logger.warning(f"context_length expected dict as first argument, got {type(context_obj)}")
            return 0

        value = get_var(context_obj, path, [])

        if isinstance(value, (list, str, dict)):
            result = len(value)
            logger.debug(f"Context length for '{path}': {result}")
            return result
        else:
            logger.debug(f"Context value at '{path}' is not measurable, returning 0")
            return 0

    # For other operators, recursively evaluate values first
    logger.debug(f"Recursively evaluating {len(values)} arguments for operator '{operator}'")
    evaluated_values = []
    for i, val in enumerate(values):
        evaluated_val = evaluate_logic(val, data)
        evaluated_values.append(evaluated_val)
        logger.debug(f"Argument {i} evaluated to: {evaluated_val}")

    # Get the operation function
    if operator not in operations:
        logger.warning(f"Unsupported operation in condition: '{operator}'")
        return False

    operation = operations[operator]

    # Apply the operation with error handling
    try:
        result = operation(*evaluated_values)
        logger.debug(f"Operation '{operator}' result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error evaluating operation '{operator}' with args {evaluated_values}: {e}")
        return False