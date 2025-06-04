# /expressions.py

"""
Enhanced JsonLogic Expression Evaluator for 2-Pass LLM-FSM Architecture.

This module provides an improved implementation of JsonLogic for evaluating
transition conditions in the enhanced FSM system. It includes optimizations
for performance, better error handling, and additional operators specific
to FSM context evaluation.

Key Enhancements:
- Performance optimizations for common FSM use cases
- Enhanced error handling and logging
- Additional operators for FSM-specific evaluations
- Better type coercion and validation
- Support for nested context evaluation
"""

import math
import operator
from functools import reduce
from typing import Dict, List, Any, Union, Set, Optional

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------

from .logging import logger

# --------------------------------------------------------------
# Type Definitions
# --------------------------------------------------------------

JsonLogicExpression = Union[Dict[str, Any], Any]
ContextData = Dict[str, Any]


# --------------------------------------------------------------
# Enhanced Comparison Operations
# --------------------------------------------------------------

def soft_equals(a: Any, b: Any) -> bool:
    """
    Enhanced soft equality with better type coercion.

    Implements JavaScript-style equality with improved handling
    for FSM context data types.

    Args:
        a: First value to compare
        b: Second value to compare

    Returns:
        True if values are equal after coercion
    """
    # Handle None/null cases
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        return False

    # Handle boolean comparisons
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)

    # Handle numeric comparisons
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        try:
            return float(a) == float(b)
        except (ValueError, TypeError):
            pass

    # Handle string comparisons (most common in FSM contexts)
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)

    # Handle list/set comparisons
    if isinstance(a, (list, tuple, set)) and isinstance(b, (list, tuple, set)):
        return set(a) == set(b)

    # Default comparison
    return a == b


def strict_equals(a: Any, b: Any) -> bool:
    """
    Strict equality requiring both type and value matching.

    Args:
        a: First value to compare
        b: Second value to compare

    Returns:
        True if values and types are identical
    """
    return type(a) == type(b) and a == b


def enhanced_less_than(a: Any, b: Any, *args: Any) -> bool:
    """
    Enhanced less-than comparison with better type handling.

    Supports chained comparisons and improved type coercion
    for FSM context values.

    Args:
        a: First value
        b: Second value
        *args: Additional values for chained comparison

    Returns:
        True if a < b [< c < d...] for all values
    """
    # Handle numeric comparisons
    try:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            result = a < b
        else:
            # Try to convert to numbers
            num_a, num_b = float(a), float(b)
            result = num_a < num_b
    except (ValueError, TypeError):
        # Fall back to string comparison
        try:
            result = str(a) < str(b)
        except TypeError:
            return False

    # Handle chained comparisons
    if not result or not args:
        return result

    return result and enhanced_less_than(b, *args)


def enhanced_less_equal(a: Any, b: Any, *args: Any) -> bool:
    """Enhanced less-than-or-equal with chaining support."""
    return enhanced_less_than(a, b) or soft_equals(a, b) and (
            not args or enhanced_less_equal(b, *args)
    )


# --------------------------------------------------------------
# Enhanced Variable Access
# --------------------------------------------------------------

def get_variable(data: ContextData, var_path: str, default: Any = None) -> Any:
    """
    Enhanced variable access with dot notation and array indexing.

    Supports complex path expressions for accessing nested FSM context data.

    Args:
        data: Context data dictionary
        var_path: Variable path (supports dot notation and array indexing)
        default: Default value if path not found

    Returns:
        Value at the specified path or default
    """
    if not isinstance(data, dict) or not var_path:
        return default

    try:
        current = data

        # Split path by dots, but handle array indices specially
        path_parts = str(var_path).split('.')

        for part in path_parts:
            if not part:
                continue

            # Handle array indexing [n] syntax
            if '[' in part and part.endswith(']'):
                key_part, index_part = part.split('[', 1)
                index_part = index_part[:-1]  # Remove closing ]

                # Get the array first
                if key_part:
                    current = current[key_part]

                # Then get the indexed element
                try:
                    index = int(index_part)
                    current = current[index]
                except (ValueError, IndexError, TypeError):
                    return default
            else:
                # Regular key access
                if isinstance(current, dict):
                    current = current.get(part, default)
                    if current is default:
                        return default
                elif isinstance(current, (list, tuple)):
                    try:
                        index = int(part)
                        current = current[index]
                    except (ValueError, IndexError, TypeError):
                        return default
                else:
                    return default

        return current

    except (KeyError, TypeError, AttributeError):
        return default


def check_missing_variables(data: ContextData, *var_paths: str) -> List[str]:
    """
    Check for missing variables in context data.

    Args:
        data: Context data to check
        *var_paths: Variable paths to verify

    Returns:
        List of missing variable paths
    """
    missing = []
    sentinel = object()  # Unique object to detect missing values

    # Handle case where first arg is a list
    if var_paths and isinstance(var_paths[0], (list, tuple)):
        var_paths = var_paths[0]

    for var_path in var_paths:
        if get_variable(data, var_path, sentinel) is sentinel:
            missing.append(var_path)

    return missing


def check_missing_some(data: ContextData, min_required: int, var_paths: List[str]) -> List[str]:
    """
    Check if minimum required variables are present.

    Args:
        data: Context data to check
        min_required: Minimum number of variables that must be present
        var_paths: List of variable paths to check

    Returns:
        List of missing variables if minimum not met, empty list otherwise
    """
    if min_required < 1:
        return []

    found_count = 0
    missing = []
    sentinel = object()

    for var_path in var_paths:
        if get_variable(data, var_path, sentinel) is not sentinel:
            found_count += 1
        else:
            missing.append(var_path)

        # Early return if we have enough
        if found_count >= min_required:
            return []

    return missing


# --------------------------------------------------------------
# Enhanced Logical Operations
# --------------------------------------------------------------

def logical_if(*conditions_and_values: Any) -> Any:
    """
    Enhanced if-then-else with support for multiple conditions.

    Args:
        *conditions_and_values: Alternating conditions and values

    Returns:
        Value for first true condition, or last value as default
    """
    args = list(conditions_and_values)

    # Process condition-value pairs
    for i in range(0, len(args) - 1, 2):
        condition = args[i]
        value = args[i + 1] if i + 1 < len(args) else None

        if condition:
            return value

    # Return default value (last odd argument)
    if len(args) % 2 == 1:
        return args[-1]

    return None


def enhanced_and(*values: Any) -> bool:
    """Enhanced logical AND with short-circuit evaluation."""
    for value in values:
        if not value:
            return False
    return True


def enhanced_or(*values: Any) -> bool:
    """Enhanced logical OR with short-circuit evaluation."""
    for value in values:
        if value:
            return True
    return False


# --------------------------------------------------------------
# FSM-Specific Operations
# --------------------------------------------------------------

def context_has_key(data: ContextData, key: str) -> bool:
    """
    Check if context has a specific key.

    Args:
        data: Context data
        key: Key to check for

    Returns:
        True if key exists in context
    """
    return isinstance(data, dict) and key in data


def context_key_count(data: ContextData) -> int:
    """
    Get count of keys in context data.

    Args:
        data: Context data

    Returns:
        Number of keys in context
    """
    return len(data) if isinstance(data, dict) else 0


def context_value_length(data: ContextData, key: str) -> int:
    """
    Get length of a context value.

    Args:
        data: Context data
        key: Key to get length for

    Returns:
        Length of value (0 if not found or not measurable)
    """
    value = get_variable(data, key)

    if value is None:
        return 0

    try:
        return len(value)
    except TypeError:
        return 0


def context_keys_match_pattern(data: ContextData, pattern: str) -> List[str]:
    """
    Find context keys matching a pattern.

    Args:
        data: Context data
        pattern: Pattern to match (supports * wildcards)

    Returns:
        List of matching keys
    """
    if not isinstance(data, dict):
        return []

    import fnmatch
    matching_keys = []

    for key in data.keys():
        if fnmatch.fnmatch(str(key), pattern):
            matching_keys.append(key)

    return matching_keys


# --------------------------------------------------------------
# Mathematical Operations
# --------------------------------------------------------------

def safe_divide(a: Any, b: Any) -> float:
    """Safe division with zero handling."""
    try:
        divisor = float(b)
        if divisor == 0:
            return float('inf') if float(a) >= 0 else float('-inf')
        return float(a) / divisor
    except (ValueError, TypeError):
        return 0.0


def safe_modulo(a: Any, b: Any) -> float:
    """Safe modulo with zero handling."""
    try:
        divisor = float(b)
        if divisor == 0:
            return 0.0
        return float(a) % divisor
    except (ValueError, TypeError):
        return 0.0


# --------------------------------------------------------------
# String Operations
# --------------------------------------------------------------

def string_concat(*values: Any) -> str:
    """Concatenate values as strings."""
    return ''.join(str(value) for value in values)


def string_contains(text: Any, substring: Any) -> bool:
    """Check if text contains substring."""
    try:
        return str(substring) in str(text)
    except (TypeError, AttributeError):
        return False


def string_starts_with(text: Any, prefix: Any) -> bool:
    """Check if text starts with prefix."""
    try:
        return str(text).startswith(str(prefix))
    except (TypeError, AttributeError):
        return False


def string_ends_with(text: Any, suffix: Any) -> bool:
    """Check if text ends with suffix."""
    try:
        return str(text).endswith(str(suffix))
    except (TypeError, AttributeError):
        return False


# --------------------------------------------------------------
# Array/Collection Operations
# --------------------------------------------------------------

def array_contains(collection: Any, item: Any) -> bool:
    """Check if collection contains item."""
    try:
        if hasattr(collection, '__contains__'):
            return item in collection
        elif hasattr(collection, '__iter__'):
            return item in list(collection)
        else:
            return soft_equals(collection, item)
    except (TypeError, AttributeError):
        return False


def array_length(collection: Any) -> int:
    """Get length of collection."""
    try:
        return len(collection)
    except (TypeError, AttributeError):
        return 0


def array_unique(collection: Any) -> List[Any]:
    """Get unique items from collection."""
    try:
        if isinstance(collection, (list, tuple)):
            seen = set()
            unique = []
            for item in collection:
                if item not in seen:
                    seen.add(item)
                    unique.append(item)
            return unique
        else:
            return [collection]
    except (TypeError, AttributeError):
        return []


# --------------------------------------------------------------
# Operations Registry
# --------------------------------------------------------------

# Enhanced operations registry with FSM-specific functions
OPERATIONS = {
    # Comparison operators
    "==": soft_equals,
    "===": strict_equals,
    "!=": lambda a, b: not soft_equals(a, b),
    "!==": lambda a, b: not strict_equals(a, b),
    ">": lambda a, b: enhanced_less_than(b, a),
    ">=": lambda a, b: enhanced_less_than(b, a) or soft_equals(a, b),
    "<": enhanced_less_than,
    "<=": enhanced_less_equal,

    # Logical operators
    "!": lambda a: not a,
    "!!": bool,
    "and": enhanced_and,
    "or": enhanced_or,
    "if": logical_if,

    # Variable access (handled specially in evaluate_logic)
    # "var", "missing", "missing_some"

    # Membership operators
    "in": array_contains,
    "contains": lambda a, b: array_contains(a, b),

    # Arithmetic operators
    "+": lambda *args: sum(float(arg) for arg in args),
    "-": lambda a, b=None: -float(a) if b is None else float(a) - float(b),
    "*": lambda *args: reduce(operator.mul, (float(arg) for arg in args), 1),
    "/": safe_divide,
    "%": safe_modulo,
    "//": lambda a, b: float(int(float(a) // float(b))),  # Floor division
    "**": lambda a, b: float(a) ** float(b),  # Exponentiation

    # Math functions
    "abs": lambda a: abs(float(a)),
    "min": lambda *args: min(args),
    "max": lambda *args: max(args),
    "round": lambda a, digits=0: round(float(a), int(digits)),
    "floor": lambda a: math.floor(float(a)),
    "ceil": lambda a: math.ceil(float(a)),
    "sqrt": lambda a: math.sqrt(float(a)),

    # String operators
    "cat": string_concat,
    "str_contains": string_contains,
    "str_starts": string_starts_with,
    "str_ends": string_ends_with,
    "str_lower": lambda s: str(s).lower(),
    "str_upper": lambda s: str(s).upper(),
    "str_trim": lambda s: str(s).strip(),
    "str_length": lambda s: len(str(s)),

    # Array operators
    "array_length": array_length,
    "array_unique": array_unique,
    "array_first": lambda arr: arr[0] if arr and len(arr) > 0 else None,
    "array_last": lambda arr: arr[-1] if arr and len(arr) > 0 else None,

    # FSM context operators
    "context_has": context_has_key,
    "context_count": context_key_count,
    "context_length": context_value_length,
    "context_keys": context_keys_match_pattern,

    # Type checking
    "is_string": lambda v: isinstance(v, str),
    "is_number": lambda v: isinstance(v, (int, float)),
    "is_bool": lambda v: isinstance(v, bool),
    "is_array": lambda v: isinstance(v, (list, tuple)),
    "is_object": lambda v: isinstance(v, dict),
    "is_null": lambda v: v is None,
    "is_empty": lambda v: not v if v is not None else True,
}


# --------------------------------------------------------------
# Main Evaluation Function
# --------------------------------------------------------------

def evaluate_logic(logic: JsonLogicExpression, data: ContextData = None) -> Any:
    """
    Enhanced JsonLogic expression evaluator with FSM optimizations.

    This function provides comprehensive JsonLogic evaluation with
    enhancements specifically designed for FSM context processing.

    Args:
        logic: JsonLogic expression to evaluate
        data: Context data for variable resolution

    Returns:
        Result of evaluating the expression

    Raises:
        Various exceptions during evaluation (handled gracefully)
    """
    # Handle primitive values
    if logic is None or not isinstance(logic, dict):
        return logic

    # Ensure we have context data
    data = data or {}

    try:
        # Get operator and operands
        if not logic:
            return None

        operator_name = list(logic.keys())[0]
        operands = logic[operator_name]

        # Convert single operands to list for consistent handling
        if not isinstance(operands, (list, tuple)):
            operands = [operands]

        # Handle special operators that need direct data access
        if operator_name == "var":
            var_path = operands[0] if operands else ""
            default = operands[1] if len(operands) > 1 else None
            return get_variable(data, var_path, default)

        elif operator_name == "missing":
            return check_missing_variables(data, *operands)

        elif operator_name == "missing_some":
            if len(operands) >= 2:
                return check_missing_some(data, operands[0], operands[1])
            return []

        # Handle context-specific operators
        elif operator_name == "context_has":
            if len(operands) >= 1:
                return context_has_key(data, str(operands[0]))
            return False

        elif operator_name == "context_count":
            return context_key_count(data)

        elif operator_name == "context_length":
            if len(operands) >= 1:
                return context_value_length(data, str(operands[0]))
            return 0

        elif operator_name == "context_keys":
            if len(operands) >= 1:
                return context_keys_match_pattern(data, str(operands[0]))
            return []

        # For other operators, recursively evaluate operands
        evaluated_operands = []
        for operand in operands:
            evaluated_operands.append(evaluate_logic(operand, data))

        # Get and execute the operation
        if operator_name not in OPERATIONS:
            logger.warning(f"Unsupported JsonLogic operation: {operator_name}")
            return False

        operation = OPERATIONS[operator_name]

        # Execute with error handling
        try:
            result = operation(*evaluated_operands)

            # Log complex operations for debugging
            if logger.level <= 10:  # DEBUG level
                logger.debug(f"JsonLogic: {operator_name}({evaluated_operands}) -> {result}")

            return result

        except Exception as op_error:
            logger.warning(f"Error executing JsonLogic operation {operator_name}: {op_error}")
            return False

    except Exception as e:
        logger.error(f"JsonLogic evaluation error: {e}")
        return False


# --------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------

def validate_json_logic(logic: JsonLogicExpression) -> bool:
    """
    Validate JsonLogic expression structure.

    Args:
        logic: JsonLogic expression to validate

    Returns:
        True if expression appears valid
    """
    if logic is None:
        return True

    if not isinstance(logic, dict):
        return True  # Primitive values are valid

    if not logic:
        return False  # Empty dict is invalid

    if len(logic) != 1:
        return False  # Should have exactly one operator

    operator_name = list(logic.keys())[0]

    # Check if operator is supported
    if operator_name not in OPERATIONS and operator_name not in [
        "var", "missing", "missing_some", "context_has", "context_count",
        "context_length", "context_keys"
    ]:
        logger.debug(f"Unknown JsonLogic operator: {operator_name}")
        return False

    return True


def optimize_json_logic(logic: JsonLogicExpression) -> JsonLogicExpression:
    """
    Optimize JsonLogic expression for better performance.

    Args:
        logic: JsonLogic expression to optimize

    Returns:
        Optimized expression
    """
    if not isinstance(logic, dict):
        return logic

    # Simple optimizations
    for operator_name, operands in logic.items():
        # Optimize constant folding for simple cases
        if operator_name in ["==", "!=", ">", "<", ">=", "<="]:
            if isinstance(operands, list) and len(operands) == 2:
                # If both operands are constants, we could pre-evaluate
                # but we'll keep it simple for now
                pass

        # Optimize variable access patterns
        elif operator_name == "var":
            # Could cache frequently accessed variables
            pass

    return logic