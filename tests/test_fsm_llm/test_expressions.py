"""
Tests for the JsonLogic expression evaluator module.

This test suite thoroughly tests the functionality of the JsonLogic expression evaluator,
covering all operators, error handling, and complex expressions that would be used in
real FSM transitions.
"""

import pytest
from llm_fsm.expressions import evaluate_logic


class TestExpressionEvaluator:
    """Test suite for JsonLogic expression evaluator."""

    def test_primitive_values(self):
        """Test that primitive values are returned as-is."""
        assert evaluate_logic(None) is None
        assert evaluate_logic(True) is True
        assert evaluate_logic(False) is False
        assert evaluate_logic(5) == 5
        assert evaluate_logic("hello") == "hello"
        assert evaluate_logic([1, 2, 3]) == [1, 2, 3]

    def test_equality_operators(self):
        """Test equality operators (==, ===, !=, !==)."""
        # Soft equality (==)
        assert evaluate_logic({"==": [1, 1]}) is True
        assert evaluate_logic({"==": [1, "1"]}) is True
        assert evaluate_logic({"==": [0, False]}) is True
        assert evaluate_logic({"==": [1, 2]}) is False
        assert evaluate_logic({"==": ["foo", "bar"]}) is False

        # Hard equality (===)
        assert evaluate_logic({"===": [1, 1]}) is True
        assert evaluate_logic({"===": [1, "1"]}) is False
        assert evaluate_logic({"===": [0, False]}) is False

        # Not equal (!=)
        assert evaluate_logic({"!=": [1, 2]}) is True
        assert evaluate_logic({"!=": [1, "1"]}) is False

        # Hard not equal (!==)
        assert evaluate_logic({"!==": [1, 2]}) is True
        assert evaluate_logic({"!==": [1, "1"]}) is True
        assert evaluate_logic({"!==": [1, 1]}) is False

    def test_comparison_operators(self):
        """Test comparison operators (<, <=, >, >=)."""
        # Less than
        assert evaluate_logic({"<": [1, 2]}) is True
        assert evaluate_logic({"<": [2, 1]}) is False
        assert evaluate_logic({"<": [1, 1]}) is False

        # Less than or equal
        assert evaluate_logic({"<=": [1, 2]}) is True
        assert evaluate_logic({"<=": [1, 1]}) is True
        assert evaluate_logic({"<=": [2, 1]}) is False

        # Greater than
        assert evaluate_logic({">": [2, 1]}) is True
        assert evaluate_logic({">": [1, 2]}) is False
        assert evaluate_logic({">": [1, 1]}) is False

        # Greater than or equal
        assert evaluate_logic({">=": [2, 1]}) is True
        assert evaluate_logic({">=": [1, 1]}) is True
        assert evaluate_logic({">=": [1, 2]}) is False

        # Chained comparisons
        assert evaluate_logic({"<": [1, 2, 3]}) is True  # 1 < 2 < 3
        assert evaluate_logic({"<": [1, 3, 2]}) is False  # 1 < 3 < 2 (false)
        assert evaluate_logic({"<=": [1, 1, 2]}) is True  # 1 <= 1 <= 2

    def test_logical_operators(self):
        """Test logical operators (!, !!, and, or)."""
        # Not
        assert evaluate_logic({"!": [True]}) is False
        assert evaluate_logic({"!": [False]}) is True
        assert evaluate_logic({"!": [0]}) is True
        assert evaluate_logic({"!": [1]}) is False

        # Double-negation (cast to boolean)
        assert evaluate_logic({"!!": [1]}) is True
        assert evaluate_logic({"!!": [0]}) is False
        assert evaluate_logic({"!!": ["hello"]}) is True
        assert evaluate_logic({"!!": [""]}) is False

        # And
        assert evaluate_logic({"and": [True, True]}) is True
        assert evaluate_logic({"and": [True, False]}) is False
        assert evaluate_logic({"and": [False, True]}) is False
        assert evaluate_logic({"and": [False, False]}) is False

        # Multiple arguments with and
        assert evaluate_logic({"and": [True, True, True]}) is True
        assert evaluate_logic({"and": [True, True, False]}) is False

        # Empty and array always returns True (all of empty set is True)
        assert evaluate_logic({"and": []}) is True

        # Or
        assert evaluate_logic({"or": [True, True]}) is True
        assert evaluate_logic({"or": [True, False]}) is True
        assert evaluate_logic({"or": [False, True]}) is True
        assert evaluate_logic({"or": [False, False]}) is False

        # Multiple arguments with or
        assert evaluate_logic({"or": [False, False, True]}) is True
        assert evaluate_logic({"or": [False, False, False]}) is False

        # Empty or array always returns False (any of empty set is False)
        assert evaluate_logic({"or": []}) is False

    def test_arithmetic_operators(self):
        """Test arithmetic operators (+, -, *, /, %)."""
        # Addition
        assert evaluate_logic({"+": [1, 2]}) == 3
        assert evaluate_logic({"+": [1, 2, 3]}) == 6
        assert evaluate_logic({"+": [-1, 1]}) == 0

        # Unary plus (converts to number)
        assert evaluate_logic({"+": ["5"]}) == 5

        # Subtraction
        assert evaluate_logic({"-": [5, 2]}) == 3
        assert evaluate_logic({"-": [2, 5]}) == -3

        # Unary minus
        assert evaluate_logic({"-": [5]}) == -5
        assert evaluate_logic({"-": [-5]}) == 5

        # Multiplication
        assert evaluate_logic({"*": [2, 3]}) == 6
        assert evaluate_logic({"*": [2, 0]}) == 0
        assert evaluate_logic({"*": [2, 3, 4]}) == 24

        # Division
        assert evaluate_logic({"/": [6, 3]}) == 2
        assert evaluate_logic({"/": [5, 2]}) == 2.5

        # Modulo
        assert evaluate_logic({"%": [5, 2]}) == 1
        assert evaluate_logic({"%": [6, 3]}) == 0

    def test_var_operator(self):
        """Test var operator for accessing context data."""
        data = {
            "user": {
                "name": "Alice",
                "age": 30,
                "address": {
                    "city": "New York"
                }
            },
            "items": [10, 20, 30],
            "enabled": True
        }

        # Basic access
        assert evaluate_logic({"var": "user.name"}, data) == "Alice"
        assert evaluate_logic({"var": "user.age"}, data) == 30
        assert evaluate_logic({"var": "enabled"}, data) is True

        # Nested access
        assert evaluate_logic({"var": "user.address.city"}, data) == "New York"

        # Array access
        assert evaluate_logic({"var": "items.1"}, data) == 20

        # Default value when path doesn't exist
        assert evaluate_logic({"var": ["user.email", "no-email"]}, data) == "no-email"
        assert evaluate_logic({"var": ["missing", 42]}, data) == 42

        # Missing without default returns None
        assert evaluate_logic({"var": "missing"}, data) is None
        assert evaluate_logic({"var": "user.phone"}, data) is None

    def test_missing_operators(self):
        """Test missing and missing_some operators."""
        data = {
            "user": {
                "name": "Alice",
                "age": 30
            },
            "items": [10, 20, 30]
        }

        # Missing operator
        assert evaluate_logic({"missing": ["user.name", "user.email"]}, data) == ["user.email"]
        assert evaluate_logic({"missing": ["user.name", "user.age"]}, data) == []
        assert evaluate_logic({"missing": ["missing.path"]}, data) == ["missing.path"]

        # missing_some operator (requires at least N elements to be present)
        # Here we require at least 2 of the 3 fields to be present
        assert evaluate_logic({"missing_some": [2, ["user.name", "user.age", "user.email"]]}, data) == []

        # Here we require at least 3 of the 3 fields to be present
        assert evaluate_logic({"missing_some": [3, ["user.name", "user.age", "user.email"]]}, data) == ["user.email"]

        # min_required = 0 always returns empty list (no missing fields reported)
        assert evaluate_logic({"missing_some": [0, ["missing.path"]]}, data) == []

    def test_if_operator(self):
        """Test if conditional operator."""
        # Basic if-then-else
        assert evaluate_logic({"if": [True, "yes", "no"]}) == "yes"
        assert evaluate_logic({"if": [False, "yes", "no"]}) == "no"

        # Computed condition
        assert evaluate_logic({"if": [{"==": [1, 1]}, "equal", "not equal"]}) == "equal"
        assert evaluate_logic({"if": [{"==": [1, 2]}, "equal", "not equal"]}) == "not equal"

        # Multiple conditions (if/elif/elif/.../else)
        assert evaluate_logic(
            {"if": [{"==": [1, 1]}, "A", {"==": [2, 2]}, "B", "C"]}
        ) == "A"  # First condition true

        assert evaluate_logic(
            {"if": [{"==": [1, 2]}, "A", {"==": [2, 2]}, "B", "C"]}
        ) == "B"  # Second condition true

        assert evaluate_logic(
            {"if": [{"==": [1, 2]}, "A", {"==": [2, 3]}, "B", "C"]}
        ) == "C"  # No conditions true, use default

        # If without else returns None when condition is false
        assert evaluate_logic({"if": [False, "yes"]}) is None

    def test_membership_operators(self):
        """Test in and contains operators."""
        # Test 'in' operator (element in collection)
        assert evaluate_logic({"in": ["a", "abc"]}) is True
        assert evaluate_logic({"in": ["z", "abc"]}) is False
        assert evaluate_logic({"in": [2, [1, 2, 3]]}) is True
        assert evaluate_logic({"in": [4, [1, 2, 3]]}) is False

        # Test 'contains' operator (collection contains element)
        assert evaluate_logic({"contains": ["abc", "a"]}) is True
        assert evaluate_logic({"contains": ["abc", "z"]}) is False
        assert evaluate_logic({"contains": [[1, 2, 3], 2]}) is True
        assert evaluate_logic({"contains": [[1, 2, 3], 4]}) is False

    def test_string_operations(self):
        """Test string operations like cat (concatenation)."""
        assert evaluate_logic({"cat": ["Hello", " ", "World"]}) == "Hello World"
        assert evaluate_logic({"cat": ["The answer is: ", 42]}) == "The answer is: 42"
        assert evaluate_logic({"cat": []}) == ""  # Empty cat returns empty string

    def test_fsm_specific_operations(self):
        """Test FSM-specific operations."""
        data = {
            "user": {
                "name": "Alice",
                "roles": ["user", "admin"]
            },
            "completed": True
        }

        # Test has_context operator
        assert evaluate_logic({"has_context": [data, "user"]}) is True
        assert evaluate_logic({"has_context": [data, "missing"]}) is False

        # Test context_length operator
        assert evaluate_logic({"context_length": [data, "user.roles"]}) == 2
        assert evaluate_logic({"context_length": [data, "user.name"]}) == 5  # Length of string
        assert evaluate_logic({"context_length": [data, "missing"]}) == 0  # Missing key returns 0

    def test_nested_complex_expressions(self):
        """Test complex nested expressions typical in FSM transitions."""
        data = {
            "customer": {
                "status": "vip",
                "lifetime_value": 6000,
                "subscription": {
                    "active": True,
                    "type": "premium"
                }
            },
            "issue": {
                "category": "billing",
                "priority": "high",
                "resolved": False,
                "resolution_time": 0
            },
            "agent": {
                "specialty": ["billing", "technical"],
                "available": True
            }
        }

        # Complex condition: VIP customer with billing issue that's high priority
        complex_expr = {
            "and": [
                {"==": [{"var": "customer.status"}, "vip"]},
                {"==": [{"var": "issue.category"}, "billing"]},
                {"==": [{"var": "issue.priority"}, "high"]}
            ]
        }
        assert evaluate_logic(complex_expr, data) is True

        # More complex: VIP customer OR premium subscription, AND high priority issue that's unresolved
        advanced_expr = {
            "and": [
                {"or": [
                    {"==": [{"var": "customer.status"}, "vip"]},
                    {"==": [{"var": "customer.subscription.type"}, "premium"]}
                ]},
                {"==": [{"var": "issue.priority"}, "high"]},
                {"==": [{"var": "issue.resolved"}, False]}
            ]
        }
        assert evaluate_logic(advanced_expr, data) is True

        # Condition with numeric comparison: VIP and lifetime value > 5000
        numeric_expr = {
            "and": [
                {"==": [{"var": "customer.status"}, "vip"]},
                {">": [{"var": "customer.lifetime_value"}, 5000]}
            ]
        }
        assert evaluate_logic(numeric_expr, data) is True

        # Condition with if and fallback values
        conditional_expr = {
            "if": [
                {"==": [{"var": "issue.resolved"}, True]},
                {"var": "issue.resolution_time"},
                {"var": ["agent.estimated_time", 30]}  # Fallback to 30 if not present
            ]
        }
        assert evaluate_logic(conditional_expr, data) == 30  # Fallback value

    def test_error_handling(self):
        """Test error handling in expression evaluation."""
        # Unsupported operator
        assert evaluate_logic({"unsupported_op": [1, 2]}) is False

        # Division by zero (should not raise exception)
        assert evaluate_logic({"/": [1, 0]}) is False

        # Invalid types in operations
        assert evaluate_logic({"<": ["not_a_number", 5]}) is False

        # Accessing properties of non-objects
        assert evaluate_logic({"var": "a.b.c"}, {"a": 5}) is None