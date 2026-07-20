"""
Tests for the JsonLogic expression evaluator module.

This test suite thoroughly tests the functionality of the JsonLogic expression evaluator,
covering all operators, error handling, and complex expressions that would be used in
real FSM transitions.
"""

from contextlib import contextmanager

from fsm_llm.expressions import evaluate_logic, hard_equals, soft_equals
from fsm_llm.logging import logger


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
            "user": {"name": "Alice", "age": 30, "address": {"city": "New York"}},
            "items": [10, 20, 30],
            "enabled": True,
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
        data = {"user": {"name": "Alice", "age": 30}, "items": [10, 20, 30]}

        # Missing operator
        assert evaluate_logic({"missing": ["user.name", "user.email"]}, data) == [
            "user.email"
        ]
        assert evaluate_logic({"missing": ["user.name", "user.age"]}, data) == []
        assert evaluate_logic({"missing": ["missing.path"]}, data) == ["missing.path"]

        # missing_some operator (requires at least N elements to be present)
        # Here we require at least 2 of the 3 fields to be present
        assert (
            evaluate_logic(
                {"missing_some": [2, ["user.name", "user.age", "user.email"]]}, data
            )
            == []
        )

        # Here we require at least 3 of the 3 fields to be present
        assert evaluate_logic(
            {"missing_some": [3, ["user.name", "user.age", "user.email"]]}, data
        ) == ["user.email"]

        # min_required = 0 always returns empty list (no missing fields reported)
        assert evaluate_logic({"missing_some": [0, ["missing.path"]]}, data) == []

    def test_if_operator(self):
        """Test if conditional operator."""
        # Basic if-then-else
        assert evaluate_logic({"if": [True, "yes", "no"]}) == "yes"
        assert evaluate_logic({"if": [False, "yes", "no"]}) == "no"

        # Computed condition
        assert evaluate_logic({"if": [{"==": [1, 1]}, "equal", "not equal"]}) == "equal"
        assert (
            evaluate_logic({"if": [{"==": [1, 2]}, "equal", "not equal"]})
            == "not equal"
        )

        # Multiple conditions (if/elif/elif/.../else)
        assert (
            evaluate_logic({"if": [{"==": [1, 1]}, "A", {"==": [2, 2]}, "B", "C"]})
            == "A"
        )  # First condition true

        assert (
            evaluate_logic({"if": [{"==": [1, 2]}, "A", {"==": [2, 2]}, "B", "C"]})
            == "B"
        )  # Second condition true

        assert (
            evaluate_logic({"if": [{"==": [1, 2]}, "A", {"==": [2, 3]}, "B", "C"]})
            == "C"
        )  # No conditions true, use default

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
            "user": {"name": "Alice", "roles": ["user", "admin"]},
            "completed": True,
        }

        # Test has_context operator (uses {"var": ""} to reference entire context)
        assert evaluate_logic({"has_context": [{"var": ""}, "user"]}, data) is True
        assert evaluate_logic({"has_context": [{"var": ""}, "missing"]}, data) is False

        # Test context_length operator
        assert (
            evaluate_logic({"context_length": [{"var": ""}, "user.roles"]}, data) == 2
        )
        assert (
            evaluate_logic({"context_length": [{"var": ""}, "user.name"]}, data) == 5
        )  # Length of string
        assert (
            evaluate_logic({"context_length": [{"var": ""}, "missing"]}, data) == 0
        )  # Missing key returns 0

    def test_nested_complex_expressions(self):
        """Test complex nested expressions typical in FSM transitions."""
        data = {
            "customer": {
                "status": "vip",
                "lifetime_value": 6000,
                "subscription": {"active": True, "type": "premium"},
            },
            "issue": {
                "category": "billing",
                "priority": "high",
                "resolved": False,
                "resolution_time": 0,
            },
            "agent": {"specialty": ["billing", "technical"], "available": True},
        }

        # Complex condition: VIP customer with billing issue that's high priority
        complex_expr = {
            "and": [
                {"==": [{"var": "customer.status"}, "vip"]},
                {"==": [{"var": "issue.category"}, "billing"]},
                {"==": [{"var": "issue.priority"}, "high"]},
            ]
        }
        assert evaluate_logic(complex_expr, data) is True

        # More complex: VIP customer OR premium subscription, AND high priority issue that's unresolved
        advanced_expr = {
            "and": [
                {
                    "or": [
                        {"==": [{"var": "customer.status"}, "vip"]},
                        {"==": [{"var": "customer.subscription.type"}, "premium"]},
                    ]
                },
                {"==": [{"var": "issue.priority"}, "high"]},
                {"==": [{"var": "issue.resolved"}, False]},
            ]
        }
        assert evaluate_logic(advanced_expr, data) is True

        # Condition with numeric comparison: VIP and lifetime value > 5000
        numeric_expr = {
            "and": [
                {"==": [{"var": "customer.status"}, "vip"]},
                {">": [{"var": "customer.lifetime_value"}, 5000]},
            ]
        }
        assert evaluate_logic(numeric_expr, data) is True

        # Condition with if and fallback values
        conditional_expr = {
            "if": [
                {"==": [{"var": "issue.resolved"}, True]},
                {"var": "issue.resolution_time"},
                {"var": ["agent.estimated_time", 30]},  # Fallback to 30 if not present
            ]
        }
        assert evaluate_logic(conditional_expr, data) == 30  # Fallback value

    def test_error_handling(self):
        """Test error handling in expression evaluation."""
        # Unsupported operator raises TransitionEvaluationError
        import pytest

        from fsm_llm.definitions import TransitionEvaluationError

        with pytest.raises(
            TransitionEvaluationError, match="Disallowed JsonLogic operation"
        ):
            evaluate_logic({"unsupported_op": [1, 2]})

        # Division by zero raises TransitionEvaluationError
        with pytest.raises(TransitionEvaluationError, match="Division by zero"):
            evaluate_logic({"/": [1, 0]})

        # Invalid types in operations
        assert evaluate_logic({"<": ["not_a_number", 5]}) is False

        # Accessing properties of non-objects
        assert evaluate_logic({"var": "a.b.c"}, {"a": 5}) is None


class TestMissingVarNeverEqualsTheStringNone:
    """T5 / D-017: an unset context var must not compare equal to the literal "None".

    `get_var` resolves a missing key to `None` (`not_found=None`), and
    `soft_equals`'s mixed str/non-str branch used to do `str(a) == str(b)`, so
    `str(None) == "None"` was True. An FSM author checking a field against the
    string "None" could not tell "never extracted" from "the LLM said None".
    """

    # --- the defect itself, at the `soft_equals` unit ---

    def test_none_is_not_the_string_none(self):
        assert soft_equals(None, "None") is False

    def test_the_string_none_is_not_none_reversed_operands(self):
        assert soft_equals("None", None) is False

    def test_none_still_equals_none(self):
        """The guard is scoped to the MIXED branch; None == None stays True."""
        assert soft_equals(None, None) is True

    # --- the seam that matters: the real `evaluate_logic` entry point ---

    def test_missing_var_does_not_match_the_string_none_through_evaluate_logic(self):
        assert evaluate_logic({"==": [{"var": "missing_key"}, "None"]}, {}) is False

    def test_missing_var_not_equal_the_string_none_is_true(self):
        """`!=` is `not soft_equals`, so it inherits the fix at the same choke point."""
        assert evaluate_logic({"!=": [{"var": "missing_key"}, "None"]}, {}) is True

    def test_a_real_none_valued_key_still_matches_the_string_none(self):
        """Do not over-correct: an explicitly-set "None" string must still match."""
        assert (
            evaluate_logic({"==": [{"var": "status"}, "None"]}, {"status": "None"})
            is True
        )

    def test_explicit_python_none_value_still_does_not_match(self):
        """A key present but holding a real None is still 'unset' for this purpose."""
        assert (
            evaluate_logic({"==": [{"var": "status"}, "None"]}, {"status": None})
            is False
        )

    # --- non-regression: every other coercion is intentional and unchanged ---

    def test_numeric_string_coercion_is_unchanged(self):
        assert soft_equals("5", 5) is True
        assert soft_equals(5, "5") is True

    def test_falsy_values_are_not_conflated_with_none(self):
        assert soft_equals(None, "") is False
        assert soft_equals(None, 0) is False
        assert soft_equals(None, False) is False
        assert soft_equals("", "") is True
        assert soft_equals(0, "0") is True

    def test_case_insensitive_string_equality_is_unchanged(self):
        assert soft_equals("Purchase", "purchase") is True

    def test_bool_string_coercion_is_unchanged(self):
        assert soft_equals(True, "true") is True
        assert soft_equals(False, "false") is True

    def test_strict_equality_was_already_correct(self):
        """`===` is type-strict, so it never had this hole. Pinned so it stays true."""
        assert hard_equals(None, "None") is False


# ---------------------------------------------------------------------------
# F-20: the unary `!` operator silently discarded extra arguments
# ---------------------------------------------------------------------------


@contextmanager
def _captured_fsm_llm_warnings():
    """Capture fsm_llm WARNING records the way an embedding app would see them.

    `fsm_llm.logging` calls `logger.disable("fsm_llm")` at import time, so a
    naive sink captures NOTHING and every "no warning was emitted" assertion
    would pass for the wrong reason. This enables the package for the duration
    and restores the import-time default afterwards.
    """
    records: list[str] = []
    logger.enable("fsm_llm")
    sink_id = logger.add(lambda msg: records.append(str(msg)), level="WARNING")
    try:
        yield records
    finally:
        logger.remove(sink_id)
        logger.disable("fsm_llm")


class TestLogicalNotArgumentArity:
    """F-20: `{"!": [a, b]}` negates `a` and drops `b` — now audibly."""

    def test_the_capture_helper_actually_captures(self):
        """Vacuity guard: an empty `records` list must mean 'nothing warned'.

        Without this, every negative case below would pass even if the sink
        were wired up wrong or the package were still log-disabled.
        """
        with _captured_fsm_llm_warnings() as records:
            evaluate_logic({"!!": [True, False]})
        assert any("!!" in r for r in records)

    def test_extra_args_still_return_the_negation_of_the_first(self):
        """SC-20: the RESULT is unchanged — this is observability, not semantics."""
        assert evaluate_logic({"!": [True, False]}) is False
        assert evaluate_logic({"!": [False, True]}) is True
        assert evaluate_logic({"!": [0, 1, 2]}) is True

    def test_extra_args_emit_a_warning_naming_the_dropped_count(self):
        """SC-20: a WARNING is emitted and names how many operands were dropped."""
        with _captured_fsm_llm_warnings() as records:
            result = evaluate_logic({"!": [True, False]})
        assert result is False
        warnings = [r for r in records if "'!'" in r]
        assert len(warnings) == 1, records
        assert "received 2" in warnings[0]
        assert "discarding 1" in warnings[0]

    def test_three_operands_report_two_dropped(self):
        """The count is derived, not a hardcoded '1'."""
        with _captured_fsm_llm_warnings() as records:
            evaluate_logic({"!": [True, False, True]})
        warnings = [r for r in records if "'!'" in r]
        assert len(warnings) == 1, records
        assert "received 3" in warnings[0]
        assert "discarding 2" in warnings[0]

    def test_correct_unary_usage_does_not_warn(self):
        """Over-correction guard: the common, correct shape must stay silent."""
        with _captured_fsm_llm_warnings() as records:
            assert evaluate_logic({"!": [True]}) is False
            assert evaluate_logic({"!": False}) is True
            assert evaluate_logic({"!": [{"var": "flag"}]}, {"flag": 0}) is True
        assert [r for r in records if "'!'" in r] == []

    def test_empty_operand_list_is_unchanged_and_silent(self):
        """`{"!": []}` returned True before and must keep doing so."""
        with _captured_fsm_llm_warnings() as records:
            assert evaluate_logic({"!": []}) is True
        assert [r for r in records if "'!'" in r] == []


# ---------------------------------------------------------------------------
# F-22: soft_equals' documented coercion rule vs. its actual one
# ---------------------------------------------------------------------------


class TestSoftEqualsBoolVersusStringRule:
    """F-22: a bool-vs-string pair is compared as LOWERCASED STRINGS.

    The docstring used to claim "if either value is a boolean, both are
    converted to booleans", which would make `soft_equals(True, "1")` True.
    It is False. These pin the rule the code actually implements.
    """

    def test_bool_against_a_numeric_string_is_a_string_comparison(self):
        """SC-21: NOT JS semantics — `True == "1"` is False here."""
        assert soft_equals(True, "1") is False
        assert soft_equals("1", True) is False
        assert soft_equals(False, "0") is False

    def test_bool_against_a_json_boolean_string_matches(self):
        """SC-21: the rule the string comparison exists to serve."""
        assert soft_equals(True, "true") is True
        assert soft_equals(True, "TRUE") is True
        assert soft_equals(False, "false") is True
        assert soft_equals(True, "false") is False

    def test_bool_against_a_non_string_still_coerces_to_bool(self):
        """The docstring's `bool()` branch is real — it just is not the str one."""
        assert soft_equals(True, 1) is True
        assert soft_equals(1, True) is True
        assert soft_equals(False, 0) is True
        assert soft_equals(True, 2) is True

    def test_bool_against_none_is_identity(self):
        """SC-21 / D-017 neighbourhood: no bool coercion of None."""
        assert soft_equals(True, None) is False
        assert soft_equals(False, None) is False
        assert soft_equals(None, "None") is False


# ---------------------------------------------------------------------------
# G-21: `_op_var` silently dropped `values[2:]`, unlike its three siblings
# ---------------------------------------------------------------------------


class TestVarArgumentArity:
    """SC-16: `{"var": [name, default, junk]}` drops `junk` — now audibly.

    `_op_var` was the only one of the four data-access operator handlers with no
    arity bound and no log line; `_op_missing_some`, `_op_has_context` and
    `_op_context_length` all bound theirs and `logger.error` on violation. This
    is convergence on that in-file pattern, not a new rule.
    """

    def test_extra_args_are_still_dropped_and_the_result_is_unchanged(self):
        """D-012: observability only. The RETURN VALUE must not move."""
        # Missing key -> the default is used; the junk argument changes nothing.
        assert evaluate_logic({"var": ["x", "d", "junk"]}, {}) == "d"
        assert evaluate_logic({"var": ["x", "d", "junk"]}, {"x": "found"}) == "found"
        assert evaluate_logic({"var": ["x", "d", "a", "b"]}, {}) == "d"
        # A present-but-None key still wins over the default (unchanged).
        assert evaluate_logic({"var": ["x", "d", "junk"]}, {"x": None}) is None
        # ...and every form is identical to the same call without the extras.
        for data in ({}, {"x": None}, {"x": "found"}):
            assert evaluate_logic({"var": ["x", "d", "junk"]}, data) == evaluate_logic(
                {"var": ["x", "d"]}, data
            )

    def test_extra_args_emit_an_error_naming_the_argument_count(self):
        """SC-16: the drop is reported, and the message names how many arrived."""
        with _captured_fsm_llm_warnings() as records:
            result = evaluate_logic({"var": ["x", "d", "junk"]}, {})

        assert result == "d"
        errors = [r for r in records if "var operator" in r]
        assert len(errors) == 1, records
        assert "ERROR" in errors[0], errors[0]
        assert "got 3" in errors[0], errors[0]

    def test_the_bound_fires_only_past_two_arguments(self):
        """Vacuity guard: the legal 1- and 2-argument forms stay silent.

        Without this, a bound of `len(values) > 0` would pass the test above.
        """
        with _captured_fsm_llm_warnings() as records:
            assert evaluate_logic({"var": "x"}, {"x": 1}) == 1
            assert evaluate_logic({"var": ["x"]}, {"x": 1}) == 1
            assert evaluate_logic({"var": ["x", "d"]}, {}) == "d"
        assert [r for r in records if "var operator" in r] == []

    def test_the_message_matches_its_siblings_shape(self):
        """The three arity-bounding siblings share one message shape; so does
        this one, so an operator log is greppable by a single pattern."""
        with _captured_fsm_llm_warnings() as records:
            evaluate_logic({"var": ["x", "d", "junk"]}, {})
            evaluate_logic({"missing_some": [1]}, {})
            evaluate_logic({"has_context": [{}, "k", "extra"]}, {})
            evaluate_logic({"context_length": [{}]}, {})

        arity_errors = [r for r in records if "requires" in r and ", got " in r]
        assert len(arity_errors) == 4, records
