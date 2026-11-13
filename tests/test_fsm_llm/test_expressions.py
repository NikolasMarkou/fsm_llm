"""
Tests for the JsonLogic expression evaluator module.

This test suite thoroughly tests the functionality of the JsonLogic expression evaluator,
covering all operators, error handling, and complex expressions that would be used in
real FSM transitions.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from fsm_llm import expressions
from fsm_llm.constants import ALLOWED_JSONLOGIC_OPERATIONS
from fsm_llm.core.definitions import (
    FSMContext,
    Transition,
    TransitionCondition,
    TransitionEvaluation,
    TransitionEvaluationResult,
)
from fsm_llm.core.transition_evaluator import TransitionEvaluator
from fsm_llm.expressions import evaluate_logic, hard_equals, soft_equals
from fsm_llm.logging import logger
from tests.test_fsm_llm.support import _a2_state, _gate_fsm, _Prov


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

        from fsm_llm.core.definitions import TransitionEvaluationError

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


class TestShortCircuitEvaluation:
    """H5: `and`/`or`/`if` evaluate ONLY the taken branch (reference JsonLogic).

    A guard idiom whose UNTAKEN branch would raise (div-by-zero / missing var)
    must return a clean verdict without ever evaluating that branch. The
    TAKEN-branch total contract is unchanged: a div-by-zero on the taken path
    still raises `TransitionEvaluationError` (see `test_error_handling`).
    """

    def test_and_short_circuits_before_div_by_zero_guard(self):
        # x=0: the `!=` guard is False, so the second operand (100/x) is never
        # evaluated. Before the fix this raised TransitionEvaluationError.
        logic = {
            "and": [
                {"!=": [{"var": "x"}, 0]},
                {"<": [{"/": [100, {"var": "x"}]}, 5]},
            ]
        }
        assert evaluate_logic(logic, {"x": 0}) is False

    def test_or_short_circuits_before_div_by_zero_guard(self):
        # x=0: the first operand is True, so 100/x is never evaluated.
        logic = {
            "or": [
                {"==": [{"var": "x"}, 0]},
                {"<": [{"/": [100, {"var": "x"}]}, 5]},
            ]
        }
        assert evaluate_logic(logic, {"x": 0}) is True

    def test_if_does_not_evaluate_the_untaken_else_branch(self):
        # mode="a": the condition matches, so the else branch (1/count with
        # count=0) is never evaluated. Before the fix this raised.
        logic = {
            "if": [
                {"==": [{"var": "mode"}, "a"]},
                "OK",
                {"/": [1, {"var": "count"}]},
            ]
        }
        assert evaluate_logic(logic, {"mode": "a", "count": 0}) == "OK"


class TestFailedArithmeticIsUndefined:
    """F-DV-02: an arithmetic operator that cannot compute (non-numeric operand,
    wrong arity) is UNDEFINED, exactly like the None-operand twin, so the
    enclosing comparison can never read it as 0."""

    def test_non_numeric_plus_never_satisfies_less_than(self):
        assert evaluate_logic({"<": [{"+": ["abc", 1]}, 5]}) is False

    def test_non_numeric_division_never_equals_zero(self):
        assert evaluate_logic({"==": [{"/": ["abc", 2]}, 0]}) is False

    def test_non_numeric_var_minus_one_never_differs_from_zero(self):
        logic = {"!=": [{"-": [{"var": "total"}, 1]}, 0]}
        assert evaluate_logic(logic, {"total": "N/A"}) is False

    def test_wrong_arity_minus_never_satisfies_less_than(self):
        assert evaluate_logic({"<": [{"-": [1, 2, 3]}, 5]}) is False

    def test_every_propagating_operator_is_unsatisfiable_on_garbage(self):
        for op in ("+", "-", "*", "/", "%", "min", "max"):
            for cmp_op, rhs in (
                ("<", 5),
                (">=", -5),
                ("==", 0),
                ("!=", 0),
                ("in", [0, 1]),
            ):
                logic = {cmp_op: [{op: ["abc", 2]}, rhs]}
                assert evaluate_logic(logic) is False, (op, cmp_op)

    def test_wrong_arity_modulo_and_division_are_unsatisfiable(self):
        assert evaluate_logic({"==": [{"%": [5]}, 0]}) is False
        assert evaluate_logic({"==": [{"/": [1, 2, 3]}, 0]}) is False

    def test_none_operand_twin_stays_false(self):
        logic = {"<": [{"-": [{"var": "t"}, 1]}, 5]}
        assert evaluate_logic(logic, {"t": None}) is False

    def test_bare_failed_arithmetic_is_none_at_the_public_boundary(self):
        assert evaluate_logic({"+": ["abc", 1]}) is None
        assert evaluate_logic({"-": [1, 2, 3]}) is None

    def test_failed_arithmetic_is_falsy_in_boolean_context(self):
        assert evaluate_logic({"!": [{"+": ["abc", 1]}]}) is True
        assert evaluate_logic({"!!": [{"+": ["abc", 1]}]}) is False
        assert evaluate_logic({"or": [{"+": ["abc", 1]}, "fallback"]}) == "fallback"

    def test_failed_arithmetic_propagates_through_nesting(self):
        logic = {"<": [{"*": [{"+": ["abc", 1]}, 2]}, 5]}
        assert evaluate_logic(logic) is False

    def test_non_propagating_operator_failure_stays_false(self):
        assert evaluate_logic({"in": [{"var": "x"}, "abc"]}, {"x": [1]}) is False
        logic = {"in": [{"var": "x"}, {"var": "y"}]}
        assert evaluate_logic(logic, {"x": [1], "y": {"a": 1}}) is False

    def test_valid_numeric_input_is_unchanged(self):
        assert evaluate_logic({"+": ["2", 1]}) == 3.0
        assert evaluate_logic({"<": [{"+": ["2", 1]}, 5]}) is True
        assert evaluate_logic({"-": [3]}) == -3.0
        assert evaluate_logic({"max": ["4", 9]}) == 9.0
        assert evaluate_logic({"cat": ["a", 1]}) == "a1"

    def test_division_by_zero_still_raises(self):
        import pytest

        from fsm_llm.core.definitions import TransitionEvaluationError

        with pytest.raises(TransitionEvaluationError):
            evaluate_logic({"/": [1, 0]})
        with pytest.raises(TransitionEvaluationError):
            evaluate_logic({"%": [1, 0]})


class TestNumericStringOrdering:
    """F-DV-06 (D-005): ``<=``/``>=`` also accept numerically equal operands
    that ``==`` refuses. The behaviour is kept; these tests state it."""

    def test_le_ge_accept_equal_numeric_strings_where_eq_does_not(self):
        """PIN (green on the parent, not a RED): the asymmetry is deliberate.

        Dropping the ``_numeric_equal`` term from ``<=``/``>=`` would flip a
        numeric-string gate such as ``>= "50"`` on an extracted ``"50.0"``
        silently; making ``==`` numeric on two strings breaks the ids rule.
        """
        # Two differently spelled numeric strings.
        for a, b in (("01", "1"), ("1", "1.0"), (".5", "0.5")):
            assert evaluate_logic({"==": [a, b]}) is False, (a, b)
            assert evaluate_logic({"<=": [a, b]}) is True, (a, b)
            assert evaluate_logic({">=": [a, b]}) is True, (a, b)
            # Strict ordering never calls the helper, so it stays False.
            assert evaluate_logic({"<": [a, b]}) is False, (a, b)
            assert evaluate_logic({">": [a, b]}) is False, (a, b)
        # A bool against a numeric string.
        assert evaluate_logic({"==": [True, "1"]}) is False
        assert evaluate_logic({"<=": [True, "1"]}) is True
        assert evaluate_logic({">=": [True, "1"]}) is True
        # The gate the kept behaviour protects.
        gate = {">=": [{"var": "score"}, "50"]}
        assert evaluate_logic(gate, {"score": "50.0"}) is True
        # A number against a numeric string is the case where all three agree.
        assert evaluate_logic({"==": [1.0, "1"]}) is True
        assert evaluate_logic({"<=": [1.0, "1"]}) is True
        # Non-numbers are never equal through the helper.
        assert evaluate_logic({"<=": ["1_000", "1000"]}) is False

    def test_module_docstring_does_not_claim_eq_le_ge_agree(self):
        import fsm_llm.expressions as expressions

        doc = " ".join((expressions.__doc__ or "").split())
        assert "so ``==``, ``<=`` and ``>=`` agree" not in doc
        # It says what the operators do differ on instead.
        assert '``"01" <= "1"``' in doc
        helper = " ".join((expressions._numeric_equal.__doc__ or "").split())
        assert "so the three agree on numeric equality" not in helper
        assert '``"01" <= "1"``' in helper


def _level_gate_fsm(op: str) -> dict:
    fsm = _gate_fsm()
    gate = fsm["states"]["gate"]
    gate["required_context_keys"] = ["level"]
    cond = gate["transitions"][0]["conditions"][0]
    cond["requires_context_keys"] = ["level"]
    cond["logic"] = (
        {op: [{"var": "level"}, 2]} if op == ">=" else {op: [2, {"var": "level"}]}
    )
    return fsm


class TestNumericStringInequalityAgreesWithEquality:
    @pytest.mark.parametrize(
        "logic",
        [
            {"<=": [1, "1.0"]},
            {">=": ["2", 2]},
            {"<=": ["1.0", 1]},
            {">=": [2, "2.0"]},
            {"<=": [1, "1.0", 2]},
        ],
    )
    def test_numeric_string_equal_pair_satisfies_the_inclusive_operator(self, logic):
        assert evaluate_logic(logic) is True

    @pytest.mark.parametrize(
        "logic",
        [
            {"<=": ["abc", 1]},
            {">=": [None, 1]},
            {"<=": [2, "1.0"]},
            {">=": [1, "1.5"]},
            {"<=": ["abc", "abd"]},
        ],
    )
    def test_non_numeric_and_unequal_pairs_keep_their_old_result(self, logic):
        expected = {"<=": ["abc", "abd"]} == logic
        assert evaluate_logic(logic) is expected

    def test_equality_operator_agrees_with_the_inclusive_operators(self):
        # Re-baselined by plan-2026-09-21T203800-8a03483a/D-013 (B3), which
        # supersedes the D-023 "== untouched" pin: `==` now accepts the same
        # numerically equal pairs `<=`/`>=` do; bools stay uncoerced.
        assert evaluate_logic({"==": [1, "1.0"]}) is True
        assert evaluate_logic({"!=": [1, "1.0"]}) is False
        assert evaluate_logic({"==": [True, "1.0"]}) is False

    @pytest.mark.parametrize("op", [">=", "<="])
    def test_transition_condition_fires_through_converse(self, op):
        with _Prov(_level_gate_fsm(op)) as p:
            p.turn(field={"level": "2.0"})
            assert p.api.get_current_state(p.cid) == "admin"


def _b4_evaluate(
    condition: TransitionCondition,
    data: dict[str, Any],
    extracted: dict[str, Any] | None = None,
) -> TransitionEvaluation:
    context = FSMContext()
    context.data.update(data)
    return TransitionEvaluator().evaluate_transitions(
        _a2_state(
            [Transition(target_state="next", description="go", conditions=[condition])]
        ),
        context,
        extracted,
    )


class TestStep09B1B4:
    """B1-B4: None/missing is one documented rule across every operator family."""

    def test_b1_minus_with_none_second_operand_is_none(self):
        """Never ``-total`` (the pre-fix unary reading). Step 9.2 (D-048):
        None, not False, so an enclosing comparison cannot read it as 0."""
        from fsm_llm.expressions import evaluate_logic

        logic = {"-": [{"var": "total"}, {"var": "discount"}]}
        assert evaluate_logic(logic, {"total": 50}) is None
        assert evaluate_logic(logic, {"total": 50, "discount": None}) is None
        assert evaluate_logic({"-": [5, None]}) is None
        # Guard: the rest of the arithmetic family is None on a None operand too.
        for op in ("+", "*", "/", "%", "min", "max"):
            assert evaluate_logic({op: [{"var": "a"}, 2]}, {}) is None, op

    def test_b1_unary_minus_unchanged(self):
        from fsm_llm.expressions import evaluate_logic

        assert evaluate_logic({"-": [5]}) == -5
        assert evaluate_logic({"-": [-5]}) == 5
        assert evaluate_logic({"-": 5}) == -5
        assert evaluate_logic({"-": [5, 2]}) == 3
        assert evaluate_logic({"-": [{"var": "x"}]}, {"x": 4}) == -4
        assert evaluate_logic({"-": [{"var": "x"}]}, {}) is None  # D-048

    def test_b2_le_ge_false_when_both_unset(self):
        from fsm_llm.expressions import evaluate_logic

        for op in ("<=", ">=", "<", ">"):
            assert evaluate_logic({op: [{"var": "x"}, {"var": "y"}]}, {}) is False, op
            assert evaluate_logic({op: [None, None]}) is False, op
            assert evaluate_logic({op: [{"var": "x"}, 5]}, {"x": None}) is False, op
        # A three-operand chain with a None link is False as well.
        assert evaluate_logic({"<=": [1, {"var": "x"}, 3]}, {}) is False
        assert evaluate_logic({"<=": [1, {"var": "x"}, 3]}, {"x": 2}) is True

    def test_b2_null_eq_null_still_true(self):
        from fsm_llm.expressions import evaluate_logic, soft_equals

        assert soft_equals(None, None) is True
        assert evaluate_logic({"==": [{"var": "x"}, {"var": "y"}]}, {}) is True
        assert evaluate_logic({"!=": [{"var": "x"}, {"var": "y"}]}, {}) is False
        assert evaluate_logic({"==": [{"var": "x"}, 0]}, {}) is False

    @pytest.mark.parametrize(
        ("a", "b"),
        [(1.0, "1"), ("1", 1.0), (2, "2.00"), ("1e3", 1000), (0.5, ".5")],
    )
    def test_b3_eq_numeric_coercion_agrees_with_le_ge(self, a, b):
        from fsm_llm.expressions import evaluate_logic

        assert evaluate_logic({"<=": [a, b]}) is True
        assert evaluate_logic({">=": [a, b]}) is True
        assert evaluate_logic({"==": [a, b]}) is True
        assert evaluate_logic({"!=": [a, b]}) is False
        assert evaluate_logic({"==": [{"var": "v"}, b]}, {"v": a}) is True

    @pytest.mark.parametrize(("a", "b"), [("01", "1"), ("1", "1.0"), ("1e3", "1000")])
    def test_b3_two_strings_are_never_numerically_coerced(self, a, b):
        """Two strings compare as strings (JS semantics): leading zeros matter."""
        from fsm_llm.expressions import evaluate_logic

        assert evaluate_logic({"==": [a, b]}) is False
        assert evaluate_logic({"!=": [a, b]}) is True
        assert evaluate_logic({"==": [{"var": "zip"}, b]}, {"zip": a}) is False

    def test_b3_unequal_numbers_stay_unequal(self):
        """Guard: coercion only equates numerically equal values."""
        from fsm_llm.expressions import evaluate_logic

        assert evaluate_logic({"==": [1.5, "1"]}) is False
        assert evaluate_logic({"==": ["01", "1.5"]}) is False
        assert evaluate_logic({"==": ["abc", "ABC"]}) is True
        assert evaluate_logic({"==": ["abc", 1]}) is False

    def test_b3_bool_not_coerced(self):
        from fsm_llm.expressions import evaluate_logic, soft_equals

        assert soft_equals(True, "1") is False
        assert soft_equals(False, "0") is False
        assert soft_equals(True, "1.0") is False
        assert soft_equals(True, "true") is True
        assert evaluate_logic({"==": [True, 1]}) is True
        assert evaluate_logic({"==": [True, "1"]}) is False

    def test_b3_missing_var_never_equals_string_None(self):
        from fsm_llm.expressions import evaluate_logic, soft_equals

        for probe in ("None", "none", "NONE", "nan", ""):
            assert evaluate_logic({"==": [{"var": "x"}, probe]}, {}) is False, probe
            assert evaluate_logic({"!=": [{"var": "x"}, probe]}, {}) is True, probe
            assert soft_equals(None, probe) is False, probe

    def test_b4_missing_treats_none_and_empty_as_missing(self):
        from fsm_llm.expressions import evaluate_logic

        data = {"a": None, "b": "", "c": 0, "d": False, "e": [], "f": "x"}
        assert evaluate_logic(
            {"missing": ["a", "b", "c", "d", "e", "f", "g"]}, data
        ) == [
            "a",
            "b",
            "g",
        ]
        assert evaluate_logic({"missing_some": [2, ["a", "b", "f"]]}, data) == [
            "a",
            "b",
        ]
        assert evaluate_logic({"missing_some": [1, ["a", "b", "f"]]}, data) == []
        nested = {"user": {"email": "", "name": "Ada"}}
        assert evaluate_logic({"missing": ["user.email", "user.name"]}, nested) == [
            "user.email"
        ]

    def test_b4_requires_context_keys_none_and_empty_block(self):
        condition = TransitionCondition(
            description="email known", requires_context_keys=["email"]
        )
        for value in (None, ""):
            result = _b4_evaluate(condition, {"email": value})
            assert result.result_type == TransitionEvaluationResult.BLOCKED, value
        for value in ("a@b.c", 0, False):
            result = _b4_evaluate(condition, {"email": value})
            assert result.result_type == TransitionEvaluationResult.DETERMINISTIC, value

    def test_b4_requires_context_keys_shares_the_missing_helper(self):
        """Guard: one predicate, not three copies of the sentinel idiom."""
        import inspect

        from fsm_llm import expressions
        from fsm_llm.core import transition_evaluator

        assert transition_evaluator.is_missing is expressions.is_missing
        source = inspect.getsource(expressions) + inspect.getsource(
            transition_evaluator
        )
        assert "_not_found = object()" not in source
        assert "not_found = object()" not in source

    def test_b4_extracted_none_does_not_overwrite_stored(self):
        condition = TransitionCondition(
            description="email is ada",
            requires_context_keys=["email"],
            logic={"==": [{"var": "email"}, "ada@example.com"]},
        )
        result = _b4_evaluate(
            condition, {"email": "ada@example.com"}, extracted={"email": None}
        )
        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC
        # A non-None extracted value still overrides the stored one.
        result = _b4_evaluate(
            condition, {"email": "ada@example.com"}, extracted={"email": "bob@x.y"}
        )
        assert result.result_type == TransitionEvaluationResult.BLOCKED


class TestStep09_2:
    """B1/B3 completion: arithmetic on a None operand is None (not False, which
    ordering operators coerced to 0), so an enclosing comparison stays False;
    ``==`` coerces only plain decimal/scientific numeric strings
    (plan-2026-09-21T203800-8a03483a/D-048)."""

    def test_b1_nested_arithmetic_with_unset_operand_compares_false(self):
        from fsm_llm.expressions import evaluate_logic

        audit = {"<": [{"-": [{"var": "total"}, {"var": "discount"}]}, 100]}
        assert evaluate_logic(audit, {"total": 150}) is False
        assert evaluate_logic(audit, {"total": 150, "discount": 60}) is True
        assert evaluate_logic({"<": [{"+": [{"var": "score"}, 0]}, 50]}, {}) is False
        assert evaluate_logic({">": [{"-": [{"var": "a"}, 5]}, -10]}, {}) is False
        assert evaluate_logic({"==": [{"-": [{"var": "a"}, 5]}, 0]}, {}) is False
        assert (
            evaluate_logic({"<": [{"max": [{"var": "a"}, {"var": "b"}]}, 5]}, {})
            is False
        )
        # None propagates through nested arithmetic too.
        nested = {"-": [{"+": [{"var": "a"}, 1]}, 2]}
        assert evaluate_logic(nested, {}) is None
        assert evaluate_logic({"<=": [nested, 0]}, {}) is False

    def test_b1_bare_arithmetic_condition_with_unset_operand_fails(self):
        """A bare arithmetic condition is None -> falsy -> the gate stays shut."""
        condition = TransitionCondition(
            description="net positive",
            logic={"-": [{"var": "total"}, {"var": "discount"}]},
        )
        result = _b4_evaluate(condition, {"total": 150})
        assert result.result_type == TransitionEvaluationResult.BLOCKED

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("1_000", 1000),
            (1000, "1_000"),
            (" 1 ", 1),
            ("1\n", 1),
            ("inf", float("inf")),
            ("Infinity", float("inf")),
            ("nan", float("nan")),
            ("0x10", 16),
            ("1e1_0", 1e10),
        ],
    )
    def test_b3_underscore_and_nonstandard_numeric_strings_not_coerced(self, a, b):
        from fsm_llm.expressions import _numeric_equal, evaluate_logic

        assert _numeric_equal(a, b) is False
        if str(a) != str(b):
            # (`"inf" == inf` and `"nan" == nan` stay True through the older
            # mixed str/number rule, `str(a) == str(b)`; that is not coercion.)
            assert evaluate_logic({"==": [a, b]}) is False
            assert evaluate_logic({"!=": [a, b]}) is True

    @pytest.mark.parametrize(
        ("a", "b"),
        [("1", 1), ("-2.5", -2.5), ("+3", 3), ("1e3", 1000), (".5", 0.5), ("5.", 5)],
    )
    def test_b3_plain_numeric_strings_still_coerced(self, a, b):
        """Guard (passes on the pre-step source)."""
        from fsm_llm.expressions import evaluate_logic

        assert evaluate_logic({"==": [a, b]}) is True


_B93_A = {"-": [{"var": "a"}, 5]}  # arithmetic on an unset operand


class TestStep09_3:
    """Review pass 2 concern 2: arithmetic on a missing operand is an internal
    UNDEFINED value that no comparison (``==``, ``!=``, ``===``, ``!==``,
    ordering, ``in``, ``contains``) can satisfy; logical operators treat it as
    falsy; ``evaluate_logic`` returns it as None. ``_PLAIN_NUMBER_RE`` is ASCII
    (plan-2026-09-21T203800-8a03483a/D-054)."""

    @pytest.mark.parametrize(
        ("logic", "data"),
        [
            (
                {"!=": [{"-": [{"var": "balance"}, {"var": "paid"}]}, 0]},
                {"balance": 10},
            ),
            ({"==": [{"+": [{"var": "a"}, 1]}, {"var": "b"}]}, {}),
            ({"==": [_B93_A, None]}, {}),
            ({"===": [_B93_A, None]}, {}),
            ({"!=": [_B93_A, 0]}, {}),
            ({"!==": [_B93_A, 0]}, {}),
            ({"in": [_B93_A, [None, 1]]}, {}),
            ({"contains": [[None, 1], _B93_A]}, {}),
            ({"<=": [_B93_A, 0]}, {}),
            ({">": [_B93_A, -10]}, {}),
            ({"!=": [{"-": [{"+": [{"var": "a"}, 1]}, 2]}, 0]}, {}),
            ({"!=": [{"max": [_B93_A, 1]}, 0]}, {}),
        ],
    )
    def test_b1_comparison_with_undefined_arithmetic_is_false(self, logic, data):
        from fsm_llm.expressions import evaluate_logic

        assert evaluate_logic(logic, data) is False

    def test_b1_undefined_is_falsy_in_logical_operators(self):
        from fsm_llm.expressions import evaluate_logic

        assert evaluate_logic({"!": [_B93_A]}, {}) is True
        assert evaluate_logic({"!!": [_B93_A]}, {}) is False
        assert evaluate_logic({"and": [_B93_A]}, {}) is None
        assert evaluate_logic({"or": [_B93_A, "x"]}, {}) == "x"
        assert evaluate_logic({"if": [_B93_A, "y", "n"]}, {}) == "n"
        # A top-level UNDEFINED comes back as None, nested arithmetic too.
        assert evaluate_logic(_B93_A, {}) is None
        assert evaluate_logic({"*": [{"+": [_B93_A, 1]}, 2]}, {}) is None

    def test_b1_undefined_arithmetic_gate_does_not_fire(self):
        """The review's "outstanding balance" gate with ``paid`` unset."""
        condition = TransitionCondition(
            description="balance outstanding",
            logic={"!=": [{"-": [{"var": "balance"}, {"var": "paid"}]}, 0]},
        )
        result = _b4_evaluate(condition, {"balance": 10})
        assert result.result_type == TransitionEvaluationResult.BLOCKED

    def test_b1_real_nulls_unchanged(self):
        """Guard (passes on the pre-step source): D-017 ``null == null`` for
        real nulls and set arithmetic are unchanged."""
        from fsm_llm.expressions import evaluate_logic

        assert evaluate_logic({"==": [{"var": "x"}, None]}, {}) is True
        assert evaluate_logic({"==": [{"var": "x"}, {"var": "y"}]}, {}) is True
        assert evaluate_logic({"!=": [{"var": "x"}, None]}, {}) is False
        assert evaluate_logic({"in": [{"var": "x"}, [None, 1]]}, {}) is True
        assert evaluate_logic({"!=": [_B93_A, 0]}, {"a": 6}) is True
        assert evaluate_logic({"==": [_B93_A, 1]}, {"a": 6}) is True

    @pytest.mark.parametrize(
        ("a", "b"), [("١٠٠٠", 1000), ("１", 1), ("٣", 3), ("1٠", 10)]
    )
    def test_b3_unicode_digits_not_numeric(self, a, b):
        from fsm_llm.expressions import _numeric_equal, evaluate_logic

        assert _numeric_equal(a, b) is False
        assert evaluate_logic({"==": [a, b]}) is False
        assert evaluate_logic({"<=": [a, b]}) is False


class TestOperatorSweep:
    def test_short_circuit_operators_never_reach_operations(self):
        short_circuit = expressions._SHORT_CIRCUIT_OPERATORS
        assert short_circuit == {"and", "or", "if"}
        assert short_circuit.isdisjoint(expressions.operations)

    def test_every_allowed_operator_has_exactly_one_dispatcher(self):
        tables = [
            set(expressions.operations),
            set(expressions._data_operators),
            set(expressions._SHORT_CIRCUIT_OPERATORS),
        ]
        assert set().union(*tables) == ALLOWED_JSONLOGIC_OPERATIONS
        assert sum(len(t) for t in tables) == len(ALLOWED_JSONLOGIC_OPERATIONS)

    @pytest.mark.parametrize("operator", sorted(expressions.operations))
    def test_every_eager_operator_is_dispatched(self, operator, monkeypatch):
        """Each ``operations`` entry is the one ``evaluate_logic`` calls."""
        sentinel = object()
        monkeypatch.setitem(expressions.operations, operator, lambda *a: sentinel)
        assert expressions.evaluate_logic({operator: [1, 2]}, {}) is sentinel

    @pytest.mark.parametrize("operator", sorted(expressions._data_operators))
    def test_every_data_operator_is_dispatched(self, operator, monkeypatch):
        sentinel = object()
        monkeypatch.setitem(expressions._data_operators, operator, lambda *a: sentinel)
        assert expressions.evaluate_logic({operator: ["x"]}, {}) is sentinel

    @pytest.mark.parametrize("operator", ["and", "or", "if"])
    def test_every_short_circuit_operator_is_dispatched(self, operator, monkeypatch):
        seen: list[str] = []

        def spy(op, values, data, depth):
            seen.append(op)
            return "spied"

        monkeypatch.setattr(expressions, "_evaluate_short_circuit", spy)
        assert expressions.evaluate_logic({operator: [True, 1, 2]}, {}) == "spied"
        assert seen == [operator]
