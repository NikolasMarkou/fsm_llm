"""The eager operator table and the short-circuit operator set."""

from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from typing import Any

from .arithmetic import _minus, _safe_div, _safe_mod
from .comparison import (
    _number,
    greater,
    greater_or_equal,
    hard_equals,
    less,
    less_or_equal,
    soft_equals,
)
from .logic_and_text import _logical_not, _membership, cat

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
    "!": _logical_not,  # Logical NOT (unary; warns on extra args)
    "!!": bool,  # Double negation (convert to boolean)
    # "and", "or", "if": short-circuit, see `_SHORT_CIRCUIT_OPERATORS` below
    # Note: Access operators handled directly in evaluate_logic()
    # "var", "missing", "missing_some", "has_context", "context_length"
    # Membership operators
    "in": _membership,
    "contains": lambda a, b: _membership(b, a),
    # Arithmetic operators
    "+": lambda *args: sum(_number(arg) for arg in args),
    "-": _minus,
    "*": lambda *args: reduce(lambda x, y: _number(x) * _number(y), args, 1.0),
    "/": _safe_div,
    "%": _safe_mod,
    # Min/max operators (with numeric coercion like other arithmetic ops)
    "min": lambda *args: min(_number(a) for a in args) if args else None,
    "max": lambda *args: max(_number(a) for a in args) if args else None,
    # String operators
    "cat": cat,
}


#: Operators that MUST evaluate their operands lazily (reference JsonLogic).
_SHORT_CIRCUIT_OPERATORS: frozenset[str] = frozenset({"and", "or", "if"})

# DECISION plan-2026-09-22T080837-8b258a25/D-012
# and/or/if live ONLY in `_evaluate_short_circuit`. Do NOT add eager entries for
# them to `operations` (the old unreachable fallbacks): the eager path evaluates
# every operand first, which breaks the D-006 guard idiom. This check fails the
# import if one is ever re-added. See decisions.md D-012.
if not _SHORT_CIRCUIT_OPERATORS.isdisjoint(operations):
    raise RuntimeError("short-circuit operators must not be in `operations`")
