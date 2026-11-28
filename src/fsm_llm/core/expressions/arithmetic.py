"""Arithmetic helpers of the operator table: safe division and modulo, unary and binary minus.

Every operand goes through ``comparison._number``, the one numeric-string rule.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.core.definitions import TransitionEvaluationError

from .comparison import _number


def _safe_div(a: Any, b: Any) -> float:
    """Division with error on zero divisor."""
    divisor = _number(b)
    if divisor == 0:
        raise TransitionEvaluationError("Division by zero in JsonLogic expression")
    return _number(a) / divisor


def _safe_mod(a: Any, b: Any) -> float:
    """Modulo with error on zero divisor.

    Python floored-modulo semantics: the result takes the sign of the
    DIVISOR, so ``{"%": [-7, 3]}`` is ``2.0`` (JavaScript's ``%``, and so
    reference JsonLogic, gives ``-1``: the sign of the dividend). Kept on
    purpose: changing it would silently flip deployed conditions on negative
    operands (audit B8).
    """
    divisor = _number(b)
    if divisor == 0:
        raise TransitionEvaluationError("Modulo by zero in JsonLogic expression")
    return _number(a) % divisor


def _minus(*args: Any) -> float:
    """``-``: unary negation for exactly one operand, else ``a - b``.

    Arity is decided by operand COUNT, never by a ``None`` default, so a
    second operand that resolved to ``None`` is not mistaken for "omitted"
    (and ``evaluate_logic`` already returns None for it). Any other arity
    raises ``TypeError``, which ``evaluate_logic`` turns into UNDEFINED
    (an arithmetic operator that cannot compute never satisfies a comparison).
    """
    if len(args) == 1:
        return -_number(args[0])
    a, b = args
    return _number(a) - _number(b)
