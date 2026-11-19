"""A total, bounded evaluator for the expression the ``simple_calculator`` strategy extracts.

``evaluate_expression(text)`` returns an ``int`` or a finite ``float``, or raises ``ValueError``
with a fixed message, for ANY value (never ``SyntaxError``, ``ZeroDivisionError``,
``OverflowError``, ``TypeError``, ``RecursionError`` or ``MemoryError``), and never calls
``eval`` or ``compile``. The text is untrusted model output: it is capped before it is
parsed and no integer grows past ``MAX_RESULT_BITS`` bits, so each call takes microseconds.
"""

from __future__ import annotations

import ast
import math
import operator

__all__ = ["calculate", "evaluate_expression"]

#: Longest accepted text, checked before ``ast.parse``.
MAX_EXPRESSION_CHARS = 200
#: Most expression nodes (numbers and operators): a sum of 50 terms passes.
MAX_EXPRESSION_NODES = 100
#: Largest ``|exponent|`` of ``**``.
MAX_EXPONENT = 1000
#: Largest integer, in bits: 617 digits, under the 640-digit floor of
#: ``sys.set_int_max_str_digits``, so ``str(result)`` never raises.
MAX_RESULT_BITS = 2048

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


# DECISION plan-2026-11-19T025425-51fde95b/D-029
# ``^`` is NOT translated to ``**``: it parses as xor and is refused with the
# fixed message. Sixteen of sixteen model outputs wrote ``**`` when the prompt
# shows ``2 ** 3``, so a silent rewrite of model text buys nothing; the refusal
# is a bounded retry (D-052: a refused expression is no solution, never the
# model's own arithmetic). Do NOT add the rewrite and do NOT widen the operators.
def evaluate_expression(text: object) -> int | float:
    """The value of ``text``: numbers, ``+ - * / // % **`` and parentheses.

    ``^`` is xor, not a power, and is refused. An integer-valued float below 2**53 is
    returned as an ``int`` (``0.15 * 240`` is ``36``); another float keeps its IEEE
    value (one that underflows is ``0``). Raises ``ValueError`` for a non-text value, a
    text over ``MAX_EXPRESSION_CHARS``, more than ``MAX_EXPRESSION_NODES`` nodes,
    anything but arithmetic (a name, call, string, ``bool``, complex), a zero divisor,
    ``|exponent| > MAX_EXPONENT``, a non-finite or complex result or an over-large int.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("no expression")
    text = text.strip()
    if len(text) > MAX_EXPRESSION_CHARS:
        raise ValueError("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise ValueError("not an arithmetic expression") from None
    if (
        sum(isinstance(node, ast.expr) for node in ast.walk(tree))
        > MAX_EXPRESSION_NODES
    ):
        raise ValueError("expression too large")
    try:
        value = _evaluate(tree.body)
    except ZeroDivisionError:
        raise ValueError("division by zero") from None
    except OverflowError:
        raise ValueError("result out of range") from None
    if isinstance(value, float) and value.is_integer() and abs(value) < 2**53:
        return int(value)
    return value


def calculate(text: object) -> tuple[int | float | None, str | None]:
    """``(value, None)``, or ``(None, reason)`` for a refused expression; never raises."""
    try:
        return evaluate_expression(text), None
    except ValueError as refusal:
        return None, str(refusal)


def _checked(value: object) -> int | float:
    """``value`` when a finite real number within the bit bound, else ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("result is not a real number")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("result out of range")
    elif value.bit_length() > MAX_RESULT_BITS:
        raise ValueError("result out of range")
    return value


def _evaluate(node: ast.expr) -> int | float:
    if isinstance(node, ast.Constant):
        return _checked(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _evaluate(node.operand)
        return _checked(-operand if isinstance(node.op, ast.USub) else operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ValueError("exponent too large")
            if isinstance(left, int) and isinstance(right, int) and right > 0:
                # refuse before computing: (2 ** 1000) ** 1000 is 1 Mbit of work
                if (left.bit_length() - 1) * right > MAX_RESULT_BITS:
                    raise ValueError("result out of range")
        return _checked(_OPERATORS[type(node.op)](left, right))
    raise ValueError("not an arithmetic expression")
