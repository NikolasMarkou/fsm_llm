"""Equality and ordering operators: ``==``, ``===``, ``<``, ``<=``, ``>``, ``>=``."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any


def soft_equals(a: Any, b: Any) -> bool:
    """
    Implement the '==' operator with loose, JavaScript-flavoured coercion.

    The coercion is *inspired by* JavaScript equality but deliberately
    diverges from it in two places (the ``None`` guards and the bool-vs-string
    rule); see Note for the rules that actually apply.

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
        >>> soft_equals(True, "true")
        True
        >>> soft_equals(True, "1")   # JavaScript says True; this does not
        False
        >>> soft_equals("hello", "hello")
        True

    Note:
        The rules are checked in this order — the FIRST match wins:

        1. Either value is a ``bool``:

           - and the other is ``None`` -> identity, so ``soft_equals(True,
             None)`` is False.
           - and the other is a ``str`` -> both are rendered with ``str()``
             and compared LOWERCASED. This is a *string* comparison, not a
             bool coercion: ``soft_equals(True, "true")`` is True but
             ``soft_equals(True, "1")`` is False. The rule exists to accept
             the lowercase JSON booleans LLMs emit, not to mimic JS.
           - otherwise -> both are coerced with ``bool()``, so
             ``soft_equals(True, 1)`` is True.

        2. Both values are ``str`` -> case-INSENSITIVE comparison (LLM
           casing varies; use ``===`` when case matters).

        3. Exactly one value is a ``str`` -> False if the other is ``None``
           (see the D-017 guard below), otherwise a case-SENSITIVE
           ``str(a) == str(b)``, so ``soft_equals("5", 5)`` is True.

        4. Otherwise -> standard Python equality.

        Rules 3-4 also accept numerically equal operands (``_numeric_equal``:
        ``1.0 == "1"``), so ``==`` agrees with ``<=``/``>=`` on mixed
        number/string operands only. ``<=``/``>=`` go further (two strings
        and a bool against a numeric string, see ``_numeric_equal``), so they
        accept some pairs that ``==`` refuses. Rule 1 (any ``bool``) is never
        numerically coerced, and rule 2 (two strings) never is either:
        ``"01" == "1"`` is False, as in JavaScript, because identifiers such
        as zip codes and account numbers are strings whose leading zeros
        matter.
    """
    # Boolean comparison — guard None, handle bool+string ("true"==True)
    if isinstance(a, bool) or isinstance(b, bool):
        if a is None or b is None:
            return a is b
        # bool + string: compare via lowercase string (JSON booleans are lowercase)
        if isinstance(a, str) or isinstance(b, str):
            return str(a).lower() == str(b).lower()
        return bool(a) == bool(b)

    # NOTE: == is intentionally case-insensitive for string comparisons to tolerate
    # LLM output casing variation. Use === for case-sensitive string equality.
    # Example: {"==": [{"var": "intent"}, "Purchase"]} matches "purchase" and "PURCHASE".
    # String comparison — case-insensitive when BOTH are strings
    # (LLMs often return "Yes"/"yes", "Buy"/"buy" inconsistently)
    # DECISION plan-2026-09-21T203800-8a03483a/D-013 (supersedes
    # plan-2026-07-19T191147-4b664252/D-023 "soft_equals is deliberately
    # untouched"): rules 3-4 below accept numerically equal operands so `==`
    # agrees with `<=`/`>=` on a number against a numeric string
    # (`1.0 == "1"`). Do NOT move this coercion above the
    # bool branch (`true == "1"` must stay False), do NOT apply it to two
    # strings (`"01" == "1"` must stay False: zip codes, ids), and do NOT let
    # it reach a None operand (`_numeric_equal` is False for None, D-017).
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    if isinstance(a, str) or isinstance(b, str):
        # DECISION plan-2026-07-18T051819-80b0bd4d/D-017 [STALE]
        # Do NOT delete this guard and do NOT "simplify" it back to a bare
        # `str(a) == str(b)`. `get_var` resolves a MISSING context key to None
        # (default `not_found=None`), so without this guard `str(None) == "None"`
        # made an unset variable compare EQUAL to the literal string "None" — every
        # JsonLogic `==` condition in the framework could fire on a variable that was
        # never extracted, and an FSM author could not distinguish "unset" from "the
        # LLM literally said None". Returning False here (rather than raising, or
        # returning a tri-state) keeps `==` total, and `!=` inherits the fix for free
        # since it is defined as `not soft_equals(...)`.
        # This guard is deliberately scoped to the MIXED str/non-str branch only:
        # `soft_equals(None, None)` must stay True (it falls through to the standard
        # equality below), and the bool branch above already guards None itself.
        # Every other coercion here is intentional JsonLogic leniency and unchanged:
        # `soft_equals("5", 5)` is still True. See decisions.md D-017.
        if a is None or b is None:
            return False
        return str(a) == str(b) or _numeric_equal(a, b)

    # Standard equality
    return bool(a == b) or _numeric_equal(a, b)


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

    Example:
        >>> less(1, 2)
        True
        >>> less(1, 2, 3)  # Equivalent to 1 < 2 < 3
        True
        >>> less("10", "2")  # Numeric coercion: float("10") < float("2")
        False

    Note:
        - Numeric conversion is attempted first: less("10", "2") compares
          as float(10) < float(2) = False. A string side must be a plain number
          (``_PLAIN_NUMBER_RE`` after ``strip()``: not "inf", "1_000" or
          non-ASCII digits); otherwise native comparison is used:
          less("abc", "def") = True.
        - Chained comparisons evaluate left-to-right
        - Never raises: operands that cannot be ordered at all (a ``TypeError``
          from the native comparison) give False, and an ``int`` past float
          range is compared as the exact ``int`` (``_as_float``)
    """
    # DECISION plan-2026-11-28T050702-8525c797/D-007: the chain is a LOOP with
    # an early exit, never a recursion: one frame per argument turned a TRUE
    # chain of 1000+ operands into a RecursionError that the evaluator logged
    # and read as False. Do NOT write it recursively again, do NOT copy the
    # tail (`args[1:]`, `(b, *args)`) per pair, and do NOT drop the early
    # return (a late violation must not cost a full pass of coercions). The
    # coerced right side is carried into the next pair, as the old recursion
    # did (`less(b, *args)` with the rebound `b`).
    previous, current = a, b
    for position in range(len(args) + 1):
        previous, current = _coerce_for_order(previous, current)
        try:
            ordered = previous < current
        except TypeError:
            return False
        if not ordered:
            return False
        if position < len(args):
            previous, current = current, args[position]
    return True


def _coerce_for_order(a: Any, b: Any) -> tuple[Any, Any]:
    """The numeric coercion of one ordered pair, shared by every chain operator.

    Interface contract: ``a``/``b`` any values; returns the pair as floats (or
    the exact ``int`` past float range, ``_as_float``) when each string side is
    a plain number after ``strip()``, else the pair unchanged. Never raises.
    """
    # Numeric conversion is attempted unless a string side is not a plain
    # number: bare `float()` accepts "inf", "1_000" and non-ASCII digits that
    # `==` refuses (D-048). Padding is trimmed, so " 25" still orders.
    if all(
        not isinstance(side, str) or _PLAIN_NUMBER_RE.fullmatch(side.strip())
        for side in (a, b)
    ):
        try:
            return _as_float(a), _as_float(b)
        except (TypeError, ValueError):
            pass  # Fall back to native comparison
    return a, b


def _holds_pairwise(
    pair: Callable[[Any, Any], bool], a: Any, b: Any, args: tuple[Any, ...]
) -> bool:
    """True when ``pair`` holds for every adjacent pair of ``a, b, *args``.

    Interface contract: ``pair(x, y) -> bool`` is total (never raises). The
    walk stops at the first pair that fails and copies nothing, so a chain of
    any length costs one pass and no recursion depth.
    """
    # DECISION plan-2026-11-28T050702-8525c797/D-007: loop with an early exit,
    # same rule as `less` (see the anchor there). Do NOT turn it back into a
    # recursive call per pair.
    if not pair(a, b):
        return False
    previous = b
    for current in args:
        if not pair(previous, current):
            return False
        previous = current
    return True


def _numeric_equal(a: Any, b: Any) -> bool:
    """True when both sides are ``float()``-able and equal (``1`` and ``"1.0"``).

    Interface contract: ``a``/``b`` any values; never raises (an uncoercible
    side, including ``None``, is simply not numerically equal). Shared by
    ``soft_equals`` (rules 3-4 only: never two strings, never a ``bool``) and
    ``<=``/``>=`` (any pair). The three agree on a number against a numeric
    string (D-013 of plan-2026-09-21T203800-8a03483a, superseding the earlier
    D-023 that kept ``soft_equals`` out of it) but NOT everywhere: ``<=``/``>=``
    also accept two numeric strings spelled differently (``"01" <= "1"``,
    ``"1" >= "1.0"``) and a ``bool`` against a numeric string (``True <= "1"``),
    all of which ``==`` rejects. ``<`` and ``>`` never use this helper, but gate
    their own string operands the same way (padding trimmed). A
    ``str`` side must be a plain decimal/scientific literal
    (``_PLAIN_NUMBER_RE``); ``"1_000"``, ``" 1 "``, ``"inf"`` are not numbers.
    """
    # DECISION plan-2026-09-21T203800-8a03483a/D-048: a string operand must
    # be a PLAIN decimal/scientific literal. Do NOT go back to bare `float()`
    # on strings: it accepts "1_000", " 1 ", "inf" and "nan", so `"1_000" ==
    # 1000` became True. See decisions.md D-048.
    for side in (a, b):
        if isinstance(side, str) and not _PLAIN_NUMBER_RE.fullmatch(side):
            return False
    try:
        return bool(_as_float(a) == _as_float(b))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> Any:
    """``float(value)``, or the exact ``int`` when it is past float range.

    Interface contract: ``value`` any object; returns a ``float``, or the same
    ``int`` when ``float()`` raises ``OverflowError`` (a 400-digit int an LLM
    extraction can carry). Python orders and compares an ``int`` against a
    ``float`` exactly, so the callers (``less`` and ``_numeric_equal``) stay
    correct. Any other ``float()`` failure (``TypeError``, ``ValueError``) is
    the caller's: both already catch it.
    """
    try:
        return float(value)
    except OverflowError:
        return value


# JSON-style number text only: optional sign, digits with an optional
# fraction (or a leading-dot fraction), optional exponent. No `_`, no
# surrounding whitespace, no inf/nan, no hex. Used with `fullmatch`.
# `re.ASCII`: `\d` must not match Unicode digits (Arabic-Indic "1000" == 1000
# was True).
_PLAIN_NUMBER_RE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII
)


def _number(value: Any) -> float:
    """The one operand coercion of the arithmetic operators.

    Interface contract: ``value`` any object; returns a ``float``. A ``str``
    must be a plain number after ``strip()`` (``_PLAIN_NUMBER_RE``, the rule
    ``<`` and ``>`` apply to a string side: not "inf", "nan", "1_000" or
    non-ASCII digits); anything else raises ``ValueError`` whose text names no
    value. Every other type goes through ``float()`` as before (``int``,
    ``float``, ``bool``), so ``None`` and containers raise ``TypeError``. The
    caller is ``evaluate_logic``, which reads either error as UNDEFINED.
    """
    # DECISION plan-2026-11-28T050702-8525c797/D-008: ONE numeric-string rule
    # for every JsonLogic operand. Do NOT go back to bare `float()` in `+ - * /
    # % min max` ("inf" and "nan" became numbers, `{">": [{"+": ["inf", 1]},
    # 5]}` was True), do NOT put the offending value in the error text (the
    # evaluator logs the exception class only, and a message would leak it),
    # and do NOT reuse `core/coerce.py` here: that layer coerces an EXTRACTED
    # field value and accepts "1_000" on purpose.
    if isinstance(value, str):
        text = value.strip()
        if not _PLAIN_NUMBER_RE.fullmatch(text):
            raise ValueError("operand is not a plain number")
        return float(text)
    return float(value)


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
        Uses less(), soft_equals() and numeric equality (``1 <= "1.0"`` is True).
        Supports the same type coercion as the less() function.
    """
    return _holds_pairwise(_less_or_equal_pair, a, b, args)


def _less_or_equal_pair(a: Any, b: Any) -> bool:
    """One ``a <= b`` pair: ``<``, ``==`` or numeric equality."""
    # DECISION plan-2026-11-06T093029-0068b055/D-005: the third term is kept on
    # purpose and is why `"01" <= "1"` is True while `"01" == "1"` is False.
    # Do NOT drop it as "redundant with soft_equals": a numeric-string gate such
    # as `>= "50"` on an extracted "50.0" would flip to False, silently. Do NOT
    # add it to `==` either (ids rule, D-013 of plan-2026-09-21T203800-8a03483a).
    return less(a, b) or soft_equals(a, b) or _numeric_equal(a, b)


def _greater_pair(a: Any, b: Any) -> bool:
    """One ``a > b`` pair."""
    return less(b, a)


def _greater_or_equal_pair(a: Any, b: Any) -> bool:
    """One ``a >= b`` pair: ``>``, ``==`` or numeric equality."""
    # DECISION plan-2026-11-06T093029-0068b055/D-005: same kept third term as
    # `_less_or_equal_pair`; see the anchor there before touching it.
    return less(b, a) or soft_equals(a, b) or _numeric_equal(a, b)


def greater(a: Any, b: Any, *args: Any) -> bool:
    """Implement '>' with chaining support (a > b > c)."""
    return _holds_pairwise(_greater_pair, a, b, args)


def greater_or_equal(a: Any, b: Any, *args: Any) -> bool:
    """Implement '>=' with chaining support (a >= b >= c)."""
    return _holds_pairwise(_greater_or_equal_pair, a, b, args)
