"""The None / UNDEFINED rule: the private UNDEFINED sentinel and the operator sets that apply it."""

from __future__ import annotations

#: Ordering operators: False when ANY evaluated operand is None (package
#: docstring, "None / missing rule"). ``==``/``!=`` are NOT here.
_NONE_IS_FALSE_OPERATORS: frozenset[str] = frozenset({"<", "<=", ">", ">="})
#: Arithmetic operators: UNDEFINED when ANY evaluated operand is None or
#: UNDEFINED, so the unset value reaches the enclosing operator.
_UNDEFINED_PROPAGATING_OPERATORS: frozenset[str] = frozenset(
    {"+", "-", "*", "/", "%", "min", "max"}
)
#: Every comparison, equality and membership operator: False when ANY
#: evaluated operand is UNDEFINED (an unset value can never satisfy one).
_UNDEFINED_IS_FALSE_OPERATORS: frozenset[str] = frozenset(
    {"==", "!=", "===", "!==", "<", "<=", ">", ">=", "in", "contains"}
)


class _Undefined:
    """Type of ``_UNDEFINED``: the result of arithmetic on a missing operand.

    Module-private. Falsy (so ``!``, ``!!``, ``and``, ``or`` and ``if`` treat
    it as false), never equal to anything but itself, and turned into
    ``None`` by ``evaluate_logic`` before it leaves the module.
    """

    __slots__ = ()

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<undefined>"


_UNDEFINED = _Undefined()
