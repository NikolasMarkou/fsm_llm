"""The remaining eager operators: ``!``, ``in`` / ``contains`` and ``cat``."""

from __future__ import annotations

from typing import Any

from fsm_llm.logging import logger


def _type_names(values: Any) -> str:
    """The distinct type names of ``values``, in first-seen order, for a log line.

    Interface contract: ``values`` any iterable; returns text such as
    ``"str, int"`` and never a value. The evaluator logs an operand's TYPE and
    never the operand: a context holds extracted user text and credentials.
    """
    return ", ".join(dict.fromkeys(type(value).__name__ for value in values))


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
    return "".join(str(arg) for arg in args)


def _membership(needle: Any, haystack: Any) -> bool:
    """``in``: is *needle* in *haystack*.

    On a ``str`` haystack an ``int`` needle (never a ``bool``) is matched by
    its decimal text, as JavaScript does (``{"in": [5, "a5b"]}`` is True);
    any other non-``str`` needle is False there. Elsewhere Python membership,
    False for a haystack without ``__contains__`` or an unhashable probe.
    Never raises (audit B8).
    """
    if isinstance(haystack, str):
        if isinstance(needle, int) and not isinstance(needle, bool):
            needle = str(needle)
        return isinstance(needle, str) and needle in haystack
    if not hasattr(haystack, "__contains__"):
        return False
    try:
        return needle in haystack
    except TypeError:
        return False


def _logical_not(*args: Any) -> bool:
    """Logical NOT — negates the FIRST argument only.

    JsonLogic's ``!`` is unary. Extra arguments are discarded, which is a
    silent footgun for FSM authors who write ``{"!": [a, b]}`` expecting
    "neither a nor b"; this logs a WARNING so the mistake is visible.

    Args:
        *args: Evaluated operands. Only ``args[0]`` participates.

    Returns:
        bool: ``not args[0]``, or ``True`` when no argument is supplied.

    Failure mode: never raises — a malformed arity warns and still returns
    a bool, keeping ``!`` total like every other operator in the registry.
    """
    if not args:
        return True
    if len(args) > 1:
        # DECISION plan-2026-07-19T191147-4b664252/D-023 [STALE]
        # Do NOT "fix" this by raising, or by folding the extra args in
        # (e.g. `not any(args)`). Either would change the RESULT of every
        # already-deployed FSM carrying this mistake, mid-conversation, on
        # a LOW-severity observability finding. The sibling `!!` raises only
        # as an accident of `bool()`'s arity, not as a designed contract —
        # it is not a precedent to copy. See decisions.md D-023.
        logger.warning(
            f"JsonLogic '!' takes a single operand but received {len(args)}; "
            f"negating the first and discarding {len(args) - 1} argument(s) of type "
            f"{_type_names(args[1:])}. Use 'and'/'or' to combine multiple conditions."
        )
    return not args[0]
