"""Constructor-limit checks and the bounded ``repr`` the tool-layer messages use."""

from __future__ import annotations

import numbers
import operator
from typing import Any

from fsm_llm.core.definitions import AgentError

from .constants import Defaults

# --- constructor limits -------------------------------------------------------
# DECISION plan-2026-11-06T093029-0068b055/D-008 (extended in step 12.8, D-020)
# One rule for every ``max_*`` / ``num_*`` constructor limit of an agent:
# ``require_int_limit``. Do NOT clamp a bad value silently (``max(1, n)``), do
# NOT re-inline an ad-hoc ``if n < 1`` check in a pattern (it let True, 1.5, NaN,
# inf and "3" through, and a bare TypeError for str and None), and do NOT
# f-string the value into a message (an int past 4300 digits makes the f-string
# itself raise ValueError): show it through ``bounded_repr``.


def bounded_repr(value: Any, limit: int = 40) -> str:
    """A short, safe ``repr`` of *value* for an error message.

    Interface contract: returns a ``str`` of at most ``limit`` characters plus a
    trailing ``...`` when cut. Never raises: an int past 64 bits is shown as
    ``<int, N bits>`` (its full decimal text may exceed Python's 4300-digit
    conversion limit), and an object whose ``repr`` raises is shown by type name.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        bits = value.bit_length()
        if bits > 64:
            return f"<int, {bits} bits>"
    try:
        text = repr(value)
    except Exception:
        return f"<{type(value).__name__}>"
    return text if len(text) <= limit else text[:limit] + "..."


def require_int_limit(
    name: str,
    value: Any,
    *,
    minimum: int,
    error: type[Exception] = AgentError,
) -> int:
    """Return *value* as an ``int`` when it is an integer limit ``>= minimum``.

    Interface contract (callers: the constructor of every agent pattern with a
    ``max_*``/``num_*`` limit, ``RemoteAgentServer``, the tool registries and
    ``SemanticMemoryStore``):
        - *value* must be an ``int`` or another ``numbers.Integral`` (a numpy
          integer); it is returned as a plain ``int``. A ``bool``, a ``float``
          (also ``1.0``, NaN, inf), a ``str``, ``None`` and every other type
          raise *error*; so does an integer below *minimum*.
        - *error* is ``AgentError`` by default; a component that always raised
          ``ValueError`` passes ``error=ValueError`` so its exception type is
          kept.
        - The message names *name*, the expectation and ``bounded_repr(value)``;
          it never holds more than a short prefix of the value, so a 5000-digit
          int cannot make the error path raise.
        - Upper bound ``Defaults.MAX_INT_LIMIT`` (2**63 - 1): above it the int
          is no real limit and would break the f-strings, log lines and JSON
          that carry it (``num_rounds=10**5000`` raised a bare ``ValueError``
          from a log line at construction). Refused, never clamped.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or not minimum <= operator.index(value) <= Defaults.MAX_INT_LIMIT
    ):
        raise error(
            f"{name} must be an int from {minimum} to {Defaults.MAX_INT_LIMIT}, "
            f"got {type(value).__name__} {bounded_repr(value)}"
        )
    return operator.index(value)
