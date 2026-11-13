"""Coercion of model-supplied values: confidence, numbers and bool words.

Layer ``core`` (plan d454fd62, step 72): split out of the old ``fsm_llm.utilities``.
Standard library only.
"""

from __future__ import annotations

import math
from typing import Any

# --------------------------------------------------------------
# Confidence coercion
# --------------------------------------------------------------


def coerce_confidence(raw: Any, default: float) -> float:
    """Coerce a model-supplied confidence to a clamped ``[0, 1]`` float.

    Contract:
        - Parameters: ``raw`` is the model-supplied value (already parsed by
          ``json.loads``); ``default`` is the fallback for non-finite input.
        - ``float(raw)`` still raises ``TypeError``/``ValueError`` on ``{...}``/
          ``null`` so the caller's parse-fallback ladder catches those exactly
          as before — this helper deliberately does NOT swallow them.
        - Only ``NaN``/``±inf`` (which ``float()`` accepts and ``min``/``max``
          leave un-clamped) and an integer past float range (``10**400``,
          whose ``float()`` raises ``OverflowError``) are mapped to
          ``default``; the result is then clamped to ``[0, 1]``.
        - Shared by llm.py (2 rungs) and classification.py (2 parsers) so the
          four confidence-parse sites can never re-diverge.

    # DECISION plan-2026-07-21T082818-4c63deac/D-001
    # Do NOT remove the NaN/inf guard or fold this back into a bare
    # `min(max(float(raw), 0.0), 1.0)` at each call site. `json.loads` accepts
    # bare `NaN`/`Infinity`, and `min`/`max` leave NaN un-clamped: in llm.py that
    # escaped as a pydantic ValidationError and failed the whole turn (G5); in
    # classification.py `max(0.0, min(1.0, nan))` silently became 1.0 (max
    # certainty), defeating `is_low_confidence` (G6). This guard is the fix.
    # See decisions.md D-001.
    """
    try:
        value = float(raw)  # may raise TypeError/ValueError — intentional
    except OverflowError:  # an integer past float range is as unreadable as inf
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return min(max(value, 0.0), 1.0)


# --------------------------------------------------------------
# Number coercion
# --------------------------------------------------------------


def coerce_int(v: Any) -> int:
    """``v`` as an ``int``; ``ValueError`` for a bool, a non-integral or non-finite float.

    Interface contract (callers: the field validation of ``pipeline.py`` and
    the unstructured rung of ``llm.py``'s field extraction, so both read a
    number the same way):
        - A ``bool`` is not a count, ``3.7`` is not 3, and ``inf`` / ``nan``
          have no integer value (``int(inf)`` and ``int(Decimal("Infinity"))``
          raise ``OverflowError``, which the callers' ``except`` does not
          name, so it is converted). ``"12"`` and ``3.0`` convert.
        - Failure mode: ``ValueError`` (``int("x")`` raises it too);
          ``TypeError`` for a value ``int()`` cannot take (``None``, a list).
    """
    if isinstance(v, bool):
        raise ValueError(f"expected an integer, got {v!r}")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and not (math.isfinite(v) and v == int(v)):
        raise ValueError(f"expected an integer, got {v!r}")
    try:
        return int(v)
    except OverflowError:  # int(Decimal("Infinity")): no integer value either
        raise ValueError(f"expected an integer, got {v!r}") from None


def coerce_float(v: Any) -> float:
    """``v`` as a finite ``float``; ``ValueError`` for a bool, ``inf`` or ``nan``.

    Same callers and failure mode as ``coerce_int``; the strings ``"nan"``,
    ``"inf"`` and ``"1e999"`` are refused like the floats they spell, and an
    integer too large for a float (``10**400``, whose ``float()`` raises
    ``OverflowError``) is a ``ValueError`` too.
    """
    if isinstance(v, bool):
        raise ValueError(f"expected a number, got {v!r}")
    try:
        result = v if isinstance(v, float) else float(v)
    except OverflowError:  # float(10**400): too large for a float
        raise ValueError(f"expected a finite number, got {str(v)[:40]!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"expected a finite number, got {v!r}")
    return result


# --------------------------------------------------------------
# Bool-word parsing
# --------------------------------------------------------------

_BOOL_TRUE_WORDS = frozenset({"true", "yes", "y", "1"})
_BOOL_FALSE_WORDS = frozenset({"false", "no", "n", "0"})


def parse_bool_word(text: str) -> bool | None:
    """Read a model-supplied string as a bool, or ``None`` when it is not one.

    Contract:
        - Parameters: ``text`` is a ``str`` (any other type is the caller's
          branch, not this helper's).
        - Returns ``True`` for ``true/yes/y/1`` and ``False`` for
          ``false/no/n/0`` after stripping whitespace, surrounding quotes and
          trailing ``.!?,;:`` and case-folding; ``None`` for anything else
          (including ``""``). Never raises.
        - Shared by ``pipeline._coerce_bool`` (which raises ``ValueError`` on
          ``None``) and the ``llm.py`` unstructured bool rung (which leaves the
          field unset on ``None``), so the two readers cannot re-diverge.

    # DECISION plan-2026-11-05T134105-5ed71cf2/D-008
    # The vocabulary is CLOSED. Do NOT add "yeah", "sure", "nope",
    # "affirmative" or a "anything else is False" default: a confident wrong
    # ``False`` from an unreadable answer gated transitions (F-LC-01). An
    # unknown word must stay invalid so the ladder re-asks. See decisions.md D-008.
    """
    word = text.strip()
    previous = None
    while word != previous:
        previous = word
        word = word.rstrip(".!?,;:").strip().strip("'\"`").strip()
    word = word.casefold()
    if word in _BOOL_TRUE_WORDS:
        return True
    if word in _BOOL_FALSE_WORDS:
        return False
    return None
