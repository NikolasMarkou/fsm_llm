"""Shared deprecation-warning machinery.

This module provides :func:`warn_deprecated` — a single canonical formatter
for ``DeprecationWarning`` emission inside fsm_llm — and
:func:`reset_deprecation_dedupe` for tests that need to assert per-call
emission.

Design notes
------------

*   We translate (do not lift) the deepagents reference. fsm_llm has no
    LangChain coupling; the implementation uses only stdlib :mod:`warnings`
    plus a process-local dedupe registry.

*   Emission is deduplicated per ``(name, since, removal)`` triple so that
    long-running sessions do not flood logs. Tests that need to observe
    multiple emissions of the same target call
    :func:`reset_deprecation_dedupe` between calls.

*   ``stacklevel`` defaults to ``2`` so the warning is attributed to the
    caller of the deprecated function (one frame above this helper). Pass
    ``stacklevel=3`` from a thin wrapper, etc.

*   We DO NOT add live warnings to I5-epoch surfaces in 0.5.x. The machinery
    ships now; the warnings flip in 0.6.0 per ``docs/lambda_fsm_merge.md``
    §3. Importing this module has no side effects.
"""

from __future__ import annotations

import threading
import warnings
from typing import Final

__all__ = [
    "warn_deprecated",
    "reset_deprecation_dedupe",
]

# Process-local dedupe registry. Keys are (name, since, removal) triples so
# that bumping the removal version forces a re-emit (which is the only thing
# users would actually want to be re-told about).
_DEDUPE_REGISTRY: set[tuple[str, str, str]] = set()
_REGISTRY_LOCK: Final = threading.Lock()


def _format_message(
    name: str,
    *,
    since: str,
    removal: str,
    replacement: str | None,
) -> str:
    """Render the canonical deprecation message text.

    Format::

        <name> is deprecated since fsm_llm <since>; it will be removed in
        <removal>. Use <replacement> instead.

    The trailing ``Use ... instead.`` clause is omitted when
    ``replacement is None``.
    """
    base = (
        f"{name} is deprecated since fsm_llm {since}; "
        f"it will be removed in {removal}."
    )
    if replacement:
        return f"{base} Use {replacement} instead."
    return base


def warn_deprecated(
    name: str,
    *,
    since: str,
    removal: str,
    replacement: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit a deduplicated :class:`DeprecationWarning` for ``name``.

    Parameters
    ----------
    name:
        Fully-qualified import path of the deprecated surface
        (e.g. ``"fsm_llm.api.API"``).
    since:
        Version string at which the deprecation was introduced
        (e.g. ``"0.6.0"``).
    removal:
        Version string at which the surface will be removed
        (e.g. ``"0.7.0"``).
    replacement:
        Optional fully-qualified replacement name. When supplied, appended to
        the message as ``Use <replacement> instead.``.
    stacklevel:
        Forwarded to :func:`warnings.warn`. Defaults to ``2`` so the warning
        is attributed to the caller of the deprecated function. Coerced to
        ``1`` when caller passes ``< 1``.

    Notes
    -----
    Emission is deduplicated per ``(name, since, removal)``. Use
    :func:`reset_deprecation_dedupe` to clear state in tests.
    """
    safe_stacklevel = stacklevel if stacklevel >= 1 else 1
    key = (name, since, removal)
    with _REGISTRY_LOCK:
        if key in _DEDUPE_REGISTRY:
            return
        _DEDUPE_REGISTRY.add(key)
    message = _format_message(
        name,
        since=since,
        removal=removal,
        replacement=replacement,
    )
    warnings.warn(message, DeprecationWarning, stacklevel=safe_stacklevel + 1)


def reset_deprecation_dedupe(*targets: str) -> None:
    """Clear the per-target dedupe registry.

    With no arguments, removes ALL recorded targets. Otherwise removes only
    entries whose ``name`` appears in ``targets``.

    Intended for tests asserting that :func:`warn_deprecated` emits on each
    call — long-running production code should not need this.
    """
    with _REGISTRY_LOCK:
        if not targets:
            _DEDUPE_REGISTRY.clear()
            return
        # Remove all keys whose `name` matches any requested target.
        wanted = set(targets)
        _DEDUPE_REGISTRY.difference_update(
            {key for key in _DEDUPE_REGISTRY if key[0] in wanted}
        )
