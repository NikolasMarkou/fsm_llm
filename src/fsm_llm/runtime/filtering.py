"""The write-time payload filter of the event log: ``filter_context``.

Layer ``runtime`` (plan 3f4530d3, step 1b): split out of ``runtime/log.py`` to keep that module
under the size guard; the bodies moved unchanged. ``filter_context`` is the one way a structured
payload (a context, a state, a delta) becomes log-safe: ``core.redaction.redact_context``, then
internal-prefixed keys dropped, then ``to_jsonable``. Imports ``core``, the kernel and
``runtime.events`` only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from fsm_llm.constants import has_internal_prefix, is_forbidden_context_entry
from fsm_llm.core.redaction import redact_context
from fsm_llm.runtime.events import to_jsonable


def _redact_arguments(text: str) -> str:
    """Redact the secret-named entries of tool-call arguments that travel as JSON TEXT.

    A transcript stores ``tool_calls[].function.arguments`` as a JSON string, which a walk by
    key name never reads. A string that parses to a JSON object goes through ``redact_context``
    and is re-dumped (compact, ASCII) only when something was redacted, so every other
    arguments string is returned byte for byte; text that is not a JSON object is returned as
    it is.
    """
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError, MemoryError):
        return text
    if not isinstance(parsed, dict):
        return text
    safe = redact_context(parsed)
    if safe == parsed:
        return text
    return json.dumps(safe, separators=(",", ":"), ensure_ascii=True)


def _scrub_in_place(node: Any) -> None:
    """Drop every internal-prefixed string key and redact the JSON text of ``arguments``
    entries, in the dicts under *node*, in place.

    Iterative with a visited set, so a shared subtree is walked once and a deep value cannot
    raise ``RecursionError``. Only ever called on a fresh copy made by ``redact_context``.
    """
    stack = [node]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if id(current) in seen:
                continue
            seen.add(id(current))
            for key in [
                k for k in current if isinstance(k, str) and has_internal_prefix(k)
            ]:
                del current[key]
            arguments = current.get("arguments")
            if isinstance(arguments, str):
                current["arguments"] = _redact_arguments(arguments)
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)


def filter_context(
    mapping: Mapping[str, Any], *, keep_internal: Iterable[str] = ()
) -> dict[str, Any]:
    """Return the log-safe copy of a structured payload (a context, a state, a delta).

    Interface contract (callers: the engine's state blob and ``context/record`` producers):
        - Secret-looking values are replaced through ``redact_context`` (names and values
          judged by ``is_forbidden_context_entry``, depth cut at ``MAX_CONTEXT_FILTER_DEPTH``,
          cycles cut); then every internal-prefixed key (``has_internal_prefix``) is dropped at
          every depth, except the TOP-LEVEL names listed in *keep_internal* (the completion
          transcript keys a definition declares; they are redacted like any value); an
          ``arguments`` string that is a JSON object (a transcript's tool call) is redacted by
          its entry names too; then ``to_jsonable`` turns what JSON cannot hold into typed
          markers.
        - The result is a new dict that shares nothing mutable with *mapping*, always passes
          ``encode_event`` as an event field, and is at most ``MAX_PAYLOAD_DEPTH - 1`` deep.
        - Never raises for content (a 100,000 deep value is cut, not recursed into);
          ``TypeError`` for a non-mapping.
    """
    if not isinstance(mapping, Mapping):
        raise TypeError(f"filter_context takes a mapping, got {type(mapping).__name__}")
    # DECISION plan-2026-11-17T100529-21e6f00d/D-009
    # Internal keys leave a payload at EVERY depth, and only the TOP-LEVEL names a caller
    # allow-lists (the definition's transcript keys, see engine/events.py) stay. Do NOT keep
    # internal keys by prefix or deny-list here: the grant `_approval_granted` lives in that
    # namespace. Secret-looking values are replaced first (`redact_context`) and never
    # restored. See decisions.md D-009.
    keep = frozenset(keep_internal)
    redacted = redact_context(dict(mapping))
    top = {
        key: value
        for key, value in redacted.items()
        if key in keep or not (isinstance(key, str) and has_internal_prefix(key))
    }
    _scrub_in_place(list(top.values()))
    safe: dict[str, Any] = to_jsonable(top)
    return safe


def visible_entries(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Return the entries of *mapping* an event may name or carry (plan 3f4530d3 D-008).

    Drops an entry whose key is not a string, is internal-prefixed or is secret-looking
    (``is_forbidden_context_entry``), so one rule decides the ``keys`` and the delta blob of the
    extraction, miss and handler events. Names only: pass ``dict.fromkeys(names)``. A new dict.
    """
    return {
        key: value
        for key, value in mapping.items()
        if isinstance(key, str)
        and not has_internal_prefix(key)
        and not is_forbidden_context_entry(key, value)
    }
