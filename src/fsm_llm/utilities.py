"""
This module provides utility functions for FSM definition loading,
JSON processing, and other common operations in the enhanced
FSM-LLM framework.

Key Features:
- Enhanced FSM definition loading with validation
- Improved JSON extraction for LLM responses
- Error handling and logging integration
- Support for new FSM definition format
"""

from __future__ import annotations

import datetime
import decimal
import json
import math
import os
import re
import uuid
from collections.abc import Callable
from typing import Any

from .constants import CONTEXT_FILTER_CYCLIC_WORK_FACTOR, MAX_CONTEXT_FILTER_NODES
from .definitions import FSMDefinition

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------
from .logging import logger

# --------------------------------------------------------------
# Reasoning-trace field resolution
# --------------------------------------------------------------


def _resolve_reasoning_trace(message: Any) -> str | None:
    """Return the model's reasoning/thinking trace, whatever field carries it.

    Some models (e.g. Qwen 3.5 via Ollama) leave ``content`` empty/``None`` and
    put the actual answer in a reasoning field. litellm's field name for that
    trace is version-dependent, so callers MUST NOT read a single hard-coded
    attribute; resolve it here, in one place, for every content reader.

    Contract:
        - Parameter: any litellm ``Message``/``Delta`` (or duck-typed stand-in).
        - Returns the first non-empty of ``reasoning_content``, legacy
          ``thinking``, then a newline-joined string of any ``thinking_blocks``
          (each block's ``thinking`` or ``text``); ``None`` when no trace exists.
        - Never raises for any input (uses ``getattr(..., None)``, never
          ``hasattr``/attribute access that could fail on a stripped object).
    """
    # DECISION plan-2026-07-21T072826-e3131cc2/D-002
    # Do NOT revert this to `hasattr(message, "thinking")` or a `.thinking`-only
    # read. The installed litellm range RENAMES the raw `thinking` field to
    # `reasoning_content` and DELETES `thinking` before building the
    # Message/Delta object, so a `.thinking`-only read is DEAD CODE for the
    # project's own DEFAULT_LLM_MODEL (ollama_chat/qwen3.5:4b). `reasoning_content`
    # is read FIRST; the legacy `thinking` string is kept so the D-023 divergence
    # tests stay green; `thinking_blocks` is a last-resort join of the provider's
    # typed reasoning segments. This is the SINGLE resolver shared by
    # llm.py::_extract_content_from_thinking and
    # classification.py::_extract_response so the two readers can never
    # re-diverge (the NL1 bug was classification.py holding its own stale copy).
    # Mirrors the original C2 anchor plan-2026-07-21T045419-9925aa3a/D-002.
    # See decisions.md D-002.
    trace = getattr(message, "reasoning_content", None) or getattr(
        message, "thinking", None
    )
    if not trace:
        blocks = getattr(message, "thinking_blocks", None)
        if blocks:
            trace = "\n".join(
                (b.get("thinking") or b.get("text") or "")
                for b in blocks
                if isinstance(b, dict)
            )
    if not trace:
        return None
    return trace


# --------------------------------------------------------------
# Reasoning-tag / code-fence stripping
# --------------------------------------------------------------


def _remove_think_blocks(content: str) -> str:
    """Remove every ``<think>...</think>`` span in one linear pass.

    Contract: takes any ``str``; returns it with each lazily-matched,
    non-overlapping ``<think>...</think>`` span (newlines included) removed.
    An opening tag with no closing tag after it is left in place, as are all
    later opening tags (none of them can have a closer either). Never raises.

    # DECISION plan-2026-09-19T175721-21cd7f8e/D-010
    # Do NOT restore ``re.sub(r"<think>.*?</think>", "", ..., re.DOTALL)``:
    # on ``"<think>a" * n`` every opening tag scans to the end of the string
    # looking for a closer that does not exist, so it is O(n^2) (50,000 tags
    # took minutes) on text that arrives straight from the provider while the
    # caller holds the per-conversation lock. Once ONE opener has no closer no
    # later opener can have one, so the scan stops there. See decisions.md D-010.
    """
    open_tag, close_tag = "<think>", "</think>"
    parts: list[str] = []
    pos = 0
    while True:
        start = content.find(open_tag, pos)
        if start == -1:
            break
        end = content.find(close_tag, start + len(open_tag))
        if end == -1:
            break
        parts.append(content[pos:start])
        pos = end + len(close_tag)
    parts.append(content[pos:])
    return "".join(parts)


def strip_think_and_fences(content: str) -> str:
    """Strip a ``<think>...</think>`` block, then any surrounding code fence.

    Some models (e.g. Qwen via Ollama) wrap a JSON reply in a ``<think>``
    trace and/or a markdown code fence; callers that then ``json.loads`` the
    result need both stripped first.

    Contract:
        - Parameter: ``content`` — the raw message content string (already
          confirmed to be a ``str`` by the caller; this function does not
          itself branch on type).
        - Returns the content with any ``<think>...</think>`` span removed
          (``re.DOTALL``, so it matches across newlines) and then any leading
          ` ```json `/``` ``` `` fence markers stripped, in that exact order.
        - Never raises for any ``str`` input.
        - Shared by ``llm.py::_parse_field_extraction_response`` and
          ``LiteLLMInterface.extract_bulk_data`` so the two content readers
          can never re-diverge — the three regex operations were previously
          byte-identical, hand-duplicated in both places.
    """
    content = _remove_think_blocks(content).strip()
    # DECISION plan-2026-09-19T175721-21cd7f8e/D-030
    # No re.MULTILINE: only a fence that STARTS the reply is a wrapper. With
    # `^` matching at every line, an inline code fence in prose
    # ('Use:\n```python\nprint(1)\n```') lost its markers. Do NOT re-add it (LS-09).
    # DECISION plan-2026-09-19T175721-21cd7f8e/D-048
    # The CLOSING fence is stripped only after a leading fence was stripped: a
    # reply that does not start with a fence is not a fenced wrapper, so a
    # trailing fence is content ('Here:\n```py\nx\n```' keeps both). Do NOT
    # strip it unconditionally, and do NOT count fences to decide (a heuristic
    # where the start-of-reply test is structural). Trade-off: a stray closing
    # fence alone ('{"a": 1}\n```') is kept here; extract_json_from_text recovers.
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", content)
    if stripped != content:
        stripped = re.sub(r"\n?```\s*$", "", stripped)
    return stripped.strip()


# --------------------------------------------------------------
# Depth-bounded context-filter tree walker
# --------------------------------------------------------------

# Sentinels for a value the walker itself drops, keyed to the ``on_drop``
# reason reported for it. Module-private -- callers never see these values,
# only the filtered result and the reason string.
_TOO_DEEP = object()
_CYCLE = object()
_DROP_REASONS = ((_TOO_DEEP, "too_deep"), (_CYCLE, "cycle"))


class ContextFilterWorkError(ValueError):
    """``filter_context_tree`` refused a cyclic input whose aliasing would
    unfold past its work ceiling. Raised instead of returning a partial value
    (see ``filter_context_tree``'s contract)."""


def _cycle_reaching_containers(source: Any) -> tuple[set[int], int]:
    """Return ``(reaching, distinct_items)`` for the dict/list/tuple graph
    under *source*.

    Contract:
        - ``reaching``: the ``id()`` of every container that lies on a cycle
          or can reach one. Every other container's subtree is acyclic, so
          nothing under it can be on the walker's active path wherever it is
          met (it would then reach itself).
        - ``distinct_items``: the summed length of every DISTINCT container.
        - Iterative Tarjan SCC (no recursion limit), linear in the distinct
          graph; never calls anything on a leaf. Never raises for a
          dict/list/tuple graph.
    """
    containers = (dict, list, tuple)
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    on_stack: set[int] = set()
    scc: list[int] = []
    # `hot`: the container has an edge into its own SCC-in-progress (a cycle)
    # or into a finished container that reaches one.
    hot: set[int] = set()
    reaching: set[int] = set()
    distinct_items = 0
    stack: list[tuple[int, Any]] = []

    def _enter(node: Any) -> None:
        nonlocal distinct_items
        node_id = id(node)
        index[node_id] = low[node_id] = len(index)
        on_stack.add(node_id)
        scc.append(node_id)
        distinct_items += len(node)
        children = node.values() if isinstance(node, dict) else node
        stack.append((node_id, iter(children)))

    _enter(source)
    while stack:
        node_id, children = stack[-1]
        descended = False
        for child in children:
            if not isinstance(child, containers):
                continue
            child_id = id(child)
            if child_id not in index:
                _enter(child)
                descended = True
                break
            if child_id in on_stack:
                low[node_id] = min(low[node_id], index[child_id])
                hot.add(node_id)
            elif child_id in reaching:
                hot.add(node_id)
        if descended:
            continue
        stack.pop()
        if low[node_id] == index[node_id]:
            members = []
            while True:
                member = scc.pop()
                on_stack.discard(member)
                members.append(member)
                if member == node_id:
                    break
            if len(members) > 1 or any(member in hot for member in members):
                reaching.update(members)
        if stack:
            parent_id = stack[-1][0]
            low[parent_id] = min(low[parent_id], low[node_id])
            if node_id in on_stack or node_id in reaching:
                hot.add(parent_id)
    return reaching, distinct_items


def _drop_reason(value: Any) -> str | None:
    """The ``on_drop`` reason for a walker sentinel (identity, never ``==``)."""
    for sentinel, reason in _DROP_REASONS:
        if value is sentinel:
            return reason
    return None


# The leaf types a prompt can render faithfully (``json.dumps`` without
# ``default=``). Containers (dict/list/tuple) are walked, never leaves.
_JSON_NATIVE_LEAF_TYPES = (str, int, float, bool)

# Stdlib value scalars whose ``str()`` IS the value (an ISO date, a number, a
# UUID) and carries no attribute fields. EXACT types only: a subclass can
# override ``__str__``. See decisions.md D-032.
_VALUE_SCALAR_TYPES = frozenset(
    {
        datetime.date,
        datetime.datetime,
        datetime.time,
        datetime.timedelta,
        decimal.Decimal,
        uuid.UUID,
    }
)


def redact_non_json_leaf(value: Any) -> Any:
    """Return *value* if it is a JSON-native leaf, else a type-name placeholder.

    Contract:
        - ``None`` and instances of ``str``/``int``/``float``/``bool`` (including
          subclasses such as ``StrEnum``/``IntEnum``) are returned unchanged,
          as are values whose EXACT type is ``datetime.date``/``datetime``/
          ``time``/``timedelta``, ``decimal.Decimal`` or ``uuid.UUID``.
        - Anything else (pydantic model, dataclass, ``MappingProxyType``, set,
          bytes, arbitrary object, a subclass of a value scalar) becomes
          ``"<redacted:TypeName>"``.
        - Never raises; never calls ``str()``/``repr()`` on *value*.
        - Shared by ``prompts.BasePromptBuilder``'s walker and
          ``context.clean_context_keys`` (as ``filter_context_tree``'s
          ``leaf`` hook) and by ``session._session_json_default`` (the
          ``FileSessionStore.save`` ``default=`` hook);
          ``fsm._strip_internal_mapping`` does NOT use it.

    # DECISION plan-2026-09-21T203800-8a03483a/D-010
    # Redact, do NOT drop (the model still learns the key exists) and do NOT
    # ``str()`` the object: its ``__str__``/``__repr__`` is exactly the path
    # that carried `password='...'` fields past every key filter. Do NOT wire
    # this into `get_data`'s walker: a handler-stored object is legitimate
    # application data there. See decisions.md D-010.
    """
    # DECISION plan-2026-09-21T203800-8a03483a/D-032
    # The stdlib value scalars stay (a date in context is ordinary data and a
    # prior plan pins it reaching Pass 2); do NOT widen this to `isinstance`
    # or to arbitrary types with a "safe-looking" `__str__`. See D-032.
    if (
        value is None
        or isinstance(value, _JSON_NATIVE_LEAF_TYPES)
        or type(value) in _VALUE_SCALAR_TYPES
    ):
        return value
    return f"<redacted:{type(value).__name__}>"


def filter_context_tree(
    source: dict[Any, Any],
    max_depth: int,
    should_drop: Callable[[Any, Any, str], str | None],
    on_drop: Callable[[str, str], None] = lambda _path, _reason: None,
    leaf: Callable[[Any], Any] | None = None,
) -> dict[Any, Any]:
    """Recursively filter a mapping's keys, bounded at ``max_depth``.

    # DECISION plan-2026-09-12T135914-45a654de/D-016
    # This is the ONE shared depth-bounded recursion (dict/list/tuple, own
    # ``_TOO_DEEP`` sentinel, fail-closed at the bound) previously
    # hand-duplicated byte-identically between `fsm.py`'s
    # `_strip_internal_value`/`_strip_internal_mapping` and `context.py`'s
    # `clean_value`/`clean_mapping` closures. It does NOT change, weaken, or
    # merge either caller's per-key policy -- see `fsm.py`'s own
    # `# DECISION plan-2026-07-20T040150-876e7164/D-010` comment (kept in
    # place at that call site) for why `fsm.py`'s predicate MUST stay a bare
    # `isinstance(key, str) and has_internal_prefix(key)` check with a
    # no-op `on_drop`: it feeds `API.get_data()`, a per-turn read accessor,
    # and must never gain `context.py`'s None-stripping, forbidden-pattern
    # check, or WARNING logging. `context.py`'s richer 5-reason predicate and
    # its logging/`removed_keys`/`warned_keys` tracking are reproduced
    # UNCHANGED inside its own `should_drop`/`on_drop` closures at its call
    # site -- this walker only owns the recursion shape, never the policy.
    # Do NOT widen `should_drop`'s contract to accept `None`,
    # `remove_none_values`, or `is_forbidden_context_entry` as walker-level
    # concepts -- that would be exactly the leak D-010 forbids. See
    # decisions.md D-016 (and the historical D-010 it respects).

    Contract:
        - ``source``: the mapping to filter (rebuilt into a new dict; never
          mutated in place).
        - ``max_depth``: the caller's own depth bound (e.g.
          ``MAX_CONTEXT_FILTER_DEPTH`` from ``constants.py`` for both current
          callers, though the walker itself has no opinion on that constant).
        - ``should_drop(key, value, full_key) -> reason | None``: called once
          per key at every mapping level. Return a reason string to drop the
          key (its value is never visited); return ``None`` to keep it (the
          value is then recursed into if it is itself a container). The
          predicate MAY have its own side effects (e.g. logging a warning for
          a non-``str`` key, or recording a "kept but flagged" key) -- the
          walker never inspects or requires anything beyond the return value.
        - ``on_drop(full_key, reason)``: called for every drop, whether from
          ``should_drop`` returning a reason or from the depth bound being
          exceeded (in which case ``reason`` is the literal string
          ``"too_deep"`` -- callers that want depth-drops to log/format
          differently from a policy drop must check for this exact value).
          Defaults to a no-op.
        - Recursion mirrors the two original implementations exactly: a
          dict value recurses at the SAME depth it was scheduled at; a
          list/tuple's own elements each get one additional depth increment
          beyond that (this asymmetry existed identically in both original
          implementations and is preserved verbatim, not "fixed").
        - Scalars pass through unchanged at any depth (or through ``leaf``
          when given); only dict/list/tuple containers are ever subject to
          the depth bound.
        - Fail-closed at the bound: a container deeper than ``max_depth`` is
          dropped, never returned unfiltered.
        - ``leaf(value) -> value``: optional, applied to every non-container
          value that is kept (``None`` = identity). Policy stays with the
          caller: it is a closure, like ``should_drop``.
        - Cycles: a container already on the ACTIVE recursion path is dropped
          with reason ``"cycle"``. A container reached twice WITHOUT a cycle
          (shared, acyclic) is filtered at every occurrence and gives the
          same value at each (equal, and when it cannot reach a cycle the
          SAME object, like ``copy.deepcopy`` preserves aliasing).
        - Never truncates: the result is either the whole filtered value or
          an exception. A linear pre-scan (``_cycle_reaching_containers``)
          finds every container that lies on or can reach a cycle. Every
          OTHER container is memoised per ``(container, depth)`` wherever it
          sits, even when the context holds a cycle elsewhere, so aliasing
          costs linear work and ``should_drop``/``on_drop`` run once per
          distinct container per depth, not once per path. Cycle-reaching
          containers are walked per path (the active-path guard makes their
          result path-dependent); if those walks charge more than
          ``MAX_CONTEXT_FILTER_NODES + CONTEXT_FILTER_CYCLIC_WORK_FACTOR *
          distinct_items`` items it raises ``ContextFilterWorkError`` (only a
          cycle that is itself heavily aliased gets there).
        - Three call sites today: `fsm.py::_strip_internal_mapping` (silent,
          bare-prefix predicate), `context.py::clean_mapping` (5-reason
          predicate, logging `on_drop`) and `pipeline.py::_json_native_values`
          (forbidden-entry predicate on the classification `context_snapshot`)
          -- a "same shape, different behavior" extraction, not a policy merge.

    # DECISION plan-2026-09-21T203800-8a03483a/D-011
    # The depth bound alone does not bound WORK: 3-way aliasing at 14 levels
    # is 3**14 paths and never finished. Guards sit NEXT TO the bound
    # (never replacing it, per D-010): an active-path `id()` set drops a true
    # cycle. Do NOT keep a global `seen` set (that drops the second
    # occurrence of a shared list). See decisions.md D-011.
    #
    # DECISION plan-2026-09-21T203800-8a03483a/D-045
    # Both callers return or COMMIT data (`get_data`, `save_session`, the
    # extracted-data commit), so this walker must NEVER truncate: do NOT put
    # a node budget back here (it silently cut a 150,000-item list and
    # dropped every later key). Aliasing is bounded by memoising per
    # (container, depth). Cyclic input over the work ceiling RAISES, never
    # returns a partial value. The truncating budget belongs to the prompt
    # walker only. See D-045.
    #
    # DECISION plan-2026-09-21T203800-8a03483a/D-052
    # Memoise per SUBTREE, not per graph: one unrelated self-referential dict
    # must not turn a 1,000-alias list into 200,000 visits and a raise. Do
    # NOT memoise a container that can reach a cycle (a cached subtree can
    # reach a container that is active at the reuse site, which must then be
    # a cycle drop), and do NOT go back to one global acyclic bit. See D-052.
    """
    reaching, distinct_items = _cycle_reaching_containers(source)
    active: set[int] = set()
    # (id, depth) -> filtered value, for containers that reach no cycle.
    memo: dict[tuple[int, int], Any] = {}
    remaining = [
        MAX_CONTEXT_FILTER_NODES + CONTEXT_FILTER_CYCLIC_WORK_FACTOR * distinct_items
    ]

    def _filter_value(value: Any, path: str, depth: int) -> Any:
        if not isinstance(value, (dict, list, tuple)):
            return value if leaf is None else leaf(value)
        if depth > max_depth:
            return _TOO_DEEP
        if id(value) in active:
            return _CYCLE
        memoisable = id(value) not in reaching
        memo_key = (id(value), depth)
        if memoisable and memo_key in memo:
            return memo[memo_key]
        if not memoisable:
            remaining[0] -= len(value) + 1
            if remaining[0] < 0:
                raise ContextFilterWorkError(
                    "context has cyclic references aliased too heavily to "
                    f"filter (more than {CONTEXT_FILTER_CYCLIC_WORK_FACTOR}x "
                    "its distinct size); refusing to return a partial value"
                )
        active.add(id(value))
        try:
            if isinstance(value, dict):
                filtered: Any = _filter_mapping(value, path, depth)
            else:
                items = []
                for index, item in enumerate(value):
                    element_path = f"{path}[{index}]"
                    filtered_item = _filter_value(item, element_path, depth + 1)
                    dropped = _drop_reason(filtered_item)
                    if dropped is not None:
                        on_drop(element_path, dropped)
                        continue
                    items.append(filtered_item)
                filtered = tuple(items) if isinstance(value, tuple) else items
        finally:
            active.discard(id(value))
        if memoisable:
            memo[memo_key] = filtered
        return filtered

    def _filter_mapping(
        mapping: dict[Any, Any], path: str, depth: int
    ) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        for key, value in mapping.items():
            full_key = f"{path}.{key}" if path else str(key)
            reason = should_drop(key, value, full_key)
            if reason is not None:
                on_drop(full_key, reason)
                continue
            filtered = _filter_value(value, full_key, depth + 1)
            dropped = _drop_reason(filtered)
            if dropped is not None:
                on_drop(full_key, dropped)
                continue
            result[key] = filtered
        return result

    active.add(id(source))
    return _filter_mapping(source, "", 0)


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
          leave un-clamped) are mapped to ``default``; the result is then
          clamped to ``[0, 1]``.
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
    value = float(raw)  # may raise TypeError/ValueError — intentional
    if math.isnan(value) or math.isinf(value):
        return default
    return min(max(value, 0.0), 1.0)


# --------------------------------------------------------------
# JSON Processing Utilities
# --------------------------------------------------------------


def _match_brace_partners(text: str, brace_positions: list[int]) -> dict[int, int]:
    """Map each ``{`` position in *text* to the index of the ``}`` balancing it.

    Contract:
        - ``brace_positions`` must be the ascending list of every ``{`` index in
          ``text`` (i.e. ``[m.start() for m in re.finditer(r"\\{", text)]``).
        - Returns ``{start: end}`` for every start whose balanced-brace scan
          terminates. Starts that never balance before end-of-text are ABSENT
          from the mapping (callers must use ``.get()``).
        - Each span is the one a JSON-string/escape-aware scan begun *at that
          start position with a fresh in-string state* would find — not the one
          a single global scan from index 0 would find. The two differ whenever
          an earlier lone ``"`` shifts the global in-string parity, e.g.
          ``'he said " {"a": 1}'``; the per-start reading is the historical
          behavior and is load-bearing (see the D-023/D-002 notes below).
        - Linear rather than quadratic: spans resolve innermost-first
          (descending start order) so an outer scan jumps over an already
          resolved nested span instead of re-walking it.
        - Never raises for any ``str`` input.
    """
    partner: dict[int, int] = {}
    text_len = len(text)

    # Innermost-first: every nested span an outer scan can meet is already known.
    for start_pos in reversed(brace_positions):
        i = start_pos + 1
        in_string = False
        escape_next = False

        while i < text_len:
            char = text[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == "\\":
                escape_next = True
                i += 1
                continue

            if char == '"':
                in_string = not in_string
                i += 1
                continue

            if not in_string:
                if char == "{":
                    nested_end = partner.get(i)
                    if nested_end is None:
                        # The nested span never balances, so neither can this
                        # one — the original scan would have run to end-of-text.
                        break
                    # Resume just past the nested span. Its closing `}` is
                    # outside a string, so the string/escape state is clean.
                    i = nested_end + 1
                    continue
                if char == "}":
                    partner[start_pos] = i
                    break

            i += 1

    return partner


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """
    Enhanced JSON extraction from text with multiple fallback strategies.

    This function handles various formats of JSON that might be returned
    by LLMs, including code blocks, partial JSON, and embedded structures.

    Args:
        text: Text potentially containing JSON data

    Returns:
        Extracted JSON object (always a ``dict``) or ``None`` if extraction
        fails. A JSON value that is not an object (``42``, ``[1, 2]``,
        ``true``, ``"hi"``, ``null``) is NOT an extraction result and yields
        ``None``; callers may rely on ``isinstance(result, dict)`` without a
        further check.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    logger.debug("Attempting enhanced JSON extraction from text")

    # Strategy 1: Direct JSON parsing
    # DECISION plan-2026-09-19T175721-21cd7f8e/D-010
    # A text that parses cleanly to a non-object returns None immediately; it
    # does NOT fall through to the brace scan. Falling through would recover
    # the first object inside a top-level array (`[{"a":1}]` -> `{"a":1}`), a
    # "recover more" change that would reach fsm_llm_harness/hardening.py,
    # whose documented contract is that a top-level array is not a payload.
    # Do NOT return the list/scalar either: the `dict | None` annotation was a
    # lie and Classifier._parse_single died on it with AttributeError.
    try:
        parsed = json.loads(text.strip())
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, RecursionError):
        # RecursionError: a deeply nested payload is undecodable, not fatal
        pass

    # Strategy 2: Extract from code blocks
    # DECISION plan-2026-09-19T175721-21cd7f8e/D-010
    # The first fenced block is located with two `str.find` calls, NOT the regex
    # ``` ```(?:json)?\s*([\s\S]*?)\s*``` ```: with an unclosed fence followed by a
    # long whitespace run that pattern retries `\s*```` from every lazy-group
    # position and is O(n^2) (50,000 spaces took minutes) on provider text. The
    # body is everything between the opener (plus an optional `json` tag) and the
    # next fence, stripped; identical to the regex's group(1).strip().
    scan_text = text
    fence_open = text.find("```")
    if fence_open != -1:
        body_start = fence_open + 3
        if text.startswith("json", body_start):
            body_start += 4
        fence_close = text.find("```", body_start)
        if fence_close != -1:
            try:
                json_str = text[body_start:fence_close].strip()
                logger.debug("Found JSON in code block")
                result = json.loads(json_str)
                # D-010: a fenced non-object (`[1,2]`) is treated exactly like
                # an undecodable fence and falls through to Strategy 3.
                if isinstance(result, dict):
                    return result
                # RA-04: the fenced span is skipped, otherwise the brace scan
                # would return the object inside a fenced array (`[{"a":1}]`).
                # DECISION plan-2026-09-19T175721-21cd7f8e/D-029
                # The span is BLANKED in place (positions preserved), not cut
                # off with `text[fence_close + 3:]`: truncation lost an object
                # BEFORE the fence. Strategy 4 reads this same string (set
                # below); do NOT let it read the original `text`, that merged
                # keys from the fenced array with an earlier object.
                scan_text = (
                    text[:fence_open]
                    + " " * (fence_close + 3 - fence_open)
                    + text[fence_close + 3 :]
                )
            except (json.JSONDecodeError, RecursionError):
                logger.debug("Code block JSON parsing failed")

    # Strategy 3: Find balanced JSON objects
    try:
        # Find all potential JSON start positions
        brace_positions = [m.start() for m in re.finditer(r"\{", scan_text)]

        # DECISION plan-2026-07-19T191147-4b664252/D-002 [STALE]
        # The closing partner of every `{` is precomputed ONCE, innermost-first,
        # instead of rescanning text[start_pos:] from every start position. That
        # rescan was O(n^2): 20,000 bare `{` characters took 18.2s of CPU on
        # text that arrives straight from the LLM provider, while the caller
        # holds the per-conversation lock (a real DoS vector).
        #
        # This changed the COMPLEXITY ONLY. Two things below are load-bearing
        # and must NOT be "simplified":
        #   1. The loop stays FIRST-wins — see the D-023 block immediately
        #      below, and decisions.md D-023. A last-wins flip was shipped and
        #      REVERTED once already.
        #   2. `_match_brace_partners` deliberately resolves each span with a
        #      scan begun AT that start position with a fresh in-string state.
        #      Do NOT replace it with one global left-to-right stack pass from
        #      index 0. That looks equivalent and is not: a single earlier lone
        #      `"` flips the global in-string parity, so a real object gets
        #      classified as string content and dropped. Probe that regresses:
        #        'x " {"a":1} " y'  ->  must return {'a': 1}, not None.
        # See decisions.md D-002.
        closing_index = _match_brace_partners(scan_text, brace_positions)

        skip_until = -1
        # DECISION plan-2026-07-18T162030-a02151fe/D-023 [STALE]
        # This strategy is FIRST-wins: it returns on the first successful parse.
        # That disagrees with Strategy 4 below and with
        # llm.py::_extract_content_from_thinking, both of which prefer the LAST
        # object. The divergence is REAL and still OPEN — which helper you hit
        # depends only on whether the provider split the reasoning trace into a
        # separate `thinking` field.
        #
        # Do NOT "fix" it by flipping this loop to last-wins. That was tried
        # (D-021) and REVERTED, because it does not work:
        #   - Strategy 2 (code fence, above) runs FIRST and is also first-match,
        #     so the fenced draft-then-final case that motivated the flip stayed
        #     broken. Only the unfenced variant changed.
        #   - It actively broke the answer-then-example shape:
        #     'The intent is {"intent": "buy"}. Schema: {"intent": "<name>"}'
        #     started returning the EXAMPLE. For classification.py that silently
        #     degrades a correct intent to `fallback_intent`.
        # Resolving this means deciding what these three helpers are FOR (is a
        # trailing object a correction, or a restated schema?), and changing
        # Strategy 2 in step with whatever is chosen. That is a design question,
        # not a one-line tie-break. See decisions.md D-023.
        for start_pos in brace_positions:
            if start_pos <= skip_until:
                continue  # Skip positions inside a previously scanned span

            end_pos = closing_index.get(start_pos)
            if end_pos is None:
                continue  # Never balances before end of text — try next start

            # Found complete JSON object
            json_str = scan_text[start_pos : end_pos + 1]
            try:
                brace_result: dict[str, Any] = json.loads(json_str)
                logger.debug(
                    "Successfully extracted JSON using balanced brace matching"
                )
                return brace_result
            except json.JSONDecodeError:
                # Skip nested positions inside this failed span
                skip_until = end_pos

    except Exception as e:
        logger.debug(f"Error during balanced brace JSON extraction: {e}")

    # Strategy 4: Extract key-value pairs using regex (fallback)
    text = scan_text  # same string as Strategy 3 (D-029)
    try:
        # Patterns for simple string values
        string_patterns = [
            r'"message"\s*:\s*"([^"]*)"',
            r'"selected_transition"\s*:\s*"([^"]*)"',
            r'"reasoning"\s*:\s*"([^"]*)"',
            # `intent` is a quoted string value, so it rides the same
            # last-match-wins string loop below. Without it, a recoverable
            # classification intent in garbled text silently degrades to
            # `fallback_intent` even though `meaningful_keys` claims to
            # cover it. See findings G4.
            r'"intent"\s*:\s*"([^"]*)"',
        ]

        extracted = {}

        for pattern in string_patterns:
            # Prefer the LAST match: a <think> trace often quotes these keys
            # before the real JSON, and the trailing occurrence is the
            # authoritative value.
            matches = re.findall(pattern, text)
            if matches:
                key = pattern.split('"')[1]
                extracted[key] = matches[-1]

        # `confidence` is an UNQUOTED number, so the quoted-string loop above
        # cannot capture it. Extract it separately, same last-match-wins
        # rationale. The pattern consumes a FULL numeric token — optional sign,
        # fraction, AND scientific-notation exponent — so `1e-3` is read as
        # 0.001, not silently truncated to "1" -> 1.0 (the G6-class silent
        # max-certainty defect; see findings CF2). The trailing negative
        # lookahead `(?![0-9.eE])` rejects malformed multi-dot / dangling-`e`
        # tokens (e.g. "1.2.3", "1e") outright rather than capturing a partial
        # prefix. The float() guard stays as a belt-and-suspenders skip. See
        # findings G4/CF2.
        confidence_matches = re.findall(
            r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?![0-9.eE])',
            text,
        )
        if confidence_matches:
            try:
                extracted["confidence"] = float(confidence_matches[-1])
            except ValueError:
                pass

        # For extracted_data, find the key and then use balanced braces
        ed_match = re.search(r'"extracted_data"\s*:\s*\{', text)
        if ed_match:
            # Start balanced brace matching from the opening brace
            brace_start = text.index("{", ed_match.start())
            depth = 0
            in_str = False
            esc = False
            for i, ch in enumerate(text[brace_start:], brace_start):
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                extracted["extracted_data"] = json.loads(
                                    text[brace_start : i + 1]
                                )
                            except json.JSONDecodeError:
                                extracted["extracted_data"] = {}
                            break

        # CF5: `intent`/`confidence` are AUXILIARY classification keys. G4 added
        # their capture here, which flipped a previously-None result into a dict
        # for garbled free text that merely mentions one of them — a real
        # cross-package hazard for non-classification callers (e.g. a lenient
        # all-optional structured-output schema in fsm_llm_agents/base.py would
        # then build a partial model from a stray `"confidence": 0.8` substring).
        # Keep a lone auxiliary key ONLY when a co-occurring PRIMARY payload key
        # is present IN THE TEXT — message/selected_transition/value/
        # extracted_data, or the OTHER member of the intent/confidence pair
        # (mutual reinforcement: a genuine classification payload carries both).
        # A malformed-but-present counterpart key (e.g. `"confidence": 1.2.3`
        # that fails to parse) still counts as co-occurring, so a recoverable
        # intent survives. See findings CF5.
        def _key_present_in_text(key: str) -> bool:
            return re.search(rf'"{key}"\s*:', text) is not None

        _primary_present = any(
            _key_present_in_text(k)
            for k in ("message", "selected_transition", "value", "extracted_data")
        )
        if not _primary_present:
            if "intent" in extracted and not _key_present_in_text("confidence"):
                extracted.pop("intent", None)
            if "confidence" in extracted and not _key_present_in_text("intent"):
                extracted.pop("confidence", None)

        # Only return if we have structurally meaningful keys, not just auxiliary ones.
        # Includes keys used by classification (intent, confidence) and response
        # generation (message, reasoning) callers — not just data extraction.
        meaningful_keys = {
            "selected_transition",
            "extracted_data",
            "message",
            "reasoning",
            "intent",
            "confidence",
        }
        if extracted and (meaningful_keys & extracted.keys()):
            logger.debug(
                f"Extracted JSON using regex fallback: {list(extracted.keys())}"
            )
            return extracted
        elif extracted:
            logger.debug(
                f"Regex fallback found only auxiliary keys {list(extracted.keys())}, treating as failed"
            )

    except Exception as e:
        logger.debug(f"Regex fallback extraction failed: {e}")

    logger.warning("All JSON extraction strategies failed")
    return None


def validate_json_structure(data: dict[str, Any], required_keys: list[str]) -> bool:
    """
    Validate that JSON data contains required keys.

    Args:
        data: JSON data to validate
        required_keys: List of required key names

    Returns:
        True if all required keys are present, False otherwise
    """
    if not isinstance(data, dict):
        return False

    missing_keys = [key for key in required_keys if key not in data]

    if missing_keys:
        logger.debug(f"JSON validation failed: missing keys {missing_keys}")
        return False

    return True


# --------------------------------------------------------------
# FSM Definition Loading
# --------------------------------------------------------------


def load_fsm_from_file(file_path: str) -> FSMDefinition:
    """
    Load FSM definition from JSON file with enhanced validation.

    Args:
        file_path: Path to JSON file containing FSM definition

    Returns:
        Validated FSM definition object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or doesn't conform to FSM structure
    """
    logger.info(f"Loading FSM definition from file: {file_path}")

    try:
        # Check file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FSM definition file not found: {file_path}")

        # Load and parse JSON
        with open(file_path, encoding="utf-8") as f:
            fsm_data = json.load(f)

        # Validate basic structure
        if not isinstance(fsm_data, dict):
            raise ValueError("FSM definition must be a JSON object")

        # Enhance with version info if missing
        if "version" not in fsm_data:
            fsm_data["version"] = "4.1"
            logger.debug("Added default version 4.1 to FSM definition")

        # Create and validate FSM definition
        fsm_definition = FSMDefinition(**fsm_data)

        logger.info(f"Successfully loaded FSM definition: {fsm_definition.name}")
        logger.debug(
            f"FSM contains {len(fsm_definition.states)} states, "
            f"initial state: {fsm_definition.initial_state}"
        )

        return fsm_definition

    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in FSM definition file: {e!s}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Error loading FSM definition from {file_path}: {e!s}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def load_fsm_definition(fsm_id_or_path: str) -> FSMDefinition:
    """
    Load FSM definition by ID or file path with fallback logic.

    Args:
        fsm_id_or_path: Either FSM ID or file path

    Returns:
        Loaded FSM definition

    Raises:
        ValueError: If FSM cannot be loaded
    """
    # Check if input looks like a file path
    if (
        os.path.exists(fsm_id_or_path)
        or "/" in fsm_id_or_path
        or "\\" in fsm_id_or_path
        or fsm_id_or_path.endswith(".json")
    ):
        return load_fsm_from_file(fsm_id_or_path)

    # Otherwise treat as FSM ID - no built-in FSM registry for now
    logger.error(f"Unknown FSM ID: {fsm_id_or_path}")
    raise ValueError(f"Unknown FSM ID: {fsm_id_or_path}")


# --------------------------------------------------------------
# Debug and Development Utilities
# --------------------------------------------------------------


def get_fsm_summary(fsm_definition: FSMDefinition) -> dict[str, Any]:
    """
    Generate summary information about an FSM definition.

    Args:
        fsm_definition: FSM definition to summarize

    Returns:
        Dictionary with summary information
    """
    states = fsm_definition.states

    # Count transitions
    total_transitions = sum(len(state.transitions) for state in states.values())

    # Find terminal states
    terminal_states = [
        state_id for state_id, state in states.items() if not state.transitions
    ]

    # Find states with conditions
    states_with_conditions = [
        state_id
        for state_id, state in states.items()
        if any(transition.conditions for transition in state.transitions)
    ]

    # Find required context keys
    all_required_keys = set()
    for state in states.values():
        if state.required_context_keys:
            all_required_keys.update(state.required_context_keys)

    return {
        "name": fsm_definition.name,
        "version": fsm_definition.version,
        "state_count": len(states),
        "initial_state": fsm_definition.initial_state,
        "terminal_states": terminal_states,
        "terminal_count": len(terminal_states),
        "total_transitions": total_transitions,
        "states_with_conditions": len(states_with_conditions),
        "unique_required_keys": sorted(all_required_keys),
        "has_persona": bool(fsm_definition.persona),
    }
