"""Context filtering and redaction: the shared depth-bounded context-tree filter and the
JSON ``default=`` hook (the leaf redaction).

Layer ``context`` (plan d454fd62, step 72): split out of the old ``fsm_llm.utilities``. The
redaction walk (``redact_context``, ``REDACTED``, ``CYCLE_PLACEHOLDER``) moved down to
``fsm_llm.core.redaction`` at step 95.
"""

from __future__ import annotations

import datetime
import decimal
import uuid
from collections.abc import Callable
from typing import Any

from fsm_llm.constants import (
    CONTEXT_FILTER_CYCLIC_WORK_FACTOR,
    MAX_CONTEXT_FILTER_NODES,
)
from fsm_llm.core.definitions import FSMError

# --------------------------------------------------------------
# Depth-bounded context-filter tree walker
# --------------------------------------------------------------

# Sentinels for a value the walker itself drops, keyed to the ``on_drop``
# reason reported for it. Module-private -- callers never see these values,
# only the filtered result and the reason string.
_TOO_DEEP = object()
_CYCLE = object()
_DROP_REASONS = ((_TOO_DEEP, "too_deep"), (_CYCLE, "cycle"))


class ContextFilterError(FSMError, ValueError):
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
          ``leaf`` hook) and by ``redacting_json_default`` (the
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


def redacting_json_default(value: Any) -> str:
    """``json.dumps`` ``default=`` hook for every writer that emits context or
    trace values outside the process: on disk (``FileSessionStore.save``,
    ``fsm_llm.agents.memory_persistence``), into an LLM prompt
    (``fsm_llm.reasoning.engine``) or over a socket
    (``fsm_llm.monitor``'s dashboard websocket).

    Returns ``str(value)`` only for an exact stdlib value scalar (the
    ``redact_non_json_leaf`` keep-set; json never calls ``default`` for its
    other kept types), else ``"<redacted:TypeName>"``. Never raises.
    """
    # DECISION plan-2026-09-22T080837-8b258a25/D-008
    # Do NOT go back to `default=str`: an object's `__str__` is the path that
    # carried a secret to disk, to a prompt and to the dashboard. Do NOT copy
    # the scalar type set here; reuse the leaf hook.
    redacted = redact_non_json_leaf(value)
    return str(value) if redacted is value else redacted


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
        - Depth: the top-level mapping is depth 0 and every nested container
          (a dict value or a list/tuple element) sits one level deeper than
          its parent; one deeper than ``max_depth`` is dropped.
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
          finds every container that lies on or can reach a cycle (skipped
          for a plain ``dict`` whose values are all leaves). Every
          OTHER container is memoised per ``(container, depth)`` wherever it
          sits, even when the context holds a cycle elsewhere, so aliasing
          costs linear work and ``should_drop``/``on_drop`` run once per
          distinct container per depth, not once per path. Cycle-reaching
          containers are walked per path (the active-path guard makes their
          result path-dependent); if those walks charge more than
          ``MAX_CONTEXT_FILTER_NODES + CONTEXT_FILTER_CYCLIC_WORK_FACTOR *
          distinct_items`` items it raises ``ContextFilterError`` (only a
          cycle that is itself heavily aliased gets there).
        - Four call sites today: `fsm.py::_strip_internal_mapping` (silent,
          bare-prefix predicate), `context.py::clean_mapping` (5-reason
          predicate, logging `on_drop`), `pipeline.py::_json_native_values`
          (forbidden-entry predicate on the classification `context_snapshot`)
          and `monitor/collector.py::redact_context` (the dashboard's display
          filter, with a `leaf` hook) -- a "same shape, different behavior"
          extraction, not a policy merge.

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
    # DECISION plan-2026-09-22T080837-8b258a25/D-037
    # A plain dict of leaves reaches no container, so `reaching` is empty and
    # `distinct_items` is len(source) either way: skip only the pre-scan. Do
    # NOT widen to Mapping/dict subclasses (the pre-scan walks a non-dict
    # Mapping's keys) and do NOT add a truncating shortcut (D-045). See D-037.
    reaching: set[int]
    if type(source) is dict and not any(
        isinstance(value, (dict, list, tuple)) for value in source.values()
    ):
        reaching, distinct_items = set(), len(source)
    else:
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
                raise ContextFilterError(
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
