"""The redaction walk: replace secret-shaped context values before they are logged.

Layer ``core`` (plan d454fd62, step 95): moved down verbatim from ``fsm_llm.context.filtering``
so the tool layer (``fsm_llm.tools.arguments.redact_secret_entries``) reads it without importing
``context``, a sibling layer. ``redact_context`` is the one redaction walk of the CLI log dump
and of the tool registry's debug logs; it replaces a matched value and keeps the key.
Imports the kernel only (``constants``, ``logging``).
"""

from __future__ import annotations

from fsm_llm.constants import MAX_CONTEXT_FILTER_DEPTH, is_forbidden_context_entry
from fsm_llm.logging import logger

REDACTED = "<redacted>"
CYCLE_PLACEHOLDER = "<redacted:cycle>"


class _RedactionWalk:
    """Per-call state of one ``redact_context`` walk (never shared)."""

    def __init__(self) -> None:
        self.active: set[int] = set()  # ids of containers on the current path
        # (id, depth) -> redacted result, only for subtrees that met no cycle.
        self.memo: dict[tuple[int, int], object] = {}
        self.met_cycle = False


# DECISION plan-2026-07-18T162030-a02151fe/D-015 [STALE]
# This duplicates the key-matching loop in ``prompts.py``
# ``_filter_context_for_security`` ON PURPOSE. Do NOT "clean this up" by
# importing that method or hoisting it into a shared helper: it is a bound
# method of ``BasePromptBuilder`` gated on ``config.filter_internal_context``,
# so reusing it from the CLI would mean either instantiating a prompt builder
# here or refactoring a security path. The REGEX list is the thing that must
# stay single-sourced, and it is — never inline a secret pattern here.
# Values are replaced rather than dropped so an operator debugging the CLI can
# still see WHICH keys exist; a dropped key looks identical to a missing one.
#
# DECISION plan-2026-07-19T191147-4b664252/D-014 [STALE]
# The recursion is NOT optional and the replace-don't-drop rule above is NOT a
# harmonization bug. `clean_context_keys` (D-010) and `prompts.py`'s
# `_filter_context_for_security` (D-011) DROP a matched key; this one keeps the
# key and redacts its VALUE, on purpose — see the paragraph above. Do NOT
# "unify" the three. What IS shared, and must stay shared, is the matching
# (`COMPILED_FORBIDDEN_CONTEXT_PATTERNS`) and the bound
# (`MAX_CONTEXT_FILTER_DEPTH`); never inline a pattern or re-declare the depth.
# Behavior AT the bound is fail-CLOSED for the same reason as D-010: a subtree
# too deep to inspect is redacted wholesale rather than logged verbatim,
# otherwise burying a secret 17 levels down prints it. See decisions.md D-014.
#
# DECISION plan-2026-09-22T080837-8b258a25/D-016
# A cycle is REPLACED by `CYCLE_PLACEHOLDER`, never dropped (the rule above).
# The (id, depth) memo caches ONLY a subtree whose walk met no placeholder: a
# placeholder depends on the active path, so sharing it is wrong (D-052). Do
# NOT key the memo by id alone (the depth bound truncates per depth), and do
# NOT route this through `utilities.filter_context_tree` (it drops, D-023).
def _redact_value(value: object, depth: int, walk: _RedactionWalk) -> object:
    """Redact one context value; recurses into dicts and into lists/tuples."""
    if not isinstance(value, (dict, list, tuple)):
        return value
    if depth > MAX_CONTEXT_FILTER_DEPTH:
        return REDACTED
    node = id(value)
    if node in walk.active:
        walk.met_cycle = True
        return CYCLE_PLACEHOLDER
    memo_key = (node, depth)
    if memo_key in walk.memo:
        return walk.memo[memo_key]
    # Each subtree reports its OWN cycle flag; it is OR-ed back into the
    # parent's afterwards, so an ancestor of a placeholder is never cached.
    outer_met_cycle, walk.met_cycle = walk.met_cycle, False
    walk.active.add(node)
    if isinstance(value, dict):
        result: object = _redact_mapping(value, depth, walk)
    else:
        items = [_redact_value(item, depth + 1, walk) for item in value]
        result = tuple(items) if isinstance(value, tuple) else items
    walk.active.discard(node)
    if not walk.met_cycle:
        walk.memo[memo_key] = result
    walk.met_cycle = walk.met_cycle or outer_met_cycle
    return result


def _redact_mapping(source: dict, depth: int, walk: _RedactionWalk) -> dict:
    """Apply the key match at one level, then recurse into the values kept."""
    result = {}
    for key, value in source.items():
        # See constants.py D-017: a non-str key cannot be pattern-matched, so it
        # bypasses redaction entirely. Log it rather than skipping silently --
        # a `bytes` key here means a secret is about to be written to a log
        # verbatim. Do NOT downgrade to debug.
        if not isinstance(key, str):
            logger.warning(
                f"Log-redaction skipped for context key {key!r} "
                f"({type(key).__name__}): only str keys can be pattern-matched"
            )
            result[key] = _redact_value(value, depth + 1, walk)
        elif is_forbidden_context_entry(key, value):
            result[key] = REDACTED
        else:
            result[key] = _redact_value(value, depth + 1, walk)
    return result


def redact_context(data: dict) -> dict:
    """Replace secret-shaped context values before they are written to a log.

    The match is applied at every nesting level — inside nested dicts and
    inside dicts nested in lists/tuples — so ``{"user": {"password": "x"}}``
    is redacted like its flat equivalent. Matched KEYS stay visible and only
    their values become ``"<redacted>"``. Recursion is bounded at
    ``MAX_CONTEXT_FILTER_DEPTH``; anything deeper is redacted wholesale. A
    container already on the recursion path (a cycle) becomes
    ``"<redacted:cycle>"``. On ACYCLIC input, work is linear in the distinct
    (container, depth) pairs, so aliased input does not blow up. A subtree
    that meets a cycle is not memoised, so input that aliases every level AND
    cycles can still cost exponential time; the CLI runner only redacts
    ``get_data()`` output, which is always acyclic. The result may share one
    redacted subtree between the places an input subtree was aliased.
    """
    walk = _RedactionWalk()
    walk.active.add(id(data))
    return _redact_mapping(data, 0, walk)
