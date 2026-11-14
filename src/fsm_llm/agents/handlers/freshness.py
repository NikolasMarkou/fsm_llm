"""Loop freshness of the agent patterns (plan d454fd62 step 88): ``make_redraft_handlers`` (a state that
redoes a draft) and ``make_fresh_keys_handler`` (a producing state that must re-extract its keys). Core
extracts a key only while it is unset, so both clear on entry. The function bodies are the old
``agents/handlers.py`` ones; the plain clear is core ``clear_keys_delta``."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from fsm_llm.agents.constants import RESULT_DROPPED_CONTEXT_KEYS
from fsm_llm.context.commit import clear_keys_delta

from .forced import is_forced_verdict, recorded_forced_reason


# DECISION plan-2026-09-24T045559-3e4eb3e5/D-013: core extracts a key only
# while it is unset (pipeline skip-if-set), so a draft or verdict left in
# context is never re-judged and
# a redo loop replays round 1. Do NOT clear these keys on entry to the JUDGING
# state instead: that erases the limiter's forced pass (PRE_TRANSITION runs
# before entry) and the loop runs to the 3x ceiling. Do NOT delete the draft
# without stashing it: the redo state's prompt must still see it.
def make_redraft_handlers(
    draft_key: str,
    previous_key: str,
    *,
    clear_on_exit: tuple[str, ...] = (),
) -> tuple[
    Callable[[dict[str, Any]], dict[str, Any]],
    Callable[[dict[str, Any]], dict[str, Any]],
]:
    """
    Build the (entry, exit) handler pair for a state that redoes a draft.

    Args:
        draft_key: The key the redo state must re-extract.
        previous_key: Visible key the old draft is moved to on entry.
        clear_on_exit: Keys deleted on exit, once the redo state has
            consumed them (e.g. feedback the next judge must rewrite).

    Returns:
        ``(on_entry, on_exit)``, for ``on_state_entry`` and ``on_state_exit``
        of the redo state. Each returns a context delta in which ``None``
        deletes a key; neither raises. ``on_exit`` restores ``draft_key``
        from ``previous_key`` when the redo state produced no new draft.
    """

    def on_entry(context: dict[str, Any]) -> dict[str, Any]:
        delta: dict[str, Any] = {draft_key: None}
        if context.get(draft_key) is not None:
            delta[previous_key] = context[draft_key]
        return delta

    def on_exit(context: dict[str, Any]) -> dict[str, Any]:
        delta: dict[str, Any] = dict.fromkeys(clear_on_exit)
        if context.get(draft_key) is None:
            delta[draft_key] = context.get(previous_key)
        return delta

    return on_entry, on_exit


# DECISION plan-2026-09-29T103145-06a5ec0a/D-008: loop freshness is ONE rule,
# built here. Register the handler on entry to the PRODUCING (redo) state,
# never on entry to the judging state: PRE_TRANSITION runs before entry, so
# that would erase the limiter's forced verdict (3e4eb3e5 D-013). Do NOT
# clear a forced True (see is_forced_verdict): the forced edge must still
# route. Do NOT stash under an internal-prefixed or unlisted key: the stash
# must stay prompt-visible and out of results (c1d5bfbc D-012).
def make_fresh_keys_handler(
    keys: Iterable[str],
    *,
    stash: Mapping[str, str] | None = None,
    keep_limiter_forced: bool = True,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build an entry handler that makes a producing state re-extract ``keys``.

    Core extracts a key only while it is unset (skip-if-set, typed
    ``field_extractions`` included), so a loop value left in context freezes
    after round 1. Clearing it on entry to the state that produces it lets
    that state's extraction run again.

    Interface contract (ReAct think, Reflexion, PlanExecute, Debate,
    PromptChain; register with ``on_state_entry(<producing state>)``; a key
    another state writes for this one to read, such as ``agent_feedback``, is
    cleared with ``on_state_exit(<consuming state>)`` instead, D-029):
        - ``keys``: non-empty; the context keys the producing state writes.
        - ``stash``: optional ``{key: stash_key}`` for keys whose previous
          value the producing prompt must still see. Every ``key`` must be in
          ``keys`` and every ``stash_key`` in ``RESULT_DROPPED_CONTEXT_KEYS``
          (visible to prompts, dropped from results).
        - Returns a handler whose delta is core ``clear_keys_delta`` (each
          listed key that holds a value set to ``None``, which core deletes)
          minus the kept keys below; a stashed key's value is also
          copied to its stash key. Unset keys are left out of the delta, so an
          older stash survives a round that produced nothing. A key holding a
          forced ``True`` (:func:`is_forced_verdict`) is left untouched. The
          handler never raises.
        - ``keep_limiter_forced``: ``False`` keeps a ``True`` only when a
          forcing handler recorded a reason (:func:`recorded_forced_reason`);
          a ``True`` forced by the bare ``max_iterations_reached`` flag is
          cleared. Only for a state whose forced edge reads the flag itself
          (ReAct-family think, D-051 of plan 06a5ec0a).
        - Raises ``ValueError`` at build time for empty ``keys`` or an invalid
          ``stash`` mapping.
    """
    fresh = tuple(dict.fromkeys(keys))
    if not fresh:
        raise ValueError("make_fresh_keys_handler needs at least one key")
    stash_map = dict(stash or {})
    unknown = sorted(set(stash_map) - set(fresh))
    if unknown:
        raise ValueError(f"stash names keys that are not refreshed: {unknown}")
    unlisted = sorted(set(stash_map.values()) - RESULT_DROPPED_CONTEXT_KEYS)
    if unlisted:
        raise ValueError(
            f"stash keys must be in RESULT_DROPPED_CONTEXT_KEYS: {unlisted}"
        )

    def kept(value: Any, context: dict[str, Any]) -> bool:
        if keep_limiter_forced:
            return is_forced_verdict(value, context)
        return value is True and recorded_forced_reason(context) is not None

    def refresh_keys(context: dict[str, Any]) -> dict[str, Any]:
        cleared = clear_keys_delta(
            [key for key in fresh if not kept(context.get(key), context)], context
        )
        delta: dict[str, Any] = {}
        for key in cleared:
            delta[key] = None
            if key in stash_map:
                delta[stash_map[key]] = context[key]
        return delta

    return refresh_keys
