"""Module-level helpers of ``FSMManager``: the caller-visible context filter, the
logger-binding decorator, the turn-failure mapping and the conversation lookup.

Layer ``engine`` (plan d454fd62, step 78): split out of the old ``fsm_llm.fsm``; the
function bodies are unchanged.
"""

from __future__ import annotations

import traceback
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

from fsm_llm.constants import MAX_CONTEXT_FILTER_DEPTH, has_internal_prefix
from fsm_llm.context.filtering import filter_context_tree
from fsm_llm.core.definitions import FSMError, FSMInstance
from fsm_llm.logging import logger

if TYPE_CHECKING:
    from .gate import TurnGate

# --------------------------------------------------------------
# Caller-visible context filter
# --------------------------------------------------------------


# DECISION plan-2026-07-20T040150-876e7164/D-010 [STALE]
# This filter DROPS an internal-prefixed key. It does NOT replace the value
# with "<redacted>" -- do NOT "harmonize" it into a redactor. `plans/SYSTEM.md`
# claims this site redacts "by original design"; that claim is FALSE at HEAD
# and was false before this change too (the pre-change body was a single
# `{k: v ... if not has_internal_prefix(k)}` comprehension). The redact-don't-
# drop site is `context.filtering.redact_context` (D-015/D-014), whose job is to show a
# CLI operator WHICH keys exist. This one feeds `API.get_data()`, whose return
# value is application DATA: injecting the literal string "<redacted>" into it
# would make a filtered key indistinguishable from a real value.
#
# Do NOT remove the depth bound and do NOT return an unfiltered sub-tree at it.
# Behaviour AT the bound is fail-CLOSED (drop), the same as
# `context.clean_context_keys` (D-010) and
# `context.layout.base.BasePromptBuilder._filter_context_for_security` (D-011) -- three
# filters, one bound, one disposition here. The bound is also what makes a
# self-referential dict (`d["self"] = d`) terminate.
#
# Do NOT "reuse" `clean_context_keys` here instead: it also drops None values
# and empty-string keys (a behaviour change for `get_data`) and logs a
# forbidden-pattern WARNING, which on a per-turn read accessor is log spam.
# What IS shared, and must stay shared, is `has_internal_prefix` and
# `MAX_CONTEXT_FILTER_DEPTH` -- never re-declare either here.
# See decisions.md D-010.
#
# DECISION plan-2026-09-12T135914-45a654de/D-016
# The depth-bounded dict/list/tuple recursion itself (the part that was
# byte-identical to `context/compactor.py`'s walker) now lives in ONE shared place,
# `context.filtering.filter_context_tree`. This call site supplies ONLY the bare
# predicate the D-010 comment above describes, plus a no-op `on_drop` -- it
# does not, and must never, reference `None`, `remove_none_values`, or
# `is_forbidden_context_entry`. See decisions.md D-016.
def _strip_internal_mapping(source: dict[Any, Any]) -> dict[Any, Any]:
    """Apply the internal-prefix drop at every level via the shared walker."""

    def _should_drop(key: Any, _value: Any, _full_key: str) -> str | None:
        # A non-`str` key carries no prefix to match and `has_internal_prefix`
        # would raise on it (constants.py D-017), so it is kept -- its VALUE
        # is still filtered.
        if isinstance(key, str) and has_internal_prefix(key):
            return "internal key prefix"
        return None

    # No `leaf` hook here (D-010 of plan-2026-09-21T203800-8a03483a): a
    # handler-stored object is application data for `get_data`, not a leak.
    # DECISION plan-2026-09-21T203800-8a03483a/D-045
    # No node budget either: this feeds `get_data` and `save_session`, which
    # must return the whole value or raise, never a truncated one. See D-045.
    return filter_context_tree(source, MAX_CONTEXT_FILTER_DEPTH, _should_drop)


# --------------------------------------------------------------


def with_conversation_context(func):
    """Bind ``conversation_id`` to a contextual logger passed as ``log=``.

    A caller that passes ``log=`` itself keeps it (a sync name that hands its own logger to
    the async body it drives).
    """

    @wraps(func)  # Preserve function metadata
    def wrapper(self, conversation_id, *args, **kwargs):
        # Bind the conversation ID to the logger
        kwargs.setdefault("log", logger.bind(conversation_id=conversation_id))

        # Call the original function with the contextual logger
        return func(self, conversation_id, *args, **kwargs)

    return wrapper


def _unexpected_turn_failure(log: Any, message: str | None, exc: Exception) -> FSMError:
    """Log an unexpected turn failure (call inside its ``except`` arm); return the FSMError to raise."""
    what = "advance conversation" if message is None else "process message"
    log.error(f"Failed to {what}: {exc!s}\n{traceback.format_exc()}")
    return FSMError(f"Failed to {what}: {exc!s}")


class _ConversationRegistry(Protocol):
    """What ``_conversation_entry`` reads of a manager: its two lookup tables.

    A structural type, so the mixins of ``FSMManager`` (which are not
    ``FSMManager``) can pass ``self`` (mypy only; a runtime no-op).
    """

    instances: dict[str, FSMInstance]
    _conversation_locks: dict[str, TurnGate]


@overload
def _conversation_entry(
    manager: _ConversationRegistry,
    conversation_id: str,
    *,
    missing_lock: Literal["raise", "keyerror"],
) -> tuple[FSMInstance, TurnGate]: ...


@overload
def _conversation_entry(
    manager: _ConversationRegistry,
    conversation_id: str,
    *,
    missing_lock: Literal["unlocked"],
) -> tuple[FSMInstance, TurnGate | None]: ...


def _conversation_entry(
    manager: _ConversationRegistry,
    conversation_id: str,
    *,
    missing_lock: Literal["raise", "keyerror", "unlocked"],
) -> tuple[FSMInstance, TurnGate | None]:
    """The instance and the per-conversation lock of ``conversation_id``.

    Interface contract: the CALLER HOLDS ``manager._lock`` (the lookup reads
    ``instances`` and ``_conversation_locks`` and takes no lock of its own, so
    the ``_lock -> conv_lock`` order is the caller's). A module function, not a
    method, so the nested ``_stream`` closure of ``_open_locked_stream`` can
    use it without a ``self._helper`` dispatch (4c63deac/D-002). Raises
    ``FSMError("Conversation <id> not found")`` for an unknown conversation.
    ``missing_lock`` is what an instance with no lock means (a broken
    create-together / remove-together invariant, L11), and each caller keeps
    the policy it had: ``"raise"`` an ``FSMError`` (reads, ``update_conversation_context``),
    ``"keyerror"`` a bare ``KeyError`` (the turn and stream bodies), ``"unlocked"``
    returns ``None`` for the lock and the caller proceeds without it
    (``set_conversation_state``, ``end_conversation``).
    """
    if conversation_id not in manager.instances:
        raise _not_found(conversation_id)
    conv_lock = manager._conversation_locks.get(conversation_id)
    if conv_lock is None:
        if missing_lock == "raise":
            raise FSMError(
                f"Conversation {conversation_id} has an instance but no "
                "per-conversation lock (broken invariant)"
            )
        if missing_lock == "keyerror":
            raise KeyError(conversation_id)
    return manager.instances[conversation_id], conv_lock


def _not_found(conversation_id: str) -> FSMError:
    """The error of an unknown or ended conversation (one text for every lookup)."""
    return FSMError(f"Conversation {conversation_id} not found")


def _require_registered(
    manager: _ConversationRegistry, conversation_id: str, instance: FSMInstance
) -> None:
    """Refuse a conversation that ended while the caller waited on its gate.

    Contract: call it AFTER ``conv_lock`` is held, on the ``instance`` the earlier lookup
    returned. Raises ``_not_found`` when ``instances`` no longer holds that very object
    (ended, or ended and started again under the id). Takes NO lock: ``end_conversation``
    removes the instance while it holds the gate, so a read here is ordered after the
    removal, and taking ``_lock`` under ``conv_lock`` would invert the ``_lock -> conv_lock``
    order (D-005 of plan 9925aa3a).
    """
    if manager.instances.get(conversation_id) is not instance:
        raise _not_found(conversation_id)
