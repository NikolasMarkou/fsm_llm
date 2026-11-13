"""Context writes: the clear delta, the commit with key rollback and the cut-off record.

Layer ``context`` (plan d454fd62, step 74): ``clear_keys_delta`` moved down from
``fsm_llm.handlers`` so that ``context`` imports no engine module, and the commit and cut-off
helpers of the old ``fsm_llm.context`` live beside it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from fsm_llm.core.definitions import (
    CONTEXT_KEY_CUT_OFF,
    FSMContext,
    FSMInstance,
    checked_key_names,
)


# DECISION plan-2026-10-01T093600-944e2692/D-051
# Do NOT put a key that is already unset (absent or None) in the delta: a round
# with nothing to clear must return an empty delta, which agents'
# make_fresh_keys_handler relies on (an older stash survives a round that
# produced nothing). Do NOT add a "keep" predicate here: the forced-verdict
# rule is agent semantics and stays in agents.
def clear_keys_delta(keys: Iterable[str], context: Mapping[str, Any]) -> dict[str, Any]:
    """The handler delta that clears ``keys`` from ``context``.

    Contract (callers: :func:`clear_keys_on_entry`, ``ContextCompactor``,
    agents ``make_fresh_keys_handler`` and the meta-builder): returns ``{key: None}``
    (core deletes the key) for each key in ``keys``, in order, whose value in
    ``context`` is not None; a key that is absent or None is left out, so
    nothing to clear gives ``{}``. Raises ``ValueError`` when ``keys`` is not
    a collection of ``str`` names (``None`` or a bare ``str``,
    ``definitions.checked_key_names``).
    """
    names = checked_key_names(keys, argument="keys")
    return {key: None for key in names if context.get(key) is not None}


def commit_with_key_rollback(
    context: FSMContext,
    committed: dict[str, Any],
    notify: Callable[[], None],
    on_failure: Callable[[Exception], None],
) -> None:
    """Write ``committed`` into ``context.data``, run ``notify``, undo only those
    keys if ``notify`` raises.

    Interface contract (callers: the Pass-1 extraction commit in
    ``engine/turn/pass1.py`` and ``FSMManager.update_conversation_context``):
    ``committed`` is the delta to write (``FSMContext.update`` writes exactly
    its keys). ``notify`` fires the CONTEXT_UPDATE handlers and takes no
    argument. On an ``Exception`` from ``notify``, ``on_failure(error)`` is
    called first (the caller logs there), then each committed key is restored
    to its prior value, or removed when it had none, and the error is
    re-raised unchanged. Returns ``None``. ``BaseException`` is not caught.

    BOUNDARY (known, deliberate, shapes 4b664252/D-005 and 876e7164/D-009):
    the snapshot is SHALLOW. Key coverage is exact (``FSMContext.update`` writes
    precisely ``committed``'s keys and derives nothing, so no committed key
    escapes the restore) but ``pre_commit`` holds the SAME objects as
    ``context.data[key]``. A CONTEXT_UPDATE handler that mutates a pre-existing
    dict/list value IN PLACE before failing is therefore NOT restored. Do not
    "fix" this with a deep copy or a full ``context.data`` restore: that
    reintroduces a per-call copy to undo a mutation handlers are contractually
    supposed to make by returning a delta, and would also discard the deltas of
    CONTEXT_UPDATE handlers that already SUCCEEDED, which survive at this
    timing point. Precedence: if a handler delta touched one of ``committed``'s
    keys, the rollback wins.
    """
    pre_commit = {key: context.data[key] for key in committed if key in context.data}
    context.update(committed)
    try:
        notify()
    except Exception as handler_err:
        on_failure(handler_err)
        for key in committed:
            if key in pre_commit:
                context.data[key] = pre_commit[key]
            else:
                context.data.pop(key, None)
        raise


def was_cut_off(response: Any) -> bool:
    """True when ``response.truncated`` is exactly ``True`` (a provider cut-off).

    Contract: any object; a response without the attribute (a test double, a
    third-party ``LLMInterface`` returning a duck-typed reply) is not cut off.
    Never raises.
    """
    return getattr(response, "truncated", False) is True


def record_cut_off(
    instance: FSMInstance, *, reply: bool = False, field: str | None = None
) -> None:
    """Note in ``context.data[CONTEXT_KEY_CUT_OFF]`` that a model reply was cut.

    Contract: ``reply=True`` records the Pass-2 reply; ``field`` appends the name
    of an extraction (a field, or a completion state's result key) once. The
    record is a new dict each call (copy-on-write: a snapshot a caller holds
    never changes) holding JSON-native values only. Written straight into
    ``context.data``: the key is internal, so it never reaches a prompt or
    ``API.get_data``. The turn bodies (``_run_turn``, ``_stream_turn``) and the
    greeting remove it before they start, so it describes the current turn only.
    A streamed reply is never flagged (the stream carries no finish reason).
    Callers: the Pass-2 reply and the bulk and field extractions in ``engine/turn/``, and
    the completion executor in ``executors.py`` (it lives here so the executors
    need no import of the pipeline).
    """
    # DECISION plan-2026-11-06T093029-0068b055/D-025: ONE key, set by the
    # pipeline from the `truncated` flag the LLM layer puts on a response. Do
    # NOT re-derive the cut in an agent from text (D-022 forbids a text-shape
    # guess) and do NOT add a second key per kind of reply. See decisions.md D-025.
    data = instance.context.data
    record = dict(data.get(CONTEXT_KEY_CUT_OFF) or {})
    if reply:
        record["reply"] = True
    if field is not None and field not in record.get("fields", []):
        record["fields"] = [*record.get("fields", []), field]
    data[CONTEXT_KEY_CUT_OFF] = record
