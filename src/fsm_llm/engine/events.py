"""The engine's side of the event log: the state blob and the turn boundary (plan 21e6f00d).

Layer ``engine``. ``FSMManager`` calls ``open_turn``, ``abegin_turn`` and ``aend_turn`` around every turn it
runs (``engine/fsm/turns.py``), so the pipeline bodies and their anchors are not edited; the
``API`` calls ``record_state`` for the records it writes out of any turn
(``engine/api/events.py``). With no scope bound (``runtime.current_scope()`` is ``None``,
the log is off) ``open_turn`` is one ``ContextVar`` read and returns ``None``, and ``aend_turn(None, ...)``
returns at once: nothing is built, nothing is written.

Interface contract:
    - ``state_blob(instance, definition) -> dict``: the JSON-safe state of one conversation:
      ``fsm_id``, ``current_state``, ``context_data`` (the snapshot's internal-key-free
      context through ``filter_context``, so secret-looking values are redacted),
      ``conversation_history`` and ``conversation_summary`` as they are,
      ``working_memory`` (buffers through ``filter_context``) and ``hidden_buffers``,
      ``provenance``, and ``internal``: the completion transcript and response format values of
      the definition's own completion states and NOTHING else (see the D-009 anchor). The
      caller holds the instance's ``conv_lock`` or owns the instance.
    - ``open_turn(manager, conversation_id, message) -> TurnHandle | None``, ``abegin_turn(handle)``
      and ``aend_turn(handle, error_type)``: ``open_turn`` runs with ``conv_lock`` held, BEFORE
      the user message is added, and builds the pre-turn state blob ON THE LOOP (it reads the live
      instance); ``abegin_turn`` appends ``turn/start`` (``state_ref``, ``state``,
      ``has_user_message``, ``source`` and, with a message, ``user_message_ref``) as ONE lane job.
      A failing write raises (``EventLogError``, the turn does not start). ``aend_turn`` runs
      AFTER the manager's rollback arms, takes no manager lock, builds the end state blob on the
      loop and appends ``turn/end`` (``outcome`` and the ``state_ref`` at the end, a checkpoint)
      as ONE lane job behind ``turn/start`` (nothing for a turn that did not start). It never
      raises for an ``Exception``: a ``turn/end`` that cannot be written is a WARNING (the
      poisoned writer refuses the next turn), because it runs in a ``finally`` that must
      release the lock and must not replace the turn's own result.
    - Outcome (``TurnOutcome``), decided by what the manager left in memory: no error is
      ``committed``; ``KeyboardInterrupt`` and ``SystemExit`` are ``interrupted``;
      ``GeneratorExit`` (a stream closed early) is ``abandoned``; any other exception is
      ``rolled_back`` when the end state blob equals the start one (same ``state_ref``, the blobs
      are content addressed) and ``abandoned`` when it does not (a partial Pass-1 commit that
      the pipeline keeps by contract). The compare covers the whole blob, history and summary
      included, so a failed turn whose only surviving change is in the history (the trim of old
      exchanges into the summary is deferred to the reply, so this needs a reply added before
      a later failure; not reproduced) would read ``abandoned``: the label is state-faithful.
    - ``record_budget_spent(error)``: append one ``decision`` event (``kind: "budget_spent"``,
      ``budget``, ``limit``, ``steps_done`` and nothing else) for a turn the lifetime budget
      refused (``fsm.budget.TurnBudget``); no-op with no scope bound, a WARNING when the write
      fails, called by the turn path after the manager lock is released.
    - ``delta_blob(delta, definition) -> dict``: the log-safe form of an out-of-turn context write
      (``API.update_context``): ``{"context": ..., "internal": ...}``. ``context`` holds the
      public keys through ``filter_context``; ``internal`` only the keys the definition's
      completion states declare (the D-009 allow-list), nothing else; a ``None`` value stays
      ``None`` (the write keeps the key with value ``None``).
    - ``record_delta(manager, scope, child_id, delta, *, source) -> int``: append the
      ``context/record`` of an out-of-turn write to conversation *child_id* under *scope*: ``source``,
      ``keys`` (the written key names, sorted) and ``delta_ref`` (the ``delta_blob`` of the
      definition that conversation runs). Reads ``manager.instances`` here, not in the ``API``
      modules (a source scan keeps the ``API`` off the manager's tables).
    - ``record_state(manager, log, conversation_id, *, source, frame=0, root_id=None, **fields)
      -> int``: append one ``context/record`` with the state of the conversation read under ONE
      ``conv_lock`` hold (out of any turn: ``start_conversation``). ``arecord_state`` is the one body
      (the read stays on the loop, the blob and the line are ONE lane job) and ``record_state`` its
      ``run_sync`` wrapper. *conversation_id* is the id
      the MANAGER knows; the event goes to the log of *root_id* (default: the same id) at
      *frame*, so a pushed frame (its own manager id, the tree's root id) can be recorded.
    - ``state_ref_of(manager, log, conversation_id) -> str``: the ``state_blob`` of the
      conversation, read under ONE ``conv_lock`` hold, stored as a blob; the reference. The
      ``stack/push`` (the child's post-greeting state) and ``stack/pop`` (the parent's
      post-merge state) producers use it.

Privacy (plan 5a6a1f80, row 1.11a): at level ``shape`` (``log.stores_values`` is False) no
producer here STORES a blob. A turn keeps the DIGEST of its start and end state (``_ref``) so
``rolled_back`` and ``abandoned`` stay distinct, and the digest is never written (the log drops
every ``*_ref``); ``abegin_turn`` stores no user message, ``record_delta`` no delta. ``record_state``
still BUILDS the state blob at ``shape`` (``_read_state_blob`` always runs ``state_blob``, once per
start, restore, fork or resume) to read ``current_state`` from it, and then drops it. At level
``values`` every blob is stored with its ``RedactionKind`` for the user's hook.

Imports ``core``, ``runtime`` and ``engine.fsm.snapshots`` only; never the pipeline or the API.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from fsm_llm.aio import run_sync
from fsm_llm.core.codes import current_message_source
from fsm_llm.core.definitions import FSMError
from fsm_llm.core.state_kinds import kind_of
from fsm_llm.runtime import (
    EventLog,
    EventType,
    RedactionKind,
    TurnOutcome,
    TurnScope,
    current_scope,
    filter_context,
    to_jsonable,
    warn_on_failure,
)

from .fsm.snapshots import snapshot_from_instance

if TYPE_CHECKING:
    from fsm_llm.core.definitions import (
        FSMDefinition,
        FSMInstance,
        RunBudgetExceededError,
    )

    from .fsm.manager import FSMManager

#: The context keys a kind config may declare as transcript or response-format holders.
_KIND_KEY_FIELDS = ("messages_key", "response_format_key")


class _TurnManager(Protocol):
    """What ``open_turn`` reads of the manager (``TurnsMixin`` declares both for mypy)."""

    instances: dict[str, FSMInstance]

    def get_fsm_definition(self, fsm_id: str) -> FSMDefinition: ...


@dataclass
class TurnHandle:
    """What ``open_turn`` remembers for ``abegin_turn`` and ``aend_turn``; ``start_ref`` stays
    ``None`` until ``turn/start`` is written, so a turn that did not start gets no ``turn/end``."""

    scope: TurnScope
    instance: FSMInstance
    definition: FSMDefinition
    start_blob: dict[str, Any]
    message: str | None
    fields: dict[str, Any]
    start_ref: str | None = None


def declared_internal_keys(definition: FSMDefinition) -> tuple[str, ...]:
    """The internal context keys the definition's own states declare (transcript and format).

    Read through the kind registry (``kind_of(state).config_field``), so the engine never
    names a kind config. Order is the definition's, duplicates dropped; empty for a definition
    with no completion state.
    """
    # DECISION plan-2026-11-17T100529-21e6f00d/D-009
    # The log carries internal keys ONLY by this allow-list, read from the running definition
    # (a completion state's `messages_key` and `response_format_key`). Do NOT carry internal
    # keys by prefix (`_`, `system_`, ...) or by a deny-list: the same namespace holds the
    # driver-only grant `_approval_granted`, `RUN_OUTPUT_KEYS`, a pattern's `_run_output_keys`
    # and counters, and a blob that carried them would be a forgery path into the HITL gate
    # at restore. Do NOT persist the transcript in `FileSessionStore` snapshots either: D-029
    # and D-039 of plan 944e2692 (anchors in `engine/kinds/completion.py`) still bind that
    # path and two pinned tests hold it. Only a restore from the LOG carries a transcript.
    names: list[str] = []
    for state in definition.states.values():
        field = kind_of(state).config_field
        config = getattr(state, field, None) if field else None
        for attribute in _KIND_KEY_FIELDS:
            name = getattr(config, attribute, None)
            if isinstance(name, str) and name and name not in names:
                names.append(name)
    return tuple(names)


def _working_memory(memory: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The buffers of a ``WorkingMemory.to_dict()`` through ``filter_context``, one by one.

    ``to_dict`` adds a ``_hidden_buffers`` list that is not a buffer (the hidden names travel
    in their own field) and an internal-prefixed buffer name would be dropped by the filter,
    so each buffer is filtered on its own.
    """
    if memory is None:
        return None
    return {
        name: filter_context(buffer)
        if isinstance(buffer, Mapping)
        else to_jsonable(buffer)
        for name, buffer in memory.items()
        if name != "_hidden_buffers"
    }


def _instance_of(manager: _TurnManager, conversation_id: str) -> FSMInstance:
    """The instance of *conversation_id*, or ``FSMError`` when it is gone.

    ``end_conversation`` ends a frame at the manager before the ``API`` drops its stack entry, so
    a producer can name a frame that has just ended. Read through ``dict.get`` (atomic), never
    through ``manager._lock``: ``open_turn`` runs under a ``conv_lock`` and ``_lock`` taken under
    one breaks the lock order (``test_fsm_lock_order.py``).
    """
    instance = manager.instances.get(conversation_id)
    if instance is None:
        raise FSMError(f"Conversation not found: {conversation_id}")
    return instance


def state_blob(instance: FSMInstance, definition: FSMDefinition) -> dict[str, Any]:
    """Return the log-safe state of *instance* (see the module docstring)."""
    snapshot = snapshot_from_instance(instance)
    keys = declared_internal_keys(definition)
    data = instance.context.data
    blob: dict[str, Any] = {
        "fsm_id": instance.fsm_id,
        "current_state": snapshot["current_state"],
        "context_data": filter_context(snapshot["context_data"]),
        "conversation_history": snapshot["conversation_history"],
        "conversation_summary": snapshot["conversation_summary"],
        "working_memory": _working_memory(snapshot["working_memory"]),
        "hidden_buffers": snapshot["hidden_buffers"],
        "provenance": filter_context(snapshot["provenance"]),
        "internal": filter_context(
            {key: data[key] for key in keys if key in data}, keep_internal=keys
        ),
    }
    safe: dict[str, Any] = to_jsonable(blob)
    return safe


def delta_blob(delta: Mapping[str, Any], definition: FSMDefinition) -> dict[str, Any]:
    """Return the log-safe form of a context write (see the module docstring)."""
    keys = declared_internal_keys(definition)
    blob: dict[str, Any] = to_jsonable(
        {
            "context": filter_context(
                {name: value for name, value in delta.items() if name not in keys}
            ),
            "internal": filter_context(
                {name: delta[name] for name in keys if name in delta},
                keep_internal=keys,
            ),
        }
    )
    return blob


def record_delta(
    manager: _TurnManager,
    scope: TurnScope,
    child_id: str,
    delta: Mapping[str, Any],
    *,
    source: str,
) -> int:
    """Append the ``context/record`` of a context write (see the module docstring)."""
    definition = manager.get_fsm_definition(_instance_of(manager, child_id).fsm_id)
    fields: dict[str, Any] = {}
    if scope.log.stores_values:
        fields["delta_ref"] = scope.log.put_json(
            delta_blob(delta, definition), kind=RedactionKind.DELTA
        )
    return scope.log.append(
        scope.root_id,
        EventType.CONTEXT_RECORD,
        frame=scope.frame,
        source=source,
        keys=sorted(str(name) for name in delta),
        **fields,
    )


def _ref(log: EventLog, blob: Mapping[str, Any]) -> str:
    """The reference of a state blob: stored at level ``values``, only its digest at ``shape``."""
    if log.stores_values:
        return log.put_json(blob, kind=RedactionKind.STATE)
    return log.digest_json(blob)


def open_turn(
    manager: _TurnManager, conversation_id: str, message: str | None
) -> TurnHandle | None:
    """Build what ``turn/start`` needs ON THE LOOP (it reads the live instance, which an orphan
    lane job must never do); ``None`` with no scope bound."""
    scope = current_scope()
    if scope is None:
        return None
    instance = manager.instances[conversation_id]
    definition = manager.get_fsm_definition(instance.fsm_id)
    fields: dict[str, Any] = {
        "state": instance.current_state,
        "has_user_message": message is not None,
    }
    if message is not None:
        fields["source"] = current_message_source().value
    return TurnHandle(
        scope, instance, definition, state_blob(instance, definition), message, fields
    )


def _write_start(handle: TurnHandle) -> None:
    """Store the start state and the message, append ``turn/start`` (one lane job)."""
    log = handle.scope.log
    start_ref = _ref(log, handle.start_blob)
    fields = {"state_ref": start_ref, **handle.fields}  # at shape the digest: dropped
    if handle.message is not None and log.stores_values:
        fields["user_message_ref"] = log.put_json(
            handle.message, kind=RedactionKind.USER_MESSAGE
        )
    log.append(
        handle.scope.root_id, EventType.TURN_START, frame=handle.scope.frame, **fields
    )
    handle.start_ref = start_ref


async def abegin_turn(handle: TurnHandle | None) -> None:
    """Append ``turn/start``: ONE lane job (the first writer open included). A failing write
    raises (``EventLogError``) and the turn does not start."""
    if handle is not None:
        await handle.scope.log.lane_call(handle.scope.root_id, _write_start, handle)


def _outcome(error: type[BaseException] | None, unchanged: bool) -> TurnOutcome:
    """The outcome rule of the module docstring."""
    if error is None:
        return TurnOutcome.COMMITTED
    if issubclass(error, (KeyboardInterrupt, SystemExit)):
        return TurnOutcome.INTERRUPTED
    if issubclass(error, GeneratorExit):
        return TurnOutcome.ABANDONED
    return TurnOutcome.ROLLED_BACK if unchanged else TurnOutcome.ABANDONED


def _write_end(
    handle: TurnHandle, end_blob: dict[str, Any], error: type[BaseException] | None
) -> None:
    """Store the end state and append ``turn/end`` (one lane job); nothing if it never started."""
    if handle.start_ref is None:
        return
    log = handle.scope.log
    end_ref = _ref(log, end_blob)
    outcome = _outcome(error, end_ref == handle.start_ref)
    log.append(
        handle.scope.root_id,
        EventType.TURN_END,
        frame=handle.scope.frame,
        outcome=outcome.value,
        state_ref=end_ref,
    )


async def aend_turn(
    handle: TurnHandle | None, error: type[BaseException] | None
) -> None:
    """Append ``turn/end`` for the turn that ran (the end state is read here, on the loop, AFTER
    the rollback arms; the write is ONE lane job); never raises an ``Exception``."""
    if handle is None:
        return
    with warn_on_failure(EventType.TURN_END):
        end_blob = state_blob(handle.instance, handle.definition)
        # DECISION plan-2026-11-24T193347-3f4530d3/D-019
        # [SUPERSEDES D-016 of this plan] A cancel that lands on this write RAISES, with or
        # without a turn error: the turn stays committed, the `turn/end` job still completes
        # (FIFO, shielded await) and the caller gets no reply, the rule of every cancel after
        # the commit point (`_settle_after`, D-026 of plan baa37f40). Do NOT swallow it (the
        # cancel was lost for the rest of a run: `cancelling()` stayed 1, a run loop took six
        # more steps, a `TaskGroup` or `asyncio.timeout` never saw it, and the reply preceded
        # the durable `turn/end`), do NOT absorb and re-raise it later (a cancel pending
        # across a reply), and do NOT shield the whole `finally` (a hung disk pins the caller).
        await handle.scope.log.lane_call(
            handle.scope.root_id, _write_end, handle, end_blob, error
        )


def record_budget_spent(error: RunBudgetExceededError) -> None:
    """Append the ``decision`` of a turn the lifetime budget refused; a no-op with no scope bound.

    The event is ``kind: "budget_spent"`` with ``budget``, ``limit`` and ``steps_done`` of
    *error* and nothing else (no user text: the refused message is never read). It runs AFTER
    the refusal is decided and outside every manager lock, so it never raises an ``Exception``:
    a write that fails is a WARNING and the caller raises the budget error as it would have.
    """
    scope = current_scope()
    if scope is None:
        return
    with warn_on_failure(EventType.DECISION):
        scope.log.append(
            scope.root_id,
            EventType.DECISION,
            frame=scope.frame,
            kind="budget_spent",
            budget=error.budget,
            limit=error.limit,
            steps_done=error.steps_done,
        )


def _read_state_blob(manager: FSMManager, conversation_id: str) -> dict[str, Any]:
    """The ``state_blob`` of *conversation_id*, read under ONE ``conv_lock`` hold.

    The definition is resolved BEFORE the hold (``_read_under_lock`` forbids resolving inside
    it), as the anchors of ``engine/fsm/`` require.
    """
    definition = manager.get_fsm_definition(
        _instance_of(manager, conversation_id).fsm_id
    )
    return manager._read_under_lock(
        conversation_id, lambda inst: state_blob(inst, definition)
    )


def state_ref_later(
    manager: FSMManager, log: EventLog, conversation_id: str
) -> Callable[[], str]:
    """Read the state of *conversation_id* now; the returned call stores its blob (a reference)."""
    blob = _read_state_blob(manager, conversation_id)
    return lambda: log.put_json(blob, kind=RedactionKind.STATE)


def state_ref_of(manager: FSMManager, log: EventLog, conversation_id: str) -> str:
    """Store the current state of *conversation_id* as a blob and return its reference."""
    return state_ref_later(manager, log, conversation_id)()


async def arecord_state(
    manager: FSMManager,
    log: EventLog,
    conversation_id: str,
    *,
    source: str,
    frame: int = 0,
    root_id: str | None = None,
    **fields: Any,
) -> int:
    """Append a ``context/record`` with the current state of *conversation_id*; return its ``seq``.

    The state is read through ``_read_state_blob`` (one ``conv_lock`` hold) on the loop; the blob
    and the line are ONE lane job. *source* names the writer (``"start"``); *fields* are extra
    event fields. The event is written to the log of *root_id* (the tree's root conversation id,
    default *conversation_id* itself, which is right for a root) at *frame*: a pushed frame has
    its own manager id and the root's log id.
    """
    root = conversation_id if root_id is None else root_id
    blob = _read_state_blob(manager, conversation_id)

    def write() -> int:
        refs: dict[str, Any] = {}
        if log.stores_values:
            refs["state_ref"] = log.put_json(blob, kind=RedactionKind.STATE)
        return log.append(
            root,
            EventType.CONTEXT_RECORD,
            frame=frame,
            source=source,
            state=blob["current_state"],
            **refs,
            **fields,
        )

    return await log.lane_call(root, write)


def record_state(
    manager: FSMManager,
    log: EventLog,
    conversation_id: str,
    *,
    source: str,
    frame: int = 0,
    root_id: str | None = None,
    **fields: Any,
) -> int:
    """The sync name of ``arecord_state`` (a ``run_sync`` wrapper: the write runs inline)."""
    return run_sync(
        arecord_state(
            manager,
            log,
            conversation_id,
            source=source,
            frame=frame,
            root_id=root_id,
            **fields,
        )
    )
