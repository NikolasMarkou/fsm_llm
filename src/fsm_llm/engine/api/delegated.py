"""The delegated part of ``settle``: start the run after the commit, the run events, the wait.

Layer ``engine`` (plan baa37f40 phase K7, step K7.4, D-052, roadmap E3.2 second half). The call
executor (``engine/kinds/call.py``) writes a delegated request in Pass 1 (``mode: delegated``, a
minted ``handle`` reserved in the launcher, a ``deadline``) and polls it on later turns. What it
cannot do is START the run: that is a side effect that must happen only after the turn committed,
which is ``settle`` (``calls.py``), and settle hands the delegated record to this module. This
module imports no ``engine.executables``: the launcher is the ``RunLauncher`` Protocol that
``API(targets=)`` may also be (``ExecutableTargets``).

Interface contract (``api`` is an ``API``, ``root_id`` its root conversation id):

- ``settle_delegated(api, root_id, frames, state, config, data, refuse) -> bool``: for a ``running``
  record of ``mode: delegated`` at the top frame of the stack (``frames[-1]``). ``True`` when it wrote
  a final record (through *refuse*, the ``_refuse`` of ``calls.py``), ``False`` when it started the
  run or left a run that is already started. Takes the target, the inputs and the budget from the
  PARENT's own definition and context (``data`` is the top frame's context), never from the record:
  the record contributes only its ``handle``, and the launcher starts that handle only when it holds
  a reservation of THIS conversation (Pass 1 made it): a forged record starts nothing and ends
  ``failed/run_lost``. ``run/start`` is appended (``checkpoint=True``, fsynced) by the ``announce``
  hook the launcher calls right before it starts the thread. A failed ``announce`` (the
  ``run/start`` write) propagates and the host puts the reservation back (``RunHost.start``,
  D-057), so the record stays ``running`` and the next entry starts the same handle; the end settle
  of a turn entry logs the failure at WARNING and the call's ``deadline`` still bounds it.
- ``flush_run_ends(api, root_id)``: appends ``run/end`` (status, reason, target, identities of the
  committed FINAL record) for each started run of this root whose frame is no longer in flight and
  whose record is final, once, then forgets the mark. Written on the calling (engine) thread only.
- ``wait_for_run(api, root_id)``: the ONE bounded wait (one helper, one semantics, D-056), at the end of
  ``aadvance`` and of the step streams (``advance_stream``): when the top
  frame has a started, unfinished run and the launcher has a ``wait`` method, wait at most
  ``DELEGATED_WAIT_SECONDS`` for it to finish, on a worker thread (``aoffload``), holding no lock.
- ``end_runs_of(api, root_id, frame_id)`` and ``cancel_delegated_runs(targets, conversation_id,
  frame_ids=())``: what ``pop_fsm`` does for a popped frame (and ``end_conversation`` for the
  ended ones, ``cancel_delegated_runs`` only): tell the launcher to stop the runs of the frame and write
  ``run/end`` (``aborted``, ``cancelled``) for each started one, at the top frame (D-056).
- ``RunMark``: what settle remembers of a started run (``CallMarks.runs``, keyed by handle).
- ``DELEGATED_WAIT_SECONDS``: the bound of that wait (owner-visible: a run loop over a delegated
  parent polls once per wait, so its ``max_steps`` must cover the seconds of the run over this).
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fsm_llm.aio import aoffload
from fsm_llm.core.codes import MessageSource, Reason, Status
from fsm_llm.core.definitions import CallStateConfig, State
from fsm_llm.engine.kinds.call import chosen_target, record_belongs
from fsm_llm.engine.kinds.targets import (
    DelegatedTarget,
    delegated_seconds,
    is_run_launcher,
)
from fsm_llm.logging import logger
from fsm_llm.runtime import EventType, RedactionKind, filter_context, warn_on_failure

if TYPE_CHECKING:
    from .stack import FSMStackFrame
    from .surface import _ApiSurface

__all__ = [
    "DELEGATED_WAIT_SECONDS",
    "RunMark",
    "cancel_delegated_runs",
    "end_runs_of",
    "flush_run_ends",
    "settle_delegated",
    "wait_for_run",
]

#: The longest ``aadvance`` waits for the started run of a delegated parent to finish (seconds).
DELEGATED_WAIT_SECONDS = 0.5

#: What ``settle`` calls to write the final record of a call it cannot run
#: (``calls._refuse``: api, root id, state id, config, reason, target, resolved).
_Refuse = Callable[..., Awaitable[None]]


@dataclass(frozen=True)
class RunMark:
    """A started delegated run: ``run/start`` is written and ``run/end`` is owed (in memory only)."""

    root_id: str
    frame_id: str
    state_id: str
    result_key: str
    target: str
    template_identity: str | None = None
    instance_identity: str | None = None


# DECISION plan-2026-11-23T122249-baa37f40/D-052
# A delegated run is STARTED by settle, after the turn committed, and only from a reservation.
# Do NOT start it from the record (the record is public data: a forged `call_result` or a restored
# snapshot would start whatever it names; the launcher's `start` consumes the reservation that
# Pass 1 made for THIS conversation, so a forged handle starts nothing and ends `failed/run_lost`),
# do NOT take the target, the inputs or the limit from the record (re-derive them from the
# parent's own state: the child receives the declared inputs only, never the grant), do NOT start
# the thread before `run/start` is on disk (the launcher calls `announce` first), do NOT put an
# `await` between the commit and the start (a cancel there would leave a reservation that is
# healed, but a run that started inside a rolled-back turn is not), do NOT let a worker write
# `run/start` or `run/end` (the log is written by the engine thread alone, so its order is the
# engine's), do NOT write `run/end` before the record that says it is final is committed (a rolled
# back turn logs nothing), and do NOT wait inside a lock (the wait is on a worker thread, after the
# turn, with the manager and the stack lock released). A failure while preparing the start (the
# resolution, the copy of an input) ends the call `push_failed` NOW, never on every entry until the
# deadline (D-058), and a result whose record landed is dropped from the host (D-057).
async def settle_delegated(
    api: _ApiSurface,
    root_id: str,
    frames: list[FSMStackFrame],
    state: State,
    config: CallStateConfig,
    data: Mapping[str, Any],
    refuse: _Refuse,
) -> bool:
    """Start the run of the delegated request at the top of the stack (see the module docstring)."""
    top = frames[-1]
    record = data.get(config.result_key)
    handle = record.get("handle") if isinstance(record, dict) else None
    if isinstance(handle, str):
        with api._stack_lock:
            if handle in api._call_marks.runs:
                return False  # started and logged: the executor polls it

    async def end(reason: Reason, found: DelegatedTarget | None) -> bool:
        await refuse(api, root_id, state.id, config, reason, target, found)
        return True

    target = chosen_target(config, data)
    launcher = api._targets
    if not is_run_launcher(launcher):
        return await end(Reason.RUN_LOST, None)  # nothing here can know the handle
    try:
        delegated = (
            None
            if target is None
            else launcher.resolve_run(
                target,
                input_keys=config.inputs,
                max_steps=config.max_steps,
                conversation_id=top.conversation_id,
            )
        )
        inputs = None if delegated is None else _declared_inputs(config, data)
    except Exception as error:
        # a failure that repeats on every entry would leave the call pending until its deadline
        # (600 seconds by default): end it now, naming the exception TYPE only (D-058)
        logger.warning(
            f"Delegated call {state.id!r} could not be prepared: {type(error).__name__}"
        )
        return await end(Reason.PUSH_FAILED, None)
    if delegated is None:
        return await end(Reason.UNKNOWN_TARGET, None)
    assert target is not None  # resolve_run answered, so a target was named
    if inputs is None:
        return await end(Reason.INPUTS_MISSING, delegated)
    if not isinstance(handle, str):
        return await end(Reason.RUN_LOST, delegated)

    def announce() -> None:
        _announce(
            api,
            root_id,
            len(frames) - 1,
            top,
            state,
            config,
            handle,
            target,
            delegated,
            inputs,
        )

    started = launcher.start(
        handle,
        delegated,
        inputs,
        conversation_id=top.conversation_id,
        max_seconds=delegated_seconds(config.max_seconds),
        max_steps=config.max_steps,
        announce=announce,
    )
    if started:
        return False
    polled = launcher.poll(handle, conversation_id=top.conversation_id)
    if polled is None:
        return await end(Reason.RUN_LOST, delegated)  # a handle the host never saw
    if polled.finished and polled.result is not None:
        # refused at start (a full host, a thread that cannot start): no run is live, so the
        # call ends now with the host's reason (D-058); a full host logged nothing, a thread
        # that could not start had already written its `run/start`
        _forget_result(api, handle, top.conversation_id)
        return await end(polled.result.reason or Reason.PUSH_FAILED, delegated)
    return False  # already running (or finished and landing): nothing to start


def _declared_inputs(
    config: CallStateConfig, data: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Private deep copies of the declared inputs, or ``None`` when one is missing."""
    inputs: dict[str, Any] = {}
    for name in config.inputs:
        value = data.get(name)
        if value is None:
            return None
        inputs[name] = copy.deepcopy(value)
    return inputs


def _announce(
    api: _ApiSurface,
    root_id: str,
    frame_index: int,
    top: FSMStackFrame,
    state: State,
    config: CallStateConfig,
    handle: str,
    target: str,
    delegated: DelegatedTarget,
    inputs: Mapping[str, Any],
) -> None:
    """Append ``run/start`` (a checkpoint: fsynced before the thread exists), then keep the mark.

    Runs on the engine thread, called by the launcher right before it starts the worker. A write
    that fails raises: the launcher then starts nothing."""
    log = getattr(api, "_event_log", None)
    if log is not None:
        fields: dict[str, Any] = {
            "handle": handle,
            "target": target,
            "template_identity": delegated.template_identity,
            "instance_identity": delegated.instance_identity,
            "source": MessageSource.RUN.value,
            "parent": top.conversation_id,
        }
        if log.stores_values:
            try:
                fields["inputs_ref"] = log.put_json(
                    filter_context(inputs), kind=RedactionKind.DELTA
                )
            except Exception as exc:  # an optional field: the event is still written
                logger.warning(
                    f"event log: run inputs not recorded ({type(exc).__name__})"
                )
        log.append(
            root_id, EventType.RUN_START, frame=frame_index, checkpoint=True, **fields
        )
    with api._stack_lock:
        api._call_marks.runs[handle] = RunMark(
            root_id,
            top.conversation_id,
            state.id,
            config.result_key,
            target,
            delegated.template_identity,
            delegated.instance_identity,
        )


#: The ``run/end`` of a run whose frame was popped: stopped from outside, never final.
_ORPHAN_END = (Status.ABORTED.value, Reason.CANCELLED.value, None, None)


def _append_run_end(
    log: Any,
    root_id: str,
    frame_index: int,
    handle: str,
    mark: RunMark,
    status: Any,
    reason: Any,
    template_identity: Any,
    instance_identity: Any,
) -> None:
    """Append one ``run/end`` at the top frame; a failed write is a WARNING (after the fact).

    ``template_identity`` and ``instance_identity`` default to those of ``run/start`` (the mark)
    when the caller has none."""
    if log is None:
        return
    with warn_on_failure(EventType.RUN_END):
        log.append(
            root_id,
            EventType.RUN_END,
            frame=frame_index,
            handle=handle,
            target=mark.target,
            status=status,
            reason=reason,
            template_identity=template_identity or mark.template_identity,
            instance_identity=instance_identity or mark.instance_identity,
        )


def end_runs_of(api: _ApiSurface, root_id: str, frame_id: str) -> None:
    """Write ``run/end`` (``aborted``/``cancelled``) for each started run of a frame that was just
    popped, at the top frame (the parent), then drop the marks.

    Contract: called by ``pop_fsm`` right after its ``stack/pop`` event, on the engine thread, with
    no lock held; a frame with no mark writes nothing; a failed append is a WARNING (D-056)."""
    marks = api._call_marks
    with api._stack_lock:
        owed = [
            (handle, mark)
            for handle, mark in marks.runs.items()
            if mark.root_id == root_id and mark.frame_id == frame_id
        ]
        for handle, _ in owed:
            marks.runs.pop(handle, None)
        frame_index = max(len(api.conversation_stacks.get(root_id) or ()) - 1, 0)
    log = getattr(api, "_event_log", None)
    for handle, mark in owed:
        _append_run_end(log, root_id, frame_index, handle, mark, *_ORPHAN_END)


def cancel_delegated_runs(
    targets: object, conversation_id: str, frame_ids: Iterable[str] = ()
) -> None:
    """Tell a ``RunLauncher`` resolver to stop the runs of an ended or popped conversation.

    Contract (K7.2, D-050; moved here by D-056, one hook for ``end_conversation`` and
    ``pop_fsm``): a ``targets`` that is no ``RunLauncher`` is left alone; ``cancel_for`` is called
    once for the id and once for each OTHER frame id (a call state in a pushed frame is run under
    that frame's manager id, D-051 (9)), never under ``_stack_lock``; a launcher that raises costs
    one WARNING naming only the exception type (the conversation is already gone, and no text of
    a worker or launcher error may reach the log).
    """
    if not is_run_launcher(targets):
        return
    for owner in dict.fromkeys([conversation_id, *frame_ids]):
        try:
            targets.cancel_for(owner)
        except Exception as error:
            logger.warning(
                f"Cancelling the delegated runs of {owner} failed: {type(error).__name__}"
            )


def _forget_result(api: _ApiSurface, handle: str, frame_id: str) -> None:
    """Tell a launcher that has a ``forget`` method to drop the finished result of a run whose
    final record is committed (D-057); an error is one WARNING naming only its type."""
    launcher = api._targets
    forget = getattr(launcher, "forget", None)
    if not is_run_launcher(launcher) or not callable(forget):
        return
    try:
        forget(handle, conversation_id=frame_id)
    except Exception as error:
        logger.warning(
            f"Forgetting a finished delegated run failed: {type(error).__name__}"
        )


def flush_run_ends(api: _ApiSurface, root_id: str) -> None:
    """Write ``run/end`` for each started run of *root_id* whose committed record is final."""
    marks = api._call_marks
    with api._stack_lock:
        stack = list(api.conversation_stacks.get(root_id) or ())
        pending = [(h, m) for h, m in marks.runs.items() if m.root_id == root_id]
    if not pending:
        return
    on_stack = {frame.conversation_id for frame in stack}
    frame_index = max(len(stack) - 1, 0)
    log = getattr(api, "_event_log", None)
    for handle, mark in pending:
        if mark.frame_id not in on_stack:
            # popped or ended: nothing will ever be final here, but the run was started and
            # logged, so its end is owed (D-056: the run is told to stop by `pop_fsm`)
            with api._stack_lock:
                marks.runs.pop(handle, None)
            _append_run_end(log, root_id, frame_index, handle, mark, *_ORPHAN_END)
            continue
        peek = api.fsm_manager.peek_turn(mark.frame_id)
        if peek is None or peek[1]:  # gone, or a turn is in flight: judge after it
            continue
        record = (api.fsm_manager.copy_raw_context_data(mark.frame_id) or {}).get(
            mark.result_key
        )
        if not isinstance(record, dict) or not record_belongs(record, mark.state_id):
            continue
        status = record.get("status")
        if status in (None, Status.RUNNING.value):
            continue
        _append_run_end(
            log,
            root_id,
            frame_index,
            handle,
            mark,
            status,
            record.get("reason"),
            record.get("template_identity"),
            record.get("instance_identity"),
        )
        with api._stack_lock:
            marks.runs.pop(handle, None)
        # the record landed: the host's copy of the result is not needed any more (D-057)
        _forget_result(api, handle, mark.frame_id)


async def wait_for_run(api: _ApiSurface, root_id: str) -> None:
    """The one bounded wait at the end of ``aadvance`` and of a step stream (module docstring)."""
    launcher = api._targets
    wait = getattr(launcher, "wait", None)
    if not is_run_launcher(launcher) or not callable(wait):
        return
    with api._stack_lock:
        stack = api.conversation_stacks.get(root_id)
        top = stack[-1].conversation_id if stack else None
        handles = [h for h, m in api._call_marks.runs.items() if m.frame_id == top]
    if top is None or not handles:
        return
    try:
        polled = launcher.poll(handles[0], conversation_id=top)
        if polled is None or polled.finished:
            return
        await aoffload(wait, handles[0], DELEGATED_WAIT_SECONDS)
    except Exception as error:
        # the step is already committed: a failing launcher must not turn it into an error. A
        # cancel (`CancelledError`, a `BaseException`) is NOT caught here (D-058)
        logger.warning(f"The wait for a delegated run failed: {type(error).__name__}")
