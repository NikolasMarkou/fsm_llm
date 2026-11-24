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
  hook the launcher calls right before it starts the thread. A ``BaseException`` propagates and
  leaves the record ``running`` with its reservation: the next entry starts it.
- ``flush_run_ends(api, root_id)``: appends ``run/end`` (status, reason, target, identities of the
  committed FINAL record) for each started run of this root whose frame is no longer in flight and
  whose record is final, once, then forgets the mark. Written on the calling (engine) thread only.
- ``wait_for_run(api, root_id)``: the ONE bounded wait, at the end of ``aadvance``: when the top
  frame has a started, unfinished run and the launcher has a ``wait`` method, wait at most
  ``DELEGATED_WAIT_SECONDS`` for it to finish, on a worker thread (``aoffload``), holding no lock.
- ``RunMark``: what settle remembers of a started run (``CallMarks.runs``, keyed by handle).
- ``DELEGATED_WAIT_SECONDS``: the bound of that wait (owner-visible: a run loop over a delegated
  parent polls once per wait, so its ``max_steps`` must cover the seconds of the run over this).
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fsm_llm.aio import aoffload
from fsm_llm.core.codes import MessageSource, Reason, Status
from fsm_llm.core.definitions import CallStateConfig, State
from fsm_llm.engine.kinds.call import chosen_target, record_belongs
from fsm_llm.engine.kinds.targets import (
    DelegatedTarget,
    RunLauncher,
    delegated_seconds,
)
from fsm_llm.logging import logger
from fsm_llm.runtime import EventType, RedactionKind, filter_context, warn_on_failure

if TYPE_CHECKING:
    from .stack import FSMStackFrame
    from .surface import _ApiSurface

__all__ = [
    "DELEGATED_WAIT_SECONDS",
    "RunMark",
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
# turn, with the manager and the stack lock released).
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
    if not isinstance(launcher, RunLauncher):
        return await end(Reason.RUN_LOST, None)  # nothing here can know the handle
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
    if delegated is None:
        return await end(Reason.UNKNOWN_TARGET, None)
    assert target is not None  # resolve_run answered, so a target was named
    inputs: dict[str, Any] = {}
    for name in config.inputs:
        value = data.get(name)
        if value is None:
            return await end(Reason.INPUTS_MISSING, delegated)
        inputs[name] = copy.deepcopy(value)
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
    if launcher.poll(handle, conversation_id=top.conversation_id) is not None:
        return False  # already running or finished: nothing to start
    return await end(Reason.RUN_LOST, delegated)


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
            root_id, top.conversation_id, state.id, config.result_key, target
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
        if (
            mark.frame_id not in on_stack
        ):  # popped or ended: nothing will ever be final here
            with api._stack_lock:
                marks.runs.pop(handle, None)
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
        if log is not None:
            with warn_on_failure(EventType.RUN_END):
                log.append(
                    root_id,
                    EventType.RUN_END,
                    frame=frame_index,
                    handle=handle,
                    target=mark.target,
                    status=status,
                    reason=record.get("reason"),
                    template_identity=record.get("template_identity"),
                    instance_identity=record.get("instance_identity"),
                )
        with api._stack_lock:
            marks.runs.pop(handle, None)


async def wait_for_run(api: _ApiSurface, root_id: str) -> None:
    """The one bounded wait at the end of ``aadvance`` (see the module docstring)."""
    launcher = api._targets
    wait = getattr(launcher, "wait", None)
    if not isinstance(launcher, RunLauncher) or not callable(wait):
        return
    with api._stack_lock:
        stack = api.conversation_stacks.get(root_id)
        top = stack[-1].conversation_id if stack else None
        handles = [h for h, m in api._call_marks.runs.items() if m.frame_id == top]
    if top is None or not handles:
        return
    polled = launcher.poll(handles[0], conversation_id=top)
    if polled is None or polled.finished:
        return
    await aoffload(wait, handles[0], DELEGATED_WAIT_SECONDS)
