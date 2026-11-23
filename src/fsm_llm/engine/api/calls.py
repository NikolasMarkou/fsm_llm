"""``settle``: what applies the record of a call state after the turn.

Layer ``engine`` (plan baa37f40, step 12, D-023, roadmap E3.2). The call executor
(``engine/kinds/call.py``) only DECIDES in Pass 1: it writes ``call_result = {status: running,
..., mode: frame, request}`` as handler-only turn data, which rolls back with the turn. Pushing
and popping a frame is the stack's, and the stack belongs to ``API``, so ``settle`` runs in the
turn entries (``turn_entries.py``) after the manager released the conversation: at the START
and the END of ``aconverse`` and ``aadvance`` and at the NORMAL end of a stream (never in a
stream ``finally``: an abandoned stream defers to the next entry's start).

Interface contract:

- ``settle(api, root_id) -> list[str]``: idempotent. Returns the greetings of the children it
  pushed (empty when it pushed none); the caller joins them to the reply of its step (D-023
  item 9). A stack with no frame whose definition holds a call state costs one list copy.
  Never blocks: it reads frames through ``FSMManager.peek_turn`` (no conversation lock) and
  does NOTHING while a turn of the frame is in flight. A second driver on the same root while
  one is settling is refused with ``FSMError`` (the root mark, released in ``finally``).
- Rule (a), the pop: the top frame T sits above a parent P whose current state is a call state
  with a ``running`` record of ``mode == frame`` (or T is in ``CallMarks.ending``): when T has
  ended (its state has no transition) or its own lifetime budget is spent, ``API.pop_fsm`` pops
  it with ONE merge of the declared ``returns`` and the final record. Rule (b), the push: T's
  state is a call state with such a record and nothing is above: the child is pushed.
- Never trusts the record (D-023): the target, the inputs and the budgets are read from the
  PARENT frame's own definition and context. A forged public ``call_result`` cannot name a
  target the state does not declare.
- Errors: ``FSMError`` for a second driver and for a pop the manager refuses (the next settle
  retries); an ``LLMResponseError`` of the child's greeting, a ``CancelledError`` and any other
  ``BaseException`` propagate and leave the record ``running``, which the next entry re-requests.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard

from fsm_llm.constants import DEFAULT_MAX_STACK_DEPTH
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.definitions import CallStateConfig, FSMError, LLMResponseError, State
from fsm_llm.core.state_kinds import kind_of
from fsm_llm.engine.kinds import executor_for
from fsm_llm.engine.kinds.call import chosen_target, record_belongs
from fsm_llm.engine.kinds.targets import TARGET_MODE_FRAME, ResolvedTarget
from fsm_llm.logging import logger

if TYPE_CHECKING:
    from .stack import FSMStackFrame
    from .surface import _ApiSurface

#: The state kinds whose records settle applies (step 20 adds ``synthesize``).
_SETTLED_KINDS = frozenset({"call"})

#: A settle step is one push or one pop; nesting is bounded by the stack depth, so a loop that
#: does not end by then is a defect, not a long run.
_MAX_ROUNDS = 4 * DEFAULT_MAX_STACK_DEPTH


@dataclass
class CallMarks:
    """What settle remembers in memory, guarded by ``API._stack_lock`` (never persisted).

    ``settling``: root ids with a settle in progress (the one-driver mark). ``ending``: ids of
    child frames whose returns are merged and whose end was refused, so the retry pops them
    with no second merge.
    """

    settling: set[str] = field(default_factory=set)
    ending: set[str] = field(default_factory=set)


# DECISION plan-2026-11-23T122249-baa37f40/D-023
# `settle` applies a call record after the turn and supersedes 5a6a1f80/D-010 for call frames
# ONLY (the engine pops THIS kind of frame at the child's end or its budget stop; it pops no
# other). Do NOT take the target, the inputs or the budgets from the record or its `request`:
# a forged public `call_result` (a snapshot, a log, a handler) would push a target the state
# does not declare or hand the child a grant; read them from the PARENT's own definition and
# context. Do NOT give the child `inherit_context=True` or any key but the declared inputs. Do
# NOT call a manager method that takes a conversation lock while a turn holds it (settle runs
# after the gate was released, reads through `peek_turn`, and does nothing while a turn is in
# flight), do NOT hold `_stack_lock` across an await or a manager call, do NOT release the root
# mark outside `finally`, and do NOT catch `BaseException` or `CancelledError` anywhere here (a
# cancel leaves the record `running` and the next entry re-requests it). Do NOT merge twice: the
# pop writes the returns and the final record in ONE `pop_fsm` merge, and a retry of a refused
# end (`ending`) pops with no merge. Do NOT run it in a stream `finally`. See D-023.
async def settle(api: _ApiSurface, root_id: str) -> list[str]:
    """Apply the call record at the top of ``root_id``'s stack (see the module docstring)."""
    with api._stack_lock:
        frames = list(api.conversation_stacks.get(root_id) or ())
    if not any(_call_states(frame) for frame in frames):
        return []
    marks = api._call_marks
    with api._stack_lock:
        if root_id in marks.settling:
            raise FSMError(
                f"Conversation {root_id} is being settled by another turn entry (a call "
                "state's child is being pushed or popped); retry after it finishes"
            )
        marks.settling.add(root_id)
    try:
        greetings: list[str] = []
        for _ in range(_MAX_ROUNDS):
            if not await _round(api, root_id, greetings):
                break
        return greetings
    finally:
        with api._stack_lock:
            marks.settling.discard(root_id)


def _call_states(frame: FSMStackFrame) -> list[State]:
    """The states of the frame's definition whose kind settle applies."""
    return [
        state
        for state in frame.fsm_definition.states.values()
        if kind_of(state).name in _SETTLED_KINDS
    ]


def _settled_state(frame: FSMStackFrame, state_id: str) -> State | None:
    state = frame.fsm_definition.states.get(state_id)
    if state is not None and kind_of(state).name in _SETTLED_KINDS:
        return state
    return None


def _is_request(record: Any, state_id: str) -> TypeGuard[dict[str, Any]]:
    """A ``running`` record of frame mode that belongs to the call state ``state_id`` (D-025):
    the only record settle applies. A record another call state wrote is never applied."""
    return (
        record_belongs(record, state_id)
        and record.get("status") == Status.RUNNING.value
        and record.get("mode") == TARGET_MODE_FRAME
    )


def _identity(record: Any, name: str) -> str | None:
    """An identity of the running record, kept only when it is text (a label, never trusted)."""
    value = record.get(name) if isinstance(record, dict) else None
    return value if isinstance(value, str) else None


def _final(
    state_id: str,
    status: Status,
    reason: Reason | None,
    target: str | None,
    template_identity: str | None,
    instance_identity: str | None,
) -> dict[str, Any]:
    """The final record: the projection the loader's sample results pin (``CALL_SAMPLES``)."""
    return {
        "status": status.value,
        "state": state_id,
        "reason": None if reason is None else reason.value,
        "target": target,
        "template_identity": template_identity,
        "instance_identity": instance_identity,
    }


async def _round(api: _ApiSurface, root_id: str, greetings: list[str]) -> bool:
    """One pop or one push; ``True`` when the stack or the record changed."""
    with api._stack_lock:
        frames = list(api.conversation_stacks.get(root_id) or ())
    if not frames:
        return False
    top = frames[-1]
    peek = api.fsm_manager.peek_turn(top.conversation_id)
    if peek is None or peek[1]:  # gone, or a turn is in flight: settle does not act
        return False
    if len(frames) >= 2 and await _try_pop(api, root_id, frames, peek[0]):
        return True
    return await _try_push(api, root_id, frames, peek[0], greetings)


async def _try_pop(
    api: _ApiSurface, root_id: str, frames: list[FSMStackFrame], top_state_id: str
) -> bool:
    top, parent = frames[-1], frames[-2]
    parent_peek = api.fsm_manager.peek_turn(parent.conversation_id)
    if parent_peek is None:
        return False
    state = _settled_state(parent, parent_peek[0])
    config = None if state is None else state.call
    if state is None or config is None:
        return False
    parent_data = api.fsm_manager.copy_raw_context_data(parent.conversation_id) or {}
    record = parent_data.get(config.result_key)
    with api._stack_lock:
        retry = top.conversation_id in api._call_marks.ending
    if not (retry or _is_request(record, state.id)):
        return False
    top_state = top.fsm_definition.states.get(top_state_id)
    ended = top_state is not None and not top_state.transitions
    spent = api.fsm_manager.turn_budget_spent(top.conversation_id)
    if not (retry or ended or spent is not None):
        return False  # the child is alive and inside its budget
    final: dict[str, Any] | None = None
    if retry:
        context_to_return = None  # the merge landed before the end was refused
    else:
        context_to_return = _outcome(
            api, top, state.id, config, record, parent_data, ended=ended
        )
        final = context_to_return[config.result_key]
    try:
        api.pop_fsm(root_id, context_to_return)
    except Exception:
        # the merge landed when the parent's record is final now: the retry must not merge again
        now = api.fsm_manager.copy_raw_context_data(parent.conversation_id) or {}
        landed = now.get(config.result_key)
        if isinstance(landed, dict) and not _is_request(landed, state.id):
            with api._stack_lock:
                api._call_marks.ending.add(top.conversation_id)
        raise
    with api._stack_lock:
        api._call_marks.ending.discard(top.conversation_id)
    if final is None:
        final = parent_data.get(config.result_key) or {}
    await executor_for(state, api._executors).on_ended(
        root_id, parent.conversation_id, top.conversation_id, state, final
    )
    return True


def _outcome(
    api: _ApiSurface,
    child: FSMStackFrame,
    state_id: str,
    config: CallStateConfig,
    record: Any,
    parent_data: dict[str, Any],
    *,
    ended: bool,
) -> dict[str, Any]:
    """What the pop merges into the parent: the declared returns (only on ``ok``) and the final
    record under the result key. The target is re-derived from the parent's own state."""
    target = chosen_target(config, parent_data)
    template = _identity(record, "template_identity")
    instance = _identity(record, "instance_identity")

    def settled(status: Status, reason: Reason | None) -> dict[str, Any]:
        return {
            config.result_key: _final(
                state_id, status, reason, target, template, instance
            )
        }

    if not ended:  # the budget of the frame is spent: partial results are dropped
        return settled(Status.ABORTED, Reason.BUDGET_SPENT)
    child_data = api.fsm_manager.copy_raw_context_data(child.conversation_id) or {}
    merged: dict[str, Any] = {}
    for parent_key, child_key in config.returns.items():
        value = child_data.get(child_key)
        if value is None:
            return settled(Status.FAILED, Reason.RETURN_KEYS_MISSING)
        merged[parent_key] = value
    return {**merged, **settled(Status.OK, None)}


async def _try_push(
    api: _ApiSurface,
    root_id: str,
    frames: list[FSMStackFrame],
    state_id: str,
    greetings: list[str],
) -> bool:
    top = frames[-1]
    state = _settled_state(top, state_id)
    config = None if state is None else state.call
    if state is None or config is None:
        return False
    data = api.fsm_manager.copy_raw_context_data(top.conversation_id) or {}
    record = data.get(config.result_key)
    if not _is_request(record, state_id):
        return False
    # Everything below comes from the parent's own definition and context, never the record.
    target = chosen_target(config, data)
    if target is None:
        await _refuse(api, root_id, state_id, config, Reason.UNKNOWN_TARGET, None, None)
        return True
    resolver = api._targets
    resolved: ResolvedTarget | None = (
        None
        if resolver is None
        else resolver.resolve_target(target, conversation_id=top.conversation_id)
    )
    if resolved is None:
        await _refuse(
            api, root_id, state_id, config, Reason.UNKNOWN_TARGET, target, None
        )
        return True
    inputs: dict[str, Any] = {}
    for name in config.inputs:
        value = data.get(name)
        if value is None:
            await _refuse(
                api, root_id, state_id, config, Reason.INPUTS_MISSING, target, resolved
            )
            return True
        inputs[name] = copy.deepcopy(value)
    if len(frames) >= DEFAULT_MAX_STACK_DEPTH:
        await _refuse(
            api, root_id, state_id, config, Reason.DEPTH_REFUSED, target, resolved
        )
        return True
    try:
        greeting = await api.apush_fsm(
            root_id,
            resolved.definition,
            context_to_pass=inputs,
            inherit_context=False,
            max_steps=config.max_steps,
            max_seconds=config.max_seconds,
        )
    except LLMResponseError:
        raise  # an outage is not a refusal: the record stays `running` and is asked again
    except (FSMError, ValueError) as error:
        logger.warning(
            f"Call state '{state_id}' could not push its target: "
            f"{type(error).__name__}: {error}"
        )
        await _refuse(
            api, root_id, state_id, config, Reason.PUSH_FAILED, target, resolved
        )
        return True
    if greeting:
        greetings.append(greeting)
    with api._stack_lock:
        child_id = api.conversation_stacks[root_id][-1].conversation_id
    await executor_for(state, api._executors).on_pushed(
        root_id, top.conversation_id, child_id, state, record
    )
    return True


async def _refuse(
    api: _ApiSurface,
    root_id: str,
    state_id: str,
    config: CallStateConfig,
    reason: Reason,
    target: str | None,
    resolved: ResolvedTarget | None,
) -> None:
    """Write the FINAL ``failed`` record of a call settle cannot run (through ``update_context``,
    so the log holds it like any context write); the failure edge fires on the next turn."""
    final = _final(
        state_id,
        Status.FAILED,
        reason,
        target,
        None if resolved is None else resolved.template_identity,
        None if resolved is None else resolved.instance_identity,
    )
    api.update_context(root_id, {config.result_key: final})
