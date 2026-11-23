"""``settle``: what applies the record of a call state after the turn.

Layer ``engine`` (plan baa37f40, step 12, D-023, roadmap E3.2). The call executor
(``engine/kinds/call.py``) only DECIDES in Pass 1: it writes ``call_result = {status: running,
..., mode: frame, request}`` as handler-only turn data, which rolls back with the turn. Pushing
and popping a frame is the stack's, and the stack belongs to ``API``, so ``settle`` runs in the
turn entries (``turn_entries.py``) after the manager released the conversation: at the START
and the END of ``aconverse`` and ``aadvance``, at the START and at the NORMAL end of a stream
(never in a stream ``finally``: an abandoned stream defers to the next entry's start). The
executor also writes the request ON ARRIVAL (D-026), so the end settle of the arriving turn
pushes the child and the next message goes to it.

Interface contract:

- ``settle(api, root_id) -> Settled``: idempotent. Returns the greetings of the children it
  pushed (a ``list`` of text, empty when it pushed none; ``changed`` is true when a round
  pushed, popped or wrote a record); the caller joins them to the reply of its step (D-023
  item 9, D-026 (4): the START greetings before it, the END greetings after it). A stack with no frame whose definition holds a call state costs one list copy.
  Never blocks: it reads frames through ``FSMManager.peek_turn`` (no conversation lock) and
  does NOTHING while a turn of the frame is in flight. A second driver on the same root while
  one is settling is refused with ``FSMError`` (the root mark, released in ``finally``).
- Rule (a), the pop: the top frame T sits above a parent P whose current state is a call state
  with a ``running`` record of ``mode == frame`` (or T is in ``CallMarks.ending``): when T has
  ended (its state has no transition) or its own lifetime budget is spent, ``API.pop_fsm`` pops
  it with ONE merge of the declared ``returns`` and the final record. Rule (b), the push: T's
  state is a call state with such a record and nothing is above: the child is pushed.
- A kind other than ``call`` (``synthesize``, plan baa37f40 step 20, D-038) is applied through its
  executor: when ``StateExecutor.serves_requests`` is set settle asks it ``aplan_push`` (a ``PushPlan``,
  the final record of a refusal, or ``None``), ``arefusal``, ``on_child_turn`` and ``aplan_pop`` (the
  one merge), with plain values from ``engine/kinds/requests.py``; the stack, the marks, the push,
  the pop and every cancel rule stay here. ``StateExecutor.watches_outage`` takes the provider-error
  position at the push (``CallMarks.errors``). An executor that does not serve requests is left alone.
- Never trusts the record (D-023): the target, the inputs and the budgets are read from the
  PARENT frame's own definition and context. A forged public ``call_result`` cannot name a
  target the state does not declare.
- The pop merges ``config.returns`` when the call declares it (every entry required: a missing
  one is ``failed/return_keys_missing``); when it declares none, the output map the PUSHED target
  carries (``ResolvedTarget.returns``, a strategy's ``results``, D-030): every non-None child
  value under its parent key, a key the child never wrote simply absent, a key that is
  internal-prefixed, secret-looking or the call's own result key dropped with a WARNING. A
  target settle cannot prove after a restart (no mark, another definition now) merges nothing.
- Errors: ``FSMError`` for a second driver and for a pop the manager refuses (the next settle
  retries); an ``LLMResponseError`` of the child's greeting is the FINAL record
  ``failed/provider_error`` (the push rolled itself back; D-026 (6)); a ``CancelledError`` and
  any other ``BaseException`` propagate and leave the record ``running``, which the next entry
  re-requests. The END settle of a turn entry runs inside ``_settle_after`` (``turn_entries.py``):
  an ``Exception`` of it never loses the committed turn's reply (D-026 (5)).
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard

from fsm_llm.constants import (
    DEFAULT_MAX_STACK_DEPTH,
    has_internal_prefix,
    is_forbidden_context_entry,
)
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.definitions import (
    CallStateConfig,
    FSMError,
    LLMResponseError,
    State,
    fsm_id_of,
)
from fsm_llm.core.state_kinds import kind_of
from fsm_llm.engine.kinds import executor_for
from fsm_llm.engine.kinds.call import chosen_target, record_belongs
from fsm_llm.engine.kinds.requests import (
    ChildView,
    PopContext,
    PushContext,
    PushPlan,
)
from fsm_llm.engine.kinds.targets import TARGET_MODE_FRAME, ResolvedTarget
from fsm_llm.logging import logger
from fsm_llm.runtime.usage import watch_provider_errors

if TYPE_CHECKING:
    from .stack import FSMStackFrame
    from .surface import _ApiSurface

#: The state kinds whose records settle applies: ``call`` it judges itself, ``synthesize`` it
#: asks its executor about (``StateExecutor.serves_requests``, plan baa37f40 D-038).
_SETTLED_KINDS = frozenset({"call", "synthesize"})

#: A settle step is one push or one pop; nesting is bounded by the stack depth, so a loop that
#: does not end by then is a defect, not a long run.
_MAX_ROUNDS = 4 * DEFAULT_MAX_STACK_DEPTH


class Settled(list[str]):
    """The greetings ``settle`` collected (a ``list`` of text, empty when it pushed none).

    ``changed`` is true when a round pushed a child, popped one or wrote a record: the entry
    that called ``settle`` saves the session again only then (D-026).
    """

    changed: bool = False


@dataclass
class CallMarks:
    """What settle remembers in memory, guarded by ``API._stack_lock`` (never persisted).

    ``settling``: root ids with a settle in progress (the one-driver mark). ``ending``: ids of
    child frames whose returns are merged and whose end was refused, so the retry pops them
    with no second merge. ``pushed``: child frame id to the ``ResolvedTarget`` settle pushed,
    the source of the identities of the final record (D-027; never the public record).
    ``errors``: child frame id to the reader of the provider errors made since its push, taken
    only for an executor with ``watches_outage`` (D-038).
    """

    settling: set[str] = field(default_factory=set)
    ending: set[str] = field(default_factory=set)
    pushed: dict[str, ResolvedTarget] = field(default_factory=dict)
    errors: dict[str, Callable[[], int]] = field(default_factory=dict)

    def forget(self, frame_ids: Iterable[str]) -> None:
        """Drop what is kept for these child frames (the caller holds ``_stack_lock``)."""
        for frame_id in frame_ids:
            self.ending.discard(frame_id)
            self.pushed.pop(frame_id, None)
            self.errors.pop(frame_id, None)


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
async def settle(api: _ApiSurface, root_id: str) -> Settled:
    """Apply the call record at the top of ``root_id``'s stack (see the module docstring)."""
    settled = Settled()
    with api._stack_lock:
        frames = list(api.conversation_stacks.get(root_id) or ())
    if not any(_call_states(frame) for frame in frames):
        return settled
    marks = api._call_marks
    with api._stack_lock:
        if root_id in marks.settling:
            raise FSMError(
                f"Conversation {root_id} is being settled by another turn entry (a call "
                "state's child is being pushed or popped); retry after it finishes"
            )
        marks.settling.add(root_id)
    try:
        for _ in range(_MAX_ROUNDS):
            if not await _round(api, root_id, settled):
                break
            settled.changed = True
        return settled
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


def _is_unknown_mode(record: Any, state_id: str) -> bool:
    """A ``running`` record of this state whose ``mode`` is not ``frame`` (a handler, an older
    snapshot, a host of a mode this phase does not build)."""
    return (
        record_belongs(record, state_id)
        and record.get("status") == Status.RUNNING.value
        and record.get("mode") != TARGET_MODE_FRAME
    )


def _pushed_target(
    api: _ApiSurface,
    child: FSMStackFrame,
    parent: FSMStackFrame,
    target: str | None,
) -> ResolvedTarget | None:
    """What settle pushed as ``child`` (D-027): the object kept at the push, else the target
    derived AGAIN from the parent's own state and kept only when its definition is the one the
    child frame runs (``fsm_id`` equal). Never read from the record."""
    with api._stack_lock:
        kept = api._call_marks.pushed.get(child.conversation_id)
    if kept is not None or target is None or api._targets is None:
        return kept
    again = api._targets.resolve_target(target, conversation_id=parent.conversation_id)
    if again is not None and fsm_id_of(again.definition) == child.fsm_id:
        return again
    return None


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
    if state is None:
        return False
    is_call = kind_of(state).name == "call"
    config = state.call
    result_key = kind_of(state).result_key(state)
    if result_key is None or (is_call and config is None):
        return False
    executor = executor_for(state, api._executors)
    if not is_call and not executor.serves_requests:
        return False
    parent_data = api.fsm_manager.copy_raw_context_data(parent.conversation_id) or {}
    record = parent_data.get(result_key)
    with api._stack_lock:
        retry = top.conversation_id in api._call_marks.ending
    if not (retry or _is_request(record, state.id)):
        return False
    top_state = top.fsm_definition.states.get(top_state_id)
    ended = top_state is not None and not top_state.transitions
    # DECISION plan-2026-11-18T071726-5a6a1f80/D-012
    # The ENGINE bounds a child: the push gave its frame `max_steps` and the manager's record for
    # THAT frame says when it is spent (`turn_budget_spent`, a fact of the frame, never the text
    # of a `RunBudgetExceededError`, so a spent HOST budget still raises). Do NOT keep a count of
    # the child's steps in settle or in an executor (it missed every turn run outside the loop and
    # restarted with each resumed run), do NOT write an abort key or wait for an abort edge (only
    # a compiled FSM had one) and do NOT merge anything of an aborted child.
    spent = api.fsm_manager.turn_budget_spent(top.conversation_id)
    view: PopContext | None = None
    if not is_call and not retry:
        view = PopContext(
            root_id,
            parent.conversation_id,
            state,
            parent.fsm_definition,
            record if isinstance(record, dict) else {},
            _child_view(api, top, top_state_id, ended=ended, spent=spent),
        )
        if not (ended or spent is not None):
            await executor.on_child_turn(view)  # alive and inside its budget: the trace
            return False
    if not (retry or ended or spent is not None):
        return False  # the child is alive and inside its budget
    final: dict[str, Any] | None = None
    context_to_return: dict[str, Any] | None
    if retry:
        context_to_return = None  # the merge landed before the end was refused
    elif view is not None:
        planned = await executor.aplan_pop(view)
        if planned is None:
            return False  # not this executor's request: the frame is left as it is
        context_to_return = dict(planned)
        final = context_to_return.get(result_key)
    else:
        assert config is not None  # is_call
        context_to_return = _outcome(
            api,
            top,
            state.id,
            config,
            _pushed_target(api, top, parent, chosen_target(config, parent_data)),
            parent_data,
            ended=ended,
        )
        final = context_to_return[config.result_key]
    try:
        api.pop_fsm(root_id, context_to_return)
    except Exception:
        # the merge landed when the parent's record is final now: the retry must not merge again
        now = api.fsm_manager.copy_raw_context_data(parent.conversation_id) or {}
        landed = now.get(result_key)
        with api._stack_lock:
            # only a frame that is still on the stack can be retried: an end that raised after
            # the teardown removed it leaves nothing for a mark to point at (D-028 (4))
            on_stack = any(
                frame.conversation_id == top.conversation_id
                for frame in api.conversation_stacks.get(root_id) or ()
            )
            if (
                on_stack
                and isinstance(landed, dict)
                and not _is_request(landed, state.id)
            ):
                api._call_marks.ending.add(top.conversation_id)
            elif not on_stack:
                api._call_marks.forget([top.conversation_id])
        raise
    with api._stack_lock:
        api._call_marks.forget([top.conversation_id])
    if final is None:
        final = parent_data.get(result_key) or {}
    await executor.on_ended(
        root_id, parent.conversation_id, top.conversation_id, state, final
    )
    return True


def _child_view(
    api: _ApiSurface,
    child: FSMStackFrame,
    state_id: str,
    *,
    ended: bool,
    spent: str | None,
) -> ChildView:
    """The child frame as an executor may see it: copies and plain values only (D-038). The
    provider-error count is read here, once, from the reader taken at the push (``0`` for an
    executor that did not ask for one, or after a restart that lost it)."""
    with api._stack_lock:
        reader = api._call_marks.errors.get(child.conversation_id)
    errors = 0
    if reader is not None:
        try:
            errors = reader()
        except Exception:  # the reader falls back by itself; a fault here is no outage
            errors = 0
    return ChildView(
        conversation_id=child.conversation_id,
        fsm_id=child.fsm_id,
        state_id=state_id,
        data=api.fsm_manager.copy_raw_context_data(child.conversation_id) or {},
        ended=ended,
        spent=spent,
        steps=api.fsm_manager.turn_budget_steps(child.conversation_id),
        errors=errors,
    )


def _outcome(
    api: _ApiSurface,
    child: FSMStackFrame,
    state_id: str,
    config: CallStateConfig,
    pushed: ResolvedTarget | None,
    parent_data: dict[str, Any],
    *,
    ended: bool,
) -> dict[str, Any]:
    """What the pop merges into the parent: the declared returns (only on ``ok``) and the final
    record under the result key. The target is re-derived from the parent's own state and the
    identities are those of what was pushed (``pushed``, ``None`` when unproven), never the
    record's."""
    target = chosen_target(config, parent_data)
    template = None if pushed is None else pushed.template_identity
    instance = None if pushed is None else pushed.instance_identity

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
    if config.returns:
        for parent_key, child_key in config.returns.items():
            value = child_data.get(child_key)
            if value is None:
                return settled(Status.FAILED, Reason.RETURN_KEYS_MISSING)
            merged[parent_key] = value
    elif pushed is not None and pushed.returns:
        merged = _mapped_returns(state_id, config, pushed.returns, child_data)
    elif pushed is None:
        logger.warning(
            f"Call state '{state_id}' ended a child whose target cannot be proven (no mark "
            "and the registry answers another definition): its own output map is unknown, "
            "nothing is merged"
        )
    return {**merged, **settled(Status.OK, None)}


def _mapped_returns(
    state_id: str,
    config: CallStateConfig,
    mapping: Mapping[str, str],
    child_data: Mapping[str, Any],
) -> dict[str, Any]:
    """The tolerant merge of the target's own output map (D-030): every non-None child value
    under its parent key. A parent key that is internal-prefixed, a secret-looking entry or the
    call's own result key is dropped with one WARNING per key (the map is registry text, never
    an engine key); a key the child never wrote is simply absent."""
    merged: dict[str, Any] = {}
    for parent_key, child_key in mapping.items():
        value = child_data.get(child_key)
        if value is None:
            continue
        if (
            parent_key == config.result_key
            or has_internal_prefix(parent_key)
            or is_forbidden_context_entry(parent_key, value)
        ):
            logger.warning(
                f"Call state '{state_id}': the output map of the target names "
                f"'{parent_key}', which is never merged into a parent context"
            )
            continue
        merged[parent_key] = value
    return merged


async def _try_push(
    api: _ApiSurface,
    root_id: str,
    frames: list[FSMStackFrame],
    state_id: str,
    greetings: list[str],
) -> bool:
    top = frames[-1]
    state = _settled_state(top, state_id)
    if state is None:
        return False
    if kind_of(state).name != "call":
        return await _try_push_requested(api, root_id, frames, state, greetings)
    config = state.call
    if config is None:
        return False
    data = api.fsm_manager.copy_raw_context_data(top.conversation_id) or {}
    record = data.get(config.result_key)
    if not _is_request(record, state_id):
        if _is_unknown_mode(record, state_id):
            # no host of another mode exists in this phase: fail, never wait forever (D-028 (2))
            target = chosen_target(config, data)
            await _refuse(
                api, root_id, state_id, config, Reason.UNKNOWN_TARGET, target, None
            )
            return True
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

    async def refuse(reason: Reason) -> None:
        await _refuse(api, root_id, state_id, config, reason, target, resolved)

    plan = PushPlan(resolved.definition, inputs, config.max_steps, config.max_seconds)
    return await _push_plan(
        api, root_id, frames, state, record, plan, resolved, refuse, greetings
    )


async def _try_push_requested(
    api: _ApiSurface,
    root_id: str,
    frames: list[FSMStackFrame],
    state: State,
    greetings: list[str],
) -> bool:
    """The push of a kind settle does not judge itself: its executor plans it (D-038)."""
    top = frames[-1]
    executor = executor_for(state, api._executors)
    result_key = kind_of(state).result_key(state)
    if not executor.serves_requests or result_key is None:
        return False
    data = api.fsm_manager.copy_raw_context_data(top.conversation_id) or {}
    record = data.get(result_key)
    unknown_mode = _is_unknown_mode(record, state.id)
    if not (unknown_mode or _is_request(record, state.id)) or not isinstance(
        record, dict
    ):
        return False
    ctx = PushContext(
        root_id, top.conversation_id, state, top.fsm_definition, data, record
    )

    async def refuse(reason: Reason) -> None:
        final = await executor.arefusal(ctx, reason)
        api.update_context(root_id, {result_key: dict(final)})

    if unknown_mode:  # no host of another mode exists: fail, never wait forever
        await refuse(Reason.UNKNOWN_TARGET)
        return True
    plan = await executor.aplan_push(ctx)
    if plan is None:
        return False  # not this executor's request: left as it is
    if not isinstance(plan, PushPlan):  # the FINAL record of a refusal it decided
        api.update_context(root_id, {result_key: dict(plan)})
        return True
    return await _push_plan(
        api, root_id, frames, state, record, plan, None, refuse, greetings
    )


async def _push_plan(
    api: _ApiSurface,
    root_id: str,
    frames: list[FSMStackFrame],
    state: State,
    record: Mapping[str, Any],
    plan: PushPlan,
    resolved: ResolvedTarget | None,
    refuse: Callable[[Reason], Awaitable[None]],
    greetings: list[str],
) -> bool:
    """The push both kinds share: the stack bound, ``apush_fsm`` with ONLY ``plan.inputs`` and
    ``inherit_context=False``, the marks, ``on_pushed``. Always ``True`` (the stack or the
    record changed); a refusal is written by ``refuse``."""
    top = frames[-1]
    if len(frames) >= DEFAULT_MAX_STACK_DEPTH:
        await refuse(Reason.DEPTH_REFUSED)
        return True
    try:
        greeting = await api.apush_fsm(
            root_id,
            plan.definition,
            context_to_pass=dict(plan.inputs),
            inherit_context=False,
            max_steps=plan.max_steps,
            max_seconds=plan.max_seconds,
        )
    except LLMResponseError:
        # the push rolled itself back (D-020): an outage of the greeting is a final `failed`
        # record, so the failure edge fires on the next turn (D-026 (6)); it is not asked again
        await refuse(Reason.PROVIDER_ERROR)
        return True
    except (FSMError, ValueError) as error:
        logger.warning(
            f"State '{state.id}' could not push its child: "
            f"{type(error).__name__}: {error}"
        )
        await refuse(Reason.PUSH_FAILED)
        return True
    if greeting:
        greetings.append(greeting)
    executor = executor_for(state, api._executors)
    with api._stack_lock:
        child_id = api.conversation_stacks[root_id][-1].conversation_id
        depth = len(api.conversation_stacks[root_id])
        if resolved is not None:
            api._call_marks.pushed[child_id] = resolved
    if executor.watches_outage:
        # DECISION plan-2026-11-18T071726-5a6a1f80/D-015
        # The outage is a POSITION taken at the push (the log, then the ledger, then the meter,
        # then 0: `runtime.usage.watch_provider_errors`), not a before and after of
        # `usage().errors`. Do NOT go back to comparing that count alone (a `reset_usage()`
        # mid-run and an interface with no meter lost the outage), do NOT add the ledger to the
        # log (one source answers) and do NOT take the position at the pop (the child has
        # already run by then). See decisions.md D-015 of plan 5a6a1f80.
        try:
            reader = watch_provider_errors(
                api.llm_interface, api._event_log, root_id, depth - 1
            )
        except Exception:  # a source that cannot start answers 0
            reader = None
        if reader is not None:
            with api._stack_lock:
                api._call_marks.errors[child_id] = reader
    await executor.on_pushed(root_id, top.conversation_id, child_id, state, record)
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
