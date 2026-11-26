"""Bounded message-free runs of ``API`` (``RunMixin``): ``run_until_terminal`` and its stream, the
budgets, the ``before_step`` hook rounds and the exempt-state check.

Layer ``engine`` (plan d454fd62, step 81): split out of the old ``fsm_llm.api``; the function and
method bodies are unchanged. The helpers that took the class (``api: _ApiSurface``) take ``_ApiSurface``
(see ``helpers.py``). The D-027 anchor below says why the one round sequence lives here.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Collection,
    Iterator,
)
from contextlib import aclosing, contextmanager

from fsm_llm.aio import SYNC, acall, acall_stream, iterate_sync, run_sync
from fsm_llm.core.definitions import (
    AdvanceResult,
    FSMDefinition,
    FSMError,
    RunBudgetExceededError,
    checked_key_names,
)
from fsm_llm.engine.fsm.budget import _check_run_budgets

from .calls import settle
from .helpers import context_run
from .surface import _ApiSurface

#: ``before_step``: called with the number of the next step; an async hook may return an
#: awaitable, which the run awaits (in sync mode it must not really suspend).
_BeforeStep = Callable[[int], Awaitable[None] | None]


async def _await(awaitable: Awaitable[None]) -> None:
    await awaitable


async def _call_hook(before_step: _BeforeStep, step: int) -> None:
    """Call ``before_step(step)`` and await what it returns when that is awaitable.

    In sync mode the awaitable is stepped with ``run_sync``: one that really suspends raises
    ``SuspendedInSyncModeError`` here, inside the round generator (a bare ``await`` leaves the
    generator "running" and the caller gets a ``RuntimeError`` from its ``aclose``).
    """
    result = before_step(step)
    if not inspect.isawaitable(result):
        return
    if SYNC.get():
        run_sync(_await(result))
    else:
        await result


def _seconds_exempt_set(states: Collection[str]) -> frozenset[str]:
    """Validate ``seconds_exempt_states`` of a bounded run; return it as a set.

    Contract: ``states`` must be a collection of state ids (``str``), not a
    single ``str`` (whose characters would be taken as ids); ``ValueError``
    otherwise (``definitions.checked_key_names``). Returns a ``frozenset``
    (empty for ``()``).
    """
    return frozenset(checked_key_names(states, argument="seconds_exempt_states"))


# DECISION plan-2026-09-30T062855-07ad3f8c/D-027
# The ONE bounded-run sequence, shared by ``API.run_until_terminal`` and
# ``API.run_until_terminal_stream``: "has it ended", both budgets, the hook,
# then the caller runs exactly one step (``advance`` / ``advance_stream``) per
# yielded number. Do NOT write this sequence a second time in either method or
# in a subpackage (agents and reasoning pass their limits and
# their hook in). Do NOT move a budget check after the step: a spent budget
# must never start a step, so the seconds budget and "has it ended" are asked
# again after ``before_step`` returns (the hook may wait, or end the
# conversation). Do NOT decide "ended" from ``AdvanceResult.ended``:
# the stream step has no result (D-025), and a handler may push or pop an FSM
# during a step, so the top of the stack is asked again every round. Do NOT
# hold a lock or an open turn across the ``yield``: every step takes and
# releases the conversation itself. See decisions.md D-027.
async def _run_rounds(
    api: _ApiSurface,
    conversation_id: str,
    max_steps: int,
    max_seconds: float | None,
    before_step: _BeforeStep | None,
    seconds_exempt: frozenset[str] = frozenset(),
    context_hook: _BeforeStep | None = None,
) -> AsyncGenerator[int, None]:
    """Yield 1, 2, ... once for each step a bounded run is allowed to take.

    Contract: the caller runs exactly one step of ``conversation_id`` per
    yielded number and then asks for the next. Each round, in order: when
    the top of the FSM stack has ended, first let ``settle`` pop it when it is the
    child of a call state (D-024; the round then starts again on the new top),
    then stop (normal return) unless it is a
    pushed frame (stack depth above 1) and ``before_step(n)``, called once
    for it, pops it (``_hook_popped_ended_frame``), in which case the round
    starts again on the new top (that call is not a step and is not gated
    by the budgets); raise
    ``RunBudgetExceededError`` when ``max_seconds`` have passed since the first
    round began, or when ``max_steps`` steps were already taken (the seconds
    budget is reported when both are spent); call ``before_step(n)``, then ask
    "ended" and the seconds budget once more (the hook may have ended the
    conversation or used up the time); yield ``n``. Whatever ``before_step``
    raises propagates unchanged and no step runs. Budgets must already be
    valid (``_check_run_budgets``). A step that raises in the caller ends the
    run: the generator is simply not resumed. A conversation ID that was never
    started is not "ended" (``has_conversation_ended`` answers ``False`` for it),
    so the first round is yielded after ``before_step(1)`` and the caller's own
    first call on that ID raises ``ValueError``. A spent
    seconds budget does not stop a round whose current state (top of the
    stack) is in ``seconds_exempt``; the steps budget still applies there.
    """
    started = time.monotonic()
    steps_done = 0

    # DECISION plan-2026-10-01T093600-944e2692/D-032: the seconds budget is
    # waived only in the states the caller names (a run's wind-down: a
    # post-loop write or repair after its answer is in hand), so a finished
    # answer is never lost to the clock between its last steps. Do NOT waive
    # the steps budget there (it still bounds the run), do NOT decide the
    # exemption in a subpackage (an agent-side clock or loop is the parallel
    # path D-027 forbids), and do NOT read the state when no state is exempt.
    def check_seconds() -> None:
        if max_seconds is None or time.monotonic() - started < max_seconds:
            return
        if seconds_exempt and (
            api.get_current_state(conversation_id) in seconds_exempt
        ):
            return
        raise RunBudgetExceededError("seconds", max_seconds, steps_done)

    while True:
        # DECISION plan-2026-10-01T093600-944e2692/D-052: an ended PUSHED
        # frame is offered to ``before_step`` once (same number as the next
        # step) so the hook can pop it; the run goes on only when the top is
        # no longer ended or the stack got shallower, so a hook that does not
        # pop (or pops and re-pushes an ended frame) ends the run as before.
        # Do NOT count this call as a step or gate it on the budgets (it runs
        # no step; the budgets are checked right after, on the new top). Do
        # NOT add this loop in a subpackage (the reasoning engine's outer
        # pop-and-rerun loop is what D-013 replaced). See D-052.
        if api.has_conversation_ended(conversation_id):
            if await _settle_ended_call_child(api, conversation_id):
                continue
            if before_step is None or not await _hook_popped_ended_frame(
                api, conversation_id, before_step, steps_done + 1
            ):
                return
            continue
        check_seconds()
        if steps_done >= max_steps:
            raise RunBudgetExceededError("steps", max_steps, steps_done)
        # DECISION plan-2026-11-26T022222-7efd5cb4/D-013: the context hook runs BEFORE the caller's
        # at each STEP, not at the ended-frame offer above (a charge there counts a step that never
        # runs). Do NOT fold it into `before_step`.
        hooks = [h for h in (context_hook, before_step) if h is not None]
        for hook in hooks:
            await _call_hook(hook, steps_done + 1)
        if hooks:
            if api.has_conversation_ended(conversation_id):
                return
            check_seconds()
        yield steps_done + 1
        steps_done += 1


def _stack_depth(api: _ApiSurface, conversation_id: str) -> int:
    """Frames on the stack of ``conversation_id``; 0 once it is closed.

    Contract: reads ``api.conversation_stacks`` under ``api._stack_lock``
    (held for that read only); never raises and refreshes no idle timer.
    """
    with api._stack_lock:
        return len(api.conversation_stacks.get(conversation_id, ()))


# DECISION plan-2026-11-23T122249-baa37f40/D-024
# An ended CALL child on the stack is the engine's to pop (D-023), and `settle` runs only inside
# the turn entries, so a run loop that returns at once on an ended top leaves a call that never
# finishes (a resume after a death between the child's end and its pop, or a pop whose end the
# manager refused) while the conversation reads as ended. Do NOT drop this call so that "an ended
# frame is the hook's" (944e2692/D-052 holds for every frame that is not a call child), do NOT pop
# here by hand or write a second pop path (settle owns the single merge), do NOT count it as a
# step or gate it on the budgets, and do NOT make `has_conversation_ended` false for the frame.
async def _settle_ended_call_child(api: _ApiSurface, conversation_id: str) -> bool:
    """Let ``settle`` pop an ended call child; True when the stack got shallower.

    Contract: called by ``_run_rounds`` only when the top of the stack has ended. Returns False
    without a call when the depth is 1 or less. Otherwise awaits ``settle`` (a no-op unless the
    top is a child of a ``running`` call record; it can only pop here, so it drops no greeting)
    and answers whether the frame is gone. What ``settle`` raises propagates unchanged.
    """
    depth = _stack_depth(api, conversation_id)
    if depth <= 1:
        return False
    await settle(api, conversation_id)
    return _stack_depth(api, conversation_id) < depth


async def _hook_popped_ended_frame(
    api: _ApiSurface,
    conversation_id: str,
    before_step: _BeforeStep,
    step: int,
) -> bool:
    """Offer an ended pushed frame to ``before_step``; True if the run goes on.

    Contract: called by ``_run_rounds`` only when the top of the stack has
    ended. Returns False without calling the hook when the stack depth is 1
    or less (the root ended, or the conversation was closed). Otherwise calls
    ``before_step(step)`` once (what it raises propagates unchanged) and
    returns True when the top is no longer ended or the stack is shallower
    than before the call (the hook popped), False otherwise. So a hook that
    pushes a live FSM on the ended frame continues the run on that FSM, while
    one that pushes an FSM that has already ended (a terminal initial state),
    with or without a pop first, ends the run: a run never loops through
    hook calls without a step.
    """
    depth = _stack_depth(api, conversation_id)
    if depth <= 1:
        return False
    await _call_hook(before_step, step)
    if not api.has_conversation_ended(conversation_id):
        return True
    return _stack_depth(api, conversation_id) < depth


def _check_exempt_states(
    exempt: frozenset[str], fsm_defs: Collection[FSMDefinition]
) -> None:
    """Refuse ``seconds_exempt_states`` ids that are states of none of ``fsm_defs``.

    Contract: ``fsm_defs`` are the definitions the run can step first: the
    one at the top of the conversation's stack when the run is called, or,
    when that top is an ended pushed frame the hook may pop, every frame
    below it. Raises ``ValueError`` naming the unknown ids (sorted) and the
    definitions; returns ``None`` otherwise.
    """
    known = {state for fsm_def in fsm_defs for state in fsm_def.states}
    unknown = sorted(exempt - known)
    if unknown:
        names = ", ".join(f"'{fsm_def.name}'" for fsm_def in fsm_defs)
        raise ValueError(
            f"seconds_exempt_states {unknown} are not states of FSM {names}"
        )


@contextmanager
def _closed_conversation_ends_run(
    api: _ApiSurface, conversation_id: str
) -> Iterator[None]:
    """Let a step's error pass silently when the conversation was closed.

    Contract: wraps exactly one step of a bounded run. When the step raises
    because ``conversation_id`` was closed (``end_conversation`` from another
    thread between the round's "ended" question and the step), the error is
    swallowed and the next round of ``_run_rounds`` ends the run normally.
    Every other error, and any error for a conversation that is still live,
    propagates unchanged.
    """
    try:
        yield
    except (ValueError, KeyError, FSMError):
        if not (
            api._conversation_gone(conversation_id)
            and api._ended_cache_entry(conversation_id) is not None
        ):
            raise


class RunMixin(_ApiSurface):
    """The bounded run loops of one ``API``."""

    def run_until_terminal(
        self,
        conversation_id: str,
        *,
        max_steps: int,
        max_seconds: float | None = None,
        before_step: Callable[[int], None] | None = None,
        seconds_exempt_states: Collection[str] = (),
    ) -> tuple[AdvanceResult, ...]:
        """Run message-free steps (``advance``) until the conversation ends.

        Each round asks the top of the FSM stack whether it has ended (so an
        FSM pushed or popped during the run is followed), checks both budgets,
        calls ``before_step`` and then runs one ``advance``. Budgets are
        checked only between steps: a step that has started is never cut short.
        When the top that has ended is a pushed FSM (stack depth above 1),
        ``before_step`` is called once for it first: if the hook pops it (or
        pushes an FSM that has not ended), the run continues on the new top;
        if not, the run returns as for any ended top.

        Args:
            conversation_id: Existing conversation ID.
            max_steps: Most steps the run may take (at least 1).
            max_seconds: Wall-clock budget for the whole run, measured from
                this call; ``None`` for no time budget.
            before_step: Called before each step, and once when a pushed
                frame has ended, with the number of the next step (1, 2,
                ...). Before a step it runs after the budget checks; it may
                write context (``update_context``) and that step sees it.
                For an ended pushed frame (top of the stack terminal, depth
                above 1) it runs before the budget checks and may pop the
                frame (``pop_fsm``): the run then continues on the new top
                (the call is not a step; the same number is passed again
                before the next step). The run also continues when the hook
                pushes an FSM that has not ended (it runs on top of the
                ended frame); otherwise (no pop, or a pushed FSM whose
                initial state is terminal) the run returns.
                What it raises propagates unchanged and no step runs.
            seconds_exempt_states: State ids in which a spent ``max_seconds``
                does not stop the run (a wind-down that must finish once
                reached); ``max_steps`` still applies, so in an exempt state
                the overrun past ``max_seconds`` is bounded only by the steps
                left times one step's duration. Default: none. Each id must
                be a state of the definition at the top of the stack when
                the run is called (``ValueError`` otherwise; not checked for
                a conversation that has already ended, which runs nothing).
                When that top is an ended pushed frame and ``before_step``
                is given, each id must be a state of some frame below it
                (the hook may pop down to any of them).
                Each round matches the CURRENT top of the stack by state id
                only: when a handler pushes a child FSM during the run, a
                child state whose id is in the set is exempt too, and the
                parent's exempt states count again once the child is
                popped.

        Returns:
            The ``AdvanceResult`` of every step, in order; ``()`` when the
            conversation had already ended. A conversation closed during the
            run (``end_conversation`` from ``before_step`` or from another
            thread) ends the run normally with the results so far.

        Raises:
            ValueError: an invalid ``max_steps``, ``max_seconds`` (wrong
                type, ``max_steps < 1``, ``max_seconds <= 0``) or
                ``seconds_exempt_states`` (a ``str``, a non-``str`` member,
                or an id that is not a state of the running definition), or a
                conversation ID that was never started.
            RunBudgetExceededError: a budget was spent before the conversation
                ended. The steps already run are kept in the conversation, but
                the error does not carry their ``AdvanceResult``s: read the
                replies from ``get_conversation_history`` and the position
                from ``get_current_state`` / ``get_data``.
            FSMError: a step failed (that step is rolled back as a
                ``converse`` turn is, see ``advance``; earlier steps are
                kept), or a turn is already running for the conversation.
        """
        return run_sync(
            self.arun_until_terminal(
                conversation_id,
                max_steps=max_steps,
                max_seconds=max_seconds,
                before_step=before_step,
                seconds_exempt_states=seconds_exempt_states,
            )
        )

    async def arun_until_terminal(
        self,
        conversation_id: str,
        *,
        max_steps: int,
        max_seconds: float | None = None,
        before_step: Callable[[int], Awaitable[None] | None] | None = None,
        seconds_exempt_states: Collection[str] = (),
    ) -> tuple[AdvanceResult, ...]:
        """The async twin of ``run_until_terminal``: each step awaits ``aadvance``, and an
        async ``before_step`` (one that returns an awaitable) is awaited."""
        _check_run_budgets(max_steps, max_seconds)
        max_steps, max_seconds, hook = context_run(
            self._context, max_steps, max_seconds
        )
        exempt = self._checked_exempt_states(
            conversation_id, seconds_exempt_states, before_step
        )
        results: list[AdvanceResult] = []
        async with aclosing(
            _run_rounds(
                self, conversation_id, max_steps, max_seconds, before_step, exempt, hook
            )
        ) as rounds:
            async for _ in rounds:
                with _closed_conversation_ends_run(self, conversation_id):
                    results.append(await acall(self, "advance", conversation_id))
        return tuple(results)

    def run_until_terminal_stream(
        self,
        conversation_id: str,
        *,
        max_steps: int,
        max_seconds: float | None = None,
        before_step: Callable[[int], None] | None = None,
        seconds_exempt_states: Collection[str] = (),
    ) -> Iterator[str]:
        """Run message-free steps until the conversation ends, streaming the
        replies.

        The stream form of ``run_until_terminal``: the same rounds, budgets
        and ``before_step`` hook, with each step run by ``advance_stream``.
        Only reply text is yielded, so silent states contribute nothing.
        Arguments and the conversation ID are checked at CALL time; nothing
        else runs before the first ``next()``, and the wall-clock budget is
        measured from it. No lock is held between steps. Closing the stream
        early releases the conversation and keeps the steps already run and
        the partial reply (as ``advance_stream`` does).

        Args:
            conversation_id: Existing conversation ID.
            max_steps: Most steps the run may take (at least 1).
            max_seconds: Wall-clock budget for the whole run; ``None`` for no
                time budget.
            before_step: Called before each step, and once when a pushed
                frame has ended, as in ``run_until_terminal`` (a hook that
                pops the ended frame, or pushes an FSM that has not ended,
                lets the run continue on the new top; otherwise it returns).
            seconds_exempt_states: As in ``run_until_terminal`` (checked
                against the running definition at call time).

        Yields:
            String chunks of each speaking state's reply as they arrive.

        Raises:
            ValueError: at call time, for an invalid ``max_steps``,
                ``max_seconds`` or ``seconds_exempt_states``, or a
                conversation ID that was never started.
            RunBudgetExceededError: while iterating, when a budget was spent
                before the conversation ended.
            FSMError: while iterating, when a step failed (after its
                rollback) or a turn is already running for the conversation.

        A conversation closed during the run ends the stream normally, as in
        ``run_until_terminal``.
        """
        return iterate_sync(
            self.arun_until_terminal_stream(
                conversation_id,
                max_steps=max_steps,
                max_seconds=max_seconds,
                before_step=before_step,
                seconds_exempt_states=seconds_exempt_states,
            )
        )

    def arun_until_terminal_stream(
        self,
        conversation_id: str,
        *,
        max_steps: int,
        max_seconds: float | None = None,
        before_step: Callable[[int], Awaitable[None] | None] | None = None,
        seconds_exempt_states: Collection[str] = (),
    ) -> AsyncGenerator[str, None]:
        """The async twin of ``run_until_terminal_stream``: a plain function (arguments and
        id are checked now) that returns a lazy async generator, each step an
        ``aadvance_stream``."""
        _check_run_budgets(max_steps, max_seconds)
        max_steps, max_seconds, hook = context_run(
            self._context, max_steps, max_seconds
        )
        # Existence check at call time, holding no lock afterwards. An ended
        # conversation (terminal, or already closed and remembered) is valid
        # and streams nothing; an unknown ID raises ``ValueError`` here.
        exempt = self._checked_exempt_states(
            conversation_id, seconds_exempt_states, before_step
        )

        # Lazy nested closure, as in ``converse_stream`` (anchor
        # plan-2026-07-21T082818-4c63deac/D-002 there).
        rounds_args = (self, conversation_id, max_steps, max_seconds, before_step)

        async def _stream() -> AsyncGenerator[str, None]:
            async with aclosing(_run_rounds(*rounds_args, exempt, hook)) as rounds:
                async for _ in rounds:
                    with _closed_conversation_ends_run(self, conversation_id):
                        async with aclosing(
                            acall_stream(self, "advance_stream", conversation_id)
                        ) as chunks:
                            async for chunk in chunks:
                                yield chunk

        return _stream()

    def _checked_exempt_states(
        self,
        conversation_id: str,
        seconds_exempt_states: Collection[str],
        before_step: _BeforeStep | None,
    ) -> frozenset[str]:
        """Validate a run's ``seconds_exempt_states`` at call time.

        Contract: shape-checks the collection (``_seconds_exempt_set``); for a
        live conversation, checks every id against the definition at the top
        of its stack (``_check_exempt_states``). When that top has ended, the
        ids are checked only if it is a pushed frame and ``before_step`` is
        given (the hook may pop it, D-052), against the frames below it; an
        ended run that steps nothing is not checked. An unknown conversation
        ID raises ``ValueError`` (from ``_get_current_fsm_conversation_id``:
        ``has_conversation_ended`` answers ``False`` for it). Returns the
        frozen set.
        """
        exempt = _seconds_exempt_set(seconds_exempt_states)
        ended = self.has_conversation_ended(conversation_id)
        if ended and before_step is None:
            return exempt
        if not ended:
            # Raises ValueError for an unknown ID or an empty (corrupted) stack.
            self._get_current_fsm_conversation_id(conversation_id)
        if exempt:
            with self._stack_lock:
                stack = list(self.conversation_stacks.get(conversation_id, ()))
            frames = stack[:-1] if ended else stack[-1:]
            if frames:
                _check_exempt_states(exempt, [frame.fsm_definition for frame in frames])
        return exempt
