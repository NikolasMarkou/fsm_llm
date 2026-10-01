"""``API.run_until_terminal`` and ``run_until_terminal_stream``: bounded runs of
message-free steps.

Pinned here: both budgets are checked before each step, ``before_step`` runs
once before each step, the top of the FSM stack is asked again every round,
and the stream form is lazy and holds no lock between steps.
"""

from __future__ import annotations

import pickle
import time as _real_time
from typing import Any

import pytest

import fsm_llm
import fsm_llm.api as api_module
from fsm_llm import (
    API,
    FSMError,
    HandlerTiming,
    RunBudgetExceededError,
    create_handler,
)
from fsm_llm.definitions import FieldExtractionRequest, FieldExtractionResponse
from tests.test_fsm_llm.test_advance import (
    _CHUNKS,
    _REPLY,
    _always,
    _ScriptedLLM,
    _start,
    _state,
    _StreamingLLM,
    _trip_fsm,
    _when_set,
)


def _loop_fsm() -> dict[str, Any]:
    """ping <-> pong for ever; ``ping`` leaves for ``done`` once ``stop`` is set."""
    return {
        "name": "loop",
        "description": "A loop that ends only on request",
        "initial_state": "ping",
        "states": {
            "ping": _state(
                "ping",
                transitions=[_when_set("done", "stop", priority=50), _always("pong")],
            ),
            "pong": _state("pong", transitions=[_always("ping")]),
            "done": _state("done", speaks=True),
        },
    }


def _sub_fsm() -> dict[str, Any]:
    """sub_work (silent) -> sub_done (terminal, silent)."""
    return {
        "name": "sub",
        "description": "A sub task",
        "initial_state": "sub_work",
        "states": {
            "sub_work": _state("sub_work", transitions=[_always("sub_done")]),
            "sub_done": _state("sub_done"),
        },
    }


def _blocked_step_calls() -> int:
    """Field-extraction calls one BLOCKED step of the trip FSM makes (the
    first attempt plus the state's extraction retries), measured not assumed."""
    llm = _ScriptedLLM()
    api, conv_id = _start(_trip_fsm(), llm)
    api.advance(conv_id)
    return len(llm.field_requests)


def _path(results: tuple[fsm_llm.AdvanceResult, ...]) -> list[tuple[str, str]]:
    return [(r.state_before, r.state_after) for r in results]


class _FakeClock:
    """Stands in for ``fsm_llm.api``'s ``time`` binding only (never the global
    ``time`` module); ``monotonic`` moves only when a test advances it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __getattr__(self, name: str) -> Any:
        return getattr(_real_time, name)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(api_module, "time", fake)
    return fake


class _SlowLLM(_ScriptedLLM):
    """Every field extraction costs ``seconds`` on the fake clock."""

    def __init__(self, clock: _FakeClock, seconds: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._clock = clock
        self._seconds = seconds

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        self._clock.advance(self._seconds)
        return super().extract_field(request)


class _SlowStreamingLLM(_StreamingLLM):
    """``_StreamingLLM`` whose field extractions cost fake-clock seconds."""

    def __init__(self, clock: _FakeClock, seconds: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._clock = clock
        self._seconds = seconds

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        self._clock.advance(self._seconds)
        return super().extract_field(request)


# ---------------------------------------------------------------------------


class TestRunToTheEnd:
    def test_returns_every_step_result_in_order(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)

        results = api.run_until_terminal(conv_id, max_steps=5)

        assert _path(results) == [("collect", "plan"), ("plan", "done")]
        assert [r.ended for r in results] == [False, True]
        assert [r.response for r in results] == [None, _REPLY]
        assert isinstance(results, tuple)
        assert api.has_conversation_ended(conv_id)

    def test_budget_equal_to_the_steps_needed_is_enough(self):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))
        assert len(api.run_until_terminal(conv_id, max_steps=2)) == 2

    def test_history_holds_no_user_exchange_and_no_marker(self):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))
        before = api.get_conversation_history(conv_id)

        api.run_until_terminal(conv_id, max_steps=5)

        assert api.get_conversation_history(conv_id) == [*before, {"system": _REPLY}]


class TestRunBudget:
    def test_endless_loop_raises_after_exactly_max_steps(self):
        api, conv_id = _start(_loop_fsm(), _ScriptedLLM())
        seen: list[int] = []

        with pytest.raises(RunBudgetExceededError) as excinfo:
            api.run_until_terminal(conv_id, max_steps=5, before_step=seen.append)

        assert seen == [1, 2, 3, 4, 5]
        # Five steps from ``ping`` leave the loop in ``pong``: step six never ran.
        assert api.get_current_state(conv_id) == "pong"
        error = excinfo.value
        assert (error.budget, error.limit, error.steps_done) == ("steps", 5, 5)

    def test_error_is_an_fsm_error_with_details_and_pickles(self):
        error = RunBudgetExceededError("seconds", 2.5, 3)

        assert isinstance(error, FSMError)
        assert error.details == {"budget": "seconds", "limit": 2.5, "steps_done": 3}
        assert "seconds budget of 2.5" in str(error)
        assert "3 step(s)" in str(error)
        clone = pickle.loads(pickle.dumps(error))
        assert (clone.budget, clone.limit, clone.steps_done) == ("seconds", 2.5, 3)
        assert str(clone) == str(error)

    def test_exported_in_static_all(self):
        assert "RunBudgetExceededError" in fsm_llm.__all__
        assert fsm_llm.RunBudgetExceededError is RunBudgetExceededError

    def test_slow_step_trips_max_seconds_before_the_next_step(self, clock):
        llm = _SlowLLM(clock, 5.0)
        api, conv_id = _start(_trip_fsm(), llm)  # ``city`` never found: BLOCKED
        seen: list[int] = []

        with pytest.raises(RunBudgetExceededError) as excinfo:
            api.run_until_terminal(
                conv_id, max_steps=100, max_seconds=3.0, before_step=seen.append
            )

        error = excinfo.value
        assert (error.budget, error.limit, error.steps_done) == ("seconds", 3.0, 1)
        # The one step that started was not cut short (it used all its
        # extraction attempts); no second step began.
        assert seen == [1]
        assert len(llm.field_requests) == _blocked_step_calls()

    def test_time_left_lets_the_run_finish(self, clock):
        llm = _SlowLLM(clock, 1.0, fields={"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)

        results = api.run_until_terminal(conv_id, max_steps=10, max_seconds=30.0)

        assert _path(results) == [("collect", "plan"), ("plan", "done")]

    def test_seconds_budget_is_reported_when_both_are_spent(self, clock):
        llm = _SlowLLM(clock, 5.0)
        api, conv_id = _start(_trip_fsm(), llm)

        with pytest.raises(RunBudgetExceededError) as excinfo:
            api.run_until_terminal(conv_id, max_steps=1, max_seconds=3.0)

        assert excinfo.value.budget == "seconds"

    def test_run_that_ends_on_its_last_allowed_step_does_not_raise(self, clock):
        # The end of the conversation is checked before either budget.
        llm = _SlowLLM(clock, 50.0, fields={"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        api.advance(conv_id)
        assert api.get_current_state(conv_id) == "plan"

        results = api.run_until_terminal(conv_id, max_steps=1, max_seconds=1.0)

        assert _path(results) == [("plan", "done")]

    def test_permanently_blocked_state_ends_in_the_error(self):
        llm = _ScriptedLLM()  # ``city`` is never extracted
        api, conv_id = _start(_trip_fsm(), llm)
        llm.reset()

        with pytest.raises(RunBudgetExceededError) as excinfo:
            api.run_until_terminal(conv_id, max_steps=3)

        assert (excinfo.value.budget, excinfo.value.steps_done) == ("steps", 3)
        assert len(llm.field_requests) == 3 * _blocked_step_calls()
        assert api.get_current_state(conv_id) == "collect"

    def test_conversation_stays_usable_after_the_error(self):
        llm = _ScriptedLLM()
        api, conv_id = _start(_trip_fsm(), llm)
        with pytest.raises(RunBudgetExceededError):
            api.run_until_terminal(conv_id, max_steps=2)
        assert conv_id not in api.fsm_manager._active_turns

        llm.fields["city"] = "Paris"
        results = api.run_until_terminal(conv_id, max_steps=2)

        assert _path(results) == [("collect", "plan"), ("plan", "done")]

    def test_failed_step_propagates_and_keeps_the_earlier_steps(self):
        llm = _ScriptedLLM({"city": "Paris"}, fail_response=True)
        api, conv_id = _start(_trip_fsm(), llm)

        with pytest.raises(FSMError) as excinfo:
            api.run_until_terminal(conv_id, max_steps=5)

        assert not isinstance(excinfo.value, RunBudgetExceededError)
        # Step 1 is kept; the failed step 2 (Pass 2 of ``done``) is rolled back.
        assert api.get_current_state(conv_id) == "plan"
        assert conv_id not in api.fsm_manager._active_turns


class TestBeforeStep:
    def test_called_once_before_each_step_with_its_number(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        calls: list[tuple[int, str, int]] = []

        def _hook(step: int) -> None:
            calls.append(
                (step, api.get_current_state(conv_id), len(llm.field_requests))
            )

        api.run_until_terminal(conv_id, max_steps=5, before_step=_hook)

        # Step 1's hook ran before any LLM work of step 1.
        assert calls == [(1, "collect", 0), (2, "plan", 1)]

    def test_not_called_for_a_step_the_budget_refuses(self):
        api, conv_id = _start(_loop_fsm(), _ScriptedLLM())
        seen: list[int] = []

        with pytest.raises(RunBudgetExceededError):
            api.run_until_terminal(conv_id, max_steps=2, before_step=seen.append)

        assert seen == [1, 2]

    def test_not_called_when_the_conversation_has_ended(self):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))
        seen: list[int] = []

        api.run_until_terminal(conv_id, max_steps=9, before_step=seen.append)

        assert seen == [1, 2]

    def test_context_it_writes_is_seen_by_that_step(self):
        llm = _ScriptedLLM()  # the model never finds ``city``
        api, conv_id = _start(_trip_fsm(), llm)

        def _hook(step: int) -> None:
            if step == 1:
                api.update_context(conv_id, {"city": "Rome"})

        results = api.run_until_terminal(conv_id, max_steps=5, before_step=_hook)

        assert _path(results)[0] == ("collect", "plan")
        assert api.get_data(conv_id)["city"] == "Rome"

    def test_it_can_end_the_run_by_writing_context(self):
        api, conv_id = _start(_loop_fsm(), _ScriptedLLM())

        def _hook(step: int) -> None:
            if step == 3:
                api.update_context(conv_id, {"stop": True})

        results = api.run_until_terminal(conv_id, max_steps=50, before_step=_hook)

        assert _path(results) == [("ping", "pong"), ("pong", "ping"), ("ping", "done")]

    def test_its_exception_propagates_and_no_step_runs(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)

        class _Halt(Exception):
            pass

        def _hook(step: int) -> None:
            raise _Halt("stop here")

        with pytest.raises(_Halt, match="stop here"):
            api.run_until_terminal(conv_id, max_steps=5, before_step=_hook)

        assert llm.field_requests == []
        assert api.get_current_state(conv_id) == "collect"
        assert conv_id not in api.fsm_manager._active_turns
        lock = api.fsm_manager._conversation_locks[conv_id]
        assert lock.acquire(blocking=False)
        lock.release()
        # The conversation is still usable.
        results = api.run_until_terminal(conv_id, max_steps=5)
        assert _path(results) == [("collect", "plan"), ("plan", "done")]

    def test_its_exception_on_a_later_step_keeps_the_earlier_steps(self):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))

        def _hook(step: int) -> None:
            if step == 2:
                raise RuntimeError("second step refused")

        with pytest.raises(RuntimeError, match="second step refused"):
            api.run_until_terminal(conv_id, max_steps=5, before_step=_hook)

        assert api.get_current_state(conv_id) == "plan"
        assert conv_id not in api.fsm_manager._active_turns


class TestRunStack:
    def _api_with_push_and_pop_handlers(self) -> tuple[API, str, list[str]]:
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))
        events: list[str] = []

        def _push(context: dict[str, Any]) -> dict[str, Any]:
            events.append("push")
            api.push_fsm(conv_id, _sub_fsm())
            return {}

        def _pop(context: dict[str, Any]) -> dict[str, Any]:
            events.append("pop")
            api.pop_fsm(conv_id)
            return {}

        api.register_handler(
            create_handler("push_sub")
            .at(HandlerTiming.POST_TRANSITION)
            .on_target_state("plan")
            .do(_push)
        )
        api.register_handler(
            create_handler("pop_sub")
            .at(HandlerTiming.POST_TRANSITION)
            .on_target_state("sub_done")
            .do(_pop)
        )
        return api, conv_id, events

    def test_loop_follows_a_sub_fsm_pushed_and_popped_by_handlers(self):
        api, conv_id, events = self._api_with_push_and_pop_handlers()

        results = api.run_until_terminal(conv_id, max_steps=10)

        assert events == ["push", "pop"]
        assert _path(results) == [
            ("collect", "plan"),
            ("sub_work", "sub_done"),
            ("plan", "done"),
        ]
        # The sub-FSM's last step reports ``ended``; the run went on because
        # the top of the stack, asked again, was the parent.
        assert [r.ended for r in results] == [False, True, True]
        assert api.get_stack_depth(conv_id) == 1
        assert api.get_current_state(conv_id) == "done"

    def test_stream_loop_follows_the_stack_too(self):
        api, conv_id = _start(_trip_fsm(), _StreamingLLM({"city": "Paris"}))
        pushed: list[str] = []

        def _push(context: dict[str, Any]) -> dict[str, Any]:
            pushed.append(api.push_fsm(conv_id, _sub_fsm()))
            return {}

        api.register_handler(
            create_handler("push_sub")
            .at(HandlerTiming.POST_TRANSITION)
            .on_target_state("plan")
            .do(_push)
        )
        states: list[str] = []

        def _hook(step: int) -> None:
            states.append(api.get_current_state(conv_id))

        chunks = list(
            api.run_until_terminal_stream(conv_id, max_steps=10, before_step=_hook)
        )

        # Nobody pops: the hook is offered the ended sub-FSM once (D-052) and
        # the run ends when the FSM on top of the stack ends.
        assert states == ["collect", "sub_work", "sub_done"]
        assert chunks == []
        assert api.get_stack_depth(conv_id) == 2
        assert api.get_current_state(conv_id) == "sub_done"
        api.pop_fsm(conv_id)
        assert "".join(api.run_until_terminal_stream(conv_id, max_steps=10)) == "".join(
            _CHUNKS
        )

    def test_sub_fsm_pushed_by_before_step_is_run(self):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))

        def _hook(step: int) -> None:
            if step == 2:
                api.push_fsm(conv_id, _sub_fsm())

        results = api.run_until_terminal(conv_id, max_steps=10, before_step=_hook)

        assert _path(results) == [("collect", "plan"), ("sub_work", "sub_done")]


def _push_sub_on_entry(api: API, conv_id: str, target: str) -> None:
    """Register a handler that pushes ``_sub_fsm`` when the root enters
    ``target`` (nobody pops it on the sub's terminal)."""

    def _push(context: dict[str, Any]) -> dict[str, Any]:
        api.push_fsm(conv_id, _sub_fsm())
        return {}

    api.register_handler(
        create_handler(f"push_sub_on_{target}")
        .at(HandlerTiming.POST_TRANSITION)
        .on_target_state(target)
        .do(_push)
    )


class _PopEndedSub:
    """``before_step`` hook: records ``(n, state, depth)`` and pops the top
    frame when it is an ended pushed frame."""

    def __init__(self, api: API, conv_id: str, *, pops: bool = True) -> None:
        self.api = api
        self.conv_id = conv_id
        self.pops = pops
        self.calls: list[tuple[int, str, int]] = []

    def __call__(self, step: int) -> None:
        depth = self.api.get_stack_depth(self.conv_id)
        self.calls.append((step, self.api.get_current_state(self.conv_id), depth))
        if self.pops and depth > 1 and self.api.has_conversation_ended(self.conv_id):
            self.api.pop_fsm(self.conv_id)


class TestBeforeStepPopsEndedFrame:
    """D-052 of plan 944e2692: when the top of the stack is an ended pushed
    frame, the round offers it to ``before_step`` once; a pop lets the run go
    on to the parent's terminal. RED on the parent: the run returned at the
    sub-FSM's terminal without calling the hook."""

    def _api(self, target: str = "plan", llm: Any = None) -> tuple[API, str]:
        api, conv_id = _start(_trip_fsm(), llm or _ScriptedLLM({"city": "Paris"}))
        _push_sub_on_entry(api, conv_id, target)
        return api, conv_id

    def test_a_pop_lets_the_run_continue_to_the_root_terminal(self):
        api, conv_id = self._api()
        hook = _PopEndedSub(api, conv_id)

        results = api.run_until_terminal(conv_id, max_steps=10, before_step=hook)

        assert _path(results) == [
            ("collect", "plan"),
            ("sub_work", "sub_done"),
            ("plan", "done"),
        ]
        # The ended-frame call carries the next step's number, which the
        # step's own call repeats after the pop.
        assert hook.calls == [
            (1, "collect", 1),
            (2, "sub_work", 2),
            (3, "sub_done", 2),
            (3, "plan", 1),
        ]
        assert api.get_stack_depth(conv_id) == 1
        assert api.has_conversation_ended(conv_id)

    def test_the_pop_round_is_not_a_step(self):
        api, conv_id = self._api()

        results = api.run_until_terminal(
            conv_id, max_steps=3, before_step=_PopEndedSub(api, conv_id)
        )

        assert len(results) == 3
        assert api.get_current_state(conv_id) == "done"

    def test_a_spent_steps_budget_still_lets_the_hook_pop_then_raises(self):
        api, conv_id = self._api()
        hook = _PopEndedSub(api, conv_id)

        with pytest.raises(RunBudgetExceededError) as excinfo:
            api.run_until_terminal(conv_id, max_steps=2, before_step=hook)

        assert (excinfo.value.budget, excinfo.value.steps_done) == ("steps", 2)
        assert hook.calls[-1] == (3, "sub_done", 2)
        assert api.get_stack_depth(conv_id) == 1
        assert api.get_current_state(conv_id) == "plan"

    def test_a_hook_that_does_not_pop_returns_at_the_sub_terminal(self):
        api, conv_id = self._api()
        hook = _PopEndedSub(api, conv_id, pops=False)

        results = api.run_until_terminal(conv_id, max_steps=10, before_step=hook)

        assert _path(results) == [("collect", "plan"), ("sub_work", "sub_done")]
        assert hook.calls[-1] == (3, "sub_done", 2)
        assert len(hook.calls) == 3
        assert api.get_stack_depth(conv_id) == 2
        assert api.get_current_state(conv_id) == "sub_done"

    def test_without_a_hook_the_run_returns_at_the_sub_terminal(self):
        api, conv_id = self._api()

        results = api.run_until_terminal(conv_id, max_steps=10)

        assert _path(results) == [("collect", "plan"), ("sub_work", "sub_done")]
        assert api.get_stack_depth(conv_id) == 2

    def test_stream_form_pops_and_continues(self):
        api, conv_id = self._api(llm=_StreamingLLM({"city": "Paris"}))
        hook = _PopEndedSub(api, conv_id)

        chunks = list(
            api.run_until_terminal_stream(conv_id, max_steps=10, before_step=hook)
        )

        assert "".join(chunks) == "".join(_CHUNKS)
        assert [state for _, state, _ in hook.calls] == [
            "collect",
            "sub_work",
            "sub_done",
            "plan",
        ]
        assert api.get_stack_depth(conv_id) == 1
        assert api.has_conversation_ended(conv_id)

    def test_what_the_hook_raises_on_the_ended_frame_propagates(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = self._api(llm=llm)

        def _hook(step: int) -> None:
            if api.has_conversation_ended(conv_id):
                raise RuntimeError("merge failed")

        with pytest.raises(RuntimeError, match="merge failed"):
            api.run_until_terminal(conv_id, max_steps=10, before_step=_hook)

        assert api.get_stack_depth(conv_id) == 2
        assert api.get_current_state(conv_id) == "sub_done"
        assert conv_id not in api.fsm_manager._active_turns

    def test_a_pop_onto_an_ended_root_returns(self):
        # The sub-FSM is pushed as the root enters its terminal ``done``.
        api, conv_id = self._api(target="done")
        hook = _PopEndedSub(api, conv_id)

        results = api.run_until_terminal(conv_id, max_steps=10, before_step=hook)

        assert _path(results) == [
            ("collect", "plan"),
            ("plan", "done"),
            ("sub_work", "sub_done"),
        ]
        assert hook.calls[-1] == (4, "sub_done", 2)
        assert api.get_stack_depth(conv_id) == 1
        assert api.get_current_state(conv_id) == "done"

    def test_closing_the_conversation_from_the_hook_ends_the_run(self):
        api, conv_id = self._api()

        def _hook(step: int) -> None:
            if api.has_conversation_ended(conv_id):
                api.end_conversation(conv_id)

        results = api.run_until_terminal(conv_id, max_steps=10, before_step=_hook)

        assert _path(results) == [("collect", "plan"), ("sub_work", "sub_done")]

    def test_a_run_called_on_an_ended_sub_frame_checks_exempt_ids_below_it(self):
        api, conv_id = self._api()
        api.run_until_terminal(conv_id, max_steps=10)  # stops at sub_done
        assert api.get_stack_depth(conv_id) == 2

        with pytest.raises(ValueError, match="seconds_exempt_states"):
            api.run_until_terminal(
                conv_id,
                max_steps=10,
                before_step=_PopEndedSub(api, conv_id),
                seconds_exempt_states={"sub_work"},
            )
        # No hook: nothing would run, so nothing is checked.
        assert (
            api.run_until_terminal(
                conv_id, max_steps=10, seconds_exempt_states={"sub_work"}
            )
            == ()
        )

        results = api.run_until_terminal(
            conv_id,
            max_steps=10,
            before_step=_PopEndedSub(api, conv_id),
            seconds_exempt_states={"plan"},
        )

        assert _path(results) == [("plan", "done")]


def _leaf_fsm() -> dict[str, Any]:
    """One state, terminal from the start: it has ended as soon as pushed."""
    return {
        "name": "leaf",
        "description": "Ended on push",
        "initial_state": "leaf_done",
        "states": {"leaf_done": _state("leaf_done")},
    }


class _PushOnEndedFrame:
    """``before_step`` hook: on the FIRST ended-frame call only, optionally
    pops, then pushes ``fsm``; records ``(n, state, depth)`` of every call."""

    def __init__(
        self, api: API, conv_id: str, fsm: dict[str, Any], *, pop_first: bool
    ) -> None:
        self.api = api
        self.conv_id = conv_id
        self.fsm = fsm
        self.pop_first = pop_first
        self.pushed = False
        self.calls: list[tuple[int, str, int]] = []

    def __call__(self, step: int) -> None:
        depth = self.api.get_stack_depth(self.conv_id)
        self.calls.append((step, self.api.get_current_state(self.conv_id), depth))
        if self.pushed or not self.api.has_conversation_ended(self.conv_id):
            return
        self.pushed = True
        if self.pop_first:
            self.api.pop_fsm(self.conv_id)
        self.api.push_fsm(self.conv_id, self.fsm)


def _run_sync(api: API, conv_id: str, hook: Any) -> None:
    api.run_until_terminal(conv_id, max_steps=10, before_step=hook)


def _run_stream(api: API, conv_id: str, hook: Any) -> None:
    list(api.run_until_terminal_stream(conv_id, max_steps=10, before_step=hook))


class TestEndedFrameHookPushes:
    """D-052 termination (review pass 14 W2): what a PUSH on the ended-frame
    call does. The run goes on only when the top is no longer ended or the
    stack got shallower; a push of an already-ended FSM must end the run
    after one hook call (no loop to the stack limit), and a push of a live
    FSM continues on it. Sync and stream share the rounds."""

    _RUNNERS = pytest.mark.parametrize("run", [_run_sync, _run_stream])

    def _api(self) -> tuple[API, str]:
        api, conv_id = _start(_trip_fsm(), _StreamingLLM({"city": "Paris"}))
        _push_sub_on_entry(api, conv_id, "plan")
        return api, conv_id

    @_RUNNERS
    def test_pushing_an_ended_fsm_without_a_pop_returns(self, run):
        api, conv_id = self._api()
        hook = _PushOnEndedFrame(api, conv_id, _leaf_fsm(), pop_first=False)

        run(api, conv_id, hook)

        # One ended-frame call (n=3), no step after it, the leaf left on top.
        assert hook.calls == [(1, "collect", 1), (2, "sub_work", 2), (3, "sub_done", 2)]
        assert api.get_stack_depth(conv_id) == 3
        assert api.get_current_state(conv_id) == "leaf_done"

    @_RUNNERS
    def test_popping_then_pushing_an_ended_fsm_returns(self, run):
        api, conv_id = self._api()
        hook = _PushOnEndedFrame(api, conv_id, _leaf_fsm(), pop_first=True)

        run(api, conv_id, hook)

        assert hook.calls == [(1, "collect", 1), (2, "sub_work", 2), (3, "sub_done", 2)]
        assert api.get_stack_depth(conv_id) == 2
        assert api.get_current_state(conv_id) == "leaf_done"

    @_RUNNERS
    def test_pushing_a_live_fsm_continues_on_it(self, run):
        api, conv_id = self._api()
        hook = _PushOnEndedFrame(api, conv_id, _sub_fsm(), pop_first=False)

        run(api, conv_id, hook)

        # The pushed sub runs one step on top of the ended one; at its own
        # terminal the hook (pushing nothing more) is offered it and the run
        # returns.
        assert hook.calls == [
            (1, "collect", 1),
            (2, "sub_work", 2),
            (3, "sub_done", 2),
            (3, "sub_work", 3),
            (4, "sub_done", 3),
        ]
        assert api.get_stack_depth(conv_id) == 3
        assert api.get_current_state(conv_id) == "sub_done"


class TestRunEnded:
    def test_terminal_conversation_returns_an_empty_tuple(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        api.run_until_terminal(conv_id, max_steps=5)
        llm.reset()
        seen: list[int] = []

        assert (
            api.run_until_terminal(conv_id, max_steps=1, before_step=seen.append) == ()
        )
        assert seen == []
        assert llm.field_requests == [] and llm.response_requests == []

    def test_terminal_conversation_streams_nothing(self):
        api, conv_id = _start(_trip_fsm(), _StreamingLLM({"city": "Paris"}))
        list(api.run_until_terminal_stream(conv_id, max_steps=5))

        assert list(api.run_until_terminal_stream(conv_id, max_steps=1)) == []

    def test_conversation_closed_with_end_conversation_returns_an_empty_tuple(self):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))
        api.run_until_terminal(conv_id, max_steps=5)
        api.end_conversation(conv_id)

        assert api.run_until_terminal(conv_id, max_steps=1) == ()
        assert list(api.run_until_terminal_stream(conv_id, max_steps=1)) == []


class TestRunStream:
    def test_yields_the_chunks_of_speaking_states_only(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)

        chunks = list(api.run_until_terminal_stream(conv_id, max_steps=5))

        # ``plan`` is silent: only ``done`` speaks.
        assert chunks == _CHUNKS
        assert len(llm.stream_requests) == 1
        assert api.has_conversation_ended(conv_id)

    def test_every_speaking_state_streams_in_order(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(plan_speaks=True), llm)

        chunks = list(api.run_until_terminal_stream(conv_id, max_steps=5))

        assert chunks == [*_CHUNKS, *_CHUNKS]
        assert api.get_conversation_history(conv_id)[-2:] == [
            {"system": "".join(_CHUNKS)},
            {"system": "".join(_CHUNKS)},
        ]

    def test_nothing_runs_before_the_first_next(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        seen: list[int] = []

        stream = api.run_until_terminal_stream(
            conv_id, max_steps=5, before_step=seen.append
        )

        assert seen == []
        assert llm.field_requests == []
        assert api.get_current_state(conv_id) == "collect"
        assert conv_id not in api.fsm_manager._active_turns
        # A never-iterated stream does not block another turn.
        assert api.advance(conv_id).state_after == "plan"
        stream.close()

    def test_wall_clock_is_measured_from_the_first_next(self, clock):
        llm = _SlowStreamingLLM(clock, 1.0, fields={"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        stream = api.run_until_terminal_stream(conv_id, max_steps=5, max_seconds=10.0)

        clock.advance(1000.0)  # time spent before anyone iterates is not counted

        assert list(stream) == _CHUNKS

    def test_closed_stream_releases_the_conversation_and_keeps_the_steps(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(plan_speaks=True), llm)
        stream = api.run_until_terminal_stream(conv_id, max_steps=5)
        assert next(stream) == _CHUNKS[0]
        assert conv_id in api.fsm_manager._active_turns

        stream.close()

        assert conv_id not in api.fsm_manager._active_turns
        lock = api.fsm_manager._conversation_locks[conv_id]
        assert lock.acquire(blocking=False)
        lock.release()
        assert api.get_current_state(conv_id) == "plan"
        assert api.get_conversation_history(conv_id)[-1] == {"system": _CHUNKS[0]}
        assert _path(api.run_until_terminal(conv_id, max_steps=5)) == [("plan", "done")]

    def test_no_turn_is_open_between_steps(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(plan_speaks=True), llm)
        open_turns: list[bool] = []

        def _hook(step: int) -> None:
            open_turns.append(conv_id in api.fsm_manager._active_turns)

        list(api.run_until_terminal_stream(conv_id, max_steps=5, before_step=_hook))

        assert open_turns == [False, False]

    def test_step_budget_raises_after_exactly_max_steps(self):
        api, conv_id = _start(_loop_fsm(), _StreamingLLM())
        seen: list[int] = []
        stream = api.run_until_terminal_stream(
            conv_id, max_steps=4, before_step=seen.append
        )

        with pytest.raises(RunBudgetExceededError) as excinfo:
            list(stream)

        assert seen == [1, 2, 3, 4]
        error = excinfo.value
        assert (error.budget, error.limit, error.steps_done) == ("steps", 4, 4)
        assert api.get_current_state(conv_id) == "ping"
        assert conv_id not in api.fsm_manager._active_turns

    def test_slow_step_trips_max_seconds_before_the_next_step(self, clock):
        llm = _SlowStreamingLLM(clock, 5.0)
        api, conv_id = _start(_trip_fsm(), llm)
        seen: list[int] = []
        stream = api.run_until_terminal_stream(
            conv_id, max_steps=100, max_seconds=3.0, before_step=seen.append
        )

        with pytest.raises(RunBudgetExceededError) as excinfo:
            list(stream)

        error = excinfo.value
        assert (error.budget, error.limit, error.steps_done) == ("seconds", 3.0, 1)
        assert seen == [1]
        assert len(llm.field_requests) == _blocked_step_calls()

    def test_before_step_context_is_seen_and_its_exception_propagates(self):
        llm = _StreamingLLM()
        api, conv_id = _start(_trip_fsm(), llm)

        def _hook(step: int) -> None:
            if step == 1:
                api.update_context(conv_id, {"city": "Rome"})
            else:
                raise RuntimeError("second step refused")

        stream = api.run_until_terminal_stream(conv_id, max_steps=5, before_step=_hook)
        with pytest.raises(RuntimeError, match="second step refused"):
            list(stream)

        assert api.get_current_state(conv_id) == "plan"
        assert conv_id not in api.fsm_manager._active_turns
        assert list(api.run_until_terminal_stream(conv_id, max_steps=5)) == _CHUNKS

    def test_mid_stream_failure_propagates_and_rolls_back_that_step(self):
        llm = _StreamingLLM({"city": "Paris"}, fail_after=1)
        api, conv_id = _start(_trip_fsm(), llm)
        stream = api.run_until_terminal_stream(conv_id, max_steps=5)

        with pytest.raises(FSMError):
            list(stream)

        assert api.get_current_state(conv_id) == "plan"
        assert conv_id not in api.fsm_manager._active_turns


class TestRunArguments:
    @pytest.mark.parametrize("max_steps", [0, -1, 1.5, True, "3", None])
    def test_invalid_max_steps_raises_value_error(self, max_steps):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))

        with pytest.raises(ValueError, match="max_steps"):
            api.run_until_terminal(conv_id, max_steps=max_steps)
        with pytest.raises(ValueError, match="max_steps"):
            api.run_until_terminal_stream(conv_id, max_steps=max_steps)
        assert api.get_current_state(conv_id) == "collect"

    @pytest.mark.parametrize("max_seconds", [0, 0.0, -1.0, float("nan")])
    def test_invalid_max_seconds_raises_value_error(self, max_seconds):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))

        with pytest.raises(ValueError, match="max_seconds"):
            api.run_until_terminal(conv_id, max_steps=1, max_seconds=max_seconds)
        with pytest.raises(ValueError, match="max_seconds"):
            api.run_until_terminal_stream(conv_id, max_steps=1, max_seconds=max_seconds)
        assert api.get_current_state(conv_id) == "collect"

    def test_max_steps_is_required(self):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM())

        with pytest.raises(TypeError):
            api.run_until_terminal(conv_id)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            api.run_until_terminal_stream(conv_id)  # type: ignore[call-arg]

    def test_budgets_and_hook_are_keyword_only(self):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM())

        with pytest.raises(TypeError):
            api.run_until_terminal(conv_id, 5)  # type: ignore[misc]
        with pytest.raises(TypeError):
            api.run_until_terminal_stream(conv_id, 5)  # type: ignore[misc]

    def test_unknown_conversation_raises_value_error(self):
        api, _ = _start(_trip_fsm(), _ScriptedLLM())

        with pytest.raises(ValueError, match="Unknown conversation ID"):
            api.run_until_terminal("nope", max_steps=1)
        # The stream form refuses at call time, before any ``next()``.
        with pytest.raises(ValueError, match="Unknown conversation ID"):
            api.run_until_terminal_stream("nope", max_steps=1)

    def test_invalid_arguments_win_over_an_ended_conversation(self):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))
        api.run_until_terminal(conv_id, max_steps=5)

        with pytest.raises(ValueError, match="max_steps"):
            api.run_until_terminal(conv_id, max_steps=0)


class TestSecondsExemptStates:
    """``seconds_exempt_states`` (D-032 of plan 944e2692): a spent seconds
    budget does not stop a round whose current state is exempt; the steps
    budget still does. RED on the parent: the keyword did not exist and the
    clock stopped every state alike."""

    def test_exempt_state_finishes_after_the_clock_is_spent(self, clock):
        # Step 1 (collect -> plan) costs 5 s of a 3 s budget.
        llm = _SlowLLM(clock, 5.0, fields={"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)

        results = api.run_until_terminal(
            conv_id, max_steps=10, max_seconds=3.0, seconds_exempt_states={"plan"}
        )

        assert _path(results) == [("collect", "plan"), ("plan", "done")]
        assert api.has_conversation_ended(conv_id)

    def test_stream_form_honours_the_exemption(self, clock):
        llm = _SlowStreamingLLM(clock, 5.0, fields={"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)

        chunks = list(
            api.run_until_terminal_stream(
                conv_id, max_steps=10, max_seconds=3.0, seconds_exempt_states=["plan"]
            )
        )

        assert "".join(chunks) == _REPLY
        assert api.has_conversation_ended(conv_id)

    def test_a_state_not_named_is_still_stopped(self, clock):
        llm = _SlowLLM(clock, 5.0, fields={"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)

        with pytest.raises(RunBudgetExceededError) as excinfo:
            api.run_until_terminal(
                conv_id, max_steps=10, max_seconds=3.0, seconds_exempt_states={"done"}
            )

        assert (excinfo.value.budget, excinfo.value.steps_done) == ("seconds", 1)
        assert api.get_current_state(conv_id) == "plan"

    def test_the_steps_budget_still_applies_in_an_exempt_state(self, clock):
        llm = _SlowLLM(clock, 5.0, fields={"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)

        with pytest.raises(RunBudgetExceededError) as excinfo:
            api.run_until_terminal(
                conv_id, max_steps=1, max_seconds=3.0, seconds_exempt_states={"plan"}
            )

        assert (excinfo.value.budget, excinfo.value.limit) == ("steps", 1)

    def test_default_exempts_nothing(self, clock):
        llm = _SlowLLM(clock, 5.0, fields={"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)

        with pytest.raises(RunBudgetExceededError):
            api.run_until_terminal(conv_id, max_steps=10, max_seconds=3.0)

    def test_state_is_not_read_while_time_is_left(self, clock, monkeypatch):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))
        reads: list[str] = []
        original = api.get_current_state

        def spy(cid: str) -> str:
            reads.append(cid)
            return original(cid)

        monkeypatch.setattr(api, "get_current_state", spy)
        api.run_until_terminal(
            conv_id, max_steps=10, max_seconds=30.0, seconds_exempt_states={"plan"}
        )

        assert reads == []

    @pytest.mark.parametrize("states", ["plan", b"plan", [1], {None}, 5])
    def test_invalid_states_raise_value_error(self, states):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))

        with pytest.raises(ValueError, match="seconds_exempt_states"):
            api.run_until_terminal(conv_id, max_steps=5, seconds_exempt_states=states)
        with pytest.raises(ValueError, match="seconds_exempt_states"):
            api.run_until_terminal_stream(
                conv_id, max_steps=5, seconds_exempt_states=states
            )
        assert api.get_current_state(conv_id) == "collect"
