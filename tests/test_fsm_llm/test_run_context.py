"""The run context (plan ace7a8c8 step 21, roadmap 1.14, contract C12): ``RunContext``.

One frozen value given to a run: the interface or a factory, the log, the budgets, the cancel
token, the host callables and the parent run reference. ``derive`` narrows and never widens.
The module is lazy: ``import fsm_llm`` does not load it (the laziness table pins that).
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path

import pytest

import fsm_llm
from fsm_llm.core.definitions import FSMError, RunBudgetExceededError
from fsm_llm.core.names import NameRegistry
from fsm_llm.engine.context import (
    CancelToken,
    RunBudget,
    RunCancelledError,
    RunContext,
)
from fsm_llm.testing import ScriptedLLM
from tests.run_context_rig import SettableClock


def test_the_module_under_test_is_this_checkout() -> None:
    root = Path(__file__).resolve().parents[2]
    assert Path(fsm_llm.__file__).resolve().is_relative_to(root / "src")


class TestRunContextFields:
    def test_a_bare_context_holds_nothing_and_a_fresh_token(self) -> None:
        context = RunContext()
        assert context.llm_interface is None
        assert context.llm_factory is None
        assert context.event_log is None
        assert context.max_steps is None
        assert context.max_seconds is None
        assert context.callables is None
        assert context.parent is None
        assert isinstance(context.cancel, CancelToken)
        assert not context.cancel.cancelled
        assert RunContext().cancel is not context.cancel

    def test_it_is_frozen(self) -> None:
        context = RunContext(max_steps=3)
        with pytest.raises(dataclasses.FrozenInstanceError):
            context.max_steps = 9  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            context.parent = "x"  # type: ignore[misc]

    def test_it_holds_no_field_beyond_the_ones_with_a_consumer(self) -> None:
        names = {f.name for f in dataclasses.fields(RunContext)}
        assert names == {
            "llm_interface",
            "llm_factory",
            "event_log",
            "max_steps",
            "max_seconds",
            "cancel",
            "callables",
            "parent",
            "registry",
            "budget",
        }

    def test_unknown_keywords_are_refused(self) -> None:
        with pytest.raises(TypeError):
            RunContext(policy=object())  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            RunContext(handlers=object())  # type: ignore[call-arg]


class TestRunContextRefusals:
    def test_an_interface_and_a_factory_together_are_refused(self) -> None:
        with pytest.raises(ValueError, match=r"llm_interface.*llm_factory"):
            RunContext(llm_interface=ScriptedLLM(), llm_factory=ScriptedLLM)

    def test_the_interface_must_be_an_llm_interface(self) -> None:
        with pytest.raises(ValueError, match="LLMInterface"):
            RunContext(llm_interface=object())  # type: ignore[arg-type]

    def test_the_factory_must_be_callable(self) -> None:
        with pytest.raises(ValueError, match="callable"):
            RunContext(llm_factory="a model")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
    def test_a_bad_step_budget_is_refused(self, bad: object) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            RunContext(max_steps=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -0.5, True, float("nan"), "3"])
    def test_a_bad_time_budget_is_refused(self, bad: object) -> None:
        with pytest.raises(ValueError, match="max_seconds"):
            RunContext(max_seconds=bad)  # type: ignore[arg-type]

    def test_the_token_and_the_registry_have_a_type(self) -> None:
        with pytest.raises(TypeError, match="CancelToken"):
            RunContext(cancel=threading.Event())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="NameRegistry"):
            RunContext(callables={"f": print})  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_a_blank_parent_is_refused(self, bad: str) -> None:
        with pytest.raises(ValueError, match="parent"):
            RunContext(parent=bad)


class TestResolveLlm:
    def test_the_interface_is_returned_as_is(self) -> None:
        llm = ScriptedLLM()
        assert RunContext(llm_interface=llm).resolve_llm() is llm

    def test_a_factory_is_called_once_per_resolve_and_its_result_is_returned(
        self,
    ) -> None:
        made: list[ScriptedLLM] = []

        def factory() -> ScriptedLLM:
            made.append(ScriptedLLM())
            return made[-1]

        context = RunContext(llm_factory=factory)
        assert made == []  # building a context calls nothing
        first = context.resolve_llm()
        assert first is made[0] and len(made) == 1
        second = context.resolve_llm()
        assert second is made[1] and len(made) == 2  # one call per run, no caching

    def test_neither_gives_none(self) -> None:
        assert RunContext().resolve_llm() is None


class TestDerive:
    def test_a_child_shares_the_parents_members_and_names_its_parent(self) -> None:
        registry: NameRegistry[object] = NameRegistry("callables")
        llm = ScriptedLLM()
        parent = RunContext(
            llm_interface=llm,
            event_log="logs",
            max_steps=10,
            max_seconds=60,
            callables=registry,
        )
        child = parent.derive(parent="conv-1")
        assert child is not parent
        assert child.parent == "conv-1"
        assert child.llm_interface is llm
        assert child.event_log == "logs"
        assert child.callables is registry
        assert child.cancel is parent.cancel  # one cancel reaches every child
        assert (child.max_steps, child.max_seconds) == (10, 60)
        assert parent.parent is None  # the parent is untouched

    def test_a_limit_is_narrowed(self) -> None:
        parent = RunContext(max_steps=10, max_seconds=60.0)
        child = parent.derive(parent="p", max_steps=4, max_seconds=5.5)
        assert (child.max_steps, child.max_seconds) == (4, 5.5)
        assert (parent.max_steps, parent.max_seconds) == (10, 60.0)

    def test_the_same_limit_is_allowed(self) -> None:
        child = RunContext(max_steps=10, max_seconds=60).derive(
            parent="p", max_steps=10, max_seconds=60
        )
        assert (child.max_steps, child.max_seconds) == (10, 60)

    def test_a_limit_is_never_widened(self) -> None:
        parent = RunContext(max_steps=10, max_seconds=60)
        with pytest.raises(ValueError, match="max_steps"):
            parent.derive(parent="p", max_steps=11)
        with pytest.raises(ValueError, match="max_seconds"):
            parent.derive(parent="p", max_seconds=60.5)

    def test_a_limit_the_parent_lacks_takes_any_valid_value_and_none_keeps(
        self,
    ) -> None:
        parent = RunContext()
        child = parent.derive(parent="p", max_steps=7)
        assert (child.max_steps, child.max_seconds) == (7, None)
        # None means "keep", never "remove": a limit cannot be dropped by deriving.
        kept = RunContext(max_steps=5).derive(parent="p", max_steps=None)
        assert kept.max_steps == 5

    def test_a_bad_value_is_refused_like_a_constructor_value(self) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            RunContext(max_steps=5).derive(parent="p", max_steps=0)
        with pytest.raises(ValueError, match="max_seconds"):
            RunContext(max_seconds=5).derive(parent="p", max_seconds=float("nan"))

    def test_the_parent_reference_is_required_and_not_blank(self) -> None:
        with pytest.raises(TypeError):
            RunContext().derive()  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="parent"):
            RunContext().derive(parent=" ")

    def test_a_grandchild_narrows_again_and_keeps_the_token(self) -> None:
        root = RunContext(max_steps=9)
        child = root.derive(parent="a", max_steps=5)
        grandchild = child.derive(parent="b", max_steps=2)
        assert (grandchild.max_steps, grandchild.parent) == (2, "b")
        assert grandchild.cancel is root.cancel
        with pytest.raises(ValueError, match="max_steps"):
            child.derive(
                parent="b", max_steps=6
            )  # below the root's 9, above the child's 5


class TestCancelToken:
    def test_it_starts_clear_and_stays_set(self) -> None:
        token = CancelToken()
        assert token.cancelled is False
        token.raise_if_cancelled()  # no raise
        token.cancel()
        assert token.cancelled is True
        token.cancel()  # idempotent
        assert token.cancelled is True

    def test_raise_if_cancelled_raises_a_named_fsm_error(self) -> None:
        token = CancelToken()
        token.cancel()
        with pytest.raises(RunCancelledError) as info:
            token.raise_if_cancelled()
        assert isinstance(info.value, FSMError)
        # it is not the asyncio cancel: a token is a request checked between steps
        import asyncio

        assert not isinstance(info.value, asyncio.CancelledError)

    def test_a_cancel_from_another_thread_is_seen(self) -> None:
        token = CancelToken()
        started = threading.Event()
        done = threading.Event()

        def canceller() -> None:
            started.wait(5)
            token.cancel()
            done.set()

        thread = threading.Thread(target=canceller)
        thread.start()
        started.set()
        assert done.wait(5)
        thread.join(5)
        assert token.cancelled
        with pytest.raises(RunCancelledError):
            token.raise_if_cancelled()

    def test_many_threads_cancel_and_read_at_once(self) -> None:
        token = CancelToken()
        barrier = threading.Barrier(8)
        seen: list[bool] = []
        lock = threading.Lock()

        def worker(index: int) -> None:
            barrier.wait(5)
            if index % 2 == 0:
                token.cancel()
            observed = token.cancelled
            with lock:
                seen.append(observed)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        assert len(seen) == 8
        assert token.cancelled  # a set token is never cleared by a reader


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> SettableClock:
    fake = SettableClock()
    monkeypatch.setattr("fsm_llm.engine.fsm.budget._monotonic", fake)
    return fake


class TestRunBudget:
    """The run-level ledger (plan 7efd5cb4 D-004): exact counts, a leaf lock, a late clock."""

    def test_n_charges_are_allowed_and_the_next_raises_with_the_count_unchanged(
        self,
    ) -> None:
        ledger = RunBudget(max_steps=3)
        for _ in range(3):
            ledger.charge()
        assert ledger.steps_taken == 3
        with pytest.raises(RunBudgetExceededError) as info:
            ledger.charge()
        assert (info.value.budget, info.value.limit, info.value.steps_done) == (
            "steps",
            3,
            3,
        )
        assert ledger.steps_taken == 3
        assert ledger.spent() == "steps"

    def test_a_ledger_of_one_step(self) -> None:
        ledger = RunBudget(max_steps=1)
        assert ledger.spent() is None
        ledger.charge()
        assert ledger.spent() == "steps"
        with pytest.raises(RunBudgetExceededError):
            ledger.charge()

    def test_a_ledger_with_no_limit_only_counts(self) -> None:
        ledger = RunBudget()
        for _ in range(50):
            ledger.charge()
        assert ledger.steps_taken == 50
        assert ledger.spent() is None

    def test_the_clock_starts_at_the_first_charge_not_at_construction(
        self, clock: SettableClock
    ) -> None:
        ledger = RunBudget(max_seconds=5)
        clock.now += 100.0  # a host that built the context early must not burn it
        assert ledger.spent() is None
        ledger.charge()  # allowed: the clock starts here
        clock.now += 4.0
        ledger.charge()
        clock.now += 1.0
        assert ledger.spent() == "seconds"
        with pytest.raises(RunBudgetExceededError) as info:
            ledger.charge()
        assert (info.value.budget, info.value.limit, info.value.steps_done) == (
            "seconds",
            5,
            2,
        )
        assert ledger.steps_taken == 2

    def test_seconds_are_named_first_when_both_are_spent(
        self, clock: SettableClock
    ) -> None:
        ledger = RunBudget(max_steps=1, max_seconds=5)
        ledger.charge()
        clock.now += 9.0
        assert ledger.spent() == "seconds"

    def test_the_last_refusal_is_the_error_of_the_latest_refused_charge_only(
        self,
    ) -> None:
        """Plan 7efd5cb4 D-011. Kills: a refusal recorded on an ACCEPTED charge (the last one at
        the limit), one that is never replaced, one that is set before the lock is released."""
        ledger = RunBudget(max_steps=2)
        assert ledger.last_refusal is None
        ledger.charge()
        ledger.charge()  # the last allowed charge: spent, not refused
        assert ledger.spent() == "steps" and ledger.last_refusal is None
        with pytest.raises(RunBudgetExceededError) as first:
            ledger.charge()
        assert ledger.last_refusal is first.value
        with pytest.raises(RunBudgetExceededError) as second:
            ledger.charge()
        assert ledger.last_refusal is second.value is not first.value
        assert ledger.steps_taken == 2 and not ledger._lock.locked()

    def test_a_context_reads_the_last_refusal_of_its_ledger_and_none_without_one(
        self,
    ) -> None:
        assert RunContext().last_refusal() is None
        ledger = RunBudget(max_steps=1)
        context = RunContext(budget=ledger)
        derived = context.derive(parent="w")
        assert context.last_refusal() is None
        ledger.charge()
        with pytest.raises(RunBudgetExceededError) as info:
            ledger.charge()
        assert context.last_refusal() is derived.last_refusal() is info.value

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
    def test_a_bad_step_limit_is_refused_like_the_engines_own(
        self, bad: object
    ) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            RunBudget(max_steps=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -0.5, True, float("nan"), "3"])
    def test_a_bad_time_limit_is_refused_like_the_engines_own(
        self, bad: object
    ) -> None:
        with pytest.raises(ValueError, match="max_seconds"):
            RunBudget(max_seconds=bad)  # type: ignore[arg-type]

    def test_the_lock_is_not_held_after_a_raise(self) -> None:
        ledger = RunBudget(max_steps=1)
        ledger.charge()
        with pytest.raises(RunBudgetExceededError):
            ledger.charge()
        assert not ledger._lock.locked()
        assert ledger.spent() == "steps"  # a reader is not blocked by the raise

    @pytest.mark.parametrize("repeat", range(8))
    def test_the_window_two_threads_one_step_exactly_one_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, repeat: int
    ) -> None:
        """Two threads race for the one step; a clock that waits for the other thread
        inside the budget check opens the window. With the lock one thread is inside at a
        time (the barrier times out harmlessly); without it both pass the check."""
        ledger = RunBudget(max_steps=1, max_seconds=10_000)
        barrier = threading.Barrier(2, timeout=0.3)

        def waiting_clock() -> float:
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
            return 1000.0

        monkeypatch.setattr("fsm_llm.engine.fsm.budget._monotonic", waiting_clock)
        outcomes: list[str] = []
        guard = threading.Lock()

        def charge() -> None:
            try:
                ledger.charge()
                result = "accepted"
            except RunBudgetExceededError:
                result = "refused"
            with guard:
                outcomes.append(result)

        threads = [threading.Thread(target=charge) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert sorted(outcomes) == ["accepted", "refused"]
        assert ledger.steps_taken == 1

    def test_many_threads_charge_exactly_the_limit(self) -> None:
        ledger = RunBudget(max_steps=400)
        barrier = threading.Barrier(8, timeout=10)
        accepted: list[int] = []
        guard = threading.Lock()

        def worker() -> None:
            barrier.wait()
            count = 0
            for _ in range(100):
                try:
                    ledger.charge()
                    count += 1
                except RunBudgetExceededError:
                    pass
            with guard:
                accepted.append(count)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert sum(accepted) == 400
        assert ledger.steps_taken == 400


class TestTheBudgetField:
    """``budget`` is a DECLARED field, so ``dataclasses.replace`` carries it by reference."""

    def test_it_is_the_last_declared_field_and_defaults_to_none(self) -> None:
        fields = dataclasses.fields(RunContext)
        assert fields[-1].name == "budget"
        assert RunContext().budget is None

    @pytest.mark.parametrize("bad", [3, "ledger", object(), {"max_steps": 3}])
    def test_a_non_ledger_is_a_type_error_naming_it(self, bad: object) -> None:
        with pytest.raises(TypeError, match="RunBudget"):
            RunContext(budget=bad)  # type: ignore[arg-type]

    def test_derive_shares_the_same_ledger_by_identity(self) -> None:
        ledger = RunBudget(max_steps=5)
        parent = RunContext(budget=ledger)
        child = parent.derive(parent="a", max_steps=3)
        grandchild = child.derive(parent="b")
        assert child.budget is ledger
        assert grandchild.budget is ledger
        child.budget.charge()  # type: ignore[union-attr]
        assert ledger.steps_taken == 1  # a child is charged to the parent

    def test_a_second_replace_keeps_it(self) -> None:
        """``ExecutableStep`` replaces a derived context once more; a ledger kept as an
        undeclared attribute would be lost there."""
        ledger = RunBudget(max_steps=5)
        child = RunContext(budget=ledger).derive(parent="a")
        again = dataclasses.replace(child, max_seconds=9)
        assert again.budget is ledger

    def test_the_context_stays_free_of_data_and_grants(self) -> None:
        names = {f.name for f in dataclasses.fields(RunContext)}
        assert not names & {"data", "context", "grant", "approval", "_approval_granted"}


class TestStepHook:
    """``step_hook()``: the cancel check first, then one charge per distinct step number."""

    def test_it_charges_once_per_distinct_step_number(self) -> None:
        ledger = RunBudget(max_steps=10)
        hook = RunContext(budget=ledger).step_hook()
        for step in (1, 2, 2, 3):  # the ended-frame offer repeats a number
            hook(step)
        assert ledger.steps_taken == 3

    def test_a_cancelled_token_raises_before_any_charge(self) -> None:
        ledger = RunBudget(max_steps=10)
        context = RunContext(budget=ledger)
        hook = context.step_hook()
        hook(1)
        context.cancel.cancel()
        with pytest.raises(RunCancelledError):
            hook(2)
        assert ledger.steps_taken == 1  # the check ran first: nothing was charged

    def test_a_cancelled_context_is_dead_for_every_later_hook(self) -> None:
        context = RunContext(budget=RunBudget(max_steps=10))
        context.cancel.cancel()
        for _ in range(2):
            with pytest.raises(RunCancelledError):
                context.step_hook()(1)
        assert context.budget.steps_taken == 0  # type: ignore[union-attr]

    def test_two_closures_keep_their_own_last_step_over_one_ledger(self) -> None:
        ledger = RunBudget(max_steps=10)
        context = RunContext(budget=ledger)
        first, second = context.step_hook(), context.step_hook()
        assert first is not second
        first(1)
        first(2)
        second(1)  # a fresh run restarts at 1 and is charged again
        second(1)
        assert ledger.steps_taken == 3

    def test_a_spent_ledger_raises_the_engines_error_and_keeps_the_count(self) -> None:
        context = RunContext(budget=RunBudget(max_steps=2))
        hook = context.step_hook()
        hook(1)
        hook(2)
        with pytest.raises(RunBudgetExceededError) as info:
            hook(3)
        assert (info.value.budget, info.value.limit, info.value.steps_done) == (
            "steps",
            2,
            2,
        )
        with pytest.raises(RunBudgetExceededError):
            context.step_hook()(1)  # the NEXT run is refused at its first step
        assert context.budget.steps_taken == 2  # type: ignore[union-attr]

    def test_with_no_ledger_it_only_checks_the_cancel(self) -> None:
        context = RunContext()
        hook = context.step_hook()
        hook(1)
        hook(2)
        context.cancel.cancel()
        with pytest.raises(RunCancelledError):
            hook(3)

    def test_budget_spent_is_a_fact_of_the_ledger(self, clock: SettableClock) -> None:
        assert RunContext().budget_spent() is None
        context = RunContext(budget=RunBudget(max_steps=1, max_seconds=5))
        assert context.budget_spent() is None
        context.step_hook()(1)
        assert context.budget_spent() == "steps"
        other = RunContext(budget=RunBudget(max_seconds=5))
        other.step_hook()(1)
        clock.now += 6.0
        assert other.budget_spent() == "seconds"


class TestStepHookInTheRunLoop:
    """The hook against the real run loop (doubles only): a sync hook serves both modes, and
    the ended-frame offer repeats a step number without being a step."""

    @staticmethod
    def _pushing_run() -> tuple[object, str]:
        from tests.test_fsm_llm.test_advance import _ScriptedLLM, _start, _trip_fsm
        from tests.test_fsm_llm.test_run_until_terminal import _push_sub_on_entry

        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))
        _push_sub_on_entry(api, conv_id, "plan")
        return api, conv_id

    @staticmethod
    def _popping(api: object, conv_id: str, hook, seen: list[int]):  # type: ignore[no-untyped-def]
        def before_step(step: int) -> None:
            seen.append(step)
            if api.get_stack_depth(conv_id) > 1 and api.has_conversation_ended(conv_id):  # type: ignore[attr-defined]
                api.pop_fsm(conv_id)  # type: ignore[attr-defined]
            hook(step)

        return before_step

    def test_an_ended_pushed_frame_offers_a_number_twice_and_charges_it_once(
        self,
    ) -> None:
        api, conv_id = self._pushing_run()
        ledger = RunBudget(max_steps=10)
        seen: list[int] = []
        hook = self._popping(api, conv_id, RunContext(budget=ledger).step_hook(), seen)
        results = api.run_until_terminal(conv_id, max_steps=10, before_step=hook)  # type: ignore[attr-defined]
        assert seen == [
            1,
            2,
            3,
            3,
        ]  # step 3 is offered to pop the ended sub-FSM, then run
        assert len(results) == 3  # the engine's own steps_done
        assert ledger.steps_taken == 3

    def test_the_async_loop_awaits_the_same_sync_hook(self) -> None:
        import asyncio

        api, conv_id = self._pushing_run()
        ledger = RunBudget(max_steps=10)
        seen: list[int] = []
        hook = self._popping(api, conv_id, RunContext(budget=ledger).step_hook(), seen)
        results = asyncio.run(
            api.arun_until_terminal(conv_id, max_steps=10, before_step=hook)  # type: ignore[attr-defined]
        )
        assert seen == [1, 2, 3, 3]
        assert len(results) == 3
        assert ledger.steps_taken == 3

    def test_a_spent_ledger_stops_the_run_before_the_step_it_forbids(self) -> None:
        from tests.test_fsm_llm.test_advance import _ScriptedLLM, _start, _trip_fsm

        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        context = RunContext(budget=RunBudget(max_steps=1))
        with pytest.raises(RunBudgetExceededError) as info:
            api.run_until_terminal(
                conv_id, max_steps=10, before_step=context.step_hook()
            )
        assert (info.value.budget, info.value.limit) == ("steps", 1)
        assert (
            api.get_current_state(conv_id) == "plan"
        )  # one step ran, the second did not
        assert context.budget.steps_taken == 1  # type: ignore[union-attr]

    def test_a_cancel_set_by_the_double_stops_the_next_step_only(self) -> None:
        from tests.test_fsm_llm.test_advance import _ScriptedLLM, _start, _trip_fsm

        context = RunContext(budget=RunBudget(max_steps=10))

        class _Cancelling(_ScriptedLLM):
            def extract_field(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                context.cancel.cancel()
                return super().extract_field(*args, **kwargs)

        api, conv_id = _start(_trip_fsm(), _Cancelling({"city": "Paris"}))
        with pytest.raises(RunCancelledError):
            api.run_until_terminal(
                conv_id, max_steps=10, before_step=context.step_hook()
            )
        assert (
            api.get_current_state(conv_id) == "plan"
        )  # step 1 committed, step 2 never began
        assert context.budget.steps_taken == 1  # type: ignore[union-attr]
