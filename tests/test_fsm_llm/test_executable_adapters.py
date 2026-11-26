"""The FSM-body executable (plan ace7a8c8 step 22.1, roadmap E2.1, decision D-059).

One adapter serves the ``fsm``, ``strategy`` and ``procedure`` kinds: it owns its own ``API``,
drives the body with the message-free step loop under a ``RunContext`` and reports a ``RunResult``
whose identity carries both digests. Rows run the REAL core with a scripted model (no network).
Cancel rows use events and bounded waits, never sleeps (D7: one turn per conversation, a cancel
rolls back whole, the conversation is ended exactly once and the gate is free).
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
import threading
from pathlib import Path
from typing import Any, ClassVar

import pytest

import fsm_llm
from fsm_llm import API
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.definitions import FSMDefinition, FSMError, LLMResponseError
from fsm_llm.core.definitions.executable import (
    InstanceIdentityMismatchError,
    InstantiationError,
    parse_executable,
)
from fsm_llm.core.definitions.executable_instance import identity_of
from fsm_llm.core.definitions.fsm import definition_digest
from fsm_llm.core.results import RunResult
from fsm_llm.engine.context import CancelToken, RunBudget, RunContext
from fsm_llm.engine.executables import (
    DEFAULT_EXECUTABLE_MAX_STEPS,
    EXECUTABLE_KINDS,
    ExecutableKindError,
    ExecutableRegistry,
    FsmBodyExecutable,
    FsmInstance,
    adapter_for,
)
from fsm_llm.engine.executables import adapters as adapters_module
from fsm_llm.llm import ModelNotConfiguredError
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm import kind_corpus as kc
from tests.test_fsm_llm.test_async_cancel import _BOUND, _gate_is_free, _Parking
from tests.test_fsm_llm.test_async_engine import _Suspending

_BOUND_SECONDS = _BOUND


def test_the_module_under_test_is_this_checkout() -> None:
    root = Path(__file__).resolve().parents[2]
    assert Path(fsm_llm.__file__).resolve().is_relative_to(root / "src")


# ---- fixtures ---------------------------------------------------------------------------------


def _fsm(**extra: Any) -> dict[str, Any]:
    """a -> b: state ``a`` extracts ``result`` and speaks, ``b`` is terminal and speaks."""
    return kc._fsm(
        "Demo",
        "a",
        kc._state(
            "a",
            extraction_instructions="Extract the result.",
            required_context_keys=["result"],
            response_instructions="A.",
            transitions=[kc._go("b")],
        ),
        kc._state("b", response_instructions="B."),
        **extra,
    )


def _long_fsm() -> dict[str, Any]:
    return kc._fsm(
        "Long",
        "a",
        kc._state("a", response_instructions="A.", transitions=[kc._go("b")]),
        kc._state("b", response_instructions="B.", transitions=[kc._go("c")]),
        kc._state("c", response_instructions="C.", transitions=[kc._go("d")]),
        kc._state("d", response_instructions="D."),
    )


def _needs_user() -> dict[str, Any]:
    """A conversational body: it leaves ``ask`` only when the user gave a ``name``."""
    return kc._fsm(
        "Needs",
        "ask",
        kc._state(
            "ask",
            extraction_instructions="Extract the name.",
            required_context_keys=["name"],
            response_instructions="Ask the name.",
            transitions=[
                kc._go(
                    "done",
                    conditions=kc._when("name", {"!=": [{"var": "name"}, ""]}, "named"),
                )
            ],
        ),
        kc._state("done", response_instructions="Thanks."),
    )


def _doc(
    kind: str = "fsm",
    ident: str = "demo",
    *,
    fsm: dict[str, Any] | None = None,
    inputs: tuple[str, ...] = ("topic",),
    outputs: tuple[str, ...] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    if outputs is None:  # a strategy declares the orchestrator-side name of its result
        outputs = ("final_solution",) if kind == "strategy" else ("result",)
    body: dict[str, Any] = {"fsm": fsm if fsm is not None else _fsm()}
    if kind == "strategy":
        body["strategy"] = {
            "results": {"final_solution": "result", "kept": "missing_key"},
            "solution_key": "final_solution",
        }
    return {
        "schema": "fsm-llm/executable",
        "kind": kind,
        "id": ident,
        "contract": {"inputs": list(inputs), "outputs": list(outputs)},
        "body": body,
        **extra,
    }


def _executable(**kw: Any) -> FsmBodyExecutable:
    exe = ExecutableRegistry().add_document(_doc(**kw))
    assert isinstance(exe, FsmBodyExecutable)
    return exe


def _llm(**kw: Any) -> ScriptedLLM:
    kw.setdefault("extraction_data", {"result": "42"})
    kw.setdefault("default_response", "r")
    return ScriptedLLM(**kw)


class _RecordingAPI(API):
    """The ``API`` of the adapter, recording what the run did to its conversations."""

    made: ClassVar[list[_RecordingAPI]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ended: list[tuple[str, bool]] = []
        self.step_calls: list[dict[str, Any]] = []
        type(self).made.append(self)

    def end_conversation(self, conversation_id: str) -> None:
        self.ended.append((conversation_id, _gate_is_free(self, conversation_id)))
        super().end_conversation(conversation_id)

    async def arun_until_terminal(self, conversation_id: str, **kwargs: Any):  # type: ignore[override]
        self.step_calls.append(kwargs)
        return await super().arun_until_terminal(conversation_id, **kwargs)


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[_RecordingAPI]:
    _RecordingAPI.made = []
    monkeypatch.setattr(adapters_module, "API", _RecordingAPI)
    return _RecordingAPI.made


def _context(llm: Any = None, **kw: Any) -> RunContext:
    return RunContext(llm_interface=llm if llm is not None else _llm(), **kw)


def _without_handle(result: RunResult) -> dict[str, Any]:
    return result.model_dump(exclude={"handle"})


# ---- the three kinds --------------------------------------------------------------------------


class TestEachKind:
    @pytest.mark.parametrize("kind", ["fsm", "strategy", "procedure"])
    def test_a_scripted_run_is_ok_and_carries_both_identities(self, kind: str) -> None:
        document = _doc(kind)
        exe = ExecutableRegistry().add_document(document)
        result = exe.run({"topic": "t"}, context=_context())
        assert (result.status, result.reason, result.success) == (
            Status.OK,
            None,
            True,
        )
        assert result.answer == ("42" if kind == "strategy" else "r")
        assert result.outputs == (
            {"final_solution": "42"} if kind == "strategy" else {"result": "42"}
        )
        envelope = parse_executable(document)
        assert (
            result.identity.template_identity == identity_of(envelope).template_identity
        )
        assert result.identity.template_identity == exe.template_identity
        assert result.identity.instance_identity == definition_digest(
            FSMDefinition(**document["body"]["fsm"])
        )
        assert isinstance(result.handle, str) and result.handle
        assert result.detail["kind"] == kind
        assert result.detail["final_state"] == "b"

    @pytest.mark.parametrize("kind", ["fsm", "strategy", "procedure"])
    def test_run_equals_arun(self, kind: str) -> None:
        exe = ExecutableRegistry().add_document(_doc(kind))
        sync = exe.run({"topic": "t"}, context=_context())
        asynchronous = asyncio.run(exe.arun({"topic": "t"}, context=_context()))
        assert _without_handle(sync) == _without_handle(asynchronous)
        assert sync.handle != asynchronous.handle  # two conversations

    def test_a_strategy_maps_results_and_skips_absent_keys(self) -> None:
        # ``kept`` reads a key the body never writes: copied only when not None (the engine's rule)
        result = _executable(kind="strategy", outputs=()).run(
            {"topic": "t"}, context=_context()
        )
        assert result.outputs == {"final_solution": "42"}
        assert result.answer == "42"  # the solution key's text

    def test_an_fsm_without_declared_outputs_returns_none(self) -> None:
        result = _executable(outputs=()).run({"topic": "t"}, context=_context())
        assert result.outputs == {} and result.status is Status.OK

    def test_the_default_step_budget_is_finite(self) -> None:
        assert 1 <= DEFAULT_EXECUTABLE_MAX_STEPS < 1000


# ---- before any call ---------------------------------------------------------------------------


class TestBeforeAnyCall:
    def test_a_missing_declared_input_fails_with_no_model_call(self) -> None:
        llm = _llm()
        result = _executable(inputs=("topic", "extra")).run(
            {"topic": "t"}, context=_context(llm)
        )
        assert (result.status, result.reason) == (Status.FAILED, Reason.INPUTS_MISSING)
        assert result.detail["missing"] == ["extra"]
        assert result.handle is None
        assert llm.call_history == []

    def test_an_undeclared_input_is_a_value_error_with_no_model_call(self) -> None:
        llm = _llm()
        with pytest.raises(ValueError, match="declares no input"):
            _executable().run({"topic": "t", "other": 1}, context=_context(llm))
        assert llm.call_history == []

    def test_an_internal_prefixed_input_is_refused_even_when_declared(self) -> None:
        llm = _llm()
        exe = _executable(inputs=("_approval_granted",))
        with pytest.raises(ValueError, match="reserved prefix"):
            exe.run({"_approval_granted": True}, context=_context(llm))
        assert llm.call_history == []

    def test_a_context_with_no_model_raises_the_resolver_error(
        self, monkeypatch: pytest.MonkeyPatch, recorded: list[_RecordingAPI]
    ) -> None:
        monkeypatch.delenv("LLM_MODEL", raising=False)
        with pytest.raises(ModelNotConfiguredError):
            _executable().run({"topic": "t"})
        assert recorded == []  # no API was built: nothing to end

    def test_a_factory_is_called_once_per_run(self) -> None:
        made: list[ScriptedLLM] = []

        def factory() -> ScriptedLLM:
            made.append(_llm())
            return made[-1]

        exe = _executable()
        context = RunContext(llm_factory=factory)
        assert len(made) == 0
        assert exe.run({"topic": "t"}, context=context).success
        assert len(made) == 1
        assert exe.run({"topic": "t"}, context=context).success
        assert len(made) == 2


# ---- mapped errors -----------------------------------------------------------------------------


class _FailsOnSecondReply(ScriptedLLM):
    """The greeting answers; the next ``generate_response`` raises ``error``."""

    def __init__(self, error: BaseException, **kw: Any) -> None:
        super().__init__(**kw)
        self.error = error
        self.replies_made = 0

    def generate_response(self, request: Any) -> Any:
        self.replies_made += 1
        if self.replies_made >= 2:
            raise self.error
        return super().generate_response(request)


class TestMappedErrors:
    def test_a_spent_step_budget_is_aborted_budget_spent(
        self, recorded: list[_RecordingAPI]
    ) -> None:
        exe = _executable(fsm=_long_fsm(), outputs=())
        result = exe.run({"topic": "t"}, context=_context(max_steps=2))
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert result.detail["budget"] == "steps"
        assert result.detail["limit"] == 2 and result.detail["steps_done"] == 2
        assert result.handle
        assert [cid for cid, _ in recorded[0].ended] == [result.handle]

    def test_the_budgets_and_the_hook_reach_the_step_loop(
        self, recorded: list[_RecordingAPI]
    ) -> None:
        context = _context(max_steps=7, max_seconds=30.0)
        assert _executable().run({"topic": "t"}, context=context).success
        call = recorded[0].step_calls[0]
        assert call["max_steps"] == 7 and call["max_seconds"] == 30.0
        assert callable(call["before_step"])
        assert _executable().run({"topic": "t"}, context=_context()).success
        assert recorded[1].step_calls[0]["max_steps"] == DEFAULT_EXECUTABLE_MAX_STEPS

    def test_a_conversational_body_that_needs_a_user_message_spends_its_budget(
        self,
    ) -> None:
        # the known limit: no stall detection, NEEDS_USER_INPUT is not emitted
        exe = _executable(fsm=_needs_user(), outputs=())
        result = exe.run(
            {"topic": "t"}, context=_context(_llm(extraction_data={}), max_steps=3)
        )
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)

    @pytest.mark.parametrize("where", ["greeting", "step"])
    def test_a_provider_failure_is_failed_provider_error_and_leaks_no_text(
        self, where: str, recorded: list[_RecordingAPI]
    ) -> None:
        error = LLMResponseError("provider down, api_key=sk-secret")
        llm = _FailsOnSecondReply(
            error, extraction_data={"result": "42"}, default_response="r"
        )
        if where == "greeting":
            llm.replies_made = 1  # the very first reply (the greeting) raises
        result = _executable(fsm=_long_fsm(), outputs=()).run(
            {"topic": "t"}, context=_context(llm)
        )
        assert (result.status, result.reason) == (Status.FAILED, Reason.PROVIDER_ERROR)
        assert result.detail["error"] == "LLMResponseError"
        assert "sk-secret" not in repr(result)
        assert len(recorded[0].ended) == (0 if where == "greeting" else 1)

    def test_a_declared_output_missing_at_the_terminal_state_fails(self) -> None:
        result = _executable().run(
            {"topic": "t"}, context=_context(_llm(extraction_data={}))
        )
        assert (result.status, result.reason) == (
            Status.FAILED,
            Reason.RETURN_KEYS_MISSING,
        )
        assert result.detail["missing"] == ["result"]
        assert result.handle

    def test_any_other_error_propagates_and_the_conversation_is_ended_once(
        self, recorded: list[_RecordingAPI]
    ) -> None:
        llm = _FailsOnSecondReply(
            RuntimeError("a bug"), extraction_data={"result": "1"}, default_response="r"
        )
        with pytest.raises(FSMError, match="a bug") as raised:
            _executable(fsm=_long_fsm(), outputs=()).run(
                {"topic": "t"}, context=_context(llm)
            )
        # the engine wraps a bug of a step in an FSMError chained from it; the adapter adds nothing
        assert isinstance(raised.value.__cause__, RuntimeError)
        assert len(recorded[0].ended) == 1 and recorded[0].ended[0][1] is True


# ---- the cancel token and the task cancel ------------------------------------------------------


class _CancelsAfterFirstReply(ScriptedLLM):
    def __init__(self, token: CancelToken, **kw: Any) -> None:
        super().__init__(**kw)
        self.token = token

    def generate_response(self, request: Any) -> Any:
        reply = super().generate_response(request)
        if len(self.calls("generate_response")) == 2:  # the first step's reply
            self.token.cancel()
        return reply


class TestCancelToken:
    def test_a_token_set_before_the_run_ends_it_with_no_model_call(
        self, recorded: list[_RecordingAPI]
    ) -> None:
        token = CancelToken()
        token.cancel()
        llm = _llm()
        result = _executable().run({"topic": "t"}, context=_context(llm, cancel=token))
        assert (result.status, result.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert llm.call_history == []
        assert recorded[0].ended == []  # nothing had started

    def test_a_token_set_mid_run_stops_at_the_next_step_boundary(
        self, recorded: list[_RecordingAPI]
    ) -> None:
        token = CancelToken()
        llm = _CancelsAfterFirstReply(token, default_response="r")
        result = _executable(fsm=_long_fsm(), outputs=()).run(
            {"topic": "t"}, context=_context(llm, cancel=token)
        )
        assert (result.status, result.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert result.handle
        # the greeting and the first step answered; no later step ran
        assert len(llm.calls("generate_response")) == 2
        assert [cid for cid, _ in recorded[0].ended] == [result.handle]


def _go(how: str, context: RunContext, **kw: Any) -> RunResult:
    """The same run through the sync name and through its async twin."""
    exe = _executable(**{"fsm": _long_fsm(), "outputs": (), **kw})
    if how == "sync":
        return exe.run({"topic": "t"}, context=context)
    return asyncio.run(exe.arun({"topic": "t"}, context=context))


class TestSharedLedger:
    """Plan 7efd5cb4 step 3 (D-004): the body is driven by the hook of the context, so one cancel
    and one ledger reach it (the long body is three steps and four model calls: the greeting and
    one reply per step)."""

    @pytest.mark.parametrize("how", ["sync", "async"])
    def test_a_ledger_smaller_than_the_body_ends_budget_spent_after_exactly_its_steps(
        self, how: str, recorded: list[_RecordingAPI]
    ) -> None:
        budget, llm = RunBudget(max_steps=2), _llm()
        result = _go(how, _context(llm, budget=budget))
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert budget.steps_taken == 2
        assert len(llm.call_history) == 3  # the greeting and the two steps it allowed
        assert result.detail["budget"] == "steps" and result.detail["limit"] == 2
        assert (
            result.detail["shared_budget"] == "steps"
        )  # a fact of the ledger, not of the text
        assert [cid for cid, _ in recorded[0].ended] == [result.handle]

    @pytest.mark.parametrize("how", ["sync", "async"])
    def test_a_ledger_as_large_as_the_body_lets_it_finish(self, how: str) -> None:
        budget = RunBudget(max_steps=3)
        result = _go(how, _context(budget=budget))
        assert result.status is Status.OK and budget.steps_taken == 3
        assert "shared_budget" not in result.detail

    @pytest.mark.parametrize("how", ["sync", "async"])
    def test_a_seconds_ledger_names_the_seconds_budget(
        self, how: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = [1000.0]
        monkeypatch.setattr("fsm_llm.engine.fsm.budget._monotonic", lambda: now[0])

        class _Slow(ScriptedLLM):
            def generate_response(self, request: Any) -> Any:
                reply = super().generate_response(request)
                if len(self.calls("generate_response")) == 2:  # the first step's reply
                    now[0] += 60.0
                return reply

        budget = RunBudget(max_seconds=30.0)
        result = _go(how, _context(_Slow(default_response="r"), budget=budget))
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert result.detail["budget"] == "seconds"
        assert result.detail["shared_budget"] == "seconds"
        assert budget.steps_taken == 1

    @pytest.mark.parametrize("how", ["sync", "async"])
    def test_two_runs_on_one_context_share_the_ledger(self, how: str) -> None:
        budget = RunBudget(max_steps=4)
        context = _context(budget=budget)
        assert _go(how, context).status is Status.OK and budget.steps_taken == 3
        second = _go(how, context)  # one step of room left
        assert (second.status, second.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert second.detail["shared_budget"] == "steps" and budget.steps_taken == 4

    @pytest.mark.parametrize("how", ["sync", "async"])
    def test_the_runs_own_ceiling_keeps_its_meaning_and_is_not_called_shared(
        self, how: str
    ) -> None:
        budget = RunBudget(max_steps=100)
        result = _go(how, _context(max_steps=2, budget=budget))
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert result.detail["limit"] == 2 and "shared_budget" not in result.detail
        assert budget.steps_taken == 2  # every step the run took was charged

    @pytest.mark.parametrize("how", ["sync", "async"])
    def test_a_token_set_mid_run_is_cancelled_before_the_next_charge(
        self, how: str, recorded: list[_RecordingAPI]
    ) -> None:
        token, budget = CancelToken(), RunBudget(max_steps=50)
        llm = _CancelsAfterFirstReply(token, default_response="r")
        result = _go(how, _context(llm, cancel=token, budget=budget))
        assert (result.status, result.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert "shared_budget" not in result.detail
        # the first step was charged; the cancel check of the second came first and charged nothing
        assert budget.steps_taken == 1 and len(llm.calls("generate_response")) == 2
        assert [cid for cid, _ in recorded[0].ended] == [result.handle]
        # the NEXT call of the same caller: the token stays set, so it ends at its first check
        again = _go(how, _context(llm, cancel=token, budget=budget))
        assert (again.status, again.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert budget.steps_taken == 1 and len(llm.calls("generate_response")) == 2
        # a fresh token on the same ledger runs on (the ledger has room for the whole body)
        fresh = _go(how, _context(_llm(), budget=budget))
        assert fresh.status is Status.OK and budget.steps_taken == 4

    @pytest.mark.parametrize("how", ["sync", "async"])
    def test_a_spent_ledger_refuses_the_run_before_the_greeting(
        self, how: str, recorded: list[_RecordingAPI]
    ) -> None:
        """Plan 7efd5cb4 step 3.1 (D-007). Kills: the check dropped (the greeting call is spent
        and a step runs), the check after the first call, a refusal that charges again."""
        budget, llm = RunBudget(max_steps=1), _llm()
        budget.charge()
        result = _go(how, _context(llm, budget=budget))
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert result.detail["shared_budget"] == "steps"
        assert llm.call_history == []  # not even the greeting
        assert budget.steps_taken == 1  # the refusal charged nothing
        assert recorded[0].step_calls == []  # the step loop never started
        assert result.handle is None

    @pytest.mark.parametrize("how", ["sync", "async"])
    def test_a_spent_seconds_ledger_refuses_the_run_the_same_way(
        self, how: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = [1000.0]
        monkeypatch.setattr("fsm_llm.engine.fsm.budget._monotonic", lambda: now[0])
        budget, llm = RunBudget(max_seconds=30.0), _llm()
        budget.charge()
        now[0] += 60.0
        result = _go(how, _context(llm, budget=budget))
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert result.detail["shared_budget"] == "seconds"
        assert llm.call_history == [] and budget.steps_taken == 1

    @pytest.mark.parametrize("how", ["sync", "async"])
    def test_an_unspent_ledger_runs_as_before_and_a_set_token_is_read_first(
        self, how: str
    ) -> None:
        budget = RunBudget(max_steps=3)
        # the run takes exactly the ledger: unspent before it, spent after it
        assert _go(how, _context(budget=budget)).status is Status.OK
        assert budget.steps_taken == 3
        token, llm = CancelToken(), _llm()
        token.cancel()
        both = _go(how, _context(llm, cancel=token, budget=budget))
        assert (both.status, both.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert "shared_budget" not in both.detail
        assert llm.call_history == [] and budget.steps_taken == 3

    def test_the_hook_of_the_step_loop_is_the_step_hook_of_the_context(
        self, recorded: list[_RecordingAPI]
    ) -> None:
        budget = RunBudget(max_steps=50)
        _go("sync", _context(budget=budget))
        hook = recorded[0].step_calls[0]["before_step"]
        hook(
            99
        )  # a distinct number of ITS closure: it charges the ledger of the context
        assert budget.steps_taken == 4


class TestATaskCancel:
    async def test_a_cancel_at_every_model_await_ends_the_conversation_once(
        self, recorded: list[_RecordingAPI]
    ) -> None:
        """Kills: ``end_conversation`` outside the ``finally`` (a cancel leaves the conversation
        open), an ``except BaseException`` around the run (the cancel is swallowed and a result
        returned), a release that raises into the cancel."""
        exe = _executable(fsm=_long_fsm(), outputs=())
        counting = _Parking(default_response="r")
        counting.park(-1)
        assert (await exe.arun({"topic": "t"}, context=_context(counting))).success
        awaits = counting.calls
        assert awaits >= 4
        for k in range(1, awaits + 1):
            recorded.clear()
            llm = _Parking(default_response="r")
            llm.park(k)
            task = asyncio.create_task(exe.arun({"topic": "t"}, context=_context(llm)))
            await asyncio.wait_for(llm.entered.wait(), _BOUND_SECONDS)
            task.cancel()
            await asyncio.wait({task}, timeout=_BOUND_SECONDS)
            assert task.cancelled(), f"k={k}: a cancel is never swallowed"
            api = recorded[0]
            assert api.list_active_conversations() == [], f"k={k}"
            assert k == 1 or len(api.ended) == 1, f"k={k}"
            assert all(free for _, free in api.ended), f"k={k}: the gate was held"

    async def test_two_runs_on_one_loop_interleave(self) -> None:
        exe = _executable(fsm=_long_fsm(), outputs=())
        beats = 0
        stop = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal beats
            while not stop.is_set():
                beats += 1
                await asyncio.sleep(0)

        beat = asyncio.create_task(heartbeat())
        results = await asyncio.gather(
            exe.arun(
                {"topic": "a"}, context=_context(_Suspending(default_response="r"))
            ),
            exe.arun(
                {"topic": "b"}, context=_context(_Suspending(default_response="r"))
            ),
        )
        stop.set()
        await beat
        assert all(r.success for r in results)
        assert results[0].handle != results[1].handle
        assert beats > 4  # the loop ran while both runs were suspended


# ---- the sync name -----------------------------------------------------------------------------


class _RecordsThread(ScriptedLLM):
    threads: set[str]

    def generate_response(self, request: Any) -> Any:
        self.threads.add(threading.current_thread().name)
        return super().generate_response(request)


class TestTheSyncName:
    def test_the_sync_run_makes_no_thread_and_needs_no_loop(self) -> None:
        llm = _RecordsThread(extraction_data={"result": "1"}, default_response="r")
        llm.threads = set()
        before = threading.active_count()
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        assert _executable().run({"topic": "t"}, context=_context(llm)).success
        assert threading.active_count() == before
        assert llm.threads == {threading.current_thread().name}

    @pytest.mark.parametrize("owner", [FsmBodyExecutable, FsmInstance])
    def test_run_is_a_docstring_and_one_run_sync_statement(self, owner: type) -> None:
        tree = ast.parse(textwrap.dedent(inspect.getsource(owner.run)))
        body = tree.body[0].body  # type: ignore[attr-defined]
        assert isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
        assert len(body) == 2
        statement = body[1]
        assert isinstance(statement, ast.Return)
        call = statement.value
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        assert call.func.id == "run_sync"
        inner = call.args[0]
        assert isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
        assert inner.func.attr == "arun"


# ---- instantiation, setup, pins, the log ----------------------------------------------------------


class _Pins:
    def __init__(self) -> None:
        self.events: list[str] = []

    def pin(self, executable: Any) -> None:
        self.events.append("pin")

    def unpin(self, executable: Any) -> None:
        self.events.append("unpin")


class TestInstanceSetupPinsLog:
    def test_variables_fill_the_body_and_the_record_rebuilds_it(self) -> None:
        fsm = _fsm()
        fsm["states"]["a"]["response_instructions"] = "Greet the {{tone}} way."
        exe = _executable(fsm=fsm, variables=["tone"])
        with pytest.raises(InstantiationError, match="tone"):
            exe.instantiate()  # a declared variable has no value
        instance = exe.instantiate(variables={"tone": "kind"})
        assert instance.record == {"options": {}, "variables": {"tone": "kind"}}
        assert instance.instance_identity == definition_digest(instance.definition)
        again = exe.rebuild(instance.record, instance.instance_identity)
        assert again.instance_identity == instance.instance_identity
        assert again.definition == instance.definition
        with pytest.raises(InstanceIdentityMismatchError):
            exe.rebuild(
                {"options": {}, "variables": {"tone": "cold"}},
                instance.instance_identity,
            )
        result = instance.run({"topic": "t"}, context=_context())
        assert result.success
        assert result.identity.instance_identity == instance.instance_identity
        # the default run of a variable-declaring envelope is refused, not guessed
        with pytest.raises(InstantiationError):
            exe.run({"topic": "t"}, context=_context())

    def test_the_executable_keeps_no_reference_to_its_instance(self) -> None:
        import gc
        import weakref

        exe = _executable()
        instance = exe.instantiate()
        ref = weakref.ref(instance)
        del instance
        gc.collect()
        assert ref() is None
        assert not any(isinstance(v, FsmInstance) for v in vars(exe).values())

    def test_setup_runs_once_per_run_on_the_fresh_api_before_the_conversation(
        self,
    ) -> None:
        seen: list[tuple[API, int]] = []

        def setup(api: API) -> None:
            seen.append((api, len(api.list_active_conversations())))

        exe = FsmBodyExecutable(parse_executable(_doc()), setup=setup)
        assert exe.run({"topic": "t"}, context=_context()).success
        assert exe.run({"topic": "t"}, context=_context()).success
        assert len(seen) == 2 and seen[0][0] is not seen[1][0]
        assert [n for _, n in seen] == [0, 0]

    @pytest.mark.parametrize("outcome", ["ok", "provider", "bug", "cancel"])
    def test_pins_are_taken_around_the_run_and_always_released(
        self, outcome: str
    ) -> None:
        pins = _Pins()
        exe = FsmBodyExecutable(
            parse_executable(_doc(fsm=_long_fsm(), outputs=())), pins=pins
        )
        if outcome == "bug":
            llm: Any = _FailsOnSecondReply(RuntimeError("x"), default_response="r")
            with pytest.raises(FSMError):
                exe.run({"topic": "t"}, context=_context(llm))
        elif outcome == "provider":
            llm = _FailsOnSecondReply(LLMResponseError("x"), default_response="r")
            assert (
                exe.run({"topic": "t"}, context=_context(llm)).status is Status.FAILED
            )
        elif outcome == "cancel":
            token = CancelToken()
            token.cancel()
            assert (
                exe.run({"topic": "t"}, context=_context(cancel=token)).status
                is Status.ABORTED
            )
        else:
            assert exe.run({"topic": "t"}, context=_context(_llm())).success
        assert pins.events == ["pin", "unpin"]

    def test_the_event_log_of_the_context_receives_the_run(
        self, tmp_path: Path
    ) -> None:
        result = _executable().run({"topic": "t"}, context=_context(event_log=tmp_path))
        assert result.handle
        assert (tmp_path / "logs" / f"{result.handle}.jsonl").is_file()


# ---- the kinds namespace -----------------------------------------------------------------------


class TestKindsNamespace:
    def test_the_three_kinds_have_their_own_factories(self) -> None:
        names = {
            k: EXECUTABLE_KINDS.resolve(k) for k in ("fsm", "strategy", "procedure")
        }
        assert len({id(f) for f in names.values()}) == 3
        envelope = parse_executable(_doc("strategy"))
        assert isinstance(adapter_for("strategy")(envelope), FsmBodyExecutable)

    def test_a_kind_with_no_adapter_raises_a_named_error(self) -> None:
        from fsm_llm.core.names import NameRegistry

        empty: NameRegistry[Any] = NameRegistry("executable_kinds")
        with pytest.raises(ExecutableKindError, match="'procedure'"):
            adapter_for("procedure", empty)
        registry = ExecutableRegistry(kinds=empty)
        with pytest.raises(ExecutableKindError):
            registry.add_document(_doc())
