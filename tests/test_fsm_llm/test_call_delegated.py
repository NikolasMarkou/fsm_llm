"""The call executor, delegated branch (plan baa37f40, step K7.3, D-051, roadmap E3.2 second half).

A ``call`` state whose target a ``RunLauncher`` resolves (a code-backed pattern or workflow, not
an FSM body) runs OFF the turn: Pass 1 writes a ``running`` request that carries ``mode:
delegated``, a minted ``handle`` reserved in the launcher and a ``deadline``; on every later turn
the executor POLLS the launcher instead of skipping, and a finished run lands as ONE extraction
(the final record and the declared ``returns``). The executor makes no model call and starts no
thread: starting is settle's (K7.4, ``test_call_delegated_settle.py``). Every row here uses a
deterministic launcher double (no thread), so what is proven is the executor alone. Each row names
the mutant it kills (census D1, D3, D6).
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from fsm_llm import API, create_handler
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.definitions import FSMDefinition, FSMError, RunBudgetExceededError
from fsm_llm.core.results import Identity, RunResult
from fsm_llm.engine.api.delegated import RunMark
from fsm_llm.engine.kinds.call import final_record
from fsm_llm.engine.kinds.targets import (
    DEFAULT_DELEGATED_MAX_SECONDS,
    MAX_DELEGATED_MAX_SECONDS,
    MAX_DELEGATED_RETURN_CHARS,
    DelegatedTarget,
    ResolvedTarget,
    RunLauncher,
    RunPoll,
    delegated_seconds,
)
from fsm_llm.handlers import HandlerTiming
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm import kind_corpus as kc
from tests.test_fsm_llm.test_call_executor import _initial_call
from tests.test_fsm_llm.test_call_state import _call, _parent

_TARGET = "probe.run"
_NOW = 1_700_000_000.0
_HANDLE = re.compile(r"^run-[0-9a-f]{32}$")


def _child() -> dict[str, Any]:
    return kc._fsm(
        "Child",
        "c1",
        kc._state("c1", response_instructions="", transitions=[kc._go("c2")]),
        kc._state("c2", response_instructions=""),
    )


class _Launcher:
    """A resolver and a launcher in one object, deterministic, no thread.

    ``delegated`` ids resolve to a ``DelegatedTarget``; ``frames`` ids resolve to a frame target;
    ``polls`` maps a handle to the answer of ``poll`` (``None``: never seen); every call is
    recorded in ``events`` as a tuple."""

    def __init__(
        self,
        *,
        delegated: tuple[str, ...] = (_TARGET,),
        frames: tuple[str, ...] = (),
        full: bool = False,
        outputs: tuple[str, ...] = ("findings",),
    ) -> None:
        self.delegated = delegated
        self.frames = frames
        self.full = full
        self.outputs = outputs
        self.events: list[tuple[Any, ...]] = []
        self.polls: dict[str, RunPoll | None] = {}
        self.claimed: dict[str, str] = {}
        self.definition = FSMDefinition(**_child())

    def of(self, kind: str) -> list[tuple[Any, ...]]:
        return [e for e in self.events if e[0] == kind]

    def resolve_target(self, target_id, *, conversation_id=None):
        self.events.append(("resolve_target", target_id, conversation_id))
        if target_id not in self.frames:
            return None
        return ResolvedTarget(self.definition, f"template:{target_id}", "instance:x")

    def resolve_run(
        self, target_id, *, input_keys=(), max_steps=None, conversation_id=None
    ):
        self.events.append(
            ("resolve_run", target_id, tuple(input_keys), max_steps, conversation_id)
        )
        if target_id not in self.delegated:
            return None
        return DelegatedTarget(
            target_id,
            "pattern",
            object(),
            f"template:{target_id}",
            None,
            self.outputs,
        )

    def claim(self, handle, *, conversation_id):
        self.events.append(("claim", handle, conversation_id))
        if self.full:
            return False
        self.claimed[handle] = conversation_id
        return True

    def start(self, handle, target, inputs, **kwargs):
        """Accepts only a handle it reserved (as the host does); no thread, no announce."""
        if handle not in self.claimed:
            self.events.append(("start_refused", handle))
            return False
        self.events.append(("start", handle, dict(inputs)))
        return True

    def poll(self, handle, *, conversation_id):
        self.events.append(("poll", handle, conversation_id))
        return self.polls.get(handle)

    def cancel(self, handle, *, conversation_id=None):
        self.events.append(("cancel", handle, conversation_id))
        return True

    def cancel_for(self, conversation_id):
        self.events.append(("cancel_for", conversation_id))
        return 0


def _record_of(run_handle: str, **over: Any) -> dict[str, Any]:
    record = {
        "status": "running",
        "state": "work",
        "target": _TARGET,
        "template_identity": f"template:{_TARGET}",
        "instance_identity": None,
        "mode": "delegated",
        "handle": run_handle,
        "deadline": _NOW + 60.0,
        "request": {
            "target": _TARGET,
            "inputs": {"topic": "why"},
            "max_steps": None,
            "max_seconds": None,
        },
    }
    record.update(over)
    return record


async def _at_the_call(
    config: dict[str, Any] | None = None,
    *,
    launcher: _Launcher | None = None,
    arrive: bool = False,
    context: dict[str, Any] | None = None,
    handlers: list[Any] | None = None,
    llm: ScriptedLLM | None = None,
    **state_overrides: Any,
):
    """A parent conversation that begins in the call state ``work`` (the next step runs the
    executor); with ``arrive`` it begins in ``start``. Returns ``(api, cid, frame, launcher,
    llm)``."""
    launcher = _Launcher() if launcher is None else launcher
    llm = ScriptedLLM(default_response="r") if llm is None else llm
    state_overrides.setdefault("response_instructions", "")
    definition = _parent(
        _call(target=_TARGET, returns={"answer": "answer"})
        if config is None
        else config,
        **state_overrides,
    )
    api = API.from_definition(
        definition if arrive else _initial_call(definition),
        llm_interface=llm,
        targets=launcher,
        handlers=handlers or [],
    )
    cid, _ = await api.astart_conversation({"topic": "why", **(context or {})})
    frame = api.conversation_stacks[cid][0].conversation_id
    return api, cid, frame, launcher, llm


def _raw(api, frame) -> dict[str, Any]:
    return api.fsm_manager.copy_raw_context_data(frame)


def _seed(api, frame, record: dict[str, Any]) -> None:
    api.fsm_manager.update_conversation_context(frame, {"call_result": record})


def _final(record: dict[str, Any]) -> dict[str, Any]:
    """The projection every final record has (``CALL_SAMPLES``), without the stamp."""
    return {k: v for k, v in record.items() if k != "state"}


class TestTheConstants:
    def test_the_default_and_the_cap_are_named_and_the_rule_is_one_function(self):
        assert DEFAULT_DELEGATED_MAX_SECONDS == 600.0
        assert MAX_DELEGATED_MAX_SECONDS == 3600.0
        assert delegated_seconds(None) == DEFAULT_DELEGATED_MAX_SECONDS
        assert delegated_seconds(2.5) == 2.5
        assert delegated_seconds(10**9) == MAX_DELEGATED_MAX_SECONDS
        assert MAX_DELEGATED_RETURN_CHARS == 8_000

    def test_the_launcher_double_is_a_launcher(self):
        assert isinstance(_Launcher(), RunLauncher)


class TestTheRequest:
    async def test_a_delegated_request_is_a_running_record_with_a_handle_and_a_deadline(
        self,
    ):
        """Kills: a record with another shape, no ``mode``, no ``handle`` or no ``deadline``."""
        launcher = _Launcher()
        api, cid, frame, _, _ = await _at_the_call(
            _call(target=_TARGET, returns={"answer": "answer"}, max_seconds=30),
            launcher=launcher,
        )
        with patch(
            "fsm_llm.engine.kinds.call.time", SimpleNamespace(time=lambda: _NOW)
        ):
            await api.aadvance(cid)
        record = _raw(api, frame)["call_result"]
        assert _HANDLE.match(record["handle"])
        assert record == {
            "status": "running",
            "state": "work",
            "target": _TARGET,
            "template_identity": f"template:{_TARGET}",
            "instance_identity": None,
            "mode": "delegated",
            "handle": record["handle"],
            "deadline": _NOW + 30,
            "request": {
                "target": _TARGET,
                "inputs": {"topic": "why"},
                "max_steps": None,
                "max_seconds": 30,
            },
        }
        # the call stays in its state: no edge passes while the status is `running`
        assert api.fsm_manager.get_conversation_state(frame) == "work"

    async def test_the_handle_is_reserved_for_the_frame_in_pass_1(self):
        """The reservation is the executor's: one ``claim`` for the frame id, the handle of the
        record, and nothing started. Kills: a handle never reserved (settle would find none)."""
        api, cid, frame, launcher, _ = await _at_the_call(
            _call(target=_TARGET, returns={"answer": "answer"})
        )
        await api.aadvance(cid)
        record = _raw(api, frame)["call_result"]
        assert launcher.of("claim") == [("claim", record["handle"], frame)]
        assert launcher.of("resolve_run")[0] == (
            "resolve_run",
            _TARGET,
            ("topic",),
            None,
            frame,
        )
        # the end settle of the same turn started it, with the declared inputs only
        assert launcher.of("start") == [("start", record["handle"], {"topic": "why"})]

    @pytest.mark.parametrize(
        ("given", "want"),
        [
            (None, DEFAULT_DELEGATED_MAX_SECONDS),
            (2.5, 2.5),
            (10**7, MAX_DELEGATED_MAX_SECONDS),
        ],
        ids=["default", "given", "capped"],
    )
    async def test_the_deadline_has_a_default_and_a_cap(self, given, want):
        """Kills: a delegated call with no bound, or one the author can lift past the cap."""
        extra = {} if given is None else {"max_seconds": given}
        api, cid, frame, *_ = await _at_the_call(
            _call(target=_TARGET, returns={"answer": "answer"}, **extra)
        )
        with patch(
            "fsm_llm.engine.kinds.call.time", SimpleNamespace(time=lambda: _NOW)
        ):
            await api.aadvance(cid)
        assert _raw(api, frame)["call_result"]["deadline"] == _NOW + want

    async def test_a_full_host_refuses_with_push_failed_in_the_same_turn(self):
        api, cid, frame, _launcher, _ = await _at_the_call(
            _call(target=_TARGET, returns={"answer": "answer"}),
            launcher=_Launcher(full=True),
        )
        step = await api.aadvance(cid)
        record = _raw(api, frame)["call_result"]
        assert record["status"] == "failed" and record["reason"] == "push_failed"
        assert record["target"] == _TARGET
        assert "handle" not in record and "mode" not in record
        assert step.state_after == "apology"  # the failure edge fired in the same turn

    async def test_a_frame_target_never_asks_the_delegated_side(self):
        """The frame path is unchanged: an id the frame resolver answers is a frame request."""
        launcher = _Launcher(delegated=(_TARGET,), frames=(_TARGET,))
        api, cid, frame, *_ = await _at_the_call(
            _call(target=_TARGET, returns={"answer": "answer"}), launcher=launcher
        )
        await api.aadvance(cid)
        assert launcher.of("resolve_run") == [] and launcher.of("claim") == []
        assert _raw(api, frame)["call_result"]["mode"] == "frame"

    async def test_a_target_nobody_resolves_is_unknown_target_and_reserves_nothing(
        self,
    ):
        launcher = _Launcher(delegated=())
        api, cid, frame, *_ = await _at_the_call(
            _call(target=_TARGET, returns={"answer": "answer"}), launcher=launcher
        )
        step = await api.aadvance(cid)
        record = _raw(api, frame)["call_result"]
        assert (record["status"], record["reason"]) == ("failed", "unknown_target")
        assert launcher.of("claim") == [] and step.state_after == "apology"

    async def test_a_missing_input_refuses_before_any_reservation(self):
        """Kills: a reservation made before the inputs are known (it would leak per refusal)."""
        api, cid, frame, launcher, _ = await _at_the_call(
            _call(target=_TARGET, inputs=["absent"], returns={"answer": "answer"})
        )
        await api.aadvance(cid)
        record = _raw(api, frame)["call_result"]
        assert (record["status"], record["reason"]) == ("failed", "inputs_missing")
        assert record["template_identity"] == f"template:{_TARGET}"
        assert launcher.of("claim") == []

    async def test_the_request_holds_only_the_declared_inputs(self):
        """Kills (D1, D3): a request built from the whole parent context. The parent holds an
        internal grant, a secret-looking entry and plain extras; only the declared name is in the
        record."""
        api, cid, frame, *_ = await _at_the_call(
            _call(target=_TARGET, returns={"answer": "answer"}),
            context={
                "_approval_granted": {"call": "x"},
                "api_key": "sk-secret-value",
                "unrelated": "do not copy me",
            },
        )
        await api.aadvance(cid)
        record = _raw(api, frame)["call_result"]
        assert set(record["request"]["inputs"]) == {"topic"}
        for leaked in ("_approval_granted", "sk-secret-value", "do not copy me"):
            assert leaked not in repr(record)

    async def test_the_executor_makes_no_model_call(self):
        api, cid, _frame, _, llm = await _at_the_call(
            _call(target=_TARGET, returns={"answer": "answer"})
        )
        before = list(llm.call_history)
        await api.aadvance(cid)
        assert list(llm.call_history) == before

    async def test_the_request_is_written_on_arrival_too(self):
        """The state is ENTERED by a transition (``aafter_transition``): the same record."""
        api, cid, frame, launcher, _ = await _at_the_call(
            _call(target=_TARGET, returns={"answer": "answer"}), arrive=True
        )
        await api.aadvance(cid)  # start -> work: the arrival decides
        record = _raw(api, frame)["call_result"]
        assert record["mode"] == "delegated" and _HANDLE.match(record["handle"])
        assert [e[1] for e in launcher.of("claim")] == [record["handle"]]

    async def test_a_pass_2_failure_rolls_the_record_back(self):
        """The record is turn data like every Pass-1 write: a failed turn leaves none."""
        llm = ScriptedLLM(default_response="r")
        api, cid, frame, *_ = await _at_the_call(
            _call(target=_TARGET, returns={"answer": "answer"}),
            llm=llm,
            response_instructions="Say something",
        )

        async def boom(request):
            raise FSMError("provider is down")

        with patch.object(llm, "agenerate_response", boom):
            with pytest.raises(FSMError):
                await api.aadvance(cid)
        assert "call_result" not in _raw(api, frame)

    async def test_a_resolver_that_is_no_launcher_never_delegates(self):
        class _FrameOnly:
            def resolve_target(self, target_id, *, conversation_id=None):
                return None

        api = API.from_definition(
            _initial_call(_parent(_call(target=_TARGET), response_instructions="")),
            llm_interface=ScriptedLLM(default_response="r"),
            targets=_FrameOnly(),
        )
        cid, _ = await api.astart_conversation({"topic": "why"})
        frame = api.conversation_stacks[cid][0].conversation_id
        await api.aadvance(cid)
        assert _raw(api, frame)["call_result"]["reason"] == "unknown_target"


async def _polling(handle="run-" + "a" * 32, **kw):
    """A parent in the call state with a ``running`` delegated record of ``handle`` seeded."""
    api, cid, frame, launcher, _llm = await _at_the_call(**kw)
    _seed(api, frame, _record_of(handle))
    # the run counts as STARTED (settle leaves it to the executor's poll): these rows are about
    # the executor; what settle does with a record is ``test_call_delegated_settle.py``
    api._call_marks.runs[handle] = RunMark(cid, frame, "work", "call_result", _TARGET)
    return api, cid, frame, launcher, handle


class TestThePoll:
    async def test_running_before_the_deadline_writes_nothing(self):
        """Kills: a poll that rewrites the record or resolves again every turn."""
        api, cid, frame, launcher, handle = await _polling()
        launcher.polls[handle] = RunPoll(False, None)
        before = _raw(api, frame)["call_result"]
        with patch(
            "fsm_llm.engine.kinds.call.time", SimpleNamespace(time=lambda: _NOW)
        ):
            step = await api.aadvance(cid)
        assert _raw(api, frame)["call_result"] == before
        assert step.state_after == "work"
        assert launcher.of("poll")[0] == ("poll", handle, frame)
        assert launcher.of("claim") == [] and launcher.of("resolve_target") == []
        assert launcher.of("cancel") == []

    async def test_a_finished_run_lands_once_with_the_final_record_and_the_returns(
        self,
    ):
        """ONE extraction: the final ``call_result`` and the declared ``returns`` reach the
        context in one CONTEXT_UPDATE pass. Kills: two writes (a window with the returns merged
        and the record still ``running``)."""
        seen: list[set[str]] = []
        watch = (
            create_handler("watch")
            .at(HandlerTiming.CONTEXT_UPDATE)
            .when_keys_updated("answer", "call_result")
            .do(lambda ctx: seen.append(set(ctx)) or {})
        )
        api, cid, frame, launcher, handle = await _polling(handlers=[watch])
        launcher.polls[handle] = RunPoll(
            True, RunResult(status=Status.OK, answer="the answer")
        )
        before = len(seen)
        step = await api.aadvance(cid)
        raw = _raw(api, frame)
        assert len(seen) - before == 1
        assert raw["answer"] == "the answer"
        assert _final(raw["call_result"]) == {
            "status": "ok",
            "reason": None,
            "target": _TARGET,
            "template_identity": f"template:{_TARGET}",
            "instance_identity": None,
        }
        assert raw["call_result"]["state"] == "work"
        assert step.state_after == "done"  # the success edge fired in the same turn

    async def test_a_declared_output_wins_over_the_pseudo_output_answer(self):
        api, cid, frame, launcher, handle = await _polling(
            config=_call(
                target=_TARGET, returns={"found": "findings", "text": "answer"}
            )
        )
        launcher.polls[handle] = RunPoll(
            True,
            RunResult(status=Status.OK, outputs={"findings": [1, 2]}, answer="prose"),
        )
        await api.aadvance(cid)
        raw = _raw(api, frame)
        assert raw["found"] == [1, 2] and raw["text"] == "prose"

    async def test_a_missing_output_fails_the_call_and_merges_nothing(self):
        """Kills: a partial merge (``returns`` are every-entry-required, as in frame mode)."""
        api, cid, frame, launcher, handle = await _polling(
            config=_call(
                target=_TARGET, returns={"found": "findings", "text": "answer"}
            )
        )
        launcher.polls[handle] = RunPoll(
            True, RunResult(status=Status.OK, answer="prose")
        )
        step = await api.aadvance(cid)
        raw = _raw(api, frame)
        assert (raw["call_result"]["status"], raw["call_result"]["reason"]) == (
            "failed",
            "return_keys_missing",
        )
        assert "found" not in raw and "text" not in raw
        assert step.state_after == "apology"

    async def test_a_call_that_declares_no_returns_merges_nothing(self):
        api, cid, frame, launcher, handle = await _polling(
            config=_call(target=_TARGET, returns={})
        )
        launcher.polls[handle] = RunPoll(
            True, RunResult(status=Status.OK, outputs={"findings": 1}, answer="x")
        )
        step = await api.aadvance(cid)
        raw = _raw(api, frame)
        assert raw["call_result"]["status"] == "ok" and step.state_after == "done"
        assert "findings" not in raw and "answer" not in raw

    @pytest.mark.parametrize(
        ("result", "want"),
        [
            (
                RunResult(status=Status.FAILED, reason=Reason.UNEXPECTED_ERROR),
                ("failed", "unexpected_error"),
            ),
            (
                RunResult(status=Status.FAILED, reason=Reason.PROVIDER_ERROR),
                ("failed", "provider_error"),
            ),
            (RunResult(status=Status.FAILED), ("failed", "unexpected_error")),
            (
                RunResult(status=Status.ABORTED, reason=Reason.CANCELLED),
                ("aborted", "cancelled"),
            ),
            (RunResult(status=Status.ABORTED), ("aborted", "budget_spent")),
            (RunResult(status=Status.ABSTAINED), ("failed", "unexpected_error")),
        ],
        ids=[
            "worker-error",
            "provider",
            "failed-no-reason",
            "cancelled",
            "aborted-no-reason",
            "other-status",
        ],
    )
    async def test_a_run_that_did_not_end_ok_keeps_its_status_and_reason(
        self, result, want
    ):
        api, cid, frame, launcher, handle = await _polling()
        launcher.polls[handle] = RunPoll(True, result)
        step = await api.aadvance(cid)
        record = _raw(api, frame)["call_result"]
        assert (record["status"], record["reason"]) == want
        assert "answer" not in _raw(api, frame)  # nothing merged on a non-ok end
        assert step.state_after == "apology"

    async def test_an_exception_text_never_reaches_the_record(self):
        """The host stores the TYPE of a worker error; whatever the result detail says never
        reaches the context (census D6)."""
        api, cid, frame, launcher, handle = await _polling()
        launcher.polls[handle] = RunPoll(
            True,
            RunResult(
                status=Status.FAILED,
                reason=Reason.UNEXPECTED_ERROR,
                detail={"error": "ValueError"},
                answer="Traceback: secret sk-123",
            ),
        )
        await api.aadvance(cid)
        assert "sk-123" not in repr(_raw(api, frame))
        assert "Traceback" not in repr(_raw(api, frame))

    async def test_past_the_deadline_the_run_is_cancelled_and_the_call_aborts(self):
        """Kills: a call that waits for ever on a run that does not end."""
        api, cid, frame, launcher, handle = await _polling()
        launcher.polls[handle] = RunPoll(False, None)
        with patch(
            "fsm_llm.engine.kinds.call.time",
            SimpleNamespace(time=lambda: _NOW + 61.0),
        ):
            step = await api.aadvance(cid)
        record = _raw(api, frame)["call_result"]
        assert (record["status"], record["reason"]) == ("aborted", "budget_spent")
        assert launcher.of("cancel") == [("cancel", handle, frame)]
        assert step.state_after == "apology"

    async def test_a_finished_run_wins_over_a_passed_deadline(self):
        api, cid, frame, launcher, handle = await _polling()
        launcher.polls[handle] = RunPoll(
            True, RunResult(status=Status.OK, answer="late")
        )
        with patch(
            "fsm_llm.engine.kinds.call.time",
            SimpleNamespace(time=lambda: _NOW + 10_000.0),
        ):
            await api.aadvance(cid)
        assert _raw(api, frame)["call_result"]["status"] == "ok"
        assert launcher.of("cancel") == []

    @pytest.mark.parametrize(
        "deadline",
        [None, "soon", float("nan"), True, [1], math.inf * -1],
        ids=["none", "text", "nan", "bool", "list", "minus-inf"],
    )
    async def test_a_deadline_that_is_no_number_fails_closed(self, deadline):
        """Kills: a forged record whose deadline never comes (a run that never stops)."""
        api, cid, frame, launcher, handle = await _polling()
        _seed(api, frame, _record_of(handle, deadline=deadline))
        launcher.polls[handle] = RunPoll(False, None)
        await api.aadvance(cid)
        record = _raw(api, frame)["call_result"]
        assert (record["status"], record["reason"]) == ("aborted", "budget_spent")

    async def test_a_handle_the_host_never_saw_is_run_lost(self):
        """A crash, a restore, a fork or a forged record: nothing is re-run."""
        api, cid, frame, launcher, _handle = await _polling()
        launcher.polls.clear()  # poll answers None
        step = await api.aadvance(cid)
        record = _raw(api, frame)["call_result"]
        assert (record["status"], record["reason"]) == ("failed", "run_lost")
        assert launcher.of("claim") == [] and launcher.of("start") == []
        assert step.state_after == "apology"

    @pytest.mark.parametrize("handle", [None, "", "   ", 5, ["x"], "h" * 129])
    async def test_a_record_with_no_usable_handle_is_run_lost(self, handle):
        api, cid, frame, _launcher, _ = await _polling()
        _seed(api, frame, _record_of("ignored", handle=handle))
        await api.aadvance(cid)
        assert _raw(api, frame)["call_result"]["reason"] == "run_lost"

    async def test_the_poll_is_scoped_to_the_frame_that_asks(self):
        """The launcher answers per conversation; the executor passes ITS frame id. Kills: an
        unscoped poll (a forged record that copies another conversation's handle adopts its
        result)."""
        api, cid, frame, launcher, handle = await _polling()
        launcher.polls[handle] = RunPoll(False, None)
        await api.aadvance(cid)
        assert {e[2] for e in launcher.of("poll")} == {frame}

    async def test_the_finished_identities_come_from_the_parents_own_state(self):
        """Kills (D3): identities (or a target) copied from a forged record."""
        api, cid, frame, launcher, handle = await _polling()
        _seed(
            api,
            frame,
            _record_of(
                handle,
                target="evil.id",
                template_identity="forged",
                instance_identity="forged",
            ),
        )
        launcher.polls[handle] = RunPoll(True, RunResult(status=Status.OK, answer="a"))
        await api.aadvance(cid)
        record = _raw(api, frame)["call_result"]
        assert record["target"] == _TARGET
        assert record["template_identity"] == f"template:{_TARGET}"
        assert record["instance_identity"] is None
        assert launcher.of("resolve_run")[-1][1] == _TARGET

    async def test_a_target_that_no_longer_resolves_leaves_the_identities_none(self):
        api, cid, frame, launcher, handle = await _polling()
        launcher.delegated = ()
        launcher.polls[handle] = RunPoll(True, RunResult(status=Status.OK, answer="a"))
        await api.aadvance(cid)
        record = _raw(api, frame)["call_result"]
        assert record["status"] == "ok"
        assert (record["template_identity"], record["instance_identity"]) == (
            None,
            None,
        )

    async def test_a_final_record_stands_and_is_not_polled_again(self):
        api, cid, frame, launcher, handle = await _polling()
        _seed(api, frame, _record_of(handle, status="ok", reason=None))
        await api.aadvance(cid)
        assert launcher.of("poll") == []

    async def test_a_running_frame_record_is_still_skipped_not_polled(self):
        """The frame request keeps its skip-if-set: only ``mode: delegated`` polls."""
        api, cid, frame, launcher, handle = await _polling()
        _seed(api, frame, _record_of(handle, mode="frame"))
        await api.aadvance(cid)
        assert launcher.of("poll") == []

    async def test_settle_leaves_a_started_delegated_record_alone(self):
        """The delegated record is not an unknown mode: settle neither refuses it nor pushes
        it. Kills: ``failed/unknown_target`` over a delegated request."""
        api, cid, frame, launcher, handle = await _polling()
        launcher.polls[handle] = RunPoll(False, None)
        with patch(
            "fsm_llm.engine.kinds.call.time", SimpleNamespace(time=lambda: _NOW)
        ):
            await api.aadvance(cid)
            await api.aadvance(cid)
        assert _raw(api, frame)["call_result"]["status"] == "running"
        assert api.get_stack_depth(cid) == 1

    async def test_a_run_loop_over_a_polling_parent_spends_a_step_per_poll(self):
        """Known limit (written, D-016): a delegated parent costs one step per poll, so the
        loop's own budget ends a run that never finishes."""
        api, cid, _frame, launcher, handle = await _polling()
        launcher.polls[handle] = RunPoll(False, None)
        with patch(
            "fsm_llm.engine.kinds.call.time", SimpleNamespace(time=lambda: _NOW)
        ):
            with pytest.raises(RunBudgetExceededError):
                await api.arun_until_terminal(cid, max_steps=3)
        assert len(launcher.of("poll")) == 3


class TestTheReturnsAreBoundedAndJsonSafe:
    async def _finish(self, value, config=None):
        api, cid, frame, launcher, handle = await _polling(config=config)
        launcher.polls[handle] = RunPoll(
            True, RunResult(status=Status.OK, outputs={"findings": value})
        )
        await api.aadvance(cid)
        return _raw(api, frame)

    async def test_a_long_text_is_cut_at_the_cap_with_a_marker(self):
        raw = await self._finish(
            "x" * (MAX_DELEGATED_RETURN_CHARS * 3),
            _call(target=_TARGET, returns={"found": "findings"}),
        )
        shown = raw["found"]
        assert len(shown) < MAX_DELEGATED_RETURN_CHARS + 200
        assert shown.startswith("x" * MAX_DELEGATED_RETURN_CHARS)
        assert "cut" in shown[MAX_DELEGATED_RETURN_CHARS:]

    async def test_a_container_over_the_cap_is_a_marker_naming_its_type_and_size(self):
        big = [str(n) * 20 for n in range(5000)]
        raw = await self._finish(
            big, _call(target=_TARGET, returns={"found": "findings"})
        )
        assert isinstance(raw["found"], str) and raw["found"].startswith("<list, ")
        assert len(raw["found"]) < 200

    async def test_a_value_json_cannot_hold_becomes_a_marker(self):
        raw = await self._finish(
            {"a": {1, 2}, "b": float("nan")},
            _call(target=_TARGET, returns={"found": "findings"}),
        )
        assert raw["found"] == {"a": "<set>", "b": "<float:nan>"}

    async def test_a_very_deep_value_does_not_raise_and_is_cut(self):
        """Walked iteratively with a depth cap (no recursive dump: a small Windows stack).
        Kills: a recursive serializer (``RecursionError`` on 100000 levels)."""
        deep: Any = "leaf"
        for _ in range(100_000):
            deep = [deep]
        raw = await self._finish(
            deep, _call(target=_TARGET, returns={"found": "findings"})
        )
        assert raw["found"] is not None
        assert "<truncated:depth>" in repr(raw["found"])


class TestTheSharedFinalRecord:
    def test_the_final_record_is_one_function_with_the_pinned_projection(self):
        record = final_record("work", Status.FAILED, Reason.RUN_LOST, "t", "tt", "ii")
        assert record == {
            "status": "failed",
            "state": "work",
            "reason": "run_lost",
            "target": "t",
            "template_identity": "tt",
            "instance_identity": "ii",
        }
        assert final_record("work", Status.OK, None, None, None, None)["reason"] is None

    def test_the_identity_type_is_still_the_envelope_of_a_result(self):
        assert Identity().template_identity is None


class TestTheSourceOfTheRules:
    def test_the_executor_imports_no_api_executables_or_context(self):
        import ast
        from pathlib import Path

        import fsm_llm

        source = (
            Path(fsm_llm.__file__).parent / "engine" / "kinds" / "call.py"
        ).read_text(encoding="utf-8")
        names = {
            node.module or ""
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
        }
        names |= {
            alias.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        banned = (
            "engine.api",
            "engine.executables",
            "engine.context",
            "engine.turn",
            "engine.fsm",
        )
        assert not [n for n in names if any(b in n for b in banned)]
