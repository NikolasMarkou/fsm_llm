"""
What a synthesis driver must do, on a real ``API`` (the runner rows of ``test_runner.py`` that do
not pin the runner's own bookkeeping, re-expressed on the rig of ``rig.py``).

Every row runs on each entry of ``rig.DRIVERS`` (the ``_driver`` fixture below), through the call
surface of a driver: ``run``, ``converse`` and the results a parent sees (``synthesis_result``, the
state and depth, the trace events). Ported, not dropped: the ok path, abort at ``max_turns``, an
outage is no fault of the procedure, return keys, a reason per failure, nesting, the trace events
being value free. What stays on ``test_runner.py``: ``_runs``, ``_refused``, ``end_conversation``
markers, ``hook``, the stream settle on close, the runner's own push, a signal during the compile.
"""

import copy
import dataclasses
import json
from typing import Any

import pytest

from fsm_llm import HandlerTiming
from fsm_llm.core.definitions import (
    CompletionStateConfig,
    FSMDefinition,
    FSMError,
    LLMResponseError,
    LLMUsage,
    RunBudgetExceededError,
)
from fsm_llm.llm.usage import _meter_of
from fsm_llm.llm.usage import _Metered as _RealMetered
from fsm_llm.logging import logger
from fsm_llm.runtime import EventLog
from fsm_llm.synthesis import (
    LibraryError,
    Procedure,
    RunTrace,
    SynthesisError,
    SynthesisPolicy,
    SynthesisRefusedError,
    TraceRecorder,
    diagnose,
    digest_of,
    strip_abort,
)
from fsm_llm.synthesis import procedures as procedures_module
from fsm_llm.synthesis.gates import GATES
from tests.test_fsm_llm_synthesis import rig as rig_module
from tests.test_fsm_llm_synthesis.conftest import ScriptedRunnerLLM
from tests.test_fsm_llm_synthesis.rig import (
    CANARIES,
    CONTEXT,
    DRIVERS,
    GOAL,
    NOISE,
    PERSONA,
    REPLIES,
    SECRET,
    Metered,
    Rig,
    make_procedure,
    make_rig,
    serve_procedure,
)


@pytest.fixture(autouse=True, params=sorted(DRIVERS))
def _driver(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run every row of this file on each driver of the rig."""
    monkeypatch.setattr(rig_module, "DEFAULT_DRIVER", request.param)
    return request.param


class _Ledgered(_RealMetered, ScriptedRunnerLLM):
    """A cheap model with a REAL ledger (``_Metered.usage``): the first ``outages`` extractions
    fail and are counted as errors, as ``LiteLLMInterface`` counts a failed request."""

    def __init__(self, outages: int, **kw: Any) -> None:
        super().__init__(**kw)
        self.outages, self.failed = outages, 0

    def extract_field(self, request: Any) -> Any:
        if self.failed < self.outages:
            self.failed += 1
            _meter_of(self).record_error("extract")
            raise LLMResponseError("provider timed out")
        return super().extract_field(request)


class _Meterless(ScriptedRunnerLLM):
    """A cheap model with no ``usage()`` at all, whose extractions always fail."""

    usage = None  # type: ignore[assignment]

    def extract_field(self, request: Any) -> Any:
        raise LLMResponseError("provider timed out")


def _two_key(valid_fsm: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(valid_fsm)
    ask = data["states"]["ask"]
    ask["field_extractions"].append(
        {"field_name": "extra", "field_type": "str", "extraction_instructions": "x"}
    )
    ask["transitions"][0]["conditions"][0]["requires_context_keys"] = [
        "answer",
        "extra",
    ]
    return data


def _nesting() -> dict[str, Any]:
    """A procedure whose first state is itself a synthesize state; it hands back
    its input ``topic`` (G3 cannot see what a nested procedure returns)."""
    return {
        "name": "Nest",
        "description": "Nests",
        "initial_state": "nest",
        "states": {
            "nest": {
                "id": "nest",
                "description": "nest",
                "purpose": "nest",
                "response_instructions": "",
                "synthesize": {
                    "goal": GOAL,
                    "inputs": ["topic"],
                    "return_keys": ["topic"],
                    "on_failure": "nest_end",
                    # not the reserved default: a compiled procedure may not
                    # read `synthesis_result` (G5), and its edge must read one
                    "result_key": "nested_result",
                },
                "transitions": [
                    {
                        "target_state": "nest_end",
                        "description": "the nested run settled",
                        "priority": 1,
                        "conditions": [
                            {
                                "description": "a result exists",
                                "requires_context_keys": ["nested_result"],
                                "logic": {
                                    "in": [
                                        {"var": "nested_result.status"},
                                        ["ok", "failed", "aborted"],
                                    ]
                                },
                            }
                        ],
                    }
                ],
            },
            "nest_end": {
                "id": "nest_end",
                "description": "end",
                "purpose": "end",
                "response_instructions": "",
            },
        },
    }


def _classified(valid_fsm: dict[str, Any], **entry: Any) -> dict[str, Any]:
    """``valid_fsm`` with one classification entry (a model setting in ``entry``)."""
    out = copy.deepcopy(valid_fsm)
    out["states"]["ask"]["classification_extractions"] = [
        {
            "field_name": "mood",
            "intents": [
                {"name": "happy", "description": "glad"},
                {"name": "sad", "description": "unhappy"},
            ],
            "fallback_intent": "happy",
            **entry,
        }
    ]
    return out


class TestOkPath:
    def test_run_until_terminal_pushes_runs_pops_and_merges(self, valid_fsm, tmp_path):
        rig = make_rig(valid_fsm, tmp_path)

        results = rig.run(20)

        assert results[-1].ended
        assert (rig.state, rig.depth) == ("booked", 1)
        assert rig.data["answer"] == "42"
        assert rig.result == {
            "status": "ok",
            "procedure_id": rig.digest,
            "version": "1.0",
            "reason": None,
        }
        assert len(rig.strong.requests) == 1
        assert rig.names == ["obtained", "pushed", "popped"]
        assert rig.events[2][1]["status"] == "ok"
        assert rig.library is not None
        assert rig.library.get_record(rig.digest).stats == {"runs": 1, "successes": 1}

    def test_the_child_runs_on_the_cheap_model_with_the_host_persona(self, valid_fsm):
        rig = make_rig(valid_fsm)
        rig.run(20)

        child_prompts = rig.cheap.shown("generate_response", "ask")
        assert "<persona>" in child_prompts and PERSONA in child_prompts
        assert rig.extractions() == 1

    def test_the_child_sees_no_parent_data(self, valid_fsm):
        rig = make_rig(valid_fsm)
        rig.run(20)

        # control: the parent's own prompt does carry its noise
        assert NOISE in rig.cheap.shown("generate_response", "start")
        for shown in (
            rig.cheap.shown("extract_field"),
            rig.cheap.shown("generate_response", "ask"),
            rig.cheap.shown("generate_response", "done"),
        ):
            assert shown and NOISE not in shown and "hunter2" not in shown


class TestConverse:
    def test_user_turns_drive_the_child_then_the_parent_moves_on(
        self, valid_fsm, tmp_path
    ):
        rig = make_rig(valid_fsm, tmp_path)

        # the parent's solve state is silent: the first reply is the child's
        assert rig.converse("hello") == "What is the answer?"
        assert (rig.state, rig.depth) == ("ask", 2)

        # the child ends on this turn, is popped, and the parent steps on
        assert rig.converse("forty-two") == "Thanks\nBooked!"
        assert (rig.state, rig.depth) == ("booked", 1)
        assert rig.data["answer"] == "42"
        assert rig.result["status"] == "ok"
        assert rig.names == ["obtained", "pushed", "popped"]
        assert rig.library is not None
        assert rig.library.get_record(rig.digest).stats["runs"] == 1

    def test_a_child_that_never_finishes_is_aborted_after_max_turns(self, valid_fsm):
        rig = make_rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=2))

        rig.converse("hello")
        rig.converse("one")
        assert rig.depth == 2 and "aborted" not in rig.names
        reply = rig.converse("two")

        assert (rig.state, rig.depth) == ("apology", 1)
        # the child's own reply to the last turn, then the parent's apology
        assert reply == "What is the answer?\nSorry"
        assert rig.result["status"] == "aborted"
        assert rig.result["reason"] == "budget_spent"
        assert "answer" not in rig.data
        assert rig.names == ["obtained", "pushed", "aborted", "popped"]


class TestAbort:
    def test_a_silent_child_is_aborted_after_max_turns_and_the_parent_apologises(
        self, valid_fsm, tmp_path
    ):
        rig = make_rig(
            valid_fsm,
            tmp_path,
            extraction={},
            policy=SynthesisPolicy(max_turns=3),
        )

        results = rig.run(30)

        assert results[-1].ended
        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.result == {
            "status": "aborted",
            "procedure_id": rig.digest,
            "version": "1.0",
            "reason": "budget_spent",
        }
        assert "answer" not in rig.data
        # the greeting and three child steps of work reply as "ask"; every child
        # step extracts twice (one retry). The engine stops the child at its third
        # turn: no fourth turn runs to reach an abort edge (it used to, so this
        # count was (3 + 1) * 2)
        assert rig.cheap.count("generate_response", "ask") == 1 + 3
        assert rig.extractions() == 3 * 2
        assert rig.names == ["obtained", "pushed", "aborted", "popped"]
        assert rig.events[2][1]["step"] == 5
        assert rig.library is not None
        assert rig.library.get_record(rig.digest).stats == {"runs": 1, "successes": 0}

    def test_a_resumed_run_still_aborts_the_child_after_max_turns_in_total(
        self, valid_fsm
    ):
        """Core restarts its step number at 1 on every ``run`` call; the runner's
        own count of child steps must not restart with it (review concern 5)."""
        rig = make_rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=5))

        for _ in range(10):
            try:
                rig.run(3)
            except RunBudgetExceededError:
                continue
            break

        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.result["status"] == "aborted"
        assert rig.names == ["obtained", "pushed", "aborted", "popped"]
        # the greeting plus exactly max_turns child steps, however the run was cut
        assert rig.cheap.count("generate_response", "ask") == 1 + 5


class TestAnOutageIsNoFault:
    """A run that spent an outage never ends on a reason ``diagnose`` can patch
    (D-061): core swallows an extraction error, so the child just stays blocked."""

    def _run(
        self, valid_fsm: dict[str, Any], tmp_path: Any, cheap: ScriptedRunnerLLM
    ) -> Rig:
        self.traces: list[RunTrace] = []
        recorder = TraceRecorder(self.traces.append)
        rig = make_rig(
            valid_fsm,
            tmp_path,
            cheap=cheap,
            policy=SynthesisPolicy(max_turns=3),
            sink=recorder,
        )
        rig.run(30)
        return rig

    def test_a_stuck_child_with_an_outage_is_provider_error_not_budget_spent(
        self, valid_fsm, tmp_path
    ):
        cheap = Metered(10_000, extraction_data={"answer": "42"}, responses=REPLIES)
        rig = self._run(valid_fsm, tmp_path, cheap)

        assert rig.result["status"] == "aborted"
        assert rig.result["reason"] == "provider_error"
        (trace,) = self.traces
        assert (trace.status, trace.reason) == ("aborted", "provider_error")
        assert (trace.stuck, trace.outcome) == ("ask", "blocked")  # a fault's shape

    def test_the_same_run_without_an_outage_stays_budget_spent(
        self, valid_fsm, tmp_path
    ):
        rig = self._run(
            valid_fsm, tmp_path, Metered(0, extraction_data={}, responses=REPLIES)
        )

        assert rig.result["reason"] == "budget_spent"
        assert self.traces[0].reason == "budget_spent"

    def test_errors_from_before_the_push_are_not_the_childs_outage(
        self, valid_fsm, tmp_path
    ):
        cheap = Metered(0, extraction_data={}, responses=REPLIES)
        cheap.errors = 5  # the meter already counts failures of earlier work
        rig = self._run(valid_fsm, tmp_path, cheap)

        assert rig.result["reason"] == "budget_spent"

    def test_a_retried_call_that_then_succeeds_is_not_a_fault_of_the_run(
        self, valid_fsm, tmp_path
    ):
        cheap = Metered(1, extraction_data={"answer": "42"}, responses=REPLIES)
        rig = self._run(valid_fsm, tmp_path, cheap)

        assert cheap.errors == 1
        assert rig.result["status"] == "ok" and rig.result["reason"] is None

    def test_an_outage_run_counts_no_run_in_the_library(self, valid_fsm, tmp_path):
        cheap = Metered(10_000, extraction_data={"answer": "42"}, responses=REPLIES)
        rig = self._run(valid_fsm, tmp_path, cheap)

        assert rig.result["reason"] == "provider_error"
        assert rig.library is not None
        stats = rig.library.get_record(self.traces[0].digest).stats
        assert stats == {"runs": 0, "successes": 0}  # no fault of the procedure

    def test_a_retried_call_that_then_succeeds_still_counts_an_ok_run(
        self, valid_fsm, tmp_path
    ):
        cheap = Metered(1, extraction_data={"answer": "42"}, responses=REPLIES)
        rig = self._run(valid_fsm, tmp_path, cheap)

        assert rig.library is not None
        stats = rig.library.get_record(self.traces[0].digest).stats
        assert stats == {"runs": 1, "successes": 1}

    def test_diagnose_never_patches_from_the_outage_runs(self, valid_fsm, tmp_path):
        cheap = Metered(10_000, extraction_data={"answer": "42"}, responses=REPLIES)
        rig = self._run(valid_fsm, tmp_path, cheap)
        stripped = strip_abort(
            rig.library.get_definition(self.traces[0].digest).model_dump(  # type: ignore[union-attr]
                exclude_defaults=True
            )
        )
        assert stripped is not None

        got = diagnose(self.traces * 3, FSMDefinition(**stripped), SynthesisPolicy())

        assert [d.code for d in got] == ["not-procedure-fault"]


class TestTheOutageIsAPositionTakenAtThePush:
    """The outage is decided from a position taken when the child is pushed, never from a
    before and after of ``usage().errors`` (D-015 of plan 5a6a1f80)."""

    def _run(
        self, valid_fsm: Any, cheap: Any, tmp_path: Any, *, log: bool, sink: Any = None
    ) -> Rig:
        opened = EventLog(tmp_path / "events") if log else None
        rig = make_rig(
            valid_fsm,
            cheap=cheap,
            policy=SynthesisPolicy(max_turns=3),
            sink=sink,
            event_log=opened,
        )
        try:
            rig.run(30)
        finally:
            if opened is not None:
                opened.close()
        return rig

    @pytest.mark.parametrize("log", [False, True])
    def test_a_reset_of_the_usage_in_the_middle_of_the_run_keeps_the_outage(
        self, valid_fsm, tmp_path, log
    ):
        cheap = _Ledgered(2, extraction_data={}, responses=REPLIES)
        seen: list[str] = []
        harvested: list[int] = []

        def sink(event: str, fields: dict[str, Any]) -> None:
            seen.append(event)
            if event == "stepped" and seen.count("stepped") == 2:
                harvested.append(
                    cheap.reset_usage().errors
                )  # another reader starts again

        rig = self._run(valid_fsm, cheap, tmp_path, log=log, sink=sink)

        assert (harvested, cheap.usage().errors) == (
            [2],
            0,
        )  # the reset really cleared the view
        assert rig.result["status"] == "aborted"
        assert rig.result["reason"] == "provider_error"

    def test_an_interface_with_no_meter_reports_the_outage_when_the_log_is_on(
        self, valid_fsm, tmp_path
    ):
        cheap = _Meterless(extraction_data={}, responses=REPLIES)

        rig = self._run(valid_fsm, cheap, tmp_path, log=True)

        assert rig.result["reason"] == "provider_error"

    def test_an_interface_with_no_meter_and_no_log_reads_zero_as_before(
        self, valid_fsm, tmp_path
    ):
        cheap = _Meterless(extraction_data={}, responses=REPLIES)

        rig = self._run(valid_fsm, cheap, tmp_path, log=False)

        assert rig.result["reason"] == "budget_spent"  # the documented limit

    def test_errors_of_the_host_before_the_push_are_not_the_childs(
        self, valid_fsm, tmp_path
    ):
        cheap = _Ledgered(0, extraction_data={}, responses=REPLIES)
        _meter_of(cheap).record_error("generate")  # earlier work of the same interface

        rig = self._run(valid_fsm, cheap, tmp_path, log=True)

        assert rig.result["reason"] == "budget_spent"


class TestReturnKeys:
    def test_missing_return_keys_merge_nothing(self, valid_fsm, monkeypatch):
        rig = make_rig(
            valid_fsm,
            extraction={"answer": "42", "extra": "x"},
            config={"return_keys": ["answer", "extra"]},
        )
        served = make_procedure(_two_key(valid_fsm), return_keys=("answer", "extra"))
        serve_procedure(rig, monkeypatch, served)

        # the child ends with both keys, but one is gone by the time it is popped:
        # a handler of the child's last transition blanks it
        def vanish(context: dict[str, Any]) -> dict[str, Any]:
            return {"extra": None}

        rig.api.register_handler(
            rig.api.create_handler("vanish")
            .at(HandlerTiming.POST_TRANSITION)
            .when(lambda timing, state, target, context, updated: target == "done")
            .do(vanish)
        )

        rig.run()

        assert rig.depth == 1
        assert "answer" not in rig.data and "extra" not in rig.data
        assert rig.result == {
            "status": "failed",
            "procedure_id": served.digest,
            "version": None,
            "reason": "return_keys_missing",
        }
        assert rig.events[-1][1]["status"] == "failed"
        assert rig.state == "apology"


class TestFailuresAreFixedReasons:
    @staticmethod
    def _chained() -> SynthesisError:
        error = SynthesisError(SECRET)
        error.__cause__ = LLMResponseError(SECRET)
        return error

    @pytest.mark.parametrize(
        ("error", "reason"),
        [
            (SynthesisRefusedError(SECRET, "approval_required"), "approval_required"),
            (SynthesisRefusedError(SECRET, "approval_denied"), "approval_denied"),
            (SynthesisRefusedError(SECRET, "reviewer_error"), "reviewer_error"),
            (SynthesisRefusedError(SECRET, "made_up_code"), "compile_failed"),
            (LibraryError(SECRET), "library_error"),
            (SynthesisError(SECRET), "compile_failed"),
            (_chained(), "provider_error"),
            (ValueError(SECRET), "compile_failed"),
        ],
        ids=[
            "approval_required",
            "approval_denied",
            "reviewer_error",
            "unknown_code",
            "library",
            "compile",
            "provider",
            "bad_arguments",
        ],
    )
    def test_each_obtain_error_class(self, valid_fsm, monkeypatch, error, reason):
        rig = make_rig(valid_fsm)

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise error

        monkeypatch.setattr(rig.synth, "obtain", refuse)
        rig.run(10)

        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.result == {
            "status": "failed",
            "procedure_id": None,
            "version": None,
            "reason": reason,
        }
        assert SECRET not in repr(rig.data) and SECRET not in repr(rig.events)
        assert rig.names == ["failed"]

    @pytest.mark.parametrize(
        "error", [TimeoutError(SECRET), IndexError(SECRET), RuntimeError(SECRET)]
    )
    def test_an_unexpected_obtain_error_is_a_final_failure_not_running(
        self, valid_fsm, monkeypatch, error
    ):
        rig = make_rig(valid_fsm)
        calls: list[int] = []

        def blow_up(*args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            raise error

        monkeypatch.setattr(rig.synth, "obtain", blow_up)
        seen: list[str] = []
        logger.enable("fsm_llm")
        handler = logger.add(lambda m: seen.append(str(m)), level="WARNING")
        try:
            rig.run(10)
        finally:
            logger.remove(handler)
            logger.disable("fsm_llm")

        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.result == {
            "status": "failed",
            "procedure_id": None,
            "version": None,
            "reason": "unexpected_error",
        }
        assert rig.names == ["failed"] and calls == [1]
        assert SECRET not in repr(rig.data) and SECRET not in repr(rig.events)
        warnings = [m for m in seen if type(error).__name__ in m]
        assert len(warnings) == 1 and SECRET not in warnings[0]

    def test_the_failure_is_final_a_second_run_obtains_nothing_more(self, valid_fsm):
        """The r7 shape: the real strong double raises a client timeout."""
        rig = make_rig(valid_fsm, replies=[TimeoutError(SECRET)])
        # the parent leaves the silent synthesize state on this very turn
        assert rig.converse("hello") == "Sorry"
        assert (rig.state, rig.result["reason"]) == ("apology", "unexpected_error")
        with pytest.raises(FSMError, match="ended"):
            rig.converse("again")

        assert (rig.state, rig.depth) == ("apology", 1)
        assert len(rig.strong.requests) == 1
        assert rig.names == ["failed"]

    def test_a_failed_pre_push_check_is_a_final_failure(self, valid_fsm, monkeypatch):
        rig = make_rig(valid_fsm)

        def blow_up(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(SECRET)

        monkeypatch.setattr("fsm_llm.synthesis.procedures.verify_stored", blow_up)
        rig.run(10)

        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.result["reason"] == "unexpected_error"
        assert rig.result["procedure_id"] == rig.digest
        assert rig.names == ["obtained", "failed"]

    def test_a_real_provider_outage(self, valid_fsm):
        rig = make_rig(valid_fsm, replies=[LLMResponseError(SECRET)])
        rig.run(10)
        assert rig.result["reason"] == "provider_error"
        assert SECRET not in repr(rig.data)

    def test_a_compile_that_never_passes(self, valid_fsm):
        rig = make_rig(
            valid_fsm,
            replies=["no json here"],
            policy=SynthesisPolicy(max_repair_rounds=0),
        )
        rig.run(10)
        assert (rig.state, rig.result["reason"]) == ("apology", "compile_failed")
        assert len(rig.strong.requests) == 1

    def test_a_missing_input_obtains_nothing(self, valid_fsm):
        rig = make_rig(valid_fsm, context={"unrelated": 1})
        rig.run(10)
        assert rig.result["reason"] == "inputs_missing"
        assert rig.strong.requests == [] and rig.names == ["failed"]

    def test_a_procedure_below_the_minimum_lifecycle_is_refused(
        self, valid_fsm, tmp_path
    ):
        rig = make_rig(valid_fsm, tmp_path, config={"min_lifecycle": "trusted"})
        rig.run(10)
        assert rig.result["reason"] == "lifecycle_too_low"
        assert rig.result["procedure_id"] == rig.digest
        assert rig.names == ["obtained", "failed"]
        assert rig.library is not None
        assert rig.library.get_record(rig.digest).stats["runs"] == 0

    def test_a_compile_that_reproduces_a_deprecated_digest_is_no_procedure(
        self, valid_fsm, tmp_path
    ):
        first = make_rig(valid_fsm, tmp_path)
        first.run(20)
        assert first.library is not None
        first.library.deprecate(first.digest)

        rig = make_rig(valid_fsm, tmp_path)
        rig.run(10)

        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.result == {
            "status": "failed",
            "procedure_id": None,
            "version": None,
            "reason": "lifecycle_too_low",
        }
        assert rig.names == ["failed"]

    @pytest.mark.parametrize(
        ("gates", "require", "reason"),
        [
            (GATES, True, "not_simulated"),
            ((*GATES, "G6b"), True, None),
            (GATES, False, None),
        ],
        ids=["unsimulated_refused", "simulated_runs", "policy_off_runs"],
    )
    def test_require_simulation_is_a_rule_of_every_run_not_only_of_obtain(
        self, valid_fsm, tmp_path, monkeypatch, gates, require, reason
    ):
        rig = make_rig(
            valid_fsm, tmp_path, policy=SynthesisPolicy(require_simulation=require)
        )
        procedure = make_procedure(valid_fsm)
        assert rig.library is not None
        record = rig.library.put(
            procedure.definition,
            goal=GOAL,
            inputs_required=("topic",),
            return_keys=("answer",),
            gates_passed=gates,
            lifecycle="verified",
        )
        serve_procedure(
            rig,
            monkeypatch,
            Procedure(procedure.digest, procedure.definition, record, "library"),
        )

        rig.run(20)

        assert rig.result["reason"] == reason
        assert rig.result["status"] == ("failed" if reason else "ok")
        assert bool(reason) == (rig.cheap.count("extract_field") == 0)

    @pytest.mark.parametrize(
        ("gates", "reason"),
        [
            (None, None),  # no record: a compile under this policy simulated it
            ([*GATES, "G6b"], None),
            ([*GATES], "not_simulated"),
            ([], "not_simulated"),
            ("G6b", "not_simulated"),  # not a list of gates
        ],
    )
    def test_runnable_reason_reads_the_gates_list_under_require_simulation(
        self, valid_fsm, gates, reason
    ):
        got = procedures_module.runnable_reason(
            make_procedure(valid_fsm).definition,
            "verified",
            min_lifecycle="verified",
            policy=SynthesisPolicy(require_simulation=True),
            inputs=("topic",),
            return_keys=("answer",),
            gates_passed=gates,
        )
        assert got == reason

    def test_a_procedure_with_a_completion_state_is_refused(
        self, valid_fsm, monkeypatch
    ):
        rig = make_rig(valid_fsm)
        procedure = make_procedure(valid_fsm)
        states = dict(procedure.definition.states)
        states["ask"] = states["ask"].model_copy(
            update={"completion": CompletionStateConfig(response_format_key="_fmt")}
        )
        definition = procedure.definition.model_copy(update={"states": states})
        serve_procedure(
            rig, monkeypatch, Procedure(procedure.digest, definition, None, "x")
        )

        rig.run(10)

        assert rig.result["reason"] == "completion_unsupported"
        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.names == ["obtained", "failed"]

    @pytest.mark.parametrize(
        "tamper",
        [
            lambda d: d.update(persona="evil"),
            lambda d: d.pop("handler_only_keys"),
            lambda d: d["states"]["ask"]["field_extractions"].append(
                {
                    "field_name": "procedure_abort",
                    "field_type": "str",
                    "extraction_instructions": "x",
                }
            ),
            lambda d: d["states"]["ask"]["field_extractions"].append(
                {
                    "field_name": "synthesis_result",
                    "field_type": "str",
                    "extraction_instructions": "x",
                }
            ),
        ],
        ids=["persona", "no_handler_only", "forged_abort", "forged_result"],
    )
    def test_a_procedure_that_fails_reverification_is_never_pushed(
        self, valid_fsm, monkeypatch, tamper
    ):
        rig = make_rig(valid_fsm)
        data = make_procedure(valid_fsm).definition.model_dump(exclude_defaults=True)
        tamper(data)
        definition = FSMDefinition(**data)
        serve_procedure(
            rig, monkeypatch, Procedure(digest_of(definition), definition, None, "x")
        )

        rig.run(10)

        assert rig.result["reason"] == "library_error"
        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.names == ["obtained", "failed"]


class TestNesting:
    def test_the_strict_default_policy_refuses_a_procedure_that_nests(
        self, valid_fsm, monkeypatch
    ):
        rig = make_rig(valid_fsm, config={"return_keys": ["topic"]})
        procedure = make_procedure(
            _nesting(),
            policy=SynthesisPolicy(max_synthesis_depth=2),
            return_keys=("topic",),
        )
        serve_procedure(rig, monkeypatch, procedure)

        rig.run(10)

        # the synthesizer's policy (depth 1) fails G5 on the stored form
        assert rig.result["reason"] == "library_error"

    def test_depth_two_nests_once_and_refuses_the_third_level(
        self, valid_fsm, monkeypatch
    ):
        policy = SynthesisPolicy(max_synthesis_depth=2)
        rig = make_rig(valid_fsm, policy=policy, config={"return_keys": ["topic"]})
        serve_procedure(
            rig,
            monkeypatch,
            make_procedure(_nesting(), policy=policy, return_keys=("topic",)),
        )

        rig.run(40)

        # level 1 obtains and nests level 2; level 2's own synthesize state is
        # refused before any obtain (no strong call), its run settles, both pop
        assert [f["reason"] for n, f in rig.events if n == "failed"] == [
            "depth_refused"
        ]
        assert rig.names == [
            "obtained",
            "pushed",
            "obtained",
            "pushed",
            "failed",
            "popped",
            "popped",
        ]
        assert (rig.state, rig.depth) == ("booked", 1)
        assert rig.result["status"] == "ok"

    def test_converse_pops_every_ended_child_in_the_same_turn(
        self, valid_fsm, monkeypatch
    ):
        """A child that a refused nested start leaves on a terminal state used to
        stay on the stack (one raising user turn per level, final reply lost)."""
        policy = SynthesisPolicy(max_synthesis_depth=2)
        rig = make_rig(valid_fsm, policy=policy, config={"return_keys": ["topic"]})
        serve_procedure(
            rig,
            monkeypatch,
            make_procedure(_nesting(), policy=policy, return_keys=("topic",)),
        )

        reply = rig.converse("hello")

        assert reply == "Booked!"  # the final reply is not lost
        assert rig.names[-3:] == ["failed", "popped", "popped"]
        assert (rig.state, rig.depth) == ("booked", 1)
        assert rig.result["status"] == "ok"


class TestTraceSink:
    def test_fields_are_small_plain_values(self, valid_fsm):
        rig = make_rig(valid_fsm)
        rig.run(20)
        for _, fields in rig.events:
            assert all(
                isinstance(v, (str, int, float, type(None))) for v in fields.values()
            ), fields

    def test_a_raising_sink_is_swallowed_with_one_warning_per_event(self, valid_fsm):
        seen: list[str] = []

        def sink(event: str, fields: dict[str, Any]) -> None:
            raise RuntimeError(SECRET)

        rig = make_rig(valid_fsm, sink=sink)
        logger.enable("fsm_llm")
        handler = logger.add(lambda m: seen.append(str(m)), level="WARNING")
        try:
            rig.run(20)
        finally:
            logger.remove(handler)
            logger.disable("fsm_llm")

        assert rig.state == "booked"
        warnings = [m for m in seen if "trace_sink" in m]
        assert len(warnings) == 3 and not any(SECRET in m for m in warnings)


class TestObtainedEvent:
    def _obtained(self, rig: Rig) -> dict[str, Any]:
        rig.run(20)
        return next(f for name, f in rig.events if name == "obtained")

    def test_it_carries_round_codes_seconds_and_token_counts(
        self, valid_fsm, monkeypatch
    ):
        rig = make_rig(valid_fsm)
        procedure = dataclasses.replace(
            make_procedure(valid_fsm),
            rounds=(("G5", "G4"), ("G6b",)),
            seconds=1.25,
            usage=LLMUsage(calls=3, prompt_tokens=40, completion_tokens=9),
        )
        serve_procedure(rig, monkeypatch, procedure)
        fields = self._obtained(rig)
        assert fields["rounds"] == "G5,G4;G6b" and fields["seconds"] == 1.25
        assert (fields["calls"], fields["prompt_tokens"]) == (3, 40)
        assert fields["completion_tokens"] == 9

    def test_a_first_pass_compile_has_empty_rounds_and_zero_usage(self, valid_fsm):
        rig = make_rig(valid_fsm)
        rig.strong.usage = None  # a strong model that keeps no meter reports zeros
        fields = self._obtained(rig)
        assert fields["rounds"] == "" and fields["calls"] == 0
        assert isinstance(fields["seconds"], float) and fields["seconds"] >= 0

    def test_two_repair_rounds_arrive_as_codes_only(self, valid_fsm):
        bad = copy.deepcopy(valid_fsm)
        bad["persona"] = SECRET
        bad["states"]["ask"]["transitions"][0]["priority"] = 0
        rig = make_rig(
            valid_fsm, replies=[SECRET, json.dumps(bad), json.dumps(valid_fsm)]
        )
        fields = self._obtained(rig)
        assert (
            fields["rounds"].count(";") == 1
            and "reply-not-json-object" in fields["rounds"]
        )
        assert SECRET not in repr(fields)

    def test_a_library_hit_is_source_library_with_empty_rounds(
        self, valid_fsm, tmp_path
    ):
        first = make_rig(valid_fsm, tmp_path)
        first.run(20)
        fields = self._obtained(make_rig(valid_fsm, tmp_path, replies=[]))
        assert (fields["source"], fields["rounds"], fields["calls"]) == (
            "library",
            "",
            0,
        )
        assert 0 <= fields["seconds"] < 5.0


class TestNoForeignInterface:
    """A compile may not move the child to a model of its own: nothing but the
    host's injected interface is ever built or called (review iter 1, concern 1)."""

    def test_a_compile_that_names_a_model_is_repaired_before_it_runs(
        self, valid_fsm, foreign_calls
    ):
        bad = json.dumps(_classified(valid_fsm, model="openai/gpt-4o"))
        rig = make_rig(valid_fsm, replies=[bad, json.dumps(valid_fsm)])

        rig.run(20)

        assert foreign_calls == []
        assert "model-setting" in str(rig.strong.requests[1].messages)
        assert (rig.state, rig.result["status"]) == ("booked", "ok")

    def test_a_compile_that_keeps_naming_one_never_runs(self, valid_fsm, foreign_calls):
        bad = json.dumps(_classified(valid_fsm, model="openai/gpt-4o"))
        rig = make_rig(valid_fsm, replies=[bad] * 4)

        rig.run(20)

        assert foreign_calls == []
        assert rig.result["status"] == "failed"
        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.names == ["failed"]

    def test_a_stored_legacy_procedure_with_a_model_is_refused_at_push(
        self, valid_fsm, monkeypatch, foreign_calls
    ):
        rig = make_rig(valid_fsm)
        data = make_procedure(valid_fsm).definition.model_dump(exclude_defaults=True)
        data["states"]["ask"]["classification_extractions"] = _classified(
            valid_fsm, model="openai/gpt-4o"
        )["states"]["ask"]["classification_extractions"]
        definition = FSMDefinition(**data)
        serve_procedure(
            rig, monkeypatch, Procedure(digest_of(definition), definition, None, "x")
        )

        rig.run(10)

        assert foreign_calls == []
        assert rig.result["reason"] == "library_error"
        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.names == ["obtained", "failed"]


class TestTraceEvents:
    """The events a ``TraceRecorder`` reads: the conversation on every one, a
    ``stepped`` event per counted child step, the stuck state and live outcome
    at an abort, the error class and round codes at a failure. Names, type
    names, state ids, counts and codes only (D-038, D-044)."""

    def test_a_canary_never_reaches_an_event_of_an_ok_run(self, valid_fsm):
        rig = make_rig(
            valid_fsm,
            extraction={"answer": CANARIES[0]},
            context={**CONTEXT, "topic": CANARIES[0]},
        )
        rig.cheap.responses["ask"] = CANARIES[2]

        rig.converse("hello")
        rig.converse(CANARIES[1])

        assert rig.result["status"] == "ok" and rig.data["answer"] == CANARIES[0]
        assert len(rig.steps) == 1  # the scan below is not vacuous
        blob = json.dumps(rig.stream)
        assert not [c for c in CANARIES if c in blob]

    def test_a_canary_never_reaches_an_event_of_an_aborted_run(self, valid_fsm):
        rig = make_rig(
            valid_fsm,
            extraction={},
            policy=SynthesisPolicy(max_turns=2),
            context={**CONTEXT, "topic": CANARIES[0]},
        )
        rig.cheap.responses["ask"] = CANARIES[2]

        for text in ("hello", CANARIES[1], CANARIES[1]):
            rig.converse(text)

        assert "aborted" in rig.names and len(rig.steps) == 2
        assert not [c for c in CANARIES if c in json.dumps(rig.stream)]

    @pytest.mark.parametrize(
        ("error", "name", "rounds"),
        [
            (RuntimeError(CANARIES[2]), "RuntimeError", None),
            (TimeoutError(CANARIES[2]), "TimeoutError", None),
            (ValueError(CANARIES[2]), "ValueError", None),
            (LibraryError(CANARIES[2]), "LibraryError", ""),
            (SynthesisError(CANARIES[2]), "SynthesisError", ""),
            (
                SynthesisError(CANARIES[2], {"rounds": [("a-b", "c"), ("d",)]}),
                "SynthesisError",
                "a-b,c;d",
            ),
            (
                SynthesisError("x", {"rounds": [("z" * 90,)] * 12}),
                "SynthesisError",
                ";".join(["z" * 90] * 12)[:200],
            ),
        ],
    )
    def test_a_failure_carries_the_error_class_and_only_codes(
        self, valid_fsm, monkeypatch, error, name, rounds
    ):
        rig = make_rig(valid_fsm)

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise error

        monkeypatch.setattr(rig.synth, "obtain", refuse)
        rig.run(10)

        (failed,) = [f for n, f in rig.events if n == "failed"]
        assert failed["error"] == name and failed["rounds"] == rounds
        assert failed["conv"] == rig.conv
        assert CANARIES[2] not in json.dumps(rig.stream)

    def test_a_real_compile_that_never_passes_reports_its_round_codes(self, valid_fsm):
        rig = make_rig(valid_fsm, replies=[CANARIES[2]] * 8)

        rig.run(10)

        (failed,) = [f for n, f in rig.events if n == "failed"]
        assert (
            failed["error"] == "SynthesisError" and failed["reason"] == "compile_failed"
        )
        rounds = failed["rounds"].split(";")
        assert len(rounds) == len(rig.strong.requests) > 1
        assert all(rounds) and len(set(rounds)) == 1
        assert CANARIES[2] not in json.dumps(rig.stream)

    def test_a_failure_without_an_exception_has_no_error(self, valid_fsm):
        rig = make_rig(valid_fsm, context={"unrelated": 1})
        rig.run(10)
        (failed,) = [f for n, f in rig.events if n == "failed"]
        assert failed["reason"] == "inputs_missing"
        assert failed["error"] is None and failed["rounds"] is None

    def test_two_conversations_on_one_runner_give_separable_streams(self, valid_fsm):
        rig = make_rig(valid_fsm, replies=[json.dumps(valid_fsm)] * 2)
        other, _ = rig.api.start_conversation(CONTEXT)

        rig.converse("hello")
        rig.converse("hello", other)
        rig.converse("forty-two", other)
        rig.converse("forty-two")

        assert all("conv" in f for _, f in rig.stream)
        for conv in (rig.conv, other):
            mine = [(n, f) for n, f in rig.stream if f["conv"] == conv]
            # the first turn is the parent's own; the second is the child's
            assert [n for n, _ in mine] == ["obtained", "pushed", "stepped", "popped"]
            assert [f["step"] for n, f in mine if n == "stepped"] == [1]
            assert len({f["digest"] for _, f in mine}) == 1
        assert {f["conv"] for _, f in rig.stream} == {rig.conv, other}

    def test_every_event_kind_carries_conv(self, valid_fsm, monkeypatch):
        ok = make_rig(valid_fsm)
        ok.run(20)
        aborted = make_rig(
            valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=2)
        )
        aborted.run(20)
        failed = make_rig(valid_fsm, context={"unrelated": 1})
        failed.run(20)
        seen = {n for r in (ok, aborted, failed) for n, _ in r.stream}
        assert seen == {"obtained", "pushed", "stepped", "aborted", "popped", "failed"}
        for rig in (ok, aborted, failed):
            assert all(f["conv"] == rig.conv for _, f in rig.stream)

    @pytest.mark.parametrize("mode", ["run", "converse"])
    def test_stepped_fires_once_per_counted_step_in_both_modes(self, valid_fsm, mode):
        rig = make_rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=3))

        if mode == "run":
            rig.run(30)
        else:
            for text in ("a", "b", "c", "d"):
                rig.converse(text)

        names = [n for n, _ in rig.stream]
        assert names.count("aborted") == 1
        # the abort comes right after the max_turns-th counted step, no later
        before = [n for n, _ in rig.stream[: names.index("aborted")]]
        assert before.count("stepped") == 3
        assert [f["step"] for f in rig.steps] == [1, 2, 3]
        for f in rig.steps:
            assert f["state"] == "ask" and f["keys"] == "topic" and f["kinds"] == ""
            assert f["digest"] == rig.digest

    def test_stepped_names_declared_keys_and_gate_read_types(self, valid_fsm):
        rig = make_rig(valid_fsm, extraction={"answer": "42", "stray": "q"})

        rig.converse("hello")
        rig.converse("forty-two")

        last = rig.steps[-1]
        assert last["state"] == "done"
        assert last["keys"] == "answer,topic"  # the input and the return key
        assert last["kinds"] == "answer:str"  # only what a gate reads
        assert "stray" not in json.dumps(rig.stream)
