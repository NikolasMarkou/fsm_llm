"""
Tests for ``fsm_llm.synthesis.runner.ProcedureRunner`` on REAL ``API`` objects.

The host is a concierge FSM with one ``synthesize`` state; the strong model is
``ScriptedStrongLLM``, the cheap model under the pushed procedure is
``ScriptedRunnerLLM`` (both in conftest). The runner is driven the two ways it
supports: ``run`` (core ``run_until_terminal`` with the runner's hook) and
``converse`` (user turns). Failure tests read the result the PARENT sees.
"""

import copy
import dataclasses
import json
from dataclasses import dataclass
from typing import Any

import pytest

from fsm_llm.api import API
from fsm_llm.definitions import (
    CompletionStateConfig,
    FSMDefinition,
    FSMError,
    LLMResponseError,
    LLMUsage,
    RunBudgetExceededError,
)
from fsm_llm.logging import logger
from fsm_llm.synthesis import (
    FSMLibrary,
    FSMSynthesizer,
    LibraryError,
    Procedure,
    ProcedureRunner,
    SynthesisError,
    SynthesisPolicy,
    SynthesisRefusedError,
    digest_of,
    run_gates,
)
from fsm_llm.synthesis import runner as runner_module
from tests.test_fsm_llm_synthesis.conftest import (
    ScriptedRunnerLLM,
    ScriptedStrongLLM,
    host_fsm,
)

GOAL = "Book a table for two"
PERSONA = "A calm hotel concierge"
SECRET = "SECRET-TEXT"
NOISE = "PARENT-ONLY-NOISE"
CONTEXT = {"topic": "tables", "noise_note": NOISE, "db_password": "hunter2"}
REPLIES = {
    "start": "Hello",
    "ask": "What is the answer?",
    "done": "Thanks",
    "booked": "Booked!",
    "apology": "Sorry",
}


def _host(**synthesize: Any) -> dict[str, Any]:
    """greet -> solve (synthesize) -> booked on ok, apology on failed/aborted."""
    return host_fsm(
        name="Concierge",
        description="Host conversation",
        persona=PERSONA,
        states=("start", "solve", "booked", "apology"),
        goal=GOAL,
        inputs=["topic"],
        return_keys=["answer"],
        greet_text="Say hello",
        work_description="run the procedure",
        ok_text="Say booked",
        fail_text="Apologise",
        edge_prefix="go to ",
        synthesize=synthesize,
    )


@dataclass
class _Rig:
    api: API
    runner: ProcedureRunner
    conv: str
    cheap: ScriptedRunnerLLM
    strong: ScriptedStrongLLM
    synth: FSMSynthesizer
    library: FSMLibrary | None
    stream: list[tuple[str, dict[str, Any]]]

    @property
    def events(self) -> list[tuple[str, dict[str, Any]]]:
        """The lifecycle events; the per-step ``stepped`` ones are ``steps``."""
        return [e for e in self.stream if e[0] != "stepped"]

    @property
    def steps(self) -> list[dict[str, Any]]:
        return [f for name, f in self.stream if name == "stepped"]

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    @property
    def state(self) -> str:
        return self.api.get_current_state(self.conv)

    @property
    def depth(self) -> int:
        return self.api.get_stack_depth(self.conv)

    @property
    def data(self) -> dict[str, Any]:
        return self.api.get_data(self.conv)

    @property
    def result(self) -> dict[str, Any]:
        return self.data["synthesis_result"]

    @property
    def digest(self) -> str:
        return next(f["digest"] for name, f in self.events if name == "obtained")

    def to_solve(self) -> None:
        """One message-free step: start -> solve (a synthesize state)."""
        self.api.advance(self.conv)
        assert self.state == "solve"

    def extractions(self) -> int:
        return sum(1 for kind, _ in self.cheap.call_history if kind == "extract_field")


def _rig(
    valid_fsm: dict[str, Any],
    tmp_path: Any = None,
    *,
    replies: list[Any] | None = None,
    extraction: dict[str, Any] | None = None,
    policy: SynthesisPolicy | None = None,
    config: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    sink: Any = None,
) -> _Rig:
    cheap = ScriptedRunnerLLM(
        {"answer": "42"} if extraction is None else extraction, dict(REPLIES)
    )
    strong = ScriptedStrongLLM([json.dumps(valid_fsm)] if replies is None else replies)
    library = FSMLibrary(tmp_path / "lib") if tmp_path is not None else None
    synth = FSMSynthesizer(strong, policy=policy, library=library)
    host = _host(**(config or {}))
    api = API.from_definition(definition=host, llm_interface=cheap)
    stream: list[tuple[str, dict[str, Any]]] = []
    runner = ProcedureRunner(
        api, host, synth, trace_sink=sink or (lambda e, f: stream.append((e, dict(f))))
    )
    conv, _ = api.start_conversation(CONTEXT if context is None else context)
    return _Rig(api, runner, conv, cheap, strong, synth, library, stream)


def _procedure(data: dict[str, Any], **kw: Any) -> Procedure:
    """A gate-passed ``Procedure`` built by hand (no strong model)."""
    kw.setdefault("inputs", ("topic",))
    kw.setdefault("return_keys", ("answer",))
    report = run_gates(copy.deepcopy(data), kw.pop("policy", SynthesisPolicy()), **kw)
    assert report.definition is not None, report.findings
    return Procedure(digest_of(report.definition), report.definition, None, "compiled")


def _serve(rig: _Rig, monkeypatch: pytest.MonkeyPatch, procedure: Procedure) -> None:
    """Make ``obtain`` hand back ``procedure`` (the compile path is tested elsewhere)."""
    monkeypatch.setattr(rig.synth, "obtain", lambda *a, **k: procedure)


class TestOkPath:
    def test_run_until_terminal_pushes_runs_pops_and_merges(self, valid_fsm, tmp_path):
        rig = _rig(valid_fsm, tmp_path)

        results = rig.runner.run(rig.conv, max_steps=20)

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
        rig = _rig(valid_fsm)
        rig.runner.run(rig.conv, max_steps=20)

        child_prompts = rig.cheap.shown("generate_response", "ask")
        assert "<persona>" in child_prompts and PERSONA in child_prompts
        assert rig.extractions() == 1

    def test_the_child_sees_no_parent_data(self, valid_fsm):
        rig = _rig(valid_fsm)
        rig.runner.run(rig.conv, max_steps=20)

        # control: the parent's own prompt does carry its noise
        assert NOISE in rig.cheap.shown("generate_response", "start")
        for shown in (
            rig.cheap.shown("extract_field"),
            rig.cheap.shown("generate_response", "ask"),
            rig.cheap.shown("generate_response", "done"),
        ):
            assert shown and NOISE not in shown and "hunter2" not in shown

    def test_the_push_hands_over_only_the_declared_inputs(self, valid_fsm):
        rig = _rig(valid_fsm)
        rig.to_solve()
        hook = rig.runner.hook(rig.conv)

        hook(2)

        assert rig.depth == 2
        child = rig.data
        assert child["topic"] == "tables"
        for absent in ("noise_note", "db_password", "synthesis_result"):
            assert absent not in child
        hook(3)  # a second call while the child runs does nothing
        assert (rig.depth, len(rig.strong.requests)) == (2, 1)
        assert rig.names == ["obtained", "pushed"]
        # the run goes on from where the manual hook left it
        rig.runner.run(rig.conv, max_steps=20)
        assert (rig.state, rig.depth) == ("booked", 1)

    def test_a_settled_result_is_never_obtained_twice(self, valid_fsm):
        rig = _rig(valid_fsm, replies=[LLMResponseError(SECRET)])
        rig.to_solve()
        hook = rig.runner.hook(rig.conv)

        hook(2)
        hook(3)

        assert len(rig.strong.requests) == 1
        assert rig.result["reason"] == "provider_error"

    def test_a_host_definition_object_is_accepted(self, valid_fsm):
        rig = _rig(valid_fsm)
        runner = ProcedureRunner(rig.api, FSMDefinition(**_host()), rig.synth)
        runner.run(rig.conv, max_steps=20)
        assert rig.state == "booked"


class TestConverse:
    def test_user_turns_drive_the_child_then_the_parent_moves_on(
        self, valid_fsm, tmp_path
    ):
        rig = _rig(valid_fsm, tmp_path)

        # the parent's solve state is silent: the first reply is the child's
        assert rig.runner.converse(rig.conv, "hello") == "What is the answer?"
        assert (rig.state, rig.depth) == ("ask", 2)

        # the child ends on this turn, is popped, and the parent steps on
        assert rig.runner.converse(rig.conv, "forty-two") == "Thanks\nBooked!"
        assert (rig.state, rig.depth) == ("booked", 1)
        assert rig.data["answer"] == "42"
        assert rig.result["status"] == "ok"
        assert rig.names == ["obtained", "pushed", "popped"]
        assert rig.library is not None
        assert rig.library.get_record(rig.digest).stats["runs"] == 1

    def test_a_child_that_never_finishes_is_aborted_after_max_turns(self, valid_fsm):
        rig = _rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=2))

        rig.runner.converse(rig.conv, "hello")
        rig.runner.converse(rig.conv, "one")
        assert rig.depth == 2 and "aborted" not in rig.names
        reply = rig.runner.converse(rig.conv, "two")

        assert (rig.state, rig.depth) == ("apology", 1)
        # the child's own reply to the last turn, then the parent's apology
        assert reply == "What is the answer?\nSorry"
        assert rig.result["status"] == "aborted"
        assert rig.result["reason"] == "budget_spent"
        assert "answer" not in rig.data
        assert rig.names == ["obtained", "pushed", "aborted", "popped"]

    def test_a_bare_converse_on_a_synthesize_state_stays_and_obtains_nothing(
        self, valid_fsm
    ):
        rig = _rig(valid_fsm)

        rig.api.converse("hello", rig.conv)
        rig.api.converse("again", rig.conv)

        assert (rig.state, rig.depth) == ("solve", 1)
        assert "synthesis_result" not in rig.data
        assert rig.strong.requests == [] and rig.events == []


class TestAbort:
    def test_a_silent_child_is_aborted_after_max_turns_and_the_parent_apologises(
        self, valid_fsm, tmp_path
    ):
        rig = _rig(
            valid_fsm,
            tmp_path,
            extraction={},
            policy=SynthesisPolicy(max_turns=3),
        )

        results = rig.runner.run(rig.conv, max_steps=30)

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
        # step extracts twice (one retry), the abort step included (Pass 1 runs
        # before the edge is chosen)
        assert rig.cheap.count("generate_response", "ask") == 1 + 3
        assert rig.extractions() == (3 + 1) * 2
        assert rig.names == ["obtained", "pushed", "aborted", "popped"]
        assert rig.events[2][1]["step"] == 5
        assert rig.library is not None
        assert rig.library.get_record(rig.digest).stats == {"runs": 1, "successes": 0}

    def test_a_spent_parent_budget_raises_and_a_later_end_cleans_up(
        self, valid_fsm, tmp_path
    ):
        rig = _rig(valid_fsm, tmp_path, extraction={})

        with pytest.raises(RunBudgetExceededError):
            rig.runner.run(rig.conv, max_steps=3)

        assert rig.depth == 2  # the child is still on top
        rig.runner.end_conversation(rig.conv)  # does not raise
        with pytest.raises(ValueError):
            rig.api.get_stack_depth(rig.conv)
        assert rig.conv not in rig.runner._runs
        assert rig.library is not None
        assert rig.library.get_record(rig.digest).stats["runs"] == 0

    def test_a_resumed_run_still_aborts_the_child_after_max_turns_in_total(
        self, valid_fsm
    ):
        """Core restarts its step number at 1 on every ``run`` call; the runner's
        own count of child steps must not restart with it (review concern 5)."""
        rig = _rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=5))

        for _ in range(10):
            try:
                rig.runner.run(rig.conv, max_steps=3)
            except RunBudgetExceededError:
                continue
            break

        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.result["status"] == "aborted"
        assert rig.names == ["obtained", "pushed", "aborted", "popped"]
        # the greeting plus exactly max_turns child steps, however the run was cut
        assert rig.cheap.count("generate_response", "ask") == 1 + 5

    def test_ending_the_conversation_with_a_child_on_top_counts_no_run(
        self, valid_fsm, tmp_path
    ):
        rig = _rig(valid_fsm, tmp_path)
        rig.to_solve()
        rig.runner.hook(rig.conv)(2)
        assert rig.depth == 2

        rig.api.end_conversation(rig.conv)  # straight on the API: no raise

        with pytest.raises(ValueError):
            rig.api.get_stack_depth(rig.conv)
        assert rig.library is not None
        assert rig.library.get_record(rig.digest).stats["runs"] == 0

    def test_a_child_popped_from_outside_is_trimmed_not_obtained_again(self, valid_fsm):
        rig = _rig(valid_fsm)
        rig.to_solve()
        hook = rig.runner.hook(rig.conv)
        hook(2)

        rig.api.pop_fsm(rig.conv)
        hook(3)

        assert rig.runner._runs[rig.conv] == []
        assert len(rig.strong.requests) == 1
        assert rig.result["status"] == "running"


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


class TestReturnKeys:
    def test_missing_return_keys_merge_nothing(self, valid_fsm, monkeypatch):
        rig = _rig(
            valid_fsm,
            extraction={"answer": "42", "extra": "x"},
            config={"return_keys": ["answer", "extra"]},
        )
        procedure = _procedure(_two_key(valid_fsm), return_keys=("answer", "extra"))
        _serve(rig, monkeypatch, procedure)
        rig.to_solve()
        hook = rig.runner.hook(rig.conv)
        hook(2)
        rig.api.advance(rig.conv)
        assert rig.api.has_conversation_ended(rig.conv)
        # the child ended, but one return key is gone by the time it is popped
        rig.api.update_context(rig.conv, {"extra": None})

        hook(3)

        assert rig.depth == 1
        assert "answer" not in rig.data and "extra" not in rig.data
        assert rig.result == {
            "status": "failed",
            "procedure_id": procedure.digest,
            "version": None,
            "reason": "return_keys_missing",
        }
        assert rig.events[-1][1]["status"] == "failed"
        rig.runner.run(rig.conv, max_steps=5)
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
        rig = _rig(valid_fsm)

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise error

        monkeypatch.setattr(rig.synth, "obtain", refuse)
        rig.runner.run(rig.conv, max_steps=10)

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
        rig = _rig(valid_fsm)
        calls: list[int] = []

        def blow_up(*args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            raise error

        monkeypatch.setattr(rig.synth, "obtain", blow_up)
        seen: list[str] = []
        logger.enable("fsm_llm")
        handler = logger.add(lambda m: seen.append(str(m)), level="WARNING")
        try:
            rig.runner.run(rig.conv, max_steps=10)
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
        rig = _rig(valid_fsm, replies=[TimeoutError(SECRET)])
        rig.runner.converse(rig.conv, "hello")
        assert (rig.state, rig.result["reason"]) == ("solve", "unexpected_error")
        rig.runner.converse(rig.conv, "again")

        assert (rig.state, rig.depth) == ("apology", 1)
        assert len(rig.strong.requests) == 1
        assert rig.names == ["failed"]

    def test_a_failed_pre_push_check_is_a_final_failure(self, valid_fsm, monkeypatch):
        rig = _rig(valid_fsm)

        def blow_up(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(SECRET)

        monkeypatch.setattr("fsm_llm.synthesis.runner.verify_stored", blow_up)
        rig.runner.run(rig.conv, max_steps=10)

        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.result["reason"] == "unexpected_error"
        assert rig.result["procedure_id"] == rig.digest
        assert rig.names == ["obtained", "failed"]

    def test_a_push_that_raises_after_pushing_leaves_the_stack_as_it_was(
        self, valid_fsm, monkeypatch
    ):
        rig = _rig(valid_fsm)
        real_push = rig.api.push_fsm

        def push_then_fail(*args: Any, **kwargs: Any) -> Any:
            real_push(*args, **kwargs)
            raise RuntimeError(SECRET)

        monkeypatch.setattr(rig.api, "push_fsm", push_then_fail)
        rig.runner.run(rig.conv, max_steps=10)

        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.result["reason"] == "unexpected_error"
        assert rig.runner._runs[rig.conv] == []

    @pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
    def test_ctrl_c_and_exit_are_never_swallowed(self, valid_fsm, monkeypatch, signal):
        rig = _rig(valid_fsm)

        def interrupt(*args: Any, **kwargs: Any) -> Any:
            raise signal()

        monkeypatch.setattr(rig.synth, "obtain", interrupt)
        with pytest.raises(signal):
            rig.runner.run(rig.conv, max_steps=10)

    @pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
    def test_an_interrupted_compile_leaves_a_final_failed_result(
        self, valid_fsm, monkeypatch, signal
    ):
        rig = _rig(valid_fsm)
        interrupted = signal()
        calls: list[int] = []

        def interrupt(*args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            raise interrupted

        monkeypatch.setattr(rig.synth, "obtain", interrupt)
        with pytest.raises(signal) as caught:
            rig.runner.run(rig.conv, max_steps=10)

        assert caught.value is interrupted
        assert rig.result["status"] == "failed"
        assert rig.result["reason"] == "unexpected_error"
        assert (rig.state, rig.depth) == ("solve", 1)
        assert rig.names == ["failed"]
        # the host catches the signal and keeps the conversation: nothing obtains again
        rig.runner.run(rig.conv, max_steps=10)
        assert calls == [1]
        assert (rig.state, rig.depth) == ("apology", 1)

    @pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
    def test_an_interrupt_after_the_push_unwinds_and_ends_failed(
        self, valid_fsm, monkeypatch, signal
    ):
        rig = _rig(valid_fsm)
        real_push = rig.api.push_fsm

        def push_then_interrupt(*args: Any, **kwargs: Any) -> Any:
            real_push(*args, **kwargs)
            raise signal()

        monkeypatch.setattr(rig.api, "push_fsm", push_then_interrupt)
        with pytest.raises(signal):
            rig.runner.run(rig.conv, max_steps=10)

        assert (rig.state, rig.depth) == ("solve", 1)
        assert rig.result["status"] == "failed"
        assert rig.result["reason"] == "unexpected_error"
        assert rig.runner._runs[rig.conv] == []
        calls = len(rig.strong.requests)
        rig.runner.run(rig.conv, max_steps=10)
        assert len(rig.strong.requests) == calls
        assert rig.state == "apology"

    def test_an_interrupted_converse_turn_is_final_too(self, valid_fsm, monkeypatch):
        rig = _rig(valid_fsm)

        def interrupt(*args: Any, **kwargs: Any) -> Any:
            raise KeyboardInterrupt()

        monkeypatch.setattr(rig.synth, "obtain", interrupt)
        with pytest.raises(KeyboardInterrupt):
            rig.runner.converse(rig.conv, "hi")
        assert rig.result["status"] == "failed"

    def test_a_failed_result_write_never_masks_the_interrupt(
        self, valid_fsm, monkeypatch
    ):
        rig = _rig(valid_fsm)
        monkeypatch.setattr(
            rig.synth, "obtain", lambda *a, **k: (_ for _ in ()).throw(SystemExit(3))
        )
        real_update = rig.api.update_context

        def refuse_failed(conv_id: str, data: dict[str, Any]) -> Any:
            if data.get("synthesis_result", {}).get("status") == "failed":
                raise RuntimeError(SECRET)
            return real_update(conv_id, data)

        monkeypatch.setattr(rig.api, "update_context", refuse_failed)
        with pytest.raises(SystemExit) as caught:
            rig.runner.run(rig.conv, max_steps=10)
        assert caught.value.code == 3

    def test_a_real_provider_outage(self, valid_fsm):
        rig = _rig(valid_fsm, replies=[LLMResponseError(SECRET)])
        rig.runner.run(rig.conv, max_steps=10)
        assert rig.result["reason"] == "provider_error"
        assert SECRET not in repr(rig.data)

    def test_a_compile_that_never_passes(self, valid_fsm):
        rig = _rig(
            valid_fsm,
            replies=["no json here"],
            policy=SynthesisPolicy(max_repair_rounds=0),
        )
        rig.runner.run(rig.conv, max_steps=10)
        assert (rig.state, rig.result["reason"]) == ("apology", "compile_failed")
        assert len(rig.strong.requests) == 1

    def test_a_missing_input_obtains_nothing(self, valid_fsm):
        rig = _rig(valid_fsm, context={"unrelated": 1})
        rig.runner.run(rig.conv, max_steps=10)
        assert rig.result["reason"] == "inputs_missing"
        assert rig.strong.requests == [] and rig.names == ["failed"]

    def test_a_procedure_below_the_minimum_lifecycle_is_refused(
        self, valid_fsm, tmp_path
    ):
        rig = _rig(valid_fsm, tmp_path, config={"min_lifecycle": "trusted"})
        rig.runner.run(rig.conv, max_steps=10)
        assert rig.result["reason"] == "lifecycle_too_low"
        assert rig.result["procedure_id"] == rig.digest
        assert rig.names == ["obtained", "failed"]
        assert rig.library is not None
        assert rig.library.get_record(rig.digest).stats["runs"] == 0

    def test_a_procedure_with_a_completion_state_is_refused(
        self, valid_fsm, monkeypatch
    ):
        rig = _rig(valid_fsm)
        procedure = _procedure(valid_fsm)
        states = dict(procedure.definition.states)
        states["ask"] = states["ask"].model_copy(
            update={"completion": CompletionStateConfig(response_format_key="_fmt")}
        )
        definition = procedure.definition.model_copy(update={"states": states})
        _serve(rig, monkeypatch, Procedure(procedure.digest, definition, None, "x"))

        rig.runner.run(rig.conv, max_steps=10)

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
        rig = _rig(valid_fsm)
        data = _procedure(valid_fsm).definition.model_dump(exclude_defaults=True)
        tamper(data)
        definition = FSMDefinition(**data)
        _serve(
            rig, monkeypatch, Procedure(digest_of(definition), definition, None, "x")
        )

        rig.runner.run(rig.conv, max_steps=10)

        assert rig.result["reason"] == "library_error"
        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.names == ["obtained", "failed"]

    def test_a_push_that_fails(self, valid_fsm, monkeypatch):
        rig = _rig(valid_fsm)

        def refuse(*args: Any, **kwargs: Any) -> str:
            raise FSMError(SECRET)

        monkeypatch.setattr(rig.api, "push_fsm", refuse)
        rig.runner.run(rig.conv, max_steps=10)

        assert rig.result["reason"] == "push_failed"
        assert SECRET not in repr(rig.data) and SECRET not in repr(rig.events)
        assert rig.names == ["obtained", "failed"]


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


class TestNesting:
    def test_the_strict_default_policy_refuses_a_procedure_that_nests(
        self, valid_fsm, monkeypatch
    ):
        rig = _rig(valid_fsm, config={"return_keys": ["topic"]})
        procedure = _procedure(
            _nesting(),
            policy=SynthesisPolicy(max_synthesis_depth=2),
            return_keys=("topic",),
        )
        _serve(rig, monkeypatch, procedure)

        rig.runner.run(rig.conv, max_steps=10)

        # the synthesizer's policy (depth 1) fails G5 on the stored form
        assert rig.result["reason"] == "library_error"

    def test_depth_two_nests_once_and_refuses_the_third_level(
        self, valid_fsm, monkeypatch
    ):
        policy = SynthesisPolicy(max_synthesis_depth=2)
        rig = _rig(valid_fsm, policy=policy, config={"return_keys": ["topic"]})
        _serve(
            rig,
            monkeypatch,
            _procedure(_nesting(), policy=policy, return_keys=("topic",)),
        )

        rig.runner.run(rig.conv, max_steps=40)

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


class TestTraceSink:
    def test_fields_are_small_plain_values(self, valid_fsm):
        rig = _rig(valid_fsm)
        rig.runner.run(rig.conv, max_steps=20)
        for _, fields in rig.events:
            assert all(
                isinstance(v, (str, int, float, type(None))) for v in fields.values()
            ), fields

    def test_a_raising_sink_is_swallowed_with_one_warning_per_event(self, valid_fsm):
        seen: list[str] = []

        def sink(event: str, fields: dict[str, Any]) -> None:
            raise RuntimeError(SECRET)

        rig = _rig(valid_fsm, sink=sink)
        logger.enable("fsm_llm")
        handler = logger.add(lambda m: seen.append(str(m)), level="WARNING")
        try:
            rig.runner.run(rig.conv, max_steps=20)
        finally:
            logger.remove(handler)
            logger.disable("fsm_llm")

        assert rig.state == "booked"
        warnings = [m for m in seen if "trace_sink" in m]
        assert len(warnings) == 3 and not any(SECRET in m for m in warnings)

    def test_a_non_callable_sink_is_refused(self, valid_fsm):
        rig = _rig(valid_fsm)
        with pytest.raises(TypeError):
            ProcedureRunner(rig.api, _host(), rig.synth, trace_sink="not callable")  # type: ignore[arg-type]


class TestObtainedEvent:
    def _obtained(self, rig: _Rig) -> dict[str, Any]:
        rig.runner.run(rig.conv, max_steps=20)
        return next(f for name, f in rig.events if name == "obtained")

    def test_it_carries_round_codes_seconds_and_token_counts(
        self, valid_fsm, monkeypatch
    ):
        rig = _rig(valid_fsm)
        procedure = dataclasses.replace(
            _procedure(valid_fsm),
            rounds=(("G5", "G4"), ("G6b",)),
            seconds=1.25,
            usage=LLMUsage(calls=3, prompt_tokens=40, completion_tokens=9),
        )
        _serve(rig, monkeypatch, procedure)
        fields = self._obtained(rig)
        assert fields["rounds"] == "G5,G4;G6b" and fields["seconds"] == 1.25
        assert (fields["calls"], fields["prompt_tokens"]) == (3, 40)
        assert fields["completion_tokens"] == 9

    def test_a_first_pass_compile_has_empty_rounds_and_zero_usage(self, valid_fsm):
        fields = self._obtained(_rig(valid_fsm))
        assert fields["rounds"] == "" and fields["calls"] == 0
        assert isinstance(fields["seconds"], float) and fields["seconds"] >= 0

    def test_two_repair_rounds_arrive_as_codes_only(self, valid_fsm):
        bad = copy.deepcopy(valid_fsm)
        bad["persona"] = SECRET
        bad["states"]["ask"]["transitions"][0]["priority"] = 0
        rig = _rig(valid_fsm, replies=[SECRET, json.dumps(bad), json.dumps(valid_fsm)])
        fields = self._obtained(rig)
        assert (
            fields["rounds"].count(";") == 1
            and "reply-not-json-object" in fields["rounds"]
        )
        assert SECRET not in repr(fields)

    def test_a_library_hit_is_source_library_with_empty_rounds(
        self, valid_fsm, tmp_path
    ):
        first = _rig(valid_fsm, tmp_path)
        first.runner.run(first.conv, max_steps=20)
        fields = self._obtained(_rig(valid_fsm, tmp_path, replies=[]))
        assert (fields["source"], fields["rounds"], fields["calls"]) == (
            "library",
            "",
            0,
        )
        assert 0 <= fields["seconds"] < 5.0


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


class TestNoForeignInterface:
    """A compile may not move the child to a model of its own: nothing but the
    host's injected interface is ever built or called (review iter 1, concern 1)."""

    def test_a_compile_that_names_a_model_is_repaired_before_it_runs(
        self, valid_fsm, foreign_calls
    ):
        bad = json.dumps(_classified(valid_fsm, model="openai/gpt-4o"))
        rig = _rig(valid_fsm, replies=[bad, json.dumps(valid_fsm)])

        rig.runner.run(rig.conv, max_steps=20)

        assert foreign_calls == []
        assert "model-setting" in str(rig.strong.requests[1].messages)
        assert (rig.state, rig.result["status"]) == ("booked", "ok")

    def test_a_compile_that_keeps_naming_one_never_runs(self, valid_fsm, foreign_calls):
        bad = json.dumps(_classified(valid_fsm, model="openai/gpt-4o"))
        rig = _rig(valid_fsm, replies=[bad] * 4)

        rig.runner.run(rig.conv, max_steps=20)

        assert foreign_calls == []
        assert rig.result["status"] == "failed"
        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.names == ["failed"]

    def test_a_stored_legacy_procedure_with_a_model_is_refused_at_push(
        self, valid_fsm, monkeypatch, foreign_calls
    ):
        rig = _rig(valid_fsm)
        data = _procedure(valid_fsm).definition.model_dump(exclude_defaults=True)
        data["states"]["ask"]["classification_extractions"] = _classified(
            valid_fsm, model="openai/gpt-4o"
        )["states"]["ask"]["classification_extractions"]
        definition = FSMDefinition(**data)
        _serve(
            rig, monkeypatch, Procedure(digest_of(definition), definition, None, "x")
        )

        rig.runner.run(rig.conv, max_steps=10)

        assert foreign_calls == []
        assert rig.result["reason"] == "library_error"
        assert (rig.state, rig.depth) == ("apology", 1)
        assert rig.names == ["obtained", "failed"]


CANARIES = ("CANARY-CONTEXT", "CANARY-USER", "CANARY-MODEL")


def _tied(valid_fsm: dict[str, Any]) -> dict[str, Any]:
    """``ask`` has two edges at one priority: ``answer`` alone reaches ``done``,
    ``answer`` plus ``extra`` also reaches ``done2`` (a tie only when both are set)."""
    data = copy.deepcopy(valid_fsm)
    ask = data["states"]["ask"]
    ask["field_extractions"].append(
        {"field_name": "extra", "field_type": "str", "extraction_instructions": "x"}
    )
    second = copy.deepcopy(ask["transitions"][0])
    second["target_state"] = "done2"
    second["conditions"][0]["requires_context_keys"] = ["answer", "extra"]
    ask["transitions"].append(second)
    data["states"]["done2"] = copy.deepcopy(data["states"]["done"])
    data["states"]["done2"]["id"] = "done2"
    return data


def _abort_at_once(rig: _Rig, monkeypatch: pytest.MonkeyPatch, fsm: dict[str, Any]):
    """Push ``fsm`` and return a ``step -> None`` hook; ``max_turns`` is 1, so
    the second hook call counts the child's first step and sets the abort."""
    _serve(rig, monkeypatch, _procedure(fsm))
    rig.to_solve()
    hook = rig.runner.hook(rig.conv)
    hook(2)
    assert rig.depth == 2
    return hook


class TestTraceEvents:
    """The events a ``TraceRecorder`` reads: the conversation on every one, a
    ``stepped`` event per counted child step, the stuck state and live outcome
    at an abort, the error class and round codes at a failure. Names, type
    names, state ids, counts and codes only (D-038, D-044)."""

    def test_a_canary_never_reaches_an_event_of_an_ok_run(self, valid_fsm):
        rig = _rig(
            valid_fsm,
            extraction={"answer": CANARIES[0]},
            context={**CONTEXT, "topic": CANARIES[0]},
        )
        rig.cheap.responses["ask"] = CANARIES[2]

        rig.runner.converse(rig.conv, "hello")
        rig.runner.converse(rig.conv, CANARIES[1])

        assert rig.result["status"] == "ok" and rig.data["answer"] == CANARIES[0]
        assert len(rig.steps) == 1  # the scan below is not vacuous
        blob = json.dumps(rig.stream)
        assert not [c for c in CANARIES if c in blob]

    def test_a_canary_never_reaches_an_event_of_an_aborted_run(self, valid_fsm):
        rig = _rig(
            valid_fsm,
            extraction={},
            policy=SynthesisPolicy(max_turns=2),
            context={**CONTEXT, "topic": CANARIES[0]},
        )
        rig.cheap.responses["ask"] = CANARIES[2]

        for text in ("hello", CANARIES[1], CANARIES[1]):
            rig.runner.converse(rig.conv, text)

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
        rig = _rig(valid_fsm)

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise error

        monkeypatch.setattr(rig.synth, "obtain", refuse)
        rig.runner.run(rig.conv, max_steps=10)

        (failed,) = [f for n, f in rig.events if n == "failed"]
        assert failed["error"] == name and failed["rounds"] == rounds
        assert failed["conv"] == rig.conv
        assert CANARIES[2] not in json.dumps(rig.stream)

    def test_an_interrupt_names_its_class(self, valid_fsm, monkeypatch):
        rig = _rig(valid_fsm)

        def interrupt(*args: Any, **kwargs: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(rig.synth, "obtain", interrupt)
        with pytest.raises(KeyboardInterrupt):
            rig.runner.run(rig.conv, max_steps=10)

        (failed,) = [f for n, f in rig.events if n == "failed"]
        assert failed["error"] == "KeyboardInterrupt"

    def test_a_real_compile_that_never_passes_reports_its_round_codes(self, valid_fsm):
        rig = _rig(valid_fsm, replies=[CANARIES[2]] * 8)

        rig.runner.run(rig.conv, max_steps=10)

        (failed,) = [f for n, f in rig.events if n == "failed"]
        assert (
            failed["error"] == "SynthesisError" and failed["reason"] == "compile_failed"
        )
        rounds = failed["rounds"].split(";")
        assert len(rounds) == len(rig.strong.requests) > 1
        assert all(rounds) and len(set(rounds)) == 1
        assert CANARIES[2] not in json.dumps(rig.stream)

    def test_a_failure_without_an_exception_has_no_error(self, valid_fsm):
        rig = _rig(valid_fsm, context={"unrelated": 1})
        rig.runner.run(rig.conv, max_steps=10)
        (failed,) = [f for n, f in rig.events if n == "failed"]
        assert failed["reason"] == "inputs_missing"
        assert failed["error"] is None and failed["rounds"] is None

    def test_two_conversations_on_one_runner_give_separable_streams(self, valid_fsm):
        rig = _rig(valid_fsm, replies=[json.dumps(valid_fsm)] * 2)
        other, _ = rig.api.start_conversation(CONTEXT)

        rig.runner.converse(rig.conv, "hello")
        rig.runner.converse(other, "hello")
        rig.runner.converse(other, "forty-two")
        rig.runner.converse(rig.conv, "forty-two")

        assert all("conv" in f for _, f in rig.stream)
        for conv in (rig.conv, other):
            mine = [(n, f) for n, f in rig.stream if f["conv"] == conv]
            # the first turn is the parent's own; the second is the child's
            assert [n for n, _ in mine] == ["obtained", "pushed", "stepped", "popped"]
            assert [f["step"] for n, f in mine if n == "stepped"] == [1]
            assert len({f["digest"] for _, f in mine}) == 1
        assert {f["conv"] for _, f in rig.stream} == {rig.conv, other}

    def test_every_event_kind_carries_conv(self, valid_fsm, monkeypatch):
        ok = _rig(valid_fsm)
        ok.runner.run(ok.conv, max_steps=20)
        aborted = _rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=2))
        aborted.runner.run(aborted.conv, max_steps=20)
        failed = _rig(valid_fsm, context={"unrelated": 1})
        failed.runner.run(failed.conv, max_steps=20)
        seen = {n for r in (ok, aborted, failed) for n, _ in r.stream}
        assert seen == {"obtained", "pushed", "stepped", "aborted", "popped", "failed"}
        for rig in (ok, aborted, failed):
            assert all(f["conv"] == rig.conv for _, f in rig.stream)

    @pytest.mark.parametrize("mode", ["run", "converse"])
    def test_stepped_fires_once_per_counted_step_in_both_modes(self, valid_fsm, mode):
        rig = _rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=3))

        if mode == "run":
            rig.runner.run(rig.conv, max_steps=30)
        else:
            for text in ("a", "b", "c", "d"):
                rig.runner.converse(rig.conv, text)

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
        rig = _rig(valid_fsm, extraction={"answer": "42", "stray": "q"})

        rig.runner.converse(rig.conv, "hello")
        rig.runner.converse(rig.conv, "forty-two")

        last = rig.steps[-1]
        assert last["state"] == "done"
        assert last["keys"] == "answer,topic"  # the input and the return key
        assert last["kinds"] == "answer:str"  # only what a gate reads
        assert "stray" not in json.dumps(rig.stream)

    def test_an_undeclared_name_in_the_child_is_never_listed(
        self, valid_fsm, monkeypatch
    ):
        rig = _rig(valid_fsm, extraction={})
        hook = _abort_at_once(rig, monkeypatch, valid_fsm)
        rig.api.update_context(rig.conv, {"answer": "x", "minted": CANARIES[2]})

        hook(3)

        assert rig.steps[-1]["keys"] == "answer,topic"
        assert "minted" not in json.dumps(rig.stream)

    @pytest.mark.parametrize(
        ("value", "kind"),
        [
            (True, "bool"),
            (3, "int"),
            (1.5, "float"),
            ("s", "str"),
            ([1], "list"),
            ({"a": 1}, "dict"),
            (type("S", (str,), {})("sub"), "other"),
            (object(), "other"),
        ],
    )
    def test_a_type_is_named_from_a_closed_table(
        self, valid_fsm, monkeypatch, value, kind
    ):
        rig = _rig(valid_fsm, extraction={})
        hook = _abort_at_once(rig, monkeypatch, valid_fsm)
        rig.api.update_context(rig.conv, {"answer": value})

        hook(3)

        assert rig.steps[-1]["kinds"] == f"answer:{kind}"
        assert "SubclassOfStr" not in json.dumps(rig.stream)

    @pytest.mark.parametrize(
        ("context", "outcome"),
        [
            ({}, "blocked"),
            ({"answer": "a"}, "deterministic"),
            ({"answer": "a", "extra": "b"}, "ambiguous"),
        ],
    )
    def test_an_abort_names_the_stuck_state_and_the_live_outcome(
        self, valid_fsm, monkeypatch, context, outcome
    ):
        rig = _rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=1))
        hook = _abort_at_once(rig, monkeypatch, _tied(valid_fsm))
        rig.api.update_context(rig.conv, context)

        hook(3)

        (aborted,) = [f for n, f in rig.events if n == "aborted"]
        assert aborted["state"] == "ask" and aborted["outcome"] == outcome
        assert aborted["conv"] == rig.conv and aborted["step"] == 3
        # the abort key was written after the outcome was read, not before
        assert rig.api.get_data(rig.conv).get("procedure_abort") is True

    def test_an_event_that_cannot_be_read_never_changes_the_run(
        self, valid_fsm, monkeypatch
    ):
        class Broken:
            def evaluate_transitions(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError(CANARIES[2])

        monkeypatch.setattr(runner_module, "TransitionEvaluator", Broken)
        rig = _rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=2))

        rig.runner.run(rig.conv, max_steps=20)

        assert (rig.state, rig.result["reason"]) == ("apology", "budget_spent")
        (aborted,) = [f for n, f in rig.events if n == "aborted"]
        assert "outcome" not in aborted and aborted["conv"] == rig.conv
        assert CANARIES[2] not in json.dumps(rig.stream)

    def test_without_a_sink_nothing_is_read(self, valid_fsm, monkeypatch):
        rig = _rig(valid_fsm, extraction={}, policy=SynthesisPolicy(max_turns=2))
        quiet = ProcedureRunner(rig.api, _host(), rig.synth)

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("read without a sink")

        monkeypatch.setattr(runner_module, "TransitionEvaluator", boom)
        quiet.run(rig.conv, max_steps=20)
        assert rig.result["status"] == "aborted"
