"""
Tests for ``fsm_llm.synthesis.runner.ProcedureRunner`` on REAL ``API`` objects.

The host is a concierge FSM with one ``synthesize`` state; the strong model is
``ScriptedStrongLLM``, the cheap model under the pushed procedure is
``ScriptedRunnerLLM`` (both in conftest). The runner is driven the two ways it
supports: ``run`` (core ``run_until_terminal`` with the runner's hook) and
``converse`` (user turns). Failure tests read the result the PARENT sees.
"""

import copy
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
from tests.test_fsm_llm_synthesis.conftest import ScriptedRunnerLLM, ScriptedStrongLLM

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
    status = {"var": "synthesis_result.status"}

    def edge(target: str, priority: int, logic: dict[str, Any]) -> dict[str, Any]:
        condition = {
            "description": f"go to {target}",
            "requires_context_keys": ["synthesis_result"],
            "logic": logic,
        }
        return {
            "target_state": target,
            "description": f"go to {target}",
            "priority": priority,
            "conditions": [condition],
        }

    def end(name: str, text: str) -> dict[str, Any]:
        return {
            "id": name,
            "description": name,
            "purpose": name,
            "response_instructions": text,
        }

    config = {
        "goal": GOAL,
        "inputs": ["topic"],
        "return_keys": ["answer"],
        "on_failure": "apology",
        **synthesize,
    }
    return {
        "name": "Concierge",
        "description": "Host conversation",
        "persona": PERSONA,
        "initial_state": "start",
        "states": {
            "start": {
                "id": "start",
                "description": "greet",
                "purpose": "greet",
                "response_instructions": "Say hello",
                "transitions": [{"target_state": "solve", "description": "go"}],
            },
            "solve": {
                "id": "solve",
                "description": "run the procedure",
                "purpose": "run the procedure",
                "response_instructions": "",
                "synthesize": config,
                "transitions": [
                    edge("booked", 1, {"==": [status, "ok"]}),
                    edge("apology", 2, {"in": [status, ["failed", "aborted"]]}),
                ],
            },
            "booked": end("booked", "Say booked"),
            "apology": end("apology", "Apologise"),
        },
    }


@dataclass
class _Rig:
    api: API
    runner: ProcedureRunner
    conv: str
    cheap: ScriptedRunnerLLM
    strong: ScriptedStrongLLM
    synth: FSMSynthesizer
    library: FSMLibrary | None
    events: list[tuple[str, dict[str, Any]]]

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
        {"answer": "42"} if extraction is None else extraction, REPLIES
    )
    strong = ScriptedStrongLLM([json.dumps(valid_fsm)] if replies is None else replies)
    library = FSMLibrary(tmp_path / "lib") if tmp_path is not None else None
    synth = FSMSynthesizer(strong, policy=policy, library=library)
    host = _host(**(config or {}))
    api = API.from_definition(definition=host, llm_interface=cheap)
    events: list[tuple[str, dict[str, Any]]] = []
    runner = ProcedureRunner(
        api, host, synth, trace_sink=sink or (lambda e, f: events.append((e, dict(f))))
    )
    conv, _ = api.start_conversation(CONTEXT if context is None else context)
    return _Rig(api, runner, conv, cheap, strong, synth, library, events)


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
                isinstance(v, (str, int, type(None))) for v in fields.values()
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
