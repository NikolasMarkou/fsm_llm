"""
End to end, offline: a concierge host FSM whose ``gather`` state is a
``synthesize`` state, on REAL ``API`` objects.

The strong model is ``ScriptedStrongLLM`` and the cheap model (the host's own
interface, under the pushed child too) is ``ScriptedRunnerLLM``, both from
conftest; the network is blocked. Every scenario runs both ways the runner
drives a conversation: ``converse`` (user turns drive the child) and ``run``
(``run_until_terminal`` with the runner's hook). One more test drives the agent
tools pair.
"""

import json
from dataclasses import dataclass
from typing import Any

import pytest

from fsm_llm.agents.definitions import ToolCall
from fsm_llm.api import API
from fsm_llm.synthesis import (
    FSMLibrary,
    FSMSynthesizer,
    ProcedureRunner,
    SynthesisPolicy,
    create_procedure_tools,
)
from fsm_llm.synthesis.constants import PROCEDURE_UNTRUSTED_LABEL as LABEL
from tests.test_fsm_llm_synthesis.conftest import (
    ScriptedRunnerLLM,
    ScriptedStrongLLM,
    host_fsm,
)

GOAL = "collect visa application data"
KEYS = ["passport_no", "travel_dates", "purpose"]
ANSWERS = {
    "passport_no": "X1234567",
    "travel_dates": "May 1 to 9",
    "purpose": "tourism",
}
LEAK = "LEAKED-MODEL-TEXT"
REPLIES = {"ask": "What are your visa details?", "done": "Details noted"}
HOST_REPLIES = {"welcome": "Welcome", "done": "All set", "apology": "Sorry"}
MODES = pytest.mark.parametrize("mode", ["converse", "run"])


def _visa(*, guaranteed: bool = True, leak: str = "") -> dict[str, Any]:
    """A compile for ``GOAL``. ``guaranteed=False`` leaves ``purpose`` out of the
    edge's ``requires_context_keys``: G3 ``return-key-not-guaranteed``."""
    fields = [
        {"field_name": k, "field_type": "str", "extraction_instructions": k}
        for k in KEYS
    ]
    needed = KEYS if guaranteed else KEYS[:2]
    condition = {"description": "has data", "requires_context_keys": needed}
    return {
        "name": "Visa" + leak,
        "description": "Collect visa application data",
        "initial_state": "ask",
        "states": {
            "ask": {
                "id": "ask",
                "description": "ask",
                "purpose": "get the details",
                "response_instructions": "Ask for the details",
                "field_extractions": fields,
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "have all",
                        "priority": 1,
                        "conditions": [condition],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "finished",
                "purpose": "wrap up",
                "response_instructions": "Confirm",
            },
        },
    }


def _text(data: dict[str, Any]) -> str:
    return json.dumps(data)


def _host() -> dict[str, Any]:
    """greet -> gather (synthesize) -> done on ok, apology on failed or aborted."""
    return host_fsm(
        name="concierge",
        description="Visa concierge",
        persona="A calm travel concierge",
        states=("greet", "gather", "done", "apology"),
        goal=GOAL,
        inputs=["applicant"],
        return_keys=KEYS,
        greet_text="Welcome the traveller",
        work_description="collect the visa data",
        ok_text="done",
        fail_text="apology",
    )


@dataclass
class _Rig:
    api: API
    runner: ProcedureRunner
    cheap: ScriptedRunnerLLM
    strong: ScriptedStrongLLM
    library: FSMLibrary | None

    def start(self) -> str:
        conv, _ = self.api.start_conversation({"applicant": "Ada", "noise": "N-1"})
        return conv

    def drive(self, conv: str, mode: str) -> None:
        """Run ``conv`` to its end: user turns, or one message-free run."""
        if mode == "run":
            assert self.runner.run(conv, max_steps=80)[-1].ended
            return
        for turn in range(12):
            if self.api.has_conversation_ended(conv):
                return
            self.runner.converse(conv, f"message {turn}")
        raise AssertionError("the conversation did not end in 12 turns")

    def finished(self, conv: str) -> tuple[str, int, dict[str, Any]]:
        api = self.api
        return (
            api.get_current_state(conv),
            api.get_stack_depth(conv),
            api.get_data(conv),
        )


def _rig(
    tmp_path: Any,
    replies: list[Any],
    *,
    extraction: dict[str, Any] | None = None,
    policy: SynthesisPolicy | None = None,
    library: bool = True,
) -> _Rig:
    cheap = ScriptedRunnerLLM(
        ANSWERS if extraction is None else extraction, {**REPLIES, **HOST_REPLIES}
    )
    strong = ScriptedStrongLLM(replies)
    lib = FSMLibrary(tmp_path / "lib") if library else None
    synth = FSMSynthesizer(strong, policy=policy, library=lib)
    host = _host()
    api = API.from_definition(definition=host, llm_interface=cheap)
    return _Rig(api, ProcedureRunner(api, host, synth), cheap, strong, lib)


def _good_after_bad() -> list[str]:
    return [_text(_visa(guaranteed=False)), _text(_visa())]


@MODES
class TestRepairThenRun:
    def test_repair_converges_in_two_strong_calls(self, tmp_path, mode):
        rig = _rig(tmp_path, _good_after_bad())
        rig.drive(rig.start(), mode)

        assert len(rig.strong.requests) == 2
        second = rig.strong.requests[1].messages
        assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
        assert second[2]["content"] == _text(_visa(guaranteed=False))
        assert "return-key-not-guaranteed" in second[3]["content"]

    def test_the_child_runs_on_the_cheap_model_and_the_parent_gets_the_keys(
        self, tmp_path, mode
    ):
        rig = _rig(tmp_path, _good_after_bad())
        conv = rig.start()
        rig.drive(conv, mode)

        state, depth, data = rig.finished(conv)
        assert (state, depth) == ("done", 1)
        assert {k: data[k] for k in KEYS} == ANSWERS
        assert data["synthesis_result"]["status"] == "ok"
        assert "abort" not in data and "procedure_abort" not in data
        # the strong model only completed; the cheap one extracted and replied
        assert all(r.tools is None for r in rig.strong.requests)
        assert rig.cheap.count("extract_field") >= len(KEYS)
        assert rig.cheap.count("generate_response", "ask") >= 1

    def test_the_library_holds_one_verified_record_with_its_gates(self, tmp_path, mode):
        rig = _rig(tmp_path, _good_after_bad())
        conv = rig.start()
        rig.drive(conv, mode)

        assert rig.library is not None
        assert len(list((rig.library.root / "records").glob("*.json"))) == 1
        digest = rig.finished(conv)[2]["synthesis_result"]["procedure_id"]
        record = rig.library.get_record(digest)
        assert record.lifecycle == "verified"
        assert record.provenance["gates_passed"] == ["G1", "G2", "G3", "G4", "G5"]
        assert record.stats == {"runs": 1, "successes": 1}

    def test_a_second_conversation_hits_the_library_with_no_strong_call(
        self, tmp_path, mode
    ):
        rig = _rig(tmp_path, _good_after_bad())
        first = rig.start()
        rig.drive(first, mode)
        assert rig.strong.replies == []  # any further strong call would raise

        second = rig.start()
        rig.drive(second, mode)

        state, depth, data = rig.finished(second)
        assert (state, depth) == ("done", 1)
        assert {k: data[k] for k in KEYS} == ANSWERS
        assert len(rig.strong.requests) == 2
        assert rig.library is not None
        digest = data["synthesis_result"]["procedure_id"]
        assert digest == rig.finished(first)[2]["synthesis_result"]["procedure_id"]
        assert rig.library.get_record(digest).stats == {"runs": 2, "successes": 2}


@MODES
class TestFailurePaths:
    def test_a_child_that_never_finishes_aborts_and_the_parent_apologises(
        self, tmp_path, mode
    ):
        rig = _rig(
            tmp_path,
            [_text(_visa())],
            extraction={},
            policy=SynthesisPolicy(max_turns=3),
        )
        conv = rig.start()
        rig.drive(conv, mode)

        state, depth, data = rig.finished(conv)
        assert (state, depth) == ("apology", 1)
        assert data["synthesis_result"]["status"] == "aborted"
        assert data["synthesis_result"]["reason"] == "budget_spent"
        assert not any(k in data for k in KEYS)
        assert rig.library is not None
        digest = data["synthesis_result"]["procedure_id"]
        assert rig.library.get_record(digest).stats == {"runs": 1, "successes": 0}

    def test_an_always_bad_compile_exhausts_the_repairs_and_the_parent_apologises(
        self, tmp_path, mode
    ):
        bad = _text(_visa(guaranteed=False, leak=LEAK))
        rig = _rig(tmp_path, [bad] * 3, policy=SynthesisPolicy(max_repair_rounds=2))
        conv = rig.start()
        rig.drive(conv, mode)

        state, depth, data = rig.finished(conv)
        assert (state, depth) == ("apology", 1)
        assert len(rig.strong.requests) == 3  # 1 + max_repair_rounds, no more
        assert data["synthesis_result"] == {
            "status": "failed",
            "procedure_id": None,
            "version": None,
            "reason": "compile_failed",
        }
        assert LEAK not in repr(data)
        assert not any(k in data for k in KEYS)
        assert rig.cheap.count("extract_field") == 0  # nothing was pushed
        assert rig.library is not None
        assert list((rig.library.root / "records").glob("*.json")) == []


class TestTools:
    def test_obtain_then_run_a_message_free_procedure(self, tmp_path):
        cheap = ScriptedRunnerLLM(ANSWERS, REPLIES)
        strong = ScriptedStrongLLM(_good_after_bad())
        library = FSMLibrary(tmp_path / "lib")
        registry = create_procedure_tools(
            FSMSynthesizer(strong, library=library), runner_llm=cheap
        )

        def call(tool: str, **params: Any) -> dict[str, str]:
            result = registry.execute(ToolCall(tool_name=tool, parameters=params))
            assert result.success, result.error
            lines = str(result.result).splitlines()
            assert lines[0] == LABEL
            return dict(line.split(": ", 1) for line in lines[1:] if ": " in line)

        receipt = call(
            "obtain_procedure", goal=GOAL, return_keys=KEYS, inputs=["applicant"]
        )
        assert len(strong.requests) == 2  # the repair ran inside obtain
        assert receipt["lifecycle"] == "verified"
        digest = receipt["procedure_id"]
        assert library.get_record(digest).provenance["gates_passed"][-1] == "G5"

        reply = call("run_procedure", procedure_id=digest, inputs={"applicant": "Ada"})
        assert reply == {"procedure_id": digest, "status": "ok", **ANSWERS}
        assert library.get_record(digest).stats == {"runs": 1, "successes": 1}
        assert cheap.count("extract_field") >= len(KEYS)
        assert len(strong.requests) == 2  # a run never calls the strong model
