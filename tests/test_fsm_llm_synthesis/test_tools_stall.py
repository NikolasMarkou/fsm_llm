"""
Tests for the early stop of ``run_procedure`` (plan 51fde95b D-027, extra
finding E3): a procedure that waits for a user cannot finish message-free, so
the run stops after ``policy.dwell_threshold`` steps that changed nothing,
tells ``needs_user_input`` and records no failed run.
"""

import json
from typing import Any

import pytest

from fsm_llm.engine.api.conversation import API
from fsm_llm.synthesis import SynthesisPolicy, create_procedure_tools
from fsm_llm.synthesis.constants import PROCEDURE_UNTRUSTED_LABEL as LABEL
from tests.test_fsm_llm_synthesis.conftest import ScriptedRunnerLLM

from .test_tools import _fields, _Outage, _rig, _run

STUCK = f"{LABEL}\nfailed: needs_user_input"


class _Late(ScriptedRunnerLLM):
    """Extracts nothing for the first ``stays`` steps, then the answer. One reply
    is written per step after the greeting of the start, so step k sees k."""

    def __init__(self, stays: int) -> None:
        super().__init__({"answer": "42"})
        self.stays = stays

    def extract_field(self, request: Any) -> Any:
        late = self.count("generate_response") > self.stays
        self.extraction_data = {"answer": "42"} if late else {}
        return super().extract_field(request)


class _Drip(ScriptedRunnerLLM):
    """Answers ``plan[k]`` fields of ``a b c d`` on step k+1 (default one more on
    each step): the state stays, the data grows."""

    def __init__(self, plan: tuple[int, ...] = (1, 2, 3, 4)) -> None:
        super().__init__({})
        self.plan = plan

    def extract_field(self, request: Any) -> Any:
        step = self.count("generate_response")  # the greeting counts: step k sees k
        have = "abcd"[: self.plan[min(step, len(self.plan)) - 1]]
        self.extraction_data = dict.fromkeys(have, "x")
        return super().extract_field(request)


def _fields_fsm() -> dict[str, Any]:
    ask = {
        "id": "ask",
        "description": "ask",
        "purpose": "ask",
        "response_instructions": "Ask",
        "field_extractions": [
            {"field_name": k, "field_type": "str", "extraction_instructions": k}
            for k in "abcd"
        ],
        "transitions": [
            {
                "target_state": "done",
                "description": "all four",
                "priority": 1,
                "conditions": [
                    {"description": "all four", "requires_context_keys": list("abcd")}
                ],
            }
        ],
    }
    done = {
        "id": "done",
        "description": "d",
        "purpose": "d",
        "response_instructions": "",
    }
    return {
        "name": "P",
        "description": "p",
        "initial_state": "ask",
        "states": {"ask": ask, "done": done},
    }


def _serve(rig: Any, cheap: Any) -> None:
    rig.cheap = cheap
    rig.registry = create_procedure_tools(rig.synth, runner_llm=cheap)


class TestAWaitingProcedureStopsEarly:
    @pytest.mark.parametrize("stays", [2, 3, 5])
    def test_it_stops_after_the_dwell_threshold_and_counts_no_run(
        self, valid_fsm, tmp_path, stays
    ):
        rig = _rig(
            valid_fsm,
            tmp_path,
            extraction={},
            policy=SynthesisPolicy(max_turns=40, dwell_threshold=stays),
        )
        rig.obtain()

        assert _run(rig) == STUCK

        # the greeting of the start, then one reply per step: not forty steps
        assert rig.cheap.count("generate_response") == stays + 1
        assert rig.library.get_record(rig.digest()).stats == {"runs": 0, "successes": 0}

    def test_the_sub_api_is_still_closed(self, valid_fsm, tmp_path, monkeypatch):
        closed: list[API] = []
        real = API.close
        monkeypatch.setattr(API, "close", lambda s: (closed.append(s), real(s))[1])
        rig = _rig(valid_fsm, tmp_path, extraction={})
        rig.obtain()
        assert _run(rig) == STUCK
        assert len(closed) == 1

    def test_a_spent_budget_below_the_threshold_still_counts_one_failure(
        self, valid_fsm, tmp_path
    ):
        rig = _rig(
            valid_fsm, tmp_path, extraction={}, policy=SynthesisPolicy(max_turns=2)
        )
        rig.obtain()
        assert _run(rig) == STUCK
        stats = rig.library.get_record(rig.digest()).stats
        assert stats == {"runs": 1, "successes": 0}

    def test_an_outage_is_still_a_provider_error(self, valid_fsm, tmp_path):
        rig = _rig(valid_fsm, tmp_path)
        rig.obtain()
        _serve(rig, _Outage(10_000, extraction_data={"answer": "42"}))
        assert _run(rig) == f"{LABEL}\nfailed: provider_error"
        assert rig.library.get_record(rig.digest()).stats == {"runs": 0, "successes": 0}
        assert rig.cheap.count("generate_response") == 4  # greeting, 3 steps


class TestAProcedureThatMovesIsNotStopped:
    def test_one_that_answers_on_its_third_step_finishes(self, valid_fsm, tmp_path):
        rig = _rig(valid_fsm, tmp_path)
        rig.obtain()
        _serve(rig, _Late(2))  # two steps that change nothing, then the answer
        assert _fields(_run(rig))["status"] == "ok"
        assert rig.library.get_record(rig.digest()).stats == {"runs": 1, "successes": 1}

    def test_one_that_answers_a_step_later_is_the_waiting_one(
        self, valid_fsm, tmp_path
    ):
        rig = _rig(valid_fsm, tmp_path)
        rig.obtain()
        _serve(rig, _Late(3))  # three steps that change nothing: it waits
        assert _run(rig) == STUCK
        assert rig.library.get_record(rig.digest()).stats == {"runs": 0, "successes": 0}

    def test_a_state_that_stays_while_its_data_grows_finishes(
        self, valid_fsm, tmp_path
    ):
        rig = _rig(valid_fsm, tmp_path, replies=[json.dumps(_fields_fsm())])
        rig.obtain(return_keys=["d"])
        _serve(rig, _Drip())
        assert _fields(_run(rig))["status"] == "ok"
        assert rig.cheap.count("generate_response") == 4

    def test_stays_of_different_runs_of_the_state_are_not_added_up(
        self, valid_fsm, tmp_path
    ):
        rig = _rig(valid_fsm, tmp_path, replies=[json.dumps(_fields_fsm())])
        rig.obtain(return_keys=["d"])
        _serve(rig, _Drip((1, 1, 1, 2, 2, 2, 3, 3, 3, 4)))  # two stays, then news
        assert _fields(_run(rig))["status"] == "ok"
