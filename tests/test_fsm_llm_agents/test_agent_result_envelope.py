"""The 0.16 adoption in agents: StopReason is an Enum, AgentResult carries status.

Plan ace7a8c8 step 10, decision D-046. The truth table below is the OLD one, as
literals: success only for ``answered`` and ``evidence``.
"""

from __future__ import annotations

import json
from enum import Enum

import pytest
from pydantic import BaseModel, ValidationError

from fsm_llm.agents import AgentResult
from fsm_llm.agents.constants import FORCED_STOP_REASONS, StopReason
from fsm_llm.agents.definitions import AgentTrace, MetaBuilderResult
from fsm_llm.agents.remote import RemoteAgentTool
from fsm_llm.core.codes import Reason, Status, status_of_stop_reason
from fsm_llm.core.results import Identity, RunResult

# (member name, wire string, old success)
OLD_TABLE = [
    ("ANSWERED", "answered", True),
    ("EVIDENCE", "evidence", True),
    ("MAX_ITERATIONS", "max_iterations", False),
    ("FORCED_PASS", "forced_pass", False),
    ("STALLED", "stalled", False),
    ("VERIFICATION_FAILED", "verification_failed", False),
    ("NO_RESULT", "no_result", False),
    ("GATE_FAILED", "gate_failed", False),
    ("ENDED", "ended", False),
    ("TRUNCATED", "truncated", False),
]
OLD_FORCED = {
    "max_iterations",
    "forced_pass",
    "stalled",
    "gate_failed",
    "ended",
    "truncated",
}


class TestStopReasonIsAnEnum:
    def test_a_str_enum_with_upper_members_and_snake_values(self):
        assert issubclass(StopReason, str) and issubclass(StopReason, Enum)
        assert [m.name for m in StopReason] == [n for n, _, _ in OLD_TABLE]

    @pytest.mark.parametrize(("name", "wire", "_ok"), OLD_TABLE)
    def test_every_member_keeps_its_wire_string(self, name, wire, _ok):
        member = StopReason[name]
        assert member.value == wire
        assert member == wire
        assert hash(member) == hash(wire)
        assert StopReason.from_string(wire) is member
        assert StopReason.from_string(member) is member

    def test_from_string_refuses_unknown_values(self):
        with pytest.raises(ValueError, match="stop_reason"):
            StopReason.from_string("brand_new")

    def test_forced_stop_reasons_is_the_old_set_and_left_the_class(self):
        assert set(FORCED_STOP_REASONS) == OLD_FORCED
        assert isinstance(FORCED_STOP_REASONS, frozenset)
        assert "FORCED" not in StopReason.__members__
        assert not hasattr(StopReason, "FORCED")
        assert "stalled" in FORCED_STOP_REASONS
        assert "answered" not in FORCED_STOP_REASONS


class TestSuccessIsDerivedFromStatus:
    @pytest.mark.parametrize(("name", "wire", "old_success"), OLD_TABLE)
    def test_success_equals_status_ok_with_the_old_truth_table(
        self, name, wire, old_success
    ):
        result = AgentResult(answer="a", stop_reason=StopReason[name])
        assert result.success is old_success
        assert result.success == (result.status is Status.OK)
        status, reason = status_of_stop_reason(wire)
        assert (result.status, result.reason) == (status, reason)

    @pytest.mark.parametrize(("name", "wire", "_ok"), OLD_TABLE)
    def test_a_wire_string_is_accepted_and_dumps_as_the_wire_string(
        self, name, wire, _ok
    ):
        result = AgentResult(answer="a", stop_reason=wire)
        assert result.stop_reason is StopReason[name]
        dumped = result.model_dump(mode="json")
        assert dumped["stop_reason"] == wire
        assert dumped["status"] == status_of_stop_reason(wire)[0].value
        assert json.loads(result.model_dump_json())["stop_reason"] == wire
        assert json.dumps({"s": result.stop_reason}) == json.dumps({"s": wire})

    def test_success_is_not_a_field_and_is_not_dumped(self):
        assert "success" not in AgentResult.model_fields
        assert "success" not in AgentResult(answer="a", status=Status.OK).model_dump()

    @pytest.mark.parametrize("flag", [True, False])
    def test_the_old_success_keyword_is_refused(self, flag):
        with pytest.raises(ValidationError):
            AgentResult(answer="a", success=flag, stop_reason=StopReason.ANSWERED)

    def test_neither_status_nor_stop_reason_raises_value_error(self):
        with pytest.raises(ValueError):
            AgentResult(answer="a")

    def test_an_unknown_stop_reason_is_refused(self):
        with pytest.raises(ValueError):
            AgentResult(answer="a", stop_reason="brand_new")

    def test_an_explicit_status_wins_over_the_derivation(self):
        result = AgentResult(
            answer="a", status=Status.OK, stop_reason=StopReason.NO_RESULT
        )
        assert result.success is True
        assert result.stop_reason is StopReason.NO_RESULT

    def test_a_status_alone_is_enough(self):
        failed = AgentResult(answer="a", status=Status.FAILED)
        assert failed.success is False and failed.stop_reason is None
        assert failed.reason is None
        assert AgentResult(answer="a", status="ok").success is True

    def test_an_explicit_reason_wins_over_the_derived_one(self):
        result = AgentResult(
            answer="a",
            stop_reason=StopReason.MAX_ITERATIONS,
            reason=Reason.CANCELLED,
        )
        assert result.status is Status.ABORTED
        assert result.reason is Reason.CANCELLED

    def test_the_budget_stop_carries_the_budget_reason(self):
        result = AgentResult(answer="a", stop_reason=StopReason.MAX_ITERATIONS)
        assert result.status is Status.ABORTED
        assert result.reason is Reason.BUDGET_SPENT

    def test_meta_builder_result_keeps_its_default_ok(self):
        assert MetaBuilderResult().success is True
        assert MetaBuilderResult(status=Status.FAILED).success is False
        with pytest.raises(ValidationError):
            MetaBuilderResult(success=False)


class TestRunResultProjection:
    def test_status_reason_and_the_detail_of_a_budget_stop(self):
        trace = AgentTrace(total_iterations=3)
        result = AgentResult(
            answer="partial", stop_reason=StopReason.MAX_ITERATIONS, trace=trace
        )
        run = result.run_result()
        assert isinstance(run, RunResult)
        status, reason = status_of_stop_reason("max_iterations")
        assert (run.status, run.reason) == (status, reason)
        assert run.answer == "partial"
        assert run.detail == {
            "stop_reason": "max_iterations",
            "iterations_used": 3,
            "tools_used": [],
        }
        assert type(run.detail["stop_reason"]) is str
        assert run.outputs == {}
        assert run.identity == Identity()
        assert run.handle is None

    def test_success_agrees_with_the_agent_result(self):
        for _, wire, old_success in OLD_TABLE:
            run = AgentResult(answer="a", stop_reason=wire).run_result()
            assert run.success is old_success

    def test_outputs_come_from_a_structured_output(self):
        class Out(BaseModel):
            city: str

        run = AgentResult(
            answer="a", stop_reason=StopReason.ANSWERED, structured_output=Out(city="X")
        ).run_result()
        assert run.outputs == {"city": "X"}
        mapping = AgentResult(
            answer="a", stop_reason=StopReason.ANSWERED, structured_output={"k": 1}
        ).run_result()
        assert mapping.outputs == {"k": 1}
        scalar = AgentResult(
            answer="a", stop_reason=StopReason.ANSWERED, structured_output=5
        ).run_result()
        assert scalar.outputs == {}

    def test_a_result_with_no_stop_reason_uses_its_status(self):
        run = AgentResult(answer="a", status=Status.FAILED).run_result()
        assert run.status is Status.FAILED
        assert run.detail["stop_reason"] is None

    def test_the_run_result_is_json_ready(self):
        run = AgentResult(answer="a", stop_reason=StopReason.ENDED).run_result()
        dumped = json.loads(run.model_dump_json())
        assert dumped["status"] == "aborted"
        assert dumped["detail"]["stop_reason"] == "ended"


class TestAPeerStopReasonStaysText:
    """A peer over A2A may send a stop_reason this version has never seen."""

    @staticmethod
    def _tool() -> RemoteAgentTool:
        return RemoteAgentTool(url="http://agent.test", name="peer", description="d")

    @pytest.mark.parametrize("stop", ["max_iterations", "brand_new_reason", None, 7])
    def test_the_peer_text_goes_to_detail_with_status_from_the_flag(self, stop):
        body = {"answer": "x", "success": False, "stop_reason": stop}
        run = self._tool()._run_result(body)
        assert run.status is Status.FAILED
        assert run.detail["stop_reason"] == stop
        assert run.answer == "x"

    def test_a_successful_peer_is_ok_and_a_missing_flag_counts_as_success(self):
        ok = self._tool()._run_result(
            {"answer": "x", "success": True, "stop_reason": "from_the_future"}
        )
        assert ok.status is Status.OK
        assert ok.detail["stop_reason"] == "from_the_future"
        assert self._tool()._run_result({"answer": "x"}).status is Status.OK

    def test_a_body_that_is_not_a_mapping_is_the_text_of_an_ok_run(self):
        run = self._tool()._run_result("plain")
        assert (run.status, run.answer) == (Status.OK, "plain")

    def test_an_unknown_stop_reason_does_not_stop_the_answer(self):
        tool = self._tool()
        body = {"answer": "x", "success": True, "stop_reason": "from_the_future"}
        assert tool._answer(body) == "x"


class TestTheServerKeepsTheWireString:
    """An A2A server answers with the plain wire string of a StopReason member."""

    @pytest.mark.parametrize("route", ["/invoke", "/stream"])
    def test_a_real_agent_result_goes_out_as_text(self, route):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from fsm_llm.agents.remote import AgentServer

        class _Agent:
            def run(self, task, initial_context=None):
                return AgentResult(answer="a", stop_reason=StopReason.MAX_ITERATIONS)

        client = TestClient(AgentServer(agent=_Agent()).app)
        resp = client.post(route, json={"task": "t"})
        assert resp.status_code == 200
        body = resp.text
        if route == "/stream":
            body = body.removeprefix("data: ").strip()
        data = json.loads(body)
        assert data["stop_reason"] == "max_iterations"
        assert data["success"] is False
