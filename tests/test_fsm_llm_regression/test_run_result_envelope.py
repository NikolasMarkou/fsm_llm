"""Every public run-result type yields the one ``RunResult`` envelope (roadmap 0.16).

Plan ace7a8c8 step 11, decision D-047. A new public run type adds a row to
``ENVELOPE_ROWS``; ``WorkflowStepResult``, ``CompileResult`` and the other
``*Result`` types are step or phase records, not runs, and are not listed.
"""

from __future__ import annotations

import copy

import pytest

from fsm_llm.agents import AgentResult
from fsm_llm.agents.constants import StopReason
from fsm_llm.agents.definitions import MetaBuilderResult
from fsm_llm.constants import has_internal_prefix
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.results import Identity, RunResult
from fsm_llm.synthesis import synthesis_run_result
from fsm_llm.synthesis.constants import (
    REASONS,
    STATUS_ABORTED,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_RUNNING,
)
from fsm_llm.workflows.models import WorkflowInstance, WorkflowStatus

DIGEST = "a" * 64


def _instance(status: WorkflowStatus, **kw) -> WorkflowInstance:
    return WorkflowInstance(
        instance_id="inst-1",
        workflow_id="wf-1",
        current_step_id="step-a",
        status=status,
        **kw,
    )


ENVELOPE_ROWS = [
    ("AgentResult", lambda: AgentResult(answer="a", stop_reason=StopReason.ANSWERED)),
    ("MetaBuilderResult", lambda: MetaBuilderResult(artifact_type="fsm")),
    ("WorkflowInstance", lambda: _instance(WorkflowStatus.COMPLETED).run_result()),
    (
        "synthesis_result dict",
        lambda: synthesis_run_result(
            {"status": "ok", "procedure_id": DIGEST, "version": 1, "reason": None}
        ),
    ),
]


class TestEveryRunTypeYieldsTheEnvelope:
    @pytest.mark.parametrize(("name", "build"), ENVELOPE_ROWS)
    def test_the_projection_is_a_run_result(self, name, build):
        produced = build()
        result = produced.run_result() if hasattr(produced, "run_result") else produced
        assert isinstance(result, RunResult), name
        assert isinstance(result.identity, Identity), name
        assert result.success == (result.status is Status.OK), name


WORKFLOW_TABLE = [
    (WorkflowStatus.PENDING, Status.RUNNING, None),
    (WorkflowStatus.RUNNING, Status.RUNNING, None),
    (WorkflowStatus.WAITING, Status.RUNNING, None),
    (WorkflowStatus.COMPLETED, Status.OK, None),
    (WorkflowStatus.FAILED, Status.FAILED, None),
    (WorkflowStatus.CANCELLED, Status.ABORTED, Reason.CANCELLED),
]


class TestWorkflowInstanceRunResult:
    def test_the_table_covers_every_workflow_status(self):
        assert {row[0] for row in WORKFLOW_TABLE} == set(WorkflowStatus)

    @pytest.mark.parametrize(("workflow_status", "status", "reason"), WORKFLOW_TABLE)
    def test_status_and_reason(self, workflow_status, status, reason):
        result = _instance(workflow_status).run_result()
        assert result.status is status
        assert result.reason is reason
        assert result.success is (status is Status.OK)

    def test_the_handle_is_the_instance_id(self):
        assert _instance(WorkflowStatus.RUNNING).run_result().handle == "inst-1"

    def test_waiting_stays_distinguishable_in_detail(self):
        detail = _instance(WorkflowStatus.WAITING).run_result().detail
        assert detail["workflow_status"] == "waiting"
        assert detail["workflow_id"] == "wf-1"
        assert detail["current_step_id"] == "step-a"

    def test_outputs_are_the_context_minus_internal_prefixed_keys(self):
        context = {
            "answer": 42,
            "nested": {"_kept": 1},
            "_private": 1,
            "System_x": 2,
            "internal_y": 3,
            "__dunder": 4,
        }
        result = _instance(WorkflowStatus.COMPLETED, context=context).run_result()
        assert result.outputs == {"answer": 42, "nested": {"_kept": 1}}
        assert not any(has_internal_prefix(k) for k in result.outputs)

    def test_outputs_are_a_copy_of_the_context(self):
        instance = _instance(WorkflowStatus.COMPLETED, context={"a": 1})
        before = copy.deepcopy(instance.context)
        result = instance.run_result()
        result.outputs["b"] = 2
        assert instance.context == before

    def test_the_error_text_lives_in_detail_only(self):
        result = _instance(WorkflowStatus.FAILED, error="step blew up").run_result()
        assert result.detail["error"] == "step blew up"
        assert result.reason is None
        assert result.answer is None

    def test_no_error_means_no_error_key(self):
        assert "error" not in _instance(WorkflowStatus.COMPLETED).run_result().detail

    @pytest.mark.parametrize("text", ["cancelled by user", "timeout", "budget_spent"])
    def test_error_text_is_never_branched_on(self, text):
        result = _instance(WorkflowStatus.FAILED, error=text).run_result()
        assert result.status is Status.FAILED
        assert result.reason is None

    def test_the_identity_is_empty(self):
        assert _instance(WorkflowStatus.RUNNING).run_result().identity == Identity()

    def test_the_instance_serializes_as_before(self):
        dump = _instance(WorkflowStatus.COMPLETED).model_dump()
        assert "run_result" not in dump
        assert "success" not in dump


class TestSynthesisProjection:
    def _dict(self, status="ok", reason=None, procedure_id=DIGEST, version=3):
        return {
            "status": status,
            "procedure_id": procedure_id,
            "version": version,
            "reason": reason,
        }

    @pytest.mark.parametrize(
        ("wire", "status"),
        [
            (STATUS_OK, Status.OK),
            (STATUS_FAILED, Status.FAILED),
            (STATUS_ABORTED, Status.ABORTED),
            (STATUS_RUNNING, Status.RUNNING),
        ],
    )
    def test_every_status_string(self, wire, status):
        assert synthesis_run_result(self._dict(wire)).status is status

    @pytest.mark.parametrize("code", REASONS)
    def test_every_reason_code(self, code):
        result = synthesis_run_result(self._dict("failed", code))
        assert result.reason is Reason(code)
        assert result.success is False

    def test_the_run_that_has_not_picked_a_procedure_has_no_identity(self):
        result = synthesis_run_result(self._dict("running", None, None, None))
        assert result.identity == Identity()
        assert result.status is Status.RUNNING

    def test_the_procedure_digest_is_the_instance_identity(self):
        result = synthesis_run_result(self._dict())
        assert result.identity.instance_identity == DIGEST
        assert result.identity.template_identity is None
        assert result.detail == {"version": 3}
        assert result.outputs == {}
        assert result.handle is None

    def test_the_dict_is_read_never_changed(self):
        wire = self._dict("failed", "compile_failed")
        before = copy.deepcopy(wire)
        synthesis_run_result(wire)
        assert wire == before

    def test_an_unknown_status_is_refused(self):
        with pytest.raises(ValueError, match="status"):
            synthesis_run_result(self._dict("done"))

    def test_a_missing_status_is_refused(self):
        with pytest.raises(ValueError, match="status"):
            synthesis_run_result({"procedure_id": DIGEST})

    def test_an_unknown_reason_is_refused(self):
        with pytest.raises(ValueError, match="reason"):
            synthesis_run_result(self._dict("failed", "because"))

    def test_a_non_mapping_is_refused(self):
        with pytest.raises(TypeError):
            synthesis_run_result("ok")  # type: ignore[arg-type]

    def test_the_function_lives_in_neither_runner_nor_library(self):
        assert synthesis_run_result.__module__ == "fsm_llm.synthesis.results"
