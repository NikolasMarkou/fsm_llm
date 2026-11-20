"""The C13 result envelope of core (plan ace7a8c8, roadmap 0.16): ``RunResult`` and ``Identity``.

These rows pin the type only. The adoption by agents, workflows and synthesis has its own
rows (``test_run_result_envelope.py``); no producer exists in this module's scope.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fsm_llm.core import results
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.results import Identity, RunResult

_PATH = Path(results.__file__)


class TestIdentity:
    def test_both_digests_are_optional(self):
        identity = Identity()
        assert identity.template_identity is None
        assert identity.instance_identity is None

    def test_the_fields_are_kept(self):
        identity = Identity(template_identity="t" * 64, instance_identity="i" * 64)
        assert identity.template_identity == "t" * 64
        assert identity.instance_identity == "i" * 64

    def test_frozen(self):
        identity = Identity(template_identity="t")
        with pytest.raises(ValidationError):
            identity.template_identity = "other"  # type: ignore[misc]

    def test_equal_by_value(self):
        assert Identity(template_identity="t") == Identity(template_identity="t")
        assert Identity(template_identity="t") != Identity(instance_identity="t")

    def test_an_unknown_keyword_raises(self):
        with pytest.raises(ValidationError):
            Identity(template="t")  # type: ignore[call-arg]

    def test_dump_is_the_two_fields(self):
        assert Identity(template_identity="t").model_dump() == {
            "template_identity": "t",
            "instance_identity": None,
        }


class TestRunResult:
    @pytest.mark.parametrize("status", list(Status))
    def test_success_is_a_property_of_status(self, status):
        assert RunResult(status=status).success is (status is Status.OK)

    def test_success_is_not_a_constructor_argument(self):
        with pytest.raises(ValidationError):
            RunResult(status=Status.OK, success=True)  # type: ignore[call-arg]

    def test_success_is_not_dumped(self):
        assert "success" not in RunResult(status=Status.OK).model_dump()

    def test_success_cannot_be_assigned(self):
        with pytest.raises((ValidationError, AttributeError)):
            RunResult(status=Status.OK).success = False  # type: ignore[misc]

    def test_defaults(self):
        result = RunResult(status=Status.FAILED)
        assert result.reason is None
        assert result.outputs == {}
        assert result.answer is None
        assert result.detail == {}
        assert result.identity == Identity()
        assert result.identity.template_identity is None
        assert result.handle is None

    def test_default_dicts_are_not_shared(self):
        first, second = RunResult(status=Status.OK), RunResult(status=Status.OK)
        first.outputs["k"] = 1
        first.detail["k"] = 1
        assert second.outputs == {} and second.detail == {}

    def test_status_is_required(self):
        with pytest.raises(ValidationError):
            RunResult()  # type: ignore[call-arg]

    def test_strings_coerce_to_the_members(self):
        result = RunResult(status="aborted", reason="cancelled")
        assert result.status is Status.ABORTED
        assert result.reason is Reason.CANCELLED

    @pytest.mark.parametrize(
        ("field", "value"),
        [("status", "bogus"), ("reason", "bogus"), ("status", None), ("status", 3)],
    )
    def test_a_value_outside_the_vocabulary_raises(self, field, value):
        data = {"status": "ok", field: value}
        with pytest.raises(ValidationError):
            RunResult(**data)

    def test_frozen(self):
        result = RunResult(status=Status.OK, answer="a")
        with pytest.raises(ValidationError):
            result.answer = "b"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            result.status = Status.FAILED  # type: ignore[misc]

    def test_an_unknown_keyword_raises(self):
        with pytest.raises(ValidationError):
            RunResult(status=Status.OK, steps=[])  # type: ignore[call-arg]

    def test_the_whole_envelope_round_trips_as_json(self):
        result = RunResult(
            status=Status.ABORTED,
            reason=Reason.BUDGET_SPENT,
            outputs={"x": [1, {"y": None}]},
            answer="partial",
            detail={"iterations_used": 3},
            identity=Identity(template_identity="t", instance_identity="i"),
            handle="conv-1",
        )
        wire = json.loads(json.dumps(result.model_dump(mode="json")))
        assert wire == {
            "status": "aborted",
            "reason": "budget_spent",
            "outputs": {"x": [1, {"y": None}]},
            "answer": "partial",
            "detail": {"iterations_used": 3},
            "identity": {"template_identity": "t", "instance_identity": "i"},
            "handle": "conv-1",
        }
        assert RunResult.model_validate(wire) == result

    def test_a_cancelled_run_is_aborted_with_the_cancelled_reason(self):
        result = RunResult(status=Status.ABORTED, reason=Reason.CANCELLED)
        assert not result.success
        assert result.reason == "cancelled"


class TestSource:
    """``core/results.py`` sits in core: it imports the kernel, the codes and pydantic only."""

    def test_imports_stay_in_core_and_the_standard_library(self):
        tree = ast.parse(_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    assert module in ("codes", "definitions", ""), module
                else:
                    assert not module.startswith("fsm_llm") or module.startswith(
                        "fsm_llm.core"
                    ), module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("fsm_llm"), alias.name

    def test_all_is_static_and_complete(self):
        assert sorted(results.__all__) == ["Identity", "RunResult"]
