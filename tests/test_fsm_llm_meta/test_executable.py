"""The meta-builder as a built-in executable (plan ace7a8c8 step 23.1, roadmap E2.4, D-061).

``meta_builder`` is a code-backed, single-shot, ``conversational`` executable: ``run(task)`` is one
type classification and one build call. A valid artifact is ``ok`` with the artifact in
``outputs``; an invalid one (``MetaBuilderResult.is_valid`` false) is ``failed`` with the
validation errors in ``detail``. The scripted double is the one of this suite's conftest.
"""

from __future__ import annotations

import json

import pytest

from fsm_llm.agents.exceptions import BuilderError
from fsm_llm.core.codes import Status
from fsm_llm.core.definitions import LLMResponseError
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import ExecutableRegistry

from .conftest import ScriptedMetaLLM
from .test_meta_conversation import _FSM_SPEC, _spec

TASK = "a support chatbot, build it"


def _meta():
    registry = ExecutableRegistry()
    registry.load_sources()
    return registry.get("meta_builder")


def test_a_valid_artifact_is_ok_with_the_artifact_in_outputs() -> None:
    llm = ScriptedMetaLLM(intents=["fsm"], builds=[json.dumps(_FSM_SPEC)])
    exe = _meta()
    result = exe.run({"task": TASK}, context=RunContext(llm_interface=llm))
    assert result.status is Status.OK and result.success
    assert result.outputs["artifact_type"] == "fsm"
    assert result.outputs["artifact"]["name"] == "SupportBot"
    assert (
        result.detail["is_valid"] is True and result.detail["validation_errors"] == []
    )
    assert result.detail["pattern"] == "meta_builder"
    assert result.identity.template_identity == exe.template_identity
    assert result.identity.instance_identity is None
    assert [r.call_type for r in llm.requests] == ["classification", "completion"]


def test_an_invalid_artifact_is_failed_with_the_errors_in_detail() -> None:
    llm = ScriptedMetaLLM(
        intents=["fsm"], builds=[_spec(_FSM_SPEC, states=[], transitions=[])]
    )
    result = _meta().run({"task": TASK}, context=RunContext(llm_interface=llm))
    assert result.status is Status.FAILED and not result.success
    assert result.detail["is_valid"] is False
    assert "At least one state is required" in result.detail["validation_errors"]


def test_a_provider_outage_propagates_as_the_builders_error() -> None:
    llm = ScriptedMetaLLM(intents=["fsm"], builds=[LLMResponseError("provider down")])
    with pytest.raises(BuilderError, match="provider down"):
        _meta().run({"task": TASK}, context=RunContext(llm_interface=llm))


def test_max_seconds_is_the_builders_timeout_and_max_steps_is_refused() -> None:
    llm = ScriptedMetaLLM(intents=["fsm"], builds=[json.dumps(_FSM_SPEC)])
    with pytest.raises(ValueError, match="max_iterations"):
        _meta().run({"task": TASK}, context=RunContext(llm_interface=llm, max_steps=3))
    assert llm.requests == []
    result = _meta().run(
        {"task": TASK}, context=RunContext(llm_interface=llm, max_seconds=60.0)
    )
    assert result.success
