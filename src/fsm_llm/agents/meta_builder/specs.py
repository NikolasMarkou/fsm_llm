"""The build reply as typed specs: untrusted model output parsed into pydantic shapes (plan d454fd62 step 90; the old ``agents/meta_builder.py``, same dotted path).

Unknown keys are ignored and nothing is coerced (``StrictStr``), so a wrong-typed field is a
``ValidationError`` that ``_shape_errors`` turns into one validation error per field; ``_given`` reads a
field only when the reply set it. ``_SPEC_MODELS`` maps an artifact type to its spec model. Read by the
assembly mixin (``assembly.py``) only.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

from fsm_llm.agents.definitions import ArtifactType


class _ReplySpec(BaseModel):
    """Base of the build-reply shapes: unknown keys ignored, no coercion."""

    model_config = ConfigDict(extra="ignore")


class _FSMStateSpec(_ReplySpec):
    state_id: StrictStr = ""
    description: StrictStr = ""
    purpose: StrictStr = ""
    extraction_instructions: StrictStr | None = None
    response_instructions: StrictStr | None = None


class _FSMTransitionSpec(_ReplySpec):
    from_state: StrictStr = ""
    target_state: StrictStr = ""
    description: StrictStr = ""


class _FSMSpec(_ReplySpec):
    name: StrictStr = "Unnamed FSM"
    description: StrictStr = ""
    persona: StrictStr | None = None
    states: list[_FSMStateSpec] = Field(default_factory=list)
    transitions: list[_FSMTransitionSpec] = Field(default_factory=list)


class _WorkflowStepSpec(_ReplySpec):
    step_id: StrictStr = ""
    step_type: StrictStr = "auto_transition"
    name: StrictStr = ""
    description: StrictStr = ""


class _WorkflowSpec(_ReplySpec):
    workflow_id: StrictStr = "wf_default"
    name: StrictStr = "Unnamed Workflow"
    description: StrictStr = ""
    steps: list[_WorkflowStepSpec] = Field(default_factory=list)


class _ToolSpec(_ReplySpec):
    name: StrictStr = "unnamed_tool"
    description: StrictStr = ""


class _AgentSpec(_ReplySpec):
    name: StrictStr = ""
    description: StrictStr = ""
    agent_type: StrictStr | None = None
    tools: list[_ToolSpec] = Field(default_factory=list)


_SPEC_MODELS: dict[ArtifactType, type[_ReplySpec]] = {
    ArtifactType.FSM: _FSMSpec,
    ArtifactType.WORKFLOW: _WorkflowSpec,
    ArtifactType.AGENT: _AgentSpec,
}


def _given(spec: BaseModel, field: str, default: Any) -> Any:
    """``spec.<field>`` when the reply set it, else ``default``."""
    return getattr(spec, field) if field in spec.model_fields_set else default


def _shape_errors(error: ValidationError) -> list[str]:
    """One validation error per wrong-typed field of a build reply."""
    errors: list[str] = []
    for item in error.errors():
        where = ".".join(str(part) for part in item["loc"])
        message = (
            "Input should be an object"
            if item["type"] in {"model_type", "dict_type"}
            else item["msg"]
        )
        errors.append(f"Build reply field '{where}': {message}")
    return errors
