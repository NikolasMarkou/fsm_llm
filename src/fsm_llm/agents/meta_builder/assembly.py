"""Assembly of one build reply on a fresh builder (plan d454fd62 step 90; the old ``agents/meta_builder.py``, same dotted path).

``_assemble_reply`` parses the untrusted reply into a typed spec (``specs.py``) and routes it to
``_assemble_fsm``, ``_assemble_workflow`` or ``_assemble_agent``; ``_create_builder`` makes the builder of an
artifact type. ``AssemblyMixin`` holds them.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise
from typing import Any

from pydantic import ValidationError

from fsm_llm.agents.constants import MetaBuildOutcome
from fsm_llm.agents.definitions import ArtifactType
from fsm_llm.agents.exceptions import BuilderError, MetaBuilderError
from fsm_llm.agents.meta_builders import (
    AgentArtifactBuilder,
    ArtifactBuilder,
    FSMArtifactBuilder,
    WorkflowArtifactBuilder,
)
from fsm_llm.llm.json_extraction import extract_json_from_text
from fsm_llm.logging import logger

from .specs import (
    _SPEC_MODELS,
    _AgentSpec,
    _FSMSpec,
    _given,
    _shape_errors,
    _WorkflowSpec,
)
from .surface import _MetaBuilderSurface


class AssemblyMixin(_MetaBuilderSurface):
    """Deterministic assembly of a build reply (typed spec to builder), and the builder factory."""

    @classmethod
    def _assemble_reply(
        cls,
        builder: ArtifactBuilder,
        artifact_type: ArtifactType,
        text: str,
        requirement: str,
    ) -> tuple[str, list[str]]:
        """Assemble one build reply on ``builder``; ``(outcome, errors)``.

        ``malformed`` (nothing assembled) for a JSON-schema echo or a
        wrong-typed field; else ``valid``/``invalid`` from the builder's own
        validation. An empty or unparseable reply assembles nothing and is
        ``invalid`` (the model answered with no artifact).
        """
        raw = extract_json_from_text(text) or {}
        if not raw:
            logger.warning("Build reply has no JSON spec; the build is incomplete")
        elif cls._is_schema_echo(raw):
            # DECISION plan_2026-05-30_26c9510a/D-001 [STALE]: reject a JSON-schema
            # echo — small models sometimes return the type definition ({"type",
            # "properties","required"}) instead of a concrete artifact. Without
            # this guard, _assemble_fsm silently emits an empty stub ("Unnamed
            # FSM", states={}) and the build reports as nominally complete.
            return MetaBuildOutcome.MALFORMED, [
                f"LLM returned a JSON schema instead of a concrete "
                f"{artifact_type.value.upper()} (keys={list(raw.keys())}). "
                f"Expected an artifact with actual values, not a type definition."
            ]
        try:
            spec = _SPEC_MODELS[artifact_type].model_validate(raw)
        except ValidationError as e:
            return MetaBuildOutcome.MALFORMED, _shape_errors(e)
        if isinstance(spec, _FSMSpec) and isinstance(builder, FSMArtifactBuilder):
            cls._assemble_fsm(spec, builder)
        elif isinstance(spec, _WorkflowSpec) and isinstance(
            builder, WorkflowArtifactBuilder
        ):
            cls._assemble_workflow(spec, builder)
        elif isinstance(spec, _AgentSpec) and isinstance(builder, AgentArtifactBuilder):
            cls._assemble_agent(spec, builder, requirement)
        errors = builder.validate_complete()
        return (MetaBuildOutcome.INVALID if errors else MetaBuildOutcome.VALID), errors

    @staticmethod
    def _assemble_fsm(spec: _FSMSpec, builder: FSMArtifactBuilder) -> None:
        """FSM assembly: overview, states in order (the first is initial), edges."""
        builder.set_overview(
            name=spec.name, description=spec.description, persona=spec.persona
        )
        state_ids: list[str] = []
        for i, state in enumerate(spec.states):
            sid = _given(state, "state_id", f"state_{i}")
            state_ids.append(sid)
            try:
                builder.add_state(
                    state_id=sid,
                    description=_given(state, "description", sid),
                    purpose=_given(state, "purpose", sid),
                    extraction_instructions=state.extraction_instructions,
                    response_instructions=state.response_instructions,
                )
            except BuilderError as e:
                logger.warning(f"Failed to add state '{sid}': {e}")
        if state_ids:
            try:
                builder.set_initial_state(state_ids[0])
            except BuilderError as e:
                logger.warning(f"set_initial_state failed for '{state_ids[0]}': {e}")
        for trans in spec.transitions:
            try:
                builder.add_transition(
                    from_state=trans.from_state,
                    target_state=trans.target_state,
                    description=trans.description,
                )
            except BuilderError as e:
                logger.warning(f"Failed to add transition: {e}")

    @staticmethod
    def _assemble_workflow(
        spec: _WorkflowSpec, builder: WorkflowArtifactBuilder
    ) -> None:
        """Workflow assembly: steps chained in reply order, the first initial."""
        builder.set_overview(
            workflow_id=spec.workflow_id, name=spec.name, description=spec.description
        )
        step_ids: list[str] = []
        for i, step in enumerate(spec.steps):
            sid = _given(step, "step_id", f"step_{i}")
            try:
                builder.add_step(
                    step_id=sid,
                    step_type=step.step_type,
                    name=_given(step, "name", sid),
                    description=step.description,
                )
                step_ids.append(sid)
            except BuilderError as e:
                logger.warning(f"Failed to add step '{sid}': {e}")
        for current, following in pairwise(step_ids):
            try:
                builder.set_step_transition(current, following)
            except BuilderError as e:
                logger.warning(f"Failed to set transition: {e}")
        if step_ids:
            try:
                builder.set_initial_step(step_ids[0])
            except BuilderError as e:
                logger.warning(f"set_initial_step failed for '{step_ids[0]}': {e}")

    @classmethod
    def _assemble_agent(
        cls, spec: _AgentSpec, builder: AgentArtifactBuilder, requirement: str
    ) -> None:
        """Agent assembly: overview, pattern, tools (first of each name kept)."""
        if spec.name or spec.description:
            builder.set_overview(
                name=spec.name or "Unnamed Agent", description=spec.description
            )
        cls._set_agent_type(builder, spec.agent_type, requirement)
        seen: set[str] = set()
        for tool_spec in spec.tools:
            if tool_spec.name in seen:
                continue
            try:
                builder.add_tool(name=tool_spec.name, description=tool_spec.description)
                seen.add(tool_spec.name)
            except BuilderError as e:
                logger.warning(f"Failed to add tool: {e}")

    @staticmethod
    def _set_agent_type(
        builder: AgentArtifactBuilder, agent_type: str | None, requirement: str
    ) -> None:
        """Set the pattern from the reply, else the first one named in ``requirement``.

        ``agent_type`` is an enum of the build schema; a missing or unknown
        one falls back to a pattern the requirement names (``plan_execute``
        or "plan execute"); with none, the builder keeps no pattern.
        """
        if agent_type is not None:
            try:
                builder.set_agent_type(agent_type)
                return
            except BuilderError as e:
                logger.warning(f"Build reply agent_type rejected: {e}")
        normalized = requirement.strip().lower()
        for pattern in sorted(builder.valid_agent_types):
            if pattern in normalized or pattern.replace("_", " ") in normalized:
                builder.set_agent_type(pattern)
                return

    @staticmethod
    def _is_schema_echo(spec: dict[str, Any]) -> bool:
        """True if the LLM echoed a JSON schema instead of a concrete spec.

        A real artifact spec carries concrete content (``states`` / ``steps``
        / ``tools``). A schema echo is dominated by JSON-schema markers
        (``type`` / ``properties`` / ``required``) with no such content.
        """
        keys = set(spec.keys())
        has_content = bool(spec.get("states") or spec.get("steps") or spec.get("tools"))
        if has_content:
            return False
        return "properties" in keys or "$schema" in keys or {"type", "required"} <= keys

    @staticmethod
    def _create_builder(
        artifact_type: ArtifactType, agent_types: Iterable[str] = ()
    ) -> FSMArtifactBuilder | WorkflowArtifactBuilder | AgentArtifactBuilder:
        if artifact_type == ArtifactType.FSM:
            return FSMArtifactBuilder()
        if artifact_type == ArtifactType.WORKFLOW:
            return WorkflowArtifactBuilder()
        if artifact_type == ArtifactType.AGENT:
            return AgentArtifactBuilder(agent_types)
        raise MetaBuilderError(f"Unknown artifact type: {artifact_type}")
