"""
MetaBuilderAgent: builds an FSM, workflow or agent artifact from a description.

The agent drives the meta-builder FSM (``build_meta_builder_fsm``) on a core
``API``: ``classify`` (the artifact type, a ``classification_extractions``
call), ``collect`` (the reply that gathers requirements, Pass 2), ``build``
(one structured completion that returns the whole artifact spec as JSON),
``build_failed`` and ``done``. Every model call is core's: this module makes
no LLM call of its own.

The conversation is the one source of session state: the FSM state says
whether the build is done, and the context holds the artifact type, the
requirements, and every build output (the assembled artifact, its validation
errors, progress and summary). Python keeps what an FSM cannot express: the
keyword hints the driver computes from each message (``build_requested``,
``type_switch``, ``keyword_type``) and writes before the turn, the handlers
that write the build request and assemble each build reply on a fresh
``ArtifactBuilder``, the send counter behind ``max_turns``, and the canned
texts (welcome, build complete, build failed, collect fallback, and the build
prompt that closes every collect reply when the model dropped it).

The tool registries in ``meta_tools.py`` (``create_fsm_tools`` and friends)
are a separate public API for driving a builder programmatically; this agent
does not use them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from itertools import pairwise
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

from fsm_llm import API
from fsm_llm.api import llm_settings_for
from fsm_llm.definitions import LLMResponseError
from fsm_llm.handlers import HandlerExecutionError, clear_keys_delta
from fsm_llm.logging import logger

from .base import _reject_misplaced_kwargs
from .constants import (
    META_BUILD_CALL_FAILED,
    META_BUILD_NEGATION_FILLERS,
    META_BUILD_NEGATIONS,
    META_BUILD_OUTPUT_KEYS,
    META_BUILD_PHRASES,
    META_BUILD_PROMPT,
    META_BUILD_TRIGGERS,
    META_SWITCH_WORDS,
    MetaBuilderStates,
    MetaBuildOutcome,
    MetaContextKeys,
    MetaErrorMessages,
    MetaHandlerNames,
    MetaLogMessages,
)
from .definitions import (
    ArtifactType,
    MetaBuilderConfig,
    MetaBuilderResult,
)
from .exceptions import BuilderError, MetaBuilderError, MetaValidationError
from .fsm_definitions import build_meta_builder_fsm
from .meta_builders import (
    AgentArtifactBuilder,
    ArtifactBuilder,
    FSMArtifactBuilder,
    WorkflowArtifactBuilder,
)
from .meta_output import format_artifact_json
from .meta_prompts import (
    build_artifact_prompt,
    build_response_format,
    build_review_presentation,
    build_welcome_message,
)

__all__ = ["MetaBuilderAgent", "MetaBuilderConfig"]

_S = MetaBuilderStates
_K = MetaContextKeys
_TYPE_VALUES = frozenset(t.value for t in ArtifactType)

# ---------------------------------------------------------------------------
# Keyword hints: word-boundary matching over a normalized message
# ---------------------------------------------------------------------------

_APOSTROPHES = str.maketrans({"‘": "'", "’": "'"})


def _normalize_message(text: str) -> str:
    """``text`` lower-cased, whitespace-collapsed, typographic apostrophes as '."""
    return " ".join(text.translate(_APOSTROPHES).split()).lower()


def _words_pattern(words: Iterable[str], *, plural: bool = False) -> re.Pattern[str]:
    """A pattern matching any of ``words`` as whole words (longest first)."""
    alternatives = "|".join(
        re.escape(w) for w in sorted(set(words), key=len, reverse=True)
    )
    suffix = "s?" if plural else ""
    return re.compile(rf"(?<!\w)(?:{alternatives}){suffix}(?!\w)")


_SWITCH_PATTERN = _words_pattern(META_SWITCH_WORDS)
_BUILD_PHRASE_PATTERN = _words_pattern(META_BUILD_PHRASES)
# A negation that governs what follows it: only filler words between it and
# the end of the text before the phrase; "why not" is not a negation.
_GOVERNING_NEGATION = re.compile(
    rf"(?<!\bwhy ){_words_pattern(META_BUILD_NEGATIONS).pattern}"
    rf"(?: {_words_pattern(META_BUILD_NEGATION_FILLERS).pattern})* $"
)
_CLAUSE_BREAK = re.compile(r"[.,;:!?\n]+")


# ---------------------------------------------------------------------------
# The build prompt that closes a collect reply (D-024, D-025)
# ---------------------------------------------------------------------------

# Variance a model's own copy of the sentence may carry and still count as
# it: any quote style, markdown emphasis, "you are", case and whitespace.
_QUOTES = str.maketrans(
    {"‘": "'", "’": "'", '"': "'", "“": "'", "”": "'", "*": None, "_": None}
)


def _prompt_text(text: str) -> str:
    """``text`` normalized for finding the build prompt in it."""
    collapsed = " ".join(text.translate(_QUOTES).split()).casefold()
    return collapsed.replace("you are", "you're")


_BUILD_PROMPT_CORE = _prompt_text(META_BUILD_PROMPT).rstrip(".!")


def _with_build_prompt(reply: str) -> str:
    """A collect reply that carries ``META_BUILD_PROMPT`` exactly once.

    A reply already containing the sentence anywhere (up to the variance
    above, so a paragraph after it does not count as a missing sentence) is
    returned unchanged; otherwise the sentence is appended as a new final
    line.
    """
    if _BUILD_PROMPT_CORE in _prompt_text(reply):
        return reply
    body = reply.rstrip()
    return f"{body}\n{META_BUILD_PROMPT}" if body else META_BUILD_PROMPT


# ---------------------------------------------------------------------------
# The build reply: untrusted model output, parsed into typed specs
# ---------------------------------------------------------------------------


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


class MetaBuilderAgent:
    """Meta-builder: classify the artifact type, collect, then build.

    A build is one structured completion (the ``build`` state of the meta
    FSM) whose JSON spec is assembled on a fresh builder and validated in
    Python. In turn-by-turn mode, ``send()`` collects requirements until a
    build trigger ("build it"); ``start()`` never builds. A build with
    validation errors keeps the session open for another try. ``is_valid``
    on a workflow or agent result means a structurally complete spec, not a
    loadable object.

    ``api_kwargs`` go to the core ``API`` (``llm_interface=`` injects the
    interface every model call goes through; other names are LLM call
    kwargs). ``model``, ``temperature`` and ``max_tokens`` belong on
    ``MetaBuilderConfig`` and raise ``TypeError`` here.

    Usage (single-shot)::

        agent = MetaBuilderAgent()
        result = agent.run("Build a customer support chatbot with 3 states")
        print(result.artifact_json)

    Usage (turn-by-turn)::

        agent = MetaBuilderAgent()
        response = agent.start()
        while not agent.is_complete():
            response = agent.send(input("> "))
        result = agent.get_result()
    """

    @classmethod
    def _build_type_aliases(cls) -> dict[str, str]:
        """Return type alias map sorted longest-first for correct matching."""
        raw: dict[str, str] = {
            "finite state machine": "fsm",
            "fsm": "fsm",
            "state machine": "fsm",
            "state_machine": "fsm",
            "conversational": "fsm",
            "conversation": "fsm",
            "help desk": "fsm",
            "onboarding": "fsm",
            "interview": "fsm",
            "chat bot": "fsm",
            "helpdesk": "fsm",
            "chatbot": "fsm",
            "dialogue": "fsm",
            "dialog": "fsm",
            "survey": "fsm",
            "quiz": "fsm",
            "faq": "fsm",
            "bot": "fsm",
            "data pipeline": "workflow",
            "processing": "workflow",
            "workflow": "workflow",
            "automation": "workflow",
            "pipeline": "workflow",
            "sequence": "workflow",
            "process": "workflow",
            "steps": "workflow",
            "batch": "workflow",
            "flow": "workflow",
            "etl": "workflow",
            "agentic": "agent",
            "research": "agent",
            "navigate": "agent",
            "browse": "agent",
            "search": "agent",
            "react": "agent",
            "agent": "agent",
            "tools": "agent",
            "tool": "agent",
        }
        return dict(sorted(raw.items(), key=lambda kv: -len(kv[0])))

    def __init__(
        self,
        config: MetaBuilderConfig | None = None,
        **api_kwargs: Any,
    ) -> None:
        _reject_misplaced_kwargs(type(self).__name__, api_kwargs)
        if config is None:
            config = MetaBuilderConfig()
        self.meta_config = config
        self._api_kwargs = api_kwargs

        # The turn-by-turn session (``start``) and the result of ``run``.
        self._api: API | None = None
        self._conversation_id: str | None = None
        self._result: MetaBuilderResult | None = None
        # Sends so far, the ``max_turns`` budget (driver bookkeeping).
        self._turn_count = 0

    # ------------------------------------------------------------------
    # Keyword hints (pure functions of one message)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_build_trigger(normalized: str) -> bool:
        """True when the normalized message asks for the build.

        A whole-message trigger ("ok", "build it"), or a build phrase that no
        negation governs ("don't build it yet" and "do not ever build it" are
        not one; "it's not perfect but build it" and "why not build it" are).
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-044: a negation blocks a
        # build phrase only when it governs it (right before it, filler words
        # allowed). Do NOT go back to "any negation earlier in the clause":
        # "don't forget to build it", "not a problem build it" and "it's not
        # perfect but build it" stopped building (they built at e1f63a9).
        if normalized.strip() in META_BUILD_TRIGGERS:
            return True
        for clause in _CLAUSE_BREAK.split(normalized):
            for match in _BUILD_PHRASE_PATTERN.finditer(clause):
                if not _GOVERNING_NEGATION.search(clause[: match.start()]):
                    return True
        return False

    @staticmethod
    def _is_type_switch(normalized: str) -> bool:
        """True when the normalized message may change the type."""
        return _SWITCH_PATTERN.search(normalized) is not None

    @classmethod
    def _detect_type_fallback(cls, text: str) -> ArtifactType:
        """The artifact type named by a keyword alias in ``text``, else FSM.

        Aliases match whole words (an optional plural ``s``). Used when the
        type classification gives no type (provider failure or no intent).
        """
        normalized = _normalize_message(text)
        for alias, type_str in cls._build_type_aliases().items():
            if _words_pattern([alias], plural=True).search(normalized):
                return ArtifactType(type_str)
        return ArtifactType.FSM

    def _turn_hints(
        self, message: str, requirements: list[str], *, build_requested: bool | None
    ) -> dict[str, Any]:
        """Driver-written context for the next turn on ``message``.

        ``build_requested`` overrides the keyword trigger (``run`` always
        builds, ``start`` never does); ``None`` reads it from the message.
        """
        normalized = _normalize_message(message)
        if build_requested is None:
            build_requested = self._is_build_trigger(normalized)
        return {
            _K.REQUIREMENTS: requirements,
            _K.LATEST_REQUEST: message,
            _K.BUILD_REQUESTED: build_requested,
            _K.TYPE_SWITCH: self._is_type_switch(normalized),
            _K.KEYWORD_TYPE: self._detect_type_fallback(message).value,
        }

    # ------------------------------------------------------------------
    # Core API and handlers
    # ------------------------------------------------------------------

    def _create_api(self) -> API:
        """The core ``API`` over the meta FSM, with the build handlers."""
        # An injected interface owns its timeout: core refuses connection
        # settings beside it (D-029), so the config timeout applies only to
        # the interface core builds.
        api_kwargs: dict[str, Any] = (
            dict(self._api_kwargs)
            if self._api_kwargs.get("llm_interface") is not None
            else {"timeout": self.meta_config.timeout_seconds, **self._api_kwargs}
        )
        api = API.from_definition(
            build_meta_builder_fsm(),
            **llm_settings_for(
                api_kwargs,
                model=self.meta_config.model,
                temperature=self.meta_config.temperature,
                max_tokens=self.meta_config.max_tokens,
            ),
            **api_kwargs,
        )
        api.register_handler(
            api.create_handler(MetaHandlerNames.CLASSIFY_ENTRY)
            .on_state_entry(_S.CLASSIFY)
            .do(self._enter_classify)
        )
        api.register_handler(
            api.create_handler(MetaHandlerNames.CLASSIFY_EXIT)
            .on_state_exit(_S.CLASSIFY)
            .do(self._resolve_artifact_type)
        )
        api.register_handler(
            api.create_handler(MetaHandlerNames.BUILD_ENTRY)
            .on_state_entry(_S.BUILD)
            .do(self._write_build_request)
        )
        api.register_handler(
            api.create_handler(MetaHandlerNames.BUILD_REPLY)
            .on_state(_S.BUILD)
            .on_context_update(_K.BUILD_REPLY)
            .critical()
            .do(self._judge_build_reply)
        )
        return api

    @staticmethod
    def _enter_classify(context: dict[str, Any]) -> dict[str, Any]:
        """``classify`` entry (a type switch): stash the type, drop the build.

        The old build's outputs belong to the old type. ``artifact_type``
        itself stays set (D-023: a clear would classify twice per switch);
        its value is stashed so the exit can restore it.
        """
        return {
            **clear_keys_delta(META_BUILD_OUTPUT_KEYS, context),
            _K.PREVIOUS_ARTIFACT_TYPE: context.get(_K.ARTIFACT_TYPE),
        }

    @staticmethod
    def _resolve_artifact_type(context: dict[str, Any]) -> dict[str, Any]:
        """``classify`` exit: the classified type, else the previous, else keyword.

        A classified ``fsm``/``workflow``/``agent`` wins. Otherwise (the
        ``unknown`` fallback, an unrecognised reply, a discarded
        low-confidence intent, or a provider failure) the type stashed on
        entry is kept; with none (the first classification) the driver's
        ``keyword_type`` is used, FSM when no alias matched.
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-023: a type switch
        # re-enters `classify` with `artifact_type` still set. Do NOT clear
        # `artifact_type` on entry: core classifies an unset classification
        # field of a newly entered state in the same turn (post-transition,
        # 8a03483a/D-006), so a clear plus the driver's `advance` (which the
        # collect reply needs) would classify twice per switch. The one
        # classification is the `advance`'s. A switch classification that
        # falls back (`unknown`, D-035), is below the threshold, or fails
        # keeps the previous type (stashed on entry). See decisions.md D-023.
        classified = context.get(_K.ARTIFACT_TYPE)
        if classified in _TYPE_VALUES:
            resolved = classified
        else:
            previous = context.get(_K.PREVIOUS_ARTIFACT_TYPE)
            keyword = context.get(_K.KEYWORD_TYPE)
            resolved = (
                previous
                if previous in _TYPE_VALUES
                else keyword
                if keyword in _TYPE_VALUES
                else ArtifactType.FSM.value
            )
            logger.debug(f"No classified artifact type; using '{resolved}'")
        return {_K.ARTIFACT_TYPE: resolved, _K.PREVIOUS_ARTIFACT_TYPE: None}

    @staticmethod
    def _write_build_request(context: dict[str, Any]) -> dict[str, Any]:
        """``build`` entry: the build request, and every old output cleared.

        Clearing ``build_reply`` on every entry makes a retry after a failed
        build call the model again: core skips a completion state whose
        result key is set.
        """
        artifact_type = ArtifactType(context[_K.ARTIFACT_TYPE])
        requirement = "\n".join(context.get(_K.REQUIREMENTS) or [])
        logger.info(
            MetaLogMessages.BUILD_STARTED.format(artifact_type=artifact_type.value)
        )
        return {
            **clear_keys_delta(META_BUILD_OUTPUT_KEYS, context),
            _K.BUILD_MESSAGES: [
                {
                    "role": "user",
                    "content": build_artifact_prompt(artifact_type, requirement),
                }
            ],
            _K.BUILD_RESPONSE_FORMAT: build_response_format(artifact_type),
        }

    @classmethod
    def _judge_build_reply(cls, context: dict[str, Any]) -> dict[str, Any]:
        """``build_reply`` committed: parse, assemble on a fresh builder, judge.

        Writes ``build_outcome``, ``artifact``, ``validation_errors``,
        ``build_progress``, ``build_summary`` and, for a valid build,
        ``review_presentation``. The reply is untrusted: a JSON-schema echo
        or a wrong-typed field is a ``malformed`` outcome whose errors name
        the problem; nothing in the reply can make this handler raise.
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-035: every reply is
        # assembled on a fresh builder and every output goes into context.
        # Do NOT keep a builder, an artifact type or a completion flag on the
        # agent: such a mirror sits outside core's rollback and drifted from
        # the FSM (a failed build's states made every retry fail as orphans;
        # `done` with `is_complete()` False). Do NOT let model output raise
        # here: a raise left the session in `build`. See decisions.md D-035.
        reply = context.get(_K.BUILD_REPLY)
        if not isinstance(reply, dict) or reply.get("kind") == META_BUILD_CALL_FAILED:
            # Cleared on entry, or the driver's record of a failed call
            # (``update_context`` runs CONTEXT_UPDATE handlers too).
            return {}
        artifact_type = ArtifactType(context[_K.ARTIFACT_TYPE])
        requirement = "\n".join(context.get(_K.REQUIREMENTS) or [])
        text = reply.get("text")
        builder = cls._create_builder(artifact_type)
        outcome, errors = cls._assemble_reply(
            builder, artifact_type, text if isinstance(text, str) else "", requirement
        )
        progress = builder.get_progress()
        valid = outcome == MetaBuildOutcome.VALID
        return {
            _K.BUILD_OUTCOME: outcome,
            _K.ARTIFACT: builder.to_dict(),
            _K.VALIDATION_ERRORS: errors,
            _K.REVIEW_PRESENTATION: (
                build_review_presentation(builder, artifact_type) if valid else None
            ),
            _K.BUILD_PROGRESS: {
                "percentage": progress.percentage,
                "completed": progress.completed,
                "total_required": progress.total_required,
                "missing": builder.get_missing_fields(),
                "warnings": progress.warnings,
            },
            _K.BUILD_SUMMARY: builder.get_summary(detail_level="standard"),
        }

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
        raw = cls._parse_extraction_response(text)
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

    # ------------------------------------------------------------------
    # Deterministic assembly (typed spec -> builder)
    # ------------------------------------------------------------------

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
        for pattern in sorted(AgentArtifactBuilder.VALID_AGENT_TYPES):
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
    def _parse_extraction_response(response: str) -> dict[str, Any]:
        """Parse JSON from the extraction LLM response."""
        text = response.strip()
        if not text:
            return {}

        # Direct parse
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Find JSON in response
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse extraction response: {text[:200]}")
        return {}

    # ------------------------------------------------------------------
    # Single-shot API
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> MetaBuilderResult:
        """Build an artifact from ``task`` in one go.

        One type classification and one build call. ``initial_context`` is
        accepted for the agent call shape and not used. The run is complete
        afterwards, valid or not.

        Raises:
            BuilderError: the build call failed (chained from core's
                ``LLMResponseError``), or its reply could not be assembled.
            MetaValidationError: the model returned no artifact of the
                schema: a JSON schema, or a field of the wrong type
                (``errors`` lists each problem).
        """
        logger.info(MetaLogMessages.META_STARTED.format(model=self.meta_config.model))
        api = self._create_api()
        conversation_id, _ = api.start_conversation(
            initial_context=self._turn_hints(task, [task], build_requested=True)
        )
        try:
            # classify -> build (entry writes the request), then the build call.
            api.converse(task, conversation_id)
            self._advance_build(api, conversation_id)
            data = api.get_data(conversation_id)
        finally:
            api.close()
        if data.get(_K.BUILD_OUTCOME) == MetaBuildOutcome.MALFORMED:
            errors = list(data.get(_K.VALIDATION_ERRORS) or [])
            raise MetaValidationError("; ".join(errors), errors=errors)
        self._result = self._result_from(data)
        return self._result

    @staticmethod
    def _advance_build(api: API, conversation_id: str) -> None:
        """Run the ``build`` state's step (the build call and the assembly).

        Raises:
            BuilderError: the build call failed or the assembly handler
                raised; the step was rolled back, so the conversation is
                still in ``build`` with no result.
        """
        try:
            api.advance(conversation_id)
        except LLMResponseError as e:
            # DECISION plan-2026-07-20T040150-876e7164/D-006 [STALE]: a provider
            # outage is an error, never an empty answer. Do NOT turn it into
            # `""` or an empty spec: `run()` would then emit a stub artifact
            # as if the build had merely underperformed. `""` keeps its one
            # meaning: the model answered with nothing (an invalid build).
            raise BuilderError(f"Meta-builder LLM call failed: {e!s}") from e
        except HandlerExecutionError as e:
            raise BuilderError(
                f"Meta-builder could not assemble the build reply: {e!s}"
            ) from e

    # ------------------------------------------------------------------
    # Turn-by-turn API (for monitor server + interactive)
    # ------------------------------------------------------------------

    def start(self, initial_message: str = "") -> str:
        """Initialize a builder session; never builds.

        With a message, the first turn classifies it and returns the
        generated collect reply, whatever the message says ("build it"
        included); without one, the welcome text.
        """
        if self._api is not None:
            raise MetaBuilderError(MetaErrorMessages.CONVERSATION_ALREADY_STARTED)
        self._api = self._create_api()
        self._conversation_id, _ = self._api.start_conversation(
            initial_context=self._turn_hints("", [], build_requested=False)
        )
        if not initial_message:
            return build_welcome_message()
        return self._turn(initial_message, build_requested=False)

    def send(self, message: str) -> str:
        """Send a message in a turn-by-turn session."""
        self._session()
        if self.is_complete():
            raise MetaBuilderError("Session has already completed")

        self._turn_count += 1
        if self._turn_count > self.meta_config.max_turns:
            raise MetaBuilderError(
                f"Maximum turns ({self.meta_config.max_turns}) exceeded"
            )
        return self._turn(message, build_requested=None)

    def _session(self) -> tuple[API, str]:
        """The started session's API and conversation id."""
        if self._api is None or self._conversation_id is None:
            raise MetaBuilderError(MetaErrorMessages.CONVERSATION_NOT_STARTED)
        return self._api, self._conversation_id

    def _session_data(self) -> dict[str, Any]:
        """The session's public context, ``{}`` without a session."""
        if self._api is None or self._conversation_id is None:
            return {}
        data: dict[str, Any] = self._api.get_data(self._conversation_id)
        return data

    def _turn(self, message: str, *, build_requested: bool | None) -> str:
        """One user turn: ``converse``, then one ``advance`` when it is due.

        The turn ends in ``classify`` after a type switch (the message-free
        step classifies ``latest_request`` and the collect reply follows) or
        in ``build`` (the step makes the build call).
        """
        api, conversation_id = self._session()
        requirements = [*(self._session_data().get(_K.REQUIREMENTS) or []), message]
        api.update_context(
            conversation_id,
            self._turn_hints(message, requirements, build_requested=build_requested),
        )
        try:
            reply = api.converse(message, conversation_id)
            state = api.get_current_state(conversation_id)
            if state == _S.CLASSIFY:
                step = api.advance(conversation_id)
                reply = step.response or ""
                state = step.state_after
        except LLMResponseError as e:
            # The turn was rolled back; the requirement is kept in context.
            logger.warning(f"Collect reply failed, using the canned reply: {e}")
            return self._collect_fallback(message)
        if state == _S.COLLECT:
            # DECISION plan-2026-10-01T093600-944e2692/D-025: the build prompt
            # is appended to the reply the user gets, not to core's stored
            # reply. Do NOT rewrite `Conversation.exchanges` from here (no
            # public core API does it; reaching into the instance would be a
            # driver-side history editor) and do NOT add more prompt wording
            # instead (the 4b model drops the sentence regardless, D-024).
            # History keeps what the model wrote. See decisions.md D-025.
            return _with_build_prompt(reply)
        if state != _S.BUILD:
            return reply
        return self._build_turn(api, conversation_id)

    def _build_turn(self, api: API, conversation_id: str) -> str:
        """The build step of a ``send``; the reply reports its outcome."""
        try:
            self._advance_build(api, conversation_id)
        except BuilderError as e:
            logger.error(f"Build execution failed: {e}")
            self._record_failed_build(api, conversation_id, e)
        data = api.get_data(conversation_id)
        if self.is_complete():
            presentation = data.get(_K.REVIEW_PRESENTATION) or ""
            return (
                f"Build complete!\n\n{presentation}\n\n"
                f"The artifact JSON has been generated."
            )

        # Build produced validation errors — keep session open.
        errors = data.get(_K.VALIDATION_ERRORS) or ["Build failed"]
        error_list = "\n".join(f"  - {e}" for e in errors)
        return (
            f"I couldn't complete the build yet:\n{error_list}\n\n"
            f"Please provide the missing information, then say 'build it' again."
        )

    @staticmethod
    def _record_failed_build(
        api: API, conversation_id: str, error: BuilderError
    ) -> None:
        """Route a ``build`` state whose step failed to ``build_failed``.

        The failed step left the conversation in ``build`` with no result.
        Recording the failure as the result (``kind`` ``call_failed``) makes
        the next step a no-call step that takes the ``build_failed`` edge, so
        the session continues like any failed build.
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-023: the failure is
        # recorded as the build state's result so the FSM's own edge takes the
        # session to `build_failed`. Do NOT leave the conversation in `build`
        # (the next `converse` there would make a build call whatever the user
        # said) and do NOT jump states with `set_conversation_state` (a
        # transition outside the FSM's rules). See decisions.md D-023.
        cause = error.__cause__
        message = (
            f"The build call failed: {error}"
            if isinstance(cause, LLMResponseError)
            else f"The build reply could not be assembled: {error}"
        )
        api.update_context(
            conversation_id,
            {
                _K.BUILD_REPLY: {
                    "kind": META_BUILD_CALL_FAILED,
                    "text": message,
                    "calls": [],
                },
                _K.BUILD_OUTCOME: MetaBuildOutcome.INVALID,
                _K.VALIDATION_ERRORS: [message],
            },
        )
        api.advance(conversation_id)

    def _collect_fallback(self, message: str) -> str:
        """The canned collect reply used when the reply call failed."""
        value = self._session_data().get(_K.ARTIFACT_TYPE)
        artifact_type = (
            ArtifactType(value)
            if value in _TYPE_VALUES
            else self._detect_type_fallback(message)
        )
        return (
            f"Got it: added to your {artifact_type.value.upper()} spec. "
            f"You can keep adding details. {META_BUILD_PROMPT}"
        )

    def is_complete(self) -> bool:
        """True once ``run`` finished or the session's build reached ``done``."""
        if self._result is not None:
            return True
        if self._api is None or self._conversation_id is None:
            return False
        state: str = self._api.get_current_state(self._conversation_id)
        return state == _S.DONE

    def get_result(self) -> MetaBuilderResult:
        """The build result; raises ``MetaBuilderError`` before completion."""
        if not self.is_complete():
            raise MetaBuilderError(
                "Build is not complete. Say 'build it' to trigger the build."
            )
        if self._result is not None:
            return self._result
        return self._result_from(self._session_data())

    def get_internal_state(self) -> dict[str, Any]:
        """Session state for the monitor: phase, turns, type, build progress.

        Read from the session's context (after ``run``, only the phase and
        completion are reported).
        """
        data = self._session_data()
        complete = self.is_complete()
        result: dict[str, Any] = {
            "phase": "complete" if complete else "collecting",
            "turn_count": self._turn_count,
            "is_complete": complete,
            "started": self._api is not None,
            "message_count": len(data.get(_K.REQUIREMENTS) or []),
        }
        if data.get(_K.ARTIFACT_TYPE) in _TYPE_VALUES:
            result["artifact_type"] = data[_K.ARTIFACT_TYPE]
        artifact = data.get(_K.ARTIFACT)
        built = isinstance(artifact, dict)
        result["builder_progress"] = data.get(_K.BUILD_PROGRESS) if built else None
        result["builder_summary"] = data.get(_K.BUILD_SUMMARY) if built else None
        result["artifact_preview"] = artifact if built else None
        result["validation_errors"] = (
            list(data.get(_K.VALIDATION_ERRORS) or []) if built else []
        )
        result["is_valid"] = data.get(_K.BUILD_OUTCOME) == MetaBuildOutcome.VALID
        return result

    def run_interactive(self) -> MetaBuilderResult:
        """Run a session on stdin/stdout until the build completes or EOF.

        Ctrl-C (``KeyboardInterrupt``) propagates to the caller, at the
        prompt as during a model call; EOF ends the session normally.
        """
        response = self.start()
        print(f"\n{response}\n")

        while not self.is_complete():
            try:
                user_input = input("> ")
            except EOFError:
                print("\nSession ended by user.")
                break
            if not user_input.strip():
                continue
            response = self.send(user_input)
            print(f"\n{response}\n")

        if self.is_complete():
            return self.get_result()
        return self._result_from(self._session_data())

    # ------------------------------------------------------------------
    # Builder creation + result
    # ------------------------------------------------------------------

    @staticmethod
    def _create_builder(
        artifact_type: ArtifactType,
    ) -> FSMArtifactBuilder | WorkflowArtifactBuilder | AgentArtifactBuilder:
        if artifact_type == ArtifactType.FSM:
            return FSMArtifactBuilder()
        if artifact_type == ArtifactType.WORKFLOW:
            return WorkflowArtifactBuilder()
        if artifact_type == ArtifactType.AGENT:
            return AgentArtifactBuilder()
        raise MetaBuilderError(f"Unknown artifact type: {artifact_type}")

    def _result_from(self, data: dict[str, Any]) -> MetaBuilderResult:
        """The result of the last build recorded in context ``data``."""
        value = data.get(_K.ARTIFACT_TYPE)
        artifact_type = (
            ArtifactType(value) if value in _TYPE_VALUES else ArtifactType.FSM
        )
        artifact = data.get(_K.ARTIFACT)
        if not isinstance(artifact, dict):
            return MetaBuilderResult(
                answer="Build was not completed",
                success=False,
                artifact_type=artifact_type,
                artifact={},
                artifact_json="{}",
                is_valid=False,
                validation_errors=["Builder was not initialized"],
                conversation_turns=self._turn_count,
                final_context={},
            )
        errors = list(data.get(_K.VALIDATION_ERRORS) or [])
        valid = data.get(_K.BUILD_OUTCOME) == MetaBuildOutcome.VALID
        artifact_json = format_artifact_json(artifact)
        return MetaBuilderResult(
            answer=artifact_json,
            success=valid,
            artifact_type=artifact_type,
            artifact=artifact,
            artifact_json=artifact_json,
            is_valid=valid,
            validation_errors=errors,
            conversation_turns=self._turn_count,
            final_context={
                "artifact_json": artifact,
                "artifact_type": artifact_type.value,
                "is_valid": valid,
                "validation_errors": errors,
            },
        )
