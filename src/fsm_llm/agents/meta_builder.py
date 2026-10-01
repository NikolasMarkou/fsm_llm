"""
MetaBuilderAgent: builds an FSM, workflow or agent artifact from a description.

The agent drives the meta-builder FSM (``build_meta_builder_fsm``) on a core
``API``: ``classify`` (the artifact type, a ``classification_extractions``
call), ``collect`` (the reply that gathers requirements, Pass 2), ``build``
(one structured completion that returns the whole artifact spec as JSON),
``build_failed`` and ``done``. Every model call is core's: this module makes
no LLM call of its own.

Python keeps what an FSM cannot express: the keyword hints the driver computes
from each message (``build_requested``, ``type_switch``, ``keyword_type``) and
writes before the turn, the handlers that write the build request and
assemble the returned spec deterministically on an ``ArtifactBuilder``
(``_assemble_fsm`` / ``_assemble_workflow`` / ``_assemble_agent``) and
validate it, the session lifecycle (``start``/``send``/``max_turns``) and the
canned texts (welcome, build complete, build failed, collect fallback, and
the build prompt that closes every collect reply when the model dropped it).

The tool registries in ``meta_tools.py`` (``create_fsm_tools`` and friends)
are a separate public API for driving a builder programmatically; this agent
does not use them.
"""

from __future__ import annotations

import json
from typing import Any, cast

from fsm_llm import API
from fsm_llm.definitions import LLMResponseError
from fsm_llm.logging import logger

from .base import _reject_misplaced_kwargs
from .constants import (
    META_BUILD_CALL_FAILED,
    META_BUILD_PROMPT,
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
from .handlers import make_fresh_keys_handler
from .meta_builders import (
    AgentBuilder,
    ArtifactBuilder,
    FSMBuilder,
    WorkflowBuilder,
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

# Words that make a message a request to change the artifact type: the turn
# routes back to ``classify`` (A-ISSUE-011: no classifier call otherwise).
_SWITCH_WORDS: tuple[str, ...] = (
    "instead",
    "actually",
    "change",
    "switch",
    "no,",
    "not a",
)

# Whole messages that ask for the build, and phrases that ask for it anywhere.
_BUILD_TRIGGERS: frozenset[str] = frozenset(
    {
        "build it",
        "build",
        "go",
        "build now",
        "create it",
        "make it",
        "generate",
        "done",
        "finish",
        "approve",
        "yes",
        "ok",
        "lgtm",
        "ship it",
        "do it",
    }
)
_BUILD_PHRASES: tuple[str, ...] = ("build it", "create it", "generate it")

# Variance a model's own build-prompt ending may carry and still count as the
# sentence: typographic quotes, closing punctuation, markdown emphasis, case
# and whitespace (seen in collect replies; none changes what the user reads).
_TYPOGRAPHIC_QUOTES = str.maketrans({"‘": "'", "’": "'"})
_TRAILING_DECORATION = " \t\r\n.!*_"


def _prompt_ending(text: str) -> str:
    """``text`` normalized for comparing its ending with the build prompt."""
    collapsed = " ".join(text.translate(_TYPOGRAPHIC_QUOTES).split())
    return collapsed.rstrip(_TRAILING_DECORATION).casefold()


_BUILD_PROMPT_ENDING = _prompt_ending(META_BUILD_PROMPT)


def _with_build_prompt(reply: str) -> str:
    """A collect reply that ends with ``META_BUILD_PROMPT`` exactly once.

    A reply already ending with it (up to the variance above) is returned
    unchanged; otherwise the sentence is appended as a new final line.
    """
    if _prompt_ending(reply).endswith(_BUILD_PROMPT_ENDING):
        return reply
    body = reply.rstrip()
    return f"{body}\n{META_BUILD_PROMPT}" if body else META_BUILD_PROMPT


class MetaBuilderAgent:
    """Meta-builder: classify the artifact type, collect, then build.

    A build is one structured completion (the ``build`` state of the meta
    FSM) whose JSON spec is assembled and validated in Python. In
    turn-by-turn mode, ``send()`` collects requirements until a build trigger
    ("build it"); a build with validation errors keeps the session open for
    another try. ``is_valid`` on a workflow or agent result means a
    structurally complete spec, not a loadable object.

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

        self._artifact_type: ArtifactType | None = None
        self._builder: ArtifactBuilder | None = None
        self._result: MetaBuilderResult | None = None
        # A schema echo found by the build handler; ``run`` raises it.
        self._build_error: MetaValidationError | None = None

        self._api: API | None = None
        self._conversation_id: str | None = None

        self._started = False
        self._complete = False
        self._messages: list[str] = []
        self._turn_count = 0

    # ------------------------------------------------------------------
    # Keyword hints (pure functions of one message)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_build_trigger(normalized: str) -> bool:
        """True when the (stripped, lower-cased) message asks for the build."""
        return normalized in _BUILD_TRIGGERS or any(
            phrase in normalized for phrase in _BUILD_PHRASES
        )

    @staticmethod
    def _is_type_switch(normalized: str) -> bool:
        """True when the (stripped, lower-cased) message may change the type."""
        return any(word in normalized for word in _SWITCH_WORDS)

    @classmethod
    def _detect_type_fallback(cls, text: str) -> ArtifactType:
        """The artifact type named by a keyword alias in ``text``, else FSM.

        Used when the type classification gives no type (provider failure or
        a low-confidence intent).
        """
        normalized = text.strip().lower()
        for alias, type_str in cls._build_type_aliases().items():
            if alias in normalized:
                return ArtifactType(type_str)
        return ArtifactType.FSM

    def _turn_hints(
        self, message: str, *, build_requested: bool | None = None
    ) -> dict[str, Any]:
        """Driver-written context for the next turn on ``message``.

        ``build_requested`` overrides the keyword trigger (``run`` always
        builds).
        """
        normalized = message.strip().lower()
        if build_requested is None:
            build_requested = self._is_build_trigger(normalized)
        return {
            _K.REQUIREMENTS: list(self._messages),
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
        api_kwargs: dict[str, Any] = {
            "timeout": self.meta_config.timeout_seconds,
            **self._api_kwargs,
        }
        api = API.from_definition(
            build_meta_builder_fsm(),
            model=self.meta_config.model,
            temperature=self.meta_config.temperature,
            max_tokens=self.meta_config.max_tokens,
            **api_kwargs,
        )
        # DECISION plan-2026-10-01T093600-944e2692/D-023: a type switch
        # re-enters `classify` with `artifact_type` still set. Do NOT clear
        # `artifact_type` on entry: core classifies an unset classification
        # field of a newly entered state in the same turn (post-transition,
        # 8a03483a/D-006), so a clear here plus the driver's `advance` (which
        # the collect reply needs) would classify twice per switch. The one
        # classification is the `advance`'s; a failed or low-confidence one
        # keeps the previous type. The old build's errors are cleared: they
        # belong to the old type. See decisions.md D-023.
        api.register_handler(
            api.create_handler(MetaHandlerNames.CLASSIFY_ENTRY)
            .on_state_entry(_S.CLASSIFY)
            .do(make_fresh_keys_handler([_K.VALIDATION_ERRORS]))
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
            .do(self._assemble_build_reply)
        )
        return api

    @staticmethod
    def _resolve_artifact_type(context: dict[str, Any]) -> dict[str, Any]:
        """``classify`` exit: keep the classified type, else the keyword type.

        The classification leaves ``artifact_type`` unset on a provider
        failure or a low-confidence intent; the driver's ``keyword_type``
        (FSM when no alias matched) fills it so every later state has a type.
        """
        valid = {t.value for t in ArtifactType}
        if context.get(_K.ARTIFACT_TYPE) in valid:
            return {}
        keyword = context.get(_K.KEYWORD_TYPE)
        fallback = keyword if keyword in valid else ArtifactType.FSM.value
        logger.debug(f"No classified artifact type; using '{fallback}'")
        return {_K.ARTIFACT_TYPE: fallback}

    @staticmethod
    def _write_build_request(context: dict[str, Any]) -> dict[str, Any]:
        """``build`` entry: the build request, and a cleared result.

        Clearing ``build_reply`` (and ``build_outcome``) on every entry makes
        a retry after a failed build call the model again: core skips a
        completion state whose result key is set.
        """
        artifact_type = ArtifactType(context[_K.ARTIFACT_TYPE])
        requirement = "\n".join(context.get(_K.REQUIREMENTS) or [])
        logger.info(
            MetaLogMessages.BUILD_STARTED.format(artifact_type=artifact_type.value)
        )
        return {
            _K.BUILD_REPLY: None,
            _K.BUILD_OUTCOME: None,
            _K.BUILD_MESSAGES: [
                {
                    "role": "user",
                    "content": build_artifact_prompt(artifact_type, requirement),
                }
            ],
            _K.BUILD_RESPONSE_FORMAT: build_response_format(artifact_type),
        }

    def _assemble_build_reply(self, context: dict[str, Any]) -> dict[str, Any]:
        """``build_reply`` committed: parse, assemble, validate, judge.

        Writes ``build_outcome`` (``valid`` iff the assembled artifact has no
        validation error), ``artifact``, ``validation_errors`` and, for a
        valid build, ``review_presentation``. A JSON-schema echo is kept on
        the agent (``run`` raises it) and reported as a validation error.
        """
        reply = context.get(_K.BUILD_REPLY)
        if not isinstance(reply, dict) or reply.get("kind") == META_BUILD_CALL_FAILED:
            # Cleared on entry, or the driver's record of a failed call
            # (``update_context`` runs CONTEXT_UPDATE handlers too).
            return {}
        artifact_type = ArtifactType(context[_K.ARTIFACT_TYPE])
        requirement = "\n".join(context.get(_K.REQUIREMENTS) or [])
        builder = self._builder_for(artifact_type)
        self._build_error = None

        spec = self._parse_extraction_response(reply.get("text") or "")
        if not spec:
            logger.warning(
                "Extraction returned empty spec — builder will be incomplete"
            )
        elif self._is_schema_echo(spec):
            # DECISION plan_2026-05-30_26c9510a/D-001 [STALE]: reject a JSON-schema
            # echo — small models sometimes return the type definition ({"type",
            # "properties","required"}) instead of a concrete artifact. Without
            # this guard, _assemble_fsm silently emits an empty stub ("Unnamed
            # FSM", states={}) and the build reports as nominally complete.
            self._build_error = MetaValidationError(
                f"LLM returned a JSON schema instead of a concrete "
                f"{artifact_type.value.upper()} (keys={list(spec.keys())}). "
                f"Expected an artifact with actual values, not a type definition."
            )
        else:
            logger.debug(f"Extracted spec keys: {list(spec.keys())}")
            if artifact_type == ArtifactType.FSM:
                self._assemble_fsm(spec, builder)
            elif artifact_type == ArtifactType.WORKFLOW:
                self._assemble_workflow(spec, builder)
            else:
                self._assemble_agent(spec, builder, requirement)

        errors = builder.validate_complete()
        if self._build_error is not None:
            errors = [str(self._build_error), *errors]
        valid = not errors
        return {
            _K.BUILD_OUTCOME: (
                MetaBuildOutcome.VALID if valid else MetaBuildOutcome.INVALID
            ),
            _K.ARTIFACT: builder.to_dict(),
            _K.VALIDATION_ERRORS: errors,
            _K.REVIEW_PRESENTATION: (
                build_review_presentation(builder, artifact_type) if valid else None
            ),
        }

    def _builder_for(self, artifact_type: ArtifactType) -> ArtifactBuilder:
        """The session's builder for ``artifact_type``.

        A retry of the same type reuses the builder (a set agent pattern and
        earlier valid fields survive); a different type starts a fresh one.
        """
        builder = self._builder
        if builder is None or self._artifact_type != artifact_type:
            builder = self._create_builder(artifact_type)
            self._builder = builder
            self._artifact_type = artifact_type
        return builder

    # ------------------------------------------------------------------
    # Deterministic assembly
    # ------------------------------------------------------------------

    def _assemble_fsm(self, spec: dict[str, Any], builder: ArtifactBuilder) -> None:
        """Deterministic FSM assembly from extracted spec."""
        # Dispatched only for ArtifactType.FSM, so the runtime type is FSMBuilder.
        builder = cast(FSMBuilder, builder)
        builder.set_overview(
            name=spec.get("name", "Unnamed FSM"),
            description=spec.get("description", ""),
            persona=spec.get("persona"),
        )

        states = spec.get("states", [])
        for i, state in enumerate(states):
            if not isinstance(state, dict):
                continue
            sid = state.get("state_id", f"state_{i}")
            try:
                builder.add_state(
                    state_id=sid,
                    description=cast(str, state.get("description", sid)),
                    purpose=cast(str, state.get("purpose", sid)),
                    extraction_instructions=state.get("extraction_instructions"),
                    response_instructions=state.get("response_instructions"),
                )
            except Exception as e:
                logger.warning(f"Failed to add state '{sid}': {e}")

        # Set initial state to first
        if states:
            first_id = states[0].get("state_id", "state_0")
            try:
                builder.set_initial_state(first_id)
            except Exception as e:
                logger.warning(f"set_initial_state failed for '{first_id}': {e}")

        # Add transitions
        for trans in spec.get("transitions", []):
            if not isinstance(trans, dict):
                continue
            try:
                builder.add_transition(
                    from_state=trans.get("from_state", ""),
                    target_state=trans.get("target_state", ""),
                    description=trans.get("description", ""),
                )
            except Exception as e:
                logger.warning(f"Failed to add transition: {e}")

    def _assemble_workflow(
        self, spec: dict[str, Any], builder: ArtifactBuilder
    ) -> None:
        """Deterministic workflow assembly from extracted spec."""
        # Dispatched only for ArtifactType.WORKFLOW → runtime type is WorkflowBuilder.
        builder = cast(WorkflowBuilder, builder)
        builder.set_overview(
            workflow_id=spec.get("workflow_id", "wf_default"),
            name=spec.get("name", "Unnamed Workflow"),
            description=spec.get("description", ""),
        )

        steps = spec.get("steps", [])
        step_ids: list[str] = []
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            sid = step.get("step_id", f"step_{i}")
            try:
                builder.add_step(
                    step_id=sid,
                    step_type=step.get("step_type", "auto_transition"),
                    name=cast(str, step.get("name", sid)),
                    description=step.get("description", ""),
                )
                step_ids.append(sid)
            except Exception as e:
                logger.warning(f"Failed to add step '{sid}': {e}")

        # Sequential transitions
        for i in range(len(step_ids) - 1):
            try:
                builder.set_step_transition(step_ids[i], step_ids[i + 1])
            except Exception as e:
                logger.warning(f"Failed to set transition: {e}")

        # Set initial step
        if step_ids:
            try:
                builder.set_initial_step(step_ids[0])
            except Exception as e:
                logger.warning(f"set_initial_step failed for '{step_ids[0]}': {e}")

    def _assemble_agent(
        self, spec: dict[str, Any], builder: ArtifactBuilder, requirement: str
    ) -> None:
        """Deterministic agent assembly from extracted spec.

        The agent pattern comes from the reply's ``agent_type`` (an enum of
        the build schema); when it is missing or unknown, the first pattern
        named in ``requirement`` (``plan_execute`` or "plan execute"), else
        the builder's current pattern is kept.
        """
        # Dispatched only for ArtifactType.AGENT → runtime type is AgentBuilder.
        builder = cast(AgentBuilder, builder)
        # Only set overview if name isn't already set (preserves on retry)
        name = spec.get("name", "")
        desc = spec.get("description", "")
        if name and (not builder.name or builder.name == "Unnamed Agent"):
            builder.set_overview(name=name, description=desc)
        elif desc and not builder.description:
            builder.set_overview(name=builder.name or "Unnamed Agent", description=desc)

        self._set_agent_type(builder, spec.get("agent_type"), requirement)

        # Add tools (skip duplicates by name)
        existing_names = {t.get("name") for t in builder.tools}
        for tool_spec in spec.get("tools", []):
            if not isinstance(tool_spec, dict):
                continue
            tool_name = tool_spec.get("name", "unnamed_tool")
            if tool_name in existing_names:
                continue
            try:
                builder.add_tool(
                    name=tool_name, description=tool_spec.get("description", "")
                )
                existing_names.add(tool_name)
            except Exception as e:
                logger.warning(f"Failed to add tool: {e}")

    @staticmethod
    def _set_agent_type(
        builder: AgentBuilder, agent_type: Any, requirement: str
    ) -> None:
        """Set the pattern from the reply, else by keyword from the requirement."""
        if isinstance(agent_type, str):
            try:
                builder.set_agent_type(agent_type)
                return
            except BuilderError as e:
                logger.warning(f"Build reply agent_type rejected: {e}")
        normalized = requirement.strip().lower()
        for pattern in sorted(AgentBuilder.VALID_AGENT_TYPES):
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
        accepted for the agent call shape and not used. The session is
        complete afterwards, valid or not.

        Raises:
            BuilderError: the build call failed (chained from core's
                ``LLMResponseError``).
            MetaValidationError: the model returned a JSON schema instead of
                an artifact.
        """
        logger.info(MetaLogMessages.META_STARTED.format(model=self.meta_config.model))
        self._messages = [task]
        api = self._create_api()
        conversation_id, _ = api.start_conversation(
            initial_context=self._turn_hints(task, build_requested=True)
        )
        try:
            # classify -> build (entry writes the request), then the build call.
            api.converse(task, conversation_id)
            self._advance_build(api, conversation_id)
            if self._build_error is not None:
                raise self._build_error
        finally:
            api.close()
        self._build_result()
        self._complete = True
        return cast(MetaBuilderResult, self._result)

    def _advance_build(self, api: API, conversation_id: str) -> None:
        """Run the ``build`` state's step (the build call and the assembly).

        Raises:
            BuilderError: the build call failed; the step was rolled back, so
                the conversation is still in ``build`` with no result.
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

    # ------------------------------------------------------------------
    # Turn-by-turn API (for monitor server + interactive)
    # ------------------------------------------------------------------

    def start(self, initial_message: str = "") -> str:
        """Initialize a builder session.

        With a message, the first turn classifies it and returns the
        generated collect reply; without one, the welcome text.
        """
        if self._started:
            raise MetaBuilderError(MetaErrorMessages.CONVERSATION_ALREADY_STARTED)
        self._started = True
        self._api = self._create_api()
        self._conversation_id, _ = self._api.start_conversation(
            initial_context=self._turn_hints("", build_requested=False)
        )
        if not initial_message:
            return build_welcome_message()
        self._messages.append(initial_message)
        return self._turn(initial_message)

    def send(self, message: str) -> str:
        """Send a message in a turn-by-turn session."""
        if not self._started:
            raise MetaBuilderError(MetaErrorMessages.CONVERSATION_NOT_STARTED)
        if self._complete:
            raise MetaBuilderError("Session has already completed")

        self._turn_count += 1
        if self._turn_count > self.meta_config.max_turns:
            raise MetaBuilderError(
                f"Maximum turns ({self.meta_config.max_turns}) exceeded"
            )

        self._messages.append(message)
        return self._turn(message)

    def _session(self) -> tuple[API, str]:
        """The started session's API and conversation id."""
        if self._api is None or self._conversation_id is None:
            raise MetaBuilderError(MetaErrorMessages.CONVERSATION_NOT_STARTED)
        return self._api, self._conversation_id

    def _turn(self, message: str) -> str:
        """One user turn: ``converse``, then one ``advance`` when it is due.

        The turn ends in ``classify`` after a type switch (the message-free
        step classifies ``latest_request`` and the collect reply follows) or
        in ``build`` (the step makes the build call).
        """
        api, conversation_id = self._session()
        api.update_context(conversation_id, self._turn_hints(message))
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
        finally:
            self._sync_artifact_type(api, conversation_id)
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
            self._record_failed_build_call(api, conversation_id, e)
        self._sync_artifact_type(api, conversation_id)
        self._build_result()
        data = api.get_data(conversation_id)
        if api.get_current_state(conversation_id) == _S.DONE:
            self._complete = True
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
    def _record_failed_build_call(
        api: API, conversation_id: str, error: BuilderError
    ) -> None:
        """Route a ``build`` state whose call failed to ``build_failed``.

        The failed step left the conversation in ``build`` with no result.
        Recording the failure as the result (``kind`` ``call_failed``) makes
        the next step a no-call step that takes the ``build_failed`` edge, so
        the session continues like any failed build.
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-023: the outage is
        # recorded as the build state's result so the FSM's own edge takes the
        # session to `build_failed`. Do NOT leave the conversation in `build`
        # (the next `converse` there would make a build call whatever the user
        # said) and do NOT jump states with `set_conversation_state` (a
        # transition outside the FSM's rules). See decisions.md D-023.
        message = f"The build call failed: {error}"
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

    def _sync_artifact_type(self, api: API, conversation_id: str) -> None:
        """Mirror the conversation's artifact type on the agent."""
        value = api.get_data(conversation_id).get(_K.ARTIFACT_TYPE)
        if value in {t.value for t in ArtifactType}:
            self._artifact_type = ArtifactType(value)

    def _collect_fallback(self, message: str) -> str:
        """The canned collect reply used when the reply call failed."""
        artifact_type = self._artifact_type or self._detect_type_fallback(message)
        self._artifact_type = artifact_type
        return (
            f"Got it — added to your {artifact_type.value.upper()} spec. "
            f"Say 'build it' when ready, or keep adding details."
        )

    def is_complete(self) -> bool:
        """True once a build produced a valid artifact (or ``run`` finished)."""
        return self._complete

    def get_result(self) -> MetaBuilderResult:
        """The build result; raises ``MetaBuilderError`` before completion."""
        if not self._complete:
            raise MetaBuilderError(
                "Build is not complete. Say 'build it' to trigger the build."
            )
        if self._result is None:
            self._build_result()
        return cast(MetaBuilderResult, self._result)

    def get_internal_state(self) -> dict[str, Any]:
        """Session state for the monitor: phase, turns, type, builder progress."""
        result: dict[str, Any] = {
            "phase": "complete" if self._complete else "collecting",
            "turn_count": self._turn_count,
            "is_complete": self._complete,
            "started": self._started,
            "message_count": len(self._messages),
        }
        if self._artifact_type is not None:
            result["artifact_type"] = self._artifact_type.value

        builder = self._builder
        if builder is not None:
            progress = builder.get_progress()
            validation_errors = builder.validate_complete()
            result["builder_progress"] = {
                "percentage": progress.percentage,
                "completed": progress.completed,
                "total_required": progress.total_required,
                "missing": builder.get_missing_fields(),
                "warnings": progress.warnings,
            }
            result["builder_summary"] = builder.get_summary(detail_level="standard")
            result["artifact_preview"] = builder.to_dict()
            result["validation_errors"] = validation_errors
            result["is_valid"] = len(validation_errors) == 0
        else:
            result["builder_progress"] = None
            result["builder_summary"] = None
            result["artifact_preview"] = None
            result["validation_errors"] = []
            result["is_valid"] = False

        return result

    def run_interactive(self) -> MetaBuilderResult:
        """Run a session on stdin/stdout until the build completes or EOF."""
        response = self.start()
        print(f"\n{response}\n")

        while not self.is_complete():
            try:
                user_input = input("> ")
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended by user.")
                break
            if not user_input.strip():
                continue
            response = self.send(user_input)
            print(f"\n{response}\n")

        if self.is_complete():
            return self.get_result()
        self._build_result()
        return cast(MetaBuilderResult, self._result)

    # ------------------------------------------------------------------
    # Builder creation + result
    # ------------------------------------------------------------------

    def _create_builder(
        self, artifact_type: ArtifactType
    ) -> FSMBuilder | WorkflowBuilder | AgentBuilder:
        if artifact_type == ArtifactType.FSM:
            return FSMBuilder()
        if artifact_type == ArtifactType.WORKFLOW:
            return WorkflowBuilder()
        if artifact_type == ArtifactType.AGENT:
            return AgentBuilder()
        raise MetaBuilderError(f"Unknown artifact type: {artifact_type}")

    def _build_result(self) -> None:
        artifact_type = self._artifact_type or ArtifactType.FSM
        builder = self._builder

        if builder is None:
            self._result = MetaBuilderResult(
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
            return

        errors = builder.validate_complete()
        artifact = builder.to_dict()
        artifact_json = format_artifact_json(artifact)

        self._result = MetaBuilderResult(
            answer=artifact_json,
            success=len(errors) == 0,
            artifact_type=artifact_type,
            artifact=artifact,
            artifact_json=artifact_json,
            is_valid=len(errors) == 0,
            validation_errors=errors,
            conversation_turns=self._turn_count,
            final_context={
                "artifact_json": artifact,
                "artifact_type": artifact_type.value,
                "is_valid": len(errors) == 0,
                "validation_errors": errors,
            },
        )
