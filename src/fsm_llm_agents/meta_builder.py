from __future__ import annotations

"""
MetaBuilderAgent -- interactively builds FSMs, Workflows, and Agents.

Uses a classification-driven FSM for routing decisions (artifact type,
review approval, revision scope) and an inner ReactAgent with builder
tools for the autonomous build phase.

Supports two interfaces:
  - ``run(task)`` for single-shot programmatic use (like other agents)
  - ``start()/send()/is_complete()/get_result()`` for turn-by-turn
    interactive use (CLI, monitor endpoints)
"""

import json
import time
from typing import Any, ClassVar

import litellm

from fsm_llm import API
from fsm_llm.logging import logger
from fsm_llm.utilities import extract_json_from_text

from .base import BaseAgent
from .constants import (
    DecisionWords,
    MetaBuilderStates,
    MetaDefaults,
    MetaErrorMessages,
    MetaLogMessages,
)
from .definitions import (
    AgentResult,
    ArtifactType,
    MetaBuilderConfig,
    MetaBuilderResult,
)
from .exceptions import MetaBuilderError
from .meta_builders import AgentBuilder, ArtifactBuilder, FSMBuilder, WorkflowBuilder
from .meta_output import format_artifact_json
from .meta_prompts import (
    BUILD_SPEC_SYSTEM_PROMPT,
    INTAKE_SYSTEM_PROMPT,
    build_followup_message,
    build_intake_user_message,
    build_output_message,
    build_review_presentation,
    build_revision_spec_prompt,
    build_spec_prompt,
    build_welcome_message,
)


class MetaBuilderAgent(BaseAgent):
    """Meta-agent that interactively builds FSMs, Workflows, and Agents.

    Uses classification_extractions on the FSM states for routing and an
    inner ReactAgent with builder tools for the autonomous build phase.

    Usage (turn-by-turn)::

        agent = MetaBuilderAgent()
        response = agent.start()
        print(response)

        while not agent.is_complete():
            user_input = input("> ")
            response = agent.send(user_input)
            print(response)

        result = agent.get_result()
        print(result.artifact_json)

    Usage (single-shot)::

        agent = MetaBuilderAgent()
        result = agent.run("Build a customer support chatbot with 3 states")
        print(result.artifact_json)

    Usage (interactive CLI)::

        agent = MetaBuilderAgent()
        result = agent.run_interactive()
        print(result.artifact_json)
    """

    def __init__(
        self,
        config: MetaBuilderConfig | None = None,
        **api_kwargs: Any,
    ) -> None:
        # Convert MetaBuilderConfig to AgentConfig for BaseAgent
        if config is None:
            config = MetaBuilderConfig()
        super().__init__(config=config, **api_kwargs)
        self.meta_config: MetaBuilderConfig = config

        # Internal state for turn-by-turn interface
        self._started: bool = False
        self._phase: str = MetaBuilderStates.INTAKE
        self._turn_count: int = 0
        self._artifact_type: ArtifactType | None = None
        self._builder: ArtifactBuilder | None = None
        self._requirements: dict[str, Any] = {}
        self._conversation_history: list[dict[str, str]] = []
        self._build_errors: list[str] = []
        self._result: MetaBuilderResult | None = None

    # ------------------------------------------------------------------
    # BaseAgent abstract method implementations
    # ------------------------------------------------------------------

    def _register_handlers(self, api: API) -> None:
        """No-op: MetaBuilderAgent uses turn-by-turn, not _standard_run."""
        pass

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run the meta-builder in single-shot mode.

        The task string is used as the initial user message describing
        what to build. The agent runs intake, build, and auto-approves
        the result.
        """
        self.start(initial_message=task)

        # Auto-approve if build completed during start
        if self.is_complete():
            return self.get_result()

        # If still in review, auto-approve
        if self._phase == MetaBuilderStates.REVIEW:
            self.send("approve")

        if self.is_complete():
            return self.get_result()

        # Build partial result
        self._build_result()
        return self._result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Turn-by-turn interface
    # ------------------------------------------------------------------

    def start(self, initial_message: str = "") -> str:
        """Initialize the conversation. Returns the first agent response.

        :param initial_message: Optional initial user message to set context
        :return: Agent's welcome/first response
        """
        if self._started:
            raise MetaBuilderError(MetaErrorMessages.CONVERSATION_ALREADY_STARTED)

        self._started = True
        logger.info(MetaLogMessages.META_STARTED.format(model=self.meta_config.model))

        if initial_message:
            return self._handle_intake(initial_message)
        return build_welcome_message()

    def send(self, message: str) -> str:
        """Send a user message and get the agent's response.

        :param message: User's message
        :return: Agent's response
        """
        if not self._started:
            raise MetaBuilderError(MetaErrorMessages.CONVERSATION_NOT_STARTED)
        if self._phase == MetaBuilderStates.OUTPUT:
            raise MetaBuilderError("Conversation has already completed")

        self._turn_count += 1
        if self._turn_count > self.meta_config.max_turns:
            raise MetaBuilderError(
                f"Maximum turns ({self.meta_config.max_turns}) exceeded"
            )

        if self._phase == MetaBuilderStates.INTAKE:
            return self._handle_intake(message)
        if self._phase == MetaBuilderStates.REVIEW:
            return self._handle_review(message)

        raise MetaBuilderError(f"Unexpected phase: {self._phase}")

    def is_complete(self) -> bool:
        """Whether the artifact has been fully built and approved."""
        return self._phase == MetaBuilderStates.OUTPUT

    def get_result(self) -> MetaBuilderResult:
        """Get the final build result.

        :return: MetaBuilderResult with the artifact
        :raises MetaBuilderError: If the build is not complete
        """
        if self._phase != MetaBuilderStates.OUTPUT:
            raise MetaBuilderError(
                "Build is not complete. Continue the conversation until the "
                "user approves the artifact."
            )
        if self._result is None:
            self._build_result()
        return self._result  # type: ignore[return-value]

    def get_internal_state(self) -> dict[str, Any]:
        """Get the current internal state for debugging/monitoring."""
        result: dict[str, Any] = {
            "phase": self._phase,
            "turn_count": self._turn_count,
            "is_complete": self.is_complete(),
            "started": self._started,
        }

        if self._artifact_type is not None:
            result["artifact_type"] = self._artifact_type.value

        if self._requirements:
            result["requirements"] = self._requirements

        result["conversation_history"] = list(self._conversation_history)
        result["build_errors"] = list(self._build_errors)

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
        """Run in interactive mode, reading from stdin.

        :return: MetaBuilderResult with the completed artifact
        """
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

        # Build partial result if interrupted
        self._build_result()
        return self._result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Phase 1: INTAKE
    # ------------------------------------------------------------------

    # Phrases that mean "stop asking, just build with defaults"
    _JUST_BUILD_PHRASES: frozenset[str] = frozenset(
        {
            "just build it",
            "just build",
            "just fill it",
            "fill it up",
            "fill it in",
            "dont care",
            "don't care",
            "whatever",
            "anything",
            "surprise me",
            "random",
            "just do it",
            "just make it",
            "just create it",
            "build something",
            "make something",
            "i dont care",
            "i don't care",
            "doesnt matter",
            "doesn't matter",
        }
    )

    def _is_just_build_request(self, message: str) -> bool:
        """Check if the user wants us to just build with defaults."""
        normalized = message.strip().lower()
        if normalized in self._JUST_BUILD_PHRASES:
            return True
        for phrase in self._JUST_BUILD_PHRASES:
            if len(phrase) > 4 and phrase in normalized:
                return True
        return False

    def _handle_intake(self, message: str) -> str:
        """Handle a message during the intake phase."""
        self._conversation_history.append({"role": "user", "content": message})

        # Quick path: "just build it" / "whatever" → defaults
        if self._is_just_build_request(message):
            if self._artifact_type is None:
                self._artifact_type = ArtifactType.FSM
            if not self._requirements.get("artifact_description"):
                self._requirements["artifact_description"] = (
                    f"A sample {self._artifact_type.value} artifact"
                )
            if not self._requirements.get("artifact_name"):
                self._requirements["artifact_name"] = (
                    f"Sample_{self._artifact_type.value.upper()}"
                )
            return self._do_build()

        # Extract requirements from all messages so far
        self._requirements = self._extract_requirements()

        # Resolve artifact type
        artifact_type = self._resolve_artifact_type(
            self._requirements.get("artifact_type")
        )
        name = self._requirements.get("artifact_name")
        description = self._requirements.get("artifact_description")

        # Auto-generate description from user messages if extraction missed it
        if not description:
            user_text = self._get_user_messages_text()
            if len(user_text.split()) > 1:
                description = user_text[:300].strip()
                self._requirements["artifact_description"] = description

        # Auto-generate name from description if missing
        if not name and description:
            name = self._generate_name(description)
            self._requirements["artifact_name"] = name
        elif not name:
            name = "Untitled"
            self._requirements["artifact_name"] = name

        # Build as soon as we know the artifact type
        if artifact_type:
            if not description:
                description = f"A {artifact_type.value} artifact"
                self._requirements["artifact_description"] = description
            self._artifact_type = artifact_type
            logger.info(
                MetaLogMessages.ARTIFACT_CLASSIFIED.format(
                    artifact_type=artifact_type.value
                )
            )
            return self._do_build()

        # Default to FSM if we have a description
        if description and len(description.split()) > 2:
            artifact_type = ArtifactType.FSM
            self._artifact_type = artifact_type
            logger.info(
                MetaLogMessages.ARTIFACT_CLASSIFIED.format(
                    artifact_type=artifact_type.value
                )
                + " (defaulted)"
            )
            return self._do_build()

        # Ask follow-up only if truly missing both type and description
        followup = build_followup_message(
            artifact_type=artifact_type,
            has_name=bool(name),
            has_description=bool(description),
        )
        self._conversation_history.append({"role": "assistant", "content": followup})
        return followup

    def _extract_requirements(self) -> dict[str, Any]:
        """Extract requirements from conversation history using LLM."""
        user_message = build_intake_user_message(self._conversation_history)

        data = self._call_llm_json(
            system_prompt=INTAKE_SYSTEM_PROMPT,
            user_message=user_message,
        )

        merged = dict(self._requirements)
        for key, value in data.items():
            if value is not None and value != "":
                merged[key] = value

        return merged

    _TYPE_ALIASES: ClassVar[dict[str, str]] = {
        "state machine": "fsm",
        "finite state machine": "fsm",
        "chatbot": "fsm",
        "chat bot": "fsm",
        "chat": "fsm",
        "conversation": "fsm",
        "conversational": "fsm",
        "bot": "fsm",
        "state_machine": "fsm",
        "dialogue": "fsm",
        "dialog": "fsm",
        "assistant": "fsm",
        "support": "fsm",
        "customer": "fsm",
        "helpdesk": "fsm",
        "help desk": "fsm",
        "faq": "fsm",
        "intake": "fsm",
        "interview": "fsm",
        "onboarding": "fsm",
        "survey": "fsm",
        "quiz": "fsm",
        "pipeline": "workflow",
        "process": "workflow",
        "automation": "workflow",
        "steps": "workflow",
        "flow": "workflow",
        "sequence": "workflow",
        "etl": "workflow",
        "data pipeline": "workflow",
        "batch": "workflow",
        "tool": "agent",
        "tools": "agent",
        "react": "agent",
        "agentic": "agent",
        "search": "agent",
        "browse": "agent",
        "navigate": "agent",
        "research": "agent",
    }

    def _resolve_artifact_type(self, raw: Any) -> ArtifactType | None:
        """Resolve a raw artifact type string to an ArtifactType enum."""
        if raw is None:
            return self._artifact_type
        if isinstance(raw, ArtifactType):
            return raw
        if not isinstance(raw, str):
            return None

        normalized = raw.strip().lower()
        normalized = self._TYPE_ALIASES.get(normalized, normalized)
        try:
            return ArtifactType(normalized)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Phase 2: BUILD
    # ------------------------------------------------------------------

    def _do_build(self) -> str:
        """Build the artifact via a single LLM call + direct builder methods."""
        assert self._artifact_type is not None

        if self._builder is None:
            self._builder = self._create_builder(self._artifact_type)

        logger.info(
            MetaLogMessages.BUILD_STARTED.format(
                artifact_type=self._artifact_type.value
            )
        )

        # Single LLM call to generate the full spec
        prompt = build_spec_prompt(
            artifact_type=self._artifact_type,
            name=self._requirements.get("artifact_name"),
            description=self._requirements.get("artifact_description"),
            persona=self._requirements.get("artifact_persona"),
            components=self._requirements.get("components"),
            user_messages=self._get_user_messages_text(),
        )

        build_error = None
        spec = self._call_llm_json(
            system_prompt=BUILD_SPEC_SYSTEM_PROMPT,
            user_message=prompt,
            temperature=MetaDefaults.BUILD_TEMPERATURE,
        )
        if not spec:
            build_error = "LLM returned empty or unparseable spec"
            logger.warning(build_error)
        else:
            try:
                self._apply_spec_to_builder(spec)
            except Exception as e:
                logger.error(f"Failed to apply spec to builder: {e}")
                build_error = str(e)

        if build_error:
            self._build_errors.append(build_error)

        # Fallback: pre-load requirements if builder is empty
        if (
            not getattr(self._builder, "states", None)
            and not getattr(self._builder, "steps", None)
            and not getattr(self._builder, "tools", None)
        ):
            self._preload_builder()

        # Transition to review
        self._phase = MetaBuilderStates.REVIEW
        presentation = build_review_presentation(self._builder, self._artifact_type)
        if build_error:
            presentation += (
                f"\n\nNote: The build process encountered an error: {build_error}\n"
                "The artifact above may be incomplete. You can approve as-is "
                "or describe changes to fix it."
            )
        return presentation

    def _do_revision(self, revision_request: str) -> str:
        """Revise the artifact via a single LLM call."""
        assert self._builder is not None
        assert self._artifact_type is not None
        logger.info(
            MetaLogMessages.REVISION_STARTED.format(revision=revision_request[:80])
        )

        current_spec = json.dumps(self._builder.to_dict(), indent=2)
        prompt = build_revision_spec_prompt(
            artifact_type=self._artifact_type,
            revision_request=revision_request,
            current_spec=current_spec,
        )

        spec = self._call_llm_json(
            system_prompt=BUILD_SPEC_SYSTEM_PROMPT,
            user_message=prompt,
            temperature=MetaDefaults.BUILD_TEMPERATURE,
        )

        if spec:
            new_builder = self._create_builder(self._artifact_type)
            old_builder = self._builder
            self._builder = new_builder
            try:
                self._apply_spec_to_builder(spec)
            except Exception as e:
                logger.error(f"Revision apply failed, restoring previous: {e}")
                self._builder = old_builder
        else:
            logger.warning("Revision returned empty spec, keeping current artifact")

        return build_review_presentation(self._builder, self._artifact_type)

    def _apply_spec_to_builder(self, spec: dict[str, Any]) -> None:
        """Apply a JSON spec to the builder using direct method calls."""
        builder = self._builder
        if builder is None:
            return

        if isinstance(builder, FSMBuilder):
            self._apply_fsm_spec(builder, spec)
        elif isinstance(builder, WorkflowBuilder):
            self._apply_workflow_spec(builder, spec)
        elif isinstance(builder, AgentBuilder):
            self._apply_agent_spec(builder, spec)

    def _extract_overview_fields(
        self, spec: dict[str, Any]
    ) -> tuple[str, str, str]:
        """Extract (name, description, persona) from requirements and spec."""
        name = self._requirements.get("artifact_name") or spec.get("name") or "Untitled"
        desc = (
            self._requirements.get("artifact_description")
            or spec.get("description")
            or ""
        )
        persona = (
            self._requirements.get("artifact_persona") or spec.get("persona") or ""
        )
        return str(name), str(desc), str(persona)

    def _apply_fsm_spec(self, builder: FSMBuilder, spec: dict[str, Any]) -> None:
        """Apply an FSM spec to the builder."""
        name, desc, persona = self._extract_overview_fields(spec)
        builder.set_overview(name=name, description=desc, persona=persona)

        # Add states -- handle both list and dict format
        states_raw = spec.get("states", [])
        state_items: list[dict[str, Any]] = []
        if isinstance(states_raw, list):
            state_items = [s for s in states_raw if isinstance(s, dict)]
        elif isinstance(states_raw, dict):
            for sid, sdata in states_raw.items():
                if isinstance(sdata, dict):
                    entry = dict(sdata)
                    if "id" not in entry:
                        entry["id"] = sid
                    state_items.append(entry)

        for s in state_items:
            state_id = str(s.get("id", s.get("state_id", "")))
            if not state_id:
                continue
            try:
                builder.add_state(
                    state_id=state_id,
                    description=str(s.get("description", ""))[:500],
                    purpose=str(s.get("purpose", s.get("description", "")))[:500],
                    extraction_instructions=str(s.get("extraction_instructions", ""))[
                        :500
                    ],
                    response_instructions=str(s.get("response_instructions", ""))[:500],
                )
            except Exception as e:
                logger.warning(f"Failed to add state '{state_id}': {e}")

        # Set initial state
        initial = spec.get("initial_state")
        if initial and str(initial) in builder.states:
            builder.set_initial_state(str(initial))

        # Add transitions -- support both top-level and per-state
        transitions: list[dict[str, Any]] = []

        top_transitions = spec.get("transitions", [])
        if isinstance(top_transitions, list):
            for t in top_transitions:
                if isinstance(t, dict):
                    transitions.append(t)

        for s in state_items:
            state_id = str(s.get("id", s.get("state_id", "")))
            embedded = s.get("transitions", [])
            if isinstance(embedded, list):
                for t in embedded:
                    if not isinstance(t, dict):
                        continue
                    target_val = (
                        t.get("target_state")
                        or t.get("target")
                        or t.get("to")
                        or t.get("to_state")
                    )
                    if target_val:
                        transitions.append(
                            {
                                "source": state_id,
                                "target": str(target_val),
                                "description": t.get("description", ""),
                                "priority": t.get("priority", 100),
                            }
                        )

        for t in transitions:
            source = str(
                t.get(
                    "source",
                    t.get(
                        "from",
                        t.get("from_state", t.get("source_state", "")),
                    ),
                )
            )
            target = str(
                t.get(
                    "target",
                    t.get("to", t.get("target_state", t.get("to_state", ""))),
                )
            )
            t_desc = str(t.get("description", ""))[:500]
            if (
                source
                and target
                and source in builder.states
                and target in builder.states
            ):
                try:
                    builder.add_transition(
                        from_state=source,
                        target_state=target,
                        description=t_desc,
                        priority=int(t.get("priority", 100)),
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to add transition '{source}' -> '{target}': {e}"
                    )

    def _apply_workflow_spec(
        self, builder: WorkflowBuilder, spec: dict[str, Any]
    ) -> None:
        """Apply a workflow spec to the builder."""
        wf_id = spec.get("workflow_id") or "workflow_1"
        name, desc, _persona = self._extract_overview_fields(spec)
        builder.set_overview(workflow_id=str(wf_id), name=name, description=desc)

        steps = spec.get("steps", [])
        if isinstance(steps, list):
            for s in steps:
                if not isinstance(s, dict):
                    continue
                step_id = str(s.get("id", ""))
                if not step_id:
                    continue
                builder.add_step(
                    step_id=step_id,
                    step_type=str(s.get("step_type", "auto_transition")),
                    name=str(s.get("name", step_id)),
                    description=str(s.get("description", "")),
                )
                next_step = s.get("next_step")
                if next_step:
                    builder.set_step_transition(step_id, str(next_step))

        initial = spec.get("initial_step_id")
        if initial and str(initial) in builder.steps:
            builder.set_initial_step(str(initial))

    def _apply_agent_spec(self, builder: AgentBuilder, spec: dict[str, Any]) -> None:
        """Apply an agent spec to the builder."""
        name, desc, _persona = self._extract_overview_fields(spec)
        builder.set_overview(name=name, description=desc)

        agent_type = spec.get("agent_type", "react")
        builder.set_agent_type(str(agent_type))

        tools = spec.get("tools", [])
        if isinstance(tools, list):
            for t in tools:
                if not isinstance(t, dict):
                    continue
                tool_name = str(t.get("name", ""))
                if tool_name:
                    builder.add_tool(
                        name=tool_name,
                        description=str(t.get("description", "")),
                    )

        config = spec.get("config")
        if isinstance(config, dict):
            builder.set_config(**config)

    # ------------------------------------------------------------------
    # Phase 3: REVIEW
    # ------------------------------------------------------------------

    def _handle_review(self, message: str) -> str:
        """Handle a message during the review phase."""
        decision = self._classify_decision(message)

        if decision == "approve":
            self._phase = MetaBuilderStates.OUTPUT
            self._build_result()
            artifact_json = self._result.artifact_json if self._result else "{}"
            return build_output_message(artifact_json)

        return self._do_revision(message)

    def _classify_decision(self, message: str) -> str:
        """Classify user message as 'approve' or 'revise'.

        Uses a 3-tier strategy:
        1. Exact match against known approval/revision phrases
        2. Word-boundary matching
        3. LLM classification for ambiguous messages
        """
        normalized = message.strip().lower()

        # Tier 1: exact match
        if normalized in DecisionWords.APPROVE:
            return "approve"
        if normalized in DecisionWords.REVISE:
            return "revise"

        # Tier 2: word-boundary matching
        words = set(normalized.split())

        approve_phrases = {
            "no changes",
            "no change",
            "change nothing",
            "nothing to change",
            "looks good",
            "sounds good",
            "all good",
            "no changes needed",
            "that's right",
            "thats right",
            "looks great",
            "looks perfect",
        }
        if any(phrase in normalized for phrase in approve_phrases):
            return "approve"

        revise_phrases = {
            "not right",
            "needs work",
            "not quite",
            "try again",
        }
        has_revise_phrase = any(phrase in normalized for phrase in revise_phrases)

        has_revise = has_revise_phrase or bool(words & DecisionWords.REVISE)
        has_approve = bool(words & DecisionWords.APPROVE)

        if "but" in words and has_approve:
            return "revise"
        if has_revise and not has_approve:
            return "revise"
        if has_approve and not has_revise:
            return "approve"

        # Tier 3: LLM classification for genuinely ambiguous messages
        if len(normalized.split()) > 3:
            data = self._call_llm_json(
                system_prompt=(
                    "You are a decision classifier. The user is reviewing "
                    "an artifact they asked to be built. Classify their "
                    "message as either approving the artifact or requesting "
                    'changes. Respond with ONLY {"decision": "approve"} or '
                    '{"decision": "revise"}.'
                ),
                user_message=f"User message: {message}",
                temperature=MetaDefaults.BUILD_TEMPERATURE,
            )
            decision = str(data.get("decision", ""))
            if decision in ("approve", "revise"):
                return decision

        # Default: short → approve, long → revise
        if len(normalized.split()) > 8:
            return "revise"
        return "approve"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Call the LLM directly and parse the response as JSON.

        Returns an empty dict on failure (never raises).
        """
        model = self.meta_config.model
        temp = temperature if temperature is not None else self.meta_config.temperature
        reserved = {"model", "messages", "temperature", "max_tokens"}
        safe_kwargs = {k: v for k, v in self._api_kwargs.items() if k not in reserved}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            t0 = time.time()
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temp,
                max_tokens=self.meta_config.max_tokens,
                **safe_kwargs,
            )
            dt = time.time() - t0
            logger.debug(f"Meta-agent LLM call completed in {dt:.2f}s")

            content = response.choices[0].message.content
            if not content:
                logger.warning("LLM returned empty content")
                return {}

            text = content.strip()
            if text.startswith("{"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

            data = extract_json_from_text(text)
            if isinstance(data, dict):
                return data

            logger.warning(f"Could not parse LLM response as JSON: {text[:200]}")
            return {}

        except Exception as e:
            logger.error(f"Meta-agent LLM call failed: {e}")
            return {}

    def _create_builder(
        self, artifact_type: ArtifactType
    ) -> FSMBuilder | WorkflowBuilder | AgentBuilder:
        """Create the appropriate builder for the artifact type."""
        if artifact_type == ArtifactType.FSM:
            return FSMBuilder()
        if artifact_type == ArtifactType.WORKFLOW:
            return WorkflowBuilder()
        if artifact_type == ArtifactType.AGENT:
            return AgentBuilder()
        raise MetaBuilderError(f"Unknown artifact type: {artifact_type}")

    @staticmethod
    def _generate_name(description: str) -> str:
        """Generate a short artifact name from a description."""
        stop = {
            "a",
            "an",
            "the",
            "for",
            "and",
            "or",
            "to",
            "is",
            "that",
            "it",
            "of",
        }
        words = [w for w in description.split() if w.lower() not in stop]
        name_words = words[:3] if words else ["Untitled"]
        return "_".join(w.capitalize() for w in name_words)

    def _preload_builder(self) -> None:
        """Pre-load extracted requirements into the builder."""
        builder = self._builder
        if builder is None:
            return
        name = self._requirements.get("artifact_name") or "Untitled"
        desc = self._requirements.get("artifact_description") or ""
        if isinstance(builder, FSMBuilder):
            persona = self._requirements.get("artifact_persona") or ""
            builder.set_overview(name=name, description=desc, persona=persona)
        elif isinstance(builder, WorkflowBuilder):
            wf_id = name.lower().replace(" ", "_")
            builder.set_overview(workflow_id=wf_id, name=name, description=desc)
        elif isinstance(builder, AgentBuilder):
            builder.set_overview(name=name, description=desc)

    def _get_user_messages_text(self) -> str:
        """Concatenate all user messages into one string."""
        parts = [
            msg["content"]
            for msg in self._conversation_history
            if msg.get("role") == "user" and msg.get("content")
        ]
        return "\n".join(parts)

    def _build_result(self) -> None:
        """Build the MetaBuilderResult from current builder state."""
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
        )
