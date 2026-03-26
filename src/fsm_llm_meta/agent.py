from __future__ import annotations

"""
MetaAgent -- interactively builds FSMs, Workflows, and Agents.

Uses a 3-phase architecture:
  1. INTAKE  -- extract requirements from user input (1-2 turns)
  2. BUILD   -- ReactAgent autonomously constructs the artifact
  3. REVIEW  -- user approves or requests revisions
"""

import json
from typing import Any

from fsm_llm.definitions import DataExtractionRequest
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.logging import logger
from fsm_llm_agents import ReactAgent
from fsm_llm_agents.definitions import AgentConfig

from .builders import AgentBuilder, FSMBuilder, WorkflowBuilder
from .constants import DecisionWords, Defaults, LogMessages, MetaPhases
from .definitions import ArtifactType, MetaAgentConfig, MetaAgentResult
from .exceptions import MetaAgentError
from .prompts import (
    INTAKE_SYSTEM_PROMPT,
    build_followup_message,
    build_intake_user_message,
    build_output_message,
    build_review_presentation,
    build_revision_prompt,
    build_task_prompt,
    build_welcome_message,
)
from .tools import create_builder_tools

# Normalize common variants of artifact type strings
_TYPE_ALIASES: dict[str, str] = {
    "state machine": "fsm",
    "finite state machine": "fsm",
    "chatbot": "fsm",
    "conversation": "fsm",
    "bot": "fsm",
    "state_machine": "fsm",
    "conversational": "fsm",
    "pipeline": "workflow",
    "process": "workflow",
    "automation": "workflow",
    "steps": "workflow",
    "tool": "agent",
    "tools": "agent",
    "react": "agent",
    "agentic": "agent",
}


class MetaAgent:
    """
    Meta-agent that interactively builds FSMs, Workflows, and Agents.

    Uses ReactAgent internally to autonomously construct artifacts
    from user requirements, minimizing unnecessary questions.

    Usage (turn-by-turn)::

        agent = MetaAgent()
        response = agent.start()
        print(response)

        while not agent.is_complete():
            user_input = input("> ")
            response = agent.send(user_input)
            print(response)

        result = agent.get_result()
        print(result.artifact_json)

    Usage (interactive CLI)::

        agent = MetaAgent()
        result = agent.run_interactive()
        print(result.artifact_json)
    """

    def __init__(
        self,
        config: MetaAgentConfig | None = None,
        **api_kwargs: Any,
    ) -> None:
        self.config = config or MetaAgentConfig()
        self._api_kwargs = api_kwargs

        # Phase tracking
        self._phase: str = MetaPhases.INTAKE
        self._turn_count: int = 0
        self._result: MetaAgentResult | None = None
        self._started: bool = False

        # Intake state
        self._conversation_history: list[dict[str, str]] = []
        self._requirements: dict[str, Any] = {}

        # Build state
        self._builder: FSMBuilder | WorkflowBuilder | AgentBuilder | None = None
        self._artifact_type: ArtifactType | None = None

        # LLM interface (lazy)
        self._llm: LiteLLMInterface | None = None

    # ------------------------------------------------------------------
    # Public API (unchanged contract)
    # ------------------------------------------------------------------

    def start(self, initial_message: str = "") -> str:
        """Initialize the conversation. Returns the first agent response.

        :param initial_message: Optional initial user message to set context
        :return: Agent's welcome/first response
        """
        if self._started:
            raise MetaAgentError("Conversation has already been started")

        self._started = True
        self._llm = LiteLLMInterface(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self._api_kwargs,
        )

        logger.info(LogMessages.META_STARTED.format(model=self.config.model))

        if initial_message:
            return self._handle_intake(initial_message)
        return build_welcome_message()

    def send(self, message: str) -> str:
        """Send a user message and get the agent's response.

        :param message: User's message
        :return: Agent's response
        """
        if not self._started:
            raise MetaAgentError(
                "Conversation has not been started. Call start() first"
            )
        if self._phase == MetaPhases.DONE:
            raise MetaAgentError("Conversation has already completed")

        self._turn_count += 1
        if self._turn_count > self.config.max_turns:
            raise MetaAgentError(f"Maximum turns ({self.config.max_turns}) exceeded")

        if self._phase == MetaPhases.INTAKE:
            return self._handle_intake(message)
        if self._phase == MetaPhases.REVIEW:
            return self._handle_review(message)

        raise MetaAgentError(f"Unexpected phase: {self._phase}")

    def is_complete(self) -> bool:
        """Whether the artifact has been fully built."""
        return self._phase == MetaPhases.DONE

    def get_result(self) -> MetaAgentResult:
        """Get the final build result.

        :return: MetaAgentResult with the artifact
        :raises MetaAgentError: If the build is not complete
        """
        if self._phase != MetaPhases.DONE:
            raise MetaAgentError(
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
        }

        if self._artifact_type is not None:
            result["artifact_type"] = self._artifact_type.value

        if self._requirements:
            result["requirements"] = self._requirements

        builder = self._builder
        if builder is not None:
            result["builder_progress"] = {
                "percentage": builder.get_progress().percentage,
                "missing": builder.get_missing_fields(),
            }
            result["builder_summary"] = builder.get_summary(detail_level="standard")
            result["artifact_preview"] = builder.to_dict()
        else:
            result["builder_progress"] = None
            result["builder_summary"] = None
            result["artifact_preview"] = None

        return result

    def run_interactive(self) -> MetaAgentResult:
        """Run in interactive mode, reading from stdin.

        :return: MetaAgentResult with the completed artifact
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

    def _handle_intake(self, message: str) -> str:
        """Handle a message during the intake phase."""
        self._conversation_history.append({"role": "user", "content": message})

        # Extract requirements from all messages so far
        self._requirements = self._extract_requirements()

        # Resolve artifact type
        artifact_type = self._resolve_artifact_type(
            self._requirements.get("artifact_type")
        )
        name = self._requirements.get("artifact_name")
        description = self._requirements.get("artifact_description")

        # Check if we have enough to build
        if artifact_type and name and description:
            self._artifact_type = artifact_type
            logger.info(
                LogMessages.ARTIFACT_CLASSIFIED.format(
                    artifact_type=artifact_type.value
                )
            )
            return self._do_build()

        # Ask for missing info
        followup = build_followup_message(
            artifact_type=artifact_type,
            has_name=bool(name),
            has_description=bool(description),
        )
        self._conversation_history.append({"role": "assistant", "content": followup})
        return followup

    def _extract_requirements(self) -> dict[str, Any]:
        """Extract requirements from conversation history using LLM."""
        assert self._llm is not None

        user_message = build_intake_user_message(self._conversation_history)

        try:
            response = self._llm.extract_data(
                DataExtractionRequest(
                    system_prompt=INTAKE_SYSTEM_PROMPT,
                    user_message=user_message,
                )
            )
            data = response.extracted_data
        except Exception as e:
            logger.warning(f"Intake extraction failed: {e}")
            data = {}

        # Merge with existing requirements (don't overwrite with None)
        merged = dict(self._requirements)
        for key, value in data.items():
            if value is not None:
                merged[key] = value

        return merged

    def _resolve_artifact_type(self, raw: Any) -> ArtifactType | None:
        """Resolve a raw artifact type string to an ArtifactType enum."""
        if raw is None:
            return self._artifact_type  # Keep previously resolved type
        if isinstance(raw, ArtifactType):
            return raw
        if not isinstance(raw, str):
            return None

        normalized = raw.strip().lower()
        normalized = _TYPE_ALIASES.get(normalized, normalized)
        try:
            return ArtifactType(normalized)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Phase 2: BUILD (autonomous via ReactAgent)
    # ------------------------------------------------------------------

    def _do_build(self) -> str:
        """Run the build phase using ReactAgent with builder tools."""
        assert self._artifact_type is not None

        # Create builder if not exists (first build)
        if self._builder is None:
            self._builder = self._create_builder(self._artifact_type)

        # Create tools for this builder type
        tools = create_builder_tools(self._builder, self._artifact_type)

        # Build the task prompt
        task = build_task_prompt(
            artifact_type=self._artifact_type,
            name=self._requirements.get("artifact_name"),
            description=self._requirements.get("artifact_description"),
            persona=self._requirements.get("artifact_persona"),
            components=self._requirements.get("components"),
            user_messages=self._get_user_messages_text(),
        )

        logger.info(
            LogMessages.BUILD_STARTED.format(
                artifact_type=self._artifact_type.value
            )
        )

        # Run ReactAgent
        agent_config = AgentConfig(
            model=self.config.model,
            max_iterations=Defaults.BUILD_MAX_ITERATIONS,
            timeout_seconds=Defaults.BUILD_TIMEOUT_SECONDS,
            temperature=Defaults.BUILD_TEMPERATURE,
            max_tokens=Defaults.BUILD_MAX_TOKENS,
        )
        react_agent = ReactAgent(tools=tools, config=agent_config)

        try:
            react_agent.run(task=task)
        except Exception as e:
            logger.error(f"ReactAgent build failed: {e}")
            # Continue to review even on failure — builder may have partial state

        # Transition to review
        self._phase = MetaPhases.REVIEW
        return build_review_presentation(self._builder, self._artifact_type)

    def _do_revision(self, revision_request: str) -> str:
        """Re-run ReactAgent with a revision task."""
        assert self._builder is not None
        assert self._artifact_type is not None

        logger.info(
            LogMessages.REVISION_STARTED.format(
                revision=revision_request[:80]
            )
        )

        tools = create_builder_tools(self._builder, self._artifact_type)
        summary = self._builder.get_summary(detail_level="full")

        task = build_revision_prompt(
            revision_request=revision_request,
            builder_summary=summary,
        )

        agent_config = AgentConfig(
            model=self.config.model,
            max_iterations=Defaults.BUILD_MAX_ITERATIONS,
            timeout_seconds=Defaults.BUILD_TIMEOUT_SECONDS,
            temperature=Defaults.BUILD_TEMPERATURE,
            max_tokens=Defaults.BUILD_MAX_TOKENS,
        )
        react_agent = ReactAgent(tools=tools, config=agent_config)

        try:
            react_agent.run(task=task)
        except Exception as e:
            logger.error(f"ReactAgent revision failed: {e}")

        return build_review_presentation(self._builder, self._artifact_type)

    # ------------------------------------------------------------------
    # Phase 3: REVIEW
    # ------------------------------------------------------------------

    def _handle_review(self, message: str) -> str:
        """Handle a message during the review phase."""
        decision = self._classify_decision(message)

        if decision == "approve":
            self._phase = MetaPhases.DONE
            self._build_result()
            artifact_json = self._result.artifact_json if self._result else "{}"
            return build_output_message(artifact_json)

        # Treat as revision request
        return self._do_revision(message)

    def _classify_decision(self, message: str) -> str:
        """Classify user message as 'approve' or 'revise'."""
        normalized = message.strip().lower()

        # Check exact and substring matches
        if normalized in DecisionWords.APPROVE:
            return "approve"
        if normalized in DecisionWords.REVISE:
            return "revise"

        # Check if any approve word is contained in the message
        for word in DecisionWords.APPROVE:
            if word in normalized and len(word) > 2:
                return "approve"

        # Default to revision if the message has substance
        return "revise"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        raise MetaAgentError(f"Unknown artifact type: {artifact_type}")

    def _get_user_messages_text(self) -> str:
        """Concatenate all user messages into one string."""
        parts = [
            msg["content"]
            for msg in self._conversation_history
            if msg.get("role") == "user" and msg.get("content")
        ]
        return "\n".join(parts)

    def _build_result(self) -> None:
        """Build the MetaAgentResult from current builder state."""
        artifact_type = self._artifact_type or ArtifactType.FSM
        builder = self._builder

        if builder is None:
            self._result = MetaAgentResult(
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
        artifact_json = json.dumps(artifact, indent=2)

        self._result = MetaAgentResult(
            artifact_type=artifact_type,
            artifact=artifact,
            artifact_json=artifact_json,
            is_valid=len(errors) == 0,
            validation_errors=errors,
            conversation_turns=self._turn_count,
        )
