from __future__ import annotations

"""
MetaAgent — interactively builds FSMs, Workflows, and Agents.

Drives a conversational loop asking the user adaptive questions
until the artifact is fully specified, then outputs a validated
JSON definition.
"""

import json
from typing import Any

from fsm_llm import API
from fsm_llm.context import ContextCompactor
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .constants import ContextKeys, HandlerNames, LogMessages, MetaStates
from .definitions import ArtifactType, MetaAgentConfig, MetaAgentResult
from .exceptions import BuilderError, MetaAgentError, MetaValidationError
from .fsm_definitions import build_meta_fsm
from .handlers import MetaHandlers


class MetaAgent:
    """
    Meta-agent that interactively builds FSMs, Workflows, and Agents.

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
        self._handlers = MetaHandlers()
        self._api: API | None = None
        self._conv_id: str | None = None
        self._turn_count: int = 0
        self._complete: bool = False
        self._result: MetaAgentResult | None = None

    def start(self, initial_message: str = "") -> str:
        """Initialize the conversation. Returns the first agent response.

        :param initial_message: Optional initial user message to set context
        :return: Agent's welcome/first response
        """
        if self._api is not None:
            raise MetaAgentError("Conversation has already been started")

        self._handlers.reset()

        # Build the meta-agent FSM
        fsm_def = build_meta_fsm()

        # Create API instance
        self._api = API.from_definition(
            fsm_def,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self._api_kwargs,
        )

        # Register handlers
        self._register_handlers()

        # Build initial context
        context: dict[str, Any] = {}
        if initial_message:
            context["_initial_request"] = initial_message

        # Start conversation
        self._conv_id, initial_response = self._api.start_conversation(context)

        logger.info(LogMessages.META_STARTED.format(model=self.config.model))

        return initial_response

    def send(self, message: str) -> str:
        """Send a user message and get the agent's response.

        :param message: User's message
        :return: Agent's response
        """
        if self._api is None or self._conv_id is None:
            raise MetaAgentError(
                "Conversation has not been started. Call start() first"
            )

        if self._complete:
            raise MetaAgentError("Conversation has already completed")

        self._turn_count += 1

        if self._turn_count > self.config.max_turns:
            raise MetaAgentError(f"Maximum turns ({self.config.max_turns}) exceeded")

        response = self._api.converse(message, self._conv_id)

        # Check if conversation ended
        if self._api.has_conversation_ended(self._conv_id):
            self._complete = True
            try:
                self._build_result()
            except (BuilderError, MetaValidationError, ValueError, TypeError) as e:
                logger.error(f"Failed to build result: {e}")
                # Create a fallback result so get_result() doesn't fail
                self._result = MetaAgentResult(
                    artifact_type=self._handlers._artifact_type or ArtifactType.FSM,
                    artifact={},
                    artifact_json="{}",
                    is_valid=False,
                    validation_errors=[f"Result build failed: {e}"],
                    conversation_turns=self._turn_count,
                )

        return response

    def is_complete(self) -> bool:
        """Whether the artifact has been fully built."""
        return self._complete

    def get_result(self) -> MetaAgentResult:
        """Get the final build result.

        :return: MetaAgentResult with the artifact
        :raises MetaAgentError: If the build is not complete
        """
        if not self._complete:
            raise MetaAgentError(
                "Build is not complete. Continue the conversation until the "
                "user approves the artifact."
            )
        if self._result is None:
            self._build_result()
        return self._result  # type: ignore[return-value]

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
    # Internal methods
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        """Register all meta-agent handlers with the API."""
        api = self._api
        if api is None:
            return

        # Context compaction: clear transient keys each turn and prune
        # phase-specific keys on state transitions.
        self._compactor = ContextCompactor(
            transient_keys={
                ContextKeys.ACTION_RESULT,
                ContextKeys.ACTION_ERRORS,
                ContextKeys.ACTION,
                ContextKeys.ACTION_PARAMS,
            },
            prune_on_entry={
                MetaStates.DESIGN_STRUCTURE: {
                    ContextKeys.ARTIFACT_NAME,
                    ContextKeys.ARTIFACT_DESCRIPTION,
                    ContextKeys.ARTIFACT_PERSONA,
                    ContextKeys.STRUCTURE_DONE,
                    ContextKeys.CONNECTIONS_DONE,
                    ContextKeys.USER_DECISION,
                    ContextKeys.VALIDATION_ERRORS,
                    ContextKeys.VALIDATION_WARNINGS,
                },
                MetaStates.DEFINE_CONNECTIONS: {ContextKeys.STRUCTURE_DONE},
                MetaStates.REVIEW: {ContextKeys.CONNECTIONS_DONE},
                MetaStates.OUTPUT: {
                    ContextKeys.USER_DECISION,
                    ContextKeys.VALIDATION_ERRORS,
                    ContextKeys.VALIDATION_WARNINGS,
                },
            },
        )

        # PRE_PROCESSING: Compact context (must run before builder injector)
        api.register_handler(
            api.create_handler(HandlerNames.CONTEXT_COMPACTOR)
            .at(HandlerTiming.PRE_PROCESSING)
            .do(self._compactor.compact)
        )

        # PRE_PROCESSING: Inject builder state before each LLM call
        api.register_handler(
            api.create_handler(HandlerNames.BUILDER_INJECTOR)
            .at(HandlerTiming.PRE_PROCESSING)
            .do(self._handlers.inject_builder_state)
        )

        # POST_TRANSITION: Prune phase-specific keys on state entry
        api.register_handler(
            api.create_handler(HandlerNames.TRANSITION_PRUNER)
            .at(HandlerTiming.POST_TRANSITION)
            .do(self._compactor.prune)
        )

        # POST_PROCESSING on classify: validate and normalize artifact type
        api.register_handler(
            api.create_handler(HandlerNames.ACTION_DISPATCHER + "_classify")
            .at(HandlerTiming.POST_PROCESSING)
            .on_state(MetaStates.CLASSIFY)
            .do(self._handlers.classify_artifact_type)
        )

        # POST_PROCESSING on gather_overview: handle overview fields
        api.register_handler(
            api.create_handler(HandlerNames.BUILDER_INJECTOR + "_overview")
            .at(HandlerTiming.POST_PROCESSING)
            .on_state(MetaStates.GATHER_OVERVIEW)
            .do(self._handlers.handle_overview)
        )

        # POST_PROCESSING on design_structure: dispatch actions
        api.register_handler(
            api.create_handler(HandlerNames.ACTION_DISPATCHER + "_structure")
            .at(HandlerTiming.POST_PROCESSING)
            .on_state(MetaStates.DESIGN_STRUCTURE)
            .do(self._handlers.dispatch_action)
        )

        # POST_PROCESSING on define_connections: dispatch actions
        api.register_handler(
            api.create_handler(HandlerNames.ACTION_DISPATCHER + "_connections")
            .at(HandlerTiming.POST_PROCESSING)
            .on_state(MetaStates.DEFINE_CONNECTIONS)
            .do(self._handlers.dispatch_action)
        )

        # POST_PROCESSING on review: normalize user_decision variants
        api.register_handler(
            api.create_handler(HandlerNames.ACTION_DISPATCHER + "_review")
            .at(HandlerTiming.POST_PROCESSING)
            .on_state(MetaStates.REVIEW)
            .do(self._handlers.normalize_decision)
        )

        # POST_TRANSITION on entering review: run validation
        api.register_handler(
            api.create_handler(HandlerNames.PROGRESS_TRACKER)
            .on_state_entry(MetaStates.REVIEW)
            .do(self._handlers.run_validation)
        )

        # POST_TRANSITION on entering output: finalize
        api.register_handler(
            api.create_handler(HandlerNames.FINALIZER)
            .on_state_entry(MetaStates.OUTPUT)
            .do(self._handlers.finalize)
        )

    def _build_result(self) -> None:
        """Build the MetaAgentResult from current state."""
        builder = self._handlers.builder
        artifact_type = self._handlers._artifact_type or ArtifactType.FSM

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
