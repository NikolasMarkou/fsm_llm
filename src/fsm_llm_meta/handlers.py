from __future__ import annotations

"""
Meta-agent handlers for builder injection, action dispatch,
progress tracking, and finalization.
"""

import json
from typing import Any, ClassVar

from fsm_llm.logging import logger

from .builders import AgentBuilder, ArtifactBuilder, FSMBuilder, WorkflowBuilder
from .constants import Actions, ContextKeys, LogMessages, MetaStates
from .definitions import ArtifactType
from .exceptions import BuilderError


class MetaHandlers:
    """Handler functions for the meta-agent FSM.

    Holds a reference to the builder and dispatches actions
    extracted by the LLM to the appropriate builder methods.
    """

    # Maps meta-agent states to builder summary detail levels.
    _DETAIL_LEVEL_MAP: ClassVar[dict[str, str]] = {
        MetaStates.DESIGN_STRUCTURE: "minimal",
        MetaStates.DEFINE_CONNECTIONS: "standard",
        MetaStates.REVIEW: "full",
    }

    # Common LLM variants for user_decision → canonical value
    _DECISION_APPROVE: ClassVar[set[str]] = {
        "approve",
        "approved",
        "yes",
        "ok",
        "okay",
        "looks good",
        "lgtm",
        "accept",
        "accepted",
        "confirm",
        "confirmed",
        "good",
        "great",
        "perfect",
        "ship it",
        "go ahead",
    }
    _DECISION_REVISE: ClassVar[set[str]] = {
        "revise",
        "revision",
        "change",
        "changes",
        "modify",
        "edit",
        "update",
        "fix",
        "no",
        "nope",
        "redo",
    }

    # Common LLM variants for action → canonical done
    _ACTION_DONE_VARIANTS: ClassVar[set[str]] = {
        "done",
        "finish",
        "finished",
        "complete",
        "completed",
        "ready",
        "next",
        "move on",
        "proceed",
        "continue",
        "that's all",
        "thats all",
        "nothing else",
        "no more",
    }

    def __init__(self) -> None:
        self.builder: ArtifactBuilder | None = None
        self._artifact_type: ArtifactType | None = None
        self._turn_count: int = 0

    def reset(self) -> None:
        """Reset handler state for a new session."""
        self.builder = None
        self._artifact_type = None
        self._turn_count = 0

    # ------------------------------------------------------------------
    # Builder initialization
    # ------------------------------------------------------------------

    def _ensure_builder(self, artifact_type_str: str) -> None:
        """Initialize the appropriate builder from artifact type string."""
        if self.builder is not None:
            return

        # Normalize common LLM variants
        normalized = artifact_type_str.strip().lower()
        _TYPE_ALIASES = {
            "state machine": "fsm",
            "finite state machine": "fsm",
            "chatbot": "fsm",
            "conversation": "fsm",
            "bot": "fsm",
            "pipeline": "workflow",
            "process": "workflow",
        }
        normalized = _TYPE_ALIASES.get(normalized, normalized)

        try:
            artifact_type = ArtifactType(normalized)
        except ValueError:
            logger.warning(f"Unknown artifact type: {artifact_type_str}")
            artifact_type = ArtifactType.FSM

        self._artifact_type = artifact_type

        if artifact_type == ArtifactType.FSM:
            self.builder = FSMBuilder()
        elif artifact_type == ArtifactType.WORKFLOW:
            self.builder = WorkflowBuilder()
        elif artifact_type == ArtifactType.AGENT:
            self.builder = AgentBuilder()

        logger.info(
            LogMessages.ARTIFACT_CLASSIFIED.format(artifact_type=artifact_type.value)
        )

    # ------------------------------------------------------------------
    # PRE_PROCESSING: Inject builder state into context + dynamic prompts
    # ------------------------------------------------------------------

    def inject_builder_state(self, context: dict[str, Any]) -> dict[str, Any]:
        """Inject builder summary and progress into context.

        Selects the appropriate summary detail level based on the current
        meta-agent state (``_current_state`` in context) to keep context
        compact during design phases and detailed during review.
        """
        self._turn_count += 1

        # Initialize builder when artifact type becomes available
        artifact_type = context.get(ContextKeys.ARTIFACT_TYPE)
        if artifact_type and self.builder is None:
            self._ensure_builder(artifact_type)

        if self.builder is None:
            return {}

        current_state = context.get("_current_state", "")
        detail_level = self._DETAIL_LEVEL_MAP.get(current_state, "minimal")

        summary = self.builder.get_summary(detail_level=detail_level)
        progress = self.builder.get_progress()
        missing = self.builder.get_missing_fields()

        return {
            ContextKeys.BUILDER_SUMMARY: summary,
            ContextKeys.BUILDER_PROGRESS: f"{progress.percentage:.0f}% complete",
            ContextKeys.BUILDER_MISSING: ", ".join(missing) if missing else "none",
        }

    # ------------------------------------------------------------------
    # POST_PROCESSING: Handle overview fields from gather_overview state
    # ------------------------------------------------------------------

    def handle_overview(self, context: dict[str, Any]) -> dict[str, Any]:
        """Process overview fields extracted in gather_overview state."""
        if self.builder is None:
            return {}

        updates: dict[str, Any] = {}
        name = context.get(ContextKeys.ARTIFACT_NAME)
        desc = context.get(ContextKeys.ARTIFACT_DESCRIPTION)
        persona = context.get(ContextKeys.ARTIFACT_PERSONA)

        if name or desc:
            if isinstance(self.builder, FSMBuilder):
                warnings = self.builder.set_overview(
                    name=name or self.builder.name or "",
                    description=desc or self.builder.description or "",
                    persona=persona,
                )
            elif isinstance(self.builder, WorkflowBuilder):
                wf_id = (name or "").lower().replace(" ", "_")
                warnings = self.builder.set_overview(
                    workflow_id=wf_id,
                    name=name or self.builder.name or "",
                    description=desc or self.builder.description or "",
                )
            elif isinstance(self.builder, AgentBuilder):
                warnings = self.builder.set_overview(
                    name=name or self.builder.name or "",
                    description=desc or self.builder.description or "",
                )
            else:
                warnings = []

            if warnings:
                updates[ContextKeys.ACTION_ERRORS] = "; ".join(warnings)

        return updates

    # ------------------------------------------------------------------
    # POST_PROCESSING: Normalize user decision for review state
    # ------------------------------------------------------------------

    def normalize_decision(self, context: dict[str, Any]) -> dict[str, Any]:
        """Normalize common LLM variants of user_decision to canonical values."""
        decision = context.get(ContextKeys.USER_DECISION)
        if not decision or not isinstance(decision, str):
            return {}

        normalized = decision.strip().lower()
        if normalized in self._DECISION_APPROVE:
            if normalized != "approve":
                logger.debug(f"Normalized user_decision '{decision}' → 'approve'")
            return {ContextKeys.USER_DECISION: "approve"}
        if normalized in self._DECISION_REVISE:
            if normalized != "revise":
                logger.debug(f"Normalized user_decision '{decision}' → 'revise'")
            return {ContextKeys.USER_DECISION: "revise"}

        # Unknown value — leave as-is, transition won't fire
        logger.warning(f"Unknown user_decision value: '{decision}'")
        return {}

    # ------------------------------------------------------------------
    # POST_PROCESSING: Dispatch extracted actions to builder
    # ------------------------------------------------------------------

    def dispatch_action(self, context: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an extracted action to the appropriate builder method."""
        action = context.get(ContextKeys.ACTION)
        if not action:
            return {}

        # Normalize action name
        if isinstance(action, str):
            action = action.strip().lower()
            if action in self._ACTION_DONE_VARIANTS:
                action = Actions.DONE

        params = context.get(ContextKeys.ACTION_PARAMS, {})
        if not isinstance(params, dict):
            params = {}

        logger.info(LogMessages.ACTION_DISPATCHED.format(action=action))

        try:
            result, updates = self._execute_action(action, params)
            logger.info(LogMessages.ACTION_SUCCEEDED.format(action=action))
            updates[ContextKeys.ACTION_RESULT] = result
            # Clear action so it doesn't re-fire
            updates[ContextKeys.ACTION] = None
            updates[ContextKeys.ACTION_PARAMS] = None
            return updates

        except BuilderError as e:
            logger.warning(
                LogMessages.ACTION_FAILED.format(action=action, error=str(e))
            )
            return {
                ContextKeys.ACTION_ERRORS: str(e),
                ContextKeys.ACTION: None,
                ContextKeys.ACTION_PARAMS: None,
            }
        except Exception as e:
            logger.error(f"Unexpected error dispatching action '{action}': {e}")
            return {
                ContextKeys.ACTION_ERRORS: f"Unexpected error: {e}",
                ContextKeys.ACTION: None,
                ContextKeys.ACTION_PARAMS: None,
            }

    def _execute_action(
        self, action: str, params: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Execute a builder action. Returns (result_message, context_updates)."""
        if self.builder is None:
            return "Builder not initialized", {}

        updates: dict[str, Any] = {}

        # Done actions
        if action == Actions.DONE:
            updates[ContextKeys.STRUCTURE_DONE] = True
            updates[ContextKeys.CONNECTIONS_DONE] = True
            return "Done with current phase", updates

        # FSM actions
        if isinstance(self.builder, FSMBuilder):
            return self._execute_fsm_action(action, params, updates)

        # Workflow actions
        if isinstance(self.builder, WorkflowBuilder):
            return self._execute_workflow_action(action, params, updates)

        # Agent actions
        if isinstance(self.builder, AgentBuilder):
            return self._execute_agent_action(action, params, updates)

        return "Unknown builder type", updates

    def _execute_fsm_action(
        self,
        action: str,
        params: dict[str, Any],
        updates: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Execute FSM-specific actions."""
        builder: FSMBuilder = self.builder  # type: ignore[assignment]

        if action == Actions.ADD_STATE:
            warnings = builder.add_state(
                state_id=params.get("state_id", ""),
                description=params.get("description", ""),
                purpose=params.get("purpose", "New state"),
                extraction_instructions=params.get("extraction_instructions"),
                response_instructions=params.get("response_instructions"),
            )
            msg = f"Added state '{params.get('state_id')}'"
            if warnings:
                msg += f" (warnings: {'; '.join(warnings)})"
            return msg, updates

        if action == Actions.REMOVE_STATE:
            removed = builder.remove_state(params.get("state_id", ""))
            return (
                f"Removed state '{params.get('state_id')}'"
                if removed
                else f"State '{params.get('state_id')}' not found"
            ), updates

        if action == Actions.UPDATE_STATE:
            state_id = params.pop("state_id", "")
            warnings = builder.update_state(state_id, **params)
            msg = f"Updated state '{state_id}'"
            if warnings:
                msg += f" (warnings: {'; '.join(warnings)})"
            return msg, updates

        if action == Actions.ADD_TRANSITION:
            warnings = builder.add_transition(
                from_state=params.get("from_state", ""),
                target_state=params.get("target_state", ""),
                description=params.get("description", ""),
                priority=params.get("priority", 100),
                conditions=params.get("conditions"),
            )
            msg = f"Added transition '{params.get('from_state')}' -> '{params.get('target_state')}'"
            if warnings:
                msg += f" (warnings: {'; '.join(warnings)})"
            return msg, updates

        if action == Actions.REMOVE_TRANSITION:
            removed = builder.remove_transition(
                from_state=params.get("from_state", ""),
                target_state=params.get("target_state", ""),
            )
            return (
                "Removed transition" if removed else "Transition not found"
            ), updates

        if action == Actions.SET_INITIAL_STATE:
            warnings = builder.set_initial_state(params.get("state_id", ""))
            return f"Set initial state to '{params.get('state_id')}'", updates

        return f"Unknown FSM action: {action}", updates

    def _execute_workflow_action(
        self,
        action: str,
        params: dict[str, Any],
        updates: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Execute workflow-specific actions."""
        builder: WorkflowBuilder = self.builder  # type: ignore[assignment]

        if action == Actions.ADD_STEP:
            warnings = builder.add_step(
                step_id=params.get("step_id", ""),
                step_type=params.get("step_type", "auto_transition"),
                name=params.get("name", ""),
                description=params.get("description", ""),
                config=params.get("config"),
            )
            msg = f"Added step '{params.get('step_id')}'"
            if warnings:
                msg += f" (warnings: {'; '.join(warnings)})"
            return msg, updates

        if action == Actions.REMOVE_STEP:
            removed = builder.remove_step(params.get("step_id", ""))
            return (
                f"Removed step '{params.get('step_id')}'"
                if removed
                else f"Step '{params.get('step_id')}' not found"
            ), updates

        if action == Actions.SET_STEP_TRANSITION:
            warnings = builder.set_step_transition(
                from_step=params.get("from_step", ""),
                to_step=params.get("to_step", ""),
                condition=params.get("condition"),
            )
            msg = f"Connected '{params.get('from_step')}' -> '{params.get('to_step')}'"
            if warnings:
                msg += f" (warnings: {'; '.join(warnings)})"
            return msg, updates

        if action == Actions.SET_INITIAL_STEP:
            warnings = builder.set_initial_step(params.get("step_id", ""))
            return f"Set initial step to '{params.get('step_id')}'", updates

        return f"Unknown workflow action: {action}", updates

    def _execute_agent_action(
        self,
        action: str,
        params: dict[str, Any],
        updates: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Execute agent-specific actions."""
        builder: AgentBuilder = self.builder  # type: ignore[assignment]

        if action == Actions.SET_AGENT_TYPE:
            warnings = builder.set_agent_type(params.get("agent_type", ""))
            msg = f"Set agent type to '{params.get('agent_type')}'"
            if warnings:
                msg += f" (warnings: {'; '.join(warnings)})"
            return msg, updates

        if action == Actions.SET_CONFIG:
            warnings = builder.set_config(**params)
            msg = "Updated agent configuration"
            if warnings:
                msg += f" (warnings: {'; '.join(warnings)})"
            return msg, updates

        if action == Actions.ADD_TOOL:
            warnings = builder.add_tool(
                name=params.get("name", ""),
                description=params.get("description", ""),
                parameter_schema=params.get("parameter_schema"),
            )
            msg = f"Added tool '{params.get('name')}'"
            if warnings:
                msg += f" (warnings: {'; '.join(warnings)})"
            return msg, updates

        if action == Actions.REMOVE_TOOL:
            removed = builder.remove_tool(params.get("name", ""))
            return (
                f"Removed tool '{params.get('name')}'"
                if removed
                else f"Tool '{params.get('name')}' not found"
            ), updates

        return f"Unknown agent action: {action}", updates

    # ------------------------------------------------------------------
    # POST_TRANSITION: Finalize on entering output state
    # ------------------------------------------------------------------

    def finalize(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run final validation and serialize the artifact."""
        if self.builder is None:
            return {ContextKeys.VALIDATION_ERRORS: "Builder not initialized"}

        errors = self.builder.validate_complete()

        if errors:
            logger.warning(
                LogMessages.VALIDATION_RUN.format(
                    error_count=len(errors), warning_count=0
                )
            )
            return {
                ContextKeys.VALIDATION_ERRORS: "; ".join(errors),
            }

        artifact = self.builder.to_dict()
        artifact_json = json.dumps(artifact, indent=2)

        logger.info(
            LogMessages.BUILD_COMPLETE.format(
                artifact_type=self.builder.artifact_type.value,
                name=artifact.get("name", "unnamed"),
            )
        )

        return {
            ContextKeys.FINAL_ARTIFACT: artifact_json,
            ContextKeys.FINAL_JSON: artifact_json,
            ContextKeys.VALIDATION_ERRORS: None,
        }

    # ------------------------------------------------------------------
    # Validation for review state
    # ------------------------------------------------------------------

    def run_validation(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run validation and inject results into context."""
        if self.builder is None:
            return {}

        errors = self.builder.validate_complete()
        warnings = self.builder.validate_partial()

        logger.info(
            LogMessages.VALIDATION_RUN.format(
                error_count=len(errors), warning_count=len(warnings)
            )
        )

        return {
            ContextKeys.VALIDATION_ERRORS: "; ".join(errors) if errors else None,
            ContextKeys.VALIDATION_WARNINGS: "; ".join(warnings) if warnings else None,
        }
