from __future__ import annotations

"""
Meta-agent handlers for builder injection, action dispatch,
progress tracking, and finalization.
"""

import json
from typing import Any, ClassVar

from fsm_llm.logging import logger
from fsm_llm_classification import ClassificationSchema, IntentDefinition

from .builders import AgentBuilder, ArtifactBuilder, FSMBuilder, WorkflowBuilder
from .constants import Actions, ContextKeys, Defaults, LogMessages, MetaStates
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
        "sounds good",
        "fine",
        "done",
        "correct",
        "right",
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
        "wrong",
        "incorrect",
        "not right",
        "needs work",
        "not quite",
        "try again",
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

    # Normalize common LLM variants of artifact type
    _TYPE_ALIASES: ClassVar[dict[str, str]] = {
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

    # Classification schema for artifact type validation.
    # Uses fsm_llm_classification to define valid types with descriptions.
    _ARTIFACT_SCHEMA: ClassVar[ClassificationSchema] = ClassificationSchema(
        intents=[
            IntentDefinition(
                name="fsm",
                description=(
                    "Finite State Machine for stateful conversations, chatbots, "
                    "or dialogue flows with states and transitions"
                ),
            ),
            IntentDefinition(
                name="workflow",
                description=(
                    "Workflow for multi-step async processes, pipelines, "
                    "automation, or sequential step-based operations"
                ),
            ),
            IntentDefinition(
                name="agent",
                description=(
                    "AI Agent with tools, ReAct loops, planning, "
                    "or autonomous task execution capabilities"
                ),
            ),
        ],
        fallback_intent="fsm",
        confidence_threshold=0.7,
    )

    def __init__(self) -> None:
        self.builder: ArtifactBuilder | None = None
        self._artifact_type: ArtifactType | None = None

    def reset(self) -> None:
        """Reset handler state for a new session."""
        self.builder = None
        self._artifact_type = None

    # ------------------------------------------------------------------
    # Builder initialization
    # ------------------------------------------------------------------

    def _ensure_builder(self, artifact_type_str: str) -> None:
        """Initialize the appropriate builder from artifact type string.

        Raises BuilderError if the artifact type cannot be resolved.
        """
        if self.builder is not None:
            return

        # Normalize common LLM variants
        normalized = artifact_type_str.strip().lower()
        normalized = self._TYPE_ALIASES.get(normalized, normalized)

        try:
            artifact_type = ArtifactType(normalized)
        except ValueError:
            logger.warning(
                f"Unknown artifact type '{artifact_type_str}', "
                f"could not resolve to fsm/workflow/agent"
            )
            raise BuilderError(
                f"Unknown artifact type: '{artifact_type_str}'. "
                f"Must be one of: fsm, workflow, agent",
                action="_ensure_builder",
            ) from None

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

    def classify_artifact_type(self, context: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize the extracted artifact_type using the
        classification schema.

        Runs at POST_PROCESSING on the classify state. If the extracted
        artifact_type is not valid, clears it so the transition condition
        does not fire and the FSM re-asks.
        """
        raw_type = context.get(ContextKeys.ARTIFACT_TYPE)
        if not raw_type or not isinstance(raw_type, str):
            return {}

        normalized = raw_type.strip().lower()
        normalized = self._TYPE_ALIASES.get(normalized, normalized)

        # Validate against the classification schema
        valid_names = self._ARTIFACT_SCHEMA.intent_names
        if normalized in valid_names:
            if normalized != raw_type:
                logger.debug(f"Normalized artifact_type '{raw_type}' → '{normalized}'")
            return {ContextKeys.ARTIFACT_TYPE: normalized}

        # Not a valid type — clear it so the FSM re-asks
        logger.warning(
            f"Extracted artifact_type '{raw_type}' is not valid "
            f"(expected one of: {', '.join(valid_names)}); clearing"
        )
        return {ContextKeys.ARTIFACT_TYPE: None}

    # ------------------------------------------------------------------
    # PRE_PROCESSING: Inject builder state into context + dynamic prompts
    # ------------------------------------------------------------------

    def inject_builder_state(self, context: dict[str, Any]) -> dict[str, Any]:
        """Inject builder summary and progress into context.

        Selects the appropriate summary detail level based on the current
        meta-agent state (``_current_state`` in context) to keep context
        compact during design phases and detailed during review.
        """
        # Initialize builder when artifact type becomes available
        artifact_type = context.get(ContextKeys.ARTIFACT_TYPE)
        if artifact_type and self.builder is None:
            try:
                self._ensure_builder(artifact_type)
            except BuilderError:
                # Cannot classify — leave builder as None, let FSM re-ask
                return {}

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
        """Process overview fields extracted in gather_overview state.

        Tolerates small-model extraction that uses generic keys like
        ``name`` instead of ``artifact_name``.
        """
        if self.builder is None:
            return {}

        updates: dict[str, Any] = {}
        # Try canonical keys first, then common LLM fallbacks
        name = (
            context.get(ContextKeys.ARTIFACT_NAME)
            or context.get("name")
            or context.get("fsm_name")
            or context.get("workflow_name")
            or context.get("agent_name")
        )
        desc = (
            context.get(ContextKeys.ARTIFACT_DESCRIPTION)
            or context.get("description")
            or context.get("fsm_description")
        )
        persona = context.get(ContextKeys.ARTIFACT_PERSONA) or context.get("persona")

        # Propagate resolved values back to canonical keys so transition
        # conditions (has_context artifact_name) can fire.
        if name and not context.get(ContextKeys.ARTIFACT_NAME):
            updates[ContextKeys.ARTIFACT_NAME] = name
        if desc and not context.get(ContextKeys.ARTIFACT_DESCRIPTION):
            updates[ContextKeys.ARTIFACT_DESCRIPTION] = desc

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
        """Normalize common LLM variants of user_decision to canonical values.

        Also checks fallback keys like ``decision`` and ``approval`` that
        small models may use instead of ``user_decision``.
        """
        decision = (
            context.get(ContextKeys.USER_DECISION)
            or context.get("decision")
            or context.get("approval")
            or context.get("review_decision")
        )
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
    # POST_PROCESSING: Infer action from flat-extracted context keys
    # ------------------------------------------------------------------

    def infer_action(self, context: dict[str, Any]) -> dict[str, Any]:
        """Detect flat-extracted keys from small models and wrap them into
        the ``action`` / ``action_params`` format the dispatch handler expects.

        Small LLMs often extract ``{"state_id": "greeting", "description": ...}``
        instead of ``{"action": "add_state", "action_params": {...}}``.  This
        handler bridges the gap by recognizing common key patterns.
        """
        # Skip if action is already properly set
        if context.get(ContextKeys.ACTION):
            return {}

        updates: dict[str, Any] = {}

        # --- FSM state patterns ---
        # Check for nested state_params dict first (small model pattern)
        for nested_key in ("state_params", "state"):
            nested = context.get(nested_key)
            if isinstance(nested, dict) and "state_id" not in context:
                for k, v in nested.items():
                    if k not in context or context[k] is None:
                        context[k] = v

        state_id = context.get("state_id") or context.get("state_name")
        if state_id and isinstance(state_id, str):
            updates[ContextKeys.ACTION] = Actions.ADD_STATE
            updates[ContextKeys.ACTION_PARAMS] = {
                "state_id": state_id,
                "description": (
                    context.get("state_description") or context.get("description") or ""
                ),
                "purpose": (
                    context.get("state_purpose")
                    or context.get("purpose")
                    or "New state"
                ),
            }
            ei = context.get("extraction_instructions")
            if ei:
                updates[ContextKeys.ACTION_PARAMS]["extraction_instructions"] = ei
            ri = context.get("response_instructions")
            if ri:
                updates[ContextKeys.ACTION_PARAMS]["response_instructions"] = ri
            logger.debug(f"Inferred add_state action for state '{state_id}'")
            return updates

        # --- FSM transition patterns ---
        # Check flat keys first, then nested dicts small models produce
        from_state = context.get("from_state") or context.get("source_state")
        target_state = (
            context.get("target_state")
            or context.get("to_state")
            or context.get("destination_state")
        )
        trans_desc = (
            context.get("transition_description") or context.get("description") or ""
        )
        # Small models may nest transition data in a params dict
        for nested_key in ("transition_params", "transition", "params"):
            nested = context.get(nested_key)
            if isinstance(nested, dict) and not from_state:
                from_state = (
                    nested.get("from_state")
                    or nested.get("from")
                    or nested.get("source_state")
                    or nested.get("source")
                )
                target_state = (
                    nested.get("target_state")
                    or nested.get("to_state")
                    or nested.get("to")
                    or nested.get("target")
                )
                trans_desc = (
                    nested.get("description")
                    or nested.get("transition_description")
                    or trans_desc
                )
        if from_state and target_state:
            updates[ContextKeys.ACTION] = Actions.ADD_TRANSITION
            updates[ContextKeys.ACTION_PARAMS] = {
                "from_state": from_state,
                "target_state": target_state,
                "description": trans_desc,
            }
            logger.debug(
                f"Inferred add_transition action: '{from_state}' → '{target_state}'"
            )
            return updates

        # --- Workflow step patterns ---
        step_id = context.get("step_id") or context.get("step_name")
        if step_id and isinstance(step_id, str):
            updates[ContextKeys.ACTION] = Actions.ADD_STEP
            updates[ContextKeys.ACTION_PARAMS] = {
                "step_id": step_id,
                "step_type": context.get("step_type", "auto_transition"),
                "name": context.get("name") or step_id,
                "description": context.get("description") or "",
            }
            logger.debug(f"Inferred add_step action for step '{step_id}'")
            return updates

        # --- Workflow transition patterns ---
        from_step = context.get("from_step")
        to_step = context.get("to_step")
        if from_step and to_step:
            updates[ContextKeys.ACTION] = Actions.SET_STEP_TRANSITION
            updates[ContextKeys.ACTION_PARAMS] = {
                "from_step": from_step,
                "to_step": to_step,
                "condition": context.get("condition"),
            }
            logger.debug(f"Inferred set_step_transition: '{from_step}' → '{to_step}'")
            return updates

        # --- Agent tool patterns ---
        tool_name = context.get("tool_name")
        if tool_name and isinstance(tool_name, str):
            updates[ContextKeys.ACTION] = Actions.ADD_TOOL
            updates[ContextKeys.ACTION_PARAMS] = {
                "name": tool_name,
                "description": context.get("tool_description")
                or context.get("description")
                or "",
            }
            logger.debug(f"Inferred add_tool action for tool '{tool_name}'")
            return updates

        # --- Agent type pattern ---
        agent_type = context.get("agent_type")
        if (
            agent_type
            and isinstance(agent_type, str)
            and not context.get(ContextKeys.ARTIFACT_TYPE)
        ):
            updates[ContextKeys.ACTION] = Actions.SET_AGENT_TYPE
            updates[ContextKeys.ACTION_PARAMS] = {"agent_type": agent_type}
            logger.debug(f"Inferred set_agent_type action: '{agent_type}'")
            return updates

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
            logger.warning(
                f"Malformed action_params (type={type(params).__name__}); "
                f"using empty dict"
            )
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
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                f"Error dispatching action '{action}': {type(e).__name__}: {e}"
            )
            return {
                ContextKeys.ACTION_ERRORS: f"Error: {e}",
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
            state_id = params.get("state_id", "")
            # Build fields dict without state_id
            fields = {k: v for k, v in params.items() if k != "state_id"}
            warnings = builder.update_state(state_id, **fields)
            msg = f"Updated state '{state_id}'"
            if warnings:
                msg += f" (warnings: {'; '.join(warnings)})"
            return msg, updates

        if action == Actions.ADD_TRANSITION:
            warnings = builder.add_transition(
                from_state=params.get("from_state", ""),
                target_state=params.get("target_state", ""),
                description=params.get("description", ""),
                priority=params.get("priority", Defaults.DEFAULT_PRIORITY),
                conditions=params.get("conditions"),
            )
            msg = (
                f"Added transition '{params.get('from_state')}' "
                f"-> '{params.get('target_state')}'"
            )
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
