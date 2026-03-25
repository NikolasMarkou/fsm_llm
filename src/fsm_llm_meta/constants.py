from __future__ import annotations

"""
Constants for the meta-agent package.
"""

from fsm_llm.constants import DEFAULT_LLM_MODEL

# ---------------------------------------------------------------------------
# Meta-agent FSM states
# ---------------------------------------------------------------------------


class MetaStates:
    """States in the meta-agent conversational FSM."""

    WELCOME = "welcome"
    CLASSIFY = "classify"
    GATHER_OVERVIEW = "gather_overview"
    DESIGN_STRUCTURE = "design_structure"
    DEFINE_CONNECTIONS = "define_connections"
    REVIEW = "review"
    OUTPUT = "output"


# ---------------------------------------------------------------------------
# Context keys
# ---------------------------------------------------------------------------


class ContextKeys:
    """Standard context keys used by the meta-agent."""

    # Artifact classification
    ARTIFACT_TYPE = "artifact_type"

    # Overview fields
    ARTIFACT_NAME = "artifact_name"
    ARTIFACT_DESCRIPTION = "artifact_description"
    ARTIFACT_PERSONA = "artifact_persona"

    # Builder state injection
    BUILDER_SUMMARY = "builder_summary"
    BUILDER_PROGRESS = "builder_progress"
    BUILDER_MISSING = "builder_missing"

    # Action dispatch
    ACTION = "action"
    ACTION_PARAMS = "action_params"
    ACTION_RESULT = "action_result"
    ACTION_ERRORS = "action_errors"

    # Review
    VALIDATION_ERRORS = "validation_errors"
    VALIDATION_WARNINGS = "validation_warnings"
    USER_DECISION = "user_decision"

    # Structure done flag
    STRUCTURE_DONE = "structure_done"
    CONNECTIONS_DONE = "connections_done"

    # Output
    FINAL_ARTIFACT = "final_artifact"
    FINAL_JSON = "final_json"


# ---------------------------------------------------------------------------
# Builder actions
# ---------------------------------------------------------------------------


class Actions:
    """Action names the LLM can extract to mutate the builder."""

    # FSM actions
    SET_OVERVIEW = "set_overview"
    ADD_STATE = "add_state"
    REMOVE_STATE = "remove_state"
    UPDATE_STATE = "update_state"
    ADD_TRANSITION = "add_transition"
    REMOVE_TRANSITION = "remove_transition"
    SET_INITIAL_STATE = "set_initial_state"
    DONE = "done"

    # Workflow actions
    ADD_STEP = "add_step"
    REMOVE_STEP = "remove_step"
    SET_STEP_TRANSITION = "set_step_transition"
    SET_INITIAL_STEP = "set_initial_step"

    # Agent actions
    SET_AGENT_TYPE = "set_agent_type"
    SET_CONFIG = "set_config"
    ADD_TOOL = "add_tool"
    REMOVE_TOOL = "remove_tool"


# ---------------------------------------------------------------------------
# Handler names
# ---------------------------------------------------------------------------


class HandlerNames:
    """Handler names for registration."""

    BUILDER_INJECTOR = "MetaBuilderInjector"
    ACTION_DISPATCHER = "MetaActionDispatcher"
    PROGRESS_TRACKER = "MetaProgressTracker"
    FINALIZER = "MetaFinalizer"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class Defaults:
    """Default configuration values."""

    MODEL = DEFAULT_LLM_MODEL
    TEMPERATURE = 0.7
    MAX_TOKENS = 2000
    MAX_TURNS = 50


# ---------------------------------------------------------------------------
# Error and log messages
# ---------------------------------------------------------------------------


class ErrorMessages:
    """Standard error messages."""

    UNKNOWN_ACTION = "Unknown action: '{action}'"
    BUILDER_NOT_INITIALIZED = "Builder has not been initialized yet"
    CONVERSATION_NOT_STARTED = "Conversation has not been started"
    CONVERSATION_ALREADY_STARTED = "Conversation has already been started"
    INVALID_ARTIFACT_TYPE = "Invalid artifact type: '{artifact_type}'"
    STATE_NOT_FOUND = "State '{state_id}' not found in builder"
    STEP_NOT_FOUND = "Step '{step_id}' not found in builder"
    TOOL_NOT_FOUND = "Tool '{tool_name}' not found in builder"


class LogMessages:
    """Standard log message templates."""

    META_STARTED = "Meta-agent started with model={model}"
    ARTIFACT_CLASSIFIED = "Artifact type classified as: {artifact_type}"
    ACTION_DISPATCHED = "Dispatching action: {action}"
    ACTION_SUCCEEDED = "Action '{action}' completed successfully"
    ACTION_FAILED = "Action '{action}' failed: {error}"
    VALIDATION_RUN = (
        "Running validation: {error_count} errors, {warning_count} warnings"
    )
    BUILD_COMPLETE = "Build complete: {artifact_type} '{name}'"
