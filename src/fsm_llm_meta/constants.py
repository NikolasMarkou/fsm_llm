from __future__ import annotations

"""
Constants for the meta-agent package.
"""

from fsm_llm.constants import DEFAULT_LLM_MODEL

# ---------------------------------------------------------------------------
# Meta-agent phases
# ---------------------------------------------------------------------------


class MetaPhases:
    """Phases of the meta-agent conversation."""

    INTAKE = "intake"
    BUILD = "build"
    REVIEW = "review"
    DONE = "done"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class Defaults:
    """Default configuration values."""

    MODEL = DEFAULT_LLM_MODEL
    TEMPERATURE = 0.7
    MAX_TOKENS = 2000
    MAX_TURNS = 50

    # Builder defaults
    DEFAULT_PRIORITY = 100
    SUMMARY_TRUNCATE_WIDTH = 80

    # ReactAgent configuration for the build phase
    BUILD_MAX_ITERATIONS = 25
    BUILD_TIMEOUT_SECONDS = 120.0
    BUILD_TEMPERATURE = 0.3
    BUILD_MAX_TOKENS = 1000

    # Agent builder defaults (for the agent artifact being built)
    AGENT_MODEL = "gpt-4o-mini"
    AGENT_MAX_ITERATIONS = 10
    AGENT_TIMEOUT_SECONDS = 300.0
    AGENT_TEMPERATURE = 0.5
    AGENT_MAX_TOKENS = 1000


# ---------------------------------------------------------------------------
# Approval/revision detection
# ---------------------------------------------------------------------------


class DecisionWords:
    """Word sets for detecting user approval or revision intent."""

    APPROVE: frozenset[str] = frozenset(
        {
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
    )

    REVISE: frozenset[str] = frozenset(
        {
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
    )


# ---------------------------------------------------------------------------
# Error and log messages
# ---------------------------------------------------------------------------


class ErrorMessages:
    """Standard error messages."""

    BUILDER_NOT_INITIALIZED = "Builder has not been initialized yet"
    CONVERSATION_NOT_STARTED = "Conversation has not been started"
    CONVERSATION_ALREADY_STARTED = "Conversation has already been started"
    INVALID_ARTIFACT_TYPE = "Invalid artifact type: '{artifact_type}'"


class LogMessages:
    """Standard log message templates."""

    META_STARTED = "Meta-agent started with model={model}"
    ARTIFACT_CLASSIFIED = "Artifact type classified as: {artifact_type}"
    BUILD_STARTED = "Build phase started for {artifact_type}"
    BUILD_COMPLETE = "Build complete: {artifact_type} '{name}'"
    REVIEW_STARTED = "Review phase: {error_count} errors, {warning_count} warnings"
    REVISION_STARTED = "Revision requested: {revision}"
