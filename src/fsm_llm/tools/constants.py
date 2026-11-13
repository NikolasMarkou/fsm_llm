"""Constants of the tool layer: the subset of the agents context keys, defaults and
message templates that the registry and the tool models read, and the tool statuses.

The agents classes of the same names (``fsm_llm.agents.constants``) assign every
attribute below from these classes, in their own position, so each value has one
source and ``vars()`` of the agents classes lists the same names as before.
"""

from __future__ import annotations

import threading


class ContextKeys:
    """The context keys a tool call is read from and the sentinel for no tool."""

    # Tool selection (extracted by LLM in think state)
    TOOL_NAME = "tool_name"
    TOOL_INPUT = "tool_input"

    # Sentinel value for "no tool selected"
    NO_TOOL = "none"


class Defaults:
    """Default configuration values of the tool layer."""

    # Largest integer an agent limit (``max_*``/``num_*``) or an artifact int
    # config field may take: 63 bits. A larger int is no real limit, and past
    # 4300 digits it breaks every f-string, log line and ``json.dumps`` of it.
    MAX_INT_LIMIT = 2**63 - 1
    MAX_OBSERVATION_LENGTH = 2000
    # Largest per-call tool ``timeout_seconds``: the most a thread wait accepts
    # (``threading.TIMEOUT_MAX``; a larger wait raises OverflowError).
    MAX_TOOL_TIMEOUT_S = threading.TIMEOUT_MAX
    # Tool-selection classification threshold (``to_classification_schema``).
    TOOL_CONFIDENCE_THRESHOLD = 0.4


class ErrorMessages:
    """Error messages of the tool layer."""

    TOOL_NOT_FOUND = "Tool '{name}' not found in registry"
    TOOL_EXECUTION_FAILED = "Tool '{name}' execution failed: {error}"
    TOOL_TIMED_OUT = (
        "Tool '{name}' timed out after {timeout} s; its outcome is unknown: it "
        "may still be running and its side effects may still happen. Do not "
        "assume it did not run"
    )
    TOOL_BAD_TIMEOUT = (
        "Tool '{name}' has an invalid timeout_seconds {timeout}: it must be a "
        "finite number above 0 and at most {limit} s; the tool was not run"
    )


class LogMessages:
    """Log message templates of the tool layer."""

    TOOL_LATE_RESULT = (
        "Tool '{name}' finished after its {timeout} s timeout; "
        "its {outcome} was discarded"
    )


class ToolObservationPrefix:
    """Prefixes that tell the model how a tool call ended (``ToolResult.observation``)."""

    # The call ran and failed (or never ran: unknown tool, bad arguments).
    FAILED = "[TOOL FAILED]"
    # The call timed out: it may still be running, its effects may happen.
    OUTCOME_UNKNOWN = "[TOOL OUTCOME UNKNOWN]"


class ToolRunStatus:
    """How a call that was sent to a tool ended (``ToolResult.status``).

    Written to ``ContextKeys.TOOL_STATUS`` by the executors that run a tool;
    the other statuses there (``skipped``, ``rejected``,
    ``awaiting_approval``) mean no tool ran.
    """

    SUCCESS = "success"
    FAILED = "failed"
    # The call timed out: it may still be running and may still take effect.
    # Distinct from FAILED so no reader treats it as "safe to run again".
    UNKNOWN = "unknown"
    # The three values below mean NO tool ran for the turn.
    SKIPPED = "skipped"
    REJECTED = "rejected"
    AWAITING_APPROVAL = "awaiting_approval"
    # Statuses that mean a tool really ran.
    RAN = frozenset({SUCCESS, FAILED, UNKNOWN})
