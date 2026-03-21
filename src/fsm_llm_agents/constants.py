from __future__ import annotations

"""
Constants for the agents package.
"""

from fsm_llm.constants import DEFAULT_LLM_MODEL


class AgentStates:
    """States in the ReAct agent FSM."""

    THINK = "think"
    ACT = "act"
    CONCLUDE = "conclude"
    AWAIT_APPROVAL = "await_approval"


class ContextKeys:
    """Standard context keys used across the agents package."""

    # Task
    TASK = "task"
    TASK_CONTEXT = "task_context"

    # Tool selection (extracted by LLM in think state)
    TOOL_NAME = "tool_name"
    TOOL_INPUT = "tool_input"
    REASONING = "reasoning"
    SHOULD_TERMINATE = "should_terminate"

    # Tool execution results
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    TOOL_STATUS = "tool_status"

    # Observations accumulated across iterations
    OBSERVATIONS = "observations"
    OBSERVATION_COUNT = "observation_count"

    # Final answer
    FINAL_ANSWER = "final_answer"

    # Budget tracking
    ITERATION_COUNT = "iteration_count"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"

    # HITL
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_ACTION = "approval_action"

    # Agent trace
    AGENT_TRACE = "agent_trace"


class HandlerNames:
    """Handler names for registration."""

    TOOL_EXECUTOR = "AgentToolExecutor"
    ITERATION_LIMITER = "AgentIterationLimiter"
    OBSERVATION_TRACKER = "AgentObservationTracker"
    HITL_GATE = "AgentHITLGate"


class Defaults:
    """Default configuration values."""

    MODEL = DEFAULT_LLM_MODEL
    TEMPERATURE = 0.5
    MAX_TOKENS = 1000
    MAX_ITERATIONS = 10
    TIMEOUT_SECONDS = 300.0
    MAX_OBSERVATION_LENGTH = 2000
    MAX_OBSERVATIONS = 20
    CONFIDENCE_THRESHOLD = 0.3


class ErrorMessages:
    """Standard error messages."""

    BUDGET_EXHAUSTED = "Agent exceeded maximum iterations ({limit})"
    TOOL_NOT_FOUND = "Tool '{name}' not found in registry"
    TOOL_EXECUTION_FAILED = "Tool '{name}' execution failed: {error}"
    APPROVAL_DENIED = "Human denied approval for: {action}"
    TIMEOUT = "Agent timed out after {seconds:.1f}s"
    NO_TOOLS = "Cannot create agent with empty tool registry"


class LogMessages:
    """Standard log message templates."""

    AGENT_STARTED = "Agent started with {tool_count} tools, model={model}"
    TOOL_SELECTED = "Selected tool: {name} with input: {input}"
    TOOL_EXECUTED = "Tool '{name}' executed successfully"
    TOOL_FAILED = "Tool '{name}' failed: {error}"
    ITERATION = "Iteration {current}/{max}"
    AGENT_COMPLETE = "Agent completed in {iterations} iterations"
    APPROVAL_REQUESTED = "Requesting approval for: {action}"
    APPROVAL_RESULT = "Approval {result} for: {action}"
    ESCALATION = "Escalating to human: {reason}"
