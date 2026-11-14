"""The one writer of a tool step for every ReAct-family executor (plan d454fd62 step 88):
``record_tool_step`` with ``next_step_number`` and ``with_feedback``. Used by ``AgentHandlers``,
ParallelReact's batch and ReasoningReact's ``reason`` path. The function bodies are the old
``agents/handlers.py`` ones."""

from __future__ import annotations

from typing import Any

from fsm_llm.agents.constants import OBSERVATION_ENTRY, ContextKeys, Defaults
from fsm_llm.agents.definitions import AgentStep
from fsm_llm.logging import logger
from fsm_llm.tools import smart_truncate


def next_step_number(trace: list[Any]) -> int:
    """Step number for the next tool observation: one past the trace length.

    Shared by the ReAct-family executors (``AgentHandlers``, ParallelReact,
    ReasoningReact's ``reason`` path). ``agent_trace`` is never pruned, so the
    number keeps rising after ``observations`` is capped at
    ``MAX_OBSERVATIONS`` (LOOP-16: counting observations repeated numbers).
    Never raises.
    """
    return len(trace) + 1


def with_feedback(message: str, delta: dict[str, Any]) -> dict[str, Any]:
    """An executor delta whose message also reaches the next think turn.

    ``tool_result`` is transient (the compactor deletes it at the next turn's
    PRE_PROCESSING, before think extracts), so the message is copied to
    ``agent_feedback``, which think's prompts list and whose think-exit
    handler clears it once read (D-029).
    """
    return {
        ContextKeys.TOOL_RESULT: message,
        ContextKeys.AGENT_FEEDBACK: message,
        **delta,
    }


def record_tool_step(
    context: dict[str, Any],
    *,
    tool_name: str,
    shown_input: Any,
    observation: str,
    status: str,
    thought: str,
    action: str,
) -> dict[str, Any]:
    """Record one tool step; return the delta that carries it.

    The one writer every ReAct-family executor uses (``AgentHandlers``,
    ParallelReact's batch, ReasoningReact's ``reason`` path).

    Interface contract:
        - ``context``: read for ``observations`` / ``agent_trace`` (a missing or
          non-list value is empty); the new trace step is appended to that list
          in place. For a batch, pass ``{**context, **previous_delta}``.
        - ``shown_input``: the REDACTED input (D-016 of plan 06a5ec0a), kept in
          the entry and as the trace step's ``tool_input``, which
          ``BaseAgent._build_trace`` turns back into ``ToolCall.parameters``.
        - ``observation``: the model-facing text; ``status``: a ``ToolRunStatus``.
        - Returns ``observations`` (the new entry cut to
          ``Defaults.MAX_OBSERVATION_LENGTH``, the list pruned to the last
          ``Defaults.MAX_OBSERVATIONS``), ``observation_count`` and
          ``agent_trace``. Never raises.
    """
    observations = context.get(ContextKeys.OBSERVATIONS)
    trace = context.get(ContextKeys.AGENT_TRACE)
    observations = observations if isinstance(observations, list) else []
    trace = trace if isinstance(trace, list) else []
    step_num = next_step_number(trace)
    entry = OBSERVATION_ENTRY.format(
        step=step_num, tool=tool_name, input=shown_input, result=observation
    )
    observations.append(smart_truncate(entry, Defaults.MAX_OBSERVATION_LENGTH))
    if len(observations) > Defaults.MAX_OBSERVATIONS:
        logger.debug(f"Pruning observations to the last {Defaults.MAX_OBSERVATIONS}")
        observations = observations[-Defaults.MAX_OBSERVATIONS :]
    step = AgentStep(
        iteration=step_num, thought=thought, action=action, observation=observation
    ).model_dump(mode="json")
    trace.append({**step, "tool_input": shown_input, ContextKeys.TOOL_STATUS: status})
    return {
        ContextKeys.OBSERVATIONS: observations,
        ContextKeys.OBSERVATION_COUNT: len(observations),
        ContextKeys.AGENT_TRACE: trace,
    }
