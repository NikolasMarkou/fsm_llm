"""The forced-stop vocabulary of the ReAct-family handlers (plan d454fd62 step 88): the delta of a turn
after a forced stop (``forced_stop_skip``), the recorded forced reason (``recorded_forced_reason``), and
the two readers that tell a forced verdict from the model's own conclusion (``concluded_on_evidence``,
``is_forced_verdict``). The function bodies are the old ``agents/handlers.py`` ones."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fsm_llm.agents.constants import (
    FORCED_STOP_REASONS,
    AgentStates,
    ContextKeys,
    ReflexionStates,
    StopReason,
    ToolRunStatus,
)
from fsm_llm.logging import logger


def forced_stop_skip(context: Mapping[str, Any]) -> dict[str, Any] | None:
    """The executor delta for a turn after a forced stop, else None.

    Shared by every ReAct-family tool executor (``AgentHandlers``,
    ReasoningReact's ``reason`` path, ParallelReact's batch): once
    ``max_iterations_reached is True`` (limiter, stall detector; seeded False,
    framework-only) no tool runs. The delta clears the selection and keeps
    ``should_terminate`` True, so the act -> conclude evidence guard routes
    the run to its answer. Never raises.
    """
    if context.get(ContextKeys.MAX_ITERATIONS_REACHED) is not True:
        return None
    logger.info("Iteration budget reached: the selected tool is not run")
    return {
        ContextKeys.TOOL_RESULT: "Iteration budget reached; no tool was run.",
        ContextKeys.TOOL_STATUS: ToolRunStatus.SKIPPED,
        ContextKeys.TOOL_NAME: None,
        ContextKeys.TOOL_INPUT: None,
        ContextKeys.SHOULD_TERMINATE: True,
    }


def recorded_forced_reason(context: Mapping[str, Any]) -> StopReason | None:
    """The ``FORCED_STOP_REASONS`` member a forcing handler recorded, else None.

    Interface contract (callers: :func:`is_forced_verdict`,
    ``AgentHandlers.check_iteration_limit``, ``BaseAgent._forced_stop_reason``,
    the Debate limiter): reads ``forced_stop_reason`` and returns it only when
    it is a string in ``FORCED_STOP_REASONS`` (returned as the member). Never
    raises.
    """
    reason = context.get(ContextKeys.FORCED_STOP_REASON)
    if isinstance(reason, str) and reason in FORCED_STOP_REASONS:
        return StopReason(reason)
    return None


# The verdict a ReAct-family turn concludes on, per state it leaves.
_CONCLUDE_VERDICTS: dict[str, str] = {
    AgentStates.THINK: ContextKeys.SHOULD_TERMINATE,
    ReflexionStates.EVALUATE: ContextKeys.EVALUATION_PASSED,
}


def concluded_on_evidence(leaving: str, context: Mapping[str, Any]) -> bool:
    """True when the turn leaving ``leaving`` concludes on its own verdict.

    Interface contract (caller: ``AgentHandlers.check_iteration_limit``):
    ``leaving`` is ``think`` with ``should_terminate is True`` or Reflexion's
    ``evaluate`` with ``evaluation_passed is True``, and ``observation_count``
    is above 0: the first disjunct of the conclude edge
    (``fsm_definitions.conclude_on_evidence_logic``), so this turn takes
    that edge. Every other state gives False. Never raises.
    """
    verdict = _CONCLUDE_VERDICTS.get(leaving)
    if verdict is None or context.get(verdict) is not True:
        return False
    count = context.get(ContextKeys.OBSERVATION_COUNT)
    return isinstance(count, int) and count > 0


def is_forced_verdict(value: Any, context: dict[str, Any]) -> bool:
    """True when ``value`` is a verdict a framework handler forced to True.

    "Forced" is read only from framework-written flags, never from the
    verdict key itself: ``value is True`` AND either ``max_iterations_reached
    is True`` (limiters, stall detector; seeded False by ``_init_context``) or
    a recorded ``forced_stop_reason`` (:func:`recorded_forced_reason`). Both
    flags are in ``RUN_OUTPUT_KEYS`` (caller context cannot seed them) and in
    ``FRAMEWORK_ONLY_KEYS`` (every agent FSM lists them as core
    ``handler_only_keys``, so no extraction writes them). Never raises.
    """
    if value is not True:
        return False
    if context.get(ContextKeys.MAX_ITERATIONS_REACHED) is True:
        return True
    return recorded_forced_reason(context) is not None
