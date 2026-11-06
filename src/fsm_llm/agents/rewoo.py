"""
REWOOAgent — Reasoning WithOut Observation agent implementation.

Uses exactly 2 LLM calls: plan all tool calls upfront (#E1, #E2 refs),
execute them sequentially (no LLM), then synthesize from evidence.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.definitions import CONTEXT_KEY_CUT_OFF
from fsm_llm.logging import logger

from .base import BaseAgent, caller_prompt_keys
from .constants import (
    ContextKeys,
    ErrorMessages,
    HandlerNames,
    HandlerPriorities,
    LogMessages,
    REWOOStates,
    StopReason,
)
from .definitions import AgentConfig, AgentResult, ToolCall
from .fsm_definitions import build_rewoo_fsm
from .handlers import make_iteration_limiter
from .tools import ToolRegistry, redact_secret_entries

# A plan id the model wrote as 1, "1", "E1" or "#E1" (any case).
_PLAN_ID = re.compile(r"#?E?(\d+)", re.IGNORECASE)


def _evidence_id(raw: Any, position: int) -> str:
    """The ``E<n>`` evidence key of a plan step.

    ``raw`` is the step's ``plan_id``: an integer (or integral float) or a
    string ``"1"``/``"E1"``/``"#E1"`` gives ``E<n>``; anything else (missing,
    bool, unparsable) gives ``E<position>``, the step's 1-based position among
    the plan's dict steps, with a WARNING when a value was present. Never raises.
    """
    if isinstance(raw, int) and not isinstance(raw, bool):
        return f"E{raw}"
    if isinstance(raw, float) and raw.is_integer():
        return f"E{int(raw)}"
    if isinstance(raw, str):
        match = _PLAN_ID.fullmatch(raw.strip())
        if match:
            return f"E{int(match.group(1))}"
    if raw is not None:
        logger.warning(f"REWOO plan_id {raw!r} is not a step number; using E{position}")
    return f"E{position}"


class REWOOAgent(BaseAgent):
    """
    REWOO agent that plans all tool calls upfront then executes them.

    Makes exactly 2 LLM calls: one to plan all tool calls with #E1/#E2
    variable references, one to synthesize the final answer from evidence.

    Usage::

        agent = REWOOAgent(tools=registry)
        result = agent.run("What is the population of France times 2?")
    """

    # Run outputs caller context may not seed (D-052 of plan 06a5ec0a).
    _run_output_keys = frozenset(
        {
            ContextKeys.PLAN_BLUEPRINT,
            ContextKeys.EVIDENCE,
            ContextKeys.EVIDENCE_STATUS,
            ContextKeys.PLAN_CUT_OFF,
        }
    )

    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        **api_kwargs: Any,
    ) -> None:
        self._require_tools(tools)

        super().__init__(config, **api_kwargs)
        self.tools = tools
        self._refuse_flagged_tools()

        logger.info(
            LogMessages.AGENT_STARTED.format(
                tool_count=len(tools), model=self.config.model
            )
        )

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Run the agent on a task.

        :param task: The task/question for the agent to solve
        :param initial_context: Optional initial context data
        :return: AgentResult with answer, trace, and metadata
        """
        self._refuse_flagged_tools()
        fsm_def = build_rewoo_fsm(
            self.tools,
            task_description=task,
            context_keys=caller_prompt_keys(initial_context, self._run_output_keys),
        )

        context = self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.EVIDENCE: {},
                ContextKeys.EVIDENCE_STATUS: [],
            },
        )

        # DECISION plan_2026-05-31_cb91a9d5/D-001 [STALE]: require non-empty tool
        # evidence — the unconditional plan_all->execute_plans transition lets
        # solve() emit a final_answer from EMPTY evidence (zero tools run) when
        # the 4b model fails to produce a valid plan_blueprint.
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-041: success reads
        # EVIDENCE_STATUS (a per-step success flag), NOT the `evidence` mapping.
        # Do NOT go back to `evidence`: failed calls store their error text
        # there, so a run whose every tool failed read as success=True.
        result = self._standard_run(
            task,
            fsm_def,
            context,
            "rewoo",
            execution_evidence_keys=[ContextKeys.EVIDENCE_STATUS],
        )
        # DECISION plan-2026-11-06T093029-0068b055/D-025: a plan cut off at
        # max_tokens ships the cause as the answer, not the `solve` state's
        # filler about evidence that does not exist. The `solve` call has
        # already been made by then (the FSM is linear; adding a branch for the
        # rare cut is not worth a second plan path). Do NOT run the cut plan's
        # prefix and do NOT retry at a higher cap here.
        if result.final_context.get(ContextKeys.PLAN_CUT_OFF) is True:
            return result.model_copy(
                update={
                    "answer": ErrorMessages.REWOO_PLAN_CUT_OFF,
                    "structured_output": None,
                }
            )
        return result

    def _register_handlers(self, api: API) -> None:
        """Register agent handlers with the API."""
        api.register_handler(
            api.create_handler(HandlerNames.REWOO_EXECUTOR)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(REWOOStates.EXECUTE_PLANS)
            .do(self._execute_all_plans)
        )
        self._register_iteration_limiter(api, self._make_iteration_limiter())

    def _execute_all_plans(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute all planned tool calls, substituting #EN variable references."""
        # DECISION plan-2026-11-06T093029-0068b055/D-025: the plan field was cut
        # off at max_tokens in THIS turn (core's cut-off record names it): the
        # plan is incomplete or lost, so run nothing and record `truncated`.
        # Do NOT fall through to the empty-plan run below (it left no cause
        # visible, live: rewoo_multi_step) and do NOT execute a salvaged prefix
        # (half a plan runs tools the full plan would have ordered or gated).
        cut = context.get(CONTEXT_KEY_CUT_OFF)
        if isinstance(cut, dict) and ContextKeys.PLAN_BLUEPRINT in cut.get(
            "fields", ()
        ):
            logger.warning(
                "REWOO plan_blueprint was cut off at max_tokens: no tool runs"
            )
            return {
                ContextKeys.EVIDENCE: {},
                ContextKeys.EVIDENCE_STATUS: [],
                ContextKeys.PLAN_CUT_OFF: True,
                ContextKeys.FORCED_STOP_REASON: StopReason.TRUNCATED,
            }
        plan_blueprint = context.get(ContextKeys.PLAN_BLUEPRINT, [])
        if not isinstance(plan_blueprint, list):
            logger.warning("plan_blueprint is not a list, skipping execution")
            return {ContextKeys.EVIDENCE: {}, ContextKeys.EVIDENCE_STATUS: []}

        evidence: dict[str, str] = {}
        status: list[dict[str, Any]] = []
        trace_entries: list[dict[str, Any]] = list(
            context.get(ContextKeys.AGENT_TRACE, [])
        )

        for step in plan_blueprint:
            if not isinstance(step, dict):
                continue

            plan_id = _evidence_id(step.get("plan_id"), len(status) + 1)
            tool_name = step.get("tool_name", "")
            tool_input = step.get("tool_input", {})
            description = step.get("description", "")

            # Substitute #EN references in tool_input
            tool_input = self._substitute_evidence_refs(tool_input, evidence)

            logger.info(
                LogMessages.PLAN_STEP.format(
                    current=plan_id,
                    total=len(plan_blueprint),
                    description=description,
                )
            )

            # Normalize tool_input to dict
            if isinstance(tool_input, str):
                tool_input = {"input": tool_input}
            elif not isinstance(tool_input, dict):
                tool_input = {"input": str(tool_input)}

            # Execute tool
            tool_call = ToolCall(
                tool_name=tool_name,
                parameters=tool_input,
                reasoning=description,
            )
            result = self.tools.execute(tool_call)

            # Store evidence (a failed call stores its error text)
            evidence[plan_id] = result.summary
            status.append(
                {
                    "id": plan_id,
                    "tool_name": tool_name,
                    "success": result.success,
                    ContextKeys.TOOL_STATUS: result.status,
                }
            )

            if result.success:
                logger.info(LogMessages.TOOL_EXECUTED.format(name=tool_name))
            else:
                logger.warning(
                    LogMessages.TOOL_FAILED.format(name=tool_name, error=result.error)
                )

            # Record in trace (include "action" key for cross-agent consistency)
            trace_entries.append(
                {
                    "action": f"{tool_name}({plan_id})",
                    "thought": description,
                    "plan_id": plan_id,
                    "tool_name": tool_name,
                    # plan-2026-09-29T103145-06a5ec0a/D-016: redacted copy.
                    "tool_input": redact_secret_entries(tool_input),
                    "description": description,
                    "result": result.summary,
                    "success": result.success,
                    ContextKeys.TOOL_STATUS: result.status,
                }
            )

        return {
            ContextKeys.EVIDENCE: evidence,
            ContextKeys.EVIDENCE_STATUS: status,
            ContextKeys.AGENT_TRACE: trace_entries,
        }

    def _substitute_evidence_refs(
        self,
        value: Any,
        evidence: dict[str, str],
    ) -> Any:
        """Recursively substitute #EN references with evidence values."""
        if isinstance(value, str):
            # Replace #E1, #E2, etc. with actual evidence
            def replace_ref(match: re.Match[str]) -> str:
                ref_key = match.group(1)
                if ref_key not in evidence:
                    logger.warning(
                        f"Evidence reference #{ref_key} not found in evidence store"
                    )
                return evidence.get(ref_key, "[unavailable]")

            return re.sub(r"#(E\d+)", replace_ref, value)

        if isinstance(value, dict):
            return {
                k: self._substitute_evidence_refs(v, evidence) for k, v in value.items()
            }

        if isinstance(value, list):
            return [self._substitute_evidence_refs(item, evidence) for item in value]

        return value

    def _make_iteration_limiter(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create the iteration limiter handler (shared rule, see handlers)."""
        # SHOULD_TERMINATE is set for consistency with every other pattern's
        # limiter. REWOO's FSM is currently linear
        # (plan_all -> execute_plans -> solve) so no transition reads it
        # today; the flag prevents an unbounded loop if a cycling state is
        # ever added (AP3-001).
        return make_iteration_limiter(
            self.config.max_iterations,
            {
                ContextKeys.MAX_ITERATIONS_REACHED: True,
                ContextKeys.SHOULD_TERMINATE: True,
            },
        )
