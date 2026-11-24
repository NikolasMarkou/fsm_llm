"""The plan executor of the REWOO pattern (plan a555dcfe step 14; bodies moved from ``rewoo.py``).

Interface contract: ``execute_plans(context, tools) -> dict`` is the body of the handler
``agents.rewoo_executor`` (``handlers/library.py`` binds it to the run's registry): it runs the
``plan_blueprint`` steps one after another with ``tools.execute`` (no model call), substituting
``#E<n>`` references, and returns the context update (``evidence``, ``evidence_status``,
``agent_trace``). A plan cut off at ``max_tokens`` runs no tool and returns ``plan_cut_off``. Never
raises for a tool: a failed call stores its error text. ``substitute_evidence_refs(value, evidence)``
replaces ``#E<n>`` in a string, dict or list (an unknown reference becomes ``[unavailable]``);
``evidence_id(raw, position)`` is the ``E<n>`` key of a plan step (never raises).
"""

from __future__ import annotations

import re
from typing import Any

from fsm_llm.agents.constants import ContextKeys, LogMessages, StopReason
from fsm_llm.core.definitions import CONTEXT_KEY_CUT_OFF
from fsm_llm.logging import logger
from fsm_llm.tools import ToolCall, ToolRegistry, redact_secret_entries

__all__ = ["evidence_id", "execute_plans", "substitute_evidence_refs"]

# A plan id the model wrote as 1, "1", "E1" or "#E1" (any case).
_PLAN_ID = re.compile(r"#?E?(\d+)", re.IGNORECASE)


def evidence_id(raw: Any, position: int) -> str:
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


def substitute_evidence_refs(value: Any, evidence: dict[str, str]) -> Any:
    """Recursively substitute #EN references with evidence values."""
    if isinstance(value, str):

        def replace_ref(match: re.Match[str]) -> str:
            ref_key = match.group(1)
            if ref_key not in evidence:
                logger.warning(
                    f"Evidence reference #{ref_key} not found in evidence store"
                )
            return evidence.get(ref_key, "[unavailable]")

        return re.sub(r"#(E\d+)", replace_ref, value)
    if isinstance(value, dict):
        return {k: substitute_evidence_refs(v, evidence) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute_evidence_refs(item, evidence) for item in value]
    return value


def execute_plans(context: dict[str, Any], tools: ToolRegistry) -> dict[str, Any]:
    """Execute all planned tool calls, substituting #EN variable references."""
    # DECISION plan-2026-11-06T093029-0068b055/D-025: the plan field was cut
    # off at max_tokens in THIS turn (core's cut-off record names it): the
    # plan is incomplete or lost, so run nothing and record `truncated`.
    # Do NOT fall through to the empty-plan run below (it left no cause
    # visible, live: rewoo_multi_step) and do NOT execute a salvaged prefix
    # (half a plan runs tools the full plan would have ordered or gated).
    cut = context.get(CONTEXT_KEY_CUT_OFF)
    if isinstance(cut, dict) and ContextKeys.PLAN_BLUEPRINT in cut.get("fields", ()):
        logger.warning("REWOO plan_blueprint was cut off at max_tokens: no tool runs")
        return {
            ContextKeys.EVIDENCE: {},
            ContextKeys.EVIDENCE_STATUS: [],
            ContextKeys.PLAN_CUT_OFF: True,
            ContextKeys.FORCED_STOP_REASON: StopReason.TRUNCATED.value,
        }
    plan_blueprint = context.get(ContextKeys.PLAN_BLUEPRINT, [])
    if not isinstance(plan_blueprint, list):
        logger.warning("plan_blueprint is not a list, skipping execution")
        return {ContextKeys.EVIDENCE: {}, ContextKeys.EVIDENCE_STATUS: []}

    evidence: dict[str, str] = {}
    status: list[dict[str, Any]] = []
    trace_entries: list[dict[str, Any]] = list(context.get(ContextKeys.AGENT_TRACE, []))

    for step in plan_blueprint:
        if not isinstance(step, dict):
            continue

        plan_id = evidence_id(step.get("plan_id"), len(status) + 1)
        tool_name = step.get("tool_name", "")
        description = step.get("description", "")
        tool_input = substitute_evidence_refs(step.get("tool_input", {}), evidence)

        logger.info(
            LogMessages.PLAN_STEP.format(
                current=plan_id, total=len(plan_blueprint), description=description
            )
        )

        # Normalize tool_input to dict
        if isinstance(tool_input, str):
            tool_input = {"input": tool_input}
        elif not isinstance(tool_input, dict):
            tool_input = {"input": str(tool_input)}

        result = tools.execute(
            ToolCall(tool_name=tool_name, parameters=tool_input, reasoning=description)
        )

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
