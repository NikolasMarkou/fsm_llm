"""``AgentHandlers``: the tool executor of the ReAct family and the HITL approval refusal (plan
d454fd62 step 88). ``approval_refusal``, ``spend_grant`` and ``consume_approval`` are the security
boundary of the driver-only grant (``_approval_granted``): the class body is the old ``agents/handlers.py``
one, byte for byte. It inherits the think-turn limiter from ``limiters``; the grant helpers it compares
against are in ``approval``."""

from __future__ import annotations

from typing import Any

from fsm_llm.agents.constants import ContextKeys, LogMessages, StopReason, ToolRunStatus
from fsm_llm.agents.tools import (
    normalize_tool_input,
    refuse_execute_without_gated,
    tool_call_from_context,
)
from fsm_llm.logging import logger
from fsm_llm.policy import ApprovalPolicy
from fsm_llm.tools import ToolCall, ToolRegistry, redact_secret_entries

from .approval import approval_grant, call_label, refusal_record
from .forced import forced_stop_skip
from .limiters import ThinkTurnLimiter
from .recording import record_tool_step, with_feedback


# DECISION plan-2026-10-01T093600-944e2692/D-042: the `gated` check lives in
# the executor that passes `gated`, not in the limiter. Do NOT build an
# AgentHandlers in an agent that only needs the limiter (ParallelReact): its
# constructor refuses an old-signature registry the agent never sends `gated`.
class AgentHandlers(ThinkTurnLimiter):
    """Collection of handler functions for agent FSM operations."""

    def __init__(
        self,
        registry: ToolRegistry,
        requires_approval: ApprovalPolicy | None = None,
    ) -> None:
        """
        :param registry: Tools the agent may run.
        :param requires_approval: The agent's HITL policy
            (``hitl.requires_approval``), passed only under the predicate that
            builds ``await_approval`` and registers the gate. When set, a
            known tool it gates runs only on a matching driver grant.

        Raises:
            AgentError: ``registry.execute`` cannot take ``gated=``
                (``refuse_execute_without_gated``).
        """
        refuse_execute_without_gated(registry)
        super().__init__()
        self.registry = registry
        self.requires_approval = requires_approval
        self._consecutive_no_tool = 0
        # Last driver grant this instance spent and how many it has spent;
        # see approval_refusal (plan-2026-09-29T103145-06a5ec0a/D-015).
        self._spent_grant: dict[str, Any] | None = None
        self._grants_spent = 0

    def _run_selected_tool(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the tool selected during the think state.

        Body of :meth:`execute_tool`, which adds approval consumption.
        """
        forced = forced_stop_skip(context)
        if forced is not None:
            return forced
        tool_name = context.get(ContextKeys.TOOL_NAME)
        tool_input = context.get(ContextKeys.TOOL_INPUT)
        if tool_input is None:
            tool_input = {}
        reasoning = context.get(ContextKeys.REASONING, "")

        # DECISION plan-2026-09-24T045559-3e4eb3e5/D-012
        # An unknown name is a no-tool turn. Do NOT send it down the real-tool
        # path: its [TOOL FAILED] observation would satisfy the conclude guard.
        unknown = bool(tool_name) and tool_name != ContextKeys.NO_TOOL
        unknown = unknown and str(tool_name) not in self.registry
        miss = f"Unknown tool '{tool_name}'. " if unknown else ""
        # Clear the selection as the real-tool path does: core re-extracts a
        # key only while it is unset, so a kept name blocks every later tool.
        clear = {ContextKeys.TOOL_NAME: None, ContextKeys.TOOL_INPUT: None}
        if not tool_name or tool_name == ContextKeys.NO_TOOL or unknown:
            # Skip stall detection for HITL agents awaiting approval
            if context.get(ContextKeys.APPROVAL_REQUIRED) and not unknown:
                return {
                    ContextKeys.TOOL_RESULT: "Awaiting approval.",
                    ContextKeys.TOOL_STATUS: ToolRunStatus.SKIPPED,
                }

            # Block premature termination: must use at least one tool
            if self._current_iteration <= 1 and context.get(
                ContextKeys.SHOULD_TERMINATE
            ):
                tool_names = [t.name for t in self.registry.list_tools()]
                return with_feedback(
                    f"{miss}You must use at least one tool before concluding. "
                    f"Available: {', '.join(tool_names)}",
                    {
                        ContextKeys.TOOL_STATUS: ToolRunStatus.REJECTED,
                        **clear,
                        ContextKeys.SHOULD_TERMINATE: False,
                    },
                )

            self._consecutive_no_tool += 1
            should_terminate = context.get(ContextKeys.SHOULD_TERMINATE)

            # Stall detection: force terminate after 3 consecutive no-tool cycles
            if self._consecutive_no_tool >= 3:
                logger.warning(
                    f"Stall detected: {self._consecutive_no_tool} consecutive "
                    f"iterations with no tool selected, forcing termination"
                )
                return {
                    ContextKeys.TOOL_RESULT: (
                        "No tool was called for 3 consecutive iterations. "
                        "Terminating — provide your best answer now."
                    ),
                    ContextKeys.TOOL_STATUS: ToolRunStatus.SKIPPED,
                    ContextKeys.SHOULD_TERMINATE: True,
                    # DECISION plan_2026-05-30_5598b755/D-004 [STALE]
                    # The act->conclude / think->conclude guards now require
                    # observation_count>0 OR max_iterations_reached. A stalled
                    # tool-free run has no observations, so flag the forced
                    # termination here too — otherwise the guard would block
                    # conclude and the loop would run to the hard budget ceiling.
                    ContextKeys.MAX_ITERATIONS_REACHED: True,
                    ContextKeys.FORCED_STOP_REASON: StopReason.STALLED.value,
                }

            if not should_terminate or unknown:
                # Instructive warning to push model toward tool use
                tool_names = [t.name for t in self.registry.list_tools()]
                return with_feedback(
                    f"{miss}WARNING: No tool was called but the task is not complete. "
                    f"You must select a tool from: {', '.join(tool_names)}. "
                    "Do not answer from memory — use a tool to gather information.",
                    {
                        ContextKeys.TOOL_STATUS: ToolRunStatus.SKIPPED,
                        **clear,
                        ContextKeys.SHOULD_TERMINATE: None,
                    },
                )

            return with_feedback(
                "No tool was selected.",
                {ContextKeys.TOOL_STATUS: ToolRunStatus.SKIPPED, **clear},
            )

        refusal = self.approval_refusal(context)
        if refusal is not None:
            return refusal

        # Reset stall counter — a tool was actually selected
        self._consecutive_no_tool = 0

        # Spend the grant and clear the selection before the tool runs, so a
        # delta core discards (handler timeout) cannot leave a reusable grant.
        spent = self.spend_grant(context)

        tool_input = normalize_tool_input(tool_input)

        # Recovery: if tool_input is empty, try to infer parameters from context.
        if not tool_input and tool_name in self.registry:
            tool_def = self.registry.get(tool_name)
            schema = tool_def.parameter_schema or {}
            props = schema.get("properties", {})
            if not isinstance(props, dict):
                props = {}
            # Handle flat {param: description} schemas (no "properties" wrapper)
            if not props and schema and "type" not in schema:
                if all(isinstance(v, str) for v in schema.values()):
                    props = {k: {} for k in schema}
            required = schema.get("required", list(props.keys()))

            # Single-param recovery: use the task as the param value
            # (most common case: search(query=task))
            if len(required) == 1:
                param_name = required[0]
                task = context.get(ContextKeys.TASK, "")
                spec = props.get(param_name)
                # A bool/str property schema carries no type: recover as base did.
                ptype = spec.get("type") if isinstance(spec, dict) else None
                # Prose in a non-string param is a TypeError; the miss names the param.
                string_ok = ptype is None or ptype == "string"
                string_ok = string_ok or (isinstance(ptype, list) and "string" in ptype)
                # DECISION plan-2026-09-24T091842-c1d5bfbc/D-030: an array param
                # gets [task]; do NOT drop it (list tools ran here pre-D-011).
                array_ok = ptype == "array"
                array_ok = array_ok or (isinstance(ptype, list) and "array" in ptype)
                if task and string_ok:
                    tool_input = {param_name: task}
                    logger.info(f"Recovered empty tool_input: {param_name}=<task>")
                elif task and array_ok:
                    tool_input = {param_name: [task]}
                    logger.info(f"Recovered empty tool_input: {param_name}=[<task>]")

        # plan-2026-09-29T103145-06a5ec0a/D-016: the tool gets `tool_input`;
        # the log, observation, action and trace get this redacted copy.
        shown_input = redact_secret_entries(tool_input)
        logger.info(LogMessages.TOOL_SELECTED.format(name=tool_name, input=shown_input))

        tool_call = ToolCall(
            tool_name=tool_name,
            parameters=tool_input,
            reasoning=reasoning,
        )

        # DECISION plan-2026-10-01T093600-944e2692/D-008: a granted call is
        # executed with `gated=True`, so a RetryingToolRegistry never re-runs it
        # (policy-gated tools included; D-052 of plan 06a5ec0a). Do NOT mark it
        # through a context variable or a `ToolCall` field (a side channel, and
        # `ToolCall` is built from model output), and do NOT pass `gated=True`
        # for any call whose grant was not spent on this turn.
        result = self.registry.execute(
            tool_call, gated=ContextKeys.APPROVALS_SPENT in spent
        )

        if result.success:
            logger.info(LogMessages.TOOL_EXECUTED.format(name=tool_name))
        else:
            logger.warning(
                LogMessages.TOOL_FAILED.format(name=tool_name, error=result.error)
            )

        # Observation string: a failure (or a timeout, outcome unknown) is
        # prefixed so the LLM can tell (`ToolResult.observation`).
        observation = result.observation

        recorded = record_tool_step(
            context,
            tool_name=tool_name,
            shown_input=shown_input,
            observation=observation,
            status=result.status,
            thought=reasoning,
            action=call_label(tool_name, tool_input),
        )

        # DECISION plan-2026-10-01T093600-944e2692/D-043: the status is
        # `ToolResult.status` (`unknown` for a timed-out call). Do NOT write
        # "failed" for a timeout: the approver's context summary and
        # PlanExecute read this key, and "failed" reads as safe to run again.
        return {
            **spent,
            ContextKeys.TOOL_RESULT: observation,
            ContextKeys.TOOL_STATUS: result.status,
            ContextKeys.TOOL_ERROR: result.error,
            **recorded,
            ContextKeys.SHOULD_TERMINATE: None,
        }

    def spend_grant(self, context: dict[str, Any]) -> dict[str, Any]:
        """Record a driver grant as spent; return the selection and grant clears.

        Call it after :meth:`approval_refusal` returned None and before the tool
        runs (``execute_tool``, ReasoningReact's ``reason`` path); merge the
        returned delta into the executor's delta. The record lives on this
        call-local instance, so it survives a delta core discards. A spent
        grant also removes the call's ``refused_actions`` entry
        (:func:`refusal_record`), if an earlier ask refused it; the key is
        deleted when no entry is left. Never raises.
        """
        spent: dict[str, Any] = {
            ContextKeys.TOOL_NAME: None,
            ContextKeys.TOOL_INPUT: None,
        }
        if context.get(ContextKeys.DRIVER_APPROVAL) is not None:
            tool_name = context.get(ContextKeys.TOOL_NAME)
            tool_input = context.get(ContextKeys.TOOL_INPUT)
            self._grants_spent += 1
            self._spent_grant = approval_grant(tool_name, tool_input)
            spent[ContextKeys.DRIVER_APPROVAL] = None
            spent[ContextKeys.APPROVALS_SPENT] = self._grants_spent
            # DECISION plan-2026-09-30T062855-07ad3f8c/D-045
            # An approved call that runs is no longer a refused action: drop
            # its record here, where the grant is spent. Do NOT refuse a
            # re-asked identical call without asking (the approver stays in
            # control and may change their mind), and do NOT leave the record
            # (the conclude prompt then tells the model to deny an action
            # that ran). See decisions.md D-045.
            refused = context.get(ContextKeys.REFUSED_ACTIONS)
            record = refusal_record(tool_name, tool_input)
            if isinstance(refused, list) and record in refused:
                kept = [entry for entry in refused if entry != record]
                spent[ContextKeys.REFUSED_ACTIONS] = kept or None
        return spent

    def execute_tool(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the tool selected during the think state.

        Called as a POST_TRANSITION handler when entering the 'act' state.
        """
        return self.consume_approval(context, self._run_selected_tool(context))

    def approval_refusal(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """Refusal delta for a gated call without its driver grant, else None.

        Only a known tool the ``requires_approval`` predicate gates is checked;
        with no predicate this always returns None.
        """
        tool_name = context.get(ContextKeys.TOOL_NAME)
        if self.requires_approval is None or not tool_name:
            return None
        if tool_name == ContextKeys.NO_TOOL or str(tool_name) not in self.registry:
            return None
        # Pre-recovery input: the driver approved the call as it was selected.
        call = approval_grant(tool_name, context.get(ContextKeys.TOOL_INPUT))
        if not self.requires_approval(tool_call_from_context(context), context):
            return None
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-015
        # A grant this instance already spent, still in context because core
        # discarded the spending delta (handler_timeout), is void: the tool ran.
        # The count the delta writes tells a discarded spend (context behind this
        # instance) from a fresh approval of the same call after a landed one.
        # Do NOT key this on the call alone (an identical call approved twice
        # must run twice) and do NOT mark the executor .critical() instead (every
        # handler timeout would fail the run).
        spent = context.get(ContextKeys.APPROVALS_SPENT, 0)
        stale = call == self._spent_grant and spent != self._grants_spent
        if context.get(ContextKeys.DRIVER_APPROVAL) == call and not stale:
            return None
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-004
        # The security boundary is HERE, not the FSM route: the public
        # approval_granted is a plain key, which a state with a bulk pass lets
        # the model write (`think` under use_classification=True), and a True
        # there skips the ask and routes await_approval -> act. Do NOT trust
        # approval_granted or approval_required here; call the predicate and
        # require the driver-only grant for this exact call. Do NOT record an
        # observation (a refused call is not conclude evidence) and do NOT
        # clear the selection (the driver asks for it next iteration). A grant
        # for another call is void.
        # Do NOT reduce the grant to a bare True: after such a refusal the
        # driver asks before the next step, and the `think` step that follows
        # can still fill an empty tool_input after the human approved the
        # empty call (D-023, pinned by TestEmptyThenFilledCall).
        # `await_approval` itself extracts nothing
        # (plan-2026-09-30T062855-07ad3f8c/D-033).
        reason = "its approval was already spent" if stale else "no approval"
        logger.warning(f"Refused gated tool '{tool_name}': {reason} for this call")
        refusal = {
            ContextKeys.TOOL_RESULT: (
                f"Tool '{tool_name}' needs human approval before it can run."
            ),
            ContextKeys.TOOL_STATUS: ToolRunStatus.AWAITING_APPROVAL,
            ContextKeys.APPROVAL_REQUIRED: True,
            ContextKeys.APPROVAL_GRANTED: None,
            ContextKeys.DRIVER_APPROVAL: None,
        }
        if stale:  # resync, so the next fresh approval of this call runs
            refusal[ContextKeys.APPROVALS_SPENT] = self._grants_spent
        return refusal

    def consume_approval(
        self, context: dict[str, Any], delta: dict[str, Any]
    ) -> dict[str, Any]:
        """Add the approval clears to a tool-executor delta (one approval = one call)."""
        # DECISION plan-2026-09-24T045559-3e4eb3e5/D-015
        # Do NOT let an approval outlive the single tool call it was granted
        # for. The driver asks only while approval_granted is unset, so a stale
        # True skipped the callback and routed await_approval -> act unasked.
        # Kept while approval_required is set (the call is still pending).
        # Extended by plan-2026-09-24T091842-c1d5bfbc/D-004: the driver grant is
        # spent by any executor turn that is not a refusal, so it covers one
        # call; a refusal clears it itself.
        pending = context.get(ContextKeys.APPROVAL_REQUIRED)
        if ContextKeys.APPROVAL_GRANTED in context and not pending:
            delta = {**delta, ContextKeys.APPROVAL_GRANTED: None}
        if ContextKeys.DRIVER_APPROVAL in context:
            refused = (
                delta.get(ContextKeys.TOOL_STATUS) == ToolRunStatus.AWAITING_APPROVAL
            )
            if not refused:
                delta = {**delta, ContextKeys.DRIVER_APPROVAL: None}
        return delta
