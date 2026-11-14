"""The HITL approval gate of ``BaseAgent`` (``HitlGateMixin``).

The one predicate (``_hitl_active``, ``_approval_predicate``), the gate registration, the refusals of flagged
tools and the driver that asks and writes the driver-only grant (``_handle_hitl_approval``). The security
boundary of the agents: read the D-004, D-005, D-023, D-049 and D-052 anchors below before editing. The
method bodies are the old ones (plan d454fd62 step 87).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.agents.constants import (
    AgentStates,
    ContextKeys,
    HandlerNames,
    HandlerPriorities,
)
from fsm_llm.agents.handlers import (
    AgentHandlers,
    approval_grant,
    call_label,
    call_ran,
    refusal_record,
)
from fsm_llm.agents.tools import make_hitl_checker, tool_call_from_context
from fsm_llm.core.definitions import AgentError
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger
from fsm_llm.policy import ApprovalPolicy, HumanInTheLoop
from fsm_llm.tools import ToolRegistry

from .helpers import _flagged_tool_policy, flagged_tool_names
from .surface import _AgentSurface


class HitlGateMixin(_AgentSurface):
    """The HITL approval gate of ``BaseAgent``: predicate, gate, refusals and the grant driver."""

    def _call_handlers(self, tools: ToolRegistry) -> AgentHandlers:
        """The call-local ``AgentHandlers`` of one HITL-capable run over *tools*.

        Refuses a flagged tool nobody can approve first (D-052: a registry can
        gain one after construction), then builds a fresh instance under the
        run's single approval predicate. Never stored on ``self`` (D-014).
        """
        self._refuse_unapprovable_flagged_tools()
        return AgentHandlers(tools, requires_approval=self._approval_predicate)

    def _register_hitl_gate(
        self,
        api: API,
        checker_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register the HITL approval-gate handler."""
        api.register_handler(
            api.create_handler(HandlerNames.HITL_GATE)
            .with_priority(HandlerPriorities.HITL_GATE)
            .at(HandlerTiming.CONTEXT_UPDATE)
            .when_keys_updated(ContextKeys.TOOL_NAME)
            .do(checker_fn)
        )

    @property
    def _hitl_active(self) -> bool:
        # Single source of truth for "this run needs HITL approval gating".
        # INVARIANT: the await_approval FSM state (include_approval_state)
        # MUST be built under exactly this predicate, because it is also the
        # predicate that registers the runtime approval gate. If the two
        # diverge, the gate can set approval_required=True with no
        # await_approval state to intercept it, and the tool executes
        # un-gated (THINK -> act runs execute_tool before the loop's
        # _handle_hitl_approval hook).
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-005
        # Moved here from ReactAgent so React, Reflexion and ReasoningReact
        # share ONE predicate for the FSM state, the gate and the
        # AgentHandlers refusal. Do NOT re-add a per-agent copy, and do NOT
        # AND it with "some tool is flagged": a policy may gate any tool.
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-004
        # With a policy, the policy alone decides (unchanged). With a callback
        # and NO policy, requires_approval=True tools are the default policy:
        # active iff some registered tool is flagged. Do NOT go back to
        # "no policy means no gating" (the flag was a no-op, SEC-03), and do
        # NOT gate a policy-less run with no flagged tool (callback-only HITL
        # with no flagged tool must keep running). See decisions.md D-004.
        hitl: HumanInTheLoop | None = getattr(self, "hitl", None)
        if hitl is None:
            return False
        if hitl.has_approval_policy:
            return True
        return hitl.has_approval_callback and bool(
            flagged_tool_names(getattr(self, "tools", None))
        )

    @property
    def _approval_predicate(self) -> ApprovalPolicy | None:
        """The approval policy for the refusal, the gate and the driver, or None.

        Set only under :attr:`_hitl_active` (plan-2026-09-24T091842-c1d5bfbc
        D-004/D-005), so the refusal never gates a run without the state. The
        HITL policy when one is set, else the flagged-tool default (D-004).
        """
        hitl: HumanInTheLoop | None = getattr(self, "hitl", None)
        if hitl is None or not self._hitl_active:
            return None
        if hitl.has_approval_policy:
            return hitl.requires_approval
        return _flagged_tool_policy(getattr(self, "tools", None))

    def _register_approval_gate(self, api: API) -> None:
        """Register the HITL gate under :attr:`_hitl_active` with the shared predicate."""
        hitl: HumanInTheLoop | None = getattr(self, "hitl", None)
        predicate = self._approval_predicate
        if hitl is None or predicate is None:
            return
        self._register_hitl_gate(api, make_hitl_checker(hitl, policy=predicate))

    def _refuse_unapprovable_flagged_tools(self) -> None:
        """Raise ``AgentError`` when flagged tools exist and nobody can decide on them.

        For the HITL patterns (ReAct, Reflexion, ReasoningReact). Call it in
        ``__init__`` after ``self.tools`` and ``self.hitl`` are set and at the
        start of ``run()``/``run_stream()`` (a registry can gain a flagged tool
        after construction). Raises when ``self.tools`` holds a
        ``requires_approval`` tool and ``self.hitl`` is ``None`` or has neither
        an approval callback nor a policy. With a policy and no callback it
        only WARNS: the policy owns the decision, and a call it gates raises
        ``ApprovalDeniedError``. No-op without a flagged tool.
        """
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-052: fail closed, as D-005
        # does for the patterns without HITL. With `hitl=None` or an empty
        # HumanInTheLoop() a requires_approval tool ran unasked (a construction
        # WARNING only). Do NOT downgrade this back to a warning, and do NOT
        # raise when a policy is set: an explicit policy owns its decision
        # (D-004 never ANDs it with the flag).
        flagged = flagged_tool_names(getattr(self, "tools", None))
        if not flagged:
            return
        hitl: HumanInTheLoop | None = getattr(self, "hitl", None)
        if hitl is None or not (hitl.has_approval_callback or hitl.has_approval_policy):
            raise AgentError(
                f"{type(self).__name__}: tools {flagged} have requires_approval="
                f"True but nobody can approve them. Pass "
                f"hitl=HumanInTheLoop(approval_callback=...), or register these "
                f"tools without requires_approval."
            )
        if not hitl.has_approval_callback:
            logger.warning(
                f"{type(self).__name__}: tools {flagged} have requires_approval="
                f"True and no HumanInTheLoop approval_callback is configured; "
                f"the approval policy decides, and a call it gates raises "
                f"ApprovalDeniedError"
            )

    def _refuse_flagged_tools(self) -> None:
        """Raise ``AgentError`` when ``self.tools`` holds a ``requires_approval`` tool.

        For patterns with no HITL path (REWOO, PlanExecute, ParallelReact,
        native_fc). Call it in ``__init__`` after ``self.tools`` is set and at
        the start of ``run()``, before any LLM call: a registry can gain a
        flagged tool after construction. No-op without a registry.
        """
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-005: fail closed. Do NOT
        # downgrade this to a warning, and do NOT drop the run() re-check: these
        # patterns execute tools with no approval gate, so a flagged tool would
        # run unapproved. Lift it only when the pattern gains real HITL (Track B).
        flagged = flagged_tool_names(getattr(self, "tools", None))
        if flagged:
            raise AgentError(
                f"{type(self).__name__} has no human-in-the-loop approval, but "
                f"tools {flagged} have requires_approval=True. Use ReactAgent, "
                f"ReflexionAgent or ReasoningReactAgent with a HumanInTheLoop, "
                f"or register these tools without requires_approval."
            )

    def _handle_hitl_approval(self, api: API, conversation_id: str) -> None:
        """Check and process HITL approval for the current context.

        Shared logic for agents that use synchronous HITL approval gates
        (ReactAgent, ReflexionAgent, ReasoningReactAgent).  Subclasses must
        set ``self.hitl`` to a :class:`HumanInTheLoop` instance (or ``None``).
        """
        hitl: HumanInTheLoop | None = getattr(self, "hitl", None)
        if hitl is None:
            return

        current_context = api.get_data(conversation_id)
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-023
        # Ask unless approval_granted is exactly True, and store a strict bool.
        # Do NOT test truthiness: await_approval routes on == True / == False
        # only, so a model-written "yes" or a callback's None parked the run
        # there, unasked, until BudgetExhaustedError.
        if current_context.get(ContextKeys.APPROVAL_GRANTED) is True:
            return

        tool_call = tool_call_from_context(current_context)
        tool_name, tool_input = tool_call.tool_name, tool_call.parameters

        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-005: ask only about a
        # registered tool the shared predicate gates. Do NOT ask on a bare
        # approval_required: the model can write it (tool "none"; approval
        # fatigue), and an extracted True for an ungated tool routes into
        # await_approval after the gate wrote False, where nothing else sets
        # approval_granted (BLOCKED until BudgetExhaustedError). Route it back.
        # The gate reads the FULL context (internal keys included), exactly as
        # AgentHandlers.approval_refusal does. Do NOT pass get_data's stripped
        # view: a policy on a `_` key then never asks while the refusal blocks.
        gate = self._approval_predicate
        tools = getattr(self, "tools", None)
        sub_id = api.get_sub_conversation_id(conversation_id)
        full = api.fsm_manager.get_complete_conversation(sub_id)["collected_data"]
        if not (tools and tool_name in tools and gate and gate(tool_call, full)):
            required = current_context.get(ContextKeys.APPROVAL_REQUIRED)
            if (
                required
                or api.get_current_state(conversation_id) == AgentStates.AWAIT_APPROVAL
            ):
                stray = (ContextKeys.APPROVAL_REQUIRED, ContextKeys.APPROVAL_GRANTED)
                api.update_context(conversation_id, dict.fromkeys(stray, False))
            return
        if not current_context.get(ContextKeys.APPROVAL_REQUIRED):
            return
        approved = hitl.request_approval(tool_call, current_context) is True
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-004
        # The public keys only route the FSM (the model can forge them). The
        # driver-only grant, bound to the exact call the human saw, is what
        # AgentHandlers.approval_refusal checks. Do NOT write a bare True here
        # and do NOT write it on denial: a grant must name its one call.
        grant = approval_grant(tool_name, tool_input) if approved else None
        api.update_context(
            conversation_id,
            {
                ContextKeys.APPROVAL_GRANTED: approved,
                ContextKeys.APPROVAL_REQUIRED: False,
                ContextKeys.DRIVER_APPROVAL: grant,
            },
        )
        if not approved:
            # LOOP-06 (D-021 of plan 06a5ec0a): the denial reaches the next think
            # turn as feedback, never as an observation: an observation would
            # bump observation_count and satisfy the D-008 conclude guard with
            # no tool result. The input shown is the redacted copy (D-016,
            # `call_label`).
            # D-034 (plan 07ad3f8c): the feedback is gone after the next think
            # turn, so the refusal is also kept as a fact for conclude. The
            # executor drops the entry if this call is approved later and runs
            # (D-045, `AgentHandlers.spend_grant`).
            # DECISION plan-2026-09-30T062855-07ad3f8c/D-049
            # A denied re-ask of a call that already ran in this run gets no
            # record: "was not performed" would be false, and the conclude
            # prompt would tell the model to deny an action that ran (a user
            # who retries then performs it twice). Do NOT record it with other
            # wording (two facts about one call in one prompt), and do NOT keep
            # a raw copy of executed calls to compare (an unredacted second
            # copy of the parameters in context, D-045); the run's trace is the
            # record of executed calls. See decisions.md D-049.
            record = refusal_record(tool_name, tool_input)
            refused = list(full.get(ContextKeys.REFUSED_ACTIONS) or [])
            ran = call_ran(full.get(ContextKeys.AGENT_TRACE), tool_name, tool_input)
            if record not in refused and not ran:
                refused.append(record)
            denial: dict[str, Any] = {
                ContextKeys.TOOL_NAME: None,
                ContextKeys.TOOL_INPUT: None,
                ContextKeys.AGENT_FEEDBACK: (
                    f"The human reviewer denied the call "
                    f"{call_label(tool_name, tool_input)}. "
                    "Do not repeat it; choose another tool or approach."
                ),
            }
            if refused:
                denial[ContextKeys.REFUSED_ACTIONS] = refused
            api.update_context(conversation_id, denial)
