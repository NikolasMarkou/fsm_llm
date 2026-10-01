"""
Agent-specific handlers for tool execution, iteration limiting,
observation tracking, and HITL gating.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from fsm_llm.logging import logger

from .constants import (
    RESULT_DROPPED_CONTEXT_KEYS,
    AgentStates,
    ContextKeys,
    Defaults,
    LogMessages,
    ReflexionStates,
    StopReason,
)
from .definitions import AgentStep, ToolCall
from .exceptions import AgentTimeoutError, BudgetExhaustedError
from .hitl import ApprovalPolicy
from .tools import ToolRegistry, normalize_tool_input, redact_secret_entries
from .truncation import smart_truncate

# Errors that end a whole run even when a sub-agent raises them inside a handler.
RUN_ENDING_ERRORS: tuple[type[Exception], ...] = (
    AgentTimeoutError,
    BudgetExhaustedError,
)


class RunEndingErrorHolder:
    """Carries a budget or timeout error out of an FSM handler to the driver.

    Core's default handler error mode (``continue``) swallows a handler's
    raise, so a sub-agent's ``AgentTimeoutError`` / ``BudgetExhaustedError``
    cannot propagate from inside the handler that ran it. The handler stores it
    here and stops starting new work; the driver re-raises it once
    ``_run_conversation_loop`` returns.

    Interface contract (ADaPT subtasks; Orchestrator workers reuse it):
        - Create one per ``run()`` call and hand it to the handler closure.
          Never store it on ``self`` (089d0ec7 D-014: agents are shared across
          threads and recursive runs).
        - ``capture(exc)``: True when ``exc`` is one of ``RUN_ENDING_ERRORS``
          (the first one is kept, later ones only logged), False otherwise, so
          the caller records any other exception as an ordinary failure.
        - ``is_set``: True once an error is held; stop starting new work.
        - ``raise_if_set()``: re-raise the held error; no-op when empty.
    """

    def __init__(self) -> None:
        self.error: Exception | None = None

    @property
    def is_set(self) -> bool:
        return self.error is not None

    def capture(self, exc: BaseException) -> bool:
        if not isinstance(exc, RUN_ENDING_ERRORS):
            return False
        if self.error is None:
            self.error = exc
        else:
            logger.warning(f"Run already ending; dropping a later error: {exc}")
        return True

    def raise_if_set(self) -> None:
        if self.error is not None:
            raise self.error


def approval_grant(tool_name: Any, tool_input: Any) -> dict[str, Any]:
    """The driver grant for one call: ``{"tool_name", "parameters"}``.

    Shared by the approval driver (``BaseAgent._handle_hitl_approval``, which
    writes it) and :meth:`AgentHandlers.approval_refusal` (which compares it),
    so both normalize the call the same way. ``tool_input`` is normalized with
    :func:`normalize_tool_input`; never raises.
    """
    return {
        "tool_name": str(tool_name),
        "parameters": normalize_tool_input(tool_input),
    }


def refusal_record(tool_name: Any, tool_input: Any) -> str:
    """The ``refused_actions`` entry for one call a human approver refused.

    Shared by the approval driver (``BaseAgent._handle_hitl_approval``, which
    appends it on a denial) and :meth:`AgentHandlers.spend_grant` (which
    removes it when the same call is later approved and runs), so both build
    the same text. The parameters are the :func:`redact_secret_entries` copy
    of the :func:`normalize_tool_input` form (secret-looking keys at any depth,
    in nested mappings and in lists, show ``<redacted>``). Never raises.
    """
    shown = redact_secret_entries(normalize_tool_input(tool_input))
    return (
        f"{tool_name}({shown}): was refused by the human approver and was "
        "not performed."
    )


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
        ContextKeys.TOOL_STATUS: "skipped",
        ContextKeys.TOOL_NAME: None,
        ContextKeys.TOOL_INPUT: None,
        ContextKeys.SHOULD_TERMINATE: True,
    }


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


class AgentHandlers:
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
        """
        self.registry = registry
        self.requires_approval = requires_approval
        self._current_iteration = 0
        self._transitions = 0
        self._consecutive_no_tool = 0
        # Last driver grant this instance spent and how many it has spent;
        # see approval_refusal (plan-2026-09-29T103145-06a5ec0a/D-015).
        self._spent_grant: dict[str, Any] | None = None
        self._grants_spent = 0

    def reset(self) -> None:
        """Reset handler state for a new run."""
        self._current_iteration = 0
        self._transitions = 0
        self._consecutive_no_tool = 0
        self._spent_grant = None
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
                    ContextKeys.TOOL_STATUS: "skipped",
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
                        ContextKeys.TOOL_STATUS: "rejected",
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
                    ContextKeys.TOOL_STATUS: "skipped",
                    ContextKeys.SHOULD_TERMINATE: True,
                    # DECISION plan_2026-05-30_5598b755/D-004 [STALE]
                    # The act->conclude / think->conclude guards now require
                    # observation_count>0 OR max_iterations_reached. A stalled
                    # tool-free run has no observations, so flag the forced
                    # termination here too — otherwise the guard would block
                    # conclude and the loop would run to the hard budget ceiling.
                    ContextKeys.MAX_ITERATIONS_REACHED: True,
                    ContextKeys.FORCED_STOP_REASON: StopReason.STALLED,
                }

            if not should_terminate or unknown:
                # Instructive warning to push model toward tool use
                tool_names = [t.name for t in self.registry.list_tools()]
                return with_feedback(
                    f"{miss}WARNING: No tool was called but the task is not complete. "
                    f"You must select a tool from: {', '.join(tool_names)}. "
                    "Do not answer from memory — use a tool to gather information.",
                    {
                        ContextKeys.TOOL_STATUS: "skipped",
                        **clear,
                        ContextKeys.SHOULD_TERMINATE: None,
                    },
                )

            return with_feedback(
                "No tool was selected.", {ContextKeys.TOOL_STATUS: "skipped", **clear}
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

        result = self.registry.execute(tool_call)

        if result.success:
            logger.info(LogMessages.TOOL_EXECUTED.format(name=tool_name))
        else:
            logger.warning(
                LogMessages.TOOL_FAILED.format(name=tool_name, error=result.error)
            )

        # Build observation string — prefix failures so LLM can distinguish
        observation = result.summary
        if not result.success:
            observation = f"[TOOL FAILED] {observation}"

        # Accumulate observations
        observations = context.get(ContextKeys.OBSERVATIONS, [])
        if not isinstance(observations, list):
            observations = []
        trace = context.get(ContextKeys.AGENT_TRACE, [])
        if not isinstance(trace, list):
            trace = []

        step_num = next_step_number(trace)
        observation_entry = (
            f"[Step {step_num}] Tool: {tool_name} | "
            f"Input: {shown_input} | "
            f"Result: {observation}"
        )
        # The raw result is capped in ToolResult.summary, but the prefix +
        # tool_input repr are unbounded; truncate the assembled entry so the
        # stored observation honors MAX_OBSERVATION_LENGTH.
        observation_entry = smart_truncate(
            observation_entry, Defaults.MAX_OBSERVATION_LENGTH
        )
        observations.append(observation_entry)

        # Prune if too many observations
        if len(observations) > Defaults.MAX_OBSERVATIONS:
            dropped = len(observations) - Defaults.MAX_OBSERVATIONS
            logger.debug(
                f"Pruning {dropped} old observations "
                f"(keeping last {Defaults.MAX_OBSERVATIONS})"
            )
            observations = observations[-Defaults.MAX_OBSERVATIONS :]

        # Track in agent trace
        trace_step = AgentStep(
            iteration=step_num,
            thought=reasoning,
            action=f"{tool_name}({shown_input})",
            observation=observation,
        ).model_dump(mode="json")
        # Preserve structured tool input so _build_trace can recover parameters
        # (mirrors REWOOAgent, which stores tool_input on the trace dict).
        trace_step["tool_input"] = shown_input
        trace.append(trace_step)

        return {
            **spent,
            ContextKeys.TOOL_RESULT: observation,
            ContextKeys.TOOL_STATUS: "success" if result.success else "failed",
            ContextKeys.TOOL_ERROR: result.error,
            ContextKeys.OBSERVATIONS: observations,
            ContextKeys.OBSERVATION_COUNT: len(observations),
            ContextKeys.AGENT_TRACE: trace,
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
        tool_call = ToolCall(
            tool_name=call["tool_name"],
            parameters=call["parameters"],
            reasoning=context.get(ContextKeys.REASONING, ""),
        )
        if not self.requires_approval(tool_call, context):
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
            ContextKeys.TOOL_STATUS: "awaiting_approval",
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
            refused = delta.get(ContextKeys.TOOL_STATUS) == "awaiting_approval"
            if not refused:
                delta = {**delta, ContextKeys.DRIVER_APPROVAL: None}
        return delta

    def check_iteration_limit(self, context: dict[str, Any]) -> dict[str, Any]:
        """Count think turns and force the stop once the budget is spent.

        Called as a PRE_TRANSITION handler on every transition. ``max_iterations``
        (``_max_iterations`` in context) counts think turns: the count rises on
        each transition out of ``think``. The stop is forced (``max_iterations_
        reached`` and ``should_terminate`` True) when a later transition closes
        a cycle with ``count >= max - 1``, so the next think turn concludes: a
        limit of N gives N think turns and N - 1 tool turns, and no tool runs
        after the flag. An approved ``await_approval -> act`` exit does not close
        the cycle (its call still runs). Independently, the stop is forced
        ``FORCED_STOP_MARGIN`` transitions (at most ``max``) before the
        ``max * FSM_BUDGET_MULTIPLIER`` loop ceiling, so a run with long cycles
        concludes instead of raising ``BudgetExhaustedError``. A limit of 1
        behaves like 2 (the first cycle's tool always runs). A turn that
        concludes on its own evidence (:func:`concluded_on_evidence`) with no
        recorded forced reason is never forced, and withdraws an earlier flag
        (``max_iterations_reached`` back to False), so the last think turn's
        own conclusion reports ``success=True``. Never raises.
        """
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-028
        # LOOP-04: count think exits here, on the instance. Do NOT move this
        # into make_iteration_limiter or give that factory a state filter
        # (3e4eb3e5 D-003, c1d5bfbc D-007: the other patterns count every turn).
        # Do NOT force the stop on the think exit itself: the think -> act
        # decision is already made, so the cycle's tool would be skipped and
        # max_iterations=1 would run no tool at all. Keep the `- 1` early rule:
        # the flag lands at the end of cycle max - 1, and think turn max
        # concludes on it. See decisions.md D-028.
        self._transitions += 1
        leaving = context.get(ContextKeys.CURRENT_STATE) or AgentStates.THINK
        if leaving == AgentStates.THINK:
            self._current_iteration += 1
        max_iterations = context.get("_max_iterations", Defaults.MAX_ITERATIONS)

        logger.debug(
            LogMessages.ITERATION.format(
                current=self._current_iteration, max=max_iterations
            )
        )

        # DECISION plan-2026-09-29T103145-06a5ec0a/D-051: a turn whose own
        # verdict concludes on evidence (think: should_terminate, which think
        # entry refreshed; Reflexion evaluate: evaluation_passed) is the
        # model's conclusion, not the budget's, even after the flag: the
        # flag is withdrawn so the run reports success. Do NOT drop the
        # recorded-reason check (a stall or reflection cap stays forced) and
        # do NOT read should_terminate on a state that does not refresh it.
        if (
            concluded_on_evidence(leaving, context)
            and recorded_forced_reason(context) is None
        ):
            update: dict[str, Any] = {
                ContextKeys.ITERATION_COUNT: self._current_iteration
            }
            if context.get(ContextKeys.MAX_ITERATIONS_REACHED) is True:
                logger.info("The model concluded on its own on the last turn")
                update[ContextKeys.MAX_ITERATIONS_REACHED] = False
            return update

        approved = (
            leaving == AgentStates.AWAIT_APPROVAL
            and context.get(ContextKeys.APPROVAL_GRANTED) is True
        )
        cycle_closed = leaving != AgentStates.THINK and not approved
        spent = cycle_closed and self._current_iteration >= max_iterations - 1
        margin = min(Defaults.FORCED_STOP_MARGIN, max_iterations)
        ceiling = max_iterations * Defaults.FSM_BUDGET_MULTIPLIER
        if spent or self._transitions >= ceiling - margin:
            return {
                ContextKeys.ITERATION_COUNT: self._current_iteration,
                ContextKeys.MAX_ITERATIONS_REACHED: True,
                ContextKeys.SHOULD_TERMINATE: True,
            }

        return {ContextKeys.ITERATION_COUNT: self._current_iteration}


# DECISION plan-2026-09-24T045559-3e4eb3e5/D-003: the 8 context-counting
# patterns build their limiter here; do NOT hand-roll a per-pattern copy (the
# copies drifted: 7 of 8 triggered at `>= max`, not `>= max - 1`), and do NOT
# fold this into AgentHandlers.check_iteration_limit (it counts on the
# instance, these count in context; merging changes ReAct semantics).
def make_iteration_limiter(
    max_iterations: int,
    forced: dict[str, Any],
    *,
    context_max_key: str | None = None,
    early: bool = True,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Build a PRE_TRANSITION iteration limiter that counts in context.

    Args:
        max_iterations: The limit, or the fallback when ``context_max_key``
            is given but absent from the context.
        forced: Context updates returned verbatim (copied once) when the limit
            is hit, e.g. ``{SHOULD_TERMINATE: True}``.
        context_max_key: Optional context key whose value overrides
            ``max_iterations`` at call time.
        early: ``True`` (default) triggers at ``limit - 1``; ``False``
            triggers at ``limit`` (maker_checker only, see D-014).

    Returns:
        A handler returning ``{ITERATION_COUNT: count}``, plus ``forced`` once
        ``count >= limit - 1`` (``count >= limit`` when not ``early``). It
        never raises.

    The transition decision is already made before a PRE_TRANSITION handler
    fires, so with ``early`` the limiter triggers one iteration early
    (``>= max - 1``) and the forced transition fires on the next iteration
    rather than overshooting by 1. The forced keys only change routing where
    a transition reads them. PRE_TRANSITION handlers do not run on a BLOCKED
    turn, so the run's step ceiling (core ``max_steps``, set by
    ``BaseAgent._run_budgets``) is the backstop bound.
    """
    forced_updates = dict(forced)

    def check_iteration_limit(context: dict[str, Any]) -> dict[str, Any]:
        count = context.get(ContextKeys.ITERATION_COUNT, 0) + 1
        limit = (
            context.get(context_max_key, max_iterations)
            if context_max_key is not None
            else max_iterations
        )
        logger.debug(LogMessages.ITERATION.format(current=count, max=limit))
        if count >= (limit - 1 if early else limit):
            return {ContextKeys.ITERATION_COUNT: count, **forced_updates}
        return {ContextKeys.ITERATION_COUNT: count}

    return check_iteration_limit


# DECISION plan-2026-09-24T045559-3e4eb3e5/D-013: core extracts a key only
# while it is unset (pipeline skip-if-set), so a draft or verdict left in
# context is never re-judged and
# a redo loop replays round 1. Do NOT clear these keys on entry to the JUDGING
# state instead: that erases the limiter's forced pass (PRE_TRANSITION runs
# before entry) and the loop runs to the 3x ceiling. Do NOT delete the draft
# without stashing it: the redo state's prompt must still see it.
def make_redraft_handlers(
    draft_key: str,
    previous_key: str,
    *,
    clear_on_exit: tuple[str, ...] = (),
) -> tuple[
    Callable[[dict[str, Any]], dict[str, Any]],
    Callable[[dict[str, Any]], dict[str, Any]],
]:
    """
    Build the (entry, exit) handler pair for a state that redoes a draft.

    Args:
        draft_key: The key the redo state must re-extract.
        previous_key: Visible key the old draft is moved to on entry.
        clear_on_exit: Keys deleted on exit, once the redo state has
            consumed them (e.g. feedback the next judge must rewrite).

    Returns:
        ``(on_entry, on_exit)``, for ``on_state_entry`` and ``on_state_exit``
        of the redo state. Each returns a context delta in which ``None``
        deletes a key; neither raises. ``on_exit`` restores ``draft_key``
        from ``previous_key`` when the redo state produced no new draft.
    """

    def on_entry(context: dict[str, Any]) -> dict[str, Any]:
        delta: dict[str, Any] = {draft_key: None}
        if context.get(draft_key) is not None:
            delta[previous_key] = context[draft_key]
        return delta

    def on_exit(context: dict[str, Any]) -> dict[str, Any]:
        delta: dict[str, Any] = dict.fromkeys(clear_on_exit)
        if context.get(draft_key) is None:
            delta[draft_key] = context.get(previous_key)
        return delta

    return on_entry, on_exit


def recorded_forced_reason(context: Mapping[str, Any]) -> str | None:
    """The ``StopReason.FORCED`` value a forcing handler recorded, else None.

    Interface contract (callers: :func:`is_forced_verdict`,
    ``AgentHandlers.check_iteration_limit``, ``BaseAgent._forced_stop_reason``,
    the Debate limiter): reads ``forced_stop_reason`` and returns it only when
    it is a string in ``StopReason.FORCED``. Never raises.
    """
    reason = context.get(ContextKeys.FORCED_STOP_REASON)
    return reason if isinstance(reason, str) and reason in StopReason.FORCED else None


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
    (``fsm_definitions._conclude_on_evidence_logic``), so this turn takes
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


# DECISION plan-2026-09-29T103145-06a5ec0a/D-008: loop freshness is ONE rule,
# built here. Register the handler on entry to the PRODUCING (redo) state,
# never on entry to the judging state: PRE_TRANSITION runs before entry, so
# that would erase the limiter's forced verdict (3e4eb3e5 D-013). Do NOT
# clear a forced True (see is_forced_verdict): the forced edge must still
# route. Do NOT stash under an internal-prefixed or unlisted key: the stash
# must stay prompt-visible and out of results (c1d5bfbc D-012).
def make_fresh_keys_handler(
    keys: Iterable[str],
    *,
    stash: Mapping[str, str] | None = None,
    keep_limiter_forced: bool = True,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build an entry handler that makes a producing state re-extract ``keys``.

    Core extracts a key only while it is unset (skip-if-set, typed
    ``field_extractions`` included), so a loop value left in context freezes
    after round 1. Clearing it on entry to the state that produces it lets
    that state's extraction run again.

    Interface contract (ReAct think, Reflexion, PlanExecute, Debate,
    PromptChain; register with ``on_state_entry(<producing state>)``; a key
    another state writes for this one to read, such as ``agent_feedback``, is
    cleared with ``on_state_exit(<consuming state>)`` instead, D-029):
        - ``keys``: non-empty; the context keys the producing state writes.
        - ``stash``: optional ``{key: stash_key}`` for keys whose previous
          value the producing prompt must still see. Every ``key`` must be in
          ``keys`` and every ``stash_key`` in ``RESULT_DROPPED_CONTEXT_KEYS``
          (visible to prompts, dropped from results).
        - Returns a handler whose delta sets each listed key that holds a
          value to ``None`` (core deletes it); a stashed key's value is also
          copied to its stash key. Unset keys are left out of the delta, so an
          older stash survives a round that produced nothing. A key holding a
          forced ``True`` (:func:`is_forced_verdict`) is left untouched. The
          handler never raises.
        - ``keep_limiter_forced``: ``False`` keeps a ``True`` only when a
          forcing handler recorded a reason (:func:`recorded_forced_reason`);
          a ``True`` forced by the bare ``max_iterations_reached`` flag is
          cleared. Only for a state whose forced edge reads the flag itself
          (ReAct-family think, D-051 of plan 06a5ec0a).
        - Raises ``ValueError`` at build time for empty ``keys`` or an invalid
          ``stash`` mapping.
    """
    fresh = tuple(dict.fromkeys(keys))
    if not fresh:
        raise ValueError("make_fresh_keys_handler needs at least one key")
    stash_map = dict(stash or {})
    unknown = sorted(set(stash_map) - set(fresh))
    if unknown:
        raise ValueError(f"stash names keys that are not refreshed: {unknown}")
    unlisted = sorted(set(stash_map.values()) - RESULT_DROPPED_CONTEXT_KEYS)
    if unlisted:
        raise ValueError(
            f"stash keys must be in RESULT_DROPPED_CONTEXT_KEYS: {unlisted}"
        )

    def kept(value: Any, context: dict[str, Any]) -> bool:
        if keep_limiter_forced:
            return is_forced_verdict(value, context)
        return value is True and recorded_forced_reason(context) is not None

    def refresh_keys(context: dict[str, Any]) -> dict[str, Any]:
        delta: dict[str, Any] = {}
        for key in fresh:
            value = context.get(key)
            if value is None or kept(value, context):
                continue
            delta[key] = None
            if key in stash_map:
                delta[stash_map[key]] = value
        return delta

    return refresh_keys
