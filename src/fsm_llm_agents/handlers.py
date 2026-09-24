"""
Agent-specific handlers for tool execution, iteration limiting,
observation tracking, and HITL gating.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.logging import logger

from .constants import ContextKeys, Defaults, LogMessages
from .definitions import AgentStep, ToolCall
from .tools import ToolRegistry, normalize_tool_input
from .truncation import smart_truncate


class AgentHandlers:
    """Collection of handler functions for agent FSM operations."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._current_iteration = 0
        self._consecutive_no_tool = 0

    def reset(self) -> None:
        """Reset handler state for a new run."""
        self._current_iteration = 0
        self._consecutive_no_tool = 0

    def _run_selected_tool(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the tool selected during the think state.

        Body of :meth:`execute_tool`, which adds approval consumption.
        """
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
                return {
                    ContextKeys.TOOL_RESULT: (
                        f"{miss}You must use at least one tool before concluding. "
                        f"Available: {', '.join(tool_names)}"
                    ),
                    ContextKeys.TOOL_STATUS: "rejected",
                    ContextKeys.SHOULD_TERMINATE: False,
                }

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
                }

            if not should_terminate or unknown:
                # Instructive warning to push model toward tool use
                tool_names = [t.name for t in self.registry.list_tools()]
                return {
                    ContextKeys.TOOL_RESULT: (
                        f"{miss}WARNING: No tool was called but the task is not complete. "
                        f"You must select a tool from: {', '.join(tool_names)}. "
                        "Do not answer from memory — use a tool to gather information."
                    ),
                    ContextKeys.TOOL_STATUS: "skipped",
                }

            return {
                ContextKeys.TOOL_RESULT: "No tool was selected.",
                ContextKeys.TOOL_STATUS: "skipped",
            }

        # Reset stall counter — a tool was actually selected
        self._consecutive_no_tool = 0

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
                if task and string_ok:
                    tool_input = {param_name: task}
                    logger.info(f"Recovered empty tool_input: {param_name}=<task>")

        logger.info(LogMessages.TOOL_SELECTED.format(name=tool_name, input=tool_input))

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

        step_num = len(observations) + 1
        observation_entry = (
            f"[Step {step_num}] Tool: {tool_name} | "
            f"Input: {tool_input} | "
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
        trace = context.get(ContextKeys.AGENT_TRACE, [])
        if not isinstance(trace, list):
            trace = []
        trace_step = AgentStep(
            iteration=step_num,
            thought=reasoning,
            action=f"{tool_name}({tool_input})",
            observation=observation,
        ).model_dump(mode="json")
        # Preserve structured tool input so _build_trace can recover parameters
        # (mirrors REWOOAgent, which stores tool_input on the trace dict).
        trace_step["tool_input"] = tool_input
        trace.append(trace_step)

        return {
            ContextKeys.TOOL_RESULT: observation,
            ContextKeys.TOOL_STATUS: "success" if result.success else "failed",
            ContextKeys.TOOL_ERROR: result.error,
            ContextKeys.OBSERVATIONS: observations,
            ContextKeys.OBSERVATION_COUNT: len(observations),
            ContextKeys.AGENT_TRACE: trace,
            # Clear tool selection for next iteration
            ContextKeys.TOOL_NAME: None,
            ContextKeys.TOOL_INPUT: None,
            ContextKeys.SHOULD_TERMINATE: None,
        }

    def execute_tool(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the tool selected during the think state.

        Called as a POST_TRANSITION handler when entering the 'act' state.
        """
        delta = self._run_selected_tool(context)
        # DECISION plan-2026-09-24T045559-3e4eb3e5/D-015
        # Do NOT let an approval outlive the single tool call it was granted
        # for. The driver asks only while approval_granted is unset, so a stale
        # True skipped the callback and routed await_approval -> act unasked.
        # Kept while approval_required is set (the call is still pending).
        pending = context.get(ContextKeys.APPROVAL_REQUIRED)
        if ContextKeys.APPROVAL_GRANTED in context and not pending:
            delta = {**delta, ContextKeys.APPROVAL_GRANTED: None}
        return delta

    def classification_tool_override(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Override should_terminate when classification confidently selects a tool.

        When use_classification=True, the pipeline stores the full classification
        result at ``_tool_name_classification``.  If the classifier picked a real
        tool but the free-form extraction set should_terminate=True, this handler
        clears should_terminate so the agent proceeds to ACT instead of CONCLUDE.
        """
        # Only act when should_terminate is True (potential conflict)
        if not context.get(ContextKeys.SHOULD_TERMINATE):
            return {}

        classification = context.get("_tool_name_classification")
        if not isinstance(classification, dict):
            return {}

        intent = classification.get("intent", ContextKeys.NO_TOOL)
        confidence = classification.get("confidence", 0.0)

        if intent != ContextKeys.NO_TOOL and confidence >= 0.5:
            logger.info(
                f"Classification override: tool={intent} "
                f"(confidence={confidence:.2f}), unsetting should_terminate"
            )
            return {ContextKeys.SHOULD_TERMINATE: False}

        return {}

    def check_iteration_limit(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Check if the iteration limit has been reached.

        Called as a PRE_TRANSITION handler. Since the transition decision
        is already made before this handler fires, we trigger one iteration
        early (>= max - 1) so the conclude transition fires on the next
        iteration rather than overshooting by 1.
        """
        self._current_iteration += 1
        max_iterations = context.get("_max_iterations", Defaults.MAX_ITERATIONS)

        logger.debug(
            LogMessages.ITERATION.format(
                current=self._current_iteration, max=max_iterations
            )
        )

        if self._current_iteration >= max_iterations - 1:
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
    fires, so the limiter triggers one iteration early (``>= max - 1``) and
    the forced transition fires on the next iteration rather than
    overshooting by 1. PRE_TRANSITION handlers do not run on a BLOCKED turn,
    so this is a near-limit nudge; the loop's hard ceiling
    (``BaseAgent._check_budgets``) is the real bound.
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
