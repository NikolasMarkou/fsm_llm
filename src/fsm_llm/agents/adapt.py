"""
ADaPTAgent — Adaptive Decomposition and Planning for Tasks.

Attempts tasks directly first, decomposes on failure, recursion bounded by max_depth.
FSM: attempt -> assess -> combine | assess -> decompose -> [recursive run()] -> combine
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .base import BaseAgent
from .constants import (
    ADaPTStates,
    ContextKeys,
    Defaults,
    HandlerNames,
    HandlerPriorities,
    LogMessages,
    StopReason,
)
from .definitions import AgentConfig, AgentResult
from .exceptions import AgentError
from .fsm_definitions import build_adapt_fsm
from .handlers import RunEndingErrorHolder, make_iteration_limiter
from .tools import ToolRegistry


class ADaPTAgent(BaseAgent):
    """
    ADaPT agent: attempt first, decompose recursively only on failure.

    Usage::

        agent = ADaPTAgent(max_depth=3)
        result = agent.run("Explain how neural networks learn")
        print(result.answer)
    """

    def __init__(
        self,
        tools: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        max_depth: int = Defaults.MAX_DECOMPOSITION_DEPTH,
        **api_kwargs: Any,
    ) -> None:
        super().__init__(config, **api_kwargs)
        self.tools = tools
        self.max_depth = max_depth

        if tools is None:
            logger.info(
                f"ADaPTAgent started in LLM-only mode (no tools), "
                f"max_depth={max_depth}, model={self.config.model}"
            )
        else:
            logger.info(
                f"ADaPTAgent started with {len(tools)} tools, "
                f"max_depth={max_depth}, model={self.config.model}"
            )

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
        _depth: int = 0,
        _start_time: float | None = None,
    ) -> AgentResult:
        """Run the ADaPT agent. _depth is internal recursion tracking."""
        start_time = _start_time or time.monotonic()
        logger.debug(LogMessages.DECOMPOSITION.format(depth=_depth))

        fsm_def = build_adapt_fsm(
            registry=self.tools,
            task_description=task[: Defaults.MAX_TASK_PREVIEW_LENGTH],
            max_depth=self.max_depth,
        )

        # Create API instance
        api = self._create_api(fsm_def)

        # Call-local: a sub-run's budget/timeout error rides out of the
        # subtask handler here (plan-2026-09-29T103145-06a5ec0a/D-014).
        holder = RunEndingErrorHolder()

        # Register handlers (needs initial_context, depth and start_time for subtask executor)
        self._register_handlers(api, initial_context, _depth, start_time, holder=holder)

        context = self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.CURRENT_DEPTH: _depth,
                ContextKeys.SUBTASK_RESULTS: [],
                "_max_iterations": self.config.max_iterations,
            },
        )

        try:
            responses, final_context, iteration = self._run_conversation_loop(
                api, context, start_time, "adapt"
            )
            # DECISION plan-2026-09-29T103145-06a5ec0a/D-014
            # Re-raise a subtask's AgentTimeoutError/BudgetExhaustedError HERE,
            # after the loop. Do NOT re-raise inside the subtask handler: core's
            # handler error mode "continue" swallows it (or wraps it as a
            # non-AgentError HandlerExecutionError). Do NOT keep the holder on
            # self: recursive and concurrent runs share this agent.
            holder.raise_if_set()

            answer = self._extract_answer(final_context, responses)
            trace = self._build_trace(final_context, iteration)

            # DECISION plan_2026-05-31_03830272/D-001 [STALE]: do NOT hard-code
            # success=True. A run that looped to the iteration limit without
            # setting final_answer and without executing a tool is degenerate —
            # the answer is _extract_answer's prose/JSON fallback. Apply the
            # _completion_is_real guard (final answer key OR a real tool call) so
            # leaked filler is success=False. ADaPT also legitimately completes
            # via a SUCCEEDED attempt (attempt_result, no separate final_answer)
            # — mirror _extract_answer's secondary source by counting
            # attempt_result as an answer key ONLY when attempt_succeeded is true
            # (a FAILED attempt's attempt_result is partial/garbage, not a real
            # completion).
            subtask_results = self._subtask_entries(final_context)
            forced = self._forced_stop_reason(final_context)
            stop_reason: str
            if forced is not None:
                # Same rule as every FSM pattern (D-011): a forced stop ships
                # its answer with success=False.
                success, stop_reason = False, forced
            elif subtask_results:
                # A decomposed run succeeds on its subtasks, not on the combine
                # text: AND needs every executed subtask, OR needs one.
                oks = [bool(entry.get("success")) for entry in subtask_results]
                operator = self._normalize_operator(final_context.get("operator"))
                success = any(oks) if operator == "OR" else all(oks)
                stop_reason = StopReason.EVIDENCE if success else StopReason.NO_RESULT
                if not success:
                    logger.warning(
                        f"ADaPT decomposition failed ({operator}: "
                        f"{sum(oks)}/{len(oks)} subtasks succeeded); "
                        "marking success=False."
                    )
            else:
                attempt_keys = (
                    [ContextKeys.ATTEMPT_RESULT]
                    if final_context.get(ContextKeys.ATTEMPT_SUCCEEDED)
                    else None
                )
                success, stop_reason = self._run_outcome(
                    final_context, trace, attempt_keys
                )
            if not success and not subtask_results and forced is None:
                logger.warning(
                    "ADaPT completed with no final_answer and no tool calls — "
                    "answer is fallback-only; marking success=False."
                )

            return AgentResult(
                answer=answer,
                success=success,
                stop_reason=stop_reason,
                trace=trace,
                final_context=self._filter_context(final_context),
            )

        except AgentError:
            raise
        except Exception as e:
            raise AgentError(
                f"ADaPT execution failed: {e}",
                details={"task": task, "depth": _depth},
            ) from e

    def _register_handlers(
        self,
        api: API,
        initial_context: dict[str, Any] | None = None,
        depth: int = 0,
        start_time: float | None = None,
        holder: RunEndingErrorHolder | None = None,
    ) -> None:
        """Register ADaPT handlers with the API.

        ``holder`` is the calling ``run()``'s error holder; ``run()`` re-raises
        what the subtask executor captures in it. Optional only to keep the
        ``BaseAgent._register_handlers(api)`` override shape: without one a
        captured error still stops further subtasks but nothing re-raises it.
        """
        if holder is None:
            holder = RunEndingErrorHolder()
        # Depth tracker: logs decomposition events
        api.register_handler(
            api.create_handler(HandlerNames.ADAPT_ASSESSOR)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(ADaPTStates.DECOMPOSE)
            .do(self._track_decomposition)
        )

        # Subtask executor: intercepts DECOMPOSE->COMBINE transition,
        # runs recursive subtasks, and injects results before COMBINE
        api.register_handler(
            api.create_handler("subtask_executor")
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .at(HandlerTiming.PRE_TRANSITION)
            .on_state(ADaPTStates.DECOMPOSE)
            .do(
                self._make_subtask_executor(
                    initial_context, depth, start_time, holder=holder
                )
            )
        )

        self._register_iteration_limiter(api, self._make_iteration_limiter())

    def _execute_subtasks(
        self,
        subtasks: list[Any],
        operator: str,
        depth: int,
        initial_context: dict[str, Any] | None,
        start_time: float | None = None,
        *,
        holder: RunEndingErrorHolder,
    ) -> list[dict[str, Any]]:
        """
        Recursively execute subtasks via self.run(). AND=all, OR=first success.

        Recursive self.run() creates a fresh FSM + handler set per subtask,
        ensuring proper isolation. Depth is bounded by max_depth, fan-out by
        ``Defaults.ADAPT_MAX_SUBTASKS``. A sub-run's ``AgentTimeoutError`` /
        ``BudgetExhaustedError`` goes into ``holder`` and no further subtask
        starts; any other exception becomes a failed entry.
        """
        results: list[dict[str, Any]] = []
        cap = Defaults.ADAPT_MAX_SUBTASKS
        if len(subtasks) > cap:
            logger.warning(
                f"ADaPT decomposition produced {len(subtasks)} subtasks at "
                f"depth {depth}; running the first {cap}."
            )
            subtasks = subtasks[:cap]

        for i, subtask in enumerate(subtasks):
            if holder.is_set:
                break
            subtask_str = str(subtask)
            logger.debug(
                f"ADaPT subtask {i + 1}/{len(subtasks)} at depth {depth}: "
                f"{subtask_str[:100]}"
            )

            try:
                sub_result = self.run(
                    task=subtask_str,
                    initial_context=initial_context,
                    _depth=depth,
                    _start_time=start_time or time.monotonic(),
                )
                results.append(
                    {
                        "subtask": subtask_str,
                        "answer": sub_result.answer,
                        "success": sub_result.success,
                        "depth": depth,
                    }
                )

                # For OR operator, stop on first success
                if self._normalize_operator(operator) == "OR" and sub_result.success:
                    break

            except Exception as e:
                if holder.capture(e):
                    logger.warning(
                        f"ADaPT subtask hit the run budget at depth {depth}; "
                        f"no further subtasks start: {e}"
                    )
                    break
                # Any other subtask failure must not crash the parent.
                logger.warning(
                    f"ADaPT subtask failed at depth {depth}: {e}", exc_info=True
                )
                results.append(
                    {
                        "subtask": subtask_str,
                        "answer": f"Subtask error: {e}",
                        "success": False,
                        "depth": depth,
                    }
                )

        return results

    def _make_subtask_executor(
        self,
        initial_context: dict[str, Any] | None,
        depth: int,
        start_time: float | None = None,
        *,
        holder: RunEndingErrorHolder,
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create handler that executes subtasks during DECOMPOSE->COMBINE transition.

        Fires as a PRE_TRANSITION handler when leaving the DECOMPOSE state.
        If subtasks were extracted, runs them recursively and injects results
        into context so the COMBINE state can synthesize them.
        """
        agent = self

        def execute_subtasks(context: dict[str, Any]) -> dict[str, Any]:
            subtasks_raw = context.get(ContextKeys.SUBTASKS)
            current_depth = context.get(ContextKeys.CURRENT_DEPTH, depth)

            if (
                not subtasks_raw
                or not isinstance(subtasks_raw, list)
                or current_depth >= agent.max_depth
                or holder.is_set
            ):
                return {}

            # Use the start_time captured from the parent run() call rather than
            # reading self._current_start_time, which is overwritten when
            # recursive subtask calls re-enter run() (A-ISSUE-005).
            subtask_results = agent._execute_subtasks(
                subtasks=subtasks_raw,
                operator=agent._normalize_operator(context.get("operator")),
                depth=current_depth + 1,
                initial_context=initial_context,
                start_time=start_time,
                holder=holder,
            )

            return {
                ContextKeys.SUBTASK_RESULTS: subtask_results,
                ContextKeys.SUBTASKS: None,
            }

        return execute_subtasks

    def _track_decomposition(self, context: dict[str, Any]) -> dict[str, Any]:
        """Track decomposition events. POST_TRANSITION on 'decompose'."""
        current_depth = context.get(ContextKeys.CURRENT_DEPTH, 0)

        logger.info(LogMessages.DECOMPOSITION.format(depth=current_depth))

        # Track in agent trace
        trace = context.get(ContextKeys.AGENT_TRACE, [])
        if not isinstance(trace, list):
            trace = []
        # "event", not "action": _build_trace turns every "action" entry into a
        # ToolCall, and a decomposition is not a tool call (no success evidence).
        trace.append(
            {
                "event": "decompose",
                "depth": current_depth,
                "max_depth": self.max_depth,
            }
        )

        return {ContextKeys.AGENT_TRACE: trace}

    def _make_iteration_limiter(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create the iteration limiter handler (shared rule, see handlers)."""
        # should_terminate alone routes to combine (priority-1 transition);
        # forcing attempt_succeeded=False here biases toward decompose at the
        # budget edge, so it is intentionally omitted.
        return make_iteration_limiter(
            Defaults.MAX_ITERATIONS,
            {ContextKeys.SHOULD_TERMINATE: True},
            context_max_key="_max_iterations",
        )

    def _extract_answer(
        self,
        final_context: dict[str, Any],
        responses: list[str],
        extra_keys: list[str] | None = None,
    ) -> str:
        """Extract the final answer from context or responses."""
        answer = final_context.get(ContextKeys.FINAL_ANSWER)
        if (
            answer
            and isinstance(answer, str)
            and len(answer) > Defaults.MIN_ANSWER_LENGTH
        ):
            return str(answer)

        # A decomposed run answers from its subtasks, never the failed attempt
        # that caused the decomposition.
        subtask_results = self._subtask_entries(final_context)
        if subtask_results:
            parts = [
                str(entry.get("answer")).strip()
                for entry in subtask_results
                if entry.get("success") and str(entry.get("answer") or "").strip()
            ]
            if parts:
                return "\n\n".join(parts)

        # Fall back to attempt_result if available
        attempt_result = final_context.get(ContextKeys.ATTEMPT_RESULT)
        if (
            not subtask_results
            and attempt_result
            and isinstance(attempt_result, str)
            and len(attempt_result) > Defaults.MIN_ANSWER_LENGTH
        ):
            return str(attempt_result)

        # DECISION plan_2026-05-31_03830272/D-001 [STALE]: skip responses that are the
        # raw bulk-extraction envelope ({"extracted_data": ...}). On weak models
        # that internal Pass-2 JSON can be the last response; returning it leaks
        # plumbing as the user-facing answer (the adapt JSON-leak bug).
        for response in reversed(responses):
            if (
                response
                and len(response.strip()) > Defaults.MIN_ANSWER_LENGTH
                and not self._is_extraction_envelope(response)
            ):
                return response.strip()

        return "ADaPT agent could not determine an answer."

    @staticmethod
    def _subtask_entries(final_context: dict[str, Any]) -> list[dict[str, Any]]:
        """The executed subtask entries of a decomposed run ([] if none ran)."""
        entries = final_context.get(ContextKeys.SUBTASK_RESULTS)
        if not isinstance(entries, list):
            return []
        return [entry for entry in entries if isinstance(entry, dict)]

    @staticmethod
    def _normalize_operator(value: Any) -> str:
        """``"OR"`` for any casing/padding of "or", else ``"AND"`` (the default)."""
        operator = str(value or "AND").strip().upper()
        if operator not in ("AND", "OR"):
            logger.warning(f"ADaPT operator {value!r} is not AND/OR; using AND.")
            return "AND"
        return operator

    @staticmethod
    def _is_extraction_envelope(text: str) -> bool:
        """True if *text* is the raw bulk-extraction envelope (internal plumbing).

        The 2-pass pipeline's bulk extractor emits ``{"extracted_data": {...}}``;
        on weak models that JSON can leak into a Pass-2 response. Such a string
        must never surface as the user-facing answer. Matches only a JSON object
        carrying the internal ``extracted_data`` key — prose that merely mentions
        the word is unaffected. Tolerates a single ```` ```json ```` code fence.
        """
        s = text.strip()
        if s.startswith("```"):
            s = s[3:]
            if s[:4].lower() == "json":
                s = s[4:]
            if s.endswith("```"):
                s = s[:-3]
            s = s.strip()
        if not (s.startswith("{") and s.endswith("}")):
            return False
        try:
            obj = json.loads(s)
        except (ValueError, TypeError):
            return False
        return isinstance(obj, dict) and "extracted_data" in obj
