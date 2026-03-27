from __future__ import annotations

"""
BaseAgent — Abstract base class for all fsm_llm agents.

Extracts the common conversation loop, budget enforcement, answer extraction,
trace building, and context filtering from the 12 agent implementations.
"""

import time
from abc import ABC, abstractmethod
from typing import Any

from fsm_llm import API
from fsm_llm.context import ContextCompactor
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .constants import (
    ContextKeys,
    Defaults,
    HandlerNames,
    HandlerPriorities,
    LogMessages,
)
from .definitions import AgentConfig, AgentResult, AgentTrace, ToolCall
from .exceptions import AgentError, AgentTimeoutError, BudgetExhaustedError


class BaseAgent(ABC):
    """Abstract base class for FSM-LLM agents.

    Provides the common conversation loop, budget enforcement, answer
    extraction, and trace building. Subclasses implement only the
    pattern-specific logic: FSM building, handler registration, and
    context setup.

    Usage for end-users is unchanged — all existing agent constructors
    and ``run()`` signatures are preserved.  Additionally, agents now
    support ``__call__``::

        result = agent("What is 2+2?")
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        **api_kwargs: Any,
    ) -> None:
        self.config = config or AgentConfig()
        self._api_kwargs = api_kwargs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run the agent on a task. Implemented by each agent pattern."""
        ...

    def __call__(
        self,
        task: str,
        **kwargs: Any,
    ) -> AgentResult:
        """Callable shorthand: ``agent("task")`` → ``agent.run("task")``."""
        return self.run(task, **kwargs)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model})"

    # ------------------------------------------------------------------
    # Common conversation loop
    # ------------------------------------------------------------------

    def _run_conversation_loop(
        self,
        api: API,
        context: dict[str, Any],
        start_time: float,
        agent_type: str,
        max_iterations: int | None = None,
    ) -> tuple[list[str], dict[str, Any], int]:
        """Run the standard FSM conversation loop.

        Returns:
            Tuple of (responses, final_context, iteration_count).
        """
        max_iters = max_iterations or self.config.max_iterations

        conv_id, initial_response = api.start_conversation(context)
        log = logger.bind(
            conversation_id=conv_id,
            package="fsm_llm_agents",
            agent_type=agent_type,
        )

        try:
            responses = [initial_response]
            iteration = 0

            while not api.has_conversation_ended(conv_id):
                iteration += 1

                self._check_budgets(start_time, iteration, max_iters)

                # Hook for mid-loop processing (e.g. HITL approval)
                self._on_loop_iteration(api, conv_id, iteration)

                response = api.converse(Defaults.CONTINUE_MESSAGE, conv_id)
                responses.append(response)

            final_context = api.get_data(conv_id)
            log.info(LogMessages.AGENT_COMPLETE.format(iterations=iteration))
            return responses, final_context, iteration

        finally:
            api.end_conversation(conv_id)

    def _on_loop_iteration(  # noqa: B027
        self,
        api: API,
        conv_id: str,
        iteration: int,
    ) -> None:
        """Hook called each loop iteration before converse().

        Override for HITL approval gates or other mid-loop logic.
        Default is a no-op.
        """

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def _check_budgets(
        self,
        start_time: float,
        iteration: int,
        max_iterations: int | None = None,
    ) -> None:
        """Raise if time or iteration budget exceeded."""
        if time.monotonic() - start_time > self.config.timeout_seconds:
            raise AgentTimeoutError(self.config.timeout_seconds)

        max_iters = max_iterations or self.config.max_iterations
        if iteration > max_iters * Defaults.FSM_BUDGET_MULTIPLIER:
            raise BudgetExhaustedError("iterations", max_iters)

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------

    def _extract_answer(
        self,
        final_context: dict[str, Any],
        responses: list[str],
        extra_keys: list[str] | None = None,
    ) -> str:
        """Extract answer with a fallback chain.

        1. Try ``ContextKeys.FINAL_ANSWER``
        2. Try each key in *extra_keys* (e.g. ``JUDGE_VERDICT``)
        3. Try responses in reverse order
        4. Return default message
        """
        # Primary: final_answer
        answer = final_context.get(ContextKeys.FINAL_ANSWER)
        if answer and isinstance(answer, str) and answer.strip():
            return str(answer)

        # Secondary: extra context keys (pattern-specific)
        for key in extra_keys or []:
            val = final_context.get(key)
            if val and isinstance(val, str) and val.strip():
                return str(val).strip()

        # Tertiary: last non-empty response
        for response in reversed(responses):
            if response and response.strip():
                return response.strip()

        return "Agent could not determine an answer."

    # ------------------------------------------------------------------
    # Trace building
    # ------------------------------------------------------------------

    def _build_trace(
        self,
        final_context: dict[str, Any],
        iteration: int,
    ) -> AgentTrace:
        """Build an AgentTrace from the AGENT_TRACE context entries.

        Agents that don't use tools get an empty trace (just iteration count).
        """
        trace_data = final_context.get(ContextKeys.AGENT_TRACE, [])
        trace = AgentTrace(
            tool_calls=[],
            total_iterations=final_context.get(ContextKeys.ITERATION_COUNT, iteration),
        )

        for step in trace_data:
            if isinstance(step, dict) and "action" in step:
                tool_name = step.get("action", "").split("(")[0]
                if tool_name and tool_name != "none":
                    trace.tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            parameters={},
                            reasoning=step.get("thought", ""),
                        )
                    )

        return trace

    # ------------------------------------------------------------------
    # Context filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_context(context: dict[str, Any]) -> dict[str, Any]:
        """Remove internal (``_``-prefixed) keys from context."""
        return {k: v for k, v in context.items() if not k.startswith("_")}

    # ------------------------------------------------------------------
    # API factory helper
    # ------------------------------------------------------------------

    def _create_api(self, fsm_def: dict[str, Any]) -> API:
        """Create an API instance from an FSM definition."""
        kwargs = dict(self._api_kwargs)
        if self.config.transition_config is not None:
            kwargs["transition_config"] = self.config.transition_config
        return API.from_definition(
            fsm_def,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Standard run() implementation
    # ------------------------------------------------------------------

    def _standard_run(
        self,
        task: str,
        fsm_def: dict[str, Any],
        context: dict[str, Any],
        agent_type: str,
        max_iterations: int | None = None,
        extra_answer_keys: list[str] | None = None,
    ) -> AgentResult:
        """Standard run() implementation shared by most agents.

        Handles API creation, handler registration, conversation loop,
        answer extraction, trace building, and error wrapping.
        """
        start_time = time.monotonic()
        api = self._create_api(fsm_def)
        self._register_handlers(api)

        # Register base handlers for lifecycle events
        api.register_handler(
            api.create_handler(HandlerNames.END_CONVERSATION)
            .with_priority(HandlerPriorities.END_CONVERSATION)
            .at(HandlerTiming.END_CONVERSATION)
            .do(
                lambda ctx: {
                    "_agent_completed": True,
                    "_agent_type": agent_type,
                }
            )
        )
        api.register_handler(
            api.create_handler(HandlerNames.ERROR)
            .with_priority(HandlerPriorities.ERROR)
            .at(HandlerTiming.ERROR)
            .do(
                lambda ctx: (
                    logger.warning(
                        f"Agent error in {agent_type}: "
                        f"state={ctx.get('_current_state', '?')}"
                    )
                    or {}
                )
            )
        )

        # Register context compactor to clean transient keys between iterations
        compactor = ContextCompactor(
            transient_keys={
                ContextKeys.TOOL_RESULT,
                ContextKeys.TOOL_STATUS,
                ContextKeys.TOOL_ERROR,
            },
        )
        api.register_handler(
            api.create_handler("AgentContextCompactor")
            .with_priority(HandlerPriorities.END_CONVERSATION)
            .at(HandlerTiming.PRE_PROCESSING)
            .do(compactor.compact)
        )

        try:
            responses, final_context, iteration = self._run_conversation_loop(
                api, context, start_time, agent_type, max_iterations
            )

            answer = self._extract_answer(final_context, responses, extra_answer_keys)
            trace = self._build_trace(final_context, iteration)

            structured = self._try_parse_structured_output(answer)

            return AgentResult(
                answer=answer,
                success=True,
                trace=trace,
                final_context=self._filter_context(final_context),
                structured_output=structured,
            )

        except (AgentTimeoutError, BudgetExhaustedError):
            raise
        except Exception as e:
            raise AgentError(
                f"{agent_type.title()} execution failed: {e}",
                details={"task": task},
            ) from e

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def _try_parse_structured_output(self, answer: str) -> Any:
        """Validate *answer* against ``config.output_schema`` if set.

        Returns a Pydantic model instance on success, ``None`` on failure
        or when no schema is configured.
        """
        schema = self.config.output_schema
        if schema is None:
            return None

        try:
            import json as _json

            from fsm_llm.utilities import extract_json_from_text

            data = extract_json_from_text(answer)
            if data is None:
                # Try direct JSON parse
                data = _json.loads(answer)

            if isinstance(data, dict):
                return schema(**data)

            logger.warning(
                f"Structured output: expected dict, got {type(data).__name__}"
            )
        except Exception as e:
            logger.warning(f"Structured output validation failed: {e}")

        return None

    @abstractmethod
    def _register_handlers(self, api: API) -> None:
        """Register pattern-specific handlers. Implemented by each agent."""
        ...
