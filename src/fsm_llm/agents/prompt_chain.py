"""
PromptChainAgent -- Fixed sequential pipeline with validation gates.

Chains a user-defined list of LLM steps, each with optional validation
gates that can short-circuit the pipeline on failure.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.logging import logger

from .base import BaseAgent
from .constants import (
    ContextKeys,
    Defaults,
    ErrorMessages,
    HandlerNames,
    HandlerPriorities,
    PromptChainStates,
    StopReason,
)
from .definitions import AgentConfig, AgentResult, ChainStep
from .exceptions import AgentError
from .fsm_definitions import build_prompt_chain_fsm
from .handlers import make_fresh_keys_handler, make_iteration_limiter


class PromptChainAgent(BaseAgent):
    """
    Sequential prompt chaining agent.

    Executes a fixed pipeline of LLM steps in order, with optional
    validation gates between steps that can terminate early on failure.

    Usage::

        from fsm_llm.agents import PromptChainAgent, ChainStep

        chain = [
            ChainStep(
                step_id="outline",
                name="Generate outline",
                extraction_instructions="Extract a structured outline as JSON.",
                response_instructions="Present the outline clearly.",
            ),
            ChainStep(
                step_id="draft",
                name="Write draft",
                extraction_instructions="Extract the full draft text.",
                response_instructions="Write a complete draft from the outline.",
                validation_fn=lambda ctx: len(ctx.get("chain_step_result", "")) > 50,
            ),
        ]

        agent = PromptChainAgent(chain=chain)
        result = agent.run("Write an essay about climate change.")
        print(result.answer)
    """

    def __init__(
        self,
        chain: list[ChainStep],
        config: AgentConfig | None = None,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize a prompt chain agent.

        :param chain: Ordered list of ChainStep definitions
        :param config: Agent configuration (defaults to AgentConfig())
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        if not chain:
            raise AgentError(ErrorMessages.EMPTY_CHAIN)

        super().__init__(config, **api_kwargs)
        self.chain = list(chain)

        logger.info(
            f"PromptChainAgent initialized with {len(self.chain)} steps, model={self.config.model}"
        )

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Run the chain on a task.

        :param task: The task/question for the agent to process
        :param initial_context: Optional initial context data
        :return: AgentResult with answer, trace, and metadata
        """
        # Build FSM from chain definition
        fsm_def = build_prompt_chain_fsm(
            self.chain,
            task_description=task,
        )

        # Build initial context
        context = self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.CHAIN_RESULTS: [],
                ContextKeys.CHAIN_STEP_INDEX: 0,
            },
        )

        # Hard ceiling on iterations based on chain length
        max_fsm_iterations = len(self.chain) * Defaults.FSM_BUDGET_MULTIPLIER

        return self._standard_run(
            task,
            fsm_def,
            context,
            "prompt_chain",
            max_iterations=max_fsm_iterations,
            # Success key only: the last step's output is kept (not cleared on
            # output entry), so a chain whose last step produced nothing
            # reports no_result. The answer comes from _extract_answer.
            extra_answer_keys=[ContextKeys.CHAIN_STEP_RESULT],
        )

    def _register_handlers(self, api: API) -> None:
        """Register chain-specific handlers with the API."""
        # On entry to step i (i >= 1) and to output, the gate checker records
        # the previous step's result and runs its gate. step_0 is the initial
        # state: nothing precedes it.
        targets = [
            f"{PromptChainStates.STEP_PREFIX}{i}" for i in range(1, len(self.chain))
        ]
        targets.append(PromptChainStates.OUTPUT)
        for index, state_id in enumerate(targets, start=1):
            suffix = "final" if state_id == PromptChainStates.OUTPUT else index
            api.register_handler(
                api.create_handler(f"{HandlerNames.CHAIN_GATE_CHECKER}_{suffix}")
                .with_priority(HandlerPriorities.TOOL_EXECUTOR)
                .on_state_entry(state_id)
                .do(self._make_gate_checker(index))
            )

        # Iteration limiter
        self._register_iteration_limiter(api, self._make_iteration_limiter())

    def _make_gate_checker(self, step_index: int) -> Any:
        """Create the entry handler of the state after step ``step_index - 1``.

        ``step_index`` is 1..len(chain); ``len(chain)`` is the output state.
        """
        chain = self.chain
        prev_step = chain[step_index - 1] if step_index > 0 else None
        # The next step extracts its own result (skip-if-set); on output the
        # last step's result stays as the success key.
        refresh = (
            None
            if step_index == len(chain)
            else make_fresh_keys_handler([ContextKeys.CHAIN_STEP_RESULT])
        )

        # DECISION plan-2026-09-29T103145-06a5ec0a/D-038
        # The gate runs on ENTRY to the next state, after core recorded the
        # step's chain_step_result; a failure writes forced_stop_reason
        # gate_failed and KEEPS chain_step_result, so the next state's turn
        # skips its extraction (skip-if-set) and its gate edge routes to
        # output. Do NOT move the gate to PRE_TRANSITION of the step (core
        # has already chosen the edge, D-031) or to CONTEXT_UPDATE (never
        # fires on a null extraction), and do NOT route on gate_passed:
        # caller context can seed it, forced_stop_reason it cannot.
        def check_gate(context: dict[str, Any]) -> dict[str, Any]:
            if context.get(ContextKeys.FORCED_STOP_REASON) == StopReason.GATE_FAILED:
                return {}  # an earlier gate stopped the chain
            updates: dict[str, Any] = {ContextKeys.CHAIN_STEP_INDEX: step_index}
            step_result = context.get(ContextKeys.CHAIN_STEP_RESULT)
            if step_result is not None:
                chain_results = list(context.get(ContextKeys.CHAIN_RESULTS, []))
                chain_results.append(step_result)
                updates[ContextKeys.CHAIN_RESULTS] = chain_results

            if prev_step is not None and prev_step.validation_fn is not None:
                passed = prev_step.validation_fn(context)
                updates[ContextKeys.GATE_PASSED] = passed
                if not passed:
                    logger.warning(
                        f"Gate failed at step '{prev_step.name}', terminating chain"
                    )
                    updates[ContextKeys.FORCED_STOP_REASON] = StopReason.GATE_FAILED
                    return updates

            if refresh is not None:
                updates.update(refresh(context))
            return updates

        return check_gate

    def _make_iteration_limiter(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Create the iteration limiter handler (shared rule, see handlers)."""
        return make_iteration_limiter(
            len(self.chain) * Defaults.FSM_BUDGET_MULTIPLIER,
            {ContextKeys.SHOULD_TERMINATE: True},
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

        # Fall back to last chain step result
        chain_results = final_context.get(ContextKeys.CHAIN_RESULTS, [])
        if chain_results:
            last = chain_results[-1]
            if isinstance(last, str) and len(last.strip()) > Defaults.MIN_ANSWER_LENGTH:
                return last.strip()

        # Fall back to last non-empty response
        for response in reversed(responses):
            if response and len(response.strip()) > Defaults.MIN_ANSWER_LENGTH:
                return response.strip()

        return "Agent could not determine an answer."
