from __future__ import annotations

"""
PromptChainAgent -- Fixed sequential pipeline with validation gates.

Chains a user-defined list of LLM steps, each with optional validation
gates that can short-circuit the pipeline on failure.
"""

from typing import Any

from fsm_llm.dialog.api import API
from fsm_llm.logging import logger

from .base import BaseAgent
from .constants import (
    ContextKeys,
    Defaults,
    ErrorMessages,
    HandlerNames,
    HandlerPriorities,
    PromptChainStates,
)
from .definitions import AgentConfig, AgentResult, ChainStep
from .exceptions import AgentError
from .fsm_definitions import build_prompt_chain_fsm


class PromptChainAgent(BaseAgent):
    """
    Sequential prompt chaining agent.

    Executes a fixed pipeline of LLM steps in order, with optional
    validation gates between steps that can terminate early on failure.

    Usage::

        from fsm_llm_agents import PromptChainAgent, ChainStep

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
        )

    def _register_handlers(self, api: API) -> None:
        """Register chain-specific handlers with the API."""
        # Gate checker: runs post-transition on every step state
        for i, _step in enumerate(self.chain):
            state_id = f"{PromptChainStates.STEP_PREFIX}{i}"

            api.register_handler(
                api.create_handler(f"{HandlerNames.CHAIN_GATE_CHECKER}_{i}")
                .with_priority(HandlerPriorities.TOOL_EXECUTOR)
                .on_state_entry(state_id)
                .do(self._make_gate_checker(i))
            )

        # Final gate: validate the last step's result on entry to output state
        api.register_handler(
            api.create_handler(f"{HandlerNames.CHAIN_GATE_CHECKER}_final")
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(PromptChainStates.OUTPUT)
            .do(self._make_gate_checker(len(self.chain)))
        )

        # Iteration limiter
        self._register_iteration_limiter(api, self._make_iteration_limiter())

    def _make_gate_checker(self, step_index: int) -> Any:
        """Create a gate checker handler for a specific step."""
        chain = self.chain

        def check_gate(context: dict[str, Any]) -> dict[str, Any]:
            # Record current step index
            updates: dict[str, Any] = {
                ContextKeys.CHAIN_STEP_INDEX: step_index,
            }

            # If the previous step had a result and validation_fn, check it
            prev_index = step_index - 1
            if prev_index >= 0 and prev_index < len(chain):
                prev_step = chain[prev_index]
                if prev_step.validation_fn is not None:
                    passed = prev_step.validation_fn(context)
                    updates[ContextKeys.GATE_PASSED] = passed
                    if not passed:
                        logger.warning(
                            f"Gate failed at step '{prev_step.name}', terminating chain"
                        )
                        updates[ContextKeys.SHOULD_TERMINATE] = True

            # Accumulate step result from previous step
            step_result = context.get(ContextKeys.CHAIN_STEP_RESULT)
            if step_result is not None:
                chain_results = list(context.get(ContextKeys.CHAIN_RESULTS, []))
                chain_results.append(step_result)
                updates[ContextKeys.CHAIN_RESULTS] = chain_results

            return updates

        return check_gate

    def _make_iteration_limiter(self) -> Any:
        """Create an iteration limiter handler."""
        max_iters = len(self.chain) * Defaults.FSM_BUDGET_MULTIPLIER

        def check_limit(context: dict[str, Any]) -> dict[str, Any]:
            count = context.get(ContextKeys.ITERATION_COUNT, 0) + 1
            result: dict[str, Any] = {ContextKeys.ITERATION_COUNT: count}
            if count >= max_iters:
                result[ContextKeys.SHOULD_TERMINATE] = True
            return result

        return check_limit

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
