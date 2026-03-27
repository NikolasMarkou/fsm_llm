from __future__ import annotations

"""
SelfConsistencyAgent -- Parallel scaling via multiple generations + majority vote.

Generates multiple independent answers to the same task at varying
temperatures, then aggregates them via majority vote (or a custom
aggregation function).
"""

import math
from collections import Counter
from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.logging import logger

from .base import BaseAgent
from .constants import (
    ContextKeys,
    Defaults,
    ErrorMessages,
    LogMessages,
)
from .definitions import AgentConfig, AgentResult, AgentTrace
from .exceptions import AgentError
from .fsm_definitions import build_self_consistency_fsm


def _majority_vote(samples: list[str]) -> str:
    """Default aggregation: return the most common answer."""
    if not samples:
        return ""
    # Normalize whitespace for comparison
    normalized = [s.strip() for s in samples if s and s.strip()]
    if not normalized:
        return ""
    counter = Counter(normalized)
    return counter.most_common(1)[0][0]


class SelfConsistencyAgent(BaseAgent):
    """
    Self-consistency agent using parallel sampling and majority vote.

    Runs the same prompt N times at different temperatures, then
    selects the final answer via majority vote or a user-supplied
    aggregation function.

    Usage::

        from fsm_llm_agents import SelfConsistencyAgent

        agent = SelfConsistencyAgent(num_samples=5)
        result = agent.run("What is the capital of France?")
        print(result.answer)

        # Custom aggregation
        agent = SelfConsistencyAgent(
            num_samples=7,
            aggregation_fn=lambda samples: max(samples, key=len),
        )
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        num_samples: int = Defaults.NUM_SAMPLES,
        aggregation_fn: Callable[[list[str]], str] | None = None,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize a self-consistency agent.

        :param config: Agent configuration (defaults to AgentConfig())
        :param num_samples: Number of independent generations (must be >= 1)
        :param aggregation_fn: Custom aggregation function (default: majority vote)
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        if num_samples < 1:
            raise AgentError(ErrorMessages.NO_SAMPLES)

        super().__init__(config, **api_kwargs)
        self.num_samples = num_samples
        self.aggregation_fn = aggregation_fn or _majority_vote

        logger.info(
            f"SelfConsistencyAgent initialized with {self.num_samples} samples, model={self.config.model}"
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
        :return: AgentResult with aggregated answer
        """
        import time

        start_time = time.monotonic()

        # Build simple single-state FSM
        fsm_def = build_self_consistency_fsm(
            task_description=task[: Defaults.MAX_TASK_PREVIEW_LENGTH]
        )

        # Compute temperatures spread across the sample range
        temp_low, temp_high = Defaults.SAMPLE_TEMPERATURE_RANGE
        if self.num_samples == 1:
            temperatures = [(temp_low + temp_high) / 2]
        else:
            temperatures = [
                temp_low + (temp_high - temp_low) * i / (self.num_samples - 1)
                for i in range(self.num_samples)
            ]

        log = logger.bind(package="fsm_llm_agents", agent_type="self_consistency")

        # Collect samples
        samples: list[str] = []
        confidences: list[float] = []

        for sample_idx in range(self.num_samples):
            # Check time budget
            self._check_budgets(start_time, sample_idx)

            temp = temperatures[sample_idx]
            logger.debug(
                f"Generating sample {sample_idx + 1}/{self.num_samples} at temperature={temp:.2f}"
            )

            try:
                answer, confidence = self._generate_single(
                    fsm_def, task, temp, initial_context
                )
                samples.append(answer)
                confidences.append(confidence)
            except Exception as e:
                # Per-sample exception handling is intentional: unlike other
                # agents, SelfConsistency benefits from partial results.
                # If 3 of 5 samples succeed, the majority vote is still valid.
                logger.warning(f"Sample {sample_idx + 1} failed: {e!s}", exc_info=True)
                continue

        if not samples:
            raise AgentError(
                "All samples failed",
                details={"task": task, "num_samples": self.num_samples},
            )

        # Aggregate
        aggregated = self.aggregation_fn(samples)

        log.info(LogMessages.AGENT_COMPLETE.format(iterations=len(samples)))

        trace = AgentTrace(
            tool_calls=[],
            total_iterations=len(samples),
        )

        structured = self._try_parse_structured_output(aggregated)

        return AgentResult(
            answer=aggregated,
            success=True,
            trace=trace,
            final_context={
                ContextKeys.SAMPLES: samples,
                ContextKeys.AGGREGATED_ANSWER: aggregated,
                ContextKeys.CONFIDENCE: (
                    sum(confidences) / len(confidences) if confidences else 0.0
                ),
                ContextKeys.TASK: task,
            },
            structured_output=structured,
        )

    def _register_handlers(self, api: API) -> None:
        """No handlers needed for self-consistency (single-state FSM)."""

    def _generate_single(
        self,
        fsm_def: dict[str, Any],
        task: str,
        temperature: float,
        initial_context: dict[str, Any] | None,
    ) -> tuple[str, float]:
        """
        Run a single generation and return (answer, confidence).

        :param fsm_def: The FSM definition dict
        :param task: The task string
        :param temperature: Temperature for this sample
        :param initial_context: Optional initial context
        :return: Tuple of (answer_string, confidence_float)
        """
        api = API.from_definition(
            fsm_def,
            model=self.config.model,
            temperature=temperature,
            max_tokens=self.config.max_tokens,
            **self._api_kwargs,
        )

        context: dict[str, Any] = dict(initial_context) if initial_context else {}
        context[ContextKeys.TASK] = task

        conv_id, initial_response = api.start_conversation(context)

        try:
            # The FSM is a single terminal state, so it should end immediately
            # after start_conversation. If not, do a few iterations.
            responses = [initial_response]
            max_iters = 5
            iteration = 0

            while not api.has_conversation_ended(conv_id) and iteration < max_iters:
                iteration += 1
                response = api.converse(Defaults.CONTINUE_MESSAGE, conv_id)
                responses.append(response)

            final_context = api.get_data(conv_id)

            # Extract answer
            answer = final_context.get(ContextKeys.FINAL_ANSWER, "")
            if not answer or not isinstance(answer, str) or len(answer.strip()) < 3:
                # Fall back to last response
                for resp in reversed(responses):
                    if resp and len(resp.strip()) > 3:
                        answer = resp.strip()
                        break

            confidence = final_context.get(ContextKeys.CONFIDENCE, 0.5)
            if (
                not isinstance(confidence, int | float)
                or math.isnan(confidence)
                or math.isinf(confidence)
            ):
                confidence = 0.5

            return str(answer), float(confidence)

        finally:
            api.end_conversation(conv_id)
