"""
SelfConsistencyAgent -- Parallel scaling via multiple generations + majority vote.

Generates multiple independent answers to the same task at varying
temperatures, then aggregates them via majority vote (or a custom
aggregation function).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.logging import logger

from .base import (
    BaseAgent,
    pattern_run_output_keys,
    prompt_overflow_error,
    strip_caller_context,
    with_instructions,
)
from .constants import (
    ContextKeys,
    Defaults,
    ErrorMessages,
    LogMessages,
    StopReason,
)
from .definitions import AgentConfig, AgentResult, AgentTrace
from .exceptions import AgentError
from .fsm_definitions import build_self_consistency_fsm

_ANSWER_LINE = re.compile(
    r"^[ \t*_#>-]*(?:final[ \t]+)?answer[ \t*_]*:(.*)$", re.IGNORECASE | re.MULTILINE
)


def _vote_key(sample: str) -> str:
    """The comparable form of a sample: its last ``Answer:`` line value (the
    whole text when it has none), casefolded, whitespace-collapsed, with
    surrounding markdown emphasis and trailing sentence punctuation dropped.
    Never raises; a blank sample gives ``""``."""
    lines = [m.strip() for m in _ANSWER_LINE.findall(sample or "") if m.strip()]
    text = lines[-1] if lines else (sample or "")
    return " ".join(text.casefold().split()).strip(" *_.!")


# DECISION plan-2026-09-29T103145-06a5ec0a/D-037
# Vote on the normalized final answer, but RETURN a full sample (the first of
# the winning group). Do NOT return the bare vote key: callers and the
# examples show the answer as prose, and a one-word key ("canberra") loses
# the case and the reasoning. Do NOT vote on whole-text equality (PAT-05):
# samples that agree in different prose then never agree.
def _majority_vote(samples: list[str]) -> str:
    """Default aggregation: the first sample of the most common final answer.

    Contract: samples are compared by :func:`_vote_key`; blank samples never
    vote; ties go to the answer seen first. Returns that group's first sample
    stripped, or ``""`` when no sample has text. Never raises.
    """
    groups: dict[str, list[str]] = {}
    for sample in samples:
        key = _vote_key(sample)
        if key:
            groups.setdefault(key, []).append(sample.strip())
    if not groups:
        return ""
    return max(groups.values(), key=len)[0]


class SelfConsistencyAgent(BaseAgent):
    """
    Self-consistency agent using parallel sampling and majority vote.

    Runs the same prompt N times at different temperatures, then
    selects the final answer via majority vote or a user-supplied
    aggregation function.

    Usage::

        from fsm_llm.agents import SelfConsistencyAgent

        agent = SelfConsistencyAgent(num_samples=5)
        result = agent.run("What is the capital of France?")
        print(result.answer)

        # Custom aggregation
        agent = SelfConsistencyAgent(
            num_samples=7,
            aggregation_fn=lambda samples: max(samples, key=len),
        )
    """

    # Run outputs caller context may not seed (D-052 of plan 06a5ec0a).
    _run_output_keys = frozenset(
        {
            ContextKeys.SAMPLES,
            ContextKeys.AGGREGATED_ANSWER,
        }
    )

    def __init__(
        self,
        config: AgentConfig | None = None,
        num_samples: int = Defaults.NUM_SAMPLES,
        aggregation_fn: Callable[[list[str]], str] | None = None,
        max_workers: int = 1,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize a self-consistency agent.

        :param config: Agent configuration (defaults to AgentConfig())
        :param num_samples: Number of independent generations (must be >= 1)
        :param aggregation_fn: Custom aggregation function (default: majority vote)
        :param max_workers: Concurrency for sampling. ``1`` (default) keeps the
            original serial behavior exactly. ``>1`` runs samples in a thread
            pool; results are still assembled in sample order so the aggregation
            is deterministic and identical to serial.
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        if num_samples < 1:
            raise AgentError(ErrorMessages.NO_SAMPLES)
        if max_workers < 1:
            raise AgentError("max_workers must be >= 1")

        super().__init__(config, **api_kwargs)
        self.num_samples = num_samples
        self.aggregation_fn = aggregation_fn or _majority_vote
        self.max_workers = max_workers

        logger.info(
            f"SelfConsistencyAgent initialized with {self.num_samples} samples, "
            f"model={self.config.model}, max_workers={self.max_workers}"
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
        # Samples build their own context (no `_init_context`), so drop the
        # run-owned keys once here: a forged `final_answer` would otherwise
        # become every sample's answer (D-002 of plan 06a5ec0a).
        initial_context = strip_caller_context(
            initial_context,
            source="SelfConsistencyAgent initial_context",
            run_keys=pattern_run_output_keys(self),
        )

        # Build simple single-state FSM
        fsm_def = build_self_consistency_fsm(task_description=task)

        # Compute temperatures spread across the sample range
        temp_low, temp_high = Defaults.SAMPLE_TEMPERATURE_RANGE
        if self.num_samples == 1:
            temperatures = [(temp_low + temp_high) / 2]
        else:
            temperatures = [
                temp_low + (temp_high - temp_low) * i / (self.num_samples - 1)
                for i in range(self.num_samples)
            ]

        log = logger.bind(package="fsm_llm.agents", agent_type="self_consistency")

        # Collect samples (serial by default; parallel when max_workers > 1).
        if self.max_workers > 1 and self.num_samples > 1:
            samples = self._collect_parallel(
                fsm_def, task, temperatures, initial_context, start_time
            )
        else:
            samples = self._collect_serial(
                fsm_def, task, temperatures, initial_context, start_time
            )

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

        # No sample text, or an empty aggregate, is no result (D-011).
        answered = any(s.strip() for s in samples) and bool(
            str(aggregated or "").strip()
        )
        # Share of samples whose final answer matches the aggregate's.
        winner = _vote_key(str(aggregated or ""))
        votes = sum(1 for s in samples if winner and _vote_key(s) == winner)
        return AgentResult(
            answer=aggregated,
            success=answered,
            stop_reason=StopReason.ANSWERED if answered else StopReason.NO_RESULT,
            trace=trace,
            final_context={
                ContextKeys.SAMPLES: samples,
                ContextKeys.AGGREGATED_ANSWER: aggregated,
                ContextKeys.CONFIDENCE: votes / len(samples),
                ContextKeys.TASK: task,
            },
            structured_output=structured,
        )

    def _collect_serial(
        self,
        fsm_def: dict[str, Any],
        task: str,
        temperatures: list[float],
        initial_context: dict[str, Any] | None,
        start_time: float,
    ) -> list[str]:
        """Original serial sampling loop (max_workers == 1)."""
        samples: list[str] = []

        for sample_idx in range(self.num_samples):
            # Check time/iteration budget. Cap iterations at num_samples so a
            # large num_samples does not trip the FSM max_iterations*3 ceiling
            # (sampling is independent of FSM iterations).
            self._check_budgets(start_time, sample_idx, max_iterations=self.num_samples)

            temp = temperatures[sample_idx]
            logger.debug(
                f"Generating sample {sample_idx + 1}/{self.num_samples} at temperature={temp:.2f}"
            )

            try:
                samples.append(
                    self._generate_single(fsm_def, task, temp, initial_context)
                )
            except Exception as e:
                # Per-sample exception handling is intentional: unlike other
                # agents, SelfConsistency benefits from partial results.
                # If 3 of 5 samples succeed, the majority vote is still valid.
                logger.warning(f"Sample {sample_idx + 1} failed: {e!s}", exc_info=True)
                continue

        return samples

    def _collect_parallel(
        self,
        fsm_def: dict[str, Any],
        task: str,
        temperatures: list[float],
        initial_context: dict[str, Any] | None,
        start_time: float,
    ) -> list[str]:
        """Concurrent sampling (max_workers > 1).

        Results are placed by sample index and then read back in order, so the
        aggregation input is identical to the serial path regardless of which
        sample finishes first. Per-sample failures are dropped (partial results
        are valid), matching the serial behavior.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # One up-front time/budget check before dispatching the batch.
        self._check_budgets(start_time, 0, max_iterations=self.num_samples)

        results: list[str | None] = [None] * self.num_samples
        workers = min(self.max_workers, self.num_samples)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(
                    self._generate_single,
                    fsm_def,
                    task,
                    temperatures[i],
                    initial_context,
                ): i
                for i in range(self.num_samples)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    logger.warning(f"Sample {idx + 1} failed: {e!s}", exc_info=True)
                    results[idx] = None

        # sample-index order → deterministic aggregation
        return [r for r in results if r is not None]

    def _register_handlers(self, api: API) -> None:
        """No handlers needed for self-consistency (single-state FSM)."""

    def _generate_single(
        self,
        fsm_def: dict[str, Any],
        task: str,
        temperature: float,
        initial_context: dict[str, Any] | None,
    ) -> str:
        """
        Run a single generation and return the sample text.

        :param fsm_def: The FSM definition dict
        :param task: The task string
        :param temperature: Temperature for this sample
        :param initial_context: Optional initial context
        :return: The last non-blank reply, stripped (``""`` if none). The
            terminal ``generate`` state never extracts, so the reply is the
            sample; its ``Answer:`` line is what the vote compares.
        """
        try:
            api = API.from_definition(
                with_instructions(fsm_def, self.config.instructions),
                model=self.config.model,
                temperature=temperature,
                max_tokens=self.config.max_tokens,
                **self._api_kwargs,
            )
        except ValueError as exc:
            error = prompt_overflow_error(exc, self.config.instructions, None)
            if error is None:
                raise
            raise error from exc

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

            return next((r.strip() for r in reversed(responses) if r and r.strip()), "")

        finally:
            api.end_conversation(conv_id)
