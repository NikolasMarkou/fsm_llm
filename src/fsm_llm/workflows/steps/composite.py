"""The steps that wrap other steps: ``ParallelStep`` and ``RetryStep``.

``ParallelStep`` refuses a wait or timer child (also behind a ``RetryStep``), so it imports ``waits``.

Part of the package (plan d454fd62 step 92; the old ``workflows/steps.py`` at the same dotted path).
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable
from typing import Any

from pydantic import Field, field_validator

from fsm_llm.constants import has_internal_prefix
from fsm_llm.logging import logger

from ..constants import PARALLEL_DEEPCOPY_WARNING_THRESHOLD
from ..models import WorkflowStepResult
from .base import WorkflowStep
from .helpers import _call_user_callable, _error_text
from .waits import TimerStep, WaitEventStep


class ParallelStep(WorkflowStep):
    """A step that executes multiple steps in parallel.

    Children get their own copy of the context (``deepcopy``, falling back
    to a shallow copy for values that cannot be deep-copied). Their
    ``next_step_id`` is ignored; their data is merged as ``step_<index>_<key>``
    (internal-prefix keys are dropped first) or by ``aggregation_function``.
    Event and timer steps cannot be children (they only pause a workflow).
    ``aggregation_function(results)`` may be sync (run in the step executor)
    or async.

    If any child fails, the step fails: it routes to ``error_step_id``, or
    FAILS the instance when none is set, and still reports the data of the
    children that succeeded.
    """

    steps: list[WorkflowStep]
    next_step_id: str
    error_step_id: str | None = None
    aggregation_function: (
        Callable[[list[WorkflowStepResult]], dict[str, Any]] | None
    ) = None

    @field_validator("steps")
    @classmethod
    def _validate_children(cls, v: list[WorkflowStep]) -> list[WorkflowStep]:
        for child in v:
            inner = child
            while isinstance(inner, RetryStep):
                inner = inner.step
            if isinstance(inner, (WaitEventStep, TimerStep)):
                raise ValueError(
                    f"ParallelStep child '{child.step_id}' is a "
                    f"{type(inner).__name__}; event and timer steps cannot run "
                    "inside a ParallelStep"
                )
        return v

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Execute multiple steps in parallel and aggregate results."""
        try:
            # Execute all steps in parallel
            results = await self._execute_parallel_steps(context)

            # Check for errors
            errors = self._collect_errors(results)
            if errors:
                error_msg = "; ".join(errors)
                # DECISION plan-2026-09-27T120000-5d1e7a3b/D-003
                # Do NOT fall back to next_step_id when error_step_id is unset: a
                # failed child must not continue down the success route. With
                # no error_step_id the failure has no next_step_id and the engine
                # FAILS the instance.
                return WorkflowStepResult(
                    success=False,
                    data=self._default_aggregate(results),
                    error=error_msg,
                    next_step_id=self.error_step_id,
                    message=f"Parallel step had {len(errors)} error(s)",
                )

            # Aggregate results
            aggregated_data = await self._aggregate_results(results)

            return WorkflowStepResult.success_result(
                data=aggregated_data,
                next_step_id=self.next_step_id,
                message="Parallel step completed successfully",
            )
        except Exception as e:
            logger.error(f"Error in parallel step {self.step_id}: {e!s}")
            return WorkflowStepResult.failure_result(
                error=_error_text(e),
                next_step_id=self.error_step_id,
                message=f"Parallel step failed: {e!s}",
            )

    def _copy_context(self, context: dict[str, Any]) -> dict[str, Any]:
        try:
            return copy.deepcopy(context)
        except Exception as e:
            logger.warning(
                f"ParallelStep [{self.step_id}] could not deep-copy the context "
                f"({e!s}); using a shallow copy"
            )
            return dict(context)

    async def _execute_parallel_steps(
        self, context: dict[str, Any]
    ) -> list[WorkflowStepResult]:
        """Execute all steps in parallel with isolated context copies."""
        if len(self.steps) > PARALLEL_DEEPCOPY_WARNING_THRESHOLD:
            logger.warning(
                f"ParallelStep [{self.step_id}] deep-copying context for "
                f"{len(self.steps)} parallel steps — consider reducing parallelism "
                "if memory usage is a concern"
            )
        contexts = [self._copy_context(context) for _ in self.steps]
        tasks = [
            step.execute(ctx) for step, ctx in zip(self.steps, contexts, strict=True)
        ]
        results = await self._with_timeout(
            asyncio.gather(*tasks, return_exceptions=True)
        )

        # Convert exceptions (including a cancelled child) to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                processed_results.append(
                    WorkflowStepResult.failure_result(
                        error=_error_text(result),
                        message=f"Parallel step {i} failed: {_error_text(result)}",
                    )
                )
            elif not isinstance(result, WorkflowStepResult):
                processed_results.append(
                    WorkflowStepResult.failure_result(
                        error=f"returned {type(result).__name__}, not a WorkflowStepResult",
                        message=f"Parallel step {i} returned an invalid result",
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def _collect_errors(self, results: list[WorkflowStepResult]) -> list[str]:
        """Collect one error message per failed result.

        # DECISION plan-2026-09-27T120000-5d1e7a3b/D-003
        # A failure counts whether or not it carries error text: do NOT go
        # back to filtering on `r.error` (an exception with an empty message,
        # such as `TimeoutError()`, then aggregated as a success).
        """
        return [
            r.error or r.message or f"step {i} failed"
            for i, r in enumerate(results)
            if not r.success
        ]

    def _default_aggregate(self, results: list[WorkflowStepResult]) -> dict[str, Any]:
        aggregated_data = {}
        for i, result in enumerate(results):
            if result.success and result.data:
                prefix = f"step_{i}_"
                for key, value in result.data.items():
                    # Drop internal keys BEFORE prefixing: "step_0__x" would no
                    # longer look internal and would slip past the engine's filter.
                    if has_internal_prefix(key):
                        continue
                    aggregated_data[f"{prefix}{key}"] = value
        return aggregated_data

    async def _aggregate_results(
        self, results: list[WorkflowStepResult]
    ) -> dict[str, Any]:
        """Aggregate results from parallel steps (``aggregation_function`` may
        be sync, run in the step executor, or async)."""
        if self.aggregation_function:
            aggregated: dict[str, Any] = await _call_user_callable(
                self.aggregation_function, results
            )
            return aggregated
        return self._default_aggregate(results)


class RetryStep(WorkflowStep):
    """Wrap another step with automatic retry on failure.

    Makes ``max_retries + 1`` attempts; the delay before retry ``n`` (1-based)
    is ``backoff_factor * n`` seconds (linear). ``timeout_seconds`` applies to each
    attempt. The last failure is returned (or re-raised) unchanged.

    Example::

        inner = api_step("call_api", "Call API", my_api, "done", "error")
        step = RetryStep(
            step_id="retry_call",
            name="Retry API Call",
            step=inner,
            max_retries=3,
            backoff_factor=2.0,
        )
    """

    step: Any = Field(exclude=True)
    """The inner ``WorkflowStep`` to retry."""
    max_retries: int = 3
    backoff_factor: float = 1.0
    """Delay multiplier between retries (seconds). Delay = backoff_factor * attempt."""

    def referenced_step_ids(self) -> set[str]:
        """The wrapped step's step ids (none for a duck-typed non-step)."""
        if isinstance(self.step, WorkflowStep):
            return self.step.referenced_step_ids()
        return set()

    @field_validator("step")
    @classmethod
    def _validate_step(cls, v: Any) -> Any:
        # Duck typing on purpose: non-WorkflowStep objects with an async
        # execute(context) are supported (see D-010 below).
        if not callable(getattr(v, "execute", None)):
            raise ValueError(
                "RetryStep.step must be a WorkflowStep (or an object with an "
                f"execute(context) method), got {type(v).__name__}"
            )
        return v

    @field_validator("max_retries")
    @classmethod
    def _validate_max_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_retries must be >= 0")
        return v

    @field_validator("backoff_factor")
    @classmethod
    def _validate_backoff_factor(cls, v: float) -> float:
        if v < 0:
            raise ValueError("backoff_factor must be >= 0")
        return v

    async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
        """Execute the inner step with retries."""
        last_result: WorkflowStepResult | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result: WorkflowStepResult = await self._with_timeout(
                    self.step.execute(context)
                )
            # DECISION plan-2026-09-12T065608-089d0ec7/D-010
            # Catch bare Exception (not just WorkflowStepError): a custom/
            # non-conforming inner WorkflowStep can raise any exception type,
            # and this must be retried the same way, matching the bare
            # `except Exception` idiom already used by 6+ other step types
            # (AutoStep, APIStep, ConditionStep,
            # LLMStep, ConversationStep, ParallelStep, AgentStep).
            # See decisions.md D-010.
            except Exception:
                # Steps that signal failure by RAISING (ConditionStep,
                # AutoStep, LLMStep, _with_timeout) must
                # also be retried, not propagated on the first attempt.
                if attempt < self.max_retries:
                    delay = self.backoff_factor * (attempt + 1)
                    logger.debug(
                        f"Retry step '{self.step_id}': attempt {attempt + 1} "
                        f"raised, retrying in {delay:.1f}s "
                        f"({self.max_retries - attempt - 1} left)"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            if result.success:
                return result
            last_result = result
            if attempt < self.max_retries:
                delay = self.backoff_factor * (attempt + 1)
                logger.debug(
                    f"Retry step '{self.step_id}': attempt {attempt + 1} failed, "
                    f"retrying in {delay:.1f}s ({self.max_retries - attempt - 1} left)"
                )
                await asyncio.sleep(delay)
        return last_result  # type: ignore[return-value]
