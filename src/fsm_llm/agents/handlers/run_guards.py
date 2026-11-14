"""Run-level guards of the agent patterns: a caller's evaluation gate that fails closed and the
holder that carries a budget or timeout error out of a handler (plan d454fd62 step 88; the function
and class bodies are the old ``agents/handlers.py`` ones)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.agents.definitions import EvaluationResult
from fsm_llm.agents.exceptions import AgentTimeoutError, BudgetExhaustedError
from fsm_llm.logging import logger


def run_evaluation_fn(
    evaluation_fn: Callable[..., Any], *args: Any
) -> EvaluationResult:
    """Run a caller's ``evaluation_fn`` so that it can only fail closed.

    Interface contract (Reflexion ``handle_evaluation``, EvaluatorOptimizer
    ``_run_evaluation``):
        - ``evaluation_fn``: the caller's gate, called as ``evaluation_fn(*args)``
          (Reflexion passes the context, EvaluatorOptimizer the output text
          and the context).
        - Returns the function's ``EvaluationResult`` unchanged; an exception,
          or any other return value (a bool, a dict, ``None``), becomes a
          FAILED evaluation (``passed=False``, ``score=0.0``, feedback
          ``"Evaluation error: ..."``) and a WARNING is logged.
        - Never raises (``Exception`` only; ``BaseException`` still passes).
    """
    # DECISION plan-2026-11-05T134105-5ed71cf2/D-012
    # Do NOT let the exception or a wrong-typed return leave the handler: core
    # runs it in handler_error_mode="continue", the verdict keys stay unset and
    # the evaluating state falls back to the model's own verdict (Reflexion)
    # or to the first draft (EvaluatorOptimizer), so an outage of the caller's
    # verifier shipped an unverified answer as a success. Do NOT register the
    # handler `.critical()` instead: that aborts the run rather than letting
    # the loop refine or reflect. See decisions.md D-012.
    try:
        result = evaluation_fn(*args)
    except Exception as e:
        logger.opt(exception=True).warning(
            "Evaluation function raised an exception: {}", e
        )
        return EvaluationResult(
            passed=False, score=0.0, feedback=f"Evaluation error: {e}"
        )
    if not isinstance(result, EvaluationResult):
        logger.warning(
            "Evaluation function returned "
            f"{type(result).__name__}, not EvaluationResult; counted as failed"
        )
        return EvaluationResult(
            passed=False,
            score=0.0,
            feedback=(
                "Evaluation error: evaluation_fn returned "
                f"{type(result).__name__}, not EvaluationResult"
            ),
        )
    return result


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
