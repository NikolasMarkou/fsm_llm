"""The Reflexion handler bodies: the evaluate-entry verdict and the reflect-exit episode record.

Interface contract (plan a555dcfe step 10): ``make_evaluation_handler(evaluation_fn)`` and
``make_reflection_handler(max_reflections)`` each return a ``(context) -> updates`` function; the
library factories ``agents.reflexion_evaluator`` and ``agents.reflexion_reflector`` wrap them in a
handler. ``EVALUATION_KEYS`` are the verdict keys evaluate produces. Failure: ``run_evaluation_fn``
treats a raising or wrong-typed ``evaluation_fn`` as a failed evaluation, never an error here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.agents.constants import ContextKeys, LogMessages, StopReason
from fsm_llm.agents.definitions import EvaluationResult, ReflexionMemory
from fsm_llm.logging import logger

from .run_guards import run_evaluation_fn

__all__ = ["EVALUATION_KEYS", "make_evaluation_handler", "make_reflection_handler"]

#: The verdict keys evaluate produces (cleared on evaluate entry).
EVALUATION_KEYS = (
    ContextKeys.EVALUATION_PASSED,
    ContextKeys.EVALUATION_SCORE,
    ContextKeys.EVALUATION_FEEDBACK,
)


def make_evaluation_handler(
    evaluation_fn: Callable[[dict[str, Any]], EvaluationResult],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The evaluate-entry handler that runs the external ``evaluation_fn``.

    Its verdict replaces the self-evaluation: the keys it writes are set,
    so evaluate extracts nothing (skip-if-set). A raising or wrong-typed
    function is a failed evaluation (``run_evaluation_fn``), never a reason
    to fall back to the model's own verdict."""

    def handle_evaluation(context: dict[str, Any]) -> dict[str, Any]:
        eval_result = run_evaluation_fn(evaluation_fn, context)
        return {
            ContextKeys.EVALUATION_PASSED: eval_result.passed,
            ContextKeys.EVALUATION_SCORE: eval_result.score,
            ContextKeys.EVALUATION_FEEDBACK: eval_result.feedback,
        }

    return handle_evaluation


def make_reflection_handler(
    max_reflections: int,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The reflect-exit handler that records the episode.

    It runs after the reflect extraction, so the memory entry holds this
    episode's reflection and lessons and the verdict evaluate produced."""

    def handle_reflection(context: dict[str, Any]) -> dict[str, Any]:
        reflection_count = context.get(ContextKeys.REFLECTION_COUNT, 0) + 1
        episodic_memory = list(context.get(ContextKeys.EPISODIC_MEMORY, []))

        reflection_text = context.get(ContextKeys.REFLECTION, "")
        lessons = context.get(ContextKeys.LESSONS, [])
        if not isinstance(lessons, list):
            lessons = [str(lessons)] if lessons else []

        memory_entry = ReflexionMemory(
            episode=reflection_count,
            task_summary=context.get(ContextKeys.TASK, ""),
            outcome=context.get(ContextKeys.EVALUATION_FEEDBACK, ""),
            reflection=str(reflection_text),
            lessons=lessons,
        )
        episodic_memory.append(memory_entry.model_dump(mode="json"))

        logger.info(
            LogMessages.REFLECTION.format(
                current=reflection_count,
                max=max_reflections,
                summary=str(reflection_text)[:80],
            )
        )

        updates: dict[str, Any] = {
            ContextKeys.REFLECTION_COUNT: reflection_count,
            ContextKeys.EPISODIC_MEMORY: episodic_memory,
        }
        if reflection_count >= max_reflections:
            # DECISION plan-2026-09-29T103145-06a5ec0a/D-051: the cap
            # stops a run whose evaluation never passed (reflect runs only
            # after a failed or unclear verdict), so it is a forced stop
            # like EvalOpt's refinement cap (D-011). Do NOT report it as
            # answered: the answer ships, success is False.
            updates[ContextKeys.SHOULD_TERMINATE] = True
            if context.get(ContextKeys.EVALUATION_PASSED) is not True:
                updates[ContextKeys.FORCED_STOP_REASON] = StopReason.FORCED_PASS.value
        return updates

    return handle_reflection
