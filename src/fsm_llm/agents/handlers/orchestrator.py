"""The worker delegator of the orchestrator pattern (plan a555dcfe step 13).

Interface contract: ``delegate_to_workers(context, worker_factory, max_workers, holder) -> dict``
is the body of the handler ``agents.orchestrator_delegator`` (``handlers/library.py`` binds it to
the run's holder and options). It runs ``worker_factory`` once per subtask, one after another, on
entry to ``delegate``; with no factory it stores a placeholder per subtask for the model to fill.
Subtasks beyond ``max_workers`` are not run: they go to ``skipped_subtasks`` (never into
``worker_results``), with a WARNING and a trace count. A worker's ``AgentTimeoutError`` or
``BudgetExhaustedError`` goes into ``holder`` and no further worker starts; any other worker
exception is a failed result. Returns the context update (``worker_results``,
``skipped_subtasks``, ``agent_trace``, ``subtasks`` cleared). Never raises for a worker.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fsm_llm.agents.constants import ContextKeys
from fsm_llm.logging import logger

if TYPE_CHECKING:
    from ..definitions import AgentResult
    from .run_guards import RunEndingErrorHolder

__all__ = ["delegate_to_workers"]


def delegate_to_workers(
    context: dict[str, Any],
    worker_factory: Callable[[str], AgentResult] | None,
    max_workers: int,
    holder: RunEndingErrorHolder,
) -> dict[str, Any]:
    """Run the subtasks of *context* through *worker_factory*; see the module docstring."""
    subtasks = context.get(ContextKeys.SUBTASKS, [])
    if not isinstance(subtasks, list):
        subtasks = [str(subtasks)]

    skipped = [str(subtask) for subtask in subtasks[max_workers:]]
    subtasks = subtasks[:max_workers]
    if skipped:
        logger.warning(
            f"Orchestrator got {len(subtasks) + len(skipped)} subtasks, "
            f"max_workers={max_workers}; skipping {len(skipped)}"
        )

    existing_results = context.get(ContextKeys.WORKER_RESULTS, [])
    if not isinstance(existing_results, list):
        existing_results = []

    new_results: list[dict[str, Any]] = []

    for i, subtask in enumerate(subtasks):
        if holder.is_set:
            break
        subtask_str = str(subtask)
        logger.debug(f"Delegating subtask {i + 1}/{len(subtasks)}: {subtask_str[:100]}")

        if worker_factory is not None:
            try:
                worker_result = worker_factory(subtask_str)
                new_results.append(
                    {
                        "subtask": subtask_str,
                        "answer": worker_result.answer,
                        "success": worker_result.success,
                    }
                )
            except Exception as e:
                if holder.capture(e):
                    logger.warning(f"Worker ended the run: {e}")
                    break
                logger.opt(exception=True).warning("Worker failed for subtask: {}", e)
                new_results.append(
                    {
                        "subtask": subtask_str,
                        "answer": f"Worker error: {e}",
                        "success": False,
                    }
                )
        else:
            # No worker_factory: store subtask for LLM inline processing
            new_results.append(
                {
                    "subtask": subtask_str,
                    "answer": f"[Pending LLM processing: {subtask_str}]",
                    "success": False,
                }
            )

    all_results = existing_results + new_results

    # DECISION plan-2026-09-29T103145-06a5ec0a/D-049: skipped subtasks go
    # to their own key, never into worker_results. Do NOT record them as
    # failed worker entries: the collect judge reads worker_results, took
    # them for unfinished work and re-delegated (hierarchical_orchestrator
    # 23 -> 66 calls). Do NOT drop them silently either (PAT-10, D-042).
    ran = {str(entry.get("subtask")) for entry in all_results}
    previous = context.get(ContextKeys.SKIPPED_SUBTASKS)
    skipped_all = [
        subtask_str
        for subtask_str in dict.fromkeys(
            [*(previous if isinstance(previous, list) else []), *skipped]
        )
        if subtask_str not in ran
    ]

    # Track in agent trace
    trace = context.get(ContextKeys.AGENT_TRACE, [])
    if not isinstance(trace, list):
        trace = []
    trace.append(
        {
            "action": "delegate",
            "subtasks_delegated": len(subtasks),
            "subtasks_skipped": len(skipped),
            "results_count": len(new_results),
        }
    )

    return {
        ContextKeys.WORKER_RESULTS: all_results,
        ContextKeys.SKIPPED_SUBTASKS: skipped_all,
        ContextKeys.AGENT_TRACE: trace,
        # Clear subtasks so orchestrate can set new ones if needed
        ContextKeys.SUBTASKS: None,
    }
