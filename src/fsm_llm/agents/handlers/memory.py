"""The memory helpers of the auto-memory hooks (moved from ``auto_memory.py``, plan a555dcfe step 21).

``MemoryBackend`` is the structural type of an auto-memory store, ``augment_task_with_memories`` the
body of the hook ``agents.memory_recall`` and ``remember_interaction`` that of ``agents.memory_remember``
(``hooks.py``). ``fsm_llm.agents.auto_memory`` re-exports the three names its callers import. Interface
contract: both functions are best-effort (a failing backend logs a warning and the task or the run goes
on); ``augment_task_with_memories`` returns the task unchanged when nothing relevant is found.
"""

from __future__ import annotations

from typing import Any, Protocol

from fsm_llm.logging import logger

__all__ = ["MemoryBackend", "augment_task_with_memories", "remember_interaction"]


class MemoryBackend(Protocol):
    """Structural type for an auto-memory backend."""

    def add(self, text: str, *args: Any, **kwargs: Any) -> Any: ...

    def search(
        self, query: str, k: int = ...
    ) -> list[tuple[str, float, dict[str, Any]]]: ...


def augment_task_with_memories(
    task: str,
    memory: MemoryBackend,
    recall_k: int = 3,
    header: str = "[Relevant things you remember]",
    min_score: float = 0.0,
) -> str:
    """Return ``task`` with a block of recalled memories prepended-as-context.

    Returns the task unchanged when nothing relevant is found or recall fails.

    ``min_score`` drops recalled memories whose similarity is below the cutoff
    BEFORE injection. ``search`` returns the top-k by score regardless of how low
    it is, so without a cutoff an unrelated stored fact (e.g. an old calculation)
    leaks into every later turn and the agent repeats it. Substring-fallback
    matches score 1.0 and always pass. (DECISION plan_2026-05-30_5598b755/D-006 [STALE])
    """
    try:
        results = memory.search(task, k=recall_k)
    except Exception as e:
        logger.warning(f"Auto-recall failed: {e}")
        return task
    if min_score > 0.0:
        results = [r for r in results if r[1] >= min_score]
    if not results:
        return task
    block = "\n".join([header, *(f"- {text}" for text, _score, _meta in results)])
    return f"{block}\n\n{task}"


def remember_interaction(
    memory: MemoryBackend,
    task: str,
    answer: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist a task/answer interaction to the memory backend (best-effort)."""
    if not answer or not answer.strip():
        return
    try:
        memory.add(f"Q: {task}\nA: {answer}", metadata=metadata)
    except TypeError:
        # Backend's add() may not accept metadata kwarg.
        try:
            memory.add(f"Q: {task}\nA: {answer}")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Auto-remember failed: {e}")
    except Exception as e:
        logger.warning(f"Auto-remember failed: {e}")
