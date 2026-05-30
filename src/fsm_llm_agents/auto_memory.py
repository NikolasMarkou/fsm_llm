from __future__ import annotations

"""
Automatic long-term memory for agents.

Small models frequently *forget to call* ``remember``/``recall`` tools, so the
memory-as-tool pattern (``create_memory_tools`` / ``create_semantic_memory_tools``)
silently no-ops on them. This module removes the model from the loop:

- **Auto-recall (before)**: relevant memories are retrieved and prepended to the
  task, so they are always in the prompt — no tool call required.
- **Auto-remember (after)**: the interaction is persisted automatically once the
  run completes.

Injection happens at the agent's ``run()`` boundary rather than via mid-FSM
handlers: the task string is *guaranteed* to reach the LLM prompt, whereas an
arbitrary injected context key is not surfaced unless the (auto-generated) FSM
prompt references it. This makes auto-memory robust across every agent pattern.

Fully additive — :class:`AutoMemoryReactAgent` is a thin :class:`ReactAgent`
subclass; the standalone helpers work with any store exposing ``add(text)`` and
``search(query, k)`` (e.g. :class:`SemanticMemoryStore`).

Example::

    from fsm_llm_agents import AgentConfig, ToolRegistry, tool
    from fsm_llm_agents.auto_memory import AutoMemoryReactAgent

    agent = AutoMemoryReactAgent(tools=registry, config=AgentConfig(model=model))
    agent.run("My favorite language is Python.")   # auto-stored
    agent.run("What language do I like?")           # auto-recalled into prompt
"""

from typing import Any, Protocol

from fsm_llm.logging import logger

from .definitions import AgentConfig, AgentResult
from .react import ReactAgent
from .semantic_memory import SemanticMemoryStore
from .tools import tool


# DECISION plan_2026-05-30_5598b755/D-005
# A first-class "answer directly" action. ReAct forces think -> act -> conclude
# and the under-call guard (D-004) requires a tool to have run before concluding,
# which is correct for tool tasks but leaves conversational / recall turns (which
# need no external tool) flailing until the stall-detector fires. `respond` gives
# those turns a valid action: the model answers from the conversation or its
# recalled memory, the answer is recorded as an observation, and the loop
# concludes cleanly. Scoped to AutoMemoryReactAgent (the conversational agent) so
# tool-only ReAct agents keep their stricter tool-first behavior.
@tool(
    name="respond",
    description=(
        "Answer the user directly when no external tool or lookup is needed — "
        "for example, replying conversationally or from facts you were given or "
        "recalled from memory. Put your complete answer in `answer`. Do NOT use "
        "this to avoid a tool that is actually required (calculations, searches, "
        "lookups): use the real tool for those."
    ),
)
def _respond(answer: str) -> str:
    """Return the model's direct answer verbatim (recorded as an observation)."""
    return answer


_RESPOND_TOOL = _respond._tool_definition


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
    matches score 1.0 and always pass. (DECISION plan_2026-05-30_5598b755/D-006)
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
    lines = [header]
    for text, _score, _meta in results:
        lines.append(f"- {text}")
    block = "\n".join(lines)
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


class AutoMemoryReactAgent(ReactAgent):
    """A :class:`ReactAgent` with automatic recall + persistence.

    Args:
        memory: Backend with ``add``/``search`` (default: a new
            :class:`SemanticMemoryStore`). Pass a persistent store to carry
            memory across sessions.
        recall_k: How many memories to inject before each run.
        auto_remember: Persist the interaction after each run.
        remember_only_on_success: When True, only runs the agent reports as
            successful are stored. Defaults to **False**: a conversational
            "remember this" turn calls no tool and is flagged ``success=False``
            by the degenerate-completion guard, so requiring success would drop
            exactly the facts a memory agent exists to keep.
        **kwargs: Forwarded to :class:`ReactAgent` (``tools``, ``config``, ...).
    """

    def __init__(
        self,
        *args: Any,
        memory: MemoryBackend | None = None,
        recall_k: int = 3,
        recall_min_score: float = 0.25,
        auto_remember: bool = True,
        remember_only_on_success: bool = False,
        enable_respond: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # NB: use `is not None`, not `or` — an empty SemanticMemoryStore has
        # __len__ == 0 and is therefore falsy, which `or` would silently
        # discard in favor of a fresh (empty) store.
        self.memory: MemoryBackend = (
            memory if memory is not None else SemanticMemoryStore()
        )
        self.recall_k = recall_k
        self.recall_min_score = recall_min_score
        self.auto_remember = auto_remember
        self.remember_only_on_success = remember_only_on_success
        self.enable_respond = enable_respond
        # DECISION plan_2026-05-30_5598b755/D-005
        # Give conversational/recall turns a valid "answer directly" action.
        # Idempotent and flag-guarded; skipped if the caller already registered
        # a `respond` tool. Mutates the passed registry by design (this agent
        # owns its conversational tool surface).
        if enable_respond and "respond" not in self.tools:
            self.tools.register(_RESPOND_TOOL)

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        augmented = augment_task_with_memories(
            task, self.memory, self.recall_k, min_score=self.recall_min_score
        )
        result: AgentResult | None = None
        try:
            result = super().run(augmented, initial_context)
            return result
        finally:
            # Persist in `finally` so a run that RAISES (budget/timeout — common
            # on slow/over-eager models) still preserves a user-stated fact
            # instead of silently dropping it.
            if self.auto_remember:
                if result is None:
                    self._remember_raw(task)
                elif result.success or not self.remember_only_on_success:
                    remember_interaction(self.memory, task, result.answer)

    def _remember_raw(self, text: str) -> None:
        """Persist the raw input when no result is available (run raised)."""
        if not text or not text.strip():
            return
        try:
            self.memory.add(text)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Auto-remember (on failure) failed: {e}")


def with_auto_memory(
    tools: Any,
    config: AgentConfig | None = None,
    memory: MemoryBackend | None = None,
    **kwargs: Any,
) -> AutoMemoryReactAgent:
    """Convenience factory for an :class:`AutoMemoryReactAgent`."""
    return AutoMemoryReactAgent(tools=tools, config=config, memory=memory, **kwargs)
