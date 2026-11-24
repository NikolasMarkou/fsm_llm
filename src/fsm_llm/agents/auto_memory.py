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
The two ends are the hooks ``agents.memory_recall`` (before) and
``agents.memory_remember`` (after, in a ``finally``) of ``patterns/auto_memory.json``
(``handlers/hooks.py``).

Fully additive — :class:`AutoMemoryReactAgent` is a thin :class:`ReactAgent`
subclass over that preset; the standalone helpers work with any store exposing
``add(text)`` and ``search(query, k)`` (e.g. :class:`SemanticMemoryStore`).

Example::

    from fsm_llm.agents import AgentConfig, ToolRegistry, tool
    from fsm_llm.agents.auto_memory import AutoMemoryReactAgent

    agent = AutoMemoryReactAgent(tools=registry, config=AgentConfig(model=model))
    agent.run("My favorite language is Python.")   # auto-stored
    agent.run("What language do I like?")           # auto-recalled into prompt
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fsm_llm.tools import tool

from .handlers.memory import (
    MemoryBackend,
    augment_task_with_memories,
    remember_interaction,
)
from .react import ReactAgent
from .semantic_memory import SemanticMemoryStore

__all__ = [
    "AutoMemoryReactAgent",
    "MemoryBackend",
    "augment_task_with_memories",
    "remember_interaction",
]


# DECISION plan_2026-05-30_5598b755/D-005 [STALE]
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

    _pattern = "auto_memory"
    # the host object the file's hooks read (D-018: a class names the objects it supplies)
    optional_callables = ("memory",)

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
        # NB: use `is not None`, not `or` — an empty SemanticMemoryStore has
        # __len__ == 0 and is therefore falsy, which `or` would silently
        # discard in favor of a fresh (empty) store.
        self.memory: MemoryBackend = (
            memory if memory is not None else SemanticMemoryStore()
        )
        self._preset_callables = {"memory": self.memory}
        self._preset_options = {
            "recall_k": recall_k,
            "recall_min_score": recall_min_score,
            "auto_remember": auto_remember,
            "remember_only_on_success": remember_only_on_success,
        }
        super().__init__(*args, **kwargs)
        self.recall_k = recall_k
        self.recall_min_score = recall_min_score
        self.auto_remember = auto_remember
        self.remember_only_on_success = remember_only_on_success
        self.enable_respond = enable_respond
        # DECISION plan_2026-05-30_5598b755/D-005 [STALE]
        # Give conversational/recall turns a valid "answer directly" action.
        # Idempotent and flag-guarded; skipped if the caller already registered
        # a `respond` tool. Mutates the passed registry by design (this agent
        # owns its conversational tool surface).
        if enable_respond and "respond" not in self.tools:
            self.tools.register(_RESPOND_TOOL)

    def run_stream(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Run :meth:`run` (memory recall and persistence included) and yield
        its final answer once; streaming the bare loop would skip both."""
        return self._stream_via_run(task, initial_context)
