"""
OrchestratorAgent: dynamic multi-agent coordination.

Decomposes tasks into subtasks, delegates to worker agents (or internal LLM),
collects results, and synthesizes a final answer.

FSM flow: orchestrate -> delegate -> collect -> synthesize
                                             -> orchestrate (if more work needed)

The machine is the pattern file ``patterns/orchestrator.json`` run by ``PatternAgent`` (plan
a555dcfe step 13); the delegator and the decision reset are the library factories
``agents.orchestrator_*`` and the error a worker ends the run with is re-raised by the
``after_run`` hook ``agents.raise_run_ending_error``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.agents.constants import ContextKeys, Defaults
from fsm_llm.logging import logger
from fsm_llm.tools import ToolRegistry, require_int_limit

from .definitions import AgentConfig, AgentResult
from .handlers import RunEndingErrorHolder
from .handlers.library import iteration_check
from .handlers.orchestrator import delegate_to_workers
from .pattern_agent import PatternAgent


class OrchestratorAgent(PatternAgent):
    """
    Orchestrator agent that decomposes tasks and delegates to workers.

    Combines FSM-LLM's core with a worker factory to implement the
    orchestrator-workers pattern. The orchestrator plans subtasks,
    delegates them to workers, collects results, and synthesizes.

    Usage::

        def my_worker(subtask: str) -> AgentResult:
            # ... solve subtask ...
            return AgentResult(answer="result", stop_reason=StopReason.ANSWERED)

        agent = OrchestratorAgent(worker_factory=my_worker)
        result = agent.run("Analyze the pros and cons of remote work")
        print(result.answer)
    """

    # Host objects an executable of this pattern reads when the run supplies them (D-018).
    optional_callables = ("tools", "worker_factory")

    # Run outputs caller context may not seed (D-052 of plan 06a5ec0a).
    _run_output_keys = frozenset(
        {
            ContextKeys.SUBTASKS,
            ContextKeys.WORKER_RESULTS,
            ContextKeys.SKIPPED_SUBTASKS,
            ContextKeys.ALL_COLLECTED,
        }
    )

    def __init__(
        self,
        worker_factory: Callable[[str], AgentResult] | None = None,
        tools: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        max_workers: int = Defaults.MAX_WORKERS,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize an Orchestrator agent.

        :param worker_factory: Callable that takes a subtask string and returns AgentResult.
            If None, the LLM handles subtasks inline (no sub-agents spawned).
        :param tools: Optional ToolRegistry (reserved for worker_factory implementations
            that need tool access; not used directly by the orchestrator)
        :param config: Agent configuration (defaults to AgentConfig())
        :param max_workers: Maximum number of subtask delegations per round
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        max_workers = require_int_limit("max_workers", max_workers, minimum=1)
        from .pattern_files import load_envelope  # lazy: reads the file

        super().__init__(
            load_envelope("orchestrator"),
            config=config,
            options={"max_workers": max_workers},
            callables={}
            if worker_factory is None
            else {"worker_factory": worker_factory},
            **api_kwargs,
        )
        self.worker_factory = worker_factory
        self.tools = tools
        self.max_workers = max_workers

        logger.info(
            f"OrchestratorAgent started with max_workers={max_workers}, model={self.config.model}"
        )

    def _context_seed(self) -> dict[str, Any]:
        """The file's seed and the iteration budget the limiter reads (``limit_key``)."""
        return {
            **super()._context_seed(),
            ContextKeys.ITERATION_BUDGET: self.config.max_iterations,
        }

    def _delegate_to_workers(
        self,
        context: dict[str, Any],
        holder: RunEndingErrorHolder | None = None,
    ) -> dict[str, Any]:
        """The delegator body for this agent's factory and cap (the handler of a run binds its own)."""
        return delegate_to_workers(
            context,
            self.worker_factory,
            self.max_workers,
            holder or RunEndingErrorHolder(),
        )

    def _make_iteration_limiter(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """The closure ``agents.iteration_limiter`` registers for this agent (the file's binding)."""
        binding = next(
            b
            for b in self.envelope.body.handlers
            if b.handler == "agents.iteration_limiter"
        )
        return iteration_check(self._handler_context(), binding)
