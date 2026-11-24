"""
REWOOAgent: Reasoning WithOut Observation.

Uses exactly 2 LLM calls: plan all tool calls upfront (#E1, #E2 refs), execute them sequentially
(no LLM), then synthesize from evidence. The machine is the pattern file ``patterns/rewoo.json``
run by ``PatternAgent`` (plan a555dcfe step 14); the executor is the library factory
``agents.rewoo_executor`` and the cut-off answer is the ``after_run`` hook
``agents.rewoo_cut_off_answer``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fsm_llm.agents.constants import ContextKeys
from fsm_llm.core.definitions import AgentError
from fsm_llm.tools import ToolRegistry

from .base import reject_misplaced_kwargs
from .definitions import AgentConfig
from .handlers.rewoo import execute_plans, substitute_evidence_refs
from .pattern_agent import PatternAgent


class REWOOAgent(PatternAgent):
    """
    REWOO agent that plans all tool calls upfront then executes them.

    Makes exactly 2 LLM calls: one to plan all tool calls with #E1/#E2
    variable references, one to synthesize the final answer from evidence.

    Usage::

        agent = REWOOAgent(tools=registry)
        result = agent.run("What is the population of France times 2?")
    """

    tools: ToolRegistry  # always set (``PatternAgent.tools`` may be None)

    # Run outputs caller context may not seed (D-052 of plan 06a5ec0a).
    _run_output_keys = frozenset(
        {
            ContextKeys.PLAN_BLUEPRINT,
            ContextKeys.EVIDENCE,
            ContextKeys.EVIDENCE_STATUS,
            ContextKeys.PLAN_CUT_OFF,
        }
    )

    # The two test seams of the old class, over the moved bodies.
    _substitute_evidence_refs = staticmethod(substitute_evidence_refs)

    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        **api_kwargs: Any,
    ) -> None:
        from .pattern_files import load_envelope  # lazy: reads the file

        # the base denylist (`hitl=` is a TypeError here): `PatternAgent` names `hitl` itself
        reject_misplaced_kwargs(type(self).__name__, api_kwargs)
        super().__init__(
            load_envelope("rewoo"), tools=tools, config=config, **api_kwargs
        )

    def _context_seed(self) -> dict[str, Any]:
        """The evidence store (a seed value of a file is a scalar or a list: not a dict) and the file's seed."""
        # DECISION plan_2026-05-31_cb91a9d5/D-001 [STALE]: require non-empty tool
        # evidence. DECISION plan-2026-09-29T103145-06a5ec0a/D-041: success reads
        # EVIDENCE_STATUS (the file's `evidence_keys`, a per-step success flag),
        # NOT the `evidence` mapping. Do NOT go back to `evidence`: failed calls
        # store their error text there, so a run whose every tool failed read as
        # success=True.
        return {ContextKeys.EVIDENCE: {}, **super()._context_seed()}

    def run_stream(
        self, task: str, initial_context: dict[str, Any] | None = None
    ) -> Iterator[str]:
        """Refused: the answer of a plan cut off at ``max_tokens`` replaces the reply after the run."""
        raise AgentError(
            "REWOOAgent does not stream (its cut-off answer is set after the run)"
        )

    def _execute_all_plans(self, context: dict[str, Any]) -> dict[str, Any]:
        """The executor body over this agent's tools (the handler of a run binds the run's)."""
        return execute_plans(context, self.tools)
