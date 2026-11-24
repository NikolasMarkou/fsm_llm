"""
PlanExecuteAgent: Plan-and-Execute.

Separates strategic planning from tactical execution:
Plan -> Execute Step -> Check Result -> Synthesize (all done)
                                      -> Replan (step failed)
                                      -> Execute Step (next step)

The machine is the pattern file ``patterns/plan_execute.json`` run by ``PatternAgent`` (plan a555dcfe
step 15); its four handlers are the library factories ``agents.plan_step_*`` and
``agents.plan_replan_counter`` (bodies in ``handlers/plan_execute.py``) and the tool executor is
``agents.tool_executor`` bound only when the run has a registry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.agents.constants import ContextKeys, Defaults
from fsm_llm.tools import ToolRegistry, require_int_limit

from .definitions import AgentConfig
from .handlers.plan_execute import (
    make_replan_handler,
    make_result_checker,
    make_step_tracker,
)
from .pattern_agent import PatternAgent


class PlanExecuteAgent(PatternAgent):
    """
    Plan-and-Execute agent that separates planning from execution.

    First creates a plan, then executes each step sequentially. If a step
    fails, the agent can revise the remaining plan. When all steps are
    complete, results are synthesized into a final answer.

    Usage::

        agent = PlanExecuteAgent(tools=registry)
        result = agent.run("Compare the populations of France and Germany")
        print(result.answer)
    """

    # Host objects an executable of this pattern reads when the run supplies them (D-018).
    optional_callables = ("tools",)

    # Run outputs caller context may not seed (D-052 of plan 06a5ec0a).
    _run_output_keys = frozenset(
        {
            ContextKeys.PLAN_STEPS,
            ContextKeys.CURRENT_STEP_INDEX,
            ContextKeys.STEP_RESULTS,
            ContextKeys.STEP_RESULT,
            ContextKeys.ALL_STEPS_COMPLETE,
            ContextKeys.STEP_FAILED,
            ContextKeys.CURRENT_STEP_DESCRIPTION,
            ContextKeys.PREVIOUS_PLAN_STEPS,
        }
    )

    def __init__(
        self,
        tools: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        max_replans: int = Defaults.MAX_REPLANS,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize a Plan-and-Execute agent.

        :param tools: Optional tool registry (executor may use LLM only)
        :param config: Agent configuration (defaults to AgentConfig())
        :param max_replans: Maximum number of replan cycles (minimum 0, which
            means no replans; a negative value raises ``AgentError``)
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        # DECISION plan-2026-11-06T093029-0068b055/D-008
        # 0 is a real setting (no replans, pinned in test_plan_execute.py) and
        # stays valid; only a negative value, which silently behaved as 0, is
        # refused. Do NOT clamp it. See decisions.md D-008.
        max_replans = require_int_limit("max_replans", max_replans, minimum=0)
        from .pattern_files import load_envelope  # lazy: reads the file

        super().__init__(
            load_envelope("plan_execute"),
            config=config,
            options={"max_replans": max_replans},
            **api_kwargs,
        )
        # the file claims no tools (an executable of it needs none: they are optional), so the registry
        # is set here, not named to the base, and the flagged-tool refusal is run again over it
        self.tools = tools
        self.max_replans = max_replans
        self._refuse_run_start()

    # DECISION plan-2026-09-19T175721-21cd7f8e/D-046
    # The file's seed `plan_steps: []` is the ONLY exit of the `plan` state (its one edge is
    # `has_context: plan_steps`, key EXISTENCE): with it a model that cannot plan ends the run as
    # success=False, without it the run loops on `plan` until BudgetExhaustedError. Do NOT remove it
    # from patterns/plan_execute.json. It is still asked for (an empty container reads as unset for
    # agent FSMs). See decisions.md D-046.
    # DECISION plan-2026-09-24T091842-c1d5bfbc/D-024
    # The file binds the tool executor on CHECK_RESULT entry, before the checker (priority +1): the
    # step's tool selection is extracted during the execute_step turn. Do NOT bind it on EXECUTE_STEP
    # entry (step 1 never ran a tool, step k ran step k-1's input).
    # DECISION plan-2026-11-05T134105-5ed71cf2/D-011
    # The file binds NO iteration limiter: it counted every PRE_TRANSITION (2k+1 for a k-step plan),
    # set a flag no edge reads and only made `forced_stop_skip` skip later tools. The bound is core's
    # step ceiling. Do NOT bind `agents.iteration_limiter` without an edge that reads its flag.
    def _context_seed(self) -> dict[str, Any]:
        """The file's seed, then the replan count (an internal key: a file may not seed one)."""
        return {**super()._context_seed(), ContextKeys.REPLAN_COUNT: 0}

    def _build_trace(self, final_context: dict[str, Any], iteration: int) -> Any:
        """The trace, with ``total_iterations`` the run's loop turns.

        No limiter counts into ``iteration_count`` here (D-011 of plan 5ed71cf2), so the seeded 0
        would be reported; *iteration* is core's step count.
        """
        trace = super()._build_trace(final_context, iteration)
        trace.total_iterations = iteration
        return trace

    # The two test seams of the old class, over the moved bodies.
    def _make_step_tracker(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        return make_step_tracker()

    def _make_result_checker(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        return make_result_checker(self.max_replans)

    def _make_replan_handler(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        return make_replan_handler()
