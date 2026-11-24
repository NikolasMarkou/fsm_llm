"""
ReflexionAgent: ReAct with evaluation, verbal self-critique, and episodic memory.

Think -> Act -> Evaluate -> Reflect (if failed) -> Think (loop)
                          -> Conclude (if passed)

The loop is the pattern file ``patterns/reflexion.json`` run by ``PatternAgent`` (plan a555dcfe
step 10); the evaluate and reflect handlers are the library factories ``agents.reflexion_*``.
This class keeps the constructor and the run-output keys.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.agents.constants import ContextKeys, Defaults
from fsm_llm.policy import HumanInTheLoop
from fsm_llm.tools import ToolRegistry, require_int_limit

from .definitions import AgentConfig, EvaluationResult
from .pattern_agent import PatternAgent


class ReflexionAgent(PatternAgent):
    """
    Reflexion agent that evaluates and self-critiques its own outputs.

    Combines the ReAct tool-use loop with an evaluation gate and episodic
    memory. If evaluation fails, the agent reflects, stores lessons in
    episodic memory, and retries with an improved strategy.

    Usage::

        agent = ReflexionAgent(tools=registry, max_reflections=3)
        result = agent.run("What year was the Eiffel Tower built?")
        print(result.answer)
    """

    # Host objects an executable of this pattern reads when the run supplies them (D-018).
    optional_callables = ("evaluation_fn",)

    # Run outputs caller context may not seed (D-052 of plan 06a5ec0a).
    _run_output_keys = frozenset(
        {
            ContextKeys.EVALUATION_PASSED,
            ContextKeys.EVALUATION_SCORE,
            ContextKeys.EVALUATION_FEEDBACK,
            ContextKeys.REFLECTION,
            ContextKeys.LESSONS,
            ContextKeys.EPISODIC_MEMORY,
            ContextKeys.REFLECTION_COUNT,
        }
    )

    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        evaluation_fn: Callable[[dict[str, Any]], EvaluationResult] | None = None,
        max_reflections: int = Defaults.MAX_REFLECTIONS,
        hitl: HumanInTheLoop | None = None,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize a Reflexion agent.

        :param tools: Tool registry with registered tools
        :param config: Agent configuration (defaults to AgentConfig())
        :param evaluation_fn: Optional external evaluation function.
            If None, the LLM self-evaluates via the FSM evaluate state.
        :param max_reflections: Maximum reflect->think cycles before forcing
            conclude (minimum 1; a smaller value raises ``AgentError``)
        :param hitl: Optional HITL manager for approval gates
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        self._require_tools(tools)
        # DECISION plan-2026-11-06T093029-0068b055/D-008
        # Refuse, never clamp: the cap check runs when reflect exits, after one
        # reflection is written, so 0 and -1 used to behave exactly like 1.
        # Do NOT turn this into `max(1, max_reflections)` (Debate did, until 12.8)
        # and do NOT "honour" 0 by changing the FSM. See decisions.md D-008.
        max_reflections = require_int_limit(
            "max_reflections", max_reflections, minimum=1
        )
        from .pattern_files import load_envelope  # lazy: reads the file

        super().__init__(
            load_envelope("reflexion"),
            tools=tools,
            config=config,
            hitl=hitl,
            options={"max_reflections": max_reflections},
            callables={} if evaluation_fn is None else {"evaluation_fn": evaluation_fn},
            **api_kwargs,
        )
        self.evaluation_fn = evaluation_fn
        self.max_reflections = max_reflections
