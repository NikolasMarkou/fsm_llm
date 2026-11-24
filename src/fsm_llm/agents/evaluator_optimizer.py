"""
EvaluatorOptimizerAgent: Generate-Evaluate-Refine agent implementation.

Uses an external evaluation function (NOT LLM self-evaluation) to
iteratively improve LLM-generated output until it passes or
maximum refinements are reached.

The machine is the pattern file ``patterns/evaluator_optimizer.json`` run by ``PatternAgent`` (plan
a555dcfe step 18); its handlers are the library factories ``agents.evalopt_evaluator``,
``agents.refine_entry``, ``agents.refine_exit`` and ``agents.iteration_limiter``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from fsm_llm.agents.constants import ContextKeys, Defaults
from fsm_llm.tools import require_int_limit

from .definitions import AgentConfig, EvaluationResult
from .handlers.evaluator_optimizer import evaluate_output
from .handlers.library_react import iteration_check
from .pattern_agent import PatternAgent


class EvaluatorOptimizerAgent(PatternAgent):
    """
    Agent that iteratively refines output using external evaluation.

    The evaluation function is NOT LLM self-evaluation -- it must be an
    external callable (e.g., running tests, checking code, validating
    against a schema, etc.).

    The loop is: generate -> evaluate -> refine -> evaluate -> ... -> output.

    Usage::

        def check_code(output: str, context: dict) -> EvaluationResult:
            passed = "def " in output and "return " in output
            return EvaluationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                feedback="Missing function definition" if not passed else "OK",
            )

        agent = EvaluatorOptimizerAgent(evaluation_fn=check_code)
        result = agent.run("Write a Python function that adds two numbers")
        print(result.answer)
    """

    # No `tools` argument: `create_agent` raises for tools given to this pattern (`accepts_tools`).
    takes_tools: ClassVar[bool] = False

    # Run outputs caller context may not seed (D-052 of plan 06a5ec0a).
    _run_output_keys = frozenset(
        {
            ContextKeys.GENERATED_OUTPUT,
            ContextKeys.EVALUATION_PASSED,
            ContextKeys.EVALUATION_RESULT,
            ContextKeys.EVALUATION_SCORE,
            ContextKeys.EVALUATION_FEEDBACK,
            ContextKeys.REFINEMENT_FEEDBACK,
            ContextKeys.REFINEMENT_COUNT,
            ContextKeys.PREVIOUS_OUTPUT,
        }
    )

    def __init__(
        self,
        evaluation_fn: Callable[[str, dict[str, Any]], EvaluationResult],
        config: AgentConfig | None = None,
        max_refinements: int = Defaults.MAX_REFINEMENTS,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize an Evaluator-Optimizer agent.

        :param evaluation_fn: External evaluation callable (output, context) -> EvaluationResult
        :param config: Agent configuration (defaults to AgentConfig())
        :param max_refinements: Maximum number of refinement iterations
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        max_refinements = require_int_limit(
            "max_refinements", max_refinements, minimum=0
        )
        from .pattern_files import load_envelope  # lazy: reads the file

        super().__init__(
            load_envelope("evaluator_optimizer"),
            config=config,
            options={"max_refinements": max_refinements},
            callables={"evaluation_fn": evaluation_fn},
            **api_kwargs,
        )
        self.evaluation_fn = evaluation_fn
        self.max_refinements = max_refinements

    # DECISION plan-2026-11-05T134105-5ed71cf2/D-010: no `_extract_answer`
    # override and no minimum answer length here. The base chain ships the
    # judged artifact when it is non-empty ("Paris", "42", "Yes" are
    # answers); a `len(x) > 5` floor replaced it with the failure sentence
    # under success=True. Do NOT re-add a length test. See decisions.md.
    def _context_seed(self) -> dict[str, Any]:
        """The file's seed (``refinement_count``) and the cap the evaluator reads."""
        return {
            **super()._context_seed(),
            ContextKeys.MAX_REFINEMENTS: self.max_refinements,
        }

    def _run_evaluation(self, context: dict[str, Any]) -> dict[str, Any]:
        """The ``agents.evalopt_evaluator`` body for this agent's function and cap."""
        return evaluate_output(context, self.evaluation_fn, self.max_refinements)

    def _make_iteration_limiter(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """The closure ``agents.iteration_limiter`` registers for this agent (the file's binding)."""
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-051: no forced reason
        # in the limiter's keys. Do NOT add FORCED_STOP_REASON back to the
        # binding of patterns/evaluator_optimizer.json: the limiter fires on
        # every transition past the limit, including the one after a genuine
        # pass, so a real pass read as forced_pass. The evaluator records it
        # only when it overrides a failing verdict.
        binding = next(
            b
            for b in self.envelope.body.handlers
            if b.handler == "agents.iteration_limiter"
        )
        return iteration_check(self._handler_context(), binding)
