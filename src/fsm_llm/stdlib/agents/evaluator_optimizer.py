from __future__ import annotations

"""
EvaluatorOptimizerAgent — Generate-Evaluate-Refine agent implementation.

Uses an external evaluation function (NOT LLM self-evaluation) to
iteratively improve LLM-generated output until it passes or
maximum refinements are reached.
"""

from collections.abc import Callable
from typing import Any

from fsm_llm.dialog.api import API
from fsm_llm.logging import logger

from .base import BaseAgent
from .constants import (
    ContextKeys,
    Defaults,
    EvalOptStates,
    HandlerNames,
    HandlerPriorities,
    LogMessages,
)
from .definitions import AgentConfig, AgentResult, EvaluationResult
from .fsm_definitions import build_evalopt_fsm


class EvaluatorOptimizerAgent(BaseAgent):
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
        super().__init__(config, **api_kwargs)
        self.evaluation_fn = evaluation_fn
        self.max_refinements = max_refinements

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Run the agent on a task.

        :param task: The task/question for the agent to solve
        :param initial_context: Optional initial context data
        :return: AgentResult with answer, trace, and metadata
        """
        # Build FSM
        fsm_def = build_evalopt_fsm(
            task_description=task,
        )

        # Build initial context
        context = self._init_context(
            task,
            initial_context,
            extra={
                ContextKeys.REFINEMENT_COUNT: 0,
                "_max_refinements": self.max_refinements,
            },
        )

        return self._standard_run(task, fsm_def, context, "evaluator_optimizer")

    def _register_handlers(self, api: API) -> None:
        """Register agent handlers with the API."""
        # Evaluation handler: runs external evaluation on entry to evaluate state
        api.register_handler(
            api.create_handler(HandlerNames.EVAL_OPT_EVALUATOR)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(EvalOptStates.EVALUATE)
            .do(self._run_evaluation)
        )

        # Iteration limiter
        self._register_iteration_limiter(api, self._check_iteration_limit)

    def _run_evaluation(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Run the external evaluation function on the generated output.

        If the evaluation passes, sets evaluation_passed=True.
        If it fails and refinement_count < max, sets evaluation_passed=False
        with feedback. If max refinements reached, forces evaluation_passed=True.
        """
        generated_output = context.get(ContextKeys.GENERATED_OUTPUT, "")
        refinement_count = context.get(ContextKeys.REFINEMENT_COUNT, 0)
        max_refinements = context.get("_max_refinements", self.max_refinements)

        # Run the external evaluation
        try:
            eval_result = self.evaluation_fn(str(generated_output), context)
        except Exception as e:
            logger.warning(
                f"Evaluation function raised an exception: {e}", exc_info=True
            )
            eval_result = EvaluationResult(
                passed=False,
                score=0.0,
                feedback=f"Evaluation error: {e}",
            )

        logger.info(
            LogMessages.EVALUATION.format(
                result="PASSED" if eval_result.passed else "FAILED",
                score=eval_result.score,
            )
        )

        # Store the evaluation result
        trace = list(context.get(ContextKeys.AGENT_TRACE, []))
        trace.append(
            {
                "type": "evaluation",
                "refinement_count": refinement_count,
                "passed": eval_result.passed,
                "score": eval_result.score,
                "feedback": eval_result.feedback,
            }
        )

        if eval_result.passed:
            return {
                ContextKeys.EVALUATION_PASSED: True,
                ContextKeys.EVALUATION_RESULT: eval_result.model_dump(mode="json"),
                ContextKeys.AGENT_TRACE: trace,
            }

        # Check if max refinements reached — force pass with best effort.
        # refinement_count is 0-indexed: allows exactly max_refinements attempts
        # before forcing pass on the (max_refinements + 1)th evaluation.
        if refinement_count >= max_refinements:
            logger.info(
                f"Max refinements ({max_refinements}) reached, "
                "forcing output with best effort"
            )
            return {
                ContextKeys.EVALUATION_PASSED: True,
                ContextKeys.EVALUATION_RESULT: eval_result.model_dump(mode="json"),
                ContextKeys.REFINEMENT_FEEDBACK: eval_result.feedback,
                ContextKeys.AGENT_TRACE: trace,
            }

        # Evaluation failed, provide feedback for refinement
        return {
            ContextKeys.EVALUATION_PASSED: False,
            ContextKeys.EVALUATION_RESULT: eval_result.model_dump(mode="json"),
            ContextKeys.REFINEMENT_FEEDBACK: eval_result.feedback,
            ContextKeys.REFINEMENT_COUNT: refinement_count + 1,
            ContextKeys.AGENT_TRACE: trace,
        }

    def _check_iteration_limit(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check if the iteration limit has been reached."""
        iteration = context.get(ContextKeys.ITERATION_COUNT, 0) + 1

        if iteration >= self.config.max_iterations:
            return {
                ContextKeys.ITERATION_COUNT: iteration,
                ContextKeys.MAX_ITERATIONS_REACHED: True,
                ContextKeys.EVALUATION_PASSED: True,
            }

        return {ContextKeys.ITERATION_COUNT: iteration}

    def _extract_answer(
        self,
        final_context: dict[str, Any],
        responses: list[str],
        extra_keys: list[str] | None = None,
    ) -> str:
        """Extract the final answer from context or responses."""
        # Priority: final_answer > generated_output > last response
        answer = final_context.get(ContextKeys.FINAL_ANSWER)
        if (
            answer
            and isinstance(answer, str)
            and len(answer) > Defaults.MIN_ANSWER_LENGTH
        ):
            return str(answer)

        output = final_context.get(ContextKeys.GENERATED_OUTPUT)
        if (
            output
            and isinstance(output, str)
            and len(output) > Defaults.MIN_ANSWER_LENGTH
        ):
            return str(output)

        for response in reversed(responses):
            if response and len(response.strip()) > Defaults.MIN_ANSWER_LENGTH:
                return response.strip()

        return "Agent could not determine an answer."
