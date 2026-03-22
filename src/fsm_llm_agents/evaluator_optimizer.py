from __future__ import annotations

"""
EvaluatorOptimizerAgent — Generate-Evaluate-Refine agent implementation.

Uses an external evaluation function (NOT LLM self-evaluation) to
iteratively improve LLM-generated output until it passes or
maximum refinements are reached.
"""

import time
from collections.abc import Callable
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .constants import (
    ContextKeys,
    Defaults,
    EvalOptStates,
    HandlerNames,
    LogMessages,
)
from .definitions import AgentConfig, AgentResult, AgentTrace, EvaluationResult
from .exceptions import AgentError, AgentTimeoutError, BudgetExhaustedError
from .fsm_definitions import build_evalopt_fsm


class EvaluatorOptimizerAgent:
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
        self.evaluation_fn = evaluation_fn
        self.config = config or AgentConfig()
        self.max_refinements = max_refinements
        self._api_kwargs = api_kwargs

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
        start_time = time.monotonic()

        # Build FSM
        fsm_def = build_evalopt_fsm(
            task_description=task[:200],
        )

        # Create API instance
        api = API.from_definition(
            fsm_def,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self._api_kwargs,
        )

        # Register handlers
        self._register_handlers(api)

        # Build initial context
        context: dict[str, Any] = dict(initial_context) if initial_context else {}
        context[ContextKeys.TASK] = task
        context[ContextKeys.AGENT_TRACE] = []
        context[ContextKeys.ITERATION_COUNT] = 0
        context[ContextKeys.REFINEMENT_COUNT] = 0
        context["_max_refinements"] = self.max_refinements

        # Start conversation
        conv_id, initial_response = api.start_conversation(context)
        log = logger.bind(
            conversation_id=conv_id, package="fsm_llm_agents", agent_type="evaluator_optimizer"
        )

        try:
            responses = [initial_response]
            iteration = 0

            while not api.has_conversation_ended(conv_id):
                iteration += 1

                # Check time budget
                elapsed = time.monotonic() - start_time
                if elapsed > self.config.timeout_seconds:
                    raise AgentTimeoutError(self.config.timeout_seconds)

                # Hard ceiling on iterations
                if iteration > self.config.max_iterations * Defaults.FSM_BUDGET_MULTIPLIER:
                    raise BudgetExhaustedError("iterations", self.config.max_iterations)

                response = api.converse(Defaults.CONTINUE_MESSAGE, conv_id)
                responses.append(response)

            # Extract final results
            final_context = api.get_data(conv_id)
            answer = self._extract_answer(final_context, responses)

            # Build trace
            trace = AgentTrace(
                tool_calls=[],
                total_iterations=final_context.get(
                    ContextKeys.ITERATION_COUNT, iteration
                ),
            )

            elapsed = time.monotonic() - start_time
            log.info(
                LogMessages.AGENT_COMPLETE.format(iterations=trace.total_iterations)
            )

            return AgentResult(
                answer=answer,
                success=True,
                trace=trace,
                final_context={
                    k: v for k, v in final_context.items() if not k.startswith("_")
                },
            )

        except (AgentTimeoutError, BudgetExhaustedError):
            raise
        except Exception as e:
            raise AgentError(
                f"Agent execution failed: {e}",
                details={"task": task, "iteration": iteration},
            ) from e
        finally:
            api.end_conversation(conv_id)

    def _register_handlers(self, api: API) -> None:
        """Register agent handlers with the API."""
        # Evaluation handler: runs external evaluation on entry to evaluate state
        api.register_handler(
            api.create_handler(HandlerNames.EVAL_OPT_EVALUATOR)
            .on_state_entry(EvalOptStates.EVALUATE)
            .do(self._run_evaluation)
        )

        # Iteration limiter
        api.register_handler(
            api.create_handler(HandlerNames.ITERATION_LIMITER)
            .at(HandlerTiming.PRE_TRANSITION)
            .do(self._check_iteration_limit)
        )

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
            logger.warning(f"Evaluation function raised an exception: {e}")
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

        # Check if max refinements reached — force pass with best effort
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
    ) -> str:
        """Extract the final answer from context or responses."""
        # Priority: final_answer > generated_output > last response
        answer = final_context.get(ContextKeys.FINAL_ANSWER)
        if answer and isinstance(answer, str) and len(answer) > 5:
            return str(answer)

        output = final_context.get(ContextKeys.GENERATED_OUTPUT)
        if output and isinstance(output, str) and len(output) > 5:
            return str(output)

        for response in reversed(responses):
            if response and len(response.strip()) > 5:
                return response.strip()

        return "Agent could not determine an answer."
