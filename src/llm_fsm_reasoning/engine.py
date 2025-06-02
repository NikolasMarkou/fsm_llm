"""
Structured Reasoning Engine for LLM-FSM
=======================================

A sophisticated reasoning framework that enables language models to perform complex reasoning
by breaking down thought processes into structured, manageable states using Finite State Machines.
"""
from typing import Dict, Any, Optional, Tuple, List

from llm_fsm import API
from llm_fsm.logging import logger
from llm_fsm.handlers import HandlerTiming


from .handlers import ReasoningHandlers
from .models import ReasoningTrace, ClassificationResult
from .utilities import load_fsm_definition, map_reasoning_type
from .constants import ReasoningType, ContextKeys, MergeStrategy, DEFAULT_MODEL


class ReasoningEngine:
    """
    Main reasoning engine that orchestrates different reasoning strategies.

    This class provides a structured approach to problem-solving by guiding
    models through appropriate reasoning patterns based on the problem type.
    """

    def __init__(self, model: str = DEFAULT_MODEL, **kwargs):
        """
        Initialize the reasoning engine.

        :param model: The LLM model to use
        :param kwargs: Additional arguments for the API
        """
        self.model = model
        self.kwargs = kwargs

        # Load FSM definitions
        self.main_fsm = load_fsm_definition("orchestrator")
        self.classification_fsm = load_fsm_definition("classifier")
        self.reasoning_fsms = self._load_reasoning_fsms()

        # Handler instance
        self.handlers = ReasoningHandlers()

        # Initialize APIs
        self._initialize_apis()

        logger.info(f"Reasoning engine initialized with model: {model}")

    def _load_reasoning_fsms(self) -> Dict[ReasoningType, Dict[str, Any]]:
        """Load all reasoning FSM definitions."""
        return {
            ReasoningType.ANALYTICAL: load_fsm_definition("analytical"),
            ReasoningType.DEDUCTIVE: load_fsm_definition("deductive"),
            ReasoningType.INDUCTIVE: load_fsm_definition("inductive"),
            ReasoningType.CREATIVE: load_fsm_definition("creative"),
            ReasoningType.CRITICAL: load_fsm_definition("critical"),
            ReasoningType.HYBRID: load_fsm_definition("hybrid")
        }

    def _initialize_apis(self):
        """Initialize the main and classification APIs with handlers."""
        # Main orchestrator API
        self.reasoner = API.from_definition(self.main_fsm, model=self.model, **self.kwargs)

        # Classification API
        self.classification_api = API.from_definition(
            self.classification_fsm,
            model=self.model,
            **self.kwargs
        )

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register all handlers for the reasoning engine."""
        # Problem classifier
        self.reasoner.register_handler(
            self.reasoner.create_handler("ProblemClassifier")
            .at(HandlerTiming.CONTEXT_UPDATE)
            .when_keys_updated(ContextKeys.PROBLEM_TYPE)
            .do(self._classify_problem)
        )

        # Strategy executor
        self.reasoner.register_handler(
            self.reasoner.create_handler("StrategyExecutor")
            .on_state_entry("execute_reasoning")
            .do(self._execute_strategy)
        )

        # Solution validator
        self.reasoner.register_handler(
            self.reasoner.create_handler("SolutionValidator")
            .at(HandlerTiming.CONTEXT_UPDATE)
            .when_keys_updated(ContextKeys.PROPOSED_SOLUTION)
            .do(self.handlers.validate_solution)
        )

        # Reasoning tracer
        self.reasoner.register_handler(
            self.reasoner.create_handler("ReasoningTracer")
            .at(HandlerTiming.POST_TRANSITION)
            .do(self.handlers.update_reasoning_trace)
        )

    def _classify_problem(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use FSM-based classification to determine the best reasoning approach.

        :param context: Current conversation context
        :return: Dictionary with problem classification
        """
        # Prepare classification context
        classification_context = {
            ContextKeys.PROBLEM_STATEMENT: context.get(ContextKeys.PROBLEM_STATEMENT, ""),
            ContextKeys.PROBLEM_TYPE: context.get(ContextKeys.PROBLEM_TYPE, ""),
            ContextKeys.PROBLEM_COMPONENTS: context.get(ContextKeys.PROBLEM_COMPONENTS, "")
        }

        # Run classification
        conv_id, _ = self.classification_api.start_conversation(classification_context)

        while not self.classification_api.has_conversation_ended(conv_id):
            self.classification_api.converse("Continue analysis", conv_id)

        # Get results
        result = self.classification_api.get_data(conv_id)
        self.classification_api.end_conversation(conv_id)

        # Create classification result
        classification = ClassificationResult(
            recommended_type=result.get("recommended_reasoning_type", "analytical"),
            justification=result.get("strategy_justification", ""),
            domain=result.get("problem_domain", ""),
            alternatives=result.get("alternative_approaches", []),
            confidence="high"
        )

        logger.info(f"Problem classified as: {classification.recommended_type}")

        return {
            ContextKeys.CLASSIFIED_PROBLEM_TYPE: classification.recommended_type,
            "classification_reasoning": classification.justification,
            "problem_domain": classification.domain,
            "alternative_approaches": classification.alternatives
        }

    def _execute_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the selected reasoning strategy.

        :param context: Current conversation context
        :return: Dictionary with strategy execution setup
        """
        # Get reasoning type
        classified_type = context.get(ContextKeys.CLASSIFIED_PROBLEM_TYPE)
        strategy = context.get(ContextKeys.REASONING_STRATEGY, "analytical")

        reasoning_type_str = classified_type or strategy
        reasoning_type = ReasoningType(map_reasoning_type(reasoning_type_str))

        logger.info(f"Executing {reasoning_type.value} reasoning strategy")

        return {
            "reasoning_fsm_to_push": self.reasoning_fsms[reasoning_type],
            "reasoning_type_selected": reasoning_type.value,
            "classification_justification": context.get("classification_reasoning", "")
        }

    def solve_problem(
        self,
        problem: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Solve a problem using structured reasoning.

        :param problem: The problem statement
        :param initial_context: Optional initial context
        :return: Tuple of (solution, reasoning_trace_info)
        """
        # Initialize context
        context = initial_context or {}
        context[ContextKeys.PROBLEM_STATEMENT] = problem

        # Start reasoning
        conv_id, initial_response = self.reasoner.start_conversation(context)
        logger.info("Started reasoning process")

        responses = [initial_response]

        # Process until complete
        while not self.reasoner.has_conversation_ended(conv_id):
            current_context = self.reasoner.get_data(conv_id)

            if current_context.get("reasoning_fsm_to_push"):
                # Push specialized reasoning FSM
                fsm_def = current_context["reasoning_fsm_to_push"]
                response = self.reasoner.push_fsm(
                    conv_id,
                    fsm_def,
                    inherit_context=True,
                    shared_context_keys=[
                        ContextKeys.PROBLEM_STATEMENT,
                        ContextKeys.PROBLEM_COMPONENTS
                    ]
                )

                # Clear flag and continue
                self.reasoner.converse("Continue with the reasoning process", conv_id)

                # Execute pushed FSM
                while self.reasoner.get_stack_depth(conv_id) > 1:
                    response = self.reasoner.converse("Continue reasoning", conv_id)
                    responses.append(response)

                # Pop back
                response = self.reasoner.pop_fsm(conv_id, merge_strategy=MergeStrategy.UPDATE)
                responses.append(response)
            else:
                # Normal flow
                response = self.reasoner.converse("Continue", conv_id)
                responses.append(response)

        # Get final results
        final_context = self.reasoner.get_data(conv_id)
        solution = final_context.get(ContextKeys.FINAL_SOLUTION, "No solution found")
        trace = final_context.get(ContextKeys.REASONING_TRACE, [])

        # Create trace info
        trace_info = ReasoningTrace(
            steps=trace,
            total_steps=len(trace),
            reasoning_types_used=[final_context.get("reasoning_type_selected", "unknown")],
            final_confidence=final_context.get("solution_confidence", 0.0)
        )

        logger.info(f"Problem solved with {trace_info.total_steps} reasoning steps")

        return solution, {
            "reasoning_trace": trace_info.dict(),
            "full_context": final_context,
            "all_responses": responses
        }