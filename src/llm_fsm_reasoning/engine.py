"""
Structured Reasoning Engine for LLM-FSM
=======================================

Enhanced with loop prevention, context management, and standardized handling.
"""
from typing import Dict, Any, Optional, Tuple, List

from llm_fsm import API
from llm_fsm.logging import logger
from llm_fsm.handlers import HandlerTiming

from .handlers import ReasoningHandlers, ContextManager, OutputFormatter
from .models import ReasoningTrace, ClassificationResult
from .utilities import load_fsm_definition, map_reasoning_type
from .constants import (
    ReasoningType, ContextKeys, MergeStrategy, Defaults,
    HandlerNames, LogMessages, ErrorMessages, OrchestratorStates
)


class ReasoningEngine:
    """
    Main reasoning engine with enhanced error handling and flow control.
    """

    def __init__(self, model: str = Defaults.MODEL, **kwargs):
        """
        Initialize the reasoning engine.

        :param model: The LLM model to use
        :param kwargs: Additional arguments for the API
        """
        self.model = model
        self.api_kwargs = kwargs.copy()

        # Load FSM definitions
        self._load_fsm_definitions()

        # Initialize components
        self.handlers = ReasoningHandlers()
        self.context_manager = ContextManager()
        self.output_formatter = OutputFormatter()

        # Initialize APIs
        self._initialize_apis()

        logger.info(LogMessages.ENGINE_INITIALIZED.format(model=model))

    def _load_fsm_definitions(self):
        """Load all FSM definitions with error handling."""
        try:
            self.main_fsm = load_fsm_definition("orchestrator")
            self.classifier_fsm = load_fsm_definition("classifier")

            # Load reasoning FSMs
            self.reasoning_fsms = {}
            for reasoning_type in ReasoningType:
                try:
                    fsm = load_fsm_definition(reasoning_type.value)
                    self.reasoning_fsms[reasoning_type] = fsm
                except Exception as e:
                    logger.warning(
                        f"Could not load FSM for {reasoning_type.value}: {e}"
                    )
        except Exception as e:
            logger.error(f"Failed to load core FSM definitions: {e}")
            raise

    def _initialize_apis(self):
        """Initialize APIs with all necessary handlers."""
        # Main orchestrator API
        self.orchestrator = API.from_definition(
            self.main_fsm,
            model=self.model,
            **self.api_kwargs
        )

        # Classification API
        self.classifier = API.from_definition(
            self.classifier_fsm,
            model=self.model,
            **self.api_kwargs
        )

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register all handlers with proper configuration."""
        # Problem classifier
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.ORCHESTRATOR_CLASSIFIER)
            .at(HandlerTiming.CONTEXT_UPDATE)
            .when_keys_updated(ContextKeys.PROBLEM_TYPE)
            .do(self._classify_problem)
        )

        # Strategy executor
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.ORCHESTRATOR_EXECUTOR)
            .on_state_entry(OrchestratorStates.EXECUTE_REASONING)
            .do(self._prepare_reasoning_execution)
        )

        # Solution validator with retry logic
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.ORCHESTRATOR_VALIDATOR)
            .at(HandlerTiming.CONTEXT_UPDATE)
            .when_keys_updated(ContextKeys.PROPOSED_SOLUTION)
            .do(self.handlers.validate_solution)
        )

        # Context pruner
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.CONTEXT_PRUNER)
            .at(HandlerTiming.PRE_TRANSITION)
            .do(self.handlers.prune_context)
        )

        # Reasoning tracer for both APIs
        tracer = (self.orchestrator.create_handler(HandlerNames.REASONING_TRACER)
                  .at(HandlerTiming.POST_TRANSITION).do(self.handlers.update_reasoning_trace))

        self.orchestrator.register_handler(tracer)
        self.classifier.register_handler(tracer)

        # Retry limiter for validation loops
        self.orchestrator.register_handler(
            self.orchestrator.create_handler(HandlerNames.RETRY_LIMITER)
            .on_state_entry(OrchestratorStates.VALIDATE_REFINE)
            .do(self._check_retry_limit)
        )

    def _classify_problem(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify problem using the classification FSM.

        :param context: Current context
        :return: Classification results
        """
        # Extract relevant context for classification
        classification_context = self.context_manager.extract_relevant_context(
            context,
            [ContextKeys.PROBLEM_STATEMENT, ContextKeys.PROBLEM_TYPE,
             ContextKeys.PROBLEM_COMPONENTS]
        )

        logger.info(LogMessages.CLASSIFICATION_STARTED.format(
            context=list(classification_context.keys())
        ))

        # Run classification
        conv_id, _ = self.classifier.start_conversation(classification_context)

        while not self.classifier.has_conversation_ended(conv_id):
            self.classifier.converse("Continue analysis", conv_id)

        # Get results
        result = self.classifier.get_data(conv_id)
        self.classifier.end_conversation(conv_id)

        # Create classification result
        classification = ClassificationResult(
            recommended_type=result.get(ContextKeys.RECOMMENDED_REASONING_TYPE, "analytical"),
            justification=result.get(ContextKeys.STRATEGY_JUSTIFICATION, ""),
            domain=result.get(ContextKeys.PROBLEM_DOMAIN, ""),
            alternatives=result.get(ContextKeys.ALTERNATIVE_APPROACHES, [])
        )

        logger.info(LogMessages.CLASSIFICATION_COMPLETE.format(
            type=classification.recommended_type
        ))

        return {
            ContextKeys.CLASSIFIED_PROBLEM_TYPE: classification.recommended_type,
            ContextKeys.CLASSIFICATION_JUSTIFICATION: classification.justification,
            "problem_domain_classified": classification.domain,
            "alternative_approaches": classification.alternatives
        }

    def _prepare_reasoning_execution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for reasoning execution with proper strategy selection.

        :param context: Current context
        :return: Execution preparation results
        """
        # Determine reasoning type
        orchestrator_strategy = context.get(ContextKeys.REASONING_STRATEGY)
        classified_type = context.get(ContextKeys.CLASSIFIED_PROBLEM_TYPE)

        # Priority: direct computation > classified type > orchestrator strategy
        if orchestrator_strategy == "direct computation":
            reasoning_type_str = "simple_calculator"
        elif classified_type:
            reasoning_type_str = map_reasoning_type(classified_type)
        elif orchestrator_strategy:
            reasoning_type_str = map_reasoning_type(orchestrator_strategy)
        else:
            reasoning_type_str = "analytical"  # Default

        # Get enum member
        try:
            reasoning_type = ReasoningType(reasoning_type_str)
        except ValueError:
            logger.warning(ErrorMessages.INVALID_REASONING_TYPE.format(
                type=reasoning_type_str
            ))
            reasoning_type = ReasoningType.ANALYTICAL

        # Get FSM definition
        fsm_def = self.reasoning_fsms.get(reasoning_type)

        if not fsm_def:
            logger.error(ErrorMessages.FSM_NOT_FOUND.format(
                name=reasoning_type.value
            ))
            # Fallback to analytical
            reasoning_type = ReasoningType.ANALYTICAL
            fsm_def = self.reasoning_fsms.get(reasoning_type)

        logger.info(LogMessages.STRATEGY_EXECUTING.format(
            type=reasoning_type.value
        ))

        return {
            ContextKeys.REASONING_FSM_TO_PUSH: fsm_def,
            ContextKeys.REASONING_TYPE_SELECTED: reasoning_type.value,
            ContextKeys.CLASSIFICATION_JUSTIFICATION: context.get(
                ContextKeys.CLASSIFICATION_JUSTIFICATION, ""
            )
        }

    def _check_retry_limit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if retry limit has been reached.

        :param context: Current context
        :return: Retry status
        """
        retry_count = context.get(ContextKeys.RETRY_COUNT, 0)
        max_reached = retry_count >= Defaults.MAX_RETRIES

        if max_reached:
            logger.warning(ErrorMessages.MAX_RETRIES_EXCEEDED)

        return {ContextKeys.MAX_RETRIES_REACHED: max_reached}

    def solve_problem(
        self,
        problem: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Solve a problem using structured reasoning.

        :param problem: The problem statement
        :param initial_context: Optional initial context
        :return: Tuple of (solution, trace_info)
        """
        # Initialize context
        context = initial_context or {}
        context[ContextKeys.PROBLEM_STATEMENT] = problem
        context[ContextKeys.REASONING_TRACE] = []
        context[ContextKeys.RETRY_COUNT] = 0

        # Start orchestrator
        conv_id, initial_response = self.orchestrator.start_conversation(context)
        logger.info(f"Started reasoning process: {conv_id}")

        responses = [initial_response]

        # Process until complete
        while not self.orchestrator.has_conversation_ended(conv_id):
            current_context = self.orchestrator.get_data(conv_id)

            # Check for FSM to push
            fsm_to_push = current_context.get(ContextKeys.REASONING_FSM_TO_PUSH)

            if fsm_to_push:
                # Clear the flag
                self.orchestrator.fsm_manager.update_conversation_context(
                    conv_id,
                    {ContextKeys.REASONING_FSM_TO_PUSH: None}
                )

                # Prepare context for sub-FSM
                sub_context = self.context_manager.extract_relevant_context(
                    current_context,
                    [
                        ContextKeys.PROBLEM_STATEMENT,
                        ContextKeys.PROBLEM_COMPONENTS,
                        ContextKeys.CONSTRAINTS,
                        ContextKeys.PROBLEM_TYPE
                    ]
                )

                # Push sub-FSM
                logger.info(LogMessages.FSM_PUSHED.format(
                    name=fsm_to_push.get("name"),
                    depth=self.orchestrator.get_stack_depth(conv_id) + 1
                ))

                sub_response = self.orchestrator.push_fsm(
                    conv_id,
                    fsm_to_push,
                    inherit_context=False,
                    context_to_pass=sub_context
                )
                responses.append(sub_response)

                # Execute sub-FSM
                sub_conv_id = self.orchestrator.conversation_stacks[conv_id][-1].conversation_id

                while self.orchestrator.get_stack_depth(conv_id) > 1:
                    if self.orchestrator.fsm_manager.has_conversation_ended(sub_conv_id):
                        break

                    response = self.orchestrator.converse("Continue reasoning", conv_id)
                    responses.append(response)

                # Get results from sub-FSM
                sub_final_context = self.orchestrator.fsm_manager.get_conversation_data(
                    sub_conv_id
                )
                reasoning_type = current_context.get(ContextKeys.REASONING_TYPE_SELECTED)

                # Map results back
                results = self.context_manager.merge_reasoning_results(
                    current_context,
                    sub_final_context,
                    reasoning_type
                )

                # Pop with results
                pop_response = self.orchestrator.pop_fsm(
                    conv_id,
                    context_to_return=results,
                    merge_strategy=MergeStrategy.UPDATE
                )
                responses.append(pop_response)

                logger.info(LogMessages.FSM_POPPED.format(
                    name=fsm_to_push.get("name"),
                    depth=self.orchestrator.get_stack_depth(conv_id)
                ))
            else:
                # Normal orchestrator progression
                response = self.orchestrator.converse("Continue", conv_id)
                responses.append(response)

        # Get final context and extract solution
        final_context = self.orchestrator.get_data(conv_id)
        solution = self.output_formatter.extract_final_solution(final_context)

        # Build trace info
        trace_steps = final_context.get(ContextKeys.REASONING_TRACE, [])
        reasoning_types = self._extract_reasoning_types(final_context, trace_steps)

        trace_info = ReasoningTrace(
            steps=trace_steps,
            total_steps=len(trace_steps),
            reasoning_types_used=list(reasoning_types),
            final_confidence=final_context.get(ContextKeys.SOLUTION_CONFIDENCE, 0.0)
        )

        logger.info(LogMessages.PROBLEM_SOLVED.format(steps=trace_info.total_steps))

        # Clean up
        self.orchestrator.end_conversation(conv_id)

        return solution, {
            "reasoning_trace": trace_info.model_dump(),
            "summary": self.output_formatter.format_reasoning_summary(
                trace_info.model_dump()
            ),
            "final_context": final_context,
            "all_responses": responses
        }

    def _extract_reasoning_types(
        self,
        final_context: Dict[str, Any],
        trace_steps: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract unique reasoning types used."""
        types = set()

        # From final context
        if ContextKeys.REASONING_TYPE_SELECTED in final_context:
            types.add(final_context[ContextKeys.REASONING_TYPE_SELECTED])

        # From trace steps
        for step in trace_steps:
            snapshot = step.get("context_snapshot", {})
            if ContextKeys.REASONING_TYPE_SELECTED in snapshot:
                types.add(snapshot[ContextKeys.REASONING_TYPE_SELECTED])

        return list(types) if types else ["unknown"]