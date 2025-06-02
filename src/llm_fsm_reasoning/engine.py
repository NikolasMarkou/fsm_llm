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
# CHANGE: Added SIMPLE_CALCULATOR to imports
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
        :param kwargs: Additional arguments for the API constructor (e.g., temperature, api_key)
        """
        self.model = model
        self.api_constructor_kwargs = kwargs.copy()

        self.main_fsm_dict = load_fsm_definition("orchestrator")
        self.classification_fsm_dict = load_fsm_definition("classifier")
        self.reasoning_fsms_dicts = self._load_reasoning_fsms_as_dicts()

        self.shared_handlers = ReasoningHandlers()
        self._initialize_apis()
        logger.info(f"Reasoning engine initialized with model: {model}")

    def _load_reasoning_fsms_as_dicts(self) -> Dict[ReasoningType, Dict[str, Any]]:
        fsms = {}
        for rt_enum in ReasoningType:
            try:
                fsm_dict = load_fsm_definition(rt_enum.value)  # This will now use the new loader
                fsms[rt_enum] = fsm_dict
            except Exception as e:
                logger.warning(
                    f"Could not load FSM definition for reasoning type '{rt_enum.value}': {e}. This strategy will be unavailable.")
        return fsms

    def _initialize_apis(self):
        self.reasoner = API.from_definition(
            self.main_fsm_dict, model=self.model, **self.api_constructor_kwargs
        )
        self.classification_api = API.from_definition(
            self.classification_fsm_dict, model=self.model, **self.api_constructor_kwargs
        )
        self._register_handlers_for_apis()

    def _register_handlers_for_apis(self):
        problem_classifier_handler = self.reasoner.create_handler("OrchestratorProblemClassifier") \
            .at(HandlerTiming.CONTEXT_UPDATE) \
            .when_keys_updated(ContextKeys.PROBLEM_TYPE) \
            .do(self._orchestrator_classify_problem_handler)
        self.reasoner.register_handler(problem_classifier_handler)

        strategy_executor_handler = self.reasoner.create_handler("OrchestratorStrategyExecutor") \
            .on_state_entry("execute_reasoning") \
            .do(self._orchestrator_execute_strategy_handler)
        self.reasoner.register_handler(strategy_executor_handler)

        solution_validator_handler = self.reasoner.create_handler("OrchestratorSolutionValidator") \
            .at(HandlerTiming.CONTEXT_UPDATE) \
            .when_keys_updated(ContextKeys.PROPOSED_SOLUTION) \
            .do(self.shared_handlers.validate_solution)
        self.reasoner.register_handler(solution_validator_handler)

        reasoning_tracer_handler_obj = self.reasoner.create_handler("ReasoningTracer") \
            .at(HandlerTiming.POST_TRANSITION) \
            .do(self.shared_handlers.update_reasoning_trace)
        self.reasoner.register_handler(reasoning_tracer_handler_obj)
        self.classification_api.register_handler(reasoning_tracer_handler_obj)

    def _orchestrator_classify_problem_handler(self, orchestrator_context: Dict[str, Any]) -> Dict[str, Any]:
        classification_run_context = {
            ContextKeys.PROBLEM_STATEMENT: orchestrator_context.get(ContextKeys.PROBLEM_STATEMENT, ""),
            ContextKeys.PROBLEM_TYPE: orchestrator_context.get(ContextKeys.PROBLEM_TYPE, ""),
            ContextKeys.PROBLEM_COMPONENTS: orchestrator_context.get(ContextKeys.PROBLEM_COMPONENTS, [])
        }
        classification_run_context = {k: v for k, v in classification_run_context.items() if v is not None}
        logger.info(f"Orchestrator's ProblemClassifier Handler: Starting classification FSM with context: {classification_run_context}")

        classifier_initial_context = classification_run_context.copy()
        classifier_initial_context[ContextKeys.REASONING_TRACE] = []

        classifier_conv_id, _ = self.classification_api.start_conversation(initial_context=classifier_initial_context)
        while not self.classification_api.has_conversation_ended(classifier_conv_id):
            self.classification_api.converse("Continue analysis", classifier_conv_id)

        classifier_result_context = self.classification_api.get_data(classifier_conv_id)
        self.classification_api.end_conversation(classifier_conv_id)

        classification = ClassificationResult(
            recommended_type=classifier_result_context.get("recommended_reasoning_type", "analytical"),
            justification=classifier_result_context.get("strategy_justification", "N/A"),
            domain=classifier_result_context.get("problem_domain", "general"),
            alternatives=classifier_result_context.get("alternative_approaches", []),
            confidence=classifier_result_context.get("confidence_level", "medium")
        )
        logger.info(f"Orchestrator's ProblemClassifier Handler: Classification complete. Recommended strategy: {classification.recommended_type}.")
        return {
            ContextKeys.CLASSIFIED_PROBLEM_TYPE: classification.recommended_type,
            "classification_reasoning": classification.justification,
            "problem_domain_classified": classification.domain,
            "alternative_approaches_classified": classification.alternatives
        }

    # CHANGE: Modified _orchestrator_execute_strategy_handler to prioritize "direct computation"
    # and use ReasoningType.SIMPLE_CALCULATOR
    def _orchestrator_execute_strategy_handler(self, orchestrator_context: Dict[str, Any]) -> Dict[str, Any]:
        classified_type = orchestrator_context.get(ContextKeys.CLASSIFIED_PROBLEM_TYPE)
        orchestrator_selected_strategy = orchestrator_context.get(ContextKeys.REASONING_STRATEGY)

        reasoning_type_to_execute_str = "analytical"  # Default strategy

        # Prioritize "direct computation" from the orchestrator's strategy selection
        if orchestrator_selected_strategy == "direct computation":
            reasoning_type_to_execute_str = "simple_calculator"
        elif classified_type:
            # Use classifier's recommendation if "direct computation" was not chosen
            reasoning_type_to_execute_str = map_reasoning_type(classified_type)
        elif orchestrator_selected_strategy:
            # Fallback to orchestrator's general strategy if classifier didn't provide one
            reasoning_type_to_execute_str = map_reasoning_type(orchestrator_selected_strategy)

        reasoning_type_enum_member = ReasoningType.ANALYTICAL  # Default enum member
        try:
            reasoning_type_enum_member = ReasoningType(reasoning_type_to_execute_str)
        except ValueError:
            logger.warning(f"Invalid reasoning type string '{reasoning_type_to_execute_str}' in orchestrator context. Defaulting to ANALYTICAL.")
            reasoning_type_enum_member = ReasoningType.ANALYTICAL # Ensure it's a valid enum

        logger.info(f"Orchestrator's StrategyExecutor Handler: Preparing to execute {reasoning_type_enum_member.value} reasoning strategy.")
        fsm_to_push_dict = self.reasoning_fsms_dicts.get(reasoning_type_enum_member)

        if not fsm_to_push_dict:
            logger.error(f"No FSM definition dictionary found for reasoning type: {reasoning_type_enum_member.value}")
            # Specific fallback if simple_calculator was intended but FSM is missing
            if reasoning_type_enum_member == ReasoningType.SIMPLE_CALCULATOR:
                logger.warning("Simple calculator FSM requested but not loaded. Attempting analytical fallback.")
                reasoning_type_enum_member = ReasoningType.ANALYTICAL # Fallback to analytical
                fsm_to_push_dict = self.reasoning_fsms_dicts.get(reasoning_type_enum_member)
                if not fsm_to_push_dict: # If analytical is also missing, then error
                    return {"error_executing_strategy": f"Missing FSM for {reasoning_type_enum_member.value} and fallback analytical."}
            else:
                return {"error_executing_strategy": f"Missing FSM for {reasoning_type_enum_member.value}"}

        return {
            "reasoning_fsm_to_push": fsm_to_push_dict,
            "reasoning_type_selected": reasoning_type_enum_member.value, # Crucial for context merging later
            "classification_justification": orchestrator_context.get("classification_reasoning", "")
        }

    def solve_problem(
        self,
        problem: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        orchestrator_initial_context = initial_context or {}
        orchestrator_initial_context[ContextKeys.PROBLEM_STATEMENT] = problem
        orchestrator_initial_context[ContextKeys.REASONING_TRACE] = []

        orchestrator_conv_id, initial_orchestrator_response = self.reasoner.start_conversation(orchestrator_initial_context)
        logger.info(f"Started reasoning process with orchestrator_conv_id: {orchestrator_conv_id}")
        all_user_facing_responses = [initial_orchestrator_response]

        while not self.reasoner.has_conversation_ended(orchestrator_conv_id):
            current_orchestrator_context = self.reasoner.get_data(orchestrator_conv_id)
            sub_fsm_def_to_push = current_orchestrator_context.get("reasoning_fsm_to_push")

            if sub_fsm_def_to_push:
                logger.info(f"Orchestrator intends to push FSM: {sub_fsm_def_to_push.get('name')}. Clearing 'reasoning_fsm_to_push' flag.")
                self.reasoner.fsm_manager.update_conversation_context(
                    orchestrator_conv_id,
                    {"reasoning_fsm_to_push": None}
                )
                logger.debug(f"Orchestrator context updated: 'reasoning_fsm_to_push' set to None for conv_id {orchestrator_conv_id}.")

                context_for_sub_fsm = {
                    ContextKeys.PROBLEM_STATEMENT: current_orchestrator_context.get(ContextKeys.PROBLEM_STATEMENT),
                    ContextKeys.PROBLEM_COMPONENTS: current_orchestrator_context.get(ContextKeys.PROBLEM_COMPONENTS),
                    ContextKeys.CONSTRAINTS: current_orchestrator_context.get(ContextKeys.CONSTRAINTS),
                    ContextKeys.PROBLEM_TYPE: current_orchestrator_context.get(ContextKeys.PROBLEM_TYPE),
                }
                context_for_sub_fsm = {k: v for k, v in context_for_sub_fsm.items() if v is not None}

                logger.info(f"Pushing sub-FSM '{sub_fsm_def_to_push.get('name')}'. Context to pass: {list(context_for_sub_fsm.keys())}")

                sub_fsm_initial_response = self.reasoner.push_fsm(
                    orchestrator_conv_id,
                    sub_fsm_def_to_push,
                    inherit_context=False,
                    context_to_pass=context_for_sub_fsm,
                    shared_context_keys=[]
                )
                all_user_facing_responses.append(sub_fsm_initial_response)

                current_sub_fsm_conv_id = self.reasoner.conversation_stacks[orchestrator_conv_id][-1].conversation_id
                logger.info(f"Sub-FSM (conv_id: {current_sub_fsm_conv_id}) pushed. Stack depth: {self.reasoner.get_stack_depth(orchestrator_conv_id)}.")

                while self.reasoner.get_stack_depth(orchestrator_conv_id) > 1:
                    if self.reasoner.fsm_manager.has_conversation_ended(current_sub_fsm_conv_id):
                        logger.info(f"Sub-FSM (conv_id: {current_sub_fsm_conv_id}) has reached its terminal state. Preparing to pop.")
                        break
                    sub_fsm_response = self.reasoner.converse("Continue reasoning", orchestrator_conv_id)
                    all_user_facing_responses.append(sub_fsm_response)

                logger.info(f"Popping sub-FSM (conv_id: {current_sub_fsm_conv_id}).")
                sub_fsm_final_context = self.reasoner.fsm_manager.get_conversation_data(current_sub_fsm_conv_id)

                results_from_sub_fsm = {}
                orchestrator_context_at_pop_time = self.reasoner.fsm_manager.get_conversation_data(orchestrator_conv_id)
                reasoning_type_executed = orchestrator_context_at_pop_time.get("reasoning_type_selected", "unknown")

                # CHANGE: Added specific context mapping for SIMPLE_CALCULATOR
                if reasoning_type_executed == ReasoningType.ANALYTICAL.value:
                    results_from_sub_fsm[ContextKeys.KEY_INSIGHTS] = sub_fsm_final_context.get("key_insights")
                    results_from_sub_fsm["integrated_analysis"] = sub_fsm_final_context.get("integrated_analysis")
                elif reasoning_type_executed == ReasoningType.DEDUCTIVE.value:
                    results_from_sub_fsm["deductive_conclusion"] = sub_fsm_final_context.get("conclusion")
                    results_from_sub_fsm["logical_validity"] = sub_fsm_final_context.get("logical_validity")
                elif reasoning_type_executed == ReasoningType.INDUCTIVE.value:
                    results_from_sub_fsm["inductive_hypothesis"] = sub_fsm_final_context.get("hypothesis")
                    results_from_sub_fsm["generalization_strength"] = sub_fsm_final_context.get("generalization_strength")
                elif reasoning_type_executed == ReasoningType.CREATIVE.value:
                    results_from_sub_fsm["best_creative_solution"] = sub_fsm_final_context.get("best_creative_solution")
                    results_from_sub_fsm["innovation_rating"] = sub_fsm_final_context.get("innovation_rating")
                elif reasoning_type_executed == ReasoningType.CRITICAL.value:
                    results_from_sub_fsm["critical_assessment"] = sub_fsm_final_context.get("critical_assessment")
                    results_from_sub_fsm["assessment_confidence"] = sub_fsm_final_context.get("confidence_rating")
                elif reasoning_type_executed == ReasoningType.SIMPLE_CALCULATOR.value: # New case
                    results_from_sub_fsm[ContextKeys.PROPOSED_SOLUTION] = sub_fsm_final_context.get("calculation_result")
                    if "calculation_error" in sub_fsm_final_context:
                        results_from_sub_fsm["calculation_error_details"] = sub_fsm_final_context.get("calculation_error")
                elif reasoning_type_executed == ReasoningType.HYBRID.value:
                     results_from_sub_fsm["final_hybrid_solution"] = sub_fsm_final_context.get("final_hybrid_solution")
                     results_from_sub_fsm["hybrid_synthesis_summary"] = sub_fsm_final_context.get("reasoning_synthesis")
                # END CHANGE

                results_from_sub_fsm = {k: v for k, v in results_from_sub_fsm.items() if v is not None}
                results_from_sub_fsm[f"{reasoning_type_executed}_reasoning_completed"] = True

                logger.info(f"Context to return from sub-FSM to orchestrator: {list(results_from_sub_fsm.keys())}")

                pop_response = self.reasoner.pop_fsm(
                    orchestrator_conv_id,
                    context_to_return=results_from_sub_fsm,
                    merge_strategy=MergeStrategy.UPDATE
                )
                all_user_facing_responses.append(pop_response)
                logger.info(f"Sub-FSM popped. Orchestrator received message: '{pop_response[:50]}...' Stack depth: {self.reasoner.get_stack_depth(orchestrator_conv_id)}")
            else:
                current_orchestrator_state = self.reasoner.get_current_state(orchestrator_conv_id)
                logger.info(f"Orchestrator (conv_id: {orchestrator_conv_id}) in state '{current_orchestrator_state}', processing 'Continue' to advance its own FSM.")
                orchestrator_response = self.reasoner.converse("Continue", orchestrator_conv_id)
                all_user_facing_responses.append(orchestrator_response)
                new_orchestrator_state = self.reasoner.get_current_state(orchestrator_conv_id)
                logger.info(f"Orchestrator advanced. State: '{current_orchestrator_state}' -> '{new_orchestrator_state}'. Response: '{orchestrator_response[:50]}...'")

        final_orchestrator_context = self.reasoner.get_data(orchestrator_conv_id)
        solution = final_orchestrator_context.get(ContextKeys.FINAL_SOLUTION, "Solution process concluded, but no explicit final solution found in context.")
        orchestrator_trace_steps = final_orchestrator_context.get(ContextKeys.REASONING_TRACE, [])

        used_reasoning_types = set()
        if "reasoning_type_selected" in final_orchestrator_context:
            used_reasoning_types.add(final_orchestrator_context["reasoning_type_selected"])
        for step_dict in orchestrator_trace_steps: # Iterate through list of dicts
            context_snapshot = step_dict.get("context_snapshot", {})
            if "reasoning_type_selected" in context_snapshot:
                used_reasoning_types.add(context_snapshot["reasoning_type_selected"])

        trace_info = ReasoningTrace(
            steps=orchestrator_trace_steps,
            total_steps=len(orchestrator_trace_steps),
            reasoning_types_used=list(used_reasoning_types) if used_reasoning_types else ["unknown"],
            final_confidence=final_orchestrator_context.get("solution_confidence", 0.0)
        )

        logger.info(f"Problem solving process complete for orchestrator_conv_id: {orchestrator_conv_id}. Total orchestrator trace steps: {trace_info.total_steps}.")
        self.reasoner.end_conversation(orchestrator_conv_id)

        return solution, {
            "reasoning_trace": trace_info.model_dump(),
            "final_orchestrator_context": final_orchestrator_context,
            "all_user_facing_responses": all_user_facing_responses
        }