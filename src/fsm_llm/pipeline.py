from __future__ import annotations

"""
MessagePipeline: The 2-pass message processing engine.

Encapsulates all LLM-driven processing logic extracted from FSMManager:
- Pass 1: Data extraction + transition evaluation + state transition
- Pass 2: Response generation from final state
- Handler execution bridge (deep-copy context, merge deltas)

FSMManager delegates to this class for all message processing.
The pipeline does not own instances or locks — those remain in FSMManager.
"""

import copy
import time
from collections.abc import Callable
from typing import Any

from .definitions import (
    DataExtractionRequest,
    DataExtractionResponse,
    FSMDefinition,
    FSMInstance,
    InvalidTransitionError,
    ResponseGenerationRequest,
    State,
    StateNotFoundError,
    TransitionDecisionRequest,
    TransitionEvaluation,
    TransitionEvaluationResult,
)
from .handlers import HandlerSystem, HandlerTiming
from .llm import LLMInterface
from .logging import logger
from .prompts import (
    DataExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
    TransitionPromptBuilder,
)
from .transition_evaluator import TransitionEvaluator


class MessagePipeline:
    """2-pass message processing pipeline.

    Handles data extraction, transition evaluation, state transitions,
    response generation, and handler execution. Stateless with respect
    to conversation instances — all state is passed as parameters.
    """

    def __init__(
        self,
        llm_interface: LLMInterface,
        data_extraction_prompt_builder: DataExtractionPromptBuilder,
        response_generation_prompt_builder: ResponseGenerationPromptBuilder,
        transition_prompt_builder: TransitionPromptBuilder,
        transition_evaluator: TransitionEvaluator,
        handler_system: HandlerSystem,
        fsm_resolver: Callable[[str], FSMDefinition],
    ):
        self.llm_interface = llm_interface
        self.data_extraction_prompt_builder = data_extraction_prompt_builder
        self.response_generation_prompt_builder = response_generation_prompt_builder
        self.transition_prompt_builder = transition_prompt_builder
        self.transition_evaluator = transition_evaluator
        self.handler_system = handler_system
        self.fsm_resolver = fsm_resolver

    def get_state(
        self, instance: FSMInstance, conversation_id: str | None = None
    ) -> State:
        """Resolve current State from FSM definition."""
        log = (
            logger.bind(conversation_id=conversation_id) if conversation_id else logger
        )

        fsm_def = self.fsm_resolver(instance.fsm_id)
        if instance.current_state not in fsm_def.states:
            error_msg = (
                f"State '{instance.current_state}' not found in FSM '{instance.fsm_id}'"
            )
            log.error(error_msg)
            raise StateNotFoundError(error_msg)

        return fsm_def.states[instance.current_state]

    # ----------------------------------------------------------
    # Handler execution bridge
    # ----------------------------------------------------------

    def execute_handlers(
        self,
        instance: FSMInstance,
        timing: HandlerTiming,
        conversation_id: str,
        current_state: str | None = None,
        target_state: str | None = None,
        updated_keys: set[str] | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        """Execute handlers at specified timing point.

        Deep-copies instance context before passing to handlers, then merges
        the delta dict back into the instance. A handler returning a key with
        value ``None`` requests deletion of that key.
        """
        context = copy.deepcopy(instance.context.data)

        if error_context:
            context.update(error_context)

        try:
            updated_context = self.handler_system.execute_handlers(
                timing=timing,
                current_state=current_state or instance.current_state,
                target_state=target_state,
                context=context,
                updated_keys=updated_keys,
            )

            if updated_context:
                for key, value in updated_context.items():
                    if value is None:
                        instance.context.data.pop(key, None)
                    else:
                        instance.context.data[key] = value

        except Exception as e:
            logger.error(f"Handler execution error at {timing.name}: {e!s}")
            if self.handler_system.error_mode == "raise":
                raise

    # ----------------------------------------------------------
    # Full 2-pass processing
    # ----------------------------------------------------------

    def process(self, instance: FSMInstance, message: str, conversation_id: str) -> str:
        """Execute the full 2-pass message processing pipeline.

        Pass 1: PRE_PROCESSING handlers → data extraction → context update →
                transition evaluation → state transition
        Pass 2: POST_PROCESSING handlers → response generation

        Args:
            instance: The FSM instance (already validated as non-terminal).
            message: User message to process.
            conversation_id: Conversation identifier.

        Returns:
            Generated response message.
        """
        # Contextualize propagates conversation_id to all downstream logger
        # calls on this thread (llm.py, transition_evaluator.py, etc.)
        with logger.contextualize(conversation_id=conversation_id, package="fsm_llm"):
            # Execute pre-processing handlers
            self.execute_handlers(
                instance,
                HandlerTiming.PRE_PROCESSING,
                conversation_id,
                current_state=instance.current_state,
            )

            # Pass 1: Data extraction + transition evaluation + execution
            extraction_response, transition_occurred, previous_state = (
                self._execute_extraction_and_transition_pass(
                    instance, message, conversation_id
                )
            )

            # Execute post-processing handlers (after potential transition)
            self.execute_handlers(
                instance,
                HandlerTiming.POST_PROCESSING,
                conversation_id,
                current_state=instance.current_state,
            )

            # Pass 2: Response generation based on final state
            return self._execute_response_generation_pass(
                instance,
                message,
                extraction_response,
                transition_occurred,
                previous_state,
                conversation_id,
            )

    # ----------------------------------------------------------
    # Initial response generation
    # ----------------------------------------------------------

    def generate_initial_response(
        self, instance: FSMInstance, conversation_id: str
    ) -> str:
        """Generate initial response for conversation start (no extraction/transition)."""
        log = logger.bind(conversation_id=conversation_id)

        current_state = self.get_state(instance, conversation_id)
        fsm_def = self.fsm_resolver(instance.fsm_id)

        system_prompt = self.response_generation_prompt_builder.build_response_prompt(
            instance=instance,
            state=current_state,
            fsm_definition=fsm_def,
            extracted_data={},
            transition_occurred=False,
            previous_state=None,
            user_message="",
        )

        request = ResponseGenerationRequest(
            system_prompt=system_prompt,
            user_message="",
            extracted_data={},
            context=instance.context.get_user_visible_data(),
            transition_occurred=False,
            previous_state=None,
        )

        response = self.llm_interface.generate_response(request)
        instance.last_response_generation = response
        instance.context.conversation.add_system_message(response.message)

        log.info("Generated initial response")
        return response.message

    # ----------------------------------------------------------
    # Pass 1: Data extraction + transition
    # ----------------------------------------------------------

    def _execute_extraction_and_transition_pass(
        self, instance: FSMInstance, user_message: str, conversation_id: str
    ) -> tuple[DataExtractionResponse, bool, str | None]:
        """Execute Pass 1: Data Extraction + Transition Evaluation + Execution."""
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing data extraction and transition pass")

        # Step 1: Data Extraction
        extraction_response = self._execute_data_extraction(
            instance, user_message, conversation_id
        )

        # Step 2: Update context with extracted data
        if extraction_response.extracted_data:
            extraction_response.extracted_data = self._clean_empty_context_keys(
                data=extraction_response.extracted_data, conversation_id=conversation_id
            )

            if extraction_response.extracted_data:
                instance.context.update(extraction_response.extracted_data)

                self.execute_handlers(
                    instance,
                    HandlerTiming.CONTEXT_UPDATE,
                    conversation_id,
                    current_state=instance.current_state,
                    updated_keys=set(extraction_response.extracted_data.keys()),
                )

        # Step 3: Transition Evaluation and Execution
        transition_occurred, previous_state = (
            self._execute_transition_evaluation_and_execution(
                instance, user_message, extraction_response, conversation_id
            )
        )

        log.debug("Data extraction and transition pass completed")
        return extraction_response, transition_occurred, previous_state

    def _execute_data_extraction(
        self, instance: FSMInstance, user_message: str, conversation_id: str
    ) -> DataExtractionResponse:
        """Execute data extraction from user input."""
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing data extraction")

        current_state = self.get_state(instance, conversation_id)
        fsm_def = self.fsm_resolver(instance.fsm_id)

        system_prompt = self.data_extraction_prompt_builder.build_extraction_prompt(
            instance, current_state, fsm_def
        )

        request = DataExtractionRequest(
            system_prompt=system_prompt,
            user_message=user_message,
            context=instance.context.get_user_visible_data(),
        )

        response = self.llm_interface.extract_data(request)
        instance.last_extraction_response = response

        log.debug(
            f"Data extraction completed: {list(response.extracted_data.keys()) if response.extracted_data else 'no data'}"
        )
        return response

    # ----------------------------------------------------------
    # Pass 1: Transition evaluation and execution
    # ----------------------------------------------------------

    def _execute_transition_evaluation_and_execution(
        self,
        instance: FSMInstance,
        user_message: str,
        extraction_response: DataExtractionResponse,
        conversation_id: str,
    ) -> tuple[bool, str | None]:
        """Evaluate transitions and execute if one is selected."""
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing transition evaluation and execution")

        current_state = self.get_state(instance, conversation_id)

        if not current_state.transitions:
            log.debug("Terminal state reached - no transitions to evaluate")
            return False, None

        previous_state_id = instance.current_state

        evaluation = self.transition_evaluator.evaluate_transitions(
            current_state, instance.context, extraction_response.extracted_data
        )

        target_state = None

        if evaluation.result_type == TransitionEvaluationResult.DETERMINISTIC:
            target_state = evaluation.deterministic_transition
            log.info(f"Deterministic transition selected: {target_state}")

        elif evaluation.result_type == TransitionEvaluationResult.AMBIGUOUS:
            target_state = self._resolve_ambiguous_transition(
                evaluation, user_message, extraction_response, instance, conversation_id
            )
            log.info(f"LLM-assisted transition selected: {target_state}")

        elif evaluation.result_type == TransitionEvaluationResult.BLOCKED:
            log.warning(f"Transitions blocked: {evaluation.blocked_reason}")
            return False, None

        if target_state:
            self._execute_state_transition(instance, target_state, conversation_id)
            return True, previous_state_id

        return False, None

    def _resolve_ambiguous_transition(
        self,
        evaluation: TransitionEvaluation,
        user_message: str,
        extraction_response: DataExtractionResponse,
        instance: FSMInstance,
        conversation_id: str,
    ) -> str:
        """Resolve ambiguous transition using LLM assistance."""
        log = logger.bind(conversation_id=conversation_id)
        log.debug(
            f"Resolving ambiguous transition with {len(evaluation.available_options)} options"
        )

        system_prompt = self.transition_prompt_builder.build_transition_prompt(
            current_state=instance.current_state,
            available_transitions=evaluation.available_options,
            context=instance.context.get_user_visible_data(),
            user_message=user_message,
            extracted_data=extraction_response.extracted_data,
        )

        request = TransitionDecisionRequest(
            system_prompt=system_prompt,
            current_state=instance.current_state,
            available_transitions=evaluation.available_options,
            context=instance.context.get_user_visible_data(),
            user_message=user_message,
            extracted_data=extraction_response.extracted_data,
        )

        response = self.llm_interface.decide_transition(request)
        instance.last_transition_decision = response

        valid_targets = {opt.target_state for opt in evaluation.available_options}
        if response.selected_transition not in valid_targets:
            raise InvalidTransitionError(
                f"LLM selected invalid transition '{response.selected_transition}'. "
                f"Valid options: {sorted(valid_targets)}"
            )

        return response.selected_transition

    def _execute_state_transition(
        self, instance: FSMInstance, target_state: str, conversation_id: str
    ) -> None:
        """Execute state transition with PRE/POST handler integration and rollback."""
        log = logger.bind(conversation_id=conversation_id)
        old_state = instance.current_state

        self.execute_handlers(
            instance,
            HandlerTiming.PRE_TRANSITION,
            conversation_id,
            current_state=old_state,
            target_state=target_state,
        )

        old_context_meta = {
            "_previous_state": instance.context.data.get("_previous_state"),
            "_current_state": instance.context.data.get("_current_state"),
            "_transition_timestamp": instance.context.data.get("_transition_timestamp"),
        }

        instance.current_state = target_state
        instance.context.data.update(
            {
                "_previous_state": old_state,
                "_current_state": target_state,
                "_transition_timestamp": time.time(),
            }
        )

        try:
            self.execute_handlers(
                instance,
                HandlerTiming.POST_TRANSITION,
                conversation_id,
                current_state=target_state,
                target_state=target_state,
            )
        except Exception as handler_err:
            log.warning(
                f"POST_TRANSITION handler failed ({type(handler_err).__name__}: {handler_err}), rolling back state from {target_state} to {old_state}"
            )
            instance.current_state = old_state
            instance.context.data.update(old_context_meta)
            raise

        log.info(f"State transition executed: {old_state} -> {target_state}")

    # ----------------------------------------------------------
    # Pass 2: Response generation
    # ----------------------------------------------------------

    def _execute_response_generation_pass(
        self,
        instance: FSMInstance,
        user_message: str,
        extraction_response: DataExtractionResponse,
        transition_occurred: bool,
        previous_state: str | None,
        conversation_id: str,
    ) -> str:
        """Execute Pass 2: Response Generation based on final state."""
        log = logger.bind(conversation_id=conversation_id)
        log.debug("Executing response generation pass")

        current_state = self.get_state(instance, conversation_id)
        fsm_def = self.fsm_resolver(instance.fsm_id)

        system_prompt = self.response_generation_prompt_builder.build_response_prompt(
            instance=instance,
            state=current_state,
            fsm_definition=fsm_def,
            extracted_data=extraction_response.extracted_data,
            transition_occurred=transition_occurred,
            previous_state=previous_state,
            user_message=user_message,
        )

        request = ResponseGenerationRequest(
            system_prompt=system_prompt,
            user_message=user_message,
            extracted_data=extraction_response.extracted_data,
            context=instance.context.get_user_visible_data(),
            transition_occurred=transition_occurred,
            previous_state=previous_state,
        )

        response = self.llm_interface.generate_response(request)
        instance.last_response_generation = response
        instance.context.conversation.add_system_message(response.message)

        log.debug("Response generation pass completed")
        return response.message

    # ----------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------

    @staticmethod
    def _clean_empty_context_keys(
        data: dict[str, Any], conversation_id: str, remove_none_values: bool = True
    ) -> dict[str, Any]:
        """Clean invalid keys from context data. Delegates to context module."""
        from .context import clean_context_keys

        return clean_context_keys(data, conversation_id, remove_none_values)
