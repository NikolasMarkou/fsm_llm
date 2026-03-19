"""
LLM Interface Module for FSM-Driven Conversational AI.

This module provides the core interface and implementation for Large Language Model (LLM)
communication within the fsm-llm library's improved 2-pass architecture. It defines how
FSM-driven applications interact with various LLM providers while maintaining clear
separation of concerns between data extraction, response generation, and transition decisions.

Architecture Overview
---------------------
The module implements a 2-pass architecture that separates LLM operations into distinct phases:

1. **Data Extraction Pass**: Extract and understand information from user input without
   generating any user-facing content. This prevents premature response generation and
   ensures all necessary data is captured before state transitions.

2. **Response Generation Pass**: Generate appropriate user-facing messages based on the
   final state context after all data extraction and transition evaluation is complete.

3. **Transition Decision Pass**: Provide LLM assistance for transition selection only
   when deterministic evaluation results in ambiguity between multiple valid options.

Key Components
--------------
LLMInterface : abc.ABC
    Abstract base class defining the contract for LLM communication with support for
    the 2-pass architecture. All LLM implementations must inherit from this interface.

LiteLLMInterface : LLMInterface
    Concrete implementation using LiteLLM for multi-provider support. Handles OpenAI,
    Anthropic, and other popular LLM providers through a unified interface.

Core Methods
------------
extract_data(request: DataExtractionRequest) -> DataExtractionResponse
    Pure data extraction from user input focused on understanding and information
    gathering without generating user-facing responses.

generate_response(request: ResponseGenerationRequest) -> ResponseGenerationResponse
    Generate user-facing messages based on final state context and extracted data.

decide_transition(request: TransitionDecisionRequest) -> TransitionDecisionResponse
    Make transition decisions when deterministic evaluation is insufficient.

Integration with FSM System
---------------------------
This module integrates with the broader fsm-llm system:

- **FSMManager** (`fsm.py`) orchestrates the overall FSM execution and calls these
  methods at appropriate points in the conversation flow.
- **PromptBuilder** (`prompts.py`) constructs the specialized prompts for each pass
  based on current FSM state, context, and conversation history.
- **API** (`api.py`) provides the high-level interface that developers use, internally
  coordinating between FSM management and LLM communication.
"""

import abc
import time
import json
from typing import Optional
from litellm import completion, get_supported_openai_params

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------

from .logging import logger
from .definitions import (
    DataExtractionRequest,
    DataExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
    TransitionDecisionRequest,
    TransitionDecisionResponse,
    LLMResponseError
)


# --------------------------------------------------------------
# Abstract Interface
# --------------------------------------------------------------

class LLMInterface(abc.ABC):
    """
    Abstract interface for LLM communication supporting improved 2-pass architecture.

    This interface defines methods for data extraction, response generation, and
    transition decision making, allowing implementations to optimize for different use cases.
    """

    @abc.abstractmethod
    def extract_data(self, request: DataExtractionRequest) -> DataExtractionResponse:
        """
        Extract data from user input without generating user-facing content.

        This method handles data extraction and understanding without generating
        any user-facing responses, preventing premature response generation.

        Args:
            request: Data extraction request with extraction-focused prompt

        Returns:
            Data extraction response with extracted information and confidence

        Raises:
            LLMResponseError: If data extraction fails
        """
        pass

    @abc.abstractmethod
    def generate_response(self, request: ResponseGenerationRequest) -> ResponseGenerationResponse:
        """
        Generate user-facing response based on final state context.

        This method generates the actual message shown to users after all data
        extraction and transition evaluation are complete.

        Args:
            request: Response generation request with final state context

        Returns:
            Response generation response with user-facing message

        Raises:
            LLMResponseError: If response generation fails
        """
        pass

    @abc.abstractmethod
    def decide_transition(self, request: TransitionDecisionRequest) -> TransitionDecisionResponse:
        """
        Decide between multiple valid transition options.

        This method is used only when deterministic evaluation results in
        ambiguity and LLM assistance is needed for transition selection.

        Args:
            request: Transition decision request with available options

        Returns:
            Transition decision response with selected target state

        Raises:
            LLMResponseError: If transition decision fails
        """
        pass


# --------------------------------------------------------------
# LiteLLM Implementation
# --------------------------------------------------------------

class LiteLLMInterface(LLMInterface):
    """
    LiteLLM-based implementation supporting multiple providers.

    This implementation uses LiteLLM to communicate with various LLM providers
    while maintaining the improved 2-pass architecture interface.
    """

    def __init__(
            self,
            model: str,
            api_key: Optional[str] = None,
            temperature: float = 0.5,
            max_tokens: int = 1000,
            **kwargs
    ):
        """
        Initialize LiteLLM interface with configuration.

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-opus")
            api_key: Optional API key (uses environment if not provided)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens for responses
            **kwargs: Additional LiteLLM parameters
        """
        if not model or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be a positive integer, got {max_tokens}")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

        # Configure API keys based on model type
        self._configure_api_keys(api_key)

        logger.info(f"Initialized LiteLLM interface with model: {model}")

    def _configure_api_keys(self, api_key: Optional[str]) -> None:
        """Configure API keys via kwargs (avoids global os.environ mutation)."""
        if not api_key:
            logger.debug("No API key provided, using environment variables")
            return

        self.kwargs["api_key"] = api_key
        logger.debug("Set API key in kwargs")

    def extract_data(self, request: DataExtractionRequest) -> DataExtractionResponse:
        """
        Extract data using LLM focused on understanding user input.

        This method creates prompts that focus purely on extracting and understanding
        information from user input without generating any user-facing responses.
        """
        try:
            start_time = time.time()

            logger.debug(f"Extracting data with {self.model}")
            logger.debug(f"User message preview: {request.user_message[:100]}...")

            # Prepare messages for data extraction
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message}
            ]

            # Get LLM response
            response = self._make_llm_call(messages, "data_extraction")
            response_time = time.time() - start_time

            logger.debug(f"Data extraction completed in {response_time:.2f}s")

            # Parse response for data extraction
            return self._parse_extraction_response(response)

        except Exception as e:
            error_msg = f"Data extraction failed: {str(e)}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e

    def generate_response(self, request: ResponseGenerationRequest) -> ResponseGenerationResponse:
        """
        Generate response using LLM focused on final state context.

        This method creates prompts that generate appropriate user-facing responses
        based on the final state context and all extracted information.
        """
        try:
            start_time = time.time()

            logger.debug(f"Generating response with {self.model}")
            logger.debug(f"Final state context: current state, transition: {request.transition_occurred}")

            # Prepare messages for response generation
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message}
            ]

            # Get LLM response
            response = self._make_llm_call(messages, "response_generation")
            response_time = time.time() - start_time

            logger.debug(f"Response generation completed in {response_time:.2f}s")

            # Parse response for response generation
            return self._parse_response_generation_response(response)

        except Exception as e:
            error_msg = f"Response generation failed: {str(e)}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e

    def decide_transition(self, request: TransitionDecisionRequest) -> TransitionDecisionResponse:
        """
        Decide transition using LLM for ambiguous cases.

        This method provides focused transition decision making without
        exposing unnecessary FSM details.
        """
        try:
            start_time = time.time()

            logger.debug(f"Deciding transition with {self.model}")
            logger.debug(f"Evaluating {len(request.available_transitions)} transition options")

            # Prepare messages for transition decision
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message}
            ]

            # Get LLM response
            response = self._make_llm_call(messages, "transition_decision")
            response_time = time.time() - start_time

            logger.debug(f"Transition decision completed in {response_time:.2f}s")

            # Parse response for transition decision
            return self._parse_transition_response(response, request.available_transitions)

        except Exception as e:
            error_msg = f"Transition decision failed: {str(e)}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e

    def _make_llm_call(self, messages: list, call_type: str) -> dict:
        """
        Make LLM API call with appropriate configuration.

        Args:
            messages: Message list for LLM
            call_type: Type of call for optimization

        Returns:
            Raw LLM response
        """
        # Check for structured output support
        supported_params = get_supported_openai_params(model=self.model)

        # Configure parameters based on call type
        # kwargs go first so explicit params cannot be overridden
        reserved_keys = {"model", "messages", "temperature", "max_tokens"}
        safe_kwargs = {k: v for k, v in self.kwargs.items() if k not in reserved_keys}
        call_params = {
            **safe_kwargs,
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Add structured output if supported and beneficial
        if supported_params and "response_format" in supported_params and call_type in ["data_extraction", "response_generation", "transition_decision"]:
            call_params["response_format"] = {"type": "json_object"}

        # Make the API call
        response = completion(**call_params)

        # Validate response structure
        if not response or not hasattr(response, 'choices') or not response.choices:
            raise LLMResponseError("Invalid response structure from LLM")

        choice = response.choices[0]
        if not hasattr(choice, 'message') or not hasattr(choice.message, 'content'):
            raise LLMResponseError("Response missing message content")

        if choice.message.content is None:
            raise LLMResponseError("LLM returned null content")

        return response

    def _parse_extraction_response(self, response) -> DataExtractionResponse:
        """
        Parse LLM response for data extraction.

        Handles both structured JSON and unstructured text responses.
        """
        content = response.choices[0].message.content

        # Handle structured response (JSON)
        if isinstance(content, dict) or self._looks_like_json(content):
            try:
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content

                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object, got array or primitive")

                return DataExtractionResponse(
                    extracted_data=data.get("extracted_data", {}),
                    confidence=data.get("confidence", 1.0),
                    reasoning=data.get("reasoning"),
                    additional_info_needed=data.get("additional_info_needed")
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse structured extraction response: {e}")

        # Handle unstructured response (plain text)
        # In this case, we assume no structured data was extracted
        logger.warning("Received unstructured response for data extraction")
        return DataExtractionResponse(
            extracted_data={},
            confidence=0.5,
            reasoning="Unstructured response - no data extraction performed"
        )

    def _parse_response_generation_response(self, response) -> ResponseGenerationResponse:
        """
        Parse LLM response for response generation.

        Handles both structured JSON and unstructured text responses.
        """
        content = response.choices[0].message.content

        # Handle structured response (JSON)
        if isinstance(content, dict) or self._looks_like_json(content):
            try:
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content

                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object, got array or primitive")

                message = data.get("message")
                if not message:
                    message = data.get("reasoning")
                if not message:
                    raise ValueError("No usable message or reasoning in response")
                return ResponseGenerationResponse(
                    message=message,
                    message_type=data.get("message_type", "response"),
                    reasoning=data.get("reasoning")
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse structured response generation response: {e}")

        # Normalize non-string content before using as message
        if not isinstance(content, str):
            content = json.dumps(content)

        # Handle unstructured response (plain text)
        # In this case, use the entire content as the message
        return ResponseGenerationResponse(
            message=content,
            message_type="response",
            reasoning="Unstructured response - used entire content as message"
        )

    def _parse_transition_response(
            self,
            response,
            available_transitions: list
    ) -> TransitionDecisionResponse:
        """
        Parse LLM response for transition decision.

        Validates that selected transition is available.
        """
        content = response.choices[0].message.content

        # Valid transition targets
        valid_targets = {t.target_state for t in available_transitions}

        # Handle structured response
        if isinstance(content, dict) or self._looks_like_json(content):
            try:
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content

                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object, got array or primitive")

                selected = data.get("selected_transition", "")

                if selected in valid_targets:
                    return TransitionDecisionResponse(
                        selected_transition=selected,
                        reasoning=data.get("reasoning")
                    )
                # Fall through to unstructured matching below

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse structured transition response: {e}")

        # Normalize non-string content before unstructured matching
        if not isinstance(content, str):
            content = json.dumps(content)

        # Handle unstructured response - try to extract state name
        content_lower = content.lower().strip()

        # Sort by length (longest first) so "collect_name_confirmation" matches before "collect_name"
        for target in sorted(valid_targets, key=len, reverse=True):
            if target.lower() in content_lower:
                logger.info(f"Extracted transition '{target}' from unstructured response")
                return TransitionDecisionResponse(
                    selected_transition=target,
                    reasoning="Extracted from unstructured response"
                )

        # If no valid transition found, raise error
        raise LLMResponseError(
            f"Could not extract valid transition from response. "
            f"Valid options: {sorted(valid_targets)}. "
            f"Response: {content[:200]}..."
        )

    def _looks_like_json(self, text: str) -> bool:
        """Check if text appears to be JSON format."""
        if not isinstance(text, str):
            return False

        text = text.strip()
        return (text.startswith('{') and text.endswith('}')) or \
            (text.startswith('[') and text.endswith(']'))


# --------------------------------------------------------------
# Response Processing Utilities
# --------------------------------------------------------------
