# /llm.py

"""
Enhanced LLM Interface supporting 2-pass architecture.

This module provides LLM communication interfaces that support:
1. Content generation requests (for user-facing responses)
2. Transition decision requests (for ambiguous state transitions)

Key Features:
- Separate request types to prevent information leakage
- Flexible response handling for different use cases
- Provider-agnostic implementation with LiteLLM
- Enhanced error handling and validation
"""

import os
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
    ContentGenerationRequest,
    ContentGenerationResponse,
    TransitionDecisionRequest,
    TransitionDecisionResponse,
    LLMResponseError,
    LLMRequestType
)


# --------------------------------------------------------------
# Abstract Interface
# --------------------------------------------------------------

class LLMInterface(abc.ABC):
    """
    Abstract interface for LLM communication supporting 2-pass architecture.

    This interface defines methods for both content generation and transition
    decision making, allowing implementations to optimize for different use cases.
    """

    @abc.abstractmethod
    def generate_content(self, request: ContentGenerationRequest) -> ContentGenerationResponse:
        """
        Generate user-facing content based on current state only.

        This method handles content generation without exposing FSM structure,
        preventing information leakage into conversations.

        Args:
            request: Content generation request with state-focused prompt

        Returns:
            Content response with message and extracted data

        Raises:
            LLMResponseError: If content generation fails
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
    while maintaining the 2-pass architecture interface.
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
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

        # Configure API keys based on model type
        self._configure_api_keys(api_key)

        logger.info(f"Initialized LiteLLM interface with model: {model}")

    def _configure_api_keys(self, api_key: Optional[str]) -> None:
        """Configure API keys based on model type."""
        if not api_key:
            logger.debug("No API key provided, using environment variables")
            return

        # Detect provider and set appropriate environment variable
        model_lower = self.model.lower()

        if "gpt" in model_lower or "openai" in model_lower:
            os.environ["OPENAI_API_KEY"] = api_key
            logger.debug("Set OpenAI API key")
        elif "claude" in model_lower or "anthropic" in model_lower:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            logger.debug("Set Anthropic API key")
        else:
            # For other providers, include in kwargs
            self.kwargs["api_key"] = api_key
            logger.debug("Set API key in kwargs for custom provider")

    def generate_content(self, request: ContentGenerationRequest) -> ContentGenerationResponse:
        """
        Generate content using LLM focused on current state only.

        This method creates prompts that avoid exposing FSM structure
        while generating appropriate user-facing responses.
        """
        try:
            start_time = time.time()

            logger.info(f"Generating content with {self.model}")
            logger.debug(f"User message preview: {request.user_message[:100]}...")

            # Prepare messages for content generation
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message}
            ]

            # Get LLM response
            response = self._make_llm_call(messages, "content_generation")
            response_time = time.time() - start_time

            logger.info(f"Content generation completed in {response_time:.2f}s")

            # Parse response for content generation
            return self._parse_content_response(response)

        except Exception as e:
            error_msg = f"Content generation failed: {str(e)}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg)

    def decide_transition(self, request: TransitionDecisionRequest) -> TransitionDecisionResponse:
        """
        Decide transition using LLM for ambiguous cases.

        This method provides focused transition decision making without
        exposing unnecessary FSM details.
        """
        try:
            start_time = time.time()

            logger.info(f"Deciding transition with {self.model}")
            logger.debug(f"Evaluating {len(request.available_transitions)} transition options")

            # Prepare messages for transition decision
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message}
            ]

            # Get LLM response
            response = self._make_llm_call(messages, "transition_decision")
            response_time = time.time() - start_time

            logger.info(f"Transition decision completed in {response_time:.2f}s")

            # Parse response for transition decision
            return self._parse_transition_response(response, request.available_transitions)

        except Exception as e:
            error_msg = f"Transition decision failed: {str(e)}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg)

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
        call_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.kwargs
        }

        # Add structured output if supported and beneficial
        if "response_format" in supported_params and call_type == "content_generation":
            call_params["response_format"] = {"type": "json_object"}

        # Make the API call
        response = completion(**call_params)

        # Validate response structure
        if not response or not hasattr(response, 'choices') or not response.choices:
            raise LLMResponseError("Invalid response structure from LLM")

        choice = response.choices[0]
        if not hasattr(choice, 'message') or not hasattr(choice.message, 'content'):
            raise LLMResponseError("Response missing message content")

        return response

    def _parse_content_response(self, response) -> ContentGenerationResponse:
        """
        Parse LLM response for content generation.

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

                return ContentGenerationResponse(
                    message=data.get("message", ""),
                    extracted_data=data.get("extracted_data", {}),
                    reasoning=data.get("reasoning")
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse structured content response: {e}")

        # Handle unstructured response (plain text)
        # In this case, we assume the entire response is the message
        return ContentGenerationResponse(
            message=content,
            extracted_data={},
            reasoning="Unstructured response - no data extraction performed"
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

                selected = data.get("selected_transition", "")

                if selected not in valid_targets:
                    raise LLMResponseError(
                        f"LLM selected invalid transition '{selected}'. "
                        f"Valid options: {sorted(valid_targets)}"
                    )

                return TransitionDecisionResponse(
                    selected_transition=selected,
                    reasoning=data.get("reasoning")
                )

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse structured transition response: {e}")

        # Handle unstructured response - try to extract state name
        content_lower = content.lower().strip()

        # Look for exact matches in the content
        for target in valid_targets:
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

def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Extract JSON object from text with various formats.

    Handles code blocks, partial JSON, and embedded JSON structures.
    """
    if not isinstance(text, str):
        return None

    logger.debug("Attempting to extract JSON from text")

    # Try direct JSON parsing first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from code blocks
    import re
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON objects in text
    brace_count = 0
    start_pos = None

    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_pos = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_pos is not None:
                try:
                    json_str = text[start_pos:i + 1]
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

    logger.warning("Could not extract valid JSON from text")
    return None