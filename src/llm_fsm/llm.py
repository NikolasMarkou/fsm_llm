import os
import time
import json
import litellm
from typing import Optional
from litellm import completion, get_supported_openai_params

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .logging import logger
from .definitions import (
    LLMRequest,
    LLMResponse,
    LLMResponseSchema,
    LLMResponseError,
    StateTransition,
    FSMInstance,
    State
)

# --------------------------------------------------------------

class LLMInterface:
    """
    Interface for communicating with LLMs.
    """

    def send_request(self, request: LLMRequest) -> LLMResponse:
        """
        Send a request to the LLM and get the response.

        Args:
            request: The LLM request

        Returns:
            The LLM's response

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement send_request")


class LiteLLMInterface(LLMInterface):
    """
    Implementation of LLMInterface using LiteLLM.

    This class uses LiteLLM to send requests to various LLM providers
    while maintaining a consistent interface.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        enable_json_validation: bool = True,
        **kwargs
    ):
        """
        Initialize the LiteLLM interface.

        Args:
            model: The model to use (e.g., "gpt-4", "claude-3-opus")
            api_key: Optional API key (will use environment variables if not provided)
            enable_json_validation: Whether to enable JSON schema validation
            **kwargs: Additional arguments to pass to LiteLLM
        """
        self.model = model
        self.kwargs = kwargs

        logger.info(f"Initializing LiteLLMInterface with model: {model}")

        # Extract provider from model name for API key setting
        if api_key:
            # Simple provider detection, can be expanded
            if "gpt" in model.lower() or "openai" in model.lower():
                os.environ["OPENAI_API_KEY"] = api_key
                logger.debug("Setting OPENAI_API_KEY environment variable")
            elif "claude" in model.lower() or "anthropic" in model.lower():
                os.environ["ANTHROPIC_API_KEY"] = api_key
                logger.debug("Setting ANTHROPIC_API_KEY environment variable")
            else:
                # For other providers, we'll need to determine the right env var
                # or pass it directly to LiteLLM
                self.kwargs["api_key"] = api_key
                logger.debug("Using API key directly in LiteLLM kwargs")
        else:
            logger.debug("No API key provided, assuming it's set in environment variables")

        # Enable JSON schema validation if needed
        if enable_json_validation:
            litellm.enable_json_schema_validation = True
            logger.debug("Enabled JSON schema validation in LiteLLM")

    def send_request(self, request: LLMRequest) -> LLMResponse:
        """
        Send a request to the LLM using LiteLLM and get the response.

        Args:
            request: The LLM request

        Returns:
            The LLM's response

        Raises:
            LLMResponseError: If there's an error processing the LLM response
        """
        try:
            start_time = time.time()

            # Log the request (truncated for brevity)
            logger.info(f"Sending request to {self.model}")
            logger.debug(f"User message: {request.user_message[:50]}{'...' if len(request.user_message) > 50 else ''}")

            # Prepare messages for LiteLLM
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message}
            ]

            # Check if the model supports structured output (response_format or json_schema)
            supported_params = get_supported_openai_params(model=self.model)
            logger.debug(f"Supported parameters for {self.model}: {', '.join(supported_params)}")

            # Decide on the response format approach
            if "response_format" in supported_params:
                # The model supports the OpenAI-style response_format
                logger.debug(f"Using response_format for {self.model}")
                response = completion(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    **self.kwargs
                )
            else:
                # For other models, try to use json_schema if possible
                # or fall back to parsing from unstructured output
                try:
                    if litellm.supports_response_schema(model=self.model):
                        logger.debug(f"Using json_schema for {self.model}")
                        response = completion(
                            model=self.model,
                            messages=messages,
                            response_format=LLMResponseSchema,
                            **self.kwargs
                        )
                    else:
                        # Fall back to unstructured output
                        # We'll add instruction to return JSON in the system prompt
                        logger.debug(f"Using enhanced prompt with JSON instructions for {self.model}")
                        enhanced_prompt = (
                            f"{request.system_prompt}\n\n"
                            "IMPORTANT: You must respond with a valid JSON object that follows this schema:\n"
                            "{\n"
                            '  "transition": {\n'
                            '    "target_state": "state_id",\n'
                            '    "context_update": {"key1": "value1", "key2": "value2"}\n'
                            '  },\n'
                            '  "message": "Your message to the user",\n'
                            '  "reasoning": "Your reasoning for this decision"\n'
                            "}\n"
                        )

                        # Update system message with enhanced prompt
                        messages[0] = {"role": "system", "content": enhanced_prompt}

                        response = completion(
                            model=self.model,
                            messages=messages,
                            **self.kwargs
                        )
                except Exception as schema_error:
                    # If schema approach fails, fall back to unstructured output
                    # with manual JSON instructions
                    logger.warning(f"JSON schema approach failed: {str(schema_error)}")
                    logger.debug("Falling back to basic approach with JSON instructions")

                    enhanced_prompt = (
                        f"{request.system_prompt}\n\n"
                        "IMPORTANT: You must respond with a valid JSON object that follows this schema:\n"
                        "{\n"
                        '  "transition": {\n'
                        '    "target_state": "state_id",\n'
                        '    "context_update": {"key1": "value1", "key2": "value2"}\n'
                        '  },\n'
                        '  "message": "Your message to the user",\n'
                        '  "reasoning": "Your reasoning for this decision"\n'
                        "}\n"
                    )

                    # Update system message with enhanced prompt
                    messages[0] = {"role": "system", "content": enhanced_prompt}

                    response = completion(
                        model=self.model,
                        messages=messages,
                        **self.kwargs
                    )

            # Calculate response time
            response_time = time.time() - start_time
            logger.info(f"Received response from {self.model} in {response_time:.2f}s")

            # Extract the response content
            content = response.choices[0].message.content

            # Handle different response types
            if hasattr(content, "model_dump"):
                # This is already a Pydantic model (likely from json_schema)
                logger.debug("Response is a Pydantic model")
                response_data = content.model_dump()
            else:
                # This is a string, try to parse as JSON
                try:
                    logger.debug("Parsing response as JSON")
                    response_data = json.loads(content)
                except json.JSONDecodeError:
                    # If not valid JSON, try to extract JSON from the text
                    logger.warning("Response is not valid JSON, attempting to extract JSON from text")
                    from .utilities import extract_json_from_text
                    extracted_json = extract_json_from_text(content)
                    if not extracted_json:
                        error_msg = f"Could not parse JSON from LLM response: {content[:100]}..."
                        logger.error(error_msg)
                        raise LLMResponseError(error_msg)
                    response_data = extracted_json
                    logger.debug("Successfully extracted JSON from text")

            # Create a StateTransition from the response
            transition_data = response_data.get("transition", {})
            transition = StateTransition(
                target_state=transition_data.get("target_state", ""),
                context_update=transition_data.get("context_update", {})
            )

            # Log the transition
            if transition.context_update:
                logger.debug(f"Context updates: {json.dumps(transition.context_update)}")

            # Create and return the LLMResponse
            return LLMResponse(
                transition=transition,
                message=response_data.get("message", ""),
                reasoning=response_data.get("reasoning", None)
            )

        except Exception as e:
            # Handle exceptions
            error_msg = f"Error processing LLM response: {str(e)}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg)
