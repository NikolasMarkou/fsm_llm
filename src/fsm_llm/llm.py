from __future__ import annotations

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
generate_response(request: ResponseGenerationRequest) -> ResponseGenerationResponse
    Generate user-facing messages based on final state context and extracted data.

extract_field(request: FieldExtractionRequest) -> FieldExtractionResponse
    Extract a single specific field from user input with custom instructions.

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
import json
import re
import time
from collections.abc import Iterator
from typing import Any

from litellm import completion, get_supported_openai_params

from .definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    LLMResponseError,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------
from .logging import logger
from .ollama import (
    apply_ollama_params,
    build_ollama_response_format,
    is_ollama_model,
)
from .utilities import extract_json_from_text

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
    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
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

    def generate_response_stream(
        self, request: ResponseGenerationRequest
    ) -> Iterator[str]:
        """Stream response tokens for Pass 2 (response generation).

        Yields individual text chunks as the LLM produces them.
        The default implementation falls back to ``generate_response``
        and yields the full message as a single chunk.

        Args:
            request: Response generation request with final state context.

        Yields:
            String chunks of the response as they arrive.
        """
        response = self.generate_response(request)
        yield response.message

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        """
        Extract a single specific field from user input.

        This method performs targeted extraction of one named field with
        custom instructions, dynamic context, and validation.  Called by
        the engine after bulk ``extract_data`` completes.

        The default implementation raises ``NotImplementedError`` so
        existing subclasses that do not need field extraction remain
        compatible.

        Args:
            request: Field extraction request with focused instructions

        Returns:
            Field extraction response with typed value and confidence

        Raises:
            LLMResponseError: If field extraction fails
            NotImplementedError: If the subclass does not implement this
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement extract_field. "
            "Override this method to support targeted field extraction."
        )


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
        api_key: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1000,
        timeout: float | None = 120.0,
        **kwargs,
    ):
        """
        Initialize LiteLLM interface with configuration.

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-opus")
            api_key: Optional API key (uses environment if not provided)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens for responses
            timeout: Timeout in seconds for LLM API calls (None for no timeout)
            **kwargs: Additional LiteLLM parameters
        """
        if not model or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {temperature}"
            )
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be a positive integer, got {max_tokens}")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs

        # Configure API keys based on model type
        self._configure_api_keys(api_key)

        logger.info(f"Initialized LiteLLM interface with model: {model}")

    def _configure_api_keys(self, api_key: str | None) -> None:
        """Configure API keys via kwargs (avoids global os.environ mutation)."""
        if not api_key:
            logger.debug("No API key provided, using environment variables")
            return

        self.kwargs["api_key"] = api_key
        logger.debug("Set API key in kwargs")

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        """
        Generate response using LLM focused on final state context.

        This method creates prompts that generate appropriate user-facing responses
        based on the final state context and all extracted information.

        When the system_prompt is empty, returns a synthetic response without
        making an LLM call. This supports the fast-path for intermediate agent
        states that skip response generation.
        """
        # Fast-path: minimal system_prompt ("." sentinel) signals skipped
        # response generation for intermediate agent states
        if request.system_prompt == ".":
            return ResponseGenerationResponse(
                message=".",
                message_type="response",
                reasoning="skipped",
            )

        try:
            start_time = time.time()

            logger.debug(f"Generating response with {self.model}")
            logger.debug(
                f"Final state context: current state, transition: {request.transition_occurred}"
            )

            # Prepare messages for response generation
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message},
            ]

            # Get LLM response
            response = self._make_llm_call(
                messages,
                "response_generation",
                response_format=request.response_format,
            )
            response_time = time.time() - start_time

            logger.debug(f"Response generation completed in {response_time:.2f}s")

            # Parse response for response generation
            return self._parse_response_generation_response(response)

        except LLMResponseError:
            raise
        except Exception as e:
            # Broad catch is intentional: wraps any litellm/network/parsing
            # error into LLMResponseError at the system boundary.
            error_msg = f"Response generation failed: {e!s}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e

    def generate_response_stream(
        self, request: ResponseGenerationRequest
    ) -> Iterator[str]:
        """Stream response tokens for Pass 2 using LiteLLM streaming.

        Pass 1 (extraction) is never streamed — it must complete fully
        for transition evaluation.  This method only streams Pass 2
        (user-facing response generation).
        """
        # Fast-path: sentinel prompt — no streaming needed
        if request.system_prompt == ".":
            yield "."
            return

        try:
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message},
            ]

            # Build call params (same as _make_llm_call but with stream=True)
            supported_params = get_supported_openai_params(model=self.model)
            reserved_keys = {"model", "messages", "temperature", "max_tokens"}
            safe_kwargs = {
                k: v for k, v in self.kwargs.items() if k not in reserved_keys
            }
            call_params = {
                **safe_kwargs,
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
            }
            if self.timeout is not None:
                call_params["timeout"] = self.timeout

            # Apply response_format if provided (schema-enforced output)
            if (
                request.response_format is not None
                and supported_params
                and "response_format" in supported_params
            ):
                call_params["response_format"] = request.response_format
            elif request.response_format is not None:
                logger.warning(
                    f"response_format requested but not supported by "
                    f"model '{self.model}'; output may not match schema"
                )

            self._apply_model_specific_params(call_params, "response_generation")

            response = completion(**call_params)

            for chunk in response:
                if (
                    hasattr(chunk, "choices")
                    and chunk.choices
                    and hasattr(chunk.choices[0], "delta")
                    and hasattr(chunk.choices[0].delta, "content")
                    and chunk.choices[0].delta.content is not None
                ):
                    yield chunk.choices[0].delta.content

        except Exception as e:
            error_msg = f"Streaming response generation failed: {e!s}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        """Extract a single specific field from user input."""
        try:
            start_time = time.time()

            logger.debug(
                f"Extracting field '{request.field_name}' "
                f"(type={request.field_type}) with {self.model}"
            )

            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message},
            ]

            response = self._make_llm_call(messages, "field_extraction")
            response_time = time.time() - start_time

            logger.debug(
                f"Field extraction for '{request.field_name}' "
                f"completed in {response_time:.2f}s"
            )

            return self._parse_field_extraction_response(response, request)

        except LLMResponseError:
            raise
        except Exception as e:
            # Broad catch is intentional: wraps any litellm/network/parsing
            # error into LLMResponseError at the system boundary.
            error_msg = f"Field extraction failed for '{request.field_name}': {e!s}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e

    def _make_llm_call(
        self,
        messages: list[dict[str, str]],
        call_type: str,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        """
        Make LLM API call with appropriate configuration.

        Args:
            messages: Message list for LLM
            call_type: Type of call for optimization
            response_format: Optional response format override for constrained
                decoding (e.g., JSON schema enforcement).

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

        if self.timeout is not None:
            call_params["timeout"] = self.timeout

        # Add structured output if supported and beneficial.
        # Do NOT force structured output for response_generation — the
        # response is user-facing natural language, not structured data
        # UNLESS an explicit response_format was provided (e.g., for
        # schema-enforced agent output).
        if (
            supported_params
            and "response_format" in supported_params
            and call_type in ["data_extraction", "field_extraction"]
        ):
            if is_ollama_model(self.model):
                # Ollama: use json_schema with explicit schema for
                # grammar-constrained output.
                ollama_fmt = build_ollama_response_format(call_type)
                if ollama_fmt is not None:
                    call_params["response_format"] = ollama_fmt
            else:
                call_params["response_format"] = {"type": "json_object"}

        # Apply explicit response_format override (e.g., from output_schema).
        # This allows schema-enforced output for response_generation calls
        # when the caller provides a JSON schema.
        if (
            response_format is not None
            and supported_params
            and "response_format" in supported_params
        ):
            call_params["response_format"] = response_format
        elif response_format is not None:
            logger.warning(
                f"response_format requested but not supported by "
                f"model '{self.model}'; output may not match schema"
            )

        self._apply_model_specific_params(call_params, call_type)

        # Make the API call
        response = completion(**call_params)

        # Validate response structure
        if not response or not hasattr(response, "choices") or not response.choices:
            raise LLMResponseError("Invalid response structure from LLM")

        choice = response.choices[0]
        if not hasattr(choice, "message") or not hasattr(choice.message, "content"):
            raise LLMResponseError("Response missing message content")

        content = choice.message.content
        if content is not None and content == "":
            content = self._extract_content_from_thinking(choice.message)
            if content is not None:
                choice.message.content = content

        if not content:
            raise LLMResponseError("LLM returned empty content")

        return response

    def _apply_model_specific_params(self, call_params: dict, call_type: str) -> None:
        """Apply model-specific parameters to the LLM call.

        Handles quirks of specific model providers (e.g. Ollama's thinking
        mode) by mutating *call_params* in place.
        """
        # Ollama: disable thinking mode and force deterministic output
        # for structured calls (data extraction, transition decisions).
        is_structured = call_type == "data_extraction"
        apply_ollama_params(call_params, self.model, structured=is_structured)

    @staticmethod
    def _extract_content_from_thinking(message) -> str | None:
        """Extract structured content from a model's ``thinking`` field.

        Some models (e.g. Qwen 3.5 via Ollama) place the actual answer in
        a ``thinking`` attribute and leave ``content`` as an empty string.
        This helper tries to recover the last JSON object from the thinking
        trace, falling back to the last non-empty line.

        Returns the extracted content string, or ``None`` if no thinking
        field is present.
        """
        if not hasattr(message, "thinking") or not message.thinking:
            return None
        logger.debug(
            "Content empty but thinking field present, extracting from thinking"
        )
        thinking = message.thinking
        # Find JSON objects using proper parsing — supports multi-line JSON
        json_candidates: list[str] = []
        # First try line-by-line for single-line JSON
        for line in thinking.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    json.loads(line)
                    json_candidates.append(line)
                except json.JSONDecodeError:
                    pass
        # If no single-line JSON found, try extracting multi-line JSON blocks
        if not json_candidates:
            depth = 0
            start = -1
            for i, ch in enumerate(thinking):
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start >= 0:
                        candidate = thinking[start : i + 1]
                        try:
                            json.loads(candidate)
                            json_candidates.append(candidate)
                        except json.JSONDecodeError:
                            pass
                        start = -1
        if json_candidates:
            # Prefer the last JSON object (most likely the final answer)
            return json_candidates[-1]
        # Fallback: use the last substantial line
        lines = [line.strip() for line in thinking.strip().split("\n") if line.strip()]
        return lines[-1] if lines else None

    def _parse_response_generation_response(
        self, response
    ) -> ResponseGenerationResponse:
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
                if not isinstance(message, str) or not message.strip():
                    message = data.get("reasoning")
                if not isinstance(message, str) or not message.strip():
                    raise ValueError("No usable message or reasoning in response")
                return ResponseGenerationResponse(
                    message=message,
                    message_type=data.get("message_type", "response"),
                    reasoning=data.get("reasoning"),
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Failed to parse structured response generation response: {e}"
                )

        # Fallback: extract JSON embedded in text (handles <think> tags, etc.)
        if isinstance(content, str):
            data = extract_json_from_text(content)
            if isinstance(data, dict) and "message" in data:
                message = data["message"]
                if isinstance(message, str) and message.strip():
                    logger.debug("Extracted response JSON via fallback")
                    return ResponseGenerationResponse(
                        message=message,
                        message_type=data.get("message_type", "response"),
                        reasoning=data.get("reasoning"),
                    )

        # Extract message from dict content if possible
        if isinstance(content, dict) and "message" in content:
            content = str(content["message"])
        elif isinstance(content, dict) or isinstance(content, list):
            # Don't expose raw JSON structures as user-facing messages
            logger.error(
                f"Response generation returned non-text content "
                f"(type={type(content).__name__}): {str(content)[:200]}; "
                f"using generic fallback message"
            )
            content = (
                "I'm sorry, I couldn't generate a proper response. Please try again."
            )
        elif not isinstance(content, str):
            content = str(content)

        # Handle unstructured response (plain text)
        # In this case, use the entire content as the message
        return ResponseGenerationResponse(
            message=content,
            message_type="response",
            reasoning="Unstructured response - used entire content as message",
        )

    def _parse_field_extraction_response(
        self, response, request: FieldExtractionRequest
    ) -> FieldExtractionResponse:
        """Parse LLM response for single-field extraction."""
        content = response.choices[0].message.content

        # Strip <think>...</think> tags that some models (e.g. Qwen) emit
        if isinstance(content, str):
            content = re.sub(
                r"<think>.*?</think>", "", content, flags=re.DOTALL
            ).strip()

        if isinstance(content, dict) or self._looks_like_json(content):
            try:
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content

                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object")

                value = data.get("value", data.get(request.field_name))
                confidence = float(data.get("confidence", 1.0))
                reasoning = data.get("reasoning")

                return FieldExtractionResponse(
                    field_name=request.field_name,
                    value=value,
                    confidence=min(max(confidence, 0.0), 1.0),
                    reasoning=reasoning,
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Failed to parse field extraction response: {e}. "
                    f"Content preview: {str(content)[:200]}"
                )

        # Fallback: extract JSON embedded in text (mirrors response gen path)
        if isinstance(content, str):
            data = extract_json_from_text(content)
            if isinstance(data, dict):
                value = data.get("value", data.get(request.field_name))
                # Nested key search: look through nested dicts (depth ≤ 3)
                if value is None:
                    value = self._find_nested_key(data, request.field_name, max_depth=3)
                confidence = float(data.get("confidence", 0.95))
                reasoning = data.get("reasoning")
                if value is not None:
                    logger.debug("Extracted field JSON via fallback")
                    return FieldExtractionResponse(
                        field_name=request.field_name,
                        value=value,
                        confidence=min(max(confidence, 0.0), 1.0),
                        reasoning=reasoning,
                    )

        # Unstructured fallback — try to coerce raw content to expected type
        if isinstance(content, str) and content.strip():
            raw = content.strip()
            coerced_value: Any = None
            if request.field_type == "str":
                coerced_value = raw
            elif request.field_type == "int":
                try:
                    coerced_value = int(raw)
                except ValueError:
                    pass
            elif request.field_type == "float":
                try:
                    coerced_value = float(raw)
                except ValueError:
                    pass
            elif request.field_type == "bool":
                if raw.lower() in ("true", "yes", "1"):
                    coerced_value = True
                elif raw.lower() in ("false", "no", "0"):
                    coerced_value = False
            elif request.field_type in ("dict", "list"):
                # Strict: only accept JSON-parsed values of the correct type
                try:
                    parsed = json.loads(raw)
                    expected = dict if request.field_type == "dict" else list
                    if isinstance(parsed, expected):
                        coerced_value = parsed
                except (json.JSONDecodeError, ValueError):
                    pass
            elif request.field_type == "any":
                try:
                    coerced_value = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    # Try extracting a quoted value first
                    quoted = re.search(r'"([^"]+)"', raw)
                    if quoted:
                        coerced_value = quoted.group(1)
                    else:
                        coerced_value = raw
            if coerced_value is not None:
                return FieldExtractionResponse(
                    field_name=request.field_name,
                    value=coerced_value,
                    confidence=0.5,
                    reasoning="Unstructured response coerced to expected type",
                )

        return FieldExtractionResponse(
            field_name=request.field_name,
            value=None,
            confidence=0.0,
            reasoning="Failed to extract field from LLM response",
            is_valid=False,
            validation_error="Extraction produced no usable value",
        )

    @staticmethod
    def _find_nested_key(data: dict, key: str, max_depth: int = 3) -> Any:
        """Search nested dicts for a key, returning first match."""
        if max_depth <= 0:
            return None
        for v in data.values():
            if isinstance(v, dict):
                if key in v:
                    return v[key]
                found = LiteLLMInterface._find_nested_key(v, key, max_depth - 1)
                if found is not None:
                    return found
        return None

    def _looks_like_json(self, text: str) -> bool:
        """Check if text appears to be JSON format."""
        if not isinstance(text, str):
            return False

        text = text.strip()
        return (text.startswith("{") and text.endswith("}")) or (
            text.startswith("[") and text.endswith("]")
        )


# --------------------------------------------------------------
# Response Processing Utilities
# --------------------------------------------------------------
