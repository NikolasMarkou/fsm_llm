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
    prepare_ollama_messages,
)
from .utilities import extract_json_from_text

# Mirrors ``ResponseGenerationResponse.message``'s ``max_length`` constraint
# (definitions.py:154).  Duplicated deliberately rather than introspected out of
# the pydantic model: the terminal raw-text rung of _parse_response_generation_
# response must be able to truncate WITHOUT importing model internals.  The two
# literals are kept in lockstep by a machine-checked cap-drift assertion in
# tests/test_fsm_llm/test_llm_parse_fallback_seam.py — if you change one and not
# the other, that test fails.
_RESPONSE_MESSAGE_MAX_LEN = 5000

# Shown to the END USER when no human-readable text can be recovered from a
# model response.  Referenced from two places in
# _parse_response_generation_response (the non-text-content branch and the
# terminal rung's envelope guard) — one string, one place to change it.
_GENERIC_FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't generate a proper response. Please try again."
)


def _safe_str(value: Any) -> str | None:
    """Coerce a MODEL-SUPPLIED value into a value the capped models will accept.

    Args:
        value: any value read off a parsed LLM JSON payload.

    Returns:
        ``value`` truncated to ``_RESPONSE_MESSAGE_MAX_LEN`` when it is a ``str``;
        ``None`` for every other type.

    Failure mode: none — total function, never raises.  It exists so that an
    optional ``str | None`` field carrying ``max_length=5000``
    (``ResponseGenerationResponse.reasoning``, ``FieldExtractionResponse.reasoning``)
    or an unannotated-but-typed field (``message_type``) cannot fail model
    construction just because the model padded it.  Callers that need a
    REQUIRED ``str`` field slice directly instead, so mypy keeps the non-optional
    type.
    """
    return value[:_RESPONSE_MESSAGE_MAX_LEN] if isinstance(value, str) else None


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
        retries: int = 0,
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
            retries: Number of ADDITIONAL attempts the provider SDK makes after a
                failed call, i.e. `retries=N` yields at most N+1 provider requests.
                Only TRANSIENT failures are retried (connection errors, timeouts,
                429, 5xx). Deterministic client errors are NOT retried: a 400 bad
                request or a 401 auth failure costs exactly one request and fails
                immediately, so a malformed prompt or a wrong API key does not
                multiply in cost.

                This parameter REPLACES the provider SDK's own retry count rather
                than adding to it. Retry is NOT off by default: with `retries=0`
                the parameter is omitted entirely and the SDK's built-in default
                (2 additional attempts on transient errors, with exponential
                backoff) still applies — byte-for-byte the historical behavior.
                The practical consequence is that `retries=1` LOWERS resilience
                below the default; only values >= 3 increase it. Measured against
                a local server returning a persistent 429: omitted -> 3 requests,
                retries=1 -> 2, retries=2 -> 3, retries=4 -> 5.

                PROVIDER-DEPENDENT: this parameter is honored by providers routed
                through the OpenAI SDK. It is a NO-OP for providers that are not
                (measured: `ollama_chat/*` and `ollama/*` make exactly 1 request
                regardless of `retries`). Setting it against ollama neither helps
                nor hurts; it is silently ignored.

                Cost: retries multiply worst-case wall clock per turn on transient
                failures, roughly (N+1) x timeout plus the SDK's backoff waits.
                Backoff behavior is the SDK's own and is not configured here.

                Values <= 0 are treated as "leave the SDK default alone" (no
                validation ceremony).
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
        self.retries = retries
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
                message="",
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
            yield ""
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

            # DECISION plan-2026-07-19T075908-70b6bdec/D-007
            # This is the SECOND of two call-param builders; generate_response_stream
            # does NOT route through _make_llm_call. Do not "fix" retries at only one
            # site — an earlier finding flagged this builder as the un-propagated twin
            # that silently leaves streaming unretried. Keep both in sync.
            # Do NOT change this back to `num_retries`: that key routes to litellm's
            # tenacity layer, which stacks ON TOP of the SDK's own retry layer (2N+1
            # requests) and retries deterministic 4xx failures that can never succeed.
            # The key must be OMITTED (not set to 0) when retries <= 0 so the default
            # is byte-for-byte the historical call. See decisions.md D-007.
            if self.retries > 0:
                call_params["max_retries"] = self.retries

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

            # Ollama: prepend /nothink and embed schema in prompt
            call_params["messages"] = prepare_ollama_messages(
                call_params["messages"],
                self.model,
                call_params.get("response_format"),
            )

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

        # DECISION plan-2026-07-19T075908-70b6bdec/D-007
        # Retries are delegated to the provider SDK's own retry layer via
        # `max_retries`. Do NOT change this to `num_retries`: that key routes to
        # litellm's tenacity layer, which sits ON TOP of the SDK layer (giving
        # 2N+1 requests, not N+1) and retries EVERYTHING, including 400/401 —
        # deterministic failures that can never succeed. `max_retries` is one
        # layer with correct error classification. Do NOT hand-roll a retry loop
        # here either. Mirrored in generate_response_stream's separate builder.
        # See decisions.md D-007.
        if self.retries > 0:
            call_params["max_retries"] = self.retries

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

        # Ollama: prepend /nothink and embed schema in prompt
        call_params["messages"] = prepare_ollama_messages(
            call_params["messages"],
            self.model,
            call_params.get("response_format"),
        )

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
        is_structured = call_type in ("data_extraction", "field_extraction")
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
                    message=message[:_RESPONSE_MESSAGE_MAX_LEN],
                    message_type=_safe_str(data.get("message_type")) or "response",
                    reasoning=_safe_str(data.get("reasoning")),
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
                    try:
                        return ResponseGenerationResponse(
                            message=message[:_RESPONSE_MESSAGE_MAX_LEN],
                            message_type=_safe_str(data.get("message_type"))
                            or "response",
                            reasoning=_safe_str(data.get("reasoning")),
                        )
                    except (ValueError, TypeError) as e:
                        # Fall THROUGH to the terminal raw-text rung below — do not
                        # swallow into a success. A legitimately oversized message
                        # (>5000 chars) or a non-str `reasoning` must degrade, not
                        # fail the turn.
                        logger.warning(
                            f"Embedded-JSON response fallback failed validation: {e}"
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
            content = _GENERIC_FALLBACK_MESSAGE
        elif not isinstance(content, str):
            content = str(content)

        # Handle unstructured response (plain text)
        # In this case, use the entire content as the message
        #
        # DECISION plan-2026-07-18T051819-80b0bd4d/D-016: this is the TERMINAL rung of the
        # degradation ladder (structured -> embedded-JSON fallback -> raw text) and
        # it MUST remain construct-safe — there is nothing below it to fall through
        # to, so anything this construction raises escapes _parse_* and fails the
        # whole turn. It builds the SAME max_length=5000-capped model as the rungs
        # above, so guarding only those would have RELOCATED the crash here rather
        # than fixing it. The slice is what makes the guarantee hold; do not remove
        # it, and do not add an uncapped or non-literal field to this construction
        # without re-checking every constraint on the model.
        #
        # DECISION plan-2026-07-18T051819-80b0bd4d/D-020: the terminal rung is the LAST
        # LINE OF DEFENCE and must be BOTH construct-safe (above) AND
        # ENVELOPE-SAFE (below).  Construct-safety alone is not enough, and
        # shipping only half of it caused a real user-facing regression:
        # D-016 capped `message` but not `reasoning`, which carries the SAME
        # max_length=5000 (definitions.py:157).  So
        # {"message": "Your booking is confirmed.", "reasoning": "R"*9000} failed
        # construction one rung up and landed here, and `content` was still the
        # whole serialized payload — the END USER was shown
        # `{"message": "Your booking is confirmed.", "reasoning": "RRRR…`.
        # Pre-D-016 that raised a catchable exception; post-D-016 it was a
        # silently-wrong success, which is WORSE than the defect being fixed.
        # Never emit a serialized envelope as user-facing text: recover the
        # human-readable `message` out of it, or say something generic.
        #
        # DECISION plan-2026-07-18T162030-a02151fe/D-022
        # RESIDUAL LEAK, DELIBERATELY LEFT OPEN. `_looks_like_json` requires the
        # text to BOTH start and end with a brace/bracket pair, so three envelope
        # shapes still pass through verbatim: prose-PREFIXED (`Here you go:
        # {...}`), prose-SUFFIXED, and markdown-FENCED (```json ... ```).
        #
        # This is a deliberate acceptance, not an oversight, and it was measured
        # rather than argued. D-016 widened this guard to also fire on "any
        # parseable non-empty JSON object appears ANYWHERE in the text" and that
        # widening was REVERTED here, because it destroyed ordinary assistant
        # replies. All four of these reached the user verbatim before the
        # widening and were replaced by _GENERIC_FALLBACK_MESSAGE after it:
        #   'Sure! To create a user, POST to /api/users with a body like
        #    {"name": "Alice", "role": "admin"}.'
        #   'The server returned {"error": "not_found"} which means the record
        #    does not exist.'
        #   'In Python you would write d = {"key": "value"} and then access
        #    d["key"].'
        #   'Here is the JSON you asked me to write: ```json\n{"order_id": 123}```'
        #
        # Do NOT re-attempt this with a smarter regex or a longer negative-case
        # list. The reason is structural: the leak `{"note": "...", "status":
        # "ok"}` and the legitimate quoted content `{"error": "not_found"}` are
        # the SAME STRING SHAPE. No text-shape discriminator can separate "an
        # envelope the model emitted by mistake" from "prose that legitimately
        # contains JSON", so an over-firing guard silently eats correct replies —
        # strictly worse than showing an obviously-wrong one. Closing this needs a
        # DIFFERENT SIGNAL entirely: schema provenance, or a response-format flag
        # recording that structured output was requested for this call.
        # See decisions.md D-022.
        #
        # Reuses this class's own `_looks_like_json` and the module's
        # `extract_json_from_text`; core must not import the equivalent
        # `_is_extraction_envelope` from fsm_llm_agents/adapt.py.
        if self._looks_like_json(content):
            parsed = extract_json_from_text(content)
            recovered = parsed.get("message") if isinstance(parsed, dict) else None
            content = (_safe_str(recovered) or "").strip() or _GENERIC_FALLBACK_MESSAGE
        return ResponseGenerationResponse(
            message=content[:_RESPONSE_MESSAGE_MAX_LEN],
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

        # Strip markdown code fences that small models sometimes emit
        if isinstance(content, str):
            content = re.sub(r"^```(?:json)?\s*\n?", "", content, flags=re.MULTILINE)
            content = re.sub(r"\n?```\s*$", "", content).strip()

        if isinstance(content, dict) or self._looks_like_json(content):
            try:
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content

                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object")

                value = data.get("value", data.get(request.field_name))
                # Handle extracted_data wrapper: some models nest the value
                if value is None and "extracted_data" in data:
                    ed = data["extracted_data"]
                    if isinstance(ed, dict):
                        value = ed.get(request.field_name)
                confidence = float(data.get("confidence", 1.0))
                # D-020: `reasoning` carries max_length=5000 here too
                # (definitions.py:347) — same trapdoor class as
                # ResponseGenerationResponse.reasoning.
                reasoning = _safe_str(data.get("reasoning"))

                return FieldExtractionResponse(
                    field_name=request.field_name,
                    value=value,
                    confidence=min(max(confidence, 0.0), 1.0),
                    reasoning=reasoning,
                )
            # D-016: TypeError added by the step-4 sweep, which found this PRIMARY
            # rung escaping too — `float(data.get("confidence", 1.0))` raises
            # TypeError (not ValueError) on a model-supplied
            # `"confidence": {...}` / `null`, so the ladder was breached one rung
            # above the two reported fallback branches. Same defect class, so it
            # is fixed in the same step.
            except (json.JSONDecodeError, ValueError, TypeError) as e:
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
                try:
                    # `float(...)` is INSIDE the guard on purpose: a model-supplied
                    # `"confidence": {...}` raises TypeError, and a non-str
                    # `reasoning` (or one over max_length=5000) raises
                    # ValidationError — both would otherwise escape the ladder from
                    # this rung just as the unguarded construction did.
                    confidence = float(data.get("confidence", 0.95))
                    reasoning = _safe_str(data.get("reasoning"))  # D-020
                    if value is not None:
                        logger.debug("Extracted field JSON via fallback")
                        return FieldExtractionResponse(
                            field_name=request.field_name,
                            value=value,
                            confidence=min(max(confidence, 0.0), 1.0),
                            reasoning=reasoning,
                        )
                except (ValueError, TypeError) as e:
                    # Fall THROUGH to the unstructured-coercion and terminal rungs.
                    logger.warning(f"Field extraction fallback failed validation: {e}")

        # Unstructured fallback — try to coerce raw content to expected type
        if isinstance(content, str) and content.strip():
            raw = content.strip()
            coerced_value: Any = None
            if request.field_type == "str":
                # DECISION plan-2026-07-18T162030-a02151fe/D-022
                # This rung hands the raw text back as the field value even when
                # that text is a prose-wrapped envelope. D-016 guarded it with
                # `None if is_envelope(raw) else raw`; that guard is REVERTED,
                # for the same reason as the response-generation rung above and
                # with a WORSE blast radius here. `None` means the key never
                # lands in context, so a state listing it in
                # `required_context_keys` never satisfies and the conversation
                # silently loops re-asking. Measured: 'The API returned
                # {"error": "not_found"} when I tried to save.' became `None`.
                # A free-text `complaint` / `error_message` field is the
                # canonical victim. See decisions.md D-022.
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
