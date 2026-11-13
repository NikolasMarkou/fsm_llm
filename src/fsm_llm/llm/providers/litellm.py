"""``LiteLLMInterface``: the provider implementation, and the ONE module of the package that
imports litellm.

Layer ``llm`` (plan d454fd62, step 73): split out of the old ``fsm_llm/llm/__init__.py`` (the old ``llm.py``); definitions unchanged.

This module owns the ``import litellm`` line with its ``LITELLM_MODE`` guard, the
``suppress_debug_info`` setting, and the ``completion`` and ``get_supported_openai_params``
bindings the interface looks up; ``embedding`` is bound here for ``providers/embedder.py``.
A test that replaces a provider call patches the module that looks it up.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from typing import Any

# DECISION plan-2026-11-12T073150-d454fd62/D-005: litellm loads a ``.env`` file
# when it is imported (``load_dotenv()``) unless ``LITELLM_MODE`` is
# ``PRODUCTION``; a stale ``.env`` then fills ``LLM_MODEL`` and keys behind the
# host's back. ``fsm_llm`` reads a ``.env`` only on request
# (``fsm_llm.load_env``, ``--env-file``). So the import below runs with the
# mode set to ``PRODUCTION`` when the caller has not set one, and the
# environment is restored right after. Do NOT move this handling out of the
# single ``import litellm`` site of this module, and do NOT leave
# ``LITELLM_MODE`` in ``os.environ`` afterwards (child processes such as eval
# examples would inherit it). An explicit ``LITELLM_MODE`` of the caller is
# respected (``DEV`` opts back in). If the host imports litellm before
# ``fsm_llm`` the file was already loaded and that cannot be undone.
_LITELLM_MODE_PREVIOUS = os.environ.get("LITELLM_MODE")
if _LITELLM_MODE_PREVIOUS is None:
    os.environ["LITELLM_MODE"] = "PRODUCTION"
try:
    import litellm

    # ``embedding`` is not looked up in this module: it is bound for
    # ``providers/embedder.py``, which takes it from here.
    from litellm import (  # noqa: F401
        completion,
        embedding,
        get_supported_openai_params,
    )
finally:
    if _LITELLM_MODE_PREVIOUS is None:
        os.environ.pop("LITELLM_MODE", None)
del _LITELLM_MODE_PREVIOUS

from fsm_llm.constants import (
    DEFAULT_TEMPERATURE,
    RESERVED_LLM_CALL_KWARGS,
    USAGE_KIND_BY_CALL_TYPE,
    USAGE_KIND_COMPLETE,
    USAGE_KIND_STREAM,
)
from fsm_llm.core.coerce import coerce_confidence
from fsm_llm.core.definitions import (
    FINISH_REASON_LENGTH,
    BulkExtractionRequest,
    CompletionRequest,
    CompletionResponse,
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    LLMResponseError,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm.interface import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_MAX_TOKENS,
    LLMInterface,
    _completion_response,
    is_malformed_tool_call_error,
)
from fsm_llm.llm.json_extraction import extract_json_from_text
from fsm_llm.llm.ollama import build_ollama_response_format, is_ollama_model
from fsm_llm.llm.parsing import (
    _GENERIC_FALLBACK_MESSAGE,
    ParseMixin,
    _finish_reason,
    _flag_cut_off,
    cut_off_note,
)
from fsm_llm.llm.text import resolve_reasoning_trace, strip_think_and_fences
from fsm_llm.llm.usage import _meter_of, _Metered
from fsm_llm.logging import logger

from .request import RequestMixin, _checked_model, _warn_reserved_kwargs

# DECISION plan-2026-09-22T080837-8b258a25/D-035: guards apology_retry_count.
# Do NOT move it into __init__: tests build LiteLLMInterface via __new__, and the
# D-020 branch would then raise AttributeError. Increments are rare, so one
# process-wide lock costs nothing.
_APOLOGY_COUNT_LOCK = threading.Lock()


# Top-level keys of the bulk-extraction envelope the pipeline asks for
# (`{"extracted_data": {...}, "confidence": ..., "reasoning": ...}`). Dropped
# from a FLAT reply's data in `extract_bulk_data` (D-009, audit D4).
_BULK_ENVELOPE_KEYS = frozenset({"confidence", "reasoning"})


# DECISION plan-2026-11-06T093029-0068b055/D-023: litellm prints a red "Give
# Feedback / Get Help" banner to STDOUT on every mapped provider exception while
# ``suppress_debug_info`` is False (its default). That is host-visible output from
# a library whose logging is meant to be off, and in `fsm-llm-eval` logs it hid
# the real error. Set it here, the one place litellm is configured, once at
# import. Do NOT set it per call or in another module (one LLM layer), and do NOT
# try to "respect" a False set earlier: False is litellm's default, so an
# explicit False cannot be told from no setting. A host that wants the banner
# sets ``litellm.suppress_debug_info = False`` AFTER importing fsm_llm.
litellm.suppress_debug_info = True


class LiteLLMInterface(RequestMixin, ParseMixin, LLMInterface, _Metered):
    """
    LiteLLM-based implementation supporting multiple providers.

    This implementation uses LiteLLM to communicate with various LLM providers
    while maintaining the 2-pass architecture interface.
    """

    # (model, result) of the last successful get_supported_openai_params call.
    _supported_params_memo: tuple[str, list[str] | None] | None = None
    # Times the D-020 apology retry fired on this instance (read-only for callers).
    apology_retry_count: int = 0

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float | None = DEFAULT_LLM_TIMEOUT_SECONDS,
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
        _checked_model(model)
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
        _warn_reserved_kwargs("LiteLLMInterface", kwargs, RESERVED_LLM_CALL_KWARGS)

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
        """
        try:
            start_time = time.time()

            logger.debug(f"Generating response with {self.model}")
            logger.debug(
                f"Final state context: current state, transition: {request.transition_occurred}"
            )

            # Prepare messages for response generation
            messages: list[dict[str, str | None]] = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message},
            ]

            # Get LLM response
            response = self._make_llm_call(
                messages,
                "response_generation",
                response_format=request.response_format,
                temperature=request.temperature,
            )
            response_time = time.time() - start_time

            logger.debug(f"Response generation completed in {response_time:.2f}s")

            # Parse response for response generation
            parsed = _flag_cut_off(
                self._parse_response_generation_response(
                    response, structured=request.response_format is not None
                ),
                response,
            )
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-020: the apology means
            # the model produced nothing usable (live: `{"message": ""}` on a
            # greeting, 1/18). Retry ONCE and return that result whatever it is:
            # no loop, and an error on the retry keeps the first apology instead
            # of failing the turn. Do NOT retry on any other outcome.
            if parsed.message == _GENERIC_FALLBACK_MESSAGE:
                logger.warning(
                    "Response generation yielded no usable text; retrying once"
                )
                with _APOLOGY_COUNT_LOCK:
                    self.apology_retry_count += 1
                try:
                    retried = self._make_llm_call(
                        messages,
                        "response_generation",
                        response_format=request.response_format,
                        temperature=request.temperature,
                    )
                    return _flag_cut_off(
                        self._parse_response_generation_response(
                            retried, structured=request.response_format is not None
                        ),
                        retried,
                    )
                except Exception as retry_error:
                    logger.warning(f"Response generation retry failed: {retry_error!s}")
            return parsed

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
        try:
            messages: list[dict[str, str | None]] = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message},
            ]

            # Build call params via the shared builder (DECISION
            # plan-2026-09-12T135914-45a654de/D-015 — this used to be a
            # second, independently-maintained ~40-line copy of
            # _make_llm_call's builder; see _build_call_params's own D-015
            # anchor for the full rationale and the D-007 max_retries
            # constraint it still preserves). `stream=True` adds
            # `"stream": True`; the forced-structured-output branch never
            # fires here because `call_type="response_generation"`.
            call_params = self._build_call_params(
                messages,
                "response_generation",
                response_format=request.response_format,
                stream=True,
                temperature=request.temperature,
            )

            response = self._send(call_params, call_type="response_generation")

            accumulated: list[str] = []
            reasoning_parts: list[str] = []
            for chunk in response:
                # DECISION plan-2026-09-20T114608-a8e47b88/D-022
                # `getattr(chunk, "choices", None)` replaces
                # `hasattr(chunk, "choices") and chunk.choices` -- a "not
                # falsy" check either way, so an ABSENT attribute (getattr's
                # None default) and a PRESENT-but-falsy one (`None`/`[]`)
                # both `continue` identically to before. Safe here because
                # nothing downstream distinguishes "attribute missing" from
                # "attribute present but falsy" -- see decisions.md D-022 for
                # the ONE flagged call site (llm.py content check) where that
                # distinction IS load-bearing and was deliberately left as
                # `hasattr()`.
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                # Accumulate reasoning the same way content is accumulated: a
                # provider may stream `reasoning_content` as one fragment per
                # chunk. Reuse the shared resolver so a Delta is read via the
                # same field precedence as a Message — never hand-roll the
                # `getattr(..., "thinking")` lookup here. See decisions.md D-002.
                frag = resolve_reasoning_trace(delta)
                if frag:
                    reasoning_parts.append(frag)
                content = getattr(delta, "content", None)
                if content is not None:
                    accumulated.append(content)
                    yield content

            # Same two tail guards _make_llm_call applies: a stream whose every
            # delta.content is "" is a FAILURE, not a silent empty reply. Recover
            # the answer from the reasoning trace ACCUMULATED across all chunks
            # (matching the non-streaming path, which sees the full assembled
            # message) — not just the final delta; otherwise a provider that
            # streams reasoning incrementally leaves only the last fragment
            # available and recovery fails where the non-streaming path succeeds.
            if not "".join(accumulated):
                full_trace = "".join(reasoning_parts)
                recovered = (
                    self._recover_content_from_trace(full_trace) if full_trace else None
                )
                if not recovered:
                    raise LLMResponseError("LLM returned empty content")
                yield recovered

        except LLMResponseError:
            raise
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

            messages: list[dict[str, str | None]] = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message},
            ]

            # Ollama only: a field-typed grammar so the model cannot wrap the
            # answer in an object (LV-01, decisions.md D-001). Other providers
            # keep the generic `json_object` format.
            typed_format = (
                build_ollama_response_format("field_extraction", request.field_type)
                if is_ollama_model(self.model)
                else None
            )
            response = self._make_llm_call(
                messages, "field_extraction", response_format=typed_format
            )
            response_time = time.time() - start_time

            logger.debug(
                f"Field extraction for '{request.field_name}' "
                f"completed in {response_time:.2f}s"
            )

            return _flag_cut_off(
                self._parse_field_extraction_response(response, request), response
            )

        except LLMResponseError:
            raise
        except Exception as e:
            # Broad catch is intentional: wraps any litellm/network/parsing
            # error into LLMResponseError at the system boundary.
            error_msg = f"Field extraction failed for '{request.field_name}': {e!s}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e

    def extract_bulk_data(
        self, request: BulkExtractionRequest
    ) -> DataExtractionResponse:
        """Extract free-form key/value data per a state's extraction_instructions.

        Interface contract (mirrors ``extract_field``'s error boundary):
            - Parses whatever the LLM returns via ``_make_llm_call``, reusing
              the same ``<think>``/fence-stripping and JSON-parsing logic as
              ``_parse_field_extraction_response``.
            - A response with no ``extracted_data``/dict-shaped payload
              (content is neither a ``str`` nor a ``dict``, or the parsed
              ``extracted_data``/top-level value isn't a ``dict``) resolves to
              an EMPTY ``DataExtractionResponse`` — this is "the model found
              nothing", not a failure, so it is NOT raised as
              ``LLMResponseError``.
            - Non-object JSON (``[1, 2]``, ``null``, ``42``) is the same
              "found nothing" case and resolves to an empty response.
            - When ``json.loads`` fails on the stripped text, the shared
              ``extract_json_from_text`` ladder is tried ONCE (prose around a
              JSON object, e.g. ``Sure! {"a": 1}``) before giving up.
            - Any actual failure (unparseable text, transport/parsing error)
              raises ``LLMResponseError``, wrapping the underlying cause.
            - Filtering of ``None``/empty-string/empty-dict values out of
              ``extracted_data`` is the CALLER's job (``engine/turn/conversation_extraction.py``'s merge
              logic), not this method's — this method returns the extraction
              verbatim, except that a FLAT reply (no ``extracted_data``
              wrapper) loses its top-level ``confidence``/``reasoning``
              envelope keys (the confidence is still read from it).
        """
        try:
            messages: list[dict[str, str | None]] = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message},
            ]

            response = self._make_llm_call(messages, "data_extraction")
            content = response.choices[0].message.content

            if isinstance(content, str):
                content = strip_think_and_fences(content)
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    # One recovery rung (D-010): prose around a JSON object.
                    # `extract_json_from_text` returns a dict or None, so a
                    # None means there is genuinely nothing parseable and the
                    # original decode error is raised below as before.
                    data = extract_json_from_text(content)
                    if data is None:
                        if _finish_reason(response) == FINISH_REASON_LENGTH:
                            raise LLMResponseError(
                                "Bulk data extraction failed: "
                                f"{cut_off_note(self.max_tokens)}; the cut text "
                                f"is not valid JSON ({e!s})"
                            ) from e
                        raise
            elif isinstance(content, dict):
                data = content
            else:
                return DataExtractionResponse(extracted_data={})

            if not isinstance(data, dict):
                # Valid JSON that is not an object (`[1,2]`, `null`, `42`):
                # "the model found nothing", per the contract above.
                return DataExtractionResponse(extracted_data={})

            # DECISION plan-2026-09-21T203800-8a03483a/D-009 (D4): a flat reply
            # (no `extracted_data` wrapper) carries the envelope's `confidence`
            # and `reasoning` at top level; they are metadata, not user data,
            # so they never enter context. A wrapped reply keeps whatever its
            # `extracted_data` holds. Do NOT return the flat object verbatim.
            if "extracted_data" in data:
                extracted = data["extracted_data"]
            else:
                extracted = {
                    k: v for k, v in data.items() if k not in _BULK_ENVELOPE_KEYS
                }
            if not isinstance(extracted, dict):
                return DataExtractionResponse(extracted_data={})

            try:
                confidence = coerce_confidence(data.get("confidence", 1.0), 1.0)
            except (TypeError, ValueError):
                # An uncoercible score (`"high"`, `null`, `{...}`) must not
                # discard the extracted data: fall back to the field default
                # (D-023). The signal is lost, the data is not.
                confidence = 1.0
            return DataExtractionResponse(
                extracted_data=extracted, confidence=confidence
            )

        except LLMResponseError:
            raise
        except Exception as e:
            # Broad catch is intentional: wraps any litellm/network/parsing
            # error into LLMResponseError at the system boundary, matching
            # extract_field's error boundary above.
            error_msg = f"Bulk data extraction failed: {e!s}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send one completion through the shared builder and read the reply.

        The request is built by ``_build_call_params`` like every other call:
        connection kwargs, timeout, the filled user turns, ``tools`` (never
        gated on litellm's supported-params list) or a ``response_format``
        (sent only when the model supports it, else a WARNING), per-call
        temperature and max_tokens, and Ollama preparation (thinking off,
        ``/nothink`` on the last user turn, the schema echoed into it; a
        ``response_format`` call runs at temperature 0). A ``seed`` is sent
        only when this interface was built with one.

        Returns:
            The normalised reply (see ``CompletionResponse``). A provider error
            that names a garbled tool call, on a request with tools, is a
            ``malformed`` reply rather than an error.

        Raises:
            LLMResponseError: the provider call failed, chained from its
                error, or the reply could not be read.
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-002: one request primitive
        # (supersedes 07ad3f8c/D-022's classifier-only method). Do NOT add a
        # second builder, a second `completion` binding or a per-kind method
        # (one for tools, one for text, one for schemas): every request
        # rule (tools XOR response_format, the Ollama gate, seed absent unless
        # set, user-turn filling) lives once in _build_call_params, and every
        # send goes through _send. A malformed tool turn is data
        # (kind="malformed", no calls), an outage raises. See decisions.md D-002.
        # Building the request is inside the boundary too (D-029): a failing
        # supported-params lookup is an LLMResponseError, never a raw error.
        try:
            call_params = self._build_call_params(
                request.messages,
                request.call_type,
                response_format=request.response_format,
                tools=request.tools,
                tool_choice=request.tool_choice,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except Exception as e:
            # Broad catch is intentional: the documented error of complete.
            error_msg = f"Completion request could not be built: {e!s}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e
        try:
            response = self._send(call_params, call_type=request.call_type)
        except Exception as e:
            # Broad catch is intentional: the provider boundary. A garbled
            # tool call is a model behaviour, every other failure an outage.
            if is_malformed_tool_call_error(e, tools_sent=request.tools is not None):
                logger.warning(f"Provider rejected a garbled tool call: {e!s}")
                return CompletionResponse(kind="malformed")
            error_msg = f"Completion call failed: {e!s}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e
        try:
            return _completion_response(response)
        except LLMResponseError:
            raise
        except Exception as e:
            # Broad catch is intentional: reading a reply is total (D-039);
            # a shape the normaliser did not foresee is a garbled reply.
            error_msg = f"Malformed LLM response shape: {e!s}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg) from e

    def _send(self, call_params: dict[str, Any], *, call_type: str) -> Any:
        """Send one built request to the provider: the one ``completion`` call.

        Every provider request of this interface (Pass 1, Pass 2, stream,
        ``complete``) goes through here; the raw provider reply (a stream
        iterator when ``call_params["stream"]``) is returned and errors
        propagate unchanged. Each request is counted once on this instance's
        usage meter, under the kind of ``call_type`` (``stream`` for a
        streamed call): a raising request as a call and an error, a streamed
        one as a call with usage missing (its chunks carry no usage on
        Ollama, and the reply is not read here).
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-004: the meter is per
        # instance and fed here only. Do NOT add a process-global counter
        # (shared state across conversations and tests), do NOT count in the
        # public methods (a second count site drifts from the one send path),
        # and do NOT read usage off a stream (its chunks carry none on Ollama).
        # Readers that want counts own (inject) the interface. See D-004.
        stream = bool(call_params.get("stream"))
        kind = (
            USAGE_KIND_STREAM
            if stream
            else USAGE_KIND_BY_CALL_TYPE.get(call_type, USAGE_KIND_COMPLETE)
        )
        meter = _meter_of(self)
        try:
            response = completion(**call_params)
        except Exception:
            # Broad catch is intentional: count the failed request, re-raise
            # it unchanged for the caller's own error boundary.
            meter.record_error(kind)
            raise
        meter.record(kind, None if stream else response)
        # DECISION plan-2026-11-06T093029-0068b055/D-024: finish_reason
        # "length" is read HERE, once per non-streamed reply (a streamed one
        # is not read, as for usage). A cut reply is warned about but never
        # raised on: it may be prose that is fine as far as it goes. Do NOT
        # add a per-method check beside this one (a second site drifts); the
        # parse paths read the same reason through `_finish_reason` to name
        # the cause in their own errors. See decisions.md D-024.
        if not stream and _finish_reason(response) == FINISH_REASON_LENGTH:
            logger.warning(
                f"Reply from '{call_params.get('model', self.model)}' "
                f"({call_type}) was cut off at "
                f"max_tokens={call_params.get('max_tokens')}; raise max_tokens "
                "(or LLM_MAX_TOKENS) if the full reply is needed"
            )
        return response

    def _supported_openai_params(self) -> list[str] | None:
        """Return litellm's supported-param list for ``self.model``, memoised.

        Only a successful result is memoised (a ``None`` "unknown model" answer
        included); an exception propagates and the next call retries. The memo
        is keyed by the model string, so reassigning ``self.model`` refreshes it.
        """
        memo = self._supported_params_memo
        if memo is not None and memo[0] == self.model:
            return memo[1]
        result = get_supported_openai_params(model=self.model)
        self._supported_params_memo = (self.model, result)
        return result

    def _make_llm_call(
        self,
        messages: list[dict[str, str | None]],
        call_type: str,
        response_format: dict[str, Any] | None = None,
        *,
        temperature: float | None = None,
    ) -> Any:
        """
        Make LLM API call with appropriate configuration.

        Args:
            messages: Message list for LLM
            call_type: Type of call for optimization
            response_format: Optional response format override for constrained
                decoding (e.g., JSON schema enforcement).
            temperature: per-call temperature (a Pass-2 request's
                ``temperature``); ``None`` keeps the interface's.

        Returns:
            Raw LLM response
        """
        call_params = self._build_call_params(
            messages,
            call_type,
            response_format=response_format,
            temperature=temperature,
        )

        # Make the API call
        response = self._send(call_params, call_type=call_type)

        # Validate response structure. D-022: `getattr(response, "choices", None)`
        # replaces `hasattr(response, "choices") and response.choices` -- see the
        # streaming loop above for why this "not falsy" shape is safe to convert.
        choices = getattr(response, "choices", None) if response else None
        if not choices:
            raise LLMResponseError("Invalid response structure from LLM")

        choice = choices[0]
        # D-022: `getattr(choice, "message", None) is not None` replaces
        # `hasattr(choice, "message")` -- safe, since a `message` attribute
        # present-but-`None` falls through to the SAME raise via the `content`
        # check below either way (`hasattr(None, "content")` is `False`).
        #
        # The SECOND check is deliberately LEFT as `hasattr()`, not converted.
        # DECISION plan-2026-09-20T114608-a8e47b88/D-022
        # `hasattr(choice.message, "content")` checks only that the `content`
        # ATTRIBUTE EXISTS, regardless of its value -- and a real litellm
        # `Message` legitimately carries `content=None` for an Ollama
        # reasoning-only reply (H3, `test_make_llm_call_recovers_from_none_content`).
        # `getattr(choice.message, "content", None) is not None` CANNOT
        # replicate this: it cannot distinguish "attribute absent" from
        # "attribute present, value None" (both collapse to the same `None`),
        # so it would raise "Response missing message content" for that
        # legitimate None-content case, before `_extract_content_from_thinking`
        # ever runs -- a real regression, not a style nit. Confirmed empirically
        # (not just reasoned): temporarily forcing this conversion during EXECUTE
        # made `test_make_llm_call_recovers_from_none_content` fail. This is the
        # `hasattr()` call D-003 already reserved: "falls back to a documented
        # exception... only if that test cannot be made to pass." See decisions.md
        # D-022 (this call site) and D-003 (the original judgment call).
        message = getattr(choice, "message", None)
        if message is None or not hasattr(message, "content"):
            raise LLMResponseError("Response missing message content")

        content = message.content
        if not content:
            # H3: `content` is falsy — empty string OR None. Ollama reasoning-
            # only replies arrive as content=None (not ""), so this must fire
            # on None too, else C2 recovery never runs in production.
            content = self._extract_content_from_thinking(choice.message)
            if content is not None:
                choice.message.content = content

        if not content:
            raise LLMResponseError("LLM returned empty content")

        return response

    @staticmethod
    def _extract_content_from_thinking(message) -> str | None:
        """Extract structured content from a model's reasoning/thinking trace.

        Some models (e.g. Qwen 3.5 via Ollama) place the actual answer in a
        reasoning field and leave ``content`` empty or ``None``. This helper
        tries to recover the last JSON object from that trace, falling back to
        the last non-empty line.

        Returns the extracted content string, or ``None`` if no reasoning
        trace is present.
        """
        # DECISION plan-2026-07-21T045419-9925aa3a/D-002
        # The reasoning-trace field-name resolution (reasoning_content ->
        # legacy thinking -> thinking_blocks) now lives in the SINGLE shared
        # `resolve_reasoning_trace` (utilities.py) so this reader and
        # classification.py::_extract_response can never re-diverge. Do NOT
        # revert to a `hasattr(message, "thinking")`/`.thinking`-only read:
        # installed litellm RENAMES `thinking` to `reasoning_content` and
        # DELETES `thinking` before building the Message/Delta object, making a
        # `.thinking`-only read dead code for the project's own
        # former default model. Rationale + anti-simplification note live at the
        # resolver's D-002 anchor. See decisions.md D-001/D-002.
        trace = resolve_reasoning_trace(message)
        if not trace:
            return None
        logger.debug("Content empty but reasoning trace present, extracting from it")
        return LiteLLMInterface._recover_content_from_trace(trace)

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
