# ruff: noqa: RUF002
from __future__ import annotations

"""
Oracle protocol + LiteLLM adapter for the λ-kernel.

The ``Oracle`` protocol is the ONLY boundary across which the executor
invokes 𝓜 (I1). It exposes three operations:

1. ``invoke(prompt, schema=None, *, model_override=None)`` — send the
   prompt, optionally constrain output to a Pydantic schema, return
   either a ``dict`` (structured) or a ``str`` (unstructured).
2. ``tokenize(text) -> int`` — return an integer token count. Used by
   SPLIT (rank-awareness) and by the ``|P| ≤ K`` guard.
3. ``context_window() -> int`` — advertise the model's K in tokens.

``LiteLLMOracle`` wraps an existing ``LiteLLMInterface`` without modifying
it (D-002). Structured-schema calls are routed through ``extract_field``;
unstructured calls through ``generate_response``.

The ``|P| > K`` guard is enforced here (E5) — never silently truncate.
"""

import importlib
import json
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from fsm_llm.dialog.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    LLMResponseError,
    ResponseGenerationRequest,
)
from fsm_llm.runtime._litellm import LLMInterface

from .constants import CHARS_PER_TOKEN_FALLBACK, DEFAULT_CONTEXT_WINDOW
from .errors import OracleError


@runtime_checkable
class Oracle(Protocol):
    """The 𝓜 interface. Only ``Leaf`` nodes hold a reference."""

    def invoke(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
        *,
        model_override: str | None = None,
        env: dict[str, Any] | None = None,
    ) -> dict[str, Any] | str:
        """Invoke the oracle. Must raise ``OracleError`` if ``|prompt|``
        exceeds ``context_window()`` in tokens, or if the underlying LLM
        call fails.

        When ``env`` is supplied, ``prompt`` is treated as a ``str.format``
        template and substituted with ``prompt.format(**env)`` before the
        underlying LLM call. When ``env`` is ``None`` (the default), ``prompt``
        is sent verbatim — preserving M1 byte-equality for all existing
        Executor-driven Leaf calls.
        """
        ...

    def tokenize(self, text: str) -> int:
        """Return the token count for ``text``."""
        ...

    def context_window(self) -> int:
        """Advertise the model's K in tokens."""
        ...


@runtime_checkable
class StreamingOracle(Protocol):
    """Optional capability extension over ``Oracle`` for streaming responses.

    The base ``Oracle`` protocol intentionally does NOT require ``invoke_stream``
    — most kernel and stdlib mock oracles only need ``invoke`` (executor Leaf
    calls are the dominant path). Streaming is a dialog-side concern: only
    pipeline.py:1185 (the user-facing Pass-2 streaming response) needs it.

    A ``StreamingOracle`` is also an ``Oracle``: it supports both
    ``invoke`` (one-shot) and ``invoke_stream`` (chunked). ``LiteLLMOracle``
    is the canonical implementation. R10 step 7 wires
    ``pipeline.py:1185`` against this narrower Protocol.
    """

    def invoke(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
        *,
        model_override: str | None = None,
        env: dict[str, Any] | None = None,
    ) -> dict[str, Any] | str:
        ...

    def invoke_stream(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
        *,
        model_override: str | None = None,
        env: dict[str, Any] | None = None,
        user_message: str = "",
    ) -> Iterator[str]:
        """Stream the oracle's response token-by-token.

        Same env/template substitution semantics as ``invoke``: when ``env``
        is supplied, ``prompt`` is treated as a ``str.format`` template and
        substituted before the underlying streaming LLM call.

        ``user_message`` is the user-turn payload; passed verbatim to the
        underlying ``LLMInterface.generate_response_stream`` request body.
        Default empty string preserves byte-equivalence with the
        ``_invoke_unstructured`` shape used by ``invoke``.

        ``schema`` is accepted for signature parity with ``invoke`` but
        forwarded to the streaming request as ``response_format`` if
        supplied — semantics depend on model support; small Ollama models
        typically ignore mid-stream schema enforcement.

        Yields ``str`` chunks. Implementations MUST raise ``OracleError``
        if ``|prompt|`` exceeds ``context_window()`` in tokens.
        """
        ...

    def tokenize(self, text: str) -> int: ...

    def context_window(self) -> int: ...


# --------------------------------------------------------------
# LiteLLM adapter
# --------------------------------------------------------------


def _resolve_schema(schema_ref: str) -> type[BaseModel]:
    """Resolve a dotted path (``'pkg.mod.Cls'``) to a pydantic BaseModel."""
    try:
        mod_path, cls_name = schema_ref.rsplit(".", 1)
    except ValueError as e:
        raise OracleError(
            f"schema_ref must be a dotted path 'module.Class', got {schema_ref!r}"
        ) from e
    try:
        mod = importlib.import_module(mod_path)
    except ImportError as e:
        raise OracleError(f"cannot import schema module {mod_path!r}: {e}") from e
    cls = getattr(mod, cls_name, None)
    if cls is None or not isinstance(cls, type) or not issubclass(cls, BaseModel):
        raise OracleError(
            f"schema_ref {schema_ref!r} does not resolve to a pydantic BaseModel"
        )
    return cls  # type: ignore[no-any-return]


def _default_token_counter(model: str, text: str) -> int:
    """Best-effort token count. Tries ``litellm.token_counter`` first, then
    falls back to ``len(text) // CHARS_PER_TOKEN_FALLBACK``."""
    try:
        import litellm

        counter = getattr(litellm, "token_counter", None)
        if counter is not None:
            return int(counter(model=model, text=text))
    except Exception:
        # Any failure in litellm's tokenizer path → fallback. We don't
        # swallow errors from the LLM call itself; this is strictly the
        # tokenizer probe.
        pass
    return max(1, len(text) // CHARS_PER_TOKEN_FALLBACK)


class LiteLLMOracle:
    """Oracle adapter over an existing ``LLMInterface``.

    Construction signature mirrors how the rest of the codebase injects
    ``LLMInterface`` (pipeline.py takes an ``LLMInterface``, not a
    concrete class). We accept any subclass.
    """

    def __init__(
        self,
        llm: LLMInterface,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW,
        model_name: str | None = None,
    ):
        self._llm = llm
        self._K = context_window_tokens
        # Prefer an explicit model_name; else fall back to the LLM's own
        # ``.model`` attribute if present (LiteLLMInterface has one).
        self._model: str = str(model_name or getattr(llm, "model", "unknown"))

    # ----- Oracle protocol -----

    def context_window(self) -> int:
        return self._K

    def tokenize(self, text: str) -> int:
        return _default_token_counter(self._model, text)

    def invoke(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
        *,
        model_override: str | None = None,
        env: dict[str, Any] | None = None,
        user_message: str = "",
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any] | str:
        # DECISION D-PIVOT-1-ORACLE (plan_2026-04-27_32652286 step 10):
        # ``user_message`` and ``response_format`` kwargs added to support
        # the canonical Pass-2 main response site (pipeline.py L2223 / D-R10-7.5).
        # The legacy ``LiteLLMInterface.generate_response`` path sends
        # ``[{system: prompt}, {user: user_message}]``; the prior
        # ``_invoke_unstructured`` pinned ``user_message=""``. Default empty
        # preserves all M1 byte-equivalence for Executor-driven Leaf calls.
        # ``response_format`` (when supplied) is forwarded to the underlying
        # ``LLMInterface._make_llm_call`` for OpenAI-format JSON-schema
        # constrained decoding (terminal-state structured output path).
        # DECISION D-005: when ``env`` is supplied, treat ``prompt`` as a
        # ``str.format`` template and substitute env vars before the LLM call.
        # This is the unified call shape used by R3 pipeline callbacks
        # (D-PLAN-09 + D-PLAN-09-RESOLUTION). Executor-driven Leaf calls
        # always pre-substitute and pass ``env=None``, so behaviour is
        # byte-identical to M1 for those callers. Missing-key / index errors
        # in the template re-raise as ``OracleError`` to keep the boundary
        # contract single-typed.
        if env is not None:
            try:
                prompt = prompt.format(**env)
            except (KeyError, IndexError) as e:
                raise OracleError(
                    f"oracle template substitution failed: missing env var {e}"
                ) from e
        # E5: |P| > K guard — never truncate silently.
        n_tokens = self.tokenize(prompt)
        if n_tokens > self._K:
            raise OracleError(
                f"|P|={n_tokens} exceeds K={self._K} (model={self._model}); "
                "refusing to truncate. Re-plan with smaller tau or larger K."
            )
        # NOTE: ``model_override`` is accepted at the protocol level for
        # M1 compatibility but not threaded through LiteLLMInterface here
        # — the upstream ``LiteLLMInterface`` binds its model at __init__.
        # A future milestone can either construct a fresh interface per
        # call or expose a per-call model setter. For M1 we log an
        # explicit OracleError if caller passes one, preventing silent
        # mis-routing.
        if model_override is not None and model_override != self._model:
            raise OracleError(
                f"model_override={model_override!r} not supported in M1; "
                f"oracle is bound to model={self._model!r}. Construct a "
                "separate LiteLLMOracle for the target model."
            )

        # DECISION D-003: single source of routing — schema-bearing calls
        # go through extract_field's direct-completion bypass (D-008);
        # unstructured calls use generate_response. Pipeline callbacks
        # (R3) reach this branch via env-bearing template invocations;
        # Executor Leaf calls reach it with already-substituted prompts
        # and env=None. Both paths land here, so there is one routing
        # site rather than two parallel implementations.
        try:
            if schema is not None:
                return self._invoke_structured(prompt, schema)
            return self._invoke_unstructured(
                prompt,
                user_message=user_message,
                response_format=response_format,
            )
        except LLMResponseError as e:
            raise OracleError(f"LLM call failed: {e}") from e

    def invoke_stream(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
        *,
        model_override: str | None = None,
        env: dict[str, Any] | None = None,
        user_message: str = "",
    ) -> Iterator[str]:
        # DECISION D-010 (R10 step 6 — see plan_2026-04-27_32652286/decisions.md):
        # streaming oracle path. Mirrors ``invoke``'s env
        # substitution + |P|>K guard + model_override rejection. Routes
        # through ``LLMInterface.generate_response_stream`` for byte-
        # equivalence with the legacy pipeline.py:1185 streaming call site
        # (R10 step 7 wires this site behind FSM_LLM_ORACLE_RESPONSE_STREAM).
        if env is not None:
            try:
                prompt = prompt.format(**env)
            except (KeyError, IndexError) as e:
                raise OracleError(
                    f"oracle template substitution failed: missing env var {e}"
                ) from e
        n_tokens = self.tokenize(prompt)
        if n_tokens > self._K:
            raise OracleError(
                f"|P|={n_tokens} exceeds K={self._K} (model={self._model}); "
                "refusing to truncate. Re-plan with smaller tau or larger K."
            )
        if model_override is not None and model_override != self._model:
            raise OracleError(
                f"model_override={model_override!r} not supported in M1; "
                f"oracle is bound to model={self._model!r}. Construct a "
                "separate LiteLLMOracle for the target model."
            )
        # Build a ResponseGenerationRequest mirroring the legacy
        # pipeline.py:1185 shape. Schema (if any) becomes response_format
        # so the underlying interface's existing format-passthrough applies;
        # mid-stream schema enforcement depends on model + provider support.
        response_format: dict[str, Any] | None = None
        if schema is not None:
            json_schema = schema.model_json_schema()
            if (
                isinstance(json_schema, dict)
                and json_schema.get("type") == "object"
                and "properties" in json_schema
                and "required" not in json_schema
            ):
                json_schema = dict(json_schema)
                json_schema["required"] = list(json_schema["properties"].keys())
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": json_schema,
                },
            }
        req = ResponseGenerationRequest(
            system_prompt=prompt,
            user_message=user_message,
            response_format=response_format,
        )
        try:
            yield from self._llm.generate_response_stream(req)
        except LLMResponseError as e:
            raise OracleError(f"LLM streaming call failed: {e}") from e

    # ----- DECISION D-PIVOT-1-ORACLE: pre-built message + field surfaces -----
    # Added in plan_2026-04-27_32652286 step 10 to unblock the 3 deferred R10
    # sites (D-STEP-7-SUMMARY → D-PIVOT-1). These are thin adapters over the
    # underlying ``LLMInterface`` that preserve byte-equivalence with the
    # legacy ``self.llm_interface.<x>`` call shapes used by pipeline.py
    # L1289 (_make_llm_call) and L1633 (extract_field).

    def invoke_messages(
        self,
        messages: list[dict[str, str]],
        *,
        schema: type[BaseModel] | None = None,
        response_format: dict[str, Any] | None = None,
        call_type: str = "data_extraction",
    ) -> Any:
        """Forward a pre-built OpenAI-style message array to the underlying
        ``LLMInterface._make_llm_call``.

        Returns the **raw litellm response object** (not the parsed content)
        so callers that need bespoke parsing (e.g. pipeline.py L1289's
        ``<think>`` strip + markdown-fence + ``extracted_data`` dict-extract)
        retain byte-equivalence with the legacy path.

        ``schema`` (Pydantic) is converted to an OpenAI ``json_schema``
        ``response_format`` dict, mirroring the helper used by
        ``invoke_stream``. ``response_format`` (raw dict) wins if both are
        supplied.

        ``call_type`` defaults to ``"data_extraction"`` to match the legacy
        L1289 site's ``_make_llm_call(messages, "data_extraction")`` shape.
        """
        # Build response_format from schema if needed (mirror invoke_stream).
        if schema is not None and response_format is None:
            json_schema = schema.model_json_schema()
            if (
                isinstance(json_schema, dict)
                and json_schema.get("type") == "object"
                and "properties" in json_schema
                and "required" not in json_schema
            ):
                json_schema = dict(json_schema)
                json_schema["required"] = list(json_schema["properties"].keys())
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": json_schema,
                },
            }
        # Only forward ``response_format`` when set — preserves ABI for
        # underlying ``_make_llm_call`` implementations whose signature
        # may not include the kwarg (mock interfaces, narrower adapters).
        # The legacy L1289 site never passed ``response_format`` at all,
        # so default-None reproduces it byte-equivalently.
        try:
            if response_format is None:
                return self._llm._make_llm_call(  # type: ignore[attr-defined]
                    messages, call_type
                )
            return self._llm._make_llm_call(  # type: ignore[attr-defined]
                messages,
                call_type,
                response_format=response_format,
            )
        except LLMResponseError as e:
            raise OracleError(f"LLM messages call failed: {e}") from e

    def invoke_field(
        self,
        request: FieldExtractionRequest,
    ) -> FieldExtractionResponse:
        """Direct passthrough to ``LLMInterface.extract_field(request)``.

        Preserves the legacy outer-envelope schema
        ``{field_name, value, confidence, reasoning, is_valid}`` that
        ``extract_field`` enforces — materially different from
        ``_invoke_structured``'s bare-schema direct-litellm path (D-008).
        Per D-PIVOT-1, the dialog field-extraction site (pipeline.py L1633)
        uses this passthrough to remain byte-equivalent with the legacy
        envelope while satisfying SC7 (no ``self.llm_interface.*`` calls
        in the dialog turn).
        """
        try:
            return self._llm.extract_field(request)
        except LLMResponseError as e:
            raise OracleError(f"LLM field-extraction call failed: {e}") from e

    # ----- internal helpers -----

    def _invoke_unstructured(
        self,
        prompt: str,
        *,
        user_message: str = "",
        response_format: dict[str, Any] | None = None,
    ) -> str:
        req = ResponseGenerationRequest(
            system_prompt=prompt,
            user_message=user_message,
            response_format=response_format,
        )
        resp = self._llm.generate_response(req)
        return resp.message

    def _invoke_structured(
        self, prompt: str, schema: type[BaseModel]
    ) -> dict[str, Any]:
        # D-008: Bypass the field-extraction wrapper for structured calls.
        # The wrapper's outer ``{field_name, value:any, ...}`` schema confuses
        # small Ollama models (qwen3.5:4b returns the bare answer string in
        # ``value`` rather than a nested object). Instead, build a direct
        # ``json_schema`` response_format from the user's Pydantic schema
        # and route through ``generate_response``. This matches how
        # ``LiteLLMInterface`` handles user-supplied output schemas.
        json_schema = schema.model_json_schema()
        # D-011 (slice 4): Ollama's grammar-constrained decoding needs a
        # ``required`` field at the top level, otherwise the model is free
        # to emit no fields at all — and small models (qwen3.5:4b) take
        # that liberty by emitting prose instead of JSON. Pydantic's
        # ``model_json_schema()`` omits ``required`` whenever every field
        # has a default. We synthesise it: all properties are required at
        # the wire level. This is invisible to validation downstream
        # because defaults still apply when the model omits a field.
        if (
            isinstance(json_schema, dict)
            and json_schema.get("type") == "object"
            and "properties" in json_schema
            and "required" not in json_schema
        ):
            json_schema = dict(json_schema)
            json_schema["required"] = list(json_schema["properties"].keys())
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": json_schema,
            },
        }
        composed_prompt = (
            f"{prompt}\n\n"
            f"Return a JSON object matching the schema for "
            f"{schema.__name__}. No prose, no markdown fences."
        )
        # D-008 follow-up (slice 4): bypass ``generate_response`` and call
        # ``_make_llm_call`` directly, then extract the raw model content.
        # ``_parse_response_generation_response`` interprets a JSON body as a
        # structured-response wrapper and looks for ``message`` / ``reasoning``
        # keys — which collides with our user schemas (e.g. ToolDecision has
        # a ``reasoning`` field), causing it to return only that field's
        # value rather than the full JSON object. We need the raw content.
        # We also force ``temperature=0`` here so Ollama's grammar-constrained
        # decoding actually constrains output; this mirrors
        # ``apply_ollama_params(structured=True)`` for non-extraction calls.
        from litellm import completion as _litellm_completion  # local import

        from fsm_llm.ollama import (
            apply_ollama_params,
            is_ollama_model,
            prepare_ollama_messages,
        )

        messages = [
            {"role": "system", "content": composed_prompt},
            {"role": "user", "content": ""},
        ]
        call_params: dict[str, Any] = {
            "model": getattr(self._llm, "model", self._model),
            "messages": messages,
            "temperature": 0,
            "max_tokens": getattr(self._llm, "max_tokens", 1000),
            "response_format": response_format,
        }
        timeout = getattr(self._llm, "timeout", None)
        if timeout is not None:
            call_params["timeout"] = timeout
        # Carry over any model-side kwargs (api_key, base_url, etc.) from
        # the underlying interface, but never overwrite our explicit values.
        for k, v in getattr(self._llm, "kwargs", {}).items():
            call_params.setdefault(k, v)
        if is_ollama_model(call_params["model"]):
            apply_ollama_params(call_params, call_params["model"], structured=True)
            call_params["messages"] = prepare_ollama_messages(
                call_params["messages"],
                call_params["model"],
                call_params.get("response_format"),
            )
        try:
            resp = _litellm_completion(**call_params)
        except Exception as e:
            raise OracleError(f"LLM call failed: {e}") from e
        raw_content = ""
        try:
            raw_content = resp.choices[0].message.content or ""
        except Exception as e:
            raise OracleError(f"LLM response missing content: {e}") from e
        raw = raw_content.strip()
        # Strip optional markdown fences (some models add them despite
        # instructions).
        if raw.startswith("```"):
            # remove first line (```json or ```) and trailing ``` if present
            lines = raw.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        try:
            value: Any = json.loads(raw)
        except (ValueError, TypeError) as e:
            raise OracleError(
                f"structured call did not return valid JSON for schema "
                f"{schema.__name__}: {e}; raw={raw[:200]!r}"
            ) from e
        if not isinstance(value, dict):
            raise OracleError(
                f"structured call returned non-dict ({type(value).__name__}) "
                f"for schema {schema.__name__}"
            )
        # Validate against the schema here so the executor receives a
        # clean dict that round-trips. Validation errors bubble up as
        # OracleError for uniform handling.
        try:
            validated = schema.model_validate(value)
        except Exception as e:
            raise OracleError(
                f"oracle response failed schema {schema.__name__} validation: {e}"
            ) from e
        return validated.model_dump()


__all__ = [
    "LiteLLMOracle",
    "Oracle",
    "StreamingOracle",
    "_resolve_schema",
]
