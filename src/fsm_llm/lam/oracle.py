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
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from fsm_llm.definitions import (
    LLMResponseError,
    ResponseGenerationRequest,
)
from fsm_llm.llm import LLMInterface

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
    ) -> dict[str, Any] | str:
        """Invoke the oracle. Must raise ``OracleError`` if ``|prompt|``
        exceeds ``context_window()`` in tokens, or if the underlying LLM
        call fails."""
        ...

    def tokenize(self, text: str) -> int:
        """Return the token count for ``text``."""
        ...

    def context_window(self) -> int:
        """Advertise the model's K in tokens."""
        ...


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
    ) -> dict[str, Any] | str:
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

        try:
            if schema is not None:
                return self._invoke_structured(prompt, schema)
            return self._invoke_unstructured(prompt)
        except LLMResponseError as e:
            raise OracleError(f"LLM call failed: {e}") from e

    # ----- internal helpers -----

    def _invoke_unstructured(self, prompt: str) -> str:
        req = ResponseGenerationRequest(
            system_prompt=prompt,
            user_message="",
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
        req = ResponseGenerationRequest(
            system_prompt=composed_prompt,
            user_message="",
            response_format=response_format,
        )
        resp = self._llm.generate_response(req)
        raw = (resp.message or "").strip()
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
    "Oracle",
    "LiteLLMOracle",
    "_resolve_schema",
]
