"""The provider request builder: ``RequestMixin`` (``_build_call_params``), the connection
parameters shared with the embedder and the model checks.

Layer ``llm`` (plan d454fd62, step 73): split out of the old ``fsm_llm/llm/__init__.py`` (the old ``llm.py``); definitions unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fsm_llm.constants import (
    EMPTY_USER_MESSAGE_TURN,
    NEUTRAL_USER_TURN,
    RESERVED_LLM_CALL_KWARGS,
)
from fsm_llm.llm.ollama import (
    apply_ollama_params,
    build_ollama_response_format,
    is_ollama_model,
    prepare_ollama_messages,
)
from fsm_llm.logging import logger

# Call types whose reply is parsed as JSON: on Ollama they run at temperature 0.
_STRUCTURED_CALL_TYPES = frozenset(
    {"data_extraction", "field_extraction", "classification"}
)


# Call types whose reply is free text even when a response_format is sent (the
# Pass-2 ``output_schema`` case keeps the user's temperature on Ollama).
_FREE_TEXT_CALL_TYPES = frozenset({"response_generation"})


def _fill_empty_user_turns(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Give every user turn without text a provider-safe content.

    Args:
        messages: the provider message list of one request. A ``user``
            content of ``None`` means there is no user message; a string is
            the user's message, empty or not.

    Returns:
        A new list of new dicts. A ``user`` content of ``None`` becomes
        ``NEUTRAL_USER_TURN``, a ``""`` or whitespace-only one becomes
        ``EMPTY_USER_MESSAGE_TURN``; every other message is copied unchanged
        (an ``assistant`` tool-call message keeps ``content: None`` and its
        ``tool_calls``). The input is not mutated.

    Failure mode: none, never raises for a list of dicts.
    """
    # DECISION plan-2026-09-30T062855-07ad3f8c/D-044: ``None`` and an empty
    # string are different facts. Do NOT collapse them (`content or ""`, one
    # constant for both): an instruction-shaped turn sent for a real empty
    # message is read by the model as the user's utterance. See decisions.md
    # D-044.
    filled: list[dict[str, Any]] = []
    for message in messages:
        copy = dict(message)
        if message.get("role") == "user":
            content = message.get("content")
            if content is None:
                copy["content"] = NEUTRAL_USER_TURN
            elif isinstance(content, str) and not content.strip():
                copy["content"] = EMPTY_USER_MESSAGE_TURN
        filled.append(copy)
    return filled


def _checked_model(model: str) -> None:
    """Raise ``ValueError`` unless *model* is a non-blank string."""
    if not model or not model.strip():
        raise ValueError("model must be a non-empty string")


def _warn_reserved_kwargs(
    owner: str, kwargs: dict[str, Any], reserved: frozenset[str]
) -> None:
    """Warn that *owner* ignores the keys of *kwargs* the framework sets per call."""
    dropped = sorted(reserved.intersection(kwargs))
    if dropped:
        logger.warning(
            f"{owner} ignores reserved call kwargs {dropped}; "
            "the framework sets them per call"
        )


def _connection_params(
    model: str,
    kwargs: dict[str, Any],
    *,
    timeout: float | None,
    retries: int,
    reserved: frozenset[str],
) -> dict[str, Any]:
    """The connection part of a provider request: model, kwargs, timeout, retries.

    Interface contract (shared by ``LiteLLMInterface._build_call_params`` and
    ``LiteLLMEmbedder.embed``):
    - ``kwargs``: the owner's constructor kwargs (``api_key``, ``api_base``,
      ...); keys in ``reserved`` are dropped, the rest go first so the
      explicit params below cannot be overridden by them.
    - ``timeout``: sent when not ``None``. ``retries``: sent as the SDK's
      ``max_retries`` when above 0, otherwise omitted (SDK default).
    - Returns a new dict; the caller adds its call-specific params. Never
      raises.
    """
    params: dict[str, Any] = {
        **{k: v for k, v in kwargs.items() if k not in reserved},
        "model": model,
    }
    if timeout is not None:
        params["timeout"] = timeout

    # DECISION plan-2026-07-19T075908-70b6bdec/D-007 [STALE]
    # Retries are delegated to the provider SDK's own retry layer via
    # `max_retries`. Do NOT change this to `num_retries`: that key routes to
    # litellm's tenacity layer, which sits ON TOP of the SDK layer (giving
    # 2N+1 requests, not N+1) and retries EVERYTHING, including 400/401 —
    # deterministic failures that can never succeed. `max_retries` is one
    # layer with correct error classification. Do NOT hand-roll a retry loop
    # here either. Now shared by every request builder via this one helper.
    # See decisions.md D-007 (historical) / D-015 (this plan's extraction).
    if retries > 0:
        params["max_retries"] = retries
    return params


class RequestMixin:
    """The provider request builder of ``LiteLLMInterface``: ``_build_call_params`` and
    the model-specific parameters (plan d454fd62, step 73; bodies unchanged).

    Mixed into ``LiteLLMInterface``; ``self`` is that interface.
    """

    if TYPE_CHECKING:
        # Provided by ``LiteLLMInterface`` (declared for mypy only).
        model: str
        temperature: float
        max_tokens: int
        timeout: float | None
        retries: int
        kwargs: dict[str, Any]

        def _supported_openai_params(self) -> list[str] | None: ...

    def _build_call_params(
        self,
        messages: list[dict[str, Any]],
        call_type: str,
        *,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Build the ``litellm.completion(**call_params)`` kwargs shared by
        ``_make_llm_call`` (non-streaming), ``generate_response_stream`` and
        ``complete``. A user turn without text is filled by
        ``_fill_empty_user_turns`` (``None``: the neutral instruction; empty
        string: the empty-message placeholder); no other message is touched
        and the caller's list is not mutated.

        # DECISION plan-2026-09-12T135914-45a654de/D-015
        # Extracted from two independently-maintained ~40-line builders
        # (_make_llm_call and generate_response_stream) that the codebase's
        # own D-007 comments (plan-2026-07-19T075908-70b6bdec, now [STALE] but
        # NOT resolved — see findings/core-reasoning-harness-fixes.md item 3)
        # already flagged as duplicated and asked to be kept in sync by hand.
        # Do NOT re-split this back into two builders "for clarity" — that is
        # exactly the shape that was drifting. Do NOT change `max_retries` to
        # `num_retries`: see the retry-layer rationale at _connection_params: this
        # is the SAME constraint D-007 documented, now enforced in one place
        # instead of two. See decisions.md D-015 (this plan) and D-007 (prior
        # plan, historical context only — not re-litigated here).
        #
        # The two deltas between callers are parameters, not hardcoded
        # branches: (a) the structured-output/response_format branch below is
        # gated on `call_type in ["data_extraction", "field_extraction"]`,
        # which is NEVER true for a streaming call (streaming call_type is
        # always "response_generation") — so passing `stream=True` alone does
        # not need its own extra guard on that branch; (b) `stream=True`
        # literally adds `"stream": True` to the dict, nothing else.
        Args:
            messages: Message list for LLM
            call_type: Type of call for optimization
            response_format: Optional response format override for constrained
                decoding (e.g., JSON schema enforcement).
            stream: Whether this call is a streaming call (adds
                ``"stream": True``; never applies the forced-structured-output
                branch, though that branch's own ``call_type`` gate already
                makes it a no-op for the streaming caller's call_type).
            tools: OpenAI function schemas (``complete`` only). Sent as given,
                with ``tool_choice`` (``"auto"`` when ``None``), and never
                gated on litellm's supported-params list: it omits tools for
                models that call them natively (``ollama/``). The caller never
                passes ``response_format`` with tools (``CompletionRequest``
                refuses it).
            tool_choice: see ``tools``.
            temperature: per-call temperature; ``None`` keeps the interface's.
            max_tokens: per-call max tokens; ``None`` keeps the interface's.

        Returns:
            The kwargs dict ready to pass to ``litellm.completion(**...)``.
        """
        # Check for structured output support
        supported_params = self._supported_openai_params()

        # The connection part (kwargs first, then model, timeout, max_retries;
        # retry-layer rationale at _connection_params) is shared with
        # LiteLLMEmbedder.
        call_params = _connection_params(
            self.model,
            self.kwargs,
            timeout=self.timeout,
            retries=self.retries,
            reserved=RESERVED_LLM_CALL_KWARGS,
        )
        call_params["messages"] = _fill_empty_user_turns(messages)
        call_params["temperature"] = (
            self.temperature if temperature is None else temperature
        )
        call_params["max_tokens"] = (
            self.max_tokens if max_tokens is None else max_tokens
        )
        if stream:
            call_params["stream"] = True
        # DECISION plan-2026-10-01T093600-944e2692/D-027: native_fc's measured
        # request rules live HERE now (its private litellm call is gone):
        # `tools`+`tool_choice` are sent only with tools and never beside a
        # `response_format` (bf7ffe24/D-002; CompletionRequest refuses both),
        # never as a present-but-None key; Ollama preparation runs below
        # gated by `is_ollama_model`, a tool turn keeping the caller's
        # temperature (bf7ffe24/D-003, measured 0/3 -> 3/3 tool calls on 4b);
        # `seed` is sent only when the interface was built with one
        # (879d04a0/D-008). Do NOT re-home any of them in an agent. D-027.
        # DECISION plan-2026-10-01T093600-944e2692/D-029: both tool rules are
        # held HERE, whatever the free-string `call_type` says: a tool turn
        # never gets a `response_format` (neither the caller's, refused, nor
        # the forced extraction format) and never the structured temperature
        # 0. Do NOT gate them on `call_type` alone (`data_extraction` or
        # `classification` with tools broke both, review area 1 concern 1).
        if tools is not None:
            if response_format is not None:
                raise ValueError(
                    "A request sends tools or a response_format, never both"
                )
            call_params["tools"] = tools
            call_params["tool_choice"] = "auto" if tool_choice is None else tool_choice

        # Add structured output if supported and beneficial.
        # Do NOT force structured output for response_generation — the
        # response is user-facing natural language, not structured data
        # UNLESS an explicit response_format was provided (e.g., for
        # schema-enforced agent output). This branch never fires for a
        # streaming call: streaming's call_type is always
        # "response_generation", never "data_extraction"/"field_extraction".
        # It never fires for a tool turn either (D-029 above).
        if (
            tools is None
            and supported_params
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

        self._apply_model_specific_params(
            call_params,
            call_type,
            response_format=response_format,
            tool_turn=tools is not None,
        )

        # Ollama: prepend /nothink and embed schema in prompt
        call_params["messages"] = prepare_ollama_messages(
            call_params["messages"],
            self.model,
            call_params.get("response_format"),
        )

        return call_params

    def _apply_model_specific_params(
        self,
        call_params: dict,
        call_type: str,
        *,
        response_format: dict[str, Any] | None = None,
        tool_turn: bool = False,
    ) -> None:
        """Apply model-specific parameters to the LLM call.

        Handles quirks of specific model providers (e.g. Ollama's thinking
        mode) by mutating *call_params* in place. A call is structured
        (temperature 0 on Ollama) when its call type parses JSON, or when it
        sends a ``response_format`` and its call type is not free text (Pass
        2 keeps the user's temperature with an ``output_schema``). A tool
        turn (``tool_turn``) is never structured, whatever its call type.
        """
        # DECISION plan-2026-10-01T093600-944e2692/D-019: on Ollama a
        # response_format call runs at temperature 0 unless its call type is
        # free text, and a tool turn keeps the caller's temperature
        # (bf7ffe24/D-003, D-029). Do NOT drop the response_format arm (the
        # classifier and structured completion turns rely on it) and do NOT
        # key the tool rule on call_type. See D-019, D-029.
        # Ollama: disable thinking mode and force deterministic output
        # for structured calls (data extraction, classification, a
        # response_format completion).
        is_structured = not tool_turn and (
            call_type in _STRUCTURED_CALL_TYPES
            or (response_format is not None and call_type not in _FREE_TEXT_CALL_TYPES)
        )
        apply_ollama_params(call_params, self.model, structured=is_structured)
