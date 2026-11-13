"""Wire models of ``LLMInterface.complete`` and the usage counters."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CompletionRequest(BaseModel):
    """One provider completion: tool calling, plain text or structured output.

    The request primitive of ``LLMInterface.complete``. ``messages`` are sent
    as given, in OpenAI chat format: a ``user`` content of ``None`` means no
    user message and a ``""`` one an empty message (the LLM layer fills each
    with its own turn); an ``assistant`` message may carry ``tool_calls`` with
    ``content: None``, and ``tool`` messages carry ``tool_call_id``.

    Fields:
      - ``tools``: OpenAI function schemas (at least one when given).
      - ``tool_choice``: ``"auto"`` (the default when ``None``), ``"required"``,
        ``"none"`` or a named-function dict; only with ``tools``.
      - ``response_format``: a provider response format (structured output).
        A request carries ``tools`` or ``response_format``, never both.
      - ``temperature`` / ``max_tokens``: per-call values; ``None`` keeps the
        interface's own.
      - ``call_type``: the call's label (``"completion"`` unless the caller
        names its call path, e.g. ``"classification"``).

    Frozen, and unknown fields are refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: list[dict[str, Any]] = Field(
        ..., min_length=1, description="Provider messages, sent in order"
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None, min_length=1, description="OpenAI function schemas"
    )
    tool_choice: str | dict[str, Any] | None = Field(
        default=None, description="Tool choice; None means 'auto'"
    )
    response_format: dict[str, Any] | None = Field(
        default=None, description="Provider response format (structured output)"
    )
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Per-call sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Per-call completion token limit"
    )
    call_type: str = Field(
        default="completion", min_length=1, description="Label of the call path"
    )

    @model_validator(mode="after")
    def _tools_xor_response_format(self) -> CompletionRequest:
        """Refuse ``tools`` with ``response_format``, and ``tool_choice`` alone."""
        if self.tools is not None and self.response_format is not None:
            raise ValueError(
                "a completion request carries tools or response_format, not both"
            )
        if self.tool_choice is not None and self.tools is None:
            raise ValueError("tool_choice requires tools")
        return self


class ModelToolCall(BaseModel):
    """One tool call a model asked for, with its arguments decoded.

    ``arguments`` is always a JSON object: a call whose arguments do not
    decode to one never becomes a ``ModelToolCall`` (the whole turn is
    ``malformed`` instead). Frozen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(
        ...,
        min_length=1,
        description="Provider id of the call (a call without one is malformed)",
    )
    name: str = Field(..., min_length=1, description="Name of the called tool")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Decoded call arguments"
    )


# The provider's ``finish_reason`` for a reply cut off at ``max_tokens``: the one
# signal that tells a cut fragment from a whole reply (plan 0068b055 D-024).
FINISH_REASON_LENGTH = "length"

# DECISION plan-2026-11-06T093029-0068b055/D-025
# The turn's cut-off record: a `context.data` key (internal prefix, so it is in
# no prompt, no `API.get_data` view and no handler-visible user data; handlers
# and `fsm_manager.get_complete_conversation(...)["collected_data"]` do see it).
# Value: a JSON-native dict `{"reply": True}` (Pass 2 reply cut by max_tokens)
# and/or `{"fields": [names]}` (extractions cut, in call order); the key is
# ABSENT when nothing was cut. The pipeline removes it at the start of every
# turn and writes it only through `context.record_cut_off` (the pipeline and the
# completion executor call it), so it describes the LAST turn only. Do NOT put
# it in the completion-state result `{kind, text, calls}` (documented shape), do
# NOT re-inline the literal, and do NOT add a second flag for the same fact. It
# lives here, not in `constants.py` (a hash-pinned bench file). See decisions.md
# D-025.
CONTEXT_KEY_CUT_OFF = "_cut_off"


class CompletionResponse(BaseModel):
    """What one ``LLMInterface.complete`` call returned.

    ``kind``:
      - ``"calls"``: the model asked for one or more tools (``calls``); any
        text it wrote beside them is kept in ``text``.
      - ``"final"``: a reply with no tool call; ``text`` is the reply, or
        ``None`` when the model returned no text at all.
      - ``"malformed"``: a tool-call turn that cannot be run (a call whose
        arguments are not a JSON object, that names no tool or that carries
        no id, or the provider's own malformed-tool-call error). It carries no calls, so
        none of the turn's calls runs, the valid ones included.

    ``finish_reason`` is the provider's own reason the reply ended (``stop``,
    ``length``, ``tool_calls``, ...), ``None`` when it sent none. ``length``
    (``truncated``) means the reply was cut off at ``max_tokens``: its ``text``
    (and any tool-call arguments) may be a fragment.

    A provider outage is not a response: ``complete`` raises. Frozen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["calls", "final", "malformed"] = Field(
        ..., description="calls, final or malformed"
    )
    text: str | None = Field(default=None, description="Reply text, if any")
    calls: tuple[ModelToolCall, ...] = Field(
        default=(), description="Requested tool calls (only for kind='calls')"
    )
    finish_reason: str | None = Field(
        default=None,
        description="Provider's finish reason (length: cut off at max_tokens)",
    )

    @property
    def truncated(self) -> bool:
        """True when the provider cut the reply off at ``max_tokens``."""
        return self.finish_reason == FINISH_REASON_LENGTH

    @model_validator(mode="after")
    def _calls_match_kind(self) -> CompletionResponse:
        """``calls`` is non-empty exactly when ``kind`` is ``"calls"``."""
        if (self.kind == "calls") != bool(self.calls):
            raise ValueError("calls must be non-empty exactly when kind is 'calls'")
        return self


class LLMCallCounts(BaseModel):
    """Provider-call counters of one kind (or of all kinds summed).

    ``calls`` counts every provider request, the failed ones included;
    ``errors`` the requests that raised; ``usage_missing`` the answered
    requests whose reply carried no token usage (every streamed call). The
    token fields sum the usage of the replies that carried one. Frozen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    calls: int = Field(default=0, ge=0, description="Provider requests sent")
    errors: int = Field(default=0, ge=0, description="Requests that raised")
    usage_missing: int = Field(
        default=0, ge=0, description="Answered requests without token usage"
    )
    prompt_tokens: int = Field(default=0, ge=0, description="Prompt tokens")
    completion_tokens: int = Field(default=0, ge=0, description="Completion tokens")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens")


class LLMUsage(LLMCallCounts):
    """A snapshot of an LLM interface's (or an embedder's) usage counters.

    The inherited fields are the totals over every kind; ``by_kind`` holds
    the same counters per call kind (``generate``, ``stream``, ``extract``,
    ``classify``, ``complete``; ``embed`` for ``LiteLLMEmbedder``; only kinds
    with at least one call appear) and
    sums to the totals. A snapshot is a copy: later calls do not change it.
    Frozen.
    """

    by_kind: dict[str, LLMCallCounts] = Field(
        default_factory=dict, description="Counters per call kind"
    )
