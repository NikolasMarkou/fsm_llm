"""Configuration models of the state kinds: context scope, completion and synthesize states."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fsm_llm.constants import (
    COMPLETION_TOOL_CHOICE_KEYWORDS,
    CONTEXT_KEY_AGENT_TRACE,
    DEFAULT_COMPLETION_MESSAGES_KEY,
    DEFAULT_COMPLETION_RESULT_KEY,
    RESERVED_CONTEXT_KEYS,
    has_internal_prefix,
    is_forbidden_context_entry,
)

from .extraction import checked_key_names
from .limits import (
    ASCII_IDENTIFIER_PATTERN,
    DEFAULT_SYNTHESIS_RESULT_KEY,
    SYNTHESIZE_GOAL_MAX_LENGTH,
    SYNTHESIZE_KEY_MAX_LENGTH,
)


class ContextScope(BaseModel):
    """Which context keys a state's prompts see.

    # DECISION plan-2026-09-21T203800-8a03483a/D-015
    # A typed model with `extra="forbid"`, not a free-form dict. Do NOT loosen
    # it back to `dict[str, Any]` or to `extra="ignore"`: a bare-string
    # `read_keys` made scoping a substring test (`"u" in "username"`), and a
    # misspelt key (`read_key`) silently disabled scoping. `State.context_scope`
    # still accepts a plain dict, which pydantic validates into this model.
    # See decisions.md D-015.
    """

    model_config = ConfigDict(extra="forbid")

    read_keys: list[str] | None = Field(
        default=None,
        description=(
            "Keys included in this state's prompts; None or [] means all "
            "user-visible context"
        ),
    )
    write_keys: list[str] | None = Field(
        default=None,
        description="Keys this state is expected to produce (advisory, not enforced)",
    )


def is_public_result_key(key: str) -> bool:
    """Whether ``key`` is a non-blank, public, non-reserved, non-``agent_trace`` name.

    Shared by the ``CompletionStateConfig`` and ``SynthesizeConfig`` validators.
    """
    return bool(
        key.strip()
        and not has_internal_prefix(key)
        and key not in RESERVED_CONTEXT_KEYS
        and key != CONTEXT_KEY_AGENT_TRACE
    )


class CompletionStateConfig(BaseModel):
    """The Pass-1 work of a completion state (``State.completion``).

    A completion state's Pass 1 is ONE ``LLMInterface.complete`` call over
    ``[system(instructions)] + context[messages_key]``: no prompt builder
    text, no conversation history and no neutral user turn. It is either a
    tool-calling turn (``tools``) or a structured turn (``response_format_key``
    names the context key holding a provider response format), never both.
    The reply is written to ``context[result_key]`` as
    ``{"kind": "calls" | "final" | "malformed", "text": str | None,
    "calls": [{"id", "name", "arguments"}]}``; transitions read
    ``<result_key>.kind``. The call is made only while ``result_key`` is unset
    (skip-if-set), so the consumer clears it once it has acted on the result.

    The transcript under ``messages_key`` is owned by the consumer (handlers
    append to it, for example with ``fsm_llm.llm.tool_exchange``); core never
    writes it and never runs a tool. A turn is refused with
    ``LLMResponseError`` and nothing is sent when:
      - the transcript is not a list, or ``fsm_llm.llm.check_tool_transcript``
        rejects it (an assistant tool-call message without exactly one tool
        result per call, an orphan tool result, a call without a non-empty
        ``str`` id or function name, a user message with ``None`` content);
      - it is a tool-calling turn (``tools``) and the transcript holds no
        ``user`` message: the task must be in the transcript, not only in
        ``instructions``. The transcript is an internal key that session
        files do not carry, so a session restored in the middle of a tool
        loop fails its next tool turn loudly instead of sending a request
        with no task;
      - a structured turn's response format is missing or not a dict.
    A reply whose tool call has no id, no tool name or arguments that are not
    a JSON object is ``kind: "malformed"`` with no calls (none of them runs).

    Fields:
      - ``tools``: OpenAI function schemas (``{"type": "function",
        "function": {"name", ...}}``), at least one, unique names.
      - ``tool_choice``: ``"auto"`` (sent when ``None``), ``"required"``,
        ``"none"``, or the name of one of ``tools`` (sent as a named-function
        choice). Only with ``tools``.
      - ``response_format_key``: internal-prefixed context key holding the
        response format of a structured turn.
      - ``instructions``: the system message; ``None`` sends none.
      - ``messages_key``: internal-prefixed context key of the transcript
        (a list of OpenAI chat messages; absent means empty, which a
        tool-calling turn refuses).
      - ``result_key``: public context key of the result (not internal, not
        reserved); the pipeline treats it as handler-only for extraction.

    Frozen; unknown fields are refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tools: list[dict[str, Any]] | None = Field(
        default=None, min_length=1, description="OpenAI function schemas"
    )
    tool_choice: str | None = Field(
        default=None,
        min_length=1,
        description="auto, required, none, or the name of a declared tool",
    )
    response_format_key: str | None = Field(
        default=None,
        min_length=1,
        description="Internal context key holding a structured turn's response format",
    )
    instructions: str | None = Field(
        default=None, description="System message of the request (None: none)"
    )
    messages_key: str = Field(
        default=DEFAULT_COMPLETION_MESSAGES_KEY,
        min_length=1,
        description="Internal context key of the consumer-owned transcript",
    )
    result_key: str = Field(
        default=DEFAULT_COMPLETION_RESULT_KEY,
        min_length=1,
        description="Public context key the result {kind, text, calls} is written to",
    )

    @model_validator(mode="after")
    def _validate_completion(self) -> CompletionStateConfig:
        """Exactly one turn kind; declared, unique tools; owned keys well placed."""
        if (self.tools is None) == (self.response_format_key is None):
            raise ValueError(
                "a completion state declares either non-empty tools or a "
                "response_format_key, exactly one"
            )
        names: list[str] = []
        for index, schema in enumerate(self.tools or []):
            function = schema.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            if (
                schema.get("type") != "function"
                or not isinstance(name, str)
                or not name
            ):
                raise ValueError(
                    f"completion tools[{index}] is not an OpenAI function schema "
                    '({"type": "function", "function": {"name": ...}})'
                )
            names.append(name)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"completion tools declare {duplicates} more than once")
        if self.tool_choice is not None:
            if self.tools is None:
                raise ValueError("completion tool_choice requires tools")
            if (
                self.tool_choice not in COMPLETION_TOOL_CHOICE_KEYWORDS
                and self.tool_choice not in names
            ):
                raise ValueError(
                    f"completion tool_choice {self.tool_choice!r} is neither one of "
                    f"{sorted(COMPLETION_TOOL_CHOICE_KEYWORDS)} nor a declared tool"
                )
        for field, key in (
            ("messages_key", self.messages_key),
            ("response_format_key", self.response_format_key),
        ):
            if key is not None and not has_internal_prefix(key):
                raise ValueError(
                    f"completion {field} {key!r} must be internal-prefixed so no "
                    "prompt, extraction or get_data reaches it"
                )
        if not is_public_result_key(self.result_key):
            raise ValueError(
                f"completion result_key {self.result_key!r} must be a public, "
                "non-reserved context key (transitions read it)"
            )
        return self


class SynthesizeConfig(BaseModel):
    """The work of a synthesize state (``State.synthesize``).

    A fixed ``goal`` (the author's text, not blank) a runner turns into a
    procedure (a child FSM), runs on the host conversation's model and merges
    back. Core only loads it: a bare ``API.converse`` finds no extraction and
    no passing transition, so it stays (BLOCKED). The runner writes
    ``context[result_key]`` (a dict with a ``status``), which the pipeline
    makes handler-only. ``inputs`` and ``return_keys`` (at least one) hold
    public names only, never ``result_key``; ``on_failure`` is a state id one
    transition reaches, and the whole FSM is checked with the real evaluator on
    sample results (``_check_synthesize_states``): no edge of the state may pass
    before the runner wrote a result (no result, or ``status`` ``running``), and
    an edge to ``on_failure`` must pass for ``status`` ``failed`` and for
    ``aborted``, so write it as ``status in [failed, aborted]`` (a bare
    ``== "failed"`` is refused). ``min_lifecycle`` is the least trusted stored
    procedure to run. Frozen, extras refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    goal: str = Field(min_length=1, max_length=SYNTHESIZE_GOAL_MAX_LENGTH)
    inputs: tuple[str, ...] = ()
    return_keys: tuple[str, ...] = Field(min_length=1)
    on_failure: str = Field(
        min_length=1, max_length=100, pattern=ASCII_IDENTIFIER_PATTERN
    )
    result_key: str = Field(
        default=DEFAULT_SYNTHESIS_RESULT_KEY,
        min_length=1,
        max_length=SYNTHESIZE_KEY_MAX_LENGTH,
    )
    # A pattern, not a Literal: the validator sweep test reads Literal values as
    # annotation names (test_validator_unit.py `_nested_models`).
    min_lifecycle: str = Field(default="verified", pattern=r"^(verified|trusted)$")

    @field_validator("inputs", "return_keys", mode="before")
    @classmethod
    def _validate_key_names(cls, value: object, info: Any) -> tuple[str, ...]:
        argument = f"synthesize {info.field_name}"
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{argument} must be a list, not {type(value).__name__}")
        names = checked_key_names(value, argument=argument)
        if not all(
            len(name) <= SYNTHESIZE_KEY_MAX_LENGTH
            and is_public_result_key(name)
            and not is_forbidden_context_entry(name, "")
            for name in names
        ):
            raise ValueError(
                f"{argument} may hold only public key names (no blank, internal, "
                f"reserved or secret-looking name, at most {SYNTHESIZE_KEY_MAX_LENGTH} "
                "characters)"
            )
        return names

    @model_validator(mode="after")
    def _validate_config(self) -> SynthesizeConfig:
        if not self.goal.strip():
            raise ValueError("synthesize goal must not be blank")
        if not is_public_result_key(self.result_key):
            raise ValueError(
                f"synthesize result_key {self.result_key!r} must be a public, "
                "non-reserved context key"
            )
        if self.result_key in self.inputs + self.return_keys:
            raise ValueError("synthesize result_key cannot be an input or return key")
        return self
