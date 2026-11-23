"""Configuration models of the state kinds: context scope, completion, synthesize and call states."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

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
    DEFAULT_CALL_RESULT_KEY,
    DEFAULT_SYNTHESIS_RESULT_KEY,
    EXECUTABLE_ID_MAX_LENGTH,
    EXECUTABLE_ID_PATTERN,
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


def checked_public_names(value: object, *, argument: str) -> tuple[str, ...]:
    """The names in *value* (a list or tuple), each a public key name; else ``ValueError``.

    Contract: *value* must be a ``list`` or ``tuple`` of ``str`` (``checked_key_names``
    drops duplicates and refuses a non-collection or a non-``str`` member); every name must
    be non-blank, at most ``SYNTHESIZE_KEY_MAX_LENGTH`` characters, not internal-prefixed or
    reserved, and not secret-looking (``is_forbidden_context_entry``). *argument* names the
    value in the message. The one rule of the host keys a ``synthesize`` or a ``call`` state
    takes or returns (census D2).
    """
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
        return checked_public_names(value, argument=f"synthesize {info.field_name}")

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


_TargetId = Annotated[
    str,
    StringConstraints(
        pattern=EXECUTABLE_ID_PATTERN, min_length=1, max_length=EXECUTABLE_ID_MAX_LENGTH
    ),
]


def _positive_number(value: object) -> object:
    """A budget is a real number: a ``bool`` or a ``str`` is no count of steps or seconds."""
    if isinstance(value, (bool, str)):
        raise ValueError(f"a budget must be a number, not {value!r}")
    return value


class CallTargetsFrom(BaseModel):
    """The declaration that ``CallChoose.targets`` was resolved from a capability.

    Core has no registry and never resolves it: a caller that has one (the reasoning
    engine) re-resolves the closed ``targets`` at build time. Frozen; extras refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    capability: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)


class CallChoose(BaseModel):
    """Which target a call state calls, picked from a closed list by a context key.

    ``key`` names the context key whose value is a target id (it may be a handler-only
    key: a handler or an earlier state writes it, a call state declares no extraction).
    ``targets`` is the CLOSED list (at least one id): a value outside it is never called.
    ``targets_from`` is a declaration kept beside the resolved list (see
    ``CallTargetsFrom``). ``fallback``, when given, is one of ``targets`` and is called
    when the value is not in the list. Frozen; extras refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1, max_length=SYNTHESIZE_KEY_MAX_LENGTH)
    targets: tuple[_TargetId, ...] = Field(min_length=1)
    targets_from: CallTargetsFrom | None = None
    fallback: _TargetId | None = None

    @field_validator("key", mode="before")
    @classmethod
    def _validate_key(cls, value: object) -> object:
        checked_public_names([value], argument="call choose key")
        return value

    @field_validator("targets", mode="before")
    @classmethod
    def _validate_targets(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"call choose targets must be a list, not {value!r}")
        return checked_key_names(value, argument="call choose targets")

    @model_validator(mode="after")
    def _validate_choose(self) -> CallChoose:
        if self.fallback is not None and self.fallback not in self.targets:
            raise ValueError(
                f"call choose fallback {self.fallback!r} must be one of its targets"
            )
        return self


class CallStateConfig(BaseModel):
    """The work of a call state (``State.call``): run one target and merge its outputs.

    Exactly one of ``target`` (an executable id) or ``choose`` (see ``CallChoose``).
    ``inputs`` are the parent context keys handed to the target (declared inputs only,
    never the whole context), ``returns`` maps a parent key to a child output name
    (``{parent_key: child_output}``, empty is allowed), ``on_failure`` is a state id one
    transition reaches. The engine writes ``context[result_key]`` (a dict with a
    ``status`` and, once known, a ``reason``, the ``target`` and both identities), which the
    pipeline makes handler-only; the loader runs the state's edges on sample results
    (``kind_call.check_result_edges``): no edge may pass before the engine wrote a result
    and an edge to ``on_failure`` must pass for ``failed`` and for ``aborted``, so write it
    as ``status in [failed, aborted]``. ``max_steps`` and ``max_seconds`` bound the call
    (the budget of the pushed frame). Every key name is public and not secret-looking and
    ``result_key`` is none of the inputs, the parent return keys or the choose key.
    Core only loads it. Frozen; extras refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: _TargetId | None = None
    choose: CallChoose | None = None
    inputs: tuple[str, ...] = ()
    returns: dict[str, str] = Field(default_factory=dict)
    on_failure: str = Field(
        min_length=1, max_length=100, pattern=ASCII_IDENTIFIER_PATTERN
    )
    result_key: str = Field(
        default=DEFAULT_CALL_RESULT_KEY,
        min_length=1,
        max_length=SYNTHESIZE_KEY_MAX_LENGTH,
    )
    max_steps: int | None = Field(default=None, ge=1, strict=True)
    max_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @field_validator("inputs", mode="before")
    @classmethod
    def _validate_inputs(cls, value: object) -> tuple[str, ...]:
        return checked_public_names(value, argument="call inputs")

    @field_validator("returns", mode="before")
    @classmethod
    def _validate_returns(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError(
                f"call returns must be a mapping, not {type(value).__name__}"
            )
        # both sides are host or child key names: public, non-secret, bounded
        checked_public_names(list(value.keys()), argument="call returns")
        checked_public_names(list(value.values()), argument="call returns")
        return value

    @field_validator("max_steps", "max_seconds", mode="before")
    @classmethod
    def _validate_budget(cls, value: object) -> object:
        return _positive_number(value)

    @model_validator(mode="after")
    def _validate_config(self) -> CallStateConfig:
        if (self.target is None) == (self.choose is None):
            raise ValueError("a call state declares exactly one of target or choose")
        if not is_public_result_key(self.result_key):
            raise ValueError(
                f"call result_key {self.result_key!r} must be a public, "
                "non-reserved context key"
            )
        taken = {*self.inputs, *self.returns}
        if self.choose is not None:
            taken.add(self.choose.key)
        if self.result_key in taken:
            raise ValueError(
                "call result_key cannot be an input, a return key or the choose key"
            )
        return self
