"""Request and response models of the two passes (extraction, response, field extraction)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fsm_llm.constants import has_internal_prefix, is_forbidden_context_entry

from .errors import LLMResponseError
from .extraction import FieldType
from .limits import RESPONSE_TEXT_MAX_LENGTH

#: Request caps: the longest system prompt and the longest user message a
#: Pass 1 / Pass 2 request carries. The user-message cap also applies when
#: ``max_message_length`` is larger (it is configurable up to 50,000).
PROMPT_MAX_CHARS = 30000
USER_MESSAGE_MAX_CHARS = 10000

_PROMPT_CAPS = {
    "system_prompt": PROMPT_MAX_CHARS,
    "user_message": USER_MESSAGE_MAX_CHARS,
}
_LARGEST_CONTEXT_ENTRIES_NAMED = 3

_PromptRequestT = TypeVar("_PromptRequestT", bound="_PromptRequest")


def _largest_public_context_keys(context: Mapping[str, Any] | None) -> str:
    """The largest public entries of ``context`` as ``key (N characters)``,
    comma-joined, or ``""`` when there are none. Names and sizes only, never
    values; internal-prefix and secret-looking entries are skipped."""
    sizes: list[tuple[int, str]] = []
    for key, value in (context or {}).items():
        if not isinstance(key, str):
            continue
        if has_internal_prefix(key) or is_forbidden_context_entry(key, value):
            continue
        try:
            sizes.append((len(str(value)), key))
        except Exception:
            continue
    sizes.sort(key=lambda item: (-item[0], item[1]))
    return ", ".join(
        f"{key} ({size} characters)"
        for size, key in sizes[:_LARGEST_CONTEXT_ENTRIES_NAMED]
    )


class _PromptRequest(BaseModel):
    """The two fields every Pass 1 / Pass 2 request carries, and their caps.

    Contract: ``system_prompt`` is required (1..``PROMPT_MAX_CHARS`` chars);
    ``user_message`` is required but may be ``None`` (the turn has no user
    message: a step, the greeting), which is not an empty message
    (0..``USER_MESSAGE_MAX_CHARS`` chars). Unknown fields are refused.
    Subclasses add their own fields after these two.

    Build a request on a turn with ``checked``, not the bare constructor: the
    only failure it rewrites is an over-cap ``system_prompt`` or
    ``user_message``, which becomes an ``LLMResponseError`` naming the field,
    its length, the cap and (system prompt) the largest context keys.
    """

    # DECISION plan-2026-09-30T062855-07ad3f8c/D-044: request models refuse
    # unknown fields. Do NOT relax this to the pydantic default ("ignore"): a
    # removed field (``skip_generation``) was then dropped silently and the
    # request went out as a paid call. See decisions.md D-044.
    model_config = ConfigDict(extra="forbid")

    system_prompt: str = Field(
        ...,
        description="The prompt the request sends as its system turn",
        min_length=1,
        max_length=PROMPT_MAX_CHARS,
    )

    user_message: str | None = Field(
        ...,
        description=(
            "The user's input for this request; None when the turn has no user "
            "message (a step, the greeting), which is not an empty message"
        ),
        min_length=0,
        max_length=USER_MESSAGE_MAX_CHARS,
    )

    # DECISION plan-2026-11-06T093029-0068b055/D-002: an over-cap prompt is a
    # typed, rolled-back error that names the field, the cap and the largest
    # PUBLIC context keys (names and sizes, never values). Do NOT truncate the
    # largest leaves here (silent data loss, a model-visible change, and the
    # data paths never truncate), do NOT raise or drop the cap, and do NOT
    # re-inline the key checks below: has_internal_prefix and
    # is_forbidden_context_entry are the one policy. Any other ValidationError
    # is re-raised unchanged.
    @classmethod
    def checked(
        cls: type[_PromptRequestT],
        *,
        prompt_context: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> _PromptRequestT:
        """Build the request; an over-cap prompt raises ``LLMResponseError``.

        Parameters: ``fields`` are the model's own fields (as for the
        constructor); ``prompt_context`` is the context mapping the system
        prompt was built from, read only to name its largest keys.
        Returns the request. Raises ``LLMResponseError`` (chained from the
        ``ValidationError``) when every validation error is an over-cap
        ``system_prompt`` / ``user_message``; any other ``ValidationError``
        propagates unchanged.
        """
        try:
            return cls(**fields)
        except ValidationError as e:
            over = [
                err["loc"][0]
                for err in e.errors()
                if err["type"] == "string_too_long"
                and len(err["loc"]) == 1
                and err["loc"][0] in _PROMPT_CAPS
            ]
            if not over or len(over) != len(e.errors()):
                raise
            parts = []
            for field in over:
                cap = _PROMPT_CAPS[field]
                length = len(fields[field])
                if field == "user_message":
                    parts.append(
                        f"the user message is {length} characters but a request "
                        f"carries at most {cap} (max_message_length may be set "
                        f"higher, the request cap still applies); send a "
                        f"shorter message"
                    )
                else:
                    named = _largest_public_context_keys(prompt_context)
                    # The advice names context entries, so it is given only when
                    # entries were named; a workflow prompt or a long user message
                    # folded into the prompt has none.
                    advice = (
                        f"; the largest context entries are {named} "
                        "(names and character counts only); shrink or remove "
                        "them, scope the state with context_scope read_keys, or "
                        "shorten the history"
                        if named
                        else "; shorten the prompt or the history"
                    )
                    parts.append(
                        f"the system prompt is {length} characters but a request "
                        f"carries at most {cap}" + advice
                    )
            raise LLMResponseError("Prompt too large: " + "; and ".join(parts)) from e


class BulkExtractionRequest(_PromptRequest):
    """Request for free-form bulk data extraction from a state's
    ``extraction_instructions`` (no per-field schema).

    Deliberately minimal (only the two fields the prompt needs): unlike
    ``FieldExtractionRequest`` this is not targeting one named field, so a
    required ``field_name``/``field_type`` would not fit. See
    ``LLMInterface.extract_bulk_data`` for the ABC contract this feeds.
    """


class DataExtractionResponse(BaseModel):
    """
    Response from data extraction containing only extracted information.

    No user-facing message is generated at this stage.
    """

    extracted_data: dict[str, Any] = Field(
        default_factory=dict, description="Data extracted from user input"
    )

    confidence: float = Field(
        default=1.0,
        description="Confidence in the extraction (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    reasoning: str | None = Field(
        None,
        description="Internal reasoning for debugging (not shown to user)",
        max_length=RESPONSE_TEXT_MAX_LENGTH,
    )

    rejected_corrections: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Values the bulk extraction proposed for an already-stored key that "
            "the provenance rule refused to apply and that the user's message "
            "contains (D-032); empty on every other turn"
        ),
    )

    extraction_failed: bool = Field(
        default=False,
        description=(
            "True when the bulk extraction call of this turn raised, so a value "
            "the user restated may not have been stored (D-050)"
        ),
    )


class ResponseGenerationRequest(_PromptRequest):
    """
    Request for generating user-facing response based on final state.

    This request generates the actual message shown to users after
    data extraction and transition evaluation are complete.

    Never built for a silent state (empty ``response_instructions``): the
    pipeline makes no LLM call there.
    """

    transition_occurred: bool = Field(
        default=False, description="Whether a state transition occurred"
    )

    response_format: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional response format for constrained decoding. "
            "When set, passed to the LLM provider to enforce structured output "
            "(e.g., JSON schema). Falls back to free-text if provider unsupported."
        ),
    )

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        strict=True,
        description=(
            "Sampling temperature of this request; None keeps the interface's "
            "own. The pipeline sets it from the context key "
            "CONTEXT_KEY_RESPONSE_TEMPERATURE; an interface must honour it."
        ),
    )


class ResponseGenerationResponse(BaseModel):
    """
    Response containing the final user-facing message.

    Generated after all data extraction and transitions are complete.
    """

    message: str = Field(
        ...,
        description="Final user-facing message",
        max_length=RESPONSE_TEXT_MAX_LENGTH,
    )

    reasoning: str | None = Field(
        None,
        description="Internal reasoning for debugging",
        max_length=RESPONSE_TEXT_MAX_LENGTH,
    )

    message_type: str = Field(
        default="response", description="Type of the response message"
    )

    truncated: bool = Field(
        default=False,
        description=(
            "True when the provider cut the reply off at max_tokens: the message "
            "is a fragment or the cut-off apology (plan 0068b055 D-025)"
        ),
    )


class FieldExtractionRequest(_PromptRequest):
    """Request for extracting a single specific field from user input."""

    field_name: str = Field(
        ...,
        description="Name of the field to extract",
        min_length=1,
        max_length=100,
    )

    field_type: FieldType = Field(
        default="str",
        description="Expected type of the extracted value",
    )

    context: dict[str, Any] | None = Field(
        default=None,
        description="Dynamic context to guide extraction",
    )

    validation_rules: dict[str, Any] | None = Field(
        default=None,
        description="Validation rules for the extracted value",
    )


class FieldExtractionResponse(BaseModel):
    """Response from single-field extraction."""

    field_name: str = Field(
        ...,
        description="Name of the extracted field",
        min_length=1,
        max_length=100,
    )

    value: Any = Field(
        default=None,
        description="The extracted value (typed according to field_type)",
    )

    confidence: float = Field(
        default=1.0,
        description="Confidence in the extraction (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    reasoning: str | None = Field(
        default=None,
        description="Reasoning for the extraction decision",
        max_length=RESPONSE_TEXT_MAX_LENGTH,
    )

    is_valid: bool = Field(
        default=True,
        description="Whether the value passed validation rules",
    )

    validation_error: str | None = Field(
        default=None,
        description="Validation error message if is_valid is False",
    )

    truncated: bool = Field(
        default=False,
        description=(
            "True when the provider cut the extraction reply off at max_tokens "
            "(plan 0068b055 D-025)"
        ),
    )
