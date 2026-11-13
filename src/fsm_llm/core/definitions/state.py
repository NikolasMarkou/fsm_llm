"""The ``State`` model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from fsm_llm.constants import (
    TRANSITION_CLASSIFICATION_THRESHOLD_KEY,
    has_internal_prefix,
)
from fsm_llm.state_kinds import kind_of, state_problem

from .classification import ClassificationExtractionConfig
from .extraction import FieldExtractionConfig
from .kinds import CompletionStateConfig, ContextScope, SynthesizeConfig
from .limits import ASCII_IDENTIFIER_PATTERN
from .transition import Transition


class State(BaseModel):
    """
    Enhanced state definition for the 2-pass architecture.

    Separates data extraction concerns from response generation.
    """

    id: str = Field(
        ...,
        description="Unique state identifier",
        min_length=1,
        max_length=100,
        pattern=ASCII_IDENTIFIER_PATTERN,
    )

    description: str = Field(
        ...,
        description="Human-readable state description",
        min_length=1,
        max_length=300,
    )

    purpose: str = Field(
        ...,
        description="What should be accomplished in this state",
        min_length=1,
        max_length=500,
    )

    extraction_instructions: str | None = Field(
        None,
        description="Instructions for data extraction in this state",
        max_length=5000,
    )

    response_instructions: str | None = Field(
        None,
        description="Instructions for response generation in this state",
        max_length=5000,
    )

    transitions: list[Transition] = Field(
        default_factory=list, description="Available transitions from this state"
    )

    required_context_keys: list[str] | None = Field(
        default=None, description="Context keys that should be collected"
    )

    extraction_retries: int = Field(
        default=1,
        ge=0,
        le=3,
        description=(
            "Number of refinement extraction passes when confidence is low "
            "or required_context_keys are missing. 0 disables multi-pass."
        ),
    )

    extraction_confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum confidence for extraction to be accepted without retry. "
            "0.0 disables confidence-based retry (only missing-key retry applies)."
        ),
    )

    transition_classification: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Custom classification config for ambiguous transition resolution. "
            "dict: user-provided config with custom intent descriptions and thresholds. "
            "None: auto-generate classification schema from transition descriptions. "
            "Classification is always-on for ambiguous transitions."
        ),
    )

    field_extractions: list[FieldExtractionConfig] | None = Field(
        default=None,
        description=(
            "Targeted field extractions to run after bulk extraction. "
            "Each entry extracts a single named field with custom instructions, "
            "dynamic context selection, and validation rules."
        ),
    )

    classification_extractions: list[ClassificationExtractionConfig] | None = Field(
        default=None,
        description=(
            "Classification-based extractions to run during Pass 1. "
            "Each entry classifies user input into predefined categories "
            "and stores the result in context for transition evaluation."
        ),
    )

    context_scope: ContextScope | None = Field(
        default=None,
        description=(
            "Optional context scoping for this state (a ContextScope or a dict "
            "with only 'read_keys' and/or 'write_keys', each a list of strings). "
            "Controls which context keys are injected into LLM prompts. "
            "When None, all user-visible context is injected (default behavior)."
        ),
    )

    # DECISION plan-2026-10-01T093600-944e2692/D-003: one optional, additive
    # field (v4.1 unchanged: every existing definition loads as before). Its
    # Pass-1 work is a `complete` call over a consumer-owned transcript in
    # context, NEVER `Conversation.exchanges` (25 readers assume str pairs).
    # Do NOT rename it `tool_calling` (it also carries the structured turn),
    # do NOT make an LLM call inside a handler instead (a second call path
    # beside Pass 1, outside rollback and the meter), and do NOT add a
    # required companion field (STOP IF 9 of plan 944e2692). See D-003.
    # Plan 9013b0e3 D-004 (appended): "v4.1 unchanged" was true for the 4.1
    # default; the definition default is now 4.2, so an explicit "version":
    # "4.1" is hashed into `fsm_id` and the library digest and differs from an
    # omitted one (omitted-version ids did not move).
    completion: CompletionStateConfig | None = Field(
        default=None,
        description=(
            "Completion state: Pass 1 is one LLM completion (tool calling or a "
            "structured turn) over a consumer-owned transcript in context; the "
            "result {kind, text, calls} is written to its result_key. Excludes "
            "field_extractions, classification_extractions, "
            "required_context_keys and extraction_instructions."
        ),
    )

    # DECISION plan-2026-11-10T194139-2cefeb61/D-002: one optional, additive
    # field (v4.1 unchanged, no existing `fsm_id` moves). Core only LOADS it; a
    # runner outside core pushes the procedure. Do NOT add a push seam to the
    # pipeline or `FSMManager` (a bare `converse` stays BLOCKED, by design), and
    # do NOT make `FSMValidator.validate()` stricter than this loader (the
    # never-stricter tests pin it; the flow gates live in `flow_analysis.py`).
    # Plan 9013b0e3 D-004 (appended): "no existing `fsm_id` moves" holds for
    # definitions that omit `version`; the default is now 4.2, so an explicit
    # "version": "4.1" definition has a new `fsm_id`.
    synthesize: SynthesizeConfig | None = Field(
        default=None,
        description=(
            "Synthesize state: a runner pushes a procedure for the goal and "
            "writes the outcome to result_key. Excludes completion and every "
            "extraction field."
        ),
    )

    @field_validator("transition_classification")
    @classmethod
    def _validate_transition_classification(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Check the shape ``_build_transition_classification_schema`` reads.

        The reserved key ``TRANSITION_CLASSIFICATION_THRESHOLD_KEY`` maps to a
        real number (not a bool) in [0, 1]. Every other key names a target state
        and maps to a dict whose only allowed key is ``description`` (a str or
        None). None (auto mode) and an empty dict pass.
        """
        if value is None:
            return value
        for key, entry in value.items():
            if key == TRANSITION_CLASSIFICATION_THRESHOLD_KEY:
                if (
                    isinstance(entry, bool)
                    or not isinstance(entry, int | float)
                    or not 0.0 <= entry <= 1.0
                ):
                    raise ValueError(
                        f"transition_classification {key!r} must be a number "
                        f"within 0.0..1.0, got {entry!r}"
                    )
                continue
            if not isinstance(entry, dict):
                raise ValueError(
                    f"transition_classification {key!r} must be a dict like "
                    f'{{"description": "..."}}, got {entry!r}'
                )
            unknown = sorted(str(k) for k in set(entry) - {"description"})
            if unknown:
                raise ValueError(
                    f"transition_classification {key!r} has unknown keys "
                    f"{unknown}; only 'description' is allowed"
                )
            description = entry.get("description")
            if description is not None and not isinstance(description, str):
                raise ValueError(
                    f"transition_classification {key!r} 'description' must be "
                    f"a str, got {description!r}"
                )
        return value

    @model_validator(mode="after")
    def _validate_extraction_configs(self) -> State:
        """Reject state configs that can never work (B10).

        # DECISION plan-2026-09-21T203800-8a03483a/D-016
        # A plain ValueError (pydantic `value_error`, promoted to ERROR by
        # validator.py) so loader and `fsm-llm-validate` agree. Do NOT downgrade
        # these to warnings: two configs of one channel writing one
        # `field_name` race for the same context key, and an empty or
        # internal-prefixed required key is never extracted (internal keys are
        # filtered out of extraction), so the state can never be satisfied.
        # DECISION plan-2026-09-21T203800-8a03483a/D-033: duplicates are checked
        # per list. Do NOT extend the check ACROSS field_extractions and
        # classification_extractions: one explicit extraction named like a
        # classification field is the supported below-threshold fallback of
        # plan-2026-09-19T175721-21cd7f8e/D-006 (now enforced in
        # `_conversation_minted_keys`, src/fsm_llm/state_kinds.py). See decisions.md
        # D-016, D-033.
        """
        for channel, configs in (
            ("field_extractions", self.field_extractions),
            ("classification_extractions", self.classification_extractions),
        ):
            names = [c.field_name for c in configs or []]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(
                    f"State '{self.id}': field_name(s) {duplicates} are declared "
                    f"more than once in {channel}"
                )
        bad_keys = [
            key
            for key in self.required_context_keys or []
            if not key.strip() or has_internal_prefix(key)
        ]
        if bad_keys:
            raise ValueError(
                f"State '{self.id}': required_context_keys {bad_keys} are empty or "
                "internal-prefixed and can never be extracted"
            )
        # The completion and synthesize rules (forbidden declarations, needs a
        # transition, on_failure must be a target) live with their kind: see
        # `state_kinds.state_problem`.
        problem = state_problem(self)
        if problem is not None:
            raise ValueError(f"State '{self.id}': {problem}")
        return self

    # DECISION plan-2026-11-12T073150-d454fd62/D-015: `kind` is a plain
    # property over the kind registry. Do NOT make it a field, a
    # `computed_field` or a `PrivateAttr`: each would reach `model_dump`, which
    # `API.process_fsm_definition` hashes with `exclude_defaults=True`, and move
    # every `fsm_id` and library digest (a plain property is also absent from
    # `model_fields` and the JSON schema; a test pins all three).
    @property
    def kind(self) -> str:
        """The state's kind name: ``conversation``, ``completion`` or
        ``synthesize`` (see ``fsm_llm.state_kinds``). Derived from the config
        fields; not part of the definition, its dump or its ``fsm_id``."""
        return kind_of(self).name
