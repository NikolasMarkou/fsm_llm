"""Classification extraction configuration, schemas and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator, model_validator

from fsm_llm.constants import (
    DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE,
    MAX_MULTI_INTENTS,
)

from .limits import _ASCII_IDENTIFIER, MAX_CLASSIFICATION_INTENTS


@dataclass(frozen=True)
class ClassificationPromptConfig:
    """Controls classification prompt generation behavior.

    Bounds are checked at construction and a violation raises ``ValueError``
    naming the field: ``max_tokens >= 1``, ``0.0 <= temperature <= 2.0`` and
    ``1 <= max_intents <= MAX_MULTI_INTENTS`` (the
    ``MultiClassificationResult.intents`` pydantic cap, so a request for more
    intents than the result model can hold is refused up front instead of being
    silently truncated).

    ``temperature`` is advisory on Ollama models: ``apply_ollama_params(...,
    structured=True)`` forces ``temperature=0`` for structured classification
    output regardless of the value set here.
    """

    include_reasoning: bool = True
    max_tokens: int = 512
    temperature: float = 0.0
    include_entities: bool = True
    multi_intent: bool = False
    max_intents: int = 3

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be within 0.0..2.0, got {self.temperature}"
            )
        if not 1 <= self.max_intents <= MAX_MULTI_INTENTS:
            raise ValueError(
                f"max_intents must be within 1..{MAX_MULTI_INTENTS} "
                f"(the MultiClassificationResult.intents cap), got {self.max_intents}"
            )


class IntentDefinition(BaseModel):
    """A single intent class within a classification schema."""

    name: str = Field(description="Snake_case identifier used for handler routing")
    description: str = Field(description="Human-readable description shown to the LLM")

    @model_validator(mode="after")
    def validate_name_format(self) -> IntentDefinition:
        # DECISION plan-2026-07-19T191147-4b664252/D-025 [STALE]: this check is a
        # `model_validator` raising ValueError ON PURPOSE. Do NOT "simplify" it to
        # `Field(pattern=_ASCII_IDENTIFIER.pattern)` to match `State.id` /
        # `Transition.target_state`, even though that reads cleaner and enforces the
        # identical charset. A `pattern=` violation is a `string_pattern_mismatch`
        # error, and `validator.py`'s ALLOW-list (see D-013) deliberately EXCLUDES
        # that type -- so the swap would make `fsm-llm-validate` report is_valid=True
        # on an FSM that `API.from_file` then refuses to load. A ValueError from a
        # `model_validator` is a `value_error`, which the ALLOW-list DOES promote to
        # ERROR tier. The regex is shared; the enforcement mechanism must not be.
        if not _ASCII_IDENTIFIER.fullmatch(self.name):
            raise ValueError(
                f"Intent name must be alphanumeric ASCII with underscores, "
                f"got '{self.name}'"
            )
        return self


class ClassificationSchema(BaseModel):
    """
    Defines the complete set of intents for a classifier.

    Enforces mutual exclusivity guidelines: at most
    ``MAX_CLASSIFICATION_INTENTS`` intents per schema and a mandatory fallback
    class.
    """

    # DECISION plan-2026-07-19T191147-4b664252/D-013 [STALE]: the 15-intent cap
    # is declared HERE, once. `ClassificationExtractionConfig` (a state's
    # `classification_extractions` entry) inherits this class, so what loads
    # and what `MessagePipeline._execute_classification_extractions` builds at
    # runtime cannot disagree: an uncapped config used to pass `API.from_file`
    # and `fsm-llm-validate` and then fail on first ENTRY to the state (F-09).
    # Do NOT re-declare `intents` / `fallback_intent` on a subclass or copy the
    # cap into another model; use `MAX_CLASSIFICATION_INTENTS`. See
    # decisions.md D-013.
    intents: list[IntentDefinition] = Field(
        min_length=2,
        max_length=MAX_CLASSIFICATION_INTENTS,
        description=f"List of intent definitions (2-{MAX_CLASSIFICATION_INTENTS})",
    )
    fallback_intent: str = Field(
        description="Name of the fallback intent for ambiguous inputs"
    )
    confidence_threshold: float = Field(
        default=DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE,
        ge=0.0,
        le=1.0,
        description="Below this threshold, the classifier signals low confidence",
    )

    @model_validator(mode="after")
    def validate_schema(self) -> ClassificationSchema:
        names = [i.name for i in self.intents]
        if len(names) != len(set(names)):
            raise ValueError("Intent names must be unique")
        if self.fallback_intent not in names:
            raise ValueError(
                f"Fallback intent '{self.fallback_intent}' must be in the intent list"
            )
        return self

    @property
    def intent_names(self) -> list[str]:
        return [i.name for i in self.intents]


class ClassificationExtractionConfig(ClassificationSchema):
    """Configuration for classification-based extraction of a categorical value.

    Declared on a :class:`State` via ``classification_extractions`` to classify
    user input into predefined categories and store the result in context.
    Runs during Pass 1 alongside field extractions, before transition evaluation.

    The intent set (``intents``, ``fallback_intent``, ``confidence_threshold``
    and their rules, the cap included) is :class:`ClassificationSchema`'s; this
    class adds where the result goes and how the classifier is run.
    """

    field_name: str = Field(
        ...,
        description="Context key to store the classified intent in",
        min_length=1,
        max_length=100,
    )

    required: bool = Field(
        default=False,
        description="If True, failed classification triggers retry logic",
    )

    model: str | None = Field(
        default=None,
        description="Override LLM model for this classification (None = use pipeline's)",
    )

    prompt_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Override ClassificationPromptConfig fields as a dict. "
            "Keys: include_reasoning, max_tokens, temperature, include_entities, "
            "multi_intent, max_intents. Bounds (max_tokens >= 1, "
            "0.0 <= temperature <= 2.0, 1 <= max_intents <= 5) and the key set "
            "are checked at load: an unknown key or out-of-range value raises."
        ),
    )

    context_keys: list[str] | None = Field(
        default=None,
        description=(
            "Context keys the classifier sees as context data and that are "
            "snapshotted alongside the classification result. If None, the "
            "classifier sees the state's read_keys-scoped visible context and "
            "no snapshot is stored."
        ),
    )

    @field_validator("prompt_config")
    @classmethod
    def _validate_prompt_config(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Reject a ``prompt_config`` that ``ClassificationPromptConfig`` refuses.

        # DECISION plan-2026-09-22T080837-8b258a25/D-011
        # The verdict IS `ClassificationPromptConfig(**value)`, the object the
        # extraction site builds. Do NOT copy its keys or bounds into Field
        # constraints here: a second copy drifts and load and extraction would
        # disagree.
        """
        if value is None:
            return value
        try:
            ClassificationPromptConfig(**value)
        except TypeError as e:
            raise ValueError(f"invalid prompt_config: {e}") from e
        return value


def _coerce_entity_values(v: Any) -> dict[str, str | None]:
    """Coerce raw LLM ``entities`` into ``{name: str | None}``.

    Shared by the ``IntentScore`` and ``ClassificationResult`` validators.
    A non-dict becomes ``{}``; a list joins with ``", "``; ``None`` stays
    ``None``; anything else is ``str()``'d. Never raises.
    """
    # DECISION plan-2026-09-22T080837-8b258a25/D-010 (supersedes 80b0bd4d D-010)
    # Both entity validators MUST call this one function; do not inline a copy.
    # Do NOT map None to str(None): "None" is truthy and defeats a handler's
    # `if entities.get(k):` check. Do NOT narrow `entities` to dict[str, str].
    if not isinstance(v, dict):
        return {}
    return {
        k: (
            ", ".join(str(i) for i in val)
            if isinstance(val, list)
            else (str(val) if val is not None else None)
        )
        for k, val in v.items()
    }


class IntentScore(BaseModel):
    """A single scored intent within a classification result."""

    intent: str = Field(description="The classified intent name")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model confidence in this classification"
    )
    entities: dict[str, str | None] = Field(
        default_factory=dict, description="Extracted entities relevant to this intent"
    )

    @field_validator("entities", mode="before")
    @classmethod
    def coerce_entity_values(cls, v: Any) -> dict[str, str | None]:
        return _coerce_entity_values(v)


class ClassificationResult(BaseModel):
    """Result of a single-intent classification."""

    reasoning: str = Field(
        description="Chain-of-thought explanation preceding the classification"
    )
    intent: str = Field(description="The primary classified intent")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model confidence in this classification"
    )
    entities: dict[str, str | None] = Field(
        default_factory=dict, description="Extracted entities relevant to the intent"
    )

    @field_validator("entities", mode="before")
    @classmethod
    def coerce_entity_values(cls, v: Any) -> dict[str, str | None]:
        return _coerce_entity_values(v)

    #: Fixed threshold for is_below_default_threshold (no schema available).
    #: For schema-aware checks, use Classifier.is_low_confidence() instead.
    DEFAULT_CONFIDENCE_THRESHOLD: ClassVar[float] = (
        DEFAULT_TRANSITION_CLASSIFICATION_CONFIDENCE
    )

    @property
    def is_below_default_threshold(self) -> bool:
        """True when confidence is below the fixed DEFAULT_CONFIDENCE_THRESHOLD.

        Ignores the schema's ``confidence_threshold``; use
        ``Classifier.is_low_confidence(result)`` for the schema-aware check.
        """
        return self.confidence < self.DEFAULT_CONFIDENCE_THRESHOLD


class MultiClassificationResult(BaseModel):
    """Result of a multi-intent classification (compound queries)."""

    reasoning: str = Field(
        description="Chain-of-thought explanation preceding the classification"
    )
    intents: list[IntentScore] = Field(
        min_length=1,
        max_length=MAX_MULTI_INTENTS,
        description="Ranked list of detected intents, most probable first",
    )

    @property
    def primary(self) -> IntentScore:
        return self.intents[0]


class HierarchicalSchema(BaseModel):
    """Top-level schema for two-stage hierarchical classification."""

    domain_schema: ClassificationSchema = Field(
        description="Stage 1: domain-level classification schema"
    )
    intent_schemas: dict[str, ClassificationSchema] = Field(
        description="Stage 2: domain -> intent schema mapping"
    )

    @model_validator(mode="after")
    def validate_domain_coverage(self) -> HierarchicalSchema:
        domain_names = set(self.domain_schema.intent_names)
        schema_keys = set(self.intent_schemas.keys())
        missing = domain_names - schema_keys - {self.domain_schema.fallback_intent}
        if missing:
            raise ValueError(f"Missing intent schemas for domains: {missing}")
        return self


class HierarchicalResult(BaseModel):
    """Result of a hierarchical (two-stage) classification."""

    domain: ClassificationResult = Field(description="Stage 1 domain classification")
    intent: ClassificationResult = Field(
        description="Stage 2 intent classification within the domain"
    )
