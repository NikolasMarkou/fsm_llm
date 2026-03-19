"""
Pydantic models for LLM-based structured classification.

Defines the schema contracts for single-intent, multi-intent, and hierarchical
classification results. All models enforce strict validation consistent with
the fsm_llm definitions module patterns.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------

class ClassificationError(Exception):
    """Base exception for classification operations."""
    pass


class SchemaValidationError(ClassificationError):
    """Raised when a classification schema is invalid."""
    pass


class ClassificationResponseError(ClassificationError):
    """Raised when the LLM returns an unparseable classification."""
    pass


# --------------------------------------------------------------
# Intent Definition
# --------------------------------------------------------------

class IntentDefinition(BaseModel):
    """A single intent class within a classification schema."""

    name: str = Field(
        description="Snake_case identifier used for handler routing"
    )
    description: str = Field(
        description="Human-readable description shown to the LLM"
    )

    @model_validator(mode="after")
    def validate_name_format(self) -> "IntentDefinition":
        if not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"Intent name must be alphanumeric with underscores, got '{self.name}'"
            )
        return self


# --------------------------------------------------------------
# Classification Schema
# --------------------------------------------------------------

class ClassificationSchema(BaseModel):
    """
    Defines the complete set of intents for a classifier.

    Enforces mutual exclusivity guidelines: max 15 intents per schema
    and a mandatory fallback class.
    """

    intents: list[IntentDefinition] = Field(
        min_length=2,
        description="List of intent definitions (2-15 recommended)"
    )
    fallback_intent: str = Field(
        description="Name of the fallback intent for ambiguous inputs"
    )
    confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Below this threshold, the classifier signals low confidence"
    )

    @model_validator(mode="after")
    def validate_schema(self) -> "ClassificationSchema":
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

    @property
    def intent_enum(self) -> type[Enum]:
        """Dynamically build an Enum from intent names."""
        return Enum("Intent", {name: name for name in self.intent_names})

    def get_intent(self, name: str) -> Optional[IntentDefinition]:
        for intent in self.intents:
            if intent.name == name:
                return intent
        return None


# --------------------------------------------------------------
# Classification Results
# --------------------------------------------------------------

class IntentScore(BaseModel):
    """A single scored intent within a classification result."""

    intent: str = Field(description="The classified intent name")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence in this classification"
    )
    entities: dict[str, str] = Field(
        default_factory=dict,
        description="Extracted entities relevant to this intent"
    )


class ClassificationResult(BaseModel):
    """Result of a single-intent classification."""

    reasoning: str = Field(
        description="Chain-of-thought explanation preceding the classification"
    )
    intent: str = Field(description="The primary classified intent")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence in this classification"
    )
    entities: dict[str, str] = Field(
        default_factory=dict,
        description="Extracted entities relevant to the intent"
    )

    @property
    def is_low_confidence(self) -> bool:
        """Check against the default threshold (0.6). Use schema-aware check in Classifier."""
        return self.confidence < 0.6


class MultiClassificationResult(BaseModel):
    """Result of a multi-intent classification (compound queries)."""

    reasoning: str = Field(
        description="Chain-of-thought explanation preceding the classification"
    )
    intents: list[IntentScore] = Field(
        min_length=1,
        max_length=5,
        description="Ranked list of detected intents, most probable first"
    )

    @property
    def primary(self) -> IntentScore:
        return self.intents[0]


# --------------------------------------------------------------
# Hierarchical Classification
# --------------------------------------------------------------

class DomainSchema(BaseModel):
    """
    Maps a domain to its intent sub-schema for hierarchical classification.

    Use when the total intent count exceeds ~15. Stage 1 classifies domain,
    stage 2 classifies intent within that domain.
    """

    domain: str = Field(description="Domain identifier (snake_case)")
    intent_schema: ClassificationSchema = Field(
        description="Intent schema for this domain"
    )


class HierarchicalSchema(BaseModel):
    """Top-level schema for two-stage hierarchical classification."""

    domain_schema: ClassificationSchema = Field(
        description="Stage 1: domain-level classification schema"
    )
    intent_schemas: dict[str, ClassificationSchema] = Field(
        description="Stage 2: domain -> intent schema mapping"
    )

    @model_validator(mode="after")
    def validate_domain_coverage(self) -> "HierarchicalSchema":
        domain_names = set(self.domain_schema.intent_names)
        schema_keys = set(self.intent_schemas.keys())
        missing = domain_names - schema_keys - {self.domain_schema.fallback_intent}
        if missing:
            raise ValueError(
                f"Missing intent schemas for domains: {missing}"
            )
        return self


class HierarchicalResult(BaseModel):
    """Result of a hierarchical (two-stage) classification."""

    domain: ClassificationResult = Field(
        description="Stage 1 domain classification"
    )
    intent: ClassificationResult = Field(
        description="Stage 2 intent classification within the domain"
    )
