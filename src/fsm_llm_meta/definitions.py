from __future__ import annotations

"""
Pydantic models for the meta-agent package.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .constants import Defaults


class ArtifactType(str, Enum):
    """Type of artifact the meta-agent can build."""

    FSM = "fsm"
    WORKFLOW = "workflow"
    AGENT = "agent"


class BuildProgress(BaseModel):
    """Progress tracking for artifact building."""

    total_required: int = 0
    completed: int = 0
    missing: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def percentage(self) -> float:
        if self.total_required == 0:
            return 0.0
        return (self.completed / self.total_required) * 100.0

    @property
    def is_complete(self) -> bool:
        return len(self.missing) == 0 and self.total_required > 0


class MetaAgentConfig(BaseModel):
    """Configuration for the meta-agent."""

    model: str = Defaults.MODEL
    temperature: float = Defaults.TEMPERATURE
    max_tokens: int = Defaults.MAX_TOKENS
    max_turns: int = Defaults.MAX_TURNS

    @field_validator("max_turns")
    @classmethod
    def validate_max_turns(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_turns must be at least 1")
        return v


class MetaAgentResult(BaseModel):
    """Result of a meta-agent build session."""

    artifact_type: ArtifactType
    artifact: dict[str, Any] = Field(default_factory=dict)
    artifact_json: str = ""
    is_valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    conversation_turns: int = 0
