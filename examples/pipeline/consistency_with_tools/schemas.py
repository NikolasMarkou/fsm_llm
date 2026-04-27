"""Schemas for consistency_with_tools pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CalcOut(BaseModel):
    """A single calculation attempt."""

    answer: float = Field(default=0.0, description="The numeric final answer.")
    steps: list[str] = Field(default_factory=list, description="Step-by-step working.")
    confidence: float = Field(default=0.0, description="0..1 confidence.")
