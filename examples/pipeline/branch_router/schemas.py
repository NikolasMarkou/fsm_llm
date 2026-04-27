"""Pydantic schemas for the branch_router pipeline example."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CalcAnswer(BaseModel):
    """Calculator branch output."""

    answer: str = Field(default="", description="Numeric answer as a string.")
    expression: str = Field(default="", description="Parsed expression.")


class FactAnswer(BaseModel):
    """Factual branch output."""

    answer: str = Field(default="", description="Concise factual answer.")
    confidence: float = Field(default=0.0, description="Confidence 0..1.")
