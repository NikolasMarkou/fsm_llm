"""Pydantic schemas for the analytical_reasoning pipeline example."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Decomposition(BaseModel):
    parts: list[str] = Field(default_factory=list)
    domain: str = Field(default="")


class Analysis(BaseModel):
    findings: list[str] = Field(default_factory=list)
    key_insight: str = Field(default="")


class Integration(BaseModel):
    answer: str = Field(default="")
    confidence: float = Field(default=0.0)
