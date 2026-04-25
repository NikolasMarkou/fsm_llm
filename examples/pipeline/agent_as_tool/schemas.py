"""Pydantic schemas for agent_as_tool λ-twin."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Context(BaseModel):
    summary: str = Field(default="", description="Relevant context or recall")
    key_points: str = Field(default="", description="2-3 key points as a single string")


class Answer(BaseModel):
    answer: str = Field(default="", description="Final answer")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
