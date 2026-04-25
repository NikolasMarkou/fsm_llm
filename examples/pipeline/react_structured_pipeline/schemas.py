"""Pydantic schemas for react_structured_pipeline λ-twin."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolDecision(BaseModel):
    tool_name: str = Field(default="search", description="One of: search, lookup")
    query: str = Field(default="", description="Argument string for the tool")
    reasoning: str = Field(default="", description="Why this tool")


class FinalAnswer(BaseModel):
    answer: str = Field(default="", description="Final answer for the user")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
