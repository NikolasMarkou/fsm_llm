"""Pydantic schemas for the fix_iterative_refine pipeline example."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Draft(BaseModel):
    text: str = Field(default="")
    iteration: int = Field(default=0)


class Critique(BaseModel):
    text: str = Field(default="")
    score: float = Field(default=0.0)
    issues: list[str] = Field(default_factory=list)
