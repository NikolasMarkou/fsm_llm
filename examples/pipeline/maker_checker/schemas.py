"""Schemas for maker-checker pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Draft(BaseModel):
    text: str = Field(default="")
    rationale: str = Field(default="")


class Review(BaseModel):
    quality_score: float = Field(default=0.0)
    feedback: list[str] = Field(default_factory=list)
    checker_passed: bool = Field(default=False)


class Final(BaseModel):
    text: str = Field(default="")
    revisions_made: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0)
