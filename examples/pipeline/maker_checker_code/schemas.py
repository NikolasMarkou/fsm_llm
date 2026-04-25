"""Schemas for maker-checker-code pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CodeDraft(BaseModel):
    code: str = Field(default="")
    rationale: str = Field(default="")


class CodeReview(BaseModel):
    quality_score: float = Field(default=0.0)
    issues: list[str] = Field(default_factory=list)
    checker_passed: bool = Field(default=False)


class CodeFinal(BaseModel):
    code: str = Field(default="")
    revisions_made: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0)
