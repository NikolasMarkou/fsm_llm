"""Schemas for pipeline-review."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Outline(BaseModel):
    sections: list[str] = Field(default_factory=list)
    summary: str = Field(default="")


class Draft(BaseModel):
    text: str = Field(default="")
    word_count: int = Field(default=0)


class Polished(BaseModel):
    text: str = Field(default="")
    improvements: list[str] = Field(default_factory=list)


class Review(BaseModel):
    quality_score: float = Field(default=0.0)
    feedback: list[str] = Field(default_factory=list)
    checker_passed: bool = Field(default=False)


class Final(BaseModel):
    text: str = Field(default="")
    review: Review = Field(default_factory=Review)
