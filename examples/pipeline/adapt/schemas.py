"""Pydantic schemas for adapt λ-twin."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Attempt(BaseModel):
    answer: str = Field(default="", description="Answer to the task")
    rationale: str = Field(default="", description="Brief reasoning")


class Evaluation(BaseModel):
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    feedback: str = Field(default="", description="What was wrong or could improve")
    passed: bool = Field(default=False, description="True if quality >= 0.7")


class Reflection(BaseModel):
    lesson: str = Field(default="", description="Lesson learned for next attempt")
    strategy: str = Field(default="", description="New strategy for retry")


class Final(BaseModel):
    answer: str = Field(default="", description="Refined answer")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
