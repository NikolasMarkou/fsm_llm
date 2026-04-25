"""Schemas for evaluator-optimizer pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    text: str = Field(default="")


class Eval(BaseModel):
    score: float = Field(default=0.0)
    passed: bool = Field(default=False)
    feedback: str = Field(default="")


class Refined(BaseModel):
    text: str = Field(default="")
    changes: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0)
