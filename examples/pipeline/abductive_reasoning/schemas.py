"""Pydantic schemas for the abductive_reasoning pipeline example."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Observation(BaseModel):
    facts: list[str] = Field(default_factory=list)
    pattern: str = Field(default="")


class Hypothesis(BaseModel):
    candidates: list[str] = Field(default_factory=list)
    primary: str = Field(default="")


class BestExplanation(BaseModel):
    answer: str = Field(default="")
    rationale: str = Field(default="")
