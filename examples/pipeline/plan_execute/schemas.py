"""Pydantic schemas for plan_execute λ-twin."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    label: str = Field(default="#E1", description="Evidence label")
    tool_name: str = Field(default="search", description="Tool name")
    args: str = Field(default="", description="Tool argument string")


class Plan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list, description="2-3 plan steps")


class FinalAnswer(BaseModel):
    answer: str = Field(default="", description="Final answer using evidence")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
