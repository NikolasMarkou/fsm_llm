"""Pydantic schemas for the deductive_reasoning pipeline example."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Premises(BaseModel):
    premises: list[str] = Field(default_factory=list)


class Inference(BaseModel):
    inference_steps: list[str] = Field(default_factory=list)
    derived_fact: str = Field(default="")


class Conclusion(BaseModel):
    answer: str = Field(default="")
    is_valid: bool = Field(default=False)
