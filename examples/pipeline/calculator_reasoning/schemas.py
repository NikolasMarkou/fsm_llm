"""Pydantic schemas for the calculator_reasoning pipeline example."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Parsed(BaseModel):
    expression: str = Field(default="")
    operands: list[str] = Field(default_factory=list)


class Computed(BaseModel):
    answer: str = Field(default="")
    work: str = Field(default="")
