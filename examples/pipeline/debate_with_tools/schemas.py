"""Schemas for debate_with_tools pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Turn(BaseModel):
    argument: str = Field(default="")
    key_point: str = Field(default="")


class Verdict(BaseModel):
    verdict: str = Field(default="")
    winner: str = Field(default="balanced")
    confidence: float = Field(default=0.0)
