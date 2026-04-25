"""Pydantic schemas for the debate pipeline example."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """A single proposer/critic turn."""

    argument: str = Field(default="", description="The argument text.")
    key_point: str = Field(default="", description="The single strongest claim.")


class Verdict(BaseModel):
    """The judge's verdict at the end of a round."""

    verdict: str = Field(default="", description="Synthesized conclusion.")
    winner: str = Field(
        default="balanced",
        description="One of: 'proposer', 'critic', 'balanced'.",
    )
    confidence: float = Field(default=0.0, description="0..1 confidence.")
