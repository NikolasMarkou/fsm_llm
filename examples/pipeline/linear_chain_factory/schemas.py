"""Pydantic schemas for the linear_chain_factory pipeline example."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FactsOut(BaseModel):
    """Step 1 output — extract key facts."""

    facts: list[str] = Field(
        default_factory=list,
        description="3-5 key facts about the topic.",
    )
    domain: str = Field(default="", description="Domain or area.")


class OutlineOut(BaseModel):
    """Step 2 output — turn facts into an outline."""

    sections: list[str] = Field(
        default_factory=list,
        description="3-5 outline section titles.",
    )
    thesis: str = Field(default="", description="One-sentence thesis.")


class SummaryOut(BaseModel):
    """Step 3 output — final summary text."""

    summary: str = Field(default="", description="Final summary text.")
    confidence: float = Field(default=0.0, description="Quality 0..1.")
