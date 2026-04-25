"""Pydantic schemas for the self-consistency pipeline example.

Each leaf in ``run.py`` references one of these classes via a dotted
``schema_ref`` string, e.g. ``"examples.pipeline.self_consistency.schemas.AnswerOut"``.
The λ-executor's ``_resolve_schema`` imports the module and validates the
LLM's JSON output against the BaseModel.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerOut(BaseModel):
    """A single sampled answer to the factual question."""

    answer: str = Field(
        ...,
        description="The answer to the question (a short noun phrase, e.g. a city name).",
    )
    confidence: float | None = Field(
        default=None,
        description="Optional self-reported confidence in 0..1.",
    )
