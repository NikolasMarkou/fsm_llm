"""Pydantic schemas for structured_output λ-twin.

Mirrors the original ``examples/agents/structured_output`` example whose
whole point is enforcing a structured Pydantic schema on the agent's
final answer (``MovieRecommendation``). The λ-twin enforces it at the
synth_leaf via ``schema_ref``.

Flat string fields with defaults — D-008/D-011 (no ``dict[str, Any]``
because Ollama's grammar-constrained decoding is unreliable on small
models).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolDecision(BaseModel):
    tool_name: str = Field(default="search_movies", description="One of: search_movies")
    query: str = Field(default="", description="Genre or keyword to search")
    reasoning: str = Field(default="", description="Why this tool")


class MovieRecommendation(BaseModel):
    title: str = Field(default="", description="Movie title")
    year: int = Field(default=2000, description="Release year")
    genre: str = Field(default="", description="Primary genre")
    reason: str = Field(default="", description="Why this movie is recommended")
    rating: float = Field(default=5.0, ge=0.0, le=10.0, description="Rating 0-10")
