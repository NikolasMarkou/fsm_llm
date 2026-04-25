"""Pydantic schemas for react_search λ-twin.

Note: Ollama's grammar-constrained decoding for ``dict[str, Any]``
fields is unreliable on small models (qwen3.5:4b). We use flat string
fields and split args downstream — this matches the shape of M4
maker_checker which works reliably.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolDecision(BaseModel):
    tool_name: str = Field(default="search", description="One of: search, calculate, lookup")
    query: str = Field(default="", description="Single string argument for the tool")
    reasoning: str = Field(default="", description="Why this tool")


class FinalAnswer(BaseModel):
    answer: str = Field(default="", description="Answer for the user")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
