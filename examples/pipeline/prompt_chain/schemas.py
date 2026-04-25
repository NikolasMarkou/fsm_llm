"""Pydantic schemas for the prompt-chain pipeline example.

Each leaf in ``run.py`` references one of these classes via a dotted
``schema_ref`` string. The λ-executor's ``_resolve_schema`` imports the
module and validates the LLM's JSON output against the BaseModel.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchOut(BaseModel):
    """Output of the research step."""

    key_points: list[str] = Field(
        default_factory=list,
        description="3-5 key facts or arguments about the topic.",
    )
    sources_mentioned: list[str] = Field(
        default_factory=list,
        description="Any sources or evidence cited.",
    )
    main_thesis: str = Field(
        default="",
        description="One-sentence summary of the core argument.",
    )


class DraftOut(BaseModel):
    """Output of the draft step."""

    outline_sections: list[str] = Field(
        default_factory=list,
        description="Section titles for the draft.",
    )
    word_count_estimate: int = Field(
        default=0,
        description="Estimated word count of the draft.",
    )
    draft_text: str = Field(
        default="",
        description="The full draft text.",
    )


class PolishOut(BaseModel):
    """Output of the polish step."""

    final_text: str = Field(
        default="",
        description="The polished final text.",
    )
    improvements_made: list[str] = Field(
        default_factory=list,
        description="List of improvements over the draft.",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score in 0..1 for the quality.",
    )
