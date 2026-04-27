"""Pydantic schemas for the switch_router pipeline example."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DomainAnswer(BaseModel):
    """Per-domain branch output."""

    answer: str = Field(default="", description="Concise answer.")
    domain: str = Field(default="", description="Domain label.")
