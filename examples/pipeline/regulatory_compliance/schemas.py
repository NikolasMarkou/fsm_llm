"""Schemas for regulatory_compliance pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Assessment(BaseModel):
    summary: str = Field(default="")
    findings: list[str] = Field(default_factory=list)
    risk_level: str = Field(default="medium")


class Review(BaseModel):
    quality_score: float = Field(default=0.0)
    issues: list[str] = Field(default_factory=list)
    checker_passed: bool = Field(default=False)


class FinalReport(BaseModel):
    summary: str = Field(default="")
    findings: list[str] = Field(default_factory=list)
    risk_level: str = Field(default="medium")
    revisions_made: list[str] = Field(default_factory=list)
