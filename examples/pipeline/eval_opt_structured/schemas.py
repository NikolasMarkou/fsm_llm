"""Schemas for eval_opt_structured pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Recipe(BaseModel):
    name: str = Field(default="")
    cuisine: str = Field(default="")
    prep_time_minutes: int = Field(default=0)
    cook_time_minutes: int = Field(default=0)
    servings: int = Field(default=0)
    difficulty: str = Field(default="Medium")
    ingredients: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    nutritional_notes: str = Field(default="")


class Eval(BaseModel):
    score: float = Field(default=0.0)
    passed: bool = Field(default=False)
    feedback: str = Field(default="")


class RefinedRecipe(Recipe):
    changes: list[str] = Field(default_factory=list)
