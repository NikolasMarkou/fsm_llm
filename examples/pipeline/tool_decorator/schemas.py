"""Pydantic schemas for tool_decorator λ-twin.

The original ``examples/agents/tool_decorator`` demos the ``@tool``
decorator + ``Annotated[T, "desc"]`` schema inference at the
*tool-registration* layer. At the λ layer that semantic novelty does
not surface in term shape — the dispatcher routes a flat ``query``.
We preserve the original task (Fahrenheit → Celsius conversion) and
return a structured ``ConversionResult`` from the synth_leaf.

Flat string fields with defaults — D-008/D-011.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolDecision(BaseModel):
    tool_name: str = Field(
        default="convert_temperature",
        description="One of: convert_temperature, convert_distance",
    )
    query: str = Field(
        default="",
        description="Single string spec, e.g. '100 fahrenheit to celsius'",
    )
    reasoning: str = Field(default="", description="Why this tool")


class ConversionResult(BaseModel):
    value: float = Field(default=0.0, description="Numeric converted value")
    unit: str = Field(default="", description="Target unit (e.g. C, F, K, km)")
    explanation: str = Field(
        default="", description="One-line explanation of the conversion"
    )
