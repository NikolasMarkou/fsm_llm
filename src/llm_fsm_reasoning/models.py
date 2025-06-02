"""
Pydantic models for the reasoning engine.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ReasoningStep(BaseModel):
    """Represents a single step in the reasoning process."""
    step_type: str = Field(..., description="Type of reasoning step")
    content: str = Field(..., description="Content of the reasoning")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence level")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    timestamp: datetime = Field(default_factory=datetime.now, description="When step occurred")


class ValidationResult(BaseModel):
    """Result of solution validation."""
    is_valid: bool = Field(..., description="Whether solution is valid")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in validation")
    checks: Dict[str, bool] = Field(default_factory=dict, description="Individual checks")
    issues: List[str] = Field(default_factory=list, description="Issues found")


class ReasoningTrace(BaseModel):
    """Complete trace of reasoning process."""
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Reasoning steps")
    total_steps: int = Field(0, description="Total number of steps")
    reasoning_types_used: List[str] = Field(default_factory=list, description="Types used")
    final_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Final confidence")


class ClassificationResult(BaseModel):
    """Result of problem classification."""
    recommended_type: str = Field(..., description="Recommended reasoning type")
    justification: str = Field(..., description="Why this type was chosen")
    domain: str = Field(..., description="Problem domain")
    alternatives: List[str] = Field(default_factory=list, description="Alternative approaches")
    confidence: str = Field("medium", description="Confidence level")