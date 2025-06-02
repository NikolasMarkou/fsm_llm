"""
Pydantic models for the reasoning engine.
Enhanced with validation and better structure.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class ReasoningStep(BaseModel):
    """Represents a single step in the reasoning process."""
    step_type: str = Field(..., description="Type of reasoning step")
    content: str = Field(..., description="Content of the reasoning")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence level")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    timestamp: datetime = Field(default_factory=datetime.now, description="When step occurred")

    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v


class ValidationResult(BaseModel):
    """Result of solution validation."""
    is_valid: bool = Field(..., description="Whether solution is valid")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in validation")
    checks: Dict[str, bool] = Field(default_factory=dict, description="Individual checks")
    issues: List[str] = Field(default_factory=list, description="Issues found")

    @property
    def passed_checks(self) -> int:
        """Number of passed validation checks."""
        return sum(1 for check in self.checks.values() if check)

    @property
    def total_checks(self) -> int:
        """Total number of validation checks."""
        return len(self.checks)


class ReasoningTrace(BaseModel):
    """Complete trace of reasoning process."""
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Reasoning steps")
    total_steps: int = Field(0, description="Total number of steps")
    reasoning_types_used: List[str] = Field(default_factory=list, description="Types used")
    final_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Final confidence")

    @validator('total_steps', pre=True, always=True)
    def set_total_steps(cls, v, values):
        """Ensure total_steps matches steps length."""
        steps = values.get('steps', [])
        return len(steps) if steps else v

    @property
    def unique_states(self) -> List[str]:
        """Get unique states visited."""
        states = set()
        for step in self.steps:
            states.add(step.get('from', ''))
            states.add(step.get('to', ''))
        return sorted(filter(None, states))


class ClassificationResult(BaseModel):
    """Result of problem classification."""
    recommended_type: str = Field(..., description="Recommended reasoning type")
    justification: str = Field(..., description="Why this type was chosen")
    domain: str = Field(..., description="Problem domain")
    alternatives: List[str] = Field(default_factory=list, description="Alternative approaches")
    confidence: str = Field("medium", description="Confidence level")

    @validator('confidence')
    def validate_confidence(cls, v):
        allowed = ['low', 'medium', 'high']
        if v not in allowed:
            raise ValueError(f'Confidence must be one of {allowed}')
        return v


class ProblemContext(BaseModel):
    """Context for a problem to be solved."""
    problem_statement: str = Field(..., description="The problem to solve")
    domain: Optional[str] = Field(None, description="Problem domain if known")
    constraints: Optional[List[str]] = Field(None, description="Any constraints")
    initial_context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    @validator('problem_statement')
    def problem_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Problem statement cannot be empty')
        return v.strip()


class SolutionResult(BaseModel):
    """Result of problem solving."""
    solution: str = Field(..., description="The solution found")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Solution confidence")
    reasoning_summary: str = Field(..., description="Summary of reasoning process")
    trace: ReasoningTrace = Field(..., description="Full reasoning trace")
    execution_time: Optional[float] = Field(None, description="Time taken in seconds")

    @property
    def is_high_confidence(self) -> bool:
        """Check if solution has high confidence."""
        return self.confidence >= 0.8

    @property
    def reasoning_depth(self) -> int:
        """Get the depth of reasoning (number of steps)."""
        return self.trace.total_steps


class EngineStatus(BaseModel):
    """Status of the reasoning engine."""
    is_ready: bool = Field(..., description="Whether engine is ready")
    active_conversations: int = Field(0, description="Number of active conversations")
    loaded_fsms: List[str] = Field(default_factory=list, description="Loaded FSM types")
    model: str = Field(..., description="LLM model being used")

    @property
    def has_active_conversations(self) -> bool:
        """Check if there are active conversations."""
        return self.active_conversations > 0


class ContextSnapshot(BaseModel):
    """Snapshot of context at a point in time."""
    timestamp: datetime = Field(default_factory=datetime.now)
    state: str = Field(..., description="Current state")
    size: int = Field(..., description="Context size in characters")
    key_count: int = Field(..., description="Number of context keys")
    important_keys: List[str] = Field(default_factory=list, description="Important context keys")

    @validator('size')
    def validate_size(cls, v):
        if v < 0:
            raise ValueError('Size cannot be negative')
        return v