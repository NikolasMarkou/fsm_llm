from __future__ import annotations

"""
Pydantic v2 models for the FSM-LLM reasoning engine.

This module defines all data models used throughout the reasoning engine,
with comprehensive validation, type safety, and computed properties.
All models are Pydantic v2 compatible with modern validation patterns.

Author: FSM-LLM Reasoning Engine
Python Version: 3.10+
Dependencies: pydantic v2, datetime, typing
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ConfidenceLevel(str, Enum):
    """Confidence levels for various assessments."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ReasoningStepType(str, Enum):
    """Types of reasoning steps in the process."""
    ANALYSIS = "analysis"
    DEDUCTION = "deduction"
    INDUCTION = "induction"
    CREATIVE = "creative"
    CRITICAL = "critical"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    TRANSITION = "transition"


class ProblemDomain(str, Enum):
    """Common problem domains."""
    MATHEMATICS = "mathematics"
    LOGIC = "logic"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    SCIENTIFIC = "scientific"
    BUSINESS = "business"
    TECHNICAL = "technical"
    GENERAL = "general"


# ============================================================================
# BASE MODELS
# ============================================================================

class TimestampedModel(BaseModel):
    """Base model with automatic timestamping."""

    model_config = ConfigDict(
        # Enable arbitrary types for datetime
        arbitrary_types_allowed=True,
        # Use enum values in serialization
        use_enum_values=True,
        # Validate assignment
        validate_assignment=True,
        # Extra fields forbidden by default
        extra='ignore',
        # Enable JSON schema generation
        json_schema_extra={
            "examples": []
        }
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this model was created",
        json_schema_extra={"example": "2024-01-01T12:00:00Z"}
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def age_seconds(self) -> float:
        """Calculate age in seconds since creation."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()


class ValidatedModel(TimestampedModel):
    """Base model with enhanced validation capabilities."""

    @model_validator(mode='after')
    def validate_model_consistency(self) -> ValidatedModel:
        """Override in subclasses for cross-field validation."""
        return self


# ============================================================================
# REASONING PROCESS MODELS
# ============================================================================

class ReasoningStep(ValidatedModel):
    """
    Represents a single step in the reasoning process with comprehensive tracking.
    """

    step_type: ReasoningStepType = Field(
        ...,
        description="Type of reasoning step being performed"
    )

    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Detailed content of the reasoning step"
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence level in this step (0.0-1.0)"
    )

    evidence: list[str] = Field(
        default_factory=list,
        description="Supporting evidence or sources for this step",
        max_length=50  # Maximum 50 pieces of evidence
    )

    context_keys_used: set[str] = Field(
        default_factory=set,
        description="Context keys that were accessed during this step"
    )

    execution_time_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Time taken to execute this step in milliseconds"
    )

    @field_validator('content')
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensure content is not just whitespace."""
        if not v.strip():
            raise ValueError('Content cannot be empty or only whitespace')
        return v.strip()

    @field_validator('evidence')
    def validate_evidence_items(cls, v: list[str]) -> list[str]:
        """Ensure all evidence items are non-empty."""
        cleaned = [item.strip() for item in v if item.strip()]
        return cleaned

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get categorical confidence level."""
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_evidence(self) -> bool:
        """Check if this step has supporting evidence."""
        return len(self.evidence) > 0


class ValidationResult(ValidatedModel):
    """
    Comprehensive result of solution validation with detailed feedback.
    """

    is_valid: bool = Field(
        ...,
        description="Overall validation result - whether solution is valid"
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the validation assessment"
    )

    checks: dict[str, bool] = Field(
        default_factory=dict,
        description="Individual validation checks performed"
    )

    issues: list[str] = Field(
        default_factory=list,
        description="Specific issues or problems found during validation"
    )

    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommendations for improving the solution"
    )

    validation_criteria: list[str] = Field(
        default_factory=list,
        description="Criteria used for validation"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed_checks(self) -> int:
        """Number of validation checks that passed."""
        return sum(1 for passed in self.checks.values() if passed)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_checks(self) -> int:
        """Total number of validation checks performed."""
        return len(self.checks)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pass_rate(self) -> float:
        """Percentage of checks that passed."""
        if self.total_checks == 0:
            return 0.0
        return self.passed_checks / self.total_checks

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_issues(self) -> bool:
        """Whether any issues were found."""
        return len(self.issues) > 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def validation_summary(self) -> str:
        """Human-readable validation summary."""
        status = "Valid" if self.is_valid else "Invalid"
        return f"{status} ({self.passed_checks}/{self.total_checks} checks passed)"


class ReasoningTrace(ValidatedModel):
    """
    Complete trace of the reasoning process with comprehensive analytics.
    """

    steps: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Sequence of reasoning steps taken"
    )

    reasoning_types_used: set[str] = Field(
        default_factory=set,
        description="Types of reasoning employed during the process"
    )

    final_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Final confidence in the overall reasoning process"
    )

    execution_time_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Total execution time in seconds"
    )

    context_evolution: list[dict[str, Any]] = Field(
        default_factory=list,
        description="How context changed throughout reasoning"
    )

    decision_points: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Key decision points in the reasoning process"
    )

    @model_validator(mode='before')
    @classmethod
    def remove_computed_fields_from_input(cls, data: Any) -> Any:
        """Remove computed fields from input data to prevent validation errors."""
        if isinstance(data, dict):
            # Remove computed field names that might be passed as input
            computed_fields = {'total_steps', 'unique_states_visited', 'reasoning_complexity', 'average_step_time'}
            return {k: v for k, v in data.items() if k not in computed_fields}
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_steps(self) -> int:
        """Total number of steps in the reasoning process."""
        return len(self.steps)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unique_states_visited(self) -> list[str]:
        """Get unique states visited during reasoning."""
        states = set()
        for step in self.steps:
            if isinstance(step, dict):
                for key in ('from', 'to'):
                    val = step.get(key)
                    if val is not None:
                        s_val = str(val).strip()
                        if s_val:
                            states.add(s_val)
        return sorted(states)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def reasoning_complexity(self) -> Literal['simple', 'moderate', 'complex', 'highly_complex']:
        """Assess complexity based on steps and reasoning types."""
        if self.total_steps < 5 and len(self.reasoning_types_used) <= 1:
            return 'simple'
        elif self.total_steps < 10 and len(self.reasoning_types_used) <= 2:
            return 'moderate'
        elif self.total_steps < 20 and len(self.reasoning_types_used) <= 3:
            return 'complex'
        else:
            return 'highly_complex'

    @computed_field  # type: ignore[prop-decorator]
    @property
    def average_step_time(self) -> float | None:
        """Average time per step if execution time is available."""
        if self.execution_time_seconds is None or self.total_steps == 0:
            return None
        return self.execution_time_seconds / self.total_steps


# ============================================================================
# CLASSIFICATION AND CONTEXT MODELS
# ============================================================================

class ReasoningClassificationResult(ValidatedModel):
    """
    Result of problem classification with comprehensive analysis.
    """

    recommended_type: str = Field(
        ...,
        min_length=1,
        description="Primary recommended reasoning type"
    )

    justification: str = Field(
        ...,
        min_length=0,
        description="Detailed justification for the recommendation"
    )

    domain: ProblemDomain | str = Field(
        default=ProblemDomain.GENERAL,
        description="Identified problem domain"
    )

    alternatives: list[str] = Field(
        default_factory=list,
        description="Alternative reasoning approaches that could work"
    )

    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Confidence in the classification"
    )

    complexity_assessment: Literal['low', 'medium', 'high', 'very_high'] = Field(
        default='medium',
        description="Assessment of problem complexity"
    )

    domain_indicators: list[str] = Field(
        default_factory=list,
        description="Specific indicators that led to domain classification"
    )

    @field_validator('alternatives')
    @classmethod
    def validate_alternatives_unique(cls, v: list[str]) -> list[str]:
        """Ensure alternative reasoning types are unique."""
        return list(dict.fromkeys(v))  # Preserve order while removing duplicates

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_alternatives(self) -> bool:
        """Whether alternative approaches are available."""
        return len(self.alternatives) > 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def classification_summary(self) -> str:
        """Human-readable classification summary."""
        alt_text = f" (with {len(self.alternatives)} alternatives)" if self.has_alternatives else ""
        return f"{self.recommended_type} reasoning for {self.domain} domain{alt_text}"


class ProblemContext(ValidatedModel):
    """
    Comprehensive context for a problem to be solved.
    """

    problem_statement: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The problem statement to be solved"
    )

    domain: ProblemDomain | str | None = Field(
        default=None,
        description="Problem domain if known or suspected"
    )

    constraints: list[str] = Field(
        default_factory=list,
        description="Any constraints, limitations, or requirements"
    )

    initial_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context data and parameters"
    )

    priority: Literal['low', 'medium', 'high', 'urgent'] = Field(
        default='medium',
        description="Priority level for solving this problem"
    )

    expected_solution_type: str | None = Field(
        default=None,
        description="Expected type or format of solution"
    )

    user_preferences: dict[str, Any] = Field(
        default_factory=dict,
        description="User preferences for reasoning approach"
    )

    @field_validator('problem_statement')
    @classmethod
    def validate_problem_not_empty(cls, v: str) -> str:
        """Ensure problem statement is meaningful."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError('Problem statement cannot be empty or only whitespace')
        if len(cleaned) < 3:
            raise ValueError('Problem statement must be at least 3 characters long')
        return cleaned

    @field_validator('constraints')
    @classmethod
    def validate_constraints(cls, v: list[str]) -> list[str]:
        """Clean and validate constraints."""
        return [constraint.strip() for constraint in v if constraint.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_constraints(self) -> bool:
        """Whether this problem has any constraints."""
        return len(self.constraints) > 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def context_size(self) -> int:
        """Size of the problem context in characters."""
        return len(self.problem_statement) + sum(len(str(v)) for v in self.initial_context.values())

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_high_priority(self) -> bool:
        """Whether this is a high priority problem."""
        return self.priority in ['high', 'urgent']


# ============================================================================
# SOLUTION MODELS
# ============================================================================

class SolutionResult(ValidatedModel):
    """
    Comprehensive result of problem solving with detailed analytics.
    """

    solution: str = Field(
        ...,
        min_length=1,
        description="The generated solution to the problem"
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the solution quality"
    )

    reasoning_summary: str = Field(
        ...,
        min_length=10,
        description="Summary of the reasoning process used"
    )

    trace: ReasoningTrace = Field(
        ...,
        description="Complete trace of the reasoning process"
    )

    execution_time_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Total time taken to generate the solution"
    )

    validation_result: ValidationResult | None = Field(
        default=None,
        description="Result of solution validation if performed"
    )

    alternative_solutions: list[str] = Field(
        default_factory=list,
        description="Alternative solutions that were considered"
    )

    key_insights: list[str] = Field(
        default_factory=list,
        description="Key insights discovered during reasoning"
    )

    used_context_keys: set[str] = Field(
        default_factory=set,
        description="Context keys that were used in the solution"
    )

    @field_validator('solution')
    @classmethod
    def validate_solution_not_empty(cls, v: str) -> str:
        """Ensure solution is meaningful."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError('Solution cannot be empty or only whitespace')
        return cleaned

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Categorical confidence level."""
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_high_confidence(self) -> bool:
        """Whether this solution has high confidence."""
        return self.confidence >= 0.8

    @computed_field  # type: ignore[prop-decorator]
    @property
    def reasoning_depth(self) -> int:
        """Depth of reasoning measured by number of steps."""
        return self.trace.total_steps

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_alternatives(self) -> bool:
        """Whether alternative solutions were generated."""
        return len(self.alternative_solutions) > 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_validated(self) -> bool:
        """Whether the solution has been validated."""
        return self.validation_result is not None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def solution_quality_summary(self) -> str:
        """Human-readable summary of solution quality."""
        validated_text = " (validated)" if self.is_validated else ""
        alt_text = f" with {len(self.alternative_solutions)} alternatives" if self.has_alternatives else ""
        return f"{self.confidence_level.value} confidence solution{validated_text}{alt_text}"


