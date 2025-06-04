"""
Pydantic v2 models for the LLM-FSM reasoning engine.

This module defines all data models used throughout the reasoning engine,
with comprehensive validation, type safety, and computed properties.
All models are Pydantic v2 compatible with modern validation patterns.

Author: LLM-FSM Reasoning Engine
Python Version: 3.11+
Dependencies: pydantic v2, datetime, typing
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union, Literal, Set
from enum import Enum

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    computed_field,
    ConfigDict
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

    @computed_field
    @property
    def age_seconds(self) -> float:
        """Calculate age in seconds since creation."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()


class ValidatedModel(TimestampedModel):
    """Base model with enhanced validation capabilities."""

    @model_validator(mode='after')
    def validate_model_consistency(self) -> 'ValidatedModel':
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

    evidence: List[str] = Field(
        default_factory=list,
        description="Supporting evidence or sources for this step",
        max_length=50  # Maximum 50 pieces of evidence
    )

    context_keys_used: Set[str] = Field(
        default_factory=set,
        description="Context keys that were accessed during this step"
    )

    execution_time_ms: Optional[float] = Field(
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
    def validate_evidence_items(cls, v: List[str]) -> List[str]:
        """Ensure all evidence items are non-empty."""
        cleaned = [item.strip() for item in v if item.strip()]
        return cleaned

    @computed_field
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

    @computed_field
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

    checks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Individual validation checks performed"
    )

    issues: List[str] = Field(
        default_factory=list,
        description="Specific issues or problems found during validation"
    )

    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving the solution"
    )

    validation_criteria: List[str] = Field(
        default_factory=list,
        description="Criteria used for validation"
    )

    @computed_field
    @property
    def passed_checks(self) -> int:
        """Number of validation checks that passed."""
        return sum(1 for passed in self.checks.values() if passed)

    @computed_field
    @property
    def total_checks(self) -> int:
        """Total number of validation checks performed."""
        return len(self.checks)

    @computed_field
    @property
    def pass_rate(self) -> float:
        """Percentage of checks that passed."""
        if self.total_checks == 0:
            return 0.0
        return self.passed_checks / self.total_checks

    @computed_field
    @property
    def has_issues(self) -> bool:
        """Whether any issues were found."""
        return len(self.issues) > 0

    @computed_field
    @property
    def validation_summary(self) -> str:
        """Human-readable validation summary."""
        status = "Valid" if self.is_valid else "Invalid"
        return f"{status} ({self.passed_checks}/{self.total_checks} checks passed)"


class ReasoningTrace(ValidatedModel):
    """
    Complete trace of the reasoning process with comprehensive analytics.
    """

    steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sequence of reasoning steps taken"
    )

    reasoning_types_used: Set[str] = Field(
        default_factory=set,
        description="Types of reasoning employed during the process"
    )

    final_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Final confidence in the overall reasoning process"
    )

    execution_time_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total execution time in seconds"
    )

    context_evolution: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="How context changed throughout reasoning"
    )

    decision_points: List[Dict[str, Any]] = Field(
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

    @computed_field
    @property
    def total_steps(self) -> int:
        """Total number of steps in the reasoning process."""
        return len(self.steps)

    @computed_field
    @property
    def unique_states_visited(self) -> List[str]:
        """Get unique states visited during reasoning."""
        states = set()
        for step in self.steps:  # self.steps is List[Dict[str, Any]]
            # Ensure step is a dictionary before trying to access keys
            if isinstance(step, dict):
                from_val = step.get('from')
                if from_val is not None:  # Check if key exists and value is not None
                    # Explicitly convert to string and add if not empty after stripping
                    s_from_val = str(from_val).strip()
                    if s_from_val:
                        states.add(s_from_val)

                to_val = step.get('to')
                if to_val is not None:
                    s_to_val = str(to_val).strip()
                    if s_to_val:
                        states.add(s_to_val)
        return sorted(list(states))  # Convert set to list before sorting

    @computed_field
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

    @computed_field
    @property
    def average_step_time(self) -> Optional[float]:
        """Average time per step if execution time is available."""
        if self.execution_time_seconds is None or self.total_steps == 0:
            return None
        return self.execution_time_seconds / self.total_steps


# ============================================================================
# CLASSIFICATION AND CONTEXT MODELS
# ============================================================================

class ClassificationResult(ValidatedModel):
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

    domain: Union[ProblemDomain, str] = Field(
        default=ProblemDomain.GENERAL,
        description="Identified problem domain"
    )

    alternatives: List[str] = Field(
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

    domain_indicators: List[str] = Field(
        default_factory=list,
        description="Specific indicators that led to domain classification"
    )

    @field_validator('alternatives')
    @classmethod
    def validate_alternatives_unique(cls, v: List[str]) -> List[str]:
        """Ensure alternative reasoning types are unique."""
        return list(dict.fromkeys(v))  # Preserve order while removing duplicates

    @computed_field
    @property
    def has_alternatives(self) -> bool:
        """Whether alternative approaches are available."""
        return len(self.alternatives) > 0

    @computed_field
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

    domain: Optional[Union[ProblemDomain, str]] = Field(
        default=None,
        description="Problem domain if known or suspected"
    )

    constraints: List[str] = Field(
        default_factory=list,
        description="Any constraints, limitations, or requirements"
    )

    initial_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context data and parameters"
    )

    priority: Literal['low', 'medium', 'high', 'urgent'] = Field(
        default='medium',
        description="Priority level for solving this problem"
    )

    expected_solution_type: Optional[str] = Field(
        default=None,
        description="Expected type or format of solution"
    )

    user_preferences: Dict[str, Any] = Field(
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
    def validate_constraints(cls, v: List[str]) -> List[str]:
        """Clean and validate constraints."""
        return [constraint.strip() for constraint in v if constraint.strip()]

    @computed_field
    @property
    def has_constraints(self) -> bool:
        """Whether this problem has any constraints."""
        return len(self.constraints) > 0

    @computed_field
    @property
    def context_size(self) -> int:
        """Size of the problem context in characters."""
        return len(self.problem_statement) + sum(len(str(v)) for v in self.initial_context.values())

    @computed_field
    @property
    def is_high_priority(self) -> bool:
        """Whether this is a high priority problem."""
        return self.priority in ['high', 'urgent']


# ============================================================================
# SOLUTION AND ENGINE STATUS MODELS
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

    execution_time_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total time taken to generate the solution"
    )

    validation_result: Optional[ValidationResult] = Field(
        default=None,
        description="Result of solution validation if performed"
    )

    alternative_solutions: List[str] = Field(
        default_factory=list,
        description="Alternative solutions that were considered"
    )

    key_insights: List[str] = Field(
        default_factory=list,
        description="Key insights discovered during reasoning"
    )

    used_context_keys: Set[str] = Field(
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

    @computed_field
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

    @computed_field
    @property
    def is_high_confidence(self) -> bool:
        """Whether this solution has high confidence."""
        return self.confidence >= 0.8

    @computed_field
    @property
    def reasoning_depth(self) -> int:
        """Depth of reasoning measured by number of steps."""
        return self.trace.total_steps

    @computed_field
    @property
    def has_alternatives(self) -> bool:
        """Whether alternative solutions were generated."""
        return len(self.alternative_solutions) > 0

    @computed_field
    @property
    def is_validated(self) -> bool:
        """Whether the solution has been validated."""
        return self.validation_result is not None

    @computed_field
    @property
    def solution_quality_summary(self) -> str:
        """Human-readable summary of solution quality."""
        validated_text = " (validated)" if self.is_validated else ""
        alt_text = f" with {len(self.alternative_solutions)} alternatives" if self.has_alternatives else ""
        return f"{self.confidence_level.value} confidence solution{validated_text}{alt_text}"


class EngineStatus(ValidatedModel):
    """
    Current status and capabilities of the reasoning engine.
    """

    is_ready: bool = Field(
        ...,
        description="Whether the engine is ready to process problems"
    )

    active_conversations: int = Field(
        default=0,
        ge=0,
        description="Number of currently active reasoning conversations"
    )

    loaded_fsms: Set[str] = Field(
        default_factory=set,
        description="Set of loaded FSM types available for reasoning"
    )

    model: str = Field(
        ...,
        min_length=1,
        description="LLM model currently being used"
    )

    version: str = Field(
        default="unknown",
        description="Version of the reasoning engine"
    )

    supported_reasoning_types: Set[str] = Field(
        default_factory=set,
        description="All supported reasoning types"
    )

    performance_metrics: Dict[str, Union[int, float]] = Field(
        default_factory=dict,
        description="Performance metrics and statistics"
    )

    configuration: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current engine configuration"
    )

    @computed_field
    @property
    def has_active_conversations(self) -> bool:
        """Whether there are active conversations."""
        return self.active_conversations > 0

    @computed_field
    @property
    def fsm_count(self) -> int:
        """Number of loaded FSMs."""
        return len(self.loaded_fsms)

    @computed_field
    @property
    def reasoning_type_count(self) -> int:
        """Number of supported reasoning types."""
        return len(self.supported_reasoning_types)

    @computed_field
    @property
    def status_summary(self) -> str:
        """Human-readable status summary."""
        status = "Ready" if self.is_ready else "Not Ready"
        active_text = f" ({self.active_conversations} active)" if self.has_active_conversations else ""
        return f"{status} - {self.fsm_count} FSMs loaded{active_text}"


class ContextSnapshot(ValidatedModel):
    """
    Point-in-time snapshot of reasoning context with analytics.
    """

    state: str = Field(
        ...,
        min_length=1,
        description="Current state of the reasoning process"
    )

    context_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Actual context data at this snapshot"
    )

    important_keys: Set[str] = Field(
        default_factory=set,
        description="Keys that are particularly important at this point"
    )

    memory_usage_bytes: Optional[int] = Field(
        default=None,
        ge=0,
        description="Estimated memory usage of this context"
    )

    @computed_field
    @property
    def size_characters(self) -> int:
        """Context size in characters."""
        import json
        try:
            return len(json.dumps(self.context_data))
        except (TypeError, ValueError):
            return len(str(self.context_data))

    @computed_field
    @property
    def key_count(self) -> int:
        """Number of context keys."""
        return len(self.context_data)

    @computed_field
    @property
    def has_important_keys(self) -> bool:
        """Whether any keys are marked as important."""
        return len(self.important_keys) > 0

    @computed_field
    @property
    def context_density(self) -> float:
        """Ratio of important keys to total keys."""
        if self.key_count == 0:
            return 0.0
        return len(self.important_keys) / self.key_count

    @model_validator(mode='after')
    def validate_important_keys_exist(self) -> 'ContextSnapshot':
        """Ensure important keys actually exist in context data."""
        if self.important_keys:
            missing_keys = self.important_keys - set(self.context_data.keys())
            if missing_keys:
                raise ValueError(f"Important keys not found in context: {missing_keys}")
        return self


# ============================================================================
# UTILITY MODELS
# ============================================================================

class ErrorReport(ValidatedModel):
    """
    Comprehensive error reporting model.
    """

    error_type: str = Field(..., description="Type of error that occurred")
    message: str = Field(..., description="Human-readable error message")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context when error occurred")
    stack_trace: Optional[str] = Field(default=None, description="Stack trace if available")
    recovery_suggestions: List[str] = Field(default_factory=list, description="Suggestions for recovery")
    severity: Literal['low', 'medium', 'high', 'critical'] = Field(default='medium', description="Error severity")

    @computed_field
    @property
    def is_critical(self) -> bool:
        """Whether this is a critical error."""
        return self.severity == 'critical'


class PerformanceMetrics(ValidatedModel):
    """
    Performance metrics for the reasoning engine.
    """

    total_problems_solved: int = Field(default=0, ge=0)
    average_solution_time: float = Field(default=0.0, ge=0.0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    average_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_type_usage: Dict[str, int] = Field(default_factory=dict)

    @computed_field
    @property
    def has_performance_data(self) -> bool:
        """Whether any performance data is available."""
        return self.total_problems_solved > 0

    @computed_field
    @property
    def most_used_reasoning_type(self) -> Optional[str]:
        """Most frequently used reasoning type."""
        if not self.reasoning_type_usage:
            return None
        return max(self.reasoning_type_usage.items(), key=lambda x: x[1])[0]


# ============================================================================
# UTILITIES
# ============================================================================

def get_model_by_name(model_name: str) -> Optional[type[BaseModel]]:
    """
    Get a model class by its name.

    Args:
        model_name: Name of the model class

    Returns:
        Model class if found, None otherwise
    """
    import sys
    current_module = sys.modules[__name__]
    return getattr(current_module, model_name, None)


def validate_model_data(model_class: type[BaseModel], data: Dict[str, Any]) -> BaseModel:
    """
    Validate data against a model class with enhanced error reporting.

    Args:
        model_class: The Pydantic model class
        data: Data to validate

    Returns:
        Validated model instance

    Raises:
        ValueError: If validation fails
    """
    try:
        return model_class.model_validate(data)
    except Exception as e:
        raise ValueError(f"Validation failed for {model_class.__name__}: {e}")