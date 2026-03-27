from __future__ import annotations

"""
Pydantic models for the agents package.
"""

from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from fsm_llm.logging import logger

from .constants import Defaults, MetaDefaults


class ToolDefinition(BaseModel):
    """Definition of a tool available to an agent."""

    name: str
    description: str
    parameter_schema: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False

    # Not serialized — runtime only
    execute_fn: Callable[..., Any] | None = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Tool name must be alphanumeric with underscores or hyphens: '{v}'"
            )
        return v


class ToolCall(BaseModel):
    """A request to invoke a tool."""

    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


class ToolResult(BaseModel):
    """Result of a tool execution."""

    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0

    @property
    def summary(self) -> str:
        if self.success:
            text = str(self.result)
            if len(text) > Defaults.MAX_OBSERVATION_LENGTH:
                logger.debug(
                    f"Tool result truncated from {len(text)} to "
                    f"{Defaults.MAX_OBSERVATION_LENGTH} chars"
                )
                return text[: Defaults.MAX_OBSERVATION_LENGTH] + "...[truncated]"
            return text
        return f"Error: {self.error}"


class AgentStep(BaseModel):
    """A single step in the agent's execution trace."""

    iteration: int
    thought: str = ""
    action: str = ""
    observation: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentTrace(BaseModel):
    """Complete trace of an agent's execution."""

    tool_calls: list[ToolCall] = Field(default_factory=list)
    total_iterations: int = 0

    @property
    def tools_used(self) -> list[str]:
        return sorted({tc.tool_name for tc in self.tool_calls})


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    model: str = Defaults.MODEL
    max_iterations: int = Defaults.MAX_ITERATIONS
    timeout_seconds: float = Defaults.TIMEOUT_SECONDS
    temperature: float = Defaults.TEMPERATURE
    max_tokens: int = Defaults.MAX_TOKENS
    """Maximum tokens per LLM response (passed to the LLM provider).

    This controls the maximum length of each individual LLM response,
    NOT the total token budget for the agent run.
    """
    output_schema: type | None = Field(default=None, exclude=True)
    """Optional Pydantic model class for structured agent output.

    When set, the agent's final answer is validated against this schema
    and the parsed model is stored in ``AgentResult.structured_output``.
    """
    transition_config: Any = Field(default=None, exclude=True)
    """Optional ``TransitionEvaluatorConfig`` to tune FSM transition evaluation.

    Controls ambiguity thresholds, minimum confidence, and strict matching.
    When None (default), uses the core library defaults.
    """

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("max_iterations")
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_iterations must be at least 1")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_tokens must be at least 1")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("output_schema")
    @classmethod
    def validate_output_schema(cls, v: type | None) -> type | None:
        if v is not None and not (isinstance(v, type) and issubclass(v, BaseModel)):
            raise ValueError("output_schema must be a Pydantic BaseModel subclass")
        return v

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout_seconds(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("timeout_seconds must be positive")
        return v


class AgentResult(BaseModel):
    """Result of an agent execution."""

    answer: str
    success: bool
    trace: AgentTrace = Field(default_factory=AgentTrace)
    final_context: dict[str, Any] = Field(default_factory=dict)
    structured_output: Any = None
    """Validated Pydantic model instance when ``AgentConfig.output_schema`` is set."""

    @property
    def iterations_used(self) -> int:
        return self.trace.total_iterations

    @property
    def tools_used(self) -> list[str]:
        return self.trace.tools_used

    def __str__(self) -> str:
        if self.structured_output is not None:
            return str(self.structured_output)
        return self.answer


class ApprovalRequest(BaseModel):
    """A request for human approval before an action."""

    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    context_summary: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pattern-specific models
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    """A single step in a plan decomposition."""

    step_id: int
    description: str
    dependencies: list[int] = Field(default_factory=list)
    status: str = "pending"
    result: str = ""


class EvaluationResult(BaseModel):
    """Result from an evaluation pass."""

    passed: bool
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    feedback: str = ""
    criteria_met: list[str] = Field(default_factory=list)


class ReflexionMemory(BaseModel):
    """An episodic memory entry from a Reflexion cycle."""

    episode: int
    task_summary: str = ""
    outcome: str = ""
    reflection: str = ""
    lessons: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DebateRound(BaseModel):
    """A single round in a debate."""

    round_num: int
    proposition: str = ""
    critique: str = ""
    counter_argument: str = ""
    judge_verdict: str = ""


class ChainStep(BaseModel):
    """A step definition for prompt chaining."""

    step_id: str
    name: str
    extraction_instructions: str
    response_instructions: str
    validation_fn: Callable[[dict[str, Any]], bool] | None = Field(
        default=None, exclude=True
    )

    model_config = {"arbitrary_types_allowed": True}


class DecompositionResult(BaseModel):
    """Result of an ADaPT-style task decomposition."""

    subtasks: list[str] = Field(default_factory=list)
    operator: str = "AND"
    depth: int = 0

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v: str) -> str:
        v = v.upper()
        if v not in ("AND", "OR"):
            raise ValueError(f"operator must be 'AND' or 'OR', got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Meta-builder models
# ---------------------------------------------------------------------------


class ArtifactType(str, Enum):
    """Type of artifact the meta-builder can build."""

    FSM = "fsm"
    WORKFLOW = "workflow"
    AGENT = "agent"


class BuildProgress(BaseModel):
    """Progress tracking for artifact building."""

    total_required: int = 0
    completed: int = 0
    missing: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def percentage(self) -> float:
        if self.total_required == 0:
            return 0.0
        return (self.completed / self.total_required) * 100.0

    @property
    def is_complete(self) -> bool:
        return len(self.missing) == 0 and self.total_required > 0


class MetaBuilderConfig(AgentConfig):
    """Configuration for the meta-builder agent."""

    temperature: float = MetaDefaults.TEMPERATURE
    max_tokens: int = MetaDefaults.MAX_TOKENS
    max_turns: int = MetaDefaults.MAX_TURNS
    build_max_iterations: int = MetaDefaults.BUILD_MAX_ITERATIONS
    build_timeout_seconds: float = MetaDefaults.BUILD_TIMEOUT_SECONDS
    build_temperature: float = MetaDefaults.BUILD_TEMPERATURE
    output_path: str | None = None

    @field_validator("max_turns")
    @classmethod
    def validate_max_turns(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_turns must be at least 1")
        return v


class MetaBuilderResult(AgentResult):
    """Result of a meta-builder session."""

    # Override AgentResult defaults for backward compat
    answer: str = ""
    success: bool = True

    artifact_type: ArtifactType = ArtifactType.FSM
    artifact: dict[str, Any] = Field(default_factory=dict)
    artifact_json: str = ""
    is_valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    conversation_turns: int = 0
