from __future__ import annotations

"""
Pydantic models for the agents package.
"""

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .constants import Defaults


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
        if not v or not v.replace("_", "").isalnum():
            raise ValueError(f"Tool name must be alphanumeric with underscores: '{v}'")
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

    steps: list[AgentStep] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    total_iterations: int = 0

    @property
    def tools_used(self) -> list[str]:
        return list({tc.tool_name for tc in self.tool_calls})


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    model: str = Defaults.MODEL
    max_iterations: int = Defaults.MAX_ITERATIONS
    timeout_seconds: float = Defaults.TIMEOUT_SECONDS
    temperature: float = Defaults.TEMPERATURE
    max_tokens: int = Defaults.MAX_TOKENS

    @field_validator("max_iterations")
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_iterations must be at least 1")
        return v


class AgentResult(BaseModel):
    """Result of an agent execution."""

    answer: str
    success: bool
    trace: AgentTrace = Field(default_factory=AgentTrace)
    final_context: dict[str, Any] = Field(default_factory=dict)

    @property
    def iterations_used(self) -> int:
        return self.trace.total_iterations

    @property
    def tools_used(self) -> list[str]:
        return self.trace.tools_used


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
    score: float = 0.0
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
        if v not in ("AND", "OR"):
            raise ValueError(f"operator must be 'AND' or 'OR', got '{v}'")
        return v
