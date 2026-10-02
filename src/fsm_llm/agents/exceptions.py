"""
Exception hierarchy for the agents package.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.definitions import BuildError, FSMError


class AgentError(FSMError):
    """Base exception for all agent-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details=details)


class ToolExecutionError(AgentError):
    """Error during tool execution."""

    def __init__(self, message: str, tool_name: str | None = None, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name


class ToolNotFoundError(AgentError):
    """Requested tool does not exist in the registry."""

    def __init__(self, tool_name: str):
        super().__init__(f"Tool not found: {tool_name}")
        self.tool_name = tool_name


class BudgetExhaustedError(AgentError):
    """Agent exceeded its iteration/token/time budget."""

    def __init__(self, budget_type: str, limit: int | float, detail: str = ""):
        suffix = f" ({detail})" if detail else ""
        super().__init__(
            f"Agent budget exhausted: {budget_type} limit ({limit}) reached{suffix}"
        )
        self.budget_type = budget_type
        self.limit = limit


class ApprovalDeniedError(AgentError):
    """Human denied approval for an agent action."""

    def __init__(self, action_description: str):
        super().__init__(f"Approval denied for action: {action_description}")
        self.action_description = action_description


class AgentTimeoutError(AgentError):
    """Agent exceeded its time budget."""

    def __init__(self, timeout_seconds: float):
        super().__init__(f"Agent timed out after {timeout_seconds:.1f} seconds")
        self.timeout_seconds = timeout_seconds


class EvaluationError(AgentError):
    """Error during evaluation (Evaluator-Optimizer, Maker-Checker)."""

    def __init__(self, message: str, evaluator: str | None = None, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.evaluator = evaluator


# ---------------------------------------------------------------------------
# Meta-builder exceptions
# ---------------------------------------------------------------------------


class MetaBuilderError(AgentError):
    """Base exception for all meta-builder-related errors."""

    pass


class BuilderError(MetaBuilderError, BuildError):
    """Error during artifact building (invalid state/step/tool operations).

    Also a core ``BuildError`` so one ``except BuildError`` covers every
    builder; ``.errors`` is ``[message]``, ``.action`` and ``.details`` as before.
    """

    def __init__(self, message: str, action: str | None = None, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.action = action


class MetaValidationError(MetaBuilderError):
    """Error during artifact validation."""

    def __init__(
        self,
        message: str,
        errors: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.errors = errors or []


class OutputError(MetaBuilderError):
    """Error during artifact output/serialization."""

    def __init__(self, message: str, path: str | None = None, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.path = path
