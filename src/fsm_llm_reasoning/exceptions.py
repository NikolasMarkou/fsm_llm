from __future__ import annotations

"""
Custom exception hierarchy for the reasoning engine.
"""

from typing import Any

from fsm_llm.definitions import FSMError


class ReasoningEngineError(FSMError):
    """Base exception for all reasoning engine errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details=details)


class ReasoningExecutionError(ReasoningEngineError):
    """Error during reasoning execution (FSM push/pop, sub-FSM processing)."""

    def __init__(self, message: str, reasoning_type: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.reasoning_type = reasoning_type


class ReasoningClassificationError(ReasoningEngineError):
    """Error during problem classification."""
    pass


class ReasoningValidationError(ReasoningEngineError):
    """Error during solution validation."""
    pass
