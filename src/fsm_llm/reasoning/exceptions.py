"""
Custom exception hierarchy for the reasoning engine.
"""

from __future__ import annotations

from fsm_llm.definitions import FSMError


class ReasoningError(FSMError):
    """Base exception for all reasoning engine errors."""


class ReasoningExecutionError(ReasoningError):
    """Error during reasoning execution (FSM push/pop, sub-FSM processing)."""

    def __init__(self, message: str, reasoning_type: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.reasoning_type = reasoning_type


class ReasoningClassificationError(ReasoningError):
    """Error during problem classification."""

    pass
