"""
Custom exception hierarchy for the reasoning engine.
"""

from __future__ import annotations

from fsm_llm.core.definitions import FSMError


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


class ReasoningRegistryError(ReasoningError, ValueError):
    """The registry given to ``ReasoningEngine(registry=)`` cannot serve a solve.

    Raised at construction when its eligible strategies (``selection.strategy_choices``) hold no
    strategy named ``analytical`` (the fallback of every solve) or fewer than two (the choice is a
    closed classification, which needs two intents).
    """
