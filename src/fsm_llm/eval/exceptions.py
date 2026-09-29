"""
Exception hierarchy for the fsm_llm.eval package.

Rooted at ``fsm_llm.definitions.FSMError`` so a caller catching the framework's
base error also catches every evaluation failure.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.definitions import FSMError


class EvalError(FSMError):
    """Base exception for all evaluation errors (also: unwritable output root)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details=details)


class EvalConfigError(EvalError):
    """An evaluation config is malformed, has unknown keys, or bad values."""


class EvalDatasetError(EvalError):
    """A conversation dataset is missing, malformed, or has duplicate case ids."""
