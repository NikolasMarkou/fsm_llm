"""
Exception hierarchy for the fsm_llm.eval package.

Rooted at ``fsm_llm.definitions.FSMError`` so a caller catching the framework's
base error also catches every evaluation failure.
"""

from __future__ import annotations

from fsm_llm.definitions import FSMError


class EvalError(FSMError):
    """Base exception for all evaluation errors (also: unwritable output root)."""


class EvalConfigError(EvalError):
    """An evaluation config is malformed, has unknown keys, or bad values."""


class EvalDatasetError(EvalError):
    """A conversation dataset is missing, malformed, or has duplicate case ids."""
