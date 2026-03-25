from __future__ import annotations

"""
Exception hierarchy for the meta-agent package.
"""

from typing import Any

from fsm_llm.definitions import FSMError


class MetaAgentError(FSMError):
    """Base exception for all meta-agent-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details=details)


class BuilderError(MetaAgentError):
    """Error during artifact building (invalid state/step/tool operations)."""

    def __init__(self, message: str, action: str | None = None, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.action = action


class MetaValidationError(MetaAgentError):
    """Error during artifact validation."""

    def __init__(
        self,
        message: str,
        errors: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.errors = errors or []


class OutputError(MetaAgentError):
    """Error during artifact output/serialization."""

    def __init__(self, message: str, path: str | None = None, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.path = path
