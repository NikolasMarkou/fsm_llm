"""
Exception hierarchy for the fsm_llm_harness package.

Rooted at ``fsm_llm.definitions.FSMError`` so a caller catching the framework's
base error also catches every harness failure.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.definitions import FSMError

from .constants import Defaults


class HarnessError(FSMError):
    """Base exception for all harness-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details=details)


class HarnessGateBlockedError(HarnessError):
    """A HARD protocol gate refused to open.

    Carries the pre-step-gate slug (one of ``GateSlug.ORDER``) so callers can
    branch on the slug rather than parsing the message.
    """

    def __init__(self, slug: str, message: str, details: dict[str, Any] | None = None):
        self.slug = slug

        error_details = details or {}
        error_details["slug"] = slug

        super().__init__(f"Gate [{slug}]: {message}", error_details)


class HarnessLeashError(HarnessError):
    """The 2-attempt autonomy leash is exhausted.

    Raised instead of dispatching a 3rd fix attempt for the same plan step.
    """

    def __init__(
        self,
        step: str,
        attempts: int,
        cap: int = Defaults.MAX_FIX_ATTEMPTS,
        details: dict[str, Any] | None = None,
    ):
        self.step = step
        self.attempts = attempts
        self.cap = cap

        error_details = details or {}
        error_details["step"] = step
        error_details["attempts"] = attempts
        error_details["cap"] = cap

        super().__init__(
            f"Step '{step}': autonomy leash exhausted "
            f"({attempts} fix attempts, cap {cap})",
            error_details,
        )


class HarnessArtifactError(HarnessError):
    """An artifact could not be read, written, parsed or schema-validated."""

    def __init__(
        self,
        artifact: str,
        message: str,
        cause: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.artifact = artifact
        self.cause = cause

        error_details = details or {}
        error_details["artifact"] = artifact
        if cause is not None:
            error_details["cause"] = str(cause)
            error_details["cause_type"] = type(cause).__name__

        super().__init__(f"Artifact '{artifact}': {message}", error_details)


class HarnessOwnershipError(HarnessError):
    """A role attempted to write an artifact it does not own.

    Enforces the File Ownership Model: exactly one writing role per artifact.
    """

    def __init__(
        self,
        artifact: str,
        role: str,
        owner: str,
        details: dict[str, Any] | None = None,
    ):
        self.artifact = artifact
        self.role = role
        self.owner = owner

        error_details = details or {}
        error_details["artifact"] = artifact
        error_details["role"] = role
        error_details["owner"] = owner

        super().__init__(
            f"Role '{role}' may not write '{artifact}' (owner: '{owner}')",
            error_details,
        )


class HarnessReentrancyError(HarnessError):
    """A worker callable re-entered the driver while a dispatch was in flight.

    Sub-agents may not spawn sub-agents: exactly one coordinator per run.
    """

    def __init__(self, role: str, details: dict[str, Any] | None = None):
        self.role = role

        error_details = details or {}
        error_details["role"] = role

        super().__init__(
            f"Worker for role '{role}' re-entered the harness driver; "
            "a dispatched worker must not drive the protocol",
            error_details,
        )


class HarnessConfinementError(HarnessError):
    """A path escaped the plan directory or the workspace root.

    Raised before any I/O is attempted, so a rejected path is never touched.
    """

    def __init__(self, path: str, root: str, details: dict[str, Any] | None = None):
        self.path = path
        self.root = root

        error_details = details or {}
        error_details["path"] = path
        error_details["root"] = root

        super().__init__(
            f"Path '{path}' resolves outside the permitted root '{root}'",
            error_details,
        )
