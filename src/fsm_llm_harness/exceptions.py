"""
Exception hierarchy for the fsm_llm_harness package.

Rooted at ``fsm_llm.definitions.FSMError`` so a caller catching the framework's
base error also catches every harness failure.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.definitions import FSMError


class HarnessError(FSMError):
    """Base exception for all harness-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details=details)


# DECISION plan-2026-07-21T125237-191b2eb2/D-059
# Do NOT re-add `HarnessGateBlockedError` or `HarnessLeashError`. They shipped
# at step 1 as speculative types and were raised NOWHERE through step 7e -- but
# the reason to delete them is not the empty grep, it is that the shipped
# driver represents both conditions as CONTEXT, on purpose:
#   - a blocked gate is the protocol's NORMAL turn outcome (`TransitionEvaluator`
#     returns BLOCKED and the FSM holds state -- assumption A2), recorded as
#     `last_gate_slug` + `halt_reason` (harness.py `_after_execute_dispatch`);
#   - an exhausted leash halts the same way and then ASKS a human
#     (`_offer_leash_continue`, D-052) -- it is a decision point, not a fault.
# Raising through the FSM handler boundary would also lose the run: core's
# handler system does not carry a harness exception back to `run()` as a
# protocol outcome. If a later step needs "the gate refused" as a value, it
# already has one -- `GateSlug` + `last_gate_slug`.
# See decisions.md D-059.


class HarnessArtifactError(HarnessError):
    """An artifact could not be read, written, parsed or schema-validated.

    Raised by ``artifacts.py`` at its parse/schema boundary and by
    ``storage.py`` at its read/write boundary.  It was kept ahead of both --
    unlike the two exception types D-059 deleted -- because a filesystem or
    schema failure genuinely IS exceptional, where a blocked gate is not.
    """

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
