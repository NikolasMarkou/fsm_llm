"""
The pure helpers of a synthesize run: the result record, the reason codes, the round
strings and the refusal a stored procedure meets before it may run.

Moved without change from the old ``runner.py`` (plan baa37f40, step 2; the runner is
gone since step 22) so the agent tools and the executor of the synthesize kind share them. Interface
contract: ``_result(status, reason, procedure)`` returns the wire dict
``{status, procedure_id, version, reason}``; ``reason_of(error)`` and
``runnable_reason(...)`` return a code of ``constants.REASONS`` (``runnable_reason``
returns ``None`` when the procedure may run); none of them reads an exception message
into a result.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from fsm_llm.core.definitions import FSMDefinition, LLMResponseError
from fsm_llm.core.state_kinds import kind_of

from .constants import (
    LIFECYCLE_TRUSTED,
    LIFECYCLE_VERIFIED,
    REASON_COMPILE_FAILED,
    REASON_COMPLETION_UNSUPPORTED,
    REASON_LIBRARY_ERROR,
    REASON_LIFECYCLE_TOO_LOW,
    REASON_NOT_SIMULATED,
    REASON_PROVIDER_ERROR,
    REASONS,
)
from .exceptions import LibraryError, SynthesisError, SynthesisRefusedError
from .gates import simulated, verify_stored
from .policy import SynthesisPolicy
from .synthesizer import Procedure

_ALLOWED_LIFECYCLES = {
    LIFECYCLE_VERIFIED: (LIFECYCLE_VERIFIED, LIFECYCLE_TRUSTED),
    LIFECYCLE_TRUSTED: (LIFECYCLE_TRUSTED,),
}


def _no_errors() -> int:
    return 0


def _result(
    status: str, reason: str | None = None, procedure: Procedure | None = None
) -> dict[str, Any]:
    """The value of the result key: codes and ids only, never exception text."""
    record = procedure.record if procedure else None
    return {
        "status": status,
        "procedure_id": procedure.digest if procedure else None,
        "version": record.version if record else None,
        "reason": reason,
    }


def reason_of(error: Exception) -> str:
    """The fixed reason for an ``obtain`` failure; the message is never read."""
    if isinstance(error, SynthesisRefusedError) and error.code in REASONS:
        return error.code
    if isinstance(error, LibraryError):
        return REASON_LIBRARY_ERROR
    if isinstance(error, SynthesisError) and isinstance(
        error.__cause__, LLMResponseError
    ):
        return REASON_PROVIDER_ERROR
    return REASON_COMPILE_FAILED


def _round_codes(error: BaseException | None) -> str | None:
    """The failure codes of each compile round of a ``SynthesisError`` (codes
    comma-joined, rounds semicolon-joined, cut at 200 characters), else ``None``."""
    if not isinstance(error, SynthesisError):
        return None
    return _round_string(error.details.get("rounds") or ())


def _round_string(rounds: Sequence[Sequence[Any]]) -> str:
    return ";".join(",".join(map(str, codes)) for codes in rounds)[:200]


def runnable_reason(
    definition: FSMDefinition,
    lifecycle: str,
    *,
    min_lifecycle: str,
    policy: SynthesisPolicy,
    inputs: Iterable[str],
    return_keys: Iterable[str],
    gates_passed: object = None,
) -> str | None:
    """The reason code a stored procedure may not run, else ``None``: a
    ``completion`` state, a lifecycle below ``min_lifecycle`` (``verified`` or
    ``trusted``), a record that never passed G6b when ``policy.require_simulation``
    (``gates_passed`` is the record's list; ``None`` means no record, which a
    compile with that policy already simulated) or a failed ``verify_stored``,
    checked in that order. Shared by the executor and the agent tools; never
    raises."""
    if any(not kind_of(s).supported_in_procedure for s in definition.states.values()):
        return REASON_COMPLETION_UNSUPPORTED
    if lifecycle not in _ALLOWED_LIFECYCLES[min_lifecycle]:
        return REASON_LIFECYCLE_TOO_LOW
    # DECISION plan-2026-11-10T194139-2cefeb61/D-062
    # `require_simulation` binds every path that runs a stored procedure, not
    # only `obtain`. Do NOT copy the G6b test inline; use `gates.simulated`.
    if policy.require_simulation and gates_passed is not None:
        if not simulated(gates_passed):
            return REASON_NOT_SIMULATED
    report = verify_stored(definition, policy, inputs=inputs, return_keys=return_keys)
    return None if report.passed else REASON_LIBRARY_ERROR
