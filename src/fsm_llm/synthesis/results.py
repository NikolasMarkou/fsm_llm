"""The ``synthesis_result`` dict as a ``RunResult`` (roadmap 0.16, D-047).

``ProcedureRunner`` writes ``{status, procedure_id, version, reason}`` to the
result key of a ``synthesize`` state; that dict is the wire contract (edges read
it) and is never replaced. ``synthesis_run_result`` reads it and returns the core
envelope.

Interface contract. ``synthesis_run_result(result)`` takes the dict (any
``Mapping``) and returns a ``RunResult``: ``status`` and ``reason`` through
``Status.from_string`` and ``Reason.from_string``, ``identity.instance_identity``
the ``procedure_id`` (the library digest, which is the ``definition_digest`` of
the procedure body), ``identity.template_identity`` ``None`` (no envelope exists
yet), ``detail`` ``{"version": ...}``. It never changes its argument. Failure
mode: ``TypeError`` for a non-mapping, ``ValueError`` for a missing or unknown
status or an unknown reason (never a guess).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.results import Identity, RunResult

__all__ = ["synthesis_run_result"]


def synthesis_run_result(result: Mapping[str, Any]) -> RunResult:
    """The ``RunResult`` of one ``synthesis_result`` dict (see the module text)."""
    if not isinstance(result, Mapping):
        raise TypeError(f"a synthesis result is a mapping, got {type(result).__name__}")
    reason = result.get("reason")
    return RunResult(
        status=Status.from_string(result.get("status")),  # type: ignore[arg-type]
        reason=None if reason is None else Reason.from_string(reason),
        detail={"version": result.get("version")},
        identity=Identity(instance_identity=result.get("procedure_id")),
    )
