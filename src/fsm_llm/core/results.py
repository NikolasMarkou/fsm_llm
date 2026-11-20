"""The result envelope of a run (roadmap 0.16, contract C13): ``RunResult`` and ``Identity``.

``RunResult`` is what a run of anything reports: ``status`` and ``reason`` from the
closed vocabularies of ``fsm_llm.core.codes``, the declared ``outputs``, the
``answer`` text, a kind-specific ``detail``, the ``identity`` of what ran and a
``handle`` to resume or inspect it. ``success`` is derived from ``status`` and is
never stored, passed or dumped.

``Identity`` carries the two digests of C11: the ``template_identity`` of the
document that was loaded and the ``instance_identity`` of the instantiated FSM
body. Either may be ``None`` (a code-backed built-in has no static instance, a
result built by a projection may know neither); ``RunResult.identity`` is
always an ``Identity``, never ``None``.

Both models are frozen and refuse an unknown keyword. Each runner exposes its own
result as a projection onto this type (plan ace7a8c8 steps C4a and C4b).

Imported by name (``from fsm_llm.core.results import RunResult``);
``fsm_llm/__init__.py`` does not import it. Imports the codes and pydantic only.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .codes import Reason, Status

__all__ = ["Identity", "RunResult"]


class Identity(BaseModel):
    """The two digests of one run: of the template that was loaded, of the instance that ran.

    ``template_identity`` is the digest of the canonical executable envelope,
    ``instance_identity`` the ``definition_digest`` of the instantiated FSM body.
    ``None`` means not known or not static for this run, never an empty string.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    template_identity: str | None = Field(
        None, description="Digest of the template (canonical envelope), if known"
    )
    instance_identity: str | None = Field(
        None, description="Digest of the instantiated body, if static"
    )


class RunResult(BaseModel):
    """The one envelope of what a run reports (contract C13).

    Fields: ``status`` and ``reason`` (``fsm_llm.core.codes``; a reason only
    where a status needs one), ``outputs`` (the declared keys only), ``answer``,
    ``detail`` (kind-specific, never branched on by library code), ``identity``
    and ``handle`` (a conversation or instance id to resume or inspect).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Status = Field(..., description="How the run ended")
    reason: Reason | None = Field(None, description="Why, when the status needs one")
    outputs: dict[str, Any] = Field(
        default_factory=dict, description="The declared output keys"
    )
    answer: str | None = Field(None, description="The answer text, if any")
    detail: dict[str, Any] = Field(
        default_factory=dict, description="Kind-specific facts about the run"
    )
    identity: Identity = Field(
        default_factory=Identity, description="Template and instance digests"
    )
    handle: str | None = Field(None, description="Id to resume or inspect the run")

    @property
    def success(self) -> bool:
        """True exactly when the run ended ``ok``."""
        return self.status is Status.OK
