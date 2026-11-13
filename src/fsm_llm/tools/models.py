"""The tool models: a tool's behaviour hints, its definition, a call to it and its result."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fsm_llm.logging import logger

from .constants import Defaults, ToolObservationPrefix, ToolRunStatus
from .truncation import smart_truncate

# DECISION plan-2026-07-20T040150-876e7164/D-008 [STALE]
# Tool names are what get sent to provider function-calling APIs, so this pattern is
# the provider contract, not a general identifier rule. Two things NOT to do here:
#
#   1. Do NOT "simplify" this back to `v.replace("_", "").replace("-", "").isalnum()`.
#      `str.isalnum()` is Unicode-aware, so that check ACCEPTED `検索`, `café_lookup`
#      and fullwidth digits (U+FF11 U+FF12 U+FF13) — all measured, G-09. No provider
#      function-calling API will accept.
#   2. Do NOT swap in core's `fsm_llm.core.definitions.ASCII_IDENTIFIER_PATTERN`. It forbids
#      hyphens, which provider function-calling APIs legitimately allow and tool authors
#      use (`web-search`). Core's pattern governs FSM state/context identifiers, which is
#      a different contract; the two are deliberately not shared.
#
# Also deliberately NOT reusing `session.SESSION_ID_RE` (same charset, no cap):
# that one guards filesystem path safety for conversation ids. Same regex today,
# unrelated contracts — coupling them would make a provider bound govern file naming.
#
# The 64 bound: CONFIRMED against the installed litellm, not merely inferred from the
# published spec — `litellm/llms/anthropic/experimental_pass_through/adapters/
# transformation.py:23` defines `OPENAI_MAX_TOOL_NAME_LENGTH = 64` and truncates to it,
# and `litellm/llms/bedrock/count_tokens/transformation.py:144` records the same 64 cap
# for Bedrock (with a narrower charset litellm normalizes itself). 64 is the tightest
# common bound, so capping here is safe for every provider litellm fans out to.
_TOOL_NAME_PATTERN = re.compile(r"[a-zA-Z0-9_-]{1,64}")


class ToolAnnotations(BaseModel):
    """Behaviour hints for a tool, mirroring the MCP tool annotations.

    Each hint is ``True``, ``False`` or ``None`` (unknown). Only an explicit
    ``True`` counts: a tool with unknown hints is treated as neither read-only
    nor idempotent (``RetryingToolRegistry`` never retries it).

    - ``read_only``: the tool does not change its environment.
    - ``destructive``: a change it makes may be irreversible.
    - ``idempotent``: calling it again with the same arguments has no
      further effect.
    - ``open_world``: it reaches outside a closed domain (network, web).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    read_only: bool | None = None
    destructive: bool | None = None
    idempotent: bool | None = None
    open_world: bool | None = None

    @property
    def retry_safe(self) -> bool:
        """True when a failed call may be re-run: idempotent or read-only."""
        return self.idempotent is True or self.read_only is True


# The hints of a tool that only reads a closed store (memory, OKF, skills).
READ_ONLY_ANNOTATIONS = ToolAnnotations(
    read_only=True, destructive=False, idempotent=True, open_world=False
)


class ToolDefinition(BaseModel):
    """Definition of a tool available to an agent.

    ``timeout_seconds`` is the per-call limit ``ToolRegistry.execute`` enforces (a
    ``timed_out`` result past it; the tool itself keeps running, a Python
    thread cannot be killed). ``None`` means no limit; otherwise a finite
    number above 0 and at most ``Defaults.MAX_TOOL_TIMEOUT_S``
    (``threading.TIMEOUT_MAX``).
    ``args_model`` is the pydantic model of the arguments, built from the
    function signature by ``@tool``/``register_function`` when every parameter
    is annotated; ``None`` for dict-style tools and explicit schemas. When set,
    ``ToolRegistry.get_json_schemas`` emits its JSON schema.
    """

    name: str
    description: str
    parameter_schema: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False
    annotations: ToolAnnotations = Field(default_factory=ToolAnnotations)
    timeout_seconds: float | None = Field(
        default=None, gt=0, le=Defaults.MAX_TOOL_TIMEOUT_S, allow_inf_nan=False
    )

    # Not serialized — runtime only
    execute_fn: Callable[..., Any] | None = Field(default=None, exclude=True)
    args_model: type[BaseModel] | None = Field(default=None, exclude=True)

    # extra="forbid": a removed spelling (``timeout_s``) must raise, not be
    # dropped, or the tool would run with no time limit.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        # DECISION plan-2026-07-20T040150-876e7164/D-008 [STALE] — see _TOOL_NAME_PATTERN above.
        # `fullmatch` on a pattern with no anchors is intentional: it anchors both ends
        # itself, so the cap cannot be escaped by a trailing newline the way `$` allows.
        if not _TOOL_NAME_PATTERN.fullmatch(v):
            raise ValueError(
                "Tool name must be 1-64 ASCII alphanumeric characters "
                f"with underscores or hyphens: '{v}'"
            )
        return v


class ToolCall(BaseModel):
    """A request to invoke a tool."""

    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


class ToolResult(BaseModel):
    """Result of a tool execution.

    ``timed_out`` marks a call stopped waiting for at its ``timeout_seconds``
    (``success`` is False): its outcome is unknown, since the tool may still
    be running and its side effects may still happen.
    """

    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    timed_out: bool = False

    @property
    def summary(self) -> str:
        if self.success:
            text = str(self.result)
            if len(text) > Defaults.MAX_OBSERVATION_LENGTH:
                logger.debug(
                    f"Tool result truncated from {len(text)} to "
                    f"~{Defaults.MAX_OBSERVATION_LENGTH} chars"
                )
                return smart_truncate(text, Defaults.MAX_OBSERVATION_LENGTH)
            return text
        return f"Error: {self.error}"

    @property
    def observation(self) -> str:
        """What the model is shown: ``summary``, prefixed when the call failed.

        A failed call is prefixed ``[TOOL FAILED]``; a timed-out one
        ``[TOOL OUTCOME UNKNOWN]`` (it may still run and take effect).
        """
        if self.success:
            return self.summary
        if self.timed_out:
            return f"{ToolObservationPrefix.OUTCOME_UNKNOWN} {self.summary}"
        return f"{ToolObservationPrefix.FAILED} {self.summary}"

    @property
    def status(self) -> str:
        """``ToolRunStatus`` of this call: success, failed, or unknown (timed out).

        The one mapping every executor writes to ``tool_status``.
        """
        if self.success:
            return ToolRunStatus.SUCCESS
        if self.timed_out:
            return ToolRunStatus.UNKNOWN
        return ToolRunStatus.FAILED
