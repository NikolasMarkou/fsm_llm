"""
Pydantic models for the agents package.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fsm_llm.agents.constants import Defaults, MetaDefaults, StopReason
from fsm_llm.core.codes import Reason, Status, status_of_stop_reason
from fsm_llm.core.definitions import ModelNotConfiguredError
from fsm_llm.core.results import RunResult
from fsm_llm.llm import PLACEHOLDER_MODEL, interface_model, resolve_model
from fsm_llm.tools.models import ToolCall


class AgentStep(BaseModel):
    """A single step in the agent's execution trace."""

    iteration: int
    thought: str = ""
    action: str = ""
    observation: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentTrace(BaseModel):
    """Complete trace of an agent's execution."""

    tool_calls: list[ToolCall] = Field(default_factory=list)
    total_iterations: int = 0

    @property
    def tools_used(self) -> list[str]:
        return sorted({tc.tool_name for tc in self.tool_calls})


def resolve_agent_model(model: str | None = None) -> str:
    """The model an agent uses: *model*, else env ``LLM_MODEL``; never a default.

    Interface contract (2 call sites: the ``AgentConfig.model`` default factory
    and ``composition.default_llm_judge``):
        - A non-empty *model* wins. Otherwise a non-blank ``LLM_MODEL`` (read
          now, stripped). Always returns a non-empty string.
        - Raises ``ModelNotConfiguredError`` (an ``FSMError`` and a
          ``ValueError``) when both are unset; pydantic lets a
          ``default_factory`` exception propagate, so ``AgentConfig()`` raises
          it too. An agent that holds an injected ``llm_interface`` does not
          come here: see ``default_agent_config``.

    # DECISION plan-2026-09-29T103145-06a5ec0a/D-013
    Do NOT make ``AgentConfig.model`` default to ``None`` and resolve later:
    ``config.model`` is read directly at more than 20 sites (native_fc,
    SelfConsistency, ReasoningReact, meta builder, log lines),
    and every one of them needs a concrete string.
    """
    return resolve_model(model)


# ``PLACEHOLDER_MODEL`` (imported above from ``fsm_llm.llm``, its one home) is
# what ``AgentConfig.model`` holds where no model was chosen: an agent with an
# injected ``llm_interface`` that states no ``model`` (the interface owns the
# model), and the second validation pass of ``AgentArtifactBuilder._config_problems``.
# ``resolve_model`` counts it as unset.


def model_or_placeholder(model: str | None = None) -> str:
    """*model*, else env ``LLM_MODEL``, else ``PLACEHOLDER_MODEL``; never raises.

    Interface contract (callers: ``default_agent_config``,
    ``composition.default_llm_judge``, ``SwarmAgent``, ``SOPRegistry``): for a
    config that must hold a model string although the model is owned elsewhere
    (an injected interface, a custom backend, member agents) or is not used at
    all (validating a template). Where an agent is really built from a
    config, ``AgentConfig()`` and ``resolve_agent_model`` keep raising
    ``ModelNotConfiguredError``. The placeholder names nothing and is never
    sent to a provider.
    """
    try:
        return resolve_agent_model(model)
    except ModelNotConfiguredError:
        return PLACEHOLDER_MODEL


class AgentConfig(BaseModel):
    """Configuration for an agent.

    Unknown fields raise (``extra="forbid"``): a typo such as ``max_iteration``
    is an error, not a silently ignored key.
    """

    model: str = Field(default_factory=resolve_agent_model)
    """litellm model id. Default: env ``LLM_MODEL`` at construction; unset
    raises ``ModelNotConfiguredError`` (there is no built-in model)."""

    max_iterations: int = Defaults.MAX_ITERATIONS
    timeout_seconds: float = Defaults.TIMEOUT_SECONDS
    temperature: float = Defaults.TEMPERATURE
    max_tokens: int = Defaults.MAX_TOKENS
    """Maximum tokens per LLM response (passed to the LLM provider).

    This controls the maximum length of each individual LLM response,
    NOT the total token budget for the agent run.
    """
    output_schema: type | None = Field(default=None, exclude=True)
    """Optional Pydantic model class for structured agent output.

    When set, the agent's final answer is validated against this schema
    and the parsed model is stored in ``AgentResult.structured_output``.
    """
    transition_config: Any = Field(default=None, exclude=True)
    """Optional ``TransitionEvaluatorConfig`` to tune FSM transition evaluation.

    Controls strict matching and diagnostic logging; transition ranking is
    by priority alone.
    When None (default), uses the core library defaults.
    """

    max_history_size: int | None = None
    """Optional cap on conversation history retained per turn.

    When None (default), the core ``API`` default (5) is used. Lower values
    bound prompt growth on long agent loops; forwarded to
    ``API.from_definition(max_history_size=...)``. Additive — does not change
    default behavior.
    """

    enable_prompt_cache: bool = False
    """When True, forwards ``caching=True`` to the litellm backend.

    Enables litellm's response caching for repeated identical LLM calls
    (the system + tool preamble re-sent each iteration). Provider/litellm-cache
    dependent; a no-op where unsupported. Default False preserves today's
    behavior.
    """

    reflect_every_n: int | None = None
    """Optional cadence for periodic self-reflection (see VerifiedReactAgent).

    When set to N>0, a reflection step is injected every N iterations. None
    (default) disables periodic reflection. Consumed by reflection-capable
    agent subclasses; ignored by agents that don't support it.
    """

    auto_summarize_after: int | None = None
    """Optional observation count after which accumulated context is summarized.

    When set to N>0, agents that opt in register a summarization handler that
    condenses observations once they exceed N entries. None (default) disables.
    N must be below ``Defaults.MAX_OBSERVATIONS`` (the length observations are
    pruned to).
    """

    verification_fn: Callable[..., Any] | None = Field(default=None, exclude=True)
    """Optional callable ``(answer, context) -> bool | dict`` to ground/verify
    an agent's answer before concluding (see VerifiedReactAgent).

    None (default) disables verification. Consumed by verification-capable
    subclasses; ignored elsewhere.
    """

    force_final_tool: str | None = None
    """Optional tool name the agent is forced to call ONCE after the ReAct loop.

    When set to a registry tool that the loop never called,
    ``NativeFunctionCallingReactAgent`` issues one post-loop completion with
    ``tool_choice`` pinned to it (no ``response_format=``), executes the returned
    call, and records it in the trace. None (default) disables (no forced-write
    finalization). Consumed only by ``NativeFunctionCallingReactAgent``.
    """

    instructions: str | None = Field(
        default=None, max_length=Defaults.MAX_INSTRUCTIONS_LENGTH
    )
    """Caller-owned standing instructions (a system prompt).

    FSM patterns prefix them to every non-empty prompt instruction of their
    FSM (Pass 2 replies and the per-field extractions where tools and answers
    are chosen); ``NativeFunctionCallingReactAgent`` uses them as its default
    ``system_policy``. ``SwarmAgent`` and ``AgentGraph`` hand work to member
    agents, so set it on the members; ``MetaBuilderAgent`` ignores it.
    ``None`` (default) changes no prompt.
    The text shares each slot's 5000-char core budget: long instructions with
    a large tool registry fail the FSM load with a length error.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator(
        "max_iterations",
        "timeout_seconds",
        "temperature",
        "max_tokens",
        "max_history_size",
        "reflect_every_n",
        "auto_summarize_after",
        mode="before",
    )
    @classmethod
    def refuse_bool_number(cls, v: Any) -> Any:
        # DECISION plan-2026-11-06T093029-0068b055/D-020: do NOT accept a bool
        # as a number. ``True`` is an ``int`` subclass and pydantic coerces it
        # to 1 / 1.0, so ``temperature=True`` or ``max_iterations=True`` would
        # run as a silent 1; it is a caller mistake, not a value. Digit strings
        # and integral floats still coerce as before.
        if isinstance(v, bool):
            raise ValueError("must be a number, not a bool")
        return v

    @field_validator(
        "max_iterations", "max_tokens", "max_history_size", "reflect_every_n"
    )
    @classmethod
    def bound_int_limit(cls, v: int | None) -> int | None:
        # DECISION plan-2026-11-06T093029-0068b055/D-030 (extends D-020,
        # step 12.35): an int field stops at ``Defaults.MAX_INT_LIMIT``, the bound
        # ``require_int_limit`` puts on every constructor limit. Do NOT accept
        # more: ``max_iterations=10**5000`` was valid and its ``repr()`` raised
        # (the 4300-digit conversion limit), as did every log line and JSON
        # dump of the config. Do NOT clamp (a refusal tells the caller);
        # ``auto_summarize_after`` is already capped far below this.
        if v is not None and v > Defaults.MAX_INT_LIMIT:
            raise ValueError(f"must be at most {Defaults.MAX_INT_LIMIT}")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model must not be blank")
        return v

    @field_validator("max_history_size", "reflect_every_n", "auto_summarize_after")
    @classmethod
    def validate_optional_positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("value must be a positive integer when set")
        return v

    @field_validator("auto_summarize_after")
    @classmethod
    def validate_auto_summarize_after(cls, v: int | None) -> int | None:
        # Observations are pruned to MAX_OBSERVATIONS: a threshold at or above
        # it never fires.
        if v is not None and v >= Defaults.MAX_OBSERVATIONS:
            raise ValueError(
                f"auto_summarize_after must be below {Defaults.MAX_OBSERVATIONS} "
                "(the observation list never grows past it)"
            )
        return v

    @field_validator("max_iterations")
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_iterations must be at least 1")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_tokens must be at least 1")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("force_final_tool")
    @classmethod
    def validate_force_final_tool(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("force_final_tool must be a non-empty tool name when set")
        return v

    @field_validator("output_schema")
    @classmethod
    def validate_output_schema(cls, v: type | None) -> type | None:
        if v is not None and not (isinstance(v, type) and issubclass(v, BaseModel)):
            raise ValueError("output_schema must be a Pydantic BaseModel subclass")
        return v

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout_seconds(cls, v: float) -> float:
        # NaN passes ``v <= 0`` and inf is no limit at all: both are refused.
        if not math.isfinite(v) or v <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        return v


_ConfigT = TypeVar("_ConfigT", bound=AgentConfig)


def default_agent_config(
    llm_interface: Any = None,
    config_cls: type[_ConfigT] = AgentConfig,  # type: ignore[assignment]
) -> _ConfigT:
    """The config an agent gets when the caller passed none.

    Interface contract (callers: ``BaseAgent.__init__``, ``create_agent``,
    ``ConfiguredAgentBuilder``, ``MetaBuilderAgent``):
        - ``llm_interface``: the interface injected into the agent, or ``None``.
        - ``config_cls``: ``AgentConfig`` or a subclass.
        - No interface: ``config_cls()``, which raises ``ModelNotConfiguredError``
          when no ``LLM_MODEL`` is set.
        - An interface: its own ``model`` string, else ``LLM_MODEL`` when set,
          else ``PLACEHOLDER_MODEL``; never raises for the model, because the
          interface, not this config, decides which model answers.
    """
    # DECISION plan-2026-11-12T073150-d454fd62/D-005
    # Do NOT build ``AgentConfig()`` eagerly for an agent that holds an injected
    # interface: with no built-in model it would raise for a user test that
    # passes only a double. Do NOT read a model from the interface any other
    # way than ``interface_model`` (a mock attribute is not a model).
    if llm_interface is None:
        return config_cls()
    return config_cls(model=interface_model(llm_interface) or model_or_placeholder())


def config_with(base: AgentConfig, **updates: Any) -> AgentConfig:
    """A validated copy of *base* (same class) with *updates* applied.

    Interface contract (callers: ``create_agent``'s ``system_prompt=``,
    ``ConfiguredAgentBuilder``): *updates* are ``base``'s own field names and
    are validated again, unlike ``model_copy(update=)``; a bad value raises
    ``ValidationError``. *base* is not modified.
    """
    fields = {name: getattr(base, name) for name in type(base).model_fields}
    return type(base)(**{**fields, **updates})


def _no_status() -> Status:
    # DECISION plan-2026-11-20T055240-ace7a8c8/D-046: ``status`` has this
    # raising default factory, so a result that gives neither ``status`` nor
    # ``stop_reason`` is a ``ValueError`` and one that gives ``stop_reason``
    # derives it first (the ``before`` validator). Do NOT make ``status``
    # required: the pydantic mypy plugin then demands ``status=`` at every call
    # site that derives it from ``stop_reason``. Do NOT give it a real default
    # (``OK``, ``FAILED``): a result built with no outcome would pass as one.
    # Do NOT put a ``success`` field back (``extra="forbid"`` refuses it).
    # ``model_copy(update=)`` does not validate: set ``status``, ``reason`` and
    # ``stop_reason`` together. See decisions.md D-046.
    raise ValueError("AgentResult needs a status or a stop_reason: give one of them")


class AgentResult(BaseModel):
    """Result of an agent execution.

    How the run ended is ``status`` (and ``reason``) from ``fsm_llm.core.codes``;
    ``success`` is derived from it (``status is Status.OK``) and is not a field,
    so ``AgentResult(success=...)`` is refused. Give ``stop_reason`` and the
    status and reason follow from it (``status_of_stop_reason``), or give
    ``status`` (an explicit ``status`` or ``reason`` wins); giving neither is a
    ``ValueError``. ``run_result()`` projects the result onto the core
    ``RunResult`` envelope.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str
    status: Status = Field(default_factory=lambda: _no_status())
    """How the run ended; derived from ``stop_reason`` when omitted."""
    reason: Reason | None = None
    """Why the status is not ``ok``, when a reason is known."""
    trace: AgentTrace = Field(default_factory=AgentTrace)
    final_context: dict[str, Any] = Field(default_factory=dict)
    structured_output: Any = None
    """Validated Pydantic model instance when ``AgentConfig.output_schema`` is set."""
    stop_reason: StopReason | None = None
    """Why the run ended, a ``constants.StopReason`` (``None`` when the result
    was built outside an agent pattern). Success only for ``answered`` and
    ``evidence``; a forced stop keeps its answer but is not ``ok``."""

    @model_validator(mode="before")
    @classmethod
    def _derive_status(cls, data: Any) -> Any:
        """Fill ``status`` and ``reason`` from ``stop_reason`` when ``status`` is omitted."""
        if not isinstance(data, dict) or data.get("status") is not None:
            return data
        stop_reason = data.get("stop_reason")
        if stop_reason is None:
            return data
        status, reason = status_of_stop_reason(StopReason.from_string(stop_reason))
        derived = {**data, "status": status}
        if data.get("reason") is None:
            derived["reason"] = reason
        return derived

    @property
    def success(self) -> bool:
        """True exactly when the run ended ``ok`` (derived from ``status``)."""
        return self.status is Status.OK

    def run_result(self) -> RunResult:
        """This result as the core ``RunResult`` envelope.

        ``status``, ``reason`` and ``answer`` carry over; ``outputs`` is
        ``structured_output`` as a dict when it is a model or a mapping, else
        ``{}``; ``detail`` holds ``stop_reason`` (the wire string or ``None``),
        ``iterations_used`` and ``tools_used``; ``identity`` is empty.
        """
        outputs: dict[str, Any] = {}
        structured = self.structured_output
        if isinstance(structured, BaseModel):
            outputs = structured.model_dump()
        elif isinstance(structured, dict):
            outputs = dict(structured)
        return RunResult(
            status=self.status,
            reason=self.reason,
            outputs=outputs,
            answer=self.answer,
            detail={
                "stop_reason": self.stop_reason.value if self.stop_reason else None,
                "iterations_used": self.iterations_used,
                "tools_used": list(self.tools_used),
            },
        )

    @property
    def iterations_used(self) -> int:
        return self.trace.total_iterations

    @property
    def tools_used(self) -> list[str]:
        return self.trace.tools_used

    def __str__(self) -> str:
        if self.structured_output is not None:
            return str(self.structured_output)
        return self.answer


# ---------------------------------------------------------------------------
# Pattern-specific models
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    """A single step in a plan decomposition."""

    step_id: int
    description: str
    dependencies: list[int] = Field(default_factory=list)
    status: str = "pending"
    result: str = ""


class EvaluationResult(BaseModel):
    """Result from an evaluation pass."""

    passed: bool
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    feedback: str = ""
    criteria_met: list[str] = Field(default_factory=list)


class ReflexionMemory(BaseModel):
    """An episodic memory entry from a Reflexion cycle."""

    episode: int
    task_summary: str = ""
    outcome: str = ""
    reflection: str = ""
    lessons: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DebateRound(BaseModel):
    """A single round in a debate."""

    round_num: int
    proposition: str = ""
    critique: str = ""
    counter_argument: str = ""
    judge_verdict: str = ""


class ChainStep(BaseModel):
    """A step definition for prompt chaining."""

    step_id: str
    name: str
    extraction_instructions: str
    response_instructions: str
    validation_fn: Callable[[dict[str, Any]], bool] | None = Field(
        default=None, exclude=True
    )

    model_config = {"arbitrary_types_allowed": True}


class DecompositionResult(BaseModel):
    """Result of an ADaPT-style task decomposition."""

    subtasks: list[str] = Field(default_factory=list)
    operator: str = "AND"
    depth: int = 0

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v: str) -> str:
        v = v.upper()
        if v not in ("AND", "OR"):
            raise ValueError(f"operator must be 'AND' or 'OR', got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Meta-builder models
# ---------------------------------------------------------------------------


class ArtifactType(str, Enum):
    """Type of artifact the meta-builder can build."""

    FSM = "fsm"
    WORKFLOW = "workflow"
    AGENT = "agent"


class BuildProgress(BaseModel):
    """Progress tracking for artifact building."""

    total_required: int = 0
    completed: int = 0
    missing: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def percentage(self) -> float:
        if self.total_required == 0:
            return 0.0
        return (self.completed / self.total_required) * 100.0

    @property
    def is_complete(self) -> bool:
        return len(self.missing) == 0 and self.total_required > 0


# AgentConfig fields the meta-builder never reads (checked against
# the agents/meta_builder/ package, which uses model, temperature, max_tokens, timeout_seconds
# and max_turns only). ``max_iterations`` (D-012) and ``instructions`` stay
# accepted.
_UNREAD_AGENT_FIELDS = (
    "enable_prompt_cache",
    "max_history_size",
    "transition_config",
    "output_schema",
    "reflect_every_n",
    "auto_summarize_after",
    "verification_fn",
    "force_final_tool",
)


class MetaBuilderConfig(AgentConfig):
    """Configuration for the meta-builder agent.

    ``model``, ``temperature`` and ``max_tokens`` configure the conversation's
    LLM interface; ``timeout_seconds`` is its per-request timeout;
    ``max_turns`` caps ``send`` calls. ``max_iterations`` is inherited from
    ``AgentConfig`` and not read by the meta-builder (kept so existing callers
    that pass it keep validating, D-012), and so is ``instructions`` (ignored,
    documented). The other inherited fields the meta-builder never reads
    (``_UNREAD_AGENT_FIELDS``) are refused when set to a non-default value.
    """

    temperature: float = MetaDefaults.TEMPERATURE
    max_tokens: int = MetaDefaults.MAX_TOKENS
    max_turns: int = MetaDefaults.MAX_TURNS

    @classmethod
    def from_agent_config(cls, config: AgentConfig) -> MetaBuilderConfig:
        """``config`` as a ``MetaBuilderConfig``.

        A ``MetaBuilderConfig`` is returned unchanged. Any other
        ``AgentConfig`` gives a new one carrying only the fields the caller set
        (``model_fields_set``), so an unset ``temperature``/``max_tokens`` keep
        the meta defaults (0.7/4096), not the ``AgentConfig`` ones. Raises
        ``ValidationError`` when a carried value is invalid.
        """
        # DECISION plan-2026-11-05T134105-5ed71cf2/D-013: the meta pattern
        # needs ``max_turns`` and the 0.7/4096 defaults, which a plain
        # AgentConfig (what create_agent / the builder pass) lacks. Do NOT
        # relax MetaBuilderConfig's extra="forbid" or copy every field of the
        # plain config: that would turn its unset defaults into explicit values.
        if isinstance(config, cls):
            return config
        return cls(**{name: getattr(config, name) for name in config.model_fields_set})

    @field_validator("max_turns")
    @classmethod
    def validate_max_turns(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_turns must be at least 1")
        return v

    @model_validator(mode="after")
    def refuse_unread_agent_fields(self) -> MetaBuilderConfig:
        """Raise ``ValueError`` naming an unread inherited field set off its default.

        Contract: runs on every construction (``from_agent_config`` and
        ``config_with`` included); returns ``self`` unchanged otherwise.
        """
        # DECISION plan-2026-11-06T093029-0068b055/D-010: the meta-builder
        # reads only model, temperature, max_tokens, timeout_seconds and
        # max_turns, so a non-default value for these fields would be accepted
        # and ignored. Refuse it (D-038, never drop). Do NOT consult
        # ``model_fields_set``: ConfiguredAgentBuilder._config_for_build marks
        # all of them "set" at their defaults, so the signal is the value
        # against the field default. Do NOT silently accept or honour them
        # here (that would copy BaseAgent._create_api logic).
        for name in _UNREAD_AGENT_FIELDS:
            default = AgentConfig.model_fields[name].get_default(
                call_default_factory=True
            )
            if getattr(self, name) != default:
                raise ValueError(
                    f"{name} is not used by the meta-builder agent and cannot "
                    f"be set (default {default!r})"
                )
        return self


class MetaBuilderResult(AgentResult):
    """Result of a meta-builder session."""

    # Defaults for the two fields AgentResult requires
    answer: str = ""
    status: Status = Status.OK

    artifact_type: ArtifactType = ArtifactType.FSM
    artifact: dict[str, Any] = Field(default_factory=dict)
    artifact_json: str = ""
    is_valid: bool = Field(
        default=True,
        description=(
            "For fsm artifacts: the artifact loads as an FSMDefinition. For "
            "workflow and agent artifacts: a structurally complete spec that is "
            "NOT directly loadable (workflow steps and agent tools need Python "
            "callables the caller must wire)."
        ),
    )
    validation_errors: list[str] = Field(default_factory=list)
    conversation_turns: int = 0
