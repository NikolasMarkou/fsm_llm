"""The built-in agent patterns as executables (roadmap E2.4; plan ace7a8c8 D-061).

``builtin_executables()`` returns one ``PatternExecutable`` per pattern of ``create_agent``
(the ``pattern_agents`` namespace, ``PATTERN_AGENTS``: the one id list, never copied here). ``fsm_llm.agents`` registers
that function as the ``agents`` source of the ``executable_sources`` namespace, so
``ExecutableRegistry().load_sources()`` serves them by id without the engine importing this
package. The executable builds the agent PER RUN through ``create_agent`` (never the class table
directly, no per-run state on the executable: plan 089d0ec7 D-014, plan 06a5ec0a D-052) and calls
its ``run``. A pattern with a file under ``agents/patterns/`` (``pattern_files.PATTERN_ORDER``) is
served by a ``PatternExecutable`` built from its file (plan a555dcfe step 9, D-015); the others stay
code-backed under the same ids. ``pattern_adapter`` is the ``pattern`` kind of the
``executable_kinds`` namespace, so a registry loads a user pattern file the same way, and
``preset_executables()`` serves the shipped presets that have no agent class.

Interface contract:

- ``PatternExecutable(pattern)``: kind ``pattern``, lifecycle ``builtin``, contract input
  ``task`` (a ``str``); every agent pattern but the meta-builder also accepts the optional input
  ``initial_context`` (a mapping; the agent strips caller grant and run-output keys itself), the
  meta-builder takes ``task`` only. ``requires.callables`` lists the host objects the constructor NEEDS (``tools``,
  ``chain``, ``evaluation_fn``, ``maker_instructions`` and ``checker_instructions``, ``agents``
  and ``entry_agent`` for swarm): a pattern that needs one is therefore not ``launchable`` (serve and
  the monitor refuse it). The optional objects of a row (``hitl``, ``config``, ``tools``
  of plan_execute, adapt and orchestrator, ``worker_factory``, ``evaluation_fn`` of reflexion,
  ``memory``) are read from ``RunContext.callables`` when present. A name is the keyword of the
  pattern's constructor; ``tools`` and ``config`` reach ``create_agent``'s own parameters.
- Run: the cancel token is checked before the run starts; a ``RunCancelledError`` the agent raises,
  or that sits in the ``__cause__`` chain of its error (eight hops), ends ``aborted/cancelled``;
  ``context.max_seconds`` becomes ``AgentConfig.timeout_seconds`` (the smaller of it and the host config's own, never wider); ``context.max_steps``
  is refused (``ValueError`` naming ``max_iterations``); the interface of the context and its event
  log are passed to the agent (a swarm takes neither: its members own theirs). The run is the
  agent's sync ``run`` on a worker thread (``aoffload``: a cancelled await leaves the worker
  running, the documented cancel hazard). ``BudgetExhaustedError`` and ``AgentTimeoutError`` end
  ``aborted/budget_spent`` (``detail["shared_budget"]`` names a spent ``RunContext.budget``, a fact
  of the context and not of the text); any other error propagates. The result is ``AgentResult.run_result()``
  with ``identity = (template digest, None)`` (a code-backed pattern has no static instance: the
  per-run tools shape the instructions) and ``handle
  None``; ``detail`` gains ``pattern``.
- HITL (census D1): the executable never builds a ``HumanInTheLoop`` and never writes an approval
  grant; the host's ``hitl`` passes through unchanged and the agent's own gate stays the only
  place a grant is made. A ``requires_approval`` tool with no ``hitl`` raises ``AgentError`` at
  construction, before any model call.
- ``PatternExecutable(pattern, envelope=None, agent_class=None, pins=None)`` with an ``envelope``: a loaded
  ``PatternEnvelope`` as an executable (``pattern_adapter`` builds it). Header fields, contract and requires come from the envelope, ``template_identity``
  is the envelope digest (``identity_of``: stable across releases, never reads a task or an option),
  there is no instance identity (the tool text depends on the per-run registry), the objects it
  reads are the envelope's ``requires.callables`` plus the optional ``config`` (and ``hitl`` for a
  file that claims the ``hitl`` capability). A run builds the agent per run: the packaged class
  through ``create_agent`` (``agent_class`` given) or a ``PatternAgent`` for a user file. ``pins``
  is accepted for the registry's call and unused (an executable that is code runs no retained
  instance). The HITL rules of this module hold unchanged: the host's ``hitl`` passes through and
  the agent's own gate is the only place a grant is made.
- The meta-builder is single-shot (``run(task)``): ``MetaBuilderResult.is_valid`` false is
  ``failed`` (no reason), the artifact in ``outputs``, the validation errors in ``detail``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from fsm_llm.agents import create_agent, fsm_definitions
from fsm_llm.agents.definitions import (
    AgentConfig,
    MetaBuilderResult,
    config_with,
    default_agent_config,
    model_or_placeholder,
)
from fsm_llm.agents.exceptions import AgentTimeoutError, BudgetExhaustedError
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.agents.pattern_files import PATTERN_ORDER, load_envelope
from fsm_llm.aio import aoffload
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.definitions.executable import (
    Capability,
    Contract,
    PatternEnvelope,
    Requires,
)
from fsm_llm.core.definitions.executable_instance import identity_of, instantiate_body
from fsm_llm.core.results import Identity, RunResult
from fsm_llm.engine.context import RunCancelledError, RunContext
from fsm_llm.engine.executables.code_backed import CodeBackedExecutable, cause_chain
from fsm_llm.engine.executables.namespaces import GENERATORS, PATTERN_AGENTS
from fsm_llm.tools import ToolRegistry, registry_from

__all__ = [
    "PatternExecutable",
    "builtin_executables",
    "pattern_adapter",
    "preset_executables",
]

_TOOLS = Capability.TOOLS
_HITL = Capability.HITL


@dataclass(frozen=True)
class _Row:
    """What one pattern needs beyond a task: see the module docstring."""

    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    tools: Literal["none", "some"] = "none"
    capabilities: tuple[Capability, ...] = ()
    initial_context: bool = True


# One row per id of the ``pattern_agents`` namespace (a test holds the key sets equal).
_ROWS: dict[str, _Row] = {
    "react": _Row(("tools",), ("hitl", "config"), "some", (_TOOLS, _HITL)),
    "rewoo": _Row(("tools",), ("config",), "some", (_TOOLS,)),
    "debate": _Row(optional=("config",)),
    "plan_execute": _Row(optional=("tools", "config")),
    "prompt_chain": _Row(("chain",), ("config",)),
    "self_consistency": _Row(optional=("config",)),
    "orchestrator": _Row(optional=("tools", "worker_factory", "config")),
    "adapt": _Row(optional=("tools", "config")),
    "evaluator_optimizer": _Row(("evaluation_fn",), ("config",)),
    "maker_checker": _Row(("maker_instructions", "checker_instructions"), ("config",)),
    "reflexion": _Row(
        ("tools",), ("hitl", "config", "evaluation_fn"), "some", (_TOOLS, _HITL)
    ),
    "meta_builder": _Row(
        optional=("config",),
        capabilities=(Capability.CONVERSATIONAL,),
        initial_context=False,
    ),
    "swarm": _Row(("agents", "entry_agent"), ("config", "memory")),
    "parallel_react": _Row(("tools",), ("config",), "some", (_TOOLS,)),
    "native_fc": _Row(("tools",), ("config",), "some", (_TOOLS,)),
    "verified_react": _Row(("tools",), ("hitl", "config"), "some", (_TOOLS, _HITL)),
    "auto_memory": _Row(
        ("tools",), ("hitl", "config", "memory"), "some", (_TOOLS, _HITL)
    ),
    "reasoning_react": _Row(("tools",), ("hitl", "config"), "some", (_TOOLS, _HITL)),
}


class PatternExecutable(CodeBackedExecutable):
    """One built-in agent pattern, served by its ``create_agent`` name."""

    refused_budgets = (
        "max_steps",
    )  # `execute` refuses it: set `AgentConfig.max_iterations`

    def __init__(
        self,
        pattern: str,
        *,
        envelope: PatternEnvelope | None = None,
        agent_class: type | None = None,
        pins: Any = None,
    ) -> None:
        self.pattern = pattern
        self.envelope = envelope
        self._agent_class = agent_class
        if envelope is not None:
            self._init_from_file(envelope, agent_class)
            return
        row = _ROWS[pattern]
        cls = PATTERN_AGENTS.resolve(pattern)
        self._row = row
        outputs = ["artifact", "artifact_type"] if pattern == "meta_builder" else []
        super().__init__(
            id=pattern,
            kind="pattern",
            impl=f"{cls.__module__}.{cls.__qualname__}",
            contract=Contract(inputs=["task"], outputs=outputs),
            requires=Requires(tools=row.tools, callables=list(row.required)),
            capabilities=row.capabilities,
            optional_inputs=("initial_context",) if row.initial_context else (),
            optional_callables=row.optional,
        )

    def _init_from_file(
        self, envelope: PatternEnvelope, agent_class: type | None
    ) -> None:
        """Header, contract and objects from a loaded pattern file (D-015).

        ``template_identity`` is the digest of the document (never a release or a caller), the
        lifecycle is the document's, and the objects read are its ``requires.callables`` plus the
        optional ``tools`` (when the file may use some), ``hitl`` (a file that claims the
        capability) and ``config``.
        """
        required = tuple(envelope.requires.callables)
        some_tools = envelope.requires.tools not in ("none", [])
        optional = tuple(
            name
            for name in ("tools", "hitl", "config")
            if name not in required
            and (name != "hitl" or Capability.HITL in envelope.capabilities)
            and (name != "tools" or some_tools)
        )
        cls = agent_class or PatternAgent
        # a class may name host objects it reads when present (reflexion: evaluation_fn); a
        # file's requires.callables are REQUIRED ones (D-018)
        optional += tuple(getattr(cls, "optional_callables", ()))
        self._row = _Row(required, optional)
        super().__init__(
            id=envelope.id,
            kind="pattern",
            impl=f"{cls.__module__}.{cls.__qualname__}",
            contract=envelope.contract,
            requires=envelope.requires,
            capabilities=envelope.capabilities,
            optional_inputs=("initial_context",),
            optional_callables=optional,
            version=envelope.version,
        )
        self.template_identity = identity_of(envelope).template_identity or ""
        self.lifecycle = envelope.lifecycle

    # DECISION plan-2026-11-25T145435-659c07c3/D-005
    # A preview is a fresh FSM dict and nothing else: no agent, model, handler, `HumanInTheLoop`,
    # event log or caller text (no argument; a variable is the fixed string `<name>`). The one stub
    # tool is never flagged and never run. Do NOT build the real agent to draw it, take a task, or
    # return a half graph (a file that cannot instantiate raises its typed error).
    def preview_definition(self) -> dict[str, Any] | None:
        """The FSM an agent of this pattern runs, as a fresh dict; ``None`` for swarm (no FSM)."""
        if self.envelope is not None:
            names = self.envelope.variables
            return instantiate_body(
                self.envelope,
                None,
                variables={name: f"<{name}>" for name in names},
                generators=GENERATORS.get,
            )
        stub = ToolRegistry().register_function(
            lambda: "", name="lookup", description="A stub tool."
        )
        build = {
            "adapt": fsm_definitions.build_adapt_fsm,
            "self_consistency": fsm_definitions.build_self_consistency_fsm,
            "meta_builder": fsm_definitions.build_meta_builder_fsm,
            "parallel_react": lambda: fsm_definitions.build_parallel_react_fsm(stub),
            "reasoning_react": lambda: fsm_definitions.build_react_fsm(stub),
        }.get(self.pattern)
        return build() if build else None

    # DECISION plan-2026-11-20T055240-ace7a8c8/D-061
    # An executable NEVER supplies an approval grant: the host's `hitl` object passes through
    # `objects` unchanged and the agent's own gate stays the only place the driver grant is
    # written, bound to one exact call. Do NOT build a `HumanInTheLoop` here (not even a default
    # or an auto-approving one), do NOT write the grant key or any run-output key into
    # `initial_context`, and do NOT filter or forward the caller's seed in a new way: the agent
    # strips grant and run-output keys itself (`strip_caller_context`). A tool with
    # `requires_approval` and no `hitl` raises `AgentError` at construction, before any model call.
    async def execute(
        self,
        inputs: Mapping[str, Any],
        objects: Mapping[str, Any],
        context: RunContext,
        identity: Identity,
    ) -> RunResult:
        if context.max_steps is not None:
            raise ValueError(
                "max_steps is not a budget of an agent pattern: set "
                "AgentConfig.max_iterations (the 'config' callable) instead"
            )
        task = inputs["task"]
        if not isinstance(task, str):
            raise ValueError(f"input 'task' must be a str, got {type(task).__name__}")
        seed = inputs.get("initial_context")
        if seed is not None and not isinstance(seed, Mapping):
            raise ValueError("input 'initial_context' must be a mapping")
        agent = self._build(objects, context)
        try:
            if self._row.initial_context:
                outcome = await aoffload(
                    agent.run, task, dict(seed) if seed is not None else None
                )
            else:
                outcome = await aoffload(agent.run, task)
        except (BudgetExhaustedError, AgentTimeoutError) as error:
            return self.stopped(
                Status.ABORTED,
                Reason.BUDGET_SPENT,
                identity,
                pattern=self.pattern,
                error=type(error).__name__,
                **context.ledger_detail(),
            )
        except Exception as error:
            # a cancel the agent wrapped in its own error is still a cancel (plan 7efd5cb4 D-005)
            if not any(isinstance(e, RunCancelledError) for e in cause_chain(error)):
                raise
            return self.stopped(
                Status.ABORTED, Reason.CANCELLED, identity, pattern=self.pattern
            )
        return self._result(outcome, identity)

    def _build(self, objects: Mapping[str, Any], context: RunContext) -> Any:
        """The agent of ONE run, built by ``create_agent`` from the resolved host objects."""
        kwargs = dict(objects)
        tools = kwargs.pop("tools", None)
        config = kwargs.pop("config", None)
        llm = context.resolve_llm()
        swarm = self.pattern == "swarm"
        if not swarm:
            if llm is not None:
                kwargs["llm_interface"] = llm
            cls = PATTERN_AGENTS.get(self.pattern) or PatternAgent
            if getattr(cls, "takes_context", False):
                kwargs["context"] = context  # log, cancel and ledger
            elif context.event_log is not None:
                kwargs["event_log"] = context.event_log
        if context.max_seconds is not None:
            # DECISION plan-2026-11-20T055240-ace7a8c8/D-068: a run budget NARROWS the host's own
            # timeout and never widens it (the derive rule of `RunContext`); with no host config
            # there is no host choice to keep. Do NOT assign `max_seconds` unconditionally.
            timeout = context.max_seconds
            if config is None:
                config = (
                    AgentConfig(model=model_or_placeholder())
                    if swarm
                    else default_agent_config(llm)
                )
            else:
                timeout = min(timeout, config.timeout_seconds)
            config = config_with(config, timeout_seconds=timeout)
        return self._make_agent(tools, config, kwargs)

    def _make_agent(self, tools: Any, config: Any, kwargs: dict[str, Any]) -> Any:
        """The agent of one run: the pattern's class built by ``create_agent``, or, for a file
        that no class stands behind, a ``PatternAgent`` over its envelope."""
        if self.envelope is not None and self._agent_class is None:
            registry = (
                tools
                if tools is None or isinstance(tools, ToolRegistry)
                else registry_from(tools)
            )
            return PatternAgent(self.envelope, tools=registry, config=config, **kwargs)
        return create_agent(self.pattern, tools, config=config, **kwargs)

    def _result(self, outcome: Any, identity: Identity) -> RunResult:
        base = outcome.run_result()
        detail = {"kind": self.kind, "pattern": self.pattern, **base.detail}
        status, outputs = base.status, base.outputs
        if isinstance(outcome, MetaBuilderResult):
            outputs = {
                "artifact": outcome.artifact,
                "artifact_type": outcome.artifact_type.value,
            }
            detail["is_valid"] = outcome.is_valid
            detail["validation_errors"] = list(outcome.validation_errors)
            if not outcome.is_valid:
                status = Status.FAILED
        return RunResult(
            status=status,
            reason=base.reason,
            outputs=outputs,
            answer=base.answer,
            detail=detail,
            identity=identity,
        )


def pattern_adapter(
    envelope: PatternEnvelope, *, pins: Any = None
) -> PatternExecutable:
    """The adapter of the ``pattern`` kind (``EXECUTABLE_KINDS``): a loaded envelope to an executable."""
    return PatternExecutable(envelope.id, envelope=envelope, pins=pins)


def builtin_executables() -> list[PatternExecutable]:
    """One executable per ``create_agent`` pattern, in the order of ``PATTERN_AGENTS``.

    A pattern with a shipped file (``pattern_files.PATTERN_ORDER``) is served by its file.
    """
    shipped = {
        name: PatternExecutable(
            name, envelope=load_envelope(name), agent_class=PATTERN_AGENTS.resolve(name)
        )
        for name in PATTERN_ORDER
        if name in PATTERN_AGENTS
    }
    return [
        shipped.get(pattern) or PatternExecutable(pattern)
        for pattern in (name for name, _ in PATTERN_AGENTS.items())
    ]


def preset_executables() -> list[PatternExecutable]:
    """One executable per shipped preset with no agent class (the ``agent_presets`` source, D-033):
    a ``PatternAgent`` runs it, ``contract.input_templates`` render the caller's arguments."""
    return [
        PatternExecutable(name, envelope=load_envelope(name))
        for name in PATTERN_ORDER
        if name not in PATTERN_AGENTS
    ]
