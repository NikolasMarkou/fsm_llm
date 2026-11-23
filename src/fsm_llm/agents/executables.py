"""The built-in agent patterns as executables (roadmap E2.4; plan ace7a8c8 D-061).

``builtin_executables()`` returns one ``PatternExecutable`` per pattern of ``create_agent``
(``fsm_llm.agents._PATTERNS``: the one id list, never copied here). ``fsm_llm.agents`` registers
that function as the ``agents`` source of the ``executable_sources`` namespace, so
``ExecutableRegistry().load_sources()`` serves them by id without the engine importing this
package. They run on today's classes: the executable builds the agent PER RUN through
``create_agent`` (never ``_PATTERNS`` directly, no per-run state on the executable: plan
089d0ec7 D-014, plan 06a5ec0a D-052) and calls its ``run``. Roadmap E4.2 later replaces the bodies
with files under the same ids.

Interface contract:

- ``PatternExecutable(pattern)``: kind ``pattern``, lifecycle ``builtin``, contract input
  ``task`` (a ``str``); the 17 agent patterns also accept the optional input ``initial_context``
  (a mapping; the agent strips caller grant and run-output keys itself), the meta-builder takes
  ``task`` only. ``requires.callables`` lists the host objects the constructor NEEDS (``tools``,
  ``chain``, ``evaluation_fn``, ``maker_instructions`` and ``checker_instructions``, ``agents``
  and ``entry_agent`` for swarm): 12 of the 18, which are therefore not ``launchable`` (serve and
  the monitor refuse them until E4.2). The optional objects of a row (``hitl``, ``config``, ``tools``
  of plan_execute, adapt and orchestrator, ``worker_factory``, ``evaluation_fn`` of reflexion,
  ``memory``) are read from ``RunContext.callables`` when present. A name is the keyword of the
  pattern's constructor; ``tools`` and ``config`` reach ``create_agent``'s own parameters.
- Run: the cancel token is checked before the run starts (a built-in is not interruptible between
  steps); ``context.max_seconds`` becomes ``AgentConfig.timeout_seconds``; ``context.max_steps``
  is refused (``ValueError`` naming ``max_iterations``); the interface of the context and its event
  log are passed to the agent (a swarm takes neither: its members own theirs). The run is the
  agent's sync ``run`` on a worker thread (``aoffload``: a cancelled await leaves the worker
  running, the documented cancel hazard). ``BudgetExhaustedError`` and ``AgentTimeoutError`` end
  ``aborted/budget_spent``; any other error propagates. The result is ``AgentResult.run_result()``
  with ``identity = (template digest, None)`` (a code-backed pattern has no static instance: the
  task sits in the FSM ``description`` and the per-run tools shape the instructions) and ``handle
  None``; ``detail`` gains ``pattern``.
- HITL (census D1): the executable never builds a ``HumanInTheLoop`` and never writes an approval
  grant; the host's ``hitl`` passes through unchanged and the agent's own gate stays the only
  place a grant is made. A ``requires_approval`` tool with no ``hitl`` raises ``AgentError`` at
  construction, before any model call.
- The meta-builder is single-shot (``run(task)``): ``MetaBuilderResult.is_valid`` false is
  ``failed`` (no reason), the artifact in ``outputs``, the validation errors in ``detail``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from fsm_llm.agents import _PATTERNS, create_agent
from fsm_llm.agents.definitions import (
    AgentConfig,
    MetaBuilderResult,
    config_with,
    default_agent_config,
    model_or_placeholder,
)
from fsm_llm.agents.exceptions import AgentTimeoutError, BudgetExhaustedError
from fsm_llm.aio import aoffload
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.definitions.executable import Capability, Contract, Requires
from fsm_llm.core.results import Identity, RunResult
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables.code_backed import CodeBackedExecutable

__all__ = ["PatternExecutable", "builtin_executables"]

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


# One row per key of ``fsm_llm.agents._PATTERNS`` (a test holds the key sets equal).
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

    def __init__(self, pattern: str) -> None:
        row = _ROWS[pattern]
        cls = _PATTERNS[pattern]
        self.pattern = pattern
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
            if _ROWS[self.pattern].initial_context:
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
            if context.event_log is not None:
                kwargs["event_log"] = context.event_log
        if context.max_seconds is not None:
            if config is None:
                config = (
                    AgentConfig(model=model_or_placeholder())
                    if swarm
                    else default_agent_config(llm)
                )
            config = config_with(config, timeout_seconds=context.max_seconds)
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


def builtin_executables() -> list[PatternExecutable]:
    """One executable per ``create_agent`` pattern, in ``_PATTERNS`` order."""
    return [PatternExecutable(pattern) for pattern in _PATTERNS]
