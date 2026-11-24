"""The reasoning strategies and ``reasoning.solve`` as executables (roadmap E2.4; plan ace7a8c8 D-061).

``builtin_executables()`` returns ten executables: the nine strategies, each the ``strategy`` document
``reasoning/strategies/<name>.json`` (ids ``reasoning.<name>``) run by the engine's ``FsmBodyExecutable``,
and ``reasoning.solve``, the whole engine as one code-backed executable. ``strategy_registry()`` is a
registry of the nine strategies alone: the ``TargetResolver`` the engine hands its orchestrator
(plan baa37f40 D-031). ``fsm_llm.reasoning``
registers this function as the ``reasoning`` source of the ``executable_sources`` namespace, so
``ExecutableRegistry().load_sources()`` serves them without the engine importing this package.

Interface contract:

- ``reasoning.<name>`` (kind ``strategy``, capability ``strategy``, lifecycle ``builtin``): the body
  is ``body.fsm`` of the file, the very FSM the engine pushes (``get_reasoning_fsm(name)``); the
  declaration is its ``body.strategy`` (aliases, results, solution key). The contract is ONE
  input, ``problem_statement`` (the problem arrives through context, never definition text, so
  the template AND the instance identity do not depend on it), and NO declared output (plan
  baa37f40 D-030): a run returns every non-None key of ``results``, so a refused calculator
  expression returns ``calculation_error_details`` instead of failing the run, and a caller with
  no ``constraints`` runs. The orchestrator hands a pushed strategy its own ``call`` inputs (the
  four problem keys); frame mode reads no contract. ``simple_calculator`` and ``hybrid``
  run with ``setup=register_strategy_handlers`` (their handlers are named in ``requires.handlers``);
  the other seven run standalone. The result carries both identities.
- ``reasoning.solve`` (kind ``pattern``): inputs ``problem`` (a ``str``, required), ``context`` (a
  mapping, optional) and ``preferred_reasoning_type`` (a ``str``, optional). One ``ReasoningEngine``
  is built per run (the context's interface and event log, else ``LLM_MODEL``) and closed in a
  ``finally``; the run is ``solve_problem`` on a worker thread (``aoffload``: a cancelled await
  leaves the worker running). ``outputs`` is ``{"solution": text}``, ``answer`` the same text,
  ``detail`` holds ``reasoning_trace`` and ``summary``. A ``ReasoningExecutionError`` whose cause is
  a ``RunBudgetExceededError`` ends ``aborted/budget_spent``, one whose cause is an
  ``LLMResponseError`` ends ``failed/provider_error``, any other error propagates. The ``max_steps``
  and ``max_seconds`` of the run context are the two keywords of ``solve_problem`` (plan baa37f40
  D-032): ``None`` keeps the engine's own derived step budget and no seconds limit. The cancel token
  is checked before the run starts. The identity has no instance (code-backed). ``solve_problem``
  itself stays a tuple.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from fsm_llm.aio import aoffload
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.definitions import LLMResponseError, RunBudgetExceededError
from fsm_llm.core.definitions.executable import (
    Contract,
    ExecutableError,
    Requires,
)
from fsm_llm.core.results import Identity, RunResult
from fsm_llm.core.verify.executable import load_executable
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import Executable, ExecutableRegistry
from fsm_llm.engine.executables.adapters import FsmBodyExecutable
from fsm_llm.engine.executables.code_backed import CodeBackedExecutable

from .constants import ContextKeys
from .engine import ReasoningEngine
from .exceptions import ReasoningExecutionError
from .handlers import register_strategy_handlers
from .strategy_files import STRATEGY_ORDER, load_document, strategy_id

__all__ = [
    "SOLVE_ID",
    "SolveExecutable",
    "builtin_executables",
    "strategy_builtins",
    "strategy_id",
    "strategy_registry",
]

SOLVE_ID = "reasoning.solve"


def _strategy_executable(name: str) -> FsmBodyExecutable:
    document = load_document(name)
    envelope, report = load_executable(document)
    if envelope is None or not report.is_valid:
        raise ExecutableError(
            f"built-in strategy {name!r} is not a valid executable: "
            + "; ".join(report.errors[:5])
        )
    return FsmBodyExecutable(
        envelope,
        setup=register_strategy_handlers if envelope.requires.handlers else None,
    )


class SolveExecutable(CodeBackedExecutable):
    """``reasoning.solve``: the whole reasoning engine behind one ``run(problem)``."""

    def __init__(self) -> None:
        super().__init__(
            id=SOLVE_ID,
            kind="pattern",
            impl=f"{ReasoningEngine.__module__}.{ReasoningEngine.__qualname__}",
            contract=Contract(inputs=["problem"], outputs=["solution"]),
            requires=Requires(),
            optional_inputs=("context", "preferred_reasoning_type"),
        )

    async def execute(
        self,
        inputs: Mapping[str, Any],
        objects: Mapping[str, Any],
        context: RunContext,
        identity: Identity,
    ) -> RunResult:
        problem = inputs["problem"]
        if not isinstance(problem, str):
            raise ValueError(
                f"input 'problem' must be a str, got {type(problem).__name__}"
            )
        seed = inputs.get("context")
        if seed is not None and not isinstance(seed, Mapping):
            raise ValueError("input 'context' must be a mapping")
        preferred = inputs.get("preferred_reasoning_type")
        if preferred is not None and not isinstance(preferred, str):
            raise ValueError("input 'preferred_reasoning_type' must be a str")
        initial_context = dict(seed) if seed is not None else {}
        if preferred is not None:
            initial_context[ContextKeys.PREFERRED_REASONING_TYPE] = preferred
        try:
            solution, trace = await aoffload(
                self._solve, problem, initial_context or None, context
            )
        except ReasoningExecutionError as error:
            return self._mapped(error, identity)
        return RunResult(
            status=Status.OK,
            outputs={"solution": solution},
            answer=solution,
            detail={
                "kind": self.kind,
                "reasoning_trace": trace.get("reasoning_trace"),
                "summary": trace.get("summary"),
            },
            identity=identity,
        )

    @staticmethod
    def _solve(
        problem: str, initial_context: dict[str, Any] | None, context: RunContext
    ) -> tuple[str, dict[str, Any]]:
        """One engine per run, closed in a ``finally`` (the sync body, run on a worker)."""
        kwargs: dict[str, Any] = {}
        llm = context.resolve_llm()
        if llm is not None:
            kwargs["llm_interface"] = llm
        if context.event_log is not None:
            kwargs["event_log"] = context.event_log
        engine = ReasoningEngine(**kwargs)
        try:
            return engine.solve_problem(
                problem,
                initial_context,
                max_steps=context.max_steps,
                max_seconds=context.max_seconds,
            )
        finally:
            engine.close()

    def _mapped(self, error: ReasoningExecutionError, identity: Identity) -> RunResult:
        """The result of a failed solve with a known cause; raise any other."""
        cause: BaseException | None = error
        for _ in range(8):
            cause = cause.__cause__ if cause is not None else None
            if isinstance(cause, RunBudgetExceededError):
                return self.stopped(
                    Status.ABORTED,
                    Reason.BUDGET_SPENT,
                    identity,
                    budget=cause.budget,
                    limit=cause.limit,
                    steps_done=cause.steps_done,
                )
            if isinstance(cause, LLMResponseError):
                # the type only: a provider message can carry text the log should not
                return self.stopped(
                    Status.FAILED,
                    Reason.PROVIDER_ERROR,
                    identity,
                    error=type(cause).__name__,
                )
        raise error


def builtin_executables() -> list[FsmBodyExecutable | SolveExecutable]:
    """The nine strategies in ``STRATEGY_ORDER``, then ``reasoning.solve``."""
    return [
        *(_strategy_executable(name) for name in STRATEGY_ORDER),
        SolveExecutable(),
    ]


def strategy_registry() -> ExecutableRegistry:
    """A registry holding the nine built-in strategies: the ``TargetResolver`` of the orchestrator.

    Contract: takes nothing and returns a NEW ``ExecutableRegistry`` on every call (the engine
    owns its own), holding exactly the nine ``reasoning.<name>`` strategies of
    ``STRATEGY_ORDER`` and not ``reasoning.solve`` (the engine is not its own target). The
    objects are the ones ``builtin_executables()`` builds, so a strategy resolves with the
    same identities in either place. Never raises for a packaged file.
    """
    return _with_builtins(ExecutableRegistry())


def strategy_builtins() -> list[Executable]:
    """The nine built-in strategy executables in ``STRATEGY_ORDER`` (new objects on every call)."""
    # FsmBodyExecutable satisfies the Executable protocol (its attributes are read-only)
    return [cast("Executable", _strategy_executable(n)) for n in STRATEGY_ORDER]


def _with_builtins(registry: ExecutableRegistry) -> ExecutableRegistry:
    for executable in strategy_builtins():
        registry.register(executable)
    return registry
