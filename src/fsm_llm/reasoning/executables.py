"""The reasoning strategies and ``reasoning.solve`` as executables (roadmap E2.4; plan ace7a8c8 D-061).

``builtin_executables()`` returns ten executables: the nine strategies of the one ``STRATEGIES``
table as ``strategy`` documents (ids ``reasoning.<name>``) run by the engine's ``FsmBodyExecutable``,
and ``reasoning.solve``, the whole engine as one code-backed executable. ``fsm_llm.reasoning``
registers this function as the ``reasoning`` source of the ``executable_sources`` namespace, so
``ExecutableRegistry().load_sources()`` serves them without the engine importing this package.

Interface contract:

- ``reasoning.<name>`` (kind ``strategy``, capability ``strategy``, lifecycle ``builtin``): the body
  is ``get_reasoning_fsm(name)``, the very FSM the engine pushes; the declaration is
  ``STRATEGIES[name]`` (aliases, results, solution key). Contract inputs are the four keys the engine
  hands a strategy (``problem_statement``, ``problem_components``, ``constraints``,
  ``problem_type``; a caller with no components passes ``[]``): the problem arrives through context,
  never definition text, so the template AND the instance identity do not depend on it. The contract
  output is the solution key (a declared output is required at the terminal state, and only declared
  outputs are returned; the other result keys wait for E3.3). ``simple_calculator`` and ``hybrid``
  run with ``setup=register_strategy_handlers`` (their handlers are named in ``requires.handlers``);
  the other seven run standalone. The result carries both identities.
- ``reasoning.solve`` (kind ``pattern``): inputs ``problem`` (a ``str``, required), ``context`` (a
  mapping, optional) and ``preferred_reasoning_type`` (a ``str``, optional). One ``ReasoningEngine``
  is built per run (the context's interface and event log, else ``LLM_MODEL``) and closed in a
  ``finally``; the run is ``solve_problem`` on a worker thread (``aoffload``: a cancelled await
  leaves the worker running). ``outputs`` is ``{"solution": text}``, ``answer`` the same text,
  ``detail`` holds ``reasoning_trace`` and ``summary``. A ``ReasoningExecutionError`` whose cause is
  a ``RunBudgetExceededError`` ends ``aborted/budget_spent``, one whose cause is an
  ``LLMResponseError`` ends ``failed/provider_error``, any other error propagates. ``max_steps`` and
  ``max_seconds`` are refused (the engine has its own budget, ``Defaults.MAX_SOLVE_STEPS``; E3.3
  maps them). The cancel token is checked before the run starts. The identity has no instance
  (code-backed). ``solve_problem`` itself stays a tuple.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
from fsm_llm.engine.executables.adapters import FsmBodyExecutable
from fsm_llm.engine.executables.code_backed import CodeBackedExecutable

from .constants import STRATEGIES, ContextKeys, Defaults, HandlerNames
from .engine import ReasoningEngine
from .exceptions import ReasoningExecutionError
from .handlers import register_strategy_handlers
from .utilities import get_reasoning_fsm

__all__ = ["SOLVE_ID", "SolveExecutable", "builtin_executables", "strategy_id"]

SOLVE_ID = "reasoning.solve"

#: The four keys the engine hands a strategy FSM (``_StrategyStack._push``).
_STRATEGY_INPUTS = [
    ContextKeys.PROBLEM_STATEMENT,
    ContextKeys.PROBLEM_COMPONENTS,
    ContextKeys.CONSTRAINTS,
    ContextKeys.PROBLEM_TYPE,
]

#: The strategies whose FSM needs handlers that only an engine registers, and their names.
_STRATEGY_HANDLERS: dict[str, list[str]] = {
    "simple_calculator": [
        HandlerNames.CALCULATOR_EXECUTOR,
        HandlerNames.CALCULATOR_FINALIZER,
    ],
    "hybrid": [HandlerNames.HYBRID_KEY_CLEARER, HandlerNames.HYBRID_LOOP_COUNTER],
}


def strategy_id(name: str) -> str:
    """The executable id of the strategy *name* (a ``ReasoningType`` value)."""
    return f"reasoning.{name}"


def _strategy_executable(name: str) -> FsmBodyExecutable:
    spec = STRATEGIES[name]
    handlers = _STRATEGY_HANDLERS.get(name, [])
    document = {
        "schema": "fsm-llm/executable",
        "kind": "strategy",
        "id": strategy_id(name),
        "name": name,
        "description": spec.description,
        "contract": {"inputs": list(_STRATEGY_INPUTS), "outputs": [spec.solution_key]},
        "requires": {"handlers": list(handlers)},
        "capabilities": ["strategy"],
        "lifecycle": "builtin",
        "body": {
            "fsm": get_reasoning_fsm(name),
            "strategy": {
                "aliases": list(spec.aliases),
                "results": dict(spec.results),
                "solution_key": spec.solution_key,
            },
        },
    }
    envelope, report = load_executable(document)
    if envelope is None or not report.is_valid:
        raise ExecutableError(
            f"built-in strategy {name!r} is not a valid executable: "
            + "; ".join(report.errors[:5])
        )
    return FsmBodyExecutable(
        envelope, setup=register_strategy_handlers if handlers else None
    )


class SolveExecutable(CodeBackedExecutable):
    """``reasoning.solve``: the whole reasoning engine behind one ``run(problem)``."""

    refused_budgets = ("max_steps", "max_seconds")  # the engine has its own budget

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
        if context.max_steps is not None or context.max_seconds is not None:
            raise ValueError(
                "reasoning.solve takes no max_steps or max_seconds: the engine has its "
                f"own budget (Defaults.MAX_SOLVE_STEPS = {Defaults.MAX_SOLVE_STEPS})"
            )
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
            return engine.solve_problem(problem, initial_context)
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
    """The nine strategies in ``STRATEGIES`` order, then ``reasoning.solve``."""
    return [*(_strategy_executable(name) for name in STRATEGIES), SolveExecutable()]
