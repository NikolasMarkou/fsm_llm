"""The reasoning strategies and ``reasoning.solve`` as executables (plan ace7a8c8 step 23.2, D-061).

``fsm_llm.reasoning`` registers one lazy source named ``reasoning`` in the ``executable_sources``
namespace. Nine ``strategy`` documents run on the engine's ``FsmBodyExecutable`` (the real core,
a scripted model, no network); ``reasoning.solve`` is the whole engine behind one code-backed
executable. Rows hold: both identities set and independent of the problem (Principle 14), the
handler seam for the two strategies that need it, a bounded hybrid loop, the error mapping of a
failed solve, one engine per run closed in a ``finally``.
"""

from __future__ import annotations

import ast
import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, ClassVar

import pytest

import fsm_llm
from fsm_llm.core.codes import Lifecycle, Reason, Status
from fsm_llm.core.definitions import (
    FSMDefinition,
    LLMResponseError,
    RunBudgetExceededError,
)
from fsm_llm.core.definitions.executable import Capability
from fsm_llm.core.definitions.fsm import definition_digest
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import (
    DEFAULT_EXECUTABLE_MAX_STEPS,
    EXECUTABLE_SOURCES,
    Executable,
    ExecutableRegistry,
    FsmBodyExecutable,
)
from fsm_llm.llm import ModelNotConfiguredError
from fsm_llm.reasoning import ReasoningEngine, ReasoningExecutionError
from fsm_llm.reasoning import executables as module
from fsm_llm.reasoning.constants import STRATEGIES, ContextKeys, Defaults
from fsm_llm.reasoning.executables import SOLVE_ID, builtin_executables, strategy_id
from fsm_llm.reasoning.handlers import register_strategy_handlers
from fsm_llm.reasoning.utilities import get_reasoning_fsm
from fsm_llm.testing import ScriptedLLM

from .test_engine_scripted import _PROBLEM, _VALID_SCRIPT, _ScriptedLLM

REPO = Path(__file__).resolve().parents[2]
INPUTS = {
    "problem_statement": "Is Socrates mortal?",
    "problem_components": ["a"],
    "constraints": [],
    "problem_type": "logic",
}
NEEDS_HANDLERS = {"simple_calculator", "hybrid"}


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(REPO / "src")
    assert Path(module.__file__).resolve().is_relative_to(REPO / "src")


def _registry() -> ExecutableRegistry:
    registry = ExecutableRegistry()
    report = registry.load_sources()
    assert not report.failed, report.failed
    return registry


def _script(name: str, **over: Any) -> dict[str, Any]:
    """A value for every required key of the strategy FSM (a bool where the FSM wants one)."""
    keys: list[str] = []
    for state in get_reasoning_fsm(name)["states"].values():
        for key in state.get("required_context_keys") or []:
            if key not in keys:
                keys.append(key)
    data: dict[str, Any] = {
        key: False
        if key == "needs_refinement"
        else (True if key == "logical_validity" else f"value of {key}")
        for key in keys
    }
    if name == "simple_calculator":
        data["calculation_expression"] = "0.15 * 240"
    return {**data, **over}


def _llm(name: str, **over: Any) -> ScriptedLLM:
    return ScriptedLLM(extraction_data=_script(name, **over), default_response="reply")


# ---- the source --------------------------------------------------------------------------------


class TestTheSource:
    def test_the_package_registers_one_source_named_reasoning(self) -> None:
        factory = EXECUTABLE_SOURCES.resolve("reasoning")
        assert factory is fsm_llm.reasoning._builtin_executables
        assert len(factory()) == 10

    def test_the_registry_lists_nine_strategies_and_solve(self) -> None:
        registry = _registry()
        strategies = {e.id for e in registry.list(kind="strategy")}
        assert strategies == {strategy_id(name) for name in STRATEGIES}
        assert registry.get(SOLVE_ID).kind == "pattern"
        assert len(STRATEGIES) == 9

    def test_every_one_is_a_builtin_and_launchable(self) -> None:
        registry = _registry()
        ids = [strategy_id(n) for n in STRATEGIES] + [SOLVE_ID]
        launchable = {e.id for e in registry.list(launchable=True)}
        for ident in ids:
            exe = registry.get(ident)
            assert exe.lifecycle is Lifecycle.BUILTIN
            assert ident in launchable
            assert isinstance(exe, Executable)

    @pytest.mark.parametrize("name", sorted(STRATEGIES))
    def test_a_strategy_declares_its_table_row(self, name: str) -> None:
        exe = _registry().get(strategy_id(name))
        spec = STRATEGIES[name]
        assert isinstance(exe, FsmBodyExecutable)
        assert Capability.STRATEGY in exe.capabilities
        decl = exe.envelope.body.strategy  # type: ignore[union-attr]
        assert (list(decl.aliases), decl.results, decl.solution_key) == (
            list(spec.aliases),
            spec.results,
            spec.solution_key,
        )
        assert exe.contract.outputs == [spec.solution_key]
        assert exe.contract.inputs == list(INPUTS)
        # the body is the engine's own FSM, not a copy that can drift
        assert exe.envelope.body.fsm == get_reasoning_fsm(name)

    def test_only_the_two_that_need_handlers_get_the_setup_seam(self) -> None:
        registry = _registry()
        for name in STRATEGIES:
            exe = registry.get(strategy_id(name))
            assert isinstance(exe, FsmBodyExecutable)
            if name in NEEDS_HANDLERS:
                assert exe.setup is register_strategy_handlers
                assert exe.requires.handlers
            else:
                assert exe.setup is None
                assert exe.requires.handlers == []

    def test_importing_the_package_loads_no_executable_module(self) -> None:
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(REPO / 'src')!r})\n"
            "import fsm_llm.reasoning\n"
            "from fsm_llm.engine.executables import EXECUTABLE_SOURCES, ExecutableRegistry\n"
            "assert 'reasoning' in EXECUTABLE_SOURCES.list(), EXECUTABLE_SOURCES.list()\n"
            "assert 'fsm_llm.reasoning.executables' not in sys.modules\n"
            "assert 'fsm_llm.agents' not in sys.modules\n"
            "report = ExecutableRegistry().load_sources()\n"
            "assert not report.failed, report.failed\n"
            "assert 'fsm_llm.reasoning.executables' in sys.modules\n"
            "print('ok')\n"
        )
        done = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO,
            env={
                "PATH": os.environ.get("PATH", ""),
                "OLLAMA_API_BASE": "http://127.0.0.1:9",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                **(
                    {"SYSTEMROOT": os.environ["SYSTEMROOT"]}
                    if "SYSTEMROOT" in os.environ
                    else {}
                ),
            },
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "ok"

    def test_the_source_returns_built_executables_in_table_order(self) -> None:
        ids = [e.id for e in builtin_executables()]
        assert ids == [strategy_id(n) for n in STRATEGIES] + [SOLVE_ID]


# ---- the nine strategies -----------------------------------------------------------------------


class TestStrategiesRunStandalone:
    @pytest.mark.parametrize("name", sorted(STRATEGIES))
    def test_each_strategy_ends_ok_with_its_solution_and_both_identities(
        self, name: str
    ) -> None:
        exe = _registry().get(strategy_id(name))
        llm = _llm(name)
        result = exe.run(INPUTS, context=RunContext(llm_interface=llm))
        spec = STRATEGIES[name]
        assert result.status is Status.OK, (result.reason, result.detail)
        assert list(result.outputs) == [spec.solution_key]
        assert result.outputs[spec.solution_key] is not None
        assert result.identity.template_identity == exe.template_identity
        assert result.identity.instance_identity  # a strategy body is static
        assert result.handle
        assert llm.unexpected == []

    def test_a_calculator_run_computes_through_its_handler(self) -> None:
        result = (
            _registry()
            .get("reasoning.simple_calculator")
            .run(INPUTS, context=RunContext(llm_interface=_llm("simple_calculator")))
        )
        assert result.outputs == {"calculation_result": 36}

    def test_the_instance_identity_is_the_digest_of_the_engines_own_fsm(self) -> None:
        exe = _registry().get("reasoning.deductive")
        result = exe.run(INPUTS, context=RunContext(llm_interface=_llm("deductive")))
        expected = definition_digest(FSMDefinition(**get_reasoning_fsm("deductive")))
        assert result.identity.instance_identity == expected

    def test_the_problem_is_data_and_moves_no_identity(self) -> None:
        exe = _registry().get("reasoning.deductive")
        first = exe.run(INPUTS, context=RunContext(llm_interface=_llm("deductive")))
        other = {**INPUTS, "problem_statement": "A completely different problem."}
        second = exe.run(other, context=RunContext(llm_interface=_llm("deductive")))
        assert first.identity == second.identity  # Principle 14

    def test_a_missing_input_fails_before_any_call(self) -> None:
        llm = _llm("deductive")
        result = (
            _registry()
            .get("reasoning.deductive")
            .run({"problem_statement": "p"}, context=RunContext(llm_interface=llm))
        )
        assert (result.status, result.reason) == (Status.FAILED, Reason.INPUTS_MISSING)
        assert llm.call_history == []

    def test_an_undeclared_input_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="declares no input"):
            _registry().get("reasoning.deductive").run({**INPUTS, "extra": 1})

    def test_the_strategies_keep_no_state_between_runs(self) -> None:
        exe = _registry().get("reasoning.deductive")
        before = dict(vars(exe))
        for _ in range(3):
            exe.run(INPUTS, context=RunContext(llm_interface=_llm("deductive")))
        assert dict(vars(exe)) == before


class TestTheHybridLoopIsBounded:
    def test_a_run_that_always_needs_refinement_ends_within_the_counters_bound(
        self,
    ) -> None:
        exe = _registry().get("reasoning.hybrid")
        llm = _llm("hybrid", needs_refinement=True)
        result = exe.run(INPUTS, context=RunContext(llm_interface=llm))
        assert result.status is Status.OK
        assert result.detail["steps"] < DEFAULT_EXECUTABLE_MAX_STEPS
        assert result.detail["final_state"] == "finalize_hybrid"

    def test_without_the_setup_seam_the_loop_spends_the_whole_budget(self) -> None:
        """The mutant of the seam: the same body with no handlers never ends on its own."""
        exe = _registry().get("reasoning.hybrid")
        assert isinstance(exe, FsmBodyExecutable)
        bare = FsmBodyExecutable(exe.envelope)  # no setup
        llm = _llm("hybrid", needs_refinement=True)
        result = bare.run(INPUTS, context=RunContext(llm_interface=llm, max_steps=25))
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)


# ---- reasoning.solve ---------------------------------------------------------------------------


def _solve(**inputs: Any) -> dict[str, Any]:
    return {"problem": _PROBLEM, **inputs}


class _SpyEngine:
    """Stands in for ``ReasoningEngine``: records its use, answers or raises on demand."""

    instances: ClassVar[list[_SpyEngine]] = []
    outcome: ClassVar[Any] = (
        "the solution",
        {"reasoning_trace": {"steps": []}, "summary": "s"},
    )

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = 0
        self.solved: tuple[str, Any] | None = None
        self.thread: int | None = None
        type(self).instances.append(self)

    def solve_problem(self, problem: str, initial_context: Any = None) -> Any:
        self.solved = (problem, initial_context)
        self.thread = threading.get_ident()
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def spy_engine(monkeypatch: pytest.MonkeyPatch) -> type[_SpyEngine]:
    _SpyEngine.instances = []
    _SpyEngine.outcome = (
        "the solution",
        {"reasoning_trace": {"steps": []}, "summary": "s"},
    )
    monkeypatch.setattr(module, "ReasoningEngine", _SpyEngine)
    return _SpyEngine


class TestSolve:
    def test_a_whole_scripted_solve_matches_solve_problem(self) -> None:
        exe = _registry().get(SOLVE_ID)
        result = exe.run(
            _solve(), context=RunContext(llm_interface=_ScriptedLLM(_VALID_SCRIPT))
        )
        with ReasoningEngine(
            model="mock", llm_interface=_ScriptedLLM(_VALID_SCRIPT)
        ) as engine:
            solution, trace = engine.solve_problem(_PROBLEM)
        assert result.status is Status.OK
        assert (result.answer, result.outputs) == (solution, {"solution": solution})
        assert result.detail["summary"] == trace["summary"]
        assert result.detail["reasoning_trace"]["steps"]
        assert result.identity.template_identity == exe.template_identity
        assert result.identity.instance_identity is None
        assert result.handle is None

    def test_solve_problem_stays_a_tuple(self) -> None:
        with ReasoningEngine(
            model="mock", llm_interface=_ScriptedLLM(_VALID_SCRIPT)
        ) as engine:
            outcome = engine.solve_problem(_PROBLEM)
        assert isinstance(outcome, tuple) and len(outcome) == 2

    def test_the_contract_inputs(self) -> None:
        exe = _registry().get(SOLVE_ID)
        assert exe.contract.inputs == ["problem"] and exe.contract.outputs == [
            "solution"
        ]
        with pytest.raises(ValueError, match="declares no input"):
            exe.run(_solve(other=1))

    def test_optional_inputs_reach_the_engine_as_the_initial_context(
        self, spy_engine: type[_SpyEngine]
    ) -> None:
        _registry().get(SOLVE_ID).run(
            _solve(context={"hint": "x"}, preferred_reasoning_type="deductive"),
            context=RunContext(llm_interface=ScriptedLLM()),
        )
        (engine,) = spy_engine.instances
        assert engine.solved == (
            _PROBLEM,
            {"hint": "x", ContextKeys.PREFERRED_REASONING_TYPE: "deductive"},
        )
        assert engine.kwargs["llm_interface"] is not None

    def test_no_optional_input_is_no_initial_context(
        self, spy_engine: type[_SpyEngine]
    ) -> None:
        _registry().get(SOLVE_ID).run(
            _solve(), context=RunContext(llm_interface=ScriptedLLM())
        )
        assert spy_engine.instances[0].solved == (_PROBLEM, None)

    def test_the_log_and_the_factory_reach_the_engine(
        self, spy_engine: type[_SpyEngine], tmp_path: Path
    ) -> None:
        made: list[ScriptedLLM] = []

        def factory() -> ScriptedLLM:
            made.append(ScriptedLLM())
            return made[-1]

        log = str(tmp_path / "logs")
        _registry().get(SOLVE_ID).run(
            _solve(), context=RunContext(llm_factory=factory, event_log=log)
        )
        (engine,) = spy_engine.instances
        assert len(made) == 1 and engine.kwargs["llm_interface"] is made[0]
        assert engine.kwargs["event_log"] == log

    def test_one_engine_per_run_closed_in_a_finally(
        self, spy_engine: type[_SpyEngine]
    ) -> None:
        exe = _registry().get(SOLVE_ID)
        context = RunContext(llm_interface=ScriptedLLM())
        exe.run(_solve(), context=context)
        spy_engine.outcome = RuntimeError("a bug in the engine")
        with pytest.raises(RuntimeError, match="a bug"):
            exe.run(_solve(), context=context)
        assert [e.closed for e in spy_engine.instances] == [1, 1]
        assert spy_engine.instances[0] is not spy_engine.instances[1]

    def test_arun_hops_to_a_worker_and_run_stays_on_the_calling_thread(
        self, spy_engine: type[_SpyEngine]
    ) -> None:
        exe = _registry().get(SOLVE_ID)
        context = RunContext(llm_interface=ScriptedLLM())
        exe.run(_solve(), context=context)
        assert spy_engine.instances[0].thread == threading.get_ident()

        async def go() -> int:
            await exe.arun(_solve(), context=context)
            return threading.get_ident()

        loop_thread = asyncio.run(go())
        assert spy_engine.instances[1].thread != loop_thread

    def test_a_spent_budget_cause_ends_aborted_budget_spent(
        self, spy_engine: type[_SpyEngine]
    ) -> None:
        failure = ReasoningExecutionError("did not finish")
        failure.__cause__ = RunBudgetExceededError("steps", 170, 170)
        spy_engine.outcome = failure
        result = (
            _registry()
            .get(SOLVE_ID)
            .run(_solve(), context=RunContext(llm_interface=ScriptedLLM()))
        )
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert result.detail["limit"] == 170 and result.detail["steps_done"] == 170
        assert result.identity.instance_identity is None
        assert spy_engine.instances[0].closed == 1

    def test_a_provider_cause_ends_failed_provider_error_with_the_type_only(
        self, spy_engine: type[_SpyEngine]
    ) -> None:
        failure = ReasoningExecutionError("failed")
        failure.__cause__ = LLMResponseError("secret text from the provider")
        spy_engine.outcome = failure
        result = (
            _registry()
            .get(SOLVE_ID)
            .run(_solve(), context=RunContext(llm_interface=ScriptedLLM()))
        )
        assert (result.status, result.reason) == (Status.FAILED, Reason.PROVIDER_ERROR)
        assert result.detail["error"] == "LLMResponseError"
        assert "secret" not in repr(result.detail)

    def test_a_cause_deeper_in_the_chain_is_found(
        self, spy_engine: type[_SpyEngine]
    ) -> None:
        middle = RuntimeError("handler failed")
        middle.__cause__ = LLMResponseError("down")
        failure = ReasoningExecutionError("failed")
        failure.__cause__ = middle
        spy_engine.outcome = failure
        result = (
            _registry()
            .get(SOLVE_ID)
            .run(_solve(), context=RunContext(llm_interface=ScriptedLLM()))
        )
        assert result.reason is Reason.PROVIDER_ERROR

    def test_any_other_reasoning_error_propagates(
        self, spy_engine: type[_SpyEngine]
    ) -> None:
        spy_engine.outcome = ReasoningExecutionError("a bug, not a budget")
        with pytest.raises(ReasoningExecutionError, match="a bug"):
            _registry().get(SOLVE_ID).run(
                _solve(), context=RunContext(llm_interface=ScriptedLLM())
            )

    @pytest.mark.parametrize("limit", [{"max_steps": 5}, {"max_seconds": 5.0}])
    def test_a_budget_the_engine_cannot_honour_is_refused(
        self, spy_engine: type[_SpyEngine], limit: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError, match="MAX_SOLVE_STEPS"):
            _registry().get(SOLVE_ID).run(
                _solve(), context=RunContext(llm_interface=ScriptedLLM(), **limit)
            )
        assert spy_engine.instances == []
        assert Defaults.MAX_SOLVE_STEPS > 0

    def test_a_set_cancel_token_ends_aborted_cancelled_before_the_engine(
        self, spy_engine: type[_SpyEngine]
    ) -> None:
        context = RunContext(llm_interface=ScriptedLLM())
        context.cancel.cancel()
        result = _registry().get(SOLVE_ID).run(_solve(), context=context)
        assert (result.status, result.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert spy_engine.instances == []

    def test_no_model_anywhere_propagates_the_resolvers_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLM_MODEL", raising=False)
        with pytest.raises(ModelNotConfiguredError):
            _registry().get(SOLVE_ID).run(_solve())

    @pytest.mark.parametrize(
        "bad",
        [
            {"problem": 3},
            {"problem": "p", "context": [1]},
            {"problem": "p", "preferred_reasoning_type": 3},
        ],
    )
    def test_a_wrongly_typed_input_is_a_value_error(
        self, bad: dict[str, Any], spy_engine: type[_SpyEngine]
    ) -> None:
        with pytest.raises(ValueError):
            _registry().get(SOLVE_ID).run(
                bad, context=RunContext(llm_interface=ScriptedLLM())
            )
        assert spy_engine.instances == []


class TestTheModule:
    def test_it_does_not_grow_the_engine_module(self) -> None:
        lines = (
            (REPO / "src/fsm_llm/reasoning/engine.py")
            .read_text(encoding="utf-8")
            .count("\n")
        )
        assert lines <= 800

    def test_it_imports_nothing_from_agents(self) -> None:
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and not node.level:
                assert not (node.module or "").startswith("fsm_llm.agents")
