"""The built-in agent patterns as executables (plan ace7a8c8 step 23.1, roadmap E2.4, D-061).

``fsm_llm.agents`` registers one lazy source named ``agents`` in the ``executable_sources``
namespace; ``ExecutableRegistry().load_sources()`` then serves the 18 patterns by id. Rows run the
REAL agent classes with scripted doubles (no network), plus a spy on ``create_agent`` for the
per-pattern mapping of host objects. Census D1 rows: an executable never supplies an approval
grant (named mutant: an executable that builds an auto-approving ``hitl``). Census D4: importing
the package loads the namespace module, never ``agents.executables``.
"""

from __future__ import annotations

import ast
import asyncio
import os
import re
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from fsm_llm.agents import create_agent
from fsm_llm.agents import executables as module
from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.definitions import (
    AgentConfig,
    AgentResult,
    ChainStep,
    EvaluationResult,
)
from fsm_llm.agents.exceptions import AgentTimeoutError, BudgetExhaustedError
from fsm_llm.agents.executables import PatternExecutable, builtin_executables
from fsm_llm.core.codes import Lifecycle, Reason, Status
from fsm_llm.core.definitions import (
    AgentError,
    DataExtractionResponse,
    FieldExtractionResponse,
    ResponseGenerationResponse,
)
from fsm_llm.core.definitions.executable import Capability, ExecutableError
from fsm_llm.core.names import NameRegistry
from fsm_llm.engine.context import RunBudget, RunCancelledError, RunContext
from fsm_llm.engine.executables import (
    EXECUTABLE_SOURCES,
    Executable,
    ExecutableRegistry,
)
from fsm_llm.engine.executables.namespaces import PATTERN_AGENTS
from fsm_llm.llm import LLMInterface
from fsm_llm.policy import HumanInTheLoop
from fsm_llm.testing import ScriptedLLM
from fsm_llm.tools import ToolRegistry

REPO = Path(__file__).resolve().parents[2]
TASK = "What is the capital of Australia?"

#: The patterns that need no host object (launchable) and those that do.
LAUNCHABLE = {
    "debate",
    "self_consistency",
    "adapt",
    "plan_execute",
    "orchestrator",
    "meta_builder",
}
NEEDS_OBJECTS = {
    "react": ["tools"],
    "rewoo": ["tools"],
    "parallel_react": ["tools"],
    "native_fc": ["tools"],
    "verified_react": ["tools"],
    "auto_memory": ["tools"],
    "reflexion": ["tools"],
    "reasoning_react": ["tools"],
    "prompt_chain": ["chain"],
    "evaluator_optimizer": ["evaluation_fn"],
    "maker_checker": ["maker_instructions", "checker_instructions"],
    "swarm": ["agents", "entry_agent"],
}


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(REPO / "src")
    assert Path(module.__file__).resolve().is_relative_to(REPO / "src")


# ---- doubles and helpers ----------------------------------------------------------------------


class _Filler(LLMInterface):
    """Answers every field from ``values`` (else ``"text"``) and counts every call."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.model = "mock-model"
        self.values = dict(values or {})
        self.calls: list[str] = []

    def extract_field(self, request: Any) -> FieldExtractionResponse:
        self.calls.append("extract_field")
        value = self.values.get(request.field_name, "text")
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0,
            reasoning="m",
            is_valid=True,
        )

    def extract_bulk_data(self, request: Any) -> DataExtractionResponse:
        self.calls.append("extract_bulk_data")
        names = re.findall(r'- "(\w+)"', request.system_prompt)
        return DataExtractionResponse(
            extracted_data={n: self.values.get(n, "text") for n in names}
        )

    def generate_response(self, request: Any) -> ResponseGenerationResponse:
        self.calls.append("generate_response")
        return ResponseGenerationResponse(
            message="ok", message_type="response", reasoning="m"
        )


_FILL = {
    "consensus_reached": True,
    "should_terminate": True,
    "final_answer": "done",
    "all_collected": True,
    "subtasks": ["a"],
    "plan_steps": ["one"],
    "tool_name": "noop",
    "tool_input": {},
}


def _callables(**named: Any) -> NameRegistry[Any]:
    registry: NameRegistry[Any] = NameRegistry("callables")
    for name, value in named.items():
        registry.register(name, value)
    return registry


def _tools(executions: list[str] | None = None, *, gated: bool = False) -> ToolRegistry:
    registry = ToolRegistry()
    sink = executions if executions is not None else []

    def noop(params: dict[str, Any]) -> str:
        sink.append("noop")
        return "noop-result"

    registry.register_function(
        noop, name="noop", description="No-op tool", requires_approval=gated
    )
    return registry


def _registry() -> ExecutableRegistry:
    registry = ExecutableRegistry()
    report = registry.load_sources()
    assert not report.failed, report.failed
    return registry


def _exe(pattern: str) -> Executable:
    return _registry().get(pattern)


def _full_objects() -> dict[str, dict[str, Any]]:
    """Every required and optional host object of every pattern, by pattern."""
    chain = [
        ChainStep(
            step_id="s",
            name="n",
            extraction_instructions="e",
            response_instructions="r",
        )
    ]
    return {
        "react": {
            "tools": _tools(),
            "hitl": HumanInTheLoop(),
            "config": AgentConfig(model="m"),
        },
        "rewoo": {"tools": _tools(), "config": AgentConfig(model="m")},
        "debate": {"config": AgentConfig(model="m")},
        "plan_execute": {"tools": _tools(), "config": AgentConfig(model="m")},
        "prompt_chain": {"chain": chain, "config": AgentConfig(model="m")},
        "self_consistency": {"config": AgentConfig(model="m")},
        "orchestrator": {
            "tools": _tools(),
            "worker_factory": lambda: None,
            "config": AgentConfig(model="m"),
        },
        "adapt": {"tools": _tools(), "config": AgentConfig(model="m")},
        "evaluator_optimizer": {
            "evaluation_fn": lambda out, ctx: EvaluationResult(passed=True),
            "config": AgentConfig(model="m"),
        },
        "maker_checker": {
            "maker_instructions": "make",
            "checker_instructions": "check",
            "config": AgentConfig(model="m"),
        },
        "reflexion": {
            "tools": _tools(),
            "hitl": HumanInTheLoop(),
            "config": AgentConfig(model="m"),
            "evaluation_fn": lambda ctx: EvaluationResult(passed=True),
        },
        "meta_builder": {"config": AgentConfig(model="m")},
        "swarm": {
            "agents": {"a": object()},
            "entry_agent": "a",
            "config": AgentConfig(model="m"),
            "memory": object(),
        },
        "parallel_react": {"tools": _tools(), "config": AgentConfig(model="m")},
        "native_fc": {"tools": _tools(), "config": AgentConfig(model="m")},
        "verified_react": {
            "tools": _tools(),
            "hitl": HumanInTheLoop(),
            "config": AgentConfig(model="m"),
        },
        "auto_memory": {
            "tools": _tools(),
            "hitl": HumanInTheLoop(),
            "config": AgentConfig(model="m"),
            "memory": object(),
        },
        "reasoning_react": {
            "tools": _tools(),
            "hitl": HumanInTheLoop(),
            "config": AgentConfig(model="m"),
        },
    }


class _SpyAgent:
    """What ``create_agent`` returns in the mapping rows: records its run, answers ok."""

    def __init__(self, pattern: str, tools: Any, kwargs: dict[str, Any]) -> None:
        self.pattern, self.tools, self.kwargs = pattern, tools, kwargs
        self.thread: int | None = None
        self.ran: tuple[Any, ...] | None = None

    def run(self, *args: Any) -> AgentResult:
        self.thread = threading.get_ident()
        self.ran = args
        return AgentResult(answer="spied", stop_reason="answered")


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> list[_SpyAgent]:
    made: list[_SpyAgent] = []

    def fake(pattern: str, tools: Any = None, **kwargs: Any) -> _SpyAgent:
        agent = _SpyAgent(pattern, tools, kwargs)
        made.append(agent)
        return agent

    monkeypatch.setattr(module, "create_agent", fake)
    return made


# ---- the source and the table -----------------------------------------------------------------


class TestTheSource:
    def test_the_agents_package_registers_one_source_named_agents(self) -> None:
        factory = EXECUTABLE_SOURCES.resolve("agents")
        assert factory is fsm_llm.agents._builtin_executables
        assert [type(e) for e in factory()] == [PatternExecutable] * 18

    def test_the_registry_lists_the_eighteen_pattern_ids(self) -> None:
        registry = _registry()
        assert {e.id for e in registry.list(kind="pattern")} >= set(
            PATTERN_AGENTS.list()
        )
        assert {e.id for e in registry.list()} >= set(PATTERN_AGENTS.list())
        for pattern in PATTERN_AGENTS.list():
            assert registry.get(pattern).id == pattern

    def test_the_table_and_create_agent_hold_the_same_patterns(self) -> None:
        assert set(module._ROWS) == set(PATTERN_AGENTS.list())
        assert [e.id for e in builtin_executables()] == [
            name for name, _ in PATTERN_AGENTS.items()
        ]

    def test_launchable_is_false_exactly_for_the_twelve_that_need_objects(
        self,
    ) -> None:
        registry = _registry()
        ids = set(PATTERN_AGENTS.list())
        launchable = {e.id for e in registry.list(launchable=True)} & ids
        refused = {e.id for e in registry.list(launchable=False)} & ids
        assert launchable == LAUNCHABLE
        assert refused == set(NEEDS_OBJECTS)

    @pytest.mark.parametrize("pattern", sorted(NEEDS_OBJECTS))
    def test_requires_callables_names_the_required_objects(self, pattern: str) -> None:
        assert _exe(pattern).requires.callables == NEEDS_OBJECTS[pattern]

    def test_the_tool_patterns_claim_tools_and_the_rest_claim_none(self) -> None:
        tool_patterns = {
            "react", "rewoo", "parallel_react", "native_fc",
            "verified_react", "auto_memory", "reflexion", "reasoning_react",
        }  # fmt: skip
        for pattern in PATTERN_AGENTS.list():
            exe = _exe(pattern)
            expected = "some" if pattern in tool_patterns else "none"
            assert exe.requires.tools == expected, pattern
            assert (Capability.TOOLS in exe.capabilities) == (pattern in tool_patterns)

    def test_the_meta_builder_is_conversational_and_single_shot(self) -> None:
        exe = _exe("meta_builder")
        assert Capability.CONVERSATIONAL in exe.capabilities
        assert exe.contract.outputs == ["artifact", "artifact_type"]
        with pytest.raises(ValueError, match="declares no input"):
            exe.run({"task": "t", "initial_context": {}})

    def test_every_pattern_is_a_builtin_with_its_class_as_impl(self) -> None:
        for pattern, cls in PATTERN_AGENTS.items():
            exe = _exe(pattern)
            assert exe.kind == "pattern"
            assert exe.lifecycle is Lifecycle.BUILTIN
            assert exe.impl == f"{cls.__module__}.{cls.__qualname__}"  # type: ignore[attr-defined]
            assert exe.contract.inputs == ["task"]
            assert isinstance(exe, Executable)

    def test_template_identities_are_distinct_and_stable(self) -> None:
        first = {e.id: e.template_identity for e in builtin_executables()}
        again = {e.id: e.template_identity for e in builtin_executables()}
        assert first == again
        assert len(set(first.values())) == 18

    def test_importing_the_package_loads_no_executable_module(self) -> None:
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(REPO / 'src')!r})\n"
            "import fsm_llm.agents\n"
            # the docstring of the package says exactly this (D-070): the engine package loads, the
            # executables of the package itself do not
            "assert 'fsm_llm.engine.executables' in sys.modules\n"
            "from fsm_llm.engine.executables import EXECUTABLE_SOURCES, ExecutableRegistry\n"
            "assert 'agents' in EXECUTABLE_SOURCES.list(), EXECUTABLE_SOURCES.list()\n"
            "assert 'fsm_llm.agents.executables' not in sys.modules\n"
            "assert 'fsm_llm.reasoning.executables' not in sys.modules\n"
            "report = ExecutableRegistry().load_sources()\n"
            "assert not report.failed, report.failed\n"
            "assert 'fsm_llm.agents.executables' in sys.modules\n"
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


# ---- runs: the six that need nothing ----------------------------------------------------------


def _direct(pattern: str, llm: LLMInterface, **kw: Any) -> AgentResult:
    return create_agent(pattern, llm_interface=llm, **kw).run(TASK)


def _inputs_only_llm(pattern: str) -> LLMInterface:
    if pattern == "self_consistency":
        return ScriptedLLM(default_response="Canberra")
    return _Filler(_FILL)


class TestRunsFromInputsOnly:
    @pytest.mark.parametrize(
        "pattern",
        ["debate", "self_consistency", "adapt", "plan_execute", "orchestrator"],
    )
    def test_the_result_is_the_agents_own_run_result(self, pattern: str) -> None:
        result = _exe(pattern).run(
            {"task": TASK}, context=RunContext(llm_interface=_inputs_only_llm(pattern))
        )
        expected = _direct(pattern, _inputs_only_llm(pattern)).run_result()
        assert (result.status, result.reason) == (expected.status, expected.reason)
        assert (result.answer, result.outputs) == (expected.answer, expected.outputs)
        for key, value in expected.detail.items():
            assert result.detail[key] == value
        assert result.detail["pattern"] == pattern
        assert result.detail["kind"] == "pattern"

    @pytest.mark.parametrize(
        "pattern",
        ["debate", "self_consistency", "adapt", "plan_execute", "orchestrator"],
    )
    def test_the_identity_has_a_template_digest_and_no_instance(
        self, pattern: str
    ) -> None:
        exe = _exe(pattern)
        result = exe.run(
            {"task": TASK}, context=RunContext(llm_interface=_inputs_only_llm(pattern))
        )
        assert result.identity.template_identity == exe.template_identity
        assert result.identity.instance_identity is None
        assert result.handle is None

    def test_a_successful_debate_and_self_consistency_end_ok(self) -> None:
        for pattern in ("debate", "self_consistency"):
            result = _exe(pattern).run(
                {"task": TASK},
                context=RunContext(llm_interface=_inputs_only_llm(pattern)),
            )
            assert result.status is Status.OK and result.success, pattern
        assert result.answer == "Canberra"

    def test_the_initial_context_reaches_the_agent(self, spy: list[_SpyAgent]) -> None:
        _exe("debate").run(
            {"task": TASK, "initial_context": {"hint": "x"}},
            context=RunContext(llm_interface=_Filler()),
        )
        assert spy[0].ran == (TASK, {"hint": "x"})

    def test_a_non_mapping_initial_context_and_a_non_str_task_are_refused(
        self, spy: list[_SpyAgent]
    ) -> None:
        with pytest.raises(ValueError, match="mapping"):
            _exe("debate").run({"task": TASK, "initial_context": [1]})
        with pytest.raises(ValueError, match="'task' must be a str"):
            _exe("debate").run({"task": 3})
        assert spy == []

    def test_an_internal_prefixed_top_level_input_is_refused(
        self, spy: list[_SpyAgent]
    ) -> None:
        with pytest.raises(ValueError):
            _exe("debate").run({"task": TASK, ContextKeys.DRIVER_APPROVAL: True})
        assert spy == []

    def test_a_missing_task_is_a_failed_result_with_no_call(
        self, spy: list[_SpyAgent]
    ) -> None:
        result = _exe("debate").run({})
        assert (result.status, result.reason) == (Status.FAILED, Reason.INPUTS_MISSING)
        assert spy == []


# ---- runs: the twelve that need host objects --------------------------------------------------


class TestRunsWithCallables:
    def test_react_runs_with_a_tool_registry(self) -> None:
        executions: list[str] = []
        llm = _Filler({**_FILL, "should_terminate": False})
        context = RunContext(
            llm_interface=llm,
            callables=_callables(
                tools=_tools(executions),
                config=AgentConfig(model="m", max_iterations=3),
            ),
        )
        result = _exe("react").run({"task": TASK}, context=context)
        assert executions == ["noop", "noop"]
        assert result.detail["tools_used"] == ["noop"]
        assert result.detail["pattern"] == "react"
        assert result.identity.instance_identity is None

    def test_react_terminates_ok_on_a_final_answer(self) -> None:
        llm = _Filler(_FILL)
        context = RunContext(llm_interface=llm, callables=_callables(tools=_tools()))
        result = _exe("react").run({"task": TASK}, context=context)
        assert result.status is Status.OK
        assert result.detail["stop_reason"] == "answered"

    def test_prompt_chain_runs_with_a_chain(self) -> None:
        chain = [
            ChainStep(
                step_id="s1",
                name="one",
                extraction_instructions="Extract text.",
                response_instructions="Say it.",
            )
        ]
        result = _exe("prompt_chain").run(
            {"task": TASK},
            context=RunContext(
                llm_interface=_Filler(_FILL), callables=_callables(chain=chain)
            ),
        )
        assert result.status is Status.OK
        assert result.detail["pattern"] == "prompt_chain"

    def test_evaluator_optimizer_runs_with_an_evaluation_function(self) -> None:
        seen: list[str] = []

        def evaluate(output: str, context: dict[str, Any]) -> EvaluationResult:
            seen.append(output)
            return EvaluationResult(passed=True, score=1.0, feedback="ok")

        result = _exe("evaluator_optimizer").run(
            {"task": TASK},
            context=RunContext(
                llm_interface=_Filler(_FILL),
                callables=_callables(evaluation_fn=evaluate),
            ),
        )
        assert result.status is Status.OK
        assert seen, "the host's evaluation function was called"

    def test_maker_checker_runs_with_the_two_instruction_strings(self) -> None:
        result = _exe("maker_checker").run(
            {"task": TASK},
            context=RunContext(
                llm_interface=_Filler(_FILL),
                callables=_callables(
                    maker_instructions="Make it.", checker_instructions="Check it."
                ),
            ),
        )
        assert result.detail["pattern"] == "maker_checker"
        assert isinstance(result.status, Status)

    def test_swarm_runs_its_members_and_takes_no_interface_of_its_own(self) -> None:
        member = create_agent("debate", llm_interface=_Filler(_FILL))
        context = RunContext(
            callables=_callables(agents={"lead": member}, entry_agent="lead")
        )
        result = _exe("swarm").run({"task": TASK}, context=context)
        assert result.detail["pattern"] == "swarm"
        assert isinstance(result.status, Status)

    @pytest.mark.parametrize("pattern", PATTERN_AGENTS.list())
    def test_each_pattern_gets_exactly_its_objects(
        self, pattern: str, spy: list[_SpyAgent]
    ) -> None:
        objects = _full_objects()[pattern]
        llm = _Filler()
        _exe(pattern).run(
            {"task": TASK},
            context=RunContext(llm_interface=llm, callables=_callables(**objects)),
        )
        (agent,) = spy
        assert agent.pattern == pattern
        assert agent.tools is objects.get("tools")
        expected = {k: v for k, v in objects.items() if k not in ("tools", "config")}
        got = dict(agent.kwargs)
        config = got.pop("config", None)
        assert config is objects["config"]  # no max_seconds: handed on as given
        if pattern == "swarm":
            assert "llm_interface" not in got
        else:
            assert got.pop("llm_interface") is llm
        assert got == expected  # the same objects, nothing added, nothing renamed

    def test_a_pattern_is_built_per_run_and_the_executable_keeps_no_state(
        self, spy: list[_SpyAgent]
    ) -> None:
        exe = _exe("debate")
        before = dict(vars(exe))
        context = RunContext(llm_interface=_Filler())
        exe.run({"task": "one"}, context=context)
        exe.run({"task": "two"}, context=context)
        assert len(spy) == 2 and spy[0] is not spy[1]
        assert dict(vars(exe)) == before


# ---- before any model call ---------------------------------------------------------------------


class TestUnresolvedCallables:
    @pytest.mark.parametrize("pattern", sorted(NEEDS_OBJECTS))
    def test_an_unresolved_name_raises_before_any_call(
        self, pattern: str, spy: list[_SpyAgent]
    ) -> None:
        llm = _Filler(_FILL)
        with pytest.raises(ExecutableError) as caught:
            _exe(pattern).run({"task": TASK}, context=RunContext(llm_interface=llm))
        for name in NEEDS_OBJECTS[pattern]:
            assert name in str(caught.value)
        assert llm.calls == []  # zero double calls
        assert spy == []  # nothing was even built

    def test_one_missing_name_of_two_is_named_alone(self) -> None:
        llm = _Filler()
        context = RunContext(
            llm_interface=llm, callables=_callables(maker_instructions="m")
        )
        with pytest.raises(ExecutableError) as caught:
            _exe("maker_checker").run({"task": TASK}, context=context)
        assert "checker_instructions" in str(caught.value)
        assert llm.calls == []

    def test_a_scripted_double_sees_zero_calls(self) -> None:
        llm = ScriptedLLM()
        with pytest.raises(ExecutableError):
            _exe("react").run({"task": TASK}, context=RunContext(llm_interface=llm))
        assert llm.call_history == [] and llm.unexpected == []

    def test_the_async_name_raises_the_same_way(self) -> None:
        llm = _Filler()
        with pytest.raises(ExecutableError):
            asyncio.run(
                _exe("react").arun(
                    {"task": TASK}, context=RunContext(llm_interface=llm)
                )
            )
        assert llm.calls == []


# ---- budgets, cancel, errors -------------------------------------------------------------------


class TestBudgetsAndErrors:
    def test_max_steps_is_refused_naming_max_iterations(
        self, spy: list[_SpyAgent]
    ) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            _exe("debate").run(
                {"task": TASK}, context=RunContext(llm_interface=_Filler(), max_steps=5)
            )
        assert spy == []

    def test_max_seconds_becomes_the_agents_timeout(self, spy: list[_SpyAgent]) -> None:
        context = RunContext(llm_interface=_Filler(), max_seconds=7.5)
        _exe("debate").run({"task": TASK}, context=context)
        config = spy[0].kwargs["config"]
        assert isinstance(config, AgentConfig) and config.timeout_seconds == 7.5

    def test_max_seconds_keeps_the_hosts_other_config_fields(
        self, spy: list[_SpyAgent]
    ) -> None:
        mine = AgentConfig(model="m", max_iterations=3, timeout_seconds=99.0)
        context = RunContext(
            llm_interface=_Filler(), max_seconds=7.5, callables=_callables(config=mine)
        )
        _exe("debate").run({"task": TASK}, context=context)
        config = spy[0].kwargs["config"]
        assert (config.max_iterations, config.timeout_seconds) == (3, 7.5)
        assert mine.timeout_seconds == 99.0  # the host's object is not changed

    def test_a_run_budget_never_widens_the_hosts_own_timeout(
        self, spy: list[_SpyAgent]
    ) -> None:
        # D-068: a host that capped its agent at 30 s keeps the cap under a 600 s run budget
        mine = AgentConfig(model="m", max_iterations=3, timeout_seconds=30.0)
        context = RunContext(
            llm_interface=_Filler(),
            max_seconds=600.0,
            callables=_callables(config=mine),
        )
        _exe("debate").run({"task": TASK}, context=context)
        config = spy[0].kwargs["config"]
        assert (config.max_iterations, config.timeout_seconds) == (3, 30.0)
        assert mine.timeout_seconds == 30.0

    def test_a_run_budget_equal_to_the_hosts_timeout_changes_nothing(
        self, spy: list[_SpyAgent]
    ) -> None:
        mine = AgentConfig(model="m", timeout_seconds=30.0)
        context = RunContext(
            llm_interface=_Filler(), max_seconds=30.0, callables=_callables(config=mine)
        )
        _exe("debate").run({"task": TASK}, context=context)
        assert spy[0].kwargs["config"].timeout_seconds == 30.0

    def test_a_swarm_with_max_seconds_builds_a_config_without_a_model(
        self, spy: list[_SpyAgent], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLM_MODEL", raising=False)
        context = RunContext(
            max_seconds=3.0, callables=_callables(agents={"a": 1}, entry_agent="a")
        )
        _exe("swarm").run({"task": TASK}, context=context)
        assert spy[0].kwargs["config"].timeout_seconds == 3.0

    @pytest.mark.parametrize(
        "error",
        [BudgetExhaustedError("iterations", 5), AgentTimeoutError(2.0)],
    )
    def test_a_spent_budget_ends_aborted_budget_spent(
        self, error: Exception, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Spent(_SpyAgent):
            def run(self, *args: Any) -> AgentResult:
                raise error

        monkeypatch.setattr(
            module, "create_agent", lambda p, t=None, **kw: _Spent(p, t, kw)
        )
        result = _exe("debate").run(
            {"task": TASK}, context=RunContext(llm_interface=_Filler())
        )
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert result.detail["error"] == type(error).__name__
        assert result.identity.instance_identity is None

    def test_another_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Broken(_SpyAgent):
            def run(self, *args: Any) -> AgentResult:
                raise AgentError("a bug")

        monkeypatch.setattr(
            module, "create_agent", lambda p, t=None, **kw: _Broken(p, t, kw)
        )
        with pytest.raises(AgentError, match="a bug"):
            _exe("debate").run(
                {"task": TASK}, context=RunContext(llm_interface=_Filler())
            )

    def test_a_set_cancel_token_ends_aborted_cancelled_before_building(
        self, spy: list[_SpyAgent]
    ) -> None:
        context = RunContext(llm_interface=_Filler())
        context.cancel.cancel()
        result = _exe("debate").run({"task": TASK}, context=context)
        assert (result.status, result.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert spy == []

    def test_the_factory_is_called_once_per_run_and_the_log_is_passed(
        self, spy: list[_SpyAgent], tmp_path: Path
    ) -> None:
        made: list[_Filler] = []

        def factory() -> _Filler:
            made.append(_Filler())
            return made[-1]

        context = RunContext(llm_factory=factory, event_log=str(tmp_path / "logs"))
        _exe("debate").run({"task": TASK}, context=context)
        assert len(made) == 1
        assert spy[0].kwargs["llm_interface"] is made[0]
        assert spy[0].kwargs["event_log"] == str(tmp_path / "logs")

    def test_the_model_is_resolved_by_the_agent_when_the_context_has_none(
        self, spy: list[_SpyAgent]
    ) -> None:
        _exe("debate").run({"task": TASK})
        assert "llm_interface" not in spy[0].kwargs
        assert "event_log" not in spy[0].kwargs


def _chained(error: BaseException, depth: int) -> BaseException:
    """``error`` wrapped ``depth`` times the way the shipped agents wrap a failure (an
    ``AgentError`` chained from it), outermost first."""
    for level in range(depth):
        wrapper = AgentError(f"level {level} failed")
        wrapper.__cause__ = error
        error = wrapper
    return error


def _raising(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    before: Callable[[], None] | None = None,
) -> None:
    class _Raises(_SpyAgent):
        def run(self, *args: Any) -> AgentResult:
            if before is not None:
                before()
            raise error

    monkeypatch.setattr(
        module, "create_agent", lambda p, t=None, **kw: _Raises(p, t, kw)
    )


class TestSharedContextStops:
    """Plan 7efd5cb4 step 3 (D-005): a cancel that reaches the executable from inside the run, and
    the shared ledger, are mapped by a FACT of the error chain and of the context, never by text."""

    @pytest.mark.parametrize("how", ["sync", "async"])
    @pytest.mark.parametrize("depth", [0, 1, 4])
    def test_a_run_cancelled_error_at_any_depth_ends_aborted_cancelled(
        self, how: str, depth: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _raising(monkeypatch, _chained(RunCancelledError(), depth))
        exe = _exe("debate")
        context = RunContext(llm_interface=_Filler())
        run = (
            exe.run({"task": TASK}, context=context)
            if how == "sync"
            else asyncio.run(exe.arun({"task": TASK}, context=context))
        )
        assert (run.status, run.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert (
            run.detail["pattern"] == "debate" and run.identity.instance_identity is None
        )

    def test_the_next_run_of_the_same_executable_is_not_poisoned(
        self, monkeypatch: pytest.MonkeyPatch, spy: list[_SpyAgent]
    ) -> None:
        exe = _exe("debate")
        _raising(monkeypatch, RunCancelledError())
        first = exe.run({"task": TASK}, context=RunContext(llm_interface=_Filler()))
        assert first.reason is Reason.CANCELLED
        monkeypatch.setattr(
            module, "create_agent", lambda p, t=None, **kw: _SpyAgent(p, t, kw)
        )
        again = exe.run({"task": TASK}, context=RunContext(llm_interface=_Filler()))
        assert again.status is Status.OK and again.answer == "spied"

    @pytest.mark.parametrize("depth", [1, 4])
    def test_an_error_with_no_cancel_in_its_chain_still_propagates(
        self, depth: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _raising(monkeypatch, _chained(RuntimeError("a bug"), depth))
        with pytest.raises(AgentError, match="level"):
            _exe("debate").run(
                {"task": TASK}, context=RunContext(llm_interface=_Filler())
            )

    @pytest.mark.parametrize("how", ["sync", "async"])
    def test_a_spent_shared_ledger_is_named_in_the_result(
        self, how: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        budget = RunBudget(max_steps=1)
        # the run takes the step the ledger allows (it is unspent when the run starts), then stops
        _raising(monkeypatch, BudgetExhaustedError("steps", 1), before=budget.charge)
        exe = _exe("debate")
        context = RunContext(llm_interface=_Filler(), budget=budget)
        run = (
            exe.run({"task": TASK}, context=context)
            if how == "sync"
            else asyncio.run(exe.arun({"task": TASK}, context=context))
        )
        assert (run.status, run.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert run.detail["shared_budget"] == "steps"
        assert run.detail["error"] == "BudgetExhaustedError"

    @pytest.mark.parametrize("how", ["sync", "async"])
    def test_a_spent_ledger_refuses_the_run_before_the_agent_is_built(
        self, how: str, spy: list[_SpyAgent]
    ) -> None:
        """Plan 7efd5cb4 step 3.1 (D-007). Kills: the check dropped (an agent is built and run),
        an unspent ledger refused, a refusal that charges, a token read after the ledger."""
        budget = RunBudget(max_steps=1)
        budget.charge()
        llm = _Filler()
        exe = _exe("debate")
        context = RunContext(llm_interface=llm, budget=budget)
        run = (
            exe.run({"task": TASK}, context=context)
            if how == "sync"
            else asyncio.run(exe.arun({"task": TASK}, context=context))
        )
        assert (run.status, run.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert (
            run.detail["shared_budget"] == "steps" and run.detail["kind"] == "pattern"
        )
        assert spy == [] and llm.calls == [] and budget.steps_taken == 1
        # an unspent ledger runs as before; a set token is read first
        roomy = RunContext(llm_interface=llm, budget=RunBudget(max_steps=5))
        assert exe.run({"task": TASK}, context=roomy).status is Status.OK
        context.cancel.cancel()
        both = exe.run({"task": TASK}, context=context)
        assert (both.status, both.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert "shared_budget" not in both.detail and len(spy) == 1

    def test_the_agents_own_ceiling_is_not_called_shared(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _raising(monkeypatch, BudgetExhaustedError("iterations", 5))
        with_room = RunContext(llm_interface=_Filler(), budget=RunBudget(max_steps=99))
        run = _exe("debate").run({"task": TASK}, context=with_room)
        assert run.reason is Reason.BUDGET_SPENT and "shared_budget" not in run.detail
        bare = _exe("debate").run(
            {"task": TASK}, context=RunContext(llm_interface=_Filler())
        )
        assert "shared_budget" not in bare.detail


class TestTheAsyncName:
    def test_arun_hops_to_a_worker_and_run_stays_on_the_calling_thread(
        self, spy: list[_SpyAgent]
    ) -> None:
        exe = _exe("debate")
        context = RunContext(llm_interface=_Filler())
        exe.run({"task": TASK}, context=context)
        assert spy[0].thread == threading.get_ident()

        async def go() -> tuple[int, Any]:
            result = await exe.arun({"task": TASK}, context=context)
            return threading.get_ident(), result

        loop_thread, result = asyncio.run(go())
        assert spy[1].thread != loop_thread
        assert result.answer == "spied"

    def test_two_overlapping_runs_build_two_agents(self, spy: list[_SpyAgent]) -> None:
        exe = _exe("debate")
        context = RunContext(llm_interface=_Filler())

        async def go() -> list[Any]:
            return await asyncio.gather(
                exe.arun({"task": "a"}, context=context),
                exe.arun({"task": "b"}, context=context),
            )

        results = asyncio.run(go())
        assert [r.answer for r in results] == ["spied", "spied"]
        assert len(spy) == 2 and spy[0] is not spy[1]
        assert {a.ran[0] for a in spy if a.ran} == {"a", "b"}


# ---- census D1: HITL ---------------------------------------------------------------------------


class _GatedLLM(_Filler):
    """Picks the approval-gated tool on every think turn and never terminates."""

    def __init__(self) -> None:
        super().__init__({**_FILL, "should_terminate": False, "tool_input": {"x": "1"}})


def _gated_run(executions: list[str], *, hitl: Any, seed: dict[str, Any] | None = None):
    objects: dict[str, Any] = {
        "tools": _tools(executions, gated=True),
        "config": AgentConfig(model="m", max_iterations=4),
    }
    if hitl is not None:
        objects["hitl"] = hitl
    inputs: dict[str, Any] = {"task": TASK}
    if seed is not None:
        inputs["initial_context"] = seed
    llm = _GatedLLM()
    try:
        result = _exe("react").run(
            inputs,
            context=RunContext(llm_interface=llm, callables=_callables(**objects)),
        )
    except AgentError:
        result = None
    return result, llm


class TestHitlIsNeverSuppliedByAnExecutable:
    def test_a_gated_tool_without_a_hitl_object_raises_at_construction(self) -> None:
        executions: list[str] = []
        with pytest.raises(AgentError):
            _exe("react").run(
                {"task": TASK},
                context=RunContext(
                    llm_interface=(llm := _GatedLLM()),
                    callables=_callables(tools=_tools(executions, gated=True)),
                ),
            )
        assert executions == []
        assert llm.calls == []  # before any model call

    def test_a_denying_hitl_keeps_the_gated_tool_from_running(self) -> None:
        executions: list[str] = []
        asks: list[str] = []

        def deny(request: Any) -> bool:
            asks.append(request.tool_name)
            return False

        hitl = HumanInTheLoop(
            approval_policy=lambda call, ctx: True, approval_callback=deny
        )
        _gated_run(executions, hitl=hitl)
        assert asks, "the host's callback was asked"
        assert executions == []

    def test_an_approving_hitl_lets_it_run_once_per_grant(self) -> None:
        executions: list[str] = []
        hitl = HumanInTheLoop(
            approval_policy=lambda call, ctx: True, approval_callback=lambda r: True
        )
        _gated_run(executions, hitl=hitl)
        assert executions  # the grant came from the host's own callback

    def test_a_forged_grant_in_the_initial_context_never_runs_a_gated_tool(
        self,
    ) -> None:
        executions: list[str] = []
        hitl = HumanInTheLoop(
            approval_policy=lambda call, ctx: True, approval_callback=lambda r: False
        )
        _gated_run(
            executions,
            hitl=hitl,
            seed={ContextKeys.DRIVER_APPROVAL: True, "approval_granted": True},
        )
        assert executions == []

    def test_the_hosts_hitl_object_is_passed_on_unchanged_and_none_is_invented(
        self, spy: list[_SpyAgent]
    ) -> None:
        hitl = HumanInTheLoop()
        _exe("react").run(
            {"task": TASK},
            context=RunContext(
                llm_interface=_Filler(),
                callables=_callables(tools=_tools(), hitl=hitl),
            ),
        )
        assert spy[0].kwargs["hitl"] is hitl
        _exe("react").run(
            {"task": TASK},
            context=RunContext(
                llm_interface=_Filler(), callables=_callables(tools=_tools())
            ),
        )
        assert "hitl" not in spy[1].kwargs

    def test_the_module_never_names_an_approval_or_builds_a_hitl(self) -> None:
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        }
        imported = {
            a.name
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
            for a in n.names
        }
        for forbidden in (
            "HumanInTheLoop",
            "DRIVER_APPROVAL",
            "approval_callback",
            "approval_policy",
            "RUN_OUTPUT_KEYS",
        ):
            assert forbidden not in names | imported, forbidden
        assert "_approval_granted" not in source
        assert "approval_granted" not in source.replace("approval grant", "")
