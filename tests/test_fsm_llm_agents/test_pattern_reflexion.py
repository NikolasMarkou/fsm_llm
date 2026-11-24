"""The file-backed ``reflexion`` (plan a555dcfe step 10, D-018).

``ReflexionAgent`` is thin on ``PatternAgent``: the loop is ``patterns/reflexion.json``, the evaluate
and reflect handlers are the library factories ``agents.reflexion_*``. Census rows: D1 (the
approval variant is the agent's choice and a flagged tool with nobody to approve it is refused at
construction and at run start: the HITL files hold the rest, unedited), D4 (the seed of a run is a
copy, nothing per run is kept), D5 (the FSM, the registration order and the request bytes are the
builder's: the differential of step 10 is recorded in progress.md, the goldens hold the bytes).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from fsm_llm.agents import ReflexionAgent, create_agent
from fsm_llm.agents.base import BaseAgent, accepts_tools
from fsm_llm.agents.constants import ContextKeys, StopReason
from fsm_llm.agents.definitions import AgentResult, AgentTrace, EvaluationResult
from fsm_llm.agents.fsm_definitions import build_reflexion_fsm
from fsm_llm.agents.handlers import library_react as lib
from fsm_llm.agents.handlers.context import HandlerContext
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.names import NameRegistry
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import ExecutableRegistry
from tests import pattern_runs as runs


class _FakeApi:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def register_handler(self, handler: Any) -> None:
        self.handlers.append(handler)


def _agent(**kw: Any) -> ReflexionAgent:
    return ReflexionAgent(
        runs._tools(flagged=kw.pop("flagged", False)),
        config=runs._config(),
        llm_interface=kw.pop("llm", runs.ScriptedModel()),
        **kw,
    )


def _ctx(**kw: Any) -> HandlerContext:
    return HandlerContext(tools=runs._tools(), **kw)


class TestReflexionIsAPatternAgent:
    def test_the_hierarchy_the_names_and_the_attributes_callers_read(self):
        assert ReflexionAgent.__mro__[:3] == (ReflexionAgent, PatternAgent, BaseAgent)
        assert "run" not in vars(ReflexionAgent)
        agent = _agent(max_reflections=2, evaluation_fn=lambda ctx: None)
        assert agent.max_reflections == 2 and callable(agent.evaluation_fn)
        assert accepts_tools(ReflexionAgent)
        assert type(create_agent("reflexion", runs._tools())) is ReflexionAgent

    def test_the_fsm_of_a_run_is_the_file_and_leads_with_the_episodic_memory(self):
        agent = _agent()

        fsm = agent._fsm_for_run("a task", {"hint": "x"})

        assert fsm == build_reflexion_fsm(agent.tools, "a task", False, ("hint",))
        keys = fsm["states"]["think"]["field_extractions"][0]["context_keys"]
        assert keys == [
            "task",
            "observations",
            "agent_feedback",
            "episodic_memory",
            "hint",
        ]

    def test_max_reflections_changes_no_byte_of_the_fsm(self):
        assert _agent(max_reflections=1)._fsm_for_run("t", None) == _agent(
            max_reflections=9
        )._fsm_for_run("t", None)

    def test_the_registration_is_what_the_file_binds_in_order(self):
        agent = _agent(hitl=runs._hitl())
        api = _FakeApi()

        agent._register_handlers(api, agent._handler_context())

        assert [h.name for h in api.handlers] == [
            "AgentToolExecutor",
            "AgentIterationLimiter",
            "AgentThinkFreshKeys",
            "AgentFeedbackConsumed",
            "AgentApprovalFreshKeys",
            "ReflexionEvaluator",
            "ReflexionFreshKeys",
            "ReflexionReflector",
            "AgentHITLGate",
        ]


class TestTheFactories:
    def test_the_evaluator_runs_the_callable_when_the_run_has_one(self):
        fn = lambda ctx: EvaluationResult(passed=True, score=0.5, feedback="ok")  # noqa: E731
        with_fn = lib.reflexion_evaluator(_ctx(callables={"evaluation_fn": fn}))[0]
        without = lib.reflexion_evaluator(_ctx())[0]

        assert with_fn.execute({}) == {
            ContextKeys.EVALUATION_PASSED: True,
            ContextKeys.EVALUATION_SCORE: 0.5,
            ContextKeys.EVALUATION_FEEDBACK: "ok",
        }
        stale = {ContextKeys.EVALUATION_PASSED: True, ContextKeys.EVALUATION_SCORE: 1}
        cleared = without.execute(dict(stale))
        assert all(cleared.get(key) is None for key in stale)

    def test_the_reflector_reads_its_cap_from_the_runs_options(self):
        handler = lib.reflexion_reflector(_ctx(options={"max_reflections": 2}))[0]

        first = handler.execute({ContextKeys.REFLECTION: "r"})
        second = handler.execute(
            {ContextKeys.REFLECTION: "r", ContextKeys.REFLECTION_COUNT: 1}
        )

        assert ContextKeys.SHOULD_TERMINATE not in first
        assert second[ContextKeys.SHOULD_TERMINATE] is True
        assert second[ContextKeys.FORCED_STOP_REASON] == StopReason.FORCED_PASS.value

    @pytest.mark.parametrize("bad", [0, -1, True, "3", 1.5])
    def test_a_cap_below_one_or_not_an_int_is_refused(self, bad):
        with pytest.raises(AgentError, match="max_reflections"):
            lib.reflexion_reflector(_ctx(options={"max_reflections": bad}))

    def test_a_binding_cannot_name_a_driver_key(self):
        named = SimpleNamespace(
            states=(), keys=("_approval_granted",), priority=None, params={}
        )

        with pytest.raises(AgentError, match="never named by a pattern file"):
            lib.reflexion_evaluator(_ctx(), named)


class TestFailClosedAndPerRun:
    def test_a_flagged_tool_with_nobody_to_approve_is_refused_at_both_places(self):
        with pytest.raises(AgentError, match="nobody can approve"):
            _agent(flagged=True)
        agent = _agent()
        agent.tools.register_function(
            lambda params: "x", name="danger", description="d", requires_approval=True
        )

        with pytest.raises(AgentError, match="nobody can approve"):
            agent.run("go")

    def test_two_runs_share_neither_the_context_nor_the_seeded_memory(self):
        agent = _agent()
        _, first, ctx_one = agent._run_inputs("a", None)
        first[ContextKeys.EPISODIC_MEMORY].append("kept?")
        _, second, ctx_two = agent._run_inputs("a", None)

        assert second[ContextKeys.EPISODIC_MEMORY] == []
        assert second[ContextKeys.REFLECTION_COUNT] == 0
        assert ctx_one is not ctx_two and ctx_one.scope is not ctx_two.scope
        assert "_handlers" not in vars(agent)

    def test_a_failed_verdict_makes_the_run_reflect_once_and_record_the_episode(self):
        verdicts = iter([False, True])
        agent = _agent(
            evaluation_fn=lambda ctx: EvaluationResult(
                passed=next(verdicts), feedback="f"
            )
        )

        result = agent.run("find it")

        memory = result.final_context[ContextKeys.EPISODIC_MEMORY]
        assert result.final_context[ContextKeys.REFLECTION_COUNT] == 1
        assert [entry["episode"] for entry in memory] == [1]


class TestTheReflexionExecutableIsTheFile:
    def _exe(self):
        registry = ExecutableRegistry()
        registry.load_sources()
        return registry.get("reflexion")

    def test_the_row_keeps_its_id_and_reads_the_evaluation_function_when_present(self):
        executable = self._exe()

        assert executable.template_identity
        assert executable.optional_callables == ("hitl", "config", "evaluation_fn")
        assert list(executable.requires.callables) == ["tools"]
        assert [c.value for c in executable.capabilities] == ["tools", "hitl"]

    def test_the_hosts_evaluation_function_reaches_the_packaged_class(
        self, monkeypatch
    ):
        seen: list[Any] = []

        def fake_run(self: Any, task: str, initial_context: Any = None) -> Any:
            seen.append(self.evaluation_fn)
            return AgentResult(
                answer="patched",
                trace=AgentTrace(),
                final_context={},
                stop_reason="answered",
            )

        monkeypatch.setattr(ReflexionAgent, "run", fake_run)
        registry: NameRegistry[Any] = NameRegistry("callables")
        fn = lambda ctx: EvaluationResult(passed=True)  # noqa: E731
        registry.register("tools", runs._tools())
        registry.register("evaluation_fn", fn)

        self._exe().run(
            {"task": "go"},
            context=RunContext(llm_interface=runs.ScriptedModel(), callables=registry),
        )

        assert seen == [fn]
