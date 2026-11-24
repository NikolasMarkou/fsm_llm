"""The file-backed ``rewoo`` (plan a555dcfe step 14, D-025).

``REWOOAgent`` is thin on ``PatternAgent``: the machine is ``patterns/rewoo.json`` (the tool section of
the plan prompt is the text variable ``plan_instructions``), the executor is the library factory
``agents.rewoo_executor`` and the cut-off answer is the hook ``agents.rewoo_cut_off_answer``, which
replaces the result in ``scope[RUN_RESULT]``. Census rows: D1 (no HITL path: a flagged tool is refused
at construction and at run start, ``hitl=`` is a ``TypeError``, a model-origin document cannot bind the
hook), D4 (the result slot and the seed are per run), D5 (the FSM, the registration order and the
request bytes are the builder's: the differential of step 14 is in progress.md, the goldens hold the
bytes; the cut-off rule itself is held end to end by ``test_cut_off_agents``).
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm.agents import REWOOAgent, create_agent
from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents.base import BaseAgent, accepts_tools
from fsm_llm.agents.constants import ContextKeys, ErrorMessages
from fsm_llm.agents.definitions import AgentResult
from fsm_llm.agents.fsm_definitions import build_rewoo_fsm
from fsm_llm.agents.handlers import library as lib
from fsm_llm.agents.handlers.context import RUN_RESULT, HandlerContext
from fsm_llm.agents.handlers.rewoo import (
    evidence_id,
    execute_plans,
    substitute_evidence_refs,
)
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.core.codes import Status
from fsm_llm.core.definitions import CONTEXT_KEY_CUT_OFF, AgentError
from fsm_llm.core.verify.executable import load_executable
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.engine.executables.namespaces import namespace_resolver
from fsm_llm.tools import ToolRegistry
from tests import pattern_runs as runs


class _FakeApi:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def register_handler(self, handler: Any) -> None:
        self.handlers.append(handler)


def _agent(**kw: Any) -> REWOOAgent:
    return REWOOAgent(
        runs._tools(flagged=kw.pop("flagged", False)),
        config=kw.pop("config", runs._config()),
        llm_interface=kw.pop("llm", runs.ScriptedModel()),
        **kw,
    )


def _result(**context: Any) -> AgentResult:
    return AgentResult(
        answer="filler",
        status=Status.OK,
        final_context=context,
        structured_output={"a": 1},
    )


class TestRewooIsAPatternAgent:
    def test_the_hierarchy_the_names_and_the_attributes_callers_read(self):
        assert REWOOAgent.__mro__[:3] == (REWOOAgent, PatternAgent, BaseAgent)
        assert "run" not in vars(REWOOAgent)
        assert accepts_tools(REWOOAgent)
        agent = _agent()
        assert agent.tools is not None and agent._agent_type == "rewoo"
        assert type(create_agent("rewoo", runs._tools())) is REWOOAgent

    def test_hitl_is_a_type_error_as_for_every_pattern_with_no_approval_path(self):
        with pytest.raises(TypeError, match="REWOOAgent does not accept"):
            REWOOAgent(runs._tools(), hitl=runs._hitl())

    def test_an_empty_registry_is_refused(self):
        with pytest.raises(AgentError, match="empty tool registry"):
            REWOOAgent(ToolRegistry())

    def test_a_flagged_tool_is_refused_at_construction_and_at_run_start(self):
        with pytest.raises(AgentError, match="search"):
            _agent(flagged=True)
        agent = _agent()
        agent.tools.register_function(
            lambda p: "x", name="pay", description="d", requires_approval=True
        )

        with pytest.raises(AgentError, match="pay"):
            agent.run("go")

    def test_the_stream_is_refused_the_cut_off_answer_cannot_reach_a_stream(self):
        with pytest.raises(AgentError, match="does not stream"):
            list(_agent().run_stream("go"))

    def test_the_file_needs_tools_and_binds_no_gate(self):
        envelope = pf.load_envelope("rewoo")

        assert envelope.requires.tools == "some"
        assert list(envelope.requires.callables) == ["tools"]
        assert "agents.approval_gate" not in envelope.requires.handlers
        assert [h.hook for h in envelope.body.run.hooks] == [
            "agents.rewoo_cut_off_answer"
        ]
        assert envelope.body.run.evidence_keys == [ContextKeys.EVIDENCE_STATUS]
        assert envelope.body.variants == {}


class TestTheFsmAndTheSeedOfARun:
    def test_the_fsm_of_a_run_is_the_file_and_leaves_out_its_own_outputs(self):
        agent = _agent()
        tools = agent.tools

        fsm = agent._fsm_for_run(
            "a task", {"hint": "x", "evidence": {}, "plan_blueprint": []}
        )

        assert fsm == build_rewoo_fsm(tools, "a task", ("hint",))
        keys = fsm["states"]["plan_all"]["field_extractions"][0]["context_keys"]
        assert keys == ["task", "observations", "hint"]

    def test_the_plan_prompt_lists_the_tools_and_the_key_names(self):
        fsm = _agent()._fsm_for_run("t", {"hint": "x"})

        text = fsm["states"]["plan_all"]["field_extractions"][0][
            "extraction_instructions"
        ]
        assert "- search: Tool search" in text
        assert "'task', 'observations', 'hint'" in text

    def test_only_solve_speaks(self):
        states = _agent()._fsm_for_run("t", None)["states"]

        assert [s for s, st in states.items() if st.get("response_instructions")] == [
            "solve"
        ]

    def test_the_registration_is_what_the_file_binds_in_order(self):
        agent = _agent()
        api = _FakeApi()

        agent._register_handlers(api, agent._handler_context())

        assert [h.name for h in api.handlers] == [
            "REWOOExecutor",
            "AgentIterationLimiter",
        ]

    def test_the_context_leads_with_the_evidence_store_and_every_run_has_its_own(self):
        agent = _agent()
        _, first, ctx_one = agent._run_inputs("a", None)
        _, second, ctx_two = agent._run_inputs("a", None)

        assert (
            first[ContextKeys.EVIDENCE] == {}
            and first[ContextKeys.EVIDENCE_STATUS] == []
        )
        assert list(first).index(ContextKeys.EVIDENCE) < list(first).index(
            ContextKeys.EVIDENCE_STATUS
        )
        first[ContextKeys.EVIDENCE_STATUS].append("kept?")
        first[ContextKeys.EVIDENCE]["E1"] = "kept?"
        assert (
            second[ContextKeys.EVIDENCE_STATUS] == []
            and second[ContextKeys.EVIDENCE] == {}
        )
        assert ctx_one.scope is not ctx_two.scope

    def test_a_caller_cannot_seed_a_run_output(self):
        _, context, _ = _agent()._run_inputs(
            "a", {ContextKeys.PLAN_CUT_OFF: True, ContextKeys.EVIDENCE: {"E1": "x"}}
        )

        assert ContextKeys.PLAN_CUT_OFF not in context
        assert context[ContextKeys.EVIDENCE] == {}

    def test_the_old_test_seams_answer(self):
        agent = _agent()

        done = agent._execute_all_plans(
            {
                ContextKeys.PLAN_BLUEPRINT: [
                    {"plan_id": 1, "tool_name": "search", "tool_input": {"x": "1"}}
                ]
            }
        )

        assert done[ContextKeys.EVIDENCE] == {"E1": "result"}
        assert agent._substitute_evidence_refs("#E1!", {"E1": "r"}) == "r!"


class TestTheLimiterOfTheFile:
    def test_at_the_budget_it_forces_the_stop_flags_the_old_limiter_forced(self):
        agent = _agent()
        binding = next(
            b
            for b in agent.envelope.body.handlers
            if b.handler == "agents.iteration_limiter"
        )
        limiter = lib.iteration_check(agent._handler_context(), binding)

        forced = limiter({ContextKeys.ITERATION_COUNT: agent.config.max_iterations})

        assert forced[ContextKeys.MAX_ITERATIONS_REACHED] is True
        assert forced[ContextKeys.SHOULD_TERMINATE] is True


class TestTheExecutorFactory:
    def test_it_runs_the_plan_with_the_tools_of_the_run(self):
        handler = lib.rewoo_executor(HandlerContext(tools=runs._tools()))[0]

        done = handler.execute(
            {
                ContextKeys.PLAN_BLUEPRINT: [
                    {"plan_id": "#E1", "tool_name": "search", "tool_input": {"x": "1"}}
                ]
            }
        )

        assert done[ContextKeys.EVIDENCE] == {"E1": "result"}
        assert done[ContextKeys.EVIDENCE_STATUS][0]["success"] is True

    def test_it_needs_the_runs_tools(self):
        with pytest.raises(AgentError, match="needs the run's tools"):
            lib.rewoo_executor(HandlerContext())

    def test_a_cut_plan_runs_no_tool_and_records_the_stop(self):
        ran: list[Any] = []
        tools = ToolRegistry()
        tools.register_function(lambda p: ran.append(p), name="search", description="d")
        cut = {CONTEXT_KEY_CUT_OFF: {"fields": [ContextKeys.PLAN_BLUEPRINT]}}

        done = execute_plans(
            {ContextKeys.PLAN_BLUEPRINT: [{"tool_name": "search"}], **cut}, tools
        )

        assert ran == []
        assert done[ContextKeys.PLAN_CUT_OFF] is True
        assert done[ContextKeys.FORCED_STOP_REASON] == "truncated"

    def test_a_cut_of_another_field_does_not_stop_the_plan(self):
        cut = {CONTEXT_KEY_CUT_OFF: {"fields": ["reasoning"]}}

        done = execute_plans(
            {ContextKeys.PLAN_BLUEPRINT: [{"tool_name": "search"}], **cut},
            runs._tools(),
        )

        assert (
            ContextKeys.PLAN_CUT_OFF not in done and "E1" in done[ContextKeys.EVIDENCE]
        )

    @pytest.mark.parametrize(
        ("raw", "want"),
        [
            (2, "E2"),
            (3.0, "E3"),
            ("e4", "E4"),
            ("#E5", "E5"),
            (True, "E9"),
            ("x", "E9"),
            (None, "E9"),
        ],
    )
    def test_the_evidence_id_of_a_plan_step(self, raw, want):
        assert evidence_id(raw, 9) == want

    def test_a_reference_to_missing_evidence_is_unavailable(self):
        assert substitute_evidence_refs({"a": ["#E1 #E2"]}, {"E1": "x"}) == {
            "a": ["x [unavailable]"]
        }


class TestTheCutOffHook:
    def _hook(self):
        ctx = HandlerContext()
        return ctx, lib.rewoo_cut_off_answer(ctx)

    def test_a_cut_plan_replaces_the_answer_and_drops_the_structured_output(self):
        ctx, hook = self._hook()
        ctx.scope[RUN_RESULT] = _result(**{ContextKeys.PLAN_CUT_OFF: True})

        hook()

        assert ctx.scope[RUN_RESULT].answer == ErrorMessages.REWOO_PLAN_CUT_OFF
        assert ctx.scope[RUN_RESULT].structured_output is None

    @pytest.mark.parametrize("marker", [False, "true", 1, None])
    def test_only_a_marker_that_is_true_counts(self, marker):
        ctx, hook = self._hook()
        original = _result(**{ContextKeys.PLAN_CUT_OFF: marker})
        ctx.scope[RUN_RESULT] = original

        hook()

        assert ctx.scope[RUN_RESULT] is original

    def test_with_no_result_the_hook_does_nothing(self):
        ctx, hook = self._hook()

        hook()

        assert RUN_RESULT not in ctx.scope


class TestTheRun:
    def test_a_run_ships_the_evidence_and_succeeds(self):
        result = _agent().run("go")

        assert result.final_context[ContextKeys.EVIDENCE] == {"E1": "result"}
        assert result.success is True

    def test_a_run_ends_with_the_cut_off_answer_when_the_marker_is_set(
        self, monkeypatch
    ):
        agent = _agent()

        def cut(context: dict[str, Any], tools: Any) -> dict[str, Any]:
            return {ContextKeys.PLAN_CUT_OFF: True, ContextKeys.EVIDENCE_STATUS: []}

        monkeypatch.setattr("fsm_llm.agents.handlers.library.execute_plans", cut)

        result = agent.run("go")

        assert result.answer == ErrorMessages.REWOO_PLAN_CUT_OFF
        assert result.success is False

    def test_a_raising_loop_raises_through_the_hook_with_no_result(self, monkeypatch):
        agent = _agent()
        seen: list[Any] = []
        real = agent._after_run

        def spy(handlers: HandlerContext) -> None:
            seen.append(RUN_RESULT in handlers.scope)
            real(handlers)

        monkeypatch.setattr(agent, "_after_run", spy)
        monkeypatch.setattr(
            agent,
            "_standard_run",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
        )

        with pytest.raises(RuntimeError, match="x"):
            agent.run("go")

        assert seen == [False]


class TestTheModelOriginRefusesTheHook:
    def test_a_model_origin_document_cannot_bind_the_hook(self):
        doc = pf.load_document("rewoo")
        doc["id"] = "model.rewoo"

        envelope, result = load_executable(
            doc, origin="model", resolver=namespace_resolver()
        )

        assert envelope is None
        assert any("body.run.hooks" in e for e in result.errors)


class TestTheRewooExecutableIsTheFile:
    def test_the_row_keeps_its_id_and_needs_tools(self):
        registry = ExecutableRegistry()
        registry.load_sources()

        executable = registry.get("rewoo")

        assert executable.template_identity
        assert list(executable.requires.callables) == ["tools"]
        assert executable.requires.tools == "some"
