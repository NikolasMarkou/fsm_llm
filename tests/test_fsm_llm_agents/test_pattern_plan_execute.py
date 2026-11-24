"""The file-backed ``plan_execute`` (plan a555dcfe step 15, D-026).

``PlanExecuteAgent`` is thin on ``PatternAgent``: the machine is ``patterns/plan_execute.json`` (the tool
sections of the four prompts are the text variables ``step_instructions``, ``step_result_instructions``,
``plan_steps_instructions`` and ``replan_steps_instructions``, empty of tools when the run has none), the
run with no tool selects the derived variant ``without_tools``, the four handlers are the library
factories ``agents.plan_step_*`` and ``agents.plan_replan_counter`` (the executor ``agents.tool_executor``
is bound with the param ``optional``). Census rows: D1 (no HITL path: a flagged tool is refused at
construction and at run start, ``hitl=`` is a ``TypeError``), D4 (the seed is per run), D5 (the FSM, the
registration order and the request bytes are the builder's: the differential of step 15 is in
progress.md, the goldens hold the bytes). The rules of the replaced builder code that no file can carry
are rows here: no iteration limiter is bound (D-011 of plan 5ed71cf2) and the seed ``plan_steps: []``
(D-046 of plan 21cd7f8e) is the one exit of the ``plan`` state.
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm.agents import PlanExecuteAgent, create_agent
from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents.base import BaseAgent, accepts_tools
from fsm_llm.agents.constants import ContextKeys, Defaults
from fsm_llm.agents.fsm_definitions import build_plan_execute_fsm
from fsm_llm.agents.handlers import library_react
from fsm_llm.agents.handlers import library_stepwise as lib
from fsm_llm.agents.handlers.context import HandlerContext
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.agents.pattern_inputs import pattern_fsm
from fsm_llm.core.definitions import AgentError
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.tools import ToolRegistry
from tests import pattern_runs as runs


class _FakeApi:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def register_handler(self, handler: Any) -> None:
        self.handlers.append(handler)


def _agent(tools: Any = "tools", **kw: Any) -> PlanExecuteAgent:
    if tools == "tools":
        tools = runs._tools(flagged=kw.pop("flagged", False))
    return PlanExecuteAgent(
        tools,
        config=kw.pop("config", runs._config()),
        llm_interface=kw.pop("llm", runs.ScriptedModel()),
        **kw,
    )


def _binding(agent: PlanExecuteAgent, name: str) -> Any:
    return next(b for b in agent.envelope.body.handlers if b.handler == name)


class TestPlanExecuteIsAPatternAgent:
    def test_the_hierarchy_the_names_and_the_attributes_callers_read(self):
        assert PlanExecuteAgent.__mro__[:3] == (
            PlanExecuteAgent,
            PatternAgent,
            BaseAgent,
        )
        assert "run" not in vars(PlanExecuteAgent)
        assert accepts_tools(PlanExecuteAgent)
        agent = _agent()
        assert agent._agent_type == "plan_execute" and agent.max_replans == 2
        assert agent.tools is not None
        assert type(create_agent("plan_execute", runs._tools())) is PlanExecuteAgent

    def test_hitl_is_a_type_error_as_for_every_pattern_with_no_approval_path(self):
        with pytest.raises(TypeError, match="PlanExecuteAgent does not accept"):
            PlanExecuteAgent(runs._tools(), hitl=runs._hitl())

    def test_a_flagged_tool_is_refused_at_construction_and_at_run_start(self):
        with pytest.raises(AgentError, match="search"):
            _agent(flagged=True)
        agent = _agent()
        agent.tools.register_function(
            lambda p: "x", name="pay", description="d", requires_approval=True
        )

        with pytest.raises(AgentError, match="pay"):
            agent.run("go")

    def test_no_registry_constructs_and_zero_replans_is_valid(self):
        assert _agent(None).tools is None
        assert _agent(max_replans=0).max_replans == 0

    @pytest.mark.parametrize("bad", [-1, True, 1.5, "2"])
    def test_a_bad_max_replans_is_refused(self, bad):
        with pytest.raises(AgentError, match="max_replans"):
            PlanExecuteAgent(max_replans=bad)

    def test_the_file_claims_no_tools_binds_no_gate_and_no_limiter(self):
        envelope = pf.load_envelope("plan_execute")

        assert envelope.requires.tools == "none" and not envelope.capabilities
        assert list(envelope.requires.callables) == []
        assert "agents.approval_gate" not in envelope.requires.handlers
        # D-011 of plan 5ed71cf2: the limiter counted transitions and routed nothing
        assert not [h for h in envelope.requires.handlers if "limiter" in h]
        assert not envelope.body.run.hooks
        assert envelope.body.run.evidence_keys == [ContextKeys.STEP_RESULTS]
        assert envelope.body.options["max_replans"] == Defaults.MAX_REPLANS
        assert list(envelope.body.variants) == ["without_tools"]


class TestTheFsmOfARun:
    def test_with_tools_the_fsm_is_the_file_with_the_tool_sections(self):
        agent = _agent()

        fsm = agent._fsm_for_run("a task", {"ignored": 1})

        assert fsm == build_plan_execute_fsm(agent.tools, "a task")
        fields = fsm["states"]["execute_step"]["field_extractions"]
        assert [f["field_name"] for f in fields] == [
            "tool_name",
            "tool_input",
            "step_result",
        ]
        assert fields[2]["required"] is False
        assert "- search: Tool search" in fields[0]["extraction_instructions"]

    @pytest.mark.parametrize("tools", [None, ToolRegistry()])
    def test_with_no_tool_the_variant_drops_the_selection_and_the_prompts_name_no_tool(
        self, tools
    ):
        fsm = _agent(tools)._fsm_for_run("a task", None)

        fields = fsm["states"]["execute_step"]["field_extractions"]
        assert [f["field_name"] for f in fields] == ["step_result"]
        assert fields[0]["required"] is True
        text = " ".join(
            f["extraction_instructions"]
            for state in fsm["states"].values()
            for f in state.get("field_extractions", [])
        )
        assert "Available tools" not in text and "{{" not in text
        assert "Add no step that needs no tool" not in text

    def test_the_variant_is_derived_never_an_option(self):
        envelope = pf.load_envelope("plan_execute")

        agent = PatternAgent(envelope, options={"without_tools": False})

        with pytest.raises(Exception, match="without_tools"):
            agent._fsm_for_run("t", None)

    def test_a_semantic_registry_is_asked_once_per_prompt_with_the_task(self):
        class Sem(ToolRegistry):
            def __init__(self) -> None:
                super().__init__()
                self.asked: list[Any] = []

            def retrieve(self, query, *a, **k):
                self.asked.append(("retrieve", query))
                return self.list_tools()

            def to_prompt_description(self, query=None):
                self.asked.append(("describe", query))
                return f"Relevant for {query}"

        sem = Sem()
        sem.register_function(lambda p: "x", name="t", description="d")

        fsm = pattern_fsm(
            pf.load_envelope("plan_execute"), tools=sem, task_description="the task"
        )

        assert sem.asked == [("describe", "the task"), ("retrieve", "the task")] * 4
        assert "Relevant for the task" in str(fsm)

    def test_the_run_leaves_the_caller_keys_out_of_the_fixed_prompt_keys(self):
        fsm = _agent()._fsm_for_run("t", {"hint": "x"})

        keys = fsm["states"]["plan"]["field_extractions"][0]["context_keys"]
        assert keys == ["task", "observations"]


class TestTheRegistration:
    def test_with_tools_the_executor_comes_first_then_the_four(self):
        agent = _agent()
        api = _FakeApi()

        agent._register_handlers(api, agent._handler_context())

        assert [(h.name, h.priority) for h in api.handlers] == [
            ("AgentToolExecutor", 100),
            ("PlanStepExecutor", 100),
            ("PlanStepFreshKeys", 100),
            ("PlanStepChecker", 101),
            ("PlanReplanCounter", 100),
        ]
        assert [sorted(h.target_states) for h in api.handlers] == [
            ["check_result"],
            ["execute_step"],
            ["execute_step"],
            ["check_result"],
            ["replan"],
        ]

    def test_with_no_registry_the_executor_is_not_bound(self):
        agent = _agent(None)
        api = _FakeApi()

        agent._register_handlers(api, agent._handler_context())

        assert [h.name for h in api.handlers] == [
            "PlanStepExecutor",
            "PlanStepFreshKeys",
            "PlanStepChecker",
            "PlanReplanCounter",
        ]

    def test_an_empty_registry_still_binds_the_executor_as_the_builder_did(self):
        agent = _agent(ToolRegistry())
        api = _FakeApi()

        agent._register_handlers(api, agent._handler_context())

        assert api.handlers[0].name == "AgentToolExecutor" and len(api.handlers) == 5

    def test_the_default_states_of_the_four_factories(self):
        ctx = HandlerContext()

        assert [
            sorted(f(ctx)[0].target_states)
            for f in (
                lib.plan_step_executor,
                lib.plan_step_fresh_keys,
                lib.plan_step_checker,
                lib.plan_replan_counter,
            )
        ] == [["execute_step"], ["execute_step"], ["check_result"], ["replan"]]

    def test_the_executor_without_the_optional_param_needs_the_registry(self):
        with pytest.raises(AgentError, match="needs the run's tools"):
            library_react.tool_executor(HandlerContext())


class TestTheFactories:
    def _checker(self, **options: Any) -> Any:
        ctx = HandlerContext(options=options)
        return lib.plan_step_checker(ctx)[0]

    def test_the_cap_is_the_option_zero_means_no_replan(self):
        failed = {
            ContextKeys.PLAN_STEPS: ["a", "b"],
            ContextKeys.CURRENT_STEP_INDEX: 0,
            ContextKeys.TOOL_STATUS: "failed",
            ContextKeys.TOOL_RESULT: "[TOOL FAILED]",
        }

        none = self._checker(max_replans=0).execute(dict(failed))
        some = self._checker(max_replans=1).execute(dict(failed))

        assert none[ContextKeys.ALL_STEPS_COMPLETE] is True
        assert none[ContextKeys.FORCED_STOP_REASON] == "no_result"
        assert (
            ContextKeys.ALL_STEPS_COMPLETE not in some and some[ContextKeys.STEP_FAILED]
        )

    def test_the_constructor_cap_reaches_the_checker_of_the_run(self):
        failed = {
            ContextKeys.PLAN_STEPS: ["a", "b"],
            ContextKeys.TOOL_STATUS: "failed",
            ContextKeys.REPLAN_COUNT: 1,
        }

        for cap, done in ((1, True), (3, False)):
            agent = _agent(max_replans=cap)
            checker = lib.plan_step_checker(agent._handler_context())[0]
            assert (
                checker.execute(dict(failed)).get(ContextKeys.ALL_STEPS_COMPLETE, False)
                is done
            )

    def test_the_default_cap_is_the_defaults_one(self):
        failed = {
            ContextKeys.PLAN_STEPS: ["a"],
            ContextKeys.TOOL_STATUS: "failed",
            ContextKeys.REPLAN_COUNT: Defaults.MAX_REPLANS,
        }

        assert self._checker().execute(failed)[ContextKeys.ALL_STEPS_COMPLETE] is True

    @pytest.mark.parametrize("bad", [-1, True, None, 2.0])
    def test_a_bad_option_is_refused(self, bad):
        with pytest.raises(AgentError, match="max_replans"):
            self._checker(max_replans=bad)

    def test_a_timed_out_step_ends_the_plan_and_is_never_replanned(self):
        done = self._checker().execute(
            {ContextKeys.PLAN_STEPS: ["a", "b"], ContextKeys.TOOL_STATUS: "unknown"}
        )

        assert done[ContextKeys.ALL_STEPS_COMPLETE] is True
        assert done[ContextKeys.STEP_FAILED] is False

    def test_the_checker_runs_after_the_executor_on_check_result(self):
        executor = library_react.tool_executor(
            HandlerContext(tools=runs._tools()),
            _binding(_agent(), "agents.tool_executor"),
        )[0]
        checker = lib.plan_step_checker(HandlerContext())[0]

        assert checker.priority == executor.priority + 1
        assert checker.target_states == executor.target_states == {"check_result"}

    def test_the_replan_counter_stashes_the_plan_and_reopens_it(self):
        counter = lib.plan_replan_counter(HandlerContext())[0]

        done = counter.execute(
            {ContextKeys.PLAN_STEPS: ["a", "b"], ContextKeys.REPLAN_COUNT: 1}
        )

        assert done == {
            ContextKeys.REPLAN_COUNT: 2,
            ContextKeys.PREVIOUS_PLAN_STEPS: ["a", "b"],
            ContextKeys.PLAN_STEPS: [],
            ContextKeys.CURRENT_STEP_INDEX: 0,
            ContextKeys.STEP_FAILED: False,
        }

    def test_the_fresh_keys_handler_clears_only_the_step_result(self):
        clear = lib.plan_step_fresh_keys(HandlerContext())[0]

        done = clear.execute(
            {ContextKeys.STEP_RESULT: "old", ContextKeys.PLAN_STEPS: ["a"]}
        )

        assert done == {ContextKeys.STEP_RESULT: None}

    def test_the_tracker_describes_the_current_step(self):
        track = lib.plan_step_executor(HandlerContext())[0]

        done = track.execute(
            {ContextKeys.PLAN_STEPS: ["a", "b"], ContextKeys.CURRENT_STEP_INDEX: 1}
        )

        assert done == {ContextKeys.CURRENT_STEP_DESCRIPTION: "Step 2/2: b"}


class TestTheSeedOfARun:
    def test_the_seed_is_the_builders_in_order_and_ends_with_the_replan_count(self):
        _, context, _ = _agent()._run_inputs("a", None)

        seed = [
            ContextKeys.OBSERVATIONS,
            ContextKeys.PLAN_STEPS,
            ContextKeys.CURRENT_STEP_INDEX,
            ContextKeys.STEP_RESULTS,
            ContextKeys.ALL_STEPS_COMPLETE,
            ContextKeys.STEP_FAILED,
            ContextKeys.REPLAN_COUNT,
        ]
        keys = list(context)
        assert [k for k in keys if k in seed] == seed
        # D-046 of plan 21cd7f8e: the `[]` is the one exit of the plan state (key existence)
        assert (
            context[ContextKeys.PLAN_STEPS] == []
            and context[ContextKeys.REPLAN_COUNT] == 0
        )
        assert ContextKeys.ITERATION_BUDGET not in context

    def test_every_run_has_its_own_lists(self):
        agent = _agent()
        _, first, ctx_one = agent._run_inputs("a", None)
        _, second, ctx_two = agent._run_inputs("a", None)

        first[ContextKeys.STEP_RESULTS].append("kept?")
        first[ContextKeys.PLAN_STEPS].append("kept?")

        assert (
            second[ContextKeys.STEP_RESULTS] == []
            and second[ContextKeys.PLAN_STEPS] == []
        )
        assert ctx_one.scope is not ctx_two.scope

    def test_a_caller_cannot_seed_a_run_output(self):
        _, context, _ = _agent()._run_inputs(
            "a", {ContextKeys.PLAN_STEPS: ["x"], ContextKeys.REPLAN_COUNT: 9, "hint": 1}
        )

        assert (
            context[ContextKeys.PLAN_STEPS] == []
            and context[ContextKeys.REPLAN_COUNT] == 0
        )
        assert context["hint"] == 1


class TestTheRun:
    def test_a_run_with_tools_walks_the_plan_and_succeeds(self):
        result = _agent().run("go")

        assert result.success is True
        assert result.final_context[ContextKeys.CURRENT_STEP_INDEX] == 2
        assert [
            r["success"] for r in result.final_context[ContextKeys.STEP_RESULTS]
        ] == [True, True]
        assert result.trace.total_iterations > 0

    def test_a_run_with_no_tool_walks_the_plan_on_the_model_alone(self):
        result = _agent(None).run("go")

        assert result.final_context[ContextKeys.ALL_STEPS_COMPLETE] is True
        assert result.final_context[ContextKeys.CURRENT_STEP_INDEX] == 2

    def test_the_stream_is_inherited(self):
        assert "".join(_agent().run_stream("go"))


class TestTheExecutableIsTheFile:
    def test_the_row_keeps_its_id_claims_no_tools_and_takes_them_when_given(self):
        registry = ExecutableRegistry()
        registry.load_sources()

        executable = registry.get("plan_execute")

        assert executable.template_identity
        assert executable.requires.tools == "none" and not executable.requires.callables
        assert "tools" in executable._row.optional
