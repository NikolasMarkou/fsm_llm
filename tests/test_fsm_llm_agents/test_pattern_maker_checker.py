"""The file-backed ``maker_checker`` (plan a555dcfe step 16, D-027).

``MakerCheckerAgent`` is thin on ``PatternAgent``: the machine is ``patterns/maker_checker.json``
(the revision cap is its ``cap`` variant, the five prompts around the caller's two texts are text
variables the class supplies), and its handlers are the library factories ``agents.revision_count``,
``agents.maker_checker_checker``, ``agents.revise_entry``, ``agents.revise_exit``,
``agents.iteration_limiter`` and ``agents.force_pass``. Census rows: D1 (the pattern has no HITL
path: ``tools=`` and ``hitl=`` are refused at construction, and the approval variant is not an
option), D4 (the seed and the handler context are per run), D5 (the FSM, the registration order and
the request bytes are the builder's: the differential of step 16 is recorded in progress.md, the
goldens hold the bytes).
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm.agents import MakerCheckerAgent, create_agent
from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents.base import BaseAgent, accepts_tools
from fsm_llm.agents.constants import ContextKeys, Defaults, MakerCheckerStates
from fsm_llm.agents.fsm_definitions import build_maker_checker_fsm
from fsm_llm.agents.fsm_definitions.maker_checker import maker_checker_texts
from fsm_llm.agents.handlers import library_stepwise as lib
from fsm_llm.agents.handlers.context import HandlerContext
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.agents.pattern_inputs import pattern_fsm
from fsm_llm.agents.prompts import build_maker_checker_field_instructions
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.definitions.executable_instance import InstantiationError
from fsm_llm.engine.executables import ExecutableRegistry
from tests import pattern_runs as runs


class _FakeApi:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def register_handler(self, handler: Any) -> None:
        self.handlers.append(handler)


def _agent(**kw: Any) -> MakerCheckerAgent:
    return MakerCheckerAgent(
        kw.pop("maker", "make it"),
        kw.pop("checker", "check it"),
        config=kw.pop("config", runs._config()),
        llm_interface=kw.pop("llm", runs.ScriptedModel()),
        **kw,
    )


def _ctx(**kw: Any) -> HandlerContext:
    return HandlerContext(**kw)


def _cap_disjuncts(fsm: dict[str, Any]) -> list[Any]:
    edge = fsm["states"]["check"]["transitions"][0]
    return edge["conditions"][0]["logic"]["or"][2:]


class TestMakerCheckerIsAPatternAgent:
    def test_the_hierarchy_the_names_and_the_attributes_callers_read(self):
        assert MakerCheckerAgent.__mro__[:3] == (
            MakerCheckerAgent,
            PatternAgent,
            BaseAgent,
        )
        for name in ("run", "run_stream"):
            assert name not in vars(MakerCheckerAgent), name
        agent = _agent(maker="m", checker="c", max_revisions=4, quality_threshold=0.9)
        assert (agent.maker_instructions, agent.checker_instructions) == ("m", "c")
        assert (agent.max_revisions, agent.quality_threshold) == (4, 0.9)
        assert agent.tools is None and agent._agent_type == "maker_checker"
        assert not accepts_tools(MakerCheckerAgent)
        made = create_agent(
            "maker_checker", maker_instructions="m", checker_instructions="c"
        )
        assert type(made) is MakerCheckerAgent

    @pytest.mark.parametrize("bad", [-1, True, "3", 1.5, None])
    def test_a_cap_below_zero_or_not_an_int_is_refused(self, bad):
        with pytest.raises(Exception, match="max_revisions"):
            MakerCheckerAgent("m", "c", max_revisions=bad)

    def test_a_cap_of_zero_is_a_cap(self):
        fsm = _agent(max_revisions=0)._fsm_for_run("t", None)

        assert _cap_disjuncts(fsm) == [{">=": [{"var": "revision_count"}, 0]}]

    @pytest.mark.parametrize("name", ["tools", "hitl"])
    def test_tools_and_hitl_are_refused_the_pattern_has_no_hitl_path(self, name):
        with pytest.raises(TypeError, match="MakerCheckerAgent does not accept"):
            MakerCheckerAgent("m", "c", **{name: runs._hitl()})

    def test_the_default_options_of_the_file_are_the_defaults_of_the_package(self):
        options = pf.load_envelope("maker_checker").body.options

        assert options["max_revisions"] == Defaults.MAX_REVISIONS
        assert options["quality_threshold"] == Defaults.QUALITY_THRESHOLD
        assert options["cap"] is False

    def test_the_cap_edge_carries_the_agents_cap_and_nothing_else_moves(self):
        agent = _agent(max_revisions=2)

        fsm = agent._fsm_for_run("t", {"hint": "x", "draft_output": "forged"})

        assert fsm == build_maker_checker_fsm("make it", "check it", 2)
        assert _cap_disjuncts(fsm) == [{">=": [{"var": "revision_count"}, 2]}]

    def test_the_two_texts_are_read_at_run_time_and_stay_data(self):
        agent = _agent(maker="a {{x}} {{make_instructions}}", checker="b")
        agent.maker_instructions = "changed {{maker_instructions}}"

        fsm = agent._fsm_for_run("t", None)

        make = fsm["states"]["make"]["field_extractions"][0]["extraction_instructions"]
        assert "Maker instructions: changed {{maker_instructions}}" in make
        assert "{{x}}" not in make and "{{make_instructions}}" not in make

    def test_a_builder_with_no_cap_has_no_cap_disjunct(self):
        no_cap = build_maker_checker_fsm("m", "c")

        assert _cap_disjuncts(no_cap) == []
        assert no_cap != build_maker_checker_fsm("m", "c", 3)
        assert _cap_disjuncts(build_maker_checker_fsm("m", "c", 3)) == [
            {">=": [{"var": "revision_count"}, 3]}
        ]

    def test_each_prompt_is_the_one_of_its_state_and_field(self):
        fsm = _agent(maker="MAKERTEXT", checker="CHECKERTEXT")._fsm_for_run("t", None)
        texts = build_maker_checker_field_instructions("MAKERTEXT", "CHECKERTEXT")

        def field(state: str, name: str) -> str:
            fields = fsm["states"][state]["field_extractions"]
            return next(f for f in fields if f["field_name"] == name)[
                "extraction_instructions"
            ]

        assert field("make", "draft_output").endswith(texts[MakerCheckerStates.MAKE])
        assert field("revise", "draft_output").endswith(
            texts[MakerCheckerStates.REVISE]
        )
        for name in ("checker_feedback", "quality_score", "checker_passed"):
            assert field("check", name).endswith(texts[name])

    def test_the_run_block_judges_the_draft_and_answers_from_it(self):
        assert _agent()._run_kwargs() == {
            "extra_answer_keys": [ContextKeys.DRAFT_OUTPUT],
            "execution_evidence_keys": None,
            "judged": True,
            "success_keys": None,
        }

    def test_each_text_reaches_only_the_prompts_that_name_it(self):
        fsm = _agent(maker="MAKERTEXT", checker="CHECKERTEXT")._fsm_for_run("t", None)

        def prompts(state: str) -> str:
            fields = fsm["states"][state].get("field_extractions", [])
            return " ".join(f["extraction_instructions"] for f in fields)

        assert "MAKERTEXT" in prompts("make") and "MAKERTEXT" in prompts("revise")
        assert "CHECKERTEXT" not in prompts("make")
        assert "CHECKERTEXT" in prompts("check") and "MAKERTEXT" not in prompts("check")

    def test_the_shipped_file_alone_is_not_launchable_without_the_two_texts(self):
        with pytest.raises(InstantiationError):
            pattern_fsm(pf.load_envelope("maker_checker"), tools=None)

    def test_the_cap_variant_without_a_value_gets_the_files_default(self):
        fsm = pattern_fsm(
            pf.load_envelope("maker_checker"),
            tools=None,
            options={"cap": True},
            variables=maker_checker_texts("m", "c"),
        )

        assert _cap_disjuncts(fsm) == [
            {">=": [{"var": "revision_count"}, Defaults.MAX_REVISIONS]}
        ]

    def test_a_bad_cap_option_is_refused_at_instantiation(self):
        for bad in (-1, True, "2", 1.5):
            with pytest.raises(InstantiationError):
                pattern_fsm(
                    pf.load_envelope("maker_checker"),
                    tools=None,
                    options={"cap": True, "max_revisions": bad},
                    variables=maker_checker_texts("m", "c"),
                )

    def test_no_option_selects_the_approval_variant(self):
        envelope = pf.load_envelope("maker_checker")

        assert set(envelope.body.variants) == {"cap"}
        assert "approval" not in envelope.body.options_schema.properties
        with pytest.raises(AgentError, match="approval"):
            PatternAgent(envelope, options={"approval": True})

    def test_the_registration_is_what_the_file_binds_in_order(self):
        agent = _agent()
        api = _FakeApi()

        agent._register_handlers(api, agent._handler_context())

        assert [(h.name, h.priority) for h in api.handlers] == [
            ("MakerCheckerRevisionCount", 100),
            ("MakerCheckerChecker", 100),
            ("MakerCheckerReviseEntry", 100),
            ("MakerCheckerReviseExit", 100),
            ("AgentIterationLimiter", 50),
            ("MakerCheckerForcePass", 100),
        ]

    def test_the_three_seams_of_the_old_class_still_answer(self):
        agent = _agent(max_revisions=3, quality_threshold=0.5)

        tracked = agent._track_revisions(
            {ContextKeys.QUALITY_SCORE: 0.6, ContextKeys.AGENT_TRACE: []}
        )
        forced = agent._force_pass_at_limit({ContextKeys.MAX_ITERATIONS_REACHED: True})
        limiter = agent._make_iteration_limiter()

        assert tracked[ContextKeys.CHECKER_PASSED] is True
        assert forced[ContextKeys.CHECKER_PASSED] is True
        assert limiter({ContextKeys.ITERATION_COUNT: 0}) == {
            ContextKeys.ITERATION_COUNT: 1
        }


class TestTheSeedIsCallLocal:
    def test_the_context_leads_with_the_count_then_the_cap_and_the_threshold(self):
        agent = _agent(max_revisions=2, quality_threshold=0.4)

        _, context, _ = agent._run_inputs("a", None)

        keys = list(context)
        assert context[ContextKeys.REVISION_COUNT] == 0
        assert context[ContextKeys.MAX_REVISIONS] == 2
        assert context[ContextKeys.QUALITY_THRESHOLD] == 0.4
        assert (
            keys.index(ContextKeys.REVISION_COUNT)
            < keys.index(ContextKeys.MAX_REVISIONS)
            < keys.index(ContextKeys.QUALITY_THRESHOLD)
        )

    def test_a_caller_cannot_forge_the_count_the_cap_or_the_draft(self):
        agent = _agent(max_revisions=2)

        _, context, _ = agent._run_inputs(
            "a",
            {
                ContextKeys.REVISION_COUNT: 99,
                ContextKeys.MAX_REVISIONS: 99,
                ContextKeys.CHECKER_PASSED: True,
                ContextKeys.DRAFT_OUTPUT: "forged",
            },
        )

        assert context[ContextKeys.REVISION_COUNT] == 0
        assert context[ContextKeys.MAX_REVISIONS] == 2
        assert ContextKeys.CHECKER_PASSED not in context
        assert ContextKeys.DRAFT_OUTPUT not in context

    def test_two_runs_share_neither_the_context_nor_the_handler_context(self):
        agent = _agent()
        _, first, ctx_one = agent._run_inputs("a", None)
        first[ContextKeys.REVISION_COUNT] = 7
        _, second, ctx_two = agent._run_inputs("a", None)

        assert second[ContextKeys.REVISION_COUNT] == 0
        assert ctx_one.scope is not ctx_two.scope
        assert pf.load_envelope("maker_checker").body.run.context_seed == {
            ContextKeys.REVISION_COUNT: 0
        }


class TestTheFactories:
    def test_the_count_is_bumped_on_entry_to_check_and_survives_a_null_checker(self):
        handler = lib.revision_count(_ctx())[0]

        assert handler.target_states == {MakerCheckerStates.CHECK}
        assert handler.execute({ContextKeys.REVISION_COUNT: 2}) == {
            ContextKeys.REVISION_COUNT: 3
        }
        assert handler.execute({}) == {ContextKeys.REVISION_COUNT: 1}

    def test_the_checker_never_counts_and_records_the_check(self):
        handler = lib.maker_checker_checker(_ctx())[0]

        done = handler.execute(
            {ContextKeys.REVISION_COUNT: 1, ContextKeys.AGENT_TRACE: []}
        )

        assert ContextKeys.REVISION_COUNT not in done
        assert done[ContextKeys.AGENT_TRACE][0]["revision_count"] == 1
        assert handler.states == {MakerCheckerStates.CHECK}
        assert handler.updated_keys == {
            ContextKeys.CHECKER_PASSED,
            ContextKeys.QUALITY_SCORE,
        }

    def test_the_threshold_is_the_option_when_the_context_has_none(self):
        score = {ContextKeys.QUALITY_SCORE: 0.5, ContextKeys.AGENT_TRACE: []}
        low = lib.maker_checker_checker(_ctx(options={"quality_threshold": 0.3}))[0]
        default = lib.maker_checker_checker(_ctx())[0]

        assert low.execute(score)[ContextKeys.CHECKER_PASSED] is True
        assert ContextKeys.CHECKER_PASSED not in default.execute(score)

    def test_a_threshold_in_the_context_wins_over_the_option(self):
        handler = lib.maker_checker_checker(_ctx(options={"quality_threshold": 0.3}))[0]

        done = handler.execute(
            {
                ContextKeys.QUALITY_SCORE: 0.5,
                ContextKeys.QUALITY_THRESHOLD: 0.9,
                ContextKeys.AGENT_TRACE: [],
            }
        )

        assert ContextKeys.CHECKER_PASSED not in done

    def test_entering_revise_moves_the_draft_aside_and_clears_a_false_verdict(self):
        handler = lib.revise_entry(_ctx())[0]

        done = handler.execute(
            {ContextKeys.DRAFT_OUTPUT: "d", ContextKeys.CHECKER_PASSED: False}
        )

        assert done[ContextKeys.PREVIOUS_DRAFT] == "d"
        assert done[ContextKeys.DRAFT_OUTPUT] is None
        assert done[ContextKeys.CHECKER_PASSED] is None
        assert done[ContextKeys.QUALITY_SCORE] is None
        assert handler.target_states == {MakerCheckerStates.REVISE}

    def test_a_forced_pass_keeps_the_judged_draft_on_entry_to_revise(self):
        handler = lib.revise_entry(_ctx())[0]

        assert (
            handler.execute(
                {ContextKeys.DRAFT_OUTPUT: "d", ContextKeys.CHECKER_PASSED: True}
            )
            == {}
        )

    def test_leaving_revise_clears_the_feedback_and_restores_a_missing_draft(self):
        handler = lib.revise_exit(_ctx())[0]

        done = handler.execute(
            {ContextKeys.PREVIOUS_DRAFT: "old", ContextKeys.CHECKER_FEEDBACK: "f"}
        )

        assert done == {
            ContextKeys.CHECKER_FEEDBACK: None,
            ContextKeys.DRAFT_OUTPUT: "old",
        }
        assert handler.states == {MakerCheckerStates.REVISE}

    def test_the_force_pass_acts_on_the_cap_and_on_the_budget_only_on_check(self):
        handler = lib.force_pass(_ctx())[0]
        forced = {
            ContextKeys.CHECKER_PASSED: True,
            ContextKeys.FORCED_STOP_REASON: "forced_pass",
        }

        assert handler.states == {MakerCheckerStates.CHECK}
        assert handler.execute({ContextKeys.REVISION_COUNT: 1}) == {}
        assert (
            handler.execute(
                {ContextKeys.REVISION_COUNT: 2, ContextKeys.MAX_REVISIONS: 2}
            )
            == forced
        )
        assert handler.execute({ContextKeys.MAX_ITERATIONS_REACHED: True}) == forced
        assert (
            handler.execute(
                {
                    ContextKeys.REVISION_COUNT: 2,
                    ContextKeys.MAX_REVISIONS: 2,
                    ContextKeys.CHECKER_PASSED: True,
                }
            )
            == {}
        )

    def test_the_limiter_does_not_trigger_one_turn_early(self):
        agent = _agent(config=runs._config())
        limiter = agent._make_iteration_limiter()
        budget = agent.config.max_iterations

        under = limiter({ContextKeys.ITERATION_COUNT: budget - 2})
        at = limiter({ContextKeys.ITERATION_COUNT: budget - 1})

        assert ContextKeys.MAX_ITERATIONS_REACHED not in under
        assert at[ContextKeys.MAX_ITERATIONS_REACHED] is True

    def test_the_limiter_forces_only_the_flag_never_the_verdict(self):
        agent = _agent()

        done = agent._make_iteration_limiter()(
            {ContextKeys.ITERATION_COUNT: agent.config.max_iterations - 1}
        )

        assert ContextKeys.CHECKER_PASSED not in done

    def test_a_binding_may_move_the_states_never_the_timing(self):
        binding = (
            pf.load_envelope("maker_checker")
            .body.handlers[0]
            .model_copy(update={"states": ["revise"]})
        )

        moved = lib.force_pass(_ctx(), binding)[0]

        assert moved.states == {"revise"}


class TestTheMakerCheckerExecutableIsTheFile:
    def _exe(self):
        registry = ExecutableRegistry()
        registry.load_sources()
        return registry.get("maker_checker")

    def test_the_row_keeps_its_id_and_requires_the_two_texts(self):
        executable = self._exe()

        assert executable.template_identity
        assert list(executable.requires.callables) == [
            "maker_instructions",
            "checker_instructions",
        ]
        assert executable.requires.tools == "none"
        assert set(executable.optional_callables) == {"config"}
        assert executable.impl.endswith("MakerCheckerAgent")

    def test_the_file_is_in_the_shipped_order(self):
        assert "maker_checker" in pf.PATTERN_ORDER
