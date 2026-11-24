"""The file-backed ``debate`` (plan a555dcfe step 17, D-028).

``DebateAgent`` is thin on ``PatternAgent``: the machine is ``patterns/debate.json`` (the round cap
is the integer hole ``max_rounds`` of the judge edge, the persona and the five field prompts around
the three personas are text variables the class supplies), and its handlers are the library factories
``agents.debate_judge``, ``agents.debate_consensus_reset`` and ``agents.debate_limiter``. Census rows:
D1 (the pattern has no HITL path: ``tools=`` and ``hitl=`` are refused at construction and the
approval variant is not an option), D4 (the seed and the handler context are per run), D5 (the FSM,
the registration order and the request bytes are the builder's: the differential of step 17 is
recorded in progress.md, the goldens hold the bytes).
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm.agents import DebateAgent, create_agent
from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents.base import BaseAgent, accepts_tools
from fsm_llm.agents.constants import (
    ContextKeys,
    DebateStates,
    Defaults,
    HandlerPriorities,
    StopReason,
)
from fsm_llm.agents.fsm_definitions import build_debate_fsm
from fsm_llm.agents.fsm_definitions.debate import debate_texts
from fsm_llm.agents.handlers import library_stepwise as lib
from fsm_llm.agents.handlers.context import HandlerContext
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.agents.pattern_inputs import pattern_fsm
from fsm_llm.agents.prompts import build_debate_field_instructions
from fsm_llm.core.definitions import AgentError
from fsm_llm.core.definitions.executable_instance import InstantiationError
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.handlers import HandlerTiming
from tests import pattern_runs as runs


class _FakeApi:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def register_handler(self, handler: Any) -> None:
        self.handlers.append(handler)


def _agent(**kw: Any) -> DebateAgent:
    return DebateAgent(
        config=kw.pop("config", runs._config()),
        llm_interface=kw.pop("llm", runs.ScriptedModel()),
        **kw,
    )


def _ctx(**kw: Any) -> HandlerContext:
    return HandlerContext(**kw)


def _cap(fsm: dict[str, Any]) -> Any:
    edge = fsm["states"]["judge"]["transitions"][0]
    return edge["conditions"][0]["logic"]["or"][1]


def _prompts(fsm: dict[str, Any]) -> dict[str, str]:
    """The extraction prompt of each debate field, by field name."""
    found: dict[str, str] = {}
    for state in fsm["states"].values():
        for field in state.get("field_extractions", []):
            found[field["field_name"]] = field["extraction_instructions"]
    return found


class TestDebateIsAPatternAgent:
    def test_the_hierarchy_the_names_and_the_attributes_callers_read(self):
        assert DebateAgent.__mro__[:3] == (DebateAgent, PatternAgent, BaseAgent)
        for name in ("run", "run_stream"):
            assert name not in vars(DebateAgent), name
        agent = _agent(num_rounds=2, proposer_persona="p", critic_persona="c")
        assert (agent.num_rounds, agent.proposer_persona) == (2, "p")
        assert agent.critic_persona == "c" and agent.judge_persona
        assert agent.tools is None and agent._agent_type == "debate"
        assert not accepts_tools(DebateAgent)
        assert type(create_agent("debate")) is DebateAgent

    @pytest.mark.parametrize("bad", [0, -3, True, "3", 1.5, None])
    def test_a_round_cap_below_one_or_not_an_int_is_refused(self, bad):
        with pytest.raises(Exception, match="num_rounds"):
            DebateAgent(num_rounds=bad)

    @pytest.mark.parametrize("name", ["tools", "hitl"])
    def test_tools_and_hitl_are_refused_the_pattern_has_no_hitl_path(self, name):
        with pytest.raises(TypeError, match="DebateAgent does not accept"):
            DebateAgent(**{name: runs._hitl()})

    def test_the_default_options_of_the_file_are_the_defaults_of_the_package(self):
        options = pf.load_envelope("debate").body.options

        assert options["max_rounds"] == Defaults.MAX_DEBATE_ROUNDS
        assert options["max_iterations"] == (
            Defaults.MAX_DEBATE_ROUNDS
            * Defaults.FSM_BUDGET_MULTIPLIER
            * Defaults.DEBATE_STATES_PER_ROUND
        )

    def test_the_default_personas_are_the_agents_and_fill_the_fsm(self):
        agent = _agent()

        fsm = agent._fsm_for_run("t", None)

        assert fsm == build_debate_fsm(
            agent.proposer_persona,
            agent.critic_persona,
            agent.judge_persona,
            Defaults.MAX_DEBATE_ROUNDS,
        )
        assert fsm["persona"] == agent.proposer_persona

    def test_a_builder_with_no_proposer_persona_has_the_default_fsm_persona(self):
        assert build_debate_fsm()["persona"].startswith("You are a thoughtful AI")
        assert build_debate_fsm("Pro")["persona"] == "Pro"

    @pytest.mark.parametrize("rounds", [1, 2, 5, 100])
    def test_the_round_cap_is_the_option_in_the_judge_edge(self, rounds):
        fsm = _agent(num_rounds=rounds)._fsm_for_run("t", {"hint": "x"})

        assert _cap(fsm) == {">=": [{"var": "current_round"}, rounds]}
        assert f"at most {rounds} in total" in _prompts(fsm)["consensus_reached"]

    @pytest.mark.parametrize("rounds", [1, 5, 100])
    def test_the_builder_wrapper_carries_its_round_cap(self, rounds):
        fsm = build_debate_fsm("P", "C", "J", rounds)

        assert _cap(fsm) == {">=": [{"var": "current_round"}, rounds]}
        assert f"at most {rounds} in total" in _prompts(fsm)["consensus_reached"]
        assert _cap(build_debate_fsm()) == {">=": [{"var": "current_round"}, 3]}

    def test_the_judge_edge_keeps_the_round_disjunct_and_the_fallback(self):
        # D-009 of plan c1d5bfbc: the cap reads current_round, never an extracted value
        edges = _agent()._fsm_for_run("t", None)["states"]["judge"]["transitions"]

        assert [(e["target_state"], e["priority"]) for e in edges] == [
            (DebateStates.CONCLUDE, 10),
            (DebateStates.PROPOSE, 300),
        ]
        assert edges[0]["conditions"][0]["logic"]["or"][0] == {
            "==": [{"var": "consensus_reached"}, True]
        }
        assert "conditions" not in edges[1] or not edges[1]["conditions"]

    def test_the_consensus_prompt_lists_the_history_and_not_the_trace(self):
        # D-054 of plan 06a5ec0a: without debate_rounds the judge declined consensus every round
        fsm = _agent()._fsm_for_run("t", None)
        field = next(
            f
            for f in fsm["states"]["judge"]["field_extractions"]
            if f["field_name"] == "consensus_reached"
        )

        assert field["context_keys"][-3:] == [
            "counter_argument",
            "current_round",
            "debate_rounds",
        ]
        assert "agent_trace" not in field["context_keys"]
        assert field["required"] is True
        verdict = fsm["states"]["judge"]["field_extractions"][0]
        assert verdict["field_name"] == "judge_verdict" and verdict["required"] is False

    def test_each_persona_reaches_only_the_prompts_that_name_it(self):
        fsm = _agent(
            proposer_persona="PROPTEXT",
            critic_persona="CRITTEXT",
            judge_persona="JUDGETEXT",
        )._fsm_for_run("t", None)
        prompts = _prompts(fsm)

        def holders(text: str) -> set[str]:
            return {name for name, value in prompts.items() if text in value}

        assert holders("PROPTEXT") == {"proposition", "counter_argument"}
        assert holders("CRITTEXT") == {"critique"}
        assert holders("JUDGETEXT") == {"judge_verdict", "consensus_reached"}
        assert fsm["persona"] == "PROPTEXT"

    def test_a_persona_stays_data_and_is_read_at_run_time(self):
        agent = _agent(proposer_persona="a {{debate_persona}} {{x}}")
        agent.critic_persona = "changed {{critique_instructions}}"

        fsm = agent._fsm_for_run("t", None)

        assert fsm["persona"] == "a {{debate_persona}} {{x}}"
        assert "Role: changed {{critique_instructions}}" in _prompts(fsm)["critique"]

    def test_each_prompt_is_the_one_of_its_state_and_field(self):
        texts = build_debate_field_instructions("P", "C", "J", 3)
        fsm = build_debate_fsm("P", "C", "J", 3)

        for key, value in texts.items():
            assert _prompts(fsm)[key].endswith(value), key
        assert set(debate_texts("P", "C", "J", 3)) == {
            "debate_persona",
            "proposition_instructions",
            "critique_instructions",
            "counter_instructions",
            "verdict_instructions",
            "consensus_instructions",
        }

    def test_the_run_block_answers_from_the_reply_and_succeeds_on_the_proposition(self):
        # D-036 of plan 06a5ec0a: no round key is an answer key
        assert _agent()._run_kwargs() == {
            "extra_answer_keys": None,
            "execution_evidence_keys": None,
            "judged": False,
            "success_keys": [ContextKeys.PROPOSITION],
        }

    def test_the_shipped_file_alone_is_not_launchable_without_the_texts(self):
        with pytest.raises(InstantiationError):
            pattern_fsm(pf.load_envelope("debate"), tools=None)

    def test_a_bad_round_option_is_refused_at_instantiation(self):
        texts = debate_texts("", "", "", 3)
        for bad in (0, -1, True, "2", 1.5):
            with pytest.raises(InstantiationError):
                pattern_fsm(
                    pf.load_envelope("debate"),
                    tools=None,
                    options={"max_rounds": bad},
                    variables=texts,
                )

    def test_no_option_selects_the_approval_variant(self):
        envelope = pf.load_envelope("debate")

        assert set(envelope.body.variants) == set()
        assert "approval" not in envelope.body.options_schema.properties
        with pytest.raises(AgentError, match="approval"):
            PatternAgent(envelope, options={"approval": True})


class TestTheRegistrationAndTheBudget:
    def test_the_registration_is_what_the_file_binds_in_order(self):
        agent = _agent()
        api = _FakeApi()

        agent._register_handlers(api, agent._handler_context())

        assert [(h.name, h.priority) for h in api.handlers] == [
            ("DebateJudge", HandlerPriorities.ITERATION_LIMITER - 1),
            ("DebateConsensusReset", HandlerPriorities.TOOL_EXECUTOR),
            ("AgentIterationLimiter", HandlerPriorities.ITERATION_LIMITER),
        ]

    def test_the_budget_is_a_pure_function_of_the_round_cap(self):
        agent = _agent(num_rounds=4)

        assert agent._fsm_budget() == 4 * 3 * 2
        assert agent._budget_kwargs() == {"max_iterations": 24}
        assert agent._options["max_iterations"] == 24

    def test_the_limiter_seam_triggers_at_the_budget_minus_one(self):
        agent = _agent(num_rounds=2)
        limiter = agent._make_iteration_limiter()
        limit = agent._fsm_budget()

        assert limiter({ContextKeys.ITERATION_COUNT: limit - 3}) == {
            ContextKeys.ITERATION_COUNT: limit - 2
        }
        at = limiter({ContextKeys.ITERATION_COUNT: limit - 2})
        assert at[ContextKeys.CONSENSUS_REACHED] is True
        assert at[ContextKeys.SHOULD_TERMINATE] is True


class TestTheSeedIsCallLocal:
    def test_the_context_leads_with_the_rounds_then_the_cap(self):
        agent = _agent(num_rounds=2)

        _, context, _ = agent._run_inputs("a", None)

        keys = list(context)
        assert context[ContextKeys.DEBATE_ROUNDS] == []
        assert context[ContextKeys.CURRENT_ROUND] == 1
        assert context[ContextKeys.MAX_ROUNDS] == 2
        assert (
            keys.index(ContextKeys.DEBATE_ROUNDS)
            < keys.index(ContextKeys.CURRENT_ROUND)
            < keys.index(ContextKeys.MAX_ROUNDS)
        )

    def test_a_caller_cannot_forge_the_rounds_the_cap_or_the_outputs(self):
        agent = _agent(num_rounds=2)

        _, context, _ = agent._run_inputs(
            "a",
            {
                ContextKeys.DEBATE_ROUNDS: ["forged"],
                ContextKeys.CURRENT_ROUND: 99,
                ContextKeys.MAX_ROUNDS: 99,
                ContextKeys.CONSENSUS_REACHED: True,
                ContextKeys.PROPOSITION: "forged",
            },
        )

        assert context[ContextKeys.DEBATE_ROUNDS] == []
        assert context[ContextKeys.CURRENT_ROUND] == 1
        assert context[ContextKeys.MAX_ROUNDS] == 2
        assert ContextKeys.CONSENSUS_REACHED not in context
        assert ContextKeys.PROPOSITION not in context

    def test_two_runs_share_neither_the_context_nor_the_handler_context(self):
        agent = _agent()
        _, first, ctx_one = agent._run_inputs("a", None)
        first[ContextKeys.DEBATE_ROUNDS].append("x")
        _, second, ctx_two = agent._run_inputs("a", None)

        assert second[ContextKeys.DEBATE_ROUNDS] == []
        assert ctx_one.scope is not ctx_two.scope
        assert pf.load_envelope("debate").body.run.context_seed == {
            ContextKeys.DEBATE_ROUNDS: [],
            ContextKeys.CURRENT_ROUND: 1,
        }


class TestTheFactories:
    def test_the_judge_runs_before_the_transition_on_the_judge_state_only(self):
        handler = lib.debate_judge(_ctx())[0]

        assert handler.states == {DebateStates.JUDGE}
        assert handler.priority == HandlerPriorities.ITERATION_LIMITER - 1
        assert handler.should_execute(
            HandlerTiming.PRE_TRANSITION, DebateStates.JUDGE, "x", {}, set()
        )
        assert not handler.should_execute(
            HandlerTiming.CONTEXT_UPDATE, DebateStates.JUDGE, "x", {}, {"a"}
        )
        assert not handler.should_execute(
            HandlerTiming.PRE_TRANSITION, DebateStates.PROPOSE, "x", {}, set()
        )

    def test_the_judge_records_the_round_without_advancing_it(self):
        handler = lib.debate_judge(_ctx(options={"max_rounds": 3}))[0]

        done = handler.execute(
            {
                ContextKeys.CURRENT_ROUND: 1,
                ContextKeys.PROPOSITION: "p",
                ContextKeys.CRITIQUE: "c",
                ContextKeys.COUNTER_ARGUMENT: "a",
                ContextKeys.JUDGE_VERDICT: "v",
            }
        )

        assert done == {
            ContextKeys.DEBATE_ROUNDS: [
                {
                    "round_num": 1,
                    "proposition": "p",
                    "critique": "c",
                    "counter_argument": "a",
                    "judge_verdict": "v",
                }
            ]
        }

    def test_an_empty_proposition_is_restored_from_the_last_recorded_one(self):
        # D-051 of plan 06a5ec0a: the success key must survive a round the proposer left empty
        handler = lib.debate_judge(_ctx())[0]
        rounds = [
            {"round_num": 1, "proposition": "first"},
            {"round_num": 2, "proposition": "second"},
            {"round_num": 3, "proposition": "  "},
        ]

        done = handler.execute(
            {
                ContextKeys.CURRENT_ROUND: 4,
                ContextKeys.DEBATE_ROUNDS: rounds,
                ContextKeys.CONSENSUS_REACHED: True,
            }
        )

        assert done[ContextKeys.PROPOSITION] == "second"
        assert done[ContextKeys.DEBATE_ROUNDS][-1]["proposition"] == ""
        assert (
            lib.debate_judge(_ctx())[0]
            .execute(
                {ContextKeys.PROPOSITION: "now", ContextKeys.DEBATE_ROUNDS: rounds}
            )
            .get(ContextKeys.PROPOSITION)
            is None
        )

    def test_the_cap_forces_consensus_as_a_forced_pass_unless_the_judge_reached_it(
        self,
    ):
        handler = lib.debate_judge(_ctx(options={"max_rounds": 2}))[0]
        forced = handler.execute({ContextKeys.CURRENT_ROUND: 2})
        own = handler.execute(
            {ContextKeys.CURRENT_ROUND: 2, ContextKeys.CONSENSUS_REACHED: True}
        )
        early = handler.execute({ContextKeys.CURRENT_ROUND: 1})

        assert forced[ContextKeys.CONSENSUS_REACHED] is True
        assert forced[ContextKeys.FORCED_STOP_REASON] == StopReason.FORCED_PASS.value
        assert own[ContextKeys.CONSENSUS_REACHED] is True
        assert ContextKeys.FORCED_STOP_REASON not in own
        assert ContextKeys.CONSENSUS_REACHED not in early

    @pytest.mark.parametrize("bad", [0, -1, True, "2", 1.5])
    def test_a_bad_round_option_is_refused_by_the_judge(self, bad):
        with pytest.raises(AgentError, match="max_rounds"):
            lib.debate_judge(_ctx(options={"max_rounds": bad}))

    def test_entering_propose_clears_the_round_and_advances_the_number(self):
        handler = lib.debate_consensus_reset(_ctx())[0]

        done = handler.execute(
            {
                ContextKeys.CURRENT_ROUND: 2,
                ContextKeys.CONSENSUS_REACHED: False,
                ContextKeys.PROPOSITION: "p",
                ContextKeys.DEBATE_ROUNDS: ["kept"],
            }
        )

        assert handler.target_states == {DebateStates.PROPOSE}
        assert done == {
            ContextKeys.CONSENSUS_REACHED: None,
            ContextKeys.PROPOSITION: None,
            ContextKeys.CURRENT_ROUND: 3,
        }
        assert ContextKeys.DEBATE_ROUNDS not in done
        assert (
            lib.debate_consensus_reset(_ctx())[0].execute({})[ContextKeys.CURRENT_ROUND]
            == 2
        )

    def test_the_limiter_records_a_forced_consensus_and_leaves_the_judges_own(self):
        agent = _agent(num_rounds=2)
        check = agent._make_iteration_limiter()
        count = {ContextKeys.ITERATION_COUNT: agent._fsm_budget() - 2}

        forced = check(dict(count))
        own = check({**count, ContextKeys.CONSENSUS_REACHED: True})
        kept = check({**count, ContextKeys.FORCED_STOP_REASON: "forced_pass"})

        assert forced[ContextKeys.FORCED_STOP_REASON] == StopReason.MAX_ITERATIONS.value
        assert ContextKeys.FORCED_STOP_REASON not in own
        assert ContextKeys.FORCED_STOP_REASON not in kept

    def test_the_limiter_limit_is_the_option_then_the_config(self):
        binding = pf.load_envelope("debate").body.handlers[2]
        by_option = lib.debate_limiter(_ctx(options={"max_iterations": 4}), binding)[0]
        by_config = lib.debate_limiter(_ctx(config=runs._config()), binding)[0]

        assert by_option.execute({ContextKeys.ITERATION_COUNT: 2})[
            ContextKeys.CONSENSUS_REACHED
        ]
        assert ContextKeys.CONSENSUS_REACHED not in by_config.execute(
            {ContextKeys.ITERATION_COUNT: 2}
        )
        assert by_option.name == "AgentIterationLimiter"
        assert by_option.priority == HandlerPriorities.ITERATION_LIMITER
        with pytest.raises(AgentError, match="limit must be an int"):
            lib.debate_limiter(_ctx(options={"max_iterations": 0}), binding)

    def test_a_binding_may_move_the_states_never_the_timing(self):
        binding = (
            pf.load_envelope("debate")
            .body.handlers[0]
            .model_copy(update={"states": ["critique"]})
        )

        moved = lib.debate_judge(_ctx(), binding)[0]

        assert moved.states == {"critique"}
        assert moved.should_execute(
            HandlerTiming.PRE_TRANSITION, "critique", "x", {}, set()
        )


class TestTheDebateExecutableIsTheFile:
    def _exe(self):
        registry = ExecutableRegistry()
        registry.load_sources()
        return registry.get("debate")

    def test_the_row_keeps_its_id_and_needs_no_host_object(self):
        executable = self._exe()

        assert executable.template_identity
        assert list(executable.requires.callables) == []
        assert executable.requires.tools == "none"
        assert executable.impl.endswith("DebateAgent")

    def test_the_file_is_in_the_shipped_order(self):
        assert "debate" in pf.PATTERN_ORDER
