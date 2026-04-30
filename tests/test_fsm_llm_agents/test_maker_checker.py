from __future__ import annotations

"""Tests for fsm_llm_agents.maker_checker module."""


from fsm_llm.dialog.definitions import FSMDefinition
from fsm_llm.stdlib.agents.constants import (
    ContextKeys,
    Defaults,
    HandlerNames,
    MakerCheckerStates,
)
from fsm_llm.stdlib.agents.definitions import AgentConfig
from fsm_llm.stdlib.agents.fsm_definitions import build_maker_checker_fsm
from fsm_llm.stdlib.agents.maker_checker import MakerCheckerAgent

# -------------------------------------------------------------------------
# MakerCheckerAgent creation
# -------------------------------------------------------------------------


class TestMakerCheckerCreation:
    """Tests for MakerCheckerAgent initialization."""

    def test_create_with_instructions(self):
        agent = MakerCheckerAgent(
            maker_instructions="Write a poem",
            checker_instructions="Check for rhyme and meter",
        )
        assert agent.maker_instructions == "Write a poem"
        assert agent.checker_instructions == "Check for rhyme and meter"
        assert agent.config is not None

    def test_create_with_empty_instructions(self):
        agent = MakerCheckerAgent(
            maker_instructions="",
            checker_instructions="",
        )
        assert agent.maker_instructions == ""
        assert agent.checker_instructions == ""

    def test_create_with_custom_config(self):
        config = AgentConfig(max_iterations=8, model="gpt-4o")
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
            config=config,
        )
        assert agent.config.max_iterations == 8
        assert agent.config.model == "gpt-4o"

    def test_create_with_max_revisions(self):
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
            max_revisions=5,
        )
        assert agent.max_revisions == 5

    def test_create_with_quality_threshold(self):
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
            quality_threshold=0.9,
        )
        assert agent.quality_threshold == 0.9

    def test_default_max_revisions(self):
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
        )
        assert agent.max_revisions == Defaults.MAX_REVISIONS

    def test_default_quality_threshold(self):
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
        )
        assert agent.quality_threshold == Defaults.QUALITY_THRESHOLD

    def test_stores_maker_instructions(self):
        agent = MakerCheckerAgent(
            maker_instructions="Write a professional email",
            checker_instructions="Check tone",
        )
        assert agent.maker_instructions == "Write a professional email"

    def test_stores_checker_instructions(self):
        agent = MakerCheckerAgent(
            maker_instructions="Write code",
            checker_instructions="Review for security and correctness",
        )
        assert agent.checker_instructions == "Review for security and correctness"

    def test_has_run_method(self):
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
        )
        assert hasattr(agent, "run")
        assert callable(agent.run)

    def test_no_tool_registry_needed(self):
        """MakerCheckerAgent does not require a ToolRegistry."""
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
        )
        assert not hasattr(agent, "tools")

    def test_config_override_works(self):
        config = AgentConfig(
            model="gpt-4",
            max_iterations=15,
            timeout_seconds=120.0,
            temperature=0.3,
        )
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
            config=config,
        )
        assert agent.config.model == "gpt-4"
        assert agent.config.max_iterations == 15
        assert agent.config.timeout_seconds == 120.0
        assert agent.config.temperature == 0.3

    def test_stores_api_kwargs(self):
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
            extra_param="test_value",
        )
        assert agent._api_kwargs == {"extra_param": "test_value"}


# -------------------------------------------------------------------------
# Maker-Checker FSM definition
# -------------------------------------------------------------------------


class TestBuildMakerCheckerFsm:
    """Tests for build_maker_checker_fsm function."""

    def test_returns_dict(self):
        fsm = build_maker_checker_fsm(
            maker_instructions="Write", checker_instructions="Check"
        )
        assert isinstance(fsm, dict)

    def test_basic_structure(self):
        fsm = build_maker_checker_fsm(
            maker_instructions="Write", checker_instructions="Check"
        )
        assert fsm["name"] == "maker_checker_agent"
        assert fsm["initial_state"] == "make"
        assert "states" in fsm
        assert "persona" in fsm

    def test_has_required_states(self):
        fsm = build_maker_checker_fsm(
            maker_instructions="Write", checker_instructions="Check"
        )
        expected = {"make", "check", "revise", "output"}
        assert set(fsm["states"].keys()) == expected

    def test_make_transitions_to_check(self):
        fsm = build_maker_checker_fsm(
            maker_instructions="Write", checker_instructions="Check"
        )
        transitions = fsm["states"]["make"]["transitions"]
        assert len(transitions) == 1
        assert transitions[0]["target_state"] == "check"

    def test_check_transitions_to_output_and_revise(self):
        fsm = build_maker_checker_fsm(
            maker_instructions="Write", checker_instructions="Check"
        )
        transitions = fsm["states"]["check"]["transitions"]
        targets = {t["target_state"] for t in transitions}
        assert targets == {"output", "revise"}

    def test_check_output_transition_has_higher_priority(self):
        """Lower priority number = higher confidence in TransitionEvaluator."""
        fsm = build_maker_checker_fsm(
            maker_instructions="Write", checker_instructions="Check"
        )
        transitions = fsm["states"]["check"]["transitions"]
        output_t = next(t for t in transitions if t["target_state"] == "output")
        revise_t = next(t for t in transitions if t["target_state"] == "revise")
        assert output_t["priority"] < revise_t["priority"]

    def test_revise_transitions_to_check(self):
        fsm = build_maker_checker_fsm(
            maker_instructions="Write", checker_instructions="Check"
        )
        transitions = fsm["states"]["revise"]["transitions"]
        assert len(transitions) == 1
        assert transitions[0]["target_state"] == "check"

    def test_output_is_terminal(self):
        fsm = build_maker_checker_fsm(
            maker_instructions="Write", checker_instructions="Check"
        )
        assert fsm["states"]["output"]["transitions"] == []

    def test_fsm_is_valid_definition(self):
        """The generated FSM should be parseable as an FSMDefinition."""
        fsm = build_maker_checker_fsm(
            maker_instructions="Write a report",
            checker_instructions="Check for clarity",
        )
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "maker_checker_agent"
        assert fsm_def.initial_state == "make"

    def test_custom_task_description(self):
        fsm = build_maker_checker_fsm(
            maker_instructions="Write",
            checker_instructions="Check",
            task_description="Draft an apology email",
        )
        assert fsm["description"] == "Draft an apology email"

    def test_default_description(self):
        fsm = build_maker_checker_fsm(
            maker_instructions="Write", checker_instructions="Check"
        )
        assert fsm["description"] == "Maker-Checker agent"

    def test_persona_mentions_quality(self):
        fsm = build_maker_checker_fsm(
            maker_instructions="Write", checker_instructions="Check"
        )
        persona_lower = fsm["persona"].lower()
        assert "quality" in persona_lower or "maker" in persona_lower

    def test_states_have_extraction_instructions(self):
        fsm = build_maker_checker_fsm(
            maker_instructions="Write", checker_instructions="Check"
        )
        for state_id in ("make", "check", "revise", "output"):
            assert "extraction_instructions" in fsm["states"][state_id], (
                f"State '{state_id}' missing extraction_instructions"
            )

    def test_states_have_response_instructions(self):
        fsm = build_maker_checker_fsm(
            maker_instructions="Write", checker_instructions="Check"
        )
        for state_id in ("make", "check", "revise", "output"):
            assert "response_instructions" in fsm["states"][state_id], (
                f"State '{state_id}' missing response_instructions"
            )

    def test_maker_instructions_in_make_state(self):
        fsm = build_maker_checker_fsm(
            maker_instructions="Write a haiku",
            checker_instructions="Check syllable count",
        )
        extraction = fsm["states"]["make"]["extraction_instructions"]
        assert "haiku" in extraction.lower() or "Write a haiku" in extraction


# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------


class TestMakerCheckerConstants:
    """Tests for MakerChecker-specific constants."""

    def test_maker_checker_states(self):
        assert MakerCheckerStates.MAKE == "make"
        assert MakerCheckerStates.CHECK == "check"
        assert MakerCheckerStates.REVISE == "revise"
        assert MakerCheckerStates.OUTPUT == "output"

    def test_context_keys_draft_output(self):
        assert hasattr(ContextKeys, "DRAFT_OUTPUT")
        assert isinstance(ContextKeys.DRAFT_OUTPUT, str)

    def test_context_keys_checker_feedback(self):
        assert hasattr(ContextKeys, "CHECKER_FEEDBACK")
        assert isinstance(ContextKeys.CHECKER_FEEDBACK, str)

    def test_context_keys_checker_passed(self):
        assert hasattr(ContextKeys, "CHECKER_PASSED")
        assert isinstance(ContextKeys.CHECKER_PASSED, str)

    def test_context_keys_revision_count(self):
        assert hasattr(ContextKeys, "REVISION_COUNT")
        assert isinstance(ContextKeys.REVISION_COUNT, str)

    def test_defaults_max_revisions(self):
        assert Defaults.MAX_REVISIONS == 3

    def test_defaults_quality_threshold(self):
        assert Defaults.QUALITY_THRESHOLD == 0.7

    def test_handler_name_maker_checker_checker(self):
        assert HandlerNames.MAKER_CHECKER_CHECKER == "MakerCheckerChecker"


# -------------------------------------------------------------------------
# Internal handler logic (unit testable without LLM)
# -------------------------------------------------------------------------


class TestMakerCheckerHandlers:
    """Tests for internal handler methods without requiring LLM."""

    def test_track_revisions_increments(self):
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
        )
        context = {
            ContextKeys.REVISION_COUNT: 0,
            ContextKeys.AGENT_TRACE: [],
            "_max_revisions": 3,
        }
        result = agent._track_revisions(context)
        assert result[ContextKeys.REVISION_COUNT] == 1

    def test_track_revisions_forces_pass_at_max(self):
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
            max_revisions=2,
        )
        context = {
            ContextKeys.REVISION_COUNT: 1,
            ContextKeys.AGENT_TRACE: [],
            "_max_revisions": 2,
        }
        result = agent._track_revisions(context)
        assert result[ContextKeys.REVISION_COUNT] == 2
        assert result[ContextKeys.CHECKER_PASSED] is True

    def test_track_revisions_no_force_under_max(self):
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
            max_revisions=5,
        )
        context = {
            ContextKeys.REVISION_COUNT: 1,
            ContextKeys.AGENT_TRACE: [],
            "_max_revisions": 5,
        }
        result = agent._track_revisions(context)
        assert ContextKeys.CHECKER_PASSED not in result

    def test_track_revisions_records_trace(self):
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
        )
        context = {
            ContextKeys.REVISION_COUNT: 0,
            ContextKeys.AGENT_TRACE: [],
            "_max_revisions": 3,
        }
        result = agent._track_revisions(context)
        trace = result[ContextKeys.AGENT_TRACE]
        assert len(trace) == 1
        assert trace[0]["type"] == "check"

    def test_check_iteration_limit_under(self):
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
        )
        context = {ContextKeys.ITERATION_COUNT: 2}
        result = agent._check_iteration_limit(context)
        assert result[ContextKeys.ITERATION_COUNT] == 3
        assert ContextKeys.MAX_ITERATIONS_REACHED not in result

    def test_check_iteration_limit_reached(self):
        config = AgentConfig(max_iterations=5)
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
            config=config,
        )
        context = {ContextKeys.ITERATION_COUNT: 4}
        result = agent._check_iteration_limit(context)
        assert result[ContextKeys.MAX_ITERATIONS_REACHED] is True
        assert result[ContextKeys.CHECKER_PASSED] is True

    def test_extract_answer_from_final_answer(self):
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
        )
        context = {ContextKeys.FINAL_ANSWER: "This is the final answer"}
        answer = agent._extract_answer(context, ["response1"])
        assert answer == "This is the final answer"

    def test_extract_answer_from_draft_output(self):
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
        )
        context = {ContextKeys.DRAFT_OUTPUT: "The draft output content here"}
        answer = agent._extract_answer(
            context, ["response1"], extra_keys=[ContextKeys.DRAFT_OUTPUT]
        )
        assert answer == "The draft output content here"

    def test_extract_answer_fallback_to_response(self):
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
        )
        context = {}
        answer = agent._extract_answer(context, ["", "A valid long response here"])
        assert answer == "A valid long response here"

    def test_extract_answer_default(self):
        agent = MakerCheckerAgent(
            maker_instructions="draft",
            checker_instructions="review",
        )
        answer = agent._extract_answer({}, ["", ""])
        assert "could not" in answer.lower()
