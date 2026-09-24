from __future__ import annotations

"""Tests for fsm_llm_agents.maker_checker module."""

from typing import Any

from fsm_llm.definitions import (
    BulkExtractionRequest,
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMDefinition,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm_agents.constants import (
    ContextKeys,
    Defaults,
    HandlerNames,
    MakerCheckerStates,
)
from fsm_llm_agents.definitions import AgentConfig
from fsm_llm_agents.fsm_definitions import build_maker_checker_fsm
from fsm_llm_agents.maker_checker import MakerCheckerAgent

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
        result = agent._make_iteration_limiter()(context)
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
        result = agent._make_iteration_limiter()(context)
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


# -------------------------------------------------------------------------
# FB-01: the check state must not BLOCK when checker_passed never extracts
# -------------------------------------------------------------------------


class _CheckerSilentLLM(LLMInterface):
    """Mock LLM whose checker never yields ``checker_passed`` or ``quality_score``.

    Every other field extracts as a plain string, so the maker produces a draft
    and the checker produces text feedback, but the gating boolean is always
    missing (as a weak model's malformed checker JSON would leave it).
    """

    _SILENT = frozenset({ContextKeys.CHECKER_PASSED, "quality_score"})

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        silent = request.field_name in self._SILENT
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=None if silent else "a draft answer",
            confidence=0.0 if silent else 0.9,
            reasoning="mock",
            is_valid=not silent,
        )

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="final answer", message_type="response", reasoning="mock"
        )


class TestCheckStateNeverBlocks:
    """Plan plan-2026-09-24T045559-3e4eb3e5 step 2 (FB-01)."""

    def test_check_has_unconditional_fallback_edge(self):
        fsm = build_maker_checker_fsm(
            maker_instructions="Write", checker_instructions="Check"
        )
        transitions = fsm["states"]["check"]["transitions"]
        fallbacks = [t for t in transitions if not t.get("conditions")]
        assert len(fallbacks) == 1
        assert fallbacks[0]["priority"] > max(
            t["priority"] for t in transitions if t.get("conditions")
        )

    def test_missing_checker_passed_reaches_output(self):
        # Pre-fix: check had only checker_passed == True/False edges, so a
        # missing value BLOCKED check every turn; no PRE_TRANSITION limiter
        # runs on a BLOCKED turn, and the run raised BudgetExhaustedError.
        max_iterations = 6
        agent = MakerCheckerAgent(
            maker_instructions="Write a haiku",
            checker_instructions="Check the syllables",
            config=AgentConfig(max_iterations=max_iterations),
            llm_interface=_CheckerSilentLLM(),
        )

        loop_counts: list[int] = []
        real_loop = agent._run_conversation_loop

        def _recording_loop(*args: Any, **kwargs: Any):
            responses, final_context, iteration = real_loop(*args, **kwargs)
            loop_counts.append(iteration)
            return responses, final_context, iteration

        agent._run_conversation_loop = _recording_loop  # type: ignore[method-assign]

        result = agent.run("Write a haiku about rain")

        # The loop only returns once the terminal ``output`` state is reached.
        assert loop_counts, "conversation loop never completed"
        assert result.answer
        # Measured 8 turns: the limiter forces checker_passed at iteration 6,
        # then revise -> check -> output. The hard ceiling is 3 x 6 = 18.
        assert loop_counts[0] <= max_iterations + 2


# -------------------------------------------------------------------------
# Step 3.1: each check round is judged afresh (core extracts a key only
# when it is unset, so a stale verdict/draft used to stick for the run)
# -------------------------------------------------------------------------


class _TwoRoundLLM(LLMInterface):
    """Maker drafts ``d1``, ``d2``, ...; the checker rejects round 1 and
    approves round 2. Records every ``checker_passed`` extraction."""

    def __init__(self) -> None:
        self.drafts = 0
        self.verdicts: list[bool] = []

    def extract_bulk_data(
        self, request: BulkExtractionRequest
    ) -> DataExtractionResponse:
        if "You are the MAKER" not in request.system_prompt:
            return DataExtractionResponse(extracted_data={})
        self.drafts += 1
        return DataExtractionResponse(
            extracted_data={ContextKeys.DRAFT_OUTPUT: f"d{self.drafts}"}
        )

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        name = request.field_name
        value: Any
        if name == ContextKeys.CHECKER_PASSED:
            value = len(self.verdicts) >= 1
            self.verdicts.append(value)
        elif name == "quality_score":
            value = 0.1
        else:
            value = f"feedback {len(self.verdicts)}"
        return FieldExtractionResponse(
            field_name=name, value=value, confidence=0.9, reasoning="m", is_valid=True
        )

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="final answer", message_type="response", reasoning="mock"
        )


class TestEachRoundIsRejudged:
    def test_second_check_sees_revised_draft_and_new_verdict(self):
        llm = _TwoRoundLLM()
        agent = MakerCheckerAgent(
            maker_instructions="Write a haiku",
            checker_instructions="Check the syllables",
            config=AgentConfig(max_iterations=10),
            llm_interface=llm,
        )
        result = agent.run("Write a haiku about rain")

        # Pre-fix: checker_passed was extracted once ([False]), revision_count
        # stayed 1 and the draft stayed d1 until the limiter forced a pass.
        assert llm.verdicts == [False, True]
        assert result.final_context[ContextKeys.REVISION_COUNT] == 2
        assert result.final_context[ContextKeys.DRAFT_OUTPUT] == "d2"
        assert not result.final_context.get(ContextKeys.MAX_ITERATIONS_REACHED)

    def test_checker_judges_at_smallest_budget(self):
        # D-014: with the shared `-1` the limiter forced a pass before the
        # first check turn at max_iterations=2, so the checker never judged.
        llm = _TwoRoundLLM()
        agent = MakerCheckerAgent(
            maker_instructions="Write a haiku",
            checker_instructions="Check the syllables",
            config=AgentConfig(max_iterations=2),
            llm_interface=llm,
        )
        agent.run("Write a haiku about rain")
        assert llm.verdicts[:1] == [False]

    def test_limiter_triggers_at_max_not_max_minus_one(self):
        agent = MakerCheckerAgent(
            maker_instructions="w",
            checker_instructions="c",
            config=AgentConfig(max_iterations=5),
        )
        limiter = agent._make_iteration_limiter()
        assert ContextKeys.CHECKER_PASSED not in limiter(
            {ContextKeys.ITERATION_COUNT: 3}
        )
        assert limiter({ContextKeys.ITERATION_COUNT: 4})[ContextKeys.CHECKER_PASSED]


class _HighScoreRejectLLM(_TwoRoundLLM):
    """Checker always says ``checker_passed=False`` but scores 0.9, at or
    above the default threshold, so ``_track_revisions`` forces a pass."""

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        name = request.field_name
        value: Any
        if name == ContextKeys.CHECKER_PASSED:
            value = False
            self.verdicts.append(value)
        elif name == "quality_score":
            value = 0.9
        else:
            value = "feedback"
        return FieldExtractionResponse(
            field_name=name, value=value, confidence=0.9, reasoning="m", is_valid=True
        )


class TestForcedPassShipsJudgedDraft:
    def test_quality_auto_pass_ships_the_scored_draft(self):
        # Step 3.2: a forced pass loses to the extracted False on its own
        # turn and routes through revise; revise entry used to move d1 aside
        # anyway, so the maker wrote d2 and check shipped it unjudged.
        llm = _HighScoreRejectLLM()
        agent = MakerCheckerAgent(
            maker_instructions="Write a haiku",
            checker_instructions="Check the syllables",
            config=AgentConfig(max_iterations=10),
            llm_interface=llm,
        )
        result = agent.run("Write a haiku about rain")

        assert llm.verdicts == [False]
        assert result.final_context[ContextKeys.DRAFT_OUTPUT] == "d1"
        assert result.answer == "d1"
