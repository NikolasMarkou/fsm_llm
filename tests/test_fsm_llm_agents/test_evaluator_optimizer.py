from __future__ import annotations

"""Tests for fsm_llm_agents.evaluator_optimizer module."""


from fsm_llm.definitions import FSMDefinition
from fsm_llm_agents.constants import ContextKeys, Defaults, EvalOptStates, HandlerNames
from fsm_llm_agents.definitions import AgentConfig, EvaluationResult
from fsm_llm_agents.evaluator_optimizer import EvaluatorOptimizerAgent
from fsm_llm_agents.fsm_definitions import build_evalopt_fsm


def _always_pass(output: str, context: dict) -> EvaluationResult:
    return EvaluationResult(passed=True, score=1.0, feedback="OK")


def _always_fail(output: str, context: dict) -> EvaluationResult:
    return EvaluationResult(passed=False, score=0.2, feedback="Needs improvement")


def _score_based(output: str, context: dict) -> EvaluationResult:
    score = 0.5 if "good" in output.lower() else 0.1
    return EvaluationResult(
        passed=score >= 0.5,
        score=score,
        feedback="Acceptable" if score >= 0.5 else "Not good enough",
    )


# -------------------------------------------------------------------------
# EvaluatorOptimizerAgent creation
# -------------------------------------------------------------------------


class TestEvaluatorOptimizerCreation:
    """Tests for EvaluatorOptimizerAgent initialization."""

    def test_create_with_evaluation_fn(self):
        agent = EvaluatorOptimizerAgent(evaluation_fn=_always_pass)
        assert agent.evaluation_fn is _always_pass
        assert agent.config is not None
        assert agent.max_refinements == Defaults.MAX_REFINEMENTS

    def test_create_with_custom_config(self):
        config = AgentConfig(max_iterations=5, model="gpt-4o-mini")
        agent = EvaluatorOptimizerAgent(evaluation_fn=_always_pass, config=config)
        assert agent.config.max_iterations == 5
        assert agent.config.model == "gpt-4o-mini"

    def test_create_with_max_refinements(self):
        agent = EvaluatorOptimizerAgent(
            evaluation_fn=_always_pass, max_refinements=7
        )
        assert agent.max_refinements == 7

    def test_create_default_max_refinements(self):
        agent = EvaluatorOptimizerAgent(evaluation_fn=_always_pass)
        assert agent.max_refinements == 3

    def test_create_default_config(self):
        agent = EvaluatorOptimizerAgent(evaluation_fn=_always_pass)
        assert agent.config.max_iterations == Defaults.MAX_ITERATIONS
        assert agent.config.temperature == Defaults.TEMPERATURE

    def test_config_override_works(self):
        config = AgentConfig(
            model="gpt-4",
            max_iterations=20,
            timeout_seconds=60.0,
            temperature=0.8,
        )
        agent = EvaluatorOptimizerAgent(evaluation_fn=_always_pass, config=config)
        assert agent.config.model == "gpt-4"
        assert agent.config.max_iterations == 20
        assert agent.config.timeout_seconds == 60.0
        assert agent.config.temperature == 0.8

    def test_has_run_method(self):
        agent = EvaluatorOptimizerAgent(evaluation_fn=_always_pass)
        assert hasattr(agent, "run")
        assert callable(agent.run)

    def test_no_tool_registry_needed(self):
        """EvaluatorOptimizerAgent does not require a ToolRegistry."""
        agent = EvaluatorOptimizerAgent(evaluation_fn=_always_pass)
        assert not hasattr(agent, "tools")

    def test_stores_api_kwargs(self):
        agent = EvaluatorOptimizerAgent(
            evaluation_fn=_always_pass,
            some_extra="value",
        )
        assert agent._api_kwargs == {"some_extra": "value"}


# -------------------------------------------------------------------------
# EvalOpt FSM definition
# -------------------------------------------------------------------------


class TestBuildEvalOptFsm:
    """Tests for build_evalopt_fsm function."""

    def test_returns_dict(self):
        fsm = build_evalopt_fsm()
        assert isinstance(fsm, dict)

    def test_basic_structure(self):
        fsm = build_evalopt_fsm()
        assert fsm["name"] == "evalopt_agent"
        assert fsm["initial_state"] == "generate"
        assert "states" in fsm
        assert "persona" in fsm

    def test_has_required_states(self):
        fsm = build_evalopt_fsm()
        expected = {"generate", "evaluate", "refine", "output"}
        assert set(fsm["states"].keys()) == expected

    def test_generate_transitions_to_evaluate(self):
        fsm = build_evalopt_fsm()
        transitions = fsm["states"]["generate"]["transitions"]
        assert len(transitions) == 1
        assert transitions[0]["target_state"] == "evaluate"

    def test_evaluate_transitions_to_output_and_refine(self):
        fsm = build_evalopt_fsm()
        transitions = fsm["states"]["evaluate"]["transitions"]
        targets = {t["target_state"] for t in transitions}
        assert targets == {"output", "refine"}

    def test_evaluate_output_transition_has_higher_priority(self):
        """Lower priority number = higher confidence in TransitionEvaluator."""
        fsm = build_evalopt_fsm()
        transitions = fsm["states"]["evaluate"]["transitions"]
        output_t = next(t for t in transitions if t["target_state"] == "output")
        refine_t = next(t for t in transitions if t["target_state"] == "refine")
        assert output_t["priority"] < refine_t["priority"]

    def test_refine_transitions_to_evaluate(self):
        fsm = build_evalopt_fsm()
        transitions = fsm["states"]["refine"]["transitions"]
        assert len(transitions) == 1
        assert transitions[0]["target_state"] == "evaluate"

    def test_output_is_terminal(self):
        fsm = build_evalopt_fsm()
        assert fsm["states"]["output"]["transitions"] == []

    def test_fsm_is_valid_definition(self):
        """The generated FSM should be parseable as an FSMDefinition."""
        fsm = build_evalopt_fsm()
        fsm_def = FSMDefinition(**fsm)
        assert fsm_def.name == "evalopt_agent"
        assert fsm_def.initial_state == "generate"

    def test_custom_task_description(self):
        fsm = build_evalopt_fsm(task_description="Write clean code")
        assert fsm["description"] == "Write clean code"

    def test_default_description(self):
        fsm = build_evalopt_fsm()
        assert fsm["description"] == "Evaluator-Optimizer agent"

    def test_persona_mentions_refinement(self):
        fsm = build_evalopt_fsm()
        persona_lower = fsm["persona"].lower()
        assert "refin" in persona_lower or "quality" in persona_lower

    def test_states_have_extraction_instructions(self):
        fsm = build_evalopt_fsm()
        for state_id in ("generate", "refine", "output"):
            assert "extraction_instructions" in fsm["states"][state_id], (
                f"State '{state_id}' missing extraction_instructions"
            )

    def test_states_have_response_instructions(self):
        fsm = build_evalopt_fsm()
        for state_id in ("generate", "evaluate", "refine", "output"):
            assert "response_instructions" in fsm["states"][state_id], (
                f"State '{state_id}' missing response_instructions"
            )


# -------------------------------------------------------------------------
# EvaluationResult model
# -------------------------------------------------------------------------


class TestEvaluationResultModel:
    """Tests for EvaluationResult Pydantic model."""

    def test_basic_creation(self):
        result = EvaluationResult(passed=True, score=0.9, feedback="Looks good")
        assert result.passed is True
        assert result.score == 0.9
        assert result.feedback == "Looks good"

    def test_defaults(self):
        result = EvaluationResult(passed=False)
        assert result.score == 0.0
        assert result.feedback == ""
        assert result.criteria_met == []

    def test_with_criteria_met(self):
        result = EvaluationResult(
            passed=True,
            score=1.0,
            feedback="All good",
            criteria_met=["correctness", "style"],
        )
        assert result.criteria_met == ["correctness", "style"]

    def test_model_dump(self):
        result = EvaluationResult(passed=True, score=0.85, feedback="Good")
        data = result.model_dump()
        assert data["passed"] is True
        assert data["score"] == 0.85
        assert data["feedback"] == "Good"
        assert data["criteria_met"] == []


# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------


class TestEvalOptConstants:
    """Tests for EvalOpt-specific constants."""

    def test_evalopt_states(self):
        assert EvalOptStates.GENERATE == "generate"
        assert EvalOptStates.EVALUATE == "evaluate"
        assert EvalOptStates.REFINE == "refine"
        assert EvalOptStates.OUTPUT == "output"

    def test_context_keys_generated_output(self):
        assert hasattr(ContextKeys, "GENERATED_OUTPUT")
        assert isinstance(ContextKeys.GENERATED_OUTPUT, str)

    def test_context_keys_evaluation_result(self):
        assert hasattr(ContextKeys, "EVALUATION_RESULT")
        assert isinstance(ContextKeys.EVALUATION_RESULT, str)

    def test_context_keys_refinement_feedback(self):
        assert hasattr(ContextKeys, "REFINEMENT_FEEDBACK")
        assert isinstance(ContextKeys.REFINEMENT_FEEDBACK, str)

    def test_context_keys_refinement_count(self):
        assert hasattr(ContextKeys, "REFINEMENT_COUNT")
        assert isinstance(ContextKeys.REFINEMENT_COUNT, str)

    def test_context_keys_evaluation_passed(self):
        assert hasattr(ContextKeys, "EVALUATION_PASSED")
        assert isinstance(ContextKeys.EVALUATION_PASSED, str)

    def test_defaults_max_refinements(self):
        assert Defaults.MAX_REFINEMENTS == 3

    def test_handler_name_eval_opt_evaluator(self):
        assert HandlerNames.EVAL_OPT_EVALUATOR == "EvalOptEvaluator"


# -------------------------------------------------------------------------
# Internal handler logic (unit testable without LLM)
# -------------------------------------------------------------------------


class TestEvalOptHandlers:
    """Tests for internal handler methods without requiring LLM."""

    def test_run_evaluation_passes(self):
        agent = EvaluatorOptimizerAgent(evaluation_fn=_always_pass)
        context = {
            ContextKeys.GENERATED_OUTPUT: "some output",
            ContextKeys.REFINEMENT_COUNT: 0,
            ContextKeys.AGENT_TRACE: [],
        }
        result = agent._run_evaluation(context)
        assert result[ContextKeys.EVALUATION_PASSED] is True

    def test_run_evaluation_fails(self):
        agent = EvaluatorOptimizerAgent(evaluation_fn=_always_fail)
        context = {
            ContextKeys.GENERATED_OUTPUT: "some output",
            ContextKeys.REFINEMENT_COUNT: 0,
            ContextKeys.AGENT_TRACE: [],
            "_max_refinements": 3,
        }
        result = agent._run_evaluation(context)
        assert result[ContextKeys.EVALUATION_PASSED] is False
        assert result[ContextKeys.REFINEMENT_COUNT] == 1

    def test_run_evaluation_forces_pass_at_max_refinements(self):
        agent = EvaluatorOptimizerAgent(
            evaluation_fn=_always_fail, max_refinements=2
        )
        context = {
            ContextKeys.GENERATED_OUTPUT: "some output",
            ContextKeys.REFINEMENT_COUNT: 2,
            ContextKeys.AGENT_TRACE: [],
            "_max_refinements": 2,
        }
        result = agent._run_evaluation(context)
        assert result[ContextKeys.EVALUATION_PASSED] is True

    def test_run_evaluation_handles_exception(self):
        def exploding_eval(output, ctx):
            raise RuntimeError("boom")

        agent = EvaluatorOptimizerAgent(evaluation_fn=exploding_eval)
        context = {
            ContextKeys.GENERATED_OUTPUT: "some output",
            ContextKeys.REFINEMENT_COUNT: 0,
            ContextKeys.AGENT_TRACE: [],
            "_max_refinements": 3,
        }
        result = agent._run_evaluation(context)
        assert result[ContextKeys.EVALUATION_PASSED] is False
        assert "boom" in result[ContextKeys.REFINEMENT_FEEDBACK]

    def test_check_iteration_limit_under(self):
        agent = EvaluatorOptimizerAgent(evaluation_fn=_always_pass)
        context = {ContextKeys.ITERATION_COUNT: 2}
        result = agent._check_iteration_limit(context)
        assert result[ContextKeys.ITERATION_COUNT] == 3
        assert ContextKeys.MAX_ITERATIONS_REACHED not in result

    def test_check_iteration_limit_reached(self):
        config = AgentConfig(max_iterations=5)
        agent = EvaluatorOptimizerAgent(
            evaluation_fn=_always_pass, config=config
        )
        context = {ContextKeys.ITERATION_COUNT: 4}
        result = agent._check_iteration_limit(context)
        assert result[ContextKeys.MAX_ITERATIONS_REACHED] is True
        assert result[ContextKeys.EVALUATION_PASSED] is True

    def test_extract_answer_from_final_answer(self):
        agent = EvaluatorOptimizerAgent(evaluation_fn=_always_pass)
        context = {ContextKeys.FINAL_ANSWER: "This is the final answer"}
        answer = agent._extract_answer(context, ["response1"])
        assert answer == "This is the final answer"

    def test_extract_answer_from_generated_output(self):
        agent = EvaluatorOptimizerAgent(evaluation_fn=_always_pass)
        context = {ContextKeys.GENERATED_OUTPUT: "The generated output here"}
        answer = agent._extract_answer(context, ["response1"])
        assert answer == "The generated output here"

    def test_extract_answer_fallback_to_response(self):
        agent = EvaluatorOptimizerAgent(evaluation_fn=_always_pass)
        context = {}
        answer = agent._extract_answer(context, ["", "A valid long response here"])
        assert answer == "A valid long response here"

    def test_extract_answer_default(self):
        agent = EvaluatorOptimizerAgent(evaluation_fn=_always_pass)
        answer = agent._extract_answer({}, ["", ""])
        assert "could not" in answer.lower()
