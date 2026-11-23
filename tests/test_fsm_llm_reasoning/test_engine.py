"""
Unit tests for the reasoning engine components.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from fsm_llm.constants import MESSAGE_TRUNCATION_SUFFIX
from fsm_llm.logging import logger
from fsm_llm.reasoning import __main__ as cli
from fsm_llm.reasoning.constants import ContextKeys, Defaults, ReasoningType
from fsm_llm.reasoning.definitions import (
    ReasoningClassificationResult,
    ReasoningStep,
    ReasoningStepType,
    ValidationResult,
)
from fsm_llm.reasoning.engine import ReasoningEngine, _strategy_for
from fsm_llm.reasoning.exceptions import (
    ReasoningClassificationError,
)
from fsm_llm.reasoning.handlers import OutputFormatter, ReasoningHandlers
from fsm_llm.reasoning.utilities import map_reasoning_type

from .test_engine_scripted import _PROBLEM, _VALID_SCRIPT, _engine, _ScriptedLLM


class TestReasoningModels:
    """Test Pydantic models."""

    def test_validation_result(self):
        """Test ValidationResult model."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.95,
            checks={"has_solution": True, "complete": True},
            issues=[],
        )

        assert result.is_valid is True
        assert result.confidence == 0.95
        assert len(result.issues) == 0

    def test_reasoning_step(self):
        """Test ReasoningStep model."""
        step = ReasoningStep(
            step_type=ReasoningStepType.ANALYSIS,
            content="Breaking down the problem",
            confidence=0.8,
            evidence=["observation1", "observation2"],
        )

        assert step.step_type == ReasoningStepType.ANALYSIS
        assert step.confidence == 0.8
        assert len(step.evidence) == 2

    def test_classification_result(self):
        """Test ReasoningClassificationResult model."""
        result = ReasoningClassificationResult(
            recommended_type="analytical",
            justification="Problem requires systematic analysis",
            domain="technical",
            alternatives=["deductive"],
            confidence="high",
        )

        assert result.recommended_type == "analytical"
        assert result.justification == "Problem requires systematic analysis"


class TestReasoningHandlers:
    """Test handler implementations."""

    def test_validate_solution(self):
        """Test solution validation handler."""
        handlers = ReasoningHandlers()

        # Valid solution
        context = {
            ContextKeys.PROPOSED_SOLUTION: "This is a detailed solution with multiple steps addressing the core problem",
            ContextKeys.KEY_INSIGHTS: ["insight1", "insight2"],
        }

        result = handlers.validate_solution(context)

        assert result[ContextKeys.VALIDATION_RESULT] is True
        assert result[ContextKeys.SOLUTION_CONFIDENCE] == 1.0
        assert result[ContextKeys.VALIDATION_CHECKS]["has_solution"] is True
        assert result[ContextKeys.VALIDATION_CHECKS]["has_insights"] is True
        assert result[ContextKeys.VALIDATION_CHECKS]["sufficient_detail"] is True

    def test_update_reasoning_trace(self):
        """Test reasoning trace update."""
        handlers = ReasoningHandlers()

        context = {
            "_current_state": "analyze",
            "_previous_state": "start",
            ContextKeys.REASONING_TRACE: [],
            ContextKeys.PROBLEM_TYPE: "technical",
        }

        result = handlers.update_reasoning_trace(context)

        assert len(result[ContextKeys.REASONING_TRACE]) == 1
        trace_step = result[ContextKeys.REASONING_TRACE][0]
        assert trace_step["from"] == "start"
        assert trace_step["to"] == "analyze"
        # Only specific ContextKeys are snapshotted
        assert ContextKeys.PROBLEM_TYPE in trace_step["context_snapshot"]
        assert "_current_state" not in trace_step["context_snapshot"]
        assert "_previous_state" not in trace_step["context_snapshot"]


class TestReasoningUtils:
    """Test utility functions."""

    def test_map_reasoning_type(self):
        """Test reasoning type mapping."""
        assert map_reasoning_type("ANALYTICAL") == "analytical"
        assert map_reasoning_type("creative") == "creative"
        assert map_reasoning_type("unknown") == "analytical"  # default


class TestReasoningEngine:
    """Test main reasoning engine."""

    def test_reasoning_type_enum(self):
        """Test ReasoningType enum."""
        assert ReasoningType.ANALYTICAL.value == "analytical"
        assert ReasoningType.CREATIVE.value == "creative"

        # All reasoning types should have corresponding values
        expected_enum_values = [
            "simple_calculator",
            "analytical",
            "deductive",
            "inductive",
            "abductive",
            "analogical",
            "creative",
            "critical",
            "hybrid",
        ]
        actual_enum_values = [rt.value for rt in ReasoningType]
        for val in expected_enum_values:
            assert val in actual_enum_values
        assert len(actual_enum_values) == len(expected_enum_values)


class TestTheStrategyIsChosenByTheClassifiedTypeThenThePreferredOne:
    """D-031 (plan baa37f40, step 15): ``_strategy_for`` is the one rule that turns the
    classifier's recommendation and ``preferred_reasoning_type`` into a strategy name. It
    replaces the choice of ``_prepare_reasoning_execution`` (and the dead branch of D-009 that
    forced ``analytical`` for an unknown preferred type)."""

    @staticmethod
    def _warnings(call, *args: Any) -> tuple[str, str]:
        from io import StringIO

        # `fsm_llm.logging.logger` is loguru, not stdlib logging, so `caplog` cannot see it
        buf = StringIO()
        logger.enable("fsm_llm")  # library logging is off by default (D-004)
        sink_id = logger.add(buf, level="WARNING")
        try:
            return call(*args), buf.getvalue()
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")

    def test_the_classified_type_is_mapped_by_value_and_alias(self):
        assert _strategy_for("deductive", None) == "deductive"
        assert _strategy_for("  Math ", None) == "simple_calculator"
        assert _strategy_for("brainstorm", "") == "creative"

    def test_an_unknown_classified_type_is_analytical_with_the_warning(self):
        chosen, warned = self._warnings(_strategy_for, "no such type", None)
        assert chosen == "analytical"
        assert "Unknown reasoning type" in warned

    def test_a_known_preferred_type_wins_over_the_classified_one(self):
        assert _strategy_for("deductive", "creative") == "creative"
        assert _strategy_for("deductive", "direct computation") == "simple_calculator"

    def test_an_unknown_preferred_type_is_ignored_with_one_warning(self):
        """The old branch was dead: ``map_reasoning_type`` turned an unknown spelling into
        ``analytical``, so an unknown ``--type`` forced ``analytical`` over the classified
        strategy. It is ignored now, as the code's own comment said."""
        chosen, warned = self._warnings(_strategy_for, "deductive", "no such type")
        assert chosen == "deductive"
        assert warned.count("names no known strategy") == 1

    def test_a_preferred_value_that_is_not_text_never_raises(self):
        assert _strategy_for("deductive", 3) == "deductive"
        assert _strategy_for("deductive", ["creative"]) == "deductive"


class TestTheEntryHandlerNamesTheStrategyTheCallRuns:
    """``_select_reasoning_type`` reads the target by the call executor's own rule
    (``chosen_target``) and never a model value (``reasoning_strategy``)."""

    @staticmethod
    def _delta(**context: Any) -> dict[str, Any]:
        engine = ReasoningEngine(model="mock", llm_interface=_ScriptedLLM({}))
        return engine._select_reasoning_type(context)

    def test_the_name_is_the_one_of_the_target_id(self):
        delta = self._delta(classified_problem_type="reasoning.deductive")
        assert delta[ContextKeys.REASONING_TYPE_SELECTED] == "deductive"

    def test_a_model_strategy_never_beats_the_classified_target(self):
        """The old F-RE-06 row: no spelling of ``reasoning_strategy`` is read at all."""
        chosen = {
            self._delta(
                reasoning_strategy=spelling,
                classified_problem_type="reasoning.deductive",
            )[ContextKeys.REASONING_TYPE_SELECTED]
            for spelling in ("direct computation", "simple_calculator", "arithmetic")
        }
        assert chosen == {ReasoningType.DEDUCTIVE.value}

    @pytest.mark.parametrize("value", [None, "", "deductive", "reasoning.nope", 3])
    def test_a_value_outside_the_closed_list_is_the_fallback(self, value):
        """The call falls back for a value that is not one of its targets, and the name
        follows the same rule: never another loaded strategy (D-009 by construction)."""
        delta = self._delta(classified_problem_type=value)
        assert delta[ContextKeys.REASONING_TYPE_SELECTED] == "analytical"

    def test_constraints_default_to_an_empty_list_and_a_callers_value_is_kept(self):
        assert self._delta()[ContextKeys.CONSTRAINTS] == []
        kept = self._delta(constraints=["no loops"])
        assert ContextKeys.CONSTRAINTS not in kept


class TestExecutionTime:
    """F-XC-03: the trace carries the measured wall-clock seconds."""

    def test_solve_reports_a_float(self):
        _solution, trace_info = _engine(_ScriptedLLM(_VALID_SCRIPT)).solve_problem(
            _PROBLEM
        )
        seconds = trace_info["reasoning_trace"]["execution_time_seconds"]
        assert isinstance(seconds, float)
        assert seconds >= 0

    def test_cli_json_metadata_shows_it(self):
        _solution, trace_info = _engine(_ScriptedLLM(_VALID_SCRIPT)).solve_problem(
            _PROBLEM
        )
        data = json.loads(cli.format_solution_output("x", trace_info, "json", False))
        assert isinstance(data["metadata"]["execution_time_seconds"], float)


class TestDirectComputationIsNotASpecialCase:
    """F-RE-06: only the spelling the prompts ask for (simple_calculator) counts,
    and it picks the strategy the same way as any other spelling."""

    def test_the_classified_calculator_spelling_maps_to_the_calculator(self):
        assert _strategy_for("direct computation", None) == "simple_calculator"

    def test_validator_reads_simple_calculator_not_direct_computation(self):
        base = {
            ContextKeys.PROPOSED_SOLUTION: "4",
            ContextKeys.PROBLEM_STATEMENT: "What is 2 plus 2?",
        }
        simple = ReasoningHandlers.validate_solution(
            {**base, ContextKeys.REASONING_STRATEGY: "simple_calculator"}
        )
        other = ReasoningHandlers.validate_solution(
            {**base, ContextKeys.REASONING_STRATEGY: "direct computation"}
        )
        assert simple[ContextKeys.VALIDATION_RESULT] is True
        assert other[ContextKeys.VALIDATION_RESULT] is False


class TestOnlyTheOrchestratorTraces:
    """F-RE-05: the tracer runs on the orchestrator API only."""

    def test_classifier_has_no_tracer(self):
        engine = ReasoningEngine(model="mock", llm_interface=_ScriptedLLM({}))
        names = lambda api: {  # noqa: E731
            h.name for h in api.handler_system.handlers if hasattr(h, "name")
        }
        assert "ReasoningTracer" in names(engine.orchestrator)
        assert "ReasoningTracer" not in names(engine.classifier)


class TestPrunedStringMarker:
    """F-RE-10: the truncation marker is core's one spelling."""

    def test_long_string_ends_in_the_core_suffix(self):
        long = "x" * (Defaults.PRUNE_STRING_MAX_LENGTH + 10)
        context = {
            ContextKeys.OBSERVATIONS: long,
            "padding": "y" * Defaults.CONTEXT_PRUNE_THRESHOLD,
        }
        pruned = ReasoningHandlers.prune_context(context)
        assert pruned[ContextKeys.OBSERVATIONS].endswith(MESSAGE_TRUNCATION_SUFFIX)


class TestClassifierCleanup:
    """F-RE-10: both conversations are ended through one helper, which warns."""

    def test_a_failed_classifier_cleanup_is_a_warning(self, monkeypatch):
        engine = ReasoningEngine(model="mock", llm_interface=_ScriptedLLM({}))

        def boom(*_args, **_kwargs):
            raise RuntimeError("cannot end")

        monkeypatch.setattr(engine.classifier, "run_until_terminal", boom)
        monkeypatch.setattr(engine.classifier, "end_conversation", boom)
        records = []
        logger.enable("fsm_llm")
        sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
        try:
            with pytest.raises(ReasoningClassificationError):
                engine._classify_problem({ContextKeys.PROBLEM_STATEMENT: "p"})
        finally:
            logger.remove(sink)
            logger.disable("fsm_llm")
        cleanup = [r for r in records if "clean up conversation" in r["message"]]
        assert [r["level"].name for r in cleanup] == ["WARNING"]


class TestMalformedUserMessage:
    """F-003, superseded by plan 944e2692 step 23: the engine sends no message.

    The engine drives every FSM with message-free core runs
    (``run_until_terminal``), so no synthetic continuation string exists to
    be malformed.
    """

    def test_engine_sends_no_synthetic_message(self):
        """No ``converse(`` call and no "Continue reasoning" text in the engine."""
        import inspect

        from fsm_llm.reasoning import engine

        source = inspect.getsource(engine)
        assert "converse(" not in source
        assert "Continue reasoning" not in source
        assert "Continue:" not in source


class TestExtractFinalSolutionKeys:
    """extract_final_solution must find results from every reasoning type
    after the call state merged them under their orchestrator-level keys."""

    def test_deductive_conclusion_found(self):
        """Deductive merge writes DEDUCTIVE_CONCLUSION, not CONCLUSION."""
        ctx = {ContextKeys.DEDUCTIVE_CONCLUSION: "Therefore X follows."}
        assert OutputFormatter.extract_final_solution(ctx) == "Therefore X follows."

    def test_inductive_hypothesis_found(self):
        ctx = {ContextKeys.INDUCTIVE_HYPOTHESIS: "Pattern suggests Y."}
        assert OutputFormatter.extract_final_solution(ctx) == "Pattern suggests Y."

    def test_critical_assessment_found(self):
        ctx = {ContextKeys.CRITICAL_ASSESSMENT: "The argument is flawed."}
        assert OutputFormatter.extract_final_solution(ctx) == "The argument is flawed."

    def test_best_explanation_found(self):
        ctx = {ContextKeys.BEST_EXPLANATION: "The best explanation is Z."}
        assert (
            OutputFormatter.extract_final_solution(ctx) == "The best explanation is Z."
        )

    def test_analogical_solution_found(self):
        ctx = {ContextKeys.ANALOGICAL_SOLUTION: "By analogy, the answer is W."}
        assert (
            OutputFormatter.extract_final_solution(ctx)
            == "By analogy, the answer is W."
        )

    def test_priority_order_final_solution_first(self):
        """FINAL_SOLUTION takes precedence over type-specific keys."""
        ctx = {
            ContextKeys.FINAL_SOLUTION: "final",
            ContextKeys.DEDUCTIVE_CONCLUSION: "deductive",
            ContextKeys.BEST_CREATIVE_SOLUTION: "creative",
        }
        assert OutputFormatter.extract_final_solution(ctx) == "final"

    def test_all_merge_keys_covered(self):
        """Every key a strategy returns (its ``results`` map) should be in the
        extract_final_solution priority list (excluding confidence/metadata keys)."""
        # Collect the primary result keys from each reasoning type's merge branch
        merge_primary_keys = {
            ContextKeys.INTEGRATED_ANALYSIS,  # analytical
            ContextKeys.DEDUCTIVE_CONCLUSION,  # deductive
            ContextKeys.INDUCTIVE_HYPOTHESIS,  # inductive
            ContextKeys.BEST_CREATIVE_SOLUTION,  # creative
            ContextKeys.CRITICAL_ASSESSMENT,  # critical
            ContextKeys.CALCULATION_RESULT,  # simple_calculator
            ContextKeys.FINAL_HYBRID_SOLUTION,  # hybrid
            ContextKeys.BEST_EXPLANATION,  # abductive
            ContextKeys.ANALOGICAL_SOLUTION,  # analogical
        }

        # Get the actual priority list by calling extract with each key
        for key in merge_primary_keys:
            ctx = {key: "test_value"}
            result = OutputFormatter.extract_final_solution(ctx)
            assert result == "test_value", (
                f"extract_final_solution does not check {key}"
            )


class TestClassificationExceptionHandling:
    """_classify_problem maps a spent classifier budget to
    ReasoningClassificationError once, before the generic wrap."""

    def test_budget_error_is_mapped_before_the_generic_wrap(self):
        """The RunBudgetExceededError branch precedes ``except Exception``."""
        import inspect

        from fsm_llm.reasoning.engine import ReasoningEngine

        source = inspect.getsource(ReasoningEngine._classify_problem)
        budget_pos = source.index("except RunBudgetExceededError as e:")
        generic_pos = source.index("except Exception as e:")
        assert budget_pos < generic_pos


class TestDeadCodeRemoved:
    """Unused utility functions of the FSM store (``strategy_files``) should not exist."""

    def test_no_get_fsm_by_name(self):
        from fsm_llm.reasoning import strategy_files

        assert not hasattr(strategy_files, "get_fsm_by_name")

    def test_no_list_available_fsms(self):
        from fsm_llm.reasoning import strategy_files

        assert not hasattr(strategy_files, "list_available_fsms")

    def test_no_get_reasoning_fsms_only(self):
        from fsm_llm.reasoning import strategy_files

        assert not hasattr(strategy_files, "get_reasoning_fsms_only")
