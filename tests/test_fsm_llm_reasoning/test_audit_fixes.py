"""
Tests verifying fixes for audit findings in fsm_llm_reasoning.
Covers: F-003 (malformed user_message), F-005 (context pruning),
        F-011 (magic numbers extracted to constants),
        senior-review fixes (extract keys, double-wrap, dead code, hardcoded keys).
"""

from __future__ import annotations

import json

from fsm_llm_reasoning.constants import ContextKeys, Defaults
from fsm_llm_reasoning.handlers import (
    ContextManager,
    OutputFormatter,
    ReasoningHandlers,
)

# ---------------------------------------------------------------------------
# F-003: Malformed user_message string
# ---------------------------------------------------------------------------


class TestMalformedUserMessage:
    """F-003: The sub-FSM continuation message must not contain ':{' artifact."""

    def test_continue_reasoning_message_is_clean(self):
        """Verify the malformed string was fixed in engine.py."""
        import inspect

        from fsm_llm_reasoning.engine import ReasoningEngine

        source = inspect.getsource(ReasoningEngine)
        # The old malformed string should NOT be present
        assert ":\\n:{" not in source
        assert "Continue reasoning:\\n:{" not in source
        # The fixed string should be present
        assert "Continue reasoning." in source


# ---------------------------------------------------------------------------
# F-005: extract_relevant_context enforces max_size
# ---------------------------------------------------------------------------


class TestContextManagerEnforcement:
    """F-005: extract_relevant_context must enforce max_size, not just warn."""

    def test_enforce_max_size_removes_keys(self):
        """Context exceeding max_size should have keys removed."""
        source = {"key": "x" * 10000}
        result = ContextManager.extract_relevant_context(source, ["key"], max_size=100)
        size = len(json.dumps(result, default=str))
        assert size <= 100

    def test_enforce_max_size_keeps_fitting_keys(self):
        """Keys that fit within budget should be kept."""
        source = {"small": "ok", "big": "x" * 10000}
        result = ContextManager.extract_relevant_context(
            source, ["small", "big"], max_size=200
        )
        # small should fit, big should be removed
        assert "small" in result
        assert "big" not in result

    def test_no_max_size_returns_all(self):
        """Without max_size, all requested keys are returned."""
        source = {"a": "x" * 10000, "b": "y" * 10000}
        result = ContextManager.extract_relevant_context(source, ["a", "b"])
        assert "a" in result
        assert "b" in result

    def test_max_size_with_empty_context(self):
        """Empty context should return empty dict regardless of max_size."""
        result = ContextManager.extract_relevant_context({}, ["key"], max_size=10)
        assert result == {}


# ---------------------------------------------------------------------------
# F-011: Magic numbers extracted to constants
# ---------------------------------------------------------------------------


class TestMagicNumberConstants:
    """F-011: Magic numbers should be in Defaults, not hardcoded."""

    def test_min_solution_length_constant_exists(self):
        assert hasattr(Defaults, "MIN_SOLUTION_LENGTH")
        assert Defaults.MIN_SOLUTION_LENGTH == 20

    def test_prune_list_max_length_constant_exists(self):
        assert hasattr(Defaults, "PRUNE_LIST_MAX_LENGTH")
        assert Defaults.PRUNE_LIST_MAX_LENGTH == 10

    def test_prune_string_max_length_constant_exists(self):
        assert hasattr(Defaults, "PRUNE_STRING_MAX_LENGTH")
        assert Defaults.PRUNE_STRING_MAX_LENGTH == 1000

    def test_prune_context_uses_list_constant(self):
        """Verify prune_context respects PRUNE_LIST_MAX_LENGTH."""
        context = {
            ContextKeys.REASONING_TRACE: list(range(50)),
            ContextKeys.PROBLEM_STATEMENT: "test problem",
        }
        # Trigger pruning by making context large enough
        big_padding = {"_padding": "x" * (Defaults.CONTEXT_PRUNE_THRESHOLD + 1)}
        full_context = {**context, **big_padding}
        result = ReasoningHandlers.prune_context(full_context)
        if ContextKeys.REASONING_TRACE in result:
            assert (
                len(result[ContextKeys.REASONING_TRACE])
                <= Defaults.PRUNE_LIST_MAX_LENGTH
            )

    def test_validate_solution_uses_min_solution_length(self):
        """Verify validate_solution uses MIN_SOLUTION_LENGTH constant."""
        context = {
            ContextKeys.PROPOSED_SOLUTION: "short",  # < MIN_SOLUTION_LENGTH
            ContextKeys.KEY_INSIGHTS: "some insight",
            ContextKeys.RETRY_COUNT: 0,
        }
        result = ReasoningHandlers.validate_solution(context)
        assert result[ContextKeys.VALIDATION_RESULT] is False


# ---------------------------------------------------------------------------
# Senior review: extract_final_solution must cover all reasoning-type keys
# ---------------------------------------------------------------------------


class TestExtractFinalSolutionKeys:
    """extract_final_solution must find results from every reasoning type
    after merge_reasoning_results maps them to orchestrator-level keys."""

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
        """Every key that merge_reasoning_results writes should be in the
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


# ---------------------------------------------------------------------------
# Senior review: double-wrapping of ReasoningClassificationError
# ---------------------------------------------------------------------------


class TestClassificationExceptionHandling:
    """The except block in _classify_problem must not double-wrap
    ReasoningClassificationError."""

    def test_no_double_wrap_in_source(self):
        """Verify the except block re-raises ReasoningClassificationError directly."""
        import inspect

        from fsm_llm_reasoning.engine import ReasoningEngine

        source = inspect.getsource(ReasoningEngine._classify_problem)
        # Must have a bare re-raise for our own exception type
        assert "except ReasoningClassificationError:" in source
        # The re-raise must come before the generic except
        rce_pos = source.index("except ReasoningClassificationError:")
        generic_pos = source.index("except Exception as e:")
        assert rce_pos < generic_pos


# ---------------------------------------------------------------------------
# Senior review: dead code removal
# ---------------------------------------------------------------------------


class TestDeadCodeRemoved:
    """Unused utility functions in reasoning_modes.py should not exist."""

    def test_no_get_fsm_by_name(self):
        from fsm_llm_reasoning import reasoning_modes

        assert not hasattr(reasoning_modes, "get_fsm_by_name")

    def test_no_list_available_fsms(self):
        from fsm_llm_reasoning import reasoning_modes

        assert not hasattr(reasoning_modes, "list_available_fsms")

    def test_no_get_reasoning_fsms_only(self):
        from fsm_llm_reasoning import reasoning_modes

        assert not hasattr(reasoning_modes, "get_reasoning_fsms_only")


# ---------------------------------------------------------------------------
# Senior review: hardcoded keys replaced with ContextKeys constants
# ---------------------------------------------------------------------------


class TestNoHardcodedKeys:
    """_classify_problem return dict must use ContextKeys constants,
    not raw strings."""

    def test_no_problem_domain_classified_string_in_source(self):
        import inspect

        from fsm_llm_reasoning.engine import ReasoningEngine

        source = inspect.getsource(ReasoningEngine._classify_problem)
        assert "problem_domain_classified" not in source

    def test_classify_return_uses_context_keys(self):
        """The return dict keys should match ContextKeys constants."""
        import inspect

        from fsm_llm_reasoning.engine import ReasoningEngine

        source = inspect.getsource(ReasoningEngine._classify_problem)
        assert "ContextKeys.PROBLEM_DOMAIN" in source
        assert "ContextKeys.ALTERNATIVE_APPROACHES" in source


# ---------------------------------------------------------------------------
# Senior review: a non-serializable context value must not break the prompt,
# and (2026-09-22 follow-up) must not reach it as its __str__ either
# ---------------------------------------------------------------------------


class TestJsonDumpsSafety:
    """Every reasoning writer that emits context out of the process (LLM
    prompt, CLI stdout, saved results file) serializes through the shared
    redacting hook, never `default=str`.
    """

    def test_continue_reasoning_uses_the_redacting_hook(self):
        import inspect

        from fsm_llm_reasoning.engine import ReasoningEngine

        source = inspect.getsource(ReasoningEngine._solve_problem_locked)
        assert "default=redacting_json_default" in source
        assert "default=str" not in source

    def test_classification_prompt_uses_the_redacting_hook(self):
        import inspect
        import re

        from fsm_llm_reasoning import engine as engine_mod

        source = inspect.getsource(engine_mod)
        # The two prompt sites are the only json.dumps calls in this module
        # that render context into a user_message.
        assert re.search(r"default=str[,)\s]", source) is None
        assert source.count("default=redacting_json_default") >= 2

    def test_size_measurements_may_still_use_default_str(self):
        """`handlers.py` only measures `len()` of the serialized text and never
        emits it, so its `default=str` is not a leak; the comment must say so.
        """
        import inspect

        from fsm_llm_reasoning import handlers as handlers_mod

        source = inspect.getsource(handlers_mod)
        assert "never emitted" in source

    def test_cli_json_output_redacts_an_objects_str(self):
        from fsm_llm_reasoning.__main__ import _format_json_output

        class Leaky:
            def __str__(self):
                return "password=hunter2"

        text = _format_json_output("done", {"summary": Leaky()})
        assert "hunter2" not in text
        assert "<redacted:Leaky>" in text

    def test_saved_results_file_redacts_an_objects_str(self, tmp_path):
        from fsm_llm_reasoning.__main__ import _save_as_json

        class Leaky:
            def __str__(self):
                return "password=hunter2"

        target = tmp_path / "results.json"
        _save_as_json(target, "problem", "solution", {"leak": Leaky()})
        written = target.read_text()
        assert "hunter2" not in written
        assert "<redacted:Leaky>" in written
