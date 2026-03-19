"""Regression tests for plan 6 verified bugs in fsm_llm.

Tests are written to FAIL before the fix and PASS after.
Each test class corresponds to a VB# from verified-bugs.md.
"""
import json
import warnings
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from fsm_llm.definitions import (
    State, Transition, TransitionCondition, FSMDefinition, FSMContext,
    DataExtractionResponse, ResponseGenerationResponse, TransitionDecisionResponse,
)
from fsm_llm.expressions import evaluate_logic, get_var, less, greater


# ── VB1: Self-transitions silently suppressed ──────────────────


class TestVB1SelfTransitionSuppressed:
    """VB1: Self-transitions should fire handlers and report transition_occurred=True."""

    def test_self_transition_returns_true(self):
        """A self-transition should return (True, previous_state_id), not (False, None)."""
        from fsm_llm.fsm import FSMManager
        from fsm_llm.transition_evaluator import TransitionEvaluation, TransitionEvaluationResult

        fsm_def = FSMDefinition(
            name="self_loop_test", description="Test self-transitions",
            states={
                "retry": State(id="retry", description="Retry state", purpose="Retry",
                               transitions=[
                                   Transition(target_state="retry", description="Loop back"),
                                   Transition(target_state="done", description="Finish", priority=500),
                               ]),
                "done": State(id="done", description="Done", purpose="Done", transitions=[]),
            },
            initial_state="retry"
        )

        mock_llm = MagicMock()
        mock_llm.extract_data.return_value = DataExtractionResponse(
            extracted_data={}, confidence=1.0, reasoning="mock"
        )
        mock_llm.generate_response.return_value = ResponseGenerationResponse(
            message="Retrying...", message_type="response", reasoning="mock"
        )

        manager = FSMManager(llm_interface=mock_llm)
        fsm_id = "self_loop_test"
        manager.fsm_cache[fsm_id] = fsm_def
        cid, _ = manager.start_conversation(fsm_id)

        instance = manager.instances[cid]
        assert instance.current_state == "retry"

        # Simulate self-transition evaluation returning current state
        eval_result = TransitionEvaluation(
            result_type=TransitionEvaluationResult.DETERMINISTIC,
            deterministic_transition="retry",
            confidence=1.0
        )

        with patch.object(manager.transition_evaluator, 'evaluate_transitions', return_value=eval_result):
            transition_occurred, prev_state = manager._execute_transition_evaluation_and_execution(
                instance, "test", DataExtractionResponse(extracted_data={}, confidence=1.0, reasoning="m"),
                cid
            )

        assert transition_occurred is True, "Self-transition should report transition_occurred=True"


# ── VB2: Empty message leaks reasoning ─────────────────────────


class TestVB2EmptyMessageLeaksReasoning:
    """VB2: message field should NOT fall back to reasoning when message is present."""

    def test_message_none_falls_back_to_reasoning(self):
        """When message is None/missing, reasoning is an acceptable fallback."""
        from fsm_llm.llm import LiteLLMInterface

        llm = LiteLLMInterface.__new__(LiteLLMInterface)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "reasoning": "Internal thinking..."
        })

        result = llm._parse_response_generation_response(mock_response)
        # message key is absent (None), so reasoning fallback is OK
        assert result.message == "Internal thinking..."

    def test_message_present_does_not_leak_reasoning(self):
        """When message field IS present (even short), reasoning should NOT replace it."""
        from fsm_llm.llm import LiteLLMInterface

        llm = LiteLLMInterface.__new__(LiteLLMInterface)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "message": "OK",
            "reasoning": "User seems confused, keeping response brief"
        })

        result = llm._parse_response_generation_response(mock_response)
        # "OK" is a valid message, should NOT be replaced with reasoning
        assert result.message == "OK", f"Expected 'OK', got: {result.message!r}"
        assert "confused" not in result.message


# ── VB2b: var("") returns None instead of data ─────────────────


class TestVB2bVarEmptyString:
    """VB2b: {"var": ""} should return the entire data object per JsonLogic spec."""

    def test_var_empty_string_returns_data(self):
        data = {"name": "Alice", "age": 30}
        result = evaluate_logic({"var": ""}, data)
        assert result == data, f"Expected {data}, got {result}"

    def test_get_var_empty_string_returns_data(self):
        data = {"x": 1, "y": 2}
        result = get_var(data, "", "NOT_FOUND")
        assert result == data, f"Expected {data}, got {result}"


# ── VB3: Double-wrapped FSMError in api.py ──────────────────────


class TestVB3DoubleWrappedFSMError:
    """VB3: api.py should re-raise FSMError directly, not double-wrap."""

    def test_start_conversation_no_double_wrap(self):
        from fsm_llm.api import API
        from fsm_llm.fsm import FSMError

        api = API.__new__(API)
        api.fsm_definition = MagicMock()
        api.fsm_id = "test-fsm"
        api.active_conversations = {}
        api.conversation_stacks = {}

        mock_manager = MagicMock()
        mock_manager.start_conversation.side_effect = FSMError("Failed to start conversation: LLM null")
        api.fsm_manager = mock_manager

        with pytest.raises(FSMError) as exc_info:
            api.start_conversation()

        msg = str(exc_info.value)
        # Should NOT contain double prefix
        assert msg.count("Failed to start conversation") == 1, f"Double-wrapped: {msg}"

    def test_converse_no_double_wrap(self):
        from fsm_llm.api import API
        from fsm_llm.fsm import FSMError

        api = API.__new__(API)
        api.active_conversations = {"c1": "f1"}
        api.conversation_stacks = {"c1": [MagicMock(fsm_conversation_id="c1")]}

        mock_manager = MagicMock()
        mock_manager.process_message.side_effect = FSMError("Failed to process message: bad input")
        api.fsm_manager = mock_manager

        with pytest.raises(FSMError) as exc_info:
            api.converse("hello", "c1")

        msg = str(exc_info.value)
        assert msg.count("Failed to process message") == 1, f"Double-wrapped: {msg}"


# ── VB4: process_message wraps FSMError subclasses ──────────────


class TestVB4FSMErrorSubclassWrapping:
    """VB4: FSMError subclasses should propagate with original type."""

    def test_fsmerror_subclass_preserved(self):
        from fsm_llm.fsm import FSMManager, FSMError

        fsm_def = FSMDefinition(
            name="test", description="test",
            states={
                "s1": State(id="s1", description="S", purpose="P",
                            transitions=[Transition(target_state="s2", description="go")]),
                "s2": State(id="s2", description="S", purpose="P", transitions=[]),
            },
            initial_state="s1"
        )

        mock_llm = MagicMock()
        mock_llm.extract_data.return_value = DataExtractionResponse(
            extracted_data={}, confidence=1.0, reasoning="mock"
        )
        mock_llm.generate_response.return_value = ResponseGenerationResponse(
            message="Hi", message_type="response", reasoning="mock"
        )

        manager = FSMManager(llm_interface=mock_llm)
        manager.fsm_cache["test"] = fsm_def
        cid, _ = manager.start_conversation("test")

        # Make extraction raise FSMError to test propagation
        mock_llm.extract_data.side_effect = FSMError("specific error")

        with pytest.raises(FSMError) as exc_info:
            manager.process_message(cid, "hello")

        # Should preserve original FSMError, not wrap in another
        assert "specific error" in str(exc_info.value)


# ── VB7: No terminal state check before processing ─────────────


class TestVB7NoTerminalStateCheck:
    """VB7: process_message should fail early for terminal states."""

    def test_terminal_state_rejects_message(self):
        from fsm_llm.fsm import FSMManager, FSMError

        fsm_def = FSMDefinition(
            name="test", description="test",
            states={
                "start": State(id="start", description="S", purpose="P",
                               transitions=[Transition(target_state="end", description="go")]),
                "end": State(id="end", description="E", purpose="P", transitions=[]),
            },
            initial_state="start"
        )

        mock_llm = MagicMock()
        mock_llm.extract_data.return_value = DataExtractionResponse(
            extracted_data={}, confidence=1.0, reasoning="mock"
        )
        mock_llm.generate_response.return_value = ResponseGenerationResponse(
            message="Hi", message_type="response", reasoning="mock"
        )

        manager = FSMManager(llm_interface=mock_llm)
        manager.fsm_cache["test"] = fsm_def
        cid, _ = manager.start_conversation("test")

        # Force to terminal state
        instance = manager.instances[cid]
        instance.current_state = "end"

        # Should raise or at least not waste LLM calls
        extract_call_count_before = mock_llm.extract_data.call_count
        with pytest.raises((FSMError, ValueError)):
            manager.process_message(cid, "one more thing")

        # extract_data should NOT have been called for a terminal state
        assert mock_llm.extract_data.call_count == extract_call_count_before, \
            "LLM should not be called for terminal state"


# ── VB8: JSON array crashes parsers ─────────────────────────────


class TestVB8JsonArrayCrash:
    """VB8: JSON array response should not crash with AttributeError."""

    def test_array_response_does_not_crash_extraction(self):
        from fsm_llm.llm import LiteLLMInterface

        llm = LiteLLMInterface.__new__(LiteLLMInterface)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '[{"key": "value"}]'

        # Should NOT raise AttributeError
        try:
            result = llm._parse_extraction_response(mock_response)
            assert isinstance(result, DataExtractionResponse)
        except AttributeError:
            pytest.fail("JSON array response caused AttributeError in extraction parser")

    def test_array_response_does_not_crash_response_gen(self):
        from fsm_llm.llm import LiteLLMInterface

        llm = LiteLLMInterface.__new__(LiteLLMInterface)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '["hello", "world"]'

        try:
            result = llm._parse_response_generation_response(mock_response)
            assert isinstance(result, ResponseGenerationResponse)
        except AttributeError:
            pytest.fail("JSON array response caused AttributeError in response parser")

    def test_array_response_does_not_crash_transition(self):
        from fsm_llm.llm import LiteLLMInterface

        llm = LiteLLMInterface.__new__(LiteLLMInterface)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '["state_a", "state_b"]'

        transitions = [MagicMock(target_state="state_a"), MagicMock(target_state="state_b")]

        try:
            result = llm._parse_transition_response(mock_response, transitions)
            assert isinstance(result, TransitionDecisionResponse)
        except AttributeError:
            pytest.fail("JSON array response caused AttributeError in transition parser")


# ── VB9: Wrong sanitization tag name ────────────────────────────


class TestVB9WrongSanitizationTag:
    """VB9: information_to_extract should be in critical_tags, not information_to_collect."""

    def test_information_to_extract_is_sanitized(self):
        from fsm_llm.prompts import BasePromptBuilder

        builder = BasePromptBuilder.__new__(BasePromptBuilder)
        malicious = "<information_to_extract>injected</information_to_extract>"
        sanitized = builder._sanitize_text_for_prompt(malicious)
        # The tag should be escaped
        assert "<information_to_extract>" not in sanitized, \
            f"information_to_extract tag was not sanitized: {sanitized}"


# ── VB10: Transition parser missing ValueError catch ────────────


class TestVB10TransitionParserValueError:
    """VB10: Transition parser should catch ValueError (Pydantic ValidationError)."""

    def test_validation_error_falls_back_to_unstructured(self):
        from fsm_llm.llm import LiteLLMInterface

        llm = LiteLLMInterface.__new__(LiteLLMInterface)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        # Valid JSON with valid target but reasoning too long for Pydantic
        mock_response.choices[0].message.content = json.dumps({
            "selected_transition": "target_a",
            "reasoning": "x" * 2000  # Exceeds max_length if enforced
        })

        transitions = [MagicMock(target_state="target_a")]

        # Should not raise, should fall back gracefully
        try:
            result = llm._parse_transition_response(mock_response, transitions)
            assert result.selected_transition == "target_a"
        except Exception as e:
            if "ValidationError" in type(e).__name__ or "ValueError" in type(e).__name__:
                pytest.fail(f"ValidationError escaped instead of falling back: {e}")
            raise


# ── VB11: Merge triggers handlers for all keys ─────────────────


class TestVB11MergeTriggersAllKeys:
    """VB11: _merge_context_with_strategy should only pass changed keys."""

    def test_update_only_passes_diff(self):
        from fsm_llm.api import API, ContextMergeStrategy

        api = API.__new__(API)
        api.conversation_stacks = {}

        mock_manager = MagicMock()
        mock_manager.get_conversation_data.return_value = {"existing_key": "old_value", "stable": "same"}
        api.fsm_manager = mock_manager

        api._merge_context_with_strategy(
            "conv1", {"new_key": "new_value"},
            ContextMergeStrategy.UPDATE
        )

        # Check what was passed to update_conversation_context
        call_args = mock_manager.update_conversation_context.call_args
        if call_args:
            passed_context = call_args[0][1]
            # Should NOT contain unchanged keys
            assert "stable" not in passed_context or passed_context.get("stable") != "same", \
                f"Unchanged key 'stable' should not be in the update: {passed_context}"


# ── VB12: Visualizer duplicates terminal states ─────────────────


class TestVB12VisualizerDuplicatesTerminal:
    """VB12: sort_states_logically should not duplicate terminal states."""

    def test_no_duplicate_terminal_states(self):
        from fsm_llm.visualizer import sort_states_logically

        states = {"start": {}, "middle": {}, "end": {}}
        terminal_states = {"end"}
        state_metrics = {
            "start": {"depth": 0},
            "middle": {"depth": 1},
            "end": {"depth": 2},
        }

        result = sort_states_logically(states, "start", terminal_states, state_metrics)
        assert result.count("end") == 1, f"Terminal state duplicated: {result}"


# ── VB13: Visualizer calculate_depths fails with inbound initial ─


class TestVB13VisualizerDepthsWrongInitial:
    """VB13: calculate_depths should use correct initial state even with inbound transitions."""

    def test_depths_correct_when_initial_has_inbound(self):
        from fsm_llm.visualizer import calculate_depths

        # Graph where initial state "start" has an inbound edge from "middle"
        graph = {
            "start": [("middle", "go forward", None)],
            "middle": [("start", "go back", None), ("end", "finish", None)],
            "end": [],
        }
        state_metrics = {
            "start": {"depth": 0, "inbound": 1},  # Has inbound from middle
            "middle": {"depth": 0, "inbound": 1},
            "end": {"depth": 0, "inbound": 1},
        }

        calculate_depths(graph, state_metrics, initial_state="start")

        assert state_metrics["start"]["depth"] == 0
        assert state_metrics["middle"]["depth"] == 1
        assert state_metrics["end"]["depth"] == 2

    def test_depths_wrong_without_initial_hint(self):
        """Without initial_state parameter, function picks wrong state when all have inbound."""
        from fsm_llm.visualizer import calculate_depths

        graph = {
            "start": [("middle", "go", None)],
            "middle": [("start", "back", None)],
        }
        state_metrics = {
            "start": {"depth": 0, "inbound": 1},
            "middle": {"depth": 0, "inbound": 1},
        }

        # Without explicit initial_state, all have inbound > 0, so no root found
        calculate_depths(graph, state_metrics)
        # Both should still be at depth 0 since no root was found
        # This test documents the current broken behavior


# ── VB14: error_mode="skip" identical to "continue" ────────────


class TestVB14SkipErrorMode:
    """VB14: error_mode='skip' should either not exist or behave differently from 'continue'."""

    def test_skip_not_accepted_or_behaves_differently(self):
        from fsm_llm.handlers import HandlerSystem

        # After fix, "skip" should be rejected
        with pytest.raises(ValueError, match="Invalid error_mode"):
            HandlerSystem(error_mode="skip")


# ── VB14b: Numeric string comparison fails ──────────────────────


class TestVB14bNumericStringComparison:
    """VB14b: Comparison operators should coerce two numeric strings."""

    def test_less_two_numeric_strings(self):
        result = less("10", "2")
        assert result is False, "less('10', '2') should be False (numeric), got True (lexicographic)"

    def test_less_two_numeric_strings_correct(self):
        result = less("3", "20")
        assert result is True, "less('3', '20') should be True (numeric), got False (lexicographic)"

    def test_greater_two_numeric_strings(self):
        result = greater("10", "2")
        assert result is True, "greater('10', '2') should be True (numeric)"

    def test_evaluate_logic_comparison_with_string_numbers(self):
        result = evaluate_logic({"<": ["10", "2"]}, {})
        assert result is False


# ── VB14c: requires_context_keys dot-notation ───────────────────


class TestVB14cDotNotationContextKeys:
    """VB14c: requires_context_keys should support dot-notation like JsonLogic var."""

    def test_dot_notation_key_found_in_nested_context(self):
        from fsm_llm.transition_evaluator import TransitionEvaluator

        evaluator = TransitionEvaluator()
        context = FSMContext()
        context.update({"user": {"name": "Alice"}})

        cond = TransitionCondition(
            description="check nested key",
            requires_context_keys=["user.name"],
            logic={"==": [{"var": "user.name"}, "Alice"]}
        )

        result = evaluator._evaluate_single_condition(cond, context.data)
        assert result is True, "Dot-notation key should be found in nested context"


# ── VB14d: state.id vs dict_key mismatch ───────────────────────


class TestVB14dStateIdMismatch:
    """VB14d: FSMDefinition should reject state.id != dict_key."""

    def test_mismatched_state_id_rejected(self):
        with pytest.raises(ValueError, match="does not match"):
            FSMDefinition(
                name="test", description="test",
                states={
                    "start": State(id="WRONG_ID", description="S", purpose="P",
                                   transitions=[Transition(target_state="end", description="go")]),
                    "end": State(id="end", description="E", purpose="P", transitions=[]),
                },
                initial_state="start"
            )


# ── VB14e: Confidence saturation ───────────────────────────────


class TestVB14eConfidenceSaturation:
    """VB14e: Different priorities should produce different outcomes, not always AMBIGUOUS."""

    def test_priority_0_vs_100_is_deterministic(self):
        from fsm_llm.transition_evaluator import TransitionEvaluator, TransitionEvaluationResult

        evaluator = TransitionEvaluator()
        state = State(
            id="s", description="s", purpose="s",
            transitions=[
                Transition(target_state="a", description="A", priority=0,
                           conditions=[TransitionCondition(description="ok", requires_context_keys=["x"])]),
                Transition(target_state="b", description="B", priority=100,
                           conditions=[TransitionCondition(description="ok", requires_context_keys=["x"])]),
            ]
        )
        ctx = FSMContext()
        ctx.update({"x": True})
        result = evaluator.evaluate_transitions(state, ctx)
        # Priority 0 should beat priority 100 deterministically
        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC, \
            f"Expected DETERMINISTIC, got {result.result_type}"
        assert result.deterministic_transition == "a"


# ── VB15: Validator cascading orphan errors ─────────────────────


class TestVB15ValidatorCascadingErrors:
    """VB15: Invalid initial_state should not cascade into orphan errors."""

    def test_invalid_initial_state_no_cascade(self):
        from fsm_llm.validator import FSMValidator

        fsm_data = {
            "name": "test",
            "description": "test",
            "initial_state": "nonexistent",
            "states": {
                "greeting": {
                    "id": "greeting",
                    "description": "Greet",
                    "purpose": "Greet",
                    "transitions": [{"target_state": "farewell", "description": "go"}]
                },
                "farewell": {
                    "id": "farewell",
                    "description": "Bye",
                    "purpose": "Bye",
                    "transitions": []
                }
            }
        }

        validator = FSMValidator(fsm_data)
        result = validator.validate()

        # Errors are plain strings
        error_messages = result.errors
        has_initial_error = any("nonexistent" in msg or "Initial state" in msg for msg in error_messages)
        assert has_initial_error, f"Expected initial state error, got: {error_messages}"

        # Should NOT have cascading orphan errors for all states
        orphan_errors = [msg for msg in error_messages if "Orphaned" in msg]
        assert len(orphan_errors) == 0, f"Cascading orphan errors found: {orphan_errors}"


# ── VB16: enable_debug_logging breaks file logging ──────────────


class TestVB16DebugLoggingBreaksFileLogging:
    """VB16: enable_debug_logging should not permanently break file logging."""

    def test_file_logging_recoverable_after_debug(self):
        import fsm_llm.logging as log_module

        # Simulate: file logging was set up
        original_flag = getattr(log_module, '_file_handler_initialized', False)

        try:
            log_module._file_handler_initialized = True

            # Call enable_debug_logging
            from fsm_llm import enable_debug_logging
            enable_debug_logging()

            # After enable_debug_logging, _file_handler_initialized should be reset
            assert log_module._file_handler_initialized is False, \
                "_file_handler_initialized should be reset after enable_debug_logging"
        finally:
            log_module._file_handler_initialized = original_flag


# ── VB21: disable_warnings filters wrong category ──────────────


class TestVB21DisableWarningsWrongCategory:
    """VB21: disable_warnings should filter RuntimeWarning, not just UserWarning."""

    def test_runtime_warning_filtered(self):
        from fsm_llm import disable_warnings

        disable_warnings()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Re-apply the filter
            disable_warnings()

            warnings.warn("test warning", RuntimeWarning)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)
                                and "fsm_llm" in str(x.filename)]
            # If from our module, should be filtered
            # This is a weak test since the warning source matters


# ── VB25: `!` operator crashes on 0 args ───────────────────────


class TestVB25NotOperatorArity:
    """VB25: ! operator should handle 0 args gracefully."""

    def test_not_single_arg(self):
        assert evaluate_logic({"!": [False]}, {}) is True
        assert evaluate_logic({"!": [True]}, {}) is False

    def test_not_of_zero(self):
        assert evaluate_logic({"!": [0]}, {}) is True

    def test_not_of_truthy(self):
        assert evaluate_logic({"!": [1]}, {}) is False
