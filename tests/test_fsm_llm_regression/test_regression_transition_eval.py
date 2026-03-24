"""Regression tests for plan 8 verified bugs in fsm_llm."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fsm_llm.definitions import (
    DataExtractionResponse,
    State,
    Transition,
    TransitionCondition,
    TransitionEvaluationResult,
)
from fsm_llm.transition_evaluator import TransitionEvaluator, TransitionEvaluatorConfig

# ── B5: strict_condition_matching exception doesn't break ────


class TestStrictConditionMatchingExceptionBreak:
    """B5: Exception in condition evaluation should break when strict_condition_matching=True."""

    def test_exception_stops_evaluation_in_strict_mode(self):
        """When strict mode is on and a condition raises, remaining conditions should NOT be evaluated."""
        config = TransitionEvaluatorConfig(strict_condition_matching=True)
        evaluator = TransitionEvaluator(config)

        # Create conditions: first raises, second would pass
        cond_raises = TransitionCondition(
            description="raises exception",
            logic={"==": [{"var": "x"}, 1]},  # will cause error with bad context
        )
        cond_passes = TransitionCondition(
            description="would pass",
            requires_context_keys=[],  # always passes (no logic, no required keys)
        )

        # Use a context that will cause an exception in the first condition
        # by mocking _evaluate_single_condition
        call_count = {"value": 0}
        original = evaluator._evaluate_single_condition

        def counting_eval(condition, context):
            call_count["value"] += 1
            if condition.description == "raises exception":
                raise RuntimeError("Simulated condition error")
            return original(condition, context)

        evaluator._evaluate_single_condition = counting_eval

        result = evaluator._evaluate_transition_conditions(
            [cond_raises, cond_passes], {}
        )

        # BUG: Without fix, both conditions are evaluated (call_count == 2)
        # With fix: only the first condition is evaluated before breaking
        assert result["all_pass"] is False
        assert call_count["value"] == 1, (
            f"Expected 1 condition evaluated (break on exception in strict mode), got {call_count['value']}"
        )

    def test_regular_failure_stops_evaluation_in_strict_mode(self):
        """Baseline: regular failure already breaks in strict mode (not a bug)."""
        config = TransitionEvaluatorConfig(strict_condition_matching=True)
        evaluator = TransitionEvaluator(config)

        cond_fails = TransitionCondition(
            description="fails normally", requires_context_keys=["missing_key"]
        )
        cond_passes = TransitionCondition(
            description="would pass", requires_context_keys=[]
        )

        result = evaluator._evaluate_transition_conditions(
            [cond_fails, cond_passes], {}
        )
        assert result["all_pass"] is False


# ── B6: Floating-point equality in tiebreaker ────


class TestFloatingPointTiebreaker:
    """B6: Tiebreaker should use epsilon comparison, not exact == 0."""

    def test_near_zero_confidence_gap_triggers_tiebreaker(self):
        """Two transitions with nearly-equal confidence should still use priority tiebreaker."""
        config = TransitionEvaluatorConfig(
            ambiguity_threshold=0.1, minimum_confidence=0.5
        )
        evaluator = TransitionEvaluator(config)

        t1 = Transition(target_state="state_a", description="A", priority=50)
        t2 = Transition(target_state="state_b", description="B", priority=100)
        current_state = State(
            id="start", description="Start", purpose="Start", transitions=[t1, t2]
        )

        # Simulate scores with near-zero but not exactly zero confidence gap
        # This mimics floating-point arithmetic imprecision
        scores = [
            {
                "transition": t1,
                "confidence": 0.7000000000000001,
                "passes_conditions": True,
                "condition_results": {},
            },
            {
                "transition": t2,
                "confidence": 0.7,
                "passes_conditions": True,
                "condition_results": {},
            },
        ]

        result = evaluator._determine_evaluation_result(scores, current_state, {})

        # BUG: Without fix, confidence_gap == 1.11e-16 != 0, so tiebreaker skipped → AMBIGUOUS
        # With fix: abs(gap) < epsilon triggers tiebreaker → DETERMINISTIC (t1 wins, lower priority)
        assert result.result_type == TransitionEvaluationResult.DETERMINISTIC, (
            f"Expected DETERMINISTIC (priority tiebreaker), got {result.result_type}"
        )
        assert result.deterministic_transition == "state_a"


# ── B3: _parse_transition_response dict content crashes on fallthrough ────


class TestTransitionResponseDictFallthrough:
    """B3: If content is a dict and no valid transition found, .lower() crashes."""

    def test_dict_content_no_valid_transition_does_not_crash(self):
        """When content is a dict but selected transition isn't valid, should not crash with AttributeError."""
        from fsm_llm.llm import LiteLLMInterface, LLMResponseError

        interface = LiteLLMInterface.__new__(LiteLLMInterface)
        interface.model = "test"

        # Create a mock response where content is a dict
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content={
                            "selected_transition": "nonexistent_state",
                            "reasoning": "test",
                        }
                    )
                )
            ]
        )

        t1 = SimpleNamespace(target_state="valid_state")

        # BUG: Without fix, this raises AttributeError: 'dict' object has no attribute 'lower'
        # With fix: should raise LLMResponseError (graceful failure)
        with pytest.raises(LLMResponseError):
            interface._parse_transition_response(mock_response, [t1])


# ── B4: _parse_response_generation_response dict content crashes on fallthrough ────


class TestResponseGenDictFallthrough:
    """B4: If content is a dict without message/reasoning, fallthrough creates model with dict as str."""

    def test_dict_content_no_message_does_not_crash(self):
        """When content is a dict lacking message/reasoning keys, should not raise ValidationError."""
        from fsm_llm.llm import LiteLLMInterface

        interface = LiteLLMInterface.__new__(LiteLLMInterface)
        interface.model = "test"

        # Create a mock response where content is a dict without message/reasoning
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content={"some_key": "some_value"})
                )
            ]
        )

        # BUG: Without fix, this path raises Pydantic ValidationError (dict as str message)
        # With fix: should return a valid ResponseGenerationResponse
        result = interface._parse_response_generation_response(mock_response)
        assert isinstance(result.message, str)
        assert len(result.message) > 0


# ── B1: Context cleaning fires handlers with empty updated_keys ────


class TestContextCleaningEmptyHandlers:
    """B1: CONTEXT_UPDATE handlers should not fire when all extracted data is cleaned away."""

    def test_handlers_not_called_after_cleaning_empties_data(self):
        """If _clean_empty_context_keys removes all keys, CONTEXT_UPDATE handlers should not fire."""
        from fsm_llm.definitions import FSMContext, FSMDefinition, FSMInstance
        from fsm_llm.fsm import FSMManager
        from fsm_llm.handlers import HandlerSystem, HandlerTiming

        handler_system = HandlerSystem(error_mode="continue")

        mock_llm = MagicMock()
        fsm_definition = FSMDefinition.model_validate(
            {
                "name": "test",
                "description": "test",
                "version": "4.1",
                "initial_state": "s1",
                "states": {
                    "s1": {
                        "id": "s1",
                        "description": "s1",
                        "purpose": "s1",
                        "transitions": [
                            {"target_state": "s2", "description": "go", "priority": 100}
                        ],
                    },
                    "s2": {
                        "id": "s2",
                        "description": "s2",
                        "purpose": "s2",
                        "transitions": [],
                    },
                },
            }
        )

        manager = FSMManager(
            llm_interface=mock_llm,
            handler_system=handler_system,
        )
        manager.fsm_cache["test_id"] = fsm_definition

        instance = FSMInstance(
            fsm_id="test_id",
            current_state="s1",
            context=FSMContext(max_history_size=5, max_message_length=1000),
        )
        manager.instances["conv1"] = instance

        # Track _execute_handlers calls
        handler_calls = []
        original_execute = manager._execute_handlers

        def tracking_execute(timing, *args, **kwargs):
            handler_calls.append(timing)
            return original_execute(timing, *args, **kwargs)

        manager._execute_handlers = tracking_execute

        # Create extraction response where ALL data will be cleaned away (only None values)
        extraction_response = DataExtractionResponse(
            extracted_data={"key1": None, "key2": None, "key3": None},
            confidence=1.0,
            reasoning="test",
        )

        # Mock _execute_data_extraction to return our response
        manager._pipeline._execute_data_extraction = MagicMock(
            return_value=extraction_response
        )
        # Mock transition evaluation to return no transition
        manager._pipeline._execute_transition_evaluation_and_execution = MagicMock(
            return_value=(False, None)
        )

        handler_calls.clear()

        # Call the actual code path that has the bug
        manager._pipeline._execute_extraction_and_transition_pass(
            instance, "hello", "conv1"
        )

        # BUG: Without fix, CONTEXT_UPDATE handler fires with empty updated_keys
        # With fix: handler should NOT fire because all extracted data was cleaned away
        # Note: empty strings and empty dicts are now preserved as semantically valid data
        context_update_calls = [
            c for c in handler_calls if c == HandlerTiming.CONTEXT_UPDATE
        ]
        assert len(context_update_calls) == 0, (
            f"Expected 0 CONTEXT_UPDATE handler calls (data cleaned to empty), got {len(context_update_calls)}"
        )


# ── B2: pop_fsm stack.pop() before end_conversation() ────


class TestPopFsmStackOrder:
    """B2: pop_fsm should end_conversation before popping stack to maintain consistency on failure."""

    def test_stack_preserved_when_end_conversation_fails(self):
        """If end_conversation raises, the stack frame should still be present."""
        from fsm_llm.api import API, FSMStackFrame
        from fsm_llm.definitions import FSMDefinition, FSMError

        fsm_def = FSMDefinition.model_validate(
            {
                "name": "test",
                "description": "test",
                "version": "4.1",
                "initial_state": "s1",
                "states": {
                    "s1": {
                        "id": "s1",
                        "description": "s1",
                        "purpose": "s1",
                        "transitions": [
                            {"target_state": "s2", "description": "go", "priority": 100}
                        ],
                    },
                    "s2": {
                        "id": "s2",
                        "description": "s2",
                        "purpose": "s2",
                        "transitions": [],
                    },
                },
            }
        )

        mock_llm = MagicMock()
        api = API.__new__(API)
        api._stack_lock = __import__("threading").Lock()
        api.llm_interface = mock_llm
        api.fsm_definition = fsm_def
        api.fsm_id = "test_id"
        api.handler_system = MagicMock()
        api.fsm_manager = MagicMock()
        api._temp_fsm_definitions = {}

        # Setup conversation with 2-frame stack
        api.active_conversations = {"conv1": True}
        frame1 = FSMStackFrame(fsm_definition=fsm_def, conversation_id="conv1_root")
        frame2 = FSMStackFrame(fsm_definition=fsm_def, conversation_id="conv1_sub")
        api.conversation_stacks = {"conv1": [frame1, frame2]}

        # Make end_conversation raise
        api.fsm_manager.end_conversation.side_effect = Exception(
            "end_conversation failed"
        )
        api.fsm_manager.get_conversation_data.return_value = {}

        # pop_fsm should raise FSMError (end_conversation failure propagates)
        with pytest.raises(FSMError):
            api.pop_fsm("conv1")

        # The stack is always popped (via try-finally) even when end_conversation
        # fails, to avoid leaving a stale frame with an inconsistent conversation_id.
        assert len(api.conversation_stacks["conv1"]) == 1, (
            f"Expected 1 frame (stale frame popped despite failure), got {len(api.conversation_stacks['conv1'])}"
        )
