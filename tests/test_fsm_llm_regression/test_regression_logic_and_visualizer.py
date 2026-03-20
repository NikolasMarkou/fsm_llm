"""Regression tests for plan 7 verified bugs in fsm_llm."""
from unittest.mock import MagicMock

import pytest

from fsm_llm.expressions import evaluate_logic


# ── VB1: AND([]) returns False instead of True ────


class TestAndEmptyArray:
    """VB1: {"and": []} should return True (vacuous truth)."""

    def test_and_empty_returns_true(self):
        """Empty AND should return True (all elements of empty set satisfy condition)."""
        result = evaluate_logic({"and": []})
        assert result is True

    def test_or_empty_returns_false(self):
        """Empty OR should return False (no elements of empty set satisfy condition)."""
        result = evaluate_logic({"or": []})
        assert result is False

    def test_and_single_truthy(self):
        """AND with single truthy value returns that value."""
        assert evaluate_logic({"and": [42]}) == 42

    def test_and_single_falsy(self):
        """AND with single falsy value returns that value."""
        assert evaluate_logic({"and": [0]}) == 0


# ── VB2: Visualizer box lines not truncated ────


class TestVisualizerBoxTruncation:
    """VB2: State boxes should have consistent width even with long state IDs."""

    def test_long_state_id_box_alignment(self):
        """Box lines should all be the same width, even if state_id exceeds box_width."""
        from fsm_llm.visualizer import create_state_boxes

        long_id = "a_very_long_state_name_that_definitely_exceeds_the_box_width_limit"
        states = {
            long_id: {
                "description": "A state",
                "purpose": "Testing",
            }
        }
        state_metrics = {long_id: {"depth": 0}}

        boxes = create_state_boxes(
            ordered_states=[long_id],
            initial_state=long_id,
            terminal_states=set(),
            states=states,
            state_metrics=state_metrics,
        )

        box_lines = boxes[long_id]
        # All lines should have the same visible width
        line_lengths = [len(line) for line in box_lines]
        assert len(set(line_lengths)) == 1, (
            f"Box lines have inconsistent widths: {line_lengths}"
        )

    def test_normal_state_id_box_alignment(self):
        """Box lines for normal-length state IDs should also be consistent."""
        from fsm_llm.visualizer import create_state_boxes

        state_id = "greeting"
        states = {state_id: {"description": "Say hello", "purpose": "Greet user"}}
        state_metrics = {state_id: {"depth": 0}}

        boxes = create_state_boxes(
            ordered_states=[state_id],
            initial_state=state_id,
            terminal_states=set(),
            states=states,
            state_metrics=state_metrics,
        )

        box_lines = boxes[state_id]
        line_lengths = [len(line) for line in box_lines]
        assert len(set(line_lengths)) == 1, (
            f"Box lines have inconsistent widths: {line_lengths}"
        )


# ── VB3: START_CONVERSATION handler failure leaks instance ────


class TestStartConversationHandlerCleanup:
    """VB3: If START_CONVERSATION handler raises with error_mode='raise', instance should be cleaned up."""

    def test_handler_failure_cleans_up_instance(self):
        """When START_CONVERSATION handler fails (error_mode=raise), instance must not leak."""
        from fsm_llm.fsm import FSMManager
        from fsm_llm.handlers import HandlerSystem, HandlerTiming, create_handler
        from fsm_llm.definitions import FSMDefinition

        fsm_def = FSMDefinition(
            name="test",
            description="Test FSM",
            initial_state="start",
            states={
                "start": {
                    "id": "start",
                    "description": "Start",
                    "purpose": "Begin",
                    "transitions": [
                        {"target_state": "end", "description": "Go to end"}
                    ],
                },
                "end": {
                    "id": "end",
                    "description": "End",
                    "purpose": "Finish",
                    "transitions": [],
                },
            },
        )

        handler_system = HandlerSystem(error_mode="raise")
        handler = (
            create_handler("failing_handler")
            .at(HandlerTiming.START_CONVERSATION)
            .do(lambda ctx: (_ for _ in ()).throw(RuntimeError("handler failed")))
        )
        handler_system.register_handler(handler)

        manager = FSMManager(
            fsm_loader=lambda fid: fsm_def,
            llm_interface=MagicMock(),
            handler_system=handler_system,
        )

        # Should raise, and instance should NOT be left in manager.instances
        with pytest.raises(Exception):
            manager.start_conversation("test_fsm")

        assert len(manager.instances) == 0, (
            "Instance leaked after START_CONVERSATION handler failure"
        )


# ── VB4: Early termination skips higher-priority transitions ────


class TestEarlyTerminationRemoved:
    """VB4: early_termination config should be removed from TransitionEvaluatorConfig."""

    def test_early_termination_config_removed(self):
        """TransitionEvaluatorConfig should not have early_termination field."""
        from fsm_llm.transition_evaluator import TransitionEvaluatorConfig

        config = TransitionEvaluatorConfig()
        assert not hasattr(config, "early_termination"), (
            "early_termination should be removed from TransitionEvaluatorConfig"
        )

    def test_all_transitions_evaluated(self):
        """All transitions should be evaluated regardless of confidence."""
        from fsm_llm.transition_evaluator import TransitionEvaluator
        from fsm_llm.definitions import (
            State, Transition, TransitionCondition, FSMContext
        )

        # Create a state with 2 transitions, all with conditions that pass.
        # Priority gap must be wide enough that confidence gap >= ambiguity_threshold.
        state = State(
            id="test_state",
            description="Test",
            purpose="Test all transitions evaluated",
            transitions=[
                Transition(
                    target_state="low_priority",
                    description="Low priority",
                    priority=500,
                    conditions=[
                        TransitionCondition(
                            description="Always true",
                            logic={"==": [1, 1]},
                        )
                    ],
                ),
                Transition(
                    target_state="high_priority",
                    description="High priority",
                    priority=10,
                    conditions=[
                        TransitionCondition(
                            description="Always true",
                            logic={"==": [1, 1]},
                        )
                    ],
                ),
            ],
        )

        evaluator = TransitionEvaluator()
        context = FSMContext()

        result = evaluator.evaluate_transitions(state, context)

        # The high_priority transition (priority=10) should win since both pass
        # and priority 10 < 200, so it should be deterministic
        assert result.deterministic_transition == "high_priority"
