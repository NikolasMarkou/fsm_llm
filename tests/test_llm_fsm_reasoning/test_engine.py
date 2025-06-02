"""
Unit tests for the reasoning engine components.
"""
import pytest
from unittest.mock import Mock, patch
from llm_fsm_reasoning import ReasoningEngine
from llm_fsm_reasoning.constants import ReasoningType, ContextKeys
from llm_fsm_reasoning.models import ValidationResult, ReasoningStep
from llm_fsm_reasoning.handlers import ReasoningHandlers
from llm_fsm_reasoning.utilities import map_reasoning_type


class TestReasoningModels:
    """Test Pydantic models."""

    def test_validation_result(self):
        """Test ValidationResult model."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.95,
            checks={"has_solution": True, "complete": True},
            issues=[]
        )

        assert result.is_valid is True
        assert result.confidence == 0.95
        assert len(result.issues) == 0

    def test_reasoning_step(self):
        """Test ReasoningStep model."""
        step = ReasoningStep(
            step_type="analytical",
            content="Breaking down the problem",
            confidence=0.8,
            evidence=["observation1", "observation2"]
        )

        assert step.step_type == "analytical"
        assert step.confidence == 0.8
        assert len(step.evidence) == 2


class TestReasoningHandlers:
    """Test handler implementations."""

    def test_validate_solution(self):
        """Test solution validation handler."""
        handlers = ReasoningHandlers()

        # Valid solution
        context = {
            ContextKeys.PROPOSED_SOLUTION: "This is a detailed solution with multiple steps addressing the core problem",
            ContextKeys.KEY_INSIGHTS: ["insight1", "insight2"]
        }

        result = handlers.validate_solution(context)

        print(result)

        assert result["solution_valid"] is True
        assert result["solution_confidence"] > 0.5

    def test_update_reasoning_trace(self):
        """Test reasoning trace update."""
        handlers = ReasoningHandlers()

        context = {
            "_current_state": "analyze",
            "_previous_state": "start",
            ContextKeys.REASONING_TRACE: []
        }

        result = handlers.update_reasoning_trace(context)

        assert len(result[ContextKeys.REASONING_TRACE]) == 1
        assert result[ContextKeys.REASONING_TRACE][0]["from"] == "start"
        assert result[ContextKeys.REASONING_TRACE][0]["to"] == "analyze"


class TestReasoningUtils:
    """Test utility functions."""

    def test_map_reasoning_type(self):
        """Test reasoning type mapping."""
        assert map_reasoning_type("ANALYTICAL") == "analytical"
        assert map_reasoning_type("creative") == "creative"
        assert map_reasoning_type("unknown") == "analytical"  # default

    @patch('builtins.open', create=True)
    def test_load_fsm_definition(self, mock_open):
        """Test FSM definition loading."""
        from llm_fsm_reasoning.utilities import load_fsm_definition

        # Mock file content
        mock_open.return_value.__enter__.return_value.read.return_value = '''
        {
            "name": "test_fsm",
            "initial_state": "start"
        }
        '''

        # This would normally load from file
        # result = load_fsm_definition("test")
        # assert result["name"] == "test_fsm"


class TestReasoningEngine:
    """Test main reasoning engine."""

    @patch('llm_fsm_reasoning.engine.API')
    def test_engine_initialization(self, mock_api):
        """Test engine initialization."""
        engine = ReasoningEngine(model="gpt-4o-mini")

        assert engine.model == "gpt-4o-mini"
        assert len(engine.reasoning_fsms) >= 6  # At least 6 reasoning types

    @patch('llm_fsm_reasoning.engine.API')
    def test_classify_problem(self, mock_api):
        """Test problem classification."""
        engine = ReasoningEngine()

        # Mock classification API
        mock_conv_id = "test-conv-123"
        engine.classification_api.start_conversation = Mock(return_value=(mock_conv_id, "Started"))
        engine.classification_api.has_conversation_ended = Mock(side_effect=[False, True])
        engine.classification_api.get_data = Mock(return_value={
            "recommended_reasoning_type": "analytical",
            "strategy_justification": "Complex system design",
            "problem_domain": "technical"
        })

        context = {
            ContextKeys.PROBLEM_TYPE: "system_design"
        }

        result = engine._classify_problem(context)

        assert result[ContextKeys.CLASSIFIED_PROBLEM_TYPE] == "analytical"
        assert "classification_reasoning" in result

    def test_reasoning_type_enum(self):
        """Test ReasoningType enum."""
        assert ReasoningType.ANALYTICAL.value == "analytical"
        assert ReasoningType.CREATIVE.value == "creative"

        # All reasoning types should have corresponding FSMs
        for reasoning_type in ReasoningType:
            assert reasoning_type.value in [
                "analytical", "deductive", "inductive",
                "creative", "critical", "hybrid",
                "abductive", "analogical"  # These might not have FSMs yet
            ]


@pytest.fixture
def mock_engine():
    """Create a mock reasoning engine for testing."""
    with patch('llm_fsm_reasoning.engine.API'):
        engine = ReasoningEngine(model="gpt-4o-mini")
        return engine


def test_solve_problem_integration(mock_engine):
    """Integration test for problem solving."""
    # This would be a more complex integration test
    # that actually runs through the full flow
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])