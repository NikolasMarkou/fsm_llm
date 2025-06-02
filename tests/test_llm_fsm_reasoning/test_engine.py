# """
# Unit tests for the reasoning engine components.
# """
# import pytest
# import json
# from unittest.mock import Mock, patch, mock_open
# from llm_fsm_reasoning import ReasoningEngine
# from llm_fsm_reasoning.constants import ReasoningType, ContextKeys
# from llm_fsm_reasoning.models import ValidationResult, ReasoningStep, ClassificationResult
# from llm_fsm_reasoning.handlers import ReasoningHandlers
# from llm_fsm_reasoning.utilities import map_reasoning_type
#
#
# class TestReasoningModels:
#     """Test Pydantic models."""
#
#     def test_validation_result(self):
#         """Test ValidationResult model."""
#         result = ValidationResult(
#             is_valid=True,
#             confidence=0.95,
#             checks={"has_solution": True, "complete": True},
#             issues=[]
#         )
#
#         assert result.is_valid is True
#         assert result.confidence == 0.95
#         assert len(result.issues) == 0
#
#     def test_reasoning_step(self):
#         """Test ReasoningStep model."""
#         step = ReasoningStep(
#             step_type="analytical",
#             content="Breaking down the problem",
#             confidence=0.8,
#             evidence=["observation1", "observation2"]
#         )
#
#         assert step.step_type == "analytical"
#         assert step.confidence == 0.8
#         assert len(step.evidence) == 2
#
#
# class TestReasoningHandlers:
#     """Test handler implementations."""
#
#     def test_validate_solution(self):
#         """Test solution validation handler."""
#         handlers = ReasoningHandlers()
#
#         # Valid solution
#         context = {
#             ContextKeys.PROPOSED_SOLUTION: "This is a detailed solution with multiple steps addressing the core problem",
#             ContextKeys.KEY_INSIGHTS: ["insight1", "insight2"]
#         }
#
#         result = handlers.validate_solution(context)
#
#         assert result["solution_valid"] is True
#         assert result["solution_confidence"] == 1.0 # Explicitly 1.0 based on logic
#         assert result["validation_checks"]["has_solution"] is True
#         assert result["validation_checks"]["has_insights"] is True
#         assert result["validation_checks"]["sufficient_detail"] is True
#
#     def test_update_reasoning_trace(self):
#         """Test reasoning trace update."""
#         handlers = ReasoningHandlers()
#
#         context = {
#             "_current_state": "analyze",
#             "_previous_state": "start",
#             ContextKeys.REASONING_TRACE: [],
#             "some_data": "value",
#             "reasoning_fsm_to_push": "some_fsm" # Should be excluded from snapshot
#         }
#
#         result = handlers.update_reasoning_trace(context)
#
#         assert len(result[ContextKeys.REASONING_TRACE]) == 1
#         trace_step = result[ContextKeys.REASONING_TRACE][0]
#         assert trace_step["from"] == "start"
#         assert trace_step["to"] == "analyze"
#         assert "some_data" in trace_step["context_snapshot"]
#         assert ContextKeys.REASONING_TRACE not in trace_step["context_snapshot"]
#         assert "_current_state" not in trace_step["context_snapshot"]
#         assert "_previous_state" not in trace_step["context_snapshot"]
#         assert "reasoning_fsm_to_push" not in trace_step["context_snapshot"]
#
#
# class TestReasoningUtils:
#     """Test utility functions."""
#
#     def test_map_reasoning_type(self):
#         """Test reasoning type mapping."""
#         assert map_reasoning_type("ANALYTICAL") == "analytical"
#         assert map_reasoning_type("creative") == "creative"
#         assert map_reasoning_type("unknown") == "analytical"  # default
#
#     @patch('llm_fsm_reasoning.utilities.open', new_callable=mock_open)
#     def test_load_fsm_definition(self, mock_file_open):
#         """Test FSM definition loading."""
#         from llm_fsm_reasoning.utilities import load_fsm_definition
#
#         mock_json_content = {
#             "name": "test_fsm",
#             "initial_state": "start",
#             "description": "A test FSM.",
#             "states": {
#                 "start": {
#                     "id": "start",
#                     "description": "Initial state",
#                     "purpose": "To begin",
#                     "transitions": []
#                 }
#             },
#             "version": "1.0"
#         }
#         mock_file_open.return_value.read.return_value = json.dumps(mock_json_content)
#
#         # The name passed to load_fsm_definition doesn't matter here because `open` is fully mocked.
#         # What matters is that `open` is called and returns the mocked content.
#         result = load_fsm_definition("any_fsm_name_since_open_is_mocked")
#
#         assert result["name"] == "test_fsm"
#         assert result["initial_state"] == "start"
#         # Check that open was called. The path used by load_fsm_definition is dynamic,
#         # so assert_called_with is tricky. assert_called_once is good enough here.
#         mock_file_open.assert_called_once()
#
#
# class TestReasoningEngine:
#     """Test main reasoning engine."""
#
#     @patch('llm_fsm_reasoning.engine.API')
#     def test_engine_initialization(self, mock_api_constructor):
#         """Test engine initialization."""
#         # Ensure that the API constructor is called for reasoner and classification_api
#         mock_api_instance = Mock()
#         mock_api_constructor.from_definition.return_value = mock_api_instance
#
#         engine = ReasoningEngine(model="gpt-4o-mini")
#
#         assert engine.model == "gpt-4o-mini"
#         # ReasoningType enum has 8 members. 2 FSMs (abductive, analogical) are missing.
#         assert len(engine.reasoning_fsms_dicts) == 6
#         assert mock_api_constructor.from_definition.call_count == 2 # For main_fsm and classification_fsm
#
#     @patch('llm_fsm_reasoning.engine.API')
#     def test_orchestrator_classify_problem_handler(self, mock_api_constructor_ignored_here):
#         """Test problem classification handler within the orchestrator."""
#         # Instantiate engine normally, its internal API instances will be real or LiteLLM
#         engine = ReasoningEngine(model="test-model")
#
#         # Mock the classification_api instance directly on the engine object
#         mock_classification_api_instance = Mock()
#         engine.classification_api = mock_classification_api_instance
#
#         mock_conv_id = "test-classifier-conv-123"
#         mock_classification_api_instance.start_conversation.return_value = (mock_conv_id, "Classifier Started")
#         mock_classification_api_instance.has_conversation_ended.side_effect = [False, True] # Simulate one loop
#         mock_classification_api_instance.converse.return_value = "Classifier Continued"
#         mock_classification_api_instance.get_data.return_value = {
#             "recommended_reasoning_type": "analytical",
#             "strategy_justification": "Complex system design",
#             "problem_domain": "technical",
#             "alternative_approaches": ["hybrid_approach"],
#             "confidence_level": "high"
#         }
#         mock_classification_api_instance.end_conversation.return_value = None
#
#         orchestrator_context = {
#             ContextKeys.PROBLEM_STATEMENT: "Design a new complex system.",
#             ContextKeys.PROBLEM_TYPE: "system_design",
#             ContextKeys.PROBLEM_COMPONENTS: ["UI", "Backend", "Database", "API"]
#         }
#
#         # Call the handler method directly
#         result_context_update = engine._orchestrator_classify_problem_handler(orchestrator_context)
#
#         # Assertions on the returned dictionary which updates the orchestrator's context
#         assert result_context_update[ContextKeys.CLASSIFIED_PROBLEM_TYPE] == "analytical"
#         assert result_context_update["classification_reasoning"] == "Complex system design"
#         assert result_context_update["problem_domain_classified"] == "technical"
#         assert result_context_update["alternative_approaches_classified"] == ["hybrid_approach"]
#
#         # Check if classification_api methods were called as expected
#         engine.classification_api.start_conversation.assert_called_once()
#         called_context = engine.classification_api.start_conversation.call_args[1]['initial_context']
#         assert called_context[ContextKeys.PROBLEM_STATEMENT] == "Design a new complex system."
#         assert called_context[ContextKeys.PROBLEM_TYPE] == "system_design"
#         assert called_context[ContextKeys.PROBLEM_COMPONENTS] == ["UI", "Backend", "Database", "API"]
#         assert ContextKeys.REASONING_TRACE in called_context # Handler adds this
#
#         engine.classification_api.converse.assert_called_once_with("Continue analysis", mock_conv_id)
#         engine.classification_api.get_data.assert_called_once_with(mock_conv_id)
#         engine.classification_api.end_conversation.assert_called_once_with(mock_conv_id)
#
#
#     def test_reasoning_type_enum(self):
#         """Test ReasoningType enum."""
#         assert ReasoningType.ANALYTICAL.value == "analytical"
#         assert ReasoningType.CREATIVE.value == "creative"
#
#         # All reasoning types should have corresponding FSMs or be known
#         # This test verifies the enum values themselves
#         expected_enum_values = [
#             "analytical", "deductive", "inductive",
#             "abductive", "analogical", "creative",
#             "critical", "hybrid"
#         ]
#         actual_enum_values = [rt.value for rt in ReasoningType]
#         for val in expected_enum_values:
#             assert val in actual_enum_values
#         assert len(actual_enum_values) == len(expected_enum_values)
#
#
# @pytest.fixture
# def mock_engine_with_mocked_apis():
#     """Create a ReasoningEngine instance where internal API instances are mocks."""
#     with patch('llm_fsm_reasoning.engine.API') as MockApiClass:
#         # Configure the mock API class to return mock instances
#         mock_reasoner_instance = Mock(name="ReasonerMock")
#         mock_classifier_instance = Mock(name="ClassifierMock")
#
#         # Side effect to return different mocks based on FSM definition name or some other logic
#         def from_definition_side_effect(fsm_definition, model, **kwargs):
#             if fsm_definition.get("name") == "reasoning_orchestrator":
#                 return mock_reasoner_instance
#             elif fsm_definition.get("name") == "problem_classifier":
#                 return mock_classifier_instance
#             return Mock() # Default mock if not orchestrator or classifier
#
#         MockApiClass.from_definition.side_effect = from_definition_side_effect
#
#         engine = ReasoningEngine(model="gpt-4o-mini")
#         # Verify that the engine's API instances are the mocks we set up
#         assert engine.reasoner is mock_reasoner_instance
#         assert engine.classification_api is mock_classifier_instance
#         return engine
#
#
#
# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])