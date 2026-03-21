from __future__ import annotations

"""Unit tests for llm.py — LiteLLMInterface with mocked litellm."""

from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.definitions import (
    DataExtractionRequest,
    LLMResponseError,
    ResponseGenerationRequest,
    TransitionDecisionRequest,
    TransitionOption,
)
from fsm_llm.llm import LiteLLMInterface, LLMInterface


class TestLiteLLMInterfaceInit:
    """Test LiteLLMInterface initialization and validation."""

    def test_requires_non_empty_model(self):
        with pytest.raises(ValueError, match="non-empty"):
            LiteLLMInterface(model="")

    def test_requires_non_empty_model_whitespace(self):
        with pytest.raises(ValueError, match="non-empty"):
            LiteLLMInterface(model="   ")

    def test_rejects_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            LiteLLMInterface(model="test", temperature=3.0)

    def test_rejects_negative_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            LiteLLMInterface(model="test", temperature=-0.1)

    def test_rejects_zero_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens"):
            LiteLLMInterface(model="test", max_tokens=0)

    def test_valid_init(self):
        llm = LiteLLMInterface(model="test-model", temperature=0.7, max_tokens=500)
        assert llm.model == "test-model"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 500

    def test_api_key_stored_in_kwargs(self):
        llm = LiteLLMInterface(model="test-model", api_key="test-key-123")
        assert llm.kwargs.get("api_key") == "test-key-123"

    def test_implements_llm_interface(self):
        assert issubclass(LiteLLMInterface, LLMInterface)


def _mock_llm_response(content: str) -> MagicMock:
    """Create a mock litellm response with given content."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_choice.message.thinking = None
    mock_response.choices = [mock_choice]
    return mock_response


class TestExtractData:
    """Test data extraction via LLM."""

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=["response_format"])
    def test_extract_data_returns_extracted_data(self, mock_params, mock_completion):
        # LiteLLM parser expects {"extracted_data": {...}} wrapper format
        mock_completion.return_value = _mock_llm_response(
            '{"extracted_data": {"name": "Alice", "age": 30}, "confidence": 0.95}'
        )

        llm = LiteLLMInterface(model="test-model")
        request = DataExtractionRequest(
            system_prompt="Extract user data",
            user_message="My name is Alice and I am 30",
            context={}
        )
        response = llm.extract_data(request)

        assert response.extracted_data == {"name": "Alice", "age": 30}
        assert response.confidence == 0.95
        mock_completion.assert_called_once()

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=["response_format"])
    def test_extract_data_handles_empty_json(self, mock_params, mock_completion):
        mock_completion.return_value = _mock_llm_response('{}')

        llm = LiteLLMInterface(model="test-model")
        request = DataExtractionRequest(
            system_prompt="Extract data",
            user_message="hello",
            context={}
        )
        response = llm.extract_data(request)
        assert response.extracted_data == {}

    @patch("fsm_llm.llm.completion", side_effect=Exception("API error"))
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=[])
    def test_extract_data_raises_on_api_error(self, mock_params, mock_completion):
        llm = LiteLLMInterface(model="test-model")
        request = DataExtractionRequest(
            system_prompt="test",
            user_message="test",
            context={}
        )
        with pytest.raises(LLMResponseError, match="Data extraction failed"):
            llm.extract_data(request)


class TestGenerateResponse:
    """Test response generation via LLM."""

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=[])
    def test_generate_response_returns_message(self, mock_params, mock_completion):
        mock_completion.return_value = _mock_llm_response("Hello! How can I help you?")

        llm = LiteLLMInterface(model="test-model")
        request = ResponseGenerationRequest(
            system_prompt="You are a helpful assistant",
            user_message="Hi",
            extracted_data={},
            context={},
            transition_occurred=False,
            previous_state=None
        )
        response = llm.generate_response(request)
        assert response.message == "Hello! How can I help you?"

    @patch("fsm_llm.llm.completion", side_effect=Exception("timeout"))
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=[])
    def test_generate_response_raises_on_error(self, mock_params, mock_completion):
        llm = LiteLLMInterface(model="test-model")
        request = ResponseGenerationRequest(
            system_prompt="test",
            user_message="test",
            extracted_data={},
            context={},
            transition_occurred=False,
            previous_state=None
        )
        with pytest.raises(LLMResponseError, match="Response generation failed"):
            llm.generate_response(request)


class TestDecideTransition:
    """Test transition decision via LLM."""

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=["response_format"])
    def test_decide_transition_returns_selected(self, mock_params, mock_completion):
        mock_completion.return_value = _mock_llm_response('{"selected_transition": "next_state"}')

        llm = LiteLLMInterface(model="test-model")
        options = [
            TransitionOption(target_state="next_state", description="Go next", conditions_met=[]),
            TransitionOption(target_state="other_state", description="Go other", conditions_met=[]),
        ]
        request = TransitionDecisionRequest(
            system_prompt="Choose transition",
            current_state="start",
            available_transitions=options,
            context={},
            user_message="I want to proceed",
            extracted_data={}
        )
        response = llm.decide_transition(request)
        assert response.selected_transition == "next_state"


class TestMakeLLMCall:
    """Test the _make_llm_call method."""

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=[])
    def test_invalid_response_structure_raises(self, mock_params, mock_completion):
        mock_completion.return_value = MagicMock(choices=[])

        llm = LiteLLMInterface(model="test-model")
        with pytest.raises(LLMResponseError, match="Invalid response"):
            llm._make_llm_call([{"role": "user", "content": "hi"}], "test")

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=[])
    def test_none_response_raises(self, mock_params, mock_completion):
        mock_completion.return_value = None

        llm = LiteLLMInterface(model="test-model")
        with pytest.raises(LLMResponseError):
            llm._make_llm_call([{"role": "user", "content": "hi"}], "test")

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=["response_format"])
    def test_json_mode_for_extraction(self, mock_params, mock_completion):
        mock_completion.return_value = _mock_llm_response('{}')

        llm = LiteLLMInterface(model="test-model")
        llm._make_llm_call([{"role": "user", "content": "hi"}], "data_extraction")

        call_kwargs = mock_completion.call_args[1] if mock_completion.call_args[1] else {}
        call_args = mock_completion.call_args
        # Check response_format was set for data_extraction
        all_kwargs = {**dict(zip([], [], strict=False)), **call_kwargs}
        if not all_kwargs:
            all_kwargs = call_args.kwargs
        assert all_kwargs.get("response_format") == {"type": "json_object"}

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=["response_format"])
    def test_no_json_mode_for_response_generation(self, mock_params, mock_completion):
        mock_completion.return_value = _mock_llm_response('Hello!')

        llm = LiteLLMInterface(model="test-model")
        llm._make_llm_call([{"role": "user", "content": "hi"}], "response_generation")

        call_kwargs = mock_completion.call_args.kwargs
        assert "response_format" not in call_kwargs
