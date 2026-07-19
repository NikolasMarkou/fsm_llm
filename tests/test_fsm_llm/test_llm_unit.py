from __future__ import annotations

"""Unit tests for llm.py — LiteLLMInterface with mocked litellm."""

from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.definitions import (
    LLMResponseError,
    ResponseGenerationRequest,
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
            previous_state=None,
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
            previous_state=None,
        )
        with pytest.raises(LLMResponseError, match="Response generation failed"):
            llm.generate_response(request)


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
    def test_json_object_for_non_ollama_extraction(self, mock_params, mock_completion):
        """Non-Ollama models use json_object format for extraction."""
        mock_completion.return_value = _mock_llm_response("{}")

        llm = LiteLLMInterface(model="test-model")
        llm._make_llm_call([{"role": "user", "content": "hi"}], "data_extraction")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs.get("response_format") == {"type": "json_object"}

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=["response_format"])
    def test_no_json_mode_for_response_generation(self, mock_params, mock_completion):
        mock_completion.return_value = _mock_llm_response("Hello!")

        llm = LiteLLMInterface(model="test-model")
        llm._make_llm_call([{"role": "user", "content": "hi"}], "response_generation")

        call_kwargs = mock_completion.call_args.kwargs
        assert "response_format" not in call_kwargs


class TestOllamaLLMCallParams:
    """Test Ollama-specific parameter handling in _make_llm_call."""

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=["response_format"])
    def test_ollama_uses_json_schema_for_extraction(self, mock_params, mock_completion):
        """Ollama models use json_schema format (not json_object) for extraction."""
        mock_completion.return_value = _mock_llm_response(
            '{"extracted_data": {}, "confidence": 0.9}'
        )

        llm = LiteLLMInterface(model="ollama_chat/qwen3.5:4b")
        llm._make_llm_call([{"role": "user", "content": "hi"}], "data_extraction")

        call_kwargs = mock_completion.call_args.kwargs
        rf = call_kwargs.get("response_format")
        assert rf is not None
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "data_extraction"

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=["response_format"])
    def test_ollama_sets_reasoning_effort_none(self, mock_params, mock_completion):
        """Ollama models set reasoning_effort=none to disable thinking."""
        mock_completion.return_value = _mock_llm_response(
            '{"extracted_data": {}, "confidence": 0.9}'
        )

        llm = LiteLLMInterface(model="ollama_chat/qwen3.5:4b")
        llm._make_llm_call([{"role": "user", "content": "hi"}], "data_extraction")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs.get("reasoning_effort") == "none"

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=["response_format"])
    def test_ollama_forces_temperature_zero_for_structured(
        self, mock_params, mock_completion
    ):
        """Ollama structured calls force temperature=0."""
        mock_completion.return_value = _mock_llm_response(
            '{"extracted_data": {}, "confidence": 0.9}'
        )

        llm = LiteLLMInterface(model="ollama_chat/qwen3.5:4b", temperature=0.7)
        llm._make_llm_call([{"role": "user", "content": "hi"}], "data_extraction")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=[])
    def test_ollama_preserves_temperature_for_response_generation(
        self, mock_params, mock_completion
    ):
        """Ollama response_generation preserves user temperature."""
        mock_completion.return_value = _mock_llm_response("Hello!")

        llm = LiteLLMInterface(model="ollama_chat/qwen3.5:4b", temperature=0.7)
        llm._make_llm_call([{"role": "user", "content": "hi"}], "response_generation")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7


class TestRetryWiring:
    """Opt-in `retries` wiring, tested at the REAL provider boundary.

    These tests deliberately do NOT patch `fsm_llm.llm.completion`. Patching it
    would replace the very retry machinery under test, so the tests would pass
    while proving nothing (a trap documented in
    plan-2026-07-18T162030-a02151fe/D-019). Instead they register a real
    `litellm.CustomLLM` provider that raises `RateLimitError` transiently, and
    count actual provider invocations.

    Retries are delegated to litellm, which requires the `tenacity` dependency
    declared in pyproject.toml. Without it litellm's sync retry path raises a
    bare `Exception("tenacity import failed")` and makes ZERO extra attempts, so
    these tests also serve as the regression guard for that dependency.
    """

    @staticmethod
    def _make_request(prompt: str = "You are a helpful assistant"):
        return ResponseGenerationRequest(
            system_prompt=prompt,
            user_message="Hi",
            extracted_data={},
            context={},
            transition_occurred=False,
            previous_state=None,
        )

    @pytest.fixture
    def flaky_provider(self):
        """Register a provider that fails `fail_times` then succeeds.

        Yields a `calls` dict whose "n" key counts provider invocations.
        """
        import litellm
        from litellm import CustomLLM
        from litellm.exceptions import RateLimitError
        from litellm.types.utils import GenericStreamingChunk

        calls = {"n": 0, "fail_times": 2}

        class FlakyLLM(CustomLLM):
            def completion(self, *args, **kwargs):
                calls["n"] += 1
                if calls["n"] <= calls["fail_times"]:
                    raise RateLimitError(
                        message="simulated 429",
                        llm_provider="flakytest",
                        model="flaky-model",
                    )
                return litellm.completion(
                    model="gpt-3.5-turbo",
                    mock_response="RETRY_SUCCESS",
                    messages=[{"role": "user", "content": "hi"}],
                )

            def streaming(self, *args, **kwargs):
                calls["n"] += 1
                if calls["n"] <= calls["fail_times"]:
                    raise RateLimitError(
                        message="simulated 429",
                        llm_provider="flakytest",
                        model="flaky-model",
                    )
                chunk: GenericStreamingChunk = {
                    "finish_reason": "stop",
                    "index": 0,
                    "is_finished": True,
                    "text": "STREAM_RETRY_SUCCESS",
                    "tool_use": None,
                    "usage": {
                        "completion_tokens": 1,
                        "prompt_tokens": 1,
                        "total_tokens": 2,
                    },
                }
                return iter([chunk])

        previous = litellm.custom_provider_map
        litellm.custom_provider_map = [
            {"provider": "flakytest", "custom_handler": FlakyLLM()}
        ]
        try:
            yield calls
        finally:
            litellm.custom_provider_map = previous

    @patch("fsm_llm.llm.get_supported_openai_params", return_value=[])
    def test_retries_two_recovers_and_makes_exactly_three_calls(
        self, mock_params, flaky_provider
    ):
        """retries=2 survives 2 transient failures and returns the SUCCESS payload."""
        llm = LiteLLMInterface(model="flakytest/flaky-model", retries=2)

        response = llm.generate_response(self._make_request())

        # Assert the DESIRED outcome, not merely that nothing raised.
        assert response.message == "RETRY_SUCCESS"
        assert flaky_provider["n"] == 3

    @patch("fsm_llm.llm.get_supported_openai_params", return_value=[])
    def test_retries_zero_makes_exactly_one_call_and_raises(
        self, mock_params, flaky_provider
    ):
        """Adversarial control: SAME provider, SAME failure script, only `retries` differs.

        Pins that the default is byte-for-byte today's behavior. The raised type
        must be the typed `LLMResponseError` — a bare `Exception` here would mean
        the `tenacity` dependency regressed out of pyproject.toml.
        """
        llm = LiteLLMInterface(model="flakytest/flaky-model")
        assert llm.retries == 0, "retries must default to 0 (opt-in)"

        with pytest.raises(LLMResponseError):
            llm.generate_response(self._make_request())

        assert flaky_provider["n"] == 1

    @patch("fsm_llm.llm.get_supported_openai_params", return_value=[])
    def test_stream_retries_two_recovers_and_makes_exactly_three_calls(
        self, mock_params, flaky_provider
    ):
        """Pins the SECOND call-param builder, inside `generate_response_stream`.

        `generate_response_stream` does NOT route through `_make_llm_call`; it
        builds its own call params. A naive one-site fix leaves streaming
        unretried and this test is the only thing that catches it. Proven at the
        provider boundary (strong evidence), not by inspecting params.
        """
        llm = LiteLLMInterface(model="flakytest/flaky-model", retries=2)

        chunks = list(llm.generate_response_stream(self._make_request()))

        assert "".join(chunks) == "STREAM_RETRY_SUCCESS"
        assert flaky_provider["n"] == 3

    @patch("fsm_llm.llm.get_supported_openai_params", return_value=[])
    def test_stream_retries_zero_makes_exactly_one_call_and_raises(
        self, mock_params, flaky_provider
    ):
        """Adversarial control for the stream site: differs from the positive only in `retries`."""
        llm = LiteLLMInterface(model="flakytest/flaky-model")

        with pytest.raises(LLMResponseError):
            list(llm.generate_response_stream(self._make_request()))

        assert flaky_provider["n"] == 1

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=[])
    def test_retries_zero_omits_num_retries_key_entirely(
        self, mock_params, mock_completion
    ):
        """Default must OMIT the key, not send `num_retries=0`.

        Params-level assertion is appropriate here: the claim under test is
        literally about the emitted params, not about retry behavior.
        """
        mock_completion.return_value = _mock_llm_response("Hello!")

        llm = LiteLLMInterface(model="test-model")
        llm._make_llm_call([{"role": "user", "content": "hi"}], "response_generation")

        assert "num_retries" not in mock_completion.call_args.kwargs

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=[])
    def test_non_positive_retries_omits_num_retries_key(
        self, mock_params, mock_completion
    ):
        """A negative `retries` takes the `<= 0` branch — no validation ceremony."""
        mock_completion.return_value = _mock_llm_response("Hello!")

        llm = LiteLLMInterface(model="test-model", retries=-1)
        llm._make_llm_call([{"role": "user", "content": "hi"}], "response_generation")

        assert "num_retries" not in mock_completion.call_args.kwargs
