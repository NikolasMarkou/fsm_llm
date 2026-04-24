from __future__ import annotations

"""Tests for fsm_llm.lam.oracle — protocol, guards, dispatch."""

from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from fsm_llm.definitions import (
    FieldExtractionResponse,
    LLMResponseError,
    ResponseGenerationResponse,
)
from fsm_llm.lam.errors import OracleError
from fsm_llm.lam.oracle import LiteLLMOracle, Oracle, _resolve_schema
from fsm_llm.llm import LLMInterface


class _SampleSchema(BaseModel):
    answer: str
    score: float


def _make_llm_mock() -> Mock:
    m = Mock(spec=LLMInterface)
    # Attach a .model attribute so LiteLLMOracle can read it.
    m.model = "ollama_chat/qwen3.5:4b"
    return m


class TestProtocolConformance:
    def test_litellm_oracle_satisfies_protocol(self) -> None:
        llm = _make_llm_mock()
        oracle = LiteLLMOracle(llm, context_window_tokens=4096)
        assert isinstance(oracle, Oracle)
        assert oracle.context_window() == 4096


class TestTokenize:
    def test_tokenize_fallback(self) -> None:
        oracle = LiteLLMOracle(_make_llm_mock(), context_window_tokens=1000)
        # Fallback is len//4 (CHARS_PER_TOKEN_FALLBACK=4).
        # litellm.token_counter may also be available; either way, must
        # return a positive int.
        n = oracle.tokenize("abcdefgh")
        assert isinstance(n, int) and n >= 1

    def test_tokenize_empty(self) -> None:
        oracle = LiteLLMOracle(_make_llm_mock(), context_window_tokens=1000)
        # Should not crash; minimum 1 from the fallback floor.
        n = oracle.tokenize("")
        assert isinstance(n, int) and n >= 0


class TestContextWindowGuard:
    def test_refuses_oversize_prompt(self) -> None:
        oracle = LiteLLMOracle(_make_llm_mock(), context_window_tokens=10)
        huge = "x" * 10_000  # far exceeds 10 tokens under any reasonable tokenizer
        with pytest.raises(OracleError, match=r"exceeds K="):
            oracle.invoke(huge)

    def test_accepts_fitting_prompt(self) -> None:
        llm = _make_llm_mock()
        llm.generate_response.return_value = ResponseGenerationResponse(
            message="pong"
        )
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)
        assert oracle.invoke("hi") == "pong"


class TestUnstructuredDispatch:
    def test_generate_response_called(self) -> None:
        llm = _make_llm_mock()
        llm.generate_response.return_value = ResponseGenerationResponse(
            message="hello world"
        )
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)

        out = oracle.invoke("Say hi.")

        assert out == "hello world"
        llm.generate_response.assert_called_once()
        assert llm.extract_field.call_count == 0

    def test_llm_failure_wrapped(self) -> None:
        llm = _make_llm_mock()
        llm.generate_response.side_effect = LLMResponseError("boom")
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)
        with pytest.raises(OracleError, match="LLM call failed"):
            oracle.invoke("test")


class TestStructuredDispatch:
    def test_extract_field_called_and_validated(self) -> None:
        llm = _make_llm_mock()
        llm.extract_field.return_value = FieldExtractionResponse(
            field_name="result",
            value={"answer": "yes", "score": 0.9},
        )
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)

        out = oracle.invoke("Q?", schema=_SampleSchema)

        assert out == {"answer": "yes", "score": 0.9}
        llm.extract_field.assert_called_once()
        assert llm.generate_response.call_count == 0

    def test_structured_non_dict_rejected(self) -> None:
        llm = _make_llm_mock()
        llm.extract_field.return_value = FieldExtractionResponse(
            field_name="result", value="not a dict"
        )
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)
        with pytest.raises(OracleError, match="non-dict"):
            oracle.invoke("Q?", schema=_SampleSchema)

    def test_structured_schema_validation_error_wrapped(self) -> None:
        llm = _make_llm_mock()
        # Missing required 'answer' field.
        llm.extract_field.return_value = FieldExtractionResponse(
            field_name="result", value={"score": 0.5}
        )
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)
        with pytest.raises(OracleError, match="schema .* validation"):
            oracle.invoke("Q?", schema=_SampleSchema)


class TestModelOverrideGuard:
    def test_same_model_override_accepted(self) -> None:
        llm = _make_llm_mock()
        llm.generate_response.return_value = ResponseGenerationResponse(message="ok")
        oracle = LiteLLMOracle(
            llm, context_window_tokens=10_000, model_name="test-model"
        )
        # Same as bound model → no error.
        assert oracle.invoke("hi", model_override="test-model") == "ok"

    def test_different_model_override_raises(self) -> None:
        llm = _make_llm_mock()
        oracle = LiteLLMOracle(
            llm, context_window_tokens=10_000, model_name="test-model"
        )
        with pytest.raises(OracleError, match="model_override"):
            oracle.invoke("hi", model_override="other-model")


class TestResolveSchema:
    def test_resolves_real_schema(self) -> None:
        cls = _resolve_schema(
            "tests.test_fsm_llm_lam.test_oracle._SampleSchema"
        )
        assert cls is _SampleSchema

    def test_bad_path_raises(self) -> None:
        with pytest.raises(OracleError):
            _resolve_schema("no_such_module.Cls")

    def test_non_basemodel_raises(self) -> None:
        with pytest.raises(OracleError):
            _resolve_schema("os.path")

    def test_malformed_path_raises(self) -> None:
        with pytest.raises(OracleError):
            _resolve_schema("justaword")
