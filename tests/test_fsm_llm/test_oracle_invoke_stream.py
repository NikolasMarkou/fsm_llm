from __future__ import annotations

"""Tests for ``LiteLLMOracle.invoke_stream`` (R10 step 6 prep).

Verifies the streaming oracle surface added as forward-compat plumbing
for the R10 pipeline.py:1185 wiring (FSM_LLM_ORACLE_RESPONSE_STREAM).

Coverage:
- Protocol conformance: ``Oracle`` runtime_checkable still admits
  ``LiteLLMOracle`` after ``invoke_stream`` is added.
- Happy path: chunks from ``LLMInterface.generate_response_stream``
  are yielded back through the oracle.
- Env template substitution mirrors ``invoke``'s contract (D-005).
- Missing env var → ``OracleError``.
- |P| > K guard fires before any underlying stream is consumed.
- ``model_override`` mismatch raises ``OracleError``.
- ``LLMResponseError`` from underlying interface re-raises as
  ``OracleError`` (single-typed boundary).
- ``user_message`` is forwarded into the underlying request body.
- Schema is materialized as ``response_format`` on the request.
"""

from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from fsm_llm._models import (
    LLMResponseError,
    ResponseGenerationRequest,
)
from fsm_llm.runtime._litellm import LLMInterface
from fsm_llm.runtime.errors import OracleError
from fsm_llm.runtime.oracle import LiteLLMOracle, Oracle, StreamingOracle


class _AnswerSchema(BaseModel):
    answer: str


def _make_llm_mock() -> Mock:
    m = Mock(spec=LLMInterface)
    m.model = "ollama_chat/qwen3.5:4b"
    m.kwargs = {}
    m.max_tokens = 1000
    m.timeout = None
    return m


def _stream(chunks: list[str]):
    def _gen(req: ResponseGenerationRequest) -> Iterator[str]:
        yield from chunks

    return _gen


class TestProtocolConformance:
    def test_oracle_protocol_still_satisfied(self) -> None:
        oracle = LiteLLMOracle(_make_llm_mock(), context_window_tokens=1000)
        assert isinstance(oracle, Oracle)
        # method exists and is callable
        assert callable(oracle.invoke_stream)

    def test_litellm_oracle_satisfies_streaming_oracle(self) -> None:
        """LiteLLMOracle is the canonical StreamingOracle."""
        oracle = LiteLLMOracle(_make_llm_mock(), context_window_tokens=1000)
        assert isinstance(oracle, StreamingOracle)

    def test_invoke_only_oracle_is_not_streaming(self) -> None:
        """Mock oracles that implement only ``invoke`` MUST still be Oracle but
        NOT StreamingOracle — guards against accidentally re-promoting
        ``invoke_stream`` to the base Protocol (which broke ~21 tests in the
        first attempt at step 6 — see decisions D-STEP-6-T1)."""

        class _InvokeOnly:
            def invoke(self, prompt, schema=None, *, model_override=None, env=None):
                return "x"

            def tokenize(self, text: str) -> int:
                return 1

            def context_window(self) -> int:
                return 1000

        m = _InvokeOnly()
        assert isinstance(m, Oracle)
        assert not isinstance(m, StreamingOracle)


class TestHappyPath:
    def test_yields_chunks_from_underlying_stream(self) -> None:
        llm = _make_llm_mock()
        llm.generate_response_stream.side_effect = _stream(["Hel", "lo, ", "world"])
        oracle = LiteLLMOracle(llm, context_window_tokens=1000)

        out = list(oracle.invoke_stream("greet the user"))

        assert out == ["Hel", "lo, ", "world"]
        # underlying interface was invoked exactly once
        assert llm.generate_response_stream.call_count == 1
        req = llm.generate_response_stream.call_args.args[0]
        assert isinstance(req, ResponseGenerationRequest)
        assert req.system_prompt == "greet the user"
        assert req.user_message == ""

    def test_user_message_threads_through(self) -> None:
        llm = _make_llm_mock()
        llm.generate_response_stream.side_effect = _stream(["ok"])
        oracle = LiteLLMOracle(llm, context_window_tokens=1000)

        list(oracle.invoke_stream("system here", user_message="hi there"))

        req = llm.generate_response_stream.call_args.args[0]
        assert req.user_message == "hi there"

    def test_empty_stream_yields_nothing(self) -> None:
        llm = _make_llm_mock()
        llm.generate_response_stream.side_effect = _stream([])
        oracle = LiteLLMOracle(llm, context_window_tokens=1000)
        assert list(oracle.invoke_stream("x")) == []


class TestEnvTemplate:
    def test_env_substitution_mirrors_invoke(self) -> None:
        llm = _make_llm_mock()
        llm.generate_response_stream.side_effect = _stream(["done"])
        oracle = LiteLLMOracle(llm, context_window_tokens=1000)

        list(oracle.invoke_stream("say hi to {name}", env={"name": "Ada"}))

        req = llm.generate_response_stream.call_args.args[0]
        assert req.system_prompt == "say hi to Ada"

    def test_missing_env_var_raises_oracle_error(self) -> None:
        oracle = LiteLLMOracle(_make_llm_mock(), context_window_tokens=1000)
        with pytest.raises(OracleError, match="missing env var"):
            list(oracle.invoke_stream("hi {name}", env={}))

    def test_env_none_passes_prompt_verbatim(self) -> None:
        llm = _make_llm_mock()
        llm.generate_response_stream.side_effect = _stream(["a"])
        oracle = LiteLLMOracle(llm, context_window_tokens=1000)
        list(oracle.invoke_stream("literal {brace} stays", env=None))
        req = llm.generate_response_stream.call_args.args[0]
        assert req.system_prompt == "literal {brace} stays"


class TestGuards:
    def test_oversize_prompt_refused(self) -> None:
        llm = _make_llm_mock()
        oracle = LiteLLMOracle(llm, context_window_tokens=10)
        huge = "x" * 10_000
        with pytest.raises(OracleError, match=r"exceeds K="):
            # consume to force generator body to execute
            list(oracle.invoke_stream(huge))
        # underlying stream must NOT have been touched
        assert llm.generate_response_stream.call_count == 0

    def test_model_override_mismatch_raises(self) -> None:
        llm = _make_llm_mock()
        oracle = LiteLLMOracle(llm, context_window_tokens=1000)
        with pytest.raises(OracleError, match="model_override"):
            list(oracle.invoke_stream("hi", model_override="gpt-4"))

    def test_model_override_match_is_ok(self) -> None:
        llm = _make_llm_mock()
        llm.generate_response_stream.side_effect = _stream(["yo"])
        oracle = LiteLLMOracle(llm, context_window_tokens=1000)
        out = list(oracle.invoke_stream("hi", model_override="ollama_chat/qwen3.5:4b"))
        assert out == ["yo"]


class TestErrorTranslation:
    def test_llm_response_error_becomes_oracle_error(self) -> None:
        llm = _make_llm_mock()

        def _raises(req):
            raise LLMResponseError("network down")
            yield  # pragma: no cover — make this a generator

        llm.generate_response_stream.side_effect = _raises
        oracle = LiteLLMOracle(llm, context_window_tokens=1000)
        with pytest.raises(OracleError, match="streaming call failed"):
            list(oracle.invoke_stream("hi"))


class TestSchemaPassthrough:
    def test_schema_materialises_response_format(self) -> None:
        llm = _make_llm_mock()
        llm.generate_response_stream.side_effect = _stream(["{}"])
        oracle = LiteLLMOracle(llm, context_window_tokens=1000)

        list(oracle.invoke_stream("answer", schema=_AnswerSchema))

        req = llm.generate_response_stream.call_args.args[0]
        assert req.response_format is not None
        assert req.response_format["type"] == "json_schema"
        assert req.response_format["json_schema"]["name"] == "_AnswerSchema"
        # required field synthesised when Pydantic omits it
        assert "required" in req.response_format["json_schema"]["schema"]
        assert "answer" in req.response_format["json_schema"]["schema"]["required"]

    def test_no_schema_leaves_response_format_none(self) -> None:
        llm = _make_llm_mock()
        llm.generate_response_stream.side_effect = _stream(["x"])
        oracle = LiteLLMOracle(llm, context_window_tokens=1000)
        list(oracle.invoke_stream("hi"))
        req = llm.generate_response_stream.call_args.args[0]
        assert req.response_format is None
