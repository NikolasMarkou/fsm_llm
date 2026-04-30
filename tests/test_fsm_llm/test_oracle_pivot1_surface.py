from __future__ import annotations

"""Tests for the D-PIVOT-1 LiteLLMOracle surface extension (step 10).

Adds three new/extended methods on ``LiteLLMOracle`` to unblock the 3
deferred R10 sites (D-STEP-7-SUMMARY → D-PIVOT-1):

1. ``invoke_messages(messages, *, schema=None, response_format=None,
   call_type='data_extraction')`` — pre-built OpenAI message array
   passthrough to ``LLMInterface._make_llm_call``. Returns the **raw
   litellm response object** so callers retain bespoke parsing.
2. ``invoke_field(request: FieldExtractionRequest)`` — direct passthrough
   to ``LLMInterface.extract_field``. Preserves the legacy outer-envelope
   schema (different from ``_invoke_structured``'s D-008 path).
3. ``invoke(prompt, *, user_message='', response_format=None, ...)`` —
   extended with ``user_message`` and ``response_format`` kwargs to
   support the canonical Pass-2 main response site (pipeline.py L2223 /
   D-R10-7.5). Default empty ``user_message`` preserves all M1
   byte-equivalence for Executor-driven Leaf calls.

Mirrors the shape of ``test_oracle_invoke_stream.py``.
"""

from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from fsm_llm.runtime._litellm import LLMInterface
from fsm_llm.runtime.errors import OracleError
from fsm_llm.runtime.oracle import LiteLLMOracle, Oracle
from fsm_llm.types import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    LLMResponseError,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)


class _AnswerSchema(BaseModel):
    answer: str


def _make_llm_mock() -> Mock:
    # NOTE: do NOT spec on LLMInterface ABC — invoke_messages reaches
    # into ``_make_llm_call`` which is concrete on LiteLLMInterface
    # (private-by-convention). A bare Mock allows that attribute access
    # while we still set the public surface explicitly.
    m = Mock()
    m.model = "ollama_chat/qwen3.5:4b"
    m.kwargs = {}
    m.max_tokens = 1000
    m.timeout = None
    return m


def _make_abc_llm_mock() -> Mock:
    """For tests of ``invoke`` (uses ``generate_response`` only) and
    ``invoke_field`` (uses ``extract_field`` only) — both are on the ABC,
    so spec=LLMInterface is appropriate."""
    m = Mock(spec=LLMInterface)
    m.model = "ollama_chat/qwen3.5:4b"
    m.kwargs = {}
    m.max_tokens = 1000
    m.timeout = None
    return m


def _fake_litellm_response(content: str):
    """Build a minimal object mimicking the litellm ``ChatCompletion``
    response shape that ``_make_llm_call`` returns."""
    msg = Mock()
    msg.content = content
    choice = Mock()
    choice.message = msg
    resp = Mock()
    resp.choices = [choice]
    return resp


# --------------------------------------------------------------------
# Capability / Protocol surface
# --------------------------------------------------------------------


class TestProtocolAndCapability:
    def test_invoke_messages_method_exists(self) -> None:
        oracle = LiteLLMOracle(_make_llm_mock())
        assert callable(getattr(oracle, "invoke_messages", None))

    def test_invoke_field_method_exists(self) -> None:
        oracle = LiteLLMOracle(_make_llm_mock())
        assert callable(getattr(oracle, "invoke_field", None))

    def test_oracle_protocol_still_satisfied(self) -> None:
        # Adding two new concrete methods + extending invoke kwargs
        # must not break the runtime-checkable Oracle Protocol.
        oracle = LiteLLMOracle(_make_llm_mock())
        assert isinstance(oracle, Oracle)


# --------------------------------------------------------------------
# invoke_messages
# --------------------------------------------------------------------


class TestInvokeMessages:
    def test_basic_passthrough_returns_raw_response(self) -> None:
        llm = _make_llm_mock()
        fake = _fake_litellm_response('{"extracted_data": {"k": "v"}}')
        llm._make_llm_call.return_value = fake
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)

        messages = [
            {"role": "system", "content": "Extract."},
            {"role": "user", "content": "k is v"},
        ]
        result = oracle.invoke_messages(messages)

        assert result is fake
        # Raw passthrough: should not parse content; caller does that.
        # When response_format is None (the default), the kwarg is omitted
        # entirely from the underlying call — preserves ABI for narrower
        # _make_llm_call signatures (e.g. test spies that match the legacy
        # L1289 signature which never passed response_format).
        llm._make_llm_call.assert_called_once_with(messages, "data_extraction")

    def test_schema_converted_to_json_schema_response_format(self) -> None:
        llm = _make_llm_mock()
        llm._make_llm_call.return_value = _fake_litellm_response("{}")
        oracle = LiteLLMOracle(llm)

        oracle.invoke_messages(
            [{"role": "user", "content": "x"}],
            schema=_AnswerSchema,
        )

        _, kwargs = llm._make_llm_call.call_args
        rf = kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "_AnswerSchema"
        # Required list is synthesised when Pydantic omits it.
        assert "answer" in rf["json_schema"]["schema"]["required"]

    def test_explicit_response_format_wins_over_schema(self) -> None:
        llm = _make_llm_mock()
        llm._make_llm_call.return_value = _fake_litellm_response("{}")
        oracle = LiteLLMOracle(llm)

        explicit = {"type": "json_object"}
        oracle.invoke_messages(
            [{"role": "user", "content": "x"}],
            schema=_AnswerSchema,
            response_format=explicit,
        )

        _, kwargs = llm._make_llm_call.call_args
        assert kwargs["response_format"] is explicit

    def test_call_type_default_matches_legacy_l1289(self) -> None:
        # Legacy site uses ``_make_llm_call(messages, "data_extraction")``;
        # default call_type must match.
        llm = _make_llm_mock()
        llm._make_llm_call.return_value = _fake_litellm_response("{}")
        oracle = LiteLLMOracle(llm)

        oracle.invoke_messages([{"role": "user", "content": "x"}])

        args, _ = llm._make_llm_call.call_args
        assert args[1] == "data_extraction"

    def test_custom_call_type_forwarded(self) -> None:
        llm = _make_llm_mock()
        llm._make_llm_call.return_value = _fake_litellm_response("{}")
        oracle = LiteLLMOracle(llm)

        oracle.invoke_messages(
            [{"role": "user", "content": "x"}],
            call_type="field_extraction",
        )

        args, _ = llm._make_llm_call.call_args
        assert args[1] == "field_extraction"

    def test_llm_response_error_wraps_to_oracle_error(self) -> None:
        llm = _make_llm_mock()
        llm._make_llm_call.side_effect = LLMResponseError("network down")
        oracle = LiteLLMOracle(llm)

        with pytest.raises(OracleError, match="messages call failed"):
            oracle.invoke_messages([{"role": "user", "content": "x"}])


# --------------------------------------------------------------------
# invoke_field
# --------------------------------------------------------------------


def _field_request() -> FieldExtractionRequest:
    return FieldExtractionRequest(
        system_prompt="Extract age.",
        user_message="I am 30",
        field_name="age",
        field_type="int",
        context={},
        validation_rules={},
    )


class TestInvokeField:
    def test_passthrough_returns_extract_field_result(self) -> None:
        llm = _make_llm_mock()
        expected = FieldExtractionResponse(
            field_name="age",
            value=30,
            confidence=0.95,
            is_valid=True,
        )
        llm.extract_field.return_value = expected
        oracle = LiteLLMOracle(llm)

        result = oracle.invoke_field(_field_request())

        assert result is expected

    def test_request_forwarded_unchanged(self) -> None:
        llm = _make_llm_mock()
        llm.extract_field.return_value = FieldExtractionResponse(
            field_name="age", value=30, confidence=1.0, is_valid=True
        )
        oracle = LiteLLMOracle(llm)
        req = _field_request()

        oracle.invoke_field(req)

        llm.extract_field.assert_called_once_with(req)
        # Same object identity — no envelope unwrapping at oracle boundary.
        assert llm.extract_field.call_args.args[0] is req

    def test_llm_response_error_wraps_to_oracle_error(self) -> None:
        llm = _make_llm_mock()
        llm.extract_field.side_effect = LLMResponseError("schema mismatch")
        oracle = LiteLLMOracle(llm)

        with pytest.raises(OracleError, match="field-extraction call failed"):
            oracle.invoke_field(_field_request())

    def test_envelope_preserved_in_response(self) -> None:
        # The legacy outer envelope (field_name, value, confidence, reasoning,
        # is_valid) reaches the caller intact — distinct from D-008's
        # _invoke_structured path which strips to a bare dict.
        llm = _make_llm_mock()
        llm.extract_field.return_value = FieldExtractionResponse(
            field_name="age",
            value=30,
            confidence=0.9,
            is_valid=True,
            reasoning="extracted from 'I am 30'",
        )
        oracle = LiteLLMOracle(llm)

        resp = oracle.invoke_field(_field_request())

        assert resp.field_name == "age"
        assert resp.value == 30
        assert resp.confidence == 0.9
        assert resp.is_valid is True
        assert resp.reasoning == "extracted from 'I am 30'"


# --------------------------------------------------------------------
# invoke(user_message=) extension
# --------------------------------------------------------------------


def _resp_gen_response(message: str) -> ResponseGenerationResponse:
    return ResponseGenerationResponse(
        message=message,
        message_type="response",
        reasoning=None,
    )


class TestInvokeUserMessageKwarg:
    def test_default_empty_preserves_m1_byte_equivalence(self) -> None:
        # Default user_message="" — must produce identical
        # ResponseGenerationRequest as the pre-pivot path.
        llm = _make_llm_mock()
        llm.generate_response.return_value = _resp_gen_response("ok")
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)

        oracle.invoke("Say hi.")

        req = llm.generate_response.call_args.args[0]
        assert isinstance(req, ResponseGenerationRequest)
        assert req.system_prompt == "Say hi."
        assert req.user_message == ""
        assert req.response_format is None

    def test_user_message_propagates_to_request(self) -> None:
        llm = _make_llm_mock()
        llm.generate_response.return_value = _resp_gen_response("ok")
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)

        oracle.invoke("System prompt.", user_message="User asks: how?")

        req = llm.generate_response.call_args.args[0]
        assert req.user_message == "User asks: how?"
        assert req.system_prompt == "System prompt."

    def test_response_format_kwarg_propagates_to_request(self) -> None:
        # Terminal-state structured-output path (pipeline.py L2202-L2213
        # ``output_response_format``) must reach the underlying
        # ResponseGenerationRequest.
        llm = _make_llm_mock()
        llm.generate_response.return_value = _resp_gen_response("ok")
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)

        rf = {"type": "json_object"}
        oracle.invoke("sys", user_message="usr", response_format=rf)

        req = llm.generate_response.call_args.args[0]
        assert req.response_format == rf

    def test_byte_equivalence_with_legacy_generate_response_call(self) -> None:
        # Wire-parity proof for D-PIVOT-1 step 11 site 7.5 rewire:
        # ``oracle.invoke(req.system_prompt, user_message=req.user_message,
        # response_format=req.response_format)`` must produce a
        # ResponseGenerationRequest with the same wire-relevant fields as
        # the legacy ``self.llm_interface.generate_response(req)`` call.
        llm = _make_llm_mock()
        llm.generate_response.return_value = _resp_gen_response("ok")
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)

        legacy_req = ResponseGenerationRequest(
            system_prompt="You are a helpful assistant.",
            user_message="What is 2+2?",
            extracted_data={"k": "v"},  # auxiliary — not on the wire
            context={"x": 1},  # auxiliary — not on the wire
            transition_occurred=True,  # auxiliary — not on the wire
            previous_state="prev",  # auxiliary — not on the wire
            response_format={"type": "json_object"},
        )

        oracle.invoke(
            legacy_req.system_prompt,
            user_message=legacy_req.user_message,
            response_format=legacy_req.response_format,
        )

        new_req = llm.generate_response.call_args.args[0]
        # The 3 wire-relevant fields match (system_prompt + user_message +
        # response_format are the only fields that reach the underlying
        # litellm.completion call per runtime/_litellm.py:254-264).
        assert new_req.system_prompt == legacy_req.system_prompt
        assert new_req.user_message == legacy_req.user_message
        assert new_req.response_format == legacy_req.response_format
