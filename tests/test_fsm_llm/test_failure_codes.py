"""``LlmFailureCode`` and ``failure_code`` of ``provider/error`` (plan 3f4530d3 step 6c, D-010).

The mapper is built from the REAL litellm exception classes (so a renamed class after a pin
refresh turns a row red) and from class names only: ``llm/failures.py`` imports no provider
library. The order is the mapping: litellm's ``Timeout`` is an ``APIConnectionError`` with status
408, ``APIConnectionError`` reports 500 and ``ContextWindowExceededError`` and
``ContentPolicyViolationError`` are ``BadRequestError`` (400), so a status-first mapper puts them
in the wrong bucket. The recorded fixtures at the end drive a refusal and a garbled tool call
through the real provider path and the log.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import litellm
import pytest

from fsm_llm.core.codes import LlmFailureCode
from fsm_llm.core.definitions import (
    CompletionRequest,
    LLMResponseError,
    ResponseGenerationRequest,
)
from fsm_llm.llm import LiteLLMInterface, failures
from fsm_llm.llm.failures import classify_failure
from fsm_llm.runtime import EventLog, bind, llm_call
from tests.test_fsm_llm.event_log_support import tree_bytes

ROOT = "root-1"
COMPLETION = "fsm_llm.llm.providers.litellm.completion"
_RESPONSE = httpx.Response(
    400, request=httpx.Request("POST", "http://provider.invalid")
)
_GARBLED = "XML syntax error on line 3: element <function> closed by </parameter>"


def _real(name: str) -> Exception:
    cls = getattr(litellm, name)
    try:
        return cls("down", model="m", llm_provider="p")
    except TypeError:  # PermissionDeniedError and others also want the response
        return cls("down", model="m", llm_provider="p", response=_RESPONSE)


def _code(exc: BaseException, *, tools: bool = False) -> LlmFailureCode:
    return classify_failure(exc, tools_sent=tools)


class TestRealLitellmClasses:
    @pytest.mark.parametrize(
        ("name", "status", "code"),
        [
            ("Timeout", 408, LlmFailureCode.TIMEOUT),
            ("APIConnectionError", 500, LlmFailureCode.TRANSPORT),
            ("ContentPolicyViolationError", 400, LlmFailureCode.CONTENT_REFUSED),
            ("ContextWindowExceededError", 400, LlmFailureCode.CONTEXT_OVERFLOW),
            ("RateLimitError", 429, LlmFailureCode.RATE_LIMITED),
            ("AuthenticationError", 401, LlmFailureCode.AUTH),
            ("PermissionDeniedError", 400, LlmFailureCode.AUTH),
            ("BadRequestError", 400, LlmFailureCode.HTTP_ERROR),
            ("NotFoundError", 404, LlmFailureCode.HTTP_ERROR),
            ("InternalServerError", 500, LlmFailureCode.HTTP_ERROR),
            ("ServiceUnavailableError", 503, LlmFailureCode.HTTP_ERROR),
        ],
    )
    def test_each_class_maps_to_its_code(
        self, name: str, status: int, code: LlmFailureCode
    ) -> None:
        exc = _real(name)
        assert exc.status_code == status  # type: ignore[attr-defined]
        assert _code(exc) is code

    def test_the_classes_the_order_depends_on_still_have_their_shape(self) -> None:
        # the premise of D-010: a pin refresh that changes it must fail here, not in a log
        # (by NAME: litellm's `Timeout` derives from openai's `APITimeoutError`, itself an
        # openai `APIConnectionError`, a different class from litellm's of the same name)
        assert "APIConnectionError" in {c.__name__ for c in litellm.Timeout.__mro__}
        assert issubclass(litellm.ContextWindowExceededError, litellm.BadRequestError)
        assert issubclass(litellm.ContentPolicyViolationError, litellm.BadRequestError)

    def test_a_status_first_mapper_would_be_wrong_for_these(self) -> None:
        # `Timeout` carries 408 and `APIConnectionError` carries 500: by status both are an
        # http error, by class they are a timeout and a transport failure
        assert _real("Timeout").status_code == 408  # type: ignore[attr-defined]
        assert _code(_real("Timeout")) is LlmFailureCode.TIMEOUT
        assert _code(_real("APIConnectionError")) is LlmFailureCode.TRANSPORT


class TestOrderAndFallbacks:
    def test_the_marker_runs_before_the_class_names(self) -> None:
        # Ollama's garbled tool call arrives as a plain `APIConnectionError`
        exc = litellm.APIConnectionError(_GARBLED, llm_provider="p", model="m")
        assert _code(exc, tools=True) is LlmFailureCode.MALFORMED_RESPONSE
        # a request without tools cannot garble one: the same text is an outage
        assert _code(exc, tools=False) is LlmFailureCode.TRANSPORT

    def test_the_marker_is_the_only_text_the_mapper_reads(self) -> None:
        assert _code(ValueError("rate limit exceeded, 429")) is LlmFailureCode.UNKNOWN
        assert _code(RuntimeError("context length exceeded")) is LlmFailureCode.UNKNOWN

    @pytest.mark.parametrize(
        ("exc", "code"),
        [
            (TimeoutError("slow"), LlmFailureCode.TIMEOUT),
            (TimeoutError(), LlmFailureCode.TIMEOUT),
            (httpx.ReadTimeout("slow"), LlmFailureCode.TIMEOUT),
            (ConnectionResetError("reset"), LlmFailureCode.TRANSPORT),
            (OSError("unreachable"), LlmFailureCode.TRANSPORT),
            (ValueError("bug"), LlmFailureCode.UNKNOWN),
            (KeyError("k"), LlmFailureCode.UNKNOWN),
        ],
    )
    def test_builtin_and_httpx_errors(
        self, exc: BaseException, code: LlmFailureCode
    ) -> None:
        assert _code(exc) is code

    def test_an_integer_status_of_400_or_more_is_an_http_error(self) -> None:
        class Odd(Exception):
            status_code = 502

        assert _code(Odd()) is LlmFailureCode.HTTP_ERROR

    @pytest.mark.parametrize("status", [200, 399, True, "502", None, 502.0])
    def test_any_other_status_is_not(self, status: Any) -> None:
        class Odd(Exception):
            pass

        odd = Odd()
        odd.status_code = status  # type: ignore[attr-defined]
        assert _code(odd) is LlmFailureCode.UNKNOWN

    def test_a_cancelled_call_is_cancelled(self) -> None:
        assert _code(asyncio.CancelledError()) is LlmFailureCode.CANCELLED

    def test_the_cause_chain_is_followed(self) -> None:
        try:
            try:
                raise _real("RateLimitError")
            except Exception as provider:
                raise LLMResponseError("wrapped") from provider
        except LLMResponseError as wrapped:
            assert _code(wrapped) is LlmFailureCode.RATE_LIMITED

    def test_a_cause_chain_that_loops_ends_in_unknown(self) -> None:
        a, b = ValueError("a"), ValueError("b")
        a.__cause__, b.__cause__ = b, a
        assert _code(a) is LlmFailureCode.UNKNOWN

    def test_the_marker_is_found_through_a_wrapper(self) -> None:
        inner = litellm.APIConnectionError(_GARBLED, llm_provider="p", model="m")
        try:
            try:
                raise inner
            except Exception as provider:
                raise LLMResponseError("wrapped") from provider
        except LLMResponseError as wrapped:
            assert _code(wrapped, tools=True) is LlmFailureCode.MALFORMED_RESPONSE

    def test_the_mapper_imports_no_provider_library(self) -> None:
        tree = ast.parse(Path(failures.__file__).read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert "litellm" not in imported and "openai" not in imported


def _events(log: EventLog, *types: str) -> list[dict[str, Any]]:
    return list(log.query(ROOT, types=types or None))


def _request() -> ResponseGenerationRequest:
    return ResponseGenerationRequest(system_prompt="Greet.", user_message="hi")


class TestRecordedFixtures:
    """A refusal and a malformed reply, through the real provider path and the log."""

    def test_a_refusal_and_a_malformed_tool_call_are_distinct_codes(
        self, log: EventLog
    ) -> None:
        tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]

        def refuse(**kwargs: Any) -> Any:
            raise _real("ContentPolicyViolationError")

        def garble(**kwargs: Any) -> Any:
            raise litellm.APIConnectionError(_GARBLED, llm_provider="p", model="m")

        llm = LiteLLMInterface(model="gpt-4o")
        with bind(log, ROOT):
            with patch(COMPLETION, refuse), pytest.raises(LLMResponseError):
                llm.generate_response(_request())
            with patch(COMPLETION, garble):
                reply = llm.complete(
                    CompletionRequest(
                        messages=[{"role": "user", "content": "hi"}], tools=tools
                    )
                )
        assert reply.kind == "malformed"
        refused, malformed = _events(log, "provider/error")
        assert refused["failure_code"] == "content_refused"
        assert refused["status_code"] == 400
        assert malformed["failure_code"] == "malformed_response"
        assert refused["failure_code"] != malformed["failure_code"]
        assert refused["purpose"] == "reply" and malformed["purpose"] == "complete"

    def test_the_same_garbled_text_without_tools_is_a_transport_failure(
        self, log: EventLog
    ) -> None:
        def garble(**kwargs: Any) -> Any:
            raise litellm.APIConnectionError(_GARBLED, llm_provider="p", model="m")

        with bind(log, ROOT), patch(COMPLETION, garble):
            with pytest.raises(LLMResponseError):
                LiteLLMInterface(model="gpt-4o").generate_response(_request())
        (error,) = _events(log, "provider/error")
        assert error["failure_code"] == "transport"

    def test_an_engine_logged_call_writes_the_code_too(self, log: EventLog) -> None:
        class Down:
            def generate_response(self, request: Any) -> Any:
                raise _real("RateLimitError")

        with bind(log, ROOT):
            with pytest.raises(litellm.RateLimitError):
                with llm_call(Down(), "response_generation", _request()):
                    Down().generate_response(_request())
        (error,) = _events(log, "provider/error")
        assert error["source"] == "engine"
        assert (error["failure_code"], error["status_code"]) == ("rate_limited", 429)

    def test_a_cancelled_engine_call_is_cancelled(self, log: EventLog) -> None:
        async def run() -> None:
            with bind(log, ROOT):
                with llm_call(object(), "response_generation", _request()):
                    raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run())
        (error,) = _events(log, "provider/error")
        assert error["failure_code"] == "cancelled"

    def test_the_code_is_a_closed_value_and_no_text_reaches_the_log(
        self, log: EventLog
    ) -> None:
        def refuse(**kwargs: Any) -> Any:
            raise litellm.ContentPolicyViolationError(
                "blocked: PRIVATE-PROMPT-TEXT", model="m", llm_provider="p"
            )

        with bind(log, ROOT), patch(COMPLETION, refuse):
            with pytest.raises(LLMResponseError):
                LiteLLMInterface(model="gpt-4o").generate_response(_request())
        (error,) = _events(log, "provider/error")
        assert LlmFailureCode.from_string(error["failure_code"])
        assert b"PRIVATE-PROMPT-TEXT" not in tree_bytes(log.root)

    def test_a_shape_log_carries_purpose_and_code_and_no_reference(
        self, tmp_path: Path
    ) -> None:
        def refuse(**kwargs: Any) -> Any:
            raise _real("ContentPolicyViolationError")

        with EventLog(tmp_path / "shape", privacy="shape") as shape:
            with bind(shape, ROOT), patch(COMPLETION, refuse):
                with pytest.raises(LLMResponseError):
                    LiteLLMInterface(model="gpt-4o").generate_response(_request())
            request, error = list(shape.query(ROOT))
        assert request["purpose"] == "reply"
        assert error["purpose"] == "reply"
        assert error["failure_code"] == "content_refused"
        assert not [key for key in (*request, *error) if key.endswith("_ref")]

    def test_a_log_written_before_the_field_has_none_and_reads_fine(
        self, log: EventLog
    ) -> None:
        # old logs carry no `failure_code`: nothing in the fold may require it
        log.append(ROOT, "provider/error", frame=0, source="provider", error_type="X")
        (error,) = _events(log, "provider/error")
        assert "failure_code" not in error
