from __future__ import annotations

"""Unit tests for llm.py — LiteLLMInterface with mocked litellm."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
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
    """Opt-in `retries` wiring, tested against a REAL local HTTP provider.

    These tests deliberately do NOT patch `fsm_llm.llm.completion` (that would
    replace the very retry machinery under test) AND deliberately do NOT use a
    `litellm.CustomLLM` provider either. An earlier version of this class used
    `CustomLLM`, and that choice is precisely what hid a defect: the `CustomLLM`
    path BYPASSES the provider SDK's own retry layer, so it is the one
    configuration in which the difference between `max_retries` (SDK layer, one
    layer, correct error classification) and `num_retries` (litellm's tenacity
    layer, stacked ON TOP of the SDK layer, retries everything) is invisible.

    Driving a real `http.server` on loopback exercises the full stack, including
    the SDK retry layer, which is where users actually live. The
    `test_retries_does_not_retry_bad_request` case below is the one that catches
    the original defect.
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
    def http_provider(self):
        """A loopback HTTP server that returns a fixed status and counts requests.

        Yields a mutable dict: set `code` to choose the status returned, read
        `n` for the number of HTTP requests the provider actually received, and
        pass `api_base` to `LiteLLMInterface`.
        """
        import http.server
        import json
        import threading

        state = {"n": 0, "code": 429}

        class Handler(http.server.BaseHTTPRequestHandler):
            # Deliberately HTTP/1.0 (the default): keep-alive against a
            # single-threaded server lets the SDK's connection pool hold a
            # socket open and deadlock the suite.
            def do_POST(self):
                state["n"] += 1
                body = json.dumps(
                    {"error": {"message": "simulated failure", "type": "test"}}
                ).encode()
                self.send_response(state["code"])
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass  # keep pytest output clean

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        state["api_base"] = f"http://127.0.0.1:{server.server_address[1]}"
        # Small poll interval: `shutdown()` blocks for up to one interval, and
        # the default 0.5s would dominate this class's teardown time.
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        thread.start()
        try:
            yield state
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    @staticmethod
    def _interface(state, code, model="openai/gpt-4", **kwargs):
        """Build an interface pointed at the fixture server, resetting its counter."""
        state["code"] = code
        state["n"] = 0
        return LiteLLMInterface(
            model=model,
            api_key="sk-test",
            api_base=state["api_base"],
            timeout=5.0,
            **kwargs,
        )

    def test_retries_two_makes_exactly_three_requests_on_rate_limit(
        self, http_provider
    ):
        """N+1 semantics on a TRANSIENT failure, measured at the HTTP boundary.

        Exactly 3 requests for retries=2 — not the 5 that the stacked
        tenacity layer produced.
        """
        llm = self._interface(http_provider, 429, retries=2)

        with pytest.raises(LLMResponseError):
            llm.generate_response(self._make_request())

        assert http_provider["n"] == 3

    def test_retries_one_lowers_attempts_below_the_sdk_default(self, http_provider):
        """`retries` REPLACES the SDK's default retry count, it does not add to it.

        This is the case that distinguishes "the kwarg is wired" from "the kwarg
        is absent": with the parameter omitted the SDK's own default gives 3
        requests, so a test at retries=2 alone would pass even if the wiring
        were deleted. retries=1 must give 2.
        """
        llm = self._interface(http_provider, 429, retries=1)

        with pytest.raises(LLMResponseError):
            llm.generate_response(self._make_request())

        assert http_provider["n"] == 2

    def test_retries_does_not_retry_bad_request(self, http_provider):
        """A 400 is deterministic — retrying it can never succeed, so it must not be retried.

        THIS IS THE LOAD-BEARING TEST OF THIS CLASS. The superseded
        `num_retries` wiring retried 400s, so a malformed prompt cost 3 round
        trips instead of 1. Any regression to `num_retries` fails here.
        """
        llm = self._interface(http_provider, 400, retries=2)

        with pytest.raises(LLMResponseError):
            llm.generate_response(self._make_request())

        assert http_provider["n"] == 1

    def test_retries_does_not_retry_auth_failure(self, http_provider):
        """A 401 is deterministic too: a wrong API key must fail fast, not 3x."""
        llm = self._interface(http_provider, 401, retries=2)

        with pytest.raises(LLMResponseError):
            llm.generate_response(self._make_request())

        assert http_provider["n"] == 1

    def test_retries_zero_matches_the_pre_change_baseline(self, http_provider):
        """Adversarial control: same server, same failure, only `retries` differs.

        With retries=0 the kwarg is omitted entirely and the provider SDK's own
        default (2 additional attempts) still applies — 3 requests. Retry is not
        "off" by default; the parameter only overrides an existing floor.
        """
        llm = self._interface(http_provider, 429)
        assert llm.retries == 0, "retries must default to 0 (opt-in)"

        with pytest.raises(LLMResponseError):
            llm.generate_response(self._make_request())

        assert http_provider["n"] == 3

    def test_stream_retries_does_not_retry_bad_request(self, http_provider):
        """Pins the SECOND call-param builder, inside `generate_response_stream`.

        `generate_response_stream` does NOT route through `_make_llm_call`; it
        builds its own call params, so a one-site fix is caught here. Same HTTP
        boundary, same deterministic-failure claim.
        """
        llm = self._interface(http_provider, 400, retries=2)

        with pytest.raises(LLMResponseError):
            list(llm.generate_response_stream(self._make_request()))

        assert http_provider["n"] == 1

    def test_stream_retries_one_lowers_attempts_below_the_sdk_default(
        self, http_provider
    ):
        """Stream site: proves the kwarg is actually wired, not merely absent."""
        llm = self._interface(http_provider, 429, retries=1)

        with pytest.raises(LLMResponseError):
            list(llm.generate_response_stream(self._make_request()))

        assert http_provider["n"] == 2

    def test_retries_is_a_documented_no_op_on_ollama(self, http_provider):
        """`retries` is honored by OpenAI-SDK-routed providers ONLY.

        Pinned, not hidden: `ollama_chat/*` makes exactly 1 request no matter
        what `retries` says, because it does not go through the SDK client whose
        `max_retries` this parameter sets. This is the project's own eval
        baseline provider, so the limitation is worth a failing test if it ever
        silently changes. The superseded `num_retries` DID retry here (5 requests
        at retries=4) — but it also retried deterministic 400s, which is why it
        was not kept. See decisions.md D-007.
        """
        llm = self._interface(
            http_provider, 429, model="ollama_chat/qwen3.5:4b", retries=4
        )

        with pytest.raises(LLMResponseError):
            llm.generate_response(self._make_request())

        assert http_provider["n"] == 1

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=[])
    def test_retries_zero_omits_the_retry_key_entirely(
        self, mock_params, mock_completion
    ):
        """Default must OMIT the key, not send `max_retries=0`.

        `max_retries=0` would DISABLE the SDK's default retries, which would be
        a silent behavior change for every existing caller. Params-level
        assertion is appropriate here: the claim under test is literally about
        the emitted params.
        """
        mock_completion.return_value = _mock_llm_response("Hello!")

        llm = LiteLLMInterface(model="test-model")
        llm._make_llm_call([{"role": "user", "content": "hi"}], "response_generation")

        assert "max_retries" not in mock_completion.call_args.kwargs
        assert "num_retries" not in mock_completion.call_args.kwargs

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=[])
    def test_non_positive_retries_omits_the_retry_key(
        self, mock_params, mock_completion
    ):
        """A negative `retries` takes the `<= 0` branch — no validation ceremony."""
        mock_completion.return_value = _mock_llm_response("Hello!")

        llm = LiteLLMInterface(model="test-model", retries=-1)
        llm._make_llm_call([{"role": "user", "content": "hi"}], "response_generation")

        assert "max_retries" not in mock_completion.call_args.kwargs
        assert "num_retries" not in mock_completion.call_args.kwargs

    @patch("fsm_llm.llm.completion")
    @patch("fsm_llm.llm.get_supported_openai_params", return_value=[])
    def test_positive_retries_emits_max_retries_not_num_retries(
        self, mock_params, mock_completion
    ):
        """Names the key explicitly: `num_retries` is the WRONG one.

        `num_retries` routes to litellm's tenacity layer, which stacks on top of
        the SDK layer (2N+1 requests) and retries deterministic 4xx failures.
        """
        mock_completion.return_value = _mock_llm_response("Hello!")

        llm = LiteLLMInterface(model="test-model", retries=3)
        llm._make_llm_call([{"role": "user", "content": "hi"}], "response_generation")

        assert mock_completion.call_args.kwargs["max_retries"] == 3
        assert "num_retries" not in mock_completion.call_args.kwargs


def _stream_chunk(**delta_fields):
    """One litellm-shaped streaming chunk with the given delta attributes."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(**delta_fields))]
    )


class TestStreamEmptyContentGuard:
    """F-02: `generate_response_stream` owes the same tail guards as `_make_llm_call`.

    A stream whose every ``delta.content`` is ``""`` (empty, NOT ``None``) used
    to yield ``['', '']`` — no exception, no log, no fallback — and the pipeline
    then persisted that blank string as an assistant turn.
    """

    @staticmethod
    def _request():
        return ResponseGenerationRequest(
            system_prompt="You are a helpful assistant",
            user_message="hi",
            extracted_data={},
            context={},
            transition_occurred=False,
        )

    def _stream(self, chunks):
        return patch("fsm_llm.llm.completion", return_value=iter(chunks)), patch(
            "fsm_llm.llm.get_supported_openai_params", return_value=[]
        )

    def test_thinking_field_is_recovered_when_every_delta_content_is_empty(self):
        """The answer lives in `delta.thinking`; it must reach the caller."""
        chunks = [
            _stream_chunk(content="", thinking=""),
            _stream_chunk(content="", thinking='{"message": "Hello there!"}'),
        ]
        completion_patch, params_patch = self._stream(chunks)
        llm = LiteLLMInterface(model="test-model")

        with completion_patch, params_patch:
            out = list(llm.generate_response_stream(self._request()))

        assert "".join(out) == '{"message": "Hello there!"}', (
            f"thinking content not recovered: {out}"
        )

    def test_all_empty_stream_without_thinking_raises(self):
        """No content and nothing to recover is an error, not a silent blank reply."""
        chunks = [_stream_chunk(content=""), _stream_chunk(content="")]
        completion_patch, params_patch = self._stream(chunks)
        llm = LiteLLMInterface(model="test-model")

        with (
            completion_patch,
            params_patch,
            pytest.raises(LLMResponseError, match="empty content"),
        ):
            list(llm.generate_response_stream(self._request()))

    def test_normal_stream_is_unchanged(self):
        """Control: a stream with real content must not gain a fallback chunk."""
        chunks = [
            _stream_chunk(content="Hel", thinking="ignored"),
            _stream_chunk(content="lo"),
        ]
        completion_patch, params_patch = self._stream(chunks)
        llm = LiteLLMInterface(model="test-model")

        with completion_patch, params_patch:
            out = list(llm.generate_response_stream(self._request()))

        assert out == ["Hel", "lo"]


def _real_delta_chunk(content, reasoning_content=None):
    """One streaming chunk wrapping a REAL litellm ``Delta``.

    The load-bearing part is the ``Delta`` object's own field resolution
    (``reasoning_content`` vs the deleted legacy ``thinking``) — the chunk /
    choice wrapper is only read as ``chunk.choices[0].delta`` and may be a
    plain namespace.
    """
    from litellm.types.utils import Delta

    delta = Delta(content=content, reasoning_content=reasoning_content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class TestStreamReasoningFragmentAccumulation:
    """G3: reasoning streamed as fragments across chunks must be reassembled.

    A provider may stream ``reasoning_content`` incrementally — one fragment per
    chunk, exactly like it streams ``content`` — while ``content`` stays empty in
    every chunk. The empty-content tail guard must recover the answer from the
    reasoning trace ACCUMULATED across all chunks (matching the non-streaming
    path, which sees the full assembled message), not merely the final delta.

    Driven by REAL ``litellm.types.utils.Delta`` objects (per plans/LESSONS.md):
    a hand-stubbed ``.thinking`` object would mask the field-resolution seam and
    the ``last_delta``-only bug both.
    """

    @staticmethod
    def _request():
        return ResponseGenerationRequest(
            system_prompt="You are a helpful assistant",
            user_message="hi",
            extracted_data={},
            context={},
            transition_occurred=False,
        )

    def _stream(self, chunks):
        return patch("fsm_llm.llm.completion", return_value=iter(chunks)), patch(
            "fsm_llm.llm.get_supported_openai_params", return_value=[]
        )

    def test_stream_recovers_reasoning_split_across_multiple_fragments(self):
        """The JSON answer is split as reasoning fragments; content empty in each.

        Every chunk's ``delta.content`` is empty/None and the JSON object
        ``{"message": "Hi"}`` arrives as ``reasoning_content`` FRAGMENTS across
        three chunks. Pre-fix code inspects only ``last_delta`` (fragment
        ``'"Hi"}'``) and cannot recover the whole object; the accumulation fix
        joins all fragments and recovers the full message.
        """
        chunks = [
            _real_delta_chunk(content="", reasoning_content='{"mess'),
            _real_delta_chunk(content=None, reasoning_content='age": '),
            _real_delta_chunk(content="", reasoning_content='"Hi"}'),
        ]
        completion_patch, params_patch = self._stream(chunks)
        llm = LiteLLMInterface(model="test-model")

        with completion_patch, params_patch:
            out = list(llm.generate_response_stream(self._request()))

        assert "".join(out) == '{"message": "Hi"}', (
            f"accumulated reasoning not recovered from fragments: {out}"
        )

    def test_stream_recovers_when_whole_trace_is_on_final_delta_only(self):
        """Defensive fallback: a provider that lumps the whole trace on the last
        delta (no per-chunk fragments) still recovers — accumulation of a single
        fragment equals that fragment, and the ``last_delta`` fallback covers a
        trace that never appears in per-chunk ``reasoning_content``.
        """
        chunks = [
            _real_delta_chunk(content="", reasoning_content=None),
            _real_delta_chunk(content="", reasoning_content='{"message": "Whole"}'),
        ]
        completion_patch, params_patch = self._stream(chunks)
        llm = LiteLLMInterface(model="test-model")

        with completion_patch, params_patch:
            out = list(llm.generate_response_stream(self._request()))

        assert "".join(out) == '{"message": "Whole"}', (
            f"single-lump reasoning not recovered: {out}"
        )


class TestReasoningContentRecovery:
    """Regression for C2 + H3: recover the answer from the reasoning field.

    These tests are driven by REAL ``litellm.types.utils.Message``/``Delta``
    objects — NOT a hand-stubbed ``.thinking`` attribute. That distinction is
    load-bearing: installed litellm renames the raw ``thinking`` field to
    ``reasoning_content`` and DELETES ``thinking`` before building the object
    (``litellm/types/utils.py``). A ``.thinking``-only stub is therefore GREEN
    on the broken pre-fix code — it never probed the gap, which is exactly how
    C2 shipped. See plan D-001/D-002.
    """

    def test_message_reasoning_content_is_recovered(self):
        """SC-1: real Message(content="", reasoning_content=json) → recovered."""
        from litellm.types.utils import Message

        msg = Message(content="", reasoning_content='{"message": "recovered"}')
        # Contract precondition: real litellm Message has no `.thinking` attr,
        # so the pre-fix `hasattr(message, "thinking")` gate returns None here.
        assert not hasattr(msg, "thinking"), (
            "real litellm Message must not expose a `.thinking` attr — if this "
            "fails the negative control no longer documents the C2 gap"
        )

        recovered = LiteLLMInterface._extract_content_from_thinking(msg)
        assert recovered == '{"message": "recovered"}'

    def test_make_llm_call_recovers_from_none_content(self):
        """SC-2: content=None + reasoning_content, driven through _make_llm_call.

        Covers H3 (guard must fire on None, not only "") and C2 (helper must
        read reasoning_content) together at the real seam.
        """
        from litellm.types.utils import Message

        msg = Message(content=None, reasoning_content='{"message": "from-reasoning"}')
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]

        with (
            patch("fsm_llm.llm.completion", return_value=response),
            patch("fsm_llm.llm.get_supported_openai_params", return_value=[]),
        ):
            llm = LiteLLMInterface(model="test-model")
            out = llm._make_llm_call([{"role": "user", "content": "hi"}], "test")

        assert out.choices[0].message.content == '{"message": "from-reasoning"}'

    def test_delta_reasoning_content_is_recovered(self):
        """SC-3: real streaming Delta(reasoning_content=json) → recovered."""
        from litellm.types.utils import Delta

        delta = Delta(reasoning_content='{"message": "x"}')
        assert not hasattr(delta, "thinking")

        recovered = LiteLLMInterface._extract_content_from_thinking(delta)
        assert recovered == '{"message": "x"}'

    def test_thinking_stub_masks_the_bug_negative_control(self):
        """SC (negative control): document WHY a .thinking stub is worthless here.

        A hand-stubbed object that carries `.thinking` is recovered even by the
        BROKEN pre-fix helper — so a `.thinking`-only test can never go RED on
        the C2 bug. A real Message carries `reasoning_content`, not `thinking`;
        that is the difference these regression tests exist to pin.
        """
        from litellm.types.utils import Message

        real = Message(content="", reasoning_content='{"message": "real"}')
        assert not hasattr(real, "thinking")
        assert getattr(real, "reasoning_content", None) == '{"message": "real"}'


def _field_request(field_name: str = "topic") -> FieldExtractionRequest:
    return FieldExtractionRequest(
        system_prompt="extract the field",
        user_message="x",
        field_name=field_name,
    )


class TestFieldExtractionNaNConfidence:
    """G5: a bare ``NaN``/``Infinity`` confidence from ``json.loads`` must be
    coerced to a clamped [0,1] default, never escape as a pydantic
    ValidationError that fails the whole turn.

    ``json.loads`` accepts bare ``NaN``/``Infinity`` tokens (no ``parse_constant``
    override in this codebase), so the raw strings below produce a real
    ``float('nan')``/``float('inf')`` inside the parser — the exact production
    shape, not a hand-stubbed value.
    """

    def test_primary_rung_nan_confidence_yields_valid_response(self):
        # Directly-parseable JSON hits the PRIMARY rung (~llm.py:867-878).
        response = _mock_llm_response('{"value": "x", "confidence": NaN}')
        llm = LiteLLMInterface(model="test-model")

        result = llm._parse_field_extraction_response(response, _field_request())

        assert isinstance(result, FieldExtractionResponse)
        assert result.value == "x"
        assert 0.0 <= result.confidence <= 1.0

    def test_primary_rung_infinity_confidence_yields_valid_response(self):
        response = _mock_llm_response('{"value": "x", "confidence": Infinity}')
        llm = LiteLLMInterface(model="test-model")

        result = llm._parse_field_extraction_response(response, _field_request())

        assert isinstance(result, FieldExtractionResponse)
        assert 0.0 <= result.confidence <= 1.0

    def test_fallback_rung_nan_confidence_yields_valid_response(self):
        # Prose-wrapped JSON skips the primary rung and is recovered by
        # extract_json_from_text on the embedded-JSON fallback (~llm.py:899-914).
        response = _mock_llm_response(
            'Here is the result: {"value": "x", "confidence": NaN} thanks.'
        )
        llm = LiteLLMInterface(model="test-model")

        result = llm._parse_field_extraction_response(response, _field_request())

        assert isinstance(result, FieldExtractionResponse)
        assert result.value == "x"
        assert 0.0 <= result.confidence <= 1.0

    def test_valid_confidence_preserved(self):
        response = _mock_llm_response('{"value": "x", "confidence": 0.7}')
        llm = LiteLLMInterface(model="test-model")

        result = llm._parse_field_extraction_response(response, _field_request())

        assert result.confidence == 0.7

    def test_dict_confidence_still_degrades_via_ladder(self):
        # A ``{...}`` confidence raises TypeError inside float() and must remain
        # caught by the existing ladder — no new exception escapes.
        response = _mock_llm_response('{"value": "x", "confidence": {"bad": 1}}')
        llm = LiteLLMInterface(model="test-model")

        result = llm._parse_field_extraction_response(response, _field_request())

        # Falls through to the unstructured str-coercion rung; must not raise.
        assert isinstance(result, FieldExtractionResponse)
