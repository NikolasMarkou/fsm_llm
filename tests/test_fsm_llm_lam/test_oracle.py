from __future__ import annotations

"""Tests for fsm_llm.lam.oracle — protocol, guards, dispatch."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from fsm_llm.runtime._litellm import LLMInterface
from fsm_llm.runtime.errors import OracleError
from fsm_llm.runtime.oracle import LiteLLMOracle, Oracle, _resolve_schema
from fsm_llm.types import (
    LLMResponseError,
    ResponseGenerationResponse,
)


class _SampleSchema(BaseModel):
    answer: str
    score: float


def _make_llm_mock() -> Mock:
    m = Mock(spec=LLMInterface)
    # Attach a .model attribute so LiteLLMOracle can read it.
    m.model = "ollama_chat/qwen3.5:4b"
    # Attach kwargs/timeout/max_tokens so the structured path's
    # call-param assembly can read them.
    m.kwargs = {}
    m.max_tokens = 1000
    m.timeout = None
    return m


def _fake_completion(content: str):
    """Build a litellm-style response object exposing ``.choices[0].message.content``."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


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
        llm.generate_response.return_value = ResponseGenerationResponse(message="pong")
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
    """D-008: Structured calls bypass ``extract_field`` and route through
    ``generate_response`` with a Pydantic-derived ``response_format``."""

    def test_generate_response_called_and_validated(self) -> None:
        # Slice 4: structured path now bypasses ``generate_response`` and
        # calls ``litellm.completion`` directly, then extracts the raw
        # ``choices[0].message.content``. We patch litellm.completion.
        llm = _make_llm_mock()
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)
        with patch(
            "litellm.completion",
            return_value=_fake_completion('{"answer": "yes", "score": 0.9}'),
        ) as fake:
            out = oracle.invoke("Q?", schema=_SampleSchema)
        assert out == {"answer": "yes", "score": 0.9}
        # extract_field is never called on the structured path.
        assert llm.extract_field.call_count == 0
        # response_format derived from the schema is threaded through.
        call_kwargs = fake.call_args.kwargs
        rf = call_kwargs.get("response_format")
        assert rf is not None
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "_SampleSchema"
        # Required fields are synthesised even when the Pydantic schema
        # has no required-from-defaults (slice 4 D-011 fix).
        assert "required" in rf["json_schema"]["schema"]
        # temperature forced to 0 for grammar-constrained decoding.
        assert call_kwargs.get("temperature") == 0

    def test_structured_strips_markdown_fences(self) -> None:
        llm = _make_llm_mock()
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)
        with patch(
            "litellm.completion",
            return_value=_fake_completion(
                '```json\n{"answer": "yes", "score": 0.9}\n```'
            ),
        ):
            out = oracle.invoke("Q?", schema=_SampleSchema)
        assert out == {"answer": "yes", "score": 0.9}

    def test_structured_non_json_rejected(self) -> None:
        llm = _make_llm_mock()
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)
        with (
            patch(
                "litellm.completion",
                return_value=_fake_completion("not a json object at all"),
            ),
            pytest.raises(OracleError, match="did not return valid JSON"),
        ):
            oracle.invoke("Q?", schema=_SampleSchema)

    def test_structured_json_non_dict_rejected(self) -> None:
        llm = _make_llm_mock()
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)
        with (
            patch(
                "litellm.completion",
                return_value=_fake_completion("[1, 2, 3]"),
            ),
            pytest.raises(OracleError, match="non-dict"),
        ):
            oracle.invoke("Q?", schema=_SampleSchema)

    def test_structured_schema_validation_error_wrapped(self) -> None:
        llm = _make_llm_mock()
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)
        with (
            patch(
                "litellm.completion",
                return_value=_fake_completion('{"score": 0.5}'),
            ),
            pytest.raises(OracleError, match=r"schema .* validation"),
        ):
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
        cls = _resolve_schema("tests.test_fsm_llm_lam.test_oracle._SampleSchema")
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


class TestInvokeEnvBranch:
    """R3 step 12+13 — DECISION D-005: when ``env`` is supplied, the prompt
    is treated as a ``str.format`` template and substituted before the LLM
    call. When ``env`` is None (default), the prompt passes through
    unchanged. Both paths land at the same routing site (D-003)."""

    def test_env_none_passes_prompt_through_unchanged(self) -> None:
        llm = _make_llm_mock()
        llm.generate_response.return_value = ResponseGenerationResponse(message="ok")
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)

        out = oracle.invoke("plain prompt", env=None)

        assert out == "ok"
        # Verify the unstructured generate_response was called with the
        # un-substituted prompt by inspecting the Request that was built.
        called_request = llm.generate_response.call_args[0][0]
        assert "plain prompt" in str(called_request)

    def test_env_substitution_replaces_named_slots(self) -> None:
        llm = _make_llm_mock()
        llm.generate_response.return_value = ResponseGenerationResponse(message="ok")
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)

        out = oracle.invoke(
            "Hello {name}, your order #{order_id} is ready.",
            env={"name": "Alice", "order_id": "42"},
        )

        assert out == "ok"
        # The substituted prompt must reach generate_response.
        called_request = llm.generate_response.call_args[0][0]
        assert "Hello Alice, your order #42 is ready." in str(called_request)

    def test_missing_env_var_raises_oracle_error(self) -> None:
        llm = _make_llm_mock()
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)

        with pytest.raises(
            OracleError, match=r"oracle template substitution failed: missing env var"
        ):
            oracle.invoke("Hello {name}", env={"other": "value"})
        # No LLM call should have been made.
        assert llm.generate_response.call_count == 0

    def test_empty_env_dict_treated_as_template(self) -> None:
        """env={} is *not* the same as env=None. With env={}, the prompt is
        still passed through ``str.format`` — so a literal `{` in the prompt
        without a substitution will raise."""
        llm = _make_llm_mock()
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)

        with pytest.raises(OracleError, match="oracle template substitution failed"):
            oracle.invoke("a literal {brace}", env={})

    def test_env_with_empty_template_no_substitution_needed(self) -> None:
        """A template with no slots and env={} substitutes to itself."""
        llm = _make_llm_mock()
        llm.generate_response.return_value = ResponseGenerationResponse(message="ok")
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)

        out = oracle.invoke("no slots here", env={})

        assert out == "ok"

    def test_env_routes_to_structured_when_schema_supplied(self) -> None:
        """The env branch is orthogonal to schema routing — both compose."""
        llm = _make_llm_mock()
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)

        # Patch the structured invocation — we only care that env
        # substitution happens BEFORE schema routing, not the structured
        # path's internals (those are exercised in TestStructuredDispatch).
        with patch.object(
            oracle, "_invoke_structured", return_value={"answer": "x", "score": 0.5}
        ) as patched:
            out = oracle.invoke(
                "Q: {question}",
                schema=_SampleSchema,
                env={"question": "what?"},
            )

        assert out == {"answer": "x", "score": 0.5}
        # The substituted prompt must reach the structured invoker.
        called_prompt = patched.call_args[0][0]
        assert called_prompt == "Q: what?"

    def test_doubled_braces_in_template_render_as_single(self) -> None:
        """Producer-level invariant: `{{` in a template renders to `{` after
        format. Critical for the to_template_and_schema escape path."""
        llm = _make_llm_mock()
        llm.generate_response.return_value = ResponseGenerationResponse(message="ok")
        oracle = LiteLLMOracle(llm, context_window_tokens=10_000)

        out = oracle.invoke('schema: {{"k": "v"}}', env={})

        assert out == "ok"
        called_request = llm.generate_response.call_args[0][0]
        assert 'schema: {"k": "v"}' in str(called_request)
