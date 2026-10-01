"""``LiteLLMEmbedder`` (plan 944e2692, step 4, D-005).

The embedder owns the one ``embedding`` binding of fsm_llm. The provider is
the ``fsm_llm.llm.embedding`` binding, patched; replies are real litellm
``EmbeddingResponse`` objects unless a test needs another shape.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litellm.types.utils import EmbeddingResponse, Usage

import fsm_llm
from fsm_llm.constants import RESERVED_EMBEDDING_CALL_KWARGS, USAGE_KIND_EMBED
from fsm_llm.definitions import LLMResponseError, LLMUsage
from fsm_llm.llm import LiteLLMEmbedder, LiteLLMInterface
from fsm_llm.logging import logger

_MODEL = "ollama/qwen3-embedding:0.6b"


def _vectors_reply(
    vectors: list[list[float]],
    *,
    prompt_tokens: int | None = 6,
    indexes: list[int] | None = None,
) -> EmbeddingResponse:
    order = indexes if indexes is not None else list(range(len(vectors)))
    kwargs: dict[str, Any] = {}
    if prompt_tokens is not None:
        kwargs["usage"] = Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens)
    return EmbeddingResponse(
        model=_MODEL,
        data=[
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in zip(order, vectors, strict=True)
        ],
        **kwargs,
    )


def _patched(provider: Any):
    return patch("fsm_llm.llm.embedding", provider)


def _echo_provider(dims: int = 3) -> MagicMock:
    """A provider that answers one distinct vector per input text."""

    def answer(**params: Any) -> EmbeddingResponse:
        texts = params["input"]
        return _vectors_reply(
            [[float(i)] * dims for i in range(len(texts))],
            prompt_tokens=2 * len(texts),
        )

    return MagicMock(side_effect=answer)


class TestEmbedRequest:
    def test_batch_is_one_request_with_one_vector_per_text_in_order(self):
        provider = _echo_provider()
        embedder = LiteLLMEmbedder(_MODEL)
        with _patched(provider):
            vectors = embedder.embed(["alpha", "beta", "gamma"])

        assert provider.call_count == 1
        assert provider.call_args.kwargs["input"] == ["alpha", "beta", "gamma"]
        assert vectors == [[0.0] * 3, [1.0] * 3, [2.0] * 3]

    def test_request_has_connection_kwargs_and_no_chat_params(self):
        provider = _echo_provider()
        embedder = LiteLLMEmbedder(
            _MODEL,
            api_key="sk-embed",
            timeout=30.0,
            retries=3,
            api_base="http://embed-host:11434",
            dimensions=256,
        )
        with _patched(provider):
            embedder.embed(["one"])

        assert provider.call_args.kwargs == {
            "model": _MODEL,
            "input": ["one"],
            "timeout": 30.0,
            "max_retries": 3,
            "api_key": "sk-embed",
            "api_base": "http://embed-host:11434",
            "dimensions": 256,
        }

    def test_defaults_send_timeout_and_leave_sdk_retries_alone(self):
        provider = _echo_provider()
        with _patched(provider):
            LiteLLMEmbedder(_MODEL).embed(["one"])
        params = provider.call_args.kwargs
        assert params["timeout"] == 120.0
        assert "max_retries" not in params
        assert "api_key" not in params

    def test_no_timeout_omits_the_key(self):
        provider = _echo_provider()
        with _patched(provider):
            LiteLLMEmbedder(_MODEL, timeout=None).embed(["one"])
        assert "timeout" not in provider.call_args.kwargs

    def test_connection_kwargs_match_the_chat_builder(self):
        """One connection builder: the chat request carries the same part."""
        conn = {"api_base": "http://h:1", "api_key": "k"}
        chat = LiteLLMInterface(
            "ollama_chat/qwen3.5:4b", timeout=9.0, retries=2, **conn
        )
        chat_params = chat._build_call_params(
            [{"role": "user", "content": "hi"}], "response_generation"
        )
        provider = _echo_provider()
        with _patched(provider):
            LiteLLMEmbedder(_MODEL, timeout=9.0, retries=2, **conn).embed(["x"])
        embed_params = provider.call_args.kwargs
        for key in ("api_base", "api_key", "timeout", "max_retries"):
            assert embed_params[key] == chat_params[key]

    def test_no_texts_is_no_request(self):
        provider = _echo_provider()
        embedder = LiteLLMEmbedder(_MODEL)
        with _patched(provider):
            assert embedder.embed([]) == []
            assert embedder.embed(()) == []
        assert provider.call_count == 0
        assert embedder.usage().calls == 0

    def test_accepts_any_sequence_of_strings(self):
        provider = _echo_provider()
        with _patched(provider):
            vectors = LiteLLMEmbedder(_MODEL).embed(("a", "b"))
        assert provider.call_args.kwargs["input"] == ["a", "b"]
        assert len(vectors) == 2

    @pytest.mark.parametrize("texts", ["a single string", ["ok", 3], [None]])
    def test_non_string_input_is_refused_before_any_request(self, texts: Any):
        provider = _echo_provider()
        with _patched(provider), pytest.raises(TypeError):
            LiteLLMEmbedder(_MODEL).embed(texts)
        assert provider.call_count == 0


class TestConstruction:
    @pytest.mark.parametrize("model", ["", "   "])
    def test_empty_model_is_refused(self, model: str):
        with pytest.raises(ValueError, match="model"):
            LiteLLMEmbedder(model)

    def test_reserved_kwargs_are_dropped_with_a_warning(self):
        records: list[Any] = []
        sink_id = logger.add(
            lambda message: records.append(message.record), level="WARNING"
        )
        logger.enable("fsm_llm")
        try:
            embedder = LiteLLMEmbedder(
                _MODEL, temperature=0.2, input=["smuggled"], response_format={}
            )
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")

        warnings = [r["message"] for r in records]
        assert len(warnings) == 1
        assert "['input', 'response_format', 'temperature']" in warnings[0]

        provider = _echo_provider()
        with _patched(provider):
            embedder.embed(["real"])
        params = provider.call_args.kwargs
        assert params["input"] == ["real"]
        assert not {"temperature", "response_format", "messages"} & params.keys()

    def test_reserved_set_covers_the_chat_params_and_input(self):
        assert {"input", "messages", "temperature", "max_tokens"} <= (
            RESERVED_EMBEDDING_CALL_KWARGS
        )

    def test_exported_from_the_package(self):
        assert fsm_llm.LiteLLMEmbedder is LiteLLMEmbedder
        assert "LiteLLMEmbedder" in fsm_llm.__all__


class TestReplyShapes:
    def test_vectors_follow_the_reply_index_not_the_list_order(self):
        reply = _vectors_reply([[2.0], [0.0], [1.0]], indexes=[2, 0, 1])
        with _patched(MagicMock(return_value=reply)):
            vectors = LiteLLMEmbedder(_MODEL).embed(["a", "b", "c"])
        assert vectors == [[0.0], [1.0], [2.0]]

    def test_object_items_and_int_components_are_read(self):
        reply = SimpleNamespace(
            data=[
                SimpleNamespace(index=0, embedding=[1, 2]),
                SimpleNamespace(index=1, embedding=[0.5, -1.5]),
            ],
            usage=None,
        )
        with _patched(MagicMock(return_value=reply)):
            vectors = LiteLLMEmbedder(_MODEL).embed(["a", "b"])
        assert vectors == [[1.0, 2.0], [0.5, -1.5]]
        assert all(isinstance(x, float) for v in vectors for x in v)

    @pytest.mark.parametrize(
        "reply",
        [
            _vectors_reply([[1.0]]),  # one vector for two texts
            _vectors_reply([[1.0], [2.0], [3.0]]),  # three for two
            SimpleNamespace(data=None),
            SimpleNamespace(data=[{"index": 0, "embedding": "nope"}, {"index": 1}]),
            SimpleNamespace(
                data=[{"index": 0, "embedding": [True]}, {"index": 1, "embedding": [1]}]
            ),
            _vectors_reply([[1.0], [2.0]], indexes=[0, 0]),
        ],
        ids=["too-few", "too-many", "no-data", "not-a-vector", "bool", "dup-index"],
    )
    def test_malformed_reply_is_an_llm_response_error(self, reply: Any):
        embedder = LiteLLMEmbedder(_MODEL)
        with _patched(MagicMock(return_value=reply)), pytest.raises(LLMResponseError):
            embedder.embed(["a", "b"])
        # The request was answered: counted as a call, not a provider error.
        assert embedder.usage().calls == 1
        assert embedder.usage().errors == 0


class TestEmbedderUsage:
    def test_answered_request_counts_tokens_under_embed(self):
        embedder = LiteLLMEmbedder(_MODEL)
        with _patched(_echo_provider()):
            embedder.embed(["a", "b", "c"])
            embedder.embed(["d"])

        usage = embedder.usage()
        assert isinstance(usage, LLMUsage)
        assert set(usage.by_kind) == {USAGE_KIND_EMBED}
        assert usage.calls == 2
        assert usage.prompt_tokens == 8
        assert usage.total_tokens == 8
        assert usage.completion_tokens == 0
        assert usage.usage_missing == 0

    def test_reply_without_usage_counts_as_usage_missing(self):
        embedder = LiteLLMEmbedder(_MODEL)
        reply = _vectors_reply([[1.0]], prompt_tokens=None)
        reply.usage = None  # litellm fills a default Usage; force the absent shape
        with _patched(MagicMock(return_value=reply)):
            embedder.embed(["a"])
        assert embedder.usage().usage_missing == 1

    def test_provider_error_propagates_unchanged_and_counts_as_error(self):
        boom = ConnectionError("embedding server down")
        embedder = LiteLLMEmbedder(_MODEL)
        with (
            _patched(MagicMock(side_effect=boom)),
            pytest.raises(ConnectionError) as info,
        ):
            embedder.embed(["a"])
        assert info.value is boom
        usage = embedder.usage()
        assert (usage.calls, usage.errors) == (1, 1)
        assert usage.by_kind[USAGE_KIND_EMBED].errors == 1

    def test_reset_usage_returns_the_held_counts_and_clears(self):
        embedder = LiteLLMEmbedder(_MODEL)
        with _patched(_echo_provider()):
            embedder.embed(["a"])
        held = embedder.reset_usage()
        assert held.calls == 1
        assert embedder.usage().calls == 0

    def test_meters_are_per_instance_and_separate_from_the_chat_interface(self):
        first = LiteLLMEmbedder(_MODEL)
        second = LiteLLMEmbedder(_MODEL)
        chat = LiteLLMInterface("ollama_chat/qwen3.5:4b")
        with _patched(_echo_provider()):
            first.embed(["a"])
        assert first.usage().calls == 1
        assert second.usage().calls == 0
        assert chat.usage().calls == 0

    def test_instance_built_without_init_still_counts(self):
        embedder = LiteLLMEmbedder.__new__(LiteLLMEmbedder)
        embedder.model = _MODEL
        embedder.timeout = None
        embedder.retries = 0
        embedder.kwargs = {}
        with _patched(_echo_provider()):
            embedder.embed(["a"])
        assert embedder.usage().calls == 1

    def test_embedding_never_touches_the_completion_binding(self):
        completion = MagicMock()
        with _patched(_echo_provider()), patch("fsm_llm.llm.completion", completion):
            LiteLLMEmbedder(_MODEL).embed(["a"])
        completion.assert_not_called()
