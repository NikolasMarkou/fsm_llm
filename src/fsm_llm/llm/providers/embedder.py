"""``LiteLLMEmbedder``: the one ``embedding`` call of fsm_llm.

Layer ``llm`` (plan d454fd62, step 73): split out of the old ``fsm_llm/llm/__init__.py`` (the old ``llm.py``); definitions unchanged.

The ``embedding`` binding comes from ``fsm_llm.llm.providers.litellm``, the one module that
imports litellm; a test that replaces it patches ``fsm_llm.llm.providers.embedder.embedding``,
the module whose code looks it up.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from fsm_llm.constants import RESERVED_EMBEDDING_CALL_KWARGS, USAGE_KIND_EMBED
from fsm_llm.core.definitions import LLMResponseError
from fsm_llm.llm.interface import DEFAULT_LLM_TIMEOUT_SECONDS
from fsm_llm.llm.usage import _meter_of, _Metered, _reply_field

from .litellm import embedding
from .request import _checked_model, _connection_params, _warn_reserved_kwargs


class LiteLLMEmbedder(_Metered):
    """Text embeddings through LiteLLM: the one ``embedding`` call of fsm_llm.

    An embedder has its own model and connection kwargs (``api_key``,
    ``api_base``, ...), separate from any chat interface, and its own usage
    counters (kind ``embed``). Ollama models (``ollama/...``) need nothing
    extra: litellm reads ``OLLAMA_API_BASE`` (default
    ``http://localhost:11434``) itself.

    Example::

        embedder = LiteLLMEmbedder("ollama/embeddinggemma-2:740m")
        vectors = embedder.embed(["first text", "second text"])
    """

    # DECISION plan-2026-10-01T093600-944e2692/D-005: one concrete embedder
    # owns the one ``embedding`` binding. Do NOT add an ``embed`` method to
    # ``LLMInterface``/``LiteLLMInterface`` (the embedding model differs from
    # the chat model and would inherit the chat model's credentials), do NOT
    # add an Embedder ABC (one implementation; consumers plug custom backends
    # in through a callable ``embed_fn``), and do NOT call
    # ``litellm.embedding`` anywhere else. See D-005.
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float | None = DEFAULT_LLM_TIMEOUT_SECONDS,
        retries: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model: Embedding model identifier (e.g. ``"ollama/embeddinggemma-2:740m"``,
                ``"text-embedding-3-small"``).
            api_key: Optional API key (the environment is used when omitted).
            timeout: Seconds per provider request (``None`` for no timeout).
            retries: The SDK's ``max_retries``; same semantics as
                ``LiteLLMInterface`` (0 leaves the SDK default; a no-op on Ollama).
            **kwargs: Additional litellm embedding parameters (``api_base``,
                ``dimensions``, ...). Keys in
                ``constants.RESERVED_EMBEDDING_CALL_KWARGS`` are ignored with a
                WARNING.

        Raises:
            ValueError: ``model`` is empty.
        """
        _checked_model(model)
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.kwargs: dict[str, Any] = dict(kwargs)
        _warn_reserved_kwargs("LiteLLMEmbedder", kwargs, RESERVED_EMBEDDING_CALL_KWARGS)
        if api_key:
            self.kwargs["api_key"] = api_key

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` in one provider request, one vector per text, in order.

        No texts means no request and ``[]``. Provider errors propagate
        unchanged (callers keep their own degrade paths); each request is
        counted once on this embedder's meter, a raising one as an error.

        Raises:
            TypeError: ``texts`` is a single string or holds a non-string.
            LLMResponseError: the reply does not hold one numeric vector per
                text.
        """
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings")
        # Materialised before the type check: checking an iterator consumed it
        # and the batch came out empty (no request, no error, review area 1).
        batch = list(texts)
        if not all(isinstance(t, str) for t in batch):
            raise TypeError("texts must be a sequence of strings")
        if not batch:
            return []
        params = _connection_params(
            self.model,
            self.kwargs,
            timeout=self.timeout,
            retries=self.retries,
            reserved=RESERVED_EMBEDDING_CALL_KWARGS,
        )
        params["input"] = batch
        meter = _meter_of(self)
        try:
            response = embedding(**params)
        except Exception:
            # Broad catch is intentional: count the failed request, re-raise
            # it unchanged for the caller's own error boundary.
            meter.record_error(USAGE_KIND_EMBED)
            raise
        meter.record(USAGE_KIND_EMBED, response)
        return _embedding_vectors(response, len(batch))


def _embedding_vectors(response: Any, expected: int) -> list[list[float]]:
    """The vectors of an embedding reply, ordered by their ``index``.

    ``data`` items may be dicts or objects. Raises ``LLMResponseError`` when
    the reply does not hold exactly ``expected`` numeric vectors, or when a
    vector is empty, holds a non-finite value or differs in length from the
    others (each would reach cosine scoring silently).
    """
    data = _reply_field(response, "data")
    if not isinstance(data, list) or len(data) != expected:
        got = len(data) if isinstance(data, list) else type(data).__name__
        raise LLMResponseError(
            f"Embedding reply holds {got} vectors for {expected} texts"
        )
    indexed: list[tuple[int, list[float]]] = []
    for position, item in enumerate(data):
        index = _reply_field(item, "index")
        vector = _reply_field(item, "embedding")
        if not isinstance(vector, list) or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in vector
        ):
            raise LLMResponseError(f"Embedding reply item {position} is not a vector")
        if not vector:
            raise LLMResponseError(f"Embedding reply item {position} is empty")
        try:
            finite = all(math.isfinite(x) for x in vector)
        except OverflowError:  # an integer past float range is no finite float
            finite = False
        if not finite:
            raise LLMResponseError(
                f"Embedding reply item {position} holds a non-finite value"
            )
        if indexed and len(vector) != len(indexed[0][1]):
            raise LLMResponseError(
                f"Embedding reply item {position} has {len(vector)} dimensions, "
                f"item 0 has {len(indexed[0][1])}"
            )
        order = (
            index
            if isinstance(index, int) and not isinstance(index, bool)
            else position
        )
        indexed.append((order, [float(x) for x in vector]))
    if sorted(order for order, _ in indexed) != list(range(expected)):
        raise LLMResponseError("Embedding reply indexes do not match the texts")
    return [vector for _, vector in sorted(indexed, key=lambda pair: pair[0])]
