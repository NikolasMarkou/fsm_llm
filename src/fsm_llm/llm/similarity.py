"""Embedding vector helpers of the LLM layer: ``EmbedTextsFn`` and ``cosine_similarity``.

Layer ``llm`` (plan c720601d, step 1b, P7.4a): moved from ``fsm_llm.agents.semantic_tools`` so a
consumer of embeddings below the pattern runtimes (``synthesis.library``, ``synthesis.synthesizer``)
no longer imports the ``agents`` package. Roadmap row 3.2 moves them again to a retrieval layer.
Pure Python, no dependency, no model call.

Interface contract:

- ``EmbedTextsFn``: ``texts -> vectors``, one vector per text, in order (the type of the batch
  embedding function; ``LiteLLMEmbedder.embed`` has this shape). A callable of this type raises
  whatever its backend raises; the caller owns the degrade path.
- ``cosine_similarity(a, b) -> float``: the cosine of two equal-length vectors, ``0.0`` when either
  has zero norm; vectors of different length score ``0.0`` and log one WARNING (never silently
  truncated); never raises on numeric input.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from fsm_llm.logging import logger

EmbedTextsFn = Callable[[Sequence[str]], list[list[float]]]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Vectors of different length (e.g. cached under another embedding model)
    are not comparable: they score ``0.0`` and log a WARNING instead of being
    silently truncated to the shorter length.
    """
    if len(a) != len(b):
        logger.warning(
            f"Embedding dimension mismatch ({len(a)} vs {len(b)}); "
            "scoring 0.0. Were the vectors made by different embedding models?"
        )
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
