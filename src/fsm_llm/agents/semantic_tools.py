"""
Semantic Tool Retrieval.

Extends ToolRegistry with embedding-based retrieval for scalable tool
selection. Embeds through core's :class:`fsm_llm.llm.LiteLLMEmbedder` (no
new dependencies) or a caller-supplied ``embed_fn``.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence

from fsm_llm.llm import LiteLLMEmbedder
from fsm_llm.logging import logger

from .definitions import ToolDefinition
from .tools import ToolRegistry

EmbedFn = Callable[[str], list[float]]
"""A custom embedding backend: one text in, its vector out."""

EmbedTextsFn = Callable[[Sequence[str]], list[list[float]]]


# DECISION plan-2026-10-01T093600-944e2692/D-005: the default backend is core's
# LiteLLMEmbedder; a custom backend plugs in as a callable ``embed_fn``. Do NOT
# import litellm or call an embedding API here, and do NOT add an Embedder ABC
# or per-class embedding code: the memory store and the tool registry share
# this one function. See D-005.
def _embedding_backend(embedding_model: str, embed_fn: EmbedFn | None) -> EmbedTextsFn:
    """The batch embedding function of a semantic store or registry.

    Shared by :class:`SemanticToolRegistry` and
    :class:`fsm_llm.agents.semantic_memory.SemanticMemoryStore`.

    Args:
        embedding_model: Model of the default :class:`LiteLLMEmbedder`, used
            when ``embed_fn`` is ``None``.
        embed_fn: Optional custom backend, called once per text.

    Returns:
        ``texts -> vectors``, one vector per text, in order. The default
        backend sends all texts in one provider request.

    Raises:
        ValueError: ``embed_fn`` is ``None`` and ``embedding_model`` is empty.
        The returned function raises whatever the backend raises; callers
        own the degrade path.
    """
    if embed_fn is None:
        return LiteLLMEmbedder(embedding_model).embed
    custom = embed_fn

    def embed_each(texts: Sequence[str]) -> list[list[float]]:
        return [list(custom(text)) for text in texts]

    return embed_each


def _tool_text(tool: ToolDefinition) -> str:
    """The text embedded for a tool."""
    return f"{tool.name}: {tool.description}"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
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


class SemanticToolRegistry(ToolRegistry):
    """ToolRegistry with embedding-based semantic retrieval.

    Embeds tool descriptions at registration time and retrieves the
    top-K most relevant tools for a given query using cosine similarity.
    Falls back to the full tool list for small registries (fewer than
    ``FALLBACK_THRESHOLD`` tools, currently 10) or when embedding fails.

    Embeds through a :class:`fsm_llm.llm.LiteLLMEmbedder` built from
    ``embedding_model``, so any LiteLLM embedding provider works (OpenAI,
    Ollama, Cohere, etc.), or through ``embed_fn`` when given (a custom
    backend; ``embedding_model`` is then only a label).

    Example::

        from fsm_llm.agents.semantic_tools import SemanticToolRegistry

        registry = SemanticToolRegistry(
            embedding_model="ollama/qwen3-embedding:0.6b",
        )
        registry.register_function(search_web, name="search", description="Search the web")
        registry.register_function(calc, name="calculator", description="Do math")

        # Retrieve top-3 tools relevant to query
        tools = registry.retrieve("what is 2+2?", top_k=3)
    """

    # Threshold below which we return all tools instead of embedding
    FALLBACK_THRESHOLD = 10

    def __init__(
        self,
        embedding_model: str = "ollama/qwen3-embedding:0.6b",
        top_k: int = 5,
        auto_embed: bool = True,
        *,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        super().__init__()
        self._embedding_model = embedding_model
        self._embed_texts = _embedding_backend(embedding_model, embed_fn)
        self._default_top_k = top_k
        self._auto_embed = auto_embed
        # DECISION plan-2026-07-20T040150-876e7164/D-005 [STALE]: `_embeddings` gets its OWN
        # non-reentrant lock, separate from the inherited `_tools_lock`, because the
        # two dicts are mutated at different times and holding one across the other
        # would create a lock-order dependency. Guard iteration as well as mutation:
        # `retrieve`'s `for ... in self._embeddings.items()` racing `register`'s write
        # is the 20/20 `RuntimeError: dictionary changed size during iteration`, and
        # dropping the lock from the iteration side alone would leave the defect open.
        # Do NOT convert either lock to `RLock` to resolve a deadlock — use the
        # snapshot-then-operate shape below (`WorkingMemory`'s shell/`_locked()` split,
        # D-007 of plan-2026-07-19T191147-4b664252). The lock is deliberately released
        # before `_get_embedding`, which makes a network LLM call.
        self._embeddings_lock = threading.Lock()
        self._embeddings: dict[str, list[float]] = {}

    def register(self, tool: ToolDefinition) -> SemanticToolRegistry:
        """Register a tool and optionally compute its embedding."""
        super().register(tool)
        if self._auto_embed:
            self._embed_tool(tool)
        return self

    def _embed_tool(self, tool: ToolDefinition) -> None:
        """Compute and cache embedding for a tool's description."""
        text = _tool_text(tool)
        try:
            # `_get_embedding` is a network call — it runs OUTSIDE the lock; only
            # the dict write below is guarded.
            embedding = self._get_embedding(text)
        except Exception as e:
            logger.warning(f"Failed to embed tool '{tool.name}': {e}")
            return
        with self._embeddings_lock:
            self._embeddings[tool.name] = embedding

    def _get_embedding(self, text: str) -> list[float]:
        """The embedding vector of one text (one backend request)."""
        return self._embed_texts([text])[0]

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[ToolDefinition]:
        """Retrieve the top-K most relevant tools for a query.

        Falls back to full tool list if:
        - Registry has fewer than FALLBACK_THRESHOLD tools
        - No embeddings are available
        - Embedding the query fails

        Args:
            query: The user query or task description.
            top_k: Number of tools to return. Defaults to self._default_top_k.

        Returns:
            List of ToolDefinitions, most relevant first.
        """
        k = top_k or self._default_top_k
        all_tools = self.list_tools()

        # Fallback for small registries
        if len(all_tools) < self.FALLBACK_THRESHOLD:
            return all_tools

        # Fallback if no embeddings
        with self._embeddings_lock:
            has_embeddings = bool(self._embeddings)
        if not has_embeddings:
            logger.debug("No tool embeddings available, returning all tools")
            return all_tools

        # Embed the query
        try:
            query_embedding = self._get_embedding(query)
        except Exception as e:
            logger.warning(f"Query embedding failed, returning all tools: {e}")
            return all_tools

        # Lazily (re-)embed any registered tool whose embedding is missing, e.g.
        # a transient failure at register time. Without this, a tool that failed
        # to embed while others succeeded is silently unreachable forever: it is
        # never scored below, and the all-tools fallback only fires when
        # _embeddings is COMPLETELY empty (AI3-004). Persistent failures are
        # logged by _embed_tool and simply remain unscored.
        with self._embeddings_lock:
            unembedded = [t for t in all_tools if t.name not in self._embeddings]
        for tool in unembedded:
            # Re-acquires the lock internally; must NOT be called while holding it.
            self._embed_tool(tool)

        # Snapshot under the lock, score outside it: `_cosine_similarity` is pure
        # CPU work over vectors that are never mutated in place, so it does not
        # need the lock, and holding it here would block every concurrent
        # `register()` for the duration of the scan.
        with self._embeddings_lock:
            embedding_snapshot = list(self._embeddings.items())

        # Score each tool
        scores: list[tuple[str, float]] = []
        for tool_name, tool_embedding in embedding_snapshot:
            score = _cosine_similarity(query_embedding, tool_embedding)
            scores.append((tool_name, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-K
        result = []
        for tool_name, _score in scores[:k]:
            try:
                result.append(self.get(tool_name))
            except Exception as e:
                logger.debug(f"Tool '{tool_name}' missing during retrieve: {e}")

        logger.debug(
            f"Semantic retrieval: query='{query[:50]}...', "
            f"top-{k} tools: {[t.name for t in result]}"
        )
        return result

    def rebuild_embeddings(self) -> int:
        """Rebuild all tool embeddings in one batch request.

        Returns the number of tools embedded: all of them, or 0 when the
        batch fails (logged at WARNING; ``retrieve`` then re-embeds lazily or
        falls back to the full list).
        """
        with self._embeddings_lock:
            self._embeddings.clear()
        tools = self.list_tools()
        try:
            # Network call: runs OUTSIDE the lock, like `_embed_tool`.
            vectors = self._embed_texts([_tool_text(tool) for tool in tools])
        except Exception as e:  # provider down / bad reply: degrade, logged
            logger.warning(f"Failed to embed {len(tools)} tools: {e}")
            return 0
        with self._embeddings_lock:
            for tool, vector in zip(tools, vectors, strict=True):
                self._embeddings[tool.name] = vector
        logger.info(f"Rebuilt embeddings for {len(tools)}/{len(self)} tools")
        return len(tools)

    def to_prompt_description(
        self, query: str | None = None, top_k: int | None = None
    ) -> str:
        """Generate prompt description, optionally filtered by semantic relevance.

        Args:
            query: If provided, only include semantically relevant tools.
            top_k: Number of tools to include when filtering.
        """
        if query is not None:
            tools = self.retrieve(query, top_k=top_k)
            if not tools:
                return "No tools available."
            lines = ["Available tools:"]
            for tool in tools:
                lines.append(f"- {tool.name}: {tool.description}")
            return "\n".join(lines)
        return super().to_prompt_description()

    @property
    def embedding_model(self) -> str:
        """Return the configured embedding model."""
        return self._embedding_model

    @property
    def embedded_tool_count(self) -> int:
        """Return the number of tools with cached embeddings."""
        with self._embeddings_lock:
            return len(self._embeddings)
