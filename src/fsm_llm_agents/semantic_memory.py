from __future__ import annotations

"""
Long-term semantic memory for agents.

``SemanticMemoryStore`` is an embedding-backed fact store: text entries are
embedded once (via litellm, the same infra ``SemanticToolRegistry`` already
uses — no new dependencies) and recalled by cosine similarity rather than
substring match. Unlike :class:`fsm_llm.memory.WorkingMemory` (ephemeral,
substring search), it is JSON-persistable so memories survive process restarts
— the missing piece for "Claude-like" cross-session recall.

It is fully additive: nothing in the framework imports it by default. Wire it
into any agent via :func:`create_semantic_memory_tools`, which returns
``remember`` / ``recall`` tools backed by the store.

Example::

    from fsm_llm_agents import ReactAgent, AgentConfig, ToolRegistry
    from fsm_llm_agents.semantic_memory import (
        SemanticMemoryStore, create_semantic_memory_tools,
    )

    store = SemanticMemoryStore(persist_path="~/.agent_memory.json")
    registry = ToolRegistry()
    for t in create_semantic_memory_tools(store):
        registry.register(t)
    agent = ReactAgent(tools=registry, config=AgentConfig(model=model))
    agent.run("Remember that my favorite language is Python.")
    # ... later process ...
    store2 = SemanticMemoryStore.load("~/.agent_memory.json")
"""

import json
import os
import threading
from collections.abc import Callable
from typing import Annotated, Any

from fsm_llm.logging import logger

from .definitions import ToolDefinition
from .semantic_tools import _cosine_similarity
from .tools import _infer_schema_from_hints

EmbedFn = Callable[[str], list[float]]


class MemoryEntry:
    """A single stored memory: text + optional metadata + cached embedding."""

    __slots__ = ("embedding", "id", "metadata", "text")

    def __init__(
        self,
        id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        self.id = id
        self.text = text
        self.metadata = metadata or {}
        self.embedding = embedding

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryEntry:
        return cls(
            id=d["id"],
            text=d["text"],
            metadata=d.get("metadata") or {},
            embedding=d.get("embedding"),
        )


class SemanticMemoryStore:
    """Embedding-backed, persistable long-term memory.

    Args:
        embedding_model: litellm embedding model id. Any litellm-supported
            provider works (OpenAI, Ollama, Cohere, ...).
        persist_path: If set, :meth:`add` / :meth:`forget` auto-save to this
            JSON file. Also used as the default path for :meth:`save`.
        embed_fn: Optional override ``(text) -> list[float]`` used instead of
            litellm. Primarily for tests and custom embedding backends.
        max_entries: Optional cap on stored entries. When set, :meth:`add`
            evicts the oldest entries (FIFO by insertion order) after appending
            until ``len <= max_entries``. Default ``None`` → unbounded (prior
            behavior, byte-identical).
    """

    def __init__(
        self,
        embedding_model: str = "ollama/qwen3-embedding:0.6b",
        persist_path: str | None = None,
        embed_fn: EmbedFn | None = None,
        max_entries: int | None = None,
    ) -> None:
        self._embedding_model = embedding_model
        self._persist_path = os.path.expanduser(persist_path) if persist_path else None
        self._embed_fn = embed_fn
        self._max_entries = max_entries
        self._entries: list[MemoryEntry] = []
        self._counter = 0
        # Single non-reentrant lock guarding the _counter/_entries mutation +
        # persistence in add/forget/clear/save. It deliberately spans the
        # blocking embedding call inside add() (coarse but safe; documented
        # latency trade-off per plan A6). Non-reentrant, so methods holding it
        # MUST NOT re-acquire it — persistence runs via _save_locked() (the
        # lock-free write body) while the lock is already held.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def _embed(self, text: str) -> list[float] | None:
        try:
            if self._embed_fn is not None:
                return list(self._embed_fn(text))
            import litellm

            response = litellm.embedding(model=self._embedding_model, input=[text])
            return list(response.data[0]["embedding"])
        except Exception as e:  # provider down / unsupported → graceful degrade
            logger.warning(f"Memory embedding failed (will store unembedded): {e}")
            return None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        entry_id: str | None = None,
    ) -> str:
        """Embed and store a memory entry. Returns its id."""
        if not text or not text.strip():
            raise ValueError("memory text must be non-empty")
        with self._lock:
            self._counter += 1
            eid = entry_id or f"mem-{self._counter}"
            entry = MemoryEntry(
                id=eid,
                text=text.strip(),
                metadata=metadata,
                embedding=self._embed(text),
            )
            self._entries.append(entry)
            # DECISION plan_2026-05-31_f08da86d/D-003: max_entries is a deliberate
            # single-use config knob (charged to the Complexity Budget), NOT a new
            # eviction-policy abstraction. Do NOT generalize this into a pluggable
            # LRU/TTL strategy — default None keeps growth unbounded (prior behavior
            # byte-identical); only FIFO-by-insertion eviction is in scope.
            if self._max_entries is not None:
                while len(self._entries) > self._max_entries:
                    self._entries.pop(0)
            self._maybe_persist_locked()
        return eid

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Return up to ``k`` most relevant memories as (text, score, metadata).

        Uses cosine similarity over embeddings. Falls back to case-insensitive
        substring matching when the query cannot be embedded or no entry has an
        embedding (so recall still works offline / when the provider is down).
        """
        if not self._entries:
            return []

        query_emb = self._embed(query)
        embedded = [e for e in self._entries if e.embedding is not None]

        if query_emb is None or not embedded:
            return self._substring_search(query, k)

        scored = [
            (e, _cosine_similarity(query_emb, e.embedding))  # type: ignore[arg-type]
            for e in embedded
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(e.text, score, e.metadata) for e, score in scored[:k]]

    def _substring_search(
        self, query: str, k: int
    ) -> list[tuple[str, float, dict[str, Any]]]:
        q = query.lower().strip()
        hits = [(e.text, 1.0, e.metadata) for e in self._entries if q in e.text.lower()]
        return hits[:k]

    def forget(self, entry_id: str) -> bool:
        """Remove an entry by id. Returns True if removed."""
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.id != entry_id]
            removed = len(self._entries) < before
            if removed:
                self._maybe_persist_locked()
        return removed

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._maybe_persist_locked()

    def all_entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_model": self._embedding_model,
            "counter": self._counter,
            "entries": [e.to_dict() for e in self._entries],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        persist_path: str | None = None,
        embed_fn: EmbedFn | None = None,
        embedding_model: str | None = None,
    ) -> SemanticMemoryStore:
        """Restore a store from a dict.

        ``embedding_model``: optional ACTIVE model to use for future query
        embeddings. When provided and it differs from the stored model, a
        warning is emitted (the cached vectors came from the stored model and
        are incompatible with a different model's query vectors). When omitted,
        the stored model is reused (prior behavior, no warning).
        """
        stored_model = data.get("embedding_model")
        active_model = (
            embedding_model
            if embedding_model is not None
            else (stored_model or "ollama/qwen3-embedding:0.6b")
        )
        store = cls(
            embedding_model=active_model,
            persist_path=persist_path,
            embed_fn=embed_fn,
        )
        store._counter = int(data.get("counter", 0))
        store._entries = [MemoryEntry.from_dict(d) for d in data.get("entries", [])]
        # Embedding-mismatch guard: stored vectors were produced by the saved
        # model; comparing them against a different active model's query vectors
        # mixes incompatible vector spaces. Warn (no behavior change on match).
        if stored_model and stored_model != active_model:
            logger.warning(
                "SemanticMemoryStore embedding model mismatch: stored "
                f"'{stored_model}' != active '{active_model}'; "
                "cached vectors may be incompatible with new query embeddings."
            )
        return store

    def save(self, path: str | None = None) -> None:
        """Persist to JSON. Uses ``persist_path`` when ``path`` is omitted."""
        with self._lock:
            self._save_locked(path)

    def _save_locked(self, path: str | None = None) -> None:
        """Lock-free write body. Callers MUST already hold ``self._lock``
        (non-reentrant): used by :meth:`save` and the locked CRUD paths via
        :meth:`_maybe_persist_locked`."""
        target = os.path.expanduser(path) if path else self._persist_path
        if not target:
            raise ValueError("no path provided and persist_path is unset")
        tmp = f"{target}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        os.replace(tmp, target)

    @classmethod
    def load(
        cls,
        path: str,
        embed_fn: EmbedFn | None = None,
    ) -> SemanticMemoryStore:
        """Load a store from a JSON file (re-uses cached embeddings)."""
        expanded = os.path.expanduser(path)
        with open(expanded, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data, persist_path=expanded, embed_fn=embed_fn)

    def _maybe_persist_locked(self) -> None:
        """Auto-persist from a locked CRUD path. Caller holds ``self._lock``;
        uses :meth:`_save_locked` so the non-reentrant lock is not re-acquired."""
        if self._persist_path:
            try:
                self._save_locked()
            except Exception as e:
                logger.warning(f"Memory auto-persist failed: {e}")


def create_semantic_memory_tools(
    store: SemanticMemoryStore,
    top_k: int = 5,
) -> list[ToolDefinition]:
    """Create ``remember`` / ``recall`` tools bound to a SemanticMemoryStore.

    Drop-in analogue of :func:`fsm_llm_agents.memory_tools.create_memory_tools`,
    but backed by embedding-based recall and optional cross-session persistence.

    Returns:
        Two ``ToolDefinition`` objects: ``remember``, ``recall``.
    """

    def remember(
        content: Annotated[str, "The fact or information to store in long-term memory"],
    ) -> str:
        """Store a fact in long-term semantic memory for later recall."""
        eid = store.add(content)
        return f"Remembered (id={eid})."

    def recall(
        query: Annotated[str, "What to search long-term memory for"],
    ) -> str:
        """Recall relevant facts from long-term semantic memory by meaning."""
        results = store.search(query, k=top_k)
        if not results:
            return f"No memories found relevant to '{query}'."
        lines = [f"Found {len(results)} relevant memories:"]
        for text, score, _meta in results:
            lines.append(f"  ({score:.2f}) {text}")
        return "\n".join(lines)

    tools: list[ToolDefinition] = []
    for fn in (remember, recall):
        tools.append(
            ToolDefinition(
                name=fn.__name__,
                description=(fn.__doc__ or "").strip().split("\n")[0],
                parameter_schema=_infer_schema_from_hints(fn),
                execute_fn=fn,
            )
        )
    return tools
