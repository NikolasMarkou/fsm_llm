from __future__ import annotations

"""
Observation summarization for long agent loops.

Agents accumulate observations turn over turn; the stock loop only hard-prunes
to the most recent ``MAX_OBSERVATIONS`` (dropping older context entirely). When
``AgentConfig.auto_summarize_after`` is set, agents instead *condense* the
oldest observations into a single summary entry once the list grows past the
threshold — retaining the gist of earlier steps instead of discarding them.

The summarizer is registered automatically by ``BaseAgent`` when the config
field is set (see ``_register_lifecycle_handlers``); this module just builds the
handler. Default summarization is deterministic (structure-preserving
truncation, no LLM call) so it is cheap and testable; pass a custom
``summarize_fn`` for an LLM-backed summary.
"""

from collections.abc import Callable
from typing import Any

from .constants import ContextKeys, Defaults
from .truncation import smart_truncate

SummarizeFn = Callable[[list[str]], str]


def _default_summary(entries: list[str]) -> str:
    """Deterministic condensation: join + structure-aware truncate."""
    joined = " | ".join(entries)
    return smart_truncate(joined, Defaults.MAX_OBSERVATION_LENGTH)


def make_observation_summarizer(
    threshold: int,
    keep_last: int | None = None,
    summarize_fn: SummarizeFn | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a PRE_PROCESSING handler that condenses old observations.

    When ``len(observations) > threshold``, all but the most recent
    ``keep_last`` entries are replaced by one summary entry.

    Args:
        threshold: Observation count above which summarization triggers.
        keep_last: Recent entries to keep verbatim (default ``max(1,
            threshold // 2)``).
        summarize_fn: Optional ``(entries) -> str`` summarizer (e.g. LLM-backed).
            Defaults to deterministic truncation.
    """
    if threshold < 1:
        raise ValueError("threshold must be >= 1")
    keep = keep_last if keep_last is not None else max(1, threshold // 2)
    summarize = summarize_fn or _default_summary

    def summarizer(context: dict[str, Any]) -> dict[str, Any]:
        obs = context.get(ContextKeys.OBSERVATIONS)
        if not isinstance(obs, list) or len(obs) <= threshold:
            return {}
        head = obs[:-keep] if keep else obs
        tail = obs[-keep:] if keep else []
        if not head:
            return {}
        summary = summarize(head)
        new_obs = [f"[Summary of {len(head)} earlier steps] {summary}", *tail]
        return {
            ContextKeys.OBSERVATIONS: new_obs,
            ContextKeys.OBSERVATION_COUNT: len(new_obs),
        }

    return summarizer
