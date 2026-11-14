"""Text helpers for model replies: the reasoning-trace field, think blocks and code fences.

Layer ``llm`` (plan d454fd62, step 72): split out of the old ``fsm_llm.utilities``.
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------
# Reasoning-trace field resolution
# --------------------------------------------------------------


def resolve_reasoning_trace(message: Any) -> str | None:
    """Return the model's reasoning/thinking trace, whatever field carries it.

    Some models (e.g. Qwen 3.5 via Ollama) leave ``content`` empty/``None`` and
    put the actual answer in a reasoning field. litellm's field name for that
    trace is version-dependent, so callers MUST NOT read a single hard-coded
    attribute; resolve it here, in one place, for every content reader.

    Contract:
        - Parameter: any litellm ``Message``/``Delta`` (or duck-typed stand-in).
        - Returns the first non-empty of ``reasoning_content``, legacy
          ``thinking``, then a newline-joined string of any ``thinking_blocks``
          (each block's ``thinking`` or ``text``); ``None`` when no trace exists.
        - Never raises for any input (uses ``getattr(..., None)``, never
          ``hasattr``/attribute access that could fail on a stripped object).
    """
    # DECISION plan-2026-07-21T072826-e3131cc2/D-002
    # Do NOT revert this to `hasattr(message, "thinking")` or a `.thinking`-only
    # read. The installed litellm range RENAMES the raw `thinking` field to
    # `reasoning_content` and DELETES `thinking` before building the
    # Message/Delta object, so a `.thinking`-only read is DEAD CODE for the
    # project's former default model (ollama_chat/qwen3.5:4b, measured).
    # `reasoning_content` is read FIRST; the legacy `thinking` string is kept so the D-023 divergence
    # tests stay green; `thinking_blocks` is a last-resort join of the provider's
    # typed reasoning segments. This is the SINGLE resolver shared by
    # llm/providers/litellm.py::_extract_content_from_thinking and llm/interface.py::_completion_response
    # (the reply reader of `complete`, which the classifier uses) so the
    # readers can never re-diverge (the NL1 bug was engine/classification.py holding
    # its own stale copy).
    # Mirrors the original C2 anchor plan-2026-07-21T045419-9925aa3a/D-002.
    # See decisions.md D-002.
    trace = getattr(message, "reasoning_content", None) or getattr(
        message, "thinking", None
    )
    if not trace:
        blocks = getattr(message, "thinking_blocks", None)
        if blocks:
            trace = "\n".join(
                (b.get("thinking") or b.get("text") or "")
                for b in blocks
                if isinstance(b, dict)
            )
    if not trace:
        return None
    return trace


# --------------------------------------------------------------
# Reasoning-tag / code-fence stripping
# --------------------------------------------------------------


def remove_think_blocks(content: str) -> str:
    """Remove every ``<think>...</think>`` span in one linear pass.

    Contract: takes any ``str``; returns it with each lazily-matched,
    non-overlapping ``<think>...</think>`` span (newlines included) removed.
    An opening tag with no closing tag after it is left in place, as are all
    later opening tags (none of them can have a closer either). Never raises.

    # DECISION plan-2026-09-19T175721-21cd7f8e/D-010
    # Do NOT restore ``re.sub(r"<think>.*?</think>", "", ..., re.DOTALL)``:
    # on ``"<think>a" * n`` every opening tag scans to the end of the string
    # looking for a closer that does not exist, so it is O(n^2) (50,000 tags
    # took minutes) on text that arrives straight from the provider while the
    # caller holds the per-conversation lock. Once ONE opener has no closer no
    # later opener can have one, so the scan stops there. See decisions.md D-010.
    """
    open_tag, close_tag = "<think>", "</think>"
    parts: list[str] = []
    pos = 0
    while True:
        start = content.find(open_tag, pos)
        if start == -1:
            break
        end = content.find(close_tag, start + len(open_tag))
        if end == -1:
            break
        parts.append(content[pos:start])
        pos = end + len(close_tag)
    parts.append(content[pos:])
    return "".join(parts)


def strip_think_and_fences(content: str) -> str:
    """Strip a ``<think>...</think>`` block, then any surrounding code fence.

    Some models (e.g. Qwen via Ollama) wrap a JSON reply in a ``<think>``
    trace and/or a markdown code fence; callers that then ``json.loads`` the
    result need both stripped first.

    Contract:
        - Parameter: ``content`` — the raw message content string (already
          confirmed to be a ``str`` by the caller; this function does not
          itself branch on type).
        - Returns the content with any ``<think>...</think>`` span removed
          (``re.DOTALL``, so it matches across newlines) and then any leading
          ` ```json `/``` ``` `` fence markers stripped, in that exact order.
        - Never raises for any ``str`` input.
        - Shared by ``llm.py::_parse_field_extraction_response`` and
          ``LiteLLMInterface.extract_bulk_data`` so the two content readers
          can never re-diverge — the three regex operations were previously
          byte-identical, hand-duplicated in both places.
    """
    content = remove_think_blocks(content).strip()
    # DECISION plan-2026-09-19T175721-21cd7f8e/D-030
    # No re.MULTILINE: only a fence that STARTS the reply is a wrapper. With
    # `^` matching at every line, an inline code fence in prose
    # ('Use:\n```python\nprint(1)\n```') lost its markers. Do NOT re-add it (LS-09).
    # DECISION plan-2026-09-19T175721-21cd7f8e/D-048
    # The CLOSING fence is stripped only after a leading fence was stripped: a
    # reply that does not start with a fence is not a fenced wrapper, so a
    # trailing fence is content ('Here:\n```py\nx\n```' keeps both). Do NOT
    # strip it unconditionally, and do NOT count fences to decide (a heuristic
    # where the start-of-reply test is structural). Trade-off: a stray closing
    # fence alone ('{"a": 1}\n```') is kept here; extract_json_from_text recovers.
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", content)
    if stripped != content:
        stripped = re.sub(r"\n?```\s*$", "", stripped)
    return stripped.strip()
