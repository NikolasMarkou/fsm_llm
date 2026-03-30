from __future__ import annotations

"""
Structure-aware text truncation for tool results.

Preserves head and tail content with a clear truncation marker in the
middle, so the LLM sees both the initial context and the final
conclusions/errors from tool output.
"""


def smart_truncate(text: str, max_length: int = 2000) -> str:
    """Truncate text while preserving structure and tail content.

    For text exceeding *max_length*, keeps the first ~60% and last ~40%
    of the content (by lines), joined by a truncation marker showing how
    many characters were removed.

    For JSON text (starts with ``{`` or ``[``), attempts to truncate at
    line boundaries so that partial key-value pairs are avoided.

    Args:
        text: The text to truncate.
        max_length: Maximum allowed length in characters.

    Returns:
        The original text if within limits, or a head+tail truncation
        with a marker in the middle.
    """
    if len(text) <= max_length:
        return text

    # Marker template — measure actual length after formatting
    dropped = len(text) - max_length
    marker = f"\n...[truncated {dropped} chars]...\n"
    available = max_length - len(marker)

    if available < 40:
        # Pathological: max_length is tiny, fall back to head-only
        return text[:max_length]

    # Split budget: 60% head, 40% tail
    head_budget = int(available * 0.6)
    tail_budget = available - head_budget

    lines = text.split("\n")

    # If single line (or very few), do character-level split
    if len(lines) <= 2:
        return text[:head_budget] + marker + text[-tail_budget:]

    # Line-aware split: accumulate lines from head and tail
    head_lines: list[str] = []
    head_chars = 0
    for line in lines:
        cost = len(line) + 1  # +1 for the newline
        if head_chars + cost > head_budget:
            break
        head_lines.append(line)
        head_chars += cost

    tail_lines: list[str] = []
    tail_chars = 0
    for line in reversed(lines):
        cost = len(line) + 1
        if tail_chars + cost > tail_budget:
            break
        tail_lines.append(line)
        tail_chars += cost
    tail_lines.reverse()

    # Guard: if we got nothing from either end, fall back to char split
    if not head_lines or not tail_lines:
        return text[:head_budget] + marker + text[-tail_budget:]

    # Check for overlap (text is mostly short lines that all fit)
    head_end_idx = len(head_lines)
    tail_start_idx = len(lines) - len(tail_lines)
    if head_end_idx >= tail_start_idx:
        # Head and tail overlap — the whole text fits in lines, just
        # truncate at char level as fallback
        return text[:head_budget] + marker + text[-tail_budget:]

    head_text = "\n".join(head_lines)
    tail_text = "\n".join(tail_lines)

    return head_text + marker + tail_text
