"""Text processing skills loaded from directory by SkillLoader."""

from fsm_llm_agents import tool


@tool
def word_count(text: str) -> str:
    """Count the number of words in a text."""
    count = len(text.split())
    return f"Word count: {count}"


@tool
def reverse_text(text: str) -> str:
    """Reverse a string of text."""
    return f"Reversed: {text[::-1]}"


@tool
def char_frequency(text: str) -> str:
    """Count frequency of each character in text."""
    from collections import Counter

    freq = Counter(text.lower().replace(" ", ""))
    top5 = freq.most_common(5)
    result = ", ".join(f"'{c}': {n}" for c, n in top5)
    return f"Top characters: {result}"
