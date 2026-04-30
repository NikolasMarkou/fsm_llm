"""Math skills loaded from directory by SkillLoader."""

from fsm_llm.stdlib.agents import tool


@tool
def add(a: float, b: float) -> str:
    """Add two numbers together."""
    return f"{a} + {b} = {a + b}"


@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers together."""
    return f"{a} * {b} = {a * b}"


@tool
def percentage(value: float, total: float) -> str:
    """Calculate what percentage value is of total."""
    if total == 0:
        return "Cannot divide by zero"
    pct = (value / total) * 100
    return f"{value} is {pct:.1f}% of {total}"
