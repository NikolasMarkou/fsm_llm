"""Helpers shared by the test files of this suite. Not collected (no test_ prefix).

Interface contract: pure builders of FSM definition dicts for `API.from_definition`.
`_state(state_id, *, say, to=None, **extra)` is one state (`say=None` leaves
`response_instructions` out, `say=""` makes a silent state, `to` adds one transition at
priority 100, `extra` overrides any field); `_chain()` is a -> b (silent) -> c (terminal), so
no user message is needed; `_stuck()` is a state that waits for a key nobody extracts, so
every turn stays in `a`.
"""

from __future__ import annotations

from typing import Any


def _state(
    state_id: str, *, say: str | None, to: str | None = None, **extra: Any
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "id": state_id,
        "description": state_id,
        "purpose": f"be {state_id}",
    }
    if say is not None:
        state["response_instructions"] = say
    if to is not None:
        state["transitions"] = [
            {"target_state": to, "description": f"go to {to}", "priority": 100}
        ]
    state.update(extra)
    return state


def _chain() -> dict[str, Any]:
    """a -> b (silent) -> c (terminal): no user message needed."""
    return {
        "name": "Chain",
        "description": "three states in a row",
        "initial_state": "a",
        "persona": "a test persona",
        "states": {
            "a": _state("a", say="Say A", to="b"),
            "b": _state("b", say="", to="c"),
            "c": _state("c", say="Say done"),
        },
    }


def _stuck() -> dict[str, Any]:
    """`a` waits for a key nobody extracts, so every turn stays in `a`."""
    gate = {
        "target_state": "b",
        "description": "when go is set",
        "priority": 100,
        "conditions": [
            {
                "description": "go is set",
                "requires_context_keys": ["go"],
                "logic": {"==": [{"var": "go"}, True]},
            }
        ],
    }
    return {
        "name": "Stuck",
        "description": "never leaves a",
        "initial_state": "a",
        "persona": "a test persona",
        "states": {
            "a": _state(
                "a",
                say="Still here.",
                extraction_instructions="Extract go.",
                required_context_keys=["go"],
                transitions=[gate],
            ),
            "b": _state("b", say="Left."),
        },
    }
