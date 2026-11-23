"""The completion kind's hooks: the lint of its raw state, its result-key and declared-tool
readers and its state check.

A leaf core module beside ``kind_call`` (plan baa37f40, step 20.1, D-037): standard library only,
never logs, imports nothing from ``fsm_llm``. Moved text for text out of ``state_kinds.py``, which
imports the six names back and builds the ``COMPLETION`` spec from them; the contract of each hook
is the module docstring of ``state_kinds`` (``lint_raw``, ``raw_result_key``, ``declared_tools``,
``raw_declared_tools``, ``check_state``).
"""

from __future__ import annotations

from typing import Any


def _completion_lint_raw(
    state_id: str, state: dict[str, Any], fsm_data: dict[str, Any], services: Any
) -> list[str]:
    """A completion state whose transitions do not cover every result kind it
    can produce (``calls``, ``final``, ``malformed`` with tools; ``final``,
    ``malformed`` for a structured turn): no unconditional transition and a kind
    no condition names as a string literal. An uncovered kind stalls the state,
    and skip-if-set makes no further call, so the step budget burns."""
    kinds = (
        services.result_kinds
        if state["completion"].get("tools")
        else tuple(k for k in services.result_kinds if k != "calls")
    )
    transitions = state.get("transitions") or []
    if any(not t.get("conditions") for t in transitions):
        return []
    named: set[str] = set()
    for transition in transitions:
        for condition in transition.get("conditions") or []:
            named |= services.string_literals(condition.get("logic"))
    missing = [kind for kind in kinds if kind not in named]
    if not missing:
        return []
    return [
        f"Completion state '{state_id}' has no transition for result "
        f"kind(s) {missing} and no unconditional transition: such a "
        "turn stays in the state and makes no further call"
    ]


def _completion_raw_result_key(state: dict[str, Any], services: Any) -> str | None:
    key: str = state["completion"].get("result_key") or services.default_result_key
    return key


def _tool_names(tools: object) -> tuple[str | None, ...]:
    """One entry per declared tool: its function name, or ``None`` for an entry
    with no readable (string, non-empty) name. ``tools`` is read as given, so a
    raw value (absent, a single dict, a string) never raises."""
    entries = () if tools is None else tools if isinstance(tools, list) else (tools,)
    names: list[str | None] = []
    for tool in entries:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        names.append(name if isinstance(name, str) and name else None)
    return tuple(names)


def _completion_declared_tools(state: Any) -> tuple[str, ...]:
    config = getattr(state, "completion", None)
    names = _tool_names(getattr(config, "tools", None))
    return tuple(name for name in names if name is not None)


def _completion_raw_declared_tools(state: dict[str, Any]) -> tuple[str | None, ...]:
    return _tool_names(state["completion"].get("tools"))


def _completion_check_state(state: Any) -> str | None:
    # A terminal state runs no turn, so its completion call would never run.
    if not getattr(state, "transitions", None):
        return "a completion state needs a transition (a terminal state runs no turn)"
    return None
