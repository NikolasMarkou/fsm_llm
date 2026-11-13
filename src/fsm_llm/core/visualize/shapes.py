"""Shape checks and small readers shared by the ASCII renderer and the graph exporters.

Layer ``core`` (plan d454fd62, step 76): split out of the old ``fsm_llm.visualizer``.
``_require_definition_shapes``, ``_require_initial_state`` and
``_require_valid_transition_targets`` are the one malformed-input contract of every
renderer; the rest reads the optional state fields.
"""

from __future__ import annotations

from typing import Any

# What an FSM without a name is called in every renderer.
_DEFAULT_NAME = "Unnamed FSM"


def _transitions_of(state: dict[str, Any]) -> list[dict[str, Any]]:
    """A state's transitions, with an explicit ``"transitions": null`` read as
    none (C9: ``.get(k, [])`` returns that ``None`` and every style crashed)."""
    return state.get("transitions") or []


def _one_line(value: Any, default: str) -> str:
    """``value`` as a single display line: whitespace runs (newlines included)
    collapse to one space, so a multi-line FSM name cannot break the header
    box (C9); a missing/blank value falls back to ``default``."""
    return " ".join(str(value).split()) if value else default


def _is_terminal(state: dict[str, Any]) -> bool:
    """The one terminal rule: a state without transitions."""
    return not _transitions_of(state)


def _known_states_hint(states: dict[str, Any]) -> str:
    """``defined states: 'a', 'b', ...`` (the first five ids, sorted)."""
    known = ", ".join(repr(s) for s in sorted(states)[:5]) or "<none>"
    suffix = ", ..." if len(states) > 5 else ""
    return f"defined states: {known}{suffix}"


def _key_list(value: Any, what: str) -> list[str]:
    """``value`` (``None`` is none) as a list of context key names, or ``ValueError``."""
    keys = value or []
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        raise ValueError(f"{what} must be a list of strings, got {value!r}")
    return keys


def _require_definition_shapes(states: Any) -> None:
    """Reject a definition whose ``states`` or their parts are not mappings.

    Args:
        states: the FSM's ``states`` value, as read from the definition.

    Raises:
        ValueError: ``states`` is not a dict, a state is not a dict, a
            state's ``transitions`` is not a list, or a transition is not a
            dict. The message names the state.
    """
    if not isinstance(states, dict):
        raise ValueError(f"'states' must be a mapping, got {type(states).__name__}")
    for state_id, state in states.items():
        if not isinstance(state, dict):
            raise ValueError(
                f"State {state_id!r} must be a mapping, got {type(state).__name__}"
            )
        transitions = _transitions_of(state)
        if not isinstance(transitions, list):
            raise ValueError(
                f"'transitions' of state {state_id!r} must be a list, "
                f"got {type(transitions).__name__}"
            )
        for transition in transitions:
            if not isinstance(transition, dict):
                raise ValueError(
                    f"A transition of state {state_id!r} must be a mapping, "
                    f"got {type(transition).__name__}"
                )


def _require_initial_state(states: dict[str, Any], initial_state: str) -> None:
    """Reject an FSM whose ``initial_state`` is not a key of ``states``.

    Args:
        states: the FSM's ``states`` mapping.
        initial_state: the FSM's declared ``initial_state`` id.

    Raises:
        ValueError: naming both fields, if the id is absent.
    """
    # DECISION plan-2026-07-19T191147-4b664252/D-021 [STALE]
    # This guard belongs HERE, ahead of `build_graph_representation`, and NOT in
    # `build_states_section`. The finding located the failure at `states[state_id]` in the
    # states section, but the FIRST raise is actually `calculate_depths`'
    # `state_metrics[initial_state]["depth"] = 0` — which runs for EVERY style, before any
    # section is built. A guard in `build_states_section` is therefore dead code that
    # leaves `style="compact"`/`"minimal"` still emitting a bare `KeyError('start')`,
    # rendered by `visualize_fsm_from_file` as the useless `"Error: 'start'"`. Do NOT
    # "fix" this instead by skipping unknown ids while sorting: that hides a malformed
    # definition behind a plausible-looking diagram. See decisions.md D-021.
    if initial_state in states:
        return
    raise ValueError(
        f"initial_state {initial_state!r} is not present in 'states' "
        f"({_known_states_hint(states)})"
    )


def _require_valid_transition_targets(states: dict[str, Any]) -> None:
    """Reject an FSM whose transitions name a ``target_state`` absent from ``states``.

    Args:
        states: the FSM's ``states`` mapping.

    Raises:
        ValueError: naming both the source state and the missing target.
    """
    # DECISION plan-2026-07-20T040150-876e7164/D-013 [STALE]
    # Placement mirrors `_require_initial_state` (D-021) for the SAME measured reason:
    # the first raise is `calculate_depths`' `state_metrics[target]["depth"] = ...`
    # (probed, not read -- it fires for all three styles), which runs before any
    # section is built. A guard inside `build_states_section` would be dead code for
    # `style="compact"`/`"minimal"`.
    #
    # Do NOT "simplify" this by wiring the visualizer into `FSMValidator`. Both
    # `FSMValidator` and `FSMDefinition`'s pydantic `model_validator` already reject a
    # dangling `target_state` as an ERROR -- the original framing of this defect ("the
    # validator passes input the visualizer crashes on") is a GHOST and was falsified at
    # HEAD. The real gap is narrower: `visualize_fsm_ascii`/`visualize_fsm_from_file` are
    # a SEPARATE entry point that never consults either layer, so `fsm-llm-visualize` on
    # invalid input printed the useless `"Error: 'nowhere'"`. Running the full validator
    # here is a different, larger design change (it would make the visualizer refuse
    # everything the validator refuses, including WARNING-only shapes it renders fine).
    #
    # Do NOT delete this guard on the theory that "the validator already catches this" --
    # nothing forces a user through the validator before the visualizer.
    #
    # This deliberately sweeps EVERY state, not only the ones `calculate_depths` reaches.
    # Measured: a dangling target on an UNREACHABLE state does not crash today -- it
    # renders an arrow to a state that does not exist. Refusing it is the D-021 rule
    # ("do not hide a malformed definition behind a plausible-looking diagram") applied
    # to the same shape, and it is a deliberate tightening. See decisions.md D-013.
    for state_id, state in states.items():
        for transition in _transitions_of(state):
            target = transition.get("target_state", "")
            if target in states:
                continue
            raise ValueError(
                f"Transition from {state_id!r} to non-existent state {target!r} "
                f"({_known_states_hint(states)})"
            )
