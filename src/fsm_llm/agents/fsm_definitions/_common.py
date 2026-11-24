"""Helpers every builder shares: the finished-FSM wrapper and the state, edge and condition constructors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fsm_llm.agents.constants import FRAMEWORK_ONLY_KEYS


def _finalize_fsm(
    name: str,
    description: str,
    initial_state: str,
    persona: str,
    states: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the top-level FSM definition dict shared by every builder here.

    Contract: returns ``{"name", "description", "initial_state", "persona",
    "states", "handler_only_keys"}`` where ``description`` is the builder's
    STATIC text, exactly as given: caller text (the task) never enters
    definition text, so one pattern has one ``fsm_id`` whatever the task
    (D-002 and D-008 of plan a555dcfe), and ``handler_only_keys`` is
    :data:`FRAMEWORK_ONLY_KEYS` (D-051 of plan 06a5ec0a). ``states`` is stored
    by reference, not copied. Never raises.
    """
    return {
        "name": name,
        "description": description,
        "initial_state": initial_state,
        "persona": persona,
        "states": states,
        "handler_only_keys": list(FRAMEWORK_ONLY_KEYS),
    }


def _is(key: str, value: Any) -> dict[str, Any]:
    """JsonLogic ``{"==": [{"var": key}, value]}``."""
    return {"==": [{"var": key}, value]}


def _cond(description: str, logic: dict[str, Any]) -> dict[str, Any]:
    """One transition condition: ``{"description", "logic"}``."""
    return {"description": description, "logic": logic}


def _edge(
    target: str,
    description: str,
    priority: int,
    *conditions: dict[str, Any],
    empty: bool = False,
) -> dict[str, Any]:
    """One transition dict, the only spelling of the edge shape here.

    Contract: ``conditions`` is absent when none are given, unless ``empty``
    asks for the explicit ``"conditions": []`` (a builder's own spelling: core
    loads ``[]`` and an absent list as different transitions and mints a
    different ``fsm_id``, so a helper must not unify them). Never raises.
    """
    edge: dict[str, Any] = {
        "target_state": target,
        "description": description,
        "priority": priority,
    }
    if conditions or empty:
        edge["conditions"] = list(conditions)
    return edge


def _when(
    target: str,
    description: str,
    priority: int,
    condition_description: str,
    logic: dict[str, Any],
) -> dict[str, Any]:
    """An edge with exactly one condition (:func:`_edge`, :func:`_cond`)."""
    return _edge(target, description, priority, _cond(condition_description, logic))


def _state(
    state_id: str,
    description: str,
    purpose: str,
    transitions: Sequence[dict[str, Any]] = (),
    *,
    response_instructions: str = "",
    required: Sequence[str] | None = None,
    fields: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """One state dict, the only spelling of the state shape here.

    Contract: always ``id``, ``description``, ``purpose``,
    ``response_instructions`` and ``transitions`` (empty = terminal);
    ``required`` adds ``required_context_keys`` (an empty list is kept);
    ``fields`` adds ``field_extractions`` with an empty state-level
    ``extraction_instructions`` (D-009 of plan 06a5ec0a: typed per-field values,
    no bulk call); ``extra`` adds any other state key and may override
    ``extraction_instructions``. Never raises.
    """
    state: dict[str, Any] = {
        "id": state_id,
        "description": description,
        "purpose": purpose,
    }
    if required is not None:
        state["required_context_keys"] = list(required)
    if fields is not None:
        state["extraction_instructions"] = ""
        state["field_extractions"] = fields
    state.update(extra)
    state["response_instructions"] = response_instructions
    state["transitions"] = list(transitions)
    return state


def _by_id(*states: dict[str, Any]) -> dict[str, Any]:
    """The ``states`` mapping of an FSM: each state under its own ``id``
    (the loader requires ``state.id == key``), in the order given."""
    return {state["id"]: state for state in states}


def _flag_condition(key: str, description: str) -> dict[str, Any]:
    """Condition that passes only when ``key`` holds the boolean ``True``.

    Strict ``===``: core's ``==`` also passes ``1``, ``1.0``, ``"true"`` and
    ``"True"`` against ``True``.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-022: the meta gates compare
    # with `===`. Do NOT relax them to `==`: core's `==` treats "true", "True",
    # 1 and 1.0 as True, so any non-bool value that reached a gate key would
    # start a build or a type switch. See decisions.md D-022.
    return {
        "description": description,
        "requires_context_keys": [key],
        "logic": {"===": [{"var": key}, True]},
    }
