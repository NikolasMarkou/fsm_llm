"""ReAct FSM (``build_react_fsm``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fsm_llm.agents.constants import AgentStates, ContextKeys
from fsm_llm.agents.prompts import (
    build_conclude_response_instructions,
    build_think_extraction_instructions,
)
from fsm_llm.tools import ToolRegistry

from ._common import _by_id, _finalize_fsm, _state
from ._loop import (
    _act_state,
    _await_approval_state,
    _conclude_state,
    _schema_field_names,
    _think_field_extractions,
    _think_transitions,
)


def build_react_fsm(
    registry: ToolRegistry,
    task_description: str = "",
    include_approval_state: bool = False,
    use_classification: bool = False,
    output_schema: type | None = None,
    context_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """
    Build a ReAct FSM definition from a tool registry.

    The FSM implements the Observe-Think-Act loop:
    - think: LLM reasons about the task and selects a tool
    - act: tool is executed (via handler), observation is recorded
    - conclude: LLM produces final answer when should_terminate is true

    Optionally includes an await_approval state for HITL patterns.

    When *use_classification* is True, tool selection in the think state
    uses a ``classification_extractions`` config (backed by the core
    ``Classifier``) instead of relying solely on extraction instructions.
    This can improve tool selection accuracy for large tool registries.

    ``think`` extracts only typed per-field values (see
    :func:`_think_field_extractions`); ``context_keys`` adds caller keys to
    those prompts (``caller_prompt_keys``).

    ``task_description`` is only the retrieval query of a semantic tool
    registry (the tool section of the prompts); the definition's name and
    description are static, so the task never enters definition text
    (D-002 and D-008 of plan a555dcfe).
    """
    # Build persona with tool awareness
    persona = (
        "You are a methodical AI agent that solves tasks by using tools step by step. "
        "Think carefully before each action. Review previous observations before deciding. "
        "Terminate when you have gathered enough information to answer the task."
    )

    think_instructions = build_think_extraction_instructions(
        registry, task_description=task_description
    )
    think_state = _state(
        AgentStates.THINK,
        "Reason about the task and select the next tool to use",
        "Analyze the task and previous observations to decide the next action",
        _think_transitions(
            "Task can be answered (a tool has run, or termination is forced)",
            include_approval=include_approval_state,
        ),
        required=[
            ContextKeys.TOOL_NAME,
            ContextKeys.TOOL_INPUT,
            ContextKeys.SHOULD_TERMINATE,
        ],
        fields=_think_field_extractions(
            think_instructions,
            include_tool_name=not use_classification,
            context_keys=context_keys,
        ),
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-009: no state-level bulk
        # call (one more LLM call per turn for keys the typed fields already
        # fill from their own narrowed context). Do NOT empty it under use_classification: tool_name is classification-owned
        # there and relies on the bulk fill when the classifier declines
        # (21cd7f8e D-019).
        extraction_instructions=think_instructions if use_classification else "",
    )

    if use_classification:
        schema = registry.to_classification_schema()
        think_state["classification_extractions"] = [
            {
                "field_name": "tool_name",
                "intents": schema["intents"],
                "fallback_intent": schema["fallback_intent"],
                "confidence_threshold": schema["confidence_threshold"],
            }
        ]

    states = _by_id(
        think_state,
        _act_state(
            "Execute the selected tool and observe the result",
            "Run the tool and record the observation",
            AgentStates.THINK,
            "Return to thinking with new observation",
        ),
        _conclude_state(
            build_conclude_response_instructions(
                refused_actions=include_approval_state
            ),
            _schema_field_names(output_schema),
        ),
    )

    if include_approval_state:
        states[AgentStates.AWAIT_APPROVAL] = _await_approval_state()

    return _finalize_fsm(
        "react_agent",
        "ReAct agent with tool use",
        AgentStates.THINK,
        persona,
        states,
    )
