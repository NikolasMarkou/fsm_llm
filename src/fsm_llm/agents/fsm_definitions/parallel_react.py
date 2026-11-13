"""Parallel ReAct FSM (``build_parallel_react_fsm``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fsm_llm.agents.constants import AgentStates, ContextKeys
from fsm_llm.agents.prompts import (
    build_conclude_response_instructions,
    build_think_terminate_instructions,
)
from fsm_llm.tools import ToolRegistry

from ._common import _by_id, _finalize_fsm, _state
from ._loop import (
    _act_state,
    _conclude_state,
    _field,
    _schema_field_names,
    _think_transitions,
)


def _build_parallel_think_instructions(
    registry: ToolRegistry, task_description: str
) -> str:
    """Per-field instructions for the parallel-think ``tool_calls`` list."""
    return (
        f"You are solving this task: {task_description}\n\n"
        f"{registry.to_prompt_description()}\n\n"
        "Decide which tools to call NEXT. You may call SEVERAL independent tools "
        f"at once to work in parallel. '{ContextKeys.TOOL_CALLS}' is a JSON list "
        "of tool calls, each an object with 'tool_name' (one of the available "
        "tools) and 'tool_input' (an object of arguments). Use an empty list if "
        "no tool is needed. Only batch tools whose inputs do not depend on each "
        "other's results."
    )


def build_parallel_react_fsm(
    registry: ToolRegistry,
    task_description: str = "",
    output_schema: type | None = None,
    context_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the think -> act(parallel) -> conclude FSM definition.

    ``think`` extracts only typed per-field values (``tool_calls`` list,
    ``should_terminate`` bool; D-009 of plan 06a5ec0a);
    their prompts list ``task``, ``observations``, ``agent_feedback`` and
    ``context_keys``, never ``agent_trace``. The FSM keeps its own name
    (``ParallelReactAgent``) and fixed description: ``task_description`` only
    reaches the think prompt.
    """
    extra = (ContextKeys.AGENT_FEEDBACK, *context_keys)

    persona = (
        "You are a methodical AI agent that solves tasks by using tools. "
        "You can call multiple independent tools at once to work in parallel. "
        "Review previous observations before deciding. Terminate when you have "
        "enough information to answer."
    )

    states = _by_id(
        _state(
            AgentStates.THINK,
            "Reason about the task and select one or more tools to run",
            "Decide the next (possibly parallel) batch of tool calls",
            _think_transitions(
                "Task can be answered with current observations",
                "Execute the selected tool batch",
            ),
            required=[ContextKeys.TOOL_CALLS, ContextKeys.SHOULD_TERMINATE],
            fields=[
                _field(
                    ContextKeys.TOOL_CALLS,
                    "list",
                    _build_parallel_think_instructions(registry, task_description),
                    extra,
                ),
                _field(
                    ContextKeys.SHOULD_TERMINATE,
                    "bool",
                    build_think_terminate_instructions(),
                    extra,
                    required=False,
                ),
            ],
        ),
        _act_state(
            "Execute the selected tools concurrently and observe",
            "Run the tool batch and record observations",
            AgentStates.THINK,
            "Return to thinking with new observations",
        ),
        _conclude_state(
            build_conclude_response_instructions(), _schema_field_names(output_schema)
        ),
    )

    return _finalize_fsm(
        "ParallelReactAgent",
        "",
        "ReAct agent with parallel tool dispatch",
        AgentStates.THINK,
        persona,
        states,
    )
