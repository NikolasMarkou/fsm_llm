"""Native function-calling FSM (``build_native_fc_fsm``): the shipped file ``patterns/native_fc.json``, instantiated."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def native_fc_options(
    schemas: Sequence[dict[str, Any]], force_final_tool: str | None, repair: bool
) -> dict[str, Any]:
    """The options of the file for one run: the tools, the forced tool (only when it names one of
    them) and the two removal variants (``without_force_final``, ``without_repair``)."""
    # DECISION plan-2026-11-24T043150-a555dcfe/D-021
    # The file holds the FULL machine and two REMOVAL variants, declared `without_repair` first
    # (each removes its highest edge index first, so no index depends on the other variant). Do NOT
    # turn them into add-variants: the patch language cannot insert a key in the middle of `states`
    # (an added state lands after `conclude` and the bytes move) and an edge index would depend on
    # which variant ran. Do NOT reorder the two variants or add an edge to the file without redoing
    # both index lists; `test_pattern_native_fc.py` holds every combination.
    forced = [
        schema
        for schema in schemas
        if force_final_tool
        and schema.get("function", {}).get("name") == force_final_tool
    ]
    return {
        "tool_schemas": [dict(schema) for schema in schemas],
        "forced_tools": forced,
        "force_final_tool": force_final_tool if forced and force_final_tool else "",
        "without_force_final": not forced,
        "without_repair": not repair,
    }


def build_native_fc_fsm(
    tool_schemas: Sequence[dict[str, Any]],
    *,
    instructions: str,
    force_final_tool: str | None = None,
    repair: bool = False,
) -> dict[str, Any]:
    """Build the native function-calling FSM (``NativeFunctionCallingReactAgent``).

    Contract: returns a v4.1 FSM definition dict that loads as ``FSMDefinition``. Every state is
    silent (no Pass 2); every model call is a core completion state sending
    ``[system(instructions)] + transcript``. ``ValueError`` for no tool schema.

    Args:
        tool_schemas: OpenAI function schemas (``ToolRegistry.get_json_schemas``), at least one.
        instructions: the system message of every model turn, sent as given.
        force_final_tool: a tool the run must end by calling. The ``force_final`` state is kept only
            when it names one of ``tool_schemas``; otherwise there is no forced turn.
        repair: keep the ``repair`` state (a structured turn whose response format is read from
            ``CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT``).

    States (``NativeFCStates``): ``call_model`` (initial, the tools, ``tool_choice`` auto),
    ``run_tools`` (handler state), ``force_final`` (optional: only the forced tool, a named
    ``tool_choice``), ``repair`` (optional: structured, no tools), ``conclude`` (terminal). The
    edges, their priorities and the handlers that flag the post-loop turns are in the file and in
    ``handlers/native_fc.py``; the handler-only keys are the file's too.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-009: native_fc is this FSM run by core; each model
    # turn is a completion state, never an LLM call in a handler or a loop in the agent. Do NOT
    # append the forced-tool or repair nudge to the shared transcript: each post-loop turn sends its
    # own copy (transcript + nudge) under its own messages key, so the repair turn never sees the
    # forced turn's nudge and the final answer is never appended (the golden requests pin both). Do
    # NOT decide here whether a post-loop turn is due: the handlers flag it when the loop ends; this
    # wrapper only decides whether the state exists (an undeclared forced tool gets no state, so no
    # forced turn: the removal variants). See decisions.md D-009, D-026.
    if not tool_schemas:
        raise ValueError("build_native_fc_fsm needs at least one tool schema")
    # lazy: pattern_inputs imports the sibling `_loop`, so a module-level import would be a cycle
    from fsm_llm.agents.pattern_files import load_envelope
    from fsm_llm.agents.pattern_inputs import pattern_fsm

    return pattern_fsm(
        load_envelope("native_fc"),
        tools=None,
        options=native_fc_options(tool_schemas, force_final_tool, repair),
        variables={"system_message": instructions},
    )
