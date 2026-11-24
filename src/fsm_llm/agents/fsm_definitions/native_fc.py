"""Native function-calling FSM (``build_native_fc_fsm``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fsm_llm.agents.constants import (
    NATIVE_FC_HANDLER_ONLY_KEYS,
    NativeFCContextKeys,
    NativeFCStates,
    NativeLoopEnd,
)
from fsm_llm.constants import CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT

from ._common import _by_id, _edge, _finalize_fsm, _flag_condition, _state


def _reply_kind_condition(kind: str) -> dict[str, Any]:
    """Condition that passes when the ``call_model`` reply is of ``kind``."""
    key = NativeFCContextKeys.MODEL_REPLY
    return {
        "description": f"The model turn was {kind}",
        "requires_context_keys": [key],
        "logic": {"===": [{"var": f"{key}.kind"}, kind]},
    }


def build_native_fc_fsm(
    tool_schemas: Sequence[dict[str, Any]],
    *,
    instructions: str,
    force_final_tool: str | None = None,
    repair: bool = False,
) -> dict[str, Any]:
    """Build the native function-calling FSM (``NativeFunctionCallingReactAgent``).

    Contract: returns a v4.1 FSM definition dict that loads as
    ``FSMDefinition``. Every state is silent (no Pass 2); every model call is
    a core completion state sending ``[system(instructions)] + transcript``.

    Args:
        tool_schemas: OpenAI function schemas (``ToolRegistry.get_json_schemas``),
            at least one; ``ValueError`` otherwise.
        instructions: the system message of every model turn, sent as given.
        force_final_tool: a tool the run must end by calling. The
            ``force_final`` state is built only when it names one of
            ``tool_schemas``; otherwise there is no forced turn.
        repair: build the ``repair`` state (a structured turn whose response
            format is read from ``CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT``).

    States (``NativeFCStates``):

    - ``call_model`` (initial): completion over ``_native_messages`` with
      every tool, ``tool_choice`` auto, result in ``model_reply``.
      ``-> run_tools`` p10 on a ``calls`` reply, ``-> force_final`` p20 /
      ``-> repair`` p30 when the handlers flagged them, ``-> conclude`` p900.
    - ``run_tools`` (handler state): ``-> force_final`` p20, ``-> repair``
      p30, ``-> conclude`` p40 once the loop has ended (``loop_end`` set),
      ``-> call_model`` p900.
    - ``force_final`` (optional): completion over ``_native_force_messages``
      with only the forced tool and a named ``tool_choice``, result in
      ``forced_reply``. ``-> repair`` p20 when flagged, ``-> conclude`` p900.
    - ``repair`` (optional): structured completion over
      ``_native_repair_messages``, result in ``repair_reply``.
      ``-> conclude`` p900.
    - ``conclude`` (terminal).

    The handlers (``native_fc.NativeFCHandlers``) seed the transcript, run
    the tools, write the nudge transcripts and the routing flags, and record
    the outcome. Their keys are ``handler_only_keys``. Never raises otherwise.
    """
    schemas = [dict(schema) for schema in tool_schemas]
    if not schemas:
        raise ValueError("build_native_fc_fsm needs at least one tool schema")
    # DECISION plan-2026-10-01T093600-944e2692/D-009: native_fc is this FSM
    # run by core; each model turn is a completion state, never an LLM call in
    # a handler or a loop in the agent. Do NOT append the forced-tool or repair
    # nudge to the shared transcript: each post-loop turn sends its own copy
    # (transcript + nudge) under its own messages key, so the repair turn
    # never sees the forced turn's nudge and the final answer is never
    # appended (the golden requests pin both). Do NOT decide here whether a
    # post-loop turn is due: the handlers flag it when the loop ends; this
    # builder only decides whether the state exists (an undeclared forced
    # tool gets no state, so no forced turn). See decisions.md D-009, D-026.
    states_ = NativeFCStates
    keys = NativeFCContextKeys
    names = [schema.get("function", {}).get("name") for schema in schemas]
    forced_schemas = [
        schema
        for schema, name in zip(schemas, names, strict=True)
        if force_final_tool and name == force_final_tool
    ]
    force_pending = _flag_condition(
        keys.FORCE_PENDING, "The run must still call its forced final tool"
    )
    repair_pending = _flag_condition(
        keys.REPAIR_PENDING, "The answer did not parse against the output schema"
    )

    force_edge: list[dict[str, Any]] = (
        [
            _edge(
                states_.FORCE_FINAL,
                "Force one call of the final tool",
                20,
                force_pending,
            )
        ]
        if forced_schemas
        else []
    )
    repair_edge: list[dict[str, Any]] = (
        [
            _edge(
                states_.REPAIR,
                "Repair the answer into the output schema",
                30,
                repair_pending,
            )
        ]
        if repair
        else []
    )

    states = _by_id(
        _state(
            states_.CALL_MODEL,
            "One model turn with the tools declared",
            "Let the model call tools or give its final answer",
            [
                _edge(
                    states_.RUN_TOOLS,
                    "The model asked for tools",
                    10,
                    _reply_kind_condition("calls"),
                ),
                *force_edge,
                *repair_edge,
                _edge(states_.CONCLUDE, "The model answered or garbled its turn", 900),
            ],
            completion={
                "tools": schemas,
                "instructions": instructions,
                "messages_key": keys.TRANSCRIPT,
                "result_key": keys.MODEL_REPLY,
            },
        ),
        _state(
            states_.RUN_TOOLS,
            "Run the model's tool calls",
            "Run each call through the tool registry and record the results",
            [
                *force_edge,
                *repair_edge,
                _edge(
                    states_.CONCLUDE,
                    "The model loop has ended",
                    40,
                    {
                        "description": "The loop ended",
                        "requires_context_keys": [keys.LOOP_END],
                        "logic": {
                            "in": [{"var": keys.LOOP_END}, list(NativeLoopEnd.ALL)]
                        },
                    },
                ),
                _edge(states_.CALL_MODEL, "Give the model the tool results", 900),
            ],
        ),
    )
    if forced_schemas:
        states[states_.FORCE_FINAL] = _state(
            states_.FORCE_FINAL,
            "One model turn that must call the final tool",
            "Make the model record its findings with the forced tool",
            [*repair_edge, _edge(states_.CONCLUDE, "The forced turn is done", 900)],
            completion={
                "tools": forced_schemas,
                "tool_choice": force_final_tool,
                "instructions": instructions,
                "messages_key": keys.FORCE_MESSAGES,
                "result_key": keys.FORCED_REPLY,
            },
        )
    if repair:
        states[states_.REPAIR] = _state(
            states_.REPAIR,
            "One structured model turn with no tools",
            "Get the final answer as JSON matching the output schema",
            [_edge(states_.CONCLUDE, "The repair turn is done", 900)],
            completion={
                "response_format_key": CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT,
                "instructions": instructions,
                "messages_key": keys.REPAIR_MESSAGES,
                "result_key": keys.REPAIR_REPLY,
            },
        )
    states[states_.CONCLUDE] = _state(
        states_.CONCLUDE,
        "The run is over",
        "Hold the answer, the trace and the run's outcome",
    )
    fsm = _finalize_fsm(
        name="native_fc",
        description=(
            "Native function calling: model turns with tools, tool turns, "
            "an optional forced final tool turn and repair turn"
        ),
        initial_state=states_.CALL_MODEL,
        persona="A capable AI agent that uses the provided tools",
        states=states,
    )
    fsm["handler_only_keys"] = [*fsm["handler_only_keys"], *NATIVE_FC_HANDLER_ONLY_KEYS]
    return fsm
