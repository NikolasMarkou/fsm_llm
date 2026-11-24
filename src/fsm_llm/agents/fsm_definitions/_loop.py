"""The think/act/conclude parts of the code-backed tool loop (parallel ReAct) and the typed-field helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from fsm_llm import typed_field_extraction
from fsm_llm.agents.constants import AgentStates, ContextKeys
from fsm_llm.constants import has_internal_prefix

from ._common import _edge, _state, _when

if TYPE_CHECKING:  # annotation only: core owns the name, agents keep no copy
    from fsm_llm.core.definitions import TypedFieldType


# DECISION plan-2026-09-29T103145-06a5ec0a/D-050
# `any` is for whole generated ARTIFACTS: EvalOpt `generated_output`,
# MakerChecker `draft_output`, PromptChain `chain_step_result`. Do NOT type
# them `str`: `any` is their baseline type (before step 20), and on a provider
# without a grammar it lets the model return a JSON deliverable as a native
# object. On Ollama it changes nothing for objects (the `any` grammar has no
# object branch, llm/ollama.py `_VALUE_TYPES_ANY`), so a JSON deliverable is still
# an escaped string that max_tokens can cut off; the protection there is core
# llm/parsing.py's envelope salvage (never ships the envelope; a cut-off value is
# logged and returned at confidence 0.3, D-056), not this type. A cut-off
# artifact can still ship as the answer (Known open: output budget, Track B).
# Short prose fields (feedback, critiques, reflections, plan step results,
# ADaPT `attempt_result`, D-057) stay `str`. Answer paths serialise a
# dict/list with base.artifact_text, which treats empty values as no answer.
# (Field types: core `fsm_llm.core.definitions.TypedFieldType`; the builder is core
# `typed_field_extraction`.)

# Context every loop-field prompt sees: the task and the tool observations.
_LOOP_FIELD_CONTEXT_KEYS: tuple[str, ...] = (ContextKeys.TASK, ContextKeys.OBSERVATIONS)


def loop_field_context_keys(extra_context_keys: Sequence[str] = ()) -> list[str]:
    """The ``context_keys`` of a loop field prompt: ``task``, ``observations``,
    then ``extra_context_keys`` (order kept, duplicates dropped).

    Every agent typed field (core ``typed_field_extraction``) and the think
    builders' tool selection configs take their ``context_keys`` from here.
    Raises ``ValueError`` for an extra key that is ``agent_trace`` or
    internal-prefixed (core reads a listed key from raw context, with no
    internal-key filter).

    # DECISION plan-2026-09-29T103145-06a5ec0a/D-017
    Do NOT drop ``context_keys`` from an agent field (``None`` dumps all
    visible context, including the unbounded ``agent_trace``, into every
    per-field prompt), and do NOT cap or rename ``agent_trace`` instead: it
    feeds ``AgentResult.trace`` and marks the FSM as agent-managed for core.
    """
    bad = [
        key
        for key in extra_context_keys
        if key == ContextKeys.AGENT_TRACE or has_internal_prefix(key)
    ]
    if bad:
        raise ValueError(f"context keys not allowed in a field prompt: {bad}")
    return list(dict.fromkeys((*_LOOP_FIELD_CONTEXT_KEYS, *extra_context_keys)))


def _field(
    name: str,
    field_type: TypedFieldType,
    instructions: str,
    extra: Sequence[str] = (),
    *,
    required: bool = True,
) -> dict[str, Any]:
    """One agent typed field: ``typed_field_extraction`` over
    :func:`loop_field_context_keys` ``(extra)``. Raises ``ValueError`` like both."""
    return typed_field_extraction(
        name,
        field_type,
        instructions,
        context_keys=loop_field_context_keys(extra),
        required=required,
    )


def conclude_on_evidence_logic(
    flag: str = ContextKeys.SHOULD_TERMINATE,
) -> dict[str, Any]:
    """JsonLogic for a tool-loop ``conclude`` edge: terminate on evidence only.

    Contract: returns a fresh ``(<flag> == True AND observation_count > 0)
    OR max_iterations_reached == True`` expression (``flag`` defaults to
    ``should_terminate``). Used on the ``think`` and ``act`` conclude edges of
    the ReAct, Reflexion and ParallelReact FSMs, and with ``evaluation_passed``
    on Reflexion's ``evaluate`` conclude edge. Never raises.

    # DECISION plan-2026-09-29T103145-06a5ec0a/D-051
    A forced stop concludes on the flag ALONE. Do NOT require ``<flag>`` for
    it again: think entry now clears a limiter-forced ``should_terminate`` so
    the last think turn is asked for its own verdict (that verdict decides
    ``success``), and a forced think turn whose model says nothing must still
    conclude.

    # DECISION plan-2026-09-24T091842-c1d5bfbc/D-008
    Do NOT gate a tool loop's conclude edge on ``should_terminate`` alone: a
    model that sets it on turn 1 answers from memory with no tool run (the B4
    under-call, plan_2026-05-30_5598b755 D-004). Both evidence keys are
    framework-written (tool executors, iteration limiter, stall detector), so
    a forced stop with zero observations still concludes. Keep the D-002
    unconditional ``think -> act`` fallback beside it, or a rejected conclude
    BLOCKS ``think``.
    """
    return {
        "or": [
            {
                "and": [
                    {"==": [{"var": flag}, True]},
                    {">": [{"var": [ContextKeys.OBSERVATION_COUNT, 0]}, 0]},
                ]
            },
            {"==": [{"var": ContextKeys.MAX_ITERATIONS_REACHED}, True]},
        ]
    }


_TERMINATE_ON_EVIDENCE = (
    "Agent decided to terminate AND a tool has run or termination is forced"
)


def _think_transitions(
    conclude_description: str,
    act_description: str = "Execute the selected tool",
) -> list[dict[str, Any]]:
    """The ``think`` edges of the parallel tool loop (no approval edge: the
    approval patterns are pattern files).

    Contract: ``think -> conclude`` p10 on :func:`conclude_on_evidence_logic`,
    then the D-002 unconditional ``think -> act`` p300. Never raises.
    """
    return [
        # DECISION plan_2026-05-30_5598b755/D-004 [STALE]
        # B4 under-call fix: do NOT conclude on the model's
        # should_terminate alone — require evidence that a tool has
        # run (observation_count > 0) OR that termination is forced
        # (max_iterations_reached, set by the iteration limiter and
        # the execute_tool stall-detector). This blocks the
        # turn-1 "answer from memory" path (think -> conclude
        # pre-empting think -> act) while preserving termination.
        # Keyed on framework-only context vars so it is immune to the
        # transition-evaluator re-merge of raw extracted should_terminate.
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-008
        # Evidence guard, see conclude_on_evidence_logic.
        _when(
            AgentStates.CONCLUDE,
            conclude_description,
            10,
            _TERMINATE_ON_EVIDENCE,
            conclude_on_evidence_logic(),
        ),
        # DECISION plan-2026-09-24T045559-3e4eb3e5/D-002
        # Unconditional lowest-priority fallback. Do NOT gate this edge on the
        # tool selection: a gated edge BLOCKS `think` on a null/unknown tool,
        # and no PRE_TRANSITION or `act`-entry handler runs on a BLOCKED turn,
        # so the run burns the 3x loop ceiling. `act` handles a missing or
        # unknown tool (an empty batch for ParallelReact).
        _edge(AgentStates.ACT, act_description, 300),
    ]


def _act_state(
    description: str, purpose: str, next_state: str, next_description: str
) -> dict[str, Any]:
    """The ``act`` state of every tool loop: conclude p1 on evidence, else
    ``next_state`` p900 (``think``, or Reflexion's ``evaluate``). Never raises."""
    return _state(
        AgentStates.ACT,
        description,
        purpose,
        [
            # DECISION plan_2026-05-30_5598b755/D-004 [STALE]
            # Mirror the think->conclude guard: a should_terminate
            # set on entry to act (e.g. the model wanted to quit
            # without a tool) must NOT conclude unless a tool has
            # run or termination is forced. The stall-detector and
            # iteration limiter set max_iterations_reached, so a
            # genuinely tool-free turn still concludes here.
            _when(
                AgentStates.CONCLUDE,
                "Terminate when framework signals completion",
                1,
                "Framework/agent decided to terminate AND a tool has run or "
                "termination is forced",
                conclude_on_evidence_logic(),
            ),
            _edge(next_state, next_description, 900),
        ],
    )


def _conclude_state(
    response_instructions: str, required: Sequence[str] | None = None
) -> dict[str, Any]:
    """The terminal ``conclude`` state of every tool loop; ``required`` is the
    output schema's field names (see :func:`_schema_field_names`) or ``None``."""
    return _state(
        AgentStates.CONCLUDE,
        "Formulate and present the final answer",
        "Synthesize all observations into a complete answer",
        response_instructions=response_instructions,
        required=required,
    )


def _schema_field_names(output_schema: type | None) -> list[str]:
    """The field names of a pydantic ``output_schema`` (``[]`` for none or a
    class without ``model_fields``)."""
    if output_schema and hasattr(output_schema, "model_fields"):
        return list(output_schema.model_fields.keys())
    return []
