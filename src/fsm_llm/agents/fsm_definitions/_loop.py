"""The think/act/conclude parts the loop patterns (ReAct, Reflexion, Plan-and-Execute, parallel ReAct) share."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from fsm_llm import typed_field_extraction
from fsm_llm.agents.constants import AgentStates, ContextKeys
from fsm_llm.agents.prompts import build_think_terminate_instructions
from fsm_llm.constants import has_internal_prefix

from ._common import _edge, _is, _state, _when

if TYPE_CHECKING:  # annotation only: core owns the name, agents keep no copy
    from fsm_llm.definitions import TypedFieldType


def _tool_selection_field_extractions(
    think_instructions: str,
    *,
    include_tool_name: bool = True,
    context_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Typed ``field_extractions`` for a think state's tool selection.

    Contract: ``think_instructions`` is the think prompt (tools, examples,
    rules); returns raw dicts for ``State(field_extractions=...)``:
    ``tool_name`` as ``str`` (omitted when ``include_tool_name`` is False, i.e.
    the classifier owns it) and ``tool_input`` as ``dict``. ``context_keys``
    (required) narrows each prompt's context to those keys (see
    :func:`loop_field_context_keys`). Never raises.

    # DECISION plan-2026-09-19T175721-21cd7f8e/D-024
    Do NOT drop these and rely on the auto-minted config from
    ``required_context_keys``: that config is ``field_type="any"``, whose grammar
    (D-001) excludes ``object``, and on qwen3.5:9b-q8_0 the model then returns
    null for both keys so no tool ever runs (live s15 A/B: 0/3 vs 3/3 with these
    configs). Do NOT widen the ``any`` union to admit ``object`` (1/3 live, and it
    reopens LV-01 for every auto-minted key). See decisions.md D-024.
    """
    fields = [("tool_input", "dict")]
    if include_tool_name:
        fields.insert(0, ("tool_name", "str"))
    return [
        {
            "field_name": name,
            "field_type": field_type,
            "extraction_instructions": f"Extract the '{name}' field. {think_instructions}",
            "context_keys": list(context_keys),
        }
        for name, field_type in fields
    ]


# DECISION plan-2026-09-29T103145-06a5ec0a/D-050
# `any` is for whole generated ARTIFACTS: EvalOpt `generated_output`,
# MakerChecker `draft_output`, PromptChain `chain_step_result`. Do NOT type
# them `str`: `any` is their baseline type (before step 20), and on a provider
# without a grammar it lets the model return a JSON deliverable as a native
# object. On Ollama it changes nothing for objects (the `any` grammar has no
# object branch, ollama.py `_VALUE_TYPES_ANY`), so a JSON deliverable is still
# an escaped string that max_tokens can cut off; the protection there is core
# llm.py's envelope salvage (never ships the envelope; a cut-off value is
# logged and returned at confidence 0.3, D-056), not this type. A cut-off
# artifact can still ship as the answer (Known open: output budget, Track B).
# Short prose fields (feedback, critiques, reflections, plan step results,
# ADaPT `attempt_result`, D-057) stay `str`. Answer paths serialise a
# dict/list with base.artifact_text, which treats empty values as no answer.
# (Field types: core `fsm_llm.definitions.TypedFieldType`; the builder is core
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


def _think_field_extractions(
    think_instructions: str,
    *,
    include_tool_name: bool = True,
    context_keys: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Every typed field a ReAct/Reflexion ``think`` turn extracts.

    Contract: the tool selection (:func:`_tool_selection_field_extractions`,
    unchanged types), then ``should_terminate`` (bool, optional: a null costs
    no retry and gates no edge). No ``reasoning`` field: that name is an
    extraction envelope key (D-034/D-035 of plan 06a5ec0a). Every prompt
    lists ``task``, ``observations``, ``agent_feedback`` and ``context_keys``,
    never ``agent_trace``. ``think_instructions`` is the think prompt; pair
    the result with an empty state-level ``extraction_instructions`` unless the
    classifier owns ``tool_name`` (D-019 of plan 21cd7f8e keeps the bulk fill
    there). Raises ``ValueError`` like :func:`loop_field_context_keys`.
    """
    keys = loop_field_context_keys((ContextKeys.AGENT_FEEDBACK, *context_keys))
    return [
        *_tool_selection_field_extractions(
            think_instructions, include_tool_name=include_tool_name, context_keys=keys
        ),
        typed_field_extraction(
            ContextKeys.SHOULD_TERMINATE,
            "bool",
            build_think_terminate_instructions(),
            context_keys=keys,
            required=False,
        ),
    ]


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


def _await_approval_state() -> dict[str, Any]:
    """The ``await_approval`` state shared by every approval-gated FSM.

    Contract: returns a fresh state dict with edges to ``conclude`` (forced
    termination), ``act`` (``approval_granted`` True) and ``think``
    (``approval_granted`` False); the builder must define those three states.
    The agent's loop driver (``BaseAgent._handle_hitl_approval``) writes the
    decision before the state's step, so the state makes no LLM call. Never
    raises.
    """
    # DECISION plan-2026-09-30T062855-07ad3f8c/D-033: this state extracts
    # nothing. Do NOT give it `extraction_instructions`, `required_context_keys`
    # or a field for `approval_granted`: the driver's callback asks the
    # approver and writes the decision before this step, the model has no
    # human reply to read, and any extraction here is a model write channel
    # inside the state that guards the approval (the old bulk pass cost one
    # LLM call per visit and let the reply fill an empty `tool_input` after
    # the ask and add keys of its own, live: `denied_tool_call`).
    # LOOP-09: an intermediate state; the driver asks the approver, so no
    # Pass-2 prose (core skips Pass 2 on empty instructions).
    return _state(
        AgentStates.AWAIT_APPROVAL,
        "Waiting for human approval before executing action",
        "Hold the selected action until the approval decision is set",
        [
            _when(
                AgentStates.CONCLUDE,
                "Terminate when framework signals completion",
                1,
                "Framework or agent decided to terminate",
                _is(ContextKeys.SHOULD_TERMINATE, True),
            ),
            _when(
                AgentStates.ACT,
                "Approval granted, proceed with action",
                10,
                "User approved the action",
                _is(ContextKeys.APPROVAL_GRANTED, True),
            ),
            _when(
                AgentStates.THINK,
                "Approval denied, reconsider approach",
                300,
                "User denied the action",
                _is(ContextKeys.APPROVAL_GRANTED, False),
            ),
        ],
    )


_TERMINATE_ON_EVIDENCE = (
    "Agent decided to terminate AND a tool has run or termination is forced"
)


def _think_transitions(
    conclude_description: str,
    act_description: str = "Execute the selected tool",
    *,
    include_approval: bool = False,
) -> list[dict[str, Any]]:
    """The ``think`` edges of every tool loop (ReAct, Reflexion, ParallelReact).

    Contract: ``think -> conclude`` p10 on :func:`conclude_on_evidence_logic`,
    then ``think -> await_approval`` p150 when ``include_approval`` (it beats
    the fallback and loses to conclude), then the D-002 unconditional
    ``think -> act`` p300. The descriptions are the prompt-visible wording each
    pattern keeps. Never raises.
    """
    transitions = [
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
        )
    ]
    if include_approval:
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-005
        # The approval edge (150) must beat the fallback below (300).
        # Do NOT drop it while the agent registers the HITL gate: without
        # it think -> act runs execute_tool before the driver asks, and
        # every gated call burns an act/evaluate/reflect cycle on the
        # D-004 refusal.
        transitions.append(
            _when(
                AgentStates.AWAIT_APPROVAL,
                "Action requires human approval before execution",
                150,
                "Approval is required for this action",
                _is(ContextKeys.APPROVAL_REQUIRED, True),
            )
        )
    # DECISION plan-2026-09-24T045559-3e4eb3e5/D-002
    # Unconditional lowest-priority fallback. Do NOT gate this edge on the
    # tool selection: a gated edge BLOCKS `think` on a null/unknown tool, and
    # no PRE_TRANSITION or `act`-entry handler runs on a BLOCKED turn, so the
    # run burns the 3x loop ceiling. `act` handles a missing/unknown tool (an
    # empty batch for ParallelReact).
    transitions.append(_edge(AgentStates.ACT, act_description, 300))
    return transitions


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
