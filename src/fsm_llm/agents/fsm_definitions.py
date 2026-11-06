"""
Pre-built FSM definitions for agent patterns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from fsm_llm import typed_field_extraction
from fsm_llm.constants import CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT, has_internal_prefix

from . import meta_prompts
from .constants import (
    FRAMEWORK_ONLY_KEYS,
    META_ARTIFACT_TYPE_INTENTS,
    META_HANDLER_ONLY_KEYS,
    META_UNKNOWN_ARTIFACT_TYPE,
    META_UNKNOWN_ARTIFACT_TYPE_DESCRIPTION,
    NATIVE_FC_HANDLER_ONLY_KEYS,
    ADaPTStates,
    AgentStates,
    ContextKeys,
    DebateStates,
    Defaults,
    EvalOptStates,
    MakerCheckerStates,
    MetaBuilderStates,
    MetaBuildOutcome,
    MetaContextKeys,
    MetaDefaults,
    NativeFCContextKeys,
    NativeFCStates,
    NativeLoopEnd,
    OrchestratorStates,
    PlanExecuteStates,
    PromptChainStates,
    ReflexionStates,
    REWOOStates,
    SelfConsistencyStates,
    StopReason,
)
from .definitions import ChainStep
from .prompts import (
    build_adapt_field_instructions,
    build_assess_response_instructions,
    build_attempt_response_instructions,
    build_chain_output_response_instructions,
    build_chain_step_field_instructions,
    build_collect_response_instructions,
    build_combine_response_instructions,
    build_conclude_response_instructions,
    build_debate_conclude_response_instructions,
    build_debate_field_instructions,
    build_decompose_response_instructions,
    build_delegate_response_instructions,
    build_evalopt_field_instructions,
    build_evalopt_output_response_instructions,
    build_evaluate_field_instructions,
    build_execute_step_instructions,
    build_generate_response_instructions,
    build_maker_checker_field_instructions,
    build_maker_checker_output_response_instructions,
    build_orchestrate_response_instructions,
    build_orchestrator_field_instructions,
    build_orchestrator_synthesize_response_instructions,
    build_plan_steps_instructions,
    build_reflect_field_instructions,
    build_rewoo_plan_field_instructions,
    build_rewoo_solve_response_instructions,
    build_synthesize_response_instructions,
    build_think_extraction_instructions,
    build_think_terminate_instructions,
)
from .tools import ToolRegistry

if TYPE_CHECKING:  # annotation only: core owns the name, agents keep no copy
    from fsm_llm.definitions import TypedFieldType

# Truncate task_description for FSM metadata (name/description fields),
# while preserving the full text for prompt builders (semantic retrieval).
_MAX_DESC = Defaults.MAX_TASK_PREVIEW_LENGTH


def _finalize_fsm(
    name: str,
    task_description: str,
    default_description: str,
    initial_state: str,
    persona: str,
    states: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the top-level FSM definition dict shared by every builder here.

    Contract: returns ``{"name", "description", "initial_state", "persona",
    "states", "handler_only_keys"}`` where ``description`` is
    ``task_description[:_MAX_DESC]``, or ``default_description`` when that
    slice is empty, and ``handler_only_keys`` is :data:`FRAMEWORK_ONLY_KEYS`
    (D-051 of plan 06a5ec0a). ``states`` is stored by reference, not copied.
    Never raises.
    """
    return {
        "name": name,
        "description": task_description[:_MAX_DESC] or default_description,
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


# ---------------------------------------------------------------------------
# Orchestrator-Workers FSM
# ---------------------------------------------------------------------------


def build_orchestrator_fsm(
    task_description: str = "", context_keys: Sequence[str] = ()
) -> dict[str, Any]:
    """
    Build an Orchestrator-Workers FSM definition.

    The FSM implements task decomposition and delegation:
    orchestrate -> delegate -> collect -> synthesize (all collected)
                                       -> orchestrate (more work needed)

    ``subtasks`` and ``all_collected`` are explicit typed fields whose prompts
    show the task, ``worker_results`` and the caller's ``context_keys``
    (never ``agent_trace`` or ``skipped_subtasks``). Raises ``ValueError``
    like :func:`loop_field_context_keys` for a disallowed context key.
    """
    persona = (
        "You are an orchestrator AI agent that solves tasks by decomposing them "
        "into subtasks and delegating to workers. You analyze results, determine "
        "if more work is needed, and synthesize a final answer from all results."
    )
    fields = build_orchestrator_field_instructions()
    # DECISION plan-2026-09-29T103145-06a5ec0a/D-053: explicit narrowed
    # configs replace core's auto-minted ones (whole context, agent_trace
    # included). `subtasks` stays `any`: the delegator also takes a lone
    # string as one subtask. Do NOT add `skipped_subtasks` here (D-049).
    # DECISION plan-2026-09-30T062855-07ad3f8c/D-031: `orchestrate` and
    # `collect` make no state-level bulk call. Their typed fields cover every
    # key the run reads (`subtasks`, `all_collected`); the bulk call only
    # added `delegation_plan` and `reasoning`, which nothing read. Do NOT add
    # bulk instructions back as a second chance for a null field: a null
    # takes the priority-900 edge to `synthesize`.
    judged = (ContextKeys.WORKER_RESULTS, *context_keys)

    states = _by_id(
        _state(
            OrchestratorStates.ORCHESTRATE,
            "Decompose the task into subtasks for delegation",
            "Analyze the task and create a delegation plan",
            [
                _when(
                    OrchestratorStates.DELEGATE,
                    "Subtasks are ready for delegation",
                    100,
                    "Subtasks have been generated",
                    {"has_context": ContextKeys.SUBTASKS},
                ),
                _edge(
                    OrchestratorStates.SYNTHESIZE,
                    "Fallback: skip to synthesis if decomposition stalls",
                    900,
                    empty=True,
                ),
            ],
            required=[ContextKeys.SUBTASKS],
            fields=[
                _field(
                    ContextKeys.SUBTASKS, "any", fields[ContextKeys.SUBTASKS], judged
                )
            ],
            response_instructions=build_orchestrate_response_instructions(),
        ),
        _state(
            OrchestratorStates.DELEGATE,
            "Delegate subtasks to workers and collect results",
            "Execute worker_factory for each subtask",
            [
                _edge(
                    OrchestratorStates.COLLECT,
                    "Workers have finished, review results",
                    100,
                )
            ],
            response_instructions=build_delegate_response_instructions(),
        ),
        _state(
            OrchestratorStates.COLLECT,
            "Review worker results and decide if more work is needed",
            "Assess completeness of gathered results",
            [
                _when(
                    OrchestratorStates.SYNTHESIZE,
                    "All results collected, produce final answer",
                    10,
                    "All needed results are collected",
                    _is(ContextKeys.ALL_COLLECTED, True),
                ),
                _when(
                    OrchestratorStates.ORCHESTRATE,
                    "More work needed, decompose further",
                    300,
                    "More subtasks are needed",
                    _is(ContextKeys.ALL_COLLECTED, False),
                ),
                _edge(
                    OrchestratorStates.SYNTHESIZE,
                    "Fallback: synthesize with available results if decision stalls",
                    900,
                    empty=True,
                ),
            ],
            fields=[
                _field(
                    ContextKeys.ALL_COLLECTED,
                    "bool",
                    fields[ContextKeys.ALL_COLLECTED],
                    judged,
                )
            ],
            response_instructions=build_collect_response_instructions(),
        ),
        _state(
            OrchestratorStates.SYNTHESIZE,
            "Synthesize all worker results into a final answer",
            "Produce a comprehensive answer from all worker results",
            response_instructions=build_orchestrator_synthesize_response_instructions(),
        ),
    )

    return _finalize_fsm(
        "orchestrator_agent",
        task_description,
        "Orchestrator-Workers agent",
        OrchestratorStates.ORCHESTRATE,
        persona,
        states,
    )


# ---------------------------------------------------------------------------
# ADaPT FSM
# ---------------------------------------------------------------------------


def build_adapt_fsm(
    registry: ToolRegistry | None = None,
    task_description: str = "",
    max_depth: int = 3,
    context_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """
    Build an ADaPT (Adaptive Decomposition and Planning for Tasks) FSM definition.

    The FSM implements try-first, decompose-on-failure:
    attempt -> assess -> combine (success)
                      -> decompose (failure) -> combine (depth limit)
                                             -> [triggers recursive run()]

    ``attempt_result`` (str), ``attempt_succeeded`` (bool), ``subtasks``
    (list) and ``operator`` (str, optional) are explicit typed fields whose
    prompts show the task, the attempt (assess, decompose) and the caller's
    ``context_keys``, never ``agent_trace``. The three loop states make no
    state-level bulk call. Raises ``ValueError`` like
    :func:`loop_field_context_keys`.

    ``registry`` is reserved and unused: ADaPT runs no tool, so none is
    advertised and the FSM built with a registry equals the one built without.
    """
    # DECISION plan-2026-11-06T093029-0068b055/D-016: ``registry`` is kept only
    # because four in-repo tests pass it (removal is carried as F-AP-16c). Do
    # NOT make it affect the FSM again (D-007) and do NOT delete it without
    # editing those tests.
    persona = (
        "You are an adaptive AI agent that attempts tasks directly first. "
        "If the direct attempt is insufficient, you decompose the task into "
        "simpler subtasks and solve them recursively. "
        "Always try the direct approach before decomposing."
    )
    fields = build_adapt_field_instructions()
    # DECISION plan-2026-09-29T103145-06a5ec0a/D-053: explicit narrowed
    # configs replace core's auto-minted `any` ones (whole context). Do NOT
    # type `subtasks` `any`: the subtask executor needs a list, and `list`
    # also parses a JSON-string list.
    # DECISION plan-2026-09-30T062855-07ad3f8c/D-036: `attempt`, `assess` and
    # `decompose` make no state-level bulk call. The typed fields are every
    # key the run reads; the bulk calls only added `confidence`, `reasoning`
    # and `evaluation_feedback`, which nothing read, and were the one channel
    # through which a model reply could write any unset key (`final_answer`
    # included, which the answer prefers). `operator`, once filled only by
    # decompose's bulk call, is a typed optional field (a null means AND).
    # Do NOT add bulk instructions back as a second chance for a null field:
    # a null takes the state's unconditional edge (3e4eb3e5/D-002).
    judged = (ContextKeys.ATTEMPT_RESULT, *context_keys)

    def iteration_limit_edge() -> dict[str, Any]:
        return _when(
            ADaPTStates.COMBINE,
            "Iteration limit reached, produce best-effort answer",
            1,
            "Should terminate due to iteration limit",
            _is(ContextKeys.SHOULD_TERMINATE, True),
        )

    states = _by_id(
        _state(
            ADaPTStates.ATTEMPT,
            "Attempt to solve the task directly",
            "Give a direct attempt at solving the task",
            [
                iteration_limit_edge(),
                _edge(ADaPTStates.ASSESS, "Evaluate the attempt quality", 100),
            ],
            required=[ContextKeys.ATTEMPT_RESULT],
            fields=[
                # DECISION plan-2026-09-29T103145-06a5ec0a/D-057
                # `str`, not D-050's artifact `any`: an attempt is a short
                # direct answer that the answer path reads only as a str
                # (ADaPTAgent._extract_answer). Do NOT widen it to `any` for
                # symmetry: a native object would then never reach the answer.
                _field(
                    ContextKeys.ATTEMPT_RESULT,
                    "str",
                    fields[ContextKeys.ATTEMPT_RESULT],
                    context_keys,
                )
            ],
            response_instructions=build_attempt_response_instructions(),
        ),
        _state(
            ADaPTStates.ASSESS,
            "Assess whether the attempt succeeded",
            "Determine if the attempt is satisfactory or needs decomposition",
            [
                iteration_limit_edge(),
                _when(
                    ADaPTStates.COMBINE,
                    "Attempt succeeded, produce final answer",
                    10,
                    "The attempt was successful",
                    _is(ContextKeys.ATTEMPT_SUCCEEDED, True),
                ),
                _when(
                    ADaPTStates.DECOMPOSE,
                    "Attempt failed, decompose into subtasks",
                    150,
                    "Attempt failed and depth allows decomposition",
                    {
                        "and": [
                            _is(ContextKeys.ATTEMPT_SUCCEEDED, False),
                            {"<": [{"var": ContextKeys.CURRENT_DEPTH}, max_depth]},
                        ]
                    },
                ),
                _when(
                    ADaPTStates.COMBINE,
                    "Attempt failed but depth limit reached, use best effort",
                    200,
                    "Depth limit reached, force best effort",
                    _is(ContextKeys.ATTEMPT_SUCCEEDED, False),
                ),
                # DECISION plan-2026-09-24T045559-3e4eb3e5/D-002
                # Unconditional lowest-priority fallback. Do NOT remove it or gate it:
                # a missing attempt_succeeded would BLOCK `assess`, and no
                # PRE_TRANSITION limiter runs on a BLOCKED turn, so the run burns the
                # 3x loop ceiling. An unjudged attempt takes the best-effort path.
                _edge(
                    ADaPTStates.COMBINE,
                    "Fallback: use best effort if the assessment is unclear",
                    900,
                ),
            ],
            required=[ContextKeys.ATTEMPT_SUCCEEDED],
            fields=[
                _field(
                    ContextKeys.ATTEMPT_SUCCEEDED,
                    "bool",
                    fields[ContextKeys.ATTEMPT_SUCCEEDED],
                    judged,
                )
            ],
            response_instructions=build_assess_response_instructions(),
        ),
        _state(
            ADaPTStates.DECOMPOSE,
            "Decompose the task into simpler subtasks",
            "Break the task down for recursive solving",
            [
                iteration_limit_edge(),
                _when(
                    ADaPTStates.COMBINE,
                    "Subtasks defined, combine after recursive solving",
                    100,
                    "Subtasks have been generated",
                    {"has_context": ContextKeys.SUBTASKS},
                ),
                # DECISION plan-2026-09-24T045559-3e4eb3e5/D-002
                # Unconditional lowest-priority fallback. Do NOT remove it or gate it:
                # a missing subtasks list would BLOCK `decompose` (no PRE_TRANSITION
                # limiter runs on a BLOCKED turn). The subtask executor returns {}
                # without subtasks, so `combine` synthesizes the attempt alone.
                _edge(
                    ADaPTStates.COMBINE,
                    "Fallback: combine without subtasks if none were produced",
                    900,
                ),
            ],
            required=[ContextKeys.SUBTASKS],
            fields=[
                _field(
                    ContextKeys.SUBTASKS, "list", fields[ContextKeys.SUBTASKS], judged
                ),
                _field(
                    ContextKeys.OPERATOR,
                    "str",
                    fields[ContextKeys.OPERATOR],
                    (*judged, ContextKeys.SUBTASKS),
                    required=False,
                ),
            ],
            response_instructions=build_decompose_response_instructions(),
        ),
        _state(
            ADaPTStates.COMBINE,
            "Combine all results into the final answer",
            "Synthesize attempt results and subtask results",
            response_instructions=build_combine_response_instructions(),
        ),
    )

    return _finalize_fsm(
        "adapt_agent",
        task_description,
        "ADaPT agent with recursive decomposition",
        ADaPTStates.ATTEMPT,
        persona,
        states,
    )


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


# ---------------------------------------------------------------------------
# Reflexion FSM
# ---------------------------------------------------------------------------


def build_reflexion_fsm(
    registry: ToolRegistry,
    task_description: str = "",
    include_approval_state: bool = False,
    context_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """
    Build a Reflexion FSM definition from a tool registry.

    Extends the ReAct loop with evaluation and self-reflection:
    think -> act -> evaluate -> reflect (if failed) -> think (loop)
                              -> conclude (if passed)

    With *include_approval_state*, a gated call goes
    think -> await_approval -> act, the same state and priorities as
    :func:`build_react_fsm`.
    """
    persona = (
        "You are a reflective AI agent that solves tasks by using tools, "
        "then evaluating and critiquing your own work. "
        "After each tool action, you evaluate whether you have a good answer. "
        "If not, you reflect on what went wrong and try a different approach. "
        "Use your episodic memory to avoid repeating past mistakes."
    )

    think_instructions = build_think_extraction_instructions(
        registry, task_description=task_description
    )
    evaluate_fields = build_evaluate_field_instructions()
    reflect_fields = build_reflect_field_instructions()
    # REACT-01: reflect reads the verdict it critiques and the earlier
    # episodes, so each episode's reflection is its own.
    reflect_context = (
        ContextKeys.EVALUATION_FEEDBACK,
        ContextKeys.EVALUATION_SCORE,
        ContextKeys.EPISODIC_MEMORY,
    )

    states = _by_id(
        _state(
            ReflexionStates.THINK,
            "Reason about the task and select the next tool to use",
            "Analyze the task, episodic memory, and previous observations",
            _think_transitions(
                "Agent decided to terminate", include_approval=include_approval_state
            ),
            required=[
                ContextKeys.TOOL_NAME,
                ContextKeys.TOOL_INPUT,
                ContextKeys.SHOULD_TERMINATE,
            ],
            # D-009 of plan 06a5ec0a: typed per-field values only, no bulk
            # call; the episodic memory stays in the think prompts.
            fields=_think_field_extractions(
                think_instructions,
                context_keys=(ContextKeys.EPISODIC_MEMORY, *context_keys),
            ),
        ),
        _act_state(
            "Execute the selected tool and observe the result",
            "Run the tool and record the observation",
            ReflexionStates.EVALUATE,
            "Evaluate the result quality",
        ),
        _state(
            ReflexionStates.EVALUATE,
            "Assess whether gathered information is sufficient",
            "Evaluate answer quality and decide whether to reflect or conclude",
            [
                # DECISION plan-2026-09-24T091842-c1d5bfbc/D-008: a pass
                # concludes only with tool evidence or a forced stop. Do
                # NOT gate on evaluation_passed alone: a self-evaluated
                # memory answer then succeeds with zero tool calls. The
                # 900 fallback to reflect keeps evaluate from BLOCKING.
                # A forced stop alone also concludes here (P2-W3): do NOT
                # route it through reflect -> think, or max_iterations=1
                # cycles to the 3-turn ceiling (BudgetExhaustedError).
                _when(
                    ReflexionStates.CONCLUDE,
                    "Evaluation passed, produce final answer",
                    10,
                    "Passed on evidence, or forced stop",
                    {
                        "or": [
                            conclude_on_evidence_logic(ContextKeys.EVALUATION_PASSED),
                            _is(ContextKeys.MAX_ITERATIONS_REACHED, True),
                        ]
                    },
                ),
                _when(
                    ReflexionStates.REFLECT,
                    "Evaluation failed, reflect on approach",
                    300,
                    "Evaluation did not pass",
                    _is(ContextKeys.EVALUATION_PASSED, False),
                ),
                _edge(
                    ReflexionStates.REFLECT,
                    "Fallback: reflect if evaluation result unclear",
                    900,
                    empty=True,
                ),
            ],
            required=[ContextKeys.EVALUATION_SCORE, ContextKeys.EVALUATION_PASSED],
            # Typed per-field values only (D-009 of plan 06a5ec0a); an
            # intermediate state, so no Pass-2 prose (LOOP-09). With an
            # evaluation_fn the verdict is set on evaluate entry and these
            # extractions are skipped (skip-if-set).
            fields=[
                _field(
                    ContextKeys.EVALUATION_PASSED,
                    "bool",
                    evaluate_fields[ContextKeys.EVALUATION_PASSED],
                ),
                _field(
                    ContextKeys.EVALUATION_SCORE,
                    "float",
                    evaluate_fields[ContextKeys.EVALUATION_SCORE],
                    required=False,
                ),
                _field(
                    ContextKeys.EVALUATION_FEEDBACK,
                    "str",
                    evaluate_fields[ContextKeys.EVALUATION_FEEDBACK],
                    required=False,
                ),
            ],
        ),
        _state(
            ReflexionStates.REFLECT,
            "Self-critique and plan a revised approach",
            "Analyze what went wrong and generate lessons for next attempt",
            [
                _edge(
                    ReflexionStates.THINK, "Return to thinking with updated memory", 100
                )
            ],
            required=[ContextKeys.REFLECTION],
            fields=[
                _field(
                    ContextKeys.REFLECTION,
                    "str",
                    reflect_fields[ContextKeys.REFLECTION],
                    reflect_context,
                ),
                _field(
                    ContextKeys.LESSONS,
                    "str",
                    reflect_fields[ContextKeys.LESSONS],
                    reflect_context,
                    required=False,
                ),
            ],
        ),
        _conclude_state(
            build_conclude_response_instructions(refused_actions=include_approval_state)
        ),
    )
    if include_approval_state:
        states[ReflexionStates.AWAIT_APPROVAL] = _await_approval_state()

    return _finalize_fsm(
        "reflexion_agent",
        task_description,
        "Reflexion agent with self-evaluation",
        ReflexionStates.THINK,
        persona,
        states,
    )


# ---------------------------------------------------------------------------
# Plan-and-Execute FSM
# ---------------------------------------------------------------------------


def build_plan_execute_fsm(
    registry: ToolRegistry | None = None, task_description: str = ""
) -> dict[str, Any]:
    """
    Build a Plan-and-Execute FSM definition.

    Separates strategic planning from tactical execution:
    plan -> execute_step -> check_result -> synthesize (all done)
                                          -> replan (step failed) -> execute_step
                                          -> execute_step (next step)
    """
    persona = (
        "You are a strategic AI agent that solves tasks by first creating a plan, "
        "then executing each step methodically. "
        "If a step fails, you can revise the remaining plan. "
        "When all steps are complete, you synthesize results into a final answer."
    )

    # Typed per-field values only and no Pass-2 prose on the intermediate
    # states (D-009 of plan 06a5ec0a, PAT-01/02). `plan_steps` is a typed
    # list: an `any` config took a string, whose characters were then
    # counted as steps. check_result extracts nothing: the step checker
    # decides `step_failed` from the tool status on entry.
    step_context = (ContextKeys.CURRENT_STEP_DESCRIPTION, ContextKeys.STEP_RESULTS)
    replan_context = (ContextKeys.STEP_RESULTS, ContextKeys.PREVIOUS_PLAN_STEPS)
    step_instructions = build_execute_step_instructions(
        registry, task_description=task_description
    )
    has_tools = registry is not None and len(registry) > 0
    step_fields: list[dict[str, Any]] = [
        _field(
            ContextKeys.STEP_RESULT,
            "str",
            build_execute_step_instructions(
                registry, task_description=task_description, step_result=True
            ),
            step_context,
            # With tools the step result is the tool observation.
            required=not has_tools,
        )
    ]
    if has_tools:
        step_fields = [
            *_tool_selection_field_extractions(
                step_instructions, context_keys=loop_field_context_keys(step_context)
            ),
            *step_fields,
        ]

    states = _by_id(
        _state(
            PlanExecuteStates.PLAN,
            "Decompose the task into a sequence of steps",
            "Create an actionable plan to solve the task",
            [
                _when(
                    PlanExecuteStates.EXECUTE_STEP,
                    "Plan is ready, begin executing steps",
                    100,
                    "Plan steps have been generated",
                    {"has_context": ContextKeys.PLAN_STEPS},
                )
            ],
            required=[ContextKeys.PLAN_STEPS],
            fields=[
                _field(
                    ContextKeys.PLAN_STEPS,
                    "list",
                    build_plan_steps_instructions(
                        registry, task_description=task_description
                    ),
                )
            ],
        ),
        _state(
            PlanExecuteStates.EXECUTE_STEP,
            "Execute the current plan step",
            "Produce a result for the current step using tools or LLM",
            [
                _edge(
                    PlanExecuteStates.CHECK_RESULT,
                    "Step executed, check the result",
                    100,
                )
            ],
            fields=step_fields,
        ),
        _state(
            PlanExecuteStates.CHECK_RESULT,
            "Assess the step result and decide next action",
            "Determine if step succeeded and whether to continue, replan, or synthesize",
            [
                _when(
                    PlanExecuteStates.SYNTHESIZE,
                    "All steps complete, synthesize final answer",
                    10,
                    "All plan steps are complete",
                    _is(ContextKeys.ALL_STEPS_COMPLETE, True),
                ),
                _when(
                    PlanExecuteStates.REPLAN,
                    "Step failed, revise the plan",
                    150,
                    "The step did not succeed",
                    _is(ContextKeys.STEP_FAILED, True),
                ),
                _edge(
                    PlanExecuteStates.EXECUTE_STEP, "Proceed to the next plan step", 300
                ),
            ],
            extraction_instructions="",
        ),
        _state(
            PlanExecuteStates.REPLAN,
            "Revise the remaining plan after a step failure",
            "Incorporate lessons from the failure into a revised plan",
            [
                _edge(
                    PlanExecuteStates.EXECUTE_STEP,
                    "Resume execution with revised plan",
                    100,
                )
            ],
            fields=[
                _field(
                    ContextKeys.PLAN_STEPS,
                    "list",
                    build_plan_steps_instructions(
                        registry, task_description=task_description, replan=True
                    ),
                    replan_context,
                )
            ],
        ),
        _state(
            PlanExecuteStates.SYNTHESIZE,
            "Combine all step results into a final answer",
            "Produce a comprehensive answer from all step results",
            response_instructions=build_synthesize_response_instructions(),
        ),
    )

    return _finalize_fsm(
        "plan_execute_agent",
        task_description,
        "Plan-and-Execute agent",
        PlanExecuteStates.PLAN,
        persona,
        states,
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
        task_description,
        "ReAct agent with tool use",
        AgentStates.THINK,
        persona,
        states,
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


# ---------------------------------------------------------------------------
# Prompt Chain FSM
# ---------------------------------------------------------------------------


def build_prompt_chain_fsm(
    chain: list[ChainStep], task_description: str = ""
) -> dict[str, Any]:
    """
    Build a Prompt Chain FSM definition from a list of ChainStep objects.

    Creates a linear pipeline of states: step_0 -> step_1 -> ... -> output.
    Each step extracts one typed ``chain_step_result`` (any, D-050) whose prompt
    shows the task and ``chain_results``; the ChainStep's instructions word
    that field, and its ``response_instructions`` stay the step's reply
    (user-owned). A step after a gated step (one with ``validation_fn``) also
    has an edge to ``output`` taken when that gate failed.
    """
    persona = (
        "You are a methodical AI assistant that processes tasks through a "
        "structured pipeline of steps. Execute each step carefully, building "
        "on the results of previous steps."
    )

    states: dict[str, Any] = {}

    for i, step in enumerate(chain):
        state_id = f"{PromptChainStates.STEP_PREFIX}{i}"
        is_last = i == len(chain) - 1
        next_state = (
            PromptChainStates.OUTPUT
            if is_last
            else f"{PromptChainStates.STEP_PREFIX}{i + 1}"
        )

        transitions: list[dict[str, Any]] = []
        if i > 0 and chain[i - 1].validation_fn is not None:
            # The gate of step i-1 runs on entry here (PromptChainAgent).
            transitions.append(
                _when(
                    PromptChainStates.OUTPUT,
                    f"Stop: the gate of {chain[i - 1].name} failed",
                    50,
                    "The previous step's gate failed",
                    _is(ContextKeys.FORCED_STOP_REASON, StopReason.GATE_FAILED),
                )
            )
        transitions.append(
            _edge(next_state, f"Proceed to {'output' if is_last else step.name}", 100)
        )

        states[state_id] = _state(
            state_id,
            f"Step {i + 1}: {step.name}",
            step.name,
            transitions,
            response_instructions=step.response_instructions,
            fields=[
                _field(
                    ContextKeys.CHAIN_STEP_RESULT,
                    "any",
                    build_chain_step_field_instructions(
                        i,
                        step.name,
                        step.response_instructions,
                        step.extraction_instructions,
                    ),
                    (ContextKeys.CHAIN_RESULTS,),
                )
            ],
        )

    # Output (terminal) state
    states[PromptChainStates.OUTPUT] = _state(
        PromptChainStates.OUTPUT,
        "Produce the final output from the chain",
        "Synthesize all step results into a final answer",
        response_instructions=build_chain_output_response_instructions(),
    )

    return _finalize_fsm(
        "prompt_chain_agent",
        task_description,
        "Prompt chain agent",
        f"{PromptChainStates.STEP_PREFIX}0",
        persona,
        states,
    )


# ---------------------------------------------------------------------------
# Self-Consistency FSM
# ---------------------------------------------------------------------------


def build_self_consistency_fsm(task_description: str = "") -> dict[str, Any]:
    """
    Build a simple single-state FSM for self-consistency sampling.

    Each invocation generates one answer. The SelfConsistencyAgent runs
    this FSM multiple times with different temperatures and aggregates.
    ``generate`` is a terminal initial state, so core never extracts there:
    it has no extraction instructions, and the sample is the reply, which ends
    with an ``Answer:`` line for the vote.
    """
    persona = (
        "You are a precise AI assistant. Answer the given task directly and concisely."
    )

    states = _by_id(
        _state(
            SelfConsistencyStates.GENERATE,
            "Generate an answer to the task",
            "Produce a direct, complete answer to the task",
            response_instructions=build_generate_response_instructions(),
            extraction_instructions="",
        )
    )

    return _finalize_fsm(
        "self_consistency_sample",
        task_description,
        "Self-consistency single sample",
        SelfConsistencyStates.GENERATE,
        persona,
        states,
    )


# ---------------------------------------------------------------------------
# Debate FSM
# ---------------------------------------------------------------------------


def build_debate_fsm(
    task_description: str = "",
    proposer_persona: str = "",
    critic_persona: str = "",
    judge_persona: str = "",
    max_rounds: int = 3,
) -> dict[str, Any]:
    """
    Build a Debate FSM definition.

    Implements a multi-round debate loop:
    propose -> critique -> counter -> judge -> propose (loop)
                                             -> conclude (consensus or max rounds)

    Each debate state extracts one typed field (judge: the verdict and the
    ``consensus_reached`` decision) whose prompt shows the task and this
    round's earlier values; state-level extraction and response instructions
    are empty, so no context-free bulk call runs and only ``conclude`` speaks.
    """
    # Use proposer persona as the top-level FSM persona since it starts
    persona = proposer_persona or (
        "You are a thoughtful AI that explores questions through structured debate. "
        "Multiple perspectives are considered to arrive at the best answer."
    )
    fields = build_debate_field_instructions(
        proposer_persona, critic_persona, judge_persona, max_rounds
    )
    round_values = (
        ContextKeys.PROPOSITION,
        ContextKeys.CRITIQUE,
        ContextKeys.COUNTER_ARGUMENT,
    )

    def _debate_field(
        name: str, extra: tuple[str, ...], *, required: bool = True
    ) -> dict[str, Any]:
        return _field(name, "str", fields[name], extra, required=required)

    states = _by_id(
        _state(
            DebateStates.PROPOSE,
            "Generate or refine a proposition for the task",
            "Present a well-reasoned argument or answer",
            [_edge(DebateStates.CRITIQUE, "Proposition ready for critique", 100)],
            fields=[
                _debate_field(ContextKeys.PROPOSITION, (ContextKeys.DEBATE_ROUNDS,))
            ],
        ),
        _state(
            DebateStates.CRITIQUE,
            "Critically analyze the current proposition",
            "Identify weaknesses, gaps, and counterpoints",
            [
                _edge(
                    DebateStates.COUNTER,
                    "Critique complete, allow counter-argument",
                    100,
                )
            ],
            fields=[_debate_field(ContextKeys.CRITIQUE, round_values[:1])],
        ),
        _state(
            DebateStates.COUNTER,
            "Address the critique with counter-arguments",
            "Strengthen the proposition by addressing criticisms",
            [_edge(DebateStates.JUDGE, "Counter-argument ready for judgment", 100)],
            fields=[_debate_field(ContextKeys.COUNTER_ARGUMENT, round_values[:2])],
        ),
        _state(
            DebateStates.JUDGE,
            "Evaluate the debate exchange and decide next action",
            "Determine whether consensus has been reached",
            [
                # DECISION plan-2026-09-24T091842-c1d5bfbc/D-009
                # Do NOT drop the round disjunct: consensus_reached
                # is now extracted every round, and before plan
                # 5ed71cf2/D-005 transition evaluation overlaid
                # this turn's extracted False on the judge
                # handler's max-round True. The cap reads
                # current_round, never an extracted value: the
                # judge's PRE_TRANSITION handler runs after this
                # gate, and a judge that extracts null must not
                # leave the cap unread. current_round advances on
                # propose entry, so at the judge of round N it is
                # N: the cap is `>=` (plan 5ed71cf2/D-009).
                _when(
                    DebateStates.CONCLUDE,
                    "Consensus reached or max rounds hit",
                    10,
                    "Consensus reached or max rounds hit",
                    {
                        "or": [
                            _is(ContextKeys.CONSENSUS_REACHED, True),
                            {">=": [{"var": ContextKeys.CURRENT_ROUND}, max_rounds]},
                        ]
                    },
                ),
                _edge(DebateStates.PROPOSE, "Another round of debate needed", 300),
            ],
            fields=[
                # The verdict is recorded, never routed on: a null costs no
                # retry call.
                _debate_field(ContextKeys.JUDGE_VERDICT, round_values, required=False),
                # Narrowed to this round's values, its number and the debate
                # history (fix 21.1): the old bool helper showed the whole
                # context, agent_trace included.
                # DECISION plan-2026-09-29T103145-06a5ec0a/D-054
                # Do NOT drop debate_rounds from this list: without the
                # earlier rounds the live judge declined consensus every
                # round (forced_pass). It is bounded by num_rounds, unlike
                # agent_trace, which must stay out.
                _field(
                    ContextKeys.CONSENSUS_REACHED,
                    "bool",
                    fields[ContextKeys.CONSENSUS_REACHED],
                    (
                        *round_values,
                        ContextKeys.CURRENT_ROUND,
                        ContextKeys.DEBATE_ROUNDS,
                    ),
                ),
            ],
        ),
        _state(
            DebateStates.CONCLUDE,
            "Produce the final answer from the debate",
            "Synthesize the debate into a definitive answer",
            response_instructions=build_debate_conclude_response_instructions(),
            # Terminal: core never extracts here; the reply is the answer.
            extraction_instructions="",
        ),
    )

    return _finalize_fsm(
        "debate_agent",
        task_description,
        "Debate agent",
        DebateStates.PROPOSE,
        persona,
        states,
    )


# ---------------------------------------------------------------------------
# REWOO FSM
# ---------------------------------------------------------------------------


def build_rewoo_fsm(
    registry: ToolRegistry, task_description: str = "", context_keys: Sequence[str] = ()
) -> dict[str, Any]:
    """
    Build a REWOO FSM definition from a tool registry.

    The FSM implements the REWOO pattern with exactly 2 LLM calls:
    - plan_all: single LLM call generates a complete plan with #E1, #E2 refs
    - execute_plans: handler executes all tool calls sequentially (no LLM)
    - solve: single LLM call synthesizes the final answer from all evidence

    plan_all and execute_plans are silent (empty response instructions): only
    solve speaks. ``plan_blueprint`` is an explicit ``list`` field whose
    prompt shows the task and the caller's ``context_keys``, never
    ``agent_trace`` (DECISION D-053 of plan 06a5ec0a, anchored in
    :func:`build_adapt_fsm`). Raises ``ValueError`` like
    :func:`loop_field_context_keys`.
    """
    persona = (
        "You are a methodical AI agent that solves tasks by planning all tool calls "
        "upfront, then executing them, and finally synthesizing an answer from the "
        "collected evidence. You think before you act, and you plan completely."
    )

    states = _by_id(
        _state(
            REWOOStates.PLAN_ALL,
            "Create a complete plan of all tool calls needed",
            "Generate a full plan with tool calls and variable references",
            [
                _edge(
                    REWOOStates.EXECUTE_PLANS,
                    "Plan is complete, proceed to execution",
                    100,
                )
            ],
            required=[ContextKeys.PLAN_BLUEPRINT],
            # No state-level bulk call (D-031 of plan 07ad3f8c, anchored in
            # build_orchestrator_fsm): `plan_blueprint` is the one key read.
            fields=[
                _field(
                    ContextKeys.PLAN_BLUEPRINT,
                    "list",
                    build_rewoo_plan_field_instructions(
                        registry, task_description=task_description
                    ),
                    context_keys,
                )
            ],
        ),
        _state(
            REWOOStates.EXECUTE_PLANS,
            "Execute all planned tool calls sequentially",
            "Run every tool call from the plan, substituting variable references",
            [
                _edge(
                    REWOOStates.SOLVE,
                    "All plans executed, proceed to synthesize answer",
                    100,
                )
            ],
        ),
        _state(
            REWOOStates.SOLVE,
            "Synthesize the final answer from all evidence",
            "Combine the task, plan, and all tool results into a final answer",
            response_instructions=build_rewoo_solve_response_instructions(),
        ),
    )

    return _finalize_fsm(
        "rewoo_agent",
        task_description,
        "REWOO agent with upfront planning",
        REWOOStates.PLAN_ALL,
        persona,
        states,
    )


# ---------------------------------------------------------------------------
# Evaluator-Optimizer FSM
# ---------------------------------------------------------------------------


def build_evalopt_fsm(task_description: str = "") -> dict[str, Any]:
    """
    Build an Evaluator-Optimizer FSM definition.

    The FSM implements the generate-evaluate-refine loop:
    - generate: LLM generates output
    - evaluate: handler runs external evaluation function
    - refine: LLM refines based on feedback
    - output: terminal state with final answer

    generate and refine each extract one typed ``generated_output`` (any) and
    are silent (empty response instructions); ``evaluation_passed`` is written
    only by the evaluation handler, never extracted.
    """
    fields = build_evalopt_field_instructions()

    persona = (
        "You are an AI agent that produces high-quality outputs through iterative "
        "refinement. You generate output, receive evaluation feedback, and improve "
        "your output until it meets the required quality standards."
    )

    states = _by_id(
        _state(
            EvalOptStates.GENERATE,
            "Generate an initial output for the task",
            "Produce the best possible first attempt at the task",
            [
                _when(
                    EvalOptStates.EVALUATE,
                    "Output generated, proceed to evaluation",
                    100,
                    "Output has been generated",
                    {"has_context": ContextKeys.GENERATED_OUTPUT},
                ),
                # DECISION plan-2026-09-24T045559-3e4eb3e5/D-002
                # Unconditional lowest-priority fallback. Do NOT remove it or gate it:
                # a missing generated_output would BLOCK `generate` (no PRE_TRANSITION
                # limiter runs on a BLOCKED turn). `evaluate` judges the empty output,
                # then `refine` retries and the refinement cap / limiter end the loop.
                _edge(
                    EvalOptStates.EVALUATE,
                    "Fallback: evaluate even if no output was extracted",
                    900,
                ),
            ],
            fields=[
                _field(
                    ContextKeys.GENERATED_OUTPUT, "any", fields[EvalOptStates.GENERATE]
                )
            ],
        ),
        _state(
            EvalOptStates.EVALUATE,
            "Evaluate the generated output",
            "Run the external evaluation function on the current output",
            [
                _when(
                    EvalOptStates.OUTPUT,
                    "Evaluation passed, produce final output",
                    10,
                    "Output passed evaluation",
                    _is(ContextKeys.EVALUATION_PASSED, True),
                ),
                _when(
                    EvalOptStates.REFINE,
                    "Evaluation failed, refine the output",
                    300,
                    "Output did not pass evaluation",
                    _is(ContextKeys.EVALUATION_PASSED, False),
                ),
                _edge(
                    EvalOptStates.OUTPUT,
                    "Fallback: produce output if evaluation stalls",
                    900,
                    empty=True,
                ),
            ],
        ),
        _state(
            EvalOptStates.REFINE,
            "Refine the output based on evaluation feedback",
            "Improve the output by addressing specific feedback points",
            [
                _edge(
                    EvalOptStates.EVALUATE,
                    "Refined output ready for re-evaluation",
                    100,
                )
            ],
            fields=[
                _field(
                    ContextKeys.GENERATED_OUTPUT,
                    "any",
                    fields[EvalOptStates.REFINE],
                    (ContextKeys.PREVIOUS_OUTPUT, ContextKeys.REFINEMENT_FEEDBACK),
                )
            ],
        ),
        _state(
            EvalOptStates.OUTPUT,
            "Present the final evaluated output",
            "Extract and present the final answer",
            response_instructions=build_evalopt_output_response_instructions(),
        ),
    )

    return _finalize_fsm(
        "evalopt_agent",
        task_description,
        "Evaluator-Optimizer agent",
        EvalOptStates.GENERATE,
        persona,
        states,
    )


# ---------------------------------------------------------------------------
# Maker-Checker FSM
# ---------------------------------------------------------------------------


def build_maker_checker_fsm(
    maker_instructions: str,
    checker_instructions: str,
    task_description: str = "",
    max_revisions: int | None = None,
) -> dict[str, Any]:
    """
    Build a Maker-Checker FSM definition.

    The FSM implements the make-check-revise loop:
    - make: maker persona generates a draft
    - check: checker persona evaluates the draft
    - revise: maker persona revises based on feedback
    - output: terminal state with final answer

    ``max_revisions`` (optional): the check -> output edge also fires once
    ``revision_count`` reaches it. The count is bumped on check entry, so the
    cap holds when the checker's extraction is null (plan 5ed71cf2/D-009).
    """
    fields = build_maker_checker_field_instructions(
        maker_instructions, checker_instructions
    )
    draft = ContextKeys.DRAFT_OUTPUT
    # The checker sees the draft under judgment and the one it replaced.
    judged = (draft, ContextKeys.PREVIOUS_DRAFT)

    persona = (
        "You are an AI agent that produces high-quality outputs through a "
        "maker-checker process. You alternate between creating content and "
        "critically evaluating it to ensure the highest quality."
    )

    states = _by_id(
        _state(
            MakerCheckerStates.MAKE,
            "Maker generates a draft output",
            "Produce a high-quality draft following the maker instructions",
            [
                _edge(
                    MakerCheckerStates.CHECK,
                    "Draft complete, proceed to checker review",
                    100,
                )
            ],
            fields=[_field(draft, "any", fields[MakerCheckerStates.MAKE])],
        ),
        _state(
            MakerCheckerStates.CHECK,
            "Checker evaluates the draft",
            "Critically evaluate the draft against quality criteria",
            [
                # DECISION plan-2026-09-24T091842-c1d5bfbc/D-007: a check
                # turn at the budget ships the draft it just judged. Do NOT
                # rely on the forced checker_passed alone: before plan
                # 5ed71cf2/D-005 this turn's extracted False overlaid it,
                # so budget 1 needed 4 turns against a 3-turn ceiling and
                # raised BudgetExhaustedError; the limiter's flag is also
                # set after the gate may have read the verdict.
                # DECISION plan-2026-11-05T134105-5ed71cf2/D-009: the
                # revision cap is read here too, from revision_count
                # (bumped on check entry), never from a handler that a
                # null extraction skips. Do NOT rely on the check
                # PRE_TRANSITION handler to route the cap: it runs after
                # this gate.
                _when(
                    MakerCheckerStates.OUTPUT,
                    "Draft passed review, produce final output",
                    10,
                    "Checker approved, or the budget is spent",
                    {
                        "or": [
                            _is(ContextKeys.CHECKER_PASSED, True),
                            _is(ContextKeys.MAX_ITERATIONS_REACHED, True),
                            *(
                                []
                                if max_revisions is None
                                else [
                                    {
                                        ">=": [
                                            {"var": ContextKeys.REVISION_COUNT},
                                            max_revisions,
                                        ]
                                    }
                                ]
                            ),
                        ]
                    },
                ),
                _when(
                    MakerCheckerStates.REVISE,
                    "Draft needs revision based on feedback",
                    300,
                    "Checker found issues with the draft",
                    _is(ContextKeys.CHECKER_PASSED, False),
                ),
                # DECISION plan-2026-09-24T045559-3e4eb3e5/D-002
                # Unconditional lowest-priority fallback. Do NOT remove it or gate it:
                # a missing checker_passed would BLOCK `check`, and no PRE_TRANSITION
                # limiter runs on a BLOCKED turn (FB-01), so the run burns the 3x
                # loop ceiling. An unjudged draft is revised and re-checked.
                _edge(
                    MakerCheckerStates.REVISE,
                    "Fallback: revise if checker result unclear",
                    900,
                    empty=True,
                ),
            ],
            # DECISION plan-2026-09-29T103145-06a5ec0a/D-039
            # One call per field and turn: feedback first (the score and the
            # verdict prompts then show it), checker_passed optional (a null
            # costs no retry and falls to the D-002 revise edge, and
            # _track_revisions passes on the score). Do NOT put these keys
            # back in required_context_keys: core mints an untyped `any`
            # config per key with the whole context, agent_trace included.
            fields=[
                _field(
                    ContextKeys.CHECKER_FEEDBACK,
                    "str",
                    fields[ContextKeys.CHECKER_FEEDBACK],
                    judged,
                ),
                _field(
                    ContextKeys.QUALITY_SCORE,
                    "float",
                    fields[ContextKeys.QUALITY_SCORE],
                    judged,
                ),
                _field(
                    ContextKeys.CHECKER_PASSED,
                    "bool",
                    fields[ContextKeys.CHECKER_PASSED],
                    judged,
                    required=False,
                ),
            ],
        ),
        _state(
            MakerCheckerStates.REVISE,
            "Maker revises the draft based on checker feedback",
            "Address all checker feedback and produce an improved draft",
            [
                _edge(
                    MakerCheckerStates.CHECK,
                    "Revised draft ready for re-evaluation",
                    100,
                )
            ],
            fields=[
                _field(
                    draft,
                    "any",
                    fields[MakerCheckerStates.REVISE],
                    (ContextKeys.PREVIOUS_DRAFT, ContextKeys.CHECKER_FEEDBACK),
                )
            ],
        ),
        _state(
            MakerCheckerStates.OUTPUT,
            "Present the final reviewed output",
            "Extract and present the final answer",
            response_instructions=build_maker_checker_output_response_instructions(),
        ),
    )

    return _finalize_fsm(
        "maker_checker_agent",
        task_description,
        "Maker-Checker agent",
        MakerCheckerStates.MAKE,
        persona,
        states,
    )


# ---------------------------------------------------------------------------
# Meta-builder FSM
# ---------------------------------------------------------------------------


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


def build_meta_builder_fsm() -> dict[str, Any]:
    """Build the meta-builder FSM: classify, collect, build, build_failed, done.

    Contract: returns a v4.1 FSM definition dict that loads as
    ``FSMDefinition`` (no arguments; the artifact type, requirements and
    build request live in context, never in the definition). States
    (``MetaBuilderStates``):

    - ``classify`` (initial, silent): ``classification_extractions`` writes
      ``artifact_type`` (fsm, workflow, agent; fallback ``unknown``, which
      the driver's exit handler resolves to a type), reading
      ``latest_request`` as its context so a message-free step classifies
      too. ``-> build`` p10 when ``build_requested`` is True, else
      ``-> collect`` p900.
    - ``collect`` (speaking, no extraction): the reply asks for details and
      ends with the build prompt; it sees only ``artifact_type``,
      ``requirements`` and ``validation_errors``. ``-> classify`` p10 on
      ``type_switch``, ``-> build`` p20 on ``build_requested`` (a message
      that does both is reclassified first, then ``classify -> build``),
      otherwise BLOCKED (stay).
    - ``build`` (silent completion state): one structured completion over
      ``_build_messages`` with the response format in
      ``_build_response_format``, result in ``build_reply``. ``-> done`` p10
      when ``build_outcome == "valid"``, else ``-> build_failed`` p900.
    - ``build_failed`` (silent): ``-> classify`` p10 on ``type_switch``,
      ``-> build`` p20 on ``build_requested``, ``-> collect`` p900.
    - ``done`` (terminal, silent).

    The driver writes ``requirements``, ``latest_request``,
    ``build_requested``, ``type_switch`` and ``keyword_type`` before each
    turn; handlers write the build request on ``build`` entry and
    ``build_outcome``, ``artifact``, ``validation_errors`` and
    ``review_presentation`` from ``build_reply``. All of those are
    ``handler_only_keys`` (``META_HANDLER_ONLY_KEYS``). Never raises.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-011: five states with the
    # artifact type in context, and the build as a core completion state.
    # Do NOT split this into per-artifact-type states (three copies of the
    # same transitions), do NOT make the build an LLM call inside a handler
    # (a second call path beside Pass 1, outside rollback and the meter), and
    # do NOT replace the driver-computed gate keys (build_requested,
    # type_switch) with a per-turn classifier: JsonLogic has no phrase
    # predicate and a classifier per turn adds the calls A-ISSUE-011 removed.
    # The gate keys and the build verdict stay handler-only so the model
    # cannot forge them. See decisions.md D-011.
    states_ = MetaBuilderStates
    keys = MetaContextKeys
    build_requested = _flag_condition(
        keys.BUILD_REQUESTED, "The user asked to build the artifact now"
    )
    type_switch = _flag_condition(
        keys.TYPE_SWITCH, "The user asked for a different artifact type"
    )
    states = _by_id(
        _state(
            states_.CLASSIFY,
            "Decide which kind of artifact the user wants",
            "Classify the request as an FSM, a workflow or an agent",
            [
                _edge(
                    states_.BUILD,
                    "The user asked to build right away",
                    10,
                    build_requested,
                ),
                _edge(states_.COLLECT, "Collect the artifact's requirements", 900),
            ],
            extraction_instructions="",
            # DECISION plan-2026-10-01T093600-944e2692/D-035: the fallback is
            # `unknown`, not an artifact type. Do NOT make `fsm` the fallback
            # again: core stores the fallback at any confidence, so a vague
            # edit ("change the tool name") classified `fsm` at 0.1 flipped an
            # agent session to an FSM, and `fsm` meant both "an FSM" and "no
            # idea". The classify exit handler resolves `unknown` (and a
            # discarded low-confidence intent) to the previous type, else the
            # keyword type. See decisions.md D-035.
            classification_extractions=[
                {
                    "field_name": keys.ARTIFACT_TYPE,
                    "intents": [
                        {"name": name, "description": description}
                        for name, description in (
                            *META_ARTIFACT_TYPE_INTENTS,
                            (
                                META_UNKNOWN_ARTIFACT_TYPE,
                                META_UNKNOWN_ARTIFACT_TYPE_DESCRIPTION,
                            ),
                        )
                    ],
                    "fallback_intent": META_UNKNOWN_ARTIFACT_TYPE,
                    "confidence_threshold": MetaDefaults.TYPE_CONFIDENCE_THRESHOLD,
                    "context_keys": [keys.LATEST_REQUEST],
                }
            ],
        ),
        _state(
            states_.COLLECT,
            "Gather the artifact's requirements from the user",
            (
                "Acknowledge each detail, ask about what is unclear and "
                "invite the user to say 'build it'"
            ),
            [
                _edge(
                    states_.CLASSIFY,
                    "The user switched to another artifact type",
                    10,
                    type_switch,
                ),
                _edge(
                    states_.BUILD,
                    "The user asked to build the artifact",
                    20,
                    build_requested,
                ),
            ],
            response_instructions=meta_prompts.build_meta_collect_response_instructions(),
            extraction_instructions="",
            context_scope={
                "read_keys": [
                    keys.ARTIFACT_TYPE,
                    keys.REQUIREMENTS,
                    keys.VALIDATION_ERRORS,
                ]
            },
        ),
        _state(
            states_.BUILD,
            "Generate the artifact spec in one structured call",
            (
                "Ask the model for the whole artifact spec as JSON matching "
                "the per-type schema"
            ),
            [
                _edge(
                    states_.DONE,
                    "The assembled artifact validated",
                    10,
                    {
                        "description": "The build produced a valid artifact",
                        "requires_context_keys": [keys.BUILD_OUTCOME],
                        "logic": {
                            "===": [{"var": keys.BUILD_OUTCOME}, MetaBuildOutcome.VALID]
                        },
                    },
                ),
                _edge(
                    states_.BUILD_FAILED,
                    "The build did not produce a valid artifact",
                    900,
                ),
            ],
            completion={
                "response_format_key": keys.BUILD_RESPONSE_FORMAT,
                "messages_key": keys.BUILD_MESSAGES,
                "result_key": keys.BUILD_REPLY,
            },
        ),
        _state(
            states_.BUILD_FAILED,
            "The last build failed validation",
            (
                "Route the user's next message: build again, switch the "
                "artifact type, or collect more details"
            ),
            [
                _edge(
                    states_.CLASSIFY,
                    "The user switched to another artifact type",
                    10,
                    type_switch,
                ),
                _edge(
                    states_.BUILD, "The user asked to build again", 20, build_requested
                ),
                _edge(
                    states_.COLLECT, "Collect more details before the next build", 900
                ),
            ],
            extraction_instructions="",
        ),
        _state(
            states_.DONE,
            "The artifact is built and valid",
            "End the session with the valid artifact",
        ),
    )
    fsm = _finalize_fsm(
        "meta_builder",
        "",
        (
            "Meta-builder: classify the artifact type, collect requirements, "
            "build the artifact with one structured completion"
        ),
        states_.CLASSIFY,
        (
            "A concise assistant that helps users design FSM-LLM artifacts: "
            "FSMs, workflows and agents"
        ),
        states,
    )
    fsm["handler_only_keys"] = list(META_HANDLER_ONLY_KEYS)
    return fsm


# ---------------------------------------------------------------------------
# Native function-calling FSM
# ---------------------------------------------------------------------------


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
        task_description="",
        default_description=(
            "Native function calling: model turns with tools, tool turns, "
            "an optional forced final tool turn and repair turn"
        ),
        initial_state=states_.CALL_MODEL,
        persona="A capable AI agent that uses the provided tools",
        states=states,
    )
    fsm["handler_only_keys"] = [*fsm["handler_only_keys"], *NATIVE_FC_HANDLER_ONLY_KEYS]
    return fsm
