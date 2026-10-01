"""
Pre-built FSM definitions for agent patterns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, get_args

from fsm_llm.constants import has_internal_prefix

from .constants import (
    FRAMEWORK_ONLY_KEYS,
    ADaPTStates,
    AgentStates,
    ContextKeys,
    DebateStates,
    Defaults,
    EvalOptStates,
    MakerCheckerStates,
    OrchestratorStates,
    PlanExecuteStates,
    PromptChainStates,
    ReflexionStates,
    REWOOStates,
    SelfConsistencyStates,
    StopReason,
)
from .definitions import ChainStep
from .tools import ToolRegistry

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


# ---------------------------------------------------------------------------
# Orchestrator-Workers FSM
# ---------------------------------------------------------------------------


def build_orchestrator_fsm(
    task_description: str = "",
    context_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """
    Build an Orchestrator-Workers FSM definition.

    The FSM implements task decomposition and delegation:
    orchestrate -> delegate -> collect -> synthesize (all collected)
                                       -> orchestrate (more work needed)

    ``subtasks`` and ``all_collected`` are explicit typed fields whose prompts
    show the task, ``worker_results`` and the caller's ``context_keys``
    (never ``agent_trace`` or ``skipped_subtasks``). Raises ``ValueError``
    like :func:`_typed_field_extraction` for a disallowed context key.
    """
    from .prompts import (
        build_collect_response_instructions,
        build_delegate_response_instructions,
        build_orchestrate_response_instructions,
        build_orchestrator_field_instructions,
        build_orchestrator_synthesize_response_instructions,
    )

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

    states: dict[str, Any] = {
        OrchestratorStates.ORCHESTRATE: {
            "id": OrchestratorStates.ORCHESTRATE,
            "description": "Decompose the task into subtasks for delegation",
            "purpose": "Analyze the task and create a delegation plan",
            "required_context_keys": [ContextKeys.SUBTASKS],
            "extraction_instructions": "",
            "field_extractions": [
                _typed_field_extraction(
                    ContextKeys.SUBTASKS,
                    "any",
                    fields[ContextKeys.SUBTASKS],
                    extra_context_keys=judged,
                )
            ],
            "response_instructions": build_orchestrate_response_instructions(),
            "transitions": [
                {
                    "target_state": OrchestratorStates.DELEGATE,
                    "description": "Subtasks are ready for delegation",
                    "priority": 100,
                    "conditions": [
                        {
                            "description": "Subtasks have been generated",
                            "logic": {"has_context": ContextKeys.SUBTASKS},
                        }
                    ],
                },
                {
                    "target_state": OrchestratorStates.SYNTHESIZE,
                    "description": "Fallback: skip to synthesis if decomposition stalls",
                    "priority": 900,
                    "conditions": [],
                },
            ],
        },
        OrchestratorStates.DELEGATE: {
            "id": OrchestratorStates.DELEGATE,
            "description": "Delegate subtasks to workers and collect results",
            "purpose": "Execute worker_factory for each subtask",
            "response_instructions": build_delegate_response_instructions(),
            "transitions": [
                {
                    "target_state": OrchestratorStates.COLLECT,
                    "description": "Workers have finished, review results",
                    "priority": 100,
                }
            ],
        },
        OrchestratorStates.COLLECT: {
            "id": OrchestratorStates.COLLECT,
            "description": "Review worker results and decide if more work is needed",
            "purpose": "Assess completeness of gathered results",
            "extraction_instructions": "",
            "field_extractions": [
                _typed_field_extraction(
                    ContextKeys.ALL_COLLECTED,
                    "bool",
                    fields[ContextKeys.ALL_COLLECTED],
                    extra_context_keys=judged,
                )
            ],
            "response_instructions": build_collect_response_instructions(),
            "transitions": [
                {
                    "target_state": OrchestratorStates.SYNTHESIZE,
                    "description": "All results collected, produce final answer",
                    "priority": 10,
                    "conditions": [
                        {
                            "description": "All needed results are collected",
                            "logic": {"==": [{"var": ContextKeys.ALL_COLLECTED}, True]},
                        }
                    ],
                },
                {
                    "target_state": OrchestratorStates.ORCHESTRATE,
                    "description": "More work needed, decompose further",
                    "priority": 300,
                    "conditions": [
                        {
                            "description": "More subtasks are needed",
                            "logic": {
                                "==": [{"var": ContextKeys.ALL_COLLECTED}, False]
                            },
                        }
                    ],
                },
                {
                    "target_state": OrchestratorStates.SYNTHESIZE,
                    "description": "Fallback: synthesize with available results if decision stalls",
                    "priority": 900,
                    "conditions": [],
                },
            ],
        },
        OrchestratorStates.SYNTHESIZE: {
            "id": OrchestratorStates.SYNTHESIZE,
            "description": "Synthesize all worker results into a final answer",
            "purpose": "Produce a comprehensive answer from all worker results",
            "response_instructions": build_orchestrator_synthesize_response_instructions(),
            "transitions": [],
        },
    }

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
    :func:`_typed_field_extraction`.
    """
    from .prompts import (
        build_adapt_field_instructions,
        build_assess_response_instructions,
        build_attempt_response_instructions,
        build_combine_response_instructions,
        build_decompose_response_instructions,
    )

    persona = (
        "You are an adaptive AI agent that attempts tasks directly first. "
        "If the direct attempt is insufficient, you decompose the task into "
        "simpler subtasks and solve them recursively. "
        "Always try the direct approach before decomposing."
    )
    fields = build_adapt_field_instructions(registry, task_description=task_description)
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

    states: dict[str, Any] = {
        ADaPTStates.ATTEMPT: {
            "id": ADaPTStates.ATTEMPT,
            "description": "Attempt to solve the task directly",
            "purpose": "Give a direct attempt at solving the task",
            "required_context_keys": [ContextKeys.ATTEMPT_RESULT],
            "extraction_instructions": "",
            "field_extractions": [
                # DECISION plan-2026-09-29T103145-06a5ec0a/D-057
                # `str`, not D-050's artifact `any`: an attempt is a short
                # direct answer that the answer path reads only as a str
                # (ADaPTAgent._extract_answer). Do NOT widen it to `any` for
                # symmetry: a native object would then never reach the answer.
                _typed_field_extraction(
                    ContextKeys.ATTEMPT_RESULT,
                    "str",
                    fields[ContextKeys.ATTEMPT_RESULT],
                    extra_context_keys=context_keys,
                )
            ],
            "response_instructions": build_attempt_response_instructions(),
            "transitions": [
                {
                    "target_state": ADaPTStates.COMBINE,
                    "description": "Iteration limit reached, produce best-effort answer",
                    "priority": 1,
                    "conditions": [
                        {
                            "description": "Should terminate due to iteration limit",
                            "logic": {
                                "==": [{"var": ContextKeys.SHOULD_TERMINATE}, True]
                            },
                        }
                    ],
                },
                {
                    "target_state": ADaPTStates.ASSESS,
                    "description": "Evaluate the attempt quality",
                    "priority": 100,
                },
            ],
        },
        ADaPTStates.ASSESS: {
            "id": ADaPTStates.ASSESS,
            "description": "Assess whether the attempt succeeded",
            "purpose": "Determine if the attempt is satisfactory or needs decomposition",
            "required_context_keys": [ContextKeys.ATTEMPT_SUCCEEDED],
            "extraction_instructions": "",
            "field_extractions": [
                _typed_field_extraction(
                    ContextKeys.ATTEMPT_SUCCEEDED,
                    "bool",
                    fields[ContextKeys.ATTEMPT_SUCCEEDED],
                    extra_context_keys=judged,
                )
            ],
            "response_instructions": build_assess_response_instructions(),
            "transitions": [
                {
                    "target_state": ADaPTStates.COMBINE,
                    "description": "Iteration limit reached, produce best-effort answer",
                    "priority": 1,
                    "conditions": [
                        {
                            "description": "Should terminate due to iteration limit",
                            "logic": {
                                "==": [{"var": ContextKeys.SHOULD_TERMINATE}, True]
                            },
                        }
                    ],
                },
                {
                    "target_state": ADaPTStates.COMBINE,
                    "description": "Attempt succeeded, produce final answer",
                    "priority": 10,
                    "conditions": [
                        {
                            "description": "The attempt was successful",
                            "logic": {
                                "==": [{"var": ContextKeys.ATTEMPT_SUCCEEDED}, True]
                            },
                        }
                    ],
                },
                {
                    "target_state": ADaPTStates.DECOMPOSE,
                    "description": "Attempt failed, decompose into subtasks",
                    "priority": 150,
                    "conditions": [
                        {
                            "description": "Attempt failed and depth allows decomposition",
                            "logic": {
                                "and": [
                                    {
                                        "==": [
                                            {"var": ContextKeys.ATTEMPT_SUCCEEDED},
                                            False,
                                        ]
                                    },
                                    {
                                        "<": [
                                            {"var": ContextKeys.CURRENT_DEPTH},
                                            max_depth,
                                        ]
                                    },
                                ]
                            },
                        }
                    ],
                },
                {
                    "target_state": ADaPTStates.COMBINE,
                    "description": "Attempt failed but depth limit reached, use best effort",
                    "priority": 200,
                    "conditions": [
                        {
                            "description": "Depth limit reached, force best effort",
                            "logic": {
                                "==": [{"var": ContextKeys.ATTEMPT_SUCCEEDED}, False]
                            },
                        }
                    ],
                },
                # DECISION plan-2026-09-24T045559-3e4eb3e5/D-002
                # Unconditional lowest-priority fallback. Do NOT remove it or gate it:
                # a missing attempt_succeeded would BLOCK `assess`, and no
                # PRE_TRANSITION limiter runs on a BLOCKED turn, so the run burns the
                # 3x loop ceiling. An unjudged attempt takes the best-effort path.
                {
                    "target_state": ADaPTStates.COMBINE,
                    "description": "Fallback: use best effort if the assessment is unclear",
                    "priority": 900,
                },
            ],
        },
        ADaPTStates.DECOMPOSE: {
            "id": ADaPTStates.DECOMPOSE,
            "description": "Decompose the task into simpler subtasks",
            "purpose": "Break the task down for recursive solving",
            "required_context_keys": [ContextKeys.SUBTASKS],
            "extraction_instructions": "",
            "field_extractions": [
                _typed_field_extraction(
                    ContextKeys.SUBTASKS,
                    "list",
                    fields[ContextKeys.SUBTASKS],
                    extra_context_keys=judged,
                ),
                _typed_field_extraction(
                    ContextKeys.OPERATOR,
                    "str",
                    fields[ContextKeys.OPERATOR],
                    extra_context_keys=(*judged, ContextKeys.SUBTASKS),
                    required=False,
                ),
            ],
            "response_instructions": build_decompose_response_instructions(),
            "transitions": [
                {
                    "target_state": ADaPTStates.COMBINE,
                    "description": "Iteration limit reached, produce best-effort answer",
                    "priority": 1,
                    "conditions": [
                        {
                            "description": "Should terminate due to iteration limit",
                            "logic": {
                                "==": [{"var": ContextKeys.SHOULD_TERMINATE}, True]
                            },
                        }
                    ],
                },
                {
                    "target_state": ADaPTStates.COMBINE,
                    "description": "Subtasks defined, combine after recursive solving",
                    "priority": 100,
                    "conditions": [
                        {
                            "description": "Subtasks have been generated",
                            "logic": {"has_context": ContextKeys.SUBTASKS},
                        }
                    ],
                },
                # DECISION plan-2026-09-24T045559-3e4eb3e5/D-002
                # Unconditional lowest-priority fallback. Do NOT remove it or gate it:
                # a missing subtasks list would BLOCK `decompose` (no PRE_TRANSITION
                # limiter runs on a BLOCKED turn). The subtask executor returns {}
                # without subtasks, so `combine` synthesizes the attempt alone.
                {
                    "target_state": ADaPTStates.COMBINE,
                    "description": "Fallback: combine without subtasks if none were produced",
                    "priority": 900,
                },
            ],
        },
        ADaPTStates.COMBINE: {
            "id": ADaPTStates.COMBINE,
            "description": "Combine all results into the final answer",
            "purpose": "Synthesize attempt results and subtask results",
            "response_instructions": build_combine_response_instructions(),
            "transitions": [],
        },
    }

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
    context_keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Typed ``field_extractions`` for a think state's tool selection.

    Contract: ``think_instructions`` is the think prompt (tools, examples,
    rules); returns raw dicts for ``State(field_extractions=...)``:
    ``tool_name`` as ``str`` (omitted when ``include_tool_name`` is False, i.e.
    the classifier owns it) and ``tool_input`` as ``dict``. ``context_keys``,
    when given, narrows each prompt's context to those keys (see
    :func:`_loop_field_context_keys`). Never raises.

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
    narrowed = {} if context_keys is None else {"context_keys": list(context_keys)}
    return [
        {
            "field_name": name,
            "field_type": field_type,
            "extraction_instructions": f"Extract the '{name}' field. {think_instructions}",
            **narrowed,
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
TypedFieldType = Literal["str", "float", "list", "bool", "any"]

# DECISION plan-2026-09-29T103145-06a5ec0a/D-035
# Keys of core's extraction reply envelopes (single-field
# `{field_name, value, confidence, reasoning}`, bulk `{extracted_data, ...}`).
# Do NOT name a typed field after one: the model answers the envelope key with
# its own meta-commentary, which then fills the field (the step-15 `reasoning`
# field fed that text into every later think prompt, D-034).
_EXTRACTION_ENVELOPE_KEYS: frozenset[str] = frozenset(
    {"reasoning", "confidence", "value", "field_name", "extracted_data"}
)

# Context every loop-field prompt sees: the task and the tool observations.
_LOOP_FIELD_CONTEXT_KEYS: tuple[str, ...] = (
    ContextKeys.TASK,
    ContextKeys.OBSERVATIONS,
)


def _loop_field_context_keys(extra_context_keys: Sequence[str] = ()) -> list[str]:
    """The ``context_keys`` of a loop field prompt: ``task``, ``observations``,
    then ``extra_context_keys`` (order kept, duplicates dropped).

    Shared by :func:`_typed_field_extraction` and the think builders' tool
    selection configs. Raises ``ValueError`` for an extra key that is
    ``agent_trace`` or internal-prefixed (core reads a listed key from raw
    context, with no internal-key filter).
    """
    bad = [
        key
        for key in extra_context_keys
        if key == ContextKeys.AGENT_TRACE or has_internal_prefix(key)
    ]
    if bad:
        raise ValueError(f"context keys not allowed in a field prompt: {bad}")
    return list(dict.fromkeys((*_LOOP_FIELD_CONTEXT_KEYS, *extra_context_keys)))


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
    there). Raises ``ValueError`` like :func:`_loop_field_context_keys`.
    """
    from .prompts import build_think_terminate_instructions

    extra = (ContextKeys.AGENT_FEEDBACK, *context_keys)
    return [
        *_tool_selection_field_extractions(
            think_instructions,
            include_tool_name=include_tool_name,
            context_keys=_loop_field_context_keys(extra),
        ),
        _typed_field_extraction(
            ContextKeys.SHOULD_TERMINATE,
            "bool",
            build_think_terminate_instructions(),
            extra_context_keys=extra,
            required=False,
        ),
    ]


def _typed_field_extraction(
    field_name: str,
    field_type: TypedFieldType,
    instructions: str,
    *,
    extra_context_keys: Sequence[str] = (),
    required: bool = True,
) -> dict[str, Any]:
    """One typed ``field_extraction`` for a loop value an agent state produces.

    Contract: returns a raw dict for ``State(field_extractions=[...])``
    declaring ``field_name`` as ``field_type`` (``str``, ``float``, ``list``,
    ``bool``, or ``any`` for a whole generated artifact; core coerces and
    rejects mismatches). The prompt context is
    narrowed to ``task``, ``observations`` and ``extra_context_keys`` (in that
    order, duplicates dropped), and the instructions tell the model to read the
    value from the task and those keys (LOOP-11). ``required`` maps to the core flag (a null required field costs
    one retry call). Raises ``ValueError`` for another ``field_type``, or for
    an extra key that is ``agent_trace`` or internal-prefixed (core reads a
    listed key from raw context, with no internal-key filter), and for a
    ``field_name`` in :data:`_EXTRACTION_ENVELOPE_KEYS`.

    Pair it with an empty state-level ``extraction_instructions`` (D-009 of
    plan 06a5ec0a): core then runs only these per-field calls, each on its own
    narrowed context, and makes no bulk call for the state.

    # DECISION plan-2026-09-29T103145-06a5ec0a/D-017
    Do NOT drop ``context_keys`` here (``None`` dumps all visible context,
    including the unbounded ``agent_trace``, into every per-field prompt), and
    do NOT cap or rename ``agent_trace`` instead: it feeds
    ``AgentResult.trace`` and marks the FSM as agent-managed for core.
    """
    if field_type not in get_args(TypedFieldType):
        raise ValueError(f"unsupported typed field type: {field_type!r}")
    if field_name in _EXTRACTION_ENVELOPE_KEYS:
        raise ValueError(
            f"typed field name {field_name!r} is an extraction envelope key; "
            "pick another name"
        )
    context_keys = _loop_field_context_keys(extra_context_keys)
    return {
        "field_name": field_name,
        "field_type": field_type,
        "extraction_instructions": (
            f"Extract the '{field_name}' field ({field_type}) from the task and "
            f"the {', '.join(repr(k) for k in context_keys)} values in the "
            f"'Already extracted:' context. {instructions}"
        ),
        "context_keys": context_keys,
        "required": required,
    }


def _conclude_on_evidence_logic(
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


def _approval_think_transition() -> dict[str, Any]:
    """The ``think -> await_approval`` edge shared by every approval-gated FSM.

    Contract: returns a fresh transition dict at priority 150, which beats the
    D-002 ``think -> act`` fallback (300) and loses to ``think -> conclude``
    (10). Never raises.
    """
    return {
        "target_state": AgentStates.AWAIT_APPROVAL,
        "description": "Action requires human approval before execution",
        "priority": 150,
        "conditions": [
            {
                "description": "Approval is required for this action",
                "logic": {"==": [{"var": ContextKeys.APPROVAL_REQUIRED}, True]},
            }
        ],
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
    return {
        "id": AgentStates.AWAIT_APPROVAL,
        "description": "Waiting for human approval before executing action",
        "purpose": "Hold the selected action until the approval decision is set",
        # LOOP-09: an intermediate state; the driver asks the approver, so no
        # Pass-2 prose (core skips Pass 2 on empty instructions).
        "response_instructions": "",
        "transitions": [
            {
                "target_state": AgentStates.CONCLUDE,
                "description": "Terminate when framework signals completion",
                "priority": 1,
                "conditions": [
                    {
                        "description": "Framework or agent decided to terminate",
                        "logic": {"==": [{"var": ContextKeys.SHOULD_TERMINATE}, True]},
                    }
                ],
            },
            {
                "target_state": AgentStates.ACT,
                "description": "Approval granted, proceed with action",
                "priority": 10,
                "conditions": [
                    {
                        "description": "User approved the action",
                        "logic": {"==": [{"var": ContextKeys.APPROVAL_GRANTED}, True]},
                    }
                ],
            },
            {
                "target_state": AgentStates.THINK,
                "description": "Approval denied, reconsider approach",
                "priority": 300,
                "conditions": [
                    {
                        "description": "User denied the action",
                        "logic": {"==": [{"var": ContextKeys.APPROVAL_GRANTED}, False]},
                    }
                ],
            },
        ],
    }


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
    from .prompts import (
        build_conclude_response_instructions,
        build_evaluate_field_instructions,
        build_reflect_field_instructions,
        build_think_extraction_instructions,
    )

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

    states: dict[str, Any] = {
        ReflexionStates.THINK: {
            "id": ReflexionStates.THINK,
            "description": "Reason about the task and select the next tool to use",
            "purpose": "Analyze the task, episodic memory, and previous observations",
            "required_context_keys": [
                ContextKeys.TOOL_NAME,
                ContextKeys.TOOL_INPUT,
                ContextKeys.SHOULD_TERMINATE,
            ],
            # D-009 of plan 06a5ec0a: typed per-field values only, no bulk
            # call; the episodic memory stays in the think prompts.
            "extraction_instructions": "",
            "field_extractions": _think_field_extractions(
                think_instructions,
                context_keys=(ContextKeys.EPISODIC_MEMORY, *context_keys),
            ),
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": ReflexionStates.CONCLUDE,
                    "description": "Agent decided to terminate",
                    "priority": 10,
                    "conditions": [
                        {
                            # DECISION plan-2026-09-24T091842-c1d5bfbc/D-008
                            # Evidence guard, see _conclude_on_evidence_logic.
                            "description": (
                                "Agent decided to terminate AND a tool has run "
                                "or termination is forced"
                            ),
                            "logic": _conclude_on_evidence_logic(),
                        }
                    ],
                },
                # DECISION plan-2026-09-24T091842-c1d5bfbc/D-005
                # The approval edge (150) must beat the fallback below (300).
                # Do NOT drop it while the agent registers the HITL gate: without
                # it think -> act runs execute_tool before the driver asks, and
                # every gated call burns an act/evaluate/reflect cycle on the
                # D-004 refusal.
                *([_approval_think_transition()] if include_approval_state else []),
                # DECISION plan-2026-09-24T045559-3e4eb3e5/D-002
                # Unconditional lowest-priority fallback. Do NOT gate this edge on the
                # tool selection: a gated edge BLOCKS `think` on a null/unknown tool, and
                # no PRE_TRANSITION or `act`-entry handler runs on a BLOCKED turn, so the
                # run burns the 3x loop ceiling. `act` handles a missing/unknown tool.
                {
                    "target_state": ReflexionStates.ACT,
                    "description": "Execute the selected tool",
                    "priority": 300,
                },
            ],
        },
        ReflexionStates.ACT: {
            "id": ReflexionStates.ACT,
            "description": "Execute the selected tool and observe the result",
            "purpose": "Run the tool and record the observation",
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": ReflexionStates.CONCLUDE,
                    "description": "Terminate when framework signals completion",
                    "priority": 1,
                    "conditions": [
                        {
                            "description": (
                                "Framework/agent decided to terminate AND a tool "
                                "has run or termination is forced"
                            ),
                            "logic": _conclude_on_evidence_logic(),
                        }
                    ],
                },
                {
                    "target_state": ReflexionStates.EVALUATE,
                    "description": "Evaluate the result quality",
                    "priority": 900,
                },
            ],
        },
        ReflexionStates.EVALUATE: {
            "id": ReflexionStates.EVALUATE,
            "description": "Assess whether gathered information is sufficient",
            "purpose": "Evaluate answer quality and decide whether to reflect or conclude",
            "required_context_keys": [
                ContextKeys.EVALUATION_SCORE,
                ContextKeys.EVALUATION_PASSED,
            ],
            # Typed per-field values only (D-009 of plan 06a5ec0a); an
            # intermediate state, so no Pass-2 prose (LOOP-09). With an
            # evaluation_fn the verdict is set on evaluate entry and these
            # extractions are skipped (skip-if-set).
            "extraction_instructions": "",
            "field_extractions": [
                _typed_field_extraction(
                    ContextKeys.EVALUATION_PASSED,
                    "bool",
                    evaluate_fields[ContextKeys.EVALUATION_PASSED],
                ),
                _typed_field_extraction(
                    ContextKeys.EVALUATION_SCORE,
                    "float",
                    evaluate_fields[ContextKeys.EVALUATION_SCORE],
                    required=False,
                ),
                _typed_field_extraction(
                    ContextKeys.EVALUATION_FEEDBACK,
                    "str",
                    evaluate_fields[ContextKeys.EVALUATION_FEEDBACK],
                    required=False,
                ),
            ],
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": ReflexionStates.CONCLUDE,
                    "description": "Evaluation passed, produce final answer",
                    "priority": 10,
                    # DECISION plan-2026-09-24T091842-c1d5bfbc/D-008: a pass
                    # concludes only with tool evidence or a forced stop. Do
                    # NOT gate on evaluation_passed alone: a self-evaluated
                    # memory answer then succeeds with zero tool calls. The
                    # 900 fallback to reflect keeps evaluate from BLOCKING.
                    # A forced stop alone also concludes here (P2-W3): do NOT
                    # route it through reflect -> think, or max_iterations=1
                    # cycles to the 3-turn ceiling (BudgetExhaustedError).
                    "conditions": [
                        {
                            "description": "Passed on evidence, or forced stop",
                            "logic": {
                                "or": [
                                    _conclude_on_evidence_logic(
                                        ContextKeys.EVALUATION_PASSED
                                    ),
                                    {
                                        "==": [
                                            {"var": ContextKeys.MAX_ITERATIONS_REACHED},
                                            True,
                                        ]
                                    },
                                ]
                            },
                        }
                    ],
                },
                {
                    "target_state": ReflexionStates.REFLECT,
                    "description": "Evaluation failed, reflect on approach",
                    "priority": 300,
                    "conditions": [
                        {
                            "description": "Evaluation did not pass",
                            "logic": {
                                "==": [{"var": ContextKeys.EVALUATION_PASSED}, False]
                            },
                        }
                    ],
                },
                {
                    "target_state": ReflexionStates.REFLECT,
                    "description": "Fallback: reflect if evaluation result unclear",
                    "priority": 900,
                    "conditions": [],
                },
            ],
        },
        ReflexionStates.REFLECT: {
            "id": ReflexionStates.REFLECT,
            "description": "Self-critique and plan a revised approach",
            "purpose": "Analyze what went wrong and generate lessons for next attempt",
            "required_context_keys": [ContextKeys.REFLECTION],
            "extraction_instructions": "",
            "field_extractions": [
                _typed_field_extraction(
                    ContextKeys.REFLECTION,
                    "str",
                    reflect_fields[ContextKeys.REFLECTION],
                    extra_context_keys=reflect_context,
                ),
                _typed_field_extraction(
                    ContextKeys.LESSONS,
                    "str",
                    reflect_fields[ContextKeys.LESSONS],
                    extra_context_keys=reflect_context,
                    required=False,
                ),
            ],
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": ReflexionStates.THINK,
                    "description": "Return to thinking with updated memory",
                    "priority": 100,
                }
            ],
        },
        ReflexionStates.CONCLUDE: {
            "id": ReflexionStates.CONCLUDE,
            "description": "Formulate and present the final answer",
            "purpose": "Synthesize all observations into a complete answer",
            "response_instructions": build_conclude_response_instructions(
                refused_actions=include_approval_state
            ),
            "transitions": [],
        },
    }
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
    registry: ToolRegistry | None = None,
    task_description: str = "",
) -> dict[str, Any]:
    """
    Build a Plan-and-Execute FSM definition.

    Separates strategic planning from tactical execution:
    plan -> execute_step -> check_result -> synthesize (all done)
                                          -> replan (step failed) -> execute_step
                                          -> execute_step (next step)
    """
    from .prompts import (
        build_execute_step_instructions,
        build_plan_steps_instructions,
        build_synthesize_response_instructions,
    )

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
        _typed_field_extraction(
            ContextKeys.STEP_RESULT,
            "str",
            build_execute_step_instructions(
                registry, task_description=task_description, step_result=True
            ),
            extra_context_keys=step_context,
            # With tools the step's result is the tool observation.
            required=not has_tools,
        )
    ]
    if has_tools:
        step_fields = [
            *_tool_selection_field_extractions(
                step_instructions,
                context_keys=_loop_field_context_keys(step_context),
            ),
            *step_fields,
        ]

    states: dict[str, Any] = {
        PlanExecuteStates.PLAN: {
            "id": PlanExecuteStates.PLAN,
            "description": "Decompose the task into a sequence of steps",
            "purpose": "Create an actionable plan to solve the task",
            "required_context_keys": [ContextKeys.PLAN_STEPS],
            "extraction_instructions": "",
            "field_extractions": [
                _typed_field_extraction(
                    ContextKeys.PLAN_STEPS,
                    "list",
                    build_plan_steps_instructions(
                        registry, task_description=task_description
                    ),
                )
            ],
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": PlanExecuteStates.EXECUTE_STEP,
                    "description": "Plan is ready, begin executing steps",
                    "priority": 100,
                    "conditions": [
                        {
                            "description": "Plan steps have been generated",
                            "logic": {"has_context": ContextKeys.PLAN_STEPS},
                        }
                    ],
                }
            ],
        },
        PlanExecuteStates.EXECUTE_STEP: {
            "id": PlanExecuteStates.EXECUTE_STEP,
            "description": "Execute the current plan step",
            "purpose": "Produce a result for the current step using tools or LLM",
            "extraction_instructions": "",
            "field_extractions": step_fields,
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": PlanExecuteStates.CHECK_RESULT,
                    "description": "Step executed, check the result",
                    "priority": 100,
                }
            ],
        },
        PlanExecuteStates.CHECK_RESULT: {
            "id": PlanExecuteStates.CHECK_RESULT,
            "description": "Assess the step result and decide next action",
            "purpose": "Determine if step succeeded and whether to continue, replan, or synthesize",
            "extraction_instructions": "",
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": PlanExecuteStates.SYNTHESIZE,
                    "description": "All steps complete, synthesize final answer",
                    "priority": 10,
                    "conditions": [
                        {
                            "description": "All plan steps are complete",
                            "logic": {
                                "==": [{"var": ContextKeys.ALL_STEPS_COMPLETE}, True]
                            },
                        }
                    ],
                },
                {
                    "target_state": PlanExecuteStates.REPLAN,
                    "description": "Step failed, revise the plan",
                    "priority": 150,
                    "conditions": [
                        {
                            "description": "The step did not succeed",
                            "logic": {"==": [{"var": ContextKeys.STEP_FAILED}, True]},
                        }
                    ],
                },
                {
                    "target_state": PlanExecuteStates.EXECUTE_STEP,
                    "description": "Proceed to the next plan step",
                    "priority": 300,
                },
            ],
        },
        PlanExecuteStates.REPLAN: {
            "id": PlanExecuteStates.REPLAN,
            "description": "Revise the remaining plan after a step failure",
            "purpose": "Incorporate lessons from the failure into a revised plan",
            "extraction_instructions": "",
            "field_extractions": [
                _typed_field_extraction(
                    ContextKeys.PLAN_STEPS,
                    "list",
                    build_plan_steps_instructions(
                        registry, task_description=task_description, replan=True
                    ),
                    extra_context_keys=replan_context,
                )
            ],
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": PlanExecuteStates.EXECUTE_STEP,
                    "description": "Resume execution with revised plan",
                    "priority": 100,
                }
            ],
        },
        PlanExecuteStates.SYNTHESIZE: {
            "id": PlanExecuteStates.SYNTHESIZE,
            "description": "Combine all step results into a final answer",
            "purpose": "Produce a comprehensive answer from all step results",
            "response_instructions": build_synthesize_response_instructions(),
            "transitions": [],
        },
    }

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
    from .prompts import (
        build_conclude_response_instructions,
        build_think_extraction_instructions,
    )

    # Build persona with tool awareness
    persona = (
        "You are a methodical AI agent that solves tasks by using tools step by step. "
        "Think carefully before each action. Review previous observations before deciding. "
        "Terminate when you have gathered enough information to answer the task."
    )

    # Think state transitions
    # NOTE: the lowest passing priority number wins in TransitionEvaluator.
    # Terminal transitions (conclude) get lowest priority numbers.
    think_transitions: list[dict[str, Any]] = [
        {
            "target_state": AgentStates.CONCLUDE,
            "description": "Task can be answered (a tool has run, or termination is forced)",
            "priority": 10,
            "conditions": [
                {
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
                    "description": (
                        "Agent decided to terminate AND a tool has run or "
                        "termination is forced"
                    ),
                    "logic": _conclude_on_evidence_logic(),
                }
            ],
        },
    ]

    if include_approval_state:
        think_transitions.append(_approval_think_transition())

    # DECISION plan-2026-09-24T045559-3e4eb3e5/D-002
    # Unconditional lowest-priority fallback. Do NOT gate this edge on the
    # tool selection: a gated edge BLOCKS `think` on a null/unknown tool, and
    # no PRE_TRANSITION or `act`-entry handler runs on a BLOCKED turn, so the
    # run burns the 3x loop ceiling. `act` handles a missing/unknown tool.
    think_transitions.append(
        {
            "target_state": AgentStates.ACT,
            "description": "Execute the selected tool",
            "priority": 300,
        }
    )

    think_instructions = build_think_extraction_instructions(
        registry, task_description=task_description
    )
    think_state: dict[str, Any] = {
        "id": AgentStates.THINK,
        "description": "Reason about the task and select the next tool to use",
        "purpose": "Analyze the task and previous observations to decide the next action",
        "required_context_keys": [
            ContextKeys.TOOL_NAME,
            ContextKeys.TOOL_INPUT,
            ContextKeys.SHOULD_TERMINATE,
        ],
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-009: no state-level bulk
        # call (one more LLM call per turn for keys the typed fields already
        # fill from their own narrowed context). Do NOT empty it under use_classification: tool_name is classification-owned
        # there and relies on the bulk fill when the classifier declines
        # (21cd7f8e D-019).
        "extraction_instructions": think_instructions if use_classification else "",
        "field_extractions": _think_field_extractions(
            think_instructions,
            include_tool_name=not use_classification,
            context_keys=context_keys,
        ),
        "response_instructions": "",
        "transitions": think_transitions,
    }

    if use_classification:
        schema = registry.to_classification_schema()
        think_state["classification_extractions"] = [
            {
                "field_name": "tool_name",
                "intents": schema["intents"],
                "fallback_intent": schema["fallback_intent"],
                "confidence_threshold": schema.get("confidence_threshold", 0.4),
            }
        ]

    states: dict[str, Any] = {
        AgentStates.THINK: think_state,
        AgentStates.ACT: {
            "id": AgentStates.ACT,
            "description": "Execute the selected tool and observe the result",
            "purpose": "Run the tool and record the observation",
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": AgentStates.CONCLUDE,
                    "description": "Terminate when framework signals completion",
                    "priority": 1,
                    "conditions": [
                        {
                            # DECISION plan_2026-05-30_5598b755/D-004 [STALE]
                            # Mirror the think->conclude guard: a should_terminate
                            # set on entry to act (e.g. the model wanted to quit
                            # without a tool) must NOT conclude unless a tool has
                            # run or termination is forced. The stall-detector and
                            # iteration limiter set max_iterations_reached, so a
                            # genuinely tool-free turn still concludes here.
                            "description": (
                                "Framework/agent decided to terminate AND a tool "
                                "has run or termination is forced"
                            ),
                            "logic": _conclude_on_evidence_logic(),
                        }
                    ],
                },
                {
                    "target_state": AgentStates.THINK,
                    "description": "Return to thinking with new observation",
                    "priority": 900,
                },
            ],
        },
        AgentStates.CONCLUDE: {
            "id": AgentStates.CONCLUDE,
            "description": "Formulate and present the final answer",
            "purpose": "Synthesize all observations into a complete answer",
            "required_context_keys": (
                list(output_schema.model_fields.keys())
                if output_schema and hasattr(output_schema, "model_fields")
                else []
            ),
            "response_instructions": build_conclude_response_instructions(
                refused_actions=include_approval_state
            ),
            "transitions": [],
        },
    }

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


# ---------------------------------------------------------------------------
# Prompt Chain FSM
# ---------------------------------------------------------------------------


def build_prompt_chain_fsm(
    chain: list[ChainStep],
    task_description: str = "",
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
    from .prompts import (
        build_chain_output_response_instructions,
        build_chain_step_field_instructions,
    )

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
                {
                    "target_state": PromptChainStates.OUTPUT,
                    "description": f"Stop: the gate of {chain[i - 1].name} failed",
                    "priority": 50,
                    "conditions": [
                        {
                            "description": "The previous step's gate failed",
                            "logic": {
                                "==": [
                                    {"var": ContextKeys.FORCED_STOP_REASON},
                                    StopReason.GATE_FAILED,
                                ]
                            },
                        }
                    ],
                }
            )
        transitions.append(
            {
                "target_state": next_state,
                "description": f"Proceed to {'output' if is_last else step.name}",
                "priority": 100,
            }
        )

        states[state_id] = {
            "id": state_id,
            "description": f"Step {i + 1}: {step.name}",
            "purpose": step.name,
            "extraction_instructions": "",
            "field_extractions": [
                _typed_field_extraction(
                    ContextKeys.CHAIN_STEP_RESULT,
                    "any",
                    build_chain_step_field_instructions(
                        i,
                        step.name,
                        step.response_instructions,
                        step.extraction_instructions,
                    ),
                    extra_context_keys=(ContextKeys.CHAIN_RESULTS,),
                )
            ],
            "response_instructions": step.response_instructions,
            "transitions": transitions,
        }

    # Output (terminal) state
    states[PromptChainStates.OUTPUT] = {
        "id": PromptChainStates.OUTPUT,
        "description": "Produce the final output from the chain",
        "purpose": "Synthesize all step results into a final answer",
        "response_instructions": build_chain_output_response_instructions(),
        "transitions": [],
    }

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


def build_self_consistency_fsm(
    task_description: str = "",
) -> dict[str, Any]:
    """
    Build a simple single-state FSM for self-consistency sampling.

    Each invocation generates one answer. The SelfConsistencyAgent runs
    this FSM multiple times with different temperatures and aggregates.
    ``generate`` is a terminal initial state, so core never extracts there:
    it has no extraction instructions, and the sample is the reply, which ends
    with an ``Answer:`` line for the vote.
    """
    from .prompts import build_generate_response_instructions

    persona = (
        "You are a precise AI assistant. Answer the given task directly and concisely."
    )

    states: dict[str, Any] = {
        SelfConsistencyStates.GENERATE: {
            "id": SelfConsistencyStates.GENERATE,
            "description": "Generate an answer to the task",
            "purpose": "Produce a direct, complete answer to the task",
            "extraction_instructions": "",
            "response_instructions": build_generate_response_instructions(),
            "transitions": [],
        },
    }

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
    from .prompts import (
        build_debate_conclude_response_instructions,
        build_debate_field_instructions,
    )

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
        name: str, context_keys: tuple[str, ...], *, required: bool = True
    ) -> dict[str, Any]:
        return _typed_field_extraction(
            name,
            "str",
            fields[name],
            extra_context_keys=context_keys,
            required=required,
        )

    states: dict[str, Any] = {
        DebateStates.PROPOSE: {
            "id": DebateStates.PROPOSE,
            "description": "Generate or refine a proposition for the task",
            "purpose": "Present a well-reasoned argument or answer",
            "extraction_instructions": "",
            "field_extractions": [
                _debate_field(ContextKeys.PROPOSITION, (ContextKeys.DEBATE_ROUNDS,))
            ],
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": DebateStates.CRITIQUE,
                    "description": "Proposition ready for critique",
                    "priority": 100,
                }
            ],
        },
        DebateStates.CRITIQUE: {
            "id": DebateStates.CRITIQUE,
            "description": "Critically analyze the current proposition",
            "purpose": "Identify weaknesses, gaps, and counterpoints",
            "extraction_instructions": "",
            "field_extractions": [
                _debate_field(ContextKeys.CRITIQUE, round_values[:1])
            ],
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": DebateStates.COUNTER,
                    "description": "Critique complete, allow counter-argument",
                    "priority": 100,
                }
            ],
        },
        DebateStates.COUNTER: {
            "id": DebateStates.COUNTER,
            "description": "Address the critique with counter-arguments",
            "purpose": "Strengthen the proposition by addressing criticisms",
            "extraction_instructions": "",
            "field_extractions": [
                _debate_field(ContextKeys.COUNTER_ARGUMENT, round_values[:2])
            ],
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": DebateStates.JUDGE,
                    "description": "Counter-argument ready for judgment",
                    "priority": 100,
                }
            ],
        },
        DebateStates.JUDGE: {
            "id": DebateStates.JUDGE,
            "description": "Evaluate the debate exchange and decide next action",
            "purpose": "Determine whether consensus has been reached",
            "extraction_instructions": "",
            "field_extractions": [
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
                _typed_field_extraction(
                    ContextKeys.CONSENSUS_REACHED,
                    "bool",
                    fields[ContextKeys.CONSENSUS_REACHED],
                    extra_context_keys=(
                        *round_values,
                        ContextKeys.CURRENT_ROUND,
                        ContextKeys.DEBATE_ROUNDS,
                    ),
                ),
            ],
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": DebateStates.CONCLUDE,
                    "description": "Consensus reached or max rounds hit",
                    "priority": 10,
                    "conditions": [
                        {
                            # DECISION plan-2026-09-24T091842-c1d5bfbc/D-009
                            # Do NOT drop the round disjunct: consensus_reached
                            # is now extracted every round, and transition
                            # evaluation overlays this turn's extracted False
                            # on the judge handler's max-round True, so the
                            # cap must read current_round (bumped by that
                            # handler, never extracted).
                            "description": "Consensus reached or max rounds hit",
                            "logic": {
                                "or": [
                                    {
                                        "==": [
                                            {"var": ContextKeys.CONSENSUS_REACHED},
                                            True,
                                        ]
                                    },
                                    {
                                        ">": [
                                            {"var": ContextKeys.CURRENT_ROUND},
                                            max_rounds,
                                        ]
                                    },
                                ]
                            },
                        }
                    ],
                },
                {
                    "target_state": DebateStates.PROPOSE,
                    "description": "Another round of debate needed",
                    "priority": 300,
                },
            ],
        },
        DebateStates.CONCLUDE: {
            "id": DebateStates.CONCLUDE,
            "description": "Produce the final answer from the debate",
            "purpose": "Synthesize the debate into a definitive answer",
            # Terminal: core never extracts here; the reply is the answer.
            "extraction_instructions": "",
            "response_instructions": build_debate_conclude_response_instructions(),
            "transitions": [],
        },
    }

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
    registry: ToolRegistry,
    task_description: str = "",
    context_keys: Sequence[str] = (),
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
    :func:`_typed_field_extraction`.
    """
    from .prompts import (
        build_rewoo_plan_field_instructions,
        build_rewoo_solve_response_instructions,
    )

    persona = (
        "You are a methodical AI agent that solves tasks by planning all tool calls "
        "upfront, then executing them, and finally synthesizing an answer from the "
        "collected evidence. You think before you act, and you plan completely."
    )

    states: dict[str, Any] = {
        REWOOStates.PLAN_ALL: {
            "id": REWOOStates.PLAN_ALL,
            "description": "Create a complete plan of all tool calls needed",
            "purpose": "Generate a full plan with tool calls and variable references",
            "required_context_keys": [ContextKeys.PLAN_BLUEPRINT],
            # No state-level bulk call (D-031 of plan 07ad3f8c, anchored in
            # build_orchestrator_fsm): `plan_blueprint` is the one key read.
            "extraction_instructions": "",
            "field_extractions": [
                _typed_field_extraction(
                    ContextKeys.PLAN_BLUEPRINT,
                    "list",
                    build_rewoo_plan_field_instructions(
                        registry, task_description=task_description
                    ),
                    extra_context_keys=context_keys,
                )
            ],
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": REWOOStates.EXECUTE_PLANS,
                    "description": "Plan is complete, proceed to execution",
                    "priority": 100,
                }
            ],
        },
        REWOOStates.EXECUTE_PLANS: {
            "id": REWOOStates.EXECUTE_PLANS,
            "description": "Execute all planned tool calls sequentially",
            "purpose": "Run every tool call from the plan, substituting variable references",
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": REWOOStates.SOLVE,
                    "description": "All plans executed, proceed to synthesize answer",
                    "priority": 100,
                }
            ],
        },
        REWOOStates.SOLVE: {
            "id": REWOOStates.SOLVE,
            "description": "Synthesize the final answer from all evidence",
            "purpose": "Combine the task, plan, and all tool results into a final answer",
            "response_instructions": build_rewoo_solve_response_instructions(),
            "transitions": [],
        },
    }

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


def build_evalopt_fsm(
    task_description: str = "",
) -> dict[str, Any]:
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
    from .prompts import (
        build_evalopt_field_instructions,
        build_evalopt_output_response_instructions,
    )

    fields = build_evalopt_field_instructions()

    persona = (
        "You are an AI agent that produces high-quality outputs through iterative "
        "refinement. You generate output, receive evaluation feedback, and improve "
        "your output until it meets the required quality standards."
    )

    states: dict[str, Any] = {
        EvalOptStates.GENERATE: {
            "id": EvalOptStates.GENERATE,
            "description": "Generate an initial output for the task",
            "purpose": "Produce the best possible first attempt at the task",
            "extraction_instructions": "",
            "field_extractions": [
                _typed_field_extraction(
                    ContextKeys.GENERATED_OUTPUT, "any", fields[EvalOptStates.GENERATE]
                )
            ],
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": EvalOptStates.EVALUATE,
                    "description": "Output generated, proceed to evaluation",
                    "priority": 100,
                    "conditions": [
                        {
                            "description": "Output has been generated",
                            "logic": {"has_context": ContextKeys.GENERATED_OUTPUT},
                        }
                    ],
                },
                # DECISION plan-2026-09-24T045559-3e4eb3e5/D-002
                # Unconditional lowest-priority fallback. Do NOT remove it or gate it:
                # a missing generated_output would BLOCK `generate` (no PRE_TRANSITION
                # limiter runs on a BLOCKED turn). `evaluate` judges the empty output,
                # then `refine` retries and the refinement cap / limiter end the loop.
                {
                    "target_state": EvalOptStates.EVALUATE,
                    "description": "Fallback: evaluate even if no output was extracted",
                    "priority": 900,
                },
            ],
        },
        EvalOptStates.EVALUATE: {
            "id": EvalOptStates.EVALUATE,
            "description": "Evaluate the generated output",
            "purpose": "Run the external evaluation function on the current output",
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": EvalOptStates.OUTPUT,
                    "description": "Evaluation passed, produce final output",
                    "priority": 10,
                    "conditions": [
                        {
                            "description": "Output passed evaluation",
                            "logic": {
                                "==": [{"var": ContextKeys.EVALUATION_PASSED}, True]
                            },
                        }
                    ],
                },
                {
                    "target_state": EvalOptStates.REFINE,
                    "description": "Evaluation failed, refine the output",
                    "priority": 300,
                    "conditions": [
                        {
                            "description": "Output did not pass evaluation",
                            "logic": {
                                "==": [{"var": ContextKeys.EVALUATION_PASSED}, False]
                            },
                        }
                    ],
                },
                {
                    "target_state": EvalOptStates.OUTPUT,
                    "description": "Fallback: produce output if evaluation stalls",
                    "priority": 900,
                    "conditions": [],
                },
            ],
        },
        EvalOptStates.REFINE: {
            "id": EvalOptStates.REFINE,
            "description": "Refine the output based on evaluation feedback",
            "purpose": "Improve the output by addressing specific feedback points",
            "extraction_instructions": "",
            "field_extractions": [
                _typed_field_extraction(
                    ContextKeys.GENERATED_OUTPUT,
                    "any",
                    fields[EvalOptStates.REFINE],
                    extra_context_keys=(
                        ContextKeys.PREVIOUS_OUTPUT,
                        ContextKeys.REFINEMENT_FEEDBACK,
                    ),
                )
            ],
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": EvalOptStates.EVALUATE,
                    "description": "Refined output ready for re-evaluation",
                    "priority": 100,
                }
            ],
        },
        EvalOptStates.OUTPUT: {
            "id": EvalOptStates.OUTPUT,
            "description": "Present the final evaluated output",
            "purpose": "Extract and present the final answer",
            "response_instructions": build_evalopt_output_response_instructions(),
            "transitions": [],
        },
    }

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
) -> dict[str, Any]:
    """
    Build a Maker-Checker FSM definition.

    The FSM implements the make-check-revise loop:
    - make: maker persona generates a draft
    - check: checker persona evaluates the draft
    - revise: maker persona revises based on feedback
    - output: terminal state with final answer
    """
    from .prompts import (
        build_maker_checker_field_instructions,
        build_maker_checker_output_response_instructions,
    )

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

    states: dict[str, Any] = {
        MakerCheckerStates.MAKE: {
            "id": MakerCheckerStates.MAKE,
            "description": "Maker generates a draft output",
            "purpose": "Produce a high-quality draft following the maker instructions",
            "extraction_instructions": "",
            "field_extractions": [
                _typed_field_extraction(draft, "any", fields[MakerCheckerStates.MAKE])
            ],
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": MakerCheckerStates.CHECK,
                    "description": "Draft complete, proceed to checker review",
                    "priority": 100,
                }
            ],
        },
        MakerCheckerStates.CHECK: {
            "id": MakerCheckerStates.CHECK,
            "description": "Checker evaluates the draft",
            "purpose": "Critically evaluate the draft against quality criteria",
            "extraction_instructions": "",
            # DECISION plan-2026-09-29T103145-06a5ec0a/D-039
            # One call per field and turn: feedback first (the score and the
            # verdict prompts then show it), checker_passed optional (a null
            # costs no retry and falls to the D-002 revise edge, and
            # _track_revisions passes on the score). Do NOT put these keys
            # back in required_context_keys: core mints an untyped `any`
            # config per key with the whole context, agent_trace included.
            "field_extractions": [
                _typed_field_extraction(
                    ContextKeys.CHECKER_FEEDBACK,
                    "str",
                    fields[ContextKeys.CHECKER_FEEDBACK],
                    extra_context_keys=judged,
                ),
                _typed_field_extraction(
                    "quality_score",
                    "float",
                    fields["quality_score"],
                    extra_context_keys=judged,
                ),
                _typed_field_extraction(
                    ContextKeys.CHECKER_PASSED,
                    "bool",
                    fields[ContextKeys.CHECKER_PASSED],
                    extra_context_keys=judged,
                    required=False,
                ),
            ],
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": MakerCheckerStates.OUTPUT,
                    "description": "Draft passed review, produce final output",
                    "priority": 10,
                    # DECISION plan-2026-09-24T091842-c1d5bfbc/D-007: a check
                    # turn at the budget ships the draft it just judged. Do NOT
                    # rely on the forced checker_passed alone: this turn's
                    # extracted False overlays it, so budget 1 needed 4 turns
                    # against a 3-turn ceiling and raised BudgetExhaustedError.
                    "conditions": [
                        {
                            "description": "Checker approved, or the budget is spent",
                            "logic": {
                                "or": [
                                    {"==": [{"var": ContextKeys.CHECKER_PASSED}, True]},
                                    {
                                        "==": [
                                            {"var": ContextKeys.MAX_ITERATIONS_REACHED},
                                            True,
                                        ]
                                    },
                                ]
                            },
                        }
                    ],
                },
                {
                    "target_state": MakerCheckerStates.REVISE,
                    "description": "Draft needs revision based on feedback",
                    "priority": 300,
                    "conditions": [
                        {
                            "description": "Checker found issues with the draft",
                            "logic": {
                                "==": [{"var": ContextKeys.CHECKER_PASSED}, False]
                            },
                        }
                    ],
                },
                # DECISION plan-2026-09-24T045559-3e4eb3e5/D-002
                # Unconditional lowest-priority fallback. Do NOT remove it or gate it:
                # a missing checker_passed would BLOCK `check`, and no PRE_TRANSITION
                # limiter runs on a BLOCKED turn (FB-01), so the run burns the 3x
                # loop ceiling. An unjudged draft is revised and re-checked.
                {
                    "target_state": MakerCheckerStates.REVISE,
                    "description": "Fallback: revise if checker result unclear",
                    "priority": 900,
                    "conditions": [],
                },
            ],
        },
        MakerCheckerStates.REVISE: {
            "id": MakerCheckerStates.REVISE,
            "description": "Maker revises the draft based on checker feedback",
            "purpose": "Address all checker feedback and produce an improved draft",
            "extraction_instructions": "",
            "field_extractions": [
                _typed_field_extraction(
                    draft,
                    "any",
                    fields[MakerCheckerStates.REVISE],
                    extra_context_keys=(
                        ContextKeys.PREVIOUS_DRAFT,
                        ContextKeys.CHECKER_FEEDBACK,
                    ),
                )
            ],
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": MakerCheckerStates.CHECK,
                    "description": "Revised draft ready for re-evaluation",
                    "priority": 100,
                }
            ],
        },
        MakerCheckerStates.OUTPUT: {
            "id": MakerCheckerStates.OUTPUT,
            "description": "Present the final reviewed output",
            "purpose": "Extract and present the final answer",
            "response_instructions": build_maker_checker_output_response_instructions(),
            "transitions": [],
        },
    }

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
      ``artifact_type`` (fsm, workflow, agent; fallback fsm), reading
      ``latest_request`` as its context so a message-free step classifies
      too. ``-> build`` p10 when ``build_requested`` is True, else
      ``-> collect`` p900.
    - ``collect`` (speaking, no extraction): the reply asks for details and
      ends with the build prompt; it sees only ``artifact_type``,
      ``requirements`` and ``validation_errors``. ``-> build`` p10 on
      ``build_requested``, ``-> classify`` p20 on ``type_switch``, otherwise
      BLOCKED (stay).
    - ``build`` (silent completion state): one structured completion over
      ``_build_messages`` with the response format in
      ``_build_response_format``, result in ``build_reply``. ``-> done`` p10
      when ``build_outcome == "valid"``, else ``-> build_failed`` p900.
    - ``build_failed`` (silent): ``-> build`` p10 on ``build_requested``,
      ``-> classify`` p20 on ``type_switch``, ``-> collect`` p900.
    - ``done`` (terminal, silent).

    The driver writes ``requirements``, ``latest_request``,
    ``build_requested``, ``type_switch`` and ``keyword_type`` before each
    turn; handlers write the build request on ``build`` entry and
    ``build_outcome``, ``artifact``, ``validation_errors`` and
    ``review_presentation`` from ``build_reply``. All of those are
    ``handler_only_keys`` (``META_HANDLER_ONLY_KEYS``). Never raises.
    """
    from .constants import (
        META_ARTIFACT_TYPE_INTENTS,
        META_HANDLER_ONLY_KEYS,
        MetaBuilderStates,
        MetaBuildOutcome,
        MetaContextKeys,
        MetaDefaults,
    )
    from .definitions import ArtifactType
    from .meta_prompts import build_collect_response_instructions

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
    states: dict[str, Any] = {
        states_.CLASSIFY: {
            "id": states_.CLASSIFY,
            "description": "Decide which kind of artifact the user wants",
            "purpose": ("Classify the request as an FSM, a workflow or an agent"),
            "extraction_instructions": "",
            "response_instructions": "",
            "classification_extractions": [
                {
                    "field_name": keys.ARTIFACT_TYPE,
                    "intents": [
                        {"name": name, "description": description}
                        for name, description in META_ARTIFACT_TYPE_INTENTS
                    ],
                    "fallback_intent": ArtifactType.FSM.value,
                    "confidence_threshold": MetaDefaults.TYPE_CONFIDENCE_THRESHOLD,
                    "context_keys": [keys.LATEST_REQUEST],
                }
            ],
            "transitions": [
                {
                    "target_state": states_.BUILD,
                    "description": "The user asked to build right away",
                    "priority": 10,
                    "conditions": [build_requested],
                },
                {
                    "target_state": states_.COLLECT,
                    "description": "Collect the artifact's requirements",
                    "priority": 900,
                },
            ],
        },
        states_.COLLECT: {
            "id": states_.COLLECT,
            "description": "Gather the artifact's requirements from the user",
            "purpose": (
                "Acknowledge each detail, ask about what is unclear and "
                "invite the user to say 'build it'"
            ),
            "extraction_instructions": "",
            "response_instructions": build_collect_response_instructions(),
            "context_scope": {
                "read_keys": [
                    keys.ARTIFACT_TYPE,
                    keys.REQUIREMENTS,
                    keys.VALIDATION_ERRORS,
                ]
            },
            "transitions": [
                {
                    "target_state": states_.BUILD,
                    "description": "The user asked to build the artifact",
                    "priority": 10,
                    "conditions": [build_requested],
                },
                {
                    "target_state": states_.CLASSIFY,
                    "description": "The user switched to another artifact type",
                    "priority": 20,
                    "conditions": [type_switch],
                },
            ],
        },
        states_.BUILD: {
            "id": states_.BUILD,
            "description": "Generate the artifact spec in one structured call",
            "purpose": (
                "Ask the model for the whole artifact spec as JSON matching "
                "the per-type schema"
            ),
            "response_instructions": "",
            "completion": {
                "response_format_key": keys.BUILD_RESPONSE_FORMAT,
                "messages_key": keys.BUILD_MESSAGES,
                "result_key": keys.BUILD_REPLY,
            },
            "transitions": [
                {
                    "target_state": states_.DONE,
                    "description": "The assembled artifact validated",
                    "priority": 10,
                    "conditions": [
                        {
                            "description": "The build produced a valid artifact",
                            "requires_context_keys": [keys.BUILD_OUTCOME],
                            "logic": {
                                "===": [
                                    {"var": keys.BUILD_OUTCOME},
                                    MetaBuildOutcome.VALID,
                                ]
                            },
                        }
                    ],
                },
                {
                    "target_state": states_.BUILD_FAILED,
                    "description": "The build did not produce a valid artifact",
                    "priority": 900,
                },
            ],
        },
        states_.BUILD_FAILED: {
            "id": states_.BUILD_FAILED,
            "description": "The last build failed validation",
            "purpose": (
                "Route the user's next message: build again, switch the "
                "artifact type, or collect more details"
            ),
            "extraction_instructions": "",
            "response_instructions": "",
            "transitions": [
                {
                    "target_state": states_.BUILD,
                    "description": "The user asked to build again",
                    "priority": 10,
                    "conditions": [build_requested],
                },
                {
                    "target_state": states_.CLASSIFY,
                    "description": "The user switched to another artifact type",
                    "priority": 20,
                    "conditions": [type_switch],
                },
                {
                    "target_state": states_.COLLECT,
                    "description": "Collect more details before the next build",
                    "priority": 900,
                },
            ],
        },
        states_.DONE: {
            "id": states_.DONE,
            "description": "The artifact is built and valid",
            "purpose": "End the session with the valid artifact",
            "response_instructions": "",
            "transitions": [],
        },
    }
    return {
        "name": "meta_builder",
        "description": (
            "Meta-builder: classify the artifact type, collect requirements, "
            "build the artifact with one structured completion"
        ),
        "initial_state": states_.CLASSIFY,
        "persona": (
            "A concise assistant that helps users design FSM-LLM artifacts: "
            "FSMs, workflows and agents"
        ),
        "states": states,
        "handler_only_keys": list(META_HANDLER_ONLY_KEYS),
    }


# ---------------------------------------------------------------------------
# Native function-calling FSM
# ---------------------------------------------------------------------------


def _reply_kind_condition(kind: str) -> dict[str, Any]:
    """Condition that passes when the ``call_model`` reply is of ``kind``."""
    from .constants import NativeFCContextKeys

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
    from fsm_llm.constants import CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT

    from .constants import (
        NATIVE_FC_HANDLER_ONLY_KEYS,
        NativeFCContextKeys,
        NativeFCStates,
        NativeLoopEnd,
    )

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
            {
                "target_state": states_.FORCE_FINAL,
                "description": "Force one call of the final tool",
                "priority": 20,
                "conditions": [force_pending],
            }
        ]
        if forced_schemas
        else []
    )
    repair_edge: list[dict[str, Any]] = (
        [
            {
                "target_state": states_.REPAIR,
                "description": "Repair the answer into the output schema",
                "priority": 30,
                "conditions": [repair_pending],
            }
        ]
        if repair
        else []
    )

    def silent(state_id: str, description: str, purpose: str) -> dict[str, Any]:
        return {
            "id": state_id,
            "description": description,
            "purpose": purpose,
            "response_instructions": "",
        }

    states: dict[str, Any] = {
        states_.CALL_MODEL: {
            **silent(
                states_.CALL_MODEL,
                "One model turn with the tools declared",
                "Let the model call tools or give its final answer",
            ),
            "completion": {
                "tools": schemas,
                "instructions": instructions,
                "messages_key": keys.TRANSCRIPT,
                "result_key": keys.MODEL_REPLY,
            },
            "transitions": [
                {
                    "target_state": states_.RUN_TOOLS,
                    "description": "The model asked for tools",
                    "priority": 10,
                    "conditions": [_reply_kind_condition("calls")],
                },
                *force_edge,
                *repair_edge,
                {
                    "target_state": states_.CONCLUDE,
                    "description": "The model answered or garbled its turn",
                    "priority": 900,
                },
            ],
        },
        states_.RUN_TOOLS: {
            **silent(
                states_.RUN_TOOLS,
                "Run the model's tool calls",
                "Run each call through the tool registry and record the results",
            ),
            "transitions": [
                *force_edge,
                *repair_edge,
                {
                    "target_state": states_.CONCLUDE,
                    "description": "The model loop has ended",
                    "priority": 40,
                    "conditions": [
                        {
                            "description": "The loop ended",
                            "requires_context_keys": [keys.LOOP_END],
                            "logic": {
                                "in": [{"var": keys.LOOP_END}, list(NativeLoopEnd.ALL)]
                            },
                        }
                    ],
                },
                {
                    "target_state": states_.CALL_MODEL,
                    "description": "Give the model the tool results",
                    "priority": 900,
                },
            ],
        },
    }
    if forced_schemas:
        states[states_.FORCE_FINAL] = {
            **silent(
                states_.FORCE_FINAL,
                "One model turn that must call the final tool",
                "Make the model record its findings with the forced tool",
            ),
            "completion": {
                "tools": forced_schemas,
                "tool_choice": force_final_tool,
                "instructions": instructions,
                "messages_key": keys.FORCE_MESSAGES,
                "result_key": keys.FORCED_REPLY,
            },
            "transitions": [
                *repair_edge,
                {
                    "target_state": states_.CONCLUDE,
                    "description": "The forced turn is done",
                    "priority": 900,
                },
            ],
        }
    if repair:
        states[states_.REPAIR] = {
            **silent(
                states_.REPAIR,
                "One structured model turn with no tools",
                "Get the final answer as JSON matching the output schema",
            ),
            "completion": {
                "response_format_key": CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT,
                "instructions": instructions,
                "messages_key": keys.REPAIR_MESSAGES,
                "result_key": keys.REPAIR_REPLY,
            },
            "transitions": [
                {
                    "target_state": states_.CONCLUDE,
                    "description": "The repair turn is done",
                    "priority": 900,
                }
            ],
        }
    states[states_.CONCLUDE] = {
        **silent(
            states_.CONCLUDE,
            "The run is over",
            "Hold the answer, the trace and the run's outcome",
        ),
        "transitions": [],
    }
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
    fsm["handler_only_keys"] = [
        *fsm["handler_only_keys"],
        *NATIVE_FC_HANDLER_ONLY_KEYS,
    ]
    return fsm
