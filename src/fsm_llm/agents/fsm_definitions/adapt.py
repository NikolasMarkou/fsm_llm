"""ADaPT FSM (``build_adapt_fsm``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fsm_llm.agents.constants import ADaPTStates, ContextKeys
from fsm_llm.agents.prompts import (
    build_adapt_field_instructions,
    build_combine_response_instructions,
)

from ._common import _by_id, _edge, _finalize_fsm, _is, _state, _when
from ._loop import _field


def build_adapt_fsm(
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

    ADaPT runs no tool, so the builder takes no registry.
    """
    # DECISION plan-2026-11-06T093029-0068b055/D-017: the ``registry``
    # parameter was removed (D-016 had kept it as a no-effect shim). Do NOT add
    # it back or any parameter the FSM does not read (D-007): ``ADaPTAgent(
    # tools=)`` is accepted and stored, never passed here.
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
    # DECISION plan-2026-11-06T093029-0068b055/D-013: `attempt`, `assess` and
    # `decompose` are silent (no `response_instructions`); only `combine`
    # speaks. Do NOT re-add narration to an intermediate state: a silent state
    # makes no model call, `attempt_result` the judges read is in
    # `<current_context>`, and the `assess` narration was written BEFORE
    # `attempt_succeeded` was judged, so it could only bias the verdict.
    # Mocked evidence only; live quality is F-AR-05-LIVE.
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
        "ADaPT agent with recursive decomposition",
        ADaPTStates.ATTEMPT,
        persona,
        states,
    )
