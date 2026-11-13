"""Maker-Checker FSM (``build_maker_checker_fsm``)."""

from __future__ import annotations

from typing import Any

from fsm_llm.agents.constants import ContextKeys, MakerCheckerStates
from fsm_llm.agents.prompts import (
    build_maker_checker_field_instructions,
    build_maker_checker_output_response_instructions,
)

from ._common import _by_id, _edge, _finalize_fsm, _is, _state, _when
from ._loop import _field


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
