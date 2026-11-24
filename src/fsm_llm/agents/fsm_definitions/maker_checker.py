"""Maker-Checker FSM (``build_maker_checker_fsm``): the shipped file ``patterns/maker_checker.json``, instantiated."""

from __future__ import annotations

from typing import Any

from fsm_llm.agents.constants import ContextKeys, MakerCheckerStates
from fsm_llm.agents.prompts import build_maker_checker_field_instructions

#: The text variable of the file for each prompt ``build_maker_checker_field_instructions`` returns.
_TEXTS = {
    "make_instructions": MakerCheckerStates.MAKE,
    "revise_instructions": MakerCheckerStates.REVISE,
    "feedback_instructions": ContextKeys.CHECKER_FEEDBACK,
    "score_instructions": ContextKeys.QUALITY_SCORE,
    "verdict_instructions": ContextKeys.CHECKER_PASSED,
}


def maker_checker_texts(
    maker_instructions: str, checker_instructions: str
) -> dict[str, str]:
    """The five text variables of the file: the field prompts built around the caller's two texts."""
    fields = build_maker_checker_field_instructions(
        maker_instructions, checker_instructions
    )
    return {name: fields[key] for name, key in _TEXTS.items()}


def build_maker_checker_fsm(
    maker_instructions: str,
    checker_instructions: str,
    max_revisions: int | None = None,
) -> dict[str, Any]:
    """
    Build a Maker-Checker FSM definition: make, check, revise (back to check), output.

    ``max_revisions`` (optional): the check -> output edge also fires once
    ``revision_count`` reaches it (the file's ``cap`` variant). The count is bumped on
    check entry, so the cap holds when the checker's extraction is null (plan
    5ed71cf2/D-009). The FSM is the pattern file ``fsm_llm/agents/patterns/maker_checker.json``
    with the two caller texts filled by ``pattern_inputs.pattern_fsm`` (plan a555dcfe step 16):
    the same dict, in the same key order, as the builder this replaced.
    """
    # imported here: pattern_inputs imports the sibling `_loop` (see build_react_fsm)
    from fsm_llm.agents.pattern_files import load_envelope
    from fsm_llm.agents.pattern_inputs import pattern_fsm

    return pattern_fsm(
        load_envelope("maker_checker"),
        tools=None,
        options=(
            {}
            if max_revisions is None
            else {"cap": True, "max_revisions": max_revisions}
        ),
        variables=maker_checker_texts(maker_instructions, checker_instructions),
    )
