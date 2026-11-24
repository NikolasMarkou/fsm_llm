"""Debate FSM (``build_debate_fsm``): the shipped file ``patterns/debate.json``, instantiated."""

from __future__ import annotations

from typing import Any

from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.prompts import build_debate_field_instructions

#: The text variable of the file for each prompt ``build_debate_field_instructions`` returns.
_TEXTS = {
    "proposition_instructions": ContextKeys.PROPOSITION,
    "critique_instructions": ContextKeys.CRITIQUE,
    "counter_instructions": ContextKeys.COUNTER_ARGUMENT,
    "verdict_instructions": ContextKeys.JUDGE_VERDICT,
    "consensus_instructions": ContextKeys.CONSENSUS_REACHED,
}

#: The persona of the FSM when the proposer has none (the proposer persona starts the debate).
_DEFAULT_FSM_PERSONA = (
    "You are a thoughtful AI that explores questions through structured debate. "
    "Multiple perspectives are considered to arrive at the best answer."
)


def debate_texts(
    proposer_persona: str, critic_persona: str, judge_persona: str, max_rounds: int
) -> dict[str, str]:
    """The six text variables of the file: the FSM persona and the field prompts around the personas."""
    fields = build_debate_field_instructions(
        proposer_persona, critic_persona, judge_persona, max_rounds
    )
    return {
        "debate_persona": proposer_persona or _DEFAULT_FSM_PERSONA,
        **{name: fields[key] for name, key in _TEXTS.items()},
    }


def build_debate_fsm(
    proposer_persona: str = "",
    critic_persona: str = "",
    judge_persona: str = "",
    max_rounds: int = 3,
) -> dict[str, Any]:
    """
    Build a Debate FSM definition: propose, critique, counter, judge (back to propose), conclude.

    The FSM is the pattern file ``fsm_llm/agents/patterns/debate.json`` with the personas filled by
    ``pattern_inputs.pattern_fsm`` and the integer option ``max_rounds`` as the hole of the judge's
    round cap (plan a555dcfe step 17): the same dict, in the same key order, as the builder this
    replaced. Each state extracts one typed field (judge: the verdict and ``consensus_reached``);
    state-level extraction and response instructions are empty, so only ``conclude`` speaks.
    """
    # imported here: pattern_inputs imports the sibling `_loop` (see build_react_fsm)
    from fsm_llm.agents.pattern_files import load_envelope
    from fsm_llm.agents.pattern_inputs import pattern_fsm

    return pattern_fsm(
        load_envelope("debate"),
        tools=None,
        options={"max_rounds": max_rounds},
        variables=debate_texts(
            proposer_persona, critic_persona, judge_persona, max_rounds
        ),
    )
