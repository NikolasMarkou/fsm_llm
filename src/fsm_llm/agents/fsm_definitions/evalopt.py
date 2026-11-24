"""Evaluator-Optimizer FSM (``build_evalopt_fsm``): the shipped file ``patterns/evaluator_optimizer.json``, instantiated."""

from __future__ import annotations

from typing import Any


def build_evalopt_fsm() -> dict[str, Any]:
    """
    Build an Evaluator-Optimizer FSM definition: generate, evaluate, refine (back to evaluate), output.

    The FSM is the pattern file ``fsm_llm/agents/patterns/evaluator_optimizer.json`` through
    ``pattern_inputs.pattern_fsm`` (plan a555dcfe step 18): the same dict, in the same key order, as
    the builder this replaced. ``evaluation_passed`` is written only by the evaluation handler.
    """
    # imported here: pattern_inputs imports the sibling `_loop` (see build_react_fsm)
    from fsm_llm.agents.pattern_files import load_envelope
    from fsm_llm.agents.pattern_inputs import pattern_fsm

    return pattern_fsm(load_envelope("evaluator_optimizer"), tools=None)
