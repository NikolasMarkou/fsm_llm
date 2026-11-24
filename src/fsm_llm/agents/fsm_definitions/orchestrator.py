"""Orchestrator-Workers FSM (``build_orchestrator_fsm``): the shipped file ``patterns/orchestrator.json``, instantiated."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def build_orchestrator_fsm(context_keys: Sequence[str] = ()) -> dict[str, Any]:
    """
    Build an Orchestrator-Workers FSM definition.

    The FSM implements task decomposition and delegation:
    orchestrate -> delegate -> collect -> synthesize (all collected)
                                       -> orchestrate (more work needed)

    ``subtasks`` and ``all_collected`` are explicit typed fields whose prompts
    show the task, ``worker_results`` and the caller's ``context_keys``
    (never ``agent_trace`` or ``skipped_subtasks``). Raises ``ValueError``
    like :func:`loop_field_context_keys` for a disallowed context key. The
    FSM is the pattern file ``fsm_llm/agents/patterns/orchestrator.json`` with
    the caller keys filled by ``pattern_inputs.pattern_fsm`` (plan a555dcfe
    step 13): the same dict, in the same key order, as the builder this replaced.
    """
    # imported here: pattern_inputs imports the sibling `_loop` (see build_react_fsm)
    from fsm_llm.agents.pattern_files import load_envelope
    from fsm_llm.agents.pattern_inputs import pattern_fsm

    return pattern_fsm(
        load_envelope("orchestrator"), tools=None, context_keys=context_keys
    )
