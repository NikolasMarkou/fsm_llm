"""
BaseAgent — Abstract base class for all fsm_llm agents.

The common conversation loop, budget enforcement, answer extraction, trace
building and context filtering shared by the FSM agent patterns.

Package layout (plan d454fd62, step 87: the old module ``fsm_llm.agents.base``
became this package at the same dotted path; the import surface is unchanged):

- ``agent`` — ``BaseAgent`` (the shell: construction, the context seed,
  ``_run_outcome``, the abstract hooks) and ``accepts_tools``
- ``hitl_gate`` — ``HitlGateMixin``: the one HITL predicate, the gate, the
  refusals of flagged tools and the driver-only approval grant
- ``outcome`` — ``OutcomeMixin``: answer extraction (``_answer_source_keys``
  beside ``_extract_answer``), the cut-off verdict, evidence, trace, structured output
- ``runs`` — ``RunsMixin``: the conversation loop on core's bounded run, the
  budgets, the API factory and the standard run with its stream form
- ``registration`` — ``RegistrationMixin``: the iteration limiter, the tool
  executor, the think-loop and lifecycle handlers
- ``helpers`` — the module-level functions (``strip_caller_context``,
  ``artifact_text``, ``reject_misplaced_kwargs``, ``_with_instructions``, ...)
- ``surface`` — ``_AgentSurface``, the declarations the mixins share (mypy only)

A private name is imported from its defining module
(``fsm_llm.agents.base.helpers._with_instructions``); a patch aims at the
module that performs the lookup (``fsm_llm.agents.base.runs.API``,
``fsm_llm.agents.base.hitl_gate.AgentHandlers``), never this package.
"""

from __future__ import annotations

from .agent import BaseAgent, accepts_tools
from .helpers import (
    artifact_text,
    caller_prompt_keys,
    flagged_tool_names,
    pattern_run_output_keys,
    reject_misplaced_kwargs,
    strip_caller_context,
)

__all__ = [
    "BaseAgent",
    "accepts_tools",
    "artifact_text",
    "caller_prompt_keys",
    "flagged_tool_names",
    "pattern_run_output_keys",
    "reject_misplaced_kwargs",
    "strip_caller_context",
]
