"""
Agent-specific handlers for tool execution, iteration limiting,
observation tracking, and HITL gating.

Package layout (plan d454fd62, step 88: the old module ``fsm_llm.agents.handlers``
became this package at the same dotted path; the import surface is unchanged):

- ``executor`` -- ``AgentHandlers``: the ReAct-family tool executor and the HITL
  approval refusal (``approval_refusal``, ``spend_grant``, ``consume_approval``)
- ``limiters`` -- ``ThinkTurnLimiter`` (counts think turns on the instance) and
  ``make_iteration_limiter`` (counts in context, the other patterns)
- ``approval`` -- the driver-grant helpers ``approval_grant``, ``call_label``,
  ``call_ran`` and ``refusal_record``
- ``forced`` -- the forced-stop vocabulary: ``forced_stop_skip``,
  ``recorded_forced_reason``, ``concluded_on_evidence``, ``is_forced_verdict``
- ``recording`` -- ``record_tool_step``, ``next_step_number``, ``with_feedback``
- ``freshness`` -- ``make_redraft_handlers`` and ``make_fresh_keys_handler``
- ``run_guards`` -- ``run_evaluation_fn``, ``RUN_ENDING_ERRORS`` and
  ``RunEndingErrorHolder``
- ``reflection`` -- the Reflexion bodies ``make_evaluation_handler`` and ``make_reflection_handler``
  (wrapped by the library factories ``agents.reflexion_*``; not in ``__all__``)
- ``context`` -- ``HandlerContext``, the per-run value the library factories read (import it from
  ``fsm_llm.agents.handlers.context``; it is not in ``__all__``)
- ``library`` -- the named factories a pattern file binds (``agents.tool_executor``, ...),
  registered at import in the ``HANDLERS`` namespace of ``fsm_llm.engine.executables.namespaces``
  (``from fsm_llm.agents.handlers.library import LIBRARY``; not in ``__all__``)

A private name is imported from its defining module; a patch aims at the module
that performs the lookup (``fsm_llm.agents.handlers.executor.logger``), never
this package; a log record is named after its module
(``fsm_llm.agents.handlers.executor``, ``.limiters``, ...).
"""

from __future__ import annotations

from . import library  # noqa: F401  (registers the factories in the HANDLERS namespace)
from .approval import approval_grant, call_label, call_ran, refusal_record
from .executor import AgentHandlers
from .forced import (
    concluded_on_evidence,
    forced_stop_skip,
    is_forced_verdict,
    recorded_forced_reason,
)
from .freshness import make_fresh_keys_handler, make_redraft_handlers
from .limiters import ThinkTurnLimiter, make_iteration_limiter
from .recording import next_step_number, record_tool_step, with_feedback
from .run_guards import RUN_ENDING_ERRORS, RunEndingErrorHolder, run_evaluation_fn

__all__ = [
    "RUN_ENDING_ERRORS",
    "AgentHandlers",
    "RunEndingErrorHolder",
    "ThinkTurnLimiter",
    "approval_grant",
    "call_label",
    "call_ran",
    "concluded_on_evidence",
    "forced_stop_skip",
    "is_forced_verdict",
    "make_fresh_keys_handler",
    "make_iteration_limiter",
    "make_redraft_handlers",
    "next_step_number",
    "recorded_forced_reason",
    "record_tool_step",
    "refusal_record",
    "run_evaluation_fn",
    "with_feedback",
]
