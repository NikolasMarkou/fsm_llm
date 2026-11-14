"""The driver-grant helpers of the HITL boundary (plan d454fd62 step 88): the one normalisation of a
call (``approval_grant``), the one way a call is shown (``call_label``), the check that a call already
ran (``call_ran``) and the ``refused_actions`` entry (``refusal_record``). The approval driver in
``agents/base/hitl_gate.py`` writes what these build and ``AgentHandlers.approval_refusal`` compares it,
so both sides normalise a call the same way. The function bodies are the old ``agents/handlers.py`` ones."""

from __future__ import annotations

from typing import Any

from fsm_llm.agents.tools import normalize_tool_input
from fsm_llm.tools import redact_secret_entries


def approval_grant(tool_name: Any, tool_input: Any) -> dict[str, Any]:
    """The driver grant for one call: ``{"tool_name", "parameters"}``.

    Shared by the approval driver (``BaseAgent._handle_hitl_approval``, which
    writes it) and :meth:`AgentHandlers.approval_refusal` (which compares it),
    so both normalize the call the same way. ``tool_input`` is normalized with
    :func:`normalize_tool_input`; never raises.
    """
    return {
        "tool_name": str(tool_name),
        "parameters": normalize_tool_input(tool_input),
    }


def call_label(tool_name: Any, tool_input: Any) -> str:
    """How one tool call is shown: ``<tool>(<redacted parameters>)``.

    The parameters are the :func:`redact_secret_entries` copy of the
    :func:`normalize_tool_input` form (secret-looking keys at any depth, in
    nested mappings and in lists, show ``<redacted>``). Shared by the
    executor's trace ``action`` (:meth:`AgentHandlers._run_selected_tool`),
    :func:`refusal_record` and :func:`call_ran`, so a refused call and a call
    that ran compare equal exactly when they are the same call. Never raises.
    """
    return f"{tool_name}({redact_secret_entries(normalize_tool_input(tool_input))})"


def call_ran(trace: Any, tool_name: Any, tool_input: Any) -> bool:
    """Whether the run's ``agent_trace`` holds an executed ``tool_name`` call
    with these parameters (compared by :func:`call_label`).

    ``trace`` is the context's ``agent_trace`` value; anything but a list is
    an empty trace. Never raises.
    """
    if not isinstance(trace, list):
        return False
    label = call_label(tool_name, tool_input)
    return any(isinstance(step, dict) and step.get("action") == label for step in trace)


def refusal_record(tool_name: Any, tool_input: Any) -> str:
    """The ``refused_actions`` entry for one call a human approver refused.

    Shared by the approval driver (``BaseAgent._handle_hitl_approval``, which
    appends it on a denial of a call that has not run in this run) and
    :meth:`AgentHandlers.spend_grant` (which removes it when the same call is
    later approved and runs), so both build the same text, the
    :func:`call_label` of the call. Never raises.
    """
    return (
        f"{call_label(tool_name, tool_input)}: was refused by the human "
        "approver and was not performed."
    )
