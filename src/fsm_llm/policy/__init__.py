"""
The policy layer: who may approve a tool call and what happens when nobody does.

A layer package: ``approval`` (``ApprovalRequest``, ``ApprovalPolicy``,
``ApprovalCallback``, ``ApprovalDeniedError``), ``hitl`` (``HumanInTheLoop``, the
approval and escalation manager, and ``EscalationCallback``) and ``constants`` (the
subset of the agents defaults and log templates the manager reads). The agents
package builds on it (``fsm_llm.agents`` re-exports the public names, and the
driver-only approval grant that gates a tool run stays in ``fsm_llm.agents.base``);
the policy layer never imports ``agents``.

Imports the kernel (``constants``, ``logging``), ``core`` (``AgentError``) and the
tool layer (``fsm_llm.tools``: the tool models, the per-call timeout and the
redaction helper) and nothing above.

This ``__init__`` is the one import surface of the package (a static ``__all__``);
a private name is imported from the module that defines it.
"""

from __future__ import annotations

from .approval import (
    ApprovalCallback,
    ApprovalDeniedError,
    ApprovalPolicy,
    ApprovalRequest,
)
from .hitl import EscalationCallback, HumanInTheLoop

__all__ = [
    "ApprovalCallback",
    "ApprovalDeniedError",
    "ApprovalPolicy",
    "ApprovalRequest",
    "EscalationCallback",
    "HumanInTheLoop",
]
