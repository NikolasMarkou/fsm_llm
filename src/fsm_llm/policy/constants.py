"""Constants of the policy layer: the subset of the agents defaults and log templates
that the HITL manager reads.

The agents classes of the same names (``fsm_llm.agents.constants``) assign every
attribute below from these classes, in their own position, so ``vars()`` of the agents
classes lists the same names as before, and link them (``mirror`` of
``fsm_llm.tools.constants``): setting or deleting a name on either class sets or deletes
it on the other, so a reader of either class sees an override made on the other.
"""

from __future__ import annotations

from fsm_llm.tools.constants import MirroredConstants


class Defaults(metaclass=MirroredConstants):
    """Default configuration values of the policy layer."""

    # Escalate to a human when an agent's confidence falls below this.
    CONFIDENCE_THRESHOLD = 0.3


class LogMessages(metaclass=MirroredConstants):
    """Log message templates of the policy layer."""

    APPROVAL_REQUESTED = "Requesting approval for: {action}"
    APPROVAL_RESULT = "Approval {result} for: {action}"
    ESCALATION = "Escalating to human: {reason}"
