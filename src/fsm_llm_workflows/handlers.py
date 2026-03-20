from __future__ import annotations

"""
Handlers for integrating workflows with FSM-LLM.

Note: AutoTransitionHandler, EventHandler, and TimerHandler were removed
because they set deferred context flags that the WorkflowEngine never consumed.
The engine manages auto-transitions, events, and timers directly through its
step execution path. See DECISION D-001.

This module is kept for backward compatibility but contains no active code.
"""

# --------------------------------------------------------------
