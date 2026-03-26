from __future__ import annotations

"""
FSM definitions for the meta-agent.

The meta-agent no longer uses a custom FSM. The build phase is driven
by a ReactAgent which auto-generates its own FSM via
``fsm_llm_agents.fsm_definitions.build_react_fsm()``.

This module is kept for backward compatibility but contains no
active definitions.
"""
