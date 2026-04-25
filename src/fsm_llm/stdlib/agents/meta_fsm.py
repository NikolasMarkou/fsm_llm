from __future__ import annotations

"""
Legacy FSM definition for the MetaBuilderAgent.

.. deprecated::
    The MetaBuilderAgent now uses a ReactAgent internally with builder
    tools. This module is kept for backward compatibility but
    ``build_meta_builder_fsm()`` is no longer called by MetaBuilderAgent.
"""

from typing import Any


def build_meta_builder_fsm() -> dict[str, Any]:
    """Build a legacy MetaBuilder FSM definition.

    .. deprecated::
        MetaBuilderAgent now uses ReactAgent with builder tools.
        This function is kept for backward compatibility only.
    """
    return {
        "name": "MetaBuilder",
        "description": "Legacy — MetaBuilderAgent now uses ReactAgent internally",
        "initial_state": "start",
        "states": {
            "start": {
                "id": "start",
                "description": "Placeholder start state",
                "purpose": "Backward compatibility only",
                "transitions": [],
            },
        },
    }
