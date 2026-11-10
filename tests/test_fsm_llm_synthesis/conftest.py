"""Shared fixtures of the synthesis suite."""

from typing import Any

import pytest


@pytest.fixture
def valid_fsm() -> dict[str, Any]:
    """A fresh compile that passes G1 to G5 with ``return_keys=("answer",)``."""
    return {
        "name": "Collect",
        "description": "Collect one answer",
        "initial_state": "ask",
        "states": {
            "ask": {
                "id": "ask",
                "description": "ask for the answer",
                "purpose": "get the answer",
                "response_instructions": "Ask for the answer",
                "field_extractions": [
                    {
                        "field_name": "answer",
                        "field_type": "str",
                        "extraction_instructions": "The user's answer",
                    }
                ],
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "answer given",
                        "priority": 1,
                        "conditions": [
                            {
                                "description": "has answer",
                                "requires_context_keys": ["answer"],
                            }
                        ],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "finished",
                "purpose": "wrap up",
                "response_instructions": "Thank the user",
            },
        },
    }
