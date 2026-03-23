from __future__ import annotations

"""Shared fixtures for core fsm_llm tests."""

import pytest


@pytest.fixture
def minimal_fsm_dict():
    """Fixture for FSM with only required fields (tests Pydantic defaults)."""
    return {
        "name": "Minimal FSM",
        "description": "FSM with only required fields",
        "initial_state": "only_state",
        "states": {
            "only_state": {
                "id": "only_state",
                "description": "The only state",
                "purpose": "Single state FSM",
                "transitions": [],
            }
        },
    }
