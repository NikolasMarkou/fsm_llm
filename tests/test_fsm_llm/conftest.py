from __future__ import annotations

"""Shared fixtures for core fsm_llm tests."""

import pytest

from fsm_llm.runtime import EventLog


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


@pytest.fixture
def log(tmp_path):
    """An ``EventLog`` under ``tmp_path / "events"`` for the event log tests, closed at
    teardown (an open handle blocks the removal of its directory on Windows)."""
    with EventLog(tmp_path / "events") as opened:
        yield opened
