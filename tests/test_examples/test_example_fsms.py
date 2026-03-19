"""Smoke tests that validate example FSM definitions load and pass validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsm_llm.definitions import FSMDefinition
from fsm_llm.validator import FSMValidator

pytestmark = pytest.mark.examples

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"

# All example FSM JSON files
EXAMPLE_FSM_FILES = sorted(EXAMPLES_DIR.glob("**/*.json"))


@pytest.fixture(params=EXAMPLE_FSM_FILES, ids=lambda p: str(p.relative_to(EXAMPLES_DIR)))
def example_fsm_path(request):
    """Parametrize over all example FSM JSON files."""
    return request.param


class TestExampleFSMDefinitions:
    """Validate that all example FSM JSON files load and pass validation."""

    def test_example_fsm_loads_as_valid_json(self, example_fsm_path):
        """Example FSM file must be valid JSON."""
        with open(example_fsm_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "name" in data
        assert "initial_state" in data
        assert "states" in data

    def test_example_fsm_parses_as_definition(self, example_fsm_path):
        """Example FSM file must parse into a valid FSMDefinition."""
        with open(example_fsm_path) as f:
            data = json.load(f)
        definition = FSMDefinition(**data)
        assert definition.name
        assert definition.initial_state in definition.states

    def test_example_fsm_passes_validation(self, example_fsm_path):
        """Example FSM file must pass structural validation."""
        with open(example_fsm_path) as f:
            data = json.load(f)
        validator = FSMValidator(data)
        result = validator.validate()
        assert result.is_valid, f"Validation failed: {result.errors}"


class TestExampleDiscovery:
    """Ensure we actually find example FSMs to test."""

    def test_example_fsms_exist(self):
        """At least 5 example FSM JSON files should exist."""
        assert len(EXAMPLE_FSM_FILES) >= 5, (
            f"Expected >=5 example FSMs, found {len(EXAMPLE_FSM_FILES)}"
        )
