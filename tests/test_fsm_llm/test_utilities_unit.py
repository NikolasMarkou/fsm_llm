"""
Unit tests for fsm_llm.utilities module.

Tests cover:
- extract_json_from_text: multiple extraction strategies
- load_fsm_from_file: file loading and validation
- load_fsm_definition: path vs ID dispatch
- truncate_text: truncation logic
- get_fsm_summary: summary generation
"""

import json

import pytest

from fsm_llm.dialog.definitions import FSMDefinition
from fsm_llm.utilities import (
    extract_json_from_text,
    get_fsm_summary,
    load_fsm_definition,
    load_fsm_from_file,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _minimal_fsm_dict():
    """Return the smallest valid FSM definition dict."""
    return {
        "name": "mini",
        "description": "Minimal FSM",
        "version": "4.1",
        "initial_state": "start",
        "states": {
            "start": {
                "id": "start",
                "description": "Start state",
                "purpose": "Begin",
                "transitions": [
                    {
                        "target_state": "end",
                        "description": "Go to end",
                    }
                ],
            },
            "end": {
                "id": "end",
                "description": "End state",
                "purpose": "Finish",
                "transitions": [],
            },
        },
    }


# ==================================================================
# extract_json_from_text
# ==================================================================


class TestExtractJsonFromText:
    def test_valid_json_object(self):
        text = '{"key": "value", "num": 42}'
        result = extract_json_from_text(text)
        assert result == {"key": "value", "num": 42}

    def test_nested_json(self):
        obj = {"outer": {"inner": [1, 2, 3]}, "flag": True}
        text = json.dumps(obj)
        result = extract_json_from_text(text)
        assert result == obj

    def test_json_in_markdown_code_block(self):
        text = 'Here is the data:\n```json\n{"answer": 42}\n```\nDone.'
        result = extract_json_from_text(text)
        assert result == {"answer": 42}

    def test_json_in_plain_code_block(self):
        text = 'Result:\n```\n{"x": 1}\n```'
        result = extract_json_from_text(text)
        assert result == {"x": 1}

    def test_completely_invalid_text(self):
        result = extract_json_from_text("no json here at all")
        assert result is None

    def test_empty_string(self):
        result = extract_json_from_text("")
        assert result is None

    def test_none_input(self):
        result = extract_json_from_text(None)
        assert result is None

    def test_json_embedded_in_prose(self):
        text = 'The result is {"selected_transition": "greeting"} as expected.'
        result = extract_json_from_text(text)
        assert result is not None
        assert result.get("selected_transition") == "greeting"

    def test_whitespace_only(self):
        result = extract_json_from_text("   \n\t  ")
        assert result is None

    def test_json_with_surrounding_whitespace(self):
        text = '   \n  {"a": 1}  \n  '
        result = extract_json_from_text(text)
        assert result == {"a": 1}


# ==================================================================
# load_fsm_from_file
# ==================================================================


class TestLoadFsmFromFile:
    def test_valid_json_file(self, tmp_path):
        fsm_dict = _minimal_fsm_dict()
        path = tmp_path / "valid.json"
        path.write_text(json.dumps(fsm_dict))

        result = load_fsm_from_file(str(path))
        assert isinstance(result, FSMDefinition)
        assert result.name == "mini"

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_fsm_from_file("/nonexistent/path/fsm.json")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_fsm_from_file(str(path))

    def test_non_dict_json(self, tmp_path):
        path = tmp_path / "array.json"
        path.write_text(json.dumps([1, 2, 3]))

        with pytest.raises(ValueError, match="must be a JSON object"):
            load_fsm_from_file(str(path))

    def test_version_default_added(self, tmp_path):
        """If version is missing it should default to '4.1'."""
        fsm_dict = _minimal_fsm_dict()
        del fsm_dict["version"]
        path = tmp_path / "no_version.json"
        path.write_text(json.dumps(fsm_dict))

        result = load_fsm_from_file(str(path))
        assert result.version == "4.1"


# ==================================================================
# load_fsm_definition
# ==================================================================


class TestLoadFsmDefinition:
    def test_load_by_file_path(self, tmp_path):
        fsm_dict = _minimal_fsm_dict()
        path = tmp_path / "def.json"
        path.write_text(json.dumps(fsm_dict))

        result = load_fsm_definition(str(path))
        assert isinstance(result, FSMDefinition)

    def test_unknown_id_raises(self):
        with pytest.raises(ValueError, match="Unknown FSM ID"):
            load_fsm_definition("some_nonexistent_id")


# ==================================================================
# get_fsm_summary
# ==================================================================


class TestGetFsmSummary:
    def test_basic_summary_content(self):
        """Verify get_fsm_summary returns correct structure and content."""
        fsm = FSMDefinition(**_minimal_fsm_dict())
        summary = get_fsm_summary(fsm)

        assert summary["name"] == "mini"
        assert summary["state_count"] == 2
        assert summary["initial_state"] == "start"
        assert "end" in summary["terminal_states"]
        assert summary["terminal_count"] == 1
        assert summary["total_transitions"] == 1
