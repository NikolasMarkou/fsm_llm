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
import time

import pytest

from fsm_llm.definitions import FSMDefinition
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

    # --------------------------------------------------------------
    # Strategy 3 (balanced-brace scan): complexity + tie-break pinning.
    # The complexity fix must NOT move the FIRST-wins tie-break (D-023).
    # --------------------------------------------------------------

    def test_unbalanced_braces_do_not_scale_quadratically(self):
        """20k bare `{` must not stall the caller (was 18.2s, O(n^2))."""
        start = time.perf_counter()
        result = extract_json_from_text("{" * 20_000)
        elapsed = time.perf_counter() - start
        assert result is None
        assert elapsed < 0.5, f"took {elapsed:.2f}s — Strategy 3 regressed to O(n^2)"

    def test_answer_then_example_returns_the_first_object(self):
        """FIRST-wins (D-023). Last-wins was shipped once and REVERTED."""
        text = 'The intent is {"intent": "buy"}. Schema: {"intent": "<name>"}'
        assert extract_json_from_text(text) == {"intent": "buy"}

    def test_fenced_final_wins_over_unfenced_draft(self):
        """Strategy 2 runs before Strategy 3 and must keep the fenced object."""
        text = 'Draft was {"answer": "no"}\n```json\n{"answer": "yes"}\n```'
        assert extract_json_from_text(text) == {"answer": "yes"}

    def test_invalid_span_is_skipped_and_next_valid_span_wins(self):
        """`skip_until` behavior: a complete-but-invalid span yields to the next."""
        text = '{"a": bad, "nested": {"x": 1}} then {"b": 2}'
        assert extract_json_from_text(text) == {"b": 2}

    def test_object_after_a_lone_quote_is_still_found(self):
        """Each `{` is scanned with a FRESH in-string state.

        A single global left-to-right pass would read the object as string
        content because of the earlier lone `"` and return None.
        """
        assert extract_json_from_text('x " {"a": 1} " y') == {"a": 1}

    # --------------------------------------------------------------
    # Strategy 4 (regex key-value fallback): intent/confidence capture.
    # The `meaningful_keys` gate lists intent+confidence, but Strategy 4
    # never CAPTURED them (finding G4). Recover them from garbled text
    # when strategies 1-3 cannot parse a balanced top-level object.
    # --------------------------------------------------------------

    def test_strategy4_recovers_intent_and_confidence_from_garbled_text(self):
        """A recoverable classification intent in un-parseable text (G4).

        No balanced top-level object parses (the trailing `{` never
        closes), so strategies 1-3 all fail. The `"intent"`/`"confidence"`
        fragments in the reasoning prose must still populate the result so
        the `meaningful_keys` gate fires — instead of degrading to
        fallback_intent/0.0.
        """
        text = (
            "<think>The user says they want to purchase a widget. "
            'The best label is "intent": "buy" with "confidence": 0.8 '
            "given the clear phrasing. Still deciding { the final answer"
        )
        result = extract_json_from_text(text)
        assert result is not None
        assert result.get("intent") == "buy"
        assert result.get("confidence") == 0.8
        assert isinstance(result["confidence"], float)

    def test_strategy4_skips_malformed_multidot_confidence(self):
        """A stray multi-dot token like "1.2.3" must not hard-fail (G4).

        The float() parse is guarded: a non-parseable confidence is
        skipped, but a well-formed intent is still recovered.
        """
        text = (
            'reasoning: "intent": "buy", "confidence": 1.2.3 '
            "and the object never closes {"
        )
        result = extract_json_from_text(text)
        assert result is not None
        assert result.get("intent") == "buy"
        assert "confidence" not in result

    def test_strategy4_message_reasoning_still_captured(self):
        """Regression: existing string-pattern keys still captured (G4)."""
        text = (
            '"message": "Hello there", "reasoning": "greeting detected" '
            "and an unbalanced tail {"
        )
        result = extract_json_from_text(text)
        assert result is not None
        assert result.get("message") == "Hello there"
        assert result.get("reasoning") == "greeting detected"

    def test_strategy4_scientific_notation_confidence_not_truncated(self):
        """CF2: `1e-3` must read as 0.001, not truncate at `e` to 1 -> 1.0.

        The old `[0-9.]+` pattern captured only the leading `1`, silently
        producing max-certainty (the G6-class silent-1.0 defect the plan set
        out to kill). The full numeric token must survive the exponent.
        """
        text = (
            'label "intent": "buy" with "confidence": 1e-3 '
            "and the object never closes {"
        )
        result = extract_json_from_text(text)
        assert result is not None
        assert result.get("intent") == "buy"
        assert result.get("confidence") == pytest.approx(0.001)
        assert result["confidence"] != 1.0

    def test_strategy4_lone_stray_confidence_returns_none(self):
        """CF5: a LONE stray `"confidence"` (no primary/classification key) must
        NOT flip a previously-None result to a dict.

        G4 loosened the SHARED parser so garbled free text mentioning only a
        confidence key produced `{'confidence': N}`, which a lenient
        all-optional structured-output schema in a non-classification caller
        would then build a partial model from. The pre-G4 None contract is
        restored for the lone-auxiliary case.
        """
        text = 'I am fairly sure — "confidence": 0.8 — but the object never closes {'
        assert extract_json_from_text(text) is None

    def test_strategy4_lone_stray_intent_returns_none(self):
        """CF5: a LONE stray `"intent"` (no primary/confidence key) returns None."""
        text = 'the label here is "intent": "buy" but nothing else parses {'
        assert extract_json_from_text(text) is None

    def test_strategy4_classification_pair_still_recovered(self):
        """CF5 boundary: intent + confidence TOGETHER is a genuine classification
        payload and must still be recovered (the pair reinforces each other)."""
        text = 'best guess "intent": "buy", "confidence": 0.9 with tail {'
        result = extract_json_from_text(text)
        assert result is not None
        assert result.get("intent") == "buy"
        assert result.get("confidence") == pytest.approx(0.9)

    def test_strategy4_auxiliary_survives_with_primary_key(self):
        """CF5 boundary: a lone auxiliary key is kept when a PRIMARY payload key
        (here `message`) co-occurs, so response-generation recovery is intact."""
        text = '"message": "hi there", "confidence": 0.7 and an unbalanced tail {'
        result = extract_json_from_text(text)
        assert result is not None
        assert result.get("message") == "hi there"
        assert result.get("confidence") == pytest.approx(0.7)


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
