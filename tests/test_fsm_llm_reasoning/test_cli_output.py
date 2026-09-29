"""
Tests for the reasoning CLI JSON writers fed by a real ReasoningTrace dump.

Earlier tests built the trace dict by hand with lists, so a set-valued
``reasoning_types_used`` written as ``"<redacted:set>"`` went unnoticed.
"""

import json

from fsm_llm.reasoning.__main__ import _format_json_output, _save_as_json
from fsm_llm.reasoning.definitions import ReasoningTrace


def _trace_info(types: set[str]) -> dict:
    trace = ReasoningTrace(
        steps=[{"from": "start", "to": "analyze"}],
        reasoning_types_used=types,
        final_confidence=0.8,
    )
    return {"reasoning_trace": trace.model_dump(), "summary": "done"}


class TestFormatJsonOutput:
    def test_reasoning_types_are_a_sorted_list(self):
        text = _format_json_output("42", _trace_info({"critical", "analytical"}))
        assert "redacted" not in text
        data = json.loads(text)
        assert data["metadata"]["reasoning_types_used"] == ["analytical", "critical"]

    def test_single_type(self):
        data = json.loads(_format_json_output("42", _trace_info({"analytical"})))
        assert data["metadata"]["reasoning_types_used"] == ["analytical"]


class TestSaveAsJson:
    def test_saved_trace_keeps_reasoning_types(self, tmp_path):
        path = tmp_path / "out.json"
        _save_as_json(path, "problem", "42", _trace_info({"b", "a"}))
        text = path.read_text(encoding="utf-8")
        assert "redacted" not in text
        data = json.loads(text)
        types = data["trace_info"]["reasoning_trace"]["reasoning_types_used"]
        assert types == ["a", "b"]
