"""Tests for smart_truncate — structure-aware text truncation."""

import json

from fsm_llm_agents.truncation import smart_truncate


class TestSmartTruncateBasics:
    """Basic behavior: under limit, at limit, empty."""

    def test_under_limit_unchanged(self):
        text = "short text"
        assert smart_truncate(text, 100) == text

    def test_at_limit_unchanged(self):
        text = "x" * 2000
        assert smart_truncate(text, 2000) == text

    def test_empty_string(self):
        assert smart_truncate("", 100) == ""

    def test_over_limit_truncated(self):
        text = "x" * 3000
        result = smart_truncate(text, 2000)
        assert len(result) <= 2100  # allow marker overhead
        assert "truncated" in result


class TestSmartTruncateHeadTail:
    """Head+tail preservation."""

    def test_preserves_head(self):
        lines = [f"Line {i}: some content here" for i in range(200)]
        text = "\n".join(lines)
        result = smart_truncate(text, 2000)
        assert result.startswith("Line 0:")

    def test_preserves_tail(self):
        lines = [f"Line {i}: some content here" for i in range(200)]
        text = "\n".join(lines)
        result = smart_truncate(text, 2000)
        assert "Line 199:" in result

    def test_conclusion_at_end_preserved(self):
        body = "Data point analysis.\n" * 150
        text = body + "CONCLUSION: The overall trend is positive."
        result = smart_truncate(text, 2000)
        assert "CONCLUSION: The overall trend is positive." in result

    def test_error_at_end_preserved(self):
        trace = "  at module.function(file.py:10)\n" * 100
        text = trace + "RuntimeError: connection refused"
        result = smart_truncate(text, 2000)
        assert "RuntimeError: connection refused" in result

    def test_marker_shows_dropped_chars(self):
        text = "x\n" * 2000  # ~4000 chars
        result = smart_truncate(text, 2000)
        assert "truncated" in result
        assert "chars" in result


class TestSmartTruncateLineAwareness:
    """Line-boundary truncation."""

    def test_no_mid_line_cuts(self):
        lines = [f"Complete line {i} with important data" for i in range(100)]
        text = "\n".join(lines)
        result = smart_truncate(text, 1500)
        # Every line in the result should be complete (start with "Complete line")
        result_lines = [ln for ln in result.split("\n") if ln and "truncated" not in ln]
        for line in result_lines:
            assert line.startswith("Complete line"), f"Broken line: {line!r}"

    def test_single_long_line_falls_back_to_char_split(self):
        text = "x" * 5000  # single line, no newlines
        result = smart_truncate(text, 2000)
        assert len(result) <= 2100
        assert "truncated" in result


class TestSmartTruncateJSON:
    """JSON-like content handling."""

    def test_json_head_preserved(self):
        data = {f"key_{i}": f"value_{i}" * 20 for i in range(50)}
        text = json.dumps(data, indent=2)
        result = smart_truncate(text, 2000)
        # Should start with JSON opening
        assert result.lstrip().startswith("{")

    def test_json_tail_has_closing_brace(self):
        data = {f"key_{i}": f"value_{i}" * 20 for i in range(50)}
        text = json.dumps(data, indent=2)
        result = smart_truncate(text, 2000)
        # Tail should include the closing brace
        assert result.rstrip().endswith("}")

    def test_list_json_preserves_brackets(self):
        data = [{"item": i, "desc": f"description {i}" * 20} for i in range(50)]
        text = json.dumps(data, indent=2)
        result = smart_truncate(text, 2000)
        assert result.lstrip().startswith("[")
        assert result.rstrip().endswith("]")


class TestSmartTruncateNumberedList:
    """Numbered/bulleted list handling."""

    def test_numbered_list_keeps_first_and_last(self):
        items = [f"{i + 1}. Item number {i + 1} with details" for i in range(100)]
        text = "\n".join(items)
        result = smart_truncate(text, 1500)
        assert "1. Item number 1" in result
        assert "100. Item number 100" in result

    def test_bullet_list_keeps_first_and_last(self):
        items = [f"- Finding {i + 1}: important detail" for i in range(100)]
        text = "\n".join(items)
        result = smart_truncate(text, 1500)
        assert "- Finding 1:" in result
        assert "- Finding 100:" in result


class TestSmartTruncateDefaultLimit:
    """Default MAX_OBSERVATION_LENGTH behavior."""

    def test_default_limit_is_2000(self):
        text = "x\n" * 2000
        result = smart_truncate(text)
        assert len(result) <= 2100
