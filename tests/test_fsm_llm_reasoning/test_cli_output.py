"""
Tests for the reasoning CLI JSON writers fed by a real ReasoningTrace dump.

Earlier tests built the trace dict by hand with lists, so a set-valued
``reasoning_types_used`` written as ``"<redacted:set>"`` went unnoticed.
"""

from __future__ import annotations

import io
import json
import sys
from typing import Any

from fsm_llm.reasoning import __main__ as cli
from fsm_llm.reasoning.__main__ import _format_json_output, _save_as_json
from fsm_llm.reasoning.definitions import ReasoningTrace
from fsm_llm.reasoning.handlers import OutputFormatter


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


def _ascii_stream() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="ascii", errors="strict")


def _a2026_11_05_trace_info(**trace: Any) -> dict[str, Any]:
    trace = {
        "steps": [{"from": "a", "to": "b"}],
        "final_confidence": 0.5,
        "total_steps": 1,
        "reasoning_types_used": ["analytical"],
        **trace,
    }
    return {
        "reasoning_trace": trace,
        "summary": OutputFormatter.format_reasoning_summary(trace),
    }


def _run_main(monkeypatch, argv, solution, trace_info):
    out, err = _ascii_stream(), _ascii_stream()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys, "argv", ["reasoning", "a problem", *argv])
    monkeypatch.setattr(
        cli, "solve_problem_with_engine", lambda *a, **k: (solution, trace_info)
    )
    code = cli.main()
    out.flush()
    return code, out.buffer.getvalue().decode("ascii")


class TestCliUnencodableOutput:
    """F-RE-01: a finished solve is printed escaped and still saved."""

    def test_non_ascii_solution_is_escaped_not_lost(self, monkeypatch, tmp_path):
        save = tmp_path / "out.txt"
        code, printed = _run_main(
            monkeypatch, ["--save", str(save)], "Café → ok", _a2026_11_05_trace_info()
        )
        assert code == 0
        assert "Caf\\xe9 \\u2192 ok" in printed
        assert "Café → ok" in save.read_text(encoding="utf-8")

    def test_detailed_output_arrows_do_not_end_the_run(self, monkeypatch):
        code, printed = _run_main(
            monkeypatch, ["--output", "detailed"], "42", _a2026_11_05_trace_info()
        )
        assert code == 0
        assert "a \\u2192 b" in printed


class TestQuietPrintsTheFormattedOutput:
    """F-RE-09: --quiet prints the stripped solution through the one formatter."""

    def test_quiet_strips_trailing_whitespace(self, monkeypatch):
        code, printed = _run_main(
            monkeypatch, ["--quiet"], "42\n\n", _a2026_11_05_trace_info()
        )
        assert code == 0
        assert printed == "42\n"

    def test_detailed_output_states_the_summary_facts_once(self):
        text = cli.format_solution_output(
            "42", _a2026_11_05_trace_info(), "detailed", False
        )
        assert text.count("Confidence") == 1
        assert "Metadata:" not in text

    def test_text_output_states_the_confidence_once(self):
        text = cli.format_solution_output(
            "42", _a2026_11_05_trace_info(), "text", False
        )
        assert text.count("Confidence") == 1


class TestJsonDumpsSafety:
    """Every reasoning writer that emits context out of the process (LLM
    prompt, CLI stdout, saved results file) serializes through the shared
    redacting hook, never `default=str`.
    """

    def test_engine_renders_no_context_into_a_message(self):
        """Plan 944e2692 step 23 removed both prompt sites: the engine sends no
        message, so it serializes no context at all."""
        import inspect

        from fsm_llm.reasoning import engine as engine_mod

        source = inspect.getsource(engine_mod)
        assert "json.dumps" not in source
        assert "default=str" not in source

    def test_size_measurements_may_still_use_default_str(self):
        """`handlers.py` only measures `len()` of the serialized text and never
        emits it, so its `default=str` is not a leak; the comment must say so.
        """
        import inspect

        from fsm_llm.reasoning import handlers as handlers_mod

        source = inspect.getsource(handlers_mod)
        assert "never emitted" in source

    def test_cli_json_output_redacts_an_objects_str(self):
        from fsm_llm.reasoning.__main__ import _format_json_output

        class Leaky:
            def __str__(self):
                return "password=hunter2"

        text = _format_json_output("done", {"summary": Leaky()})
        assert "hunter2" not in text
        assert "<redacted:Leaky>" in text

    def test_saved_results_file_redacts_an_objects_str(self, tmp_path):
        from fsm_llm.reasoning.__main__ import _save_as_json

        class Leaky:
            def __str__(self):
                return "password=hunter2"

        target = tmp_path / "results.json"
        _save_as_json(target, "problem", "solution", {"leak": Leaky()})
        written = target.read_text(encoding="utf-8")
        assert "hunter2" not in written
        assert "<redacted:Leaky>" in written
