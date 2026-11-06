"""Reasoning audit of plan 5980fbb7 (step 18): each behavioural fix has a test
that fails on the parent commit."""

from __future__ import annotations

import io
import json
import sys
from typing import Any

import pytest

from fsm_llm.constants import MESSAGE_TRUNCATION_SUFFIX
from fsm_llm.logging import logger
from fsm_llm.reasoning import __main__ as cli
from fsm_llm.reasoning.constants import (
    STRATEGIES,
    ContextKeys,
    Defaults,
    ReasoningType,
)
from fsm_llm.reasoning.engine import ReasoningEngine
from fsm_llm.reasoning.exceptions import ReasoningClassificationError
from fsm_llm.reasoning.handlers import OutputFormatter, ReasoningHandlers
from fsm_llm.reasoning.reasoning_modes import ALL_REASONING_FSMS
from fsm_llm.reasoning.utilities import get_reasoning_fsm

from .test_engine_scripted import _PROBLEM, _VALID_SCRIPT, _engine, _ScriptedLLM


def _ascii_stream() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="ascii", errors="strict")


def _trace_info(**trace: Any) -> dict[str, Any]:
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
            monkeypatch, ["--save", str(save)], "Café → ok", _trace_info()
        )
        assert code == 0
        assert "Caf\\xe9 \\u2192 ok" in printed
        assert "Café → ok" in save.read_text(encoding="utf-8")

    def test_detailed_output_arrows_do_not_end_the_run(self, monkeypatch):
        code, printed = _run_main(
            monkeypatch, ["--output", "detailed"], "42", _trace_info()
        )
        assert code == 0
        assert "a \\u2192 b" in printed


class TestQuietPrintsTheFormattedOutput:
    """F-RE-09: --quiet prints the stripped solution through the one formatter."""

    def test_quiet_strips_trailing_whitespace(self, monkeypatch):
        code, printed = _run_main(monkeypatch, ["--quiet"], "42\n\n", _trace_info())
        assert code == 0
        assert printed == "42\n"

    def test_detailed_output_states_the_summary_facts_once(self):
        text = cli.format_solution_output("42", _trace_info(), "detailed", False)
        assert text.count("Confidence") == 1
        assert "Metadata:" not in text

    def test_text_output_states_the_confidence_once(self):
        text = cli.format_solution_output("42", _trace_info(), "text", False)
        assert text.count("Confidence") == 1


class TestExecutionTime:
    """F-XC-03: the trace carries the measured wall-clock seconds."""

    def test_solve_reports_a_float(self):
        _solution, trace_info = _engine(_ScriptedLLM(_VALID_SCRIPT)).solve_problem(
            _PROBLEM
        )
        seconds = trace_info["reasoning_trace"]["execution_time_seconds"]
        assert isinstance(seconds, float)
        assert seconds >= 0

    def test_cli_json_metadata_shows_it(self):
        _solution, trace_info = _engine(_ScriptedLLM(_VALID_SCRIPT)).solve_problem(
            _PROBLEM
        )
        data = json.loads(cli.format_solution_output("x", trace_info, "json", False))
        assert isinstance(data["metadata"]["execution_time_seconds"], float)


class TestDirectComputationIsNotASpecialCase:
    """F-RE-06: only the spelling the prompts ask for (simple_calculator) counts,
    and it picks the strategy the same way as any other spelling."""

    @staticmethod
    def _chosen(**context: Any) -> str:
        engine = ReasoningEngine(model="mock", llm_interface=_ScriptedLLM({}))
        return engine._prepare_reasoning_execution(context)[
            ContextKeys.REASONING_TYPE_SELECTED
        ]

    def test_direct_computation_does_not_beat_the_classified_type(self):
        chosen = self._chosen(
            reasoning_strategy="direct computation",
            classified_problem_type=ReasoningType.DEDUCTIVE.value,
        )
        assert chosen == ReasoningType.DEDUCTIVE.value

    def test_every_spelling_agrees_with_a_classified_type(self):
        chosen = {
            self._chosen(
                reasoning_strategy=spelling,
                classified_problem_type=ReasoningType.DEDUCTIVE.value,
            )
            for spelling in (
                "direct computation",
                "Direct computation",
                "simple_calculator",
                "arithmetic",
            )
        }
        assert chosen == {ReasoningType.DEDUCTIVE.value}

    def test_the_strategy_alone_still_maps_to_the_calculator(self):
        assert (
            self._chosen(reasoning_strategy="direct computation")
            == ReasoningType.SIMPLE_CALCULATOR.value
        )

    def test_validator_reads_simple_calculator_not_direct_computation(self):
        base = {
            ContextKeys.PROPOSED_SOLUTION: "4",
            ContextKeys.PROBLEM_STATEMENT: "What is 2 plus 2?",
        }
        simple = ReasoningHandlers.validate_solution(
            {**base, ContextKeys.REASONING_STRATEGY: "simple_calculator"}
        )
        other = ReasoningHandlers.validate_solution(
            {**base, ContextKeys.REASONING_STRATEGY: "direct computation"}
        )
        assert simple[ContextKeys.VALIDATION_RESULT] is True
        assert other[ContextKeys.VALIDATION_RESULT] is False


class TestOnlyTheOrchestratorTraces:
    """F-RE-05: the tracer runs on the orchestrator API only."""

    def test_classifier_has_no_tracer(self):
        engine = ReasoningEngine(model="mock", llm_interface=_ScriptedLLM({}))
        names = lambda api: {  # noqa: E731
            h.name for h in api.handler_system.handlers if hasattr(h, "name")
        }
        assert "ReasoningTracer" in names(engine.orchestrator)
        assert "ReasoningTracer" not in names(engine.classifier)


class TestPrunedStringMarker:
    """F-RE-10: the truncation marker is core's one spelling."""

    def test_long_string_ends_in_the_core_suffix(self):
        long = "x" * (Defaults.PRUNE_STRING_MAX_LENGTH + 10)
        context = {
            ContextKeys.OBSERVATIONS: long,
            "padding": "y" * Defaults.CONTEXT_PRUNE_THRESHOLD,
        }
        pruned = ReasoningHandlers.prune_context(context)
        assert pruned[ContextKeys.OBSERVATIONS].endswith(MESSAGE_TRUNCATION_SUFFIX)


class TestClassifierCleanup:
    """F-RE-10: both conversations are ended through one helper, which warns."""

    def test_a_failed_classifier_cleanup_is_a_warning(self, monkeypatch):
        engine = ReasoningEngine(model="mock", llm_interface=_ScriptedLLM({}))

        def boom(*_args, **_kwargs):
            raise RuntimeError("cannot end")

        monkeypatch.setattr(engine.classifier, "run_until_terminal", boom)
        monkeypatch.setattr(engine.classifier, "end_conversation", boom)
        records = []
        logger.enable("fsm_llm")
        sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
        try:
            with pytest.raises(ReasoningClassificationError):
                engine._classify_problem({ContextKeys.PROBLEM_STATEMENT: "p"})
        finally:
            logger.remove(sink)
            logger.disable("fsm_llm")
        cleanup = [r for r in records if "clean up conversation" in r["message"]]
        assert [r["level"].name for r in cleanup] == ["WARNING"]


class TestOneStrategyTable:
    """F-RE-02, F-RE-03, F-RE-07: tables derive from one registry."""

    def test_strategies_cover_exactly_the_enum(self):
        assert set(STRATEGIES) == {t.value for t in ReasoningType}

    def test_every_state_id_is_its_key(self):
        for fsm in ALL_REASONING_FSMS.values():
            assert all(s["id"] == key for key, s in fsm["states"].items())

    def test_get_reasoning_fsm_returns_a_copy_and_names_the_missing(self):
        copy = get_reasoning_fsm("orchestrator")
        copy["states"].clear()
        assert ALL_REASONING_FSMS["orchestrator"]["states"]
        with pytest.raises(KeyError, match="FSM definition not found: nope"):
            get_reasoning_fsm("nope")
