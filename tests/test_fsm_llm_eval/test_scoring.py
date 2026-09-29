"""Golden parity tests for fsm_llm.eval.scoring.classify_result.

Every row of _GOLDEN was produced by calling the ORIGINAL
classify_result in scripts/eval.py at commit 0facf56 (before the move
into fsm_llm.eval.scoring) on the listed inputs. The moved function must
return exactly the same (score, failures) for each row: new example scores
stay comparable with the historical runs in EVALUATE.md. Do NOT regenerate
this table from the new function; a mismatch means the rubric changed.

The rows cover every return statement: the four score-0 crash branches, both
timeout returns, each output signal (traceback, tool, extraction keywords,
budget, parse), the extraction-rate, [MISSING], stuck-state and handler
checks, nonzero exit with long/short stdout, and every exit-0 severity branch.
"""

from __future__ import annotations

import pytest

from fsm_llm.eval import ExampleResult, classify_result
from fsm_llm.eval.constants import MAX_SCORE, SCORE_LABELS

_LONG_300 = "y" * 300
_LONG_400 = "x" * 400

# (id, exit_code, stdout, stderr, duration, timed_out, score, failures)
_GOLDEN = [
    ("crash_runner_error", -1, "", "boom", 0.1, False, 0, ["F-CODE"]),
    (
        "crash_fast_import",
        1,
        "",
        "ModuleNotFoundError: no module",
        0.5,
        False,
        0,
        ["F-CODE"],
    ),
    (
        "crash_fast_validation",
        1,
        "",
        "pydantic ValidationError: bad",
        0.5,
        False,
        0,
        ["F-SCHEMA"],
    ),
    (
        "crash_fast_pydantic_lower",
        1,
        "",
        "PYDANTIC failure",
        0.5,
        False,
        0,
        ["F-SCHEMA"],
    ),
    ("crash_fast_other", 2, "  \n", "segfault", 1.0, False, 0, ["F-CODE"]),
    ("slow_nonzero_empty_stdout", 1, "", "some error text", 5.0, False, 1, ["F-CODE"]),
    ("timeout_long_stdout", None, _LONG_300, "", 120.0, True, 1, ["F-LOOP"]),
    ("timeout_short_stdout", None, "short", "", 120.0, True, 1, ["F-LOOP"]),
    (
        "traceback_error_exit0",
        0,
        "Traceback (most recent call last):\nValueError: x",
        "",
        10.0,
        False,
        3,
        ["F-CODE"],
    ),
    ("tool_not_selected", 0, "No tool was selected", "", 10.0, False, 3, ["F-TOOL"]),
    ("tool_status_skipped", 0, "tool_status: skipped", "", 10.0, False, 3, ["F-TOOL"]),
    ("extract_keyword", 0, "0 keys extracted", "", 10.0, False, 2, ["F-EXTRACT"]),
    (
        "budget_exhausted",
        0,
        "Budget exhausted after 10 steps",
        "",
        10.0,
        False,
        3,
        ["F-LOOP"],
    ),
    ("parse_error", 0, "JSONDecodeError at line 1", "", 10.0, False, 3, ["F-PARSE"]),
    (
        "parse_error_with_traceback",
        0,
        "Traceback: parse error in json",
        "",
        10.0,
        False,
        3,
        ["F-CODE"],
    ),
    (
        "extraction_rate_zero",
        0,
        "Extraction rate: 0/5 (0%)",
        "",
        10.0,
        False,
        1,
        ["F-EXTRACT"],
    ),
    (
        "extraction_rate_partial",
        0,
        "Extraction rate: 2/5 (40%)",
        "",
        10.0,
        False,
        2,
        ["F-EXTRACT"],
    ),
    ("extraction_rate_ok", 0, "Extraction rate: 3/5 (60%)", "", 10.0, False, 4, []),
    (
        "all_missing",
        0,
        "name [MISSING]\nage [MISSING]",
        "",
        10.0,
        False,
        2,
        ["F-EXTRACT"],
    ),
    ("some_missing", 0, "name [EXTRACTED]\nage [MISSING]", "", 10.0, False, 4, []),
    ("single_state_only", 0, "State: start\nState: start", "", 10.0, False, 4, []),
    (
        "stuck_with_zero_extraction",
        0,
        "State: start\nState: start\nExtraction rate: 0/4",
        "",
        10.0,
        False,
        1,
        ["F-EXTRACT", "F-TRANS"],
    ),
    (
        "stuck_with_partial_extraction",
        0,
        "State: a\nState: a\nExtraction rate: 1/4",
        "",
        10.0,
        False,
        1,
        ["F-EXTRACT", "F-TRANS"],
    ),
    (
        "handler_empty_lists_with_missing",
        0,
        "HANDLER stats\nphases: []\nstages: []\nx [MISSING]",
        "",
        10.0,
        False,
        1,
        ["F-EXTRACT", "F-TRANS"],
    ),
    (
        "hook_zero_metrics_with_missing",
        0,
        "HOOK counts\nupdates: 0\nentries: 0\nx [MISSING]",
        "",
        10.0,
        False,
        1,
        ["F-EXTRACT", "F-TRANS"],
    ),
    (
        "handler_empty_no_extract_fail",
        0,
        "HANDLER\nphases: []\nstages: []",
        "",
        10.0,
        False,
        4,
        [],
    ),
    ("nonzero_long_stdout", 1, _LONG_400, "", 20.0, False, 2, ["F-CODE"]),
    ("nonzero_short_stdout", 1, "partial run", "", 20.0, False, 1, ["F-CODE"]),
    (
        "nonzero_long_with_tool_fail",
        3,
        _LONG_400 + " 0 tools called",
        "",
        20.0,
        False,
        2,
        ["F-TOOL"],
    ),
    (
        "two_failures_exit0",
        0,
        "No tool was selected; budget exhausted",
        "",
        10.0,
        False,
        2,
        ["F-TOOL", "F-LOOP"],
    ),
    (
        "clean_pass",
        0,
        "State: start\nState: done\nExtraction rate: 5/5",
        "",
        12.3,
        False,
        4,
        [],
    ),
    (
        "stderr_only_signal",
        0,
        "fine",
        "json.decoder failure",
        10.0,
        False,
        3,
        ["F-PARSE"],
    ),
]


def _result(exit_code, stdout, stderr, duration, timed_out) -> ExampleResult:
    return ExampleResult(
        name="cat/example",
        category="cat",
        exit_code=exit_code,
        duration=duration,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )


class TestGoldenParity:
    @pytest.mark.parametrize(
        ("exit_code", "stdout", "stderr", "duration", "timed_out", "score", "failures"),
        [row[1:] for row in _GOLDEN],
        ids=[row[0] for row in _GOLDEN],
    )
    def test_matches_original_classify_result(
        self, exit_code, stdout, stderr, duration, timed_out, score, failures
    ):
        result = _result(exit_code, stdout, stderr, duration, timed_out)
        assert classify_result(result) == (score, failures)

    def test_table_reaches_every_score(self):
        assert {row[6] for row in _GOLDEN} == set(range(MAX_SCORE + 1))

    def test_ids_are_unique(self):
        ids = [row[0] for row in _GOLDEN]
        assert len(ids) == len(set(ids))


class TestScoreLabels:
    def test_one_label_per_score(self):
        assert len(SCORE_LABELS) == MAX_SCORE + 1
        assert SCORE_LABELS[0] == "CRASH"
        assert SCORE_LABELS[MAX_SCORE] == "PASS"

    def test_classify_does_not_read_precomputed_score(self):
        result = _result(0, "clean", "", 1.0, False)
        result.score, result.failures = 0, ["F-CODE"]
        assert classify_result(result) == (4, [])
