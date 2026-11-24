"""The requests each built-in pattern sends to a scripted model (plan a555dcfe, step 8).

The census invariant D5 (request bytes) for the conversion of the pattern builders to files
(E4.2, steps 9 to 18): a scripted run of every file-able pattern, the sha256 of every request the
model received (extract_field, extract_bulk_data, generate_response), with the clock frozen and
timestamps and uuids normalised, plus the outcome of the run. The FSM text fixtures
(``test_executables_corpus.py``) prove the dict equal; this proves what the model SEES equal, which
the dict does not (a handler that writes a context key a prompt reads moves a byte with the dict
unchanged).

The golden is ``tests/fixtures/executables/pattern_request_bytes.json``, written by
``python -m tests.executables_corpus --write`` on the PARENT tree of the conversion (nothing
converted yet at step 8) and holds a hash per request, as ``kind_equality_baseline.json`` does for
long values; the differential script of the plan compares full text against a git-archive copy of
the parent. ``native_fc`` is pinned by ``test_native_fc_golden.py`` (twelve scenarios, two
providers, full request text), which this file only checks is still in the D5 set.

Do NOT regenerate the fixture to make a failure go away: a difference is a pattern that now sends
the model different bytes (STOP IF 2 of the plan).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests import pattern_runs as runs
from tests.executables_corpus import REQUEST_BYTES_PATH

_GOLDEN = json.loads(REQUEST_BYTES_PATH.read_text(encoding="utf-8"))
_ROOT = Path(__file__).resolve().parents[2]


def _first_difference(got: list[dict], want: list[dict]) -> str:
    for index, (a, b) in enumerate(zip(got, want, strict=False)):
        if a != b:
            return f"request {index} ({a['label']} vs {b['label']}) differs"
    return f"{len(got)} requests against {len(want)} recorded"


class TestGoldenFixture:
    def test_the_fixture_covers_every_case_but_native_fc(self):
        expected = {name for name, case in runs.CASES().items() if not case.native}
        assert set(_GOLDEN) == expected

    def test_every_recorded_request_is_a_hash_of_a_known_call(self):
        for name, row in _GOLDEN.items():
            assert row["requests"], name
            for request in row["requests"]:
                assert request["call"] in {
                    "extract_field",
                    "extract_bulk_data",
                    "generate_response",
                }
                assert len(request["sha256"]) == 64

    def test_native_fc_is_held_by_its_own_golden_in_the_d5_set(self):
        gate = (_ROOT / "scripts" / "gate.sh").read_text(encoding="utf-8")
        row = next(r for r in gate.splitlines() if r.lstrip().startswith('"D5 '))
        assert "test_native_fc_golden.py" in row
        assert "test_pattern_request_bytes.py" in row
        scenarios = json.loads(
            (
                _ROOT
                / "tests/test_fsm_llm_agents/fixtures/native_fc_golden_requests.json"
            ).read_text(encoding="utf-8")
        )["scenarios"]
        assert any(name.startswith("one_tool_call/") for name in scenarios)

    def test_the_approval_runs_ran_the_flagged_tool_through_the_gate(self):
        for name in ("react_approval", "reflexion_approval"):
            tools = _GOLDEN[name]["outcome"]["tools"]
            assert tools and tools[0][0] == "search", name

    def test_the_flows_reached_their_tool_and_reply_steps(self):
        reached = {
            name: {r["label"] for r in row["requests"]} for name, row in _GOLDEN.items()
        }
        assert "field:tool_name" in reached["react"]
        assert "field:plan_blueprint" in reached["rewoo"]
        assert "field:tool_calls" in reached["parallel_react"]
        assert "field:plan_steps" in reached["plan_execute"]
        for name, labels in reached.items():
            assert "reply" in labels, name


class TestGoldenRequests:
    @pytest.mark.parametrize("name", sorted(_GOLDEN))
    def test_the_requests_equal_the_golden(self, name):
        run = runs.run_case(name)
        want = _GOLDEN[name]
        assert run["requests"] == want["requests"], (
            f"{name}: {_first_difference(run['requests'], want['requests'])}"
        )
        assert run["outcome"] == want["outcome"], f"{name}: the outcome moved"

    @pytest.mark.parametrize("name", sorted(_GOLDEN))
    def test_a_second_run_sends_the_same_bytes(self, name):
        first = runs.run_case(name)
        second = runs.run_case(name)
        assert first["requests"] == second["requests"]
        assert first["outcome"] == second["outcome"]

    def test_a_different_clock_day_sends_the_same_bytes(self, monkeypatch):
        """The prompt names today's date: the harness freezes it, so no day moves a hash."""
        import datetime

        real = datetime.date

        class _Later(real):
            @classmethod
            def today(cls):
                return real(2031, 1, 2)

        monkeypatch.setattr(datetime, "date", _Later)
        assert runs.run_case("react")["requests"] == _GOLDEN["react"]["requests"]
