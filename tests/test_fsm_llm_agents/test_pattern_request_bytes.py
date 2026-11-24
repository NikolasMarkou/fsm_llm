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
import re
from pathlib import Path

import pytest

from tests import pattern_runs as runs
from tests.executables_corpus import REQUEST_BYTES_PATH
from tests.test_fsm_llm.kind_corpus import _frozen_today

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


# ---- the blind spots of the first golden (plan a555dcfe, step 28.5, D-035) -----------------------
#
# The review found the golden blind to the retry note of ``verified_react``, the recall and remember
# texts of ``auto_memory``, every ``run_stream`` path, a semantic registry under ``rewoo`` and
# ``plan_execute`` and custom debate personas. Each is now a case of the harness whose golden entry was
# written from a ``git archive`` of the tree BEFORE the conversion (commit 1a87fb4e, the old classes with
# this harness copied in), with the full request text compared and equal there: the entries are the
# parent's, never the tree's.

_STREAMS_WITHOUT_A_PARENT = [
    "reflexion",
    "plan_execute",
    "maker_checker",
    "debate",
    "evaluator_optimizer",
    "prompt_chain",
    "orchestrator",
]


def _texts(name: str) -> list[str]:
    """The text of every request the scripted model of case *name* received (not hashed)."""
    case = {**runs.CASES(), **runs._new_stream_cases()}[name]
    model = runs.ScriptedModel(case.tool)
    agent = case.build(model)
    if case.stream:
        list(agent.run_stream(case.task))
    else:
        agent.run(case.task)
    return [
        runs.normalise(json.dumps(request, sort_keys=True))
        for _call, request in model.log
    ]


class TestTheBlindSpotsAreCovered:
    """A case that never reached the text it guards would pass a wrong golden: check it did."""

    def test_the_retry_note_reaches_the_second_attempt_only(self):
        texts = _texts("verified_react_retry")
        marked = [i for i, t in enumerate(texts) if "rejected by verification" in t]

        assert marked and marked[0] > 0
        assert _GOLDEN["verified_react_retry"]["outcome"]["status"] == "ok"
        assert _GOLDEN["verified_react_rejected"]["outcome"]["stop_reason"] == (
            "verification_failed"
        )

    def test_the_recall_block_is_sent_and_the_interaction_is_remembered(self):
        texts = _texts("auto_memory")

        assert any("[Relevant things you remember]" in t for t in texts)
        assert any("second shelf" in t for t in texts)
        remembered = _GOLDEN["auto_memory"]["outcome"]["memory"]
        assert any(m.startswith("Q: Find the alpha thing") for m in remembered)

    def test_the_recall_floor_keeps_only_the_memory_above_it(self):
        texts = " ".join(_texts("auto_memory_floor"))
        everything = " ".join(_texts("auto_memory"))

        assert "second shelf" in texts and "Beta is unrelated" not in texts
        assert "Beta is unrelated" in everything

    @pytest.mark.parametrize("name", ["rewoo_semantic", "plan_execute_semantic"])
    def test_a_semantic_registry_lists_the_retrieved_tool_in_the_prompt(self, name):
        assert any("t0" in t for t in _texts(name))
        assert _GOLDEN[name]["outcome"]["tools"][0][0] == "t0"

    def test_the_three_custom_personas_reach_the_model(self):
        joined = " ".join(_texts("debate_personas"))

        for persona in (
            "cautious systems engineer",
            "bold product designer",
            "neutral technical editor",
        ):
            assert persona in joined

    @pytest.mark.parametrize(
        "name", ["react_stream", "verified_react_stream", "auto_memory_stream"]
    )
    def test_a_stream_ends_with_one_chunk_of_the_answer(self, name):
        assert _GOLDEN[name]["outcome"]["chunks"] == ["Answer: ok"]

    def test_a_stream_through_run_sends_what_run_sends(self):
        for stream, plain in (
            ("verified_react_stream", "verified_react_retry"),
            ("auto_memory_stream", "auto_memory"),
        ):
            assert _GOLDEN[stream]["requests"] == _GOLDEN[plain]["requests"], stream


class TestAStreamSendsTheRequestsOfRun:
    """A stream asks for a plain-text reply: each reply request differs from ``run``'s in the
    ``<response_format>`` block and nowhere else (every other request is the same)."""

    @staticmethod
    def _without_format(texts: list[str]) -> list[str]:
        return [re.sub(r"<response_format>.*?</response_format>", "", t) for t in texts]

    @pytest.mark.parametrize("pattern", ["react", *_STREAMS_WITHOUT_A_PARENT])
    def test_the_stream_differs_from_run_in_the_response_format_only(self, pattern):
        with _frozen_today():
            streamed = _texts(f"{pattern}_stream")
            plain = _texts(pattern)

        assert len(streamed) == len(plain)
        assert streamed != plain  # the replies ask for plain text
        assert self._without_format(streamed) == self._without_format(plain)


class TestStreamsThatNoOldClassHad:
    """The old ``reflexion``, ``plan_execute``, ``maker_checker``, ``debate``, ``evaluator_optimizer``,
    ``prompt_chain`` and ``orchestrator`` had no ``run_stream`` (``AttributeError``), so there is no
    parent golden of a stream: its rule is the one above against the same pattern's ``run``, whose golden
    IS the parent's (and every request that is not a reply is hashed equal to it here)."""

    @pytest.mark.parametrize("pattern", _STREAMS_WITHOUT_A_PARENT)
    def test_every_request_that_is_not_a_reply_equals_the_golden_of_run(self, pattern):
        streamed = runs.run_case(f"{pattern}_stream")

        assert "raised" not in streamed["outcome"], streamed["outcome"]
        want = _GOLDEN[pattern]["requests"]
        assert len(streamed["requests"]) == len(want), pattern
        for got, golden in zip(streamed["requests"], want, strict=True):
            if (
                golden["call"] != "generate_response"
            ):  # a reply asks for plain text in a stream
                assert got == golden, pattern
        assert streamed["outcome"]["chunks"], pattern

    def test_these_streams_have_no_golden_entry_of_their_own(self):
        for pattern in _STREAMS_WITHOUT_A_PARENT:
            assert f"{pattern}_stream" not in _GOLDEN
