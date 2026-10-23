"""Offline tests for ``scripts/skills_bench.py``: scripted agents drive the REAL skill
tools; each docstring names the defect it guards against."""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import skills_bench as sb

import fsm_llm.agents as agents_pkg
import fsm_llm.agents.agent_skills as adapter
import fsm_llm.llm as core_llm
from fsm_llm.agents import ToolCall
from fsm_llm.agents.agent_skills import create_skill_tools
from fsm_llm.constants import SKILL_READ_HINT

Q = {q.id: q for q in sb.QUESTIONS}
BAD = sb.hb.BenchDataError
BID = "skills-disclosure"
FIXTURE_SHA = "a62f855bc2b0254edc65a329304c498adec7a763e5b784ec60ab8835bcec6ecc"


def _trace(q, read=True):
    """The reference run on *q*: skill_load, then skill_read of the exact path."""
    calls = [("skill_load", {"name": q.skill})]
    calls += [("skill_read", {"name": q.skill, "path": q.path})] * read
    return [{"tool": t, "params": p} for t, p in calls]


def _exec(registry, step):
    call = ToolCall(tool_name=step["tool"], parameters=step["params"])
    return registry.execute(call)


def _arm(k, bad=0):
    return {"n_units": 10, "k_g1": k, "invalid_cells": bad}


class _Scripted:
    """A create_agent stand-in: runs the reference calls, sends each reply."""

    def __init__(self, registry, mode):
        self.registry, self.mode = registry, mode

    def run(self, task):
        if self.mode == "crash":
            raise RuntimeError("provider down")
        q = next(q for q in sb.QUESTIONS if task.endswith(q.text))
        steps = _trace(q, read=not q.control)
        for step in steps:
            reply = _exec(self.registry, step).result
            core_llm.completion(messages=[{"role": "tool", "content": reply}])
        trace = SimpleNamespace(total_iterations=len(steps))
        result = SimpleNamespace(answer=f"It is {q.token}.", success=True, trace=trace)
        return SimpleNamespace(**vars(result), stop_reason="answered")


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    return sb.fixture_catalog(tmp_path_factory.mktemp("skills"))


@pytest.fixture
def scripted(monkeypatch):
    """A fake provider and a scripted agent; returns the mode setter."""
    mode = {"value": "solve"}
    monkeypatch.setattr(core_llm, "completion", lambda **kw: "ok")
    make = lambda kind, registry, **kw: _Scripted(registry, mode["value"])  # noqa: E731
    monkeypatch.setattr(agents_pkg, "create_agent", make)
    return lambda value: mode.update(value=value)


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """Bench data and raw dirs in tmp; digest and commit need no daemon or git."""
    monkeypatch.setattr(sb, "BENCH_DATA", tmp_path / "data")
    monkeypatch.setattr(sb, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(sb.hb, "_model_digest", lambda tag: {"tag": tag, "digest": "d"})
    monkeypatch.setattr(sb.hb, "_git_commit", lambda: "c" * 40)
    return tmp_path / "data" / BID


class TestFixture:
    def test_register_time_asserts_hold_and_the_shape_is_as_designed(self, catalog):
        """Guards: a fixture that cannot be answered, leaks its answers or moved."""
        kinds = Counter(q.kind for q in sb.QUESTIONS)  # 2+2+2+4 read-needed, 2 controls
        assert sb.check_fixture(catalog) == [] and len(catalog.names) == 16
        assert sorted(kinds.values()) == [2, 2, 2, 2, 4] and kinds["control"] == 2
        assert len({q.skill for q in sb.QUESTIONS}) == 10  # 6 more are distractors
        assert sum(len(files) == 4 for *_, files in sb.SKILLS) == 2
        assert sb.fixture_sha256() == FIXTURE_SHA  # an edit is a new fixture

    def test_a_token_in_a_body_is_reported(self, catalog, monkeypatch):
        """Guards: an answer readable without the disclosure step."""
        name, desc, body, files = sb.SKILLS[0]
        monkeypatch.setattr(sb, "SKILLS", ((name, desc, body + " 7341", files),))
        assert any("token is in" in p for p in sb.check_fixture(catalog))


class TestGrader:
    def test_the_reference_solver_passes_all_questions(self, catalog):
        """Guards: a G1 no run can reach; the token is read from the real reply."""
        for q in sb.QUESTIONS:
            seen, registry = [], create_skill_tools(catalog)
            sb.spy_registry(registry, seen)
            for step in _trace(q, read=not q.control):
                _exec(registry, step)
            assert q.token in sb.squash(seen[-1]["reply"])
            grades = sb.grade(q, f"It is {q.token}.", seen)
            assert grades["g1"] and grades["g4"]
            assert sb.shape(q, "", seen, grades, "answered") == "pass"

    def test_the_grader_can_fail_a_run_and_normalises_the_answer(self):
        """Guards: a guess or a token dump counted as success; '7,341' a miss."""
        q, no_read = Q["q01"], _trace(Q["q01"], read=False)
        assert sb.grade(q, q.token, [])["g1"] is False
        g = sb.grade(q, q.token, no_read)
        assert (g["g2"], g["g1"], g["g3"]) == (True, False, False)
        assert sb.shape(q, q.token, no_read, g, "answered") == "other"
        assert all(sb.grade(q, a, _trace(q))["g1"] for a in ("7,341 KB", "7 341"))
        dump = sb.grade(Q["q05"], f"8801 or {q.token}", _trace(Q["q05"], read=False))
        assert (dump["g1"], dump["g4"], dump["g3"]) == (True, False, None)

    @pytest.mark.parametrize(
        ("calls", "answer", "stop", "want"),
        [
            ([("skill_load", "gateway-ops", "")], "about 90 KB", "answered", "invented number"),
            ([("skill_load", "gateway-ops", "")], "I cannot tell.", "answered", "refusal"),
            ([("skill_read", "gateway-ops", "x.md")], "5", "answered", "wrong file"),
            ([("skill_load", "pdf", "")], "5", "answered", "wrong skill"),
            ([("skill_load", "gateway-ops", "")] * 3, "5", "answered", "load-loop"),
            ([("skill_load", "gateway-ops", "")], "5", "max_iterations", "budget"),
        ],
    )  # fmt: skip
    def test_failure_shapes(self, calls, answer, stop, want):
        """Guards: every failure collapsing into one 'wrong' bucket."""
        trace = [{"tool": t, "params": {"name": n, "path": p}} for t, n, p in calls]
        grades = sb.grade(Q["q01"], answer, trace)
        assert sb.shape(Q["q01"], answer, trace, grades, stop) == want


class TestHintAblation:
    def _replies(self, catalog, hint_on):
        registry = create_skill_tools(catalog)
        with sb.hint_ablation(hint_on):
            steps = (
                {"tool": "skill_load", "params": {"name": n}} for n in catalog.names
            )
            return {s["params"]["name"]: _exec(registry, s).result for s in steps}

    def test_off_differs_from_on_by_exactly_the_hint_line(self, catalog):
        """Guards: an ablation that moves the file list or claims, or leaks on."""
        before = adapter._first_page
        on, off = self._replies(catalog, True), self._replies(catalog, False)
        assert adapter._first_page is before and SKILL_READ_HINT  # restored
        for name in catalog.names:
            lines = on[name].split("\n")
            assert lines.count(SKILL_READ_HINT) == 1
            assert SKILL_READ_HINT not in off[name]
            assert [x for x in lines if x != SKILL_READ_HINT] == off[name].split("\n")
            assert all(path in off[name] for path in catalog.get(name).resources)


class TestRunTrial:
    def test_hint_seen_follows_the_arm_and_a_failed_ablation_is_invalid(
        self, catalog, scripted, bench, monkeypatch
    ):
        """Guards: hint_seen read from the wrong place; a failed ablation counted."""
        raw = bench.parent / "raw.json"

        def flags(arm):
            row = sb.run_trial(arm, Q["q01"], 1, catalog, raw)
            assert row["g1"] and row["shape"] == "pass"
            return [row[k] for k in ("hint_seen", "hint_in_reply", "hint_valid")]

        assert flags("hint_on") == [True, True, True]
        assert flags("hint_off") == [False, False, True]
        assert flags("native_4b") == [True, True, True]
        assert flags("native_4b_hint_off") == [False, False, True]  # native too
        assert json.loads(raw.read_text())["wire"], "raw requests are kept"
        monkeypatch.setattr(sb, "hint_ablation", lambda on: contextlib.nullcontext())
        assert flags("hint_off") == [True, True, False]  # kept, flagged invalid


class TestBlocks:
    def test_register_once_and_run_refusals(self, bench, scripted):
        """Guards: a rewritten pre-registration, a manifest-less or drifted run."""
        with pytest.raises(BAD, match="register first"):
            sb.run_block(BID, "B0")
        sb.register_block(BID, "B0")
        paths = sb.register_block(BID, "B1")
        assert [p.stem[9:] for p in paths] == list(sb.BLOCKS["B1"])
        manifest = json.loads(paths[2].read_text())
        assert manifest["model"] == sb.M9B and manifest["fixture_hash"] == FIXTURE_SHA
        assert set(manifest["src_sha256"]) == set(sb.SRC_PINS)
        with pytest.raises(BAD, match="ONCE"):  # registered once
            sb.register_block(BID, "B1")
        path = bench / "B0" / "manifest_hint_on.json"
        manifest = json.loads(path.read_text())
        manifest["src_sha256"]["src/fsm_llm/constants.py"] = "0" * 64
        path.write_text(json.dumps(manifest))
        with pytest.raises(BAD, match="src_sha256"):
            sb.run_block(BID, "B0")
        (bench / "B1" / "rows_react_9b.jsonl").write_text("{}\n")
        with pytest.raises(BAD, match="ONCE"):
            sb.run_block(BID, "B1")

    def test_b0_report_flags_a_tampered_summary(self, bench, scripted, capsys):
        """Guards: a report that trusts the committed summary."""
        sb.register_block(BID, "B0")
        out = sb.run_block(BID, "B0")
        assert {s["status"] for s in out.values()} == {"complete"}
        assert len(sb.hb.read_rows(bench / "B0" / "rows_hint_on.jsonl")) == 24
        assert sb.report(BID) == 0
        assert "no effect shown (diff 0, p=1.0000) -> REMOVE" in capsys.readouterr().out
        path = bench / "B0" / "summary_hint_off.json"
        summary = json.loads(path.read_text())
        summary["metrics"]["k_g1"] = 3
        path.write_text(json.dumps(summary))
        assert sb.report(BID) == 1 and "MISMATCH k_g1" in capsys.readouterr().out

    def test_three_errored_rows_abort_and_keep_the_rows(self, bench, scripted):
        """Guards: an outage burning a block that cannot be re-run (B1: model-major)."""
        scripted("crash")
        sb.register_block(BID, "B1")
        with pytest.raises(BAD, match="consecutive errored"):
            sb.run_block(BID, "B1")
        summary = json.loads((bench / "B1" / "summary_native_4b.json").read_text())
        assert summary["status"].startswith("aborted")
        metrics = summary["metrics"]
        assert (metrics["n_rows"], metrics["shapes"]) == (2, {"budget": 2})
        assert not (bench / "B1" / "rows_react_9b.jsonl").exists()
        assert sb.plan("B0")[4][0] == "hint_off"  # the first arm flips per question

    @pytest.mark.parametrize(
        ("k_on", "k_off", "want"),
        [(0, 0, "floor, inconclusive KEEP"), (1, 1, "floor, inconclusive KEEP"), (1, 0, "floor, inconclusive KEEP"),
         (0, 1, "floor, inconclusive KEEP"), (0, 4, "inconclusive KEEP"), (2, 1, "no effect shown REMOVE"),
         (10, 10, "no effect shown REMOVE"), (10, 4, "effect shown KEEP"), (9, 3, "effect shown KEEP"), (4, 10, "hint harmful REMOVE"),
         (10, 6, "inconclusive KEEP"), (6, 10, "inconclusive KEEP"), (4, 7, "inconclusive KEEP"),
         (9, 9, "no effect shown REMOVE"), (10, 9, "no effect shown REMOVE"), (3, 3, "no effect shown REMOVE")],
    )  # fmt: skip
    def test_the_pre_registered_rule(self, k_on, k_off, want):
        """Guards: a rule unlike the registered text (10/6: p > .05; floor <= 1 both: D-032)."""
        verdict, action = sb.decide_b0(_arm(k_on), _arm(k_off), True)
        assert f"{verdict.split(' (')[0]} {action}" == want

    def test_b1_pairs_native_4b_with_its_hint_off_arm_and_prints_a_fisher_p(
        self, bench, scripted, capsys
    ):
        """Guards: a hint-off pair not adjacent, not alternating or gated (D-032)."""
        order, arms = sb.plan("B1"), [a for a, *_ in sb.plan("B1")]
        assert len(order) == 96 and len(sb.plan("B0")) == 48
        assert arms[:8] == ["native_4b", "native_4b_hint_off"] * 2 + ["native_4b_hint_off", "native_4b"] * 2  # fmt: skip
        assert set(arms[48:72]) == {"react_9b"} and arms[72] == "native_9b"
        on, off = sb.ARMS["native_4b"], sb.ARMS["native_4b_hint_off"]
        assert on == ("native_fc", sb.M4B, True) and off == (*on[:2], False)
        sb.register_block(BID, "B1")
        sb.run_block(BID, "B1")
        assert len(sb.hb.read_rows(bench / "B1" / "rows_native_4b_hint_off.jsonl")) == 24  # fmt: skip
        assert sb.report(BID) == 0
        out = capsys.readouterr().out
        assert "Fisher p=1.0000 (informational, no gate)" in out and "B0 rule" not in out  # fmt: skip

    def test_an_incomplete_or_invalid_block_keeps_the_hint(self):
        """Guards: removing the hint on a block that proved nothing."""
        assert sb.decide_b0(_arm(5, 2), _arm(5, 1), True)[0].startswith("block invalid")
        assert sb.decide_b0(_arm(5), _arm(5), False)[0] == "block incomplete"


class TestHygiene:
    def test_import_opens_no_socket_and_loads_no_fsm_llm(self):
        """Guards: the offline report fetching litellm's cost map."""
        code = (
            "import socket, sys\nsocket.socket = None\n"
            f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\nimport skills_bench\n"
            "assert 'fsm_llm' not in sys.modules and 'litellm' not in sys.modules\n"
        )
        argv = [sys.executable, "-c", code]
        run = subprocess.run(argv, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr

    def test_no_second_copy_of_the_shared_helpers_and_ascii_only(self):
        """Guards: a copy of wilson/fisher/append_row (agents_bench D-002)."""
        source = Path(sb.__file__).read_text(encoding="utf-8")
        shared = r"^def (wilson_ci|fisher_exact_two_sided|append_row|read_rows)\b"
        assert not re.search(shared, source, re.M)
        assert source.isascii() and chr(0x2014) not in source
