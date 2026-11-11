"""Offline tests for ``scripts/synthesis_bench.py``: scripted models drive the REAL
host FSM, runner, library, gates and simulator; each docstring names the defect
it guards against. No network, no ``fsm_llm`` import by the script itself."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import synthesis_bench as sb

import fsm_llm.agents as agents_pkg
from fsm_llm.agents import ToolCall
from fsm_llm.definitions import (
    DataExtractionResponse,
    FieldExtractionResponse,
    FSMDefinition,
    LLMUsage,
)
from fsm_llm.synthesis import FSMLibrary, SynthesisPolicy, digest_of, run_gates
from fsm_llm.synthesis.gates import GATES
from tests.test_fsm_llm_synthesis.conftest import ScriptedRunnerLLM, ScriptedStrongLLM

BAD = sb.bc.BenchDataError
FIXTURE_SHA = "2f48c7469cf78549dcca25a36a7648d89bf583ee03c6de2918e6f679f4f8b0db"
TASKS = {t.id: t for t in sb.TASKS}
FAMILIES = ("collect", "route", "branch", "retry")


class _Strong(ScriptedStrongLLM):
    """The compile (and scenario writer) model: the reference child, then prose."""

    model = sb.MODEL

    def usage(self):
        return LLMUsage(calls=len(self.requests), prompt_tokens=7 * len(self.requests))


def _truth(task, field, message):
    """What a perfect model extracts from the user's message."""
    gold = task.gold.get(field)
    if gold is None or task.family == "route":
        return gold
    if task.family == "retry":
        found = re.search(r"\d+", message)
        return int(found[0]) if found else None
    return gold if gold.lower() in message.lower() else None


def _texts(task):
    """state id -> the reply the scripted model gives there: the state's own
    instruction, so the scripted user's rules see the real wording."""
    docs = (sb.host_definition(task), sb.hand_fsm(task))
    return {s: v["response_instructions"] for d in docs for s, v in d["states"].items()}


class _Run(ScriptedRunnerLLM):
    """The run model, perfect: extracts what the user said, asks what the state says."""

    model = sb.MODEL

    def __init__(self, task):
        super().__init__({}, _texts(task))
        self.task = task

    def extract_field(self, request):
        self.call_history.append(("extract_field", request))
        value = _truth(self.task, request.field_name, request.user_message or "")
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=float(value is not None),
            reasoning="",
            is_valid=value is not None,
        )

    def extract_bulk_data(self, request):
        """The bulk pass, the only channel for a later-turn correction."""
        self.call_history.append(("extract_bulk_data", request))
        found = {
            k: _truth(self.task, k, request.user_message or "") for k in self.task.gold
        }
        return DataExtractionResponse(
            extracted_data={k: v for k, v in found.items() if v is not None}
        )


class _Sim(ScriptedRunnerLLM):
    """The G6b model: answers every extraction with the gold value."""

    model = sb.MODEL

    def __init__(self, task):
        gold = {k: int(v) if v.isdigit() else v for k, v in task.gold.items()}
        super().__init__(gold, _texts(task))


class _Agent:
    """A ``create_agent`` stand-in: asks through the REAL ``ask_user`` tool."""

    def __init__(self, registry, task, asks, answer):
        self.registry, self.task, self.asks, self.answer = registry, task, asks, answer

    def run(self, prompt):
        for question in self.asks:
            call = ToolCall(tool_name="ask_user", parameters={"question": question})
            assert self.registry.execute(call).success
        trace = SimpleNamespace(total_iterations=len(self.asks) + 1)
        done = SimpleNamespace(answer=self.answer, success=True, trace=trace)
        return SimpleNamespace(**vars(done), stop_reason="answered")


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """Data and raw dirs in tmp; digest, commit and git state need no daemon or git."""
    monkeypatch.setattr(sb, "BENCH_DATA", tmp_path / "data")
    monkeypatch.setattr(sb, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(sb.bc, "_model_digest", lambda tag: {"tag": tag, "digest": "d"})
    monkeypatch.setattr(sb.bc, "_git_commit", lambda: "c" * 40)
    monkeypatch.setattr(sb.bc, "_git_dirty", lambda: False)
    return tmp_path / "data" / sb.BENCH_ID


@pytest.fixture
def rig(monkeypatch):
    """Scripted models for whichever task ``run_trial`` is on; returns the state."""
    now = {"compile_replies": None, "asks": None, "answer": None}
    real, real_make = sb.run_trial, sb.make_llm

    def spy(arm, task, *args, **kwargs):
        now["task"] = task
        return real(arm, task, *args, **kwargs)

    def make(role):
        task = now.get("task")
        if task is None:  # registering, not running
            return real_make(role)
        if role == "compile":
            replies = now["compile_replies"]
            return _Strong(
                [json.dumps(sb.hand_fsm(task)), "no scenarios"]
                if replies is None
                else replies
            )
        return (_Sim if role == "sim" else _Run)(task)

    def agent(kind, registry, **kwargs):
        asks = now["asks"] or [
            sb.hand_fsm(now["task"])["states"]["ask"]["response_instructions"]
        ]
        answer = now["answer"] or json.dumps(now["task"].gold)
        return _Agent(registry, now["task"], asks, answer)

    monkeypatch.setattr(sb, "run_trial", spy)
    monkeypatch.setattr(sb, "make_llm", make)
    monkeypatch.setattr(agents_pkg, "create_agent", agent)
    return now


class _Client:
    """A fake Ollama: ``ps_script`` answers (the last repeats), ``lines`` is the
    journal (``None`` = unreadable), ``probes`` the (wall, served) seconds."""

    def __init__(self, ps_script=(None,), lines=(), probes=((0.3, 0.2),) * 3):
        self.ps_script, self.lines, self.probes, self.calls = (
            list(ps_script),
            lines,
            list(probes),
            [],
        )

    def ps(self):
        self.calls.append("ps")
        models = self.ps_script.pop(0) if len(self.ps_script) > 1 else self.ps_script[0]
        if isinstance(models, Exception):
            raise models
        return {"models": models or []}

    def journal(self, minutes):
        self.calls.append("journal")
        return None if self.lines is None else list(self.lines)

    def probe(self):
        self.calls.append("probe")
        return self.probes.pop(0) if len(self.probes) > 1 else self.probes[0]


OURS = [{"name": "bonsai-2:27b", "context_length": 32768}]
OTHER = {"name": "llama3:8b", "context_length": 8192}
BIG = [{"name": "bonsai-2:27b", "context_length": 65536}]
CHAT = '[GIN] 2026/11/11 | 200 | 1m5s | POST "/api/chat"'


class TestFixture:
    @staticmethod
    def _gates(sub, tid, tmp_path):
        """The ``gates_passed`` the library recorded for task *tid*'s child."""
        task = TASKS[tid]
        report = run_gates(
            sb.hand_fsm(task), SynthesisPolicy(**sb.POLICY), return_keys=task.gold
        )
        library = FSMLibrary(tmp_path / tid / sub)
        return library.get_record(digest_of(report.definition)).provenance[
            "gates_passed"
        ]

    def test_the_shape_and_the_pin(self):
        """Guards: a fixture that is not 4 x 4, repeats an id or moved without a new block."""
        assert [t.family for t in sb.TASKS] == [f for f in FAMILIES for _ in range(4)]
        assert len(TASKS) == 16 and sb.fixture_sha256() == FIXTURE_SHA

    def test_every_hand_child_and_host_passes_the_real_gates_and_loads(self):
        """Guards: a reference FSM the compiler's own gates would refuse."""
        policy = SynthesisPolicy(**sb.POLICY)
        for t in sb.TASKS:
            report = run_gates(sb.hand_fsm(t), policy, return_keys=t.gold)
            assert report.passed, (t.id, report.repair_text())
            FSMDefinition(**sb.host_definition(t))

    def test_the_hand_children_are_four_builders_and_pass_every_task_offline(
        self, rig, tmp_path
    ):
        """Guards (the harness check): a bench no arm could pass, a scripted user
        that cannot answer, a grader that cannot be satisfied."""
        for t in sb.TASKS:
            row = sb.run_trial("hand-fsm", t, 1, tmp_path / t.id)
            assert row["correct"] and row["phase"] == "pass", (t.id, row)
            assert row["confused_turns"] == 0 and row["compile_calls"] == 0
            assert row["status"] == "ok" and row["source"] == "library"
        assert self._gates("hand1", "c1", tmp_path) == list(GATES)  # G6b never claimed

    def test_the_synthesized_arm_passes_through_the_simulator_and_reuses(
        self, rig, tmp_path
    ):
        """Guards: S without G6b, trials 2 and 3 compiling again, a lost library."""
        for tid in ("c1", "r1", "b1", "y1"):
            task = TASKS[tid]
            rows = [
                sb.run_trial("synth-fsm", task, n, tmp_path / tid) for n in (1, 2, 3)
            ]
            assert [r["correct"] for r in rows] == [True] * 3, (tid, rows[0])
            assert [r["source"] for r in rows] == ["compiled", "library", "library"]
            assert rows[0]["compile_calls"] == 2 and rows[1]["compile_calls"] == 0
            assert rows[0]["sim_calls"] > 0 and rows[2]["rounds"] == []
            assert self._gates("synth", tid, tmp_path) == [*GATES, "G6b"]

    def test_react_reads_its_answer_and_the_user_through_ask_user(self, rig, tmp_path):
        """Guards: R graded on prose it cannot give, or never talking to the user."""
        rig.update(asks=["Please share the passport and dates and purpose"])
        row = sb.run_trial("react", TASKS["c1"], 1, tmp_path)
        assert row["correct"] and row["turns"] == 1 and row["compile_calls"] == 0
        prose = "The passport is X1234567, dates 2026-05-01, purpose tourism."
        rig.update(answer=prose)
        assert sb.run_trial("react", TASKS["c1"], 1, tmp_path)[
            "correct"
        ]  # lenient tokens
        rig.update(answer="all collected")
        assert sb.run_trial("react", TASKS["c1"], 1, tmp_path)["phase"] == "wrong_value"


class TestUserAndGrader:
    def test_every_matching_rule_answers_in_order_and_the_fallback_counts(self):
        """Guards: a user that answers unasked facts, forgets, or hides confusion."""
        user = sb.User(TASKS["y1"].users)
        assert user.reply("Hi") == sb.FALLBACK and user.confused == 1
        assert "12" in user.reply("How many guests?") and "6" in user.reply("How many?")
        assert user.used[0] == 2 and "6" in user.reply("How many?")  # the last repeats
        both = sb.User(TASKS["c1"].users).reply("Your passport and travel dates?")
        assert "X1234567" in both and "2026-05-01" in both and "tourism" not in both

    @pytest.mark.parametrize("squashed", ["X-1234 567", "x1234567", "(X1234567)."])
    def test_squash_ignores_case_space_and_punctuation(self, squashed):
        """Guards: a correct value marked wrong over formatting."""
        assert sb.squash(squashed) == "x1234567"
        assert sb.squash(6.0) == sb.squash("6") == "6"

    def test_the_gold_passes_and_each_shape_can_fail(self):
        """Guards: a grader that cannot fail, or that fails a right answer."""
        c, b, y, r = (TASKS[i] for i in ("c1", "b1", "y1", "r1"))
        assert all(sb.grade(t, t.gold, [], 2)["correct"] for t in (c, b, y, r))
        wrong = {**c.gold, "purpose": "business"}
        g = sb.grade(c, wrong, [], 0)
        assert not g["correct"] and g["per_key"] == {"passport_no": True, "travel_dates": True, "purpose": False}  # fmt: skip
        assert not sb.grade(c, {**c.gold, "purpose": None}, [], 0)["correct"]
        assert not sb.grade(r, {r.id: r.gold[next(iter(r.gold))]}, [], 0)["correct"]
        assert not sb.grade(b, b.gold, ["Where is your destination?"], 0)["correct"]
        assert sb.grade(b, b.gold, ["What is your company?"], 0)["correct"]
        assert not sb.grade(y, y.gold, [], 1)["correct"]  # never re-asked


_BASE = dict(correct=False, arm="hand-fsm", family="collect", error=None, status=None, runner_reason=None, stop_reason=None)  # fmt: skip


class TestPhase:
    @pytest.mark.parametrize(
        ("change", "want"),
        [
            (dict(correct=True), "pass"),
            (dict(runner_reason="compile_failed"), "compile"),
            (dict(runner_reason="provider_error"), "provider"),
            (dict(error="LLMResponseError: timed out"), "provider"),
            (dict(error="KeyError: 'x'"), "host"),
            (dict(runner_reason="budget_spent"), "run_budget"),
            (dict(status="running"), "run_budget"),
            (dict(runner_reason="return_keys_missing"), "contract"),
            (dict(status="ok", family="route"), "wrong_route"),
            (dict(status="ok"), "wrong_value"),
            (dict(runner_reason="push_failed"), "host"),
            (dict(status=None), "host"),
            (dict(arm="react", stop_reason="max_iterations"), "run_budget"),
            (dict(arm="react", stop_reason="answered", error="Timeout"), "provider"),
            (dict(arm="react", stop_reason="answered", family="route"), "wrong_route"),
            (dict(arm="react", stop_reason="answered"), "wrong_value"),
        ],
    )  # fmt: skip
    def test_the_table(self, change, want):
        """Guards: every failure collapsing into one bucket, or the wrong one."""
        assert sb.phase({**_BASE, **change}) == want


def _arm(k, bad=0, n=16):
    return {"k": k, "n_units": n, "infra_first": bad}


class TestVerdict:
    @pytest.mark.parametrize(
        ("s", "r", "h", "bad", "want"),
        [
            (12, 4, 14, 0, "SUPPORTED"),
            (4, 12, 14, 0, "REFUTED"),
            (8, 6, 14, 0, "NOT SUPPORTED"),
            (10, 4, 14, 0, "NOT SUPPORTED"),  # p = 0.0732
            (7, 1, 10, 0, "SUPPORTED"),  # p = 0.0373, h at the floor
            (12, 4, 9, 0, "INCONCLUSIVE"),  # the hand-written arm cannot finish
            (12, 4, 14, 3, "INCONCLUSIVE"),  # S failures were infrastructure
            (12, 4, 14, 2, "SUPPORTED"),
            (15, 15, 16, 0, "INCONCLUSIVE"),  # ceiling
            (16, 15, 16, 0, "INCONCLUSIVE"),
            (16, 14, 16, 0, "NOT SUPPORTED"),
            (0, 0, 14, 0, "NOT SUPPORTED"),
        ],
    )  # fmt: skip
    def test_the_pre_registered_table(self, s, r, h, bad, want):
        """Guards: a verdict that ignores h, the ceiling, infrastructure failures or p."""
        arms = {"synth-fsm": _arm(s, bad), "react": _arm(r), "hand-fsm": _arm(h)}
        assert sb.decide(arms, True)[0] == want

    def test_an_incomplete_block_never_decides(self):
        """Guards: a verdict read off an aborted or short block."""
        arms = {"synth-fsm": _arm(12), "react": _arm(4), "hand-fsm": _arm(14)}
        assert sb.decide(arms, False)[0] == "INCONCLUSIVE"
        arms["react"] = _arm(4, n=15)
        assert sb.decide(arms, True)[0] == "INCONCLUSIVE"

    def test_the_minimum_gap_is_computed_not_typed(self):
        """Guards: a hand-typed gap table in the pre-registration."""
        assert (
            sb.min_gaps()[0] == 5 and sb.min_gaps()[3] == 10 and sb.min_gaps()[10] == 16
        )

    def test_the_one_model_sentence_is_the_registered_wording(self):
        """Guards: a report that hides that every role is one model."""
        for word in ("NO strong/cheap split", "same model under three scaffolds", "Q2_0", "n = 16"):  # fmt: skip
            assert word in sb.ONE_MODEL


class TestPreflight:
    @pytest.mark.parametrize(
        ("client", "confirm", "want"),
        [
            (_Client(), False, ""),
            (_Client([OURS]), False, ""),
            (_Client([[OTHER]]), False, "llama3:8b"),
            (_Client([BIG]), False, "65536"),
            (_Client([[{"name": "bonsai-2:27b"}]]), False, "context"),
            (_Client([OURS], lines=["x", CHAT]), False, "request line"),
            (_Client(lines=None), False, "journal"),
            (_Client(lines=None), True, ""),
            (_Client(probes=[(0.3, 0.2), (4.0, 0.5), (0.3, 0.2)]), False, "queue wait"),
            (_Client([OSError("refused")]), False, "/api/ps"),
        ],
    )  # fmt: skip
    def test_the_table(self, client, confirm, want):
        """Guards: a preflight that passes on a busy, foreign or unreadable server."""
        waits = []
        found = sb.preflight(client, confirm_exclusive=confirm, sleep=waits.append)
        assert (want in " ".join(found)) if want else found == []
        if not want:
            assert client.calls.count("probe") == 3 and waits == [sb.PROBE_SPACING] * 2

    @pytest.mark.parametrize("client", [_Client([[OTHER]]), _Client(lines=["x", CHAT])])
    def test_a_refusal_sends_no_probe(self, client):
        """Guards: a probe that forces the other job's model to reload."""
        assert sb.preflight(client, sleep=lambda s: None)
        assert "probe" not in client.calls

    def test_the_guard_reads_ps_and_names_a_foreign_load(self):
        """Guards: a block that goes on after the server changed under it."""
        assert sb.guard(_Client([OURS])) == [
            {"name": "bonsai-2:27b", "context_length": 32768}
        ]
        assert sb.guard(_Client([[]])) == []
        for ps in ([OTHER], BIG, [*OURS, OTHER]):
            with pytest.raises(sb.ForeignLoad):
                sb.guard(_Client([ps]))

    def test_no_context_is_forced_and_the_roles_have_their_timeouts(self):
        """Guards: a num_ctx override (it reloads the model), a missing timeout."""
        with sb.bc.captured_wire() as seen:
            for role in ("compile", "sim", "run"):
                with pytest.raises(Exception):  # the stub refuses the send
                    sb.make_llm(role).complete(_request())
        assert [s["timeout"] for s in seen] == [600.0, 300.0, 300.0]
        assert all(
            s["model"] == sb.MODEL and "num_ctx" not in json.dumps(s) for s in seen
        )
        with pytest.raises(BAD, match="one model"):
            sb.require_model("ollama_chat/qwen3.5:9b-q8_0")


def _request():
    from fsm_llm.definitions import CompletionRequest

    return CompletionRequest(messages=[{"role": "user", "content": "hi"}])


class TestWireSpy:
    def test_it_stamps_role_seconds_and_tokens_and_restores(self, monkeypatch):
        """Guards: a latency meter that loses the role, or a binding left replaced."""
        import fsm_llm.llm as core_llm

        reply = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3)
        )
        monkeypatch.setattr(core_llm, "completion", lambda **kw: reply)
        before = core_llm.completion
        with sb.wire_spy() as wire:
            core_llm.completion(timeout=600.0, messages=[])
            core_llm.completion(timeout=300.0, messages=[])
        assert core_llm.completion is before
        assert [w["role"] for w in wire] == ["compile", "run"]
        assert wire[0]["prompt_tokens"] == 11 and wire[0]["completion_tokens"] == 3
        assert all(w["seconds"] >= 0 for w in wire)


class TestRunTrial:
    def test_a_foreign_model_and_an_h_strong_call_are_errored_rows(
        self, rig, tmp_path, monkeypatch
    ):
        """Guards: an S row on another model, an H arm that quietly compiled."""
        make = sb.make_llm

        def foreign(role):
            llm = make(role)
            if role == "compile":
                llm.model = "ollama_chat/qwen3.5:9b-q8_0"
            return llm

        monkeypatch.setattr(sb, "make_llm", foreign)
        row = sb.run_trial("synth-fsm", TASKS["c1"], 1, tmp_path / "a")
        assert row["error"].startswith("one model only") and row["models_ok"] is False
        assert row["correct"] is False and row["turns"] == 0 and row["source"] is None
        monkeypatch.setattr(sb, "make_llm", make)
        monkeypatch.setattr(FSMLibrary, "put", lambda *a, **k: None)  # nothing seeded
        row = sb.run_trial("hand-fsm", TASKS["c1"], 1, tmp_path / "b")
        assert row["error"] == "hand-fsm made a strong call" and row["correct"] is False

    def test_an_exception_is_a_failed_row_and_raw_requests_are_kept(
        self, rig, tmp_path, monkeypatch
    ):
        """Guards: a crashed trial that ends the block, or a lost raw record."""

        def boom(*args, **kwargs):
            raise RuntimeError("provider down")

        monkeypatch.setattr(agents_pkg, "create_agent", boom)
        raw = tmp_path / "raw.json"
        row = sb.run_trial("react", TASKS["c1"], 1, tmp_path, raw)
        assert row["error"].startswith("RuntimeError") and row["correct"] is False
        assert row["phase"] == "provider"  # any error of an agent row is the model call
        assert json.loads(raw.read_text())["row"]["task_id"] == "c1"


class TestBlocks:
    @pytest.fixture(autouse=True)
    def small(self, monkeypatch):
        """Two tasks and two trials keep a whole block fast; the plan is the real one."""
        monkeypatch.setattr(
            sb, "TASKS", tuple(t for t in sb.TASKS if t.id in ("c1", "y1"))
        )
        monkeypatch.setattr(sb, "TRIALS", 2)

    def test_plan_keeps_a_task_together_and_alternates_the_first_arm(self):
        """Guards: S trials split apart (no reuse), or a fixed arm order (drift bias)."""
        order = sb.plan("s0")
        assert (
            len(order) == 12 and [t.id for _, t, _ in order] == ["c1"] * 6 + ["y1"] * 6
        )
        assert [a for a, _, _ in order[:3]] == ["synth-fsm", "hand-fsm", "react"]
        assert [a for a, _, _ in order[6:9]] == ["react", "hand-fsm", "synth-fsm"]

    def test_register_once_and_every_refusal(self, bench, monkeypatch):
        """Guards: a rewritten pre-registration, a manifest-less or tampered run."""
        client = _Client()
        with pytest.raises(BAD, match="register first"):
            sb.run_block("s0", client=client, sleep=lambda s: None)
        paths = sb.register_block("s0")
        assert [p.stem[9:] for p in paths] == list(sb.BLOCKS["s0"])
        manifest = json.loads(paths[0].read_text())
        assert manifest["model"] == sb.MODEL and manifest["trials"] == 2
        assert manifest["fixture_hash"] == sb.fixture_sha256()
        assert (
            set(manifest["src_sha256"]) == set(sb.SRC_PINS) and len(sb.SRC_PINS) == 11
        )
        assert set(manifest["wire"]) == {"compile", "run"} and "num_ctx" not in json.dumps(manifest)  # fmt: skip
        with pytest.raises(BAD, match="ONCE"):
            sb.register_block("s0")
        (bench / "s0" / "rows_react.jsonl").write_text("{}\n")
        with pytest.raises(BAD, match="ONCE"):
            sb.run_block("s0", client=client, sleep=lambda s: None)
        (bench / "s0" / "rows_react.jsonl").unlink()
        for key, value in (("src_sha256", {"src/flow_analysis.py": "0" * 64}), ("fixture_hash", "0")):  # fmt: skip
            path = bench / "s0" / "manifest_synth-fsm.json"
            saved = path.read_text()
            manifest = json.loads(saved)
            manifest[key] = (
                {**manifest.get(key, {}), **value} if isinstance(value, dict) else value
            )
            path.write_text(json.dumps(manifest))
            with pytest.raises(BAD, match=key):
                sb.run_block("s0", client=client, sleep=lambda s: None)
            path.write_text(saved)

    def test_run_refuses_a_busy_server_before_any_row(self, bench, rig, monkeypatch):
        """Guards: a run whose preflight was dropped (a queued run measures the queue)."""
        sb.register_block("s0")
        busy = _Client([[OTHER]])
        with pytest.raises(sb.Blocked, match="llama3:8b"):
            sb.run_block("s0", client=busy, sleep=lambda s: None)
        assert not list((bench / "s0").glob("rows_*")) and "probe" not in busy.calls
        monkeypatch.setattr(sb, "OllamaClient", lambda: busy)
        assert sb.main(["run", "s0"]) == sb.EXIT_BLOCKED

    def test_a_block_runs_summarises_and_the_report_recounts(self, bench, rig, capsys):
        """Guards: a report that trusts the committed summary; rows lost on a run."""
        sb.register_block("s0")
        out = sb.run_block("s0", client=_Client(), sleep=lambda s: None)
        assert {s["status"] for s in out.values()} == {"complete"}
        assert [len(sb.bc.read_rows(bench / "s0" / f"rows_{a}.jsonl")) for a in sb.BLOCKS["s0"]] == [4] * 3  # fmt: skip
        assert sb.report("s0") == 0
        text = capsys.readouterr().out
        assert "INCONCLUSIVE" in text and sb.ONE_MODEL in text  # n is 2 here, not 16
        path = bench / "s0" / "summary_hand-fsm.json"
        summary = json.loads(path.read_text())
        summary["metrics"]["k"] = 0
        path.write_text(json.dumps(summary))
        assert sb.report("s0") == 1 and "MISMATCH k" in capsys.readouterr().out

    def test_report_of_a_registered_block_says_so(self, bench, capsys):
        """Guards: a report that fails, or invents a result, before the run."""
        sb.register_block("s0")
        assert sb.report("s0") == 0
        out = capsys.readouterr().out
        assert "registered" in out and "NOT RUN" in out and sb.ONE_MODEL in out

    def test_three_errored_rows_abort_and_keep_the_rows(self, bench, rig):
        """Guards: an outage burning a block that cannot be re-run."""
        sb.register_block("s0")
        rig.update(compile_replies=[])  # every compile call raises
        real = sb.run_trial

        def crash(arm, task, *a, **k):
            row = real(arm, task, *a, **k)
            return {**row, "error": "LLMResponseError: down", "phase": "provider"}

        sb.run_trial = crash
        try:
            with pytest.raises(BAD, match="consecutive errored"):
                sb.run_block("s0", client=_Client(), sleep=lambda s: None)
        finally:
            sb.run_trial = real
        summary = json.loads((bench / "s0" / "summary_synth-fsm.json").read_text())
        rows = sum(len(sb.bc.read_rows(p)) for p in (bench / "s0").glob("rows_*"))
        assert summary["status"].startswith("aborted") and rows == 3

    def test_a_foreign_load_mid_run_aborts_and_keeps_the_rows(self, bench, rig):
        """Guards: a block that measures another job's load after it arrived."""
        sb.register_block("s0")
        client = _Client(
            [[], [], [], [], OURS, [OTHER]]
        )  # preflight, then 3 trials, then ours, then foreign
        with pytest.raises(sb.ForeignLoad):
            sb.run_block("s0", client=client, sleep=lambda s: None)
        summary = json.loads((bench / "s0" / "summary_synth-fsm.json").read_text())
        assert summary["status"] == "aborted: foreign-load"
        assert (
            1
            <= sum(len(sb.bc.read_rows(p)) for p in (bench / "s0").glob("rows_*"))
            < 12
        )

    def test_the_report_decides_from_first_trial_rows_of_a_complete_block(
        self, bench, capsys, monkeypatch
    ):
        """Guards: a verdict not read from rows, or from trials after the first."""
        monkeypatch.undo()  # the whole fixture of 16 tasks, not the two of this class
        monkeypatch.setattr(sb, "BENCH_DATA", bench.parent)
        (bench / "s0").mkdir(parents=True)
        for arm, k in (("synth-fsm", 12), ("hand-fsm", 14), ("react", 4)):
            for i, task in enumerate(sb.TASKS):
                for trial in (1, 2):
                    ok = trial == 1 and i < k
                    row = dict(task_id=task.id, family=task.family, arm=arm, trial=trial, correct=ok, phase="pass" if ok else "wrong_value", source="compiled", error=None)  # fmt: skip
                    sb.bc.append_row(bench / "s0" / f"rows_{arm}.jsonl", row)
            sb.write_summary(bench / "s0", arm, "complete", "t", {})
        assert sb.report("s0") == 0
        out = capsys.readouterr().out
        assert (
            "verdict: SUPPORTED" in out
            and "Fisher p = 0.0121" in out
            and sb.RULE in out
        )


class TestSmoke:
    @pytest.fixture(autouse=True)
    def small(self, monkeypatch):
        monkeypatch.setattr(sb, "TRIALS", 1)

    def test_it_refuses_like_run_and_writes_no_bench_data(self, bench, rig, capsys):
        """Guards: a smoke that skips the preflight, or leaves rows behind."""
        with pytest.raises(sb.Blocked):
            sb.smoke(["c1"], client=_Client([[OTHER]]), sleep=lambda s: None)
        rows = sb.smoke(["c1"], client=_Client(), sleep=lambda s: None)
        assert [r["arm"] for r in rows] == list(sb.BLOCKS["s0"])
        assert all(r["correct"] and r["ps"] == [] for r in rows)
        assert (sb.RAW_DIR / "smoke" / "synth-fsm__c1.json").is_file()
        assert not bench.exists() and sb.ONE_MODEL in capsys.readouterr().out
        with pytest.raises(BAD, match="no such task"):
            sb.smoke(["nope"], client=_Client())


class TestHygiene:
    def test_import_opens_no_socket_and_loads_no_fsm_llm_or_litellm(self):
        """Guards: the offline register and report touching the network or litellm."""
        code = (
            "import socket, sys\nsocket.socket = None\n"
            f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\nimport synthesis_bench\n"
            "assert 'fsm_llm' not in sys.modules and 'litellm' not in sys.modules\n"
        )
        run = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert run.returncode == 0, run.stderr

    def test_no_second_copy_of_the_shared_helpers_no_process_control_ascii_only(self):
        """Guards: a copy of wilson/fisher/append_row; code that could touch another
        job's process; a non-ASCII character or em dash."""
        source = Path(sb.__file__).read_text(encoding="utf-8")
        shared = r"^def (wilson_ci|fisher_exact_two_sided|append_row|read_rows)\b"
        assert not re.search(shared, source, re.M)
        process = r"\bkill|os\.kill|\bimport signal|\bfrom signal|\bsignal\.\w+\(|\bterminate\b"
        assert not re.search(process, source)  # the prose "a weaker signal" is allowed
        assert not re.search(r"confirm_exclusive\W*=\W*True", source)
        assert source.isascii() and chr(0x2014) not in source
        assert (
            "import agents_bench" not in source and "import skills_bench" not in source
        )
        assert source.count("LiteLLMInterface(") == 1  # the one construction site

    def test_register_and_report_never_build_a_client(self, bench, monkeypatch):
        """Guards: an offline command reaching the server (register reads tags only)."""
        monkeypatch.setattr(
            sb, "OllamaClient", lambda: pytest.fail("a client was built")
        )
        sb.register_block("s0")
        assert sb.report("s0") == 0
