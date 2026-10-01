"""Offline sanity tests for ``scripts/agents_bench.py`` (E1).

Pure-offline and fast: no live gate, no slow marker, no LLM. Each test's
docstring names the defect it guards against (repo convention). Patterns
follow ``tests/test_harness_bench.py``; decisions D-002/D-003 of
plan-2026-09-29T184639-65baa765.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import agents_bench as ab
import harness_bench as hb

#: Top-level imports the bench module may hold; everything heavier is lazy.
_STDLIB_ALLOWED = {
    "__future__",
    "argparse",
    "ast",
    "collections",
    "dataclasses",
    "datetime",
    "hashlib",
    "importlib",
    "inspect",
    "json",
    "math",
    "pathlib",
    "re",
    "statistics",
    "sys",
    "time",
    "typing",
}
_DIGEST = {"tag": "qwen3.5:4b", "digest": "2a654d98e6fb"}


def _row(task_id: str, trial: int, correct: bool, **extra) -> dict:
    """One synthetic row carrying every per-row fact the bench records."""
    row = {
        "task_id": task_id,
        "category": extra.pop("category", "single_tool"),
        "arm": "legacy",
        "trial": trial,
        "correct": correct,
        "success": extra.pop("success", correct),
        "stop_reason": extra.pop("stop_reason", "answered"),
        "iterations": 2,
        "tool_calls": 1,
        "tools_used": ["calculator"],
        "llm_calls": extra.pop("llm_calls", 3),
        "llm_errors": 0,
        "usage_missing": 0,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": extra.pop("total_tokens", 120),
        "latency_s": extra.pop("latency_s", 1.0),
        "error": None,
        "answer": "42",
    }
    row.update(extra)
    return row


def _synthetic_manifest(arm: str = "legacy", **overrides) -> dict:
    """A manifest carrying every required field, no live query needed."""
    manifest = {
        "bench_id": "synthetic",
        "block": "B0",
        "n_tasks": 2,
        "trials": 3,
        "model": "ollama_chat/qwen3.5:4b",
        "created_at": "2026-09-29T00:00:00Z",
        "prompt_bytes_sha256": "0" * 64,
        "tool_surface": {"arm_factory": arm},
        "fixture_hash": "1" * 64,
        "tasks_sha256": "1" * 64,
        "model_digest": dict(_DIGEST),
        "arm": {"name": arm, "factory": "synthetic"},
        "git_commit": "deadbeef",
        "temperature": 0.5,
        "limits": dict(ab.LIMITS),
        "wrapper_version": ab.WRAPPER_VERSION,
    }
    manifest.update(overrides)
    return manifest


#: Two tasks x 3 trials: t1 passes every trial, t2 fails its first trial only.
_ROWS = [
    _row("t1", 1, True, latency_s=1.0),
    _row("t2", 1, False, success=True, stop_reason="max_iterations", latency_s=4.0),
    _row("t1", 2, True, latency_s=2.0),
    _row("t2", 2, True, latency_s=3.0),
    _row("t1", 3, True, latency_s=5.0),
    _row("t2", 3, True, latency_s=6.0, category="typed_args"),
]


def _make_block(bdir: Path, arm: str, rows: list[dict], **manifest) -> None:
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / f"manifest_{arm}.json").write_text(
        json.dumps(_synthetic_manifest(arm, **manifest)), encoding="utf-8"
    )
    for row in rows:
        ab.append_row(bdir / f"rows_{arm}.jsonl", row)


class TestImportIsInert:
    """Offline `report` must never load fsm_llm/litellm (they open sockets)."""

    def test_plain_import_opens_no_socket_and_loads_no_fsm_llm(self):
        """Defect guarded: a module-scope `import fsm_llm` or litellm making
        the offline report fetch litellm's cost map over HTTP. Sockets are
        disabled in a fresh interpreter; the import and list-tasks must pass
        and leave fsm_llm/litellm unloaded."""
        code = (
            "import socket\n"
            "def boom(*a, **k):\n"
            "    raise AssertionError('socket opened at import time')\n"
            "socket.socket = boom\n"
            "socket.create_connection = boom\n"
            "import sys\n"
            f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
            "import agents_bench\n"
            "assert agents_bench.list_tasks(verify=True) == 0\n"
            "assert 'fsm_llm' not in sys.modules, 'fsm_llm loaded'\n"
            "assert 'litellm' not in sys.modules, 'litellm loaded'\n"
            "print('IMPORT-OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, proc.stderr
        assert "IMPORT-OK" in proc.stdout

    def test_top_level_imports_are_stdlib_only(self):
        """Defect guarded: someone adds `import litellm` (or fsm_llm) at module
        scope, re-opening the import-time socket path."""
        tree = ast.parse(Path(ab.__file__).read_text(encoding="utf-8"))
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        assert names <= _STDLIB_ALLOWED, names - _STDLIB_ALLOWED

    def test_stats_and_io_helpers_are_harness_benchs_not_copies(self):
        """Defect guarded (D-002): a third hand-kept copy of wilson/fisher/
        jsonl IO drifting from the pair test_bench_parity.py holds equal."""
        assert ab.wilson_ci is hb.wilson_ci
        assert ab.fisher_exact_two_sided is hb.fisher_exact_two_sided
        assert ab.append_row is hb.append_row
        assert ab.read_rows is hb.read_rows
        assert ab.BenchDataError is hb.BenchDataError
        source = Path(ab.__file__).read_text(encoding="utf-8")
        assert "def wilson_ci" not in source
        assert "def fisher_exact_two_sided" not in source

    def test_manifest_fields_extend_harness_six(self):
        """Defect guarded: an agent manifest missing one of the six fields
        that make two blocks comparable, or the agent-bench extras."""
        assert ab.MANIFEST_FIELDS[:6] == hb.MANIFEST_FIELDS
        for field in ("tasks_sha256", "trials", "temperature", "limits"):
            assert field in ab.MANIFEST_FIELDS
        assert "wrapper_version" in ab.MANIFEST_FIELDS


class TestTaskSet:
    """The fixture: ground truth that a reference solver reproduces."""

    def test_every_expected_answer_is_reproduced_by_its_reference_solver(self):
        """Defect guarded: a mistyped expected value turning a task into one
        no agent can pass (or every agent passes). Each reference solver
        calls the SAME in-file tools the agent gets, fresh per task."""
        assert ab.verify_tasks() == []

    @pytest.mark.parametrize("task", ab.TASKS, ids=lambda t: t.id)
    def test_reference_answer_grades_correct(self, task):
        """Defect guarded: as above, per task, so a failure names the task."""
        tools = ab.make_tools()
        answer = task.reference({name: tools[name] for name in task.tools})
        assert ab.grade(task.grader, answer), (task.id, answer)

    def test_task_set_shape(self):
        """Defect guarded: a category silently emptied or the set shrunk
        below the pre-registered size (plan: ~38 tasks in 7 categories)."""
        cats = {t.category for t in ab.TASKS}
        assert cats == set(ab.CATEGORIES)
        assert 36 <= len(ab.TASKS) <= 40
        assert len({t.id for t in ab.TASKS}) == len(ab.TASKS)
        for task in ab.TASKS:
            if task.category == "distractor_tools":
                assert set(ab.DISTRACTORS) <= set(task.tools)

    def test_numeric_answers_never_appear_in_their_prompt(self):
        """Defect guarded: the numeric grader accepting an answer that only
        echoes the prompt's own numbers."""
        for task in ab.TASKS:
            if task.grader["kind"] != "numeric":
                continue
            in_prompt = ab.numbers_in(task.prompt)
            for value in task.grader["values"]:
                assert all(abs(n - value) > task.grader["tol"] for n in in_prompt)

    def test_flaky_tool_fails_once_per_fresh_tool_set(self):
        """Defect guarded: tool state leaking across trials, so trial 2 of
        er-flaky never sees the error it exists to measure."""
        for _ in range(2):
            tools = ab.make_tools()
            with pytest.raises(RuntimeError, match="retry"):
                tools["station_reading"]("ST-7")
            assert json.loads(tools["station_reading"]("ST-7"))["celsius"] == 18.4

    def test_error_tools_raise_instructive_errors(self):
        """Defect guarded: an error_recovery task whose error message gives
        the model nothing to recover with."""
        tools = ab.make_tools()
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            tools["days_between"]("March 5, 2024", "2024-04-25")
        with pytest.raises(ValueError, match="UPPERCASE"):
            tools["check_inventory"]("ef-400")
        with pytest.raises(ValueError, match="lb"):
            tools["convert_units"](150, "pounds", "kg")
        with pytest.raises(ValueError, match="thousands"):
            tools["calculator"]("2,480 * 0.15")

    def test_calculator_refuses_non_arithmetic(self):
        """Defect guarded: the calculator evaluating arbitrary Python."""
        calc = ab.make_tools()["calculator"]
        for expr in ("__import__('os')", "open('x')", "2 ** 1000"):
            with pytest.raises(ValueError):
                calc(expr)
        assert calc("(2 + 3) * 4 / 8") == "2.5"


class TestTasksSha256:
    """tasks_sha256 pins tasks, graders and tool source; nothing else."""

    def test_hash_is_deterministic_hex(self):
        """Defect guarded: a hash that wobbles between calls un-pins every
        manifest."""
        h1 = ab.tasks_sha256()
        assert h1 == ab.tasks_sha256()
        assert len(h1) == 64 and int(h1, 16) >= 0

    def test_registering_an_arm_does_not_change_the_hash(self, monkeypatch):
        """Defect guarded: adding an arm later changing the task hash, so B1
        could never be compared with B0."""
        before = ab.tasks_sha256()
        monkeypatch.setitem(ab.ARMS, "new_arm", lambda tools, model: None)
        monkeypatch.setattr(ab, "LIMITS", {**ab.LIMITS, "max_iterations": 99})
        assert ab.tasks_sha256() == before

    def test_hashed_region_holds_the_fixture_and_not_the_arms(self):
        """Defect guarded: the markers drifting so the hash covers the arm
        registry (changes with every arm) or misses the tasks/graders."""
        text = Path(ab.__file__).read_text(encoding="utf-8")
        region = text[text.index("# --- BEGIN TASKS") : text.index("# --- END TASKS")]
        for needle in ("def make_tools", "TASKS: tuple", "def grade", "COUNTRIES"):
            assert needle in region
        for needle in ("ARMS", "_fsm_advance_arm", "LIMITS"):
            assert needle not in region


class TestGrader:
    """Checker table: every kind, both outcomes."""

    @pytest.mark.parametrize(
        ("grader", "answer", "expected"),
        [
            ({"kind": "exact", "value": "Maskett"}, "Maskett.", True),
            ({"kind": "exact", "value": "Maskett"}, "  maskett ", True),
            ({"kind": "exact", "value": "Maskett"}, "The capital is Maskett", False),
            ({"kind": "numeric", "values": [47104101], "tol": 0.5},
             "The product is 47,104,101.", True),
            ({"kind": "numeric", "values": [46], "tol": 0},
             "From 2024-01-15 to 2024-03-01 there are 46 days.", True),
            ({"kind": "numeric", "values": [46], "tol": 0},
             "From 2024-01-15 to 2024-03-01 there are 45 days.", False),
            ({"kind": "numeric", "values": [99.5, 76.2], "tol": 0.01},
             "99.5 F and 76.2 m", True),
            ({"kind": "numeric", "values": [99.5, 76.2], "tol": 0.01},
             "99.5 F", False),
            ({"kind": "numeric", "values": [0.025], "tol": 0.0005}, "0.025 ORC", True),
            ({"kind": "tokens", "all_of": ["chidi", "okafor"]}, "Chidi Okafor.", True),
            ({"kind": "tokens", "all_of": ["chidi", "okafor"]}, "Chidi", False),
            ({"kind": "contains", "any_of": ["not found"]}, "Zandoria: NOT found!", True),
            ({"kind": "contains", "any_of": ["not found"]}, "The capital is X", False),
            ({"kind": "contains", "any_of": ["not found"], "none_of": ["research"]},
             "Not found, maybe Research", False),
            ({"kind": "contains", "any_of": ["couldn't find"]}, "I couldnt find it", True),
        ],
    )  # fmt: skip
    def test_checker_table(self, grader, answer, expected):
        """Defect guarded: a grader that is too strict (formatting noise
        fails a right answer) or too lax (a wrong number passes)."""
        assert ab.grade(grader, answer) is expected

    def test_unknown_kind_raises(self):
        """Defect guarded: a typo'd grader kind silently grading False."""
        with pytest.raises(ValueError, match="unknown grader kind"):
            ab.grade({"kind": "fuzzy"}, "x")


class TestMetricsMath:
    """Percentiles, pass@1, pass^k and the success x correct cross-tab."""

    def test_nearest_rank_percentile(self):
        """Defect guarded: an interpolating percentile (not the pre-registered
        nearest-rank one) shifting every latency number."""
        values = [5.0, 1.0, 4.0, 2.0, 3.0]
        assert ab.percentile_nearest_rank(values, 50) == 3.0
        assert ab.percentile_nearest_rank(values, 95) == 5.0
        assert ab.percentile_nearest_rank(list(range(1, 21)), 95) == 19
        assert ab.percentile_nearest_rank([7.0], 0) == 7.0
        assert ab.percentile_nearest_rank([], 50) is None

    def test_first_trial_pass1_mean_and_pass_hat_k(self):
        """Defect guarded: pass@1 computed over all rows (correlated trials)
        instead of first trials, or pass^k counting a task with a failure."""
        m = ab.compute_metrics(_ROWS, trials=3)
        assert (m["n_tasks"], m["n_rows"]) == (2, 6)
        assert m["k_pass1_first_trial"] == 1  # t2 trial 1 failed
        assert m["k_correct_rows"] == 5
        assert m["k_pass_hat_k"] == 1  # only t1 is correct on all 3
        assert m["wilson_pass1_first_trial"] == [
            round(v, 4) for v in hb.wilson_ci(1, 2)
        ]

    def test_pass_hat_k_needs_all_trials_present(self):
        """Defect guarded: an aborted block counting a task with 1 of 3 rows
        as passing all three."""
        m = ab.compute_metrics([_row("t1", 1, True)], trials=3)
        assert m["k_pass_hat_k"] == 0

    def test_first_trial_is_the_lowest_trial_not_file_order(self):
        """Defect guarded: 'first' read as first row in the file."""
        rows = [_row("t1", 2, False), _row("t1", 1, True)]
        assert ab.compute_metrics(rows, trials=2)["k_pass1_first_trial"] == 1

    def test_cross_tab_histogram_latency_and_categories(self):
        """Defect guarded: E5 at bench level -- a run claiming success on a
        wrong answer disappearing into an aggregate."""
        m = ab.compute_metrics(_ROWS, trials=3)
        assert m["success_vs_correct"] == {
            "success_correct": 5,
            "success_incorrect": 1,
            "fail_correct": 0,
            "fail_incorrect": 0,
        }
        assert m["stop_reasons"] == {"answered": 5, "max_iterations": 1}
        assert m["latency_p50_s"] == 3.0
        assert m["latency_p95_s"] == 6.0
        assert m["llm_calls_mean"] == 3.0 and m["llm_calls_median"] == 3.0
        cats = m["per_category"]
        assert cats["single_tool"]["tasks"] == 2  # a task's first row names it
        assert cats["single_tool"]["first_trial_correct"] == 1

    def test_empty_rows_do_not_crash(self):
        """Defect guarded: a block aborted before row 1 crashing its summary."""
        m = ab.compute_metrics([], trials=3)
        assert m["n_rows"] == 0 and m["latency_p50_s"] is None


class TestManifestGate:
    """An unmanifested block is not evidence."""

    def test_summary_refused_without_manifest(self, tmp_path: Path):
        """Defect guarded: rows committed without a manifest passing as
        evidence."""
        for row in _ROWS:
            ab.append_row(tmp_path / "rows_legacy.jsonl", row)
        with pytest.raises(ab.BenchDataError, match="manifest"):
            ab.write_summary(tmp_path, "legacy", status="complete")
        assert not (tmp_path / "summary_legacy.json").exists()

    @pytest.mark.parametrize("field", ["model_digest", "tasks_sha256", "trials"])
    def test_summary_refused_when_manifest_lacks_a_field(self, tmp_path, field):
        """Defect guarded: a manifest without the digest or task hash making
        two incomparable blocks look comparable."""
        manifest = _synthetic_manifest()
        del manifest[field]
        (tmp_path / "manifest_legacy.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with pytest.raises(ab.BenchDataError, match=field):
            ab.write_summary(tmp_path, "legacy", status="complete")

    def test_build_manifest_carries_every_field(self, monkeypatch):
        """Defect guarded: the builder drifting from the field list its own
        summary writer enforces."""
        monkeypatch.setattr(
            hb, "_model_digest", lambda tag: {"tag": tag, "digest": "x"}
        )
        manifest = ab.build_manifest(
            bench_id="x", block="B1", arm_name="fsm_advance", trials=3, model=ab.MODEL
        )
        assert all(field in manifest for field in ab.MANIFEST_FIELDS)
        assert manifest["arm"]["factory"].startswith('create_agent("react"')
        assert manifest["tasks_sha256"] == ab.tasks_sha256()
        assert manifest["n_preregistered"] == len(ab.TASKS) * 3
        assert manifest["model_digest"]["tag"] == "qwen3.5:4b"


class TestReport:
    """`report` recounts from raw rows and reads arms from file names."""

    def test_summary_matches_recount(self, tmp_path, monkeypatch, capsys):
        """Defect guarded: a summary number drifting from its own rows."""
        monkeypatch.setattr(ab, "BENCH_DATA", tmp_path)
        bdir = tmp_path / "synthetic" / "B0"
        _make_block(bdir, "legacy", _ROWS)
        summary = ab.write_summary(bdir, "legacy", status="complete")
        assert summary["metrics"]["k_pass1_first_trial"] == 1
        assert ab.report("synthetic") == 0
        out = capsys.readouterr().out
        assert "pass@1 first trial: 1/2" in out
        assert "MISMATCH" not in out

    def test_tampered_summary_is_a_mismatch_and_exit_one(
        self, tmp_path, monkeypatch, capsys
    ):
        """Defect guarded: a hand-edited summary going unnoticed."""
        monkeypatch.setattr(ab, "BENCH_DATA", tmp_path)
        bdir = tmp_path / "synthetic" / "B0"
        _make_block(bdir, "legacy", _ROWS)
        summary = ab.write_summary(bdir, "legacy", status="complete")
        summary["metrics"]["k_pass1_first_trial"] = 2  # the lie
        (bdir / "summary_legacy.json").write_text(json.dumps(summary), encoding="utf-8")
        assert ab.report("synthetic") == 1
        assert "MISMATCH k_pass1_first_trial" in capsys.readouterr().out
        assert ab.main(["report", "synthetic"]) == 1

    def test_unregistered_arm_block_still_recounts(self, tmp_path, monkeypatch, capsys):
        """Defect guarded: `report` reading arm labels from ARMS, so B0 rows
        of an arm deleted later (native_fc at step 11) stop recounting."""
        monkeypatch.setattr(ab, "BENCH_DATA", tmp_path)
        bdir = tmp_path / "synthetic" / "B0"
        _make_block(bdir, "retired_arm", _ROWS)
        ab.write_summary(bdir, "retired_arm", status="complete")
        assert "retired_arm" not in ab.ARMS
        assert ab.report("synthetic") == 0
        assert "[retired_arm]" in capsys.readouterr().out

    def test_pair_fisher_matches_direct_call(self, tmp_path, monkeypatch, capsys):
        """Defect guarded: the printed comparison using other arithmetic than
        the pre-registered Fisher rule (D-003)."""
        monkeypatch.setattr(ab, "BENCH_DATA", tmp_path)
        _make_block(tmp_path / "synthetic" / "B0", "legacy", _ROWS)
        all_pass = [dict(r, correct=True) for r in _ROWS]
        _make_block(tmp_path / "synthetic" / "B1", "runtime_native", all_pass)
        rc = ab.report("synthetic", ["B0", "B1"], ["B1/runtime_native:legacy"])
        assert rc == 0
        out = capsys.readouterr().out
        expected = hb.fisher_exact_two_sided(2, 2, 1, 2)
        assert f"pass@1 first trial: 2/2 vs 1/2 p={expected:.4f}" in out

    def test_pair_across_digests_is_refused(self, tmp_path, monkeypatch, capsys):
        """Defect guarded: a re-pulled model making blocks incomparable while
        the report compares away."""
        monkeypatch.setattr(ab, "BENCH_DATA", tmp_path)
        _make_block(tmp_path / "synthetic" / "B0", "legacy", _ROWS)
        _make_block(
            tmp_path / "synthetic" / "B1",
            "runtime_native",
            _ROWS,
            model_digest={"tag": "qwen3.5:4b", "digest": "other"},
        )
        assert ab.report("synthetic", pairs=["runtime_native:legacy"]) == 1
        out = capsys.readouterr().out
        assert "REFUSING" in out and "p=" not in out

    def test_ambiguous_bare_arm_in_pair_raises(self, tmp_path, monkeypatch):
        """Defect guarded: a bare arm label present in two blocks silently
        resolving to one of them."""
        monkeypatch.setattr(ab, "BENCH_DATA", tmp_path)
        _make_block(tmp_path / "synthetic" / "B0", "legacy", _ROWS)
        _make_block(tmp_path / "synthetic" / "B1", "legacy", _ROWS)
        with pytest.raises(ab.BenchDataError, match="BLOCK/ARM"):
            ab.report("synthetic", pairs=["legacy:legacy"])

    def test_missing_bench_raises_and_main_exits_one(self, tmp_path, monkeypatch):
        """Defect guarded: an empty path reported as a clean zero-block run."""
        monkeypatch.setattr(ab, "BENCH_DATA", tmp_path)
        with pytest.raises(ab.BenchDataError, match="no such bench"):
            ab.report("never-registered")
        assert ab.main(["report", "never-registered"]) == 1


class TestCallMeter:
    """The bench-local completion wrapper (token and call capture)."""

    def test_counts_calls_and_reads_usage_defensively(self):
        """Defect guarded: litellm renaming or dropping `usage` fields (they
        vary across its range) crashing a live block or zeroing the count."""
        responses = [
            SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=12, completion_tokens=20, total_tokens=32
                )
            ),
            {"usage": {"prompt_tokens": 5, "completion_tokens": 1}},
            SimpleNamespace(),  # streamed: no usage at all
            SimpleNamespace(usage=SimpleNamespace(prompt_tokens=None)),
        ]
        fake = SimpleNamespace(completion=lambda **kw: responses[kw["i"]])
        meter = ab.CallMeter()
        restore = ab.install_meter(meter, [(fake, "completion")])
        for i in range(4):
            fake.completion(i=i)
        restore()
        snap = meter.snapshot()
        assert snap["llm_calls"] == 4
        assert snap["usage_missing"] == 1
        assert (snap["prompt_tokens"], snap["completion_tokens"]) == (17, 21)
        assert snap["total_tokens"] == 38

    def test_async_binding_and_raising_call_are_counted_and_restored(self):
        """Defect guarded: `acompletion` wrapped as a sync function (returns
        an un-awaited coroutine) or a failing call vanishing from the count."""
        import asyncio

        async def acompletion(**kw):
            return {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        def completion(**kw):
            raise RuntimeError("provider down")

        fake = SimpleNamespace(acompletion=acompletion, completion=completion)
        meter = ab.CallMeter()
        restore = ab.install_meter(meter, [(fake, "acompletion"), (fake, "completion")])
        asyncio.run(fake.acompletion())
        with pytest.raises(RuntimeError):
            fake.completion()
        restore()
        assert fake.acompletion is acompletion and fake.completion is completion
        snap = meter.snapshot()
        assert (snap["llm_calls"], snap["llm_errors"], snap["total_tokens"]) == (
            2,
            1,
            2,
        )

    def test_completion_targets_cover_every_binding(self):
        """Defect guarded: a binding left unwrapped (core `from litellm import
        completion` in llm.py), so legacy calls go uncounted."""
        import litellm

        import fsm_llm.llm

        targets = {(mod.__name__, attr) for mod, attr in ab._completion_targets()}
        assert targets == {
            (litellm.__name__, "completion"),
            (litellm.__name__, "acompletion"),
            (fsm_llm.llm.__name__, "completion"),
        }

    def test_one_classifier_call_is_one_count(self, monkeypatch):
        """Defect guarded: the classifier, which sends through the LLM layer's
        binding, going uncounted or counted twice by the real target list."""
        import litellm

        import fsm_llm.llm
        from fsm_llm.classification import Classifier
        from fsm_llm.definitions import ClassificationSchema, IntentDefinition

        def fake_completion(**kw):
            message = SimpleNamespace(content='{"intent": "buy", "confidence": 0.9}')
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
            )

        monkeypatch.setattr(litellm, "completion", fake_completion)
        monkeypatch.setattr(fsm_llm.llm, "completion", fake_completion)
        monkeypatch.setattr(
            fsm_llm.llm, "get_supported_openai_params", lambda model: []
        )
        schema = ClassificationSchema(
            intents=[
                IntentDefinition(name="buy", description="b"),
                IntentDefinition(name="browse", description="x"),
            ],
            fallback_intent="browse",
        )
        meter = ab.CallMeter()
        restore = ab.install_meter(meter, ab._completion_targets())
        try:
            result = Classifier(schema, model="gpt-4o").classify("I want it")
        finally:
            restore()
        assert result.intent == "buy"
        snap = meter.snapshot()
        assert snap["llm_calls"] == 1
        assert (snap["prompt_tokens"], snap["completion_tokens"]) == (7, 3)


class _FakeResult:
    def __init__(self, answer: str, *, calls: int, success: bool = True):
        self.answer = answer
        self.success = success
        self.stop_reason = "answered" if success else "max_iterations"
        self.trace = SimpleNamespace(
            tool_calls=[SimpleNamespace(tool_name="calculator")] * calls,
            total_iterations=calls + 1,
        )


class TestRunEndToEnd:
    """run_block with a mock arm: rows, manifest, summary, refusals."""

    @pytest.fixture
    def offline(self, tmp_path, monkeypatch):
        """BENCH_DATA in tmp, a stub digest/commit, a fake completion module."""
        monkeypatch.setattr(ab, "BENCH_DATA", tmp_path)
        monkeypatch.setattr(hb, "_model_digest", lambda tag: dict(_DIGEST))
        monkeypatch.setattr(hb, "_git_commit", lambda: "cafebabe")
        fake = SimpleNamespace(
            completion=lambda **kw: {
                "usage": {"prompt_tokens": 10, "completion_tokens": 2}
            }
        )
        monkeypatch.setattr(ab, "_completion_targets", lambda: [(fake, "completion")])
        two = tuple(t for t in ab.TASKS if t.id in ("st-capital", "er-flaky"))
        monkeypatch.setattr(ab, "TASKS", two)
        return fake

    def _arm(self, fake):
        """A mock agent: solves with the reference, calls the fake LLM once;
        crashes on er-flaky's first trial to exercise the error path."""
        seen: list[str] = []

        def factory(tools, model):
            def run(prompt):
                task = next(t for t in ab.TASKS if t.prompt == prompt)
                fake.completion(model=model)
                seen.append(task.id)
                if task.id == "er-flaky" and seen.count("er-flaky") == 1:
                    raise TimeoutError("AgentTimeoutError stand-in")
                return _FakeResult(task.reference(tools), calls=len(tools))

            return SimpleNamespace(run=run)

        return factory

    def test_rows_manifest_and_summary(self, offline, tmp_path, monkeypatch):
        """Defect guarded: the live path writing rows that the offline recount
        cannot reproduce, or losing the per-row facts the plan lists."""
        monkeypatch.setitem(ab.ARMS, "mock", self._arm(offline))
        summary = ab.run_block("agents-react", "B9", "mock", trials=2)
        bdir = tmp_path / "agents-react" / "B9"
        rows = ab.read_rows(bdir / "rows_mock.jsonl")
        assert [(r["task_id"], r["trial"]) for r in rows] == [
            ("st-capital", 1),
            ("er-flaky", 1),
            ("st-capital", 2),
            ("er-flaky", 2),
        ]
        crashed = rows[1]
        assert crashed["correct"] is False and crashed["stop_reason"] == "timeout"
        assert crashed["error"].startswith("TimeoutError")
        good = rows[3]
        assert good["correct"] is True and good["llm_calls"] == 1
        assert good["total_tokens"] == 12 and good["tool_calls"] == 1
        for key in ("iterations", "latency_s", "answer", "success", "category"):
            assert key in good
        manifest = json.loads((bdir / "manifest_mock.json").read_text())
        assert manifest["git_commit"] == "cafebabe" and manifest["trials"] == 2
        assert summary["status"] == "complete"
        assert summary["metrics"]["k_pass1_first_trial"] == 1
        assert summary["metrics"]["k_pass_hat_k"] == 1
        assert ab.report("agents-react") == 0

    def test_existing_rows_refuse_a_second_run(self, offline, tmp_path, monkeypatch):
        """Defect guarded (D-002): re-sampling a block until a number looks
        good. The refusal fires before the meter or any arm loads."""
        monkeypatch.setitem(ab.ARMS, "mock", self._arm(offline))
        bdir = tmp_path / "agents-react" / "B0"
        bdir.mkdir(parents=True)
        (bdir / "rows_mock.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(ab.BenchDataError, match="run ONCE"):
            ab.run_block("agents-react", "B0", "mock")

    def test_registered_manifest_is_kept_and_checked(
        self, offline, tmp_path, monkeypatch
    ):
        """Defect guarded: `run` silently overwriting the committed
        pre-registration, or running against a changed task set."""
        monkeypatch.setitem(ab.ARMS, "mock", self._arm(offline))
        path = ab.register_block("agents-react", "B0", "mock", trials=1)
        created = json.loads(path.read_text())["created_at"]
        with pytest.raises(ab.BenchDataError, match="registered ONCE"):
            ab.register_block("agents-react", "B0", "mock", trials=1)
        with pytest.raises(ab.BenchDataError, match="trials"):
            ab.run_block("agents-react", "B0", "mock", trials=2)
        ab.run_block("agents-react", "B0", "mock", trials=1)
        assert json.loads(path.read_text())["created_at"] == created

    def test_unknown_arm_is_refused(self, offline):
        """Defect guarded: a typo'd --arm minting an unregistered arm."""
        with pytest.raises(ab.BenchDataError, match="unknown arm"):
            ab.run_block("x", "B0", "legacyy")
        with pytest.raises(ab.BenchDataError, match="unknown arm"):
            ab.register_block("x", "B0", "legacyy")


class TestArms:
    """The registered arms build their agents without an LLM call."""

    @pytest.mark.parametrize(("arm", "cls"), [("fsm_advance", "ReactAgent")])
    def test_arm_builds_its_agent_class(self, arm, cls):
        """Defect guarded: an arm factory broken by a signature change,
        found only after the live block has started."""
        tools = ab.make_tools()
        agent = ab.ARMS[arm]({"calculator": tools["calculator"]}, ab.MODEL)
        assert type(agent).__name__ == cls
        assert agent.config.max_iterations == ab.LIMITS["max_iterations"]
        assert agent.config.timeout_seconds == ab.LIMITS["timeout_seconds"]
        assert agent.config.temperature == ab.LIMITS["temperature"]
        assert agent.config.max_tokens == ab.LIMITS["max_tokens"]

    def test_b1_comparison_limits_equal_b0(self):
        """Defect guarded: B1 registered with limits or a task set that differ
        from B0's committed manifest, so the B1/B0 pair compares two things."""
        b0 = json.loads(
            (ab.BENCH_DATA / "agents-react" / "B0" / "manifest_legacy.json").read_text(
                encoding="utf-8"
            )
        )
        assert ab.tasks_sha256() == b0["tasks_sha256"]
        assert ab.LIMITS == b0["limits"]
        assert ab.TRIALS == b0["trials"]
        assert ab.MODEL == b0["model"]
        assert ab.WRAPPER_VERSION == b0["wrapper_version"]

    @pytest.mark.parametrize("label", ["legacy", "native_fc"])
    def test_b0_labels_are_never_reused_for_new_rows(
        self, label, tmp_path, monkeypatch
    ):
        """Defect guarded: new rows of changed code written under a B0 arm
        label, so `--pair` compares a label with itself across code."""
        monkeypatch.setattr(ab, "BENCH_DATA", tmp_path)
        assert label not in ab.ARMS
        with pytest.raises(ab.BenchDataError, match="unknown arm"):
            ab.register_block("agents-react", "B9", label)
        with pytest.raises(ab.BenchDataError, match="unknown arm"):
            ab.run_block("agents-react", "B9", label)
        assert not any(tmp_path.iterdir())


class TestCLI:
    @pytest.mark.parametrize(
        "argv",
        [["--help"], ["run", "--help"], ["register", "--help"], ["report", "--help"],
         ["list-tasks", "--help"]],
    )  # fmt: skip
    def test_help_exits_zero(self, argv, capsys):
        """Defect guarded: a --help that crashes mid-block."""
        with pytest.raises(SystemExit) as excinfo:
            ab.build_parser().parse_args(argv)
        assert excinfo.value.code == 0
        capsys.readouterr()

    def test_run_requires_a_registered_arm_choice(self, capsys):
        """Defect guarded: an unconstrained --arm string."""
        with pytest.raises(SystemExit) as excinfo:
            ab.build_parser().parse_args(
                ["run", "--bench-id", "x", "--block", "B0", "--arm", "gpt"]
            )
        assert excinfo.value.code != 0
        capsys.readouterr()

    def test_list_tasks_verify_exits_zero(self, capsys):
        """Defect guarded: a task set whose own verification fails."""
        assert ab.main(["list-tasks", "--verify"]) == 0
        out = capsys.readouterr().out
        assert "verify: ok" in out and f"{len(ab.TASKS)} tasks" in out
