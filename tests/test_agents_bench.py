"""Offline sanity tests for ``scripts/agents_bench.py`` (E1).

Pure-offline and fast: no live gate, no slow marker, no LLM. Each test's
docstring names the defect it guards against (repo convention). Patterns
follow ``tests/test_harness_bench.py``; decisions D-002/D-003 of
plan-2026-09-29T184639-65baa765.
"""

from __future__ import annotations

import ast
import hashlib
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
        "llm_request": {"timeout": 120.0, "retries": 0, "extra_kwargs": []},
        "agent_class": "fsm_llm.agents.react.ReactAgent",
        "run_cap": {"max_steps": 24, "formula": "x", "max_seconds": 180.0},
        "tool_schemas_sha256": {"t1": "2" * 64, "t2": "3" * 64},
        "first_request": {"t1": None, "t2": None},
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
        monkeypatch.setitem(ab.ARMS, "new_arm", lambda tools, model, llm: None)
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
        two = tuple(t for t in ab.TASKS if t.id in ("st-capital", "er-flaky"))
        monkeypatch.setattr(ab, "TASKS", two)
        manifest = ab.build_manifest(
            bench_id="x", block="B1", arm_name="fsm_advance", trials=3, model=ab.MODEL
        )
        assert all(field in manifest for field in ab.MANIFEST_FIELDS)
        assert manifest["arm"]["factory"].startswith('create_agent("react"')
        assert manifest["tasks_sha256"] == ab.tasks_sha256()
        assert manifest["n_preregistered"] == len(ab.TASKS) * 3
        assert manifest["model_digest"]["tag"] == "qwen3.5:4b"


def _native_arm(tools, model, llm_interface):
    """create_agent("native_fc", ...): the shape step 19's arm will take."""
    from fsm_llm.agents import create_agent

    return create_agent(
        "native_fc",
        ab.build_registry(tools),
        config=ab._agent_config(model),
        llm_interface=llm_interface,
    )


_TWO_IDS = ("st-capital", "ty-order")


class TestManifestDisclosure:
    """A new manifest records what its requests carried beyond the task text
    (review pass 7 concerns 2-4, D-029 item 18.4, D-036)."""

    @pytest.fixture
    def two_tasks(self, monkeypatch):
        """Two tasks (one with the order_total schema step 14 changed), a
        stub digest, and a provider that fails the test if anything is sent."""
        import fsm_llm.llm

        def no_send(**kw):
            raise AssertionError("the disclosure probe sent a provider request")

        monkeypatch.setattr(fsm_llm.llm, "completion", no_send)
        monkeypatch.setattr(hb, "_model_digest", lambda tag: dict(_DIGEST))
        two = tuple(t for t in ab.TASKS if t.id in _TWO_IDS)
        monkeypatch.setattr(ab, "TASKS", two)
        return two

    def test_manifest_records_timeout_class_cap_and_digests(self, two_tasks):
        """Defect guarded: a B2-vs-B0 pair that cannot show the per-request
        timeout (none at B0, 120 s now), the code path or the step ceiling,
        because the manifest never recorded them."""
        manifest = ab.build_manifest(
            bench_id="x", block="B9", arm_name="fsm_advance", trials=3, model=ab.MODEL
        )
        for field in ab.DISCLOSURE_FIELDS:
            assert field in manifest, field
        settings = manifest["llm_request"]
        # The FINAL request (D-040): core's Ollama preparation is in it.
        assert settings["timeout"] == 120.0
        assert settings["max_tokens"] == ab.LIMITS["max_tokens"]
        assert settings["reasoning_effort"] == "none"
        assert settings["temperature"] == 0  # an extraction call on Ollama
        assert settings["model"] == ab.MODEL
        assert "messages" not in settings and "tools" not in settings
        assert manifest["agent_class"] == "fsm_llm.agents.react.ReactAgent"
        assert manifest["run_cap"]["max_steps"] == 24
        assert manifest["run_cap"]["max_seconds"] == ab.LIMITS["timeout_seconds"]
        assert set(manifest["tool_schemas_sha256"]) == set(_TWO_IDS)
        first = manifest["first_request"]["st-capital"]
        assert first["tools_sha256"] is None
        for digest in ("system_sha256", "user_sha256", "settings_sha256"):
            assert len(first[digest]) == 64
        assert first["run_error"] == "AgentError"

    def test_tool_schema_digest_is_the_exact_unsorted_registry_bytes(self, two_tasks):
        """Defect guarded: a digest taken over key-sorted JSON (how the step-14
        record missed the `items`/`type` reorder and the added
        `additionalProperties`), so a schema byte change goes unrecorded."""
        tools = ab.make_tools()
        task = two_tasks[1]
        schemas = ab.build_registry(
            {n: tools[n] for n in task.tools}
        ).get_json_schemas()
        disclosure = ab.request_disclosure("fsm_advance", ab.MODEL)
        expected = hashlib.sha256(json.dumps(schemas).encode("utf-8")).hexdigest()
        assert disclosure["tool_schemas_sha256"][task.id] == expected
        reordered = {"items": {"type": "number"}, "type": "array"}
        original = {"type": "array", "items": {"type": "number"}}
        assert reordered == original
        assert ab.sha256_json(reordered) != ab.sha256_json(original)

    def test_native_first_request_sends_the_registry_schema_bytes(
        self, two_tasks, monkeypatch
    ):
        """Defect guarded: the first-request digest reading something other
        than what the agent hands its interface: for a native tool-calling
        arm its `tools=` must be the registry's schema bytes, and its system
        prompt the same for every task."""
        monkeypatch.setitem(ab.ARMS, "native_probe", _native_arm)
        disclosure = ab.request_disclosure("native_probe", ab.MODEL)
        assert disclosure["agent_class"].endswith("NativeFunctionCallingReactAgent")
        firsts = disclosure["first_request"]
        for task_id in _TWO_IDS:
            assert (
                firsts[task_id]["tools_sha256"]
                == disclosure["tool_schemas_sha256"][task_id]
            )
        assert len({firsts[t]["system_sha256"] for t in _TWO_IDS}) == 1
        assert disclosure["run_cap"]["formula"].startswith("2 x max_iterations")

    def test_an_arm_without_a_step_ceiling_records_none(self, two_tasks, monkeypatch):
        """Defect guarded: a non-agent arm (or one that sends nothing) crashing
        the registration instead of recording what it can."""

        def silent(tools, model, llm_interface):
            return SimpleNamespace(run=lambda prompt: None)

        monkeypatch.setitem(ab.ARMS, "silent", silent)
        disclosure = ab.request_disclosure("silent", ab.MODEL)
        assert disclosure["llm_request"] is None
        assert disclosure["run_cap"] is None
        assert disclosure["first_request"] == {t: None for t in _TWO_IDS}
        assert disclosure["agent_class"] == "types.SimpleNamespace"

    def test_run_refuses_a_block_whose_request_settings_drifted(
        self, two_tasks, tmp_path, monkeypatch
    ):
        """Defect guarded: a block registered at one per-request timeout and
        run at another, with the manifest still claiming the first."""
        monkeypatch.setattr(ab, "BENCH_DATA", tmp_path)
        monkeypatch.setattr(hb, "_git_commit", lambda: "cafebabe")
        ab.register_block("agents-react", "B9", "fsm_advance", trials=1)
        original = ab.metered_interface

        def shorter(model):
            llm = original(model)
            llm.timeout = 30.0
            return llm

        monkeypatch.setattr(ab, "metered_interface", shorter)
        with pytest.raises(ab.BenchDataError, match="llm_request"):
            ab.run_block("agents-react", "B9", "fsm_advance", trials=1)
        assert not (
            tmp_path / "agents-react" / "B9" / "rows_fsm_advance.jsonl"
        ).exists()

    def test_recorded_manifests_predate_the_disclosures(self):
        """Defect guarded: a recorded B0/B1 manifest edited to carry fields
        it never recorded (blocks are never edited; report prints them as
        "not recorded")."""
        recorded = sorted(
            (ab.BENCH_DATA / "agents-react").glob("B[01]/manifest_*.json")
        )
        assert len(recorded) == 3
        for path in recorded:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            assert not set(ab.DISCLOSURE_FIELDS) & set(manifest), path


class TestPairDisclosure:
    """`report --pair` prints every manifest difference before any number."""

    def test_manifest_differences_walks_nested_keys_and_absent_fields(self):
        """Defect guarded: a nested change (one digest inside model_digest, one
        limit) or a field one side never recorded going unprinted."""
        a = _synthetic_manifest(wrapper_version="2")
        b = _synthetic_manifest(
            wrapper_version="1",
            limits={**ab.LIMITS, "max_iterations": 10},
            model_digest={"tag": "qwen3.5:4b", "digest": "other"},
        )
        del b["llm_request"]
        lines = ab.manifest_differences(a, b)
        assert lines == [
            "limits.max_iterations: 8 vs 10",
            'llm_request: {"timeout": 120.0, "retries": 0, "extra_kwargs": []} '
            "vs not recorded",
            'model_digest.digest: "2a654d98e6fb" vs "other"',
            'wrapper_version: "2" vs "1"',
        ]
        assert ab.manifest_differences(a, dict(a)) == []

    def test_pair_prints_every_differing_field_before_the_numbers(
        self, tmp_path, monkeypatch, capsys
    ):
        """Defect guarded (review pass 7 concern 3): a pair that prints only
        Fisher lines, so a cross-meter, cross-timeout, cross-schema comparison
        reads as like for like."""
        monkeypatch.setattr(ab, "BENCH_DATA", tmp_path)
        _make_block(tmp_path / "synthetic" / "B0", "native_fc", _ROWS)
        old = tmp_path / "synthetic" / "B0" / "manifest_native_fc.json"
        recorded = json.loads(old.read_text(encoding="utf-8"))
        for field in ab.DISCLOSURE_FIELDS:
            del recorded[field]
        recorded.update(wrapper_version="1", git_commit="73d7a6c")
        old.write_text(json.dumps(recorded), encoding="utf-8")
        _make_block(
            tmp_path / "synthetic" / "B2",
            "fsm_toolcall",
            _ROWS,
            block="B2",
            tasks_sha256="9" * 64,
        )
        rc = ab.report("synthetic", ["B0", "B2"], ["B2/fsm_toolcall:B0/native_fc"])
        assert rc == 0
        out = capsys.readouterr().out
        head = out.index("Manifest differences, B2/fsm_toolcall vs B0/native_fc")
        assert head < out.index("Fisher two-sided")
        diff = out[head : out.index("Fisher two-sided")]
        for line in (
            'wrapper_version: "2" vs "1"',
            'git_commit: "deadbeef" vs "73d7a6c"',
            f'tasks_sha256: "{"9" * 64}" vs "{"1" * 64}"',
            'arm.name: "fsm_toolcall" vs "native_fc"',
            "vs not recorded",
        ):
            assert line in diff, line
        for field in ab.DISCLOSURE_FIELDS:
            assert f"  {field}: " in diff, field

    def test_differences_are_printed_even_when_the_pair_is_refused(
        self, tmp_path, monkeypatch, capsys
    ):
        """Defect guarded: a refused pair hiding which manifest field made it
        incomparable."""
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
        assert 'model_digest.digest: "other" vs "2a654d98e6fb"' in out
        assert out.index("Manifest differences") < out.index("REFUSING")

    def test_recorded_b1_b0_pair_discloses_its_differences(self, capsys):
        """Defect guarded: the recorded pair (manifests without the new
        fields) breaking the report, or its known differences (code commit,
        arm) going unprinted."""
        rc = ab.report("agents-react", ["B0", "B1"], ["B1/fsm_advance:B0/legacy"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Manifest differences, B1/fsm_advance vs B0/legacy: 6 field(s)" in out
        assert 'git_commit: "1c8572e1dda702b48944aa12006d73fb29747fff" vs ' in out
        assert "pass@1 first trial: 32/38 vs 28/38" in out


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


def _provider_reply(text: str, usage: dict | None) -> SimpleNamespace:
    """A litellm-shaped completion reply; ``usage=None`` leaves it out."""
    message = SimpleNamespace(content=text, tool_calls=None)
    reply = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    if usage is not None:
        reply.usage = usage
    return reply


def _one_call(llm_interface) -> None:
    """One plain request through the trial's injected interface."""
    from fsm_llm import CompletionRequest

    llm_interface.complete(
        CompletionRequest(messages=[{"role": "user", "content": "hi"}])
    )


class TestRunEndToEnd:
    """run_block with a mock arm: rows, manifest, summary, refusals."""

    @pytest.fixture
    def offline(self, tmp_path, monkeypatch):
        """BENCH_DATA in tmp, a stub digest/commit, a scripted provider
        behind core's one send binding."""
        import fsm_llm.llm

        monkeypatch.setattr(ab, "BENCH_DATA", tmp_path)
        monkeypatch.setattr(hb, "_model_digest", lambda tag: dict(_DIGEST))
        monkeypatch.setattr(hb, "_git_commit", lambda: "cafebabe")
        monkeypatch.setattr(hb, "_git_dirty", lambda: False)
        monkeypatch.setattr(
            fsm_llm.llm,
            "completion",
            lambda **kw: _provider_reply(
                "ok", {"prompt_tokens": 10, "completion_tokens": 2}
            ),
        )
        two = tuple(t for t in ab.TASKS if t.id in ("st-capital", "er-flaky"))
        monkeypatch.setattr(ab, "TASKS", two)

    def _arm(self):
        """A mock agent: solves with the reference, sends one request through
        the injected interface; crashes on er-flaky's first trial to exercise
        the error path."""
        seen: list[str] = []

        def factory(tools, model, llm_interface):
            def run(prompt):
                task = next(t for t in ab.TASKS if t.prompt == prompt)
                _one_call(llm_interface)
                seen.append(task.id)
                if task.id == "er-flaky" and seen.count("er-flaky") == 1:
                    raise TimeoutError("AgentTimeoutError stand-in")
                return _FakeResult(task.reference(tools), calls=len(tools))

            return SimpleNamespace(run=run)

        return factory

    def test_rows_manifest_and_summary(self, offline, tmp_path, monkeypatch):
        """Defect guarded: the live path writing rows that the offline recount
        cannot reproduce, or losing the per-row facts the plan lists."""
        monkeypatch.setitem(ab.ARMS, "mock", self._arm())
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
        assert summary["run"] == {"git_commit": "cafebabe", "git_dirty": False}
        assert summary["metrics"]["k_pass1_first_trial"] == 1
        assert summary["metrics"]["k_pass_hat_k"] == 1
        assert ab.report("agents-react") == 0

    def test_existing_rows_refuse_a_second_run(self, offline, tmp_path, monkeypatch):
        """Defect guarded (D-002): re-sampling a block until a number looks
        good. The refusal fires before the meter or any arm loads."""
        monkeypatch.setitem(ab.ARMS, "mock", self._arm())
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
        monkeypatch.setitem(ab.ARMS, "mock", self._arm())
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


def _scripted_react_provider():
    """A scripted provider for a ReactAgent run on the bench tools.

    Answers each typed field by name (one lookup, then conclude), writes the
    Pass-2 reply, varies the usage shape (object fields, a dict, absent) and
    raises for st-convert's requests, so parity covers tokens, usage-missing
    and errors. Returns ``(completion, calls)``.
    """
    calls: list[int] = []

    def completion(**kw):
        calls.append(1)
        system = kw["messages"][0]["content"]
        if "26.2 miles" in system:
            raise RuntimeError("provider down")
        seen = "Result:" in system
        if "field 'tool_name'" in system:
            value = "none" if seen else "lookup_country"
        elif "field 'tool_input'" in system:
            value = {"name": "Veloria"}
        elif "field 'should_terminate'" in system:
            value = seen
        else:
            return _provider_reply("Maskett", None)
        text = json.dumps({"value": value, "confidence": 0.9})
        n = len(calls)
        usage = (
            SimpleNamespace(
                prompt_tokens=10 + n, completion_tokens=3, total_tokens=None
            )
            if n % 2
            else {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
        )
        return _provider_reply(text, usage)

    return completion, calls


class TestMeterV2:
    """wrapper_version "2": core counters on the trial's injected interface,
    and their parity with the "1" patching meter (D-004)."""

    @pytest.fixture
    def provider(self, monkeypatch):
        """The scripted react provider behind core's one send binding."""
        import fsm_llm.llm

        completion, calls = _scripted_react_provider()
        monkeypatch.setattr(fsm_llm.llm, "completion", completion)
        monkeypatch.setattr(
            fsm_llm.llm, "get_supported_openai_params", lambda model: []
        )
        return calls

    def test_metered_interface_equals_the_one_the_agent_builds(self):
        """Defect guarded: the injected interface differing from the one the
        agent builds itself (temperature, max_tokens, timeout, kwargs), so a
        "2" block sends different requests than B0/B1 did."""
        from fsm_llm.agents.fsm_definitions import build_react_fsm

        tools = ab.make_tools()
        agent = ab.ARMS["fsm_advance"](
            {"calculator": tools["calculator"]}, ab.MODEL, None
        )
        own = agent._create_api(build_react_fsm(agent.tools)).llm_interface
        assert vars(ab.metered_interface(ab.MODEL)) == vars(own)

    def test_usage_fields_are_the_v1_row_fields(self):
        """Defect guarded: "2" rows carrying other keys than "1" rows, so
        `report` cannot recount B0/B1 and a "2" block alike."""
        from fsm_llm import LLMUsage

        usage = LLMUsage(
            calls=3,
            errors=1,
            usage_missing=1,
            prompt_tokens=20,
            completion_tokens=4,
            total_tokens=24,
        )
        fields = ab.usage_fields(usage)
        assert set(fields) == set(ab.CallMeter().snapshot())
        assert fields == {
            "llm_calls": 3,
            "llm_errors": 1,
            "usage_missing": 1,
            "prompt_tokens": 20,
            "completion_tokens": 4,
            "total_tokens": 24,
        }

    def test_each_trial_counts_only_its_own_calls(self, provider, monkeypatch):
        """Defect guarded: one interface (or one counter) shared across
        trials, so a row carries the calls of the trials before it."""
        two = tuple(t for t in ab.TASKS if t.id == "st-capital")
        monkeypatch.setattr(ab, "TASKS", two)
        first = ab._trial_row(ab.TASKS[0], 1, "fsm_advance", "gpt-4o-mini")
        after_first = len(provider)
        second = ab._trial_row(ab.TASKS[0], 2, "fsm_advance", "gpt-4o-mini")
        assert first["correct"] is True and second["correct"] is True
        assert first["llm_calls"] == after_first > 0
        assert second["llm_calls"] == len(provider) - after_first
        assert first["total_tokens"] > 0

    def test_parity_with_the_patching_meter_on_a_scripted_react_run(self, provider):
        """Defect guarded: the "2" meter counting a different number of calls,
        tokens, usage-missing replies or errors than the "1" meter B0/B1 were
        counted with, so a "2" block is not comparable with them (D-004)."""
        results = ab.meter_parity(
            ["st-capital", "st-convert"], "fsm_advance", "gpt-4o-mini"
        )
        by_id = {r["task_id"]: r for r in results}
        assert [r["task_id"] for r in results] == ["st-capital", "st-convert"]
        assert all(r["equal"] for r in results), results
        ran = by_id["st-capital"]
        assert ran["error"] is None
        assert ran["v2"]["llm_calls"] >= 4
        assert ran["v2"]["usage_missing"] >= 1
        assert ran["v2"]["prompt_tokens"] > 0
        failed = by_id["st-convert"]
        assert failed["v2"]["llm_errors"] >= 1
        assert sum(r["v1"]["llm_calls"] for r in results) == len(provider)

    def test_parity_flags_an_arm_that_bypasses_the_injected_interface(
        self, provider, monkeypatch
    ):
        """Defect guarded (anti-vacuity): a parity check that passes whatever
        the arm does. An arm that lets the agent build its own interface
        sends real calls the "2" meter never sees."""

        def bypass(tools, model, llm_interface):
            return ab._fsm_advance_arm(tools, model, None)

        monkeypatch.setitem(ab.ARMS, "bypass", bypass)
        (result,) = ab.meter_parity(["st-capital"], "bypass", "gpt-4o-mini")
        assert result["equal"] is False
        assert result["v2"]["llm_calls"] == 0
        assert result["v1"]["llm_calls"] == len(provider) > 0

    def test_parity_refuses_unknown_arm_or_task_before_running(self, provider):
        """Defect guarded: a typo'd smoke running nothing and reporting an
        empty, vacuous parity."""
        with pytest.raises(ab.BenchDataError, match="unknown arm"):
            ab.meter_parity(["st-capital"], "fsm_advanc")
        with pytest.raises(ab.BenchDataError, match="unknown task"):
            ab.meter_parity(["st-capital", "st-nope"], "fsm_advance")
        assert provider == []

    def test_recorded_b0_b1_blocks_recount_unchanged(self, capsys):
        """Defect guarded: the meter switch altering how recorded "1" rows
        recount (B0 legacy 28/38 and native_fc 37/38, B1 fsm_advance 32/38
        at 10.97 calls), or a summary no longer matching its rows."""
        assert ab.report("agents-react", blocks=["B0", "B1"]) == 0
        out = capsys.readouterr().out
        assert "MISMATCH" not in out
        sections = out.split("agents-react ")
        legacy = next(s for s in sections if s.startswith("B0 [legacy]"))
        native = next(s for s in sections if s.startswith("B0 [native_fc]"))
        advance = next(s for s in sections if s.startswith("B1 [fsm_advance]"))
        assert "pass@1 first trial: 28/38" in legacy
        assert "pass@1 first trial: 37/38" in native
        assert "llm calls mean/median: 2.3947/" in native
        assert "pass@1 first trial: 32/38" in advance
        assert "llm calls mean/median: 10.9737/" in advance


class TestArms:
    """The registered arms build their agents without an LLM call."""

    @pytest.mark.parametrize(("arm", "cls"), [("fsm_advance", "ReactAgent")])
    def test_arm_builds_its_agent_class(self, arm, cls):
        """Defect guarded: an arm factory broken by a signature change,
        found only after the live block has started."""
        tools = ab.make_tools()
        llm_interface = ab.metered_interface(ab.MODEL)
        agent = ab.ARMS[arm](
            {"calculator": tools["calculator"]}, ab.MODEL, llm_interface
        )
        assert type(agent).__name__ == cls
        assert agent.config.max_iterations == ab.LIMITS["max_iterations"]
        assert agent.config.timeout_seconds == ab.LIMITS["timeout_seconds"]
        assert agent.config.temperature == ab.LIMITS["temperature"]
        assert agent.config.max_tokens == ab.LIMITS["max_tokens"]
        assert agent._api_kwargs["llm_interface"] is llm_interface

    def test_new_blocks_keep_b0_limits_and_switch_the_meter(self):
        """Defect guarded: a new block registered with limits or a task set
        that differ from B0's committed manifest, so a pair with B0/B1
        compares two things; or the meter switch left unrecorded, so a "2"
        block's manifest claims B0's patching meter (D-004)."""
        block_dir = ab.BENCH_DATA / "agents-react"
        b0 = json.loads(
            (block_dir / "B0" / "manifest_legacy.json").read_text(encoding="utf-8")
        )
        assert ab.tasks_sha256() == b0["tasks_sha256"]
        assert ab.LIMITS == b0["limits"]
        assert ab.TRIALS == b0["trials"]
        assert ab.MODEL == b0["model"]
        recorded = sorted(block_dir.glob("B[01]/manifest_*.json"))
        assert len(recorded) == 3
        for path in recorded:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            assert manifest["wrapper_version"] == "1", path
        assert ab.WRAPPER_VERSION == "2"

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


class TestWireDisclosure:
    """Review pass 10 warnings 1 and 2, notes 6 and 7 (D-037 item 18.5, D-040):
    the disclosure is read off the FINAL request at core's send binding, is
    stable across days, and is pinned at ``run``."""

    @pytest.fixture
    def native_two(self, monkeypatch):
        """Two tasks, a stub digest/commit, the native arm registered, and a
        provider that fails the test if anything is sent."""
        import fsm_llm.llm

        def no_send(**kw):
            raise AssertionError("the disclosure probe sent a provider request")

        monkeypatch.setattr(fsm_llm.llm, "completion", no_send)
        monkeypatch.setattr(hb, "_model_digest", lambda tag: dict(_DIGEST))
        monkeypatch.setattr(hb, "_git_commit", lambda: "cafebabe")
        monkeypatch.setitem(ab.ARMS, "native_probe", _native_arm)
        two = tuple(t for t in ab.TASKS if t.id in _TWO_IDS)
        monkeypatch.setattr(ab, "TASKS", two)

    def test_a_tool_turn_temperature_rule_change_moves_the_disclosure(
        self, native_two, monkeypatch
    ):
        """Defect guarded: forcing temperature 0 on a tool turn (the
        bf7ffe24/D-003 rule, measured 0/3 vs 3/3 tool calls) moved no
        manifest field, because the digest stopped at the interface."""
        from fsm_llm import LiteLLMInterface

        before = ab.request_disclosure("native_probe", ab.MODEL)
        assert before["llm_request"]["temperature"] == ab.LIMITS["temperature"]
        original = LiteLLMInterface._apply_model_specific_params

        def greedy_tools(self, call_params, call_type, **kw):
            original(self, call_params, call_type, **kw)
            call_params["temperature"] = 0

        monkeypatch.setattr(
            LiteLLMInterface, "_apply_model_specific_params", greedy_tools
        )
        after = ab.request_disclosure("native_probe", ab.MODEL)
        assert after["llm_request"]["temperature"] == 0
        assert hb.manifest_differences(before, after)
        for task_id in _TWO_IDS:
            assert (
                before["first_request"][task_id]["settings_sha256"]
                != after["first_request"][task_id]["settings_sha256"]
            )

    def test_dropping_the_ollama_user_turn_prefix_moves_the_disclosure(
        self, native_two, monkeypatch
    ):
        """Defect guarded: core's Ollama message preparation (``/nothink`` on
        the last user turn) going unrecorded."""
        import fsm_llm.llm

        before = ab.request_disclosure("native_probe", ab.MODEL)
        monkeypatch.setattr(
            fsm_llm.llm, "prepare_ollama_messages", lambda messages, *a: messages
        )
        after = ab.request_disclosure("native_probe", ab.MODEL)
        for task_id in _TWO_IDS:
            assert (
                before["first_request"][task_id]["user_sha256"]
                != after["first_request"][task_id]["user_sha256"]
            )

    def test_extra_settings_are_recorded_with_values_and_secrets_redacted(
        self, native_two, monkeypatch
    ):
        """Defect guarded: a seed 7 vs seed 8 block (or an api_base change)
        looking identical because only kwarg NAMES were recorded."""
        from fsm_llm import LiteLLMInterface

        def seeded(seed):
            def interface(model):
                return LiteLLMInterface(
                    model=model,
                    temperature=ab.LIMITS["temperature"],
                    max_tokens=ab.LIMITS["max_tokens"],
                    seed=seed,
                    api_key="sk-live-abcdef0123456789abcdef",
                )

            return interface

        monkeypatch.setattr(ab, "metered_interface", seeded(7))
        seven = ab.request_disclosure("native_probe", ab.MODEL)
        monkeypatch.setattr(ab, "metered_interface", seeded(8))
        eight = ab.request_disclosure("native_probe", ab.MODEL)
        assert seven["llm_request"]["seed"] == 7
        assert seven["llm_request"]["api_key"] == "<redacted>"
        assert "sk-live" not in json.dumps(seven)
        assert "llm_request.seed: 7 vs 8" in hb.manifest_differences(seven, eight)

    def test_digests_do_not_move_with_the_date(self, native_two, monkeypatch):
        """Defect guarded: a prompt-mode system prompt carries today's date,
        so its digest moved overnight (or with TZ) and could not be pinned
        at ``run``."""
        import datetime as real_datetime

        def on(day):
            class _Day(real_datetime.date):
                @classmethod
                def today(cls):
                    return cls(2026, 10, day)

            monkeypatch.setattr(real_datetime, "date", _Day)
            return ab.request_disclosure("fsm_advance", ab.MODEL)["first_request"]

        assert on(1) == on(2)

    def test_run_refuses_a_block_whose_native_system_prompt_changed(
        self, native_two, tmp_path, monkeypatch
    ):
        """Defect guarded (pass 10 warning 2): ``run`` accepted a native block
        whose system prompt changed after registration (first_request was not
        pinned)."""
        from fsm_llm.agents import native_fc

        monkeypatch.setattr(ab, "BENCH_DATA", tmp_path)
        ab.register_block("agents-react", "B9", "native_probe", trials=1)
        monkeypatch.setattr(
            native_fc, "_SYSTEM_PROMPT", "You are a terse agent. Never call tools."
        )
        with pytest.raises(ab.BenchDataError, match="first_request"):
            ab.run_block("agents-react", "B9", "native_probe", trials=1)
        assert not (
            tmp_path / "agents-react" / "B9" / "rows_native_probe.jsonl"
        ).exists()

    def test_field_comparison_distinguishes_types(self):
        """Defect guarded (pass 10 note 6): Python ``==`` hid a ``retries: 0``
        vs ``false`` or a ``timeout: 120`` vs ``120.0`` change."""
        lines = ab.manifest_differences(
            {"a": 1, "b": {"t": 120}}, {"a": True, "b": {"t": 120.0}}
        )
        assert lines == ["a: 1 vs true", "b.t: 120 vs 120.0"]
        assert ab.manifest_differences is hb.manifest_differences
