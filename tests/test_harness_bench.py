"""Offline sanity tests for ``scripts/harness_bench.py``.

Pure-offline and fast on purpose: no live gate, no slow marker, no LLM. Each
test's docstring names the defect it guards against (repo convention), most of
them defects the predecessor plan actually recorded (plans/LESSONS.md [I:4],
decisions.md D-001/D-002 of plan-2026-07-22T114536-879d04a0).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import harness_bench as hb

#: Top-level imports the bench module may hold. Everything heavier is lazy,
#: which is what keeps a plain import socket-free (assumption A6).
_STDLIB_ALLOWED = {
    "__future__",
    "argparse",
    "datetime",
    "hashlib",
    "json",
    "math",
    "pathlib",
    "subprocess",
    "sys",
    "tempfile",
    "time",
    "typing",
}


def _synthetic_manifest(**overrides) -> dict:
    """A manifest carrying all 6 required fields, no live query needed."""
    manifest = {
        "bench_id": "synthetic",
        "block": "B0",
        "n_preregistered": 5,
        "seed": None,
        "model": "ollama_chat/qwen3.5:4b",
        "created_at": "2026-07-22T00:00:00Z",
        "prompt_bytes_sha256": "0" * 64,
        "tool_surface": {"native_function_calling": True},
        "fixture_hash": "1" * 64,
        "model_digest": {"tag": "qwen3.5:4b", "digest": "2a654d98e6fb"},
        "arm": {"native": True, "display": "native"},
        "git_commit": "deadbeef",
    }
    manifest.update(overrides)
    return manifest


def _synthetic_rows(flags: list[tuple[bool, bool, bool, bool]]) -> list[dict]:
    """Rows whose four k-metrics are set from *flags* tuples."""
    rows = []
    for i, (wrote, bytes_, matched, success) in enumerate(flags, start=1):
        rows.append(
            {
                "bench_id": "synthetic",
                "block": "B0",
                "arm": "native_fc (package default)",
                "native": True,
                "run": i,
                "ts": "2026-07-22T00:00:00Z",
                "elapsed_s": 1.0,
                "tool_calls": 3,
                "write_tool_issued": wrote,
                "bytes_on_disk": bytes_,
                "content_matched": matched,
                "success": success,
                "tool_trace": [{"tool": "write_file", "ok": True}],
                "seed": None,
            }
        )
    return rows


def _make_block(
    bdir: Path, arm: str, flags: list[tuple[bool, bool, bool, bool]], **manifest
) -> None:
    """A complete synthetic block arm on disk: manifest + rows."""
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / f"manifest_{arm}.json").write_text(
        json.dumps(_synthetic_manifest(**manifest)), encoding="utf-8"
    )
    for row in _synthetic_rows(flags):
        hb.append_row(bdir / f"rows_{arm}.jsonl", row)


class TestImportIsInert:
    """Guards assumption A6: importing the bench must never open a socket."""

    def test_plain_import_opens_no_socket_even_with_live_env_armed(self):
        """Defect guarded: test_live_ollama's skipif gate probes the Ollama
        socket AT IMPORT TIME whenever FSM_LLM_HARNESS_LIVE=1 is exported; a
        bench that imported it at module scope would inherit that probe. The
        subprocess disables sockets outright, arms the env var, and the import
        must still succeed."""
        code = (
            "import socket\n"
            "def boom(*a, **k):\n"
            "    raise AssertionError('socket opened at import time')\n"
            "socket.socket = boom\n"
            "socket.create_connection = boom\n"
            "import sys\n"
            f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
            "import harness_bench\n"
            "print('IMPORT-OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "FSM_LLM_HARNESS_LIVE": "1"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "IMPORT-OK" in proc.stdout

    def test_top_level_imports_are_stdlib_only(self):
        """Defect guarded: someone adds `import litellm` (or the test module)
        at module scope, silently re-opening the import-time socket path the
        lazy-import design closed."""
        tree = ast.parse(Path(hb.__file__).read_text(encoding="utf-8"))
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        assert names <= _STDLIB_ALLOWED, names - _STDLIB_ALLOWED


class TestWilsonCI:
    """The CI every summary carries; wrong bounds misreport every block."""

    def test_known_value_33_of_40(self):
        """Defect guarded: a mis-implemented denominator quietly narrows the
        interval; 33/40 has a hand-checkable reference value."""
        lo, hi = hb.wilson_ci(33, 40)
        assert lo == pytest.approx(0.6805, abs=1e-3)
        assert hi == pytest.approx(0.9125, abs=1e-3)

    def test_bounds_stay_inside_unit_interval_at_the_edges(self):
        """Defect guarded: the Wald interval (the naive choice) escapes [0,1]
        at k=0 and k=n; Wilson must not."""
        lo0, hi0 = hb.wilson_ci(0, 40)
        lon, hin = hb.wilson_ci(40, 40)
        assert lo0 == 0.0 and hi0 < 0.15
        assert lon > 0.85 and hin == 1.0

    def test_n_zero_returns_vacuous_interval(self):
        """Defect guarded: ZeroDivisionError on an empty (aborted) block."""
        assert hb.wilson_ci(0, 0) == (0.0, 1.0)

    def test_impossible_counts_raise(self):
        """Defect guarded: k>n silently producing a 'probability' above 1."""
        with pytest.raises(ValueError):
            hb.wilson_ci(5, 4)
        with pytest.raises(ValueError):
            hb.wilson_ci(-1, 4)


class TestFisherExact:
    """The pre-registered B0-vs-B1 decision rule runs through this."""

    def test_known_value_2_5_vs_0_5(self):
        """Defect guarded: one-sided-by-accident. Reference: 2/5 vs 0/5 is
        p=20/45~=0.444 two-sided (the predecessor's D-049 quoted value)."""
        assert hb.fisher_exact_two_sided(2, 5, 0, 5) == pytest.approx(0.4444, abs=1e-3)

    def test_known_value_5_10_vs_0_10(self):
        """Defect guarded: hypergeometric support enumerated wrongly.
        Reference: 5/10 vs 0/10 is p=504/15504~=0.0325 two-sided."""
        assert hb.fisher_exact_two_sided(5, 10, 0, 10) == pytest.approx(
            0.0325, abs=1e-3
        )

    def test_identical_arms_give_p_one(self):
        """Defect guarded: a p<1 on identical arms would fabricate an effect
        out of nothing."""
        assert hb.fisher_exact_two_sided(20, 40, 20, 40) == pytest.approx(1.0)

    def test_symmetry_in_arm_order(self):
        """Defect guarded: swapping B0/B1 changing the verdict."""
        assert hb.fisher_exact_two_sided(32, 40, 20, 40) == pytest.approx(
            hb.fisher_exact_two_sided(20, 40, 32, 40)
        )

    def test_empty_arm_raises(self):
        """Defect guarded: n=0 silently 'comparing' against nothing."""
        with pytest.raises(ValueError):
            hb.fisher_exact_two_sided(1, 2, 0, 0)


class TestManifestGate:
    """An unmanifested block is not evidence -- the writer must refuse."""

    def test_summary_refused_without_manifest(self, tmp_path: Path):
        """Defect guarded: plan.md Failure Modes, git row -- a block committed
        without its manifest silently passing as evidence."""
        for row in _synthetic_rows([(True, True, True, True)]):
            hb.append_row(tmp_path / "rows_native.jsonl", row)
        with pytest.raises(hb.BenchDataError, match="manifest"):
            hb.write_summary(tmp_path, "native", status="complete")
        assert not (tmp_path / "summary_native.json").exists()

    def test_summary_refused_when_manifest_lacks_a_required_field(self, tmp_path: Path):
        """Defect guarded: a 5-field manifest (say, no model_digest) making
        two incomparable blocks look comparable."""
        manifest = _synthetic_manifest()
        del manifest["model_digest"]
        (tmp_path / "manifest_native.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with pytest.raises(hb.BenchDataError, match="model_digest"):
            hb.write_summary(tmp_path, "native", status="complete")

    def test_manifest_fields_constant_names_exactly_six(self):
        """Defect guarded: the required-field list drifting from the 6 the
        findings pinned (findings/bench-measurement-architecture.md)."""
        assert len(hb.MANIFEST_FIELDS) == 6


class TestJsonlRoundTrip:
    """Rows are the raw evidence; a lossy writer corrupts every recount."""

    def test_rows_survive_append_and_read_back_unchanged(self, tmp_path: Path):
        """Defect guarded: any serialization loss between what a dispatch
        measured and what `report` recounts later."""
        rows = _synthetic_rows(
            [(True, True, False, True), (False, False, False, False)]
        )
        path = tmp_path / "rows_native.jsonl"
        for row in rows:
            hb.append_row(path, row)
        assert hb.read_rows(path) == rows

    def test_reading_a_missing_rows_file_yields_empty(self, tmp_path: Path):
        """Defect guarded: an aborted-before-first-row block crashing the
        summary writer instead of recording n=0."""
        assert hb.read_rows(tmp_path / "absent.jsonl") == []


class TestReportRecompute:
    """Verification check 3: the report must reproduce summaries from RAW rows."""

    FLAGS_B0 = [(True, True, True, True)] * 3 + [(True, True, False, True)] * 2
    FLAGS_B1 = [(True, True, True, True)] * 5

    def test_summary_written_by_run_path_matches_report_recount(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        """Defect guarded: a summary k that drifts from its own rows (the
        predecessor's numbers could not be recomputed at all)."""
        monkeypatch.setattr(hb, "BENCH_DATA", tmp_path)
        bdir = tmp_path / "synthetic" / "B0"
        _make_block(bdir, "native", self.FLAGS_B0)
        summary = hb.write_summary(bdir, "native", status="complete")
        assert summary["n"] == 5
        assert summary["k_content_matched"] == 3
        assert summary["k_write_tool_issued"] == 5
        assert hb.report("synthetic") == 0
        out = capsys.readouterr().out
        assert "content_matched: 3/5" in out
        assert "MISMATCH" not in out

    def test_tampered_summary_is_flagged_as_mismatch(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        """Defect guarded: a hand-edited summary going unnoticed because
        nothing ever recounts the raw rows."""
        monkeypatch.setattr(hb, "BENCH_DATA", tmp_path)
        bdir = tmp_path / "synthetic" / "B0"
        _make_block(bdir, "native", self.FLAGS_B0)
        summary = hb.write_summary(bdir, "native", status="complete")
        summary["k_content_matched"] = 5  # the lie
        (bdir / "summary_native.json").write_text(json.dumps(summary), encoding="utf-8")
        assert hb.report("synthetic") == 1
        assert "MISMATCH" in capsys.readouterr().out

    def test_a_frozen_block_without_the_ast_metric_reports_clean(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        """Defect guarded: adding ``content_matched_ast`` to K_METRICS turning
        ``report l4-execute-write`` red over FROZEN blocks -- their committed
        summaries lack ``k_content_matched_ast`` and their rows never measured
        it, so the report must skip the absent metric (no fabricated 0/n line,
        no MISMATCH) rather than re-score frozen evidence (D-006)."""
        monkeypatch.setattr(hb, "BENCH_DATA", tmp_path)
        bdir = tmp_path / "synthetic" / "B0"
        _make_block(bdir, "native", self.FLAGS_B0)
        summary = hb.write_summary(bdir, "native", status="complete")
        for key in ("k_content_matched_ast", "wilson_content_matched_ast"):
            del summary[key]
        (bdir / "summary_native.json").write_text(json.dumps(summary), encoding="utf-8")
        assert hb.report("synthetic") == 0
        out = capsys.readouterr().out
        assert "content_matched_ast" not in out
        assert "MISMATCH" not in out

    def test_two_blocks_get_a_fisher_p_matching_a_direct_call(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        """Defect guarded: the printed comparison silently using different
        arithmetic than the pre-registered fisher_exact_two_sided rule."""
        monkeypatch.setattr(hb, "BENCH_DATA", tmp_path)
        _make_block(tmp_path / "synthetic" / "B0", "native", self.FLAGS_B0)
        _make_block(tmp_path / "synthetic" / "B1", "native", self.FLAGS_B1)
        assert hb.report("synthetic") == 0
        out = capsys.readouterr().out
        expected = hb.fisher_exact_two_sided(3, 5, 5, 5)
        assert f"content_matched: 3/5 vs 5/5 p={expected:.4f}" in out

    def test_cross_digest_comparison_is_refused(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        """Defect guarded: plan.md Failure Modes -- a silently re-pulled model
        digest making B0 and B1 incomparable while the report compares away."""
        monkeypatch.setattr(hb, "BENCH_DATA", tmp_path)
        _make_block(tmp_path / "synthetic" / "B0", "native", self.FLAGS_B0)
        _make_block(
            tmp_path / "synthetic" / "B1",
            "native",
            self.FLAGS_B1,
            model_digest={"tag": "qwen3.5:4b", "digest": "someothersha"},
        )
        assert hb.report("synthetic") == 1
        out = capsys.readouterr().out
        assert "REFUSING" in out
        assert "p=" not in out

    def test_missing_bench_raises(self, tmp_path: Path, monkeypatch):
        """Defect guarded: reporting an empty path as a clean zero-block run."""
        monkeypatch.setattr(hb, "BENCH_DATA", tmp_path)
        with pytest.raises(hb.BenchDataError, match="no such bench"):
            hb.report("never-registered")


class TestRunRefusesResample:
    """D-002: a block is run ONCE; re-sampling is the defect this plan fixes."""

    def test_existing_rows_refuse_a_second_run_before_any_socket(
        self, tmp_path: Path, monkeypatch
    ):
        """Defect guarded: the predecessor re-sampled n=5 blocks until a
        number looked stable. The refusal must fire BEFORE the live machinery
        import, so this test needs no daemon and no env gate."""
        monkeypatch.setattr(hb, "BENCH_DATA", tmp_path)
        bdir = tmp_path / "l4-execute-write" / "B0"
        bdir.mkdir(parents=True)
        (bdir / "rows_native.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(hb.BenchDataError, match="run ONCE"):
            hb.run_block("l4-execute-write", "B0", "native", 40)

    def test_unknown_arm_is_refused(self, tmp_path: Path, monkeypatch):
        """Defect guarded: a typo'd --arm minting a third, unregistered arm."""
        monkeypatch.setattr(hb, "BENCH_DATA", tmp_path)
        with pytest.raises(hb.BenchDataError, match="unknown arm"):
            hb.run_block("x", "B0", "nativ", 40)


class TestMachineryImport:
    """D-001: the scripts->tests import direction actually works offline."""

    def test_fixture_hash_is_deterministic_and_hex(self, monkeypatch):
        """Defect guarded: the D-001 import direction breaking silently (a
        renamed test helper) or the fixture hash wobbling between calls,
        which would un-pin every manifest."""
        monkeypatch.delenv("FSM_LLM_HARNESS_LIVE", raising=False)
        live = hb._live()
        h1 = hb._fixture_hash(live)
        assert h1 == hb._fixture_hash(live)
        assert len(h1) == 64 and int(h1, 16) >= 0

    def test_prompt_hash_is_deterministic_and_hex(self, monkeypatch):
        """Defect guarded: placeholder paths leaking per-run tmpdirs into the
        prompt hash, making every block 'differ' in prompt bytes."""
        monkeypatch.delenv("FSM_LLM_HARNESS_LIVE", raising=False)
        live = hb._live()
        h1 = hb._prompt_hash(live)
        assert h1 == hb._prompt_hash(live)
        assert len(h1) == 64 and int(h1, 16) >= 0

    def test_tool_surface_names_a_write_tool_and_the_arm_flag(self, monkeypatch):
        """Defect guarded: a manifest that cannot show the dispatch even HELD
        a write tool -- the exact ambiguity manifests exist to remove."""
        monkeypatch.delenv("FSM_LLM_HARNESS_LIVE", raising=False)
        live = hb._live()
        surface = hb._tool_surface(live, native=True)
        assert surface["native_function_calling"] is True
        assert "write_file" in surface["declared_tools"]
        assert surface["retry_attempts"] == 1
        assert surface["timeout_seconds"] == 600

    def test_build_manifest_carries_all_six_fields(self, monkeypatch):
        """Defect guarded: a manifest builder that drifts from the 6-field
        contract its own summary writer enforces."""
        monkeypatch.delenv("FSM_LLM_HARNESS_LIVE", raising=False)
        monkeypatch.setattr(
            hb,
            "_model_digest",
            lambda tag=hb.MODEL_TAG: {"tag": tag, "digest": "stubbed"},
        )
        live = hb._live()
        manifest = hb.build_manifest(
            live, bench_id="x", block="B0", arm_name="react", n=40, seed=None
        )
        assert all(field in manifest for field in hb.MANIFEST_FIELDS)
        assert manifest["arm"] == {"native": False, "display": "react"}


class TestCLI:
    """The argparse surface the five pre-declared live events will drive."""

    @pytest.mark.parametrize(
        "argv",
        [["--help"], ["probe-seed", "--help"], ["run", "--help"], ["report", "--help"]],
    )
    def test_help_exits_zero(self, argv, capsys):
        """Defect guarded: a --help that crashes is a CLI nobody can operate
        mid-block; argparse must exit 0 on every help path."""
        with pytest.raises(SystemExit) as excinfo:
            hb.build_parser().parse_args(argv)
        assert excinfo.value.code == 0
        capsys.readouterr()  # swallow the help text

    def test_run_requires_a_registered_arm_choice(self, capsys):
        """Defect guarded: an unconstrained --arm string minting arms the
        ARMS table does not know."""
        with pytest.raises(SystemExit) as excinfo:
            hb.build_parser().parse_args(
                ["run", "--bench-id", "x", "--block", "B0", "--arm", "gpt"]
            )
        assert excinfo.value.code != 0
        capsys.readouterr()

    def test_main_maps_bench_errors_to_exit_one(self, tmp_path: Path, monkeypatch):
        """Defect guarded: a BenchDataError escaping as a traceback, which a
        shell wrapper cannot distinguish from a crash."""
        monkeypatch.setattr(hb, "BENCH_DATA", tmp_path)
        assert hb.main(["report", "never-registered"]) == 1
