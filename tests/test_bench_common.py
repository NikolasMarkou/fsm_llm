"""Offline sanity tests for ``scripts/bench_common.py``.

Pure-offline and fast on purpose: no live gate, no slow marker, no LLM. Each
test's docstring names the defect it guards against (repo convention), most of
them defects the predecessor plan actually recorded (plans/LESSONS.md [I:4],
decisions.md D-002 of plan-2026-07-22T114536-879d04a0).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bench_common as bc

#: Top-level imports the bench module may hold. Everything heavier is lazy,
#: which is what keeps a plain import socket-free.
_STDLIB_ALLOWED = {
    "__future__",
    "collections",
    "contextlib",
    "copy",
    "datetime",
    "hashlib",
    "json",
    "math",
    "pathlib",
    "re",
    "subprocess",
    "typing",
}


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


class TestImportIsInert:
    """Importing the shared bench helpers must never open a socket."""

    def test_plain_import_opens_no_socket(self):
        """Defect guarded: a module-scope import of litellm (or anything that
        probes the network) would run in every offline bench command. The
        subprocess disables sockets outright and the import must still
        succeed."""
        code = (
            "import socket\n"
            "def boom(*a, **k):\n"
            "    raise AssertionError('socket opened at import time')\n"
            "socket.socket = boom\n"
            "socket.create_connection = boom\n"
            "import sys\n"
            f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
            "import bench_common\n"
            "print('IMPORT-OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
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
        tree = ast.parse(Path(bc.__file__).read_text(encoding="utf-8"))
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        assert names <= _STDLIB_ALLOWED, names - _STDLIB_ALLOWED

    def test_no_direct_litellm_reference_anywhere_in_the_module(self):
        """Defect guarded: a second provider path beside core's LLM layer
        (user direction: no litellm import outside it), lazy or not."""
        tree = ast.parse(Path(bc.__file__).read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Name):
                names = [node.id]
            else:
                continue
            offenders += [n for n in names if n.split(".")[0] == "litellm"]
        assert offenders == []


class TestWilsonCI:
    """The CI every summary carries; wrong bounds misreport every block."""

    def test_known_value_33_of_40(self):
        """Defect guarded: a mis-implemented denominator quietly narrows the
        interval; 33/40 has a hand-checkable reference value."""
        lo, hi = bc.wilson_ci(33, 40)
        assert lo == pytest.approx(0.6805, abs=1e-3)
        assert hi == pytest.approx(0.9125, abs=1e-3)

    def test_bounds_stay_inside_unit_interval_at_the_edges(self):
        """Defect guarded: the Wald interval (the naive choice) escapes [0,1]
        at k=0 and k=n; Wilson must not."""
        lo0, hi0 = bc.wilson_ci(0, 40)
        lon, hin = bc.wilson_ci(40, 40)
        assert lo0 == 0.0 and hi0 < 0.15
        assert lon > 0.85 and hin == 1.0

    def test_n_zero_returns_vacuous_interval(self):
        """Defect guarded: ZeroDivisionError on an empty (aborted) block."""
        assert bc.wilson_ci(0, 0) == (0.0, 1.0)

    def test_impossible_counts_raise(self):
        """Defect guarded: k>n silently producing a 'probability' above 1."""
        with pytest.raises(ValueError):
            bc.wilson_ci(5, 4)
        with pytest.raises(ValueError):
            bc.wilson_ci(-1, 4)


class TestFisherExact:
    """The pre-registered B0-vs-B1 decision rule runs through this."""

    def test_known_value_2_5_vs_0_5(self):
        """Defect guarded: one-sided-by-accident. Reference: 2/5 vs 0/5 is
        p=20/45~=0.444 two-sided (the predecessor's D-049 quoted value)."""
        assert bc.fisher_exact_two_sided(2, 5, 0, 5) == pytest.approx(0.4444, abs=1e-3)

    def test_known_value_5_10_vs_0_10(self):
        """Defect guarded: hypergeometric support enumerated wrongly.
        Reference: 5/10 vs 0/10 is p=504/15504~=0.0325 two-sided."""
        assert bc.fisher_exact_two_sided(5, 10, 0, 10) == pytest.approx(
            0.0325, abs=1e-3
        )

    def test_identical_arms_give_p_one(self):
        """Defect guarded: a p<1 on identical arms would fabricate an effect
        out of nothing."""
        assert bc.fisher_exact_two_sided(20, 40, 20, 40) == pytest.approx(1.0)

    def test_symmetry_in_arm_order(self):
        """Defect guarded: swapping B0/B1 changing the verdict."""
        assert bc.fisher_exact_two_sided(32, 40, 20, 40) == pytest.approx(
            bc.fisher_exact_two_sided(20, 40, 32, 40)
        )

    def test_empty_arm_raises(self):
        """Defect guarded: n=0 silently 'comparing' against nothing."""
        with pytest.raises(ValueError):
            bc.fisher_exact_two_sided(1, 2, 0, 0)


class TestManifestFields:
    """The comparability fields every manifest carries."""

    def test_manifest_fields_constant_names_exactly_six(self):
        """Defect guarded: the required-field list drifting from the 6 the
        findings pinned (findings/bench-measurement-architecture.md)."""
        assert len(bc.MANIFEST_FIELDS) == 6


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
            bc.append_row(path, row)
        assert bc.read_rows(path) == rows

    def test_reading_a_missing_rows_file_yields_empty(self, tmp_path: Path):
        """Defect guarded: an aborted-before-first-row block crashing the
        summary writer instead of recording n=0."""
        assert bc.read_rows(tmp_path / "absent.jsonl") == []


class TestWireDisclosure:
    """Review pass 10 warning 3 (D-037 item 18.5, D-040): an L4 manifest
    discloses what its first request carried at core's send binding."""

    @pytest.fixture
    def no_send(self, monkeypatch):
        import fsm_llm.llm

        def no_send(**kw):
            raise AssertionError("the disclosure probe sent a provider request")

        monkeypatch.setattr(fsm_llm.llm, "completion", no_send)

    def test_the_binding_is_restored_when_the_block_raises(self, no_send):
        import fsm_llm.llm

        with pytest.raises(RuntimeError), bc.captured_wire():
            raise RuntimeError("boom")
        assert fsm_llm.llm.completion.__name__ == "no_send"

    def test_the_digest_masks_the_date_and_the_task(self):
        def params(day: str, task: str) -> dict:
            return {
                "model": "m",
                "messages": [
                    {"role": "system", "content": f"Today's date: {day}\nRules"},
                    {"role": "user", "content": f"/nothink\n{task}"},
                ],
            }

        a = bc.wire_digest(params("2026-10-01", "task one"), "task one")
        b = bc.wire_digest(params("2027-01-31", "task two"), "task two")
        assert a == b
        c = bc.wire_digest(
            {
                **params("2026-10-01", "x"),
                "messages": [{"role": "user", "content": "x"}],
            },
            "x",
        )
        assert c["user_sha256"] != a["user_sha256"]


class TestManifestDifferences:
    """``manifest_differences`` compares leaves as JSON text."""

    def test_field_comparison_distinguishes_types(self):
        assert bc.manifest_differences(
            {"t": 120, "r": 0}, {"t": 120.0, "r": False}
        ) == [
            "r: 0 vs false",
            "t: 120 vs 120.0",
        ]


class TestPinnedDrift:
    """``pinned_drift`` is the one run-time drift check (D-049)."""

    def test_pinned_drift_ignores_key_order_but_not_types(self):
        """Defect guarded (review pass 12 note 3): a manifest read back from
        its sort_keys file drifting from the in-memory one on key order
        inside a list, so ``run`` refuses a valid block; a type change must
        still be drift."""
        current = {
            "llm_request": {"anyOf": [{"type": "string", "description": "x"}]},
            "model_digest": {"digest": "d"},
        }
        recorded = json.loads(json.dumps(current, sort_keys=True))
        assert bc.pinned_drift(recorded, current, ("llm_request",)) == []
        typed = {**current, "llm_request": {"anyOf": 1.0}}
        stored = {**recorded, "llm_request": {"anyOf": 1}}
        assert bc.pinned_drift(stored, typed, ("llm_request",)) == ["llm_request"]
        moved = {**current, "model_digest": {"digest": "e"}}
        assert bc.pinned_drift(recorded, moved, ()) == ["model_digest"]
