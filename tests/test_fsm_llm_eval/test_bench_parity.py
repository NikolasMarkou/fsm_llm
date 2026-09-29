"""Parity between ``scripts/harness_bench.py`` and its ``fsm_llm.eval`` twins.

The bench script keeps stdlib copies of seven helpers so it stays offline
(D-008 of plan 581c2634: importing ``fsm_llm`` pulls litellm, which opens a
socket). These tests are what keeps the two copies from drifting.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fsm_llm.eval import records, stats

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "scripts" / "harness_bench.py"

_FISHER_TABLES = [
    (0, 1, 0, 1),
    (0, 5, 5, 5),
    (5, 5, 0, 5),
    (1, 5, 3, 5),
    (3, 10, 7, 10),
    (2, 40, 40, 40),
    (4, 5, 5, 5),
    (10, 30, 20, 30),
    (0, 3, 1, 7),
    (13, 17, 2, 9),
]


def _load_bench():
    spec = importlib.util.spec_from_file_location("_harness_bench_parity", BENCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hb():
    return _load_bench()


class TestStatsParity:
    @pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 10, 30, 40, 100])
    def test_wilson_equal_over_every_k(self, hb, n):
        """Defect guarded: an edit to one Wilson copy but not the other."""
        for k in range(n + 1):
            assert hb.wilson_ci(k, n) == stats.wilson_ci(k, n)
            assert hb.wilson_ci(k, n, 2.576) == stats.wilson_ci(k, n, 2.576)

    @pytest.mark.parametrize("table", _FISHER_TABLES)
    def test_fisher_equal(self, hb, table):
        """Defect guarded: an edit to one Fisher copy but not the other."""
        assert hb.fisher_exact_two_sided(*table) == stats.fisher_exact_two_sided(*table)

    @pytest.mark.parametrize(
        ("name", "args"),
        [
            ("wilson_ci", (6, 5)),
            ("wilson_ci", (-1, 5)),
            ("fisher_exact_two_sided", (1, 0, 1, 1)),
            ("fisher_exact_two_sided", (3, 2, 1, 1)),
        ],
    )
    def test_same_rejections(self, hb, name, args):
        """Both copies refuse impossible counts."""
        for impl in (getattr(hb, name), getattr(stats, name)):
            with pytest.raises(ValueError):
                impl(*args)


class TestRecordsParity:
    def test_append_read_round_trip_is_interchangeable(self, hb, tmp_path):
        """Rows written by either copy read back identically through both."""
        rows = [{"run": 1, "ok": True}, {"run": 2, "text": "café", "x": None}]
        a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        for row in rows:
            hb.append_row(a, row)
            records.append_row(b, row)
        assert a.read_bytes() == b.read_bytes()
        assert hb.read_rows(a) == records.read_rows(a) == rows
        assert hb.read_rows(tmp_path / "missing.jsonl") == []
        assert records.read_rows(tmp_path / "missing.jsonl") == []

    def test_write_json_bytes_identical(self, hb, tmp_path):
        obj = {"b": [1, 2.5, None], "a": {"z": "café", "y": True}}
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        hb._write_json(a, obj)
        records.write_json(b, obj)
        assert a.read_bytes() == b.read_bytes()

    def test_utc_now_same_format(self, hb):
        a, b = hb._utc_now(), records.utc_now()
        assert len(a) == len(b) == 20 and a.endswith("Z") and b.endswith("Z")
        assert a[:16] <= b[:16]

    def test_git_commit_equal(self, hb):
        assert hb._git_commit() == records.git_commit(cwd=ROOT)


def test_bench_statistics_work_with_sockets_disabled():
    """Defect guarded (D-008): a delegate to fsm_llm.eval imports litellm,
    which fetches its model-cost map over HTTP, so the offline ``report``
    command opened a socket. The bench helpers must run with sockets off."""
    code = (
        "import socket\n"
        "def boom(*a, **k):\n"
        "    raise AssertionError('socket opened')\n"
        "socket.socket = boom\n"
        "socket.create_connection = boom\n"
        "import sys\n"
        f"sys.path.insert(0, {str(BENCH.parent)!r})\n"
        "import harness_bench as hb\n"
        "assert hb.wilson_ci(3, 5)\n"
        "assert hb.fisher_exact_two_sided(1, 5, 4, 5) > 0\n"
        "print('OFFLINE-OK', 'fsm_llm' in sys.modules)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "FSM_LLM_HARNESS_LIVE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OFFLINE-OK False"
