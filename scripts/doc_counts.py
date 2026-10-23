#!/usr/bin/env python
"""Fail when a test count written in a tracked .md file differs from a collection.

Run `.venv/bin/python scripts/doc_counts.py` (or `make counts`); exit 1 on a
problem. `measure()` runs ONE child `pytest --collect-only` of `tests/` (the
command and env scrub of `tests/test_packaging.py::measured`) that loads this
module as a plugin to read each item's markers; every number comes from that
run or a glob. `scan(measured)` ties each count token of each tracked .md file
to a measured number by an explicit rule and returns the problems; a token no
rule names is UNCLASSIFIED, also a problem. HISTORICAL files are skipped.

The documented counts are the counts WITHOUT the optional `mcp` package, so
with `mcp` importable `main` and the slow test in `tests/test_packaging.py` skip.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ENV = "DOC_COUNTS_OUT"
MARKERS = ("slow", "integration", "real_llm", "examples")
SCRUB = ("PYTEST_ADDOPTS", "FSM_LLM_HARNESS_LIVE", "SKIP_SLOW_TESTS", "TEST_REAL_LLM")
MCP_INSTALLED = importlib.util.find_spec("mcp") is not None
MCP_SKIP = "skipped: `mcp` is importable; the documented counts are the no-mcp counts"
HISTORICAL = {
    "CHANGELOG.md": "release history",
    "docs/agents_roadmap.md": "dated roadmap and bench reports",
    "plans/": "retired plan directories",
    "evaluation/": "dated eval runs",
    "scripts/bench_data/": "pre-registered bench blocks, never edited",
}
NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
TOK = re.compile(
    rf"(?P<n>{NUM})\s+(?:offline )?(?P<kind>collected tests|collected|tests|test files"
    rf"|`?test_\*\.py`? files|test modules)\b"
    rf"|(?:collected|collects) (?P<n2>{NUM})\b|suite count \((?P<n3>{NUM})\)"
)
CELL = re.compile(r"\|\s*(\d{1,3}(?:,\d{3})*)(?: \([^|)]*\))?\s*(?=\|)")
RUN = re.compile(r"pytest (?:\S+ )*?tests/(test_[a-z_]+)")
LEAD_FILE = re.compile(r"^\s*[|-]\s*`((?:test_[a-z0-9_]+|conftest)\.py)`")
LEAD_SUITE = re.compile(r"^\s*\|\s*`(test_[a-z_]+)/`")
TOTAL_LINE = re.compile(
    "make test|Run all tests|Run full test suite|full test tree"
    "|whole pytest test tree|The full pytest tree"
)
CLASS = re.compile(r"`(Test\w+)`")
#: (kind, pattern with group n): a token that carries its own context.
PHRASES = (
    ("marker", rf"`(?P<k>{'|'.join(MARKERS)})` \(?(?P<n>{NUM})(?! s)\b"),
    ("not_slow", rf"-m \"not slow\"`? (?:collects|runs) (?P<n>{NUM})"),
    ("suite_sum", rf"sum to (?P<n>{NUM})"),
    ("slow_here", rf"(?P<n>{NUM}) (?:tests are )?(?:marked `slow`|`?slow\b)"),
    ("test_files", rf"tests,? (?:in )?(?P<n>{NUM}) files"),
    ("live", rf"\b(?P<n>{NUM}) (?:live|gated)\b"),
    ("slow_here", rf"deselects (?P<n>{NUM}) tests"),
    ("slow_here", rf"as a whole \((?P<n>{NUM}) tests\)"),
    ("slow_lead", rf"(?P<n>{NUM}) of them `slow`"),
    ("slow_file", r"`(?P<f>test_[a-z0-9_]+\.py)` (?P<n>\d+)\b"),
)


def pytest_collection_finish(session):
    """Child side (`-p doc_counts`): write each item's node id and markers.

    `live` is not a marker: it tags a `real_llm` test that also has a `skipif`
    (the gated Ollama tests, off by default)."""
    rows = []
    for i in session.items:
        names = {m.name for m in i.iter_markers()}
        live = {"live"} if {"real_llm", "skipif"} <= names else set()
        rows.append([i.nodeid, sorted(names & set(MARKERS) | live)])
    Path(os.environ[OUT_ENV]).write_text(json.dumps(rows))


def _suite(path: str) -> str | None:
    parts = path.split("/")
    return parts[1] if len(parts) >= 3 else None


def _num(text: str) -> int:
    return int(text.replace(",", ""))


def measure() -> dict:
    """One child collection of `tests/`; returns the measured numbers.

    Keys: total; files, slow_files, live_files (Counter path -> tests; test files
    with none included); classes (Counter (suite, class) -> tests); markers (Counter,
    plus "not slow"); suites, slow_suites, live_suites (key None = root files); test_files (suite
    -> number of test_*.py); by_base (file name -> repo paths). A failed child
    raises RuntimeError with the output tails.
    """
    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "scripts")] + ([env["PYTHONPATH"]] if "PYTHONPATH" in env else [])
    )
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts="]
    cmd += ["-p", "no:cacheprovider", "-p", "doc_counts", "tests/"]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "items.json"
        env[OUT_ENV] = str(out)
        run = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=600
        )
        if run.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"collection failed: {run.stdout[-1500:]}{run.stderr[-1500:]}"
            )
        rows = json.loads(out.read_text())
    files, slow_files, live_files = Counter(), Counter(), Counter()
    classes, markers = Counter(), Counter()
    for nodeid, marks in rows:
        parts = nodeid.split("[", 1)[0].split("::")
        files[parts[0]] += 1
        slow_files[parts[0]] += "slow" in marks
        live_files[parts[0]] += "live" in marks
        if len(parts) > 2:
            classes[(_suite(parts[0]), parts[1])] += 1
        markers.update(marks)
    markers["not slow"] = len(rows) - markers["slow"]
    by_base: dict[str, list[str]] = {}
    test_files = Counter()
    for p in sorted((ROOT / "tests").rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if p.name.startswith("test_") or p.name == "conftest.py":
            by_base.setdefault(p.name, []).append(rel)
            files.setdefault(rel, 0)
            test_files[_suite(rel)] += p.name.startswith("test_")
    suites, slow_suites, live_suites = Counter(), Counter(), Counter()
    for counter, into in (
        (files, suites),
        (slow_files, slow_suites),
        (live_files, live_suites),
    ):
        for path, n in counter.items():
            into[_suite(path)] += n
    return {
        "total": len(rows), "files": files, "slow_files": slow_files,
        "classes": classes, "markers": markers, "suites": suites,
        "slow_suites": slow_suites, "live_files": live_files,
        "live_suites": live_suites, "by_base": by_base, "test_files": test_files,
    }  # fmt: skip


class _Doc:
    """The count tokens of one .md file against the measured numbers `m`."""

    def __init__(self, m: dict, md: str):
        parts = md.split("/")
        self.m, self.md, self.bad = m, md, []
        self.suite = parts[1] if parts[0] == "tests" and _suite(md) else None
        self.src_suite = self.suite
        if parts[:2] == ["src", "fsm_llm"] and len(parts) >= 4:
            self.src_suite = f"test_fsm_llm_{parts[2]}"

    def file_in(self, base: str) -> str | None:
        """The one repo path a file name means here (same folder, suite, else any)."""
        cands, here = self.m["by_base"].get(base, []), str(Path(self.md).parent)
        near = [c for c in cands if str(Path(c).parent) == here]
        near = near or [c for c in cands if f"/{self.suite}/" in c] or cands
        return near[0] if len(near) == 1 else None

    def put(self, at: int, tok: str, got: int, want: tuple[int, str] | None) -> None:
        """Record one comparison; `want` None means no rule names the token."""
        if want is None:
            self.bad.append(f"UNCLASSIFIED {self.md}:{at} {tok!r}: no rule")
        elif got != want[0]:
            self.bad.append(
                f"MISMATCH {self.md}:{at} {tok!r}: documented {got}, "
                f"measured {want[0]} ({want[1]})"
            )

    def phrase(self, kind: str, x: re.Match, slowish: bool, lead, lead_suite):
        m, s = self.m, (lead_suite[1] if lead_suite else self.suite)
        if kind == "marker":
            return m["markers"][x["k"]], f"marker {x['k']}"
        if kind == "not_slow":
            n = m["suites"][s] - m["slow_suites"][s] if s else m["markers"]["not slow"]
            return n, "not slow collection here"
        if kind == "suite_sum":
            return m["total"] - m["suites"][None], "the ten suites summed"
        if kind == "test_files":
            return m["test_files"][s], f"test files in {s}"
        if kind in ("slow_here", "live"):  # the file's, else the suite's, else all
            c = "slow" if kind == "slow_here" else "live"
            if lead:
                return m[f"{c}_files"][lead], f"{c} tests in {lead}"
            by_suite = m[f"{c}_suites"]
            return by_suite[s] if s else sum(by_suite.values()), f"{c} tests in {s}"
        f = lead if kind == "slow_lead" else self.file_in(x["f"])
        return (m["slow_files"][f], f"slow tests in {f}") if f and slowish else None

    def token(self, tm: re.Match, line: str, first: int, lead, lead_suite):
        """(number, label) a count token must equal; None when no rule names it."""
        m, s, kind = self.m, lead_suite[1] if lead_suite else None, tm["kind"] or ""
        run, classes = RUN.search(line), set(CLASS.findall(line))
        if kind in ("test files", "test modules") or "test_*.py" in kind:
            sub = self.src_suite
            ok = sub in m["suites"]
            return (m["test_files"][sub], f"test files in {sub}") if ok else None
        if TOTAL_LINE.search(line):
            return m["total"], "total collection"
        if run and first == 0:
            return m["suites"][run[1]], f"suite {run[1]}"
        if s and CELL.search(line):  # a sub-count in a suite row: one file of it
            ones = {n for p, n in m["files"].items() if f"/{s}/" in p}
            val = _num(tm["n"] or tm["n2"] or tm["n3"])
            return (val if val in ones else -1), f"one file of {s}: {sorted(ones)}"
        if s and first == 0:
            return m["suites"][s], f"suite {s}"
        if lead and first == 0:
            return m["files"][lead], f"file {lead}"
        if (
            len(classes) == 1 and m["classes"][(self.suite, *classes)]
        ):  # "fails 2 tests"
            return m["classes"][(self.suite, *classes)], f"class {classes}"
        return (m["suites"][self.suite], f"suite {self.suite}") if self.suite else None

    def line(self, ln: int, line: str) -> None:
        lead_suite, lead_file = LEAD_SUITE.match(line), LEAD_FILE.match(line)
        lead = self.file_in(lead_file[1]) if lead_file else None
        slowish, owned = bool(re.search("slow|deselect", line)), []
        for kind, pat in PHRASES:
            for x in re.finditer(pat, line):
                if want := self.phrase(kind, x, slowish, lead, lead_suite):
                    self.put(ln, x[0], _num(x["n"]), want)
                    owned.append(x.span())
        for cm in CELL.finditer(line) if (lead or lead_suite) else ():
            name = lead_suite[1] if lead_suite else lead
            want = self.m["suites" if lead_suite else "files"][name]
            self.put(ln, f"cell {cm[1]}", _num(cm[1]), (want, name))
        for first, tm in enumerate(TOK.finditer(line)):
            if not any(a <= tm.start() < b for a, b in owned):
                got = _num(tm["n"] or tm["n2"] or tm["n3"])
                self.put(ln, tm[0], got, self.token(tm, line, first, lead, lead_suite))


def scan(measured: dict) -> list[str]:
    """Problems found in the tracked .md files against `measured`; [] when clean."""
    bad: list[str] = []
    names = subprocess.check_output(["git", "ls-files", "*.md"], cwd=ROOT, text=True)
    for md in names.splitlines():
        if md in HISTORICAL or md.startswith(tuple(HISTORICAL)):
            continue
        doc = _Doc(measured, md)
        lines = (ROOT / md).read_text(errors="replace").splitlines()
        for ln, line in enumerate(lines, 1):
            doc.line(ln, line)
        bad += doc.bad
    return bad


def main() -> int:
    if MCP_INSTALLED:
        print(MCP_SKIP)
        return 0
    start = time.monotonic()
    m = measure()
    bad = scan(m)
    print(f"measured total={m['total']} markers={dict(m['markers'])}")
    print(f"suites={dict(m['suites'])}; not compared: {sorted(HISTORICAL)}")
    print("\n".join(bad))
    print(f"{len(bad)} problems, {time.monotonic() - start:.1f} s")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
