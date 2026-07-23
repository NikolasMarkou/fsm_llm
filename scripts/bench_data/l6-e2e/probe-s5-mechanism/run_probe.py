"""S5 scratch diagnosis probe: name the mechanism behind B6 run 2's lost credit.

METHOD RECORD (plan-2026-07-23T173454-2c22e5f6, Step 2).  NON-pre-registered:
one-look discipline binds pre-registered B-blocks only; this is a bounded
scratch probe (<= 5 runs) whose deliverable is a MECHANISM VERDICT, not a
graded row.  It writes ONLY under this directory -- never into B6/ or B7/.

The question: B6 run 2 (seed 20260722101) reached REFLECT with an ASSIGNED
EXECUTE target (`uploader.py`, `assigned-prose`) and a real workspace change
(`workspace_files_changed=['uploader.py']`), yet the frozen floor's
`verified_write` stayed False.  The raw per-dispatch observation list was not
retained in B6 (the instrumentation gap Step 1 closed at commit 3a682fb), so
the mechanism could not be read off the committed artifacts.  This probe
reproduces the B6 run-2 configuration WITH the retention active and partitions
every EXECUTE-state record by mechanism:

  (a) bundled-dispatch loss -- the write tool call never appears in the
      returning dispatch's trace (budget exhausted mid-bookkeeping, or a
      later non-writing redispatch is the record that survives);
  (b) fails-after-write / exception branch -- roles.py:1278-1301 hard-codes
      ``write_evidence_paths: ()`` + ``write_required: False`` and re-raises;
      harness.py:1589-1596 catches it and spends a leash attempt;
  (c) label-spelling -- a captured label RESOLVES to the changed file but its
      spelling fails the frozen ``_normalized_ws_path`` membership test;
  wrong-state -- a NON-EXECUTE record's labels name the changed workspace file.

DESIGN: direct reuse of the real test plumbing.  We import
``tests.test_fsm_llm_harness.test_live_ollama`` and call its ``_one_e2e_run``
unmodified, so the probe measures the IDENTICAL configuration (same GOAL, same
factory, same seeds convention, same frozen floor predicate grading the row).
Two module globals are patched, both OUTSIDE the four hash-frozen objects:

  * ``L6_BLOCK = "probe-s5-mechanism"`` -- redirects `_one_e2e_run`'s guarded
    diagnostic retention (artifacts/run-{k}/ + observations.json) into THIS
    directory instead of the committed B6/ block dir.
  * ``E2E_SEED_BASE = seed - (k - 1)`` -- so `_one_e2e_run(tmp, k)` computes
    exactly the scheduled seed for probe run k (run 1 replays B6 run 2's
    20260722101) while keeping distinct artifacts/run-{k}/ retention dirs.
  * ``E2E_WALL_CLOCK_CEILING_S = 520.0`` -- probe-specific DEVIATION from the
    block's 1800 s, forced by the probe's own outer 600 s-per-invocation
    bound: the row must land on disk (disk-first) before the invocation is
    killed.  B6 run 2 -- the configuration under diagnosis -- finished in
    207.5 s, so the EXECUTE-reaching path this probe exists to observe fits
    with >2x headroom; only would-be explore-cap stragglers (B6 run 3: 540.8 s)
    can be truncated, and a truncated run records honestly as ``timed_out``.

USAGE (one full run per invocation, disk-first):
    FSM_LLM_HARNESS_LIVE=1 .venv/bin/python \
        scripts/bench_data/l6-e2e/probe-s5-mechanism/run_probe.py run <k> <seed>
    .venv/bin/python \
        scripts/bench_data/l6-e2e/probe-s5-mechanism/run_probe.py analyze

``run`` appends the full rubric row (which now carries "observations") to
probe-rows.jsonl and writes probe-run-{k}-observations.json, then prints a
one-line summary.  ``analyze`` re-reads probe-rows.jsonl and partitions every
EXECUTE record by the mechanisms above, testing each captured label against
the REAL frozen functions (imported, not re-implemented).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROBE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROBE_DIR.parents[3]
ROWS = PROBE_DIR / "probe-rows.jsonl"

#: Probe-local wall ceiling (seconds); see module docstring for why 520.
PROBE_CEILING_S = 520.0


def _module() -> Any:
    """Import the real live-test module with the probe patches applied."""
    os.environ.setdefault("FSM_LLM_HARNESS_LIVE", "1")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import tests.test_fsm_llm_harness.test_live_ollama as t

    # Redirect _one_e2e_run's guarded diagnostic retention into THIS probe
    # directory.  Never B6/ or B7/.
    t.L6_BLOCK = PROBE_DIR.name
    return t


def _run(k: int, seed: int) -> None:
    t = _module()
    t.E2E_SEED_BASE = seed - (k - 1)  # _one_e2e_run(tmp, k) -> exactly `seed`
    t.E2E_WALL_CLOCK_CEILING_S = PROBE_CEILING_S

    tmp = Path(tempfile.mkdtemp(prefix=f"s5-probe-{k}-"))
    row = t._one_e2e_run(tmp, k)
    assert row["seed"] == seed, (row["seed"], seed)

    # Disk-first: the row lands before any reporting.
    with ROWS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    (PROBE_DIR / f"probe-run-{k}-observations.json").write_text(
        json.dumps(row["observations"], indent=2), encoding="utf-8"
    )

    print(
        f"probe run {k}: seed={row['seed']} wall={row['wall_clock_s']}s "
        f"timed_out={row['timed_out']} furthest={row['furthest_state']} "
        f"verified_write={row['verified_write']} halt_slug={row['halt_slug']} "
        f"dispatch_counts={row['dispatch_counts']} "
        f"ws_changed={row['workspace_files_changed']} "
        f"targets={row['execute_assigned_targets']} "
        f"reasons={row['execute_target_reasons']}"
    )
    reached = "execute" in row["dispatch_counts"] or row["furthest_state"] in (
        "execute",
        "reflect",
        "pivot",
        "close",
    )
    n_exec_records = sum(
        1 for r in row["observations"] if r.get("state") == "execute"
    )
    print(
        f"probe run {k}: reached_execute={reached} "
        f"execute_records_retained={n_exec_records}"
    )


def _analyze() -> None:
    t = _module()
    rows = [
        json.loads(line)
        for line in ROWS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        changed = set(row["workspace_files_changed"])
        print(
            f"\n=== run {row['run']} seed={row['seed']} wall={row['wall_clock_s']}s "
            f"furthest={row['furthest_state']} verified_write={row['verified_write']} "
            f"halt_slug={row['halt_slug']} dispatch_counts={row['dispatch_counts']} "
            f"ws_changed={sorted(changed)} targets={row['execute_assigned_targets']}"
        )
        observations = row["observations"]
        execute_records = [
            r for r in observations if r.get("state") == "execute"
        ]
        n_dispatched = row["dispatch_counts"].get("execute", 0)
        if len(execute_records) != n_dispatched:
            print(
                f"  [absent-record] EXECUTE dispatches={n_dispatched} but "
                f"retained EXECUTE records={len(execute_records)}"
            )
        for i, r in enumerate(execute_records, 1):
            print(
                f"  EXECUTE record {i}: success={r['success']} "
                f"agent_success={r['agent_success']} "
                f"failure_reason={r['failure_reason']!r} "
                f"write_required={r['write_required']} "
                f"write_evidence={r['write_evidence']} "
                f"write_evidence_workspace={r['write_evidence_workspace']} "
                f"write_evidence_plan={r['write_evidence_plan']} "
                f"write_evidence_paths={r['write_evidence_paths']!r} "
                f"answer_chars={r['answer_chars']} "
                f"missing_keys={r['missing_keys']!r} "
                f"elapsed_s={round(r['elapsed_s'], 1)}"
            )
            # Mechanism (b): the exception-branch hard-coded shape.
            if str(r["failure_reason"] or "").startswith("exception:"):
                print(
                    "    -> mechanism (b) shape: exception branch "
                    "(roles.py:1278-1301 hard-codes empty evidence: "
                    f"write_evidence={r['write_evidence']} "
                    f"write_required={r['write_required']} "
                    f"answer_chars={r['answer_chars']})"
                )
            # Mechanism (a): a returning record with NO write tool call in its
            # own trace while the workspace changed.
            if (
                not str(r["failure_reason"] or "").startswith("exception:")
                and not any(
                    str(label).startswith("workspace:")
                    for label in r["write_evidence_paths"]
                )
                and changed
            ):
                print(
                    "    -> mechanism (a) shape: no workspace write in THIS "
                    "dispatch's trace (write tool call absent/unverified) "
                    "while the workspace did change"
                )
            # Mechanism (c): each label tested against the REAL frozen pair.
            for label in r["write_evidence_paths"]:
                root, sep, path = str(label).partition(":")
                member = bool(
                    sep
                    and root == "workspace"
                    and t._normalized_ws_path(path) in changed
                )
                print(
                    f"    label {label!r}: root={root!r} path={path!r} "
                    f"normalized={t._normalized_ws_path(path)!r} "
                    f"frozen-membership-pass={member}"
                )
                if not member and sep and root == "workspace":
                    print(
                        "    -> mechanism (c) candidate: workspace label whose "
                        "spelling fails the frozen membership test"
                    )
        # Wrong-state hypothesis: a NON-EXECUTE record naming a changed
        # workspace file.
        for r in observations:
            if r.get("state") == "execute":
                continue
            for label in r["write_evidence_paths"]:
                root, sep, path = str(label).partition(":")
                if sep and root == "workspace" and (
                    t._normalized_ws_path(path) in changed
                ):
                    print(
                        f"  [wrong-state] {r['state']} record carries "
                        f"{label!r} which names changed file "
                        f"{t._normalized_ws_path(path)!r}"
                    )
        # Cross-check: replay the frozen floor predicate on the retained list.
        replay = t._verified_execute_workspace_write(
            observations, sorted(changed)
        )
        print(
            f"  frozen-floor replay on retained observations: {replay} "
            f"(row said {row['verified_write']})"
        )


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        _run(int(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "analyze":
        _analyze()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
