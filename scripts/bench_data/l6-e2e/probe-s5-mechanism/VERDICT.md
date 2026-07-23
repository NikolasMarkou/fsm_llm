# S5 Scratch Probe — Mechanism Verdict

*Plan: plan-2026-07-23T173454-2c22e5f6, Step 2. NON-pre-registered scratch
probe (one-look discipline binds pre-registered B-blocks only). Nothing here
touches `B6/` or `B7/`.*

## Verdict

**Mechanism (c) — label-spelling.** The EXECUTE dispatch's write WAS issued,
DID land bytes, and WAS credited internally (D-016 passed, `success=True`),
but the model passed the **full absolute tmp path** as the `write_file`
`path` parameter, so the raw-parameter label recorded at `roles.py:944` reads
`workspace:/tmp/s5-probe-1-0wob0yk7/e2e-1/workspace/uploader.py`. The frozen
`_normalized_ws_path` strips only the leading `/workspace` SENTINEL spelling
(first path component after `/` must be literally `workspace`; here it is
`tmp`), so the label normalizes to
`tmp/s5-probe-1-0wob0yk7/e2e-1/workspace/uploader.py`, which is not a member
of `workspace_files_changed=['uploader.py']` — the frozen floor fails CLOSED
and `verified_write=False` despite a real, assigned, verified write.

Mechanisms (a) bundled-dispatch loss, (b) fails-after-write/exception, and
the wrong-state hypothesis are each REFUTED for this run by retained record
fields (below). Note the OBSERVED (c) spelling is the absolute-REAL-path
variant, not the whitespace/`./` sub-variant the findings hypothesized
(`findings/s5-credit-layer-mechanism.md`, candidate (c)) and not the
`/workspace/...` sentinel (which `_normalized_ws_path` does repair).

## Probe design

- Method record: `run_probe.py` (this directory). Direct reuse of the real
  test plumbing: imports `tests.test_fsm_llm_harness.test_live_ollama` and
  calls `_one_e2e_run` unmodified — same GOAL, same
  `build_default_worker_factory` (native, seed-pinned), same
  `DiskEvidenceApprovals`, same frozen floor predicate grading the row, with
  the Step-1 (commit 3a682fb) raw-observation retention active.
- Patched module globals, all outside the four hash-frozen objects:
  `L6_BLOCK="probe-s5-mechanism"` (redirects `_one_e2e_run`'s guarded
  diagnostic retention into this directory), `E2E_SEED_BASE` (per-run seed
  control), `E2E_WALL_CLOCK_CEILING_S=520.0` (probe-specific deviation from
  the block's 1800 s, forced by the probe's 600 s-per-invocation outer bound
  so the row always lands disk-first; irrelevant in the event — the run
  finished in 170.9 s).
- Seed schedule: 20260722101 (the B6 run-2 seed) first, then
  20260723201..204, ≤5 runs, STOP EARLY at the first EXECUTE-reaching run
  with retained EXECUTE observations.
- **Stopped after run 1**: the stop condition fired on the first run.
  Seeds 20260723201..204 were NOT run.

## Per-run table

| run | seed | wall (s) | timed_out | furthest_state | verified_write | halt_slug | dispatch_counts | EXECUTE records |
|---|---|---|---|---|---|---|---|---|
| 1 | 20260722101 | 170.9 | false | reflect | **false** | None (REFLECT stall) | explore 3, plan 1, execute 1, reflect 1 | 1 (success=True, 2 workspace write labels) |

Congruence with B6 run 2 (the run under diagnosis; same seed, same goal):
identical `dispatch_counts` (`{'explore': 3, 'plan': 1, 'execute': 1,
'reflect': 1}`), identical `execute_assigned_targets=['uploader.py']` /
`execute_target_reasons=['assigned-prose']`, identical
`workspace_files_changed=['uploader.py']`, identical
`write_evidence_dispatches=4`, identical `verified_write=false`, identical
slugless REFLECT-stall ending (`halt_slug=None`), wall 170.9 s vs 207.5 s.
The anomaly REPRODUCED on the first attempt in the same shape end to end.

## Mechanism partition — evidence, quoted from retained records

Source of truth: `probe-rows.jsonl` row 1 / `probe-run-1-observations.json`
(identical list also under `artifacts/run-1/observations.json`). The run's
single EXECUTE record, in full:

```json
{
  "role": "executor",
  "state": "execute",
  "success": true,
  "failure_reason": null,
  "missing_keys": [],
  "top_level_objects": 1,
  "agent_success": true,
  "answer_chars": 198,
  "elapsed_s": 11.637934698999743,
  "write_evidence": 2,
  "write_evidence_workspace": 2,
  "write_evidence_plan": 0,
  "write_evidence_paths": [
    "workspace:/tmp/s5-probe-1-0wob0yk7/e2e-1/workspace/uploader.py",
    "workspace:/tmp/s5-probe-1-0wob0yk7/e2e-1/workspace/uploader.py"
  ],
  "write_required": true,
  "claimed_findings_count": null,
  "derived_findings_count": null
}
```

- **(b) fails-after-write / exception branch — REFUTED for this run.** The
  exception-branch shape (`roles.py:1278-1301`) is
  `success=false, failure_reason="exception:…", write_evidence=0,
  write_required=false, answer_chars=0`. The retained record has
  `success=true`, `failure_reason=null`, `write_required=true`,
  `write_evidence=2`, `answer_chars=198`. No absent record either:
  `dispatch_counts['execute']=1` and exactly 1 EXECUTE record was retained.
- **(a) bundled-dispatch loss — REFUTED for this run.** The write tool calls
  DID appear in the returning dispatch's own trace:
  `write_evidence_workspace=2` with two `workspace:…uploader.py` labels, and
  the driver log shows both writes inside the one dispatch
  (`workspace wrote uploader.py (492 chars)` then `(468 chars)`). Nothing was
  lost across dispatches; there was only one EXECUTE dispatch.
- **(c) label-spelling — CONFIRMED.** Each captured label tested against the
  REAL frozen functions (imported, not re-implemented; see `run_probe.py`
  `analyze`):
  - label `workspace:/tmp/s5-probe-1-0wob0yk7/e2e-1/workspace/uploader.py`
    → `root='workspace'`,
    `_normalized_ws_path(path)='tmp/s5-probe-1-0wob0yk7/e2e-1/workspace/uploader.py'`,
    membership in `{'uploader.py'}` → **False** (both labels, identically).
  - The label RESOLVES to the changed file: `Workspace.resolve()` accepted
    the in-root absolute path (the write landed; the run-level sha256 diff
    shows `uploader.py` changed) and `has_bytes` verified it — which is why
    `write_evidence_workspace=2` and the dispatch is `success=true`.
  - Replaying the frozen floor on the retained list reproduces the row:
    `_verified_execute_workspace_write(observations, ['uploader.py'])` →
    `False` (row said `false`) — the retention is faithful to what the floor
    graded.
- **Wrong-state hypothesis — REFUTED.** No non-EXECUTE record carries a
  workspace-root label naming the changed file. The other records'
  `write_evidence_paths`: explore ×3 → `plan:findings/…` (one each), plan →
  `[]`, reflect → `[]`.

## Why the model had an absolute path to echo

Not required for the verdict, recorded for Step 3: the dispatch context and
tool feedback expose the REAL workspace root (e.g.
`workspace_root=/tmp/…/workspace` in the request context snapshot), and
`Workspace.resolve()` deliberately accepts in-root absolute paths, so an
echo-back of the absolute spelling both works AND produces a raw label the
frozen normalizer cannot credit. The mismatch is between what
`_verified_writes` verifies (bytes at the RESOLVED path, `roles.py:940-943`)
and the RAW `call.parameters["path"]` string used for the label
(`roles.py:937,944`) — exactly the gap the findings called "structurally
real" for candidate (c), in a different spelling than hypothesized.

## Implication (gates Step 3)

Verdict (c) selects **Branch B**: normalize at the label construction site
(`roles.py:944`, NOT frozen) so the label carries the workspace-relative
resolved spelling — feeding the frozen predicate a clean label is the
sanctioned path; the four frozen floor objects stay byte-identical. The
normalization must cover the absolute-real-path spelling observed here (not
just whitespace/`./`).

## Honest caveats

- n=1 EXECUTE-reaching probe run (stop-early rule). Mechanisms (a)/(b)
  remain structurally present code paths (`roles.py:1278-1301`,
  `harness.py:1589-1596`); they are refuted as the mechanism OF THIS RUN,
  not as ever-possible shapes.
- B6 run 2 itself is not byte-confirmed (its observations were never
  retained — that is why this probe exists). The attribution to B6 run 2
  rests on same-seed same-config reproduction with a field-for-field
  congruent row signature.
- The probe run also reproduced the slugless REFLECT stall
  (`halt_slug=None`, `honest_halt=false`, `halt_reason="Stalled in REFLECT
  for 3 turns with no progress…"`) — a known unbudgeted stall class
  (REFLECT/PIVOT/CLOSE), out of scope for this step, recorded here so it is
  not lost.
- The PLAN record shows `failure_reason="unverified-write"` with zero plan
  write evidence while `plan_md_bytes=3037`: expected under the
  driver-rendered structured plan (the driver, not a model tool call, writes
  `plan.md`), not a new anomaly.
