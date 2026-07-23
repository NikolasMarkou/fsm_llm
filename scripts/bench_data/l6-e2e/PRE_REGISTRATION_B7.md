# L6 `l6-e2e/B7` — pre-registration (S5 label normalization + S4b reflect-cap)

*Plan: plan-2026-07-23T173454-2c22e5f6 (harness iteration 9 / successors S5 + S4b).*
*Committed BEFORE the B7 live run (git log: this file precedes any `B7/` rows
commit). One-look / no-`--force` / no-re-sample (established one-look
discipline). The decision rule below is fixed BEFORE any row exists.*

## What changed since B6 (the things under test)

Exactly TWO driver-side behavior changes land between B6 and B7. They target
DISJOINT floor clauses, so per-clause attribution in the retained per-run rows
stays clean (D-003 trade-off):

1. **S5 write-evidence label normalization** (commit `aa0ecb6`, D-002 —
   targets the **`verified_write`** clause). B6 run 2 reached EXECUTE with an
   assigned prose target (`uploader.py`) and the workspace sha256 diff showed
   the file changed, yet `verified_write=false`. The mechanism was MEASURED
   (not inferred) by a scratch instrumented reproduction at the same seed
   (`probe-s5-mechanism/VERDICT.md`, retained-record evidence, anomaly
   reproduced 1/1): mechanism **(c) label-spelling, absolute-real-path
   variant** — the model passed the full in-root ABSOLUTE tmp path as
   `write_file`'s `path`; `Workspace.resolve()` accepted it and bytes landed,
   but the raw-parameter label
   (`workspace:/tmp/…/workspace/uploader.py`) normalizes under the frozen
   `_normalized_ws_path` to a non-member of the changed-file set — the floor
   fails CLOSED. Mechanisms (a) bundled-dispatch loss and (b)
   exception-branch empty evidence were REFUTED for that run by the same
   retained record. The fix routes every verified write's LABEL through the
   SAME resolve chokepoint that verified the bytes (`roles.py` `_evidence_path`;
   the frozen predicate is fed a clean root-relative label — the frozen floor
   itself is untouched). Red-before regression class
   `TestWriteEvidenceLabelNormalization` (commit `333b134`) includes a
   frozen-floor replay: the exact retained probe observation list now grades
   `verified_write=True`.
2. **S4b honest reflect-cap budget** (commit `66df5c3`, D-003 — targets the
   **`honest_halt`** clause). Every EXECUTE-reaching run ever observed (B5
   run 1, B6 run 2, the S5 probe run) ended in a SLUGLESS REFLECT stall
   (`halt_slug=None`, `honest_halt=false`). The measured stall variant is a
   `success=True` but UNROUTABLE verifier reply (not a worker failure), so
   the budget is keyed on ROUTABILITY: `Defaults.MAX_REFLECT_REDISPATCHES=3`
   + `GateSlug.REFLECT_CAP` (`"reflect-cap"`), mirroring the proven
   S2/plan-cap machinery — driver run-state counter, reset only in
   `_run_once`, unreachable from approval callbacks. A REFLECT-stuck run now
   redispatches up to the cap, then HALTS HONESTLY with the `reflect-cap`
   slug.

Additionally (instrumentation, not a behavior change): **raw per-dispatch
observation retention** is now active (commit `3a682fb`) — each row carries
the full `observations` list (additive row field) and a guarded
`artifacts/run-{n}/observations.json`, so any credit-layer anomaly in B7 is
diagnosable from the committed block WITHOUT a re-run (the gap that forced
the S5 scratch probe).

## Floor-clause identity (HARD)

`_verified_execute_workspace_write` + `_normalized_ws_path` + `_bench_defect` +
the floor-loop test body (`test_three_full_runs_grade_at_or_above_the_floor`)
hash to
`cbeeb6aaa2a5a2e5910fed7abb77e1b44651c8d81715301e2c4b547509b453ce` — verified
IDENTICAL to B1-B6 both BEFORE and AFTER the `L6_BLOCK` bump ("B6"→"B7") in
this registration commit (the `inspect.getsource` sha256 command, run at HEAD
before the edit and re-run after). Only `L6_BLOCK` changed in
`test_live_ollama.py` for this block; the four floor-graded objects are
byte-unchanged. The S5 fix touched only `roles.py` (label construction, NOT
the frozen normalizer); the S4b fix touched only `constants.py`/`harness.py`
(driver-side) — NOT the floor grading. Feeding the frozen predicate a clean
label is the sanctioned path; the bar itself never moved.

## What is UNCHANGED (the disk-truth gates + ethos)

Model `ollama_chat/qwen3.5:4b` (digest `2a654d98e6fb`, same as B0-B6). Same
GOAL, same worker factory, same `DiskEvidenceApprovals`, same seed schedule
(`E2E_SEED_BASE=20260722100`, per-row `base+run-1`) — the same goal/config
lineage as B6. `derive_execute_target` + prose fallback, `_plan_is_approvable`,
`_plan_has_content`, D-016 all byte-unchanged since B6. The model authors all
content; the driver assigns an existing-file target, renders the structured
plan, and now (S5) spells the evidence label from the resolved path — it
invents no prose and cannot open a gate with a sentence.

## Block parameters

Bench id/block `l6-e2e`/**B7**; n=3 (`RUNS_E2E=3`); real workers
(`_one_e2e_run`), `E2E_WALL_CLOCK_CEILING_S=1800.0` per run. One look, no
`--force`, no interim row reads; the one-look guard refuses an existing
`B7/rows.jsonl`. Failed-artifact retention active → `B7/artifacts/run-{n}/`,
now including `observations.json` per run.

## Pre-registered grading (decision rule, fixed before any row)

Each run graded on the same three frozen floor clauses: (1) reached
`>= EXECUTE`, (2) `verified_write` — an EXECUTE-state dispatch's OWN written
workspace path intersects the sha256-diffed changed-file set, (3)
`honest_halt`. Plus `_bench_defect`.

**Per-clause attribution branches** (each fix is graded by its own clause):

- **(i) S5 / `verified_write`**: any run reaching `>= EXECUTE` with a
  workspace write should now show `verified_write=True`. If an
  EXECUTE-reaching run that changed workspace files still shows
  `verified_write=False`, the S5 fix is **REFUTED in-loop** — diagnose
  directly from the retained `observations` (the labels are in the row; no
  re-run needed).
- **(ii) S4b / `honest_halt`**: any REFLECT-stuck run should either route
  onward or halt HONESTLY with `halt_slug="reflect-cap"`. A SLUGLESS stall
  (`halt_slug=None` after reaching REFLECT) **REFUTES the S4b fix**.
- **(iii) The floor (founding criterion)**: a floor row = `>= EXECUTE` AND
  `verified_write` AND `honest_halt`. The floor clause of the frozen body
  clears only per its own (byte-identical) rule — B0-B6 were all 0/3 at this
  floor. **The aimed outcome** is >= 1/3 floor rows; a 3/3 EXECUTE-reach is
  NOT expected (S4c explore/plan variance, below).

**Pre-committed HONEST-NEGATIVE branch**: the floor may stay **0/3** without
either fix being refuted — e.g. S4c explore-cap variance keeps runs from
EXECUTE entirely (B6: 2/3 explore-cap halts; EXECUTE-reach has run ~1/3 per
block). In that case the block's deliverables are the per-clause validations
on whatever states WERE reached, plus the retained raw observations for every
dispatch of every run. That outcome is a pre-blessed deliverable, not a
failure, and will be reported as measured.

**Pre-Mortem #3 defect-halt branch (frozen body)**: if runs 1 AND 2 both end
in a crash/hang/SLUGLESS stall, the block halts early without spending run 3
(bench-defect rule, byte-identical since B1). NOTE, pre-registered: the S4b
fix specifically targets the slugless class — this branch FIRING via two
slugless stalls would itself be evidence AGAINST the S4b fix, and will be
reported as such.

**No capability-ceiling claims** will be made from this block regardless of
outcome — in particular, neither "4b can drive the harness" (if the floor
clears) nor "4b cannot" (if it stays 0/3). n=3, one configuration, one goal.

## Run command (one look)

```
FSM_LLM_HARNESS_LIVE=1 .venv/bin/python -m pytest \
  "tests/test_fsm_llm_harness/test_live_ollama.py::TestL6EndToEndRealWorkers::test_three_full_runs_grade_at_or_above_the_floor" -q -s
```
(after `L6_BLOCK = "B7"`; same pytest node id as B6's run command).
