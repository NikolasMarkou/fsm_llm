# L6 `l6-e2e/B4` — pre-registration (response_format STRUCTURED plan, e2e)

*Plan: plan-2026-07-23T124347-09045e6e (harness iteration 7 / S3).*
*Committed BEFORE the B4 live run (git log: this file precedes the `B4/` rows
commit). One-look / no-`--force` / no-re-sample (D-002).*

## What changed since B3 (the thing under test)
Iteration-7 fix (D-001/D-002, commits `00c4289` + `ac40e3c` + `ffd461b` +
`7f4f6e2`): the PLAN role now authors the 11 `PlanSchema.SECTIONS` as required
`response_format` fields. On a PLAN dispatch the model returns the 11 sections as
structured string fields under a `response_format` json_schema — reusing the
EXISTING D-002 repair turn (no edit to `native_fc.py`/`base.py`). The driver
renderer `_render_plan_from_structured` reads `result.structured_output`, maps the
11 fields into a `PlanDoc` in `PlanSchema.SECTIONS` order, and writes `plan.md`
via `PlanDirectory.write_artifact`. Because each field is placed under ITS heading
BY CONSTRUCTION, content distributes into all 11 sections — the exact failure the
iter-6 scaffold+append machinery could NOT achieve (`append_plan_file` appends to
the file END; content could not distribute; source-REFUTED, floor 0/3 four times
B0-B3). That refuted machinery (`_seed_plan_scaffold`, `_empty_plan_scaffold`,
`_plan_deliverable_line`, PLAN `force_final_tool=append_plan_file`) is REMOVED in
this iteration. The driver RENDERS the model's fields but NEVER invents filler: an
empty/placeholder field renders an empty section body that the UNCHANGED
`_plan_is_approvable` denies → redispatch budget → honest `plan-cap` halt.

This targets B3's residual: B3 confirmed 4b writes substantial plan content
(15-18 KB) and CLOSED the S2 slugless stall (3/3 honest `plan-cap`), but the
append mechanism concentrated content into one section / produced duplicate
headers, so the plan was never a valid 11-section approvable document and the
floor stayed 0/3. B4 measures whether the response_format structured render
produces a distribution-correct, approvable plan and advances past PLAN.

## Floor-clause identity (HARD)
`_verified_execute_workspace_write` + `_normalized_ws_path` + `_bench_defect` +
the floor-loop test body (`test_three_full_runs_grade_at_or_above_the_floor`)
hash to
`cbeeb6aaa2a5a2e5910fed7abb77e1b44651c8d81715301e2c4b547509b453ce` — verified
IDENTICAL to B1/B2/B3 both BEFORE and AFTER the `L6_BLOCK` bump. Only `L6_BLOCK`
("B3"→"B4") and an ADDITIVE, diagnostic-only failed-artifact retention block in
`_one_e2e_run` (copies each run's `plan.md`/`decisions.md`/`state.md` to
`B4/artifacts/run-{n}/` before tmp_path teardown, wrapped so a copy failure never
perturbs the grade) changed in `test_live_ollama.py`; the four floor-graded
functions are byte-unchanged. B4's manifest `prompt_bytes_sha256` legitimately
differs (the fix changes the PLAN dispatch); `model_digest` unchanged
(4b `2a654d98e6fb`).

## What is UNCHANGED (the S2 aligned-gates machinery)
`_plan_is_approvable` / `_plan_has_content` / `DiskEvidenceApprovals` are
BYTE-UNCHANGED — format-agnostic, they read the driver-rendered file off disk, so
"a confident sentence cannot open a gate" stays INTACT. The D-002 repair-turn
plumbing (`native_fc.py`, `base.py`) is REUSED, not modified. Model stays
`ollama_chat/qwen3.5:4b`. Structural fix only.

## Block parameters (identical to B3 except the block name + the fixed product)
Bench id/block `l6-e2e`/**B4**; n=3 (`RUNS_E2E=3`); real workers (`_one_e2e_run`),
`E2E_WALL_CLOCK_CEILING_S` per run; model `ollama_chat/qwen3.5:4b`
(digest `2a654d98e6fb`, same as B0-B3). One look, no `--force`, no interim row
reads.

## Pre-registered grading (the floor — transcribed from B1/B2/B3, byte-identical)
Each run graded on the same three floor clauses: (1) `reached >= EXECUTE`,
(2) `verified_write` — an EXECUTE-state dispatch's OWN written workspace path
intersects the sha256-diffed changed-file set, (3) `honest_halt`. Plus
`_bench_defect`.

**Success criterion (the floor)**: ≥1 of 3 runs reaches ≥EXECUTE with a verified
EXECUTE-state workspace write + honest halt (B0/B1/B2/B3 were all 0/3 at this
floor). Report all three rubric vectors regardless.

**Pre-committed honest-negative branch**: recording "floor still 0/3, wall moved
(now blocks at approval/EXECUTE despite distributed sections) OR held at the
PLAN-writer" is a VALID deliverable. A falsifiable criterion failing honestly is
not a defect (LESSONS). Both outcomes are the block's output.

**Pre-registered PARTIAL-advance shapes** (report explicitly if observed):
- A run reaches EXECUTE but leaves no verified EXECUTE-state workspace write
  (floor clause 2 false) — advance without floor clearance.
- Plans now DISTRIBUTE into all 11 sections (the B3 append flaw is gone) yet
  approval or EXECUTE still blocks — mechanism moved, floor not yet met.
- All 11 sections render non-empty but the plan is present-but-THIN /
  unapprovable (Pre-Mortem #4). **Pre-approved recovery** (USER-SANCTIONED, no
  new fork): split the single response_format completion into N per-section
  response_format calls, one field per call, same repair-turn machinery, same
  renderer, same unchanged gate. Contingency only — NOT built now; single-call is
  the F4-probe-validated 5/5 primary path; triggered solely by this measurement.

## W3 (falsifier grading, if reached)
If any run reaches PLAN and redispatches, grade `plan_redispatches` vs
`MAX_PLAN_REDISPATCHES=3`. If a run reaches EXECUTE, note the EXECUTE-state
behavior (the tightened verified-write floor clause).

## Ethos check (post-run)
Confirm from the B4 rows (and the retained `B4/artifacts/run-{n}/plan.md`) that
any run reaching EXECUTE did so on a model-authored, all-sections-non-placeholder
plan.md — the driver only FORMATS the model's 11 fields and invents no filler; an
empty field renders an empty section the unchanged `_plan_is_approvable` denies.

## Run command (one look)
```
FSM_LLM_HARNESS_LIVE=1 .venv/bin/python -m pytest \
  "tests/test_fsm_llm_harness/test_live_ollama.py::TestL6EndToEndRealWorkers::test_three_full_runs_grade_at_or_above_the_floor" -q -s
```
(after `L6_BLOCK = "B4"`).
