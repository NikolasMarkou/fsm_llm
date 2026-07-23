# L6 `l6-e2e/B5` — pre-registration (completion-fix: PLAN gate keys on rendered-disk-content)

*Plan: plan-2026-07-23T124347-09045e6e (harness iteration 7 / S3, same-iteration completion-fix D-002).*
*Committed BEFORE the B5 live run (git log: this file precedes the `B5/` rows commit). One-look / no-`--force` / no-re-sample (D-002/D-003 one-look discipline).*

## What changed since B4 (the thing under test)
B4 (`9745b3a`) measured floor **0/3** but the result was **CONFOUNDED — the
configuration could not have succeeded** (decisions.md D-002, root cause
MEASURED via run log + live `litellm.completion` repro + source read): the
response_format PLAN dispatch DOES produce a valid 14-field `structured_output`
(repro: clean AND polluted contexts both return valid JSON; the harness worker
preserves it at `roles.py:1429`), but the dispatch's `result.success` is **False**
because the D-016 write-obligation (`roles.py:1360-1363`: a role holding a write
tool must show a verified byte-write) fires — under response_format the model
calls NO write tool (the driver renders instead) — and the renderer
`_render_plan_from_structured` was gated on `result.success`, so it DISCARDED the
valid plan → plan.md 0 bytes → `_plan_has_content` False → redispatch → `plan-cap`
(3/3, `plan_md_bytes=0`, `plan_redispatches=3`).

The B5 completion-fix (commit `ed67f1d`, D-002) makes the **driver-rendered
plan.md on disk** the PLAN gate authority instead of the write-obligation-polluted
`result.success`. Two surgical `harness.py` edits:
1. `_render_plan_from_structured` renders whenever `structured_output` carries the
   fields — the `not result.success` guard is dropped (all other guards kept:
   None result, no `model_dump`, no plan dir, no-filler `_demote_heading_lines`).
2. `_after_plan_dispatch`'s redispatch/approval gate keys on `_plan_has_content`
   (disk truth) when a plan directory exists, deferring to `result.success` only
   on the no-plan-dir DEGRADE path (edge-case d preserved).

B5 measures whether, with the valid response_format plan now actually written to
disk and gated on, a run advances past PLAN toward the ≥EXECUTE floor.

## Floor-clause identity (HARD)
`_verified_execute_workspace_write` + `_normalized_ws_path` + `_bench_defect` +
the floor-loop test body (`test_three_full_runs_grade_at_or_above_the_floor`)
hash to
`cbeeb6aaa2a5a2e5910fed7abb77e1b44651c8d81715301e2c4b547509b453ce` — verified
IDENTICAL to B1/B2/B3/B4 both BEFORE and AFTER the `L6_BLOCK` bump ("B4"→"B5").
Only `L6_BLOCK` changed in `test_live_ollama.py` for this block; the four
floor-graded functions are byte-unchanged. The completion-fix touched only
`harness.py` (`_render_plan_from_structured`, `_after_plan_dispatch`) + tests +
docs — NOT the floor grading. `model_digest` unchanged (4b `2a654d98e6fb`).

## What is UNCHANGED (the disk-truth gates + ethos)
`_plan_is_approvable` / `_plan_has_content` / `DiskEvidenceApprovals` are
BYTE-UNCHANGED (AST-verified) — they read the driver-rendered file off disk, so
"a confident sentence cannot open a gate" stays INTACT. D-016 is UNTOUCHED and
stays fully in force for EXPLORE/EXECUTE (which still require real model
tool-writes). The D-002 repair-turn plumbing (`native_fc.py`, `base.py`) is
REUSED, not modified. Model stays `ollama_chat/qwen3.5:4b`. The fix only
recognizes that PLAN's deliverable moved to driver-render, so a hollow/empty
field still renders a placeholder section the unchanged gate DENIES → honest
`plan-cap`.

## Block parameters (identical to B4 except the block name + the fixed product)
Bench id/block `l6-e2e`/**B5**; n=3 (`RUNS_E2E=3`); real workers (`_one_e2e_run`),
`E2E_WALL_CLOCK_CEILING_S` per run; model `ollama_chat/qwen3.5:4b`
(digest `2a654d98e6fb`, same as B0-B4). One look, no `--force`, no interim row
reads. Failed-artifact retention (from B4) still active → `B5/artifacts/run-{n}/`.

## Pre-registered grading (the floor — byte-identical to B0-B4)
Each run graded on the same three floor clauses: (1) `reached >= EXECUTE`,
(2) `verified_write` — an EXECUTE-state dispatch's OWN written workspace path
intersects the sha256-diffed changed-file set, (3) `honest_halt`. Plus
`_bench_defect`.

**Success criterion (the floor)**: ≥1 of 3 runs reaches ≥EXECUTE with a verified
EXECUTE-state workspace write + honest halt (B0-B4 were all 0/3 at this floor).
Report all three rubric vectors regardless.

**Pre-committed honest-negative branch**: recording "floor still 0/3, wall moved"
is a VALID deliverable. A falsifiable criterion failing honestly is not a defect
(LESSONS). Both outcomes are the block's output.

**Pre-registered PARTIAL-advance shapes** (report explicitly if observed):
- A run now reaches PLAN-approval and/or EXECUTE (the B4 could-not-succeed defect
  is gone) but leaves no verified EXECUTE-state workspace write (floor clause 2
  false) — advance without floor clearance. This alone would VALIDATE the D-002
  fix even if the floor stays 0/3.
- Plans render into all 11 distributed sections and are approved, but EXECUTE
  blocks (a NEW, deeper wall — the founding e2e goal still unmet).
- All 11 sections render non-empty but the plan is present-but-THIN /
  unapprovable (Pre-Mortem #4). **Pre-approved recovery** (USER-SANCTIONED, no
  new fork): split the single response_format completion into N per-section
  response_format calls, one field per call, same repair-turn machinery, same
  renderer, same unchanged gate. Contingency only — NOT built now; triggered
  solely by this measurement.

## W3 (falsifier grading, if reached)
If any run reaches PLAN and redispatches, grade `plan_redispatches` vs
`MAX_PLAN_REDISPATCHES=3`. If a run reaches EXECUTE, note the EXECUTE-state
behavior (the tightened verified-write floor clause). A run reaching approval or
EXECUTE is the direct falsification of the B4 could-not-succeed diagnosis.

## Ethos check (post-run)
Confirm from the B5 rows (and the retained `B5/artifacts/run-{n}/plan.md`) that
any run reaching approval/EXECUTE did so on a model-authored,
all-sections-non-placeholder plan.md — the driver only FORMATS the model's 11
fields and invents no filler; an empty field renders an empty section the
unchanged `_plan_is_approvable` denies.

## Run command (one look)
```
FSM_LLM_HARNESS_LIVE=1 .venv/bin/python -m pytest \
  "tests/test_fsm_llm_harness/test_live_ollama.py::TestL6EndToEndRealWorkers::test_three_full_runs_grade_at_or_above_the_floor" -q -s
```
(after `L6_BLOCK = "B5"`).
