# L6 `l6-e2e/B3` — pre-registration (PLAN scaffold + honest approval, e2e)

*Plan: plan-2026-07-23T095051-a6dcb40d (harness iteration 6).*
*Committed BEFORE the B3 live run (git log: this file precedes the `B3/` rows
commit). One-look / no-`--force` / no-re-sample (D-002).*

## What changed since B2 (the thing under test)
Iteration-6 fix (D-001, commits `32f71a3`+`f296f4b`): the driver seeds `plan.md`
with the 11 `PlanSchema.SECTIONS` headers (empty bodies, STRUCTURE ONLY) at PLAN
entry; the plan-writer FILLS each section by appending (`force_final_tool=
append_plan_file` for the PLAN role + a deliverable-line "the scaffold exists,
append under each header" instruction); `_plan_has_content` and the approval gate
now share ONE bar — `_plan_is_approvable` = valid `PlanDoc` AND every section
NON-placeholder. This targets B2's PLAN wall (B2: 3/3 reached PLAN but 0/3 past
it — the 4b plan-writer could not emit a valid/complete plan.md; run 1
non-empty-invalid→denied→slugless stall, runs 2/3 empty→plan-cap).

## DECLARED CONFOUND (accepted by the user, F4)
B3 tightens `DiskEvidenceApprovals` (the e2e approval stub) to DENY a
placeholder/empty plan, not just a structurally-invalid one — so an empty
driver-seeded scaffold does NOT clear the gate (the gate opens only on
model-authored, all-sections-filled content; "a confident sentence cannot open a
gate" INTACT). This is a change to the APPROVAL stub between B2 and B3, NOT to the
L6 FLOOR grading clauses (those stay byte-identical, hash verified below). The
confound is HONEST strengthening: B2's empty/invalid plans FAIL under the tighter
approval too, so B2's 0/3 stands; B3 measures whether the FIXED product produces a
complete, approvable plan and advances. This is disclosed, not hidden.

## Floor-clause identity (HARD)
`_verified_execute_workspace_write` + `_normalized_ws_path` + `_bench_defect` +
the floor-loop test body hash to
`cbeeb6aaa2a5a2e5910fed7abb77e1b44651c8d81715301e2c4b547509b453ce` — verified
IDENTICAL to B1/B2 before AND after the `L6_BLOCK` bump. B3's manifest
`prompt_bytes_sha256` legitimately differs (the fix changes the PLAN dispatch);
`model_digest` unchanged (4b `2a654d98e6fb`).

## Block parameters (identical to B2 except the block name + the fixed product + the tighter approval)
Bench id/block `l6-e2e`/**B3**; n=3 (`RUNS_E2E=3`); real workers (`_one_e2e_run`),
`E2E_WALL_CLOCK_CEILING_S` per run; model `ollama_chat/qwen3.5:4b`.

## Pre-registered grading (the floor, transcribed from B1/B2 — UNCHANGED, byte-identical)
Each run graded on the same three floor clauses: (1) `reached >= EXECUTE`,
(2) `verified_write` (an EXECUTE-state dispatch's own write path intersects the
sha256-diffed changed-file set), (3) `honest_halt`. Plus `_bench_defect`.

**Success criterion for this iteration**: ≥1 of 3 runs reaches ≥EXECUTE with a
verified EXECUTE-state workspace write + honest halt (B0/B1/B2 were all 0/3 at
this floor). Report all three rubric vectors regardless. This is the founding
end-goal's first honest test with the PLAN wall addressed.

## W3 (falsifier grading, if reached)
If any run reaches PLAN and redispatches, grade the `plan_redispatches` field vs
`MAX_PLAN_REDISPATCHES=3`. If a run reaches EXECUTE, note the EXECUTE-state
behavior (the tightened verified-write floor clause).

## Ethos check (post-run)
Confirm from the B3 rows that any run reaching EXECUTE did so on a
FILLED (all-sections-non-placeholder) model-authored plan.md, NOT an empty
scaffold (the honest approval enforces this by construction; verify).

## Run command (one look)
```
FSM_LLM_HARNESS_LIVE=1 .venv/bin/python -m pytest \
  "tests/test_fsm_llm_harness/test_live_ollama.py::TestL6EndToEndRealWorkers::test_three_full_runs_grade_at_or_above_the_floor" -q -s
```
(after `L6_BLOCK = "B3"`).
