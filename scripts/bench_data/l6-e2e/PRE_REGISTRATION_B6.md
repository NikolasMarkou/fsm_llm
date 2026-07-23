# L6 `l6-e2e/B6` — pre-registration (S4a: existence-gated prose EXECUTE target)

*Plan: plan-2026-07-23T155204-fdc2d181 (harness iteration 8 / successor S4a).*
*Committed BEFORE the B6 live run (git log: this file precedes the `B6/` rows commit). One-look / no-`--force` / no-re-sample (D-002/D-003 one-look discipline).*

## What changed since B5 (the thing under test)
B5 run-1 was the FIRST run in package history to clear PLAN and reach REFLECT on
a VALID 11-section plan.md, but missed the founding floor's `verified_write`
clause (S4a): `execute_assigned_targets=[null]`, reason `no-target-token`, yet the
workspace changed (`config.py`, `uploader.py`) — `verified_write=false`.

Root cause MEASURED this iteration (findings F1-F7, plan-2026-07-23T155204-fdc2d181):
1. `derive_execute_target` (harness.py) harvests ONLY backticked path tokens
   (`_TICKED_RE`). The real 4b plan writes `uploader.py`/`config.py` as PROSE, so
   it returns `None` → `no-target-token` → the executor prompt drops the
   "WRITE IT TO: <path>" line → the model writes the workspace on its own → its
   raw-path write label fails to normalize into the sha256-changed set → the
   HASH-FROZEN floor predicate scores `verified_write=false` (the L4-B0 regime).
2. **F4 (live, n=3/arm)**: assigning target `uploader.py` flips
   `_verified_execute_workspace_write` **1/3 (false-positive) → 3/3 clean** on live
   `qwen3.5:4b` — the L4-B1 driver-assigned-target mechanism (2/40→40/40) transfers
   to a real retained plan.md.
3. **F5 (live, n=5/arm, + independent re-run)**: amending the response_format
   "Files To Modify" FIELD DESCRIPTION to instruct backticking is a NO-OP over
   Ollama (0/5 both arms, byte-identical output) — litellm forwards the schema
   (descriptions included) as Ollama's `format` = a grammar constraint, NOT
   model-visible prompt text. So model-steering the plan to backtick is not viable.
4. **F7 (offline, deterministic)**: an existence-gated prose parser extracts
   `uploader.py` from the REAL B5 run-1 plan.md and rejects `requests.post` even
   when it appears first (order-independent via the workspace-existence gate).

**The B6 fix (commits `1228c96` + `8ce53e8`, D-001)**: an ADDITIVE, deterministic
prose fallback. `_assign_execute_target` calls a new pure
`_derive_prose_target(plan, step, existing_files)` ONLY when the strict
`derive_execute_target` returns `None`; it keeps path-shaped, extension-bearing
prose tokens that name a file EXISTING under the workspace root (the gate makes
selection order-independent and forbids pointing the executor at a non-file), and
reports the diagnostic reason `EXECUTE_TARGET_ASSIGNED_PROSE`.
`derive_execute_target` is BYTE-UNCHANGED (D-010 + its tests preserved). A plan
naming no real file yields no assignment (fail-open, byte-identical prompt).

B6 measures whether, with the EXECUTE target now assigned from the real prose
plan, a run that reaches EXECUTE leaves a floor-crediting verified workspace write.

## Floor-clause identity (HARD)
`_verified_execute_workspace_write` + `_normalized_ws_path` + `_bench_defect` +
the floor-loop test body (`test_three_full_runs_grade_at_or_above_the_floor`)
hash to
`cbeeb6aaa2a5a2e5910fed7abb77e1b44651c8d81715301e2c4b547509b453ce` — verified
IDENTICAL to B1/B2/B3/B4/B5 both BEFORE and AFTER the `L6_BLOCK` bump ("B5"→"B6").
Only `L6_BLOCK` changed in `test_live_ollama.py` for this block; the four
floor-graded functions are byte-unchanged. The fix touched only `harness.py`
(additive) + `test_harness_agent.py` (new tests) — NOT the floor grading.
`model_digest` unchanged (4b `2a654d98e6fb`).

## What is UNCHANGED (the disk-truth gates + ethos)
`derive_execute_target` / `_plan_is_approvable` / `_plan_has_content` /
`DiskEvidenceApprovals` are BYTE-UNCHANGED. D-016 is UNTOUCHED and stays fully in
force for EXPLORE/EXECUTE. The model authors the plan AND names the files; the
driver only assigns an EXISTING-file target and names it in the prompt — it invents
no filler and cannot open a gate with a sentence. Model stays
`ollama_chat/qwen3.5:4b`.

## Block parameters (identical to B5 except the block name + the fixed product)
Bench id/block `l6-e2e`/**B6**; n=3 (`RUNS_E2E=3`); real workers (`_one_e2e_run`),
`E2E_WALL_CLOCK_CEILING_S` per run; model `ollama_chat/qwen3.5:4b`
(digest `2a654d98e6fb`, same as B0-B5). One look, no `--force`, no interim row
reads. Failed-artifact retention active → `B6/artifacts/run-{n}/`.

## Pre-registered grading (the floor — byte-identical to B0-B5)
Each run graded on the same three floor clauses: (1) `reached >= EXECUTE`,
(2) `verified_write` — an EXECUTE-state dispatch's OWN written workspace path
intersects the sha256-diffed changed-file set, (3) `honest_halt`. Plus
`_bench_defect`.

**Success criterion (the floor)**: >= 1 of 3 runs reaches >= EXECUTE with a verified
EXECUTE-state workspace write + honest halt (B0-B5 were all 0/3 at this floor).

**PRIMARY validation target (this iteration's charter, S4a)**: >= 1 run reaches
>= EXECUTE with **`verified_write=True`**. This VALIDATES the S4a fix even if the
founding floor stays 0/3, because a run reaching EXECUTE then advances to REFLECT
and (S4b, a residual DEFERRED per D-003) slugless-stalls there → `honest_halt=False`.
Report all three rubric vectors regardless.

**Pre-committed honest-negative branch**: recording "S4a validated (a run reached
EXECUTE with verified_write=True) but floor still 0/3 because honest_halt fails at
the S4b slugless REFLECT stall" is a VALID deliverable. A falsifiable criterion
failing honestly, with the wall moved from verified_write to honest_halt, is not a
defect (LESSONS). Both outcomes are the block's output.

**Pre-registered shapes** (report explicitly if observed):
- A run reaches EXECUTE with `execute_assigned_targets=['uploader.py']` (reason
  `assigned-prose`) and `verified_write=True` → the DIRECT falsification of B5's
  S4a `no-target-token`/`verified_write=false` result. **This is the aimed outcome.**
- A run reaches EXECUTE with an assigned target but `verified_write=False` → the
  F4 3/3 measurement did not transfer to the full e2e loop; report and diagnose
  (Pre-Mortem #2).
- 0/3 runs reach EXECUTE (S4c: single-call PLAN approval variance, as B5 run 2/3)
  → the S4a fix could not be observed this block; NOT a refutation. Report and
  note S4c (per-section multi-call fallback) as the standing successor.
- Pre-Mortem #3 (`_bench_defect`) early-halt if runs 1 AND 2 both end in a slugless
  REFLECT stall (S4b) → the block reads defect-shaped though the verified_write=True
  rows are still committed. Mitigated by S4c honest cap-halts (as in B5).

## W3 (falsifier grading, if reached)
If a run reaches EXECUTE, record the assigned target, its reason
(`assigned-prose` vs `assigned`), and the tightened verified-write floor clause.
A run reaching EXECUTE with `verified_write=True` off a prose-derived target is
the direct falsification of B5's S4a diagnosis.

## Ethos check (post-run)
Confirm from the B6 rows (and retained `B6/artifacts/run-{n}/plan.md`) that any
run reaching EXECUTE did so on a model-authored plan whose Files-To-Modify NAMED
`uploader.py` in prose — the driver only SELECTED an existing-file token and named
it; it invented no target. An empty/no-real-file section yields no assignment.

## Run command (one look)
```
FSM_LLM_HARNESS_LIVE=1 .venv/bin/python -m pytest \
  "tests/test_fsm_llm_harness/test_live_ollama.py::TestL6EndToEndRealWorkers::test_three_full_runs_grade_at_or_above_the_floor" -q -s
```
(after `L6_BLOCK = "B6"`).
