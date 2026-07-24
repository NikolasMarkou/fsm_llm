# L6 `l6-e2e/B8` — pre-registration (β close-cap budget + α honest-set grading extension)

*Plan: plan-2026-07-24T032539-032ae337 (harness iteration 10 / residuals α + β).*
*Committed BEFORE the B8 live run (git log: this file precedes any `B8/` rows
commit). One-look / no-`--force` / no-re-sample (established one-look
discipline). The decision rule below is fixed BEFORE any row exists.*

## What changed since B7 (the things under test)

Exactly TWO changes land between B7 and B8: one driver-side behavior change
(β) and one declared bench-side GRADING-SEMANTICS change (α). Both touch the
**`honest_halt`** clause, so the attribution rule is stated precisely up
front (see the end of this section) rather than claimed disjoint-by-clause.

1. **β — bounded close-cap budget for the denied-CLOSE approval loop**
   (commit `8ff8ecf`, D-001 — targets the `honest_halt` clause's MECHANISM:
   it changes what halt a denied-close run PRODUCES). B7 run 3 reached
   REFLECT, the verifier claimed 4/4 criteria PASS, the CLOSE approval was
   DENIED ("verification.md is absent or empty"), and the run slugless-stalled
   (`halt_slug=null`, `honest_halt=false`): the denied branch returned
   unconditionally from `_after_reflect_dispatch`, bypassing every budget.
   The fix mints `GateSlug.CLOSE_CAP` (`"close-cap"`, excluded from
   `GateSlug.ORDER` like PLAN_CAP/REFLECT_CAP) with a dedicated
   `Defaults.MAX_CLOSE_DENIALS = 3` budget and a `self._close_denials` driver
   run-state counter (declared in `__init__`, reset only in `_run_once`,
   structurally unreachable from approval callbacks — D-029 pattern),
   mirroring the plan-cap machinery: each denial re-dispatches the verifier
   (reopening the SAME REFLECT `_dispatch_key` ledger entry — the denial
   evidence is exactly what a re-dispatched verifier, holding the write tool
   for `verification.md`, can genuinely repair); at the cap the driver
   pre-writes the honest `close-cap` slug + halt reason BEFORE the stall
   detector (which hard-codes `slug=None`) can fire. B7 run 3's slugless
   stall shape can no longer occur on this path. The `REFLECT -> CLOSE` HARD
   gate (`close_confirmed AND all_criteria_pass`) and the disk-truth approval
   predicates are unchanged; no flag is fabricated at cap exhaustion.
2. **α — GRADING-SEMANTICS change, declared**: `HONEST_HALT_SLUGS` extended
   (commit `2e8a4a6`, D-001, pre-authorized by name in B7's D-005 pin —
   targets how the `honest_halt` clause GRADES a cap halt). The bench-side,
   NON-frozen allowlist `HONEST_HALT_SLUGS` gains `GateSlug.REFLECT_CAP` and
   `GateSlug.CLOSE_CAP`. B7 measured gap α precisely because the S4b
   reflect-cap budget shipped an honest-BY-DESIGN slug the bench could not
   yet grade as honest (B7 run 2: honest reflect-cap halt,
   `honest_halt=false` by construction); D-005 pinned that exclusion as an
   explicit recorded decision whose extension "must ride its own
   pre-registered block (B8)". This is that block. The extension applies to
   **B8 rows ONLY** — B0-B7 rows are never retroactively regraded; B7 run
   2's `honest_halt=false` stands as measured under B7's registration.
   `CLOSE_CAP` joins in the same edit per the two-file rule (slug
   declaration AND allowlist membership land together, the PLAN_CAP
   precedent).

**Attribution rule (both items touch `honest_halt`)**: α changes how a
reflect-cap/close-cap halt GRADES; β changes what halt a denied-close run
PRODUCES. A run that halts `close-cap` and grades honest exercises BOTH. A
slugless stall (`halt_slug=null` ∧ `close_reached=false`) REFUTES β
regardless of α — no allowlist membership can make a slugless stall grade
honest (`None` is not in the set, pinned by test).

## Floor-clause identity (HARD)

`_verified_execute_workspace_write` + `_normalized_ws_path` + `_bench_defect` +
the floor-loop test body (`test_three_full_runs_grade_at_or_above_the_floor`)
hash to
`cbeeb6aaa2a5a2e5910fed7abb77e1b44651c8d81715301e2c4b547509b453ce` — verified
IDENTICAL to B1-B7 both BEFORE and AFTER the step-3 α allowlist edit, and
verified again BEFORE and AFTER the `L6_BLOCK` bump ("B7"→"B8") in this
registration commit (the `inspect.getsource` sha256 one-liner, outputs
recorded in the plan's verification.md). `HONEST_HALT_SLUGS` and `L6_BLOCK`
are module constants OUTSIDE the frozen four objects — editing them is a
legitimate non-floor edit, NOT a change to the frozen bar itself. The α
extension changes what the (non-frozen) allowlist CONTAINS; the frozen
`honest_halt` formula, the `verified_write` predicate, the defect rule, and
the 3/3 floor-loop body are byte-unchanged. The β fix touched only
`constants.py`/`harness.py` (driver-side) — NOT the floor grading.

## What is UNCHANGED (the disk-truth gates + ethos)

Model `ollama_chat/qwen3.5:4b` (digest
`2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`, tag
`qwen3.5:4b` — same as B0-B7 per `B7/manifest.json`). Same GOAL, same worker
factory, same `DiskEvidenceApprovals` (the CLOSE denial predicate
"`verification.md` is absent or empty" is byte-unchanged), same seed schedule
(`E2E_SEED_BASE=20260722100`, per-row `base+run-1`, native arm — B8 seeds
derive exactly as B7's did). `derive_execute_target` + existence-gated prose
fallback, `_plan_is_approvable`, `_plan_has_content`, the S5
`_evidence_path` label normalization, the forced-write EXPLORE fix, the
`response_format` structured plan, and D-016/D-010 all byte-unchanged since
B7. `Defaults.MAX_PLAN_REDISPATCHES = 3` and
`Defaults.MAX_REFLECT_REDISPATCHES = 3` unchanged (the new
`MAX_CLOSE_DENIALS = 3` is β itself, declared above). Raw per-dispatch
observation retention stays ON (`artifacts/run-{n}/observations.json` per
run). The model authors all content; the driver assigns targets, renders the
structured plan, and spells evidence labels from resolved paths — it invents
no prose and cannot open a gate with a sentence.

## Block parameters

Bench id/block `l6-e2e`/**B8**; n=3 (`RUNS_E2E=3`); real workers
(`_one_e2e_run`), `E2E_WALL_CLOCK_CEILING_S=1800.0` per run. One look, no
`--force`, no interim row reads; the one-look guard refuses an existing
`B8/rows.jsonl`: "an L6 block runs ONCE (D-002); a new question needs a new
pre-registered block and decision entry". Failed-artifact retention active →
`B8/artifacts/run-{n}/`, including `observations.json` per run.

## Pre-registered grading (decision rule, fixed before any row)

Each run graded on the same three frozen floor clauses: (1) reached
`>= EXECUTE` by `E2E_STATE_RANK`, (2) `verified_write` — an EXECUTE-state
dispatch's OWN written workspace path (via
`_verified_execute_workspace_write`) intersects the sha256-diffed
changed-file set, (3) `honest_halt` = `error is None` ∧ `not timed_out` ∧
(`close_reached` ∨ `halt_slug ∈ HONEST_HALT_SLUGS`). Bar: ALL 3 runs must
clear all 3 clauses (`floor_misses == []`). Plus `_bench_defect`.

**Per-clause attribution branches** (each graded by its own refute
condition):

- **(i) β / slugless-stall class**: β is **REFUTED** if any B8 row shows
  `halt_slug=null` ∧ `close_reached=false` (a slugless stall — regardless of
  α, per the attribution rule above). β's mechanism is
  **VALIDATED-in-loop** if a `close-cap` halt row shows ≥1 denied
  `harness.confirm_close` approval in its `approvals` field and `1 + cap`
  REFLECT dispatches in its retained `observations` (attribution to the
  denial-redispatch loop, not some other path).
- **(ii) α / cap-halt grading**: α is **exercised** iff some row halts
  `reflect-cap` or `close-cap`. For any such row, `honest_halt` MUST grade
  `True` — if it grades `False`, the allowlist wiring is defective (the
  exact B7 gap α, reproduced) and that is the finding.
- **(iii) The floor (founding criterion)**: 3/3 rows clearing all three
  clauses per the frozen body. B0-B7 were all 0/3 at this floor.

**Per-clause × per-shape passability** (fixed before any row; post-α+β no
clause is unpassable by construction for any expected shape — the voiding
condition "some clause is unpassable for every expected shape" does NOT
hold):

| Shape | reached>=EXECUTE | verified_write | honest_halt |
|---|---|---|---|
| (a) explore-cap halt | impossible (furthest=explore, rank 1 < 3) — honest S4c variance, not a defect | impossible (no EXECUTE observation exists) | passable (`explore-cap` ∈ set) |
| (b) reflect-cap halt | passable (EXECUTE en route) | passable | passable — POST-α (`reflect-cap` now ∈ set; B7 run 2's shape would now grade honest) |
| (c) close-cap after denials | passable | passable | passable — POST-β (the denied-close path now PRODUCES `close-cap`) + POST-α (the set now GRADES it honest) |
| (d) full traverse to CLOSE | passable (CLOSE ranks 5) | passable | passable (`close_reached` disjunct, no slug needed) |

Shape (a) fails clauses 1-2 honestly (explore-cap variance) while grading
`honest_halt=True`; shapes (b)/(c)/(d) can pass all three. Unlike B7 (where
shape (c) was structurally unpassable on `honest_halt`), B8 is a
configuration in which every clause is passable for every
EXECUTE-reaching shape.

## Pre-committed HONEST-NEGATIVE branch

The floor may stay **0/3** (or land 1/3, 2/3) with ALL halts honest, without
either change being refuted — S4c explore-cap variance has kept ~1/3-2/3 of
runs per block from reaching EXECUTE across B5-B7. That outcome is a
pre-blessed deliverable, not a failure: the block's deliverables are the
per-clause graded result (branches i-iii against their OWN refute
conditions), ZERO slugless stalls, and the retained raw observations for
every dispatch of every run. It will be reported as measured.

## Pre-Mortem #3 defect-halt branch (frozen body)

If runs 1 AND 2 both end in a crash/hang/SLUGLESS stall, the block halts
early without spending run 3 (bench-defect rule `_bench_defect`,
byte-identical since B1: `error is not None or timed_out or
(not close_reached and halt_slug is None)`). NOTE, pre-registered: post-α+β
a slugless stall REFUTES β — this branch FIRING via two slugless stalls
would itself be evidence AGAINST the β fix, and will be reported as such.

## No capability-ceiling claims

No capability-ceiling claims will be made from this block regardless of
outcome — in particular, neither "4b can drive the harness" (if the floor
clears) nor "4b cannot" (if it stays 0/3). n=3, one configuration, one goal.

## Run command (one look)

```
FSM_LLM_HARNESS_LIVE=1 .venv/bin/python -m pytest \
  "tests/test_fsm_llm_harness/test_live_ollama.py::TestL6EndToEndRealWorkers::test_three_full_runs_grade_at_or_above_the_floor" -q -s
```
(after `L6_BLOCK = "B8"`; same pytest node id as B7's run command).
