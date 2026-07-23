# L6 `l6-e2e/B7` — grading (one look, graded strictly against PRE_REGISTRATION_B7.md)

*Plan: plan-2026-07-23T173454-2c22e5f6, iter-1/step-6. Registration commit
`18a77e9` (also the manifest's `git_commit`) precedes every row. The block ran
EXACTLY once (`RUNS_E2E=3`, seeds `20260722100+run-1`); the frozen pytest node
`test_three_full_runs_grade_at_or_above_the_floor` FAILED on the floor assert,
which is the pre-registered honest-negative shape, not a bench defect. Every
claim below cites a committed row/observation field in `rows.jsonl` /
`artifacts/run-{n}/observations.json`. No re-run, no row was hand-corrected.*

## Row summary (read directly from `rows.jsonl`)

| run | seed | furthest_state | verified_write | halt_slug | honest_halt | workspace_files_changed | dispatches (by state) | observations retained |
|---|---|---|---|---|---|---|---|---|
| 1 | 20260722100 | explore | false | `explore-cap` | **true** | [] | 10 (explore 10) | 10 records |
| 2 | 20260722101 | reflect | **true** | `reflect-cap` | false | ["uploader.py"] | 9 (explore 3, plan 1, execute 1, reflect 4) | 9 records |
| 3 | 20260722102 | reflect | **true** | None | false | ["uploader.py"] | 8 (explore 4, plan 2, execute 1, reflect 1) | 8 records |

Floor rows (`>= EXECUTE` AND `verified_write` AND `honest_halt`): **0/3**
(Wilson 95% CI [0.000, 0.562]). `_bench_defect` never fired; Pre-Mortem #3
(frozen defect-halt: runs 1 AND 2 both crash/hang/slugless) did NOT fire —
run 1 halted honestly on `explore-cap`, run 2 carried the `reflect-cap` slug.

## (i) S5 / `verified_write` — **2/2 EXECUTE-reaching runs credited in-loop; fix proven load-bearing OFFLINE (probe record), not attributed in-loop**

*(Heading and scope qualified at step 7 per the adversarial review, concern 1;
the row citations below are unchanged.)*

Both EXECUTE-reaching runs (2 and 3) show `verified_write=true` with the exact
normalized label the D-002 fix (`roles.py` `_evidence_path`, commit `aa0ecb6`)
was built to produce:

- run 2, EXECUTE observation (obs[4]): `"state": "execute", "success": true,
  "failure_reason": null, "write_required": true,
  "write_evidence_paths": ["workspace:uploader.py"]`; row
  `workspace_files_changed=["uploader.py"]`, `verified_write=true`.
- run 3, EXECUTE observation (obs[6]): identical clean shape —
  `"write_evidence_paths": ["workspace:uploader.py"]`; row
  `workspace_files_changed=["uploader.py"]`, `verified_write=true`.

Per the pre-registered branch (i), the refute condition (an EXECUTE-reaching
run that changed workspace files yet shows `verified_write=false`) did not
occur. **Scope the claim precisely:** the retained `write_evidence_paths`
labels are POST-normalization — `_evidence_path` runs INSIDE
`_verified_writes`, upstream of the retention observer — so B7's committed
records cannot show what raw spelling the model typed, and therefore CANNOT
prove the fix was load-bearing in runs 2/3. Run 2's single clean
`workspace:uploader.py` with `write_evidence=1` is a DIFFERENT EXECUTE shape
from the probe/B6 record (`write_evidence=2`, absolute path) and is
consistent with the model simply typing a relative path unaided. The
load-bearing proof is OFFLINE: the frozen-floor replay of the retained n=1
probe record (probe-run-1) flips `verified_write` False→True under the fix
(verification.md M3). The honest statement is therefore: the S5 credit layer
no longer blocks floor credit for any in-root spelling (proven offline on the
measured probe record); B7 shows 2/2 EXECUTE-reaching runs credited in-loop —
an existence proof, not attribution of those two credits to the fix. The
failure mode B6 run 2 exhibited (bytes landed, credit lost to a raw
absolute-path label) did not recur in either EXECUTE-reaching run. This is
the first block in the B0-B7 lineage with ANY `verified_write=true` row
(B6: 0/1 EXECUTE-reaching; ad hoc per O1: verified_write 0/3 vs 2/3, Fisher
two-sided p=0.4; Wilson 95% CI for 2/3 is [0.208, 0.939]). n=2
EXECUTE-reaching runs, one goal, one configuration; a future recurrence of
the exact S5 class would be invisible in blocks that retain only the
normalized label — retaining the raw `call.parameters["path"]` alongside it
is a named successor instrumentation item.

## (ii) S4b / `honest_halt` — **PARTIALLY validated, two measured gaps**

**The targeted variant fired and halted honestly.** Run 2 hit exactly the
stall class D-003 budgeted: 4 REFLECT dispatches (obs[5..8]), ALL
`"success": false, "failure_reason": "unparseable"` — the verifier never
produced a routable verdict — and the run halted with
`halt_slug="reflect-cap"` and `halt_reason`:

> "Reflect cap: 3 extra verifier dispatch(es) spent (cap 3) and none produced
> a routable verdict (no criteria verdict, no routing flag), so no REFLECT
> edge can fire. Characterize the verifier failure before raising the budget.
> …"

The B5/B6/probe signature this fix targeted — an unroutable-verifier REFLECT
stall ending `halt_slug=None` — did not recur for that variant.

**Gap (α), a measured B7 bench-side confound (honest_halt clause only):**
run 2's row still grades `honest_halt=false` BY CONSTRUCTION. The bench's
`HONEST_HALT_SLUGS` allowlist (`test_live_ollama.py:1393-1395` — a module
constant, NOT one of the four hash-frozen floor objects) is
`{*GateSlug.ORDER, EXPLORE_CAP, PLAN_CAP}` and was never extended with
`REFLECT_CAP`, so a reflect-cap halt was ungradeable-as-honest in this block
no matter what the driver did. This is bench allowlist lag, a one-line fix in
a non-frozen constant — but whether `reflect-cap` JOINS the honest set is a
successor/REFLECT-stage decision (it was not pre-registered for B7 and is not
made here, and nothing in this block is regraded).

**Gap (β), the pre-named out-of-scope residual, now measured live:** run 3's
single REFLECT dispatch SUCCEEDED (obs[7]: `"success": true`), the verifier
graded `criteria_pass_count=4/4` and routed toward CLOSE — and the close
approval gate DENIED on disk evidence (row `approvals`:
`{"gate": "harness.confirm_close", "approved": false, "evidence":
"verification.md is absent or empty"}`; `verification_md_bytes=0`). A denied
CLOSE is not a worker failure and does not consume the reflect budget (D-003
Anchor-Refs, `harness.py:2640-2669`: "close-approval denial deliberately out
of scope"), so the run ended in a slugless stall (`halt_slug=None`,
`halt_reason` "Stalled in REFLECT for 3 turns with no progress…").

**Strict grading note:** the pre-registered branch (ii) literally says a
slugless stall after reaching REFLECT "REFUTES the S4b fix". That condition
FIRED on run 3 as written. Attribution from the retained records is
unambiguous, though: the mechanism is not the budgeted unroutable-verifier
class (which run 2 shows halting honestly) but the close-approval-denied
residual that the fix's own decision record named as out of scope BEFORE the
block ran (commit `66df5c3` precedes registration `18a77e9`). Verdict:
**PARTIALLY validated** — validated on the variant it covers, and B7 measured
live that the S4b budget's coverage boundary is real: the denied-CLOSE loop
still stalls sluglessly. That residual is now a measured successor item, not
a hypothesis.

## (iii) Floor — **0/3; the pre-committed honest-negative branch fires**

No run satisfies all three frozen clauses:

- run 1: never reached EXECUTE — an honest `explore-cap` halt after 10
  explore dispatches with `findings_nonempty=2` of 3 (S4c explore variance,
  same class as B6 runs 1/3; `honest_halt=true` but the other clauses false).
- run 2: `>=EXECUTE` true, `verified_write` true, `honest_halt` false — gap (α).
- run 3: `>=EXECUTE` true, `verified_write` true, `honest_halt` false — gap (β).

Per the pre-registration, the deliverables for this branch are the per-clause
validations above plus the retained raw observations — committed in full:
each row carries its `observations` list (10/9/8 records) and each
`artifacts/run-{n}/` holds `observations.json` (+ `state.md`, and `plan.md`
where written). Both honest_halt losses are diagnosable from the committed
block alone; no re-run is needed — the retention instrumentation did the job
it was added for.

## New positive fact (named, no ceiling claim)

Run 3 went functionally deeper than any prior lineage run: EXPLORE → PLAN
(approved on disk evidence, 2853 bytes, 11 sections) → EXECUTE
(verified write) → REFLECT (verifier SUCCEEDED, 4/4 criteria) → routed toward
CLOSE — and the close gate held: it denied on disk truth (`verification.md`
absent/empty), exactly the honesty-gate design (the model cannot close with a
sentence; only bytes on disk open gates). The run's failure point is now the
protocol's own evidence requirement, not the model's inability to traverse.

## What this block does NOT show

No capability-ceiling claims, per the pre-registration: neither "4b can drive
the harness" (floor did not clear) nor "4b cannot" (n=3, one configuration,
one goal; 2/3 EXECUTE-reach vs the ~1/3 historical rate is Fisher p=1.0 at
this n). The S5 validation is n=2. Nothing here regrades B0-B6.

## Measured successor items

1. (α) `HONEST_HALT_SLUGS` + `REFLECT_CAP` bench allowlist decision (one-line,
   non-frozen constant; requires its own decision on whether a reflect-cap
   halt is honest-by-design like explore-cap/plan-cap).
2. (β) the close-approval-denied slugless stall (run 3): the denied-CLOSE loop
   needs either budget coverage or its own honest slug — pre-named in D-003,
   now measured live for the first time.
3. S4c explore variance (run 1) remains the top-of-funnel loss, unchanged
   since B5/B6 — not addressed by this plan and not newly measured here.
