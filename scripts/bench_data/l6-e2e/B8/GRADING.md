# L6 `l6-e2e/B8` — grading (one look, graded strictly against PRE_REGISTRATION_B8.md)

*Plan: plan-2026-07-24T032539-032ae337, iter-1/step-6. Registration commit
`d49409e` (also the manifest's `git_commit`) precedes every row (rows commit
`b678257`). The block ran EXACTLY once (`RUNS_E2E=3`, seeds `20260722100+run-1`);
the frozen pytest node `test_three_full_runs_grade_at_or_above_the_floor` FAILED
on the 3/3 floor assert (exit 1), which is the pre-registered honest-negative
shape, not a bench defect. Every claim below cites a committed row/observation
field in `rows.jsonl` / `artifacts/run-{n}/observations.json`. No re-run, no row
was hand-corrected. Floor hash `cbeeb6aa…` verified identical before and after
every bench-file edit (registration "Floor-clause identity").*

## Row summary (read directly from `rows.jsonl`)

| run | seed | furthest_state | verified_write | halt_slug | honest_halt | workspace_files_changed | dispatches (by state) | observations retained |
|---|---|---|---|---|---|---|---|---|
| 1 | 20260722100 | reflect | **true** | `close-cap` | **true** | ["uploader.py"] | 9 (explore 3, plan 1, execute 1, reflect 4) | 9 records |
| 2 | 20260722101 | plan | false | `plan-cap` | **true** | [] | 7 (explore 3, plan 4) | 7 records |
| 3 | 20260722102 | reflect | **true** | `reflect-cap` | **true** | ["uploader.py"] | 9 (explore 3, plan 1, execute 1, reflect 4) | 9 records |

Floor rows (`>= EXECUTE` AND `verified_write` AND `honest_halt`): **2/3**
(Wilson 95% CI [0.208, 0.939]) — the frozen 3/3 bar is **NOT MET**.
`_bench_defect` never fired; Pre-Mortem #3 (frozen defect-halt: runs 1 AND 2
both crash/hang/slugless) did NOT fire — no timeout, no error, and **zero
slugless stalls** in the block (no row with `halt_slug=null` ∧
`close_reached=false`).

## (i) β / slugless-stall class — **not refuted; mechanism VALIDATED-in-loop (run 1)**

**Refute condition (registration): any row with `halt_slug=null` ∧
`close_reached=false`.** It did NOT fire — halt slugs are
`close-cap`/`plan-cap`/`reflect-cap`, all non-null. The B7 run 3 shape (denied
CLOSE → slugless stall) can no longer be observed on this configuration and was
not.

**Validation condition (registration): a `close-cap` row with ≥1 denied
`harness.confirm_close` and `1 + cap` REFLECT dispatches.** Run 1 is exactly
that row:

- `approvals`: 1 granted `harness.approve_plan` ("plan.md carries 2673 bytes …
  all 11 sections filled") + **4 denied `harness.confirm_close`**, each with
  evidence "verification.md is absent or empty" (`verification_md_bytes=0`).
- observations: obs[5] REFLECT `"success": true` — the verifier verdict routed
  toward CLOSE (row `criteria_pass_count=1/1`; note: B7 run 3's verdict was
  4/4 — the criteria count is verdict-dependent, the routing shape is the
  same); obs[6..8] are the three budget-funded verifier redispatches, ALL
  `"failure_reason": "empty-reply"`.
- `dispatch_counts.reflect=4` = 1 + cap (`MAX_CLOSE_DENIALS=3`), and
  `halt_reason` opens "Close cap: 3 extra verifier dispatch(es) spent (cap 3)
  and the human CLOSE approval is still denied on an all-criteria-pass
  verdict…" — the honest pre-written slug, not the stall detector.

State precisely what is and is NOT shown. β is validated as **bounding the
denied-CLOSE loop and halting it honestly**: each denial re-dispatched the
verifier, the budget spent to exactly the cap, and the halt carries the
`close-cap` slug with the denial evidence in the reason. β does **not** make
the verifier repair the denial: all three redispatched verifiers empty-replied
and `verification.md` stayed at 0 bytes, so the retry loop's "genuinely
productive" premise (the verifier holds the write tool for verification.md) was
funded but never used by the model — a named residual (successor s1 below).
The `REFLECT -> CLOSE` HARD gate held throughout: 4 denials on disk truth, no
flag fabricated at cap (row `close_reached=false`, `success=false`).

## (ii) α / cap-halt grading — **exercised twice, graded honest both times**

**Condition (registration): α is exercised iff some row halts `reflect-cap` or
`close-cap`; any such row MUST grade `honest_halt=true`, else the allowlist
wiring is defective.** Exercised twice: run 1 (`close-cap`) and run 3
(`reflect-cap`) — **both grade `honest_halt=true`** under the B8-declared
allowlist. The exact B7 gap α (an honest cap halt grading `false` BY
CONSTRUCTION) did not reproduce. Run 3 is shape (b) of the passability table —
the same shape as B7 run 2 (4/4 REFLECT dispatches all
`"failure_reason": "unparseable"`, obs[5..8], reflect-cap halt) which B7 graded
`honest_halt=false`; under B8's registration the same halt shape grades honest.
Per the declared scope, the extension applies to **B8 rows only**: no B0-B7 row
was touched or regraded; B7 run 2's `honest_halt=false` stands as measured
under B7's registration. `HONEST_HALT_SLUGS` is a non-frozen module constant;
the frozen floor formula that CONSUMES it hashed identical (`cbeeb6aa…`).

**Attribution rule, applied as registered**: run 1 exercises BOTH α and β (β
PRODUCED the `close-cap` halt; α GRADED it honest). Run 3 exercises α only on
the grading side — its `reflect-cap` halt was produced by the pre-existing B7
reflect budget (plan-2c22e5f6 D-003), not by anything that changed in B8.

## (iii) Floor (founding criterion) — **2/3; 3/3 bar NOT MET; the pre-committed honest-negative branch fires**

The bar requires ALL 3 runs to clear all 3 frozen clauses; run 2 misses two:

- run 1: `>=EXECUTE` true (furthest=reflect), `verified_write` true
  (obs[4] EXECUTE `"write_evidence_paths": ["workspace:uploader.py"]`,
  `workspace_files_changed=["uploader.py"]`), `honest_halt` true (`close-cap`)
  — **clears all three**.
- run 2: never left PLAN — all 4 plan-writer dispatches
  `"failure_reason": "empty-reply"` (obs[3..6]), `plan_md_bytes=0`,
  `plan_redispatches=3=MAX_PLAN_REDISPATCHES`, honest `plan-cap` halt.
  `reached>=EXECUTE` false, `verified_write` false, `honest_halt` true.
- run 3: `>=EXECUTE` true, `verified_write` true (obs[6] EXECUTE
  `"write_evidence_paths": ["workspace:uploader.py"]`), `honest_halt` true
  (`reflect-cap`) — **clears all three**.

Historic per-run fact, recorded precisely: **runs 1 and 3 are the FIRST rows in
the B0-B8 lineage to clear the full per-run floor conjunction**
(`reached>=EXECUTE` ∧ `verified_write` ∧ `honest_halt`). Every prior block was
0/3 at that conjunction (B7's two EXECUTE-reaching runs each lost `honest_halt`
to gap α/β). The block-level 3/3 bar still FAILS honestly on run 2 — a
top-of-funnel variance loss, S4c-adjacent but at PLAN rather than EXPLORE (the
plan-writer, not the explorer, empty-replied out its budget).

**Seed observation (observation only, no causal claim)**: run 2's per-row seed
`20260722101` is the same seed as B7 run 2, which did NOT plan-cap — it wrote
an approvable plan and reached REFLECT (`reflect-cap`, `verified_write=true`,
per `B7/GRADING.md`'s row table). So there is no seed-level plan-cap pattern;
the same seed produced divergent trajectories across blocks (B8 run 2's
divergence is visible as early as EXPLORE — obs[0] carries a doubled
`plan:findings/problem-scope.md` write-evidence entry, unlike B7's row). The
per-row seed does not pin the trajectory.

## What this block does NOT show

No capability-ceiling claims, per the registration: neither "4b can drive the
harness" (the 3/3 bar did not clear) nor "4b cannot" (n=3, one configuration,
one goal). 2/3 at the per-run conjunction is not attribution of those two
clears to α or β individually beyond the registered attribution rule: run 1
exercises BOTH (β produced its halt, α graded it); run 3 exercises α only on
the grading side. Neither run proves the model can produce verification.md
CONTENT — `verification_md_bytes=0` in all three rows (run 1's redispatched
verifiers empty-replied 3x; run 3's unparseable 4x). Nothing here regrades
B0-B7. The `MAX_CLOSE_DENIALS=3` value is β's declared unmeasured placeholder,
not a tuned constant.

## Measured successor items

1. **(s1) The verifier never writes `verification.md` — now the dominant
   measured wall for full traverse AND for the 3/3 floor.** Both deep runs
   died on verifier-side content: run 1's close-denial loop funded 3 verifier
   redispatches and all empty-replied (obs[6..8]); run 3's verifier was
   unparseable 4/4 (obs[5..8]). The denial evidence names exactly the file the
   verifier holds the write tool for, and across 7 funded REFLECT dispatches
   in this block the model never wrote it (`verification_md_bytes=0`
   everywhere). This is a verifier-content/conformance problem, the same
   failure class the PLAN wall had before the `response_format` structured
   plan — that mechanism is the obvious candidate and is NOT chosen here.
2. **(s2) PLAN-side variance** (run 2: honest `plan-cap`, empty-reply x4,
   `plan_md_bytes=0`) — the top-of-funnel loss class persists (S4c-adjacent,
   here at PLAN); ~1 run per block still dies before EXECUTE, B5-B8.
3. **(s3) Budget values are unmeasured placeholders**:
   `MAX_REFLECT_REDISPATCHES=3` and `MAX_CLOSE_DENIALS=3` both fired at their
   caps in this block, but no evidence exists that a larger budget would
   convert either run — B8's observations (empty-reply/unparseable at every
   funded retry) suggest the verifier, not the budget size, is the lever.
   Tuning requires its own pre-registered bench.

## Step-7 amendment (post-review, D-002) — wording/attribution corrections; graded verdicts UNCHANGED

*Appended at plan-032ae337 iter-1/step-7 after the adversarial review
(`findings/review-iter-1.md`). Per the B7 D-005 precedent this is a MARKED
amendment: nothing above this section is rewritten, no row is regraded, and
the graded verdicts stand unchanged — β validated as BOUNDING the denied-CLOSE
loop and halting it honestly, α exercised twice and grading honest, floor 2/3
with the 3/3 bar honestly NOT MET, zero slugless stalls. What follows corrects
two attributions the registration and section (i)/s1 carried.*

**W1 — the registration's "genuinely productive redispatch" premise is FALSE
on this configuration.** The registration (and D-001) justified
redispatch-over-immediate-halt on the claim that the verifier "holds the write
tool for verification.md" and can repair the denial. It does not: the REFLECT
role receives `READ_ONLY_TOOLS + SHELL_TOOLS` only (roles.py:322; the B8
manifest's reflect tool surface has no write tool), VERIFICATION is owned by
PLAN_WRITER/ORCHESTRATOR (rules.py:122), and no driver code merges the
verifier reply into verification.md. Dispositive in-block proof: run 1 obs[5]
was a PARSEABLE `success=true` verdict, yet `verification_md_bytes=0` and the
first `confirm_close` was still denied "verification.md is absent or empty."
The close-denial redispatch loop is therefore futile-for-repair by
construction here; what B8 run 1 validated is the bounded HONEST `close-cap`
halt, not repair. D-001 stands as written (append-only); D-002 in the plan's
decisions.md is the correcting record.

**s1 reframed (W2) — the dominant wall is a HARNESS PLUMBING gap, not
"verifier content".** Section s1 above aims the successor at a verifier-side
`response_format` fix; that cannot work alone, because REFLECT has no write
path to `verification.md` at all (neither a verifier tool nor a driver
merge). The named successor options are driver-side: (1) a driver merge of a
structured verifier reply into `verification.md`, or (2) seeding
`verification.md` at the PLAN render (the rules.py:438 obligation — reviewer
N4). Until one lands, the close-approval path cannot open on this
configuration — which is why run 1's cap spend was futile-for-repair yet
correctly honest.

**N4, the plumbing gap's second face (same fix surface)** —
`verification.md` is never seeded during PLAN in any B7/B8 run: the
plan-writer's rules.py:438 obligation ("Seed verification.md with one row per
success criterion") is unmet by the response_format plan render
(`verification_md_bytes=0` in every row despite plan.md rendering 2673/3559
bytes). This predates β and is the root of every "absent or empty" close
denial in B7/B8.
