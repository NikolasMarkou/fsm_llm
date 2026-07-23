# L6 `l6-e2e/B2` — pre-registration (forced-write fix, end-to-end)

*Plan: plan-2026-07-23T073649-bb230f18 (harness iteration 5).*
*Committed BEFORE the B2 live run (git log: this file precedes the `B2/` rows
commit). One-look / no-`--force` / no-re-sample (D-002).*

## What changed since B1 (the thing under test)
The forced-write finalization fix (D-003, `9446d08`+`3da96ba`), the SAME product
L8 B1 measured shifting EXPLORE from `gate_cleared` 0/10 → **9/10** (Fisher
p=0.00012, D-004). B1 (the prior L6 block) was **0/3**, every run blocked at
EXPLORE (`furthest_state=explore`, `halt_slug=explore-cap`). B2 tests whether the
now-clearing EXPLORE gate lets a full real-worker traverse advance past EXPLORE.

## Floor-clause identity (HARD)
The three floor clauses + the floor-loop assertion body + `HONEST_HALT_SLUGS` /
`E2E_STATE_RANK` / `GOAL` / `SEED_FILES` are BYTE-IDENTICAL to B1. Verified: the
`inspect.getsource` sha256 of
`_verified_execute_workspace_write`+`_normalized_ws_path`+`_bench_defect`+the
floor-loop test body ==
`cbeeb6aaa2a5a2e5910fed7abb77e1b44651c8d81715301e2c4b547509b453ce` at
registration (the iteration-5 fix touched no L6 floor clause — it changed
`fsm_llm_agents` + the EXPLORE wiring, not the L6 grading). NOTE: B2's manifest
`prompt_bytes_sha256` will legitimately DIFFER from B1's — the fix changes the
EXPLORE dispatch's behavior/prompt round-trip; the FLOOR GRADING clauses are what
must match, not the prompt hash. `report()`'s `_digests_comparable` only gates on
`model_digest`, which is unchanged (4b `2a654d98e6fb`).

## Block parameters (identical to B1 except the block name + the fixed product)
- Bench id / block: `l6-e2e` / **B2**. n = 3 runs (`RUNS_E2E=3`).
- Real workers (`_one_e2e_run`, `native_function_calling=True`), DENY-default
  disk-bound approval stub, `E2E_WALL_CLOCK_CEILING_S` per run.
- Model `ollama_chat/qwen3.5:4b`, digest pinned in `B2/manifest.json`.

## Pre-registered grading (the floor, transcribed from B1 — unchanged)
Each run is graded on the SAME three floor clauses:
1. `reached >= EXECUTE` (`furthest_state` ranks ≥ EXECUTE).
2. `verified_write` (an EXECUTE-state dispatch's OWN write path intersects the
   sha256-diffed changed-file set).
3. `honest_halt` (`error is None and not timed_out and (close_reached or
   halt_slug in HONEST_HALT_SLUGS)`).
Plus the `_bench_defect` circuit-breaker (unchanged).

**Success criterion for this iteration**: ≥1 of the 3 runs reaches ≥EXECUTE with
a verified write + honest halt (B1 was 0/3, blocked at EXPLORE). Report all three
rubric vectors regardless. This is the founding end-goal's first live test above
EXPLORE.

## W3 (falsifier grading, if reached)
If ANY run reaches PLAN, grade the `plan_redispatches` row field against
`MAX_PLAN_REDISPATCHES=3` (report; it has had zero live evidence).

## Run command (one look)
```
FSM_LLM_HARNESS_LIVE=1 .venv/bin/python -m pytest \
  "tests/test_fsm_llm_harness/test_live_ollama.py::TestL6EndToEndRealWorkers::test_three_full_runs_grade_at_or_above_the_floor" -q -s
```
(after `L6_BLOCK = "B2"`).
