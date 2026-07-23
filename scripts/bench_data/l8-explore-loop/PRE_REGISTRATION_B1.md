# L8 `l8-explore-loop/B1` — pre-registration (forced-write fix re-measure)

*Plan: plan-2026-07-23T073649-bb230f18 (harness iteration 5).*
*Committed BEFORE the B1 live run, so the decision rule provably predates the
data (git log: this file's commit precedes the `B1/` rows commit). Same
one-look / no-`--force` / no-re-sample discipline as B0 (D-002).*

## What changed since B0 (the thing under test)
The forced-write finalization fix (D-003, commits `9446d08` + `3da96ba`): the
EXPLORE worker now sets `AgentConfig.force_final_tool = "write_plan_file"`, so
`NativeFunctionCallingReactAgent.run()` issues ONE post-loop forced-`tool_choice`
turn when the ReAct loop concluded without calling the write tool. The MODEL
emits a genuine `write_plan_file` call (executed through the normal confined
tool path; `_verified_writes` stays honest — no driver salvage, no ethos shift).
This targets B0's measured dominant mechanism: **never-called-a-write-tool =
75/89 (84%)**, `gate_cleared` 0/10.

## Block parameters (identical to B0 except the block name)
- Bench id / block: `l8-explore-loop` / **B1**.
- n = 10 runs (`RUNS_L8=10`), each a full EXPLORE loop up to `MAX_EXPLORE_REDISPATCHES=9`.
- Model: `ollama_chat/qwen3.5:4b`, digest pinned in `B1/manifest.json`.
- Same runner (`_one_explore_loop`), same classifier (`classify_failed_dispatch`),
  same 6-bucket→3-family vocabulary, same partition hard-gate assertion. The ONLY
  difference vs B0 is the product under test (forced-write fix is now live).

## Pre-registered decision rule (fixed BEFORE the run)
The fix is judged to SHIFT THE MECHANISM (→ proceed to spend the L6 B2 e2e block) iff EITHER:
- **(A) capability signal**: `gate_cleared` rises above B0's **0/10** (≥1 of 10
  runs reaches the 3-findings EXPLORE gate), OR
- **(B) mechanism signal**: the `never-called` (family i) share of failed
  dispatches drops **materially** below B0's 84% (75/89) — operationalized as
  never-called family share < 50% of failed dispatches AND the total findings
  landed (successful dispatches) rises materially above B0's 11/100.

If NEITHER (A) nor (B) holds → **honest-negative branch**: the forced turn did
not convert the mechanism into gate progress (e.g. the provider dropped the
forced choice after a long read history, or writes land but the loop still can't
accumulate 3 DISTINCT topics). Do NOT spend the L6 B2 block; record the honest
negative and aim the next lever accordingly.

Report regardless: pooled `family_counts`, `mechanism_counts`, `iii_empty` /
`iii_unparseable`, `k_gate_cleared` with Wilson CI, `runs_reaching_plan`, total
dispatches, and a per-arm B0-vs-B1 comparison (Wilson CIs + Fisher on
`gate_cleared` and on never-called share). `hb.report()` cannot auto-compare
flat `rows.jsonl`, so the compare is an ad-hoc `python -c` reusing
`hb.wilson_ci` / `hb.fisher_exact_two_sided`.

## Run command (one look)
```
FSM_LLM_HARNESS_LIVE=1 .venv/bin/python -m pytest \
  "tests/test_fsm_llm_harness/test_live_ollama.py::TestL8ExploreLoop::test_explore_loop_mechanism_block" -q -s
```
(after `L8_BLOCK = "B1"`).
