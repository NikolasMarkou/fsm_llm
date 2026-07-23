# L8 `l8-explore-loop/B0` — pre-registration

*Plan: plan-2026-07-23T050609-8787a3ca (harness iteration 4, W1).*
*Committed BEFORE the live block runs, so the decision rule provably predates
the data (git log: this file's commit precedes the `B0/` rows commit). `plans/`
is gitignored, so this tracked file is the tamper-evident pre-registration of
the rule that also lives in plan.md.*

## What this block measures
The `ollama_chat/qwen3.5:4b` EXPLORE **redispatch-loop** collapse, characterized
per-tool-call. A NEW single-state runner (`_one_explore_loop`) drives the
harness's OWN EXPLORE loop (`_after_explore_dispatch`) over a bare `mkdir` plan
dir, multi-dispatch, bounded by the real `MAX_EXPLORE_REDISPATCHES=9`, stopping
at EXPLORE (faithfulness proven offline: it matches a real `agent.run()` collapse
— furthest=EXPLORE, halt=explore-cap, 10 dispatches). A spy on
`ToolRegistry.execute` records `(tool, ok, params, error)` per call, and a
deterministic classifier partitions every FAILED dispatch into exactly one
mechanism bucket.

- **Bench id / block**: `l8-explore-loop` / `B0`
- **n (pre-registered, fixed, one look)**: 10 runs (`RUNS_L8=10`); each run is a
  full EXPLORE loop of up to 10 dispatches.
- **Model**: `ollama_chat/qwen3.5:4b`, digest pinned in `B0/manifest.json`.
- **Primary artifact**: `B0/summary.json` — pooled mechanism family/bucket
  distribution over all dispatches, `runs_reaching_plan`, and the
  empty-vs-unparseable split within family (iii).

## The 6 mechanism buckets → 3 families (fixed vocabulary)
- `wrong-root` → **(ii)**
- `never-called` → **(i)**
- `accepted-no-bytes`, `empty-reply`, `unparseable`, `other-write-failure` → **(iii)**

This block asserts NO capability/gate-clear bar (it is a characterization block,
like L7). The only assertion is the partition hard-gate: every failed dispatch
maps to exactly one bucket.

## Pre-registered W1→W2 decision rule (structural 4b fixes only — model swap is OFF the table)
The DOMINANT mechanism family across pooled failed dispatches selects the
follow-on fix (a LATER iteration). Fixed before the data:

| Dominant mechanism | Pre-committed W2 follow-on (later iteration) |
|---|---|
| **(iii)-empty** (family iii, `empty-reply` the largest bucket) AND the D-002 repair turn returns empty | driver-side lever to force a usable 4b final answer: make `response_format` PRIMARY (reorder `native_fc.py:424`) and/or a redispatch-shape escalation after N empty-replies. NOT a model swap. Then fresh L6 B2. |
| **(iii)-unparseable** (family iii, `unparseable` largest) | make `response_format` PRIMARY not fallback, then fresh L6 B2 (n=3, floor sha256-identical to B1). |
| **(i) never-called** | driver-side forced-write EXPLORE target (mirror EXECUTE 2/40→40/40), then fresh L6 B2. |
| **(ii) wrong-root** | driver-side target/root repair (D-004 closed bare-sentinel), then fresh L6 B2. |
| **no single dominant family** | successor takes the largest family, states the residual, re-measures the same way (honest partial). |

## Run command (one look, D-002: no re-sample, no --force)
```
FSM_LLM_HARNESS_LIVE=1 .venv/bin/python -m pytest \
  "tests/test_fsm_llm_harness/test_live_ollama.py::TestL8ExploreLoop::test_explore_loop_mechanism_block" -q -s
```
