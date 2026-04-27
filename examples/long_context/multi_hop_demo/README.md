# Multi-Hop Demo (M5 slice 3)

Two-hop retrieval over a long document via a `Let` chain of independent
niah-style λ-term sweeps.

## What it does

Synthesises a 2048-char document with two planted chunks — an entity
introduction (`Project Helix`) at offset 512 and a fact about that entity
(its launch date `March 15, 2026`) at offset 1280 — plus filler chunks.
Runs `fsm_llm.stdlib.long_context.multi_hop` with τ=256, k=2, hops=2:

```
let hop_0_result = fix(λself. λP.
   case size_bucket(P) of
     "small" → leaf(<entity-finding prompt>, P)
     _      → reduce_(best, fmap(self, split(P, k)))) document in
let hop_1_result = fix(λself. λP.
   case size_bucket(P) of
     "small" → leaf(<fact-finding prompt referencing hop_0_result>, P)
     _      → reduce_(best, fmap(self, split(P, k)))) document in
hop_1_result
```

Each hop re-traverses the full 8-leaf tree (8 oracle calls per hop). The
hop-1 leaf prompt closes over `hop_0_result` via a `Var` reference threaded
by `Let` — so hop 1's question is grounded in hop 0's finding. Total
oracle calls: `hops * predicted_calls = 2 * 8 = 16`.

## Run

```bash
# OpenAI (default model gpt-4o-mini):
export OPENAI_API_KEY=your-key-here
python examples/long_context/multi_hop_demo/run.py

# Ollama:
export LLM_MODEL=ollama_chat/qwen3.5:4b
python examples/long_context/multi_hop_demo/run.py
```

Exit 0 iff:
- `oracle_calls_match_2hop_planner` —
  `ex.oracle_calls == hops * plan(...).predicted_calls` (the hard
  theorem-2 gate; must hold).

The `launch_date_found` check is reported but does NOT gate the exit
code — it is a heuristic that depends on the live model surfacing
`March 15, 2026` (or a normalised variant) in its hop-1 answer.
Theorem 2 is invariant of model quality; fact-extraction quality is not.

## Multi-hop shape (D-S3-002)

`multi_hop` is `hops` independent `Fix` calls in a `Let` chain — NOT a
single `Fix` with a query-state monad. This keeps each hop within the
per-Fix Theorem-2 cost equality individually; the total cost across all
hops is exactly `hops * predicted_calls` (additive, not amortised).

Confidence-gated dynamic termination shipped in **M5 slice 6** as
`multi_hop_dynamic` (factory) + `make_dynamic_hop_runner` (host
orchestrator) + `not_found_gate` (predicate). The dynamic variant lifts
iteration to the host so the gate can short-circuit; Theorem-2
reformulates as `actual_oracle_calls == actual_hops · predicted_calls`
(strict per actual hops) AND `≤ max_hops · predicted_calls` (loose).
See `src/fsm_llm/stdlib/long_context/CLAUDE.md` for the helpers and
`docs/lambda.md` §13 for slice-6 status.

## Difference from `niah_demo` and `pairwise_demo`

| | `niah` | `pairwise` | `multi_hop` | `multi_hop_dynamic` (slice 6) |
|---|---|---|---|---|
| Term shape | single `Fix` | single `Fix` | `Let` chain of `hops` `Fix` calls | host orchestrator wrapping niah-shaped `Fix` per hop |
| Reduce | `best_answer_op()` | `compare_op()` (== best) | `best_answer_op()` | `best_answer_op()` |
| Leaf prompt | "find this needle" | "pick most-relevant segment" | hop-0 entity-find; hop-1 fact-find referencing hop-0 | same, but iteration is host-driven and gated |
| Verification | `needle_found` (ground truth) | `topic_a_selected` (heuristic) | `launch_date_found` (heuristic) | `launch_date_found` + per-actual-hops T2 |
| Oracle calls | `k^d` | `k^d` | `hops · k^d` | `actual_hops · k^d` (strict); `≤ max_hops · k^d` (loose) |

## Type Note

`multi_hop(question, *, hops, tau, k)` returns a `Let`-chain whose body is `Var("hop_{N-1}_result")`. Each `Let` binds a fresh `Fix` whose body's leaf prompt closes over the previous hops' results via env-bound `Var` references. This is how a query mutates per level without breaking `predicted_calls` — each individual `Fix` is independently planner-bounded. See `src/fsm_llm/stdlib/long_context/CLAUDE.md` for the alternative dynamic shape.
