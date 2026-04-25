# Aggregate Demo (M5 slice 2)

Synthesise an answer across ALL chunks of a long document via the
λ-substrate.

## What it does

Synthesises a 2048-char document with 4 topical sentences (codename, lead,
deadline, budget) placed at chunk-aligned offsets (0, 512, 1024, 1536).
Runs `fsm_llm.stdlib.long_context.aggregate` with τ=256 and k=2:

```
fix(λself. λP.
   case size_bucket(P) of
     "small" → leaf(<extract relevant facts from this chunk>, P)
     _      → reduce_(merge, fmap(self, split(P, k))))
```

8 leaves, each on a 256-char chunk. Reduce step is a pure-Python bullet
joiner (`aggregate_op()`) — zero extra oracle calls — so Theorem 2 holds
identically to `niah`.

## Run

```bash
# OpenAI (default model gpt-4o-mini):
export OPENAI_API_KEY=your-key-here
python examples/long_context/aggregate_demo/run.py

# Ollama:
export LLM_MODEL=ollama_chat/qwen3.5:4b
python examples/long_context/aggregate_demo/run.py
```

Exit 0 iff:
- `oracle_calls_match_planner` — `ex.oracle_calls == plan(...).predicted_calls`.
- `output_nontrivial` — result is a non-trivial bullet-joined synthesis
  (length > 50, not the sentinel).

## Caveat — verification is telemetry, not correctness

Aggregate output is free-form synthesis. There is no ground-truth string
to match per chunk. The `output_nontrivial` check is a weak heuristic; it
does not score quality. Theorem 2 (cost equality) is the only hard
correctness gate. Quality scoring against labelled benchmarks is a slice
4+ concern.

## Difference from `niah_demo`

| | `niah` | `aggregate` |
|---|---|---|
| Reduce | `best_answer_op()` (pick longest) | `aggregate_op()` (bullet-join) |
| Verification | `needle_found` (ground truth) | `output_nontrivial` (heuristic) |
| Oracle calls | `k^d` | `k^d` (same — Theorem 2) |
