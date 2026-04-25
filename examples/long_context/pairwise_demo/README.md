# Pairwise Demo (M5 slice 3)

Pick the segment most relevant to a question via recursive k-ary
tournament reduction over the λ-substrate.

## What it does

Synthesises a 2048-char document with two competing topical segments —
Topic A (deep-sea hydrothermal vents) at offset 512 and Topic B (medieval
guilds) at offset 1280 — plus filler chunks. Runs
`fsm_llm.stdlib.long_context.pairwise` with τ=256 and k=2:

```
fix(λself. λP.
   case size_bucket(P) of
     "small" → leaf(<pairwise prompt asking about Topic A>, P)
     _      → reduce_(compare, fmap(self, split(P, k))))
```

8 leaves, each on a 256-char chunk. Each leaf asks the oracle to return
the most-relevant segment of its chunk for the Topic A question (or
`NOT_FOUND`). The reduce step is `compare_op()` —
"longer-non-sentinel-wins" per D-S3-001 — so Theorem 2 holds identically
to `niah`/`aggregate`.

## Run

```bash
# OpenAI (default model gpt-4o-mini):
export OPENAI_API_KEY=your-key-here
python examples/long_context/pairwise_demo/run.py

# Ollama:
export LLM_MODEL=ollama_chat/qwen3.5:4b
python examples/long_context/pairwise_demo/run.py
```

Exit 0 iff:
- `oracle_calls_match_planner` — `ex.oracle_calls == plan(...).predicted_calls`
  (the hard theorem-2 gate; must hold).

The `topic_a_selected` check is reported but does NOT gate the exit
code — it is a heuristic that depends on the live model picking the
Topic A segment over the Topic B decoy. Theorem 2 is invariant of model
quality; segment-pick quality is not.

## Caveat — slice-3 op limitation (D-S3-001)

The slice-3 `compare_op` is functionally equivalent to `best_answer_op`
(longer-non-sentinel-wins). Pairwise's differentiation from `niah` lives
in the *leaf prompt template* (asks the oracle to pick between segments)
and the *demo content* (two competing topical segments), not in the op
math. A true oracle-mediated comparison op is deferred to slice 4.

## Difference from `niah_demo` and `aggregate_demo`

| | `niah` | `aggregate` | `pairwise` |
|---|---|---|---|
| Reduce | `best_answer_op()` | `aggregate_op()` (bullet-join) | `compare_op()` (== best) |
| Leaf prompt | "find this needle" | "extract relevant facts" | "pick most-relevant segment" |
| Verification | `needle_found` (ground truth) | `output_nontrivial` (heuristic) | `topic_a_selected` (heuristic) |
| Oracle calls | `k^d` | `k^d` | `k^d` (Theorem 2) |
