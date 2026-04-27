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

## Caveat — slice-3 op limitation (D-S3-001, historical)

The slice-3 `compare_op` (default `--mode length`) is functionally
equivalent to `best_answer_op` (longer-non-sentinel-wins). Pairwise's
differentiation from `niah` in this mode lives in the *leaf prompt
template* (asks the oracle to pick between segments) and the *demo
content* (two competing topical segments), not in the op math. The
oracle-mediated variant shipped in M5 slice 5 — see below.

## Oracle-mediated mode (M5 slice 5)

A real oracle-mediated pairwise tournament op is available via
`--mode oracle`. Each reduce step asks the oracle to pick the winner of
a pair against the question:

```bash
export LLM_MODEL=ollama_chat/qwen3.5:4b
python examples/long_context/pairwise_demo/run.py --mode oracle
```

Theorem-2 cost equality changes shape:

| Mode | Predicted oracle calls | Doc τ=256, k=2, len=2048 |
|---|---|---|
| `length` (default) | `k^d` (leaves only; reduce is pure) | 8 |
| `oracle` | `2·k^d − 1` (leaves + reduce pairs) | 15 |

The planner is informed by passing `reduce_calls_per_node=1` to
`PlanInputs` in oracle mode — the executor's `_oracle_calls` counter
ticks once per pair invocation, so `oracle_calls_match_planner` remains
the hard gate in both modes.

Sentinel short-circuit: when one arm is sentinel/empty, the op returns
the other arm without an oracle call. Strict T2 equality therefore
requires every reduce input to have two non-sentinel arms at the leaf
level (the demo's two-topic doc satisfies this in the path leading to
the winner; sparse cases relax T2 to an upper bound).

## Difference from `niah_demo` and `aggregate_demo`

| | `niah` | `aggregate` | `pairwise` (length) | `pairwise` (oracle) |
|---|---|---|---|---|
| Reduce | `best_answer_op()` | `aggregate_op()` (bullet-join) | `compare_op()` (== best) | `oracle_compare_op` (LLM picks winner) |
| Leaf prompt | "find this needle" | "extract relevant facts" | "pick most-relevant segment" | "pick most-relevant segment" |
| Verification | `needle_found` (ground truth) | `output_nontrivial` (heuristic) | `topic_a_selected` (heuristic) | same |
| Oracle calls | `k^d` | `k^d` | `k^d` | `2·k^d − 1` |
| Theorem-2 form | strict | strict | strict | strict (dense haystack) / upper bound (sparse) |

## Type Note

In `length` mode the reduce slot is a pure `Combinator` (`compare_op()`); in `oracle` mode it's a `Leaf` (`oracle_compare_op` closes over `executor` and increments `_oracle_calls` directly — D-S5-001). Theorem-2 holds in both modes — the planner shape changes via `PlanInputs.reduce_calls_per_node`. See `src/fsm_llm/lam/CLAUDE.md` (Planner) and `src/fsm_llm/stdlib/long_context/CLAUDE.md` (factory table).
