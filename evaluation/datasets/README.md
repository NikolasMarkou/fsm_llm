# `evaluation/datasets/`

Dataset fixtures for `scripts/bench_long_context.py --dataset PATH`. Each fixture is shaped to exercise one of the M5 long-context **λ-term factories** (`fsm_llm.stdlib.long_context`).

## Fixture → Factory Map

| Fixture | Records | Factory | Theorem-2 form | Demo |
|---------|--------:|---------|----------------|------|
| `oolong_synth.jsonl` (`task: "niah"`) | 10 | `niah` | strict `k^d` | `examples/long_context/niah_demo` |
| `oolong_synth.jsonl` (`task: "aggregate"`) | 10 | `aggregate` | strict `k^d` | `examples/long_context/aggregate_demo` |
| `oolong_synth.jsonl` (`task: "multi_hop"`) | 10 | `multi_hop_dynamic` | per-actual-hops strict; `≤ max_hops · k^d` | `examples/long_context/multi_hop_demo` |
| `oolpairs_synth.jsonl` | 10 | `pairwise` | `k^d` (length mode) or `2·k^d − 1` (oracle mode) | `examples/long_context/pairwise_demo` |
| OOLONG real (`oolong_synth_real_*.jsonl`, `oolong_real_*.jsonl`) | per `--limit-per-task` | `aggregate` (per D-S7-003) | strict `k^d` | (slice 7 — bench-only) |

The bench harness reads `task` per record and routes via `_FACTORY_TASK_MAP`. Slice-6 telemetry (`actual_hops`, `oracle_calls`, `theorem2_holds`, `padded_size`) is recorded per-record in scorecard JSONs.

## Tracked synthetic fixtures (M5 slice 6)

These are **toy fixtures** authored in-tree, unrelated to the real
OOLONG benchmark. They exercise the bench harness on small, fully
self-contained inputs.

- `oolong_synth.jsonl` — 30 records (10 niah + 10 aggregate +
  10 multi_hop). Schema: `{id, task, question, answer, document, metadata}`.
  Used by slice-6 SC8/SC11 evidence runs.
- `oolpairs_synth.jsonl` — 10 pairwise records. Each holds
  `segment_a` + `segment_b` + the labelled winner. The bench harness
  joins them via `<SEP>`.

These files are committed to the repo. The "OOLONG" name in
`oolong_synth.jsonl` predates the slice-7 OOLONG ingestion path; the
file content is unrelated to upstream `oolongbench/oolong-synth`.

## Real OOLONG ingestion (M5 slice 7)

OOLONG is a long-context benchmark from arXiv 2511.02817 (Bertsch
et al.) hosted on HuggingFace at `oolongbench/oolong-{synth,real}`.
Per **D-S7-002** (license-conservative posture), we do NOT
redistribute OOLONG records. The loader pulls into your local HF
cache; output is gitignored.

### Install + run

```bash
# 1. Install the [oolong] extra (adds `datasets>=3.0.0`).
pip install -e ".[oolong]"

# 2. Pull a small subset (HF streaming, no 12 GB bulk download).
python scripts/datasets/oolong_loader.py \
    --subset synth --split validation \
    --limit-per-task 6 \
    --out evaluation/datasets/oolong_synth_real_subset.jsonl

# 3. Run the slice-6 bench harness over the converted JSONL.
python scripts/bench_long_context.py \
    --dataset evaluation/datasets/oolong_synth_real_subset.jsonl \
    --models ollama_chat/qwen3.5:4b \
    --factories aggregate \
    --tau 256 --k 2 --doc-size 1024 \
    --score-mode substring \
    --out evaluation/slice7_oolong_synth.json
```

### Subsets

- `--subset synth` → `oolongbench/oolong-synth`. 1024-token contexts.
  ICL aggregation tasks: MOST_FREQ, LEAST_FREQ, REPRESENTED_N_TIMES,
  RELATIVE_FREQ, NUMERIC_ONE_CLASS.
- `--subset real` → `oolongbench/oolong-real`. D&D campaign
  transcripts (very long contexts). Numeric counting answers.

### Output schema

Per **D-S7-003**, every OOLONG record becomes `task: "aggregate"` in the
internal schema; the original OOLONG task is preserved in
`metadata.oolong_task`.

```json
{
  "id": "oolong-synth-validation-counting-MOST_FREQ-0",
  "task": "aggregate",
  "question": "...",
  "answer": "...",
  "document": "...",
  "metadata": {
    "source": "oolongbench/oolong-synth",
    "oolong_id": 0,
    "oolong_task_group": "counting",
    "oolong_task": "MOST_FREQ",
    "oolong_answer_type": "LABEL",
    "context_len": 1024
  }
}
```

### License

OOLONG license is not stated explicitly on the HuggingFace dataset
cards. Refer to the [paper](https://arxiv.org/abs/2511.02817) and
the upstream HuggingFace dataset cards
(`oolongbench/oolong-synth`, `oolongbench/oolong-real`) for the
authoritative answer. We default to non-redistribution.

### Accuracy expectations

OOLONG is a hard benchmark. Per the paper, even GPT-5 achieves
< 50 % accuracy at 128K. Slice 7 demonstrates that our infrastructure
(planner, executor, bench harness) absorbs the data correctly and
that Theorem-2 holds; **we do not chase OOLONG accuracy numbers in
this slice**. Higher accuracy needs frontier models, custom prompting,
or a richer factory shape — slice 8+ candidates.
