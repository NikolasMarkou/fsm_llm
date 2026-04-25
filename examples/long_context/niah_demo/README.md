# NIAH Demo (M5 slice 1)

Recursive needle-in-haystack QA via the λ-substrate.

## What it does

Synthesises a 2048-char document with a single needle (`ACCESS_CODE: SECRET-7421`)
at offset 1024, then runs `fsm_llm.stdlib.long_context.niah` with τ=256 and
k=2. The factory builds the recursive λ-term

```
fix(λself. λP.
   case size_bucket(P) of
     "small" → leaf(<question prompt>, P)
     _      → reduce_(best, fmap(self, split(P, k))))
```

which produces 8 leaf oracle calls (each on a 256-char chunk) and reduces
with a "best answer" op that discards `NOT_FOUND` sentinels and picks the
longer informative response.

## Run

```bash
# OpenAI (default model gpt-4o-mini):
export OPENAI_API_KEY=your-key-here
python examples/long_context/niah_demo/run.py

# Ollama:
export LLM_MODEL=ollama_chat/qwen3.5:4b
python examples/long_context/niah_demo/run.py
```

Exit 0 iff both verification checks pass:
- `needle_found` — the answer string contains `SECRET-7421`.
- `oracle_calls_match_planner` — `ex.oracle_calls == plan(...).predicted_calls`.

## Why this is the M5 milestone

This is the smallest end-to-end demonstration of the paper's Category-C
recursive decomposition wired into fsm_llm. It exercises every kernel
piece (`fix`, `split`, `fmap`, `reduce_`, `case_`, `Leaf`) plus the
`LiteLLMOracle` and `CostAccumulator` end-to-end, and proves Theorem 2
(pre-computable cost) on real LLM traffic — not a scripted mock.

Future M5 slices add `aggregate`, `pairwise`, `multi_hop`, OOLONG-style
factories, and multi-model benchmark fan-out (Qwen3 / Llama / Mistral).
