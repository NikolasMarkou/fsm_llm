# NIAH-Padded Demo (M5 slice 4)

Padded needle-in-haystack QA over a non-τ·k^d-aligned document via the λ-substrate.

## What it does

Synthesises a **2000-char** document (deliberately unaligned: 1024 < 2000 < 2048
= τ·k^3 with τ=256, k=2) containing a needle (`ACCESS_CODE: SECRET-7421`) at
offset 1024, then runs `fsm_llm.stdlib.long_context.niah_padded`. The factory
internally pads the raw document up to `N* = aligned_size(2000, 256, 2) = 2048`
via a `Let`-bound pad callable, then delegates to the canonical recursive body:

```
let document = pad_to_aligned(raw_document) in
  fix(λself. λP.
    case size_bucket(P) of
      "small" → leaf(<question prompt>, P)
      _      → reduce_(best, fmap(self, split(P, k))))
```

This produces 8 leaf oracle calls on the *padded* size — exactly matching
`plan(PlanInputs(n=2048, ...)).predicted_calls`. The padded chunks (whitespace
filler) usually return `NOT_FOUND`; `best_answer_op` reduces those out.

## Run

```bash
# OpenAI (default model gpt-4o-mini):
export OPENAI_API_KEY=your-key-here
python examples/long_context/niah_padded_demo/run.py

# Ollama:
export LLM_MODEL=ollama_chat/qwen3.5:4b
python examples/long_context/niah_padded_demo/run.py
```

Exit 0 iff `oracle_calls_match_planner == True` (the hard theorem-2 contract).
`needle_found` is best-effort: small models may emit prose on whitespace chunks
instead of the `NOT_FOUND` sentinel — `best_answer_op` filters most of that, but
the demo treats it as a non-blocking observation.

## Pad-budget caveat

Worst-case padding factor is **k×** when `n = τ·k^d + 1`. For τ=256, k=2, the
practical inflation is ≤2× even on adversarial inputs; the demo's 2000→2048
case is only **1.024×** (48 pad chars). Document length matters: pick a raw
`n` close to but not above a `τ·k^d` boundary to minimise overhead.

## Type Note

`niah_padded(question, *, tau, k, pad_char=" ")` builds a `Let` node that binds `document = pad_to_aligned(raw_document)` (via an env-bound pad callable) and then delegates to the canonical `niah`-style recursive body. The pad callable is bound by the demo at `Executor.run` time. See `src/fsm_llm/stdlib/long_context/CLAUDE.md` for the helpers (`aligned_size`, `pad_to_aligned`, `make_pad_callable`).

## Why this is the M5 slice-4 milestone

`docs/lambda.md` §13 names `niah_padded` as the slice that closes the
non-τ·k^d-aligned input gap. The canonical `niah` factory's cost-equality
contract holds *only* when `n` is exactly aligned; `niah_padded` removes that
constraint so callers can ship arbitrary user input lengths with deterministic
predicted call counts. Compare against `niah_demo/` for the aligned baseline.
