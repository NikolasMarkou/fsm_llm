# fsm_llm.stdlib.long_context — Category-C Long-Context Factories

Recursive long-context λ-term factories. The home of M5. Each factory builds a `Fix` node whose Theorem-2 cost is closed-form per `plan(...)`.

Per `docs/lambda.md` §3 (Category C): "recursive decomposition over long inputs (λ-native from day one)" — long-document QA, aggregation across large corpora, pairwise tournaments, multi-hop retrieval. The shape is `SPLIT → FMAP(self) → REDUCE`.

**Purity invariant**: imports only from `fsm_llm.runtime`. AST-walk unit tests per module enforce.

- **Slices shipped**: M5 1, 2, 3, 4, 5, 6, 7
- **Install**: included in core (no extra needed for the factories themselves; the OOLONG dataset loader needs `pip install fsm-llm[oolong]`).

## File Map

```
long_context/
├── niah.py                   # niah factory + best_answer_op + make_size_bucket — needle-in-haystack QA (slice 1)
├── aggregate.py              # aggregate factory + aggregate_op — synthesise across all chunks (slice 2)
├── pairwise.py               # pairwise factory + compare_op + oracle_compare_op — k-ary tournament (slice 3 + 5)
├── multi_hop.py              # multi_hop + multi_hop_dynamic + make_dynamic_hop_runner + not_found_gate (slice 3 + 6)
├── niah_padded.py            # niah_padded + aligned_size + pad_to_aligned + make_pad_callable — non-aligned n (slice 4)
├── _recursive.py             # _recursive_long_context helper — shared body for niah/aggregate/pairwise (slice 3)
└── __init__.py               # 14 public exports
```

## Public Exports

```python
# Factories
niah, aggregate, pairwise, multi_hop, multi_hop_dynamic, niah_padded,

# Helpers
make_size_bucket, best_answer_op, aggregate_op,
compare_op, oracle_compare_op,
make_dynamic_hop_runner, not_found_gate,
make_pad_callable, aligned_size, pad_to_aligned,
```

## Factories

| Factory | Slice | Cost (per `plan`) | Comment |
|---------|------:|--------------------|---------|
| `niah(question, *, tau, k)` | 1 | `predicted_calls = k^d` (strict) | `n = τ·k^d` aligned |
| `aggregate(question, *, tau, k)` | 2 | `predicted_calls = k^d` (strict) | Free-form synthesis; no per-chunk ground truth |
| `pairwise(question, *, tau, k, reduce_op_name)` | 3 + 5 | `compare_op`: `k^d`; `oracle_compare_op`: `2·k^d − 1` | Sentinel short-circuit relaxes T2 to upper bound on sparse-needle inputs |
| `multi_hop(question, *, hops, tau, k)` | 3 | `hops · k^d` (additive) | Let-chain of independent Fix calls — each hop is a niah-shaped Fix |
| `multi_hop_dynamic(question, *, max_hops, tau, k, confidence_gate)` | 6 | `actual_hops · k^d` (per-actual-hops strict); `≤ max_hops · k^d` (upper bound) | Confidence-gated; iteration lifted to host via `make_dynamic_hop_runner` |
| `niah_padded(question, *, tau, k, pad_char=" ")` | 4 | `predicted_calls = k^d` against `N* = aligned_size(n, τ, k)` | Worst-case k× pad overhead when `n = τ·k^d + 1` |

## Theorem-2 Contract

For any factory, executing on a τ·k^d-aligned input gives `ex.oracle_calls == plan(...).predicted_calls`. Strict equality is the headline invariant; bench scorecards under `evaluation/bench_long_context_*.json` record per-(model × factory) cells with `theorem2_holds`.

Three relaxations are documented:
1. **`oracle_compare_op` sentinel short-circuit** (slice 5): when an arm is empty / `NOT_FOUND`, the oracle is bypassed → actual < predicted on sparse-needle inputs. Use **dense-haystack fixtures** (every τ-chunk holds a distinct concrete factual sentence) for live strict-T2 verification.
2. **`multi_hop_dynamic` runtime termination** (slice 6): early-exit means `actual_hops < max_hops`. Strict T2 holds per actual hops; loose T2 against `max_hops`.
3. **`niah_padded` worst-case overhead** (slice 4): when `n = τ·k^d + 1`, padding to `N* = τ·k^(d+1)` costs k× more leaf calls.

## Helpers

- `make_size_bucket(tau)` — pure callable `int → str` returning `"leaf"` if `n ≤ τ` else `"recurse"`. Used as the `Case` discriminator inside niah/aggregate/pairwise. **Do NOT extend `BUILTIN_OPS`** — bind via env (LESSONS.md "Closed Enum + Strategy Dispatch").
- `best_answer_op()` — a `ReduceOp` for niah (longest non-sentinel string wins).
- `aggregate_op()` — pure-Python joiner; zero oracle calls.
- `compare_op()` — slice-3 length-heuristic placeholder for pairwise (kept for back-compat).
- `oracle_compare_op(question, executor)` — slice-5 oracle-mediated tournament op. **Closure mutates `executor._oracle_calls` directly** (D-S5-001 — single-counter T2 demonstration). See LESSONS.md "Oracle-mediated reduce ops".
- `make_dynamic_hop_runner(executor, factory_kwargs, max_hops, confidence_gate, peer_env, actual_iters_cell)` — host orchestrator that calls `executor._eval(hop_term, env, _fix_depth=0)` per hop. **MUST use `_eval`, not `run`** (`run` resets `_oracle_calls` per call). LESSONS.md "Host-callable orchestrator with `_eval` bypass".
- `not_found_gate(threshold)` — confidence gate that returns `True` (continue) on weird LLM output via `try/except` defensive default.
- `aligned_size(n, tau, k)` / `pad_to_aligned(doc, tau, k, pad_char=" ")` / `make_pad_callable(tau, k, pad_char=" ")` — niah_padded helpers.

## Scripted-Oracle Test Pattern

```python
# tests/test_fsm_llm_long_context/test_niah.py
def scripted_oracle(prompt: str, schema=None):
    # Match by prompt substring; return per-chunk fixture.
    ...

ex = Executor(oracle=scripted_oracle)
term = niah(question="Where?", tau=256, k=2)
ex.run(term, env={"document": doc_τ_k_d_aligned})
assert ex.oracle_calls == plan(PlanInputs(n=len(doc), tau=256, k=2)).predicted_calls
```

## Live Demos

5 runnable demos under `examples/long_context/` — each ships a `VERIFICATION` block with hard `oracle_calls_match_planner` and (where applicable) `needle_found` gates. See each demo's README.

## Bench Harness

`scripts/bench_long_context.py` runs (model × factory) cells with optional `--workers N` ProcessPool parallelism, `--mode {default, oracle, dynamic}`, `--dataset PATH` (slice 6), `--score-mode {exact, substring, f1_token}`, `--max-hops N`, `--max-context-len N`. Cloud-model env-var preflight built in.

## OOLONG Dataset (slice 7)

```bash
pip install fsm-llm[oolong]
python scripts/datasets/oolong_loader.py --subset synth --limit-per-task 10 --max-context-len 2048
python scripts/bench_long_context.py --models ollama_chat/qwen3.5:4b \
    --factories aggregate --dataset evaluation/datasets/oolong_synth_real_synth.jsonl \
    --score-mode substring
```

License-conservative: HuggingFace streaming-mode load (no bulk download). Converted records gitignored under `evaluation/datasets/`. Telemetry JSONs (record IDs + cost data, no benchmark content) ARE committed. See `evaluation/datasets/README.md`.

## Testing

```bash
pytest tests/test_fsm_llm_long_context/    # Scripted-oracle factory tests
TEST_REAL_LLM=1 pytest -m real_llm tests/test_fsm_llm_long_context/   # Live smokes on qwen3.5:4b
```

## Related

- **`fsm_llm.lam.planner`** — Theorem-2 contract source. `plan(PlanInputs)` returns `Plan` with `predicted_calls`.
- **`fsm_llm.lam.executor`** — runs the term; `ex.oracle_calls` is the counter to assert against.
- **`evaluation/`** — bench scorecards (`bench_long_context_*.json`), dataset fixtures, OOLONG telemetry.
- **`examples/long_context/*_demo/`** — 5 runnable demos with hard T2 gates.
