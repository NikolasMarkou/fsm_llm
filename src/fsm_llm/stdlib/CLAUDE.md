# fsm_llm.stdlib — Standard Library of λ-Term Factories

The standard library: **named λ-term factories** organised by domain. Each subpackage exposes pure factory functions that return `fsm_llm.runtime.Term` nodes — closures over no Python state. Bind dynamic values (host callables, predicates, classifiers) via the env at execution time.

Per `docs/lambda.md` §11: this is the post-unification home of what used to be `fsm_llm_reasoning`, `fsm_llm_workflows`, and `fsm_llm_agents`. Those top-level siblings remain as `sys.modules` shims that resolve here, but in 0.6.0 they emit `DeprecationWarning(since="0.6.0", removal="0.7.0")` on import — new code should `from fsm_llm.stdlib.<reasoning|workflows|agents> import …` directly.

**Purity invariant**: every `lam_factories.py` imports **only from `fsm_llm.runtime`**. AST-walk unit tests per subpackage enforce this. Class-based legacy code (e.g. `ReactAgent`, `ReasoningEngine`, `WorkflowEngine`) coexists in the same subpackage as the factories — both paths are active.

## Subpackages

| Subpackage | Slice | Factories | Reference |
|------------|-------|-----------|-----------|
| `agents/` | M3 slice 1 | `react_term`, `rewoo_term`, `reflexion_term`, `memory_term` | `agents/CLAUDE.md` |
| `reasoning/` | M3 slice 2 | 11 factories (9 strategies + classifier + solve orchestrator) | `reasoning/CLAUDE.md` |
| `workflows/` | M3 slice 3 | `linear_term`, `branch_term`, `switch_term`, `parallel_term`, `retry_term` | `workflows/CLAUDE.md` |
| `long_context/` | M5 (slices 1-7) | `niah_term`, `aggregate_term`, `pairwise_term`, `multi_hop_term`, `multi_hop_dynamic_term`, `niah_padded_term` + helpers (bare names `niah`, `aggregate`, … remain as deprecated aliases through 0.6.x) | `long_context/CLAUDE.md` |

## Design Pattern (canonical)

```python
# Inside src/fsm_llm/stdlib/<pkg>/lam_factories.py
from fsm_llm.runtime import Term, leaf, let_, var, ...

def react_term(
    *,
    decide_prompt: str,
    synth_prompt: str,
    tool_dispatch_var: str = "tool_dispatch",
    decision_schema: type[BaseModel] = ToolDecision,
    input_var: str = "question",
) -> Term:
    """Build a ReAct-shape λ-term: 2 oracle calls."""
    return let_(
        "decision", leaf(prompt=decide_prompt, schema=decision_schema, input_var=input_var),
        let_(
            "observation", app(var(tool_dispatch_var), var("decision")),
            leaf(prompt=synth_prompt, input_var="observation"),
        ),
    )
```

Caller wires the env:

```python
ex.run(react_term(decide_prompt=..., synth_prompt=...),
       env={"question": "...", "tool_dispatch": tool_runner})
```

## Theorem-2 Forms by Factory Shape

| Shape | Form | Example |
|-------|------|---------|
| **let-chain** (linear/agent/reasoning) | strict equality `oracle_calls == sum(leaves)` | `react_term`, `analytical_term`, `linear_term` |
| **case-using** (branch/switch) | runtime arm-only `oracle_calls == leaves(taken_arm)` | `branch_term`, `switch_term` |
| **fix-using** (retry / long-context recursion) | bounded recursion: closed-form per `plan(...)`, e.g. `k^d` | `niah_term`, `aggregate_term`, `retry_term` |
| **0-leaf** (host-callable body) | `oracle_calls == 0` (trivial) | `retry_term` with no-leaf body |
| **oracle-mediated reduce** (long-context slice 5) | `2·k^d − 1` strict; sentinel short-circuit relaxes to upper bound on sparse-needle | `pairwise_term` with `oracle_compare_op` |
| **dynamic recursion** (multi-hop slice 6) | strict `actual == actual_hops · predicted_per_hop`; loose `≤ max_hops · predicted_per_hop` | `multi_hop_dynamic_term` |

Bench scorecards under `evaluation/m3_slice*_*_scorecard.json` and `evaluation/bench_long_context_*.json` capture these per-cell.

## Verification Pattern (per subpackage)

1. **AST-walk purity test** — assert the `lam_factories` module imports only from `fsm_llm.runtime`.
2. **Per-factory shape test** — count Leaves, walk let-binding names, assert root node kind.
3. **5-cell live smoke** (`@pytest.mark.real_llm`, gated by `TEST_REAL_LLM=1`) — execute on `ollama_chat/qwen3.5:4b`, assert `ex.oracle_calls == expected`.
4. **Bench scorecard JSON** — per-(model × factory) cell with `theorem2_holds` boolean.

## Naming Conventions

- Factory names end in `_term` (e.g. `react_term`, `analytical_term`, `linear_term`).
- Helpers (size buckets, reduce ops, predicates, gates) take the underlying mechanism name (e.g. `make_size_bucket`, `best_answer_op`, `not_found_gate`).
- Private `_chain(*pairs) -> Term` helpers are duplicated per subpackage by design — purity is per-module (LESSONS.md "M3 Stdlib Factory Layers — Trinity Pattern").

## Related

- **`fsm_llm.runtime`** — the kernel these factories build on. See `runtime/CLAUDE.md`.
- **`examples/pipeline/`** — 47 Category-B examples authored as inline λ-terms; M4 evidence corpus.
- **`examples/long_context/`** — 5 demos exercising long-context factories with hard Theorem-2 gates.
