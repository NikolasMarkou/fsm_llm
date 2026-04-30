# Architecture

`fsm-llm` `0.8.0` is a **typed λ-calculus runtime** with two surface syntaxes (FSM JSON and λ-DSL) and one verb (`Program.invoke`). This page tours how the pieces fit together. For the formal thesis, see [`lambda.md`](lambda.md). For invariants and falsification gates, see [`lambda_fsm_merge.md`](lambda_fsm_merge.md).

## One runtime, two surfaces, one verb

```
        FSM JSON (Category A)              λ-DSL (Category B / C)
              │                                    │
              ▼  fsm_llm.dialog.compile_fsm        ▼  fsm_llm.runtime.dsl
        ┌─────────────────────────────────────────────────────┐
        │                  λ-AST (typed Term)                 │
        │  Var · Abs · App · Let · Case · Combinator · Fix    │
        │                       · Leaf                        │
        └─────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────┐
        │ Executor (β-reduction, depth-bounded)    │
        │ Planner  (closed-form k*, τ*, d, calls)  │
        │ Oracle   (one per Program — uniform)     │
        │ Session  (per-conversation persistence)  │
        │ Cost     (per-leaf accumulator)          │
        └──────────────────────────────────────────┘
                                │
                                ▼
                       Program.invoke(...)  →  Result
```

Both surfaces compile to the **same** typed AST. There is no separate "FSM engine" and "agent engine" and "workflow engine" — there is one β-reduction interpreter (`runtime/executor.py`) and everything reduces through it. The standard library (`stdlib/`) is a collection of named factory functions that produce `Term` nodes in the canonical shapes (let-chains, `Case`-using, `Fix`-using); class-based legacy code (e.g. `ReactAgent`, `ReasoningEngine`, `WorkflowEngine`) coexists with the factories, and both paths run on the same kernel.

## The four layers

The public surface is partitioned into four layers plus a Legacy block. The partitions are computed in `src/fsm_llm/__init__.py` as `_LAYER_L1..L4` and asserted disjoint + cover by `tests/test_fsm_llm/test_layering.py`.

| Layer | Role | Imports | Names |
|-------|------|---------|-------|
| **L1 REDUCE** | Substrate | nothing upstream (closed against `dialog/`) | `Term`, AST nodes, DSL builders, `Executor`, `Plan`/`plan`, `Oracle`/`LiteLLMOracle`, kernel exceptions, `LeafCall`/`CostAccumulator` |
| **L2 COMPOSE** | Pure Term→Term transforms + construction-time data | L1 | `compose`, full `Handler*` surface, profiles surface |
| **L3 AUTHOR** | Term producers | L1, L2 | stdlib factories (`react_term`, `niah_term`, `analytical_term`, `linear_term`, …); `compile_fsm`, `compile_fsm_cached` |
| **L4 INVOKE** | One verb | L1–L3 | `Program`, `Result`, `ExplainOutput`, `ProgramModeError` |
| **Legacy** | FSM dialog front-end + utilities | L1–L3 | `API`, `FSMManager`, definitions, classifiers, prompt builders, sessions, validators, visualisers, `Memory`, `ContextCompactor` |

Adding a new public name requires a layer assignment. The import-audit test (`test_layering.py::TestImportAudit`) enforces no upward edges from L1.

## Package layout

```
src/fsm_llm/
├── runtime/                # M1 — typed λ-AST + Executor + Planner + Oracle + cost
│   ├── ast.py              #   Var, Abs, App, Let, Case, Combinator, Fix, Leaf, Term, is_term
│   ├── dsl.py              #   var, abs_, app, let_, case_, fix, leaf, split, peek, fmap, ffilter, reduce_, concat, cross, host_call
│   ├── combinators.py      #   ReduceOp + BUILTIN_OPS (closed registry)
│   ├── executor.py         #   β-reduction interpreter
│   ├── planner.py          #   plan() — closed-form
│   ├── oracle.py           #   Oracle Protocol + LiteLLMOracle
│   ├── _litellm.py         #   LiteLLMInterface (private-by-convention adapter)
│   ├── _handlers_ast.py    #   private — compose() + AST splicers (moved from handlers.py at 0.8.0)
│   ├── cost.py             #   per-Leaf telemetry
│   ├── errors.py           #   LambdaError hierarchy
│   └── constants.py
│
├── dialog/                 # FSM dialog front-end
│   ├── api.py              #   API class
│   ├── fsm.py              #   FSMManager (per-conversation RLocks)
│   ├── turn.py             #   MessagePipeline — compiled-path 2-pass body
│   ├── extraction.py       #   private — ExtractionEngine (Pass-1 cluster, extracted from turn.py at 0.8.0)
│   ├── compile_fsm.py      #   FSMDefinition → Term + lru_cache(64) front door
│   ├── prompts.py          #   *PromptBuilder + BasePromptBuilder helpers (dedup at 0.8.0) + to_template_and_schema producers
│   ├── classification.py   #   Classifier, HierarchicalClassifier, IntentRouter
│   ├── transition_evaluator.py
│   ├── definitions.py      #   Pydantic models (FSMDefinition, State, Transition, …)
│   └── session.py          #   SessionStore ABC + FileSessionStore
│
├── stdlib/                 # Named λ-term factories
│   ├── agents/             #   react_term, rewoo_term, reflexion_term, memory_term + 12 class agents
│   ├── reasoning/          #   11 strategy factories + classifier_term + solve_term + ReasoningEngine (descriptive parameter names since 0.8.0)
│   ├── workflows/          #   linear/branch/switch/parallel/retry term factories + WorkflowEngine
│   └── long_context/       #   niah_term, aggregate_term, pairwise_term, multi_hop_term, multi_hop_dynamic_term, niah_padded_term
│
├── types.py                # Neutral types layer (since 0.7.0) — FSMError + runtime-touching Pydantic models
├── program.py              # L4 facade (Program, Result, ExplainOutput, ProgramModeError)
├── handlers.py             # HandlerSystem, HandlerBuilder, BaseHandler, LambdaHandler, HandlerTiming. compose + splicers re-exported from runtime/_handlers_ast.py.
├── profiles.py             # HarnessProfile + ProviderProfile + registries + apply_to_term
├── _api/deprecation.py     # warn_deprecated + reset_deprecation_dedupe (private)
└── __init__.py             # Layered __all__ — see L1..L4 partition
```

The `runtime/` kernel is **closed against `dialog/`** as of D-001 (`plan_2026-04-27_5d8a038b`): it imports nothing from `dialog/`. The `compile_fsm` / `compile_fsm_cached` symbols live in `dialog/compile_fsm.py` and surface at the top level (`from fsm_llm import compile_fsm`) for convenience.

## How `Program.invoke` reaches the executor

### FSM mode

```
Program.from_fsm(defn)
     ├── construct API (compiles defn via compile_fsm_cached)
     ├── apply HarnessProfile to compiled term (if profile=)
     └── compose handlers into compiled term

Program.invoke(message=..., conversation_id=...)
     └── API.converse(...)
            ├── FSMManager.process_message(...)
            ├── MessagePipeline runs compiled term step
            │      └── Executor.run(term, env)
            └── Session persistence (if SessionStore configured)
```

A "turn" is one β-reduction step on the cached compiled term. The dialog surface composes the FSM-JSON schema, the prompt builders, and the transition evaluator into a single λ-term per state; the executor traverses it and emits Leaf calls.

### Term / factory mode

```
Program.from_term(t) | Program.from_factory(f, ...)
     ├── apply HarnessProfile (if profile=)
     └── compose handlers (if handlers=)

Program.invoke(inputs={...})
     └── Executor.run(self._term, env=inputs)
            ├── reduces β-style
            ├── increments self.oracle_calls per Leaf call
            └── populates self.cost (CostAccumulator)
```

`Result.leaf_calls` and `Result.oracle_calls` come straight off the executor.

## The Theorem-2 contract

For every `Fix` subtree in a program, the executor's `oracle_calls` equals the planner's `predicted_calls` strictly when the input is τ·k^d-aligned. Universal-by-default for non-terminal FSM programs (post-A.M3c) and for every λ-DSL factory in the stdlib.

```
plan(PlanInputs(n=10000, K=8192, tau=512, k=2)).predicted_calls   # closed-form
        ⇕  for τ·k^d-aligned input
ex.run(term, env).oracle_calls
```

Three documented relaxations:

1. **`oracle_compare_op` sentinel short-circuit** — empty / `NOT_FOUND` arms bypass the oracle on sparse-needle inputs. Use dense-haystack fixtures for live strict-T2 verification.
2. **`multi_hop_dynamic_term` runtime termination** — early-exit means `actual_hops < max_hops`. Strict per actual hops; loose against `max_hops`.
3. **`niah_padded_term` worst-case overhead** — when `n = τ·k^d + 1`, padding to `N* = τ·k^(d+1)` costs k× more leaf calls.

Bench scorecards under `evaluation/bench_long_context_*.json` capture per-(model × factory) cells with `theorem2_holds`.

### The terminal `output_schema_ref` caveat

The single residual exception to Theorem-2 universality on the FSM side is **terminal non-cohort states with `output_schema_ref` unset**: the compiler emits a host-callable fallback (`App(CB_RESPOND, instance)`) instead of a `Leaf`, and the host call is intentionally invisible to `predicted_calls` (host-side semantics — handler invocation, generator returns, exception escapes — are out of scope for `plan(...)`). Authors who need strict-T2 on a terminal state set `State.output_schema_ref = MyPydanticModel`; the compiler routes the terminal branch to a structured Leaf and Theorem-2 strict equality holds end-to-end. This is the **A.D5 opt-in** documented in [`lambda_fsm_merge.md`](lambda_fsm_merge.md) §4 CAND-C.

## The M3c default-flip (now structural)

Theorem-2 universal-by-default for non-terminal FSM programs is structural since 0.8.0. Pre-0.7.0 history: `plan_2026-04-29_0f87b9c4` flipped the `State._emit_response_leaf_for_non_cohort` default from `False` to `True`, leaving the field as an opt-OUT gate exercised by regression-coverage helpers. **Phase A of the 0.8.0 cleanup removed both the field and the `_disable_leaf` / `_legacy_defn` test helpers**, simplifying the compiler's `_compile_state` branch significantly. Every non-terminal non-cohort FSM state now emits a real `Leaf` for response generation — there is no per-state knob to flip back. This closes invariant **I6** in the merge contract.

The legacy `App(CB_RESPOND, instance)` shape now appears **only** for the D3 conservative terminal-non-cohort fallback (terminal states without `output_schema_ref`); it retires together with M3d-wide cleanup once stdlib agents universally adopt `output_schema_ref`.

## Key cross-cutting decisions

- **D-001 — Kernel closure.** `runtime/` imports nothing from `dialog/`. The `compile_fsm` / `compile_fsm_cached` re-exports live in `dialog/compile_fsm.py` and surface at the top level. The kernel↔dialog import allow-list shrunk from 5 entries to **0** at 0.7.0 — the `FSMError` hierarchy and the runtime-touching Pydantic models moved to a neutral `fsm_llm.types` layer (the lam-shim back-reference was deleted at 0.6.0).
- **D-005 — Streaming ⊥ schema.** A `Leaf` cannot carry both `streaming=True` and `schema_ref != None`; mid-stream schema enforcement is unreliable. Enforced at compile time.
- **D-008 — `LiteLLMOracle._invoke_structured` bypasses subclass overrides.** A subclass of `LiteLLMInterface` overriding `generate_response` is **not** invoked on Executor-driven structured Leaf calls. Implement the `Oracle` protocol directly for full control. See `runtime/oracle.py` and [`api_reference.md`](api_reference.md) for the escape hatch.
- **D-009 — `LiteLLMInterface` private (formalised at 0.7.0).** No top-level re-export. Compose through `LiteLLMOracle(llm)` or `from fsm_llm.runtime._litellm import LiteLLMInterface`.
- **D-PIVOT-1-R13 — Three-epoch deprecation calendar.** R13-epoch shims (`fsm_llm.api`, `fsm_llm.lam`, …) were deleted at `0.6.0`. I5-epoch surfaces (`Program.run`/`converse`/`register_handler`, `fsm_llm.API`, sibling shim packages, long-context bare names) warned in `0.6.x` and were **removed at 0.7.0** — accessing them now raises `AttributeError` / `ImportError`. The Z8-epoch back-compat ballast (`Handler` alias, top-level `LLMInterface`/`BUILTIN_OPS`, `has_*`/`get_*` helpers, `dialog/definitions.py` type re-exports, the `State._emit_response_leaf_for_non_cohort` gate, and the hidden `Program.__init__` kwargs) was hard-removed at `0.8.0` with no warn cycle per the explicit no-back-compat directive. See [`migration_0.7_to_0.8.md`](migration_0.7_to_0.8.md) and the archived [`migration_0.6_to_0.7.md`](archive/migration_0.6_to_0.7.md).

## Where execution lives

There is exactly one runtime: **`fsm_llm.runtime.Executor`**. Both surfaces compile to the same AST.

```
       FSM JSON  →  compile_fsm()  →┐
                                    │
                                    ▼
                                  Term  →  Executor.run(env)  →  result
                                    ▲
       λ-DSL    →  dsl builders   →┘
```

`API.converse()` is one β-reduction step on a cached compiled term. `MessagePipeline` is the body of that term for Category-A FSM dialogs; its public `process` / `process_stream` entry points were retired in M2 S11 — there is no "legacy path" to maintain.

Stdlib factories build on the kernel and are executed by the same `Executor`. `WorkflowEngine`, `ReasoningEngine`, and the class-based agents wrap the factories with their own state; they don't replace the kernel.

## Reading the code

If you're new, walk the layers in order:

1. `runtime/ast.py` and `runtime/dsl.py` — the AST and the builders. Twenty minutes.
2. `runtime/executor.py` — the β-reduction loop. Thirty minutes.
3. `runtime/planner.py` — Theorems 2 and 4 in code. Thirty minutes.
4. `program.py` — the L4 facade. Fifteen minutes.
5. `dialog/compile_fsm.py` — FSM JSON → Term. The link from FSM authoring to the kernel.
6. `stdlib/long_context/niah.py` — the canonical recursive factory. SPLIT → FMAP(self) → REDUCE.

That's the whole architecture. Everything else is implementation detail in service of those six files.

## Related reading

- [`lambda.md`](lambda.md) — the architectural thesis, Theorems 1–5.
- [`lambda_fsm_merge.md`](lambda_fsm_merge.md) — invariants I1–I6, falsification gates G1–G5, the deprecation calendar, the unified-API specification.
- [`handlers.md`](handlers.md) — the handler lifecycle, AST-side vs host-side timings.
- [`fsm_design.md`](fsm_design.md) — patterns and anti-patterns for FSM JSON authoring.
- [`threat_model.md`](threat_model.md) — trust boundaries, threats, dismissed proposals.
- `src/fsm_llm/CLAUDE.md` and per-subpackage `CLAUDE.md` files — file maps and key classes for maintainers.
