# Architecture

FSM-LLM is a **typed λ-calculus runtime** with two surface syntaxes (FSM JSON and λ-DSL) and one verb (`Program.invoke`). This page is a tour of how the pieces fit together.

For the architectural thesis (why λ-calculus, Theorems 1–5), see [`docs/lambda.md`](lambda.md). For the merge contract (invariants, falsification gates, deprecation calendar), see [`docs/lambda_fsm_merge.md`](lambda_fsm_merge.md).

## Layered architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ L4  INVOKE     Program.invoke(message=|inputs=, *, ...) → Result     │
│                  one verb · construction-time mode · one error class │
├──────────────────────────────────────────────────────────────────────┤
│ L3  AUTHOR     Term producers — pure functions returning Term         │
│                  compile_fsm(defn) | react_term(...) | niah(...)      │
│                  | leaf | fix | let_ | case_ | abs_ | app | var       │
├──────────────────────────────────────────────────────────────────────┤
│ L2  COMPOSE    Pure AST→AST transforms                                │
│                  compose(term, [handler, instrument, …])              │
├──────────────────────────────────────────────────────────────────────┤
│ L1  REDUCE     Typed substrate                                        │
│                  Term · Executor · Plan · Oracle · CostAccumulator    │
└──────────────────────────────────────────────────────────────────────┘
```

A reader of a single `from fsm_llm import …` line can identify their layer. The package surface tells the architecture, not the historical order. An import-audit test (`tests/test_fsm_llm/test_layering.py`) enforces L1→L4 ordering on every module.

## The substrate (L1) — `fsm_llm.runtime`

The kernel is a typed λ-AST plus an evaluator. Both surfaces compile to the same `Term` and both run on the same `Executor`.

| File | Role |
|---|---|
| `runtime/ast.py` | `Var`, `Abs`, `App`, `Let`, `Case`, `Combinator`, `Fix`, `Leaf`, `Term`. Pydantic v2 frozen models with `model_rebuild()` for the recursive types. |
| `runtime/dsl.py` | `var`, `abs_`, `app`, `let_`, `case_`, `fix`, `leaf`, `split`, `peek`, `fmap`, `ffilter`, `reduce_`, `concat`, `cross` |
| `runtime/combinators.py` | `ReduceOp` enum + `BUILTIN_OPS` registry (closed) |
| `runtime/executor.py` | `Executor` — β-reduction, depth-bounded, per-Leaf cost; `Executor.run(term, env, *, stream=False)` |
| `runtime/planner.py` | `plan()`, `Plan`, `PlanInputs` — closed-form `(k*, τ*, d, predicted_calls)` per `Fix` subtree |
| `runtime/oracle.py` | `Oracle` Protocol (5 call shapes), `LiteLLMOracle` adapter, `StreamingOracle` Protocol |
| `runtime/_litellm.py` | `LLMInterface` ABC + `LiteLLMInterface` (litellm; 100+ providers) |
| `runtime/cost.py` | `CostAccumulator`, `LeafCall` |
| `runtime/errors.py` | `LambdaError` → `ASTConstructionError`, `TerminationError`, `PlanningError`, `OracleError` |

The kernel is **closed against `dialog/`** — `runtime/*` does not import from the FSM front-end (the only exception is `runtime/oracle.py` and `runtime/_litellm.py` reading request/response Pydantic models from `dialog/definitions.py`; they're leaves on the dialog side).

## Theorem-2 — predictable cost

For every `Fix` node, the planner pre-computes `(k*, τ*, d, predicted_calls)` from the AST shape. The executor then delivers exactly that many oracle calls when input is τ·k^d-aligned:

```python
ex = Executor(oracle=...)
ex.run(term, env=env)
assert ex.oracle_calls == plan(...).predicted_calls   # strict equality
```

**Universality**: post-A.M3c (2026-04-29), Theorem-2 holds **by default for non-terminal FSM programs** as well — every non-terminal non-cohort state lifts its response generation to a real `Leaf` so the executor's `CostAccumulator` sees it. The single residual caveat is terminal non-cohort states without `output_schema_ref`, which still use the host-callable `CB_RESPOND` fallback. See `docs/lambda_fsm_merge.md` §3 (I6) for the full statement.

Long-context demos under `examples/long_context/` and the bench scorecards under `evaluation/` carry hard `oracle_calls_match_planner` assertions; cost regressions fail CI.

## The dialog front-end — `fsm_llm.dialog`

```
dialog/
├── api.py              # API class (legacy entry; routed through compiled term cache)
├── fsm.py              # FSMManager — per-conversation locks
├── turn.py             # MessagePipeline — compiled-path 2-pass body (was pipeline.py pre-R13)
├── prompts.py          # DataExtractionPromptBuilder, ResponseGenerationPromptBuilder, FieldExtractionPromptBuilder
├── classification.py   # Classifier, HierarchicalClassifier, IntentRouter
├── transition_evaluator.py  # DETERMINISTIC | AMBIGUOUS | BLOCKED
├── definitions.py      # Pydantic models: State, Transition, FSMDefinition, FSMContext, FSMInstance, Conversation
├── session.py          # SessionStore ABC + FileSessionStore (atomic writes)
└── compile_fsm.py      # compile_fsm() : FSMDefinition → Term + lru_cache(64) compile_fsm_cached
```

### One conversation turn

`Program.invoke(message=...)` → `API.converse(...)` → `FSMManager.process_message(...)` (acquires per-conversation lock) → `MessagePipeline` runs the **compiled λ-term body** on the executor:

```
Pass 1 (extract + transition):
  PRE_PROCESSING handlers
  → Oracle.invoke(extraction_prompt, schema=…)         [one Leaf eval]
  → CONTEXT_UPDATE handlers (when keys changed)
  → TransitionEvaluator: DETERMINISTIC | AMBIGUOUS | BLOCKED
  → If transitioning: PRE_TRANSITION → state change → POST_TRANSITION

Pass 2 (response):
  POST_PROCESSING handlers
  → Oracle.invoke(response_prompt[, schema_ref])       [one Leaf eval]
  → return reply
```

The 2-pass shape is preserved as the **body** of the compiled term — but `MessagePipeline.process` and `process_stream` are internal post-M2 S11. There is one execution path.

### Compiled-term cache

`compile_fsm_cached(fsm, fsm_id)` is `lru_cache(maxsize=64)` keyed on `(fsm_id, fsm.model_dump_json())`. `FSMManager.get_compiled_term` is a thin shim. Compile happens at load time; subsequent turns are pure β-reduction on the cached term.

## The standard library (L3) — `fsm_llm.stdlib`

Named λ-term factories organised by domain:

| Subpackage | Factories |
|---|---|
| `stdlib.agents` | `react_term`, `rewoo_term`, `reflexion_term`, `memory_term` (+ 12 class agents) |
| `stdlib.reasoning` | 11 strategy factories + `classifier_term` + `solve_term` (+ `ReasoningEngine`) |
| `stdlib.workflows` | `linear_term`, `branch_term`, `switch_term`, `parallel_term`, `retry_term` (+ `WorkflowEngine`) |
| `stdlib.long_context` | `niah`, `aggregate`, `pairwise`, `multi_hop`, `niah_padded` |

**Stdlib purity invariant**: every `lam_factories.py` module imports **only from `fsm_llm.runtime`** (or its `fsm_llm.lam` shim). AST-walk unit tests enforce this per subpackage.

## Handlers — `fsm_llm.handlers`

Eight timing points: `START_CONVERSATION`, `PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `END_CONVERSATION`, `ERROR`.

- **AST-side timings** (2 of 8): `PRE_PROCESSING` and `POST_PROCESSING` are spliced into the term via `compose(term, handlers)` (L2). They become real reduction steps.
- **Host-side timings** (6 of 8): the rest stay outside the AST permanently per merge spec §8 — they hook into the host-side conversation loop in `dialog/api.py` / `dialog/fsm.py`. This is the final word; not deferred.

`HandlerBuilder` provides a fluent surface; `HandlerSystem` orchestrates execution by priority (lower runs first, default 100). Error mode: `"continue"` (skip failed) or `"raise"` (propagate). Critical handlers always raise.

## Sessions — `fsm_llm.dialog.session`

`SessionStore` ABC + `FileSessionStore` (JSON files via atomic temp+rename). `SessionState` is a Pydantic model: `conversation_id`, `fsm_id`, `current_state`, `context_data`, `conversation_history`, `stack_depth`, `saved_at`, `metadata`. Path-traversal protection on session IDs.

## Context — `fsm_llm.context`, `fsm_llm.memory`

```python
class FSMContext:
    data: dict                 # User-defined (extracted data, handler outputs)
    conversation: Conversation # Message history with max_history_size
    metadata: dict
    working_memory: WorkingMemory  # 4 named buffers (core, scratch, environment, reasoning)
```

Internal keys (prefixed `_`, `system_`, `internal_`, `__`): `_conversation_id`, `_current_state`, `_previous_state`, `_timestamp`, `_user_input`. Stripped by `clean_context_keys()` from user-visible context.

`ContextCompactor`: `compact(ctx)` (clear transient keys), `prune(ctx)` (on transition), `summarize(conversation)`.

### FSM stacking

`API.push_fsm(conv_id, child)` pushes a child FSM onto the stack with `context_to_pass` and `shared_context_keys`. `pop_fsm(conv_id)` merges back via strategy:

- `UPDATE` — child overwrites parent
- `PRESERVE` — only new keys added to parent

## Security

- **Prompt injection prevention**: XML tag sanitization on all user input
- **Context isolation**: per-conversation isolated state, no cross-conversation access
- **Input validation**: length limits, sanitization
- **Forbidden patterns**: regex filtering for passwords, secrets, API keys, tokens
- **Internal key prefixes**: stripped from user-facing context

## Performance

- **Conversation history limits**: `max_history_size` (default 5), `max_message_length` (default 1000)
- **FSM definition caching**: LRU cache (max 64) for loaded definitions; compiled-term cache same size
- **Handler pre-filtering**: only check handlers matching current timing
- **Thread safety**: per-conversation `RLock`s in `FSMManager`

## The merge contract — six invariants

These are the falsification-gated promises the package makes (`docs/lambda_fsm_merge.md` §3):

- **I1** — every LLM call goes through `Oracle`; no fifth method, no escape hatch.
- **I2** — `Program` owns exactly one `Oracle`; dialog and term paths read from it.
- **I3** — mode is fixed at construction; `.invoke` cannot switch modes at runtime.
- **I4** — layering is enforced by an import-audit AST walk, not convention.
- **I5** — back-compat surfaces follow a two-epoch deprecation calendar (R13: warning since 0.3.0, removal in 0.6.0; I5: silent in 0.5.x, warning in 0.6.0, removal in 0.7.0).
- **I6** — Theorem-2 holds for every Program a user can construct from public APIs.

CI tests `G1`–`G5` enforce these. New features that would violate an invariant fail loudly at the gate.

## Extension integration

| Package | Mechanism |
|---|---|
| Classification | Built into core, LLM-backed via litellm |
| Reasoning | FSM stacking — orchestrator pushes strategy FSMs onto the stack |
| Workflows | Async engine + `ConversationStep` (creates a `Program` for FSM conversations) |
| Agents | Auto-generated FSMs + handlers (tool execution at `POST_TRANSITION`) |
| Monitor | Observer handlers at all 8 timings (priority 9999), never modifies state |

Extension subpackages all live under `fsm_llm.stdlib.*`. The legacy top-level `fsm_llm_*` names are silent shims in 0.5.x.
