# Refactoring Report: Re-rooting fsm_llm on the λ-Substrate

**Status**: Refactoring proposal (v1.0, 2026-04-27).
**Scope**: Architectural integration of `fsm_llm/lam/` (kernel) and the legacy FSM front-end (`api.py`, `fsm.py`, `pipeline.py`, `handlers.py`, …).
**Companion**: `docs/lambda.md` (the thesis — what was promised). This document is the audit (what was delivered) and the plan (how to close the gap).

---

## 0. Executive Summary

`docs/lambda.md` declares: *λ-calculus is the substrate; FSM is one front-end*. The kernel under `src/fsm_llm/lam/` (≈1.9 kLOC, 12 modules) is, in fact, a clean implementation of that substrate. The stdlib factories (`agents/`, `reasoning/`, `workflows/`, `long_context/`) honour the substrate's purity invariant.

But the FSM front-end **was not re-rooted on the substrate** — it was *bolted onto* it. M2 Slice 11 retired the duplicate `MessagePipeline.process` entry point, which collapsed the dual execution path; everything beyond that has stayed put. The result is a codebase where:

- The λ-runtime exists, is correct, and is universal — but `pipeline.py` is still 2,032 lines of FSM-specific callback machinery whose contents are conceptually `Leaf` nodes that *aren't* `Leaf` nodes.
- Handlers are invoked **outside** the compiled λ-term despite §6.3's claim of term-native composition.
- Two different code paths assemble prompts and call litellm (`LiteLLMInterface.generate_response/extract_field` for FSM; `LiteLLMOracle.invoke` for λ-DSL) for the same model.
- The user-facing `__init__.py` exports 90+ names, mostly FSM-flavoured. The kernel exports 29 names. There is no unified `Program` abstraction; users must know whether they want `API.converse(...)` or `Executor.run(...)`.
- The CLI surface (`fsm-llm`, `fsm-llm-validate`, `fsm-llm-visualize`) is FSM-only. There is no `lam-run` for stdlib factories or user-authored λ-DSL programs.

**Verdict**: clean substrate, asymmetric front-end. The fix is not to rewrite the kernel; it is to **collapse the FSM front-end onto the substrate it has already declared**. Concrete proposal below — 7 refactors, each independently shippable, ordered by payoff/risk.

---

## 1. The Seven Loci of Disjointedness

Each locus below names: *what the thesis says* (T), *what the code does* (C), *the gap* (G).

### L1. The Compile-Once Cache Lives in the Wrong Layer

- **T**: §6.1 — "Legacy FSM JSON is compiled at load time by a small, pure function."
- **C**: `FSMManager._compiled_terms: OrderedDict[str, Term]` — `fsm.py:119`. LRU cache (max 64), thread-safe (D-S7-03). Lookup at `fsm.py:188-221`.
- **G**: Compilation caching is a **kernel concern** that lives in the **FSM-frontend**. A future user calling `compile_fsm()` directly from a script gets no caching. `lam.fsm_compile.compile_fsm` is pure — good — but the natural memoization layer is missing from `lam/`.

### L2. MessagePipeline Is a Leaf-Library Pretending to Be a Pipeline

- **T**: §5 — "`Leaf` is the **only** node that invokes 𝓜. Everything else is symbolic. This is Assumption A2 encoded in the type system."
- **C**: `pipeline.py:151` — `MessagePipeline` is 2,032 LOC. Its 7 `_cb_*` callbacks (`_cb_extract`, `_cb_field_extract`, `_cb_class_extract`, `_cb_eval_transit`, `_cb_resolve_ambig`, `_cb_transit`, `_cb_respond`) each call `self.llm_interface.generate_response(...)` or `self.llm_interface.extract_field(...)`. The compiled FSM term is a `Case(state_id) → Let(c', App(cb_extract, ...), Let(s', App(cb_eval, ...), Let(o, App(cb_respond, ...), ...)))`. The `App` targets are **host-callable Vars** (D-003 in `fsm_compile.py:6-10`) — not `Leaf` nodes.
- **G**: The FSM-side "leaves" are not `Leaf` nodes. They are Python closures invoked through `App(Var(...))`. Two consequences:
  1. Assumption A2 (only `Leaf` invokes 𝓜) is **violated** — every `App` over a `cb_*` Var also invokes 𝓜, transitively.
  2. The kernel's per-`Leaf` cost telemetry, schema enforcement, and structured-decode bypass logic (`LiteLLMOracle._invoke_structured`, D-008) **does not apply** to FSM oracle calls. Theorem 2 (closed-form cost) holds for stdlib factories but is silently bypassed for FSM dialogs.

### L3. Handlers Are Cross-Cutting Middleware, Not Term-Native

- **T**: §6.3 — "The 8 `HandlerTiming` hooks become term transformers. […] composing it at the right point is a straightforward AST rewrite."
- **C**: 8 explicit `self.execute_handlers(HandlerTiming.X, …)` call-sites in `pipeline.py` (lines 291, 336, 371, 412, 629, 806, 1865, 1886). All execute *outside* the compiled term, in Python wrappers around `Executor.run()`. The `lam/` kernel has no `Handler` AST node; `lam/__init__.py` does not import from `fsm_llm.handlers`.
- **G**: §6.3 was promised; §6.3 was not delivered. Handlers remain Python middleware. This is the largest single source of "FSM and λ feel disjoined" — because they *are* disjoined here.

### L4. Two Oracle Paths, One LLM

- **T**: §10 — "Same code path for all three [Category A/B/C]. This is the payoff."
- **C**: Two prompt-assembly + LLM-invocation paths exist:

  | Path | Caller | Prompt builder | LLM method |
  |---|---|---|---|
  | FSM extract | `MessagePipeline._cb_extract` | `DataExtractionPromptBuilder` | `LiteLLMInterface.generate_response(ResponseGenerationRequest)` |
  | FSM field-extract | `_cb_field_extract` | `FieldExtractionPromptBuilder` | `LiteLLMInterface.extract_field(FieldExtractionRequest)` |
  | FSM respond | `_cb_respond` | `ResponseGenerationPromptBuilder` | `LiteLLMInterface.generate_response(...)` |
  | FSM classify | `_cb_class_extract` | `ClassificationPromptBuilder` | `LiteLLMInterface.generate_response(...)` |
  | λ-DSL leaf (unstructured) | `Leaf` evaluation | `template.format(**env)` (literal) | `LiteLLMInterface.generate_response(...)` (via oracle) |
  | λ-DSL leaf (structured) | `Leaf` with schema | `template.format(...)` | **raw `litellm.completion()`** with synthesised `required` (`oracle.py:241-256`, D-011) |

  Six paths, three call-shapes, one model. Same provider, six different request constructions.

- **G**: The promise of "one runtime" was kept at the *executor* layer; the promise of "one I/O surface to the LLM" was not kept anywhere. `LiteLLMOracle` is not a wrapper over `LiteLLMInterface`; it is a *peer*.

### L5. No Unified Program Abstraction

- **T**: §4 — "Single executor. […] Single planner. […] One runtime. Two surface syntaxes."
- **C**: Two parallel entry-point families that do not converge until the final `Executor.run` call:

  ```
  FSM:   API.from_file(path) → API.converse(msg, conv_id) → FSMManager.process_message → MessagePipeline.process_compiled → Executor.run
  λ-DSL: factory(args) → Term;   user constructs Executor(oracle=LiteLLMOracle(LiteLLMInterface(model))).run(term, env)
  ```

  There is no class that wraps `(term, oracle, optional_session)` and exposes a uniform `.run()` / `.converse()` surface. Stdlib factory examples in `examples/long_context/*/run.py` build the executor by hand each time.
- **G**: Users must know which surface they need *before* they pick an import. The "two surface syntaxes" the thesis promotes have no shared facade.

### L6. CLI is FSM-Only

- **C**: Five console scripts (`pyproject.toml`):
  - `fsm-llm` — interactive FSM runner.
  - `fsm-llm-validate` — operates directly on FSM JSON; **does not compile**.
  - `fsm-llm-visualize` — operates directly on FSM JSON; **does not compile**.
  - `fsm-llm-monitor` — web dashboard over running FSM conversations.
  - `fsm-llm-meta` — meta-builder; runs stdlib agents internally but is itself λ-DSL.
- **G**: No `fsm-llm run my_pipeline:react_term` style entry point for stdlib factories or user λ-DSL modules. No `fsm-llm explain <fsm.json>` that shows planner/cost output. Validator/visualizer's avoidance of compilation means they cannot detect problems that only surface after compilation (e.g., reserved-name collisions, schema mismatches).

### L7. Naming and Module-Layout Asymmetry

- **C**:
  - The kernel lives at `fsm_llm.lam` (a substrate-name as a sub-module).
  - The FSM front-end lives at top level (`fsm_llm.api`, `fsm_llm.fsm`, `fsm_llm.pipeline`, `fsm_llm.handlers`, `fsm_llm.classification`, `fsm_llm.transition_evaluator`).
  - The stdlib lives at `fsm_llm.stdlib.*` (correct).
  - `fsm_llm/__init__.py` exports 90+ names, dominated by FSM types (`FSMDefinition`, `FSMInstance`, `State`, `Transition`, `FSMContext`, `Conversation`, `HandlerSystem`, `HandlerBuilder`, …); 0 of the 90 are λ types.
  - `fsm_llm.lam.__init__.py` exports 29 names, all kernel.
  - `LLMInterface` ↔ `Oracle`; `extract_field` ↔ `invoke(schema=…)`; `extraction_instructions` (State) ↔ `template` (Leaf) — same concept, different vocabulary.
- **G**: A reader of the public surface has no way to tell that the substrate is the substrate. The package layout encodes the historical order ("FSM was here first") rather than the architectural truth ("λ is the substrate").

---

## 2. Target Architecture (Clean Levels)

The proposal is to flatten the conceptual hierarchy from "two front-ends each speaking to a kernel" into **four crisp layers**. Each layer is a strict client of the layer below.

```
┌──────────────────────────────────────────────────────────────────┐
│ L4  USER API                                                     │
│     fsm_llm.Program — single facade                              │
│       Program.from_fsm(fsm_def) → Program                        │
│       Program.from_term(term)   → Program                        │
│       Program.from_factory(fn, **kw) → Program                   │
│       .run(input) → output              [stateless]              │
│       .converse(msg, conv_id) → reply   [stateful, opt session]  │
│       .explain() → Plan + AST diagram                            │
│       .register_handler(handler)  [composes into AST at compile] │
└──────────────────────────────────────────────────────────────────┘
                                │
┌──────────────────────────────────────────────────────────────────┐
│ L3  FACTORIES — Term producers                                   │
│     fsm_llm.dialog.compile_fsm(fsm_def)        [FSM JSON → Term] │
│     fsm_llm.stdlib.agents.react_term(...)      [Category B]      │
│     fsm_llm.stdlib.long_context.niah(...)      [Category C]      │
│     fsm_llm.lam.dsl.{var,abs_,leaf,fix,...}    [user-authored]   │
│     All paths produce a closed `Term`. Period.                   │
└──────────────────────────────────────────────────────────────────┘
                                │
┌──────────────────────────────────────────────────────────────────┐
│ L2  HANDLERS / TRANSFORMS — pure AST→AST                         │
│     fsm_llm.handlers.compose(term, handlers) → Term              │
│     Each HandlerTiming becomes a wrapper combinator that         │
│     splices `let_("_h_x", App(handler_term, ctx), inner)` into   │
│     the right binding. No runtime middleware.                    │
└──────────────────────────────────────────────────────────────────┘
                                │
┌──────────────────────────────────────────────────────────────────┐
│ L1  RUNTIME — typed λ-substrate                                  │
│     fsm_llm.lam.{ast,dsl,combinators,planner,executor,           │
│                  cost,oracle,errors,constants}                   │
│     Executor   — β-reduction, depth-bounded                      │
│     Oracle     — single LLM adapter; ALL 𝓜 calls go here         │
│     Planner    — closed-form (k*, τ*, d, predicted_calls)        │
│     Cost       — per-Leaf telemetry (universal — works for FSM   │
│                  too once L3.compile_fsm emits real Leaves)      │
└──────────────────────────────────────────────────────────────────┘
```

### Layer invariants (enforceable by `make lint`)

- **L1 imports nothing from L2/L3/L4**. (Already true.)
- **L2 imports only from L1**. (New: `handlers.py` currently imports from `fsm` and `pipeline`.)
- **L3 factories import only from L1**. (Already true for stdlib; `compile_fsm` is the holdout — see R3.)
- **L4 (`Program`) imports from L1+L2+L3 but holds **no business logic** of its own**.

The `pipeline.py` module disappears. `MessagePipeline` is replaced by a ~150-LOC compile-time helper inside `dialog/compile_fsm.py`. The 8 `execute_handlers` call-sites become 8 AST-rewrite rules in `handlers.compose`.

---

## 3. The Seven Refactors (ranked by payoff/risk)

Each refactor is independently shippable, gated, and reversible. They are ordered so that landing R1–R3 alone already removes most of the user-visible disjointedness; R4–R7 are deeper but build on those foundations.

### R1. Introduce `Program` — the unified facade   (HIGH payoff, LOW risk)

**Why first**: closes L5 (no unified abstraction). Pure addition; nothing else changes.

```python
# src/fsm_llm/program.py  (new)

class Program:
    """Unified facade: a Term + an Oracle + optional Session."""

    def __init__(
        self,
        term: Term,
        *,
        oracle: Oracle | None = None,
        session: SessionStore | None = None,
        handlers: HandlerSystem | None = None,
    ): ...

    @classmethod
    def from_fsm(cls, fsm: FSMDefinition | str | Path, **kw) -> "Program":
        from fsm_llm.dialog import compile_fsm_cached
        return cls(compile_fsm_cached(load_fsm(fsm)), **kw)

    @classmethod
    def from_term(cls, term: Term, **kw) -> "Program":
        return cls(term, **kw)

    @classmethod
    def from_factory(cls, factory: Callable[..., Term], /, *args, **kw) -> "Program":
        return cls(factory(*args, **kw), **{k: kw.pop(k) for k in ("oracle", "session", "handlers") if k in kw})

    def run(self, **env) -> Any:
        """Stateless invocation: one β-reduction, no session."""

    def converse(self, message: str, conversation_id: str) -> str:
        """Stateful dialog: load session, β-reduce, save session."""

    def explain(self) -> ExplainOutput:
        """Return Plan + AST shape + per-Leaf schema map."""

    def register_handler(self, handler) -> None:
        """Adds the handler. The compiled term is rebuilt lazily on next .run/.converse."""
```

`API` becomes a thin back-compat shim: `API.converse → Program.converse`. Existing user code keeps working; new code uses `Program` directly. Net new LOC: ~200. Removed LOC: ~0 (additive).

### R2. Move the compile cache into the kernel   (HIGH payoff, LOW risk)

**Why second**: closes L1. Tiny mechanical move, removes one of `FSMManager`'s two reasons to exist.

```python
# src/fsm_llm/lam/fsm_compile.py  (extension)
@lru_cache(maxsize=64)
def _compile_fsm_by_id(fsm_id: str, fsm_json: str) -> Term: ...

def compile_fsm_cached(fsm: FSMDefinition) -> Term:
    return _compile_fsm_by_id(fsm.fsm_id, fsm.model_dump_json())
```

`FSMManager._compiled_terms` and the 30-odd lines of LRU bookkeeping in `fsm.py` go away. Threading: `lru_cache` is GIL-safe for our access pattern; the double-checked locking that `fsm.py:119-221` carries is a per-conversation-state lock, *not* a compile-cache lock — so removing the compile cache from `FSMManager` does not weaken correctness.

### R3. Unify the Oracle — `LiteLLMInterface` becomes a private detail of `LiteLLMOracle`   (HIGH payoff, MEDIUM risk)

**Why third**: closes L4. Reduces 6 LLM call-shapes to 1. Touches `pipeline.py` callbacks but does so *additively* — the callbacks invoke the oracle instead of the interface.

Concretely:

1. Move `LiteLLMInterface` to `fsm_llm.lam._litellm` (private). Keep its public class for back-compat re-export from `fsm_llm`.
2. Define `Oracle.invoke(template, *, env, schema)` as the one method that accepts a template + variables + optional schema — the union of `extract_field` and `generate_response`. Internal routing (structured vs. unstructured, D-008 bypass) stays inside `LiteLLMOracle`.
3. Rewrite each `_cb_*` in `pipeline.py` to call `self.oracle.invoke(...)` instead of `self.llm_interface.generate_response/extract_field`. The four `*PromptBuilder` classes (DataExtraction, ResponseGeneration, FieldExtraction, Classification) become *template producers* that emit the same `template`/`schema` pair a `Leaf` would carry.
4. After this refactor, the `Leaf-vs-callback` asymmetry from L2 is reduced to "is the prompt assembled inside a Python function or inside the AST?" — an addressable distinction (R5).

Risk is medium because every FSM dialog flows through these callbacks. Mitigation: the existing 688 core tests are the regression gate; T5 (semantic preservation) is what they enforce.

### R4. Module reorganisation — promote the substrate, demote the front-end   (HIGH payoff, MEDIUM risk)

**Why fourth**: closes L7. Renames are noisy but the change is mechanical.

```
BEFORE                              AFTER
fsm_llm/                            fsm_llm/
├── api.py                          ├── __init__.py        (re-exports Program + a few essentials)
├── fsm.py                          ├── program.py         (R1)
├── pipeline.py                     ├── runtime/           (was lam/)
├── handlers.py                     │   ├── ast.py
├── classification.py               │   ├── dsl.py
├── transition_evaluator.py         │   ├── executor.py
├── prompts.py                      │   ├── planner.py
├── llm.py                          │   ├── oracle.py
├── definitions.py                  │   ├── cost.py
├── session.py                      │   └── _litellm.py    (was llm.py)
├── memory.py / context.py          ├── dialog/            (was: api, fsm, pipeline, prompts, classification, transition_evaluator, definitions)
├── lam/                            │   ├── definitions.py
│   ├── ast.py / dsl.py / ...       │   ├── compile_fsm.py (was lam/fsm_compile.py)
│   ├── fsm_compile.py              │   ├── classification.py
│   └── oracle.py                   │   ├── transition_evaluator.py
├── stdlib/                         │   ├── prompts.py
└── ...                             │   └── session.py
                                    ├── handlers.py        (now AST transformer — R5)
                                    ├── stdlib/            (unchanged)
                                    └── memory.py / context.py
```

The `lam` → `runtime` rename is the headline change: the substrate gets a name that says "I am the substrate". `dialog/` collects everything that is FSM-specific (which is what it has always been). `MessagePipeline` does not reappear — its body has been absorbed into `dialog/compile_fsm.py` and `handlers.py` per R3 + R5.

Back-compat: `from fsm_llm.lam import …` and `from fsm_llm.api import API` keep working through `sys.modules` shims for one minor version, then warn, then remove in 0.5.0. Same pattern already used for `fsm_llm_reasoning` / `fsm_llm_workflows` / `fsm_llm_agents`.

### R5. Handlers as AST transformers   (HIGH payoff, MEDIUM risk) — **SHIPPED in 0.4.0 (`r5-green`, plan v1) — refinement scope ARCHITECTURALLY HOST-SIDE (plan v2 D-S1-02, 2026-04-27)**

> **Status (2026-04-27, plan_2026-04-27_1b5c3b2f)**: R5-narrowed remains the ceiling. The follow-up "splicer extension to PRE_TRANSITION + POST_TRANSITION + CONTEXT_UPDATE" was investigated in plan v2 and **falsified at gate-before-edits**: each of the 3 deferred timings has structural barriers to clean term-side splicing.
>
> **Architecturally infeasible host-side (3 timings)** — these timings stay host-side permanently, NOT due to a kernel limitation alone but because the cardinality + conditional-firing semantics they require cannot be matched by a flat structural splicer:
>
> - **POST_TRANSITION**: `_execute_state_transition` rolls back state mutation on POST_TRANSITION handler failure (`pipeline.py:2049-2059`). The kernel has no exception combinator; rollback semantics cannot be expressed term-side.
> - **CONTEXT_UPDATE**: dual-fire per turn (bulk-extract `pipeline.py:785-791` + post-transition re-extract `pipeline.py:962-968`), each with its own `updated_keys` set. Single env binding can't carry per-Let key-sets without structural changes.
> - **PRE_TRANSITION**: fires inside `_execute_state_transition`, called from BOTH the deterministic path (`pipeline.py:898`) AND the AMBIGUOUS-resolved path (`pipeline.py:850`, gated by `if target and target != instance.current_state`). Cardinality is (DETERMINISTIC) ∪ (AMBIGUOUS-resolved-to-valid-target), zero on (BLOCKED) ∪ (AMBIGUOUS-resolved-to-fallback) ∪ (AMBIGUOUS-resolved-to-same-state). A flat term-splice cannot honour this — splicing on "advanced" branch only under-fires; funneling through CB_TRANSIT + unconditional splice over-fires. A conditional `Case(host_call("_has_transition_target", ...), {...})` splice would work but requires HOST_CALL Case-gating machinery out of plan scope.
>
> **The shipped 2-of-8 splice surface remains the practical ceiling.** All 8 timings still route through one `make_handler_runner` so execution semantics (priority order, error_mode, timeout, `should_execute`) are uniform regardless of call-site location. See `plans/plan_2026-04-27_1b5c3b2f/decisions.md` D-S1-02.

**Why fifth**: closes L3. Delivers §6.3 for real. Removes 8 `execute_handlers` call-sites and the `MessagePipeline` Python-middleware role.

Each `HandlerTiming` maps to an AST splice point. Sketch:

```python
# src/fsm_llm/handlers.py  (post-refactor — AST mode)

def compose(term: Term, handlers: list[Handler]) -> Term:
    """Pure AST→AST. Wraps `term` with handler invocations at the appropriate seams."""
    for h in handlers:
        term = _splice(h.timing, term, h)
    return term

_SPLICES: dict[HandlerTiming, Callable[[Term, Handler], Term]] = {
    HandlerTiming.PRE_PROCESSING:  _wrap_outermost_let,
    HandlerTiming.POST_PROCESSING: _wrap_outermost_let_post,
    HandlerTiming.PRE_TRANSITION:  _wrap_transition_case,
    HandlerTiming.POST_TRANSITION: _wrap_transition_case_post,
    HandlerTiming.CONTEXT_UPDATE:  _wrap_each_let,
    HandlerTiming.START_CONVERSATION: _wrap_program,
    HandlerTiming.END_CONVERSATION:   _wrap_program_post,
    HandlerTiming.ERROR:           _wrap_with_catch,   # executor-level
}
```

A handler's body is a Python function `(ctx) → ctx`. To make it AST-native, we wrap it in a `Leaf` variant that does **not** invoke 𝓜 — instead, it invokes a registered host-callable (a `Combinator(op=HOST_CALL, args=[...])` — the one new closed-set op this refactor adds). This is the same pattern stdlib already uses for `oracle_compare_op` (LESSONS.md "new op via env"). Cost: one new `CombinatorOp` value (`HOST_CALL`) — a controlled extension of the closed set, justified because it is the only mechanism we accept for non-𝓜 side effects.

After R5, `pipeline.py` exists in the diff only as a renamed `dialog/compile_fsm_body.py` with all callbacks replaced by AST builders. Estimated LOC reduction: 2,032 → ~400.

### R6. Lift FSM callbacks to first-class `Leaf` nodes   (HIGH payoff, HIGH risk) — **SHIPPED narrow-cohort in 0.4.x (`r6-green`, plan_2026-04-27_1b5c3b2f, 2026-04-27)**

> **Status (2026-04-27, plan_2026-04-27_1b5c3b2f)**: shipped for the **terminal-cohort** (response-only states with no extractions) under **opt-in gate `FSM_LLM_COHORT_EMISSION=1`**. Theorem-2 strict equality `Executor.oracle_calls == plan(...).predicted_calls` holds for cohort states. Default-OFF preserves byte-equivalent legacy behavior; production rollout to default-ON deferred to a future plan once production validation completes.
>
> **What shipped**:
> - `dialog/prompts.py`: `ResponseGenerationPromptBuilder.to_compile_time_template((state, fsm_def)) -> (template, input_vars, schema_ref)` additive compile-time emitter. Degenerate single-placeholder design (D-S1-03): template = `"{response_prompt_rendered}"`, runtime pre-renders the full prompt and binds it.
> - `dialog/compile_fsm.py`: `_is_cohort_state(state, fsm_def)` predicate; cohort terminal states emit `Leaf` instead of `App(CB_RESPOND, instance)`. `COHORT_RESPONSE_PROMPT_VAR` reserved env name. `FSM_LLM_COHORT_EMISSION` env-var gate.
> - `dialog/pipeline.py`: cohort env binding via `build_response_prompt`; Executor wired with `LiteLLMOracle(self.llm_interface)`.
> - `runtime/planner.py`: `PlanInputs.fmap_leaf_count` additive extension (default 0 → backward-compatible).
> - 49 new tests across `test_prompt_bytes_parity.py` (17), `test_compile_fsm_cohort.py` (13), `test_planner_fmap.py` (12), `test_compiler_theorem2.py` (7).
>
> **Cohort definition (terminal-only v1)** — a state is in the cohort iff:
> ```python
> not state.transitions
> and not state.required_context_keys
> and not state.field_extractions
> and not state.classification_extractions
> and not (state.extraction_instructions and state.extraction_instructions.strip())
> ```
>
> **Coverage boundary (architecturally documented)**: Theorem-2 universality across ALL FSM states is **architecturally impossible**. Three reasons:
> 1. The `skip-if-in-context` filter (`pipeline.py:1373-1374`) makes oracle-call count turn-state-dependent; this filter is fundamental to FSM stateful design.
> 2. `extraction_retries > 0` is LLM-output-dependent (number of retries varies based on what the LLM returns).
> 3. `_cb_extract`'s multi-call orchestration (per-field `fmap` + per-classification `fmap` + retry `Fix` + bulk-vs-field `Case`) cannot be flattened to a single Leaf without significant kernel-design work — including a `Fix` retry encoding the planner can score, which depends on runtime-dependent termination.
>
> **What's deferred** to a future plan:
> - **Non-cohort states** keep the legacy host-callback path. R6.x widening (e.g. cohort states with simple deterministic transitions; cohort states with `field_extractions` and `extraction_retries == 0`) is a clear follow-up.
> - **Schema enforcement**: cohort Leaf currently uses `schema_ref=None` to preserve `CB_RESPOND`'s string-returning contract. Pydantic decoding via `ResponseGenerationResponse` requires pipeline output-unwrap support (~+50 LOC).
> - **Default-ON gate flip**: production validation under representative FSM load + a documented benchmark gating, then flip default to ON.
>
> Stdlib factories (Category B/C) continue to satisfy Theorem-2 unchanged. See `plans/plan_2026-04-27_1b5c3b2f/decisions.md` D-S1-03 for the degenerate single-placeholder rationale and D-S1-04 for the opt-in gate.

**Why sixth**: closes L2 — the last and largest piece. Makes Theorem 2's cost model apply to FSM dialogs too.

The current `_cb_extract` is conceptually:

```
Leaf(
  oracle="default",
  template=DataExtractionPromptBuilder.render(state.extraction_instructions, ...),
  input_var="message",
  extra_input_vars=("context", "instance"),
  schema=schema_for(state.classification_extractions, state.field_extractions),
)
```

The reason D-003 (`fsm_compile.py:6-10`) chose host-callable Vars instead is that **the prompt template depends on runtime instance state** (the per-state extraction instructions), which a static AST cannot inline. The fix is **per-state Leaf specialisation at compile time**: `compile_fsm` emits one `Leaf` per `(state_id, role)` pair, with the template materialised from the FSM definition. Runtime variability that genuinely depends on per-message data (current context dict, recent conversation) is threaded through `extra_input_vars` — exactly what `Leaf` was designed for.

The risk is high because:
- Schema construction is per-state — the compiler must know how to build a Pydantic model from `(field_extractions, classification_extractions)` for every state.
- Field-extraction loops (a state with multiple `field_extractions`) become `fmap` over a list of Leaves — a structural change that the test suite will exercise heavily.
- `_cb_resolve_ambig` (the Classifier-fallback path) is a *conditional* Leaf — `Case` over `(DETERMINISTIC | AMBIGUOUS | BLOCKED)` with the AMBIGUOUS branch holding a real `Leaf`. Already supported by `Case` semantics; just needs careful compilation.

Payoff: T2 (closed-form cost) becomes universal. Per-Leaf cost telemetry, currently silent for FSM dialogs, lights up. The 5–10% planner-vs-actual cost gap that the bench scorecards record on stdlib factories becomes the gap for *every* fsm_llm program. And `pipeline.py`'s callback machinery — kept alive by R3, slimmed by R5 — finally goes to zero.

### R7. CLI unification   (MEDIUM payoff, LOW risk) — **SHIPPED in 0.4.0 (`r7-green`, plan v1)**

> **Status (2026-04-27)**: shipped. The unified `fsm-llm` binary exposes 6 subcommands (`run / explain / validate / visualize / meta / monitor`); the legacy 4 console scripts (`fsm-llm-validate`, `fsm-llm-visualize`, `fsm-llm-meta`, `fsm-llm-monitor`) continue to work as aliases per D-PLAN-04 (silent in 0.4.x; deprecation in 0.5.0; removal in 0.6.0). `Program.explain(n=…, K=…, plan_kwargs=…)` is the new kw-only overload that populates `plans` per `Fix` subtree; FSM-mode programs return `plans=[]` until R6 ships (no `Fix` subtrees in compiled FSMs today). 45 new tests in `tests/test_fsm_llm/test_cli_unified.py`.

**Why last**: closes L6. Easy once R1 (`Program`) lands.

```
fsm-llm run <target>            # auto-detect: *.json → from_fsm; pkg:factory → from_factory
fsm-llm explain <target>        # show plan(...) + ASCII AST + per-Leaf schemas
fsm-llm validate <target>       # compile + validate (replaces current validator)
fsm-llm visualize <target>      # ASCII for both FSM and λ-DSL (was visualizer)
fsm-llm meta                    # unchanged (already λ-native)
fsm-llm monitor                 # unchanged (web dashboard)
```

`run`/`explain`/`validate`/`visualize` are subcommands of one binary; the old per-binary console scripts are kept as aliases for one minor version, then removed.

---

## 4. Migration Sequencing

Each refactor lands as one PR with green CI. Tests are the contract.

| # | Refactor | Touches | Removes | Adds | Tests gate |
|---|---|---|---|---|---|
| R1 | `Program` facade | new file `program.py` | 0 LOC | ~200 LOC | New `tests/test_program/` (~30 tests) + existing 688 core via `API → Program` shim |
| R2 | Compile-cache to kernel | `lam/fsm_compile.py`, `fsm.py` | ~30 LOC | ~20 LOC | `tests/test_fsm_llm_lam/test_fsm_compile.py` |
| R3 | Unify oracle | `lam/oracle.py`, `pipeline.py`, `prompts.py` | ~50 LOC | ~80 LOC | All 688 core (T5 semantic preservation) |
| R4 | Module reorganisation | rename `lam/` → `runtime/`, split `api+fsm+pipeline+...` → `dialog/` | 0 LOC net | ~50 LOC of shims | All tests (mechanical) |
| R5 | Handlers as AST transformers | `handlers.py`, `pipeline.py` (gutted) | ~400 LOC | ~250 LOC | All 688 core + new `tests/test_handlers_ast/` |
| R6 | Lift callbacks to Leaves | `dialog/compile_fsm.py`, kill `pipeline.py` | ~1600 LOC | ~500 LOC | All 688 core + new per-state Leaf cost assertions |
| R7 | CLI unification | `__main__.py`, `runner.py`, `validator.py`, `visualizer.py` | ~50 LOC | ~150 LOC | New `tests/test_cli/` (~20 tests) |

**Net delta after all seven refactors**: roughly **−1,400 LOC** in the FSM front-end, **+700 LOC** of cleaner abstractions, **+50 tests**. Of the deleted lines, ~1,200 come from `pipeline.py`'s callback bodies after R6 absorbs them into AST-emit code in `dialog/compile_fsm.py`.

**Total elapsed time at ~1 PR/week**: 7 weeks. R1–R3 (3 weeks) yield most of the user-visible disjointedness fix; R4–R7 are deeper but lower-risk on top.

---

## 5. What This Refactor Does Not Do

- **Does not change FSM JSON v4.1 semantics.** T5 (`docs/lambda.md` §12) is the contract.
- **Does not deprecate FSMs.** Category-A dialogs remain a first-class surface; the `Program.from_fsm(...)` path is permanent.
- **Does not introduce full System F typing.** Per §14: monomorphisation at parse time stays.
- **Does not touch the planner or its theorems.** Planner is already universal; R6 is what makes it *applied* universally.
- **Does not change the public `API` class signature.** It becomes a 50-line back-compat shim over `Program`. Existing user code is unchanged.
- **Does not require the M5 long-context work to land first.** Independent.

---

## 6. Validation Plan

For each refactor:

1. **Test**: existing test count must not drop; new tests must accompany the new abstraction.
2. **Cost**: bench scorecards (`evaluation/bench_long_context_*.json`) must not regress on Theorem-2 holds.
3. **Performance**: `pytest tests/test_fsm_llm/` wall-clock must stay within +10% of baseline. AST overhead is real but bounded; if it exceeds 10%, R6's per-state `Leaf` specialisation needs the term-cache hint that `fsm.py:188-221` is currently doing manually.
4. **Public API**: `python -c "from fsm_llm import API, Program; …"` works after every refactor. Back-compat shims are tested.
5. **Examples**: all 152 examples (`examples/`) green via `scripts/eval.py`. The 90.8% baseline (Run 004, 2026-04-02) is the floor.

After R7, `docs/lambda.md` should be re-read with this report and any §s that R5/R6 turned from aspirational to concrete should have their language tightened. Specifically: §6.3 ("hooks become term transformers") moves from future tense to present.

---

## 7. Summary

The codebase has a clean λ-substrate. It also has a 2,032-line `pipeline.py` whose contents are conceptually `Leaf` nodes, an 8-callsite handler middleware that the thesis already declared should be AST transforms, and a public surface where the substrate is buried inside `fsm_llm.lam` while FSM types fill the top level. None of these is a bug; each is a point where the FSM front-end did not finish moving onto the substrate it has officially endorsed.

The fix is seven refactors, ordered by payoff/risk, each independently shippable, with the 688-test core suite as the regression gate. After they land:

- One facade (`Program`), one compile cache, one Oracle.
- Handlers compose into the term, not around it.
- FSM callbacks are real `Leaf` nodes; Theorem 2's cost model is universal.
- The substrate is named `runtime/`; the front-end is named `dialog/`; the public surface tells the truth.

One runtime. Two surface syntaxes. **And every line of code that handles them looks like it.**

---

## References

- `docs/lambda.md` — the architectural thesis. This document is the audit + plan.
- `src/fsm_llm/lam/CLAUDE.md` — the kernel's file map.
- `src/fsm_llm/CLAUDE.md` — the package map (post-this-refactor: needs revision).
- `plans/LESSONS.md` — D-003, D-008, D-011, "AST attribute names matter", "Host-callable orchestrator with `_eval` bypass", "new op via env".
- `evaluation/bench_long_context_*.json` — Theorem-2 evidence per (model × factory) cell (currently stdlib-only; R6 makes this also FSM).
