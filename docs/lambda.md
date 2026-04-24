# λ-Calculus as the fsm_llm Substrate

**Status**: Architectural thesis (v0.2, 2026-04-24).
**Source**: Distilled from Roy, Tutunov, Ji, Zimmer, Bou-Ammar, *The Y-Combinator
for LLMs: Solving Long-Context Rot with λ-Calculus* (arXiv:2603.20105, Mar 2026),
fused with the existing fsm_llm architecture.

This document proposes **λ-calculus as the primary substrate of fsm_llm**. FSMs
remain — but as *one surface syntax* over that substrate, not the substrate itself.
Most programs that today ceremonially wrap themselves in FSM JSON do not actually
need dialog state, and should be expressed directly as λ-terms. Programs that
*do* need dialog state continue to work unchanged: the existing FSM JSON compiles
down to λ-terms and runs on the same executor as native λ-programs.

One runtime. Two surface syntaxes. FSMs where they earn their keep; λ-terms
everywhere else.

---

## 1. Thesis

> **Every fsm_llm program is already a λ-term.** The FSM JSON is a serialisation
> of a per-turn function `step : (state, input, context) → (state', output, context')`
> with a switch on `state_id` — which is precisely a simply-typed λ-expression with
> a case analysis. The runtime never loops; the host application does, via
> `converse()`. Making λ-terms primary lets us:
>
> 1. **Delete FSM overhead from programs that don't need state.** Most pipelines,
>    agentic patterns, and one-shot tasks are stateless — the FSM wrapping is
>    ceremony.
> 2. **Unify the five sub-packages.** `fsm_llm_reasoning`, `fsm_llm_workflows`,
>    `fsm_llm_agents` all become named combinator patterns in a single standard
>    library, not separate execution engines.
> 3. **Inherit the paper's theorems** (termination, closed-form cost, polynomial
>    accuracy decay, optimal k*=2) for every program, not just the ones that opt
>    into a special "λ-substrate" extension.
> 4. **Preserve every existing program.** The FSM JSON compiler runs before M1
>    ships; no example, no handler, no production deployment breaks.

The design move is not "add λ-substrate alongside FSM". It is: **the runtime is a
typed λ-term interpreter; FSM JSON is one front-end; a direct λ-DSL is another.**

---

## 2. An FSM *is* a λ-term — constructively

Let *S* be the set of state IDs, *I* the input alphabet (user messages), *O* the
output alphabet (assistant responses), *C* a context dict. The current fsm_llm
runtime executes this function per `converse()` call:

$$
\text{step} : S \times I \times C \to S \times O \times C
$$

In the existing 2-pass architecture this decomposes as:

$$
\text{step}(s, i, c) = \big(\delta(s, c'),\; \rho(s', c')\big)\quad\text{where}\quad c' = \text{extract}_s(i, c)
$$

- `extract_s` — Pass 1 data extraction at state *s* (leaf β-reduction of 𝓜)
- `δ`  — transition evaluator (deterministic over *c*′; ambiguity resolved by a
  bounded `Classifier` call, which is itself a leaf β-reduction)
- `ρ`  — Pass 2 response generation at the post-transition state (leaf β-reduction)

Every one of these is a function; their composition is a λ-term. Writing it out
in combinators (where `[[s]]` denotes state *s*'s specification):

```
step = λ(s, i, c).
  let c' = extract_leaf([[s]].extraction_instructions, i ⊕ c)  -- one 𝓜 call
  let s' = dispatch(c', [[s]].transitions)                     -- pure + optional 𝓜 via Classifier
  let o  = response_leaf([[s']].response_instructions, c')     -- one 𝓜 call
  in (s', o, c')
```

Host-side recursion (the conversation loop) is external:

```
conversation = fix(λloop. λ(s, c).
  let i = await_user()
  let (s', o, c') = step(s, i, c)
  emit(o)
  if [[s']].terminal then halt(c') else loop(s', c'))
```

So `converse()` *is* one β-reduction step of the conversational fix. The FSM JSON
is a serialisation of the `[[s]]` table and the `δ` dispatch. Everything runs in
λ already; we have just not named it that.

**Consequence**: expressing a program directly as a λ-term (no FSM JSON) is a
strict simplification for any program where the switch-on-`s` is degenerate (one
state, or linear transitions). That turns out to be most of `examples/`.

---

## 3. Taxonomy: when FSMs earn their keep, when they don't

We audit the 80 examples against the question "does this program need a runtime
switch on persistent dialog state?". Three categories emerge.

### Category A — Genuinely stateful dialog (FSM surface is the right fit)

State must persist **across user turns**, and the transition graph is genuinely
non-linear (user can redirect mid-conversation).

Examples: `form_filling`, `adaptive_quiz`, `e_commerce` (FSM stacking used), most
dialog bots in `examples/basic` and `examples/intermediate`, `smart_helpdesk`,
the `classification/intent_routing` family.

**Recommendation**: keep FSM JSON. The compiler (§6) translates to λ; nothing
else changes for these.

### Category B — Pipelines with implicit state (state is fake)

The "state" is really just a stage in a linear pipeline. No user re-entry, no
branching on persistent context. The FSM is ceremony for what is structurally a
function composition.

Examples: most of `examples/agents/*` — `prompt_chain`, `plan_execute`,
`evaluator_optimizer`, `react_search`, `reflexion`, `debate`, `maker_checker`,
`rewoo`, `orchestrator`, `eval_opt_structured`, etc. (roughly 30 of 48 agent
examples). Also `basic/story_time`, `basic/simple_greeting`, most of
`examples/reasoning`.

**Recommendation**: express directly as λ-terms. Keep FSM version as a
compatibility shim during migration; remove after M4.

### Category C — Recursive decomposition over long inputs (λ-native from day one)

The paper's home turf. Long-document QA, aggregation over large corpora,
pairwise comparison, multi-hop retrieval. fsm_llm currently has **no good
expression** of these — `push_fsm` stacking approximates bounded recursion but
gives no termination proof, no cost model, no optimal partition.

Examples: none today; likely additions per paper benchmarks — S-NIAH, OOLONG,
OOL-Pairs, CodeQA-style tasks.

**Recommendation**: express as λ-terms with `fix` and the combinator library.
This is where the paper's +21.9 pp / 4.1× wins live.

**Rough split**: A ≈ 20 examples, B ≈ 50, C ≈ 10 (new). The substrate change
shrinks the FSM-authoring surface by ~75% without breaking anything in the
remaining 25%.

---

## 4. Unified architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ SURFACE  (authoring)                                               │
│                                                                    │
│   FSM JSON v4.1       λ-DSL (Python)       λ-JSON (serialised)     │
│   (legacy,            (preferred for       (portable,              │
│    Category A)         Category B/C)        optional)              │
│          │                     │                   │               │
│          └──── compile ────────┴────── parse ──────┘               │
│                                 │                                  │
│                                 ▼                                  │
│ ══════════════════════════  λ-AST  ════════════════════════════   │
│   Typed terms: Var · Abs · App · Let · Case                        │
│                Combinator(op, args) · Fix · Leaf(𝓜, template)     │
│                                                                    │
│                                 │                                  │
│                                 ▼                                  │
│ RUNTIME                                                            │
│                                                                    │
│   Planner    — computes (k*, τ*, d, ⊕, π) for any Fix node         │
│   Executor   — β-reduction with bounded 𝓜 at Leaf nodes           │
│   Oracle     — fsm_llm.LiteLLMInterface adapter (unchanged)        │
│   Session    — persistence for Category A dialog (unchanged)       │
│   Monitor    — per-term span tracing, depth/cost histograms        │
└────────────────────────────────────────────────────────────────────┘
```

Two crucial architectural properties:

1. **Single executor.** There is no "FSM path" and "λ path". Every program is a
   λ-AST. `converse(user_msg, conv_id)` does one β-reduction of the program's
   per-turn term, threading `(state_id, context)` through the session store for
   Category A programs, or purely functionally for Category B/C.
2. **Single planner.** Every `Fix` node — whether it originated from an FSM's
   `push_fsm` stacking or from a native `fix` combinator — goes through the same
   planner to get a termination proof, a cost estimate, and an execution plan.

---

## 5. λ-AST

The AST is deliberately small. Keeping it small is what makes the theorems
compositional.

```python
# src/fsm_llm/lam/ast.py  (becomes part of core)
class Term: ...

class Var(Term):          name: str
class Abs(Term):          param: str; body: Term
class App(Term):          fn: Term; arg: Term
class Let(Term):          name: str; value: Term; body: Term
class Case(Term):         scrutinee: Term; branches: dict[str, Term]   # finite discrimination

class Combinator(Term):   op: Literal["SPLIT","PEEK","MAP","FILTER","REDUCE","CONCAT","CROSS"]
                          args: list[Term]
class Fix(Term):           body: Abs                                 # body is a self-application λf. ...
class Leaf(Term):          oracle: OracleRef                         # 𝓜 call, schema-typed
                           template: PromptTemplate
                           input_vars: list[str]
                           schema: type[BaseModel] | None             # if set, structured extraction
```

Notes on design:

- `Case` replaces the FSM's transition table. A compiled FSM has exactly one
  top-level `Case` on `state_id`; each branch is the λ-term for that state.
- `Leaf` is the **only** node that invokes 𝓜. Everything else is symbolic. This
  is Assumption A2 encoded in the type system.
- `Fix` appears exactly where the program wants bounded recursion — in Category
  C programs natively, in Category A programs only if stacking is used.
- `Combinator`'s `op` is a closed enum. New combinators are added by extending the
  enum + the executor; users cannot introduce arbitrary operators, which is what
  makes the library "pre-verified" in the paper's sense.

### Typing (pragmatic, not full System F)

Each `Leaf` declares its schema (pydantic type) for typed IO at oracle
boundaries. Combinators are monomorphised at parse time from the concrete
types flowing through the tree. This catches the common errors (passing a list
where a string is expected, `REDUCE` with a non-associative operator on
multi-chunk output) without the complexity of full inference.

---

## 6. FSM JSON → λ-AST compiler

Legacy FSM JSON is compiled at load time by a small, pure function.
**No FSM is ever executed except via this compile → run path** after M2 —
`FSMManager` and `MessagePipeline` become thin adapters over the λ-executor.

### 6.1 Compilation scheme

For an FSM with states *s*₁…*s*ₙ and initial *s*₀:

```
compile(fsm) =
  λ(state_id, user_msg, context).
    Case state_id of
      "s1" → compile_state(s1, user_msg, context)
      "s2" → compile_state(s2, user_msg, context)
      ...

compile_state(s, user_msg, context) =
  Let c' = extract_for(s, user_msg, context)            -- Leaf with s.extraction_instructions schema
  Let s' = eval_transitions(s.transitions, c')          -- pure case analysis + optional Classifier Leaf
  Let o  = respond_for(s', c')                          -- Leaf with s'.response_instructions
  (s', o, c')

extract_for(s, i, c) =
  Leaf(oracle=𝓜, template=s.extraction_instructions,
       input_vars=[i, c], schema=schema_from(s.classification_extractions))

eval_transitions(ts, c) =
  Case (all_guards_evaluated(ts, c)) of
    DETERMINISTIC(s')  → s'
    BLOCKED            → emit_blocked_response
    AMBIGUOUS(cands)   → Leaf(oracle=Classifier, template=disambig(cands), input_vars=[c])

push_fsm(sub) =
  Fix(λloop. compile(sub))   -- each push is a bounded fix invocation
```

### 6.2 Semantic preservation

For every FSM `F`, running the compiled λ-term via the executor produces the
same `(state', output, context')` triple as the current `FSMManager.process`
for every `(state, input, context)` triple in the valid input space. This is a
theorem we prove via structural induction on the FSM AST, not a hope.

**Regression test requirement**: the existing 2,349 tests run against the
λ-executor with no changes. Any deviation is a compiler bug, not a behavioural
change.

### 6.3 Handler hooks survive compilation

The 8 `HandlerTiming` hooks become term transformers:

- `START_CONVERSATION` / `END_CONVERSATION` — wrap the outermost `conversation = fix(...)` loop
- `PRE_PROCESSING` — compose before `extract_for`
- `POST_PROCESSING` — compose after `respond_for`
- `PRE_TRANSITION` / `POST_TRANSITION` — compose around `eval_transitions`
- `CONTEXT_UPDATE` — compose after every `Let c' = ...`
- `ERROR` — catch at the executor

A handler is just a function `context → context` (or equivalent); composing it
at the right point is a straightforward AST rewrite. The existing handler
registration API is unchanged.

---

## 7. The λ-DSL — direct authoring for Category B and C

For programs that don't need FSM ceremony, write λ-terms directly in Python
via a fluent combinator API. No JSON round-trip required.

```python
from fsm_llm.lam import split, fmap, ffilter, reduce_, fix, leaf, oracle

# Category B: an evaluator-optimizer loop (was fsm_llm_agents/evaluator_optimizer)
draft  = leaf(template="Write a draft response to: {q}",           schema=str)
score  = leaf(template="Score the draft (0-10): {draft}",          schema=float)
refine = leaf(template="Improve this draft: {draft}\nFeedback: {fb}", schema=str)

evaluator_optimizer = fix(lambda self: lambda q, draft_so_far:
    leaf_if(score(draft_so_far) >= 8.0,
            then=draft_so_far,
            else_=self(q, refine(draft_so_far, feedback=score.explanation))))

# Category C: long-document NIAH (paper's S-NIAH plan)
niah = fix(lambda self: lambda P:
    leaf_answer(P) if size(P) <= TAU else
    reduce_(best, fmap(self, split(P, K_STAR))))
```

The Python API is an `Abs`/`App` builder — its `__call__` returns AST nodes,
not values. Execution is explicit: `executor.run(niah, document)`. This keeps
the semantics referentially transparent and makes the term serialisable.

Optional JSON surface (`λ-JSON`) exists for programs that want to be stored,
diffed, or shared across language runtimes — but is secondary to the Python
DSL.

---

## 8. The combinator library (Layer 1)

Same library as the paper, now part of the runtime kernel rather than an
extra. Invariants (tokenizer-aware `split`, total/deterministic ℒ∖{𝓜},
associative `reduce`) are enforced at AST construction time, not at runtime.

```python
split(P: str, k: int, *, boundary: BoundaryFn = token_aware) -> list[str]
peek(P: str, start: int, end: int) -> str
fmap(f, xs)        # (α→β) × [α] → [β]
ffilter(pred, xs)  # (α→𝔹) × [α] → [α]
reduce_(op, xs, *, unit=None)   # associative fold
concat(xs, sep="")
cross(xs, ys)      # Cartesian product
```

Composition operators ⊕ are a registry of named `ReduceOp` values, each with an
associativity flag, an optional `cost(k)` method, and an
`accuracy_preserving_prob` used by the planner.

---

## 9. Planner and executor (Layers 2 & 3)

Unchanged from the v0.1 design in function, but now the *default* path for
every program, not an opt-in.

- **Planner** (`src/fsm_llm/lam/planner.py`): pure function from `PlanInputs`
  (|P|, K, task_type, α, cost_config) to `Plan` (k\*, τ\*, d, ⊕, π,
  predicted_cost, predicted_calls, accuracy_floor). Zero LLM calls. Closed-form
  per paper's Theorems 2 & 4. Runs at `Fix` node entry.
- **Executor** (`src/fsm_llm/lam/executor.py`): standard β-reduction
  interpreter. Recurses on AST, delegates `Combinator` to the library,
  `Leaf` to the oracle. For `Fix` it either uses a trampoline (headless) or
  the session-stack (conversational) — the two cases pick the same algorithm,
  just with different state-persistence backends.
- **Oracle** (`src/fsm_llm/lam/oracle.py`): adapter from `LiteLLMInterface` to
  an `Oracle` protocol enforcing A1 (`|P| ≤ K`) at call time.

Cost model (`CostAccumulator`) wraps `LiteLLMInterface` transparently and
emits per-leaf usage to `Monitor`.

---

## 10. Request lifecycle — unified

For a Category A dialog (existing FSM JSON):

```
API.converse(user_msg, conv_id)
  ▸ session.load(conv_id) → (state_id, context)
  ▸ program = COMPILED[fsm_id]                      -- cached λ-term
  ▸ (state', output, context') =
      executor.run(program, (state_id, user_msg, context))
      ├── Case state_id → branch for current state
      ├── Leaf extract_for     → 1 𝓜 call           Pass 1
      ├── Case eval_transitions → 0 or 1 𝓜 call     Pass 1 (classifier)
      └── Leaf respond_for     → 1 𝓜 call           Pass 2
  ▸ session.save(conv_id, (state', context'))
  ▸ return output
```

For a Category B pipeline (direct λ-DSL, no session):

```
executor.run(my_pipeline, inputs)
  ▸ planner.plan(...) at each Fix node
  ▸ β-reduction traverses AST
  ▸ leaves invoke 𝓜; combinators execute symbolically
  ▸ return final result
```

For a Category C long-context task (paper's home turf):

```
executor.run(niah_program, document)
  ▸ top-level Fix → planner computes (k*=2, τ*=K, d=⌈log₂(n/K)⌉)
  ▸ predicted_cost logged; budget check
  ▸ SPLIT → MAP → FILTER → MAP(𝓜) → REDUCE chain
  ▸ exactly (k*)^d + 1 oracle calls, guaranteed
```

Same code path for all three. This is the payoff.

---

## 11. What happens to the existing sub-packages

The five sub-packages reorganise into one kernel + a standard library of
named λ-terms.

| Today | Post-unification | Notes |
|---|---|---|
| `fsm_llm/` (FSM, 2-pass pipeline, classifier, handlers, session, LLM iface) | `fsm_llm/` = kernel (λ-AST + compiler + executor + oracle + planner + session) | 2-pass becomes a λ-term schema; FSM JSON is compiled; LLM iface unchanged |
| `fsm_llm_reasoning` (structured reasoning engine) | `fsm_llm/stdlib/reasoning.py` | Reasoning steps become named λ-terms; engine is a scheduler over them, which is itself a λ-term |
| `fsm_llm_workflows` (DAG orchestration) | `fsm_llm/stdlib/workflows.py` | DAG is a λ-term built by `.then` / `.parallel` / `.branch` combinators; the engine dissolves |
| `fsm_llm_agents` (12 patterns + graph, swarm, MCP, A2A, SOPs, semantic tools, meta) | `fsm_llm/stdlib/agents/` (one module per pattern) | Each pattern becomes a named λ-term factory returning an `Abs`; the class-per-pattern proliferation collapses |
| `fsm_llm_monitor` | `fsm_llm_monitor/` (unchanged externally; gains λ-term span exporter) | Trace granularity shifts from FSM state → AST node; the OTEL schema is richer but the consumer API is the same |

The meta-builder (`fsm_llm_agents/meta_*`) targets λ-terms as its output
format instead of FSM JSON.

The `examples/` tree reorganises to reflect the taxonomy:

```
examples/
├── dialog/            (was basic/ intermediate/ advanced/ — Category A survivors)
├── pipeline/          (was agents/, most of reasoning/ — Category B)
├── long_context/      (new — Category C: NIAH, aggregate, pairwise, multi_hop)
└── mixed/             (programs that push sub-FSMs from within pipelines)
```

---

## 12. Preserved and extended theorems

The four theorems from the paper now apply universally.

**T1 (Termination).** Every λ-term without `Fix` halts in bounded β-reductions.
Every term with `Fix` halts in exactly (k\*)^*d* + 1 oracle calls, where *d* is
the planner-computed depth, provided (i) `split` strictly reduces rank, (ii) all
combinators in ℒ ∖ {𝓜} are total, (iii) session depth ≤ `max_stack_depth`.

**T2 (Cost is pre-computable).** For every `Fix` node, `plan.predicted_cost`
bounds actual spend within 5% given matched pricing constants. For terms
without `Fix`, cost is the sum of `Leaf` per-call costs — trivially bounded
by a single traversal.

**T3 (Accuracy floor).** `plan.accuracy_floor` ≥ 𝒜(τ\*)^{*n·k\**/τ\*} · 𝒜_⊕^*d*.
For decomposable tasks (𝒜_⊕ = 1), per-query accuracy is constant in *n* —
polynomial, not exponential, decay with input length.

**T4 (Default k\*=2).** Under linear cost and composition, the planner selects
k\* = 2 unless an accuracy constraint tightens it.

**T5 (FSM semantic preservation).** For any FSM *F*, `execute(compile(F), x)`
≡ `FSMManager.process(F, x)` for all valid inputs *x*. Proven by induction
on FSM structure, tested by running the existing 2,349 tests against the
λ-executor.

T1–T4 are carried from the paper. T5 is what lets us swap the runtime without
breaking a single production deployment.

---

## 13. Migration path

Five milestones. Each delivers working software and can land independently.

| Milestone | Scope | Ships |
|---|---|---|
| **M1 — Kernel** | `fsm_llm/lam/` (AST, parser for λ-DSL, combinator library, planner, executor, oracle adapter) | λ-DSL runs end-to-end for Category B & C programs. FSM path untouched. |
| **M2 — FSM compiler + executor unification** | `fsm_llm/lam/fsm_compile.py`; `FSMManager` delegates to λ-executor; T5 regression (all 2,349 tests pass). | Existing programs run on the new substrate. No user-visible change. |
| **M3 — Stdlib** | Agents, reasoning, workflows patterns reimplemented as λ-term factories; old sub-packages re-export the names for backward compat. | Category B examples migrate — smaller, faster, theorems apply. |
| **M4 — Category B example migration** | Rewrite the ~30 Category B examples as direct λ-DSL; deprecate (don't delete) their FSM JSON. | Ceremonial FSM usage measurably shrinks. |
| **M5 — Long-context library + benchmarks** | Category C programs (NIAH, OOLONG, OOL-Pairs equivalents) as λ-terms; publish benchmarks on Qwen3/Llama/Mistral paralleling the paper. | fsm_llm becomes a *superset* of the paper's λ-RLM, with conversational interop. |

At no point does an existing program break. At every point a new kind of
program becomes expressible.

---

## 14. What we explicitly do **not** claim

- **FSMs aren't disappearing.** Category A programs (real dialog state) are a
  first-class surface. We're not arguing for "λ-only"; we're arguing for "λ as
  substrate, FSM as dialog surface".
- **The substrate does not subsume free-form code generation.** Paper's
  Table 8: for CodeQA with strong models, free-form agents win. fsm_llm keeps
  the ability to embed an unstructured agent as a `Leaf` — that's what makes it
  strictly more expressive than λ-RLM-as-published.
- **Streaming is still open.** The paper is batch. Leaf-level streaming works
  for `CONCAT` and monotone `REDUCE` variants; neural ⊕ buffers. Design for
  partial streaming in M4.
- **Full System F typing is not on the table for v1.** Monomorphisation at
  parse time catches the common bugs. Richer types if M5 needs them.

---

## 15. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| T5 semantic-preservation proof has a hole; some FSM behaviour diverges under the compiler | Medium | All 2,349 tests must pass pre-M2 merge; CI gate. Any divergence is a ship-blocker. |
| Performance regression from AST interpretation overhead | Low-medium | Benchmark M2 vs baseline; if >5% regression, compile hot paths (per-FSM cache of reduced forms). |
| User confusion from two surface syntaxes | Medium | Migration guide + decision tree ("do you need dialog state? → FSM; otherwise → λ-DSL"). Don't deprecate FSM JSON. |
| Scope creep — substrate-wide change invites unrelated refactoring | High | Milestones are strict; each merge is narrow; stdlib reorganisation (M3) is separate from kernel (M1) and compiler (M2). |
| Tokenizer mismatch breaks T1 termination | Medium | `Oracle` protocol exposes `tokenize()`; `split` uses the oracle's tokenizer by default; integration test per provider. |
| `Fix`-via-`push_fsm` per-level overhead exceeds trampoline | Low | Trampoline is default; stacked executor only when dialog interop is needed. Benchmarked. |

---

## 16. Summary

The proposal is not to add λ-substrate beside FSMs. The proposal is: **λ-calculus
is the substrate**. FSMs become a surface syntax, kept for dialog-state programs
where they earn their keep, compiled to λ-terms everywhere else. Most of
`fsm_llm_agents`, most of `fsm_llm_reasoning`, and most of `examples/` stop
being FSMs and become what they always structurally were — λ-terms with
combinators. The paper's theorems (termination, closed-form cost, polynomial
accuracy decay, optimal k\*=2) stop being opt-in guarantees for a sub-package
and become universal properties of the runtime. And long-context tasks, which
the current FSM model can't express well, gain a native, provably-bounded home.

One runtime. Two surface syntaxes. A smaller, stronger, more honest framework.

---

## References

- Roy, Tutunov, Ji, Zimmer, Bou-Ammar. *The Y-Combinator for LLMs: Solving
  Long-Context Rot with λ-Calculus*. arXiv:2603.20105, 20 Mar 2026.
- Barendregt, *The Lambda Calculus: Its Syntax and Semantics*, 1984.
- Plotkin, *Call-by-name, call-by-value and the λ-calculus*, TCS 1975.
- fsm_llm CLAUDE.md — framework architecture, 2-pass flow, handler timing,
  FSM stacking, classification.
