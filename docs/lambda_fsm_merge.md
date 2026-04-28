# The FSM ↔ λ Merge — A Unified-API Specification

**Status**: v1.0 (2026-04-28). Successor deliverable to `docs/lambda_integration.md` v2.0.
**Companion**: `docs/lambda.md` (the architectural thesis — what the merge serves).
**Audience**: an engineer asked to finish the merge without re-reading the two prior docs end-to-end.
**Methodology**: Produced via a LITE-tier epistemic-deconstruction pass (`analyses/analysis_2026-04-27_13dc12d8/`); five design candidates (CAND-A–E) were derived by abductive expansion against HEAD (2026-04-28), not inherited from v2.0's R8–R13 frame.

---

## 0. Why this document exists, and what it is not

The two prior docs answered different questions:

- `docs/lambda.md` (v0.2, 2026-04-24) is the **thesis**: λ-calculus is the substrate; FSMs are one surface; a single executor runs both. It does not specify a user-facing API.
- `docs/lambda_integration.md` v2.0 (2026-04-27) is the **audit + plan**: R1–R7 shipped narrowed; eight loci of disjointedness (L1–L8); R8–R13 to ship in a clean v0.5.0. It names *what to do* but stops at the PR level.

This document is the **merge contract**: the user-facing API, the invariants that defend it, the falsification gates that detect regressions, and the sequenced commits that finish the work HEAD has already started. It is not a status report and not a roadmap — it is the specification an engineer implements against.

> **Crucial fact about HEAD**: between v2.0's publication (2026-04-27) and this doc (2026-04-28), R8 (`Program.invoke`), R9c (cohort gate dropped), R11 (substrate exports promoted), R12 (factory CLI), and partial R10/R13 already shipped. The merge is in mid-air, not pre-flight. The work below picks up from that state.

---

## Status — Implementation progress (as of 2026-04-28)

Plan `plans/plan_2026-04-28_6597e394/` shipped **M1 + M2 + M4 + M3a + M3b**. M3c halted on Pre-Mortem Scenario A (default-flip surfaced 4 dialog runtime divergences including an unforeseen streaming-sibling break). M3d is unreachable while M3c is unlanded; both are deferred to a follow-on plan, alongside M5 and M6.

| Milestone | Status | Commit | Notes |
|---|---|---|---|
| M1 — `Result` everywhere | ✅ SHIPPED | `f3b6a1a` | `Program.invoke` returns `Result` in all modes; 3 FSM-mode tests inverted+renamed; CHANGELOG entry. |
| M2 — Layered `__all__` + import-audit | ✅ SHIPPED | `2f080f5` | `_LAYER_L1..L4` frozensets; `test_layering.py` (12 tests). **Allow-list landed at 5 entries, not the spec's 4** — `handlers.py` is a 5th `FSMError`-inheritance coupling found mid-EXECUTE (D-007-SURPRISE); same pattern, same future "decouple-kernel-from-dialog" plan. |
| M4 — Oracle ownership on Program | ✅ SHIPPED | `da7d03c` | `Program → API → FSMManager → MessagePipeline` thread one `Oracle`; 7 (not 5 — count drifted) `LiteLLMOracle(...)` calls in `dialog/turn.py` collapsed to `self._oracle` field-reads; `test_oracle_ownership.py` ships (5 tests, G2 AST-gate + identity propagation). |
| M3a — Private opt-in field scaffolding | ✅ SHIPPED | `a25b899` | `_emit_response_leaf_for_non_cohort: bool = False` Pydantic-private State attr; `compile_fsm._compile_state` plumbed via `getattr`; byte-equivalent at default. |
| M3b — Non-cohort Leaf emission (opt-in) | ✅ SHIPPED | `7a1e506` | `Let(NONCOHORT_RESPONSE_PROMPT_VAR, App(CB_RENDER_RESPONSE_PROMPT, instance), Leaf("{...}"))`; `_make_cb_render_response_prompt` host factory; +31 strict-Theorem-2 tests; default-False preserves all 837 dialog tests. **I6 holds for opt-in non-cohort programs.** |
| M3c — Flip default to True | 🟡 HALTED → DEFERRED | (reverted) | Pre-Mortem Scenario A fired. 4 dialog runtime divergences: empty-`response_instructions` short-circuit, `add_system_message` history append, structured-output `_output_response_format`, **and the unforeseen `converse_stream` streaming sibling** (`process_compiled_stream` rebinds `CB_RESPOND` with `_make_cb_respond_stream`; bare Leaf has no streaming counterpart). See follow-on plan §"Deferred work" below. |
| M3d — Retire `_cb_respond` | ⬜ DEFERRED | — | Cannot land while M3c is unlanded; `_cb_respond` is load-bearing for the 4 divergences (esp. streaming). |
| M5 — Deprecation calendar | ⬜ DEFERRED | — | Independent of structural work. Silent-in-0.5.0 work scheduled in follow-on. |
| M6 — Out-of-scope payload | ⬜ DEFERRED | — | Monitor span schema, session migration, sibling shim deprecation hook, third-party LLMInterface contract. |

**What this means for I6 today**: Theorem-2 universality holds for term/factory programs (already true at HEAD pre-M3) and for any FSM program that opts in via the M3a private field. For default FSM programs, `App(CB_RESPOND, ...)` still emits and the I6 caveat clause from §3 below remains in force. The merge is **structurally complete** (mode-fixed Program, one Oracle, layered exports, audit-enforced) but **Theorem-2 is not yet universal by default**.

---

## 1. The merge contract in one paragraph

> A user constructs a `Program` from one of three sources — FSM JSON, a λ-term, or a factory function — and gets back an object with **one execution verb**: `program.invoke(...)`. The verb is mode-aware (FSM mode accepts `message=`; term/factory mode accepts `inputs=`); mode is fixed at construction; misuse raises one error class (`ProgramModeError`) with a redirect message; the return type is always `Result`. The Program owns **one Oracle**; every LLM call across both surfaces flows through that Oracle's four methods (`invoke`, `invoke_stream`, `invoke_messages`, `invoke_field`) and nothing else; per-Leaf cost telemetry is emitted uniformly. Theorem-2 (`oracle_calls == predicted_calls`) holds for every Program a user can construct from public APIs. The four-layer architecture (substrate → composition → authoring → invocation) is enforced by an import-audit test, not by convention. Back-compat surfaces (`.run`, `.converse`, `register_handler`, `API`, sibling shim packages) survive through 0.5.x with no warning, gain `DeprecationWarning` in 0.6.0, and are removed in 0.7.0.

The rest of this document defines each clause precisely enough to implement and to test.

---

## 2. The four-layer architecture (the picture every PR is checked against)

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

The merge contract says: **a reader of a single `from fsm_llm import …` line knows which layer they are in.** The package surface tells the architecture, not the historical order in which features arrived.

---

## 3. The six invariants (these defend the merge against the next narrowing)

These are not aspirations. Each is enforced by a falsification gate in §6.

**Invariant status (2026-04-28):**
- **I1** ⬜ DEFERRED — needs M6/oracle-call-paths audit (G1); not yet gated. M4 shipped ownership but not the AST-walk enforcement.
- **I2** ✅ SHIPPED via M4 (commit `da7d03c`); G2 test enforces zero `LiteLLMOracle(...)` calls in `dialog/turn.py`.
- **I3** ✅ Mode is fixed at construction; no runtime mode-switch. Held since R8.
- **I4** ✅ partial — G3 ships via M2 (commit `2f080f5`) with a documented **5-entry** allow-list (one more than spec). 5th entry is `handlers.py` (`HandlerSystemError(FSMError)` MRO). All 5 entries point to a future "decouple-kernel-from-dialog" plan.
- **I5** ⬜ DEFERRED — M5 calendar work not started; aliases ship silent in 0.5.x as the spec prescribes (no behaviour change required to satisfy 0.5.0 row).
- **I6** 🟡 PARTIAL — Theorem-2 strict tests pass under default-False (M3b ships +31 tests with strict equality); default-True flip blocked by 4 dialog divergences including streaming. Universal-by-default deferred to follow-on plan.

### I1 — All LLM calls go through Oracle. No exceptions.

The kernel `Oracle` protocol exposes exactly four call-shapes:

```python
class Oracle(Protocol):
    def invoke(self, prompt: str, *, env: dict | None = None,
               schema: type[BaseModel] | None = None,
               model_override: str | None = None) -> Any: ...
    def invoke_stream(self, prompt: str, *, user_message: str | None = None) -> Iterator[str]: ...
    def invoke_messages(self, messages: list[dict], *, call_type: str) -> Any: ...
    def invoke_field(self, request: FieldExtractionRequest) -> Any: ...
    def tokenize(self, text: str) -> list[int]: ...   # for Theorem-1 split bound
```

No fifth method. No `_make_llm_call` escape hatch. No `LLMInterface.{generate_response, extract_field, generate_response_stream}` symbol referenced anywhere under `src/fsm_llm/dialog/`, `src/fsm_llm/runtime/` (other than `_litellm.py`'s implementation), or `src/fsm_llm/stdlib/`.

### I2 — Program owns exactly one Oracle. Dialog and term paths read from it; they do not construct.

```python
class Program:
    _oracle: Oracle      # set in __init__, frozen for the program's lifetime
                         # (per-call model_override goes through invoke(..., model_override=))
```

`MessagePipeline` (now `dialog/turn.py`) does not call `LiteLLMOracle(self.llm_interface)` — it accepts the oracle as a constructor argument from `FSMManager`, which receives it from `API`, which receives it from `Program`. A static check (§6 G2) asserts no `LiteLLMOracle(...)` constructor call exists in `dialog/turn.py`.

This closes the smoking gun discovered at HEAD: five separate `LiteLLMOracle(self.llm_interface)` instantiations across `turn.py:441, 1190, 1235, 1290, 1635, 2174`. The wiring landed without ownership; the merge needs the ownership.

### I3 — Mode is fixed at construction. There is no runtime mode-switch.

```python
Program.from_fsm(defn, ..., handlers=...)        # ⇒ FSM mode permanently
Program.from_term(term, ..., handlers=...)       # ⇒ term mode permanently
Program.from_factory(factory, *args, **kwargs)   # ⇒ term mode permanently
```

`.invoke(message=...)` on a term-mode program raises `ProgramModeError` with the redirect "build with `from_fsm` or call `.invoke(inputs=...)`". `.invoke(inputs=...)` on an FSM-mode program raises with the redirect to `.invoke(message=..., conversation_id=...)`.

There is no `Program.switch_mode()`, no late binding of `_api`/`_term`. The pair `(_api, _term)` is invariant: exactly one of the two is non-None for the program's lifetime.

### I4 — Layering is enforced by import-audit, not convention.

A test (`tests/test_fsm_llm/test_layering.py`) AST-walks every module in `src/fsm_llm/` and asserts:

| Module path | Permitted to import from |
|---|---|
| `fsm_llm.runtime.*` (excluding `_litellm.py`) | nothing under `fsm_llm.dialog`, `fsm_llm.stdlib`, `fsm_llm.handlers`, `fsm_llm.program` |
| `fsm_llm.handlers` (L2) | only `fsm_llm.runtime.*` |
| `fsm_llm.stdlib.*` and `fsm_llm.dialog.compile_fsm` (L3) | only `fsm_llm.runtime.*` and `fsm_llm.handlers` |
| `fsm_llm.program` (L4) | any of the above |

The L6 back-reference (`fsm_llm.lam.fsm_compile` shim resolving into `fsm_llm.dialog.compile_fsm`) is explicitly permitted in the shim layer (`fsm_llm/lam/__init__.py`) and *only* there; the test allow-lists that one file by path.

> **Implementation note (2026-04-28)**: the shipped `_PRE_EXISTING_DIALOG_IMPORT_ALLOWLIST` in `tests/test_fsm_llm/test_layering.py` contains **five** entries, not the four named in I4's table:
> 1. `runtime/_litellm.py` (the `LiteLLMInterface` implementation)
> 2. `runtime/oracle.py` (Request objects from `dialog/definitions`)
> 3. `runtime/errors.py` (`LambdaError(FSMError)` MRO)
> 4. `stdlib/workflows/exceptions.py` (workflow exceptions inheriting `FSMError`)
> 5. **`handlers.py`** (`HandlerSystemError(FSMError)` MRO — found mid-EXECUTE during M2 as D-007-SURPRISE; same kernel↔dialog coupling pattern as the four above)
>
> All five entries are visibly grandfathered debt with code-comment pointers to a future "decouple-kernel-from-dialog" plan (relocate `FSMError` out of `dialog/definitions.py`). The audit catches future violations; pre-existing debt is named, not silently accepted. This is a known limitation of the current I4 enforcement; spec §3 I4 should read "five" once that follow-up plan retires entries.

### I5 — Back-compat alias contract: ship date is 0.5.0, deprecation is 0.6.0, removal is 0.7.0.

Aliases preserved through their full life:

| Surface | 0.5.x | 0.6.x | 0.7.x |
|---|---|---|---|
| `Program.run(**env)` | works silently | `DeprecationWarning` | removed |
| `Program.converse(msg, conv_id)` | works silently | `DeprecationWarning` | removed |
| `Program.register_handler(h)` | works silently | `DeprecationWarning` | removed (use `handlers=` constructor arg) |
| `from fsm_llm import API` | works silently | `DeprecationWarning` | removed (use `Program.from_fsm`) |
| `from fsm_llm.lam import …` | works silently | `DeprecationWarning` | removed (use `from fsm_llm.runtime`) |
| `from fsm_llm.{api,fsm,pipeline,prompts,classification,…} import …` | works silently | `DeprecationWarning` | removed (use `fsm_llm.dialog.*`) |
| `from fsm_llm.dialog.pipeline import MessagePipeline` | works silently | `DeprecationWarning` | removed (use `fsm_llm.dialog.turn`) |
| `import fsm_llm_reasoning / fsm_llm_workflows / fsm_llm_agents` | works silently | `DeprecationWarning` | removed (use `fsm_llm.stdlib.*`) |

The calendar slips by **one minor** versus v2.0's plan because R8/R9c/R11/R12 already shipped without the alias-deprecation gate. The 0.5.0 release ships the merge complete; 0.6.0 is the warning release; 0.7.0 is the removal release. A CI test (§6 G4) asserts the calendar entries match what the deprecation warnings actually emit.

### I6 — Theorem-2 holds for every Program a user can construct from public APIs.

For every `(model, program_kind)` cell in the regression bench, `oracle_calls == predicted_calls` strictly (or `oracle_calls ≤ predicted_calls` for cases with documented sentinel-arm short-circuit, e.g. `multi_hop_dynamic`). `program_kind ∈ {"fsm", "term", "factory"}`. The bench scorecard schema gains the `program_kind` axis; the `theorem2_holds: true` field is required for every cell.

**The asterisk this exposes**: today, non-cohort FSM states (any state with extractions OR transitions OR field-extractions referenced in the response prompt) still emit `App(CB_RESPOND, ...)` instead of a `Leaf`, so I6 fails for them. The merge is not done until R10.next (§5, M3) lifts CB_RESPOND for non-cohort states. **Until R10.next ships, I6 carries an explicit exception clause documented in `compile_fsm.py` and the bench scorecard.**

---

## 4. The five load-bearing design decisions

These are the choices the v2.0 plan did not make. The merge cannot ship without committing each.

### CAND-A — `Result` is the universal return type. `.invoke()` always returns `Result`.

```python
@dataclass(frozen=True)
class Result:
    value: Any                         # term reduction OR FSM reply string OR streaming-iter sentinel
    conversation_id: str | None = None # populated in FSM mode only
    plan: Plan | None = None           # populated when explain=True
    leaf_calls: int = 0                # populated by the executor's CostAccumulator
    oracle_calls: int = 0              # ditto; should equal plan.predicted_calls under I6
    explain: ExplainOutput | None = None  # for richer .explain() output
```

`.invoke()` returns `Result` in every mode. The R8 compromise of `Result | str` is fixed: callers needing the raw string in FSM mode write `.invoke(message=...).value`. The legacy `.converse(...)` alias still returns `str` (one-line unwrap of `result.value`); the legacy `.run(**env)` still returns the bare reduction value. The merge spec commits the union shape only on the *aliases*, not on the canonical verb.

### CAND-B — `dialog/turn.py` is judged by **path count**, not LOC.

The v2.0 plan promised `pipeline.py: 2236 → ~400 LOC`. HEAD shows 2295 LOC after R10 wiring. The honest invariant is **how many distinct LLM call paths exist**, not how many lines back the file. The merge declares: there are exactly **four** call paths, one per Oracle method (I1). Each is reachable by exactly one code path in `dialog/turn.py`. The `LiteLLMInterface` symbol does not appear in the file outside the bound `self._oracle` field (I2 enforces ownership; this clause enforces visibility).

LOC is permitted to stay near 2,000 if the path-count invariant holds. The spec does not commit a LOC target.

### CAND-C — Non-cohort FSM states get a Leaf-emission story (R10.next).

Today's compile_fsm.py emits `App(CB_RESPOND, instance)` for any state with extractions or transitions; only "cohort-eligible" states (terminal, no transitions, no extractions, no extracted-field references in the response prompt) get a real `Leaf`. The R10.next commit (M3 below) extends Leaf emission to non-cohort states by:

1. Compile the response generation as `Leaf(template=rendered_response_prompt, schema=str, env={...extracted+context})` after the extraction Leaf and (if present) classifier Leaf in a `let_` chain.
2. The `_cb_respond` Python callable is retired; the AST itself carries the response-generation work.
3. The `App(CB_*, ...)` host-callable splices for *non-LLM* host hooks (session save, error boundary) survive — those are not LLM calls and do not violate I1.

Cost: medium. Risk: T5 semantic preservation across 837 dialog tests. Mitigation: per-state opt-in via a private `_emit_response_leaf: bool = False` field on State for the first PR; flip default in the second PR; remove the field in the third.

### CAND-D — Oracle ownership lives on Program, not per-callsite.

Concretely:

```python
class Program:
    def __init__(self, *, oracle: Oracle, term: Term | None, api: API | None, ...):
        self._oracle = oracle
        # ... no other oracle construction anywhere

    @classmethod
    def from_fsm(cls, defn: FSMDefinition, *, llm: LLMInterface | None = None,
                 oracle: Oracle | None = None, handlers: list[FSMHandler] = ()) -> Program:
        # exactly-one-of llm / oracle:
        if (llm is None) == (oracle is None):
            raise ValueError("provide exactly one of llm= or oracle=")
        if oracle is None:
            oracle = LiteLLMOracle(llm)
        api = API(...)  # internally accepts oracle, threads through FSMManager → MessagePipeline
        return cls(oracle=oracle, api=api, term=None, ...)
```

`MessagePipeline.__init__` gains a required `oracle: Oracle` parameter. The five `LiteLLMOracle(self.llm_interface)` constructions in `dialog/turn.py` collapse to one `self._oracle = oracle` field-read. `API.from_definition(defn, oracle=oracle)` is the new signature; the old `API.from_definition(defn, llm_interface=llm)` is preserved as a `DeprecationWarning`-on-call alias.

This is the structural change that makes I1+I2 a property of the code, not a code-review aspiration.

### CAND-E — `__all__` is layer-ordered with a discoverable convention.

```python
# fsm_llm/__init__.py — v0.5.0
"""fsm_llm — typed-λ runtime with FSM dialog and λ-DSL surfaces.

Layers, in import order:
    L4 INVOKE   — Program, Result, ExplainOutput, ProgramModeError
    L3 AUTHOR   — compile_fsm, stdlib factories (react_term, niah, …),
                  raw DSL (leaf, fix, let_, case_, …)
    L2 COMPOSE  — compose, Handler, HandlerTiming, HandlerBuilder
    L1 REDUCE   — Term, Executor, Plan, Oracle, LiteLLMOracle, …
    Legacy      — FSMDefinition, FSMInstance, State, Transition, API, …
                  (preserved for back-compat; see I5 calendar)
"""

__all__ = [
    # === L4 INVOKE ===
    "Program", "Result", "ExplainOutput", "ProgramModeError",
    # === L3 AUTHOR ===
    "compile_fsm",
    "react_term", "rewoo_term", "reflexion_term", "memory_term",
    "niah", "aggregate", "pairwise", "multi_hop", "niah_padded",
    "leaf", "fix", "let_", "case_", "var", "abs_", "app",
    "split", "fmap", "ffilter", "reduce_", "concat", "cross", "peek",
    # === L2 COMPOSE ===
    "compose", "Handler", "HandlerTiming", "HandlerBuilder",
    # === L1 REDUCE ===
    "Term", "Executor", "Plan", "PlanInputs", "plan",
    "Oracle", "LiteLLMOracle", "CostAccumulator", "LeafCall",
    # === Legacy (deprecation 0.6.0; removal 0.7.0) ===
    "FSMDefinition", "FSMInstance", "State", "Transition",
    "API", "FSMManager", "Conversation",
    "HandlerSystem",          # legacy class-based handler builder
    "Classifier", "HierarchicalClassifier", "IntentRouter",
    "MessagePipeline",        # internal; will become private in 0.7.0
    "LLMInterface", "LiteLLMInterface",
    # ... infrastructure (validators, visualizers, memory, sessions, logging, exceptions)
]

# Layer markers for the import-audit test (I4):
_LAYER_L4 = frozenset({"Program", "Result", "ExplainOutput", "ProgramModeError"})
_LAYER_L3 = frozenset({"compile_fsm", "react_term", ..., "peek"})
_LAYER_L2 = frozenset({"compose", "Handler", "HandlerTiming", "HandlerBuilder"})
_LAYER_L1 = frozenset({"Term", "Executor", "Plan", "PlanInputs", "plan",
                       "Oracle", "LiteLLMOracle", "CostAccumulator", "LeafCall"})
```

A test (§6 G3) asserts every public name is in exactly one of `_LAYER_L1..L4` or in the legacy block, and no L1 name's defining module imports from an L2/L3/L4 module.

---

## 5. Sequenced commits (M1 → M6) — what an engineer ships, in order

Each Mi is one PR. The order matters: M1 and M2 are independent and small; M3 is the structural deep work; M4–M6 are cleanup and safety.

### M1 — `Result` everywhere (closes CAND-A)

**Status**: ✅ SHIPPED (commit `f3b6a1a`, 2026-04-28).

**Touches**: `program.py`, `tests/test_fsm_llm/test_program.py`.
**Concretely**:
- `Program.invoke` returns `Result` in every mode (FSM mode populates `value=<reply_str>, conversation_id=...`).
- `.run(**env)` and `.converse(msg, conv_id)` aliases unchanged (return raw types).
- New tests assert `isinstance(program.invoke(message="hi"), Result)` for FSM mode.
**Net LOC**: +30 / −5. **Risk**: low.
**Validation**: 837 dialog + 202 kernel tests pass; new ~5 tests assert return-type uniformity.

### M2 — Layer-ordered `__all__` + import-audit test (closes CAND-E + I4)

**Status**: ✅ SHIPPED (commit `2f080f5`, 2026-04-28). Allow-list landed at 5 entries (see I4 implementation note).

**Touches**: `src/fsm_llm/__init__.py`, `tests/test_fsm_llm/test_layering.py` (new).
**Concretely**:
- Reorder `__all__` per CAND-E.
- Add `_LAYER_L1..L4` frozensets.
- Add `test_layering.py` that AST-walks `src/fsm_llm/` and asserts the import permissions in I4.
- The lone allow-listed module is `src/fsm_llm/lam/__init__.py` (the back-compat shim).
**Net LOC**: +120 / −0 (mostly the test). **Risk**: low; might surface one-off violations to fix.
**Validation**: existing 2,899 tests + new ~15 layering tests pass.

### M3 — Lift response Leaf for non-cohort states (closes CAND-C, finishes R10, makes I6 universal)

**Status**: 🟡 PARTIAL — M3a (`a25b899`) and M3b (`7a1e506`) shipped; M3c HALTED on Pre-Mortem Scenario A; M3d unreachable. Default flip + retirement deferred to follow-on plan (see "Deferred work" below).

**Touches**: `dialog/compile_fsm.py`, `dialog/turn.py`, `dialog/CLAUDE.md`, `evaluation/bench_long_context_*.json` schema, `tests/test_fsm_llm/test_compile_fsm.py`.
**Concretely** (sub-PRs to bound risk):
- **M3a**: Add `_emit_response_leaf_for_non_cohort: bool = False` State field. Plumb through compiler. No behaviour change.
- **M3b**: Implement Leaf-emission for non-cohort states, gated on the field. 837 dialog tests with field=False (default) preserved byte-equivalent. New ~30 tests with field=True assert `oracle_calls == predicted` strictly.
- **M3c**: Flip default to True. Run smoke battery on representative production FSMs. T5 must hold (semantic preservation across all 152 examples).
- **M3d**: Remove the field. Retire `_cb_respond` Python callable. Bench scorecard adds `program_kind: "fsm"` cells.
**Net LOC**: +200 / −400 (turn.py shrinks once `_cb_respond` and its dispatch retire). **Risk**: HIGH — every dialog program flows through this path. Mitigation: 4-step incremental rollout above.
**Validation**: 837 dialog + 202 kernel + 273 regression tests; bench Theorem-2 universality confirmed.

### M4 — Oracle ownership on Program (closes CAND-D + I2)

**Status**: ✅ SHIPPED (commit `da7d03c`, 2026-04-28). 7 (not 5 — count drifted in HEAD between v2.0 and ship) `LiteLLMOracle(...)` calls collapsed to `self._oracle` field-reads.

**Touches**: `program.py`, `dialog/api.py`, `dialog/fsm.py`, `dialog/turn.py`.
**Concretely**:
- `Program.__init__` stores `_oracle: Oracle`; `from_*` constructors accept exactly-one of `llm=` or `oracle=`.
- `API.__init__` gains required `oracle: Oracle`; old `llm_interface=` signature preserved with `DeprecationWarning`.
- `FSMManager.__init__` accepts `oracle`; threads to `MessagePipeline`.
- `MessagePipeline.__init__` gains required `oracle`; the five `LiteLLMOracle(self.llm_interface)` calls in `turn.py` collapse to `self._oracle = oracle` field-reads at the call sites.
- A static check in `test_layering.py` asserts no `LiteLLMOracle(...)` constructor call exists in `dialog/turn.py`.
**Net LOC**: +80 / −40. **Risk**: medium (signature changes ripple through tests); mitigation: deprecation alias on every old signature.
**Validation**: all 2,899 tests pass; new ~10 tests assert oracle-identity propagation through `Program → API → FSMManager → MessagePipeline`.

### M5 — Back-compat deprecation calendar (closes I5)

**Status**: ⬜ DEFERRED to follow-on plan. Independent of structural work; aliases ship silent in 0.5.x with no behaviour change required.

**Touches**: every module bearing a deprecation alias; `pyproject.toml` (version bump); new `tests/test_fsm_llm/test_deprecation_calendar.py`.
**Concretely**:
- All aliases in §3 I5 ship silent in 0.5.0. (No code change beyond the version bump and the test scaffolding.)
- The deprecation warnings themselves land in 0.6.0 (a separate release commit; out of scope for this merge but the calendar is committed in this PR).
- `test_deprecation_calendar.py` reads `__version__` and asserts:
  - if `version < (0, 6, 0)`: aliases work silently (no warning)
  - if `(0, 6, 0) ≤ version < (0, 7, 0)`: aliases work + emit `DeprecationWarning`
  - if `version ≥ (0, 7, 0)`: aliases raise `AttributeError`
- A documentation entry under `CHANGELOG.md` commits the calendar.
**Net LOC**: +60 / −0. **Risk**: zero in 0.5.0 (no behaviour change). The hard work is in the 0.6.0 and 0.7.0 release commits, but the schedule is fixed here.
**Validation**: the calendar test passes against the current version; will activate the warning-presence and removal assertions in subsequent releases.

### M6 — Out-of-scope items: monitor, sessions, sibling shims, third-party LLMInterface (the H_S_prime payload)

**Status**: ⬜ DEFERRED to follow-on plan.

**Touches**: `fsm_llm_monitor/CLAUDE.md`, `fsm_llm/dialog/session.py`, `fsm_llm_{reasoning,workflows,agents}/__init__.py`, top-level `CLAUDE.md`.
**Concretely** (each sub-item is one short PR or a doc-only commit):
- **M6a Monitor span schema**: After M3, OTEL spans emit per-Leaf, not per-FSM-state. Document the schema migration in `fsm_llm_monitor/CLAUDE.md`. Add a `span_schema_version: 2` field; consumers reading v1 must update. Provide a one-time backfill script if production sessions are mid-flight.
- **M6b Session-store migration**: Cohort-OFF states wrote a `transition_context` dict that cohort-emission no longer produces. Provide `scripts/migrate_session_store.py` that walks a `FileSessionStore` and rewrites legacy entries to the post-M3 shape. Idempotent.
- **M6c Sibling shim deprecation hook**: `fsm_llm_{reasoning,workflows,agents}/__init__.py` gain a one-line `warnings.warn("import from fsm_llm.stdlib.<pkg> instead", DeprecationWarning, stacklevel=2)` — but **the warning is gated on the I5 calendar** (silent in 0.5.x).
- **M6d Third-party LLMInterface contract**: A user who subclassed `LiteLLMInterface` to add an in-house provider must continue to work. Document in CLAUDE.md: "Any `LLMInterface` subclass passed to `Program.from_fsm(llm=...)` is wrapped in `LiteLLMOracle` automatically; subclass behaviour is preserved through the wrap. Custom oracles bypass `LiteLLMOracle` — implement the `Oracle` protocol directly and pass via `oracle=`."
**Net LOC**: +200 / −0 (mostly migration script and docs). **Risk**: low.
**Validation**: a smoke test with a `MockLLMInterface(LiteLLMInterface)` subclass; a session-migration round-trip test on a fixture session file.

### Recommended ship order
1. **M1** (low-risk, user-visible win — completes the `.invoke` story).
2. **M2** (zero-risk; lights up the discipline that prevents future drift).
3. **M4** (oracle ownership — must precede M3 because M3 leans on owned-oracle semantics).
4. **M3** (the deep work; behind sub-PRs M3a–d).
5. **M5** (deprecation calendar — once the merge content is final, lock the dates).
6. **M6** (out-of-scope payload — can ship in parallel with M5).

**Total elapsed at one PR/week**: ~7 weeks. M1+M2+M4 (3 weeks) make `Program.invoke` honest; M3 (3 weeks across sub-PRs) makes Theorem-2 universal; M5+M6 (1 week, parallel) lock the contract.

---

## 6. Falsification gates (the tests that detect a regression of the merge)

These are not test ideas — they are the spec's enforcement mechanisms. Without them, every invariant is decorative.

**Gate status (2026-04-28):** G2 ✅ shipped (M4); G3 ✅ shipped (M2, with 5-entry allow-list); G1 ⬜, G4 ⬜, G5 ⬜ deferred to follow-on plan.

### G1 — `test_oracle_call_paths_only.py` (enforces I1)

AST-walks `src/fsm_llm/dialog/`, `src/fsm_llm/runtime/` (excluding `_litellm.py`), and `src/fsm_llm/stdlib/`. Asserts:
- No `Attribute` node with `.attr ∈ {"generate_response", "extract_field", "generate_response_stream", "_make_llm_call"}` whose value type is `LLMInterface` or `LiteLLMInterface`.
- The set of attribute accesses with value type `Oracle` ⊆ `{"invoke", "invoke_stream", "invoke_messages", "invoke_field", "tokenize"}`.

A new method on `Oracle` requires updating this test — that is the point.

### G2 — `test_oracle_ownership.py` (enforces I2)

AST-walks `src/fsm_llm/dialog/turn.py` and asserts no `Call` node whose `func` is `LiteLLMOracle` or `_LiteLLMOracle`. Asserts `MessagePipeline.__init__` has a parameter named `oracle` typed as `Oracle`.

### G3 — `test_layering.py` (enforces I4 + checks CAND-E)

For every `.py` module under `src/fsm_llm/` (excluding the allow-listed `lam/__init__.py`):
- Resolve the module's layer from its path (`runtime/` → L1; `handlers.py` → L2; `stdlib/`, `dialog/compile_fsm.py` → L3; `program.py` → L4; `dialog/*` other than `compile_fsm.py` → "frontend-internal").
- Walk `Import` and `ImportFrom` nodes; assert the imported module's layer is permitted by the table in I4.
- Walk `__all__` in `fsm_llm/__init__.py`; assert each name is in exactly one of the four layer frozensets or in the legacy block.

### G4 — `test_deprecation_calendar.py` (enforces I5)

Reads `fsm_llm.__version__`. For each alias listed in I5:
- If `version < (0, 6, 0)`: invoking the alias emits no warning.
- If `(0, 6, 0) ≤ version < (0, 7, 0)`: invoking the alias emits exactly one `DeprecationWarning` containing the recommended replacement.
- If `version ≥ (0, 7, 0)`: importing or invoking the alias raises `AttributeError` or `ImportError`.

### G5 — Theorem-2 universality bench (enforces I6)

`evaluation/bench_long_context_*.json` schema gains `program_kind ∈ {"fsm", "term", "factory"}`. For every cell:
- `theorem2_holds: true` is required.
- For `program_kind == "fsm"`, until M3d ships, the cell may be marked `theorem2_holds_with_caveat: "non_cohort_response_via_callback"` — but the caveat list is logged in `decisions.md` with a removal date, and the merge is not declared "complete" until the caveat list is empty.

`scripts/bench_long_context.py` gains a `--program-kind {fsm,term,factory,all}` filter. CI runs the FSM cell on a representative subset of `examples/dialog/`.

---

## 6b. Deferred work — follow-on plan(s)

The 2026-04-28 plan (`plans/plan_2026-04-28_6597e394/`) shipped the structural merge core (M1+M2+M4+M3a+M3b) and stopped at the M3c default flip. The remaining work decomposes into two follow-on plans, sequenced.

### Follow-on Plan A — "Finish M3 (default flip + retire `_cb_respond`)"

The four divergences blocking M3c (per D-009 in the 2026-04-28 plan's `decisions.md`) are independently designable sub-PRs. Each must close before the default flip can land:

- **D1 — Empty-`response_instructions` short-circuit** (`turn.py:2171-2187`, D-R10-7.4). The legacy path constructs a `"."`-sentinel oracle.invoke that LiteLLM treats as a no-op + returns a synthetic `f"[{state.id}]"` for `add_system_message`. The bare Leaf always issues a real call. Lift the sentinel into either a compile-time emission split (cohort-style guard) or a Leaf-side `schema_ref` + post-process callback.
- **D2 — `add_system_message` history append**. The legacy `_cb_respond` updates `instance.context.conversation` with the response; the bare Leaf does not. Thread conversation-history append via a follow-on Let after the response Leaf, or via an outer wrap.
- **D3 — Structured-output `_output_response_format` enforcement** on terminal states (`turn.py:2210-2213`). Legacy enforces a Pydantic schema from `instance.context.data`; bare Leaf has `schema_ref=None` and emits unstructured. Resolve via `schema_ref` on the response Leaf for terminal states.
- **D4 — Streaming sibling — `process_compiled_stream`** rebinds `CB_RESPOND` with `_make_cb_respond_stream` (`turn.py:577`); the bare Leaf path has no streaming counterpart, so `converse_stream` returns a non-streaming string under default=True. **This is the gating decision**: design a `LeafStream` AST variant or a streaming-aware oracle protocol. NOT foreseen in spec §4 CAND-C; surfaced only at M3c HALT. Theorem-2 universality literally cannot include streaming under the current single-Leaf-returns-single-string design.

D4 may justify a kernel-design plan-cycle of its own before D1–D3 can be sequenced cleanly.

### Follow-on Plan B — M5 (deprecation calendar) + M6 (out-of-scope payload)

Both calendars in spec §3 I5 and §5 M6 are independent of the structural merge. Recommended decomposition:

- **M5** — single PR: `tests/test_fsm_llm/test_deprecation_calendar.py` + `CHANGELOG.md` calendar entry. Silent in 0.5.0 (no warning code). Warning-emit landing PRs are the 0.6.0 release sequence.
- **M6a** — Monitor span schema migration. After M3 ships universal, OTEL spans emit per-Leaf; document the v1→v2 schema bump in `fsm_llm_monitor/CLAUDE.md`. Provide one-time backfill script if production sessions are mid-flight.
- **M6b** — Session-store migration. `scripts/migrate_session_store.py` walks a `FileSessionStore` and rewrites legacy `transition_context` entries to the post-M3 shape. Idempotent.
- **M6c** — Sibling shim deprecation hook. One-line `warnings.warn(...)` in `fsm_llm_{reasoning,workflows,agents}/__init__.py`, gated on the I5 calendar (silent in 0.5.x).
- **M6d** — Third-party `LLMInterface` contract documentation. CLAUDE.md note clarifying `LiteLLMOracle` wrap-and-preserve behaviour for subclasses + the `oracle=` escape hatch for non-LiteLLM custom oracles.

### Future "decouple-kernel-from-dialog" plan

The 5-entry `_PRE_EXISTING_DIALOG_IMPORT_ALLOWLIST` in `test_layering.py` is grandfathered debt. Each of the five entries currently imports `FSMError` (or a request/response model) from `fsm_llm.dialog.definitions`. Relocating `FSMError` to a neutral module (e.g. `fsm_llm.runtime.errors` or `fsm_llm.errors`) would let the allow-list drop to 1 entry (the `lam/__init__.py` shim). Out of scope for the merge; called out here as the natural follow-up that would make I4 strict-by-default. Estimated impact: 28+ file ripple (per D-003 in the 2026-04-28 plan).

---

## 7. The CLI dispatch table (closes L5; pins R12)

`fsm-llm run <target>` dispatches by target shape:

| Target shape | Resolves to | Notes |
|---|---|---|
| `*.json` (path with `.json` suffix) | `Program.from_fsm(load_definition(target))` | Interactive REPL by default; `--non-interactive --message "..."` for one-shot |
| `pkg.module:fn[(...)]` (contains `:`) | `Program.from_factory(import_callable(target.split(':')[0]+':'+target.split(':')[1]), *args, **kwargs)` | `--inputs '{"q": "..."}'` or `--inputs path.json` populates env |
| anything else | error: `"target must be either a .json FSM path or a pkg.module:fn factory reference"` |

Subcommands: `run`, `explain`, `validate`, `visualize`, `meta`, `monitor` (unchanged from R7). Aliases: `fsm-llm-run`, `fsm-llm-explain`, etc., kept silent in 0.5.x; warn in 0.6.x; removed in 0.7.x per I5.

`fsm-llm explain <target> --n <int> --K <int>` returns the planner's `Plan` for every `Fix` subtree. For FSM targets, `--n / --K` are ignored (FSM compiled terms have no Fix subtrees today; M3d will enable Plan output for the per-state Leaf chain).

---

## 8. What this merge does **not** do

- **Does not change FSM JSON v4.1 semantics.** T5 (`docs/lambda.md` §12) remains the contract through M3.
- **Does not deprecate FSMs.** Category-A dialogs are first-class; `Program.from_fsm(...)` is permanent.
- **Does not introduce full System F typing.** Per `docs/lambda.md` §14: monomorphisation at parse time stays.
- **Does not add async.** `oracle.invoke_stream` is sync-iterator. Async is a separate workstream.
- **Does not rewrite stdlib.** M3 stdlib (`agents/`, `reasoning/`, `workflows/`, `long_context/`) is already λ-native and Theorem-2-clean.
- **Does not move the M5 long-context work earlier.** Independent.
- **Does not remove `LiteLLMInterface`.** It is the implementation `LiteLLMOracle` wraps; users with custom subclasses keep working (M6d).
- **Does not solve the 6-host-side-handler-timing problem.** R5's narrowing (`START/END_CONVERSATION`, `PRE/POST_TRANSITION`, `CONTEXT_UPDATE`, `ERROR`) stays host-side **permanently** — the merge spec elevates this from "deferred" to "contract". The 2 AST-side timings (`PRE/POST_PROCESSING`) are the universe of what's term-side. Documented in `lambda.md` §6.3 as the final word.

---

## 9. Migration guide for users (one-page)

A user on 0.4.x with this code:

```python
from fsm_llm import API, LiteLLMInterface, HandlerBuilder
api = API.from_file("my_bot.json", llm_interface=LiteLLMInterface(model="gpt-4o-mini"))
handler = HandlerBuilder().on_pre_processing(my_func).build()
api.register_handler(handler)
conv_id, greeting = api.start_conversation()
reply = api.converse("hello", conv_id)
```

migrates to 0.5.0 incrementally. Step 1: keep working unchanged (no warnings). Step 2 (0.6.0 prep): rewrite as

```python
from fsm_llm import Program, Handler, LiteLLMOracle, LiteLLMInterface
oracle = LiteLLMOracle(LiteLLMInterface(model="gpt-4o-mini"))
program = Program.from_fsm("my_bot.json", oracle=oracle, handlers=[Handler.on_pre_processing(my_func)])
result = program.invoke(message="hello")  # auto-starts conversation
print(result.value)                       # the reply
print(result.conversation_id)             # for the next turn
```

Stateless / pipeline / long-context users get the substrate at the top of the import:

```python
from fsm_llm import Program, niah
program = Program.from_factory(niah, question="What is the capital?", tau=256, k=2)
result = program.invoke(inputs={"document": doc})
print(result.value)
print(result.oracle_calls, "==", result.plan.predicted_calls)  # Theorem-2 in user code
```

A `MIGRATION.md` shipped with 0.5.0 documents the call-site rewrites for the four common patterns (FSM dialog, term pipeline, factory invocation, handler registration).

---

## 10. Closing argument — what changes for a reader

After M1–M6 land:

- **A user reading `from fsm_llm import …` knows which layer they touch.** L4 (`Program`), L3 (`compile_fsm`, `react_term`, `niah`, `leaf`, `fix`), L2 (`compose`, `Handler`), L1 (`Term`, `Executor`, `Oracle`). The architecture is the import statement.
- **A user calling `program.invoke(...)` cannot construct a misuse the type checker doesn't catch.** Mode is fixed at construction; the verb dispatches; the return type is `Result`.
- **A user inspecting Theorem-2 evidence sees one bench schema.** `program_kind ∈ {fsm, term, factory}`; `theorem2_holds: true` is required for every cell.
- **An engineer adding a new feature cannot accidentally re-introduce disjointedness.** The five falsification gates (G1–G5) fail loudly on the next narrowing.
- **A maintainer reading this document instead of `lambda.md` + `lambda_integration.md` v2.0 knows what to ship and when.** The seven channels, six invariants, five design decisions, six commits, five gates — and the deprecation calendar — are spec, not narrative.

One runtime. Two surface syntaxes. One verb. One Oracle. One Result. One bench schema. One import organisation. One deprecation calendar.

**The merger is the contract that makes all six "ones" enforceable.**

---

## 11. References

- `docs/lambda.md` — architectural thesis (v0.2, 2026-04-24).
- `docs/lambda_integration.md` v2.0 — audit + R8–R13 plan (2026-04-27); superseded by this document for the unified-API contract.
- HEAD source verified 2026-04-28: `program.py:300` (Program.invoke), `runtime/oracle.py` (Oracle protocol), `dialog/turn.py` (post-R10/R13 wiring), `dialog/compile_fsm.py:135` (R9c gate-drop), `cli/main.py:56` (R12 factory dispatch).
- `analyses/analysis_2026-04-27_13dc12d8/` — the LITE-tier epistemic-deconstruction pass that produced this spec (Phase 0 setup, Phase 1 boundary mapping with HEAD audit, Phase 1.5 abductive expansion yielding CAND-A–E, Phase 5 synthesis).
- `plans/LESSONS.md` — D-003 / D-008 / D-011 (oracle structured-decode, prompt-builder reuse, Pydantic schema patches) — load-bearing context for I1's contract.
- `plans/plan_2026-04-27_a426f667/decisions.md` — D-PLAN-09 family (R3 narrowing rationale that informed CAND-D).
- `plans/plan_2026-04-27_32652286/` — the in-flight plan that landed R8/R9c/R11/R12 (HEAD's mid-merge state).
