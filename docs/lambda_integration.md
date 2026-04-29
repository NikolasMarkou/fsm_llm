# Refactoring Report v2.0 — Closing the λ/FSM Disjointedness

> **⚠ SUPERSEDED.** This document is preserved for historical reference. The canonical document for the merge — invariants, falsification gates, deprecation calendar, and the unified-API specification an engineer implements against — is now [`docs/lambda_fsm_merge.md`](lambda_fsm_merge.md) (2026-04-29). It supersedes this report's R8–R13 plan with a sequenced commit-level spec; M1–M6 (minus M6b deferred-with-cause) and A.D4/A.D5/A.M3c/A.M3d-narrowed have all shipped per `git log` since 2026-04-28. New work goes against `lambda_fsm_merge.md`.

**Status**: v2.0 (2026-04-27). Supersedes v1.0 (preserved in git history at commit `6ef05b1` and earlier — see `git log -- docs/lambda_integration.md`).
**Scope**: Architectural integration of `fsm_llm/runtime/` (substrate) and `fsm_llm/dialog/` (FSM front-end), as the codebase stands at HEAD after v1.0's seven refactors all shipped *narrowed*.
**Companion**: `docs/lambda.md` (the thesis — what was promised at the architecture level).

---

## 0. Executive Summary

v1.0 promised seven refactors (R1–R7) to collapse the FSM front-end onto the λ-substrate. **All seven shipped.** Three shipped at full scope (R1 Program facade, R2 compile-cache, R7 CLI). Four shipped narrowed (R3 oracle, R4 module rename, R5 handlers, R6 cohort Leaf). The narrowing left a codebase where:

- The substrate (`runtime/`) is correct and universal — Theorem-2 holds for every λ-DSL Fix node.
- The dialog front-end (`dialog/`) is a parallel-callback architecture sitting *next to* the substrate, not *on* it. **Six call sites in `dialog/pipeline.py` still invoke `LiteLLMInterface` directly**, bypassing the unified Oracle (lines 1185, 1227, 1270, 1606, 2146, 2188).
- The `Program` facade is mode-bifurcated: `.run` raises in FSM mode; `.converse` raises in term mode; `.register_handler` raises in term mode; `.explain()` returns `plans=[]` for FSM (no Fix subtrees yet).
- `pipeline.py` *grew* from 2,032 LOC to **2,236 LOC** post-v1.0 (cohort + R5 splicing added paths without removing the legacy ones).
- Top-level `__init__.py` exports **93 names — zero are λ-flavoured**. The substrate is invisible at the public surface.
- The R6 cohort-Leaf gate is **default-OFF** (`FSM_LLM_COHORT_EMISSION=0`). Theorem-2's cost model therefore does *not* apply to FSM dialogs in production today.
- Forward-compat plumbing — `to_template_and_schema` on three prompt builders, `oracle.invoke(env=)` — exists, is tested, and **has no live caller**.

**Diagnosis**: v1.0 navigated the structural change correctly. The narrowing was risk-driven, principled, and well-documented (D-S1-02 thru D-S1-04, D-PLAN-09). But the cumulative effect is a codebase whose surface still tells two stories. A reader importing `fsm_llm` cannot tell which layer they are in. A user invoking `Program.from_fsm(...).run(...)` gets `NotImplementedError` for what reads like the natural call.

**This report (v2.0) names eight remaining loci of disjointedness (§1), proposes six refactors (R8–R13) for a clean v0.5.0 (§3), and ends with a public-surface contract where the import statement itself reveals the layer.**

The unifying principle is *one verb per layer*. The verb a user reaches for tells the runtime which layer they meant. Today the verb is buried under namespace; v2.0 promotes it to the import surface.

---

## 1. v1.0 Refactor Status — What Actually Shipped

| # | Refactor | Status | Evidence | Residual gap |
|---|---|---|---|---|
| R1 | `Program` facade | **Narrow** | `program.py` (543 LOC). `from_fsm`/`from_term`/`from_factory` all real. `run`/`converse`/`register_handler` mode-bifurcated; `explain()` returns `plans=[]` for FSM by contract. | One facade with three modal-failure surfaces. Users hit `NotImplementedError` on reasonable calls. |
| R2 | Compile-cache to kernel | **Full** | `dialog/compile_fsm.compile_fsm_cached` (lru_cache(64), keyed on `(fsm_id, json_dump)`). `FSMManager.get_compiled_term` is a 3-line shim. | None. Closed. |
| R3 | Unify oracle | **Narrow / forward-compat only** | `runtime/oracle.py:143-165` — `LiteLLMOracle.invoke(prompt, schema, *, model_override, env)`; D-008 structured-decode bypass at `oracle.py:211-214`. `dialog/prompts.py` builders gained `to_template_and_schema` (lines 565, 915, 946, 1446, 1478). | **Pipeline callbacks do NOT use the oracle.** Six direct `LiteLLMInterface` call sites remain (`pipeline.py:1185, 1227, 1270, 1606, 2146, 2188`). Forward-compat plumbing has no live caller. |
| R4 | Module reorganisation | **Full + shims** | `runtime/` (was `lam/`); `dialog/` (was top-level). 9 module shims (`fsm_llm.api`, …) and 10 lam submodule shims. Identity contract verified by `tests/test_fsm_llm/test_module_shims.py`. | Gross structure correct. Closed at the file-system level. |
| R5 | Handlers as AST transforms | **Narrow (2 of 8 timings)** | `handlers.py:1117-1167`. Term-side: PRE_PROCESSING, POST_PROCESSING (HOST_CALL splices). Host-side: 6 remaining timings (architecturally infeasible per D-S1-02). | The 6 host-side timings stay host-side permanently. Closed within the architecturally feasible envelope; closed enough. |
| R6 | Lift FSM callbacks to Leaf | **Narrow + DEFAULT-OFF** | `compile_fsm.py:143-321`. `_is_cohort_state`: terminal-only (no transitions, no extractions). `FSM_LLM_COHORT_EMISSION` env gate, default false. | Theorem-2 cost model not applied to FSM dialogs in production. Seven `CB_*` host-callables remain (lines 104-110). |
| R7 | CLI unification | **Narrow** | `cli/main.py` (379 LOC) — 6 subcommands: `run`, `explain`, `validate`, `visualize`, `meta`, `monitor`. | `fsm-llm run pkg.module:factory_term --inputs '{...}'` does **not** work. CLI accepts `--fsm path.json` only. Stdlib factories and user λ-DSL programs are not runnable from the shell. |

**Observation**: Four refactors *shipped at half scope under different names* — risk-tagged narrowings, principled and reversible, but each leaves a gap the next clean-up plan must close.

---

## 2. Eight Loci of Remaining Disjointedness

Each below names: *what the thesis (T) or v1.0 contract states*, *what the code at HEAD does (C)*, *the gap (G)*.

### L1. The Facade Lies By Mode

- **T**: `Program` is the single user-facing facade. *"One runtime. Two surface syntaxes."*
- **C**: `program.py:241-298`. `Program.from_fsm(d).run()` raises `NotImplementedError` (line 264). `Program.from_term(t).converse(m, c)` raises `NotImplementedError` (line 292). `Program.from_term(t).register_handler(h)` raises (line 466). `Program.explain()` returns `plans=[]` for FSM-mode programs because no Fix subtrees exist in compiled FSMs today.
- **G**: The "single facade" exposes three different shape-classes that each accept and reject different methods. A user cannot reach for `program.run(...)` without first remembering whether they constructed it from FSM or factory. This is the inverse of a facade — the user must already know the implementation to use the interface.

### L2. The Oracle Is Built But Not Mounted

- **T**: §10 of `docs/lambda.md` — *"Same code path for all three [Categories]. This is the payoff."* R3 in v1.0: *"Six paths, three call-shapes, one model — collapse to one."*
- **C**: At HEAD, `LiteLLMOracle.invoke(prompt, schema=, *, env=)` exists, is tested, and includes the D-008 structured-decode bypass (`oracle.py:211-214`). Three `*PromptBuilder` classes have `to_template_and_schema` producers (`prompts.py:565, 915, 1446`). The free `classification_template` lives at `prompts.py:1478`. **None of these are called by the FSM hot path.** The six direct `LiteLLMInterface` call sites in `pipeline.py` are byte-equivalent to pre-R3 code.
- **G**: The unified-oracle abstraction was *built* but not *wired*. From the user's perspective, R3 has not shipped. From the test suite's perspective, R3 ships duplicate behaviour. From the cost-model's perspective, FSM dialogs are silent: per-Leaf telemetry, schema enforcement, and the structured-decode bypass do not apply to *any* FSM running with the default cohort gate off.

### L3. Theorem-2's Cost Model Is Off By Default

- **T**: §12 of `docs/lambda.md`, T2 — *"For every Fix node, plan.predicted_cost bounds actual spend within 5%."* Universal property of the runtime.
- **C**: `compile_fsm.py:143-153`. `FSM_LLM_COHORT_EMISSION` reads as bool from env, default `False`. Cohort eligibility is the strictest possible: terminal state, no transitions, no extractions. Most FSMs in `examples/` have *zero* cohort-eligible states.
- **G**: T2 is a property of programs that opt in. The "universal" claim in §12 is conditional on a flag set on a subset of states in a minority of FSMs. The shipped path preserves byte-equivalence — but at the cost of the headline guarantee.

### L4. The Public Surface Hides The Substrate

- **C**: `src/fsm_llm/__init__.py` (448 LOC) exports **93 names**. The split:
  - 38 FSM-flavoured (`FSMDefinition`, `FSMInstance`, `State`, `Transition`, `API`, `FSMManager`, `Conversation`, `HandlerSystem`, `HandlerBuilder`, `HandlerTiming`, `Classifier`, `HierarchicalClassifier`, `IntentRouter`, `MessagePipeline`, …).
  - 55 shared infrastructure (`LLMInterface`, `LiteLLMInterface`, exceptions, validators, visualizers, memory, sessions, logging, utilities).
  - **0 λ-flavoured.** No `Term`, no `Executor`, no `leaf`, no `fix`, no `Program`. Users discover the substrate by reading `docs/lambda.md`, not by tab-completing `fsm_llm.<TAB>`.
- **G**: A reader of the public surface infers from the surface itself that this is a stateful-dialog framework with a hidden λ extension. The architectural truth is the inverse. The package surface tells the historical order, not the architecture.

### L5. The CLI Cannot Run Factories

- **C**: `cli/main.py:241`. The `run` subcommand parser (`_add_run_subparser`, line 56) accepts `--fsm path.json`. There is no `--factory pkg.module:fn`, no `--inputs '{...}'`, no `--explain`. Stdlib factories (`react_term`, `niah`, `aggregate`, etc.) and user-authored λ-DSL programs ship with no shell entry point.
- **G**: R7 closed L6 from v1.0 only halfway. A user wanting to invoke `fsm_llm.stdlib.long_context.niah` from the shell must write a Python script. The CLI's existence implies "you can run programs from the shell"; the actual answer is "FSM JSON only."

### L6. The Kernel Namespace Points Into The Front-End

- **C**: `runtime/__init__.py` re-exports `compile_fsm` and `compile_fsm_cached` (per `runtime/CLAUDE.md` — *"the actual file lives at `dialog/compile_fsm.py`"*). The `fsm_llm.lam.fsm_compile` shim resolves to `fsm_llm.dialog.compile_fsm`.
- **G**: The substrate namespace points into the dialog namespace. Layer L1 (substrate) imports from L3 (dialog factory) via re-export. The folder rename was correct; the cross-layer back-reference contradicts the layering it implements. Every CLAUDE.md has to footnote this.

### L7. Forward-Compat Plumbing Without A Live Caller

- **C**: `dialog/prompts.py` `to_template_and_schema` (4 entry points, ~250 LOC); `runtime/oracle.py` `invoke(env=)` branch (~30 LOC); `dialog/prompts.py` `classification_template` free function (~35 LOC). Tested. **No production caller.**
- **G**: ~315 LOC of dead code from the user's perspective. The plumbing exists because R3 was narrowed (D-PLAN-09-RESOLUTION-step14-narrowed). Either it gets wired (R10 below) or it gets deleted. Today it does neither — it lives in the codebase as an unfunded promissory note.

### L8. `pipeline.py` Grew, Did Not Shrink

- **C**: 2,236 LOC at HEAD (was 2,032 LOC at the v1.0 report). Cohort emission added paths; R5 splicing added paths. Legacy callback bodies preserved for byte-equivalence.
- **G**: v1.0's prediction was "2,032 → ~400 after R5 + R6". The actual delta is +204 LOC. The seven `CB_*` host-callables (`compile_fsm.py:104-110`) and their 7 factory functions (`pipeline.py:709-865+`) and their 6 `LiteLLMInterface` call sites (lines 1185-2188) all coexist with the new oracle path. The reason is policy (D-S1-04 default-OFF gate), not code (R6 mechanically works for cohort states). The legacy path is preserved; the new path is dormant.

---

## 3. Target Architecture — *One Verb Per Layer*

The four-layer model from v1.0 is correct. Its v1.0 phrasing — namespaces — is incomplete. v2.0 phrases each layer as **one verb the user reaches for**, and arranges the public import surface so the verb is what the user types.

```
                    ┌──────────────────────────────────────────────┐
   L4  INVOKE       │  Program(...).invoke(...)                    │
                    │   one method, mode-detected at construction  │
                    │   verb = invoke                              │
                    └────────────────┬─────────────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────────────┐
   L3  AUTHOR       │  Term producers — return Term, no I/O        │
                    │    compile_fsm(defn) | react_term(...)       │
                    │    | niah(...) | leaf/fix/let_/...           │
                    │   verb = build a Term                        │
                    └────────────────┬─────────────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────────────┐
   L2  COMPOSE      │  Pure AST→AST transforms                     │
                    │    compose(term, [handler1, instrument, …])  │
                    │   verb = wrap a Term                         │
                    └────────────────┬─────────────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────────────┐
   L1  REDUCE       │  Typed substrate                             │
                    │    Executor.run(term, env) → result          │
                    │    Planner.plan(...) → Plan                  │
                    │    Oracle.invoke(template, env, schema)      │
                    │   verb = β-reduce                            │
                    └──────────────────────────────────────────────┘
```

### Three invariants that make this layering enforceable

1. **L1 imports nothing.** (Already true.)
2. **L2 imports only L1.** (`handlers.py` currently imports from `dialog.pipeline` — see R10/R13.)
3. **L3 imports only L1 + L2.** Specifically: `dialog/compile_fsm.py` imports from `runtime/`, never the reverse. (Currently violated by `runtime/__init__.py` re-exporting `compile_fsm` — see R13.)

A reader who knows the verb knows the layer. A user who knows the layer knows what to import:

```python
# L4 — the dominant case
from fsm_llm import Program

# L3 — when authoring directly
from fsm_llm import compile_fsm           # FSM JSON
from fsm_llm import react_term, niah      # stdlib
from fsm_llm import leaf, fix, let_       # raw DSL

# L2 — wrapping
from fsm_llm import compose, instrument

# L1 — when implementing
from fsm_llm import Term, Executor, Plan, Oracle
```

**Every name in the example above is a top-level import.** That is the v0.5.0 contract.

---

## 4. The Single-Verb Unification — `Program.invoke`

L4 today exposes three modal-failure verbs (`run`, `converse`, `register_handler`). v2.0 collapses them into one mode-aware entry:

```python
class Program:
    def invoke(
        self,
        message: str | None = None,             # dialog turn
        *,
        inputs: dict | None = None,             # term env
        conversation_id: str | None = None,     # opt-in dialog state
        explain: bool = False,                  # → returns ExplainOutput
    ) -> Result:
        """One entry. Mode is detected from how the program was built."""

    # Back-compat aliases (deprecated 0.5.0, removed 0.6.0)
    def run(self, **env): return self.invoke(inputs=env).output
    def converse(self, msg, conv_id=None): return self.invoke(message=msg, conversation_id=conv_id).output
```

**Mode detection is unambiguous because it lives at construction time**, not call time:
- `Program.from_fsm(...)` ⇒ dialog mode. `invoke(message=..., conversation_id=...)`.
- `Program.from_term(...)` / `Program.from_factory(...)` ⇒ term mode. `invoke(inputs=...)`.

A misuse (`from_fsm(...).invoke(inputs=...)`) raises a *single* descriptive error class (`ProgramModeError`) with the right method named in the message — not five different `NotImplementedError`s scattered across the API.

`register_handler` becomes a constructor parameter, not a post-construction mutation:

```python
program = Program.from_fsm(defn, handlers=[my_handler, another])      # ← new
program = Program.from_term(term, handlers=[trace_instrument])         # ← new (works, because handlers are AST transforms — R5)
```

Existing `program.register_handler(h)` continues to work in dialog mode by appending to the handler list and re-compiling. In term mode it ceases to be NotImplementedError — handler composition (R5's HOST_CALL Combinator) is already universal, so the term-mode case is the easier path.

`Result` is a small dataclass: `(output, plan, leaf_calls, oracle_calls, conversation_id)`. The `explain=True` short-circuit returns the same shape with `output=None` and the plan populated.

---

## 5. The Six Refactors (R8–R13)

Each is independently shippable. R8 is the user-visible win. R9–R10 close the cost-model and oracle gaps. R11–R13 are cleanup that drops LOC.

### R8. Single-verb facade — `Program.invoke` and ambient mode  (HIGH payoff, LOW risk)

**Closes**: L1.

**Concretely**:
- Add `Program.invoke(...)` per §4. Old `.run`/`.converse`/`.register_handler` keep working as aliases.
- Add `ProgramModeError(FSMError)` for misuse.
- Move `handlers=` to a constructor param on all three `from_*` classmethods. `register_handler` keeps working but is documented as a post-construction convenience.
- Make `.explain()` return runtime plans when invoked on a term-mode program with `inputs=` supplied (currently only static `(n,K)` form works).
- Add `Result` dataclass; `invoke(...)` returns `Result`, not raw output.

**LOC**: ~+250 in `program.py`, ~+30 in `__init__.py`, ~+40 tests.

**Validation**: existing `program.run(**env)` and `program.converse(msg, id)` continue to pass; new tests assert `Program.from_fsm(d).invoke(inputs=...)` raises `ProgramModeError` (not `NotImplementedError`); FSM-mode `register_handler` byte-equals `handlers=[h]` constructor path.

### R9. Universal Leaf emission — flip the cohort gate to default-ON, then drop it  (HIGH payoff, MEDIUM risk)

**Closes**: L3 (and removes a layer of L8 ceremony).

**Step 9a (one PR)**: Flip `FSM_LLM_COHORT_EMISSION` default from `False` → `True`. Run the full 2,899-test suite + a representative production FSM smoke battery. Theorem-2 strict equality `oracle_calls == predicted_calls` must hold for cohort states; non-cohort states preserve byte-equivalence. The current narrow cohort (terminal-only, no extractions) is the only definition that ships in this step.

**Step 9b (later PR)**: *Widen* the cohort definition to "any state whose response generation does not depend on classification or field-extraction" (most response-only states with deterministic transitions qualify). This requires emitting `Leaf` for `_cb_respond` even when transitions exist — straightforward because the response Leaf does not consume transition output.

**Step 9c (R10's prerequisite)**: Drop the gate entirely. Cohort emission becomes the only path for response generation. Non-response callbacks (extraction, classification) remain host-callables until R10.

**LOC**: 9a/9b ~+50/-20; 9c removes the gate (~-30 LOC).

**Validation**: bench scorecards under `evaluation/bench_long_context_*.json` extend to FSM examples after 9c. Theorem-2 evidence per (model × FSM) cell.

### R10. Pipeline-callback collapse to `oracle.invoke`  (HIGH payoff, HIGH risk)

**Closes**: L2 fully, L7 (wires the dead plumbing), most of L8.

This is the v1.0 R3 + R6 work that was deferred. The unified-oracle abstraction (`runtime/oracle.py:143-165`) and the template-producer surface (`dialog/prompts.py:565+`) are *already shipped and tested*; what's missing is the wiring.

**Concretely** (six call sites at HEAD):

| File:Line | Today | After R10 |
|---|---|---|
| `pipeline.py:1185` | `self.llm_interface.generate_response_stream(req)` | `self.oracle.invoke_stream(template, env, schema)` |
| `pipeline.py:1227` | `self.llm_interface.generate_response(req)` (response gen) | `self.oracle.invoke(template, env, schema)` |
| `pipeline.py:1270` | `self.llm_interface._make_llm_call(messages, "data_extraction")` | `self.oracle.invoke(template, env, schema=DataExtraction)` |
| `pipeline.py:1606` | `self.llm_interface.extract_field(req)` | `self.oracle.invoke(template, env, schema=field_schema)` |
| `pipeline.py:2146` | `self.llm_interface.generate_response(req)` (classifier) | `self.oracle.invoke(template, env, schema=ClassificationResponse)` |
| `pipeline.py:2188` | `self.llm_interface.generate_response(req)` (classifier resp) | `self.oracle.invoke(template, env, schema=ClassificationResponse)` |

For each, the template comes from the corresponding `*PromptBuilder.to_template_and_schema()` (already shipped at R3 step 14).

**Risk** is high because every FSM dialog flows through these callbacks; T5 (semantic preservation) is the regression gate. Mitigation: ship behind a per-callback feature flag (`FSM_LLM_ORACLE_<callback>=1`), one callback per PR, with the existing 837 dialog tests as the gate. After all six flags ship green, drop the flags.

**LOC delta**: pipeline.py shrinks from 2,236 → ~1,300 (~-900 LOC). The legacy LLM path inside `LiteLLMInterface` (request/response wrappers) becomes `runtime/oracle.py`'s implementation detail; `LiteLLMInterface` becomes private (`runtime/_litellm.py` already, just stop re-exporting).

**Payoff**: Theorem-2 universal across all FSM call shapes — extraction, classification, response, ambiguity resolution all become Leaf nodes with cost telemetry. The "six paths, three call-shapes, one model" of L2 collapses to "one path, one call-shape, one model."

### R11. Promote the substrate at the public surface  (HIGH payoff, LOW risk)

**Closes**: L4.

`fsm_llm/__init__.py` adds the substrate names, prominently:

```python
# fsm_llm/__init__.py — proposed top of __all__ (v0.5.0)
__all__ = [
    # L4 — the dominant case
    "Program", "Result", "ExplainOutput", "ProgramModeError",
    # L3 — authoring
    "compile_fsm", "react_term", "rewoo_term", "reflexion_term", "memory_term",
    "niah", "aggregate", "pairwise", "multi_hop",
    "leaf", "fix", "let_", "case_", "var", "abs_", "app",
    "split", "fmap", "ffilter", "reduce_", "concat", "cross", "peek",
    # L2 — composition
    "compose", "Handler", "HandlerTiming", "HandlerBuilder",
    # L1 — substrate
    "Term", "Executor", "Plan", "PlanInputs", "plan",
    "Oracle", "LiteLLMOracle", "CostAccumulator", "LeafCall",
    # legacy FSM types (still public, but moved to bottom of the list)
    "FSMDefinition", "FSMInstance", "State", "Transition", "API", ...
]
```

The 38 FSM-flavoured names stay public (back-compat) but lose their position-of-honour. Users tab-completing `fsm_llm.` see `Program` first.

**LOC**: ~+30 lines of `__all__` reorganisation, ~+40 lines of new re-exports.

**Validation**: `python -c "from fsm_llm import Program, Term, leaf, fix, react_term, niah, compile_fsm; …"` works without intermediate paths.

### R12. Factory-target CLI runner — `fsm-llm run pkg.mod:fn`  (MEDIUM payoff, LOW risk)

**Closes**: L5.

```bash
fsm-llm run examples/dialog/form_filling/fsm.json                    # FSM JSON (existing)
fsm-llm run fsm_llm.stdlib.long_context:niah --inputs inputs.json    # factory
fsm-llm run my_pipeline:my_term --inputs '{"q": "..."}'              # user term
fsm-llm explain my_pipeline:my_term --n 4096 --K 2048                # planner output
```

The dispatch rule: if the target ends in `.json` → `Program.from_fsm`; if it contains `:` → `Program.from_factory(import_path)`; else → error with usage hint.

**LOC**: ~+150 in `cli/run.py`, ~+25 tests.

### R13. Cleanup — kernel back-reference, dead plumbing, module shims  (LOW payoff, LOW risk)

**Closes**: L6, L7, L8 residuals.

- **L6 fix**: Move `compile_fsm`'s top-level re-export from `fsm_llm.lam`/`fsm_llm.runtime` into `fsm_llm` (the package root, where R11 places it anyway). `runtime/__init__.py` stops re-exporting it; the back-reference is broken. `fsm_llm.lam.fsm_compile` shim retires per the 0.6.0 schedule.
- **L7 fix**: After R10 wires the forward-compat plumbing, the dead-code claim becomes moot — those ~315 LOC are now exercised by every FSM call.
- **L8 fix**: After R9c + R10, `pipeline.py` drops to ~400 LOC of pure orchestration (turn-routing + session glue). Rename to `dialog/turn.py` and inline the remaining helper into `dialog/compile_fsm.py`.
- **Module shim retirement**: per the 0.5.0/0.6.0 schedule. `fsm_llm.lam` emits `DeprecationWarning` in 0.5.0; removed in 0.6.0. Same for the 9 dialog shims.

**LOC**: net -200 to -400 depending on inlining choices.

---

## 6. Migration Sequencing

Each refactor is one PR. The six combine into a clean v0.5.0 release.

| # | Refactor | Touches | Net LOC | Tests gate |
|---|---|---|---|---|
| R8 | `Program.invoke` | `program.py`, `__init__.py` | +250 / +30 | Existing 837 dialog + new ~40 |
| R9a | Cohort gate default-ON | `compile_fsm.py:143-153` | ~+5 / -2 | Existing 837 dialog + 202 kernel; smoke battery |
| R9b | Widen cohort definition | `compile_fsm.py:154-200` | ~+80 / -20 | 837 + 202 + new ~30 |
| R9c | Drop the gate | `compile_fsm.py` | -30 | 837 + 202 |
| R10 | Pipeline → oracle | `pipeline.py`, `prompts.py`, `oracle.py` | +150 / -1100 | All 2,899 (T5 semantic preservation) |
| R11 | Public-surface promotion | `__init__.py` (only) | +70 / -0 | New ~15 import tests |
| R12 | Factory CLI | `cli/run.py`, `cli/main.py` | +150 / -10 | New ~25 |
| R13 | Cleanup | several | -300 / +50 | All 2,899 |

**Net delta after all six**: roughly **−1,000 LOC** (driven mostly by R10's pipeline collapse), **+50 tests**, **public surface tells the architectural truth**.

**Recommended order**:
1. R8 (user-visible win, no risk)
2. R11 (zero risk; once R8 lands, the substrate names are useful)
3. R12 (parallel to R11; small)
4. R9a → R9b → R9c (each behind the prior's smoke)
5. R10 (the deep work; depends on R9c so the cohort path is the only response-emission shape)
6. R13 (cleanup; depends on R10)

**Total elapsed time at one PR/week**: ~9 weeks. R8 + R11 + R12 (3 weeks) yield the user-visible cleanliness. R9 + R10 (4 weeks) yield the cost-model universality. R13 (1–2 weeks) compacts the codebase.

---

## 7. What This Refactor Does Not Do

- **Does not change FSM JSON v4.1 semantics.** T5 (`docs/lambda.md` §12) remains the contract.
- **Does not deprecate FSMs.** Category-A dialogs stay first-class; `Program.from_fsm(...)` is permanent.
- **Does not remove `API`.** It becomes a thin shim over `Program.from_fsm(...)` (already half-true) and stays exported.
- **Does not rewrite `MessagePipeline`.** R10 collapses its callback bodies; the orchestration shell becomes `dialog/turn.py`. The class survives renamed and shrunk.
- **Does not introduce full System F typing.** Per `docs/lambda.md` §14: monomorphisation at parse time stays.
- **Does not require the M5 long-context work to land first.** Independent.
- **Does not add async.** Streaming via `oracle.invoke_stream` is sync-iterator. Async is a separate concern.

---

## 8. Validation Plan

For each of R8–R13:

1. **Test count**: existing 2,899 tests must not drop; new tests accompany each new abstraction.
2. **T5 semantic preservation**: `Program.from_fsm(d).invoke(message=m, conversation_id=c)` byte-equals legacy `API.from_definition(d).converse(m, c)` for every (FSM, input) pair in the regression corpus.
3. **Theorem-2 universality** (post-R10): bench scorecards extend to FSM cells. `evaluation/bench_long_context_*.json` schema gains a `program_kind: "fsm" | "term" | "factory"` axis. `theorem2_holds: true` required for every cell.
4. **Performance**: `pytest tests/test_fsm_llm/` wall-clock within +10% of v0.4.x baseline. AST-emit overhead is real but bounded; if R9b/R10 push past +10%, a per-FSM warmed-cache hint goes in `compile_fsm_cached`.
5. **Public API**: `python -c "from fsm_llm import Program, Term, leaf, fix, react_term, niah, compile_fsm, Executor, Oracle; …"` succeeds after R11.
6. **CLI**: `fsm-llm run examples/dialog/form_filling/fsm.json`, `fsm-llm run fsm_llm.stdlib.long_context:niah --inputs ...`, `fsm-llm explain ... --n 4096 --K 2048` all return exit 0.
7. **Examples**: all 152 examples (`examples/`) green via `scripts/eval.py`. The 90.8% baseline (Run 004, 2026-04-02) is the floor.

After R13, this document should be closed out (renamed to `docs/lambda_integration_history.md` or merged into `docs/lambda.md` §13). The architectural thesis stops needing an audit companion.

---

## 9. Summary

The codebase ships v1.0's seven refactors. Three at full scope; four narrowed. The cumulative effect is a substrate that is correct and a front-end that sits next to the substrate rather than on it.

The fix is six refactors, sequenced over one release cycle:

- **R8** gives users one verb (`invoke`) that does not lie about mode.
- **R9** flips the cost-model from opt-in to default and then drops the gate.
- **R10** wires the unified-oracle plumbing that v1.0 R3 shipped dormant — six pipeline call sites collapse to one `oracle.invoke`.
- **R11** promotes the substrate to the public surface so tab-completion teaches the architecture.
- **R12** lets the CLI run the factories the kernel was built for.
- **R13** drops the back-references, the dead plumbing, and most of `pipeline.py`.

After they land:

- **One facade** with one verb, mode-aware at construction.
- **One oracle** for every LLM call across both surfaces.
- **One Theorem-2** that holds universally — FSM and λ-DSL alike.
- **One public surface** where `from fsm_llm import Program, Term, leaf, react_term, compile_fsm` works, and the order of names matches the order of layers.

One runtime. Two surface syntaxes. **And the import statement says so.**

---

## References

- `docs/lambda.md` — the architectural thesis. This document is its v2.0 audit + plan.
- v1.0 of this report — preserved in `git log -- docs/lambda_integration.md`.
- `src/fsm_llm/runtime/CLAUDE.md` — kernel file map (post-R4).
- `src/fsm_llm/dialog/CLAUDE.md` — dialog file map (post-R4).
- `src/fsm_llm/CLAUDE.md` — package-level file map.
- `plans/plan_2026-04-27_a426f667/decisions.md` — R1/R2/R3-narrow/R4 decisions (D-PLAN-02, D-PLAN-09-RESOLUTION-step14-narrowed, D-PLAN-10).
- `plans/plan_2026-04-27_1b5c3b2f/decisions.md` — R5-narrow/R6-narrow decisions (D-S1-02, D-S1-03, D-S1-04).
- `plans/plan_2026-04-27_43d56276/decisions.md` — D-STEP-08-RESOLUTION (R6+ deferral rationale).
- `plans/LESSONS.md` — D-003, D-008, D-011 (oracle structured-decode, prompt-builder reuse, Pydantic schema patches).
- `evaluation/bench_long_context_*.json` — Theorem-2 evidence per (model × factory) cell. Post-R10, this also covers FSM cells.
