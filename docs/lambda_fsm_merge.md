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

> **Crucial fact about HEAD (2026-04-29, post `plan_0f87b9c4` close)**: M1 + M2 + M4 + M3a + M3b + D1 + D2 + D3 + **A.D4** + **A.D5** + **A.M3c** + **A.M3d-narrowed** + M5 + M6a + M6c + M6d have all shipped. **The merge is now Theorem-2-universal-by-default for non-terminal FSM programs.** The D5-AGENT divergence was resolved on the test-infrastructure side (option (b'-narrow) in `plan_0f87b9c4`/decisions.md D-001): no stdlib code touch, no env-var gate — just per-test mock refactors + a generalised wire-shape assertion. What remains is **B.M5/M6 (0.6.0 release sequence — live deprecation warnings + sibling shim hooks + R13-epoch removals)**, **B.M6b (session-store migration, was M3c-blocked, now ready to re-scope)**, and the **future kernel/dialog decoupling** (Plan C — 28-file ripple). See §6b for full follow-on plan details.

---

## Status — Implementation progress (as of 2026-04-29, post `plan_0f87b9c4` close)

Five plans have shipped, in order:

- **`plans/plan_2026-04-28_6597e394/`** shipped **M1 + M2 + M4 + M3a + M3b**. M3c halted on Pre-Mortem Scenario A (default-flip surfaced 4 dialog runtime divergences including an unforeseen streaming-sibling break).
- **`plans/plan_2026-04-28_f1003066/`** shipped **D1 + D2 + D3** (3 of the 4 divergences) behind the M3a opt-in flag, plus **M5 + M6a + M6c + M6d** as docs/test scaffold. The opt-in path is now semantically complete for non-streaming, non-terminal, non-empty-instructions states.
- **`plans/plan_2026-04-28_ca542489/`** shipped **A.D4** — the streaming kernel design (candidate (b): `Leaf.streaming` capability flag + `Executor.run(*, stream: bool = False)` per-call mode + secondary `StreamingOracle` Protocol with `isinstance` check; base `Oracle` Protocol untouched per D-STEP-6-T1 mock-conformance precedent). 5 commits, +22 tests (3094 → 3116). The 4th D-009 divergence (streaming) is now closed; opt-in users get full Theorem-2-correct streaming via the new D2 `Leaf(streaming=True)` path.
- **`plans/plan_2026-04-28_90d0824f/`** shipped **A.D5** — the terminal-state Leaf-emission compile-time pathway (`State.output_schema_ref` field + `compile_fsm._compile_state` D3 branch routing to `Leaf(schema_ref=<dotted-path>, streaming=False)` when set). 5 commits, +6 tests (3158 → 3164). Pre-Mortem Scenario A trigger fired on the M3c default flip step (3 behavioural regressions in `tests/test_fsm_llm_agents/test_bug_fixes.py` confirmed the named hypothesis: stdlib agents' construction path depends on default-OFF shape). Plan halted iter-2 cleanly per D-005 protocol; iter-3 ran NARROWED (steps 5+M3d-narrowed BLOCKED; steps 6–8 doc-only).
- **`plans/plan_2026-04-29_0f87b9c4/`** shipped **D5-AGENT resolution + A.M3c + A.M3d-narrowed** atomically. The D5-AGENT divergence dissolved on closer inspection: the 3 "behavioural regressions" in `test_fsm_llm_agents/test_bug_fixes.py` were artefacts of a brittle `SequenceMockLLM` whose cursor advanced per-`generate_response` call (D1 short-circuits empty-`response_instructions` states to 0 oracle calls under the new path, drifting the cursor). The 4th was a wire-shape assertion that pinned the legacy `user_message=` kwarg shape. **No stdlib code touch. No env-var gate.** Resolution: per-test `FunctionalMockLLM` (call-count-keyed callables) + a generalised "user message reaches the LLM via either wire field or system_prompt" assertion. Then the flip + the `_make_cb_respond_stream` retirement landed cleanly. 4 commits (`30e2cda` test fixes; `1046443` wire-shape; `d85b614` flip + 23 shape-assertion inversions; `e40399d` M3d-narrowed drop, ~-110 net source LOC). Suite: 3164 → ~3164 (test renames; ruff clean). 0 PIVOT, 0 autonomy-leash hits, Pre-Mortem A NOT triggered. **I6 universal-by-default for non-terminal FSM programs.**

| Milestone | Status | Commit(s) | Notes |
|---|---|---|---|
| M1 — `Result` everywhere | ✅ SHIPPED | `f3b6a1a` | `Program.invoke` returns `Result` in all modes; 3 FSM-mode tests inverted+renamed; CHANGELOG entry. |
| M2 — Layered `__all__` + import-audit | ✅ SHIPPED | `2f080f5` | `_LAYER_L1..L4` frozensets; `test_layering.py` (12 tests). **Allow-list landed at 5 entries, not the spec's 4** — `handlers.py` is a 5th `FSMError`-inheritance coupling found mid-EXECUTE (D-007-SURPRISE); same pattern, same future "decouple-kernel-from-dialog" plan. |
| M4 — Oracle ownership on Program | ✅ SHIPPED | `da7d03c` | `Program → API → FSMManager → MessagePipeline` thread one `Oracle`; 7 (not 5 — count drifted) `LiteLLMOracle(...)` calls in `dialog/turn.py` collapsed to `self._oracle` field-reads; `test_oracle_ownership.py` ships (5 tests, G2 AST-gate + identity propagation). |
| M3a — Private opt-in field scaffolding | ✅ SHIPPED | `a25b899` | `_emit_response_leaf_for_non_cohort: bool = False` Pydantic-private State attr; `compile_fsm._compile_state` plumbed via `getattr`; byte-equivalent at default. |
| M3b — Non-cohort Leaf emission (opt-in) | ✅ SHIPPED | `7a1e506` | `Let(NONCOHORT_RESPONSE_PROMPT_VAR, App(CB_RENDER_RESPONSE_PROMPT, instance), Leaf("{...}"))`; `_make_cb_render_response_prompt` host factory; +31 strict-Theorem-2 tests; default-False preserves all 837 dialog tests. **I6 holds for opt-in non-cohort programs.** |
| M3.D1 — Empty-`response_instructions` synthetic gate (opt-in) | ✅ SHIPPED | `a78334b` | `CB_RESPOND_SYNTHETIC` constant + `_make_cb_respond_synthetic` host factory + compile-time gate. **0 oracle calls** for opt-in states with empty `response_instructions` (cleaner than legacy's 1 sentinel `oracle.invoke(".")`). 6 new tests. |
| M3.D2 — Outer `Let` with curried `CB_APPEND_HISTORY` (opt-in) | ✅ SHIPPED | `f4d127e` | Wraps the M3b inner `Let` so the response Leaf's return value is appended to `instance.context.conversation` via `App(App(CB_APPEND_HISTORY, instance), value)`. Theorem-2 strict equality preserved (host App uncounted). 9 existing M3b shape-tests updated via `_inner_m3b_let` walker; 3 new D2 tests. |
| M3.D3 — Terminal opt-in fallback to legacy `CB_RESPOND` | ✅ SHIPPED | `115fcdf` | Compile-time guard: `if not state.transitions:` → `App(CB_RESPOND, instance)`. Conservative because `_output_response_format` is runtime-injected via `instance.context.data` (uninspectable at compile time). 3 new D3 tests. |
| M3.A.D4 — Streaming kernel design (`Leaf.streaming` + `StreamingOracle`) | ✅ SHIPPED | `fe22e9c`, `949d6f9`, `7032101`, `ded4147`, `e4480ae` | Candidate (b) per `plan_ca542489` D-001: `Leaf.streaming: bool = False` capability flag + `Executor.run(*, stream: bool = False)` per-call mode kwarg + secondary `StreamingOracle` Protocol (base `Oracle` immutable; 0 mock-conformance break). Compile-time mutual exclusion `streaming=True ⊥ schema_ref != None` (D-005). D2 `Leaf(streaming=True)` for opt-in non-cohort. `process_stream_compiled` rewired. +22 tests (3094 → 3116). Closes the 4th D-009 divergence. |
| M3.A.D5 — Terminal opt-in `State.output_schema_ref` → Leaf | ✅ SHIPPED | `e15c53c`, `c3501a1`, `7b85e45`, `14b5a16`, `f649eeb` | New `State.output_schema_ref: Any = None` field; when set to a `BaseModel` subclass (or pre-formatted dotted-path string), `compile_fsm._compile_state` routes the D3 terminal-non-cohort branch to `leaf(template, schema_ref=<dotted>, streaming=False)` inside the D2 outer-Let. Compile-time validation raises `ASTConstructionError` on malformed input. D-007-SURPRISE caught pre-commit: kernel `Leaf.schema_ref` is `str | None` (dotted path), not a class — compiler converts class → `f"{cls.__module__}.{cls.__qualname__}"` automatically. Default `None` preserves D3 byte-equivalent for every State at HEAD (stdlib agents migration deferred per D-002 inherited from `plan_ca542489`). +6 tests (3158 → 3164). |
| M3c — Flip default to True | ✅ SHIPPED | `d85b614` (`plan_0f87b9c4`) | Default flipped 2026-04-29. D5-AGENT resolved on the test-infrastructure side (option (b'-narrow) per `plan_0f87b9c4`/decisions.md D-001): added `FunctionalMockLLM` to `test_bug_fixes.py` (per-call-count callables, decoupled from `generate_response` cadence), generalised `test_site_7_5_invoke_preserves_user_message_on_wire` to assert "user message reaches the LLM via wire field OR system_prompt". 23 shape-assertion tests (10 in core, 12 in kernel, 1 in pipeline) updated to call `_disable_leaf` / `_legacy_defn` for explicit-False regression coverage. Pre-Mortem A NOT triggered. **I6 universal-by-default** for non-terminal FSM programs. The D3 conservative terminal-non-cohort fallback (`App(CB_RESPOND, instance)` for terminal states without `output_schema_ref`) survives — gated on universal `output_schema_ref` adoption, scope of M3d-wide. |
| M3d-narrowed — Drop `_make_cb_respond_stream` + env-rebind | ✅ SHIPPED | `e40399d` (`plan_0f87b9c4`) | The streaming sibling closure factory + its support method `_stream_response_generation_pass` retired entirely (~110 net source LOC dropped from `dialog/turn.py`). Streaming now flows through Executor.run(stream=True) on the D2 `Leaf(streaming=True)` chain via `StreamingOracle.invoke_stream`. Residual `App(CB_RESPOND, instance)` paths (D3 terminal fallback + explicit-False regression coverage) yield a single `str` wrapped to a single-chunk Iterator[str] by `process_stream_compiled` — chunked streaming unavailable for those minority paths but functional parity preserved. Test marker `D-R10-7.6 (step 8 finalised)` updated to assert the stronger contract. |
| M5 — Deprecation calendar | ✅ SHIPPED | `0d10ef8` (spec), `ed5fb53` (test+CHANGELOG) | Two-epoch reconciliation (R13 rows already-warning + I5 rows silent-then-warn) — see I5. `tests/test_fsm_llm/test_deprecation_calendar.py`: 20 active + 19 future-version-skipped assertions. **Live warning emit for the 4 I5-epoch surfaces is the 0.6.0 release commit** — out of scope for 0.5.0. |
| M6a — Monitor span schema doc | ✅ SHIPPED | `3327f95` | `src/fsm_llm_monitor/CLAUDE.md` v1/v2 `span_schema_version` table. Live per-Leaf span routing (v2) was M3c-blocked; now ready to ship as a follow-on (M3c shipped). |
| M6b — Session-store migration | ⬜ DEFERRED — **spec error**, now M3c-unblocked | — | `transition_context` (named in spec §5) has zero source occurrences. M3c has shipped; the schema delta (if any) can now be re-scoped against the post-flip `SessionState`. Concrete delta is up to a fresh plan. |
| M6c — Sibling shim deprecation hook | ✅ SHIPPED (test scaffold) | `ed5fb53` | The 0.5.x silent-assertion is exercised by the M5 calendar test (rows 8: `import fsm_llm_{reasoning,workflows,agents}` silent at 0.3.x). Live `warnings.warn(...)` lands in 0.6.0 release. |
| M6d — Third-party `LLMInterface` contract doc | ✅ SHIPPED | `ac7b21e` | `src/fsm_llm/runtime/CLAUDE.md` documents the wrap-and-preserve behaviour for ABC overrides + `_invoke_structured` bypass caveat (`oracle.py:449-572`) + escape-hatch via `oracle=` kwarg. |

**What this means for I6 today**: Theorem-2 universality holds for term/factory programs (already true at HEAD pre-M3), AND for **every non-terminal FSM program by default** (post-A.M3c). The full **D1 + D2 + D3 + A.D4 (streaming) + A.D5 (terminal `output_schema_ref`)** semantics now fire universally, with no opt-in flag required from authors. The single residual caveat is the **D3 conservative terminal-non-cohort fallback**: terminal states with neither `output_schema_ref` nor empty `response_instructions` still emit `App(CB_RESPOND, instance)` because `_output_response_format` is runtime-injected via `instance.context.data` (uninspectable at compile time). This caveat retires when stdlib agents adopt `output_schema_ref` and M3d-wide can drop `_make_cb_respond` and the `CB_RESPOND` constant — a documentation-driven migration over several minor releases, not a single plan. **The merge is now Theorem-2-universal-by-default for the dominant FSM cohort** (any FSM with at least one non-terminal state); the explicit-False path remains queryable via `state._emit_response_leaf_for_non_cohort = False` and is exercised by the regression-coverage test sets (`_disable_leaf` / `_legacy_defn` helpers).

**Suite count**: 3060 (post-`6597e394`) → 3094 (post-`f1003066`) → 3116 (post-`ca542489`) → 3164 (post-`90d0824f`) → **~3164** (post-`0f87b9c4`; 23 shape-assertion tests updated rather than added). Explicit-False path byte-equivalent under `_disable_leaf` / `_legacy_defn` regression coverage. No PIVOT in any of the five plans; the prior M3c HALTs in `6597e394` and `90d0824f` were both clean Pre-Mortem A triggers reverted per protocol — `0f87b9c4` finally shipped the flip cleanly because the diagnosis pinpointed test-infrastructure brittleness (mock cursor coupling, wire-shape over-pinning) rather than stdlib semantics. Same institutional discipline (D-009/D-005, "trust the protocol, don't fix-forward") survived; zero autonomy-leash hits across all five plans; zero stdlib code touched in `0f87b9c4` despite shipping the flip.

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

**Invariant status (2026-04-29, post `plan_0f87b9c4` close):**
- **I1** ⬜ DEFERRED — needs oracle-call-paths AST audit (G1); not yet gated. M4 shipped ownership; G1 enforcement remains future work.
- **I2** ✅ SHIPPED via M4 (commit `da7d03c`); G2 test enforces zero `LiteLLMOracle(...)` calls in `dialog/turn.py`.
- **I3** ✅ Mode is fixed at construction; no runtime mode-switch. Held since R8.
- **I4** ✅ partial — G3 ships via M2 (commit `2f080f5`) with a documented **5-entry** allow-list (one more than spec). 5th entry is `handlers.py` (`HandlerSystemError(FSMError)` MRO). All 5 entries point to a future "decouple-kernel-from-dialog" plan.
- **I5** ✅ partial (two-epoch reconciled) — `tests/test_fsm_llm/test_deprecation_calendar.py` (commit `ed5fb53`) gates HEAD behaviour: R13-epoch rows (lam + 9 dialog/llm shims + dialog/pipeline) already warn at 0.3.0 with "removal in 0.6.0" text; I5-epoch rows (Program methods + sibling shim packages) silent at 0.3.0. **Live warning emit for the 4 I5-epoch surfaces is 0.6.0 release work** — separate commit, schedule fixed in this plan.
- **I6** ✅ **UNIVERSAL by default for non-terminal FSM programs** (post-`plan_0f87b9c4`). All four D-009 divergences (D1/D2/D3/D4) closed, A.D5 closes the structured-terminal path under opt-in `output_schema_ref`, and A.M3c flipped the default. The single residual caveat is the **D3 conservative terminal-non-cohort fallback** for terminal states without `output_schema_ref` — `_make_cb_respond` and `CB_RESPOND` survive there; M3d-wide retirement is gated on universal `output_schema_ref` adoption (documentation-driven migration, separate plan).

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

### I5 — Back-compat alias contract: two calendar epochs (R13 contract + new I5 contract).

**Calendar epochs.** Investigation during plan `plan_2026-04-28_f1003066` (cf. `findings/m5-deprecation-calendar.md`) found that three of the eight alias rows ship a `DeprecationWarning` ALREADY at HEAD (0.3.0) under the R13 contract — predating this spec. Re-authoring those warnings to the I5 schedule would change the user-facing contract for code that has already shipped. The merge spec accordingly recognises **two calendar epochs**:

- **R13 epoch** (silent → warn at 0.3.0+ → remove at 0.6.0). Applies to rows whose warnings ship today. The 0.6.0 removal date is the contract these users have been promised by the warning text in `lam/__init__.py:34-47` and the 10 dialog/llm shim files.
- **I5 epoch** (silent in 0.5.x → warn at 0.6.0 → remove at 0.7.0). Applies to rows that ship NO warning today; the merge spec adds the schedule.

| Surface | Epoch | 0.3.x–0.5.x | 0.6.x | 0.7.x |
|---|---|---|---|---|
| `Program.run(**env)` | I5 | works silently | `DeprecationWarning` | removed |
| `Program.converse(msg, conv_id)` | I5 | works silently | `DeprecationWarning` | removed |
| `Program.register_handler(h)` | I5 | works silently | `DeprecationWarning` | removed (use `handlers=` constructor arg) |
| `from fsm_llm import API` | I5 | works silently | `DeprecationWarning` | removed (use `Program.from_fsm`) |
| `from fsm_llm.lam import …` | **R13** | **already warns** | warns | **removed (R13 contract)** |
| `from fsm_llm.{api,fsm,pipeline,prompts,classification,…} import …` | **R13** | **already warns** | warns | **removed (R13 contract)** |
| `from fsm_llm.dialog.pipeline import MessagePipeline` | **R13** | **already warns** | warns | **removed (R13 contract)** |
| `import fsm_llm_reasoning / fsm_llm_workflows / fsm_llm_agents` | I5 | works silently | `DeprecationWarning` | removed (use `fsm_llm.stdlib.*`) |

The 0.5.0 release ships the merge complete (`Program`, layered surface, oracle ownership); 0.6.0 is the warning release for the four I5 rows AND the removal release for the three R13 rows; 0.7.0 is the removal release for the four I5 rows. A CI test (§6 G4) — landed as `tests/test_fsm_llm/test_deprecation_calendar.py` per plan `plan_2026-04-28_f1003066` — asserts both epochs against `__version__`.

### I6 — Theorem-2 holds for every Program a user can construct from public APIs.

For every `(model, program_kind)` cell in the regression bench, `oracle_calls == predicted_calls` strictly (or `oracle_calls ≤ predicted_calls` for cases with documented sentinel-arm short-circuit, e.g. `multi_hop_dynamic`). `program_kind ∈ {"fsm", "term", "factory"}`. The bench scorecard schema gains the `program_kind` axis; the `theorem2_holds: true` field is required for every cell.

**The residual caveat** (post-`plan_0f87b9c4`): the default for `_emit_response_leaf_for_non_cohort` is now **True**, so non-terminal non-cohort states automatically lift their response generation to a real `Leaf` — D1 covers empty-instructions synthetic responses, D2 covers standard non-cohort responses (with curried `CB_APPEND_HISTORY` for history append), and A.D4 covers streaming (per-Leaf `streaming=True` capability + per-call `Executor.run(stream=True)` mode + `StreamingOracle` Protocol). For **terminal** non-cohort states, A.D5 lifts the response to `Leaf(schema_ref=<dotted>)` when `State.output_schema_ref` is set; otherwise the **D3 conservative fallback** keeps `App(CB_RESPOND, instance)` (because `_output_response_format` is runtime-injected via `instance.context.data` and uninspectable at compile time). The opt-in field is now an opt-OUT field: setting `_emit_response_leaf_for_non_cohort=False` per-state restores legacy emission (used by the regression-coverage test sets via `_disable_leaf` / `_legacy_defn`). Full retirement of `_make_cb_respond` and `CB_RESPOND` (M3d-wide) is gated on universal `output_schema_ref` adoption — a documentation-driven migration over several minor releases, not a single plan.

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

### CAND-C — Non-cohort FSM states get a Leaf-emission story (✅ SHIPPED universal-by-default for non-terminal across M3a/b + D1/D2/D3 + A.D4 + A.D5 + A.M3c + A.M3d-narrowed).

Pre-M3, `compile_fsm._compile_state` emitted `App(CB_RESPOND, instance)` for any state with extractions or transitions; only "cohort-eligible" states (terminal, no transitions, no extractions, no extracted-field references in the response prompt) got a real `Leaf`. The M3 sub-PR train extended Leaf emission to **every** non-cohort state under the `_emit_response_leaf_for_non_cohort` opt-in:

1. **M3a** (`a25b899`) added the `_emit_response_leaf_for_non_cohort: bool = False` State field as the migration gate.
2. **M3b** (`7a1e506`) implemented the standard non-cohort case: `Let(NONCOHORT_RESPONSE_PROMPT_VAR, App(CB_RENDER_RESPONSE_PROMPT, instance), Leaf("{...}"))`.
3. **D1** (`a78334b`) handled empty-`response_instructions` via a `CB_RESPOND_SYNTHETIC` host-callable (0 oracle calls).
4. **D2** (`f4d127e`) wrapped M3b's inner `Let` in an outer `Let(NONCOHORT_RESPONSE_VAR, _inner, App(App(CB_APPEND_HISTORY, instance), value))` for history append. Theorem-2 strict equality preserved (host App uncounted).
5. **D3** (`115fcdf`) added the conservative terminal-state fallback to legacy `App(CB_RESPOND, instance)` because `_output_response_format` is runtime-injected and uninspectable at compile time.
6. **A.D4** (`fe22e9c` → `e4480ae`) closed the streaming divergence: `Leaf.streaming` capability flag + `Executor.run(stream=)` per-call mode + secondary `StreamingOracle` Protocol. D2's response Leaf is emitted with `streaming=True` for opt-in non-cohort. Compile-time mutual exclusion `streaming=True ⊥ schema_ref != None` (D-005).
7. **A.D5** (`e15c53c` → `f649eeb`) lifted D3's conservative terminal fallback for migrated states: `State.output_schema_ref` (default None) routes the terminal-non-cohort branch to `Leaf(schema_ref=<dotted>, streaming=False)` when set; default None preserves D3 byte-equivalent.

8. **A.M3c** (`d85b614`, `plan_0f87b9c4`) flipped the default to True. The "D5-AGENT" gate from `plan_90d0824f` dissolved on closer inspection — the failures were test-infrastructure artifacts (mock cursor coupling under D1's 0-call short-circuit, wire-shape over-pinning), not stdlib semantics. Resolution: per-test `FunctionalMockLLM` (call-count-keyed callables) + a generalised wire-shape assertion. NO stdlib code touched.
9. **A.M3d-narrowed** (`e40399d`, same plan) atomically retired `_make_cb_respond_stream` and `_stream_response_generation_pass` (~110 net source LOC removed from `dialog/turn.py`). Streaming flows through Executor.run(stream=True) on the D2 `Leaf(streaming=True)` chain via `StreamingOracle.invoke_stream`.

The original spec said "remove the field in the third PR"; reality is that the `_emit_response_leaf_for_non_cohort` field is **transitional**: the default is True (the old "opt-in" is now the default) but the field still accepts `False` for explicit-legacy regression coverage. It retires together with M3d-wide. The `_cb_respond` Python callable for terminal-without-`output_schema_ref` states is **temporarily permanent** (D3 conservative path can't be lifted without universal `output_schema_ref` adoption).

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

**Status**: ✅ SHIPPED across 5 plans. M3a (`a25b899`) + M3b (`7a1e506`) opt-in scaffolding + Leaf emission. D1/D2/D3/A.D4/A.D5 added the divergence-closing variants behind the opt-in flag. **A.M3c flipped the default to True (`d85b614`, `plan_0f87b9c4`)** after a clean re-diagnosis dissolved the prior D5-AGENT halt into test-infrastructure artifacts — no stdlib code touched. **A.M3d-narrowed retired `_make_cb_respond_stream` (`e40399d`, same plan)** atomically with the flip. Theorem-2 universal-by-default for non-terminal FSM programs. M3d-wide (full `_make_cb_respond` + `CB_RESPOND` retirement) remains gated on universal `output_schema_ref` adoption — separate doc-driven migration.

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

**Status (2026-04-28)**: ✅ SHIPPED in `plan_2026-04-28_f1003066` (commits `0d10ef8` spec amendment + `ed5fb53` test+CHANGELOG). Two-epoch contract (R13 already-warning + I5 silent-then-warn) — see I5 above.

**Touches**: `tests/test_fsm_llm/test_deprecation_calendar.py` (new), `CHANGELOG.md` (calendar entry).
**Concretely**:
- The four I5-epoch aliases ship silent in 0.5.0; the three R13-epoch aliases continue to warn (already shipped at 0.3.0).
- The I5-epoch warnings land in 0.6.0; the R13-epoch removals also land in 0.6.0 (separate release commits).
- `test_deprecation_calendar.py` reads `__version__` and asserts the **two-epoch** schedule from §3 I5:
  - if `version < (0, 6, 0)`: I5-epoch aliases silent; R13-epoch aliases warn (matches HEAD).
  - if `(0, 6, 0) ≤ version < (0, 7, 0)`: I5-epoch aliases warn; R13-epoch aliases raise `AttributeError`.
  - if `version ≥ (0, 7, 0)`: I5-epoch aliases raise `AttributeError`.
- A documentation entry under `CHANGELOG.md` commits both calendars.
**Net LOC**: +120 / −0 (test + CHANGELOG + spec table reconciliation). **Risk**: zero in 0.5.0 — the test asserts current-HEAD behaviour, and the two epochs are codified rather than the spec being silently ignored.
**Validation**: the calendar test passes against 0.3.0 today; the future-version branches are exercise-ready and will activate at the next two version bumps.

### M6 — Out-of-scope items: monitor, sessions, sibling shims, third-party LLMInterface (the H_S_prime payload)

**Status (2026-04-28)**: ✅ PARTIAL SHIPPED in `plan_2026-04-28_f1003066`. M6a (commit `3327f95`), M6c (covered by M5 test `ed5fb53`), and M6d (commit `ac7b21e`) shipped. **M6b DEFERRED — spec error**: `transition_context` precondition is factually wrong at HEAD; see §6b for re-scope.

**Touches**: `src/fsm_llm_monitor/CLAUDE.md` (M6a doc), `src/fsm_llm/runtime/CLAUDE.md` (M6d doc), `tests/test_fsm_llm/test_deprecation_calendar.py` (M6c — covered by the M5 test).
**Concretely** (each sub-item is one short PR or a doc-only commit):
- **M6a Monitor span schema**: Document a `span_schema_version` table in `fsm_llm_monitor/CLAUDE.md` — `v1` is the current FSM-level schema (`conversation_start`/`state_transition`/`pre_processing` etc., `otel.py:188-265`); `v2` is the planned per-Leaf schema landing AFTER M3c flips. The live OTEL routing change is M3c-blocked (the executor's `CostAccumulator` does not emit per-Leaf spans yet); the doc table is shippable today and signals the schema bump to consumers in advance.
- **M6b Session-store migration**: ⬜ **DEFERRED — spec error.** Investigation in `findings/m6-payload.md` found that `transition_context` is a string that appears nowhere in `SessionState` (`dialog/session.py:41-53`) or anywhere else in the source tree at HEAD. The spec's stated precondition is factually wrong; no migration script can be written until M3c (and possibly D2's `add_system_message` lift) defines a real schema delta. Re-scope this sub-item alongside the eventual M3c default flip.
- **M6c Sibling shim deprecation hook**: `fsm_llm_{reasoning,workflows,agents}/__init__.py` gain a `warnings.warn(..., DeprecationWarning, stacklevel=2)` line **in the 0.6.0 release commit** — gated on the I5 calendar (silent in 0.5.x). The 0.5.x assertion (silent at 0.3.0) is exercised by `test_deprecation_calendar.py` rows 8 (M5 covers M6c).
- **M6d Third-party LLMInterface contract**: Document in `runtime/CLAUDE.md`: "Any `LLMInterface` subclass passed to `Program.from_fsm(llm=...)` is wrapped in `LiteLLMOracle` automatically; subclass overrides of the ABC's `generate_response` / `extract_field` / `generate_response_stream` are preserved (`oracle.py:185-195, 341, 428, 446`). **Caveat**: `LiteLLMOracle._invoke_structured` (`oracle.py:449-572`) bypasses `self._llm.generate_response` for Executor structured-Leaf calls — it calls `litellm.completion` directly via `getattr(self._llm, ...)`. Subclasses that override `generate_response` to add provider-side logic do NOT get that override invoked on structured Leaves. Custom oracles bypass `LiteLLMOracle` entirely — implement the `Oracle` protocol directly and pass via `oracle=`."
**Net LOC**: +60 / −0 (docs only). **Risk**: zero (no code change).
**Validation**: doc-review only.

### Actual ship trajectory (as of 2026-04-29, post `plan_0f87b9c4` close)

The originally-recommended order held, with the M3 sub-decomposition expanded twice under live constraint (D1–D4 in `plan_f1003066` and `plan_ca542489`; D5 in `plan_90d0824f`; D5-AGENT resolution + the long-deferred A.M3c+A.M3d-narrowed in `plan_0f87b9c4`):

1. **M1** ✅ shipped (`f3b6a1a`) — `Program.invoke` returns `Result` uniformly.
2. **M2** ✅ shipped (`2f080f5`) — layered `__all__` + import-audit (5-entry allow-list).
3. **M4** ✅ shipped (`da7d03c`) — Program-owned Oracle threaded through API → FSMManager → MessagePipeline.
4. **M3** ✅ **complete** — M3a (`a25b899`) + M3b (`7a1e506`) shipped opt-in scaffolding + Leaf emission. M3c flip HALTED on Pre-Mortem A in `plan_6597e394` (4 divergences). `plan_f1003066` closed **D1 (`a78334b`) + D2 (`f4d127e`) + D3 (`115fcdf`)** behind the opt-in flag. **A.M3c flipped + A.M3d-narrowed retired in `plan_0f87b9c4`** (`d85b614`, `e40399d`).
5. **M5** ✅ shipped (`0d10ef8` spec + `ed5fb53` test+CHANGELOG) — two-epoch calendar reconciled with shipped R13 contract.
6. **M6** ✅ partial shipped — M6a (`3327f95`) + M6c (covered by M5 test `ed5fb53`) + M6d (`ac7b21e`). M6b deferred for cause (spec error: `transition_context` never existed); now M3c-unblocked for re-scope.
7. **A.D4** ✅ shipped in `plan_ca542489` (`fe22e9c` → `e4480ae`, 5 commits) — streaming kernel design candidate (b): `Leaf.streaming` flag + `Executor.run(stream=)` mode + `StreamingOracle` Protocol. Closes the 4th D-009 divergence under opt-in. Suite 3094 → 3116.
8. **A.D5** ✅ shipped in `plan_90d0824f` (`e15c53c` → `f649eeb`, 5 commits) — `State.output_schema_ref` field + compile-time D3 lift to `Leaf(schema_ref=<dotted>, streaming=False)` for migrated terminal states. D-007-SURPRISE caught and resolved pre-commit (kernel `Leaf.schema_ref` is `str | None` dotted-path, not class — compiler converts). Suite 3158 → 3164.
9. **A.M3c default flip** ✅ shipped in `plan_0f87b9c4` (`d85b614`) after the long M3c HALT in `plan_6597e394` and `plan_90d0824f`. The diagnosis was that the failures during the prior flip experiments were test-infrastructure artifacts (mock cursor coupling, wire-shape over-pinning), not stdlib semantics — see "Follow-on Plan D5-AGENT" in §6b. 23 shape-assertion tests updated, `_disable_leaf` / `_legacy_defn` helpers added for explicit-False regression coverage. Pre-Mortem A NOT triggered. Suite ~3164.
10. **A.M3d-narrowed** ✅ shipped in `plan_0f87b9c4` (`e40399d`) atomically with A.M3c — dropped `_make_cb_respond_stream`, `_stream_response_generation_pass`, and the `D-STEP-5-NARROWED` env-rebind block. ~110 net source LOC removed.

**Remaining sequence** (per §6b): **B.M5/M6** (0.6.0 release: live warnings + R13 removals) → **G1** (oracle-call-paths AST audit, independent) → **G5** (FSM `program_kind` cells in bench scorecard) → **B.M6b** (session migration re-scope) → **Plan C** (decouple kernel from dialog; 28-file ripple) → **A.M3d-wide** (eventual `output_schema_ref`-universal cleanup).

---

## 6. Falsification gates (the tests that detect a regression of the merge)

These are not test ideas — they are the spec's enforcement mechanisms. Without them, every invariant is decorative.

**Gate status (2026-04-29, post `plan_0f87b9c4` close):**
- **G2** ✅ shipped (M4, `da7d03c`).
- **G3** ✅ shipped (M2, `2f080f5` — with 5-entry allow-list).
- **G4** ✅ shipped (M5, `ed5fb53` — `tests/test_fsm_llm/test_deprecation_calendar.py`, two-epoch).
- **G1** ⬜ DEFERRED — oracle-call-paths AST audit. Plumbing is in place (M4); the AST-walk enforcement test is future work. Independent — can ship anytime.
- **G5** ⬜ DEFERRED — Theorem-2 universality bench schema (`program_kind` axis on `evaluation/bench_long_context_*.json`). M3c-unblocked (post-`plan_0f87b9c4`); the bench-side work that adds an FSM-flavoured driver to `bench_long_context.py` is now meaningful but still out of scope here.

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

Five prior plans shipped: `plans/plan_2026-04-28_6597e394/` (structural core M1+M2+M4+M3a+M3b; stopped at M3c default flip), `plans/plan_2026-04-28_f1003066/` (D1+D2+D3 behind opt-in + M5 + M6a/c/d docs/test), `plans/plan_2026-04-28_ca542489/` (A.D4 streaming kernel design candidate (b)), `plans/plan_2026-04-28_90d0824f/` (A.D5 terminal opt-in pathway; M3c HALT + iter-3 NARROWED doc-only), and **`plans/plan_2026-04-29_0f87b9c4/`** (D5-AGENT resolution + A.M3c default flip + A.M3d-narrowed retirement, all atomic). The remaining work decomposes into the follow-on plans below.

### Follow-on Plan A.D4 — ✅ SHIPPED in `plan_2026-04-28_ca542489/`

Candidate (b) chosen per `plan_ca542489` D-001: `Leaf.streaming: bool = False` capability flag (per-Leaf opt-in) + `Executor.run(*, stream: bool = False)` per-call execution-mode kwarg + secondary `StreamingOracle` Protocol with `isinstance` check at `_eval_leaf`. Base `Oracle` Protocol untouched per D-STEP-6-T1 mock-conformance precedent. Compile-time mutual exclusion `streaming=True ⊥ schema_ref != None` (D-005). 5 commits (`fe22e9c` → `e4480ae`), +22 tests (3094 → 3116), 0 PIVOT, 0 autonomy-leash hits.

The historical alternatives — (a) `LeafStream` AST sibling (4-file coupled change + new `_eval` return-type union), and (c) permanent HOST_CALL bypass for streaming (would require an I6/§4 CAND-C amendment carrying a permanent footnote) — are no longer in play. The kernel-design substrate from candidate (b) supports the M3c default flip atomically with the M3d-narrowed retirement of `_make_cb_respond_stream` once D5-AGENT (below) is resolved.

### Follow-on Plan D5-AGENT — ✅ SHIPPED in `plan_2026-04-29_0f87b9c4/`

**Resolution: option (b'-narrow) — test-side fixes, no stdlib code touch, no env-var gate.**

The local re-experiment in `plan_0f87b9c4` (running the M3c flip on a controlled checkpoint, gating the failure list, then reverting) showed that the 4 named failures from `plan_90d0824f` iter-2 were NOT stdlib-construction-path semantics — they were **test-infrastructure artifacts**:

- 3 failures in `test_fsm_llm_agents/test_bug_fixes.py` (`TestReflexionEvaluationFnActuallyCalled`, `TestReasoningReactAgentInterception`) used a brittle `SequenceMockLLM` whose cursor advanced per-`generate_response` call. Under M3c's D1 short-circuit (empty-`response_instructions` → 0 oracle calls), `act` and `reflect` states no longer fire `generate_response`, drifting the cursor out of step with state progression. The mock — not the FSM — broke.
- 1 failure in `test_pipeline_oracle_parity_pivot1.py::test_site_7_5_invoke_preserves_user_message_on_wire` pinned the legacy `user_message=` kwarg shape; under D2 the executor's `oracle.invoke(prompt)` embeds the user message in `system_prompt` instead. Same end-to-end visibility, different wire field.

**Reframed strategy from `plan_0f87b9c4`/decisions.md D-001**: rejected (a) migrate stdlib agents (no actual semantic divergence existed) and (c) env-var gating (deferred-only; doesn't unblock M3d-narrowed). Picked option (b'-narrow): added `FunctionalMockLLM` helper (per-field call-count callables, decoupled from gen-response cadence) and rewrote the 3 brittle tests; generalised the wire-shape assertion. Then the M3c flip + M3d-narrowed drop landed cleanly atop these test fixes.

The original options enumerated in `plan_2026-04-28_90d0824f`/decisions.md D-008-PIVOT (option (a) migrate ~723 tests; option (b) per-fixture; option (c) env-var) are preserved for institutional record but no longer in play — the diagnosis pinpointed test infrastructure, not stdlib semantics, so a much narrower fix sufficed.

### Follow-on Plan A.M3c+A.M3d-narrowed — ✅ SHIPPED in `plan_2026-04-29_0f87b9c4/`

Atomic with D5-AGENT in the same plan. 4 commits: `30e2cda` (FunctionalMockLLM + 3 brittle test rewrites), `1046443` (wire-shape generalisation), `d85b614` (the flip + 23 shape-assertion test inversions across `test_compile_fsm_m3b.py`, `test_pipeline.py`, `test_fsm_compile.py`; added `_disable_leaf` and `_legacy_defn` helpers preserving explicit-False regression coverage), `e40399d` (M3d-narrowed: dropped `_make_cb_respond_stream` factory + `_stream_response_generation_pass` support method + the `D-STEP-5-NARROWED` env-rebind, ~110 net source LOC removed from `dialog/turn.py`). Pre-Mortem A NOT triggered. 0 PIVOT, 0 autonomy-leash hits, 0 stdlib code touched.

**A.M3d-wide** ("retire `_make_cb_respond` + remove `CB_RESPOND` constant") REMAINS deferred. The D3 conservative terminal-non-cohort fallback keeps `_make_cb_respond` and `CB_RESPOND` live for terminal states without `output_schema_ref`. A.D5's compile-time pathway is shipped, but agents/users have to actually set `output_schema_ref` for the lift to occur. Full M3d-wide retirement is gated on universal `output_schema_ref` adoption — a documentation-driven migration over several minor releases, not a single plan.

The bench scorecard schema (G5) can now gain `program_kind: "fsm"` cells with `theorem2_holds: true` for any non-terminal FSM program — separate plan.

### Follow-on Plan B.M5/M6 — "0.6.0 release sequence (live deprecation warnings + sibling shim hooks)"

The M5 calendar test (`test_deprecation_calendar.py`) is in place; the **live `warnings.warn(...)` emit** for the four I5-epoch surfaces is the 0.6.0 release commit:
- `Program.run(**env)` (`program.py:459`) — top-of-method `warnings.warn(...)`.
- `Program.converse(msg, conv_id)` (`program.py:494`) — same.
- `Program.register_handler(h)` (`program.py:678`) — same.
- `from fsm_llm import API` — needs a `__getattr__` hook in top-level `__init__.py` (the hardest one; `API` is currently a direct attribute in `__all__`).
- `import fsm_llm_{reasoning,workflows,agents}` — one-line `warnings.warn(...)` after the last `sys.modules` alias registration in each shim's `__init__.py` (M6c).

The 0.6.0 release also lands the **R13-epoch removals** (lam shim, 8 dialog shims, llm shim, dialog/pipeline shim — per the message text those shims have been emitting since 0.3.0). Test `test_deprecation_calendar.py` will activate its `(0,6,0) ≤ version < (0,7,0)` skipif branches automatically on the version bump. Single PR, low-risk, parallelisable with A.D4.

### Follow-on Plan B.M6b — "Session-store migration (after the schema delta is real)"

M6b is **deferred for cause**: investigation in `plans/plan_2026-04-28_f1003066/findings/m6-payload.md` found that `transition_context` is a string with zero occurrences in the source tree at HEAD; the spec invented a precondition that does not hold. M3c has now shipped (`plan_0f87b9c4`); the migration can be re-scoped against the post-flip `SessionState` if a concrete delta exists. Likely outcome: a small or empty migration since D2's `add_system_message` continues to write into `instance.context.conversation` byte-equivalently.

### Future "decouple-kernel-from-dialog" plan

The 5-entry `_PRE_EXISTING_DIALOG_IMPORT_ALLOWLIST` in `test_layering.py` is grandfathered debt. Each of the five entries currently imports `FSMError` (or a request/response model) from `fsm_llm.dialog.definitions`. Relocating `FSMError` to a neutral module (e.g. `fsm_llm.runtime.errors` or `fsm_llm.errors`) would let the allow-list drop to 1 entry (the `lam/__init__.py` shim). Out of scope for the merge; called out here as the natural follow-up that would make I4 strict-by-default. Estimated impact: 28+ file ripple (per D-003 in `plan_2026-04-28_6597e394/decisions.md`).

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
