# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] — 2026-04-29

### Removed (R13 epoch — shim layer deleted)
- **Top-level shim modules deleted**: `fsm_llm.api`, `fsm_llm.fsm`, `fsm_llm.pipeline`, `fsm_llm.prompts`, `fsm_llm.definitions`, `fsm_llm.llm`, `fsm_llm.session`, `fsm_llm.classification`, `fsm_llm.transition_evaluator`. These have warned with `DeprecationWarning` since 0.5.0 (R13/D-PIVOT-1-R13). Users must import from canonical homes: `fsm_llm.dialog.<x>` (or `fsm_llm.runtime._litellm` for `llm`).
- **`fsm_llm.lam` shim package deleted** (was redirecting to `fsm_llm.runtime`). Users must import from `fsm_llm.runtime` or, for `compile_fsm` / `compile_fsm_cached`, from the top level `fsm_llm`.
- Sibling shim packages (`fsm_llm_reasoning`, `fsm_llm_workflows`, `fsm_llm_agents`) **remain** as sys.modules redirects but now emit `DeprecationWarning(since="0.6.0", removal="0.7.0")` once at first import per the M6c calendar entry. Users should migrate to `fsm_llm.stdlib.<reasoning|workflows|agents>`.

### Changed (I5 epoch — silent → warning)
- **Active deprecation warnings on legacy `Program` aliases**: `Program.run(**env)`, `Program.converse(message, conversation_id)`, `Program.register_handler(handler)` now emit `DeprecationWarning(since="0.6.0", removal="0.7.0")` once per process via the deduped `fsm_llm._api.deprecation.warn_deprecated` helper. Callers should migrate to `Program.invoke(...)` and the `handlers=` constructor kwarg respectively.
- **Top-level `from fsm_llm import API` warns**: served via module-level `__getattr__` that dispatches to `warn_deprecated("fsm_llm.API", since="0.6.0", removal="0.7.0", replacement="Program.from_fsm")`. Users should migrate to the `Program` facade.

### Changed (Public surface consistency)
- **Long-context factories renamed to the `*_term` convention** (matches every other stdlib slice): `niah → niah_term`, `aggregate → aggregate_term`, `pairwise → pairwise_term`, `multi_hop → multi_hop_term`, `multi_hop_dynamic → multi_hop_dynamic_term`, `niah_padded → niah_padded_term`. Bare names remain reachable via module-level `__getattr__` and emit `DeprecationWarning(since="0.6.0", removal="0.7.0")` on first access; will be removed in 0.7.0.
- **Top-level `__all__` L3 export completeness**: every stdlib factory is now reachable directly from `fsm_llm` — all 11 reasoning factories (`analytical_term`, `deductive_term`, …, `solve_term`), all 5 workflow factories (`linear_term`, `branch_term`, `switch_term`, `parallel_term`, `retry_term`), and all 6 long-context factories (`*_term` forms). Previously users had to reach into subpackages.
- **L2 `Handler` placement clarified**: `FSMHandler`, `BaseHandler`, and `create_handler` join the L2 COMPOSE block in `__all__` (alongside the existing `Handler` alias and `compose`/`HandlerTiming`/`HandlerBuilder`). The L2 block is now self-contained.
- **`HandlerBuilder.do()` return type narrowed** from `BaseHandler` to `LambdaHandler` (the actual returned class). Cosmetic; no behavior change.
- **`LiteLLMOracle._invoke_structured` docstring expanded** with explicit caveat: subclasses of `LiteLLMInterface` overriding `generate_response` are NOT invoked on structured-Leaf calls. Escape hatch: implement the `Oracle` Protocol directly and pass it via `oracle=`. (No code change; doc-only fix for D-008.)

### Documentation
- **Full rewrite of user-facing docs against the post-cleanup surface**: `README.md`, `docs/quickstart.md`, `docs/api_reference.md`, `docs/architecture.md`, `docs/handlers.md`, and `CLAUDE.md` (root). All references to the legacy `API` import path, the R13 shim modules, and the `lam/` shim are gone. Technical references (`docs/lambda.md`, `docs/lambda_fsm_merge.md`, `docs/threat_model.md`, `docs/deepagents.md`, strands phase docs) are unchanged.
- **Archived**: `docs/lambda_integration.md` → `docs/archive/lambda_integration.md` (was already marked SUPERSEDED in 0.5.0).
- **Removed root artifacts**: `book_recommender.log` (stale example run log) and `REVIEW_REPORT.md` (pre-refactor audit superseded by current state).

### Added (deepagents absorb — 2026-04-29)
- **Deprecation machinery** (`fsm_llm._api.deprecation`): private package shipping `warn_deprecated(name, *, since, removal, replacement, stacklevel)` and `reset_deprecation_dedupe(*targets)`. Stdlib `warnings`-only (no langchain coupling); process-local dedupe registry keyed on `(name, since, removal)`; thread-safe via RLock; defensive `stacklevel >= 1` floor. **Machinery only** — I5-epoch surfaces (`Program.run`, `.converse`, `.register_handler`, `from fsm_llm import API`, `import fsm_llm_{reasoning,workflows,agents}`) remain SILENT in 0.5.x per `docs/lambda_fsm_merge.md` §3. Live warnings flip in 0.6.0. 11 tests in `tests/test_fsm_llm/test_deprecation.py`.
- **`pyproject.toml` filterwarnings policy**: explicit allow-list for our own DeprecationWarnings; ignore-by-source for pydantic/litellm/httpx/asyncio noise — with rationale comments.
- **Threat model** (`docs/threat_model.md`, 543 lines): trust boundaries, threats, dismissed proposals. 11 threats T-01..T-11 (prompt injection, secrets, format-string, path traversal, concurrent-write, unbounded cost, provider injection, handler privilege, monitor disclosure, dependency confusion, stale dedupe). 6 trust boundaries TB-1..TB-7. 6 dismissed proposals ID-01..ID-06 (sandboxing, path-traversal protection, prompt-injection filtering, monitor auth, TLS pinning, encryption-at-rest). File:line citations into `src/`. Cross-link from `docs/api_reference.md`.
- **Harness profiles** (`fsm_llm.profiles`, L2 COMPOSE): construction-time data bundles applied apply-once at `Program.from_*`. New public surface — `HarnessProfile`, `ProviderProfile`, `register_harness_profile`, `register_provider_profile`, `get_harness_profile`, `get_provider_profile`. Both profile types are frozen Pydantic. Resolution: `provider:model` → bare `provider` fallback for HarnessProfile; `provider/model` slash-prefix extraction for ProviderProfile. `apply_to_term(term, profile)` rewrites only `Leaf.template` strings via `Term.model_copy` — no AST schema changes, no Leaf cardinality changes, **Theorem-2 strict equality preserved** (`tests/test_fsm_llm/test_profiles.py::TestTheorem2WithProfile`). `assemble_system_prompt` joins fragments in canonical `USER → (BASE | CUSTOM) → SUFFIX` order. `LiteLLMInterface.__init__` consults `get_provider_profile(model)` and merges `extra_kwargs` under caller-supplied kwargs (caller wins). `Program.from_fsm/from_term/from_factory` gain `profile=` kwarg accepting `HarnessProfile | str | None`. 29 tests in `tests/test_fsm_llm/test_profiles.py`. Layering test `test_layer_l2_exact` updated.

### Documentation (2026-04-29)
- **Full doc refresh against `docs/lambda_fsm_merge.md` (the merge contract).** README.md and CLAUDE.md rewritten — both lead with `Program` as the unified entry point, both stay under 400 lines. `docs/quickstart.md`, `docs/api_reference.md`, and `docs/architecture.md` rewritten around the four-layer architecture (L1-L4), `Program.invoke` as the single verb, and `Result` as the uniform return type. `docs/handlers.md` gains a `Program(handlers=...)` registration example. `docs/lambda.md` gets a companion-document banner. `docs/lambda_integration.md` v2.0 marked SUPERSEDED (preserved for historical reference). Stale claims purged: "Program mode-bifurcated", "six call sites bypass Oracle", outdated test counts.

### Added (0.5.0 A.M3c + A.M3d-narrowed — Theorem-2 universal-by-default for FSM)
- **A.M3c (FSM↔λ merge §3 I6, plan_2026-04-29_0f87b9c4)**: `State._emit_response_leaf_for_non_cohort` default flipped `False → True` in `dialog/definitions.py:635`. Every non-terminal non-cohort FSM state now emits a real `Leaf` for response generation by default — Theorem-2 strict equality `Executor.oracle_calls == plan(...).predicted_calls` holds **universally** for the dominant FSM cohort, no opt-in flag required. Closes invariant **I6** in `docs/lambda_fsm_merge.md`. Field semantics inverted from "opt-IN to leaf emission" to "opt-OUT to legacy `App(CB_RESPOND, instance)`" (used only by the regression-coverage test sets via the new `_disable_leaf` / `_legacy_defn` helpers).
- **D5-AGENT divergence dissolved on diagnosis** (D-001 in plan decisions.md): the 4 named failures from `plan_2026-04-28_90d0824f` iter-2 were **test-infrastructure artifacts**, not stdlib semantics. 3 failures in `test_fsm_llm_agents/test_bug_fixes.py` came from a brittle `SequenceMockLLM` whose cursor advanced per-`generate_response` call — under the new path, D1's 0-call short-circuit for empty-`response_instructions` states (`act`/`reflect`) drifts the cursor out of step with state progression. The 4th was a wire-shape assertion that pinned the legacy `user_message=` kwarg shape. Resolution: added `FunctionalMockLLM` helper (per-field call-count callables, decoupled from gen-response cadence) + rewrote the 3 brittle tests; generalised the wire-shape assertion to "user message reaches the LLM via either wire field or system_prompt". **Zero stdlib code touched**, no env-var gate, no fixture-by-fixture migration.
- **A.M3d-narrowed (FSM↔λ merge §6b)**: dropped `_make_cb_respond_stream` factory, `_stream_response_generation_pass` support method, and the `D-STEP-5-NARROWED` env-rebind block in `dialog/turn.py:599`. Net: -110 source LOC. Streaming now flows through the executor's stream-mode branch on the D2 `Leaf(streaming=True)` chain via `StreamingOracle.invoke_stream`. Residual `App(CB_RESPOND, instance)` paths (D3 terminal fallback when `output_schema_ref` is unset; explicit-False regression coverage) yield a single `str` that `process_stream_compiled` wraps in `iter([result])` — chunked streaming unavailable for those minority paths but functional parity preserved.
- **23 shape-assertion tests updated** across `tests/test_fsm_llm/test_compile_fsm_m3b.py` (9), `tests/test_fsm_llm_lam/test_fsm_compile.py` (12), `tests/test_fsm_llm/test_pipeline.py` (1) + the streaming marker assertion in `test_pipeline_oracle_parity_7_6.py`. New helpers `_disable_leaf(state)` / `_legacy_defn(dict)` preserve regression coverage for the explicit-False path until M3d-wide retires the field together with the D3 conservative fallback. **Pre-Mortem A NOT triggered** in this plan (the fix landed atomic with the diagnosis); 0 PIVOT, 0 autonomy-leash hits across 4 commits.
- **Doc surface updated** (`docs/lambda_fsm_merge.md`): Status table, §3 I6, §4 CAND-C, §6b follow-on plans, §6 G5 status, ship-trajectory ledger. **M3 row in §5 → ✅ SHIPPED** across 5 plans. The "M3c HALTED on D5-AGENT" caveat removed from `src/fsm_llm/dialog/CLAUDE.md` A.D5 note.
- **A.M3d-wide remains deferred** (separate plan, doc-driven migration): full retirement of `_make_cb_respond` + `CB_RESPOND` constant is gated on universal `output_schema_ref` adoption in stdlib agents.

### Added (0.5.0 A.D5 — Terminal opt-in `State.output_schema_ref` → Leaf)
- **A.D5 (FSM↔λ merge §4 CAND-C, plan_2026-04-28_90d0824f)**: new `State.output_schema_ref: Any = None` field opts a terminal non-cohort state into structured `Leaf` emission. When set to a Pydantic `BaseModel` subclass (or a pre-formatted dotted-path string `"module.Class"`), `compile_fsm._compile_state` routes the D3 terminal-non-cohort branch from the conservative legacy `App(CB_RESPOND, instance)` to a real `Leaf(template, schema_ref=<dotted>, streaming=False)` inside the D2 outer-Let. The executor enforces the schema via `oracle._invoke_structured` and Theorem-2 strict equality `Executor.oracle_calls == predicted_calls` holds end-to-end for migrated terminals. Default `None` preserves D3 byte-equivalent for every State at HEAD (stdlib agents migration deferred per D-002 inherited from `plan_ca542489`).
- **D-005 mutual exclusion preserved**: A.D5 emits `streaming=False` unconditionally (mid-stream schema enforcement is unreliable per `runtime/oracle.py:120-128`); the streaming entry `process_stream_compiled` degrades terminal opt-in responses to a single-chunk iterator via the entry-point `iter([result])` normalisation shipped in plan_ca542489 step 5.
- **D-007-SURPRISE caught pre-commit**: kernel `Leaf.schema_ref` is typed `str | None` (dotted path resolved by `runtime/oracle.py:_resolve_schema`), not `type[BaseModel]` — the AST is JSON-roundtrippable. The compiler accepts EITHER a `BaseModel` subclass (preferred — auto-converted to `f"{cls.__module__}.{cls.__qualname__}"`) OR a pre-formatted dotted-path string (must contain `.`). Validation gate raises `ASTConstructionError` loudly on malformed input (string-without-dot, dict, plain class, BaseModel instance).
- **6 new tests** under `TestD5TerminalSchemaRefEmission` in `tests/test_fsm_llm/test_compile_fsm_m3b.py`: default-None preserves D3 byte-equivalent, terminal opt-in with schema_ref emits `Leaf(schema_ref=...)`, D-005 global mutual-exclusion sweep across the whole compiled term, default-OFF gate ignores `output_schema_ref` (outer `else` branch fires before the D3 sub-gate), non-terminal opt-in ignores the field (terminal-only by design), and the validation gate raises loudly. Total suite **3158 → 3164 passing** (+6 net).

### Added (0.5.0 A.D4 — Streaming kernel design candidate (b))
- **A.D4 (FSM↔λ merge §6b, plan_2026-04-28_ca542489)** — streaming kernel design candidate (b) chosen over (a) `LeafStream` AST sibling and (c) HOST_CALL bypass. New `Leaf.streaming: bool = False` capability flag (per-Leaf opt-in: "this Leaf is allowed to stream") + `Executor.run(*, stream: bool = False)` per-call execution-mode kwarg (caller's intent) + secondary `StreamingOracle` Protocol with `isinstance` check at `_eval_leaf`. Base `Oracle` Protocol untouched (per D-STEP-6-T1 mock-conformance precedent — broadening the runtime-checkable Protocol breaks 21 mock-conformance tests). Compile-time mutual exclusion `streaming=True ⊥ schema_ref != None` (D-005) — enforced at the compiler boundary because mid-stream schema enforcement is unreliable.
- **D2 streaming-aware** (plan_ca542489 step 4): `compile_fsm` emits `Leaf(streaming=True)` for the opt-in non-cohort response. Cohort/D1/D3 paths unchanged. **D1 streaming-aware** via per-turn `_TurnState.stream` field (closure-captured in `_make_cb_respond_synthetic`). **D2 `CB_APPEND_HISTORY` iterator-aware**: `_make_cb_append_history` inner closure dispatches on `isinstance(value, str)` — string path unchanged; `Iterator[str]` path returns a tee-on-exhaustion generator that yields chunks while accumulating, then on exhaustion or `GeneratorExit` calls `add_system_message("".join(chunks))`. Theorem-2 strict equality preserved (host App still uncounted).
- **`process_stream_compiled` rewired** (NARROWED per D-008): `Executor` now receives `oracle=self._oracle` so the new D2 streaming Leaf can reach `oracle.invoke_stream`; `Executor.run(stream=True)` is set; string-or-iterator return normalisation (`iter([result])` if string) handles cohort terminal Leaves. **`env[CB_RESPOND] = self._make_cb_respond_stream(...)` rebind RETAINED** through A.D4 — drops with M3d-narrowed once M3c default flip eliminates the default-OFF chunking dependency. `# DECISION D-STEP-5-NARROWED` source anchor at the rebind site.
- **22 new tests** across `tests/test_fsm_llm_lam/test_executor.py` (10) and `tests/test_fsm_llm/test_compile_fsm_m3b.py` (12 across `TestD2AppendHistoryIteratorAware`, `TestD1*`, `TestD4StreamingLeafFlagEmission`). Suite **3094 → 3116 passing** (+22 net).

### Halted (A.M3c default flip — D5-AGENT divergence)
- **A.M3c HALTED in `plan_2026-04-28_90d0824f` iter-2**: flipping `_emit_response_leaf_for_non_cohort: False → True` in `dialog/definitions.py:635` triggered Pre-Mortem Scenario A. **3 behavioural regressions in `tests/test_fsm_llm_agents/test_bug_fixes.py`** (e.g. `solve_problem` called 5× instead of 1× on a React/Reflexion path) confirmed the named hypothesis: stdlib agents' construction path depends on default-OFF AST shape for per-turn LLM call counts. Reverted clean per D-005 protocol (no fix-forward, no autonomy-leash retry); logged as D-008-PIVOT.
- **D5-AGENT** is the new gating divergence for the default flip. Three resolution options enumerated in `plans/plan_2026-04-28_90d0824f/decisions.md` D-008-PIVOT: (a) migrate `stdlib/agents/base.py` + reasoning engines (~723 agent tests at risk; D-002 deferred work); (b) per-fixture amendment of the failing tests via `_enable_leaf` helper; (c) env-var gating analogous to the R6 `FSM_LLM_COHORT_EMISSION` pattern. Each option requires its own EXPLORE+PLAN cycle.
- **A.M3d-narrowed** (drop `_make_cb_respond_stream` + `D-STEP-5-NARROWED` env-rebind) is M3c-blocked per `plan_ca542489` D-008: until M3c flips, default-OFF programs remain the streaming-traffic majority and the rebind covers their chunking. Will ship atomically with the D5-AGENT-resolution + M3c-flip plan.

### Added (0.5.0 M5 + M6a/c/d — Deprecation calendar + payload docs)
- **Deprecation calendar test (`tests/test_fsm_llm/test_deprecation_calendar.py`)** — codifies the **two-epoch** alias contract from `docs/lambda_fsm_merge.md` §3 I5:
  - **R13 epoch** (`fsm_llm.lam`, the eight `fsm_llm.{api,fsm,pipeline,prompts,classification,transition_evaluator,definitions,session}` shims, `fsm_llm.llm`, `fsm_llm.dialog.pipeline` — 11 module shims): already emit `DeprecationWarning` at HEAD (R13/D-PIVOT-1-R13); REMOVED at 0.6.0.
  - **I5 epoch** (`Program.run`, `Program.converse`, `Program.register_handler`, `from fsm_llm import API`, `import fsm_llm_{reasoning,workflows,agents}` — 7 surfaces total): silent at HEAD; WARN at 0.6.0; REMOVED at 0.7.0.
  - 20 active assertions at 0.3.0; 19 future-version branches gated by `pytest.skipif` and active at 0.6.0 / 0.7.0 thresholds.
- **M6a Monitor span schema doc (`src/fsm_llm_monitor/CLAUDE.md`)** — documents the v1 (current FSM-level) vs v2 (planned post-M3c per-Leaf) span schema contract for OTEL consumers. Live v2 routing is M3c-blocked; the doc table signals the schema bump in advance so dashboards can gate parser selection.
- **M6d Third-party `LLMInterface` subclass contract (`src/fsm_llm/runtime/CLAUDE.md`)** — documents which subclass overrides survive the `Program.from_fsm(llm=...)` → `LiteLLMOracle(llm)` wrap (preserved: ABC surface; NOT preserved: `_invoke_structured` bypass for structured Leaf calls). Escape hatch: implement the `Oracle` Protocol directly and pass via `oracle=` kwarg.
- **Spec amendment (`docs/lambda_fsm_merge.md`)** — §3 I5 reconciled to two-epoch table; §5 M6b marked DEFERRED with reason (`transition_context` never existed in `SessionState`; spec precondition was factually wrong); §5 M5 + §6b updated to reflect plan_f1003066 in-flight scope. **No contract change** for users — the spec now matches HEAD.

### Changed (0.5.0 M1 — Result-uniform Program.invoke)
- **M1 (FSM↔λ merge §M1, plan plan_2026-04-28_6597e394)**: `Program.invoke()` now returns `Result` in **every** mode — eliminating the pre-M1 `Result | str` union leak in the public surface. FSM-mode `Program.invoke(message=..., conversation_id=...)` returns `Result(value=<reply_str>, conversation_id=<id>, plan=None, leaf_calls=0, oracle_calls=0, explain=None)`; the `plan` / `leaf_calls` / `oracle_calls` placeholders will be populated when M3 lifts response generation into a `Leaf`. Term/factory mode now surfaces executor accounting on `Result.leaf_calls` (= `cost_accumulator.total_calls`) and `Result.oracle_calls` (= `executor.oracle_calls`). The legacy `.run(**env)` and `.converse(msg, conv_id)` aliases keep their pre-M1 return types (raw value / `str`) by unwrapping `result.value` — out-of-tree callers using the legacy aliases are unaffected. **Breaking change** for any external caller that relied on `prog.invoke(message=...)` returning a bare `str`; migrate to `result.value`. `Result` dataclass extended with 4 new fields, all defaulted (purely additive on construction). Three FSM-mode test sites in `test_program_invoke.py` inverted+renamed per D-STEP-2-T1 precedent (audit trail preserved).

### Added (0.5.0 R8-R13 — facade + cohort-default + oracle-collapse + DeprecationWarnings)
- **R8 — `Program.invoke()` unified verb**: single user-visible entry over both surface modes. `Program.from_fsm(d).invoke(message="hi", conversation_id=None)` auto-starts a conversation and returns the response string. `Program.from_term(t).invoke(inputs={"x": 1})` returns a `Result(value=..., explain=None)`. Mode mismatch raises `ProgramModeError`. Legacy `.run`/`.converse` survive as 3-line deprecation aliases delegating to `.invoke`. `Program.explain(inputs=, n=, K=)` runtime-plan path emits one `Plan` per `Fix` subtree. `register_handler` now works in term mode.
- **R11 — Substrate first-class at top-level**: `from fsm_llm import Term, leaf, fix, react_term, niah, compile_fsm, Executor, Oracle, LiteLLMOracle, Plan, PlanInputs, plan, Result, ProgramModeError, …` — substrate names appear before FSM-front-end names in `__all__`. `Handler = FSMHandler` alias. `compose` re-exported.
- **R12 — `--inputs FILE` on `fsm-llm run`**: factory targets accept JSON-loaded inputs unpacked as `**env` to `Program.from_factory(...).invoke(...)`.
- **R9a/9b/9c — Cohort emission default-ON, env gate dropped**: `_cohort_emission_enabled` flipped default-ON in step 3, then the gate function + `FSM_LLM_COHORT_EMISSION` env reads were removed entirely in step 5. Legacy `App(CB_RESPOND, instance)` shape retired for cohort-eligible terminal states; cohort `Leaf` is the only path. R9b cohort-widening predicate added with STOP-IF guard for byte-equivalence with `CB_RESPOND` output (D-PLAN-05 worst-case: predicate stays effectively terminal-only when widening introduces transition-info dependency).
- **R10 — All 6 dialog LLM call sites collapsed to `oracle.invoke[_stream|_messages|_field]`**: every `self.llm_interface.<call>` in `dialog/turn.py` (renamed from `pipeline.py`) now routes through `LiteLLMOracle`. Three new oracle surface methods (`invoke_messages`, `invoke_field`, `invoke(user_message=)`) added in the D-PIVOT-1 surface-extension to unblock the 3 deferred sites (data-extraction `_make_llm_call`, field-extraction `extract_field`, canonical Pass-2 main response). `LiteLLMInterface` un-exported from `fsm_llm.__all__` (D-009 — kept importable as a back-compat module attribute). `Oracle.invoke_stream` Protocol added via narrower `StreamingOracle` capability extension (D-STEP-6) to preserve mock-oracle conformance.
- **R13 — `dialog/pipeline.py` → `dialog/turn.py` rename + 10-shim DeprecationWarnings**: the 2-pass body module renamed to reflect its post-R10 single-turn-of-dialog role. The 10 module shims (`fsm_llm.lam`, `fsm_llm.{api,fsm,pipeline,prompts,classification,transition_evaluator,definitions,session,llm}`) now emit `DeprecationWarning` on import (silent in 0.4.x → warn in 0.5.0 → removal in 0.6.0 per D-004 / D-PLAN-10). All internal callers migrated to canonical paths so `import fsm_llm` is warning-free. Identity contracts preserved: `import fsm_llm.lam as A; import fsm_llm.runtime as B; A is B` and the 9 dialog-shim equivalents (verified by `tests/test_fsm_llm/test_module_shims.py`).
- **3032 tests passing** at the close of plan_2026-04-27_32652286 (vs 2937 at R8 start; +95 from new test surface). 14 pre-existing real-LLM Ollama-down failures unchanged.

### Added (0.4.0 R5/R7 — Handlers as AST + unified CLI)
- **R5 — Handlers compose into the compiled λ-term**: `Program.register_handler(...)` (and legacy `API.register_handler(...)`) now splice the handler into the FSM's compiled `Term` via the new `fsm_llm.handlers.compose(term, handlers)` AST rewriter. PRE_PROCESSING and POST_PROCESSING timings are real AST splices that invoke the handler via the new `Combinator(op=HOST_CALL, ...)` op (kernel addition). The other 6 timings (PRE/POST_TRANSITION, CONTEXT_UPDATE, START/END_CONVERSATION, ERROR) keep their host-side dispatch sites for cardinality / conditional-firing reasons (D-STEP-04-RESOLUTION) — all 8 timings still route through one `make_handler_runner` callable so the execution semantics (priority, error_mode, timeout, `should_execute`) are unchanged. `r5-green` tagged at commit `9208b8a`.
- **R6 narrow-cohort shipped** in plan_2026-04-27_1b5c3b2f (`r6-green`). Cohort terminal states (response-only, no extractions) compile to a real `Leaf` instead of `App(CB_RESPOND, instance)`, lighting up per-Leaf cost telemetry + Theorem-2 strict equality `Executor.oracle_calls == plan(...).predicted_calls` for the cohort. Opt-in via `FSM_LLM_COHORT_EMISSION=1` (default OFF preserves byte-equivalent legacy behavior; production rollout to default-ON deferred). New compile-time producer `ResponseGenerationPromptBuilder.to_compile_time_template((state, fsm_def))`; new predicate `dialog/compile_fsm._is_cohort_state`; planner extension `PlanInputs.fmap_leaf_count` (additive, default 0). 49 new tests across byte-parity, cohort emission, planner, and Theorem-2 invariant. R5 splicer-refinement (PRE/POST_TRANSITION + CONTEXT_UPDATE term-side) and R6 universality remain architecturally infeasible — see `docs/lambda_integration.md` for the documented coverage boundary. Stdlib factories (Category B/C) continue to satisfy Theorem-2 unchanged.
- **R7 — Unified `fsm-llm` CLI**: single binary now exposes 6 subcommands: `run / explain / validate / visualize / meta / monitor`. Legacy console scripts (`fsm-llm-validate`, `fsm-llm-visualize`, `fsm-llm-meta`, `fsm-llm-monitor`) survive as aliases (D-PLAN-04 — silent in 0.4.x; deprecation in 0.5.0; removal in 0.6.0). `fsm-llm run <target>` auto-detects FSM JSON paths vs `pkg.mod:factory_name` factory strings. `fsm-llm explain <target>` prints AST skeleton + leaf schemas + (optionally, when `--n N --K K` are supplied) one `Plan` per `Fix` subtree. New kw-only overload: `Program.explain(n=…, K=…, plan_kwargs=…)` — populates `plans` per Fix subtree. R1 no-arg contract preserved (`plans=[]`). 45 new tests under `tests/test_fsm_llm/test_cli_unified.py`. `r7-green` tag pending final type-check.

### Security
- **litellm supply chain compromise**: litellm versions 1.82.7 and 1.82.8 were compromised
  with credential-stealing malware via `.pth` file injection. These versions are now
  explicitly excluded from the dependency specification (`!=1.82.7,!=1.82.8`).
  - **Impact**: Any Python invocation in an environment with the compromised versions would
    exfiltrate environment variables, SSH keys, AWS credentials, Kubernetes configs, and git
    credentials to an attacker-controlled server. No import of litellm was required.
  - **Action for users**: If you installed litellm 1.82.7 or 1.82.8 at any time, treat all
    credentials in that environment as compromised and rotate them immediately.
  - **Current status**: PyPI has quarantined the entire litellm package. Existing installs of
    safe versions (<=1.82.6) continue to work.
- Added `.pth` file audit in CI pipeline and local `make audit` / `scripts/audit_pth.py`
- Added `constraints.txt` for dependency version locking in dev/CI builds

### Changed
- **Skip Pass 2 for intermediate agent states** — States with `response_instructions=""` now skip
  the response generation LLM call entirely. The pipeline sends a minimal sentinel to the LLM
  interface (for cycle tracking) and the real LLM returns immediately without an API call. This
  halves the number of LLM calls for agent iterations, cutting wall time ~50% and eliminating
  all F-LOOP timeout failures. Applied to: think/act (ReAct, Reflexion), evaluate (EvalOpt),
  check (MakerChecker).
- **Stall detection threshold** reduced from 3 to 2 consecutive no-tool iterations before
  forced termination, saving ~20s per stall event.

### Fixed
- **MakerChecker quality_score extraction** — When the LLM embeds quality_score inside the
  checker_feedback dict instead of as a separate context field, `_track_revisions` now recovers
  it from the dict. Previously quality_score defaulted to 0.0, forcing max revisions.
- Evaluation health score improved from 95.7% to **100%** (70/70 PASS) on `ollama_chat/qwen3.5:4b`.

### Added
- **BaseAgent ABC** for all 12 agent implementations — shared conversation loop, budget enforcement,
  answer extraction, trace building, context filtering, and `__call__` syntax (`agent("task")`)
- **Enhanced `@tool` decorator** — supports bare `@tool` (no parentheses) with auto-schema inference
  from type hints (`str→string`, `int→integer`, `float→number`, `bool→boolean`, `list→array`, `dict→object`).
  Supports `typing.Annotated[T, "description"]` for per-parameter descriptions. Backward compatible
  with explicit `parameter_schema` overrides.
- **Structured output** — `AgentConfig(output_schema=PydanticModel)` validates agent answers against
  Pydantic models. Parsed result stored in `AgentResult.structured_output`. Graceful fallback on
  validation failure.
- **`create_agent()` factory** — create agents in one line: `create_agent(tools=[search], pattern="react")`
- **`ToolRegistry.register_agent()`** — register agents as tools for supervisor/orchestrator patterns
- **`AgentResult.__str__`** — returns structured_output if available, else raw answer
- `ollama.py` module — centralized Ollama helpers for structured output compatibility
  - `is_ollama_model()` — model detection
  - `apply_ollama_params()` — disables thinking via `reasoning_effort="none"`, forces `temperature=0` for structured calls
  - `build_ollama_response_format()` — builds `json_schema` response format with extraction/transition schemas
  - `EXTRACTION_JSON_SCHEMA`, `TRANSITION_JSON_SCHEMA` — JSON Schema constants for structured output
- `fsm_llm_agents` extension package for ReAct and Human-in-the-Loop agentic patterns
  - `ReactAgent` — ReAct loop agent with auto-generated FSM from tool registry (think → act → observe → conclude)
  - `ToolRegistry` — tool management with schema descriptions, prompt generation, and execution
  - `HumanInTheLoop` — configurable approval gates, confidence-based escalation, and human override
  - `@tool` decorator for simple tool registration
  - Pydantic models: `ToolDefinition`, `ToolCall`, `ToolResult`, `AgentStep`, `AgentTrace`, `AgentConfig`, `AgentResult`, `ApprovalRequest`
  - `AgentError` exception hierarchy (7 error types)
  - 109 unit tests across 8 test files
- `has_agents()` / `get_agents()` extension checks in `fsm_llm`
- `MessagePipeline` class extracted from FSMManager — encapsulates all 2-pass message processing
- `context.py` module extracted from FSMManager — stateless context cleaning utilities
- `ConversationStep` added to workflows — embeds full FSM conversations within workflow steps
- Handler execution timeout support (`DEFAULT_HANDLER_TIMEOUT = 30s`)
- Workflow step async timeout support (`DEFAULT_STEP_TIMEOUT = 120s`)
- Workflow-level timeout, conversation timeout, and event listener expiration
- `critical` flag on `BaseHandler` — errors always raise regardless of error_mode
- `FORBIDDEN_CONTEXT_PATTERNS` enforcement for password/secret/token key filtering
- 5 new examples combining sub-packages (reasoning, workflows, classification)
- 20 new complex examples (70 total) focused on agentic patterns and meta builders:
  - **Agents (14)**: debate_with_tools (evidence-based debate), reflexion_code_gen (self-improving code
    generation with test runner), orchestrator_specialist (multi-specialist ReactAgents), pipeline_review
    (PromptChain + MakerChecker QA), adapt_with_memory (ADaPT + WorkingMemory), rewoo_multi_step (complex
    multi-dependency planning), eval_opt_structured (EvaluatorOptimizer + Pydantic validation),
    plan_execute_recovery (replanning on tool failure), consistency_with_tools (SelfConsistency for
    multi-step reasoning), maker_checker_code (code review pattern), hierarchical_orchestrator (nested
    multi-level delegation), agent_memory_chain (multi-task continuity via WorkingMemory),
    react_structured_pipeline (ReAct → structured output → PromptChain), multi_debate_panel (parallel
    debates with synthesis)
  - **Meta (4)**: build_workflow (interactive workflow builder), build_agent (interactive agent builder),
    meta_review_loop (FSMBuilder + MakerChecker quality review), meta_from_spec (programmatic
    FSM/workflow/agent from text specs)
  - **Workflows (2)**: conditional_branching (condition-based routing), workflow_agent_loop (quality-gated
    agent execution with retry)
- Automated evaluation baseline: 95.7% health score (70 examples, ollama_chat/qwen3.5:4b)
- Tests for MessagePipeline, handler timeout, step timeout, context, logging, runner, LiteLLMInterface
- Audit verification tests across all packages

### Changed
- Ollama structured output uses `json_schema` response format instead of `json_object` for grammar-constrained output
- Ollama thinking mode disabled via `reasoning_effort="none"` (litellm >=1.82 maps this to Ollama's `think: false`)
- Ollama structured calls (data extraction, transition decision) force `temperature=0` for deterministic output
- Classification `Classifier._call_llm()` now applies Ollama params via shared `fsm_llm.ollama` helpers
- Minimum litellm version bumped from 1.68.1 to 1.82.0 (required for proper Ollama `think` parameter forwarding)
- FSMManager delegates message processing to MessagePipeline
- `push_fsm`/`pop_fsm` decomposed into focused sub-methods
- `evaluate_logic()` refactored with dispatch pattern
- Runner refactored to use API; workflows drops phantom FSMManager dependency
- Exception handling standardized across codebase (chaining with `from e`)
- Regex patterns pre-compiled for performance
- Test fixtures deduplicated across test suites
- mypy enforcement enabled in CI with pydantic plugin
- All 118 mypy errors fixed

### Removed
- `fsm_llm_classification` deprecation shim package (use `from fsm_llm import Classifier` directly)
- `LLMInterface.decide_transition()` deprecated method
- `LLMInterface.extract_data()` deprecated method
- `FSMManager` `transition_prompt_builder` parameter
- `WorkflowEngine` `fsm_manager` and `llm_interface` parameters
- `DataExtractionRequest` class
- `State._coerce_and_warn()` boolean coercion for `transition_classification`
- `State` `instructions` field deprecation warning
- `has_classification()` and `get_classification()` helper functions
- Empty `fsm_llm_workflows.handlers` compatibility shim
- 7 forwarding methods from FSMManager (moved to MessagePipeline)
- Dead workflow handler code (AutoTransitionHandler, EventHandler, TimerHandler)
- Dead code and empty extras across multiple packages

### Fixed
- Ollama/Qwen3 thinking mode corrupting structured JSON output (ollama/ollama#10538)
- Integration test `test_pre_processing_handler_fires` using wrong HandlerBuilder API (`.on_timing()` → `.at()`, `.execute().build()` → `.do()`)
- Race condition in conversation lock retrieval
- Conversation lock leak with cleanup methods
- Event listener race condition in workflows
- Confidence collapse with additive boost in classification
- MockLLM2Interface crash on empty transitions
- Classifier thinking hacks and multi-intent prompt mismatch
- Classification confidence handling and dead code
- Workflow step error paths and type safety
- Workflow engine safety issues
- Security gaps in handlers, context, and prompts
- JSON regex fallback validation (requires meaningful keys)
- Multi-key JsonLogic expression error
- Reasoning engine bugs and magic number extraction
- Algorithm and logic issues across codebase

### Security
- Safety limits and validation guards added
- Security gaps fixed in handlers, context, and prompts
- Context key security filtering (internal prefixes, forbidden patterns)

## [0.3.0] - 2026-03-19

### Added
- `fsm_llm_classification` extension package for LLM-backed structured classification
  - `Classifier` for single-intent and multi-intent classification
  - `HierarchicalClassifier` for two-stage domain-then-intent classification (>15 classes)
  - `IntentRouter` for mapping classified intents to handler functions with low-confidence fallback
  - Pydantic models: `ClassificationSchema`, `IntentDefinition`, `ClassificationResult`, `MultiClassificationResult`, `HierarchicalSchema`
  - Prompt and JSON schema builders with reasoning-first field ordering (mitigates constrained-decoding distortion)
  - Structured output support via `response_format` when the LLM provider supports it
- `has_classification()` / `get_classification()` extension checks in `fsm_llm`
- 39 unit tests for classification package
- Classification extension documentation (README, examples, architecture docs)
- `timeout` parameter on `LiteLLMInterface` (default 120s) to prevent indefinite hangs on network issues
- `pytest-mock` added to dev extras in pyproject.toml
- `[tool.ruff.lint]` configuration in pyproject.toml to suppress false E402 from `__future__` annotations
- 21 regression tests for codebase review fixes (`test_regression_review.py`)
- 15 new `ContextKeys` constants for reasoning sub-FSM result keys (deductive, inductive, abductive, analogical, critical, hybrid)

### Fixed
- Version number aligned to 0.3.0 across `pyproject.toml` and `__version__.py` (was still 0.2.1)
- Context pruning log now reports actual new size instead of repeating the original size
- Hard-coded context keys in `merge_reasoning_results` replaced with `ContextKeys` constants (prevents silent `None` on key mismatch)
- Duplicate `import re` removed from `llm.py` `_make_llm_call()` (leftover from Qwen3.5 workaround)
- Extraction parse failure now returns `confidence=0.0` instead of `0.5` (callers can distinguish failure from low-confidence extraction)
- `requirements.txt` aligned with `pyproject.toml` core deps (removed dev deps, fixed `python-dotenv` version pin)

### Removed
- Unused async handler support from `handlers.py` (asyncio import, `AsyncExecutionLambda` type, `is_async` detection, ThreadPoolExecutor fallback) — no async handlers existed in the codebase
- `MergeStrategy` alias from `reasoning/constants.py` — engine now imports `ContextMergeStrategy` directly
- Dynamic `__all__.extend()` / `__all__.append()` calls from `__init__.py` — consolidated into single `__all__` definition
- Dead `[testenv:docs]` sphinx environment from `tox.ini`

## [0.2.1] - 2026-03-19

### Added
- `[tool.pytest.ini_options]` in pyproject.toml
- `[tool.mypy]` configuration in pyproject.toml
- Python 3.12 support in CI and tox
- CHANGELOG.md (this file)
- examples/README.md with example index and learning path

### Changed
- Python minimum version updated to 3.10 (was 3.8)
- Package-data now includes `fsm_llm_reasoning`
- Pre-commit hooks replaced: pytest-on-commit removed, ruff + standard hooks added
- Makefile expanded from 3 to 8 targets (added help, lint, format, type-check, install-dev)
- CI workflow installs from pyproject.toml instead of requirements.txt
- tox.ini aligned with CI (consistent flake8 config, added mypy env)

### Fixed
- CLI entry point now correctly resolves `fsm-llm` command
- Exception chaining (`from e`) added to all catch-and-reraise blocks for proper traceback preservation
- `__main__.py` docstring placement (was after imports, not recognized by Python)
- Workflows package version now imported from main package instead of hardcoded
- LLM interface log levels demoted from INFO to DEBUG (less noisy)
- Input validation added to `LiteLLMInterface` (model, temperature, max_tokens)

## [0.2.0] - 2026-03-18

### Changed
- Project renamed from `llm-fsm` to `fsm-llm` across all packages, tests, docs, and examples

## [0.1.0] - 2026-03-07

### Added
- Initial release with 2-pass architecture
- FSM stacking with push/pop operations
- Handler system with builder pattern
- JsonLogic expression evaluator
- LiteLLM multi-provider support
- CLI tools: fsm-llm, fsm-llm-visualize, fsm-llm-validate
- 7 examples (basic, intermediate, advanced)
- Comprehensive documentation

[Unreleased]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NikolasMarkou/fsm_llm/releases/tag/v0.1.0
