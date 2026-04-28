# fsm_llm.dialog — FSM Dialog Surface (R4)

The FSM dialog front-end. Houses everything that turns a Category-A FSM JSON definition into a compiled λ-term and runs it turn-by-turn: the user-facing `API` class, `FSMManager` orchestrator, `MessagePipeline` 2-pass body, classifiers, transition evaluator, prompt builders, Pydantic definitions, sessions, and the FSM → λ compiler itself.

> **Created in plan v3 R4 step 21 (D-PLAN-08).** Before R4, all of these modules lived at the top level under `fsm_llm/`. Old import paths (`from fsm_llm.api import API`, etc.) keep working through 0.4.x via sys.modules shims at `src/fsm_llm/{api,fsm,pipeline,prompts,classification,transition_evaluator,definitions,session}.py`. Deprecation in 0.5.0; removal in 0.6.0 per D-004 / D-PLAN-10.

## File Map

```
dialog/
├── api.py                  # API — primary user-facing entry (from_file, from_definition, converse, push/pop_fsm). Was fsm_llm/api.py.
├── fsm.py                  # FSMManager — per-conversation orchestration, RLocks, FSM-definition cache. Compiled-term cache lives in compile_fsm.py via lru_cache. Was fsm_llm/fsm.py.
├── pipeline.py             # MessagePipeline — compiled-path 2-pass body (extract → evaluate → respond). Internal post-M2 S11. Was fsm_llm/pipeline.py.
├── classification.py       # Classifier, HierarchicalClassifier, IntentRouter, HandlerFn type alias. Was fsm_llm/classification.py.
├── transition_evaluator.py # TransitionEvaluator + TransitionEvaluatorConfig — DETERMINISTIC | AMBIGUOUS | BLOCKED. Was fsm_llm/transition_evaluator.py.
├── prompts.py              # DataExtractionPromptBuilder, ResponseGenerationPromptBuilder, FieldExtractionPromptBuilder + classification_template + to_template_and_schema producers (R3 step 14). Was fsm_llm/prompts.py.
├── definitions.py          # Pydantic v2 models: State, Transition, FSMDefinition, FSMContext, FSMInstance, Conversation, classification/extraction request+response models, exceptions. Was fsm_llm/definitions.py.
├── session.py              # SessionStore ABC + FileSessionStore — atomic JSON writes. Was fsm_llm/session.py.
├── compile_fsm.py          # M2 — compile_fsm(FSMDefinition) → Term + R2 compile_fsm_cached(fsm, fsm_id) — lru_cache(64). Was fsm_llm/lam/fsm_compile.py.
└── __init__.py             # Re-exports the surface. Importing `from fsm_llm.dialog import API, FSMManager, ...` works.
```

## Public surface (from `fsm_llm.dialog`)

```python
from fsm_llm.dialog import (
    API,
    FSMManager,
    MessagePipeline,
    Classifier, HierarchicalClassifier, IntentRouter,
    TransitionEvaluator, TransitionEvaluatorConfig,
    DataExtractionPromptBuilder, ResponseGenerationPromptBuilder, FieldExtractionPromptBuilder,
    FSMDefinition, FSMContext, FSMInstance, State, Transition, Conversation,
    ClassificationSchema, ClassificationResult, ClassificationExtractionConfig, FieldExtractionConfig,
    SessionStore, FileSessionStore, SessionState,
    compile_fsm, compile_fsm_cached,
)
```

The same names also resolve via `fsm_llm` directly: `from fsm_llm import API, FSMManager, LiteLLMInterface, Program` (top-level convenience exports listed in `src/fsm_llm/__init__.py`).

## Cross-package boundaries

- **Reads from `fsm_llm.runtime`**: `runtime.ast.Term` (in `pipeline.py`, `fsm.py`, `compile_fsm.py`), `runtime.executor.Executor` (in `pipeline.py`, `fsm.py`), `runtime.dsl.{abs_, app, case_, let_, var}` and `runtime.errors.ASTConstructionError` (in `compile_fsm.py`).
- **Reads from top-level `fsm_llm.<x>`**: `fsm_llm.constants`, `fsm_llm.handlers`, `fsm_llm.logging`, `fsm_llm.context`, `fsm_llm.expressions`, `fsm_llm.utilities`, `fsm_llm.ollama`, `fsm_llm.llm` (the `LiteLLMInterface` adapter).
- **No imports from `fsm_llm.dialog.<x>`** in `runtime/` *except* `runtime/__init__.py` re-exporting `compile_fsm`/`compile_fsm_cached` and the `fsm_compile` module alias for back-compat with `from fsm_llm.lam import compile_fsm` and `from fsm_llm.lam.fsm_compile import compile_fsm_cached`. `runtime/oracle.py` and `runtime/_litellm.py` do import `fsm_llm.dialog.definitions` for the request/response Pydantic models; these are leaves on the dialog side (no upstream dialog deps).

## Internal import order (dependency chain)

When editing dialog/ modules, keep the chain acyclic:

```
definitions  ← session
            ← prompts        ← classification
            ← prompts/transition_evaluator/compile_fsm
                                          ← pipeline ← fsm ← api
```

The 9 files were moved into `dialog/` in this order (plan v3 step 21) and their relative imports rewritten consistently:
- Sibling-in-dialog imports stay relative: `from .definitions import …`, `from .prompts import …`, etc.
- Cross-package to top-level `fsm_llm.<x>` use parent-relative: `from ..constants import …`, `from ..handlers import …`, etc.
- Cross-package to `fsm_llm.runtime` use parent-relative: `from ..runtime.ast import Term`, `from ..runtime.executor import Executor`.

## Key surfaces

- **`API`** — `from_file(path, **kwargs)`, `from_definition(fsm_def, **kwargs)`, `start_conversation`, `converse`, `push_fsm`, `pop_fsm`, `register_handler`, `create_handler`. See `docs/api_reference.md` and `src/fsm_llm/CLAUDE.md`.
- **`FSMManager`** — orchestration with per-conversation RLocks. As of R2, the compiled-term cache lives in `compile_fsm.compile_fsm_cached` (lru_cache(64) keyed on `(fsm_id, fsm.model_dump_json())`); `FSMManager.get_compiled_term` is a 3-line shim.
- **`MessagePipeline`** — 2-pass body: data extraction → field extractions → classification extractions → transition evaluation → state transition → response generation. Internal post-M2 S11.
- **`Classifier` / `HierarchicalClassifier` / `IntentRouter`** — LLM-backed intent classification.
- **`TransitionEvaluator`** — rule-based transition resolution (`DETERMINISTIC` | `AMBIGUOUS` | `BLOCKED`).
- **Prompt builders** — also expose `to_template_and_schema(...) -> (template_str, env, schema)` per R3 step 14 (narrowed). The pipeline.py callbacks at HEAD still use `build_*_prompt`; the callback collapse to `oracle.invoke` (lifting `_cb_*` to `Leaf` nodes for Theorem-2 universality) is deferred to a fresh R6 plan — see `plans/plan_2026-04-27_43d56276/decisions.md` D-STEP-08-RESOLUTION. The producer signature, multi-Leaf-per-state via `fmap`, and `Fix` retry encoding are kernel-level concerns that need a dedicated PLAN cycle.
- **`compile_fsm` / `compile_fsm_cached`** — FSM JSON → λ-Term. Cache key `(fsm_id, fsm.model_dump_json())` (D-PLAN-07, D-002). Inspect via `_compile_fsm_by_id.cache_info()`.

### R5 — Handlers compose into the compiled term (post-r5-green)

`Program.register_handler` and `API.register_handler` now splice the handler into the compiled FSM term via `fsm_llm.handlers.compose(term, handlers)`. PRE_PROCESSING and POST_PROCESSING timings are real AST splices via `Combinator(op=HOST_CALL, ...)` (see `runtime/CLAUDE.md`). The other 6 timings (PRE/POST_TRANSITION, CONTEXT_UPDATE, START/END_CONVERSATION, ERROR) keep their host-side dispatch sites in `pipeline.py` and `fsm.py` for cardinality / conditional-firing reasons (D-STEP-04-RESOLUTION) — all 8 still route through one `make_handler_runner` callable so execution semantics (priority, error_mode, timeout, `should_execute`) are unchanged. The composed-term cache lives on `FSMManager` keyed on `(fsm_id, _handlers_version)` with FIFO eviction at 128 entries (D-STEP-03).

Refinement of PRE_TRANSITION + POST_TRANSITION + CONTEXT_UPDATE term-side splicing was investigated in plan_2026-04-27_1b5c3b2f and **falsified** — these are now documented as **architecturally host-side** (POST_TRANSITION rollback, CONTEXT_UPDATE dual-fire, PRE_TRANSITION cardinality requires HOST_CALL Case-gating). See `docs/lambda_integration.md` §R5 for rationale.

### R6 — Cohort Leaf emission (post-r6-green, opt-in)

`compile_fsm._is_cohort_state(state, fsm_def)` predicate identifies **terminal cohort states** (response-only states with no transitions, extractions, or required_context_keys). For these states, `_compile_state` emits a real `Leaf("{response_prompt_rendered}", input_vars=("response_prompt_rendered",), schema_ref=None)` instead of `App(Var(CB_RESPOND), Var(VAR_INSTANCE))`. The pipeline pre-renders the full response prompt at env-build time via `ResponseGenerationPromptBuilder.build_response_prompt` and binds it under `COHORT_RESPONSE_PROMPT_VAR`. This lights up per-Leaf cost telemetry and Theorem-2 strict equality `Executor.oracle_calls == plan(...).predicted_calls` for the cohort.

**Opt-in gate**: `FSM_LLM_COHORT_EMISSION=1` (default OFF preserves byte-equivalent legacy behavior). Default-ON rollout deferred to a future plan once production validation completes. Non-cohort states (transitions, extractions, classifications, required_keys, extraction_instructions) keep the legacy host-callback path unchanged. Theorem-2 universality across ALL FSM states is architecturally impossible — `skip-if-in-context` filtering + LLM-output-dependent retries make oracle-call count turn-state-dependent. See `docs/lambda_integration.md` §R6 for the full coverage boundary.

Producer surface: `ResponseGenerationPromptBuilder.to_compile_time_template((state, fsm_def)) -> (template, input_vars, schema_ref)` is the additive compile-time emitter; `classification_compile_time_template` is forward-compat plumbing for richer placeholder schemas.

### A.D5 — Terminal opt-in via `State.output_schema_ref` (plan_2026-04-28_90d0824f, opt-in)

`State.output_schema_ref` (default `None`) lets a Category-A FSM author opt a **terminal non-cohort** state into structured Leaf emission. When set to a Pydantic `BaseModel` subclass (or a pre-formatted dotted-path string `"module.Class"`), `compile_fsm._compile_state` routes the terminal-non-cohort branch from the conservative D3 fallback `App(CB_RESPOND, instance)` to the same D2 outer-Let shape used by non-terminal opt-in states, with the response Leaf carrying `schema_ref=<resolved-dotted-path>` and `streaming=False`. The executor enforces the schema via `oracle._invoke_structured` and Theorem-2 strict equality `Executor.oracle_calls == predicted_calls` holds end-to-end for migrated terminals.

**D-005 mutual exclusion** (inherited from plan_2026-04-28_ca542489): `streaming=True ⊥ schema_ref != None` on any Leaf — enforced at the compiler boundary because mid-stream schema enforcement is unreliable (`runtime/oracle.py:120-128`). The streaming entry `process_stream_compiled` degrades terminal opt-in responses to a single-chunk iterator via the entry-point `iter([result])` normalisation (A.D4 step 5).

**Validation**: invalid `output_schema_ref` (anything other than a `BaseModel` subclass or a dotted-path string containing `.`) raises `ASTConstructionError` at compile time — fail loud, not deep in `_invoke_structured` at runtime. See `# DECISION D-007-SURPRISE` in `compile_fsm.py` for the class-to-dotted-path conversion rationale (kernel `Leaf.schema_ref` is `str | None` for JSON-roundtrippability of the AST).

**Default-False gate retained**: A.D5 is gated on `_emit_response_leaf_for_non_cohort=True` (the same M3a opt-in flag used by D1/D2/D3/D4). At HEAD this remains False by default. **A.M3c default-flip is HALTED** pending D5-AGENT resolution: the stdlib agents construction path (`stdlib/agents/base.py`-led FSMs) depends on the legacy default-OFF shape. Flipping the default at HEAD breaks 3 tests in `tests/test_fsm_llm_agents/test_bug_fixes.py` with a `solve_problem`-called-5x-instead-of-1x behavioural regression. See `plans/plan_2026-04-28_90d0824f/decisions.md` D-008-PIVOT for the full surface and the named follow-on plan options.

## Testing

```bash
pytest tests/test_fsm_llm/                  # core — 837 tests (post-R1+R2+R3+R4+R5+R7)
pytest tests/test_fsm_llm/test_module_shims.py  # R4 SC29-SC32 identity tests
```

## Related Subpackages

- **`fsm_llm.runtime`** — λ-calculus kernel (was `fsm_llm.lam`). The substrate that dialog/ runs on.
- **`fsm_llm.handlers`** — top-level; `HandlerSystem`, `HandlerBuilder`, `HandlerTiming`. Composed into the compiled λ-term per `docs/lambda.md` §6.3.
- **`fsm_llm.stdlib`** — named λ-term factories. Independent of dialog/ — uses runtime/ directly.
