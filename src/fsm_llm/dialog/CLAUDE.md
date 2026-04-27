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
- **Prompt builders** — also expose `to_template_and_schema(...) -> (template_str, env, schema)` per R3 step 14 (narrowed). The pipeline.py callbacks at HEAD still use `build_*_prompt`; the callback collapse to `oracle.invoke` is deferred to R6.
- **`compile_fsm` / `compile_fsm_cached`** — FSM JSON → λ-Term. Cache key `(fsm_id, fsm.model_dump_json())` (D-PLAN-07, D-002). Inspect via `_compile_fsm_by_id.cache_info()`.

## Testing

```bash
pytest tests/test_fsm_llm/                  # core (688 → 730+ post-R1+R2+R3+R4)
pytest tests/test_fsm_llm/test_module_shims.py  # R4 SC29-SC32 identity tests
```

## Related Subpackages

- **`fsm_llm.runtime`** — λ-calculus kernel (was `fsm_llm.lam`). The substrate that dialog/ runs on.
- **`fsm_llm.handlers`** — top-level; `HandlerSystem`, `HandlerBuilder`, `HandlerTiming`. Composed into the compiled λ-term per `docs/lambda.md` §6.3.
- **`fsm_llm.stdlib`** — named λ-term factories. Independent of dialog/ — uses runtime/ directly.
