# FSM-LLM — Claude Code Instructions

## Project Overview

FSM-LLM (`0.9.0`) is a Python framework for building stateful LLM programs on a typed **λ-calculus runtime**. Two surface syntaxes share one executor:

- **FSM JSON (Category A)** — dialog programs with persistent per-turn state. Compiled to λ-terms at load time.
- **λ-DSL (Category B/C)** — pipelines, agents, reasoning chains, long-context recursion. Authored as λ-terms directly.

Both surfaces flow through one verb: **`Program.invoke(...)` → `Result`**. One Oracle. One bench schema. See `docs/lambda_fsm_merge.md` (the merge contract — canonical) and `docs/lambda.md` (the architectural thesis).

- **License**: GPL-3.0-or-later
- **Python**: 3.10, 3.11, 3.12
- **Core deps**: `loguru`, `litellm` (>=1.82,<2.0, excluding 1.82.7/1.82.8), `pydantic` (>=2.0), `python-dotenv`
- **Virtual environment**: Always use `.venv` — run with `.venv/bin/python` or activate first.

## What changed in 0.9.0

The namespace-cleanup release. The flat ~140-name top-level surface from 0.8.0 was reorganised into themed sub-namespaces; the misleading "Legacy" tier label was retired; the 6 module-level profile-registry functions collapsed into a single `ProfileRegistry` class; the `HandlerBuilder` got a coherent `.when_*` prefix for all conditions; a few DSL builders shed their inconsistent trailing underscore. **Every change is a hard break — no aliases, no deprecation warnings.** Things future Claude sessions need up front:

- **Sub-namespaces** — the substrate now lives under descriptive paths (the top-level convenience exports remain for the high-traffic surface, but everything else moved):
  - `fsm_llm.ast` — `Term`, `Var`, `Abs`, `App`, `Let`, `Case`, `Combinator`, `CombinatorOp`, `ReduceOp`, `Leaf`, `Fix`, `LeafCall`, `is_term`.
  - `fsm_llm.dsl` — `leaf`, `var`, `abs_`, `app`, `let`, `case_`, `fix`, plus combinator builders.
  - `fsm_llm.combinators` — closed combinator builders + `BUILTIN_OPS` + `ReduceOp`.
  - `fsm_llm.factories` — every `*_term` stdlib factory (agents/reasoning/workflows/long-context).
  - `fsm_llm.errors` — full exception hierarchy. Only `FSMError` and `LambdaError` roots remain at the top level.
  - `fsm_llm.debug` — `enable_debug_logging`, `disable_warnings`, `BUFFER_METADATA`.
- **DSL renames** — naming convention now: trailing underscore = collision with Python keyword or always-loaded builtin.
  - `let_` → `let`, `reduce_` → `reduce` (in `fsm_llm.runtime.dsl` and `fsm_llm.dsl`).
  - `abs_` and `case_` keep the underscore (`abs` is a builtin; `case` is a soft keyword in match statements).
  - `fmap`, `ffilter` keep the `f`-prefix (`map` and `filter` are true builtins users would shadow).
- **Profile registry collapse** — six module-level functions replaced by a single `ProfileRegistry` class with a module-level `profile_registry` singleton. The kind is now a kwarg: `profile_registry.register(name, profile, kind="harness")`.
- **HandlerBuilder method renames** — `.on_state` → `.when_state`, `.not_on_state` → `.not_when_state`, `.on_state_entry` → `.when_state_entry`, `.on_state_exit` → `.when_state_exit`. All conditions now share `.when_*`. `.at(timing)` and `.do(action)` keep their names.
- **`HandlerSystem.execute_handlers` privatized** to `._execute_handlers` (internal plumbing post-R5).
- **`Program.from_term` and `from_factory`** accept the same explicit LLM kwargs as `from_fsm` — `model`, `api_key`, `temperature`, `max_tokens`, `**llm_kwargs`. Passing both `oracle=` and any LLM kwarg raises `ValueError`.
- **`HandlerSystem` dropped from top-level** (still importable from `fsm_llm.handlers`).
- **`compile_fsm_cached` dropped from top-level** (still importable from `fsm_llm.dialog.compile_fsm`). Top-level `compile_fsm` stays.
- **`get_version_info()` deleted** — use `fsm_llm.__version__` directly.
- **`fsm_llm.types` deleted** — split into `fsm_llm._models` (private: Pydantic models + enums) and `fsm_llm.errors` (public: `FSMError` hierarchy).
- **"Legacy" tier label retired** — dialog-front-end names stay top-level but in thematic comment groups (FSM dialog core, Classification & extraction, Pydantic models, Prompt builders, Transition evaluation, Context & memory, Session persistence, Validation/visualization/loaders).
- **MetaBuilderStates legacy alias class deleted** in `stdlib/agents/constants.py`.
- **`docs/archive/strands_features*.md`** (3 files) deleted (preserved in git history; superseded).

`tests/test_fsm_llm/test_deprecation_calendar.py::TestN9EpochHardRemovedAt090` is the executable source of truth — every removed surface is asserted to raise the expected exception at `__version__ >= 0.9.0`. See `docs/migration_0.8_to_0.9.md` for the upgrade walkthrough.

## What changed in 0.8.0

The post-0.7.0 cleanup release. Eight removals, three structural moves, one parameter rename. Every change is a hard removal at the source-tree level — the 0.7.0 deferred items shipped, no new deprecation cycle was introduced. Things future Claude sessions need to know up front so we don't try to use surfaces that no longer exist:

- **Removed top-level aliases.** Accessing any of these now raises `AttributeError` / `ImportError`:
  - `from fsm_llm import Handler` — was an alias for `FSMHandler`. Use `from fsm_llm import FSMHandler` (or `BaseHandler`) directly.
  - `from fsm_llm import LLMInterface` — top-level re-export. Canonical path is `from fsm_llm.runtime._litellm import LLMInterface`.
  - `from fsm_llm import BUILTIN_OPS` — top-level re-export. Canonical path is `from fsm_llm.runtime import BUILTIN_OPS` (still architecturally closed; bind new ops through env at the call site).
  - `has_workflows()` / `has_reasoning()` / `has_agents()` / `get_workflows()` / `get_reasoning()` / `get_agents()` extension-check helpers — the stdlib subpackages are not optional since 0.7.0.
- **Removed re-exports in `dialog/definitions.py`.** The 0.7.0 back-compat re-exports for the `FSMError` hierarchy + 5 runtime-touching Pydantic request/response models + 2 enums (`LLMRequestType`, `TransitionEvaluationResult`) are gone. Canonical home: `from fsm_llm.types import FSMError, ...`. Existing `from fsm_llm.dialog.definitions import FSMError` callsites must update.
- **Removed State field.** The Pydantic-private `_emit_response_leaf_for_non_cohort` gate on `State` is gone. Non-cohort states **always** emit a Leaf — Theorem-2 universal-by-default for non-terminal FSM programs is now a structural property, not a default-value flip.
- **Program facade tightening.** `Program.__init__` no longer accepts `_api` / `_profile` (the public ctor is term-mode only). `Program.from_fsm` now takes explicit kwargs (`model`, `api_key`, `temperature`, `max_tokens`, `max_history_size`, `max_message_length`, `handler_error_mode`, `transition_config`) plus `**llm_kwargs` for LiteLLM passthrough. The pre-0.8.0 `**api_kwargs` catch-all is gone.
- **New private modules.** `src/fsm_llm/dialog/extraction.py` (extracted from `turn.py`; holds `ExtractionEngine` — the Pass-1 extraction cluster) and `src/fsm_llm/runtime/_handlers_ast.py` (holds `compose` + AST splicers, moved from `handlers.py`). Public API unchanged: `from fsm_llm import compose` and `from fsm_llm.handlers import compose` continue to work as re-exports.
- **Reasoning factory parameter rename.** `prompt_a` / `prompt_b` / `prompt_c` (and the matching `*_input_vars` / `*_schema_ref` kwargs) are renamed to descriptive names matching each factory's bind_names. E.g. `analytical_term(decomposition_prompt, analysis_prompt, integration_prompt)` instead of `analytical_term(prompt_a, prompt_b, prompt_c)`. Every reasoning factory in `stdlib/reasoning/lam_factories.py` was migrated.
- **Prompts dedup.** `dialog/prompts.py` `*PromptBuilder` classes share a `BasePromptBuilder` helper for repeated section logic; no public-API change.

`tests/test_fsm_llm/test_deprecation_calendar.py` is the executable source of truth — its assertions auto-flip per `__version__` thresholds. The Z8 epoch row (the 0.8.0 cleanup) now fires at 0.8.0+; the test currently passes because every removed surface raises the right exception. See `docs/migration_0.7_to_0.8.md` for the upgrade walkthrough.

## Quick Commands

```bash
make test           # pytest -v (verify count with: pytest --collect-only -q | tail -3)
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make type-check     # mypy across all packages
make build          # python -m build (wheel + sdist)
make coverage       # pytest with coverage report
make install-dev    # pip install -c constraints.txt -e ".[dev,workflows,reasoning,agents,monitor,oolong]" + pre-commit install
make audit          # audit site-packages for suspicious .pth files

# CLI console scripts (5 entries in pyproject.toml)
fsm-llm --mode {run,validate,visualize} --fsm <path.json>   # legacy single-binary CLI
fsm-llm-validate  --fsm <path.json>                         # subcommand-specific aliases
fsm-llm-visualize --fsm <path.json>
fsm-llm-monitor                                             # web dashboard
fsm-llm-meta                                                # interactive artifact builder
```

## Architecture — One Runtime, Two Surfaces, One Verb

```
        FSM JSON  (Category A)             λ-DSL  (Category B / C)
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
        │ Executor   (β-reduction, depth-bounded)  │
        │ Planner    (k*, τ*, d, predicted_calls)  │
        │ Oracle     (one per Program — uniform)   │
        │ Session    (per-conversation persistence)│
        │ Cost       (CostAccumulator, LeafCall)   │
        └──────────────────────────────────────────┘
                                │
                                ▼
                         Program.invoke(...)  →  Result
```

### Public API — Layered

| Layer | Names | Purpose |
|---|---|---|
| **L4 INVOKE** | `Program`, `Result`, `ExplainOutput`, `ProgramModeError` | One verb. Mode fixed at construction. |
| **L3 AUTHOR** | `compile_fsm`; stdlib factories — agents (4) `react_term`, `rewoo_term`, `reflexion_term`, `memory_term`; reasoning (11) `analytical_term`, …, `solve_term`; workflows (5) `linear_term`, …, `retry_term`; long-context (6) `niah_term`, …, `niah_padded_term`; raw DSL (`leaf`, `fix`, `let_`, `case_`, `var`, `abs_`, `app`, `split`, `fmap`, `reduce_`, …) | Term producers. |
| **L2 COMPOSE** | `compose`, handler surface (`FSMHandler`, `BaseHandler`, `HandlerTiming`, `HandlerBuilder`, `HandlerSystem`, `create_handler`), profiles surface (`HarnessProfile`, `ProviderProfile`, `register_*`, `get_*`) | Pure AST→AST transforms + construction-time data. |
| **L1 REDUCE** | `Term`, `Executor`, `Plan`, `Oracle`, `LiteLLMOracle`, `CostAccumulator`, `LeafCall` | Typed substrate. |

Plus a **Legacy** block (`FSMManager`, `MessagePipeline`, `FSMDefinition`, `State`, `Transition`, classifiers, etc.) — preserved at the top level for back-compat. The 0.7.0 I5 epoch surfaces (`API`, sibling shim packages, deprecated `Program` aliases, long-context bare names, `LiteLLMInterface` top-level re-export) and the 0.8.0 Z8 epoch surfaces (`Handler`, top-level `LLMInterface`, top-level `BUILTIN_OPS`, `has_*`/`get_*` extension helpers, `dialog/definitions.py` type re-exports) are all hard removals — see "What changed in 0.8.0" above. Layering invariants asserted by `tests/test_fsm_llm/test_layering.py` (kernel↔dialog allow-list at 0 since 0.7.0).

### `Program` — the unified facade

```python
from fsm_llm import Program

# Three constructors — mode fixed at construction
Program.from_fsm(
    defn,
    *,
    oracle=None, session=None, handlers=None, profile=None,
    # Explicit LLM/API config (since 0.8.0):
    model=None, api_key=None, temperature=None, max_tokens=None,
    max_history_size=5, max_message_length=1000,
    handler_error_mode="continue", transition_config=None,
    **llm_kwargs,            # forwarded to LiteLLMInterface (top_p, …)
)
Program.from_term(term, *, oracle=None, session=None, handlers=None, profile=None)
Program.from_factory(factory, factory_args=(), factory_kwargs=None, *, oracle=None, session=None, handlers=None, profile=None)

# One verb, returns Result
result = program.invoke(message="hi", conversation_id=None)   # FSM mode
result = program.invoke(inputs={"x": 1, ...})                 # term/factory mode

# Result fields
result.value              # str (FSM) or term reduction (term/factory)
result.conversation_id    # FSM only
result.plan               # populated when explain=True
result.leaf_calls
result.oracle_calls       # equals plan(...).predicted_calls under Theorem 2
```

`Program.invoke(message=...)` on a term-mode Program raises `ProgramModeError` with a redirect; vice versa. Mode is invariant. The pre-0.7.0 legacy aliases (`.run(**env)`, `.converse(msg, conv_id)`, `.register_handler(h)`) were removed at 0.7.0 — accessing them now raises `AttributeError`. Pass handlers via `handlers=[...]` at construction. The pre-0.8.0 `**api_kwargs` catch-all was replaced with explicit kwargs + `**llm_kwargs`.

### Theorem-2 (cost model)

For every `Fix` node, `oracle_calls == plan(...).predicted_calls` strictly when input is τ·k^d-aligned. **Universal-by-default for non-terminal FSM programs** (post-A.M3c; structural since 0.8.0 with the `_emit_response_leaf_for_non_cohort` field removed). Single residual caveat: terminal non-cohort states without `output_schema_ref` still use the host-callable fallback.

## Key Modules in `src/fsm_llm/`

- **`types.py`** — neutral Pydantic-models layer (since 0.7.0). Hosts `FSMError` hierarchy + 5 runtime-touching request/response models + 2 enums (`LLMRequestType`, `TransitionEvaluationResult`). The canonical home for everything that the runtime kernel + handlers + stdlib subpackages share. (Pre-0.8.0 `dialog/definitions.py` re-exports were removed in 0.8.0.)
- **`runtime/`** — typed λ-kernel. AST, DSL builders, Executor, Planner, Oracle, `_litellm.py` (private adapter), cost tracker, `_handlers_ast.py` (private — holds `compose` + AST splicers; moved from `handlers.py` in 0.8.0). **The substrate.** Closed against `dialog/` per D-001 (kernel↔dialog allow-list at 0 since 0.7.0). See `src/fsm_llm/runtime/CLAUDE.md`.
- **`dialog/`** — FSM dialog front-end. `API`, `FSMManager`, `MessagePipeline` (in `dialog/turn.py`), `ExtractionEngine` (in `dialog/extraction.py`; extracted from turn.py in 0.8.0), classifiers, `TransitionEvaluator`, prompt builders, `compile_fsm`/`compile_fsm_cached`, definitions, sessions. See `src/fsm_llm/dialog/CLAUDE.md`.
- **`stdlib/`** — named λ-term factories organised by domain (`agents/`, `reasoning/`, `workflows/`, `long_context/`). Each subpackage's `lam_factories.py` exposes pure factory functions returning `Term`. See `src/fsm_llm/stdlib/CLAUDE.md`. Reasoning factories use descriptive parameter names since 0.8.0 (e.g. `analytical_term(decomposition_prompt, analysis_prompt, integration_prompt)`).
- **`program.py`** — **`Program` facade** with `from_fsm`/`from_term`/`from_factory` constructors and the `.invoke(...)` verb. Public `__init__` is term-mode only (FSM-mode construction is `from_fsm` exclusively since 0.8.0). Internal `_api: API | None` and `_term: Term | None` are mode-invariant.
- **`handlers.py`** — `HandlerSystem`, `HandlerBuilder`, `HandlerTiming` (8 timing points). Two timings (`PRE/POST_PROCESSING`) are AST-side via `compose`; the other six stay host-side per merge spec §8. The `compose` function and AST splicers moved to `runtime/_handlers_ast.py` in 0.8.0; this module re-exports them so `from fsm_llm.handlers import compose` and `from fsm_llm import compose` continue to work unchanged. Fresh-name generation uses `itertools.count()` (thread-safe; 0.7.0).
- **`profiles.py`** — `HarnessProfile`, `ProviderProfile`, registries, `apply_to_term`. Apply-once at construction; **Theorem-2 strict equality preserved**.
- **`_api/deprecation.py`** — `warn_deprecated(name, *, since, removal, replacement)` + `reset_deprecation_dedupe(*targets)`. The canonical formatter; deduplicates per `(name, since, removal)` triple. Use this; do not write parallel warning shims.
- **`runtime/_litellm.py`** — `LLMInterface` ABC + `LiteLLMInterface` (litellm; 100+ providers). Private-by-convention since 0.7.0 (D-009 formalised; no top-level re-export). Subclasses passed to `Program.from_fsm(model=..., **llm_kwargs)` are auto-wrapped in `LiteLLMOracle`. **D-008 caveat**: `LiteLLMOracle._invoke_structured` bypasses the user-supplied `generate_response` for structured Leaves; subclasses needing custom provider logic on every call should pass an `Oracle` directly via `oracle=`.
- **`memory.py`, `context.py`, `dialog/session.py`** — `WorkingMemory` (4 named buffers), `ContextCompactor` (transient-key clearing), `FileSessionStore` (atomic writes via temp→rename).

## Package Map

```
src/
├── fsm_llm/                       # The kernel + dialog surface + standard library
│   ├── runtime/                   # M1 — typed λ-AST + executor + planner + oracle + _litellm
│   │   ├── _handlers_ast.py       #   private — compose + AST splicers (moved from handlers.py at 0.8.0)
│   │   └── ...                    #   ast.py, dsl.py, combinators.py, executor.py, planner.py, oracle.py, _litellm.py, cost.py, errors.py
│   ├── dialog/                    # FSM dialog surface — API, FSMManager, MessagePipeline (turn.py),
│   │   ├── extraction.py          #   private — ExtractionEngine (extracted from turn.py at 0.8.0)
│   │   └── ...                    #   prompts, classifiers, transition_evaluator, definitions, session, compile_fsm
│   ├── stdlib/                    # M3 — named λ-term factories
│   │   ├── agents/                #   react_term, rewoo_term, reflexion_term, memory_term + 12 class agents
│   │   ├── reasoning/             #   11 strategy factories + classifier_term + solve_term + ReasoningEngine
│   │   │                          #   (descriptive parameter names since 0.8.0)
│   │   ├── workflows/             #   linear/branch/switch/parallel/retry term factories + WorkflowEngine
│   │   └── long_context/          #   niah_term, aggregate_term, pairwise_term, multi_hop_term,
│   │                              #   multi_hop_dynamic_term, niah_padded_term + helpers
│   ├── _api/                      # Private deprecation machinery
│   ├── types.py                   # Neutral types layer (since 0.7.0) — FSMError + request/response models
│   ├── program.py                 # Program facade
│   ├── handlers.py                # HandlerSystem + HandlerBuilder + HandlerTiming (compose re-exported from runtime/_handlers_ast.py)
│   └── profiles.py                # HarnessProfile + ProviderProfile + registries
│
└── fsm_llm_monitor/               # Native top-level package — web dashboard + OTEL exporter

# Removed at 0.7.0 (R13/I5 epoch closure): src/fsm_llm_reasoning/,
# src/fsm_llm_workflows/, src/fsm_llm_agents/ sibling-shim packages.
# Use the canonical fsm_llm.stdlib.{reasoning,workflows,agents} paths.
```

## Optional Extras

| Extra | Command | Dependencies |
|-------|---------|-------------|
| `reasoning` | `pip install fsm-llm[reasoning]` | None (stdlib subpackage) |
| `agents` | `pip install fsm-llm[agents]` | None (stdlib subpackage) |
| `workflows` | `pip install fsm-llm[workflows]` | None (stdlib subpackage) |
| `monitor` | `pip install fsm-llm[monitor]` | fastapi, uvicorn, jinja2 |
| `mcp` | `pip install fsm-llm[mcp]` | mcp (>=1.0.0) |
| `otel` | `pip install fsm-llm[otel]` | opentelemetry-api, opentelemetry-sdk (>=1.20.0) |
| `a2a` | `pip install fsm-llm[a2a]` | httpx (>=0.24.0) |
| `oolong` | `pip install fsm-llm[oolong]` | datasets (>=3.0.0) |
| `all` | `pip install fsm-llm[all]` | All of the above |

Each subpackage that has a `CLAUDE.md` exposes its own file map. Start at `src/fsm_llm/CLAUDE.md` and `src/fsm_llm/stdlib/CLAUDE.md`.

## Code Conventions

- **Linting / Formatting**: ruff (target Python 3.10, line-length 88). Ignored: E402, E501, RUF013, RUF001, RUF022.
- **Type hints**: Used throughout. mypy with `disallow_untyped_defs=false`, pydantic plugin enabled.
- **Models**: Pydantic v2 `BaseModel` with `model_validator` for complex validation. Recursive AST models work with `ConfigDict(frozen=True)` + `model_rebuild()`.
- **Logging**: loguru via `from fsm_llm.logging import logger`.
- **Exports**: Single `__all__` in each `__init__.py` — no dynamic extend/append. The top-level `__all__` is layered (L1–L4 + Legacy); an import-audit test (`tests/test_fsm_llm/test_layering.py`) enforces it.
- **Stdlib purity**: Modules under `src/fsm_llm/stdlib/<pkg>/lam_factories.py` import **only from `fsm_llm.runtime`**. AST-walk unit tests enforce this per subpackage.
- **Deprecations**: Use `fsm_llm._api.deprecation.warn_deprecated(name, *, since, removal, replacement)` — do not write parallel warning shims.
- **Exceptions**:
  - Core: `FSMError` → `StateNotFoundError`, `InvalidTransitionError`, `LLMResponseError`, `TransitionEvaluationError`, `ClassificationError` → `SchemaValidationError`, `ClassificationResponseError`. **`ProgramModeError(FSMError)`** for mode mismatches on `.invoke`.
  - λ-kernel: `LambdaError` → `ASTConstructionError`, `TerminationError`, `PlanningError`, `OracleError`.
  - Handlers: `HandlerSystemError(FSMError)` → `HandlerExecutionError`.
  - Reasoning: `ReasoningEngineError` → `ReasoningExecutionError`, `ReasoningClassificationError`.
  - Workflows: `WorkflowError` → `WorkflowDefinitionError`, `WorkflowStepError`, `WorkflowInstanceError`, `WorkflowTimeoutError`, `WorkflowValidationError`, `WorkflowStateError`, `WorkflowEventError`, `WorkflowResourceError`.
  - Agents: `AgentError` → `ToolExecutionError`, `ToolNotFoundError`, `ToolValidationError`, `BudgetExhaustedError`, `ApprovalDeniedError`, `AgentTimeoutError`, `EvaluationError`, `DecompositionError`. Meta: `MetaBuilderError(AgentError)` → `BuilderError`, `MetaValidationError`, `OutputError`.
  - Monitor: `MonitorError(Exception)` → `MonitorInitializationError`, `MetricCollectionError`, `MonitorConnectionError`. (Inherits from `Exception`, not `FSMError`.)
- **Constants**: Centralised in `constants.py` per package.
- **Security**: Internal context key prefixes (`_`, `system_`, `internal_`, `__`). Forbidden patterns for passwords/secrets/tokens. XML tag sanitisation in prompts.

## FSM Definition Format (JSON v4.1) — Category A surface

Compiled to a λ-term at load time. FSM JSON is the authoring format for dialog programs that need persistent per-turn state and non-linear transition graphs. Stateless / pipeline / long-context programs use the λ-DSL directly (`Program.from_term` / `Program.from_factory`).

```json
{
  "name": "MyBot",
  "initial_state": "start",
  "persona": "A friendly assistant",
  "states": {
    "start": {
      "id": "start",
      "description": "Brief state description",
      "purpose": "What should be accomplished",
      "extraction_instructions": "What data to extract",
      "response_instructions": "How to respond",
      "required_context_keys": ["key1"],
      "classification_extractions": [{
        "field_name": "user_intent",
        "schema": {"intents": [{"name": "buy", "description": "User wants to purchase"}],
                    "fallback_intent": "browse"},
        "confidence_threshold": 0.7
      }],
      "transitions": [{
        "target_state": "next",
        "description": "When this transition should fire",
        "priority": 100,
        "conditions": [{
          "description": "Human-readable condition",
          "requires_context_keys": ["key1"],
          "logic": {"==": [{"var": "key1"}, "expected_value"]}
        }]
      }]
    }
  }
}
```

## Testing

Verify counts via `.venv/bin/python -m pytest --collect-only -q | tail -3`.

```bash
pytest                                     # All tests (~3302 passing at 0.8.0)
pytest tests/test_fsm_llm/                # Core package
pytest tests/test_fsm_llm_lam/            # λ-kernel (Executor / Planner / DSL / FSM compiler)
pytest tests/test_fsm_llm_long_context/   # M5 long-context factories
pytest tests/test_fsm_llm_reasoning/      # Reasoning
pytest tests/test_fsm_llm_workflows/      # Workflows
pytest tests/test_fsm_llm_agents/         # Agents
pytest tests/test_fsm_llm_monitor/        # Monitor
pytest tests/test_fsm_llm_meta/           # Meta builder
pytest tests/test_fsm_llm_regression/     # Regression suite
pytest tests/test_examples/               # Example validation
pytest tests/test_scripts/                # bench / eval / loader scripts
pytest -m "not slow"                      # Skip slow
pytest -m integration                     # Integration only
pytest -m real_llm                        # Live LLM smokes (gated by TEST_REAL_LLM=1)
```

**Conventions**:
- Test files: `test_<module>.py` and `test_<module>_elaborate.py` for extended scenarios.
- Test classes: `class Test<Feature>`. Helper functions: prefixed with `_`.
- **Markers**: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.examples`, `@pytest.mark.real_llm`.
- **Mock LLMs**: `Mock(spec=LLMInterface)` (simple) and `MockLLM2Interface` (2-pass) in `conftest.py`. For agent tests under M3c, use `FunctionalMockLLM` (call-count-keyed callables) — see `tests/test_fsm_llm_agents/test_bug_fixes.py`.
- **Fixtures**: `sample_fsm_definition_v2` (v4.1), `mock_llm2_interface`.
- **Environment**: `SKIP_SLOW_TESTS`, `TEST_REAL_LLM`, `TEST_LLM_MODEL`, `OPENAI_API_KEY`.
- λ-kernel tests assert `ex.oracle_calls == plan(...).predicted_calls` (Theorem-2 contract). Long-context bench scorecards under `evaluation/` record `theorem2_holds` per cell.

## Examples

**172 examples across 10 trees** — verify with `find examples -mindepth 2 -maxdepth 2 -type d | wc -l`.

**Legacy (Category A — FSM JSON, read-only baselines)**:
- `basic/`, `intermediate/`, `advanced/`, `classification/`, `agents/`, `reasoning/`, `workflows/`

**λ-native (Category B/C, plus meta builder)**:
- `pipeline/` — λ-DSL twins of `agents/*`. Each is a 3-file shape: `__init__.py`, `schemas.py`, `run.py` with an inline λ-term and a `VERIFICATION` block parsed by the eval harness.
- `long_context/` — `niah_demo`, `niah_padded_demo`, `aggregate_demo`, `pairwise_demo`, `multi_hop_demo`. Each ships a hard `oracle_calls_match_planner` Theorem-2 gate.
- `meta/` — `build_fsm`, `build_workflow`, `build_agent`, `meta_review_loop`, `meta_from_spec`.

All examples support OpenAI and Ollama fallback. Run with: `python examples/<tree>/<name>/run.py`.

**IMPORTANT — Do NOT modify existing examples unless explicitly asked.** Examples (especially under the legacy 7 trees and `examples/pipeline/`) serve as stable evaluation baselines.

### Evaluation

Automated evaluation via `scripts/eval.py` runs examples in parallel and produces scorecards. Long-context bench scorecards under `evaluation/bench_long_context_*.json` document Theorem-2 evidence per `(model × factory)` cell. See `EVALUATE.md` for methodology.

## Documentation

- `README.md` — Public-facing project overview + quick start.
- `docs/lambda_fsm_merge.md` — **The merge contract (canonical)**: invariants, falsification gates, the unified-API specification, deprecation calendar.
- `docs/lambda.md` — **Architectural thesis**: λ-calculus as substrate, FSM as one surface, Theorems 1–5.
- `docs/quickstart.md` — Five-minute tour.
- `docs/api_reference.md` — Complete API (L1–L4 + Legacy).
- `docs/architecture.md` — System design, layered architecture, Theorem-2.
- `docs/handlers.md` — Handler lifecycle reference.
- `docs/fsm_design.md` — FSM design patterns, anti-patterns.
- `docs/migration_0.7_to_0.8.md` — Migration guide for callers upgrading from 0.7.x.
- `docs/threat_model.md` — Trust boundaries, T-01..T-11, dismissed proposals.
- `docs/deepagents.md` — deepagents reverse-engineering analysis.
- `docs/archive/` — superseded docs preserved for historical context (incl. archived `strands_features*.md` Phase 1/2 implementation logs).
- `CHANGELOG.md` — Release notes.

## Pre-commit & CI

- **Pre-commit**: trailing whitespace, EOF fixer, YAML/JSON validation, ruff (with `--fix`), pytest pre-push.
- **CI**: GitHub Actions on push/PR to `main` — tests on Python 3.10, 3.11, 3.12.
- **Tox**: Multi-version testing + lint + mypy environments.

## Critical Instructions for Claude

1. **Always use `.venv`** for Python invocations. `.venv/bin/python` or activate first.
2. **Never modify existing examples** unless the user explicitly asks. They are evaluation baselines.
3. **No new `docs/monitor.md`** — monitor docs go in `docs/api_reference.md`.
4. **Numbers in this file may drift** — when you need exact figures (test count, examples count), verify with the commands above before quoting them.
5. **The merge contract is `docs/lambda_fsm_merge.md`** — when in doubt about API/architecture, that's the source of truth.
6. **Layered imports**: top-level `__all__` is layered. New public names need a layer assignment (L1/L2/L3/L4 or Legacy) and a corresponding entry in the layering test (`tests/test_fsm_llm/test_layering.py`) if they bridge layers. The kernel↔dialog allow-list is at 0 since 0.7.0; if a new entry is required, surface it as a D-NNN-SURPRISE (don't silently extend).
7. **Deprecation calendar** is testable. The asserter `tests/test_fsm_llm/test_deprecation_calendar.py` flips its expectations automatically per `__version__`. Use `warn_deprecated` from `_api/deprecation.py` for any new deprecation; do not write parallel warning shims.
8. **`fsm_llm.lam` is gone (R13 epoch).** Don't import from it. Canonical paths are `fsm_llm.runtime` (kernel) and `fsm_llm.dialog.compile_fsm` (FSM compiler), with the convenience exports at the top level (`from fsm_llm import compile_fsm, Executor, Term, leaf, …`).
9. **The I5 epoch is closed (0.7.0).** All five surfaces — `Program.run/.converse/.register_handler`, `from fsm_llm import API`, `import fsm_llm_{reasoning,workflows,agents}`, long-context bare names, top-level `LiteLLMInterface` — are hard removals. Don't try to re-introduce them; use the canonical paths from the migration guides.
10. **The Z8 epoch is closed (0.8.0).** Eight more removals — `Handler` top-level alias, top-level `LLMInterface` / `BUILTIN_OPS`, `has_*` / `get_*` extension-check helpers, `dialog/definitions.py` type re-exports, `_emit_response_leaf_for_non_cohort` State field, `Program.__init__` `_api`/`_profile` kwargs, `Program.from_fsm` `**api_kwargs` catch-all. Don't re-introduce; see "What changed in 0.8.0" for canonical paths.
11. **`fsm_llm.types` is the canonical home** for the shared `FSMError` hierarchy + runtime-touching Pydantic models. New code: `from fsm_llm.types import FSMError`. The 0.7.0 back-compat re-exports from `fsm_llm.dialog.definitions` were removed in 0.8.0 — update existing imports.
