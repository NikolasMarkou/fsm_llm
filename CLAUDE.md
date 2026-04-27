# FSM-LLM — Claude Code Instructions

## Project Overview

FSM-LLM (v0.3.0) is a Python framework for building stateful LLM programs on a typed **λ-calculus runtime**. Two surface syntaxes share one executor:

- **FSM JSON (Category A)** — dialog programs with persistent per-turn state. Compiled to λ-terms at load time.
- **λ-DSL (Category B/C)** — pipelines, agents, reasoning chains, long-context recursion. Authored as λ-terms directly.

The substrate is the typed λ-AST in `src/fsm_llm/lam/` (M1 kernel). FSM is one front-end (M2 compiler); λ-DSL is the other. See `docs/lambda.md` for the full architectural thesis (§1 motivation, §11 package map, §13 milestone status).

- **License**: GPL-3.0-or-later
- **Python**: 3.10, 3.11, 3.12
- **Core deps**: `loguru`, `litellm` (>=1.82,<2.0, excluding 1.82.7/1.82.8), `pydantic` (>=2.0), `python-dotenv`
- **Virtual environment**: Always use `.venv` — run with `.venv/bin/python` or activate first.

## Quick Commands

```bash
make test           # pytest -v (currently 2,899 tests across all packages)
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make type-check     # mypy across all packages
make build          # python -m build (wheel + sdist)
make coverage       # pytest with coverage report
make install-dev    # pip install -c constraints.txt -e ".[dev,workflows,reasoning,agents,monitor,oolong]" + pre-commit install
make audit          # audit site-packages for suspicious .pth files

# CLI tools — unified `fsm-llm` binary (R7) with subcommand dispatch
fsm-llm run <target>                # Run FSM JSON (interactive) or factory (pkg.mod:fn)
fsm-llm explain <target> [--n N --K K]  # Print AST shape, leaf schemas, and Plans (per Fix subtree when (n,K) supplied)
fsm-llm validate --fsm <path.json>  # Validate FSM definition (also: fsm-llm-validate alias)
fsm-llm visualize --fsm <path.json> # ASCII visualization (also: fsm-llm-visualize alias)
fsm-llm meta                        # Interactive artifact builder (also: fsm-llm-meta alias)
fsm-llm monitor                     # Launch web monitoring dashboard (also: fsm-llm-monitor alias)

# Legacy --mode-flagged binary (kept for back-compat, both surfaces ship in 0.4.x):
fsm-llm --fsm <path.json>           # Run FSM interactively (compiled λ-path)

# Legacy console-script aliases — re-pointed at fsm_llm.cli.main in R7 (D-PLAN-04):
fsm-llm-visualize --fsm <path.json>
fsm-llm-validate --fsm <path.json>
fsm-llm-monitor
fsm-llm-meta
```

## Architecture — One Runtime, Two Surfaces

```
        FSM JSON  (Category A)             λ-DSL  (Category B / C)
              │                                    │
              ▼  fsm_llm.lam.fsm_compile           ▼  fsm_llm.lam.dsl
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
        │ Oracle     (LiteLLMOracle ← LiteLLMInterface)
        │ Session    (per-conversation persistence)│
        │ Cost       (CostAccumulator, LeafCall)   │
        └──────────────────────────────────────────┘
```

### Category A — FSM dialog (single-path, post-M2 S11)

`API.converse(msg, conv_id)` loads a compiled λ-term from cache, evaluates one β-reduction step, persists `(state_id, context)` to the session store. The 2-pass shape (data extraction → transition eval → response generation) is preserved as the body of the compiled term — but `MessagePipeline.process` and `process_stream` are **retired**, and `FSMManager.use_compiled` is **removed**. Single execution path.

### Category B/C — λ-DSL programs

Pipelines (B) and long-context recursion (C) are written directly as λ-terms via the kernel DSL:

```python
from fsm_llm.lam import let_, leaf, fix, case_, var, split, fmap, reduce_
from fsm_llm.lam import Executor, LiteLLMOracle
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.stdlib.long_context import niah  # M5 slice 1

term = niah(question="What animal is the protagonist?", tau=256, k=2)
ex = Executor(oracle=LiteLLMOracle(LiteLLMInterface(model="ollama_chat/qwen3.5:4b")))
result = ex.run(term, env={"document": long_doc})
assert ex.oracle_calls == plan(...).predicted_calls   # Theorem-2 holds
```

Per-Fix node, the planner pre-computes `(k*, τ*, d, predicted_calls)`; the executor delivers exactly that many oracle calls when input is τ·k^d-aligned (Theorem 2). See `docs/lambda.md` §12 for the full theorem set.

## Key Modules in `src/fsm_llm/`

- **`runtime/`** — typed λ-kernel (11 modules). AST, DSL builders, Executor, Planner, Oracle, `_litellm.py` (was top-level `llm.py`), cost tracker. **The substrate.** See `src/fsm_llm/runtime/CLAUDE.md`. Renamed from `lam/` in plan v3 R4 — old `from fsm_llm.lam import …` paths still work via a sys.modules shim at `src/fsm_llm/lam/__init__.py`.
- **`dialog/`** — FSM dialog front-end (R4). Houses `API`, `FSMManager`, `MessagePipeline`, classifiers, `TransitionEvaluator`, prompt builders, `compile_fsm`/`compile_fsm_cached`, definitions, and sessions. See `src/fsm_llm/dialog/CLAUDE.md`. Old top-level paths (`from fsm_llm.api import API`, etc.) keep working via shims.
- **`stdlib/`** — named λ-term factories organised by domain (`agents/`, `reasoning/`, `workflows/`, `long_context/`). Each subpackage's `lam_factories.py` exposes pure factory functions returning `Term`. See `src/fsm_llm/stdlib/CLAUDE.md`.
- **`program.py`** — **`Program` facade (R1)** — unified entry over `(term, oracle, optional_session, optional_handlers)`. Three constructors: `Program.from_fsm(defn)` (delegates to `API`), `Program.from_term(term)`, `Program.from_factory(factory, factory_args=(), factory_kwargs={})`. Surface: `.run(**env)` (term/factory mode) → calls `Executor.run` byte-equivalently; `.converse(msg, conversation_id)` (FSM mode) → delegates to `API.converse`, auto-starts conversation if id is None and caches it on the Program; `.explain()` → `ExplainOutput(plans=[], leaf_schemas, ast_shape)` (R1 returns `plans=[]` — runtime `(n, K)` not yet wired); `.register_handler(handler)` → FSM-mode delegates to `API.register_handler`, term-mode raises (R5 territory). When `oracle=` is supplied to `from_fsm`, must be a `LiteLLMOracle` (unwrapped to its underlying `LLMInterface` for API); non-LiteLLM oracles raise `TypeError`. See `# DECISION D-001` in `program.py` and `plans/plan_2026-04-27_a426f667/decisions.md` D-PLAN-02.
- **`dialog/api.py`** — `API` class. Entry: `from_file()`, `from_definition()`. Conversation: `converse()`, `start_conversation()`, `end_conversation()`. FSM stacking: `push_fsm()`, `pop_fsm()`. Handlers: `register_handler()`, `create_handler()`. Internally routes through compiled λ-term cache.
- **`dialog/fsm.py`** — `FSMManager`. Per-conversation thread locks. The compiled-term cache lives in `dialog/compile_fsm.py` as of R2 (plan v3 step 8): `FSMManager.get_compiled_term(fsm_id)` is a 3-line shim routing to `compile_fsm_cached`. The previous per-manager `_compiled_terms` OrderedDict was removed. Thin adapter over the λ-executor.
- **`dialog/pipeline.py`** — `MessagePipeline`. The compiled-path 2-pass body (extract → evaluate → respond). Internal; `process`/`process_stream` retired in M2 S11. **R3 status (plan v3 step 14, narrowed)**: pipeline callbacks remain on `LiteLLMInterface.{generate_response, extract_field}` at HEAD — collapse to `oracle.invoke` is deferred to R6 (per-state Leaf specialisation). The unified `Oracle.invoke(template, *, env, schema)` shape (with R3 env branch, see `runtime/CLAUDE.md`) and template-producer surface in `dialog/prompts.py` (`*PromptBuilder.to_template_and_schema` + free `classification_template`) ship in this PR as forward-compat plumbing. See `plans/plan_2026-04-27_a426f667/decisions.md` D-PLAN-09-RESOLUTION-step14-narrowed.
- **`handlers.py`** — `HandlerSystem`, `HandlerBuilder`, `HandlerTiming` (8 hook points: `START_CONVERSATION`, `PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `END_CONVERSATION`, `ERROR`). Hooks compose into the compiled λ-term per `docs/lambda.md` §6.3.
- **`dialog/classification.py`** — `Classifier`, `HierarchicalClassifier`, `IntentRouter` for ambiguity resolution at transitions.
- **`dialog/transition_evaluator.py`** — Rule-based transition evaluation: `DETERMINISTIC` / `AMBIGUOUS` / `BLOCKED`.
- **`runtime/_litellm.py`** — `LLMInterface` ABC + `LiteLLMInterface` (litellm; 100+ providers). `generate_response`, `extract_field`, `generate_response_stream`. Was `fsm_llm/llm.py` pre-R4; old import path kept via shim. **Use `generate_response` + Pydantic `response_format` for small models** — see LESSONS for `extract_field` caveats with qwen3.5:4b.
- **`dialog/session.py`** — `SessionStore` ABC, `FileSessionStore` (atomic writes via temp→rename).
- **`memory.py`** — `WorkingMemory` (4 named buffers: core, scratch, environment, reasoning).
- **`context.py`** — `ContextCompactor` (transient-key clearing, pruning, summarisation).
- **`dialog/definitions.py`** — Pydantic v2 models: `State`, `Transition`, `FSMDefinition`, `FSMContext`, `FSMInstance`, `Conversation`.

## Package Map (post-unification per `docs/lambda.md` §11)

```
src/
├── fsm_llm/                       # The kernel + dialog surface + standard library
│   ├── runtime/                   # M1 — typed λ-AST + executor + planner + oracle + _litellm. Was lam/ pre-R4.
│   ├── dialog/                    # FSM dialog surface (R4) — API, FSMManager, MessagePipeline, prompts, classifiers, transition_evaluator, definitions, session, compile_fsm
│   ├── stdlib/                    # M3 — named λ-term factories
│   │   ├── agents/                #   slice 1: react_term, rewoo_term, reflexion_term, memory_term
│   │   ├── reasoning/             #   slice 2: 11 strategy factories + classifier_term + solve_term
│   │   ├── workflows/             #   slice 3: linear/branch/switch/parallel/retry term factories
│   │   └── long_context/          #   M5: niah, aggregate, pairwise, multi_hop, niah_padded + helpers
│   ├── lam/                       # sys.modules shim → fsm_llm.runtime (R4 D-004; deprecation 0.5.0; removal 0.6.0)
│   └── (api, fsm, pipeline, prompts, classification, transition_evaluator, definitions, session, llm)  # all sys.modules shims → fsm_llm.dialog.<x> (or fsm_llm.runtime._litellm for llm). Same R4 D-004 / D-PLAN-10 timeline.
│
├── fsm_llm_reasoning/             # sys.modules shim → fsm_llm.stdlib.reasoning
├── fsm_llm_workflows/             # sys.modules shim → fsm_llm.stdlib.workflows
├── fsm_llm_agents/                # sys.modules shim → fsm_llm.stdlib.agents
└── fsm_llm_monitor/               # Native top-level package — web dashboard + OTEL exporter
```

The three `fsm_llm_*` siblings (reasoning / workflows / agents) are **silent back-compat shims** — `from fsm_llm_agents import ReactAgent` resolves to the same object as `from fsm_llm.stdlib.agents import ReactAgent`. No deprecation warning. New code should import from `fsm_llm.stdlib.<pkg>`.

Inside `fsm_llm/`, the R4 reorganisation introduced two parallel shim layers:
- **`fsm_llm.lam` ↔ `fsm_llm.runtime`** — kernel rename. 10 submodule paths preserved (`fsm_llm.lam.executor` is the same module object as `fsm_llm.runtime.executor`). `fsm_llm.lam.fsm_compile` resolves to `fsm_llm.dialog.compile_fsm` because the FSM compiler moved to the dialog side per `docs/lambda.md` §11.
- **`fsm_llm.{api,fsm,pipeline,prompts,classification,transition_evaluator,definitions,session}` ↔ `fsm_llm.dialog.<x>`** — front-end move. 8 module paths preserved. `fsm_llm.llm` ↔ `fsm_llm.runtime._litellm`. Identity contract verified by `tests/test_fsm_llm/test_module_shims.py`. Silent in 0.4.x; deprecation in 0.5.0; removal in 0.6.0.

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
| `oolong` | `pip install fsm-llm[oolong]` | datasets (>=3.0.0) — M5 slice 7 OOLONG benchmark loader |
| `all` | `pip install fsm-llm[all]` | All of the above |

Each subpackage that has a CLAUDE.md exposes its own file map. Start at `src/fsm_llm/CLAUDE.md` (kernel) and `src/fsm_llm/stdlib/CLAUDE.md` (factory layers).

## Code Conventions

- **Linting/Formatting**: ruff (target Python 3.10, line-length 88). Ignored: E402, E501, RUF013, RUF001, RUF022.
- **Type hints**: Used throughout. mypy with `disallow_untyped_defs=false`, pydantic plugin enabled.
- **Models**: Pydantic v2 `BaseModel` with `model_validator` for complex validation. Recursive AST models work with `ConfigDict(frozen=True)` + `model_rebuild()`.
- **Logging**: loguru via `from fsm_llm.logging import logger`.
- **Exports**: Single `__all__` in each `__init__.py` — no dynamic extend/append.
- **Stdlib purity**: Modules under `src/fsm_llm/stdlib/<pkg>/lam_factories.py` import **only from `fsm_llm.lam`**. AST-walk unit test enforces this per subpackage.
- **Exceptions**:
  - Core: `FSMError` → `StateNotFoundError`, `InvalidTransitionError`, `LLMResponseError`, `TransitionEvaluationError`, `ClassificationError` → `SchemaValidationError`, `ClassificationResponseError`.
  - λ-kernel: `LambdaError` → `ASTConstructionError`, `TerminationError`, `PlanningError`, `OracleError`.
  - Handlers: `HandlerSystemError(FSMError)` → `HandlerExecutionError`.
  - Reasoning: `ReasoningEngineError` → `ReasoningExecutionError`, `ReasoningClassificationError`.
  - Workflows: `WorkflowError` → `WorkflowDefinitionError`, `WorkflowStepError`, `WorkflowInstanceError`, `WorkflowTimeoutError`, `WorkflowValidationError`, `WorkflowStateError`, `WorkflowEventError`, `WorkflowResourceError`.
  - Agents: `AgentError` → `ToolExecutionError`, `ToolNotFoundError`, `ToolValidationError`, `BudgetExhaustedError`, `ApprovalDeniedError`, `AgentTimeoutError`, `EvaluationError`, `DecompositionError`. Meta: `MetaBuilderError(AgentError)` → `BuilderError`, `MetaValidationError`, `OutputError`.
  - Monitor: `MonitorError(Exception)` → `MonitorInitializationError`, `MetricCollectionError`, `MonitorConnectionError`. (Inherits from `Exception`, not `FSMError`.)
- **Constants**: Centralised in `constants.py` per package.
- **Security**: Internal context key prefixes (`_`, `system_`, `internal_`, `__`). Forbidden patterns for passwords/secrets/tokens. XML tag sanitisation in prompts.

## FSM Definition Format (JSON v4.1) — Category A surface

Compiled to a λ-term at load time (M2 S11; single-path runtime). FSM JSON is the authoring format for dialog programs that need persistent per-turn state and non-linear transition graphs. Stateless / pipeline / long-context programs should use the λ-DSL directly.

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
        "schema": {"intents": [{"name": "buy", "description": "User wants to purchase"}], "fallback_intent": "browse"},
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

Total: **2,899 tests** (verified via `pytest --collect-only`). Per-package collected counts:

```bash
pytest                                     # All tests
pytest tests/test_fsm_llm/                # Core package — 837 tests
pytest tests/test_fsm_llm_lam/            # λ-kernel — 202 tests (Executor / Planner / DSL / FSM compiler)
pytest tests/test_fsm_llm_long_context/   # M5 long-context factories — 61 tests
pytest tests/test_fsm_llm_reasoning/      # Reasoning — 134 tests
pytest tests/test_fsm_llm_workflows/      # Workflows — 155 tests
pytest tests/test_fsm_llm_agents/         # Agents — 723 tests
pytest tests/test_fsm_llm_monitor/        # Monitor — 245 tests
pytest tests/test_fsm_llm_meta/           # Meta builder — 205 tests
pytest tests/test_fsm_llm_regression/     # Regression suite — 273 tests
pytest tests/test_examples/               # Example validation — 44 tests
pytest tests/test_scripts/                # bench / eval / loader scripts — 8 tests
pytest -m "not slow"                      # Skip slow
pytest -m integration                     # Integration only
pytest -m real_llm                        # Live LLM smokes (gated by TEST_REAL_LLM=1)
```

**Conventions**:
- Test files: `test_<module>.py` and `test_<module>_elaborate.py` for extended scenarios.
- Test classes: `class Test<Feature>`. Helper functions: prefixed with `_`.
- **Markers**: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.examples`, `@pytest.mark.real_llm`.
- **Mock LLMs**: `Mock(spec=LLMInterface)` (simple) and `MockLLM2Interface` (2-pass) in `conftest.py`.
- **Fixtures**: `sample_fsm_definition_v2` (v4.1), `mock_llm2_interface`.
- **Environment**: `SKIP_SLOW_TESTS`, `TEST_REAL_LLM`, `TEST_LLM_MODEL`, `OPENAI_API_KEY`.
- λ-kernel tests assert `ex.oracle_calls == plan(...).predicted_calls` (Theorem-2 contract); long-context bench scorecards under `evaluation/` record `theorem2_holds` per cell.

## Examples

**152 examples across 10 trees** — both legacy (FSM) and λ-native:

**Legacy (Category A — FSM JSON, read-only baselines)** — 95 examples:
- `basic/` (14), `intermediate/` (3), `advanced/` (17), `classification/` (4), `agents/` (48), `reasoning/` (1), `workflows/` (8)

**λ-native (Category B/C, plus meta builder)** — 57 examples:
- `pipeline/` (47) — M4 Category-B λ-DSL twins of `agents/*`. Each is a 3-file shape: `__init__.py`, `schemas.py`, `run.py` with an inline λ-term and a `VERIFICATION` block parsed by the eval harness.
- `long_context/` (5) — M5 Category-C demos: `niah_demo`, `niah_padded_demo`, `aggregate_demo`, `pairwise_demo`, `multi_hop_demo`. Each ships a hard `oracle_calls_match_planner` Theorem-2 gate.
- `meta/` (5) — Meta-builder examples: `build_fsm`, `build_workflow`, `build_agent`, `meta_review_loop`, `meta_from_spec`.

All examples support OpenAI and Ollama fallback. Run with: `python examples/<tree>/<name>/run.py`.

**IMPORTANT**: Do NOT modify existing examples unless explicitly asked. Examples (especially under the legacy 7 trees and `examples/pipeline/`) serve as stable evaluation baselines. The M4 corpus under `examples/pipeline/` was the regression evidence that M4 closed.

### Evaluation

Automated evaluation via `scripts/eval.py` runs examples in parallel and produces scorecards. **Last published baseline: 90.8% health score on `ollama_chat/qwen3.5:4b`** (Run 004, 2026-04-02, 100 examples). The current discoverable inventory is 152 examples → a fresh eval is pending. Long-context bench scorecards under `evaluation/bench_long_context_*.json` and slice-specific `evaluation/m3_slice*_*_scorecard.json` document Theorem-2 evidence per (model × factory) cell. See `EVALUATE.md` for methodology.

## Documentation

- `README.md` — Public-facing project overview + quick start.
- `docs/lambda.md` — **Architectural thesis**: λ-calculus as substrate, FSM as one surface. Authoritative for §11 package map and §13 milestone status (M1-M5 slice-by-slice).
- `docs/quickstart.md` — Getting started.
- `docs/api_reference.md` — Complete `API` class documentation.
- `docs/architecture.md` — System design, 2-pass flow within compiled λ-terms, security, performance.
- `docs/fsm_design.md` — FSM design patterns, anti-patterns.
- `docs/handlers.md` — Handler development guide (8 timing points).
- `CHANGELOG.md` — Version history (current: 0.3.0).

## Pre-commit & CI

- **Pre-commit**: trailing whitespace, EOF fixer, YAML/JSON validation, ruff (with `--fix`), pytest pre-push.
- **CI**: GitHub Actions on push/PR to `main` — tests on Python 3.10, 3.11, 3.12.
- **Tox**: Multi-version testing + lint + mypy environments.
