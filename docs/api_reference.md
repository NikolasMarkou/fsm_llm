# API Reference

Complete API for `fsm-llm` `0.7.0`. The public surface is **layered** (L1–L4 plus a Legacy block); the canonical contract is in [`lambda_fsm_merge.md`](lambda_fsm_merge.md) §4. New code should reach for the L4 verb (`Program`).

> **Security.** For trust boundaries, threats, and assumptions integrators are expected to honor, see [`threat_model.md`](threat_model.md).

## Layers at a glance

| Layer | Purpose | Headline names |
|-------|---------|----------------|
| **L4 INVOKE** | One verb, one return type | `Program`, `Result`, `ExplainOutput`, `ProgramModeError` |
| **L3 AUTHOR** | Term producers | stdlib factories (`react_term`, `niah_term`, `analytical_term`, `linear_term`, …); `compile_fsm`, `compile_fsm_cached` |
| **L2 COMPOSE** | Pure Term→Term transforms | `compose`, `Handler`, `HandlerBuilder`, `HandlerTiming`, `HandlerSystem`, `HarnessProfile`, `ProviderProfile`, `register_*`, `get_*` |
| **L1 REDUCE** | Typed substrate | `Term`, `Var`, `Abs`, `App`, `Let`, `Case`, `Combinator`, `Fix`, `Leaf`; `Executor`, `Plan`, `plan`, `Oracle`, `LiteLLMOracle`; DSL builders (`var`, `leaf`, `let_`, …) |
| **Legacy** | FSM dialog front-end + utilities | `API`, `FSMManager`, definitions, classifiers, etc. |

The layering is asserted by `tests/test_fsm_llm/test_layering.py` — disjoint partitions plus an exhaustive cover of `__all__`.

---

## L4 — INVOKE

The four-name layer that carries the verb.

### `Program`

The unified facade. **Mode is fixed at construction**; `.invoke(...)` returns a `Result` in every mode.

```python
from fsm_llm import Program
```

#### Constructors

```python
Program.from_fsm(
    fsm_definition: FSMDefinition | dict | str | Path,
    *,
    oracle: LiteLLMOracle | None = None,
    session: SessionStore | None = None,
    handlers: list[FSMHandler] | None = None,
    profile: HarnessProfile | str | None = None,
    **api_kwargs,
) -> Program
```
Build from FSM JSON. Internally constructs an `API` and delegates `.invoke` / `.register_handler` to it. When `oracle=` is supplied, it must be a `LiteLLMOracle` (we unwrap to its underlying `LiteLLMInterface`); non-`LiteLLMOracle` values raise `TypeError`. Default oracle is constructed lazily — building the Program never touches the network.

```python
Program.from_term(
    term: Term,
    *,
    oracle: Oracle | None = None,
    session: SessionStore | None = None,
    handlers: list[FSMHandler] | None = None,
    profile: HarnessProfile | str | None = None,
) -> Program
```
Wrap a pre-authored λ-term. Stateless one-shot evaluations.

```python
Program.from_factory(
    factory: Callable[..., Term],
    factory_args: tuple = (),
    factory_kwargs: dict | None = None,
    *,
    oracle: Oracle | None = None,
    session: SessionStore | None = None,
    handlers: list[FSMHandler] | None = None,
    profile: HarnessProfile | str | None = None,
) -> Program
```
Call a factory at construction time and wrap its result. `factory_args` and `factory_kwargs` are explicit (separate from facade kwargs).

#### `.invoke(...)`

```python
# FSM mode
result = prog.invoke(message: str, conversation_id: str | None = None) -> Result

# Term/factory mode
result = prog.invoke(inputs: dict[str, Any] | None = None) -> Result
```

Mode mismatch (calling `message=` on a term-mode Program or `inputs=` on an FSM-mode Program) raises `ProgramModeError`.

#### `.explain(...)`

```python
explanation = prog.explain(
    *,
    inputs: dict[str, Any] | None = None,
    n: int | None = None,
    K: int | None = None,
    plan_kwargs: dict | None = None,
) -> ExplainOutput
```
Static analysis: walks the AST and returns one `Plan` per discovered `Fix` subtree (when both `n` and `K` are supplied), all `Leaf` schemas, and a string rendering of the term skeleton. Zero LLM calls.

#### Removed at 0.7.0

`Program.register_handler(h)`, `Program.run(**env)`, and `Program.converse(msg, conversation_id)` were removed at the I5 epoch closure. Migrate to:

- `Program(handlers=[h1, h2, ...])` at construction (replaces `.register_handler`)
- `Program.invoke(inputs={...}).value` (replaces `.run`)
- `Program.invoke(message=..., conversation_id=...).value` (replaces `.converse`)

See [`migration_0.6_to_0.7.md`](migration_0.6_to_0.7.md) for the detailed walkthrough.

### `Result`

```python
@dataclass(frozen=True)
class Result:
    value: Any = None
    conversation_id: str | None = None
    plan: Plan | None = None
    leaf_calls: int = 0
    oracle_calls: int = 0
    explain: ExplainOutput | None = None
```

Uniform return type for every `Program.invoke` mode. In FSM mode, `value` is the response string and `conversation_id` is the (auto-started or echoed) session id. In term/factory mode, `value` is whatever the term reduces to; `leaf_calls` and `oracle_calls` come from the executor.

### `ExplainOutput`

```python
@dataclass(frozen=True)
class ExplainOutput:
    plans: list[Plan] = []          # one per Fix subtree (when n, K supplied)
    leaf_schemas: dict[str, type | None] = {}
    ast_shape: str = ""             # indented term skeleton
```

`leaf_schemas` is keyed by synthesised `leaf_NNN_<template-prefix>` ids.

### `ProgramModeError`

Subclass of `FSMError`. Raised when `.invoke(...)` arguments don't match the Program's mode.

---

## L3 — AUTHOR

Term producers: stdlib factories and the FSM compiler. All factory names end in `*_term`.

### Stdlib factories

#### Agents (4)

```python
from fsm_llm import react_term, rewoo_term, reflexion_term, memory_term
```

| Factory | Shape | Reference |
|---------|-------|-----------|
| `react_term(*, decide_prompt, synth_prompt, ...)` | 2-leaf decide → tool → synth | `stdlib/agents/lam_factories.py` |
| `rewoo_term(*, plan_prompt, work_prompt, solve_prompt, ...)` | 3-leaf plan → work → solve | same |
| `reflexion_term(*, attempt_prompt, evaluate_prompt, reflect_prompt, retry_prompt, max_attempts)` | `Fix`-bounded retry | same |
| `memory_term(*, retrieve_prompt, answer_prompt, ...)` | retrieve → answer | same |

See `src/fsm_llm/stdlib/agents/CLAUDE.md` for the full surface (12 class-based agents coexist with the 4 factories).

#### Reasoning (11)

```python
from fsm_llm import (
    analytical_term, deductive_term, inductive_term, abductive_term,
    analogical_term, creative_term, critical_term, hybrid_term,
    calculator_term, classifier_term, solve_term,
)
```

Each is a let-chain of leaves modeling a reasoning strategy. `solve_term` is the orchestrator; `classifier_term` routes to a strategy at runtime.

#### Workflows (5)

```python
from fsm_llm import linear_term, branch_term, switch_term, parallel_term, retry_term
```

| Factory | Shape |
|---------|-------|
| `linear_term(*pairs)` | right-nested `Let` chain |
| `branch_term(condition_var, then_term, else_term)` | `Case` with `"true"` / default |
| `switch_term(scrutinee_var, branches, default=None)` | full `Case` |
| `parallel_term(*pairs)` | `Combinator(CONCAT, [pairs])` |
| `retry_term(body, *, max_attempts, success_predicate, ...)` | `Fix`-bounded |

#### Long-context (6)

```python
from fsm_llm import (
    niah_term, aggregate_term, pairwise_term,
    multi_hop_term, multi_hop_dynamic_term, niah_padded_term,
)
```

| Factory | Cost (Theorem 2) |
|---------|------------------|
| `niah_term(question, *, tau, k, reduce_op_name='best_answer')` | `k^d` strict on τ·k^d-aligned |
| `aggregate_term(question, *, tau, k)` | `k^d` strict |
| `pairwise_term(question, *, tau, k, reduce_op_name)` | `k^d` (compare_op) or `2·k^d − 1` (oracle_compare_op) |
| `multi_hop_term(question, *, hops, tau, k)` | `hops · k^d` |
| `multi_hop_dynamic_term(question, *, max_hops, tau, k, confidence_gate)` | `actual_hops · k^d` (per-actual); `≤ max_hops · k^d` (loose) |
| `niah_padded_term(question, *, tau, k, pad_char=' ')` | `k^d` against `aligned_size(n, τ, k)` |

The bare names (`niah`, `aggregate`, `pairwise`, `multi_hop`, `multi_hop_dynamic`, `niah_padded`) were removed at 0.7.0 — accessing them raises `AttributeError`. Use the `*_term` canonical names exclusively.

### FSM compiler

```python
from fsm_llm import compile_fsm, compile_fsm_cached

term = compile_fsm(fsm_def)                          # one-shot
term = compile_fsm_cached(fsm_def, fsm_id="my_fsm")  # lru_cache(64) keyed on (fsm_id, json)
```

`compile_fsm_cached` is the canonical path used by `FSMManager` and transitively by `Program.from_fsm`. Inspect the cache via `_compile_fsm_by_id.cache_info()` for `(hits, misses, currsize, maxsize=64)`.

---

## L2 — COMPOSE

Pure Term→Term transforms and construction-time data bundles.

### Handlers

```python
from fsm_llm import (
    compose, Handler, HandlerBuilder, HandlerTiming, HandlerSystem,
    FSMHandler, BaseHandler, create_handler,
)
```

`HandlerTiming` enumerates 8 timing points (`START_CONVERSATION`, `PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `END_CONVERSATION`, `ERROR`). `PRE_PROCESSING` and `POST_PROCESSING` are spliced into the AST by `compose`; the other six dispatch host-side. See [`handlers.md`](handlers.md) for the full lifecycle and a builder cookbook.

`Handler` is an alias for `FSMHandler` (the Protocol). `BaseHandler` is the canonical base class. `LambdaHandler` (returned by `HandlerBuilder.do(...)`) is a public concrete implementation.

```python
compose(term: Term, handlers: list[FSMHandler]) -> Term
```
Pure AST→AST splice. Identity when `handlers in (None, [])`.

### Profiles

```python
from fsm_llm import (
    HarnessProfile, ProviderProfile,
    register_harness_profile, register_provider_profile,
    get_harness_profile, get_provider_profile,
)
```

```python
class HarnessProfile(BaseModel, frozen=True):
    system_prompt_base: str | None = None
    system_prompt_custom: str | None = None
    system_prompt_suffix: str | None = None
    leaf_template_overrides: dict[str, str] = {}
    provider_profile_name: str | None = None

class ProviderProfile(BaseModel, frozen=True):
    extra_kwargs: dict[str, Any] = {}
```

Registries are thread-safe (`RLock`-backed) and support `provider:model` → `provider` fallback on lookup. Application is **apply-once at construction**: `apply_to_term(term, profile)` rewrites only `Leaf.template` strings via `Term.model_copy` — no AST schema changes, no Leaf cardinality changes — so **Theorem-2 strict equality is preserved**.

---

## L1 — REDUCE

The typed λ-substrate. Use directly when you want kernel-level control.

### AST nodes (frozen Pydantic)

```python
from fsm_llm import Term, Var, Abs, App, Let, Case, Combinator, CombinatorOp, Fix, Leaf, is_term
```

| Node | Fields | Role |
|------|--------|------|
| `Var(name)` | `name: str` | variable reference |
| `Abs(param, body)` | | λ-abstraction |
| `App(fn, arg)` | | application (note: `fn`, not `func`) |
| `Let(name, value, body)` | | eager let-binding (sequencing primitive) |
| `Case(scrutinee, branches, default?)` | `branches: dict[str, Term]` | finite discrimination on `str(value)` |
| `Combinator(op, args)` | `op: CombinatorOp; args: list[Term]` | closed-set ops (SPLIT/PEEK/MAP/FILTER/REDUCE/CONCAT/CROSS/HOST_CALL) |
| `Fix(body)` | `body: Abs` | bounded recursion |
| `Leaf(template, input_vars, schema_ref=None, streaming=False)` | | the only node that invokes `𝓜` |

`is_term(obj)` is a duck-type check for validators.

### DSL builders

```python
from fsm_llm import var, abs_, app, let_, case_, fix, leaf, split, peek, fmap, ffilter, reduce_, concat, cross, host_call
```

Closures over no Python state. All return immutable AST nodes. See `runtime/dsl.py` for the full signatures.

### Combinators

```python
from fsm_llm import ReduceOp, BUILTIN_OPS
```

`ReduceOp` is a closed `str`/`Enum`. `BUILTIN_OPS` is **architecturally closed** — new ops bind through env at the call site (factory pattern), not by extending the registry.

### Planner

```python
from fsm_llm import PlanInputs, Plan, plan
```

Pure function `plan(PlanInputs) -> Plan`. Zero LLM calls. Closed-form per [`lambda.md`](lambda.md) Theorems 2 & 4.

```python
@dataclass class PlanInputs:
    n: int
    K: int = 8192
    tau: int = 256
    k: int = 2
    alpha: float = 1.0
    leaf_accuracy: float = 0.99
    combine_accuracy: float = 1.0
    reduce_calls_per_node: int = 0
    fmap_leaf_count: int = 0

@dataclass class Plan:
    k_star: int
    tau_star: int
    depth: int
    leaf_calls: int           # k^d
    reduce_calls: int
    predicted_calls: int      # leaf_calls + reduce_calls
    predicted_cost: float
    accuracy_floor: float
    composition_op: ReduceOp
```

**Theorem-2 contract**: for τ·k^d-aligned input, `Executor.run(term, env).oracle_calls == plan(...).predicted_calls`. Strict equality.

### Oracle

```python
from fsm_llm import Oracle, LiteLLMOracle
```

```python
class Oracle(Protocol):
    def invoke(self, prompt: str, schema: type[BaseModel] | None = None,
               *, model_override: str | None = None, env: dict | None = None) -> Any: ...
    def tokenize(self, text: str) -> int: ...
    def context_window(self) -> int: ...

class LiteLLMOracle:
    def __init__(self, llm: LiteLLMInterface, *, max_tokens: int = 8192): ...
```

#### D-008 caveat — `LiteLLMOracle._invoke_structured` bypasses subclass overrides

When a `Leaf` carries a `schema_ref`, `LiteLLMOracle._invoke_structured` calls `litellm.completion` **directly** via `self._llm.model`, `self._llm.max_tokens`, `self._llm.kwargs` — it does **not** call `self._llm.generate_response`. This bypass exists for D-011 reasons: the ABC's response-format wrapper breaks structured output for small Ollama models (e.g. qwen3.5:4b).

**Implication.** A subclass of `LiteLLMInterface` overriding `generate_response` (e.g. for an in-house provider, retry policy, or prompt scrubber) will **not** be invoked on Executor-driven structured Leaf calls.

**Escape hatch.** Implement the `Oracle` Protocol directly and pass via `Program.from_*(oracle=my_custom_oracle)`. Custom oracles bypass `LiteLLMOracle` and `_invoke_structured` entirely. The Protocol surface is small: `invoke`, `tokenize`, `context_window`, plus optional `StreamingOracle.invoke_stream`.

### Executor

```python
from fsm_llm import Executor

ex = Executor(oracle=LiteLLMOracle(LiteLLMInterface(model="...")), max_fix_depth=64)
result = ex.run(term, env={"document": doc})
ex.oracle_calls   # integer counter
ex.cost           # CostAccumulator
```

`Executor.peer_env` (constructor kwarg) lets callers pass extra env bindings into a host orchestrator without runner-attribute mutation. `Executor._eval(term, env, *, _fix_depth=0)` is the single-step interpreter; host-callable orchestrators that need to call sub-terms WITHOUT resetting `oracle_calls` should call `_eval` directly (`run` resets the counter).

### Cost telemetry

```python
from fsm_llm import LeafCall, CostAccumulator
```

```python
@dataclass class LeafCall:
    leaf_id: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    schema_name: str | None

class CostAccumulator:
    def add(self, call: LeafCall) -> None: ...
    def total_tokens(self) -> int: ...
    def total_cost(self) -> float: ...
    def rows(self) -> list[LeafCall]: ...
```

### Kernel exceptions

```python
LambdaError
├── ASTConstructionError    # AST built incorrectly (e.g. Fix body not Abs)
├── TerminationError        # depth limit, combinator that fails to reduce rank
├── PlanningError           # invalid PlanInputs, k > K, etc.
└── OracleError             # oracle invocation failed (network, schema, parse)
```

---

## Legacy

The full FSM dialog front-end. Use `Program.from_fsm` for new code; these names remain for back-compat.

### Removed surfaces (I5 epoch closed at 0.7.0)

The deprecation-warning surfaces from 0.6.x are hard removals at 0.7.0. Each row below now raises `AttributeError` / `ImportError`:

| Removed name | Replacement |
|--------------|-------------|
| `from fsm_llm import API` | `Program.from_fsm` (or `from fsm_llm.dialog.api import API`) |
| `Program.run(**env)` | `Program.invoke(inputs=env).value` |
| `Program.converse(msg, conv_id)` | `Program.invoke(message=msg, conversation_id=conv_id).value` |
| `Program.register_handler(h)` | `handlers=[...]` at construction |
| `import fsm_llm_{reasoning,workflows,agents}` | `from fsm_llm.stdlib.<x> import ...` |
| Long-context bare names (`niah`, `aggregate`, …, `niah_padded`) | `*_term` forms |
| `from fsm_llm import LiteLLMInterface` | `from fsm_llm.runtime._litellm import LiteLLMInterface` |
| `quick_start("bot.json")` | `Program.from_fsm("bot.json")` |

See [`migration_0.6_to_0.7.md`](migration_0.6_to_0.7.md) for before/after code per row.

### `API` (canonical home `fsm_llm.dialog.api.API`)

User-facing FSM entry point.

```python
# Construction
API.from_file(path: str, **kwargs) -> API
API.from_definition(fsm_def: FSMDefinition, **kwargs) -> API

# Conversation
api.start_conversation(initial_context: dict | None = None) -> tuple[str, str]
api.converse(message: str, conversation_id: str) -> str
api.end_conversation(conversation_id: str) -> None
api.has_conversation_ended(conversation_id: str) -> bool

# Queries
api.get_data(conversation_id: str) -> dict
api.get_current_state(conversation_id: str) -> str
api.get_conversation_history(conversation_id: str) -> list[dict]
api.list_active_conversations() -> list[str]

# FSM stacking
api.push_fsm(conversation_id: str, new_fsm: FSMDefinition) -> None
api.pop_fsm(conversation_id: str, merge_strategy: ContextMergeStrategy = ...) -> None

# Handlers
api.register_handler(handler: FSMHandler) -> None
api.create_handler(name: str) -> HandlerBuilder
```

### `FSMManager`

```python
from fsm_llm import FSMManager
```
Per-conversation orchestration with `RLock` thread-safety. The compiled-term cache lives in `compile_fsm_cached`.

### `MessagePipeline`

Internal post-M2 S11. Lives at `fsm_llm.dialog.turn`.

### `LiteLLMInterface`, `LLMInterface`

```python
from fsm_llm.runtime._litellm import LiteLLMInterface, LLMInterface
```
The provider adapter. Subclassable, but see the D-008 caveat above.

### Definitions, classifiers, transitions

```python
from fsm_llm.dialog.definitions import (
    FSMDefinition, FSMContext, FSMInstance, State, Transition,
    TransitionCondition, Conversation, ClassificationSchema,
    ClassificationResult, FieldExtractionConfig, ClassificationExtractionConfig,
    # … and all paired Request/Response models
)
from fsm_llm.dialog.classification import Classifier, HierarchicalClassifier, IntentRouter
from fsm_llm.dialog.transition_evaluator import TransitionEvaluator, TransitionEvaluatorConfig
from fsm_llm.dialog.session import SessionStore, FileSessionStore, SessionState
```

### Memory and context

```python
from fsm_llm import WorkingMemory, ContextCompactor
```

### Prompt builders

```python
from fsm_llm import (
    DataExtractionPromptBuilder, ResponseGenerationPromptBuilder,
    FieldExtractionPromptBuilder, ClassificationPromptConfig,
    DataExtractionPromptConfig, ResponsePromptConfig, FieldExtractionPromptConfig,
    build_classification_json_schema, build_classification_system_prompt,
)
```

### Exceptions

```
FSMError (top of dialog hierarchy)
├── StateNotFoundError
├── InvalidTransitionError
├── LLMResponseError
├── TransitionEvaluationError
├── ClassificationError
│   ├── SchemaValidationError
│   └── ClassificationResponseError
├── HandlerSystemError
│   └── HandlerExecutionError
└── ProgramModeError
```

Stdlib exceptions follow the same pattern in `fsm_llm.stdlib.{reasoning,workflows,agents}.exceptions`.

---

## CLI surface

Five console scripts ship with the package:

```bash
fsm-llm --mode {run,validate,visualize} --fsm <path.json> [--style ...]
fsm-llm-validate  --fsm <path.json>
fsm-llm-visualize --fsm <path.json>
fsm-llm-monitor                          # web dashboard (requires fsm-llm[monitor])
fsm-llm-meta                             # interactive artifact builder
```

## Versioning and stability

`fsm_llm.__version__` is `"0.7.0"`. Public-API stability is governed by the deprecation calendar in [`lambda_fsm_merge.md`](lambda_fsm_merge.md) §3:

- **R13 epoch (removed at 0.6.0)** — the `fsm_llm.{api,fsm,pipeline,prompts,definitions,llm,session,classification,transition_evaluator,lam}` shim modules have been deleted.
- **I5 epoch (closed at 0.7.0)** — `Program.run`, `Program.converse`, `Program.register_handler`, `from fsm_llm import API`, `import fsm_llm_{reasoning,workflows,agents}`, and long-context bare names are all removed; accessing them raises `AttributeError` / `ImportError`. Plus the D-009 formalisation: `from fsm_llm import LiteLLMInterface` is private-only via the runtime adapter, and the undocumented `quick_start()` helper was deleted. See [`migration_0.6_to_0.7.md`](migration_0.6_to_0.7.md).

The deprecation-calendar test (`tests/test_fsm_llm/test_deprecation_calendar.py`) flips its assertions automatically per `__version__` thresholds — at 0.7.0+ the calendar asserts every removed row raises the right exception.
