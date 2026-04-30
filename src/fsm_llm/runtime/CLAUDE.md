# fsm_llm.runtime — λ-Calculus Kernel (M1)

> **History.** Originally `fsm_llm.lam`; renamed to `fsm_llm.runtime` in plan v3 R4 (D-PLAN-08, D-PLAN-10). The `fsm_llm.lam` sys.modules shim was **removed in 0.6.0 (R13 epoch)** — `fsm_llm.runtime` is the canonical and only path.

The substrate. A typed λ-AST + Python builder DSL + closed combinator library + closed-form planner + β-reduction executor + oracle adapter + per-leaf cost accumulator. The FSM-JSON → λ compiler (`compile_fsm` / `compile_fsm_cached`) lives in `fsm_llm.dialog.compile_fsm`; the kernel is **closed against `dialog/`** as of D-001 (`plan_2026-04-27_5d8a038b`). `runtime/__init__.py` does not import anything from `dialog/`. Use `from fsm_llm.dialog import compile_fsm, compile_fsm_cached` — or the top-level convenience export `from fsm_llm import compile_fsm`.

Per `docs/lambda.md`: **every fsm_llm program is already a λ-term**. This package is the runtime that proves it. M1 ships the kernel; M2 the FSM compiler; M3 the stdlib of named factories on top.

**Purity invariant (post-D-001)**: `runtime/` imports **nothing** from `fsm_llm.dialog.*`. Adapters live here (`oracle.py` over `LiteLLMInterface`; `_litellm.py` is the moved `llm.py`) and at the dialog boundary (`fsm_llm.dialog.compile_fsm` over `FSMDefinition`). New code should import from `fsm_llm.runtime` (or `from fsm_llm import …` per the top-level convenience exports).

## File Map

```
runtime/                  # was lam/ pre-R4 (lam shim removed in 0.6.0)
├── ast.py                # Frozen Pydantic AST nodes
├── dsl.py                # Builder functions returning AST nodes
├── combinators.py        # ReduceOp enum + BUILTIN_OPS dict (closed registry)
├── executor.py           # β-reduction interpreter
├── planner.py            # plan() — closed-form (k*, τ*, d, predicted_calls, accuracy_floor)
├── oracle.py             # Oracle Protocol + LiteLLMOracle adapter (R3 env branch — see D-005)
├── _litellm.py           # LiteLLMInterface (was top-level llm.py; private-by-convention adapter)
├── _handlers_ast.py      # private — compose() + 8 _splice_<timing> functions + handler-runner env-binding helpers. Moved from handlers.py at 0.8.0; re-exported from there for back-compat (so `from fsm_llm.handlers import compose` and `from fsm_llm import compose` continue to work).
├── cost.py               # CostAccumulator + LeafCall — per-leaf cost telemetry
├── errors.py             # Exception hierarchy
├── constants.py          # K_DEFAULT, TAU_DEFAULT, depth limits
└── __init__.py           # exports — see below. Kernel is closed against `dialog/` as of D-001; `compile_fsm` / `compile_fsm_cached` are NOT re-exported here. Import them from `fsm_llm.dialog` or the top-level `fsm_llm` package.
```

`fsm_compile.py` itself moved to `fsm_llm/dialog/compile_fsm.py` in R4 step 21 (per `docs/lambda.md` §11 layout). Import the FSM compiler from `fsm_llm.dialog.compile_fsm` (or via the top-level convenience export `from fsm_llm import compile_fsm`).

## Public Surface

```python
from fsm_llm.runtime import (
    # AST nodes (frozen Pydantic; structural equality)
    Var, Abs, App, Let, Case, Combinator, CombinatorOp, Fix, Leaf, Term, is_term,

    # DSL builders
    var, abs_, app, let_, case_, fix, leaf,
    split, peek, fmap, ffilter, reduce_, concat, cross,

    # Combinators
    ReduceOp, BUILTIN_OPS,

    # Planner
    PlanInputs, Plan, plan,

    # Oracle
    Oracle, LiteLLMOracle,

    # Cost
    LeafCall, CostAccumulator,

    # Executor
    Executor,

    # Errors
    LambdaError, ASTConstructionError, TerminationError, PlanningError, OracleError,
)

# FSM compiler (M2) lives in dialog — kernel is closed against dialog/.
from fsm_llm.dialog import compile_fsm, compile_fsm_cached  # or: from fsm_llm import compile_fsm
```

## AST Nodes (`ast.py`)

| Node | Fields | Role |
|------|--------|------|
| `Var(name)` | `name: str` | Variable reference |
| `Abs(param, body)` | `param: str; body: Term` | Lambda abstraction |
| `App(fn, arg)` | `fn: Term; arg: Term` | Application |
| `Let(name, value, body)` | `name: str; value: Term; body: Term` | Eager let-binding (sequencing primitive) |
| `Case(scrutinee, branches, default?)` | `scrutinee: Term; branches: dict[str, Term]; default: Term \| None` | Finite discrimination on `str(value)` |
| `Combinator(op, args)` | `op: CombinatorOp; args: list[Term]` | Closed-set operations (SPLIT/PEEK/MAP/FILTER/REDUCE/CONCAT/CROSS) |
| `Fix(body)` | `body: Abs` | Bounded recursion (planner-bounded depth) |
| `Leaf(oracle, template, input_var, schema?, extra_input_vars?)` | see below | The **only** node that invokes 𝓜 |

`Leaf` fields:
- `oracle: OracleRef` — handle to a registered oracle (typically `default`)
- `template: PromptTemplate` — formattable string with `{var}` slots
- `input_var: str` — primary env-bound input
- `extra_input_vars: tuple[str, ...] = ()` — additional env bindings spliced into the prompt
- `schema: type[BaseModel] | None = None` — when set, structured Pydantic decode

`is_term(obj)` — duck-type check used by validators.

**Field naming gotcha**: `App.fn` (not `func`); `Combinator.args` (not `operands`); `Case.scrutinee` / `Case.branches` / `Case.default`. (Per LESSONS.md "AST attribute names matter" — past misnames cost fix attempts.)

## DSL Builders (`dsl.py`)

Thin constructors. All return immutable AST nodes; closures over no Python state.

```python
var("x")                                                   # → Var
abs_("x", body)                                            # → Abs
app(fn, arg)                                               # → App
let_("name", value, body)                                  # → Let
case_(scrutinee, {"a": term_a, "b": term_b}, default=t)   # → Case
fix(abs_("self", body))                                    # → Fix
leaf(prompt, *, input_var, schema=None, extra_input_vars=())  # → Leaf

# Combinator shortcuts (resolve to BUILTIN_OPS)
split(term, k)
peek(term, start, end)
fmap(fn, xs)
ffilter(pred, xs)
reduce_(op, xs, *, unit=None)            # op is a ReduceOp enum value
concat(xs, sep="")
cross(xs, ys)
```

## Combinators (`combinators.py`)

`ReduceOp` is a closed `str`/`Enum`. **`BUILTIN_OPS` is architecturally closed** (LESSONS line 95). New ops bind through env at the call site (factory pattern); they are **not** added to the registry. See e.g. `oracle_compare_op` in `stdlib/long_context/pairwise.py` for the canonical "new op via env" pattern.

### `HOST_CALL` op (R5, D-PLAN-02)

`CombinatorOp.HOST_CALL` is the one new closed-set op added in plan v1 R5 (handlers as AST transformers). A `Combinator(op=HOST_CALL, args=[Var(callable_name), *args])` evaluates by looking up `callable_name` in the env (resolves to a Python callable bound by the host), calling it with the evaluated `args`, and threading the result back. Unlike `Leaf`, `HOST_CALL` is **not** counted by the planner (`predicted_calls` ignores it) — it is intentionally invisible to the cost model because host-side semantics (handler invocation, generator returns, exception escapes) are out of scope for `plan(...)`. Adding `HOST_CALL` is the **only** closed-set extension in plan v1; the door is now closed again.

## Planner (`planner.py`)

Pure function from `PlanInputs` to `Plan`. Zero LLM calls. Closed-form per `docs/lambda.md` Theorems 2 & 4.

```python
@dataclass class PlanInputs:
    n: int                          # |document|
    tau: int                        # τ — leaf budget
    k: int                          # branching factor
    K: int = 8192                   # context budget
    alpha: float = 1.0              # cost slope
    cost_per_token: float = 0.0
    leaf_accuracy: float = 0.99
    combine_accuracy: float = 1.0   # 1.0 for decomposable tasks
    reduce_calls_per_node: int = 0  # 0 = pure reduce; >0 = oracle-mediated reduce
    fmap_leaf_count: int = 0        # R6.3 — additive Leaf-call count for fmap-over-N shapes

@dataclass class Plan:
    k_star: int                     # Theorem 4: defaults to 2
    tau_star: int
    depth: int
    leaf_calls: int                 # k^d
    reduce_calls: int               # (k^d - 1) * reduce_calls_per_node
    predicted_calls: int            # leaf_calls + reduce_calls
    predicted_cost: float
    accuracy_floor: float
    composition_op: ReduceOp
```

**Theorem-2 contract**: for a τ·k^d-aligned input, `Executor.run(term, env).oracle_calls == plan(...).predicted_calls`. Strict equality. See `evaluation/bench_long_context_*.json` for live evidence per (model × factory) cell.

## Oracle (`oracle.py`)

```python
class Oracle(Protocol):
    def invoke(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
        *,
        model_override: str | None = None,
        env: dict[str, Any] | None = None,
    ) -> Any: ...
    def tokenize(self, text: str) -> int: ...

class LiteLLMOracle:
    def __init__(self, llm: LiteLLMInterface, *, max_tokens: int = 8192): ...
```

`LiteLLMOracle._invoke_structured` bypasses `LiteLLMInterface.generate_response`'s outer-schema wrapper that breaks structured output for small Ollama models. See LESSONS.md "M4 Slice 4 — D-011 oracle 3-bug fix" for the canonical reasoning. Forces `temperature=0` on structured calls; synthesises `required = list(properties)` when Pydantic omits it.

**R3 env branch (D-005)**: when `env` is supplied, `prompt` is treated as a `str.format` template and substituted with `prompt.format(**env)` before the LLM call. Missing env keys raise `OracleError` (no silent passthrough). When `env` is `None` (the default), `prompt` flows through unchanged — this is the path taken by Executor-driven Leaf calls (planner-bounded paths always pre-substitute and pass `env=None`, so M1 behaviour is byte-identical for those callers). The env branch is forward-compat plumbing for stdlib factories that want to ship a `(template, env, schema)` triple as authored by the new template producers in `fsm_llm.prompts` (`*PromptBuilder.to_template_and_schema` + free `classification_template`). The 4 pipeline.py callbacks at HEAD remain on `LiteLLMInterface.{generate_response, extract_field}`; the callback collapse to `oracle.invoke` is deferred to R6 (per-state Leaf specialisation) per `D-PLAN-09-RESOLUTION-step14-narrowed`.

### Third-party `LLMInterface` subclass contract (M6d, `docs/lambda_fsm_merge.md` §5)

A user who has subclassed `LiteLLMInterface` to add an in-house provider, retry policy, prompt scrubber, etc. needs to know which of their overrides survive the `Program.from_fsm(llm=...)` → `LiteLLMOracle(llm)` wrap. The contract is:

- **Preserved**: subclass overrides of the ABC surface — `generate_response(request)` (`oracle.py:446`), `extract_field(request)` (`oracle.py:428`), `generate_response_stream(request)` (`oracle.py:341`). The `LiteLLMOracle` wrapper holds the subclass instance as `self._llm` and dispatches through the ABC methods on every non-structured call path.
- **NOT preserved (caveat)**: `LiteLLMOracle._invoke_structured` (`oracle.py:449-572`) bypasses `self._llm.generate_response` entirely. It calls `litellm.completion` directly via `getattr(self._llm, "model", ...)`, `getattr(self._llm, "max_tokens", ...)`, `getattr(self._llm, "kwargs", {})` — only static attributes are read. A subclass that overrides `generate_response` to inject provider-side logic does NOT get that logic invoked on Executor-driven structured Leaf calls (any Leaf with `schema_ref != None`). This bypass exists for D-011 reasons — the ABC's response-format wrapper breaks structured output for small Ollama models. See LESSONS.md "M4 Slice 4 — D-011 oracle 3-bug fix".

**Escape hatch**: subclasses that need their structured-call logic preserved should not subclass `LiteLLMInterface` — they should implement the `Oracle` Protocol directly and pass via `Program.from_fsm(oracle=my_custom_oracle)`. The Protocol surface is small (`invoke`, `tokenize`, plus the secondary `StreamingOracle` / `MessagesOracle` Protocols if needed); custom oracles bypass `LiteLLMOracle` and `_invoke_structured` entirely.

## Executor (`executor.py`)

β-reduction interpreter with depth limits and per-leaf cost tracking.

```python
ex = Executor(oracle=LiteLLMOracle(LiteLLMInterface(model)), max_fix_depth=64)
result = ex.run(term, env={"document": doc})
ex.oracle_calls   # integer counter — successful invocations
ex.cost           # CostAccumulator — per-leaf cost rows
```

Internal: `Executor._eval(term, env, *, _fix_depth=0)` is the single-step interpreter. Host-callable orchestrators that need to call sub-terms WITHOUT resetting `oracle_calls` should call `_eval` directly (`run` resets the counter). Canonical precedent: `make_dynamic_hop_runner` in `stdlib/long_context/multi_hop.py` (LESSONS.md "Host-callable orchestrator with `_eval` bypass").

`Executor.peer_env` (constructor kwarg) lets callers pass extra env bindings into a host orchestrator without runner-attribute mutation.

## Cost (`cost.py`)

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

## FSM Compiler (`fsm_llm.dialog.compile_fsm`, M2 + R2 cache)

```python
from fsm_llm.dialog import compile_fsm, compile_fsm_cached
# or, equivalently, the top-level convenience export:
# from fsm_llm import compile_fsm

# Pure compile (no cache) — used for one-off / no-reuse paths.
term = compile_fsm(fsm_def)

# Memoised compile (R2) — kernel-level lru_cache(maxsize=64). Use this
# from FSMManager, Program.from_fsm (transitively), and any stdlib
# script caller that compiles the same FSM more than once.
term = compile_fsm_cached(fsm_def, fsm_id="my_fsm")
# fsm_id is optional; when None, derived as f"defn_{sha256(json)[:8]}".
```

Output shape: top-level `Case` on `state_id`; each branch is `Let("c'", extract_leaf, Let("s'", transition_case, Let("o", respond_leaf, ...)))`. Handler hooks compose at the appropriate `Let` boundary per `docs/lambda.md` §6.3.

**R2 cache key (D-PLAN-07, D-002)**: `(fsm_id, fsm.model_dump_json())`. The JSON is the content fingerprint and the actual identity; `fsm_id` is along for log/telemetry coherence. Two callers with the same JSON but different `fsm_id` strings get independent cache slots — intentional, so bench/log telemetry stays per-source. Inspect via `_compile_fsm_by_id.cache_info()` (`hits`, `misses`, `currsize`, `maxsize=64`).

**Reserved env names**: see `RESERVED_VARS: frozenset[str]` exported from this module — names the executor binds in env (user, current state, context, etc.) Tests assert closure on this set so the rewrite milestone stays drift-free.

## Errors (`errors.py`)

```
LambdaError
├── ASTConstructionError    # AST built incorrectly (e.g. Fix body not Abs)
├── TerminationError        # depth limit, combinator that fails to reduce rank
├── PlanningError           # invalid PlanInputs, k > K, etc.
└── OracleError             # oracle invocation failed (network, schema, parse)
```

## Constants (`constants.py`)

- `K_DEFAULT = 8192` — default token-context budget
- `TAU_DEFAULT = 256` — default leaf chunk size
- `MAX_FIX_DEPTH = 64` — hard depth limit on Fix
- `DEFAULT_REDUCE_CALLS_PER_NODE = 0`

## Testing

```bash
pytest tests/test_fsm_llm_lam/         # Kernel unit tests
pytest tests/test_fsm_llm_long_context # M5 factories — exercise Executor + Planner end-to-end
```

**Theorem-2 unit-test pattern**: build the term via the appropriate factory or DSL, run the executor with a scripted oracle (no LLM), assert `ex.oracle_calls == plan(...).predicted_calls`. Live `@pytest.mark.real_llm` smokes assert the same equality on `ollama_chat/qwen3.5:4b`. Bench scorecards under `evaluation/` capture (model × factory) cells.

## Related Subpackages

- **`fsm_llm.stdlib`** — named λ-term factories built on this kernel (M3+).
- **`fsm_llm.dialog`** — FSM dialog surface: `API`, `FSMManager`, `MessagePipeline` (in `dialog/turn.py`), classifiers, `TransitionEvaluator`, prompt builders, definitions, sessions, plus `compile_fsm` / `compile_fsm_cached`. Wraps `Executor` for Category-A FSM programs.
- **`fsm_llm.handlers`** — composes hooks into compiled λ-terms.

## Removed back-compat shims (0.6.0 / R13 epoch)

Plan v3 R4 reorganised the package per `docs/lambda.md` §11; the old import paths were kept as sys.modules shims through 0.5.x and **removed in 0.6.0**. The mapping for reference (do **not** import the left-hand paths — they no longer exist):

| Old path (gone)                                                                                                                                                     | Canonical home (use this)                              |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| `fsm_llm.lam`, `fsm_llm.lam.<sub>` (entire package)                                                                                                                 | `fsm_llm.runtime`, `fsm_llm.runtime.<sub>`             |
| `fsm_llm.lam.fsm_compile` / `from fsm_llm.runtime import compile_fsm`                                                                                               | `fsm_llm.dialog.compile_fsm` (or `from fsm_llm import compile_fsm`) |
| `fsm_llm.api`, `fsm_llm.fsm`, `fsm_llm.pipeline`, `fsm_llm.prompts`, `fsm_llm.classification`, `fsm_llm.transition_evaluator`, `fsm_llm.definitions`, `fsm_llm.session` | `fsm_llm.dialog.<name>` (note: `pipeline.py` was renamed to `turn.py` in R13) |
| `fsm_llm.llm`                                                                                                                                                       | `fsm_llm.runtime._litellm`                             |

Top-level convenience exports remain: `from fsm_llm import API, FSMManager, LiteLLMInterface, Program, Executor, Term, leaf, compile_fsm, ...` keeps working unchanged.
