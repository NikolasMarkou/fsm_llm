# API Reference

Complete API documentation for FSM-LLM. The public surface is layered (L1–L4 plus a Legacy block) per `docs/lambda_fsm_merge.md` §4. New code should reach for the L4 verb (`Program`).

## L4 — INVOKE: `Program`, `Result`, `ExplainOutput`, `ProgramModeError`

### `Program`

The unified facade. Mode is fixed at construction.

```python
from fsm_llm import Program

# FSM mode — message-driven, persistent per-turn state
Program.from_fsm(
    fsm_definition,                       # FSMDefinition | dict | str (path or JSON)
    *,
    oracle: Oracle | None = None,         # built lazily from llm_kwargs if absent
    session: SessionStore | None = None,
    handlers: list[FSMHandler] | None = None,
    **api_kwargs,                         # model, temperature, max_tokens, …
) -> Program

# Term mode — pre-authored λ-term, inputs-driven
Program.from_term(
    term: Term,
    *,
    oracle: Oracle | None = None,
    handlers: list[Handler] | None = None,
    **llm_kwargs,
) -> Program

# Factory mode — invoke a stdlib factory and wrap its term
Program.from_factory(
    factory: Callable[..., Term],
    factory_args: tuple = (),
    factory_kwargs: dict | None = None,
    *,
    oracle: Oracle | None = None,
    **llm_kwargs,
) -> Program
```

The verb:

```python
# FSM mode
result = program.invoke(message="hi", conversation_id=None, *, explain=False)

# Term / factory mode
result = program.invoke(inputs={"x": 1, "y": 2}, *, explain=False)
```

`Program.invoke(message=...)` on a term-mode Program raises `ProgramModeError` with a redirect; vice versa.

### `Result`

```python
@dataclass(frozen=True)
class Result:
    value: Any                         # FSM: str reply; term/factory: term reduction
    conversation_id: str | None = None # FSM mode only
    plan: Plan | None = None           # populated when explain=True
    leaf_calls: int = 0
    oracle_calls: int = 0              # equals plan.predicted_calls under Theorem-2
    explain: ExplainOutput | None = None
```

### `ExplainOutput`

```python
@dataclass(frozen=True)
class ExplainOutput:
    plans: list[Plan]                  # one per Fix subtree
    leaf_schemas: dict[str, type | None]
    ast_shape: str                     # indented multi-line term skeleton
```

### `ProgramModeError`

`FSMError` subclass raised on `.invoke` mode mismatches. Its message points users at the correct verb shape.

## L3 — AUTHOR: `compile_fsm`, stdlib factories, raw DSL

### FSM compiler

```python
from fsm_llm import compile_fsm, compile_fsm_cached
term = compile_fsm(fsm_definition)                 # one-off compile
term = compile_fsm_cached(fsm_definition, fsm_id)  # lru_cache(maxsize=64), keyed on (fsm_id, JSON dump)
```

### Raw DSL

```python
from fsm_llm import (
    leaf, fix, let_, case_, var, abs_, app,
    split, fmap, ffilter, reduce_, concat, cross, peek,
)
```

`leaf(prompt: str, *, schema: type[BaseModel] | None = None, input_var: str | None = None, streaming: bool = False, schema_ref: str | None = None)` — produces a `Leaf` term.

`let_(name, value_term, body_term)` — local binding.

`fix(λ-loop)` — recursive fixed-point combinator (M5 long-context primitive).

`case_(scrutinee, branches)` — case analysis on a discriminated union.

`split`, `fmap`, `ffilter`, `reduce_`, `concat`, `cross`, `peek` — combinator-side operations on chunked content.

### Stdlib factories

| Subpackage | Factory | Brief |
|---|---|---|
| `stdlib.agents` | `react_term`, `rewoo_term`, `reflexion_term`, `memory_term` | λ-native agent patterns |
| `stdlib.reasoning` | `solve_term`, `classifier_term`, 11 strategy factories | reasoning chains |
| `stdlib.workflows` | `linear_term`, `branch_term`, `switch_term`, `parallel_term`, `retry_term` | workflow shapes |
| `stdlib.long_context` | `niah`, `aggregate`, `pairwise`, `multi_hop`, `niah_padded` | M5 closed-form-cost primitives |

## L2 — COMPOSE: `compose`, `Handler`, `HandlerTiming`, `HandlerBuilder`

### `compose(term, handlers)`

Pure AST→AST transform. Splices each handler into the term at its declared timing point. Idempotent for empty handler lists.

### `HandlerBuilder`

Fluent API:

| Method | Description |
|--------|-------------|
| `.at(*timings)` | Specify HandlerTiming values |
| `.on_state(*states)` | Execute only in these states |
| `.not_on_state(*states)` | Exclude these states |
| `.on_target_state(*states)` | Execute only when transitioning TO these states |
| `.not_on_target_state(*states)` | Exclude transitions TO these states |
| `.when_context_has(*keys)` | Require these context keys |
| `.when_keys_updated(*keys)` | Execute when these keys change |
| `.on_state_entry(*states)` | Shorthand: `.at(POST_TRANSITION).on_target_state()` |
| `.on_state_exit(*states)` | Shorthand: `.at(PRE_TRANSITION).on_state()` |
| `.on_context_update(*keys)` | Shorthand: `.at(CONTEXT_UPDATE).when_keys_updated()` |
| `.when(condition)` | Custom condition lambda |
| `.with_priority(n)` | Execution priority (lower runs first, default 100) |
| `.do(fn)` | Set handler function and build |

### `HandlerTiming` enum

`START_CONVERSATION`, `PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `END_CONVERSATION`, `ERROR`.

Two timings (`PRE/POST_PROCESSING`) are AST-side via `compose`. The other six remain host-side per `docs/lambda_fsm_merge.md` §8.

## L1 — REDUCE: `Term`, `Executor`, `Plan`, `Oracle`, …

### `Executor`

```python
from fsm_llm import Executor, LiteLLMOracle
from fsm_llm.runtime._litellm import LiteLLMInterface

ex = Executor(oracle=LiteLLMOracle(LiteLLMInterface(model="gpt-4o-mini")))
result = ex.run(term, env={"q": "What is photosynthesis?"}, *, stream=False)
print(ex.oracle_calls, ex.leaf_calls)
```

### `Oracle` Protocol

```python
class Oracle(Protocol):
    def invoke(self, prompt: str, *, env: dict | None = None,
               schema: type[BaseModel] | None = None,
               model_override: str | None = None) -> Any: ...
    def invoke_stream(self, prompt: str, *, user_message: str | None = None) -> Iterator[str]: ...
    def invoke_messages(self, messages: list[dict], *, call_type: str) -> Any: ...
    def invoke_field(self, request: FieldExtractionRequest) -> Any: ...
    def tokenize(self, text: str) -> list[int]: ...
```

`StreamingOracle` is a secondary Protocol for streaming-capable oracles; `Executor.run(stream=True)` `isinstance`-checks for it.

### `LiteLLMOracle`

Adapter wrapping any `LLMInterface` (the ABC in `fsm_llm.runtime._litellm`). Default `LiteLLMInterface` covers 100+ providers via litellm.

### Planner

```python
from fsm_llm import plan, PlanInputs

p = plan(PlanInputs(n=4096, tau=256, k=2))
print(p.predicted_calls)        # closed-form cost from AST shape
```

### `CostAccumulator`, `LeafCall`

Per-leaf telemetry record. Available on `Executor.cost_accumulator` after a run.

## Legacy block: `API`, `LLMInterface`, classifiers, …

Preserved silently in 0.5.x. Deprecation in 0.6.0; removal in 0.7.0 per the I5 calendar.

### `API` class

```python
from fsm_llm import API

api = API.from_file("path/to/fsm.json", model="gpt-4o-mini")
api = API.from_definition(fsm_dict, model="gpt-4o-mini")

conv_id, hello = api.start_conversation(initial_context={"key": "val"})
response = api.converse("user message", conv_id)
api.end_conversation(conv_id)
api.has_conversation_ended(conv_id)
api.close()

api.get_current_state(conv_id)
api.get_data(conv_id)
api.get_conversation_history(conv_id)
api.list_active_conversations()
api.update_context(conv_id, {"k": "v"})
api.cleanup_stale_conversations(max_idle_seconds=3600)

# FSM stacking
api.push_fsm(conv_id, new_fsm,
    context_to_pass={"step": "details"},
    shared_context_keys=["user_id"],
    preserve_history=True, inherit_context=True)
api.pop_fsm(conv_id, context_to_return={"complete": True}, merge_strategy="update")
api.get_stack_depth(conv_id)
```

### `LLMInterface`

```python
class LLMInterface(ABC):
    @abstractmethod
    def generate_response(self, request: ResponseGenerationRequest) -> ResponseGenerationResponse: ...
    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse: ...
    def generate_response_stream(self, request: ResponseGenerationRequest) -> Iterator[str]: ...
```

Custom subclasses are auto-wrapped in `LiteLLMOracle` when passed to `Program.from_fsm(llm=...)`. Caveat: `LiteLLMOracle._invoke_structured` bypasses `generate_response` for structured Leaves; subclasses needing custom behavior on every call should pass an `Oracle` directly via `oracle=`.

## Classification (built into core)

```python
from fsm_llm import Classifier, ClassificationSchema, IntentDefinition, IntentRouter

schema = ClassificationSchema(
    intents=[IntentDefinition(name="billing", description="Billing questions")],
    fallback_intent="general",
)
classifier = Classifier(schema, model="gpt-4o-mini")
result = classifier.classify("Where is my invoice?")
# result.intent, result.confidence, result.is_low_confidence

# Multi-intent
result = classifier.classify_multi("Check order and update billing")

# Hierarchical (two-stage domain → intent)
from fsm_llm import HierarchicalClassifier
h_classifier = HierarchicalClassifier(domains=[...], model="gpt-4o-mini")

# Intent routing
router = IntentRouter(schema)
router.register("billing", handle_billing)
response = router.route(message, classification_result)
```

## Stdlib (subpackages)

### `fsm_llm.stdlib.agents`

```python
from fsm_llm.stdlib.agents import create_agent, ReactAgent, tool, AgentConfig, HumanInTheLoop

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

agent = create_agent("react", model="gpt-4o-mini", tools=[search])
out = agent("task")    # out.answer, out.success, out.trace, out.structured_output

agent = ReactAgent(model="gpt-4o-mini", tools=[search],
                   config=AgentConfig(output_schema=MyPydanticModel))
```

12 patterns: `react`, `rewoo`, `reflexion`, `plan_execute`, `prompt_chain`, `self_consistency`, `debate`, `orchestrator`, `adapt`, `evaluator_optimizer`, `maker_checker`, `reasoning_react`.

### `fsm_llm.stdlib.reasoning`

```python
from fsm_llm.stdlib.reasoning import ReasoningEngine, ReasoningType
engine = ReasoningEngine(model="gpt-4o-mini")
solution, trace = engine.solve_problem("problem text", initial_context={})
```

11 strategies: `SIMPLE_CALCULATOR`, `ANALYTICAL`, `DEDUCTIVE`, `INDUCTIVE`, `CREATIVE`, `CRITICAL`, `HYBRID`, `ABDUCTIVE`, `ANALOGICAL`, plus the core orchestrator and classifier.

### `fsm_llm.stdlib.workflows`

```python
from fsm_llm.stdlib.workflows import WorkflowEngine, create_workflow, auto_step, condition_step

workflow = create_workflow("my_workflow", "My Workflow")
workflow.with_initial_step(auto_step("start", "Start", next_state="check"))
workflow.with_step(condition_step("check", "Check", condition=fn,
                                  true_state="ok", false_state="fail"))

engine = WorkflowEngine(max_concurrent_workflows=100)
engine.register_workflow(workflow)
instance_id = await engine.start_workflow("my_workflow", initial_context={})
await engine.advance_workflow(instance_id)
status = engine.get_workflow_status(instance_id)
await engine.shutdown()
```

11 step types.

### `fsm_llm.stdlib.long_context`

```python
from fsm_llm.stdlib.long_context import niah, aggregate, pairwise, multi_hop, niah_padded
```

Each has closed-form `predicted_calls` available via `plan(PlanInputs(...))`.

## Monitor

```python
from fsm_llm_monitor import MonitorBridge, EventCollector, create_server

collector = EventCollector(max_events=1000, max_logs=5000)
bridge = MonitorBridge()
bridge.connect(api, collector)
app = create_server(bridge, collector)
# Run with: uvicorn.run(app, host="127.0.0.1", port=8420)
# Or CLI: fsm-llm monitor
```

The monitor registers as observer handlers at all 8 timing points (priority 9999) and emits structured events. `span_schema_version` is `v1` (FSM-level events) at HEAD; `v2` (per-Leaf spans, emitted directly by the executor's `CostAccumulator`) is the next bump. See `src/fsm_llm_monitor/CLAUDE.md`.

## Exception Hierarchy

```
FSMError
├── StateNotFoundError
├── InvalidTransitionError
├── LLMResponseError
├── TransitionEvaluationError
├── ProgramModeError                     # Program.invoke mode mismatch
├── ClassificationError → SchemaValidationError, ClassificationResponseError
├── HandlerSystemError → HandlerExecutionError
├── ReasoningEngineError → ReasoningExecutionError, ReasoningClassificationError
├── WorkflowError → Definition, Step, Instance, Timeout, Validation, State, Event, Resource
└── AgentError → ToolExecution, ToolNotFound, ToolValidation, Budget, Approval,
                  Timeout, Evaluation, Decomposition
    └── MetaBuilderError → Builder, MetaValidation, Output

LambdaError                              # kernel
├── ASTConstructionError
├── TerminationError
├── PlanningError
└── OracleError

Exception
└── MonitorError → MonitorInitialization, MetricCollection, MonitorConnection
```

## Constants

```python
from fsm_llm.constants import (
    DEFAULT_LLM_MODEL,        # "ollama_chat/qwen3.5:4b"
    DEFAULT_TEMPERATURE,      # 0.5
    DEFAULT_MAX_HISTORY_SIZE, # 5
    DEFAULT_MAX_MESSAGE_LENGTH, # 1000
    DEFAULT_MAX_STACK_DEPTH,  # 10
)
```

## See also

- [`docs/lambda_fsm_merge.md`](lambda_fsm_merge.md) — the merge contract (canonical)
- [`docs/lambda.md`](lambda.md) — architectural thesis (Theorems 1–5)
- [`docs/architecture.md`](architecture.md) — execution model + layered architecture
- [`docs/handlers.md`](handlers.md) — handler development guide
