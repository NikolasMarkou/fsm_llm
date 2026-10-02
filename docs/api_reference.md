# API Reference

> Covers FSM-LLM v0.11.0

Complete API documentation for FSM-LLM and its extension subpackages.

## API Class (`fsm_llm.API`)

Main interface for working with FSM-LLM.

### Constructor

```python
API(
    fsm_definition: FSMDefinition | dict | str,
    llm_interface: LLMInterface | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_history_size: int = 5,
    max_message_length: int = 1000,
    handlers: list[FSMHandler] | None = None,
    handler_error_mode: str = "continue",
    transition_config: TransitionEvaluatorConfig | None = None,
    session_store: SessionStore | None = None,
    handler_timeout: float | None = None,
    max_fsm_cache_size: int = 64,
    **llm_kwargs,
)
```

`handler_timeout` (seconds, `None` disables it) goes to the API's single `HandlerSystem`, so the cap of 4 still-running timed-out handler threads (`constants.MAX_TIMED_HANDLER_STRAGGLERS`) is shared by every conversation of that `API`. `max_fsm_cache_size` bounds `FSMManager`'s FSM definition cache and must be at least 1 (`ValueError` otherwise). The prompt builders and the FSM loader are not configurable through `API`; construct `FSMManager` directly for those.

With `llm_interface`, the interface owns every LLM setting: `API` raises `ValueError` when `api_key`, a non-None `temperature` or `max_tokens`, a `model` other than the interface's own, or any `**llm_kwargs` are given beside it. It also raises `ValueError` (in `__init__` and `push_fsm`) when a definition's `classification_extractions` would classify through an interface that does not implement `complete`. `fsm_id` is `fsm_<name>_<8 hex>` from a sha256 of `model_dump(exclude_defaults=True)`, so only a change of the definition document changes it.

### Factory Methods

```python
api = API.from_file("path/to/fsm.json", model="gpt-4o-mini")
api = API.from_definition(fsm_dict, model="gpt-4o-mini")
```

`from_definition(fsm_definition=None, *, definition=None, **kwargs)` takes the definition positionally or as `fsm_definition=` / `definition=`; give exactly one.

### Conversation Lifecycle

```python
conv_id, response = api.start_conversation(initial_context={"key": "val"})
response = api.converse("user message", conv_id)
for chunk in api.converse_stream("user message", conv_id):  # -> Iterator[str]
    print(chunk, end="", flush=True)
api.end_conversation(conv_id)
api.has_conversation_ended(conv_id)  # -> bool
api.close()                          # cleanup all resources
```

A state with empty `response_instructions` is silent: it makes no Pass-2 LLM call, `start_conversation` and `converse` return `""` for it, a stream yields nothing, and nothing is added to the history. A `converse` turn still records the user message.

### Steps Without a User Message

```python
result = api.advance(conv_id)            # -> AdvanceResult
result.state_before, result.state_after  # state ids
result.transition_outcome                # DETERMINISTIC, AMBIGUOUS or BLOCKED
result.response                          # reply text, or None for a silent state
result.ended                             # True when state_after is terminal

for chunk in api.advance_stream(conv_id):  # same step, reply streamed
    print(chunk, end="")

results = api.run_until_terminal(        # -> tuple[AdvanceResult, ...]
    conv_id, max_steps=20, max_seconds=60.0,
    before_step=lambda n: None,          # called before step n (1, 2, ...)
    seconds_exempt_states=(),            # state ids that may run past max_seconds
)
for chunk in api.run_until_terminal_stream(conv_id, max_steps=20):
    print(chunk, end="")
```

`advance` runs the same turn as `converse` (one turn at a time, rollback on failure, every handler timing, extraction, transition evaluation, Pass 2 from the post-transition state, session auto-save) with no user message: nothing is added to the history for a user, and the extraction, classification and reply prompts use their no-message wording. The bulk extraction prompt is built from the state's scoped, security-filtered context and fills only unset keys. The LLM layer sends the fixed `NEUTRAL_USER_TURN` as the provider user turn; an empty `converse("")` message is a different fact and goes out as `EMPTY_USER_MESSAGE_TURN` (`"(empty message)"`). `advance` on a terminal state raises `FSMError`.

`run_until_terminal` checks, before each step: has the conversation (top of the FSM stack) ended, is the `max_seconds` budget spent, is the `max_steps` budget spent, then calls `before_step(n)` and runs one `advance`. Budgets are checked only between steps. A spent budget raises `RunBudgetExceededError` (`budget` `"steps"` or `"seconds"`, `limit`, `steps_done`); the steps already run are kept, read them from `get_conversation_history` and `get_current_state`. A conversation closed during the run (from `before_step` or another thread) ends the run normally. When the top of the stack is an ended pushed FSM (stack depth above 1), `before_step(n)` is called once first: that call is not a step and is not budget-gated, `n` is the next step's number (the step itself calls it again with the same `n`; tell the two apart with `has_conversation_ended`), and if the hook pops the frame the run continues on the parent, otherwise it returns. `seconds_exempt_states` lets a round whose current state id is in the set run after `max_seconds` is spent (the steps budget still applies); ids are checked against the running definition (`ValueError`). `max_steps` must be an int of at least 1 and `max_seconds` `None` or a positive number (`ValueError` otherwise). The agents, the harness and workflow `AgentStep`s run on these loops.

### Session Persistence

Pass a `session_store` to persist conversations across process restarts (state auto-saves after each `converse`):

```python
from fsm_llm import API, FileSessionStore

api = API.from_file("bot.json", model="gpt-4o-mini",
                    session_store=FileSessionStore("./sessions"))
api.save_session(conv_id)                              # -> None
state = api.load_session(session_id)                   # -> SessionState | None  (inspect only)
new_conv_id, state = api.restore_session(session_id)   # -> (conv_id, SessionState) | None  (resume)
```

`save_session` also stores the pipeline's extraction provenance (one digest per extracted key, not the values) in `SessionState.metadata["pipeline_extracted"]`, and `restore_session` re-seeds it, so a later-turn correction still lands after a restart. A handler-set or `update_context` value has no digest and is never overwritten. The map is also visible (as `_pipeline_extracted`) in `FSMManager.get_complete_conversation()['metadata']` and in the session file. A session file without the key restores an empty map.

### State & Data Queries

```python
api.get_current_state(conv_id)           # -> str
api.get_data(conv_id)                    # -> dict
api.get_conversation_history(conv_id)    # -> list[dict]
api.list_active_conversations()          # -> list[str]
api.update_context(conv_id, {"k": "v"})
api.cleanup_stale_conversations(max_idle_seconds=3600)  # -> list[str] (ends idle conversations)
api.get_llm_interface()                  # -> LLMInterface
```

`API.cleanup_stale_conversations` ends conversations idle longer than `max_idle_seconds`. It is unrelated to `FSMManager.prune_orphaned_locks()`, which only drops per-conversation locks that have no live instance and ends nothing. `FSMManager` has no `cleanup_stale_conversations`.

### FSM Stacking

```python
response = api.push_fsm(conv_id, new_fsm,
    context_to_pass={"step": "details"},
    shared_context_keys=["user_id"],
    preserve_history=True, inherit_context=True)
# push_fsm returns the sub-FSM's first message (a str), not an id. All calls take the ROOT id.
response = api.pop_fsm(conv_id,
    context_to_return={"complete": True},
    merge_strategy=ContextMergeStrategy.UPDATE)  # or "preserve"
# pop_fsm returns a resume message. Only the child's values for shared_context_keys plus
# return_context / context_to_return are merged back; the strategy only arbitrates collisions.
api.get_stack_depth(conv_id)           # -> int
api.get_sub_conversation_id(conv_id)   # -> str, an internal id for extensions; other API calls reject it
```

### Handler Registration

```python
api.register_handler(handler)
api.register_handlers([handler1, handler2])
builder = api.create_handler("MyHandler")
```

## HandlerBuilder

Fluent API returned by `api.create_handler()`:

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
| `.critical(value=True)` | Mark the handler critical: its failure raises `HandlerExecutionError` even in `error_mode="continue"` |
| `.do(fn)` | Set handler function and build |
| `.build()` | Build the handler from the current configuration (`.do(fn)` calls it) |

## Builders and `BuildError`

One convention for every builder (`APIBuilder`, `FSMManagerBuilder`, `HandlerBuilder`, `AgentGraphBuilder`, `ConfiguredAgentBuilder`, `HarnessAgentBuilder`, `workflows.WorkflowBuilder`, the meta artifact builders): `<Product>Builder`; `set_`, `add_` and `remove_` mutators only record and return the builder; `build()` is the only validator, copies what the builder owns, and raises `BuildError`. `BuildError` is exported from `fsm_llm`; it is an `FSMError` and a `ValueError`, `.errors` lists the problems, and the domain or constructor error that caused it is its `__cause__`. Objects you pass in (an `LLMInterface`, a session store, handlers, agents, registries) stay shared by reference. `HandlerBuilder` keeps its `at`/`on_state`/`do` vocabulary, and the meta artifact builders still refuse a bad call at call time with `BuilderError` (also a `BuildError`) and report warnings with `take_warnings()`. They also keep `update_state`, and their tools serialise calls per builder.

```python
from fsm_llm import APIBuilder, BuildError

try:
    api = (
        APIBuilder()
        .set_definition("bot.json")
        .set_model("ollama_chat/qwen3.5:4b")
        .set_temperature(0.3)
        .set_llm_option("seed", 7)       # open-ended litellm kwarg, never filtered
        .build()
    )
except BuildError as exc:
    print(exc.errors, exc.__cause__)
```

- `APIBuilder`: one setter per `API` parameter (`set_definition`, `set_llm_interface`, `set_model`, `set_api_key`, `set_temperature`, `set_max_tokens`, `set_llm_option(name, value)` (`build()` refuses a name that repeats a named `API` parameter and names the typed setter), `add_handler`, `set_handler_error_mode`, `set_transition_config`, `set_session_store`, `set_handler_timeout`, `set_max_history_size`, `set_max_message_length`, `set_max_fsm_cache_size`). Only set values reach `API(...)`, so `API`'s own rules (an `llm_interface` refuses other LLM settings) apply unchanged and come back as `BuildError`. No definition is a `BuildError`.
- `FSMManagerBuilder`: `set_fsm_loader`, `set_llm_interface`, the three prompt builders, `set_transition_evaluator`, `set_max_history_size`, `set_max_message_length`, `set_handler_system`, `set_handler_error_mode`, `set_max_fsm_cache_size`, `build() -> FSMManager`.
- `fsm_llm.agents.ConfiguredAgentBuilder`: `set_pattern`, `add_tool` (or `set_tool_registry`, not both), `set_config`, `set_config_option`, typed config setters (`set_model`, `set_temperature`, `set_max_tokens`, `set_max_iterations`, `set_timeout_seconds`), `set_system_prompt` (a `create_agent` argument, not an `AgentConfig` field), `set_hitl`, `set_option(name, value)`; `build()` calls `create_agent` on a shallow copy of the config (callables inside it stay shared). `set_option` passes any other `create_agent` keyword; `build()` refuses a name that repeats a named parameter (`pattern`, `tools`, `config`, `system_prompt`, ...) and names the typed setter to use.
- `fsm_llm.harness.HarnessAgentBuilder`: one `set_<parameter>` per `HarnessAgent` parameter and `set_api_option(name, value)`; `build() -> HarnessAgent` from a shallow copy of the config (callables inside it stay shared). `build()` refuses a `set_api_option` name that repeats a constructor parameter and names the typed setter to use.

## HandlerTiming Enum

`START_CONVERSATION`, `PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `END_CONVERSATION`, `ERROR`

## LLMInterface (`fsm_llm.llm`)

```python
class LLMInterface(ABC):
    @abstractmethod
    def generate_response(self, request: ResponseGenerationRequest) -> ResponseGenerationResponse: ...

    # The base class raises NotImplementedError; implement it for per-field extraction.
    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse: ...

    # Same default: called for every state with extraction_instructions.
    # Without it the pipeline logs "Bulk extraction fallback failed" and extracts
    # nothing for that free-text instruction.
    def extract_bulk_data(self, request: BulkExtractionRequest) -> DataExtractionResponse: ...

    def generate_response_stream(self, request: ResponseGenerationRequest) -> Iterator[str]: ...

    # Same default (NotImplementedError). Needed for classification and completion states.
    def complete(self, request: CompletionRequest) -> CompletionResponse: ...
```

`CompletionRequest{messages, tools=None, tool_choice=None, response_format=None, temperature=None, max_tokens=None, call_type="completion"}` (frozen, unknown fields refused): `tools` (OpenAI function schemas) and `response_format` never together, `tool_choice` only with `tools`; `None` temperature/max_tokens keep the interface's. `CompletionResponse{kind, text, calls}` (frozen): `kind` is `"calls"`, `"final"` or `"malformed"`, `calls` a tuple of `ModelToolCall{id, name, arguments: dict}`. A tool call without an id or name, or whose arguments are not a JSON object, and a provider "malformed tool call" error, give `kind="malformed"` with no calls. Every other failure raises `LLMResponseError`.

```python
from fsm_llm import CompletionRequest, LiteLLMInterface, LiteLLMEmbedder, tool_exchange

llm = LiteLLMInterface("ollama_chat/qwen3.5:4b")
reply = llm.complete(CompletionRequest(
    messages=[{"role": "user", "content": "What is the weather in Paris?"}],
    tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
))
if reply.kind == "calls":
    results = [str(run_tool(c.name, c.arguments)) for c in reply.calls]
    messages_to_append = tool_exchange(reply.text, reply.calls, results)
llm.usage()          # LLMUsage{calls, errors, usage_missing, prompt/completion/total_tokens, by_kind}
llm.reset_usage()    # returns the snapshot it cleared

vectors = LiteLLMEmbedder("ollama/qwen3-embedding:0.6b").embed(["a", "b"])  # one request
```

Usage counters count every provider call of that instance once, per kind (`generate`, `extract`, `classify`, `stream`, `complete`; `embed` for the embedder): a raising call counts as a call and an error, a streamed call with usage missing. `LiteLLMEmbedder(model, *, api_key=None, timeout=120.0, retries=0, **kwargs)` has its own model, credentials and counters; no texts means no request and `[]`; a missing, uneven or non-finite vector raises `LLMResponseError`.

### Completion state

A state may declare one optional `completion` field (`CompletionStateConfig`; the definition stays v4.1):

```json
"call_model": {
  "id": "call_model",
  "description": "One tool-calling model turn",
  "purpose": "Let the model call a tool or answer",
  "completion": {
    "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
    "tool_choice": "auto",
    "instructions": "You are a weather assistant.",
    "messages_key": "_completion_messages",
    "result_key": "completion_result"
  },
  "transitions": [
    {"target_state": "run_tools", "description": "The model called tools", "priority": 10,
     "conditions": [{"description": "Calls", "logic": {"==": [{"var": "completion_result.kind"}, "calls"]}}]},
    {"target_state": "done", "description": "Anything else", "priority": 900}
  ]
}
```

Its Pass 1 is one `complete` call over `[system(instructions)] + context[messages_key]`: a tool-calling turn with `tools`, or a structured turn with `response_format_key` (an internal context key holding the response format), exactly one of them. Core adds no prompt text, no history and no neutral user turn, and runs no tool. The result `{kind, text, calls}` is written to `result_key` (public, handler-only for extraction) and the call is made only while that key is unset, so the handler that acts on it clears it. The transcript under `messages_key` (internal) is the consumer's: handlers append `tool_exchange(...)` to it; it never enters `get_conversation_history`. A transcript with an unpaired tool call, or a tool-calling turn whose transcript has no `user` message, raises `LLMResponseError` before sending and rolls the turn back. A completion state declares no `field_extractions`, `classification_extractions`, `required_context_keys` or `extraction_instructions` and is not terminal; `fsm-llm-validate` warns when its transitions do not cover every result kind.

A state with empty `response_instructions` is never sent to the interface: no `generate_response` call is made for it. `ResponseGenerationRequest` carries `system_prompt`, `user_message`, `transition_occurred` and `response_format`; the prompt is the only context channel. All three request models (`ResponseGenerationRequest`, `FieldExtractionRequest`, `BulkExtractionRequest`) refuse unknown fields (`extra="forbid"`) and carry `user_message: str | None`: `None` means there is no user message (a message-free step, the greeting, a context-only classifier call), a string (including `""`) is what the user sent. `LiteLLMInterface` fills the provider user turn for both cases (`NEUTRAL_USER_TURN`, `EMPTY_USER_MESSAGE_TURN`); a custom interface must accept `None`. `FieldExtractionRequest.context` and `.validation_rules` are filled for third-party interfaces even though `LiteLLMInterface` does not read them.

`DataExtractionResponse` (the `extract_bulk_data` return) carries `rejected_corrections: dict[str, Any]` (default `{}`): a bulk value for an already-set key that the provenance rule refused and that the user's message states as a whole word of at least 3 characters (a 1 or 2 character value such as `US` never grounds). The stored value is unchanged; the pipeline hands the dict to the Pass-2 prompt as a `<rejected_corrections>` block so the reply says the change was not applied. It also carries `extraction_failed: bool` (default `False`): set by the pipeline when the bulk extraction call raised, so `build_response_prompt(..., extraction_failed=True)` adds one plain line saying a restated value may not have been stored. A custom `LLMInterface` never needs to set either field.

`FSMDefinition.handler_only_keys: list[str]` (default `[]`, opt-in): keys a user message can never write. They are dropped from the bulk return, the per-field configs and the post-transition configs; a handler write, `update_context` and `initial_context` still work. Only listed keys are covered: an unlisted gate key stays writable and a classification-owned key is not covered. A stacked child FSM uses its own list. `fsm-llm-validate` reports a listed key that no state references as INFO and warns (never errors) when a listed key is a `classification_extractions` field name.

`LiteLLMInterface` is the built-in implementation supporting 100+ providers via litellm; it implements `generate_response_stream` (the Pass-2 streaming path behind `API.converse_stream`) and `complete`. It is the only place the package calls litellm. Constructor kwargs named in `constants.RESERVED_LLM_CALL_KWARGS` (`model`, `messages`, `temperature`, `max_tokens`, `stream`, `response_format`, `tools`, `tool_choice`, `functions`, `function_call`, `n`) are ignored with a WARNING. Every Pass-2 request carries `ResponseGenerationRequest.temperature` (the conversation's internal `_response_temperature`, else `None` for the interface's own); a custom interface should honour it.

## WorkingMemory (`fsm_llm`)

```python
from fsm_llm import WorkingMemory, BUFFER_METADATA

wm = WorkingMemory()
wm.set("core", "goal", "book a flight")   # set(buffer, key, value)
wm.get("core", "goal")                    # get(buffer, key, default=None)
wm.get_all_data()                         # flattened view across buffers
wm.hidden_buffers                         # read-only frozenset of hidden buffer names
```

Named buffers: `core`, `scratch`, `environment`, `reasoning` (constants `BUFFER_CORE`, `BUFFER_SCRATCH`, `BUFFER_ENVIRONMENT`, `BUFFER_REASONING`, `DEFAULT_BUFFERS`, `DEFAULT_HIDDEN_BUFFERS` are exported from `fsm_llm`). The `BUFFER_METADATA` (`"metadata"`) buffer is hidden by default and excluded from `get_all_data()`/`get_user_visible_data()`. Reach: when attached as `FSMContext.working_memory`, buffer data enters only the Pass-1 per-field extraction prompt (default `context_keys`); the Pass-2 response prompt is built from `context.data` alone, so a value the response model must see goes in `context.data`.

## TransitionEvaluatorConfig

```python
@dataclass
class TransitionEvaluatorConfig:
    strict_condition_matching: bool = True
    detailed_logging: bool = False
```

Transitions are ranked by `priority` alone; no confidence score is computed (`TransitionEvaluation` has no `confidence` field). The removed fields `ambiguity_threshold`, `minimum_confidence` and `evidence_conditions_normalizer` raise `TypeError`.

## Classification (`fsm_llm`)

Built into core -- no separate install needed.

```python
from fsm_llm import Classifier, ClassificationSchema, IntentDefinition, IntentRouter

schema = ClassificationSchema(
    intents=[  # at least two intents; fallback_intent must be one of them
        IntentDefinition(name="billing", description="Billing questions"),
        IntentDefinition(name="general", description="Anything else"),
    ],
    fallback_intent="general",
)
classifier = Classifier(schema, model="gpt-4o-mini")
result = classifier.classify("Where is my invoice?")
# result.intent, result.confidence
if classifier.is_low_confidence(result):  # compares against schema.confidence_threshold
    ...
```

The `ClassificationResult.is_below_default_threshold` property compares against a fixed default of 0.6 (`DEFAULT_CONFIDENCE_THRESHOLD`), not the schema's `confidence_threshold`; use `classifier.is_low_confidence(result)` for the schema-aware check. `ClassificationResult` has no `is_low_confidence`.

`Classifier(schema, model=None, *, llm=None, api_key=None, config=None, **llm_kwargs)` sends one `complete` request per call. With `llm=` every request goes to that interface (`model` defaults to its model; an `api_key`, extra kwargs, a different `model` or an object that is not an `LLMInterface` implementing `complete` raise `ValueError`). Without it, it builds its own `LiteLLMInterface` for `model` (default `DEFAULT_LLM_MODEL`; timeout default 120 s; `retries` is the SDK's `max_retries`; a model without `response_format` support logs a WARNING per call). Inside a conversation, the AMBIGUOUS-transition classifier and every `classification_extractions` entry use the conversation's own `llm_interface`; only an entry that names a different `model` gets a private interface, which inherits no connection kwargs. Every failure of the classifier's call is `ClassificationError` (a reply text that does not parse: `ClassificationResponseError`), and only those make a turn stay with a WARNING; any other exception fails the turn. `HierarchicalClassifier(..., llm=None)` takes the same `llm=`. `classify`/`classify_multi` accept `None` for a context-only call. The ambiguous-transition record is stored only in `context.metadata["transition_classification"]` (read it with `api.fsm_manager.get_complete_conversation(conv_id)["metadata"]`).

```python
# Multi-intent
result = classifier.classify_multi("Check order and update billing")

# Hierarchical (two-stage domain -> intent, for >15 intents)
from fsm_llm import HierarchicalClassifier, HierarchicalSchema
h_schema = HierarchicalSchema(
    domain_schema=ClassificationSchema(
        intents=[
            IntentDefinition(name="billing", description="Billing domain"),
            IntentDefinition(name="support", description="Support domain"),
        ],
        fallback_intent="support",
    ),
    intent_schemas={"billing": schema, "support": schema},  # one schema per domain
)
h_classifier = HierarchicalClassifier(h_schema, model="gpt-4o-mini")

# Intent routing
router = IntentRouter(schema)
router.register("billing", handle_billing)
response = router.route(message, classification_result)
```

## Visualization (`fsm_llm`)

```python
from fsm_llm import build_fsm_graph, to_dot, to_mermaid, visualize_fsm_ascii

graph = build_fsm_graph(fsm_dict)   # -> FSMGraph{name, initial_state, nodes, edges} (frozen)
print(to_mermaid(graph))            # Mermaid stateDiagram text
print(to_dot(graph))                # Graphviz DOT text
print(visualize_fsm_ascii(fsm_dict, style="compact"))
```

`FSMGraphNode{id, description, purpose, is_initial, is_terminal}`, `FSMGraphEdge{source, target, description, priority}`. `build_fsm_graph` accepts an `FSMDefinition` or a dict and raises `ValueError` for a definition whose initial state or a transition target is missing, or whose shape is malformed. CLI: `fsm-llm-visualize --fsm F --format ascii|mermaid|dot` (`--style` applies to ASCII only; `fsm-llm --mode visualize` prints ASCII). The monitor's FSM graphs read the same function.

## ReasoningEngine (`fsm_llm.reasoning`)

```python
from fsm_llm.reasoning import ReasoningEngine, ReasoningType

engine = ReasoningEngine(model="gpt-4o-mini")
solution, trace = engine.solve_problem("problem text", initial_context={})
# solution: str, trace: dict (run metadata: steps, reasoning_types_used, final_confidence, execution_time_seconds)
```

The engine sends no user message: the classifier FSM and the orchestrator run on core `run_until_terminal`, and one orchestrator run (at most 170 steps, `Defaults.MAX_SOLVE_STEPS`) pushes and pops the strategy FSMs from its `before_step` hook. A failed solve, a spent budget included, raises `ReasoningExecutionError` with `details={"conversation_id", "responses_so_far", "partial_context"}`. `ReasoningEngine(llm_interface=...)` sends every call to that interface (`model` is then not used).

9 strategies: `SIMPLE_CALCULATOR`, `ANALYTICAL`, `DEDUCTIVE`, `INDUCTIVE`, `CREATIVE`, `CRITICAL`, `HYBRID`, `ABDUCTIVE`, `ANALOGICAL`.

## Agents (`fsm_llm.agents`)

```python
from fsm_llm.agents import create_agent, ReactAgent, tool, ToolRegistry, HumanInTheLoop, AgentConfig

# @tool decorator auto-generates JSON schema from type hints
@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

# Create agent: create_agent(pattern="react", tools=None, *, config=None, system_prompt=None, **kwargs)
agent = create_agent("react", [search], config=AgentConfig(model="gpt-4o-mini"),
                     system_prompt="Cite your sources.")  # stored as AgentConfig.instructions
result = agent("task")  # or agent.run("task")
# result.answer, result.success, result.stop_reason, result.trace, result.structured_output
for chunk in agent.run_stream("task"):  # model text only
    print(chunk, end="")

# Structured output (agent classes take a ToolRegistry, not a list)
registry = ToolRegistry().register(search._tool_definition)
agent = ReactAgent(tools=registry,
                   config=AgentConfig(model="gpt-4o-mini", output_schema=MyPydanticModel))

# Human-in-the-loop (ReactAgent, ReflexionAgent, ReasoningReactAgent)
# approval_policy(call, context) -> bool picks the gated calls; each approval covers one call
hitl = HumanInTheLoop(approval_policy=lambda call, ctx: call.tool_name == "search",
                      approval_callback=fn)
agent = ReactAgent(tools=registry, config=AgentConfig(model="gpt-4o-mini"), hitl=hitl)
# With a callback and no policy, @tool(requires_approval=True) tools are the gated ones
```

- `AgentResult{answer, success, trace, final_context, structured_output, stop_reason=None}`. `success` is `True` only when the run reached its goal. A forced stop still returns its last answer, with `success=False`. `stop_reason` is one of the `StopReason` values (exported from `fsm_llm.agents`): `answered`, `evidence` (planner patterns with real executed work), `max_iterations`, `forced_pass` (a failing evaluator/checker/debate verdict overridden at its limit, or Reflexion's `max_reflections` reached without a pass; a genuine pass on the last round is `answered`), `stalled` (three turns with no tool), `verification_failed`, `no_result`, `gate_failed` (a PromptChain gate failed), `ended` (the conversation was closed from outside before a terminal state). `AgentServer` responses carry it too. The answer is the reply of the last speaking state or a pattern's own answer key; `final_answer` in the context is never read.
- `AgentConfig{model, max_iterations=10, timeout_seconds=300.0, temperature=0.5, max_tokens=1000, output_schema, instructions=None, ...}` rejects unknown fields. `model` defaults to env `LLM_MODEL` (read when the config is built), then `DEFAULT_LLM_MODEL`. `instructions` (max 2,000 chars) is prefixed to every non-empty state and per-field instruction of FSM patterns (not `classification_extractions`, the transition classifier or the reasoning engine; a slot overflowing core's 5,000-character limit raises `AgentError` at `run()`) and is `NativeFunctionCallingReactAgent`'s default `system_policy`; Swarm and meta_builder do not use it.
- ReAct family: `max_iterations=N` counts think turns; for N >= 2 a run that never concludes gets N think turns and N - 1 tool calls, and N = 1 behaves like N = 2. A conclusion the model makes itself on the last think turn, backed by a tool result, is `success=True`; the loop ceiling `N * 3` FSM steps raises `BudgetExhaustedError`, `timeout_seconds` raises `AgentTimeoutError` (both budgets are enforced by core `API.run_until_terminal`; the agent error's `__cause__` is core's `RunBudgetExceededError`; a slow approver uses up the timeout). Both also propagate from ADaPT subtasks and Orchestrator workers.
- Constructors raise `TypeError` for `hitl=`, `tools=`, `evaluation_fn=` or any `HumanInTheLoop` argument (`approval_policy=`, `approval_callback=`, `on_escalation=`, `confidence_threshold=`, `approval_timeout=`) on a pattern that does not take them, and for `model=`, `temperature=`, `max_tokens=` (set them on `AgentConfig`). Other keyword arguments (`seed=`, `timeout=`, `llm_interface=`, `handlers=`, ...) go to `API`. With `llm_interface=` the interface owns the model settings: `AgentConfig.model/temperature/max_tokens` are not applied, any LLM setting beside it is refused by core, and `enable_prompt_cache=True` raises `AgentError`. `REWOOAgent`, `PlanExecuteAgent`, `ParallelReactAgent` and `NativeFunctionCallingReactAgent` have no approval step and raise `AgentError` when their registry holds a `requires_approval` tool. `ReactAgent`, `ReflexionAgent` and `ReasoningReactAgent` raise `AgentError` for such a tool when `hitl` has neither an approval callback nor a policy.
- `initial_context` cannot set run-owned keys (`final_answer`, `should_terminate`, `observation_count`, tool and approval keys, `refused_actions`, a pattern's own outputs such as ADaPT `operator`) or the driver grant `_approval_granted`: they are dropped with a warning.
- Tools: `@tool(name=, description=, parameter_schema=, requires_approval=, annotations=ToolAnnotations(read_only=, destructive=, idempotent=, open_world=), timeout_s=)`. Native tool schemas and prompt descriptions come from the function's own pydantic model (exact types: `Optional[int]` is `integer or null`). `ToolRegistry.execute(call, *, gated=False)` never raises; a `timeout_s` call runs in a worker thread and a timeout returns a failed `ToolResult` with `timed_out=True` and `status` `unknown` (the tool keeps running). `RetryingToolRegistry` re-runs only tools annotated `idempotent` or `read_only`, never a granted (`gated`) call. A custom registry's `execute` must accept `gated`.
- `NativeFunctionCallingReactAgent(tools, config=None, system_policy=None, *, seed=None, **api_kwargs)` uses the provider's native tool calls as an FSM on core completion states (`build_native_fc_fsm`); inject an interface with `llm_interface=` (there is no `complete_fn`). `config.force_final_tool` adds one forced tool turn and `output_schema` one repair turn.
- `MetaBuilderAgent(config=MetaBuilderConfig(model, temperature=0.7, max_tokens=4096, max_turns=50, timeout_seconds=...), **api_kwargs)`: `run(task)`, `start(message="")`, `send(message)`, `is_complete()`, `get_result()`, `get_internal_state()`, `run_interactive()`. It is an FSM run by core (type classification, collect replies, one structured build call); a malformed build raises `MetaValidationError` from `run`, a build-call outage `BuilderError`. The artifact builders it uses are `FSMArtifactBuilder`, `WorkflowArtifactBuilder` and `AgentArtifactBuilder` (mutators, including `update_state`, return the builder; `take_warnings()` returns and clears the accumulated warnings; `build()` validates and raises `BuildError`).
- Agents drive their FSM with core's `run_until_terminal` / `run_until_terminal_stream`: no synthetic user message, and silent intermediate states make no reply call. On a HITL denial the driver writes one sentence per refused call that did not run to `final_context["refused_actions"]` (`ContextKeys.REFUSED_ACTIONS`, built by `handlers.refusal_record` from `handlers.call_label`), and approval-gated conclude prompts tell the model those actions were not performed; a call approved and run later loses its entry, and a denied repeat of a call that already ran (`handlers.call_ran`) adds none.

18 `create_agent()` patterns: `react`, `rewoo`, `debate`, `plan_execute`, `prompt_chain`, `self_consistency`, `orchestrator`, `adapt`, `evaluator_optimizer`, `maker_checker`, `reflexion`, `meta_builder`, `swarm`, `parallel_react`, `native_fc`, `verified_react`, `auto_memory`, `reasoning_react`. The source of truth is `_PATTERNS` in `src/fsm_llm/agents/__init__.py`; an unknown pattern raises `ValueError` listing the available names. `tools=` for a pattern that takes none (`debate`, `prompt_chain`, `self_consistency`, `evaluator_optimizer`, `maker_checker`, `meta_builder`, `swarm`) raises `TypeError`. Pattern names are matched after `strip().lower()`; any other first argument (including the removed `create_agent("You are ...", tools)` form) raises `ValueError`. Pass instructions as `system_prompt=`.

Multi-agent coordination and integrations (constructed directly, not via the factory): `SwarmAgent`, `AgentGraph` / `AgentGraphBuilder` (DAG orchestration; `build()` raises `BuildError`, a duplicate node name included), `ConfiguredAgentBuilder` (fluent `create_agent`), `MCPToolProvider` (MCP tools), `AgentServer` / `RemoteAgentTool` (A2A), `SemanticToolRegistry` (embedding-based tool retrieval through core's `LiteLLMEmbedder`, or `embed_fn=`), `SOPRegistry` / `load_builtin_sops` (reusable agent templates).

User guide: `src/fsm_llm/agents/README.md`. Audit record, adjusted decisions and deferred work: `docs/agents_roadmap.md`.

## WorkflowEngine (`fsm_llm.workflows`)

```python
from fsm_llm.workflows import WorkflowEngine, create_workflow, auto_step, condition_step

workflow = create_workflow("my_workflow", "My Workflow")
workflow.with_initial_step(auto_step("start", "Start", next_state="check"))
workflow.with_step(condition_step("check", "Check", condition=fn, true_state="ok", false_state="fail"))

engine = WorkflowEngine(max_concurrent_workflows=100)
engine.register_workflow(workflow)
instance_id = await engine.start_workflow("my_workflow", initial_context={})
await engine.advance_workflow(instance_id)
status = engine.get_workflow_status(instance_id)
await engine.shutdown()
```

`workflow_builder(id, name)` returns a `WorkflowBuilder` (`add_step`, `set_initial_step`, `add_metadata`); its `build()` always validates, returns a new `WorkflowDefinition` that later builder calls do not change, and raises `BuildError` chained from `WorkflowDefinitionError` or `WorkflowValidationError`. It has no `validate` argument.

11 step types: `auto_step`, `api_step`, `condition_step`, `llm_step`, `wait_event_step`, `timer_step`, `parallel_step`, `conversation_step`, `agent_step`, `retry_step`, `switch_step`.

Events and loops: `await engine.process_event(WorkflowEvent(event_type="paid", payload={...}))` wakes waiting instances (set `instance_id=` to target one; `wait_event_step(..., correlation_key=...)` matches a payload key against the instance context). Loops must pass through a `timer_step` or `wait_event_step`; purely synchronous cycles are rejected by `register_workflow`. A step failure with no error route FAILS the instance. `engine.add_hook(fn)` observes step and status changes. `WorkflowEngine(*, max_concurrent_workflows=100, max_completed_instances=1000, max_steps_per_run=1000, executor=None)` takes keyword arguments only. A `conversation_step` publishes `last_response`/`final_answer` as the last reply that was actually spoken (silent states say nothing), and adds neither key when nothing was spoken.

## Harness (`fsm_llm.harness`)

The iterative-planner protocol as a 6-state FSM over a plan directory. Requires
`pip install -e ".[harness]"` from a clone (the `harness` extra pulls `agents`). The public surface is one literal `__all__`; the
load-bearing names are below.

### HarnessAgent -- the driver

```python
from fsm_llm.harness import HarnessAgent, Workspace, build_default_worker_factory, ContextKeys

workspace = Workspace("./src")                       # confined source-tree root
agent = HarnessAgent(
    worker_factory=build_default_worker_factory(workspace),
    approval_callback=None,      # None => a callback that DENIES every human gate
    revert_callback=None,        # None => compute the leash revert, execute nothing
    findings_threshold=3,        # EXPLORE -> PLAN
    max_fix_attempts=2,          # the autonomy leash
    max_leash_grants=2,          # human leash-continues honoured per plan step
    iteration_hard_cap=6,        # PLAN -> EXECUTE
    max_explore_redispatches=9,  # extra EXPLORE dispatches per run while blocked
)
# Same agent, fluent: HarnessAgentBuilder().set_worker_factory(...).set_max_fix_attempts(2).build()
result = agent.run(
    "add a retry to the uploader",
    initial_context={
        ContextKeys.PLAN_DIR: "plans/plan-2026-07-22T101500-1a2b3c4d",
        ContextKeys.WORKSPACE_ROOT: "./src",
    },
)
agent.presentations   # list[Presentation] -- the Presentation Contracts emitted
agent.reverts         # list[RevertDirective] -- computed, not executed
agent.audit_issues    # list[Issue] | None -- the CLOSE audit, if CLOSE was reached
```

Executor dispatches on one plan step are bounded by
`max_fix_attempts * (1 + max_leash_grants)` for **any** sequence of approvals --
an approving callback cannot raise it. `worker_factory=None` is a diagnostic mode:
the FSM turns but no gate opens.

Worker seam: `WorkerFactory = Callable[[RoleRequest], AgentResult]`. `RoleSpec`
(one per state) carries `tool_scope`, `plan_tool_scope`, `owned_artifacts`,
`output_schema`, `expected_keys`, `writable_keys`, `max_iterations`;
`ROLE_SPECS` / `get_role_spec(state)` expose them.

### Plan directory, gate and audit

```python
from fsm_llm.harness import PlanDirectory, Role, ArtifactNames, StateDoc, pre_step_gate, audit

directory = PlanDirectory.create("plans", role=Role.ORCHESTRATOR)   # mints plan-<ts>-<hex8>
directory.write_text(ArtifactNames.STATE, StateDoc(state="explore").to_markdown())  # atomic
run_state = directory.load_run_state()          # RunState(plan_id, doc); .state == "explore"

gate = pre_step_gate(directory.path)            # GateResult(passed, slug, detail, exit_code)
issues = audit(directory.path, workspace_root=".")     # list[Issue]; [] means clean
```

`directory.path` is the plan directory; `directory.root` is its **parent** (the
confinement root `PlanMemory` was given), so pass `path` to anything that expects
a plan directory. `mint_plan_id()` is exported separately for callers that want
the id without a directory.

`pre_step_gate` evaluates 4 slugs in `GateSlug.ORDER` (`no-plan`, `wrong-state`,
`leash-cap`, `iteration-cap`), short-circuits on the first failure, and never
raises -- an unreadable `state.md` **is** the `no-plan` answer. Every failure is
HARD and carries `exit_code == 2`. `audit` runs the 30 checks in `CHECKS` and
reports a check that raises as an ERROR rather than letting it suppress the rest.

CLOSE-phase size policies, all pure and non-writing: `evict_lessons(doc)` and
`apply_sliding_window(doc)` return `(trimmed_doc, report)`, while
`check_system_cap(doc)` returns a `CapReport` only -- `SYSTEM.md` is measured and
REFUSED over its cap, never trimmed, because it carries no eviction order and all
six of its sections are required. `PlanDirectory.enforce_lessons_cap` /
`enforce_system_cap` / `apply_sliding_window` are the wrappers that write.

### Confined tools

```python
from fsm_llm.harness import Workspace, PlanMemory, build_workspace_tools, build_plan_tools, Role

workspace = Workspace("./src")                          # confinement only
memory = PlanMemory("plans/plan-...", role=Role.EXPLORER)  # confinement + ownership
registry = build_workspace_tools(workspace, allowed=("read_file", "grep_files"))
build_plan_tools(memory, allowed=("write_plan_file",), registry=registry)
```

Tool name groups: `READ_ONLY_TOOLS`, `WRITE_TOOLS`, `SHELL_TOOLS`,
`PLAN_READ_TOOLS`, `PLAN_WRITE_TOOLS`. `run_command` is off unless
`Workspace(..., allow_shell=True)`; `COMMAND_ALLOWLIST` is
`cat grep head ls tail wc` (`git` is deliberately excluded), and
`VERIFICATION_COMMANDS` (`git make mypy pytest ruff`) is an opt-in set.
An escape raises `HarnessConfinementError`; an ownership violation raises
`HarnessOwnershipError`.

### Artifacts and hardening

`ARTIFACT_MODELS` maps each of the 15 artifact names to its pydantic model -- 14
classes, since `FINDINGS.md` and `DECISIONS.md` share `ConsolidatedDoc`:
`StateDoc`, `PlanDoc`, `DecisionsDoc`, `FindingsIndexDoc`, `FindingsTopicDoc`,
`ProgressDoc`, `VerificationDoc`, `ChangelogDoc`, `CheckpointDoc`, `SummaryDoc`,
`ConsolidatedDoc`, `LessonsDoc`, `SystemAtlasDoc`, `IndexDoc`. Every one carries
`from_markdown()` / `to_markdown()`. Contract tables: `DECISION_ENTRY_SCHEMAS`,
`PRESENTATION_CONTRACTS`, `MANDATORY_ADDITIONAL_CHECKS`, `VERDICT_BULLETS`,
`VERDICT_RECOMMENDATIONS`, `REJECTED_EVIDENCE`.

Small-model reply recovery (`hardening`): `strip_model_noise`,
`parse_json_payload`, `parse_role_output` -> `RoleOutput`, `coerce_worker_output`,
`type_matches`, `as_int`, `retry`. All fail CLOSED -- a garbled reply is never
retried into a pass.

### CLI

```bash
fsm-llm-harness new "add a retry to the uploader" [--create-only] [--model M] [--workspace .]
fsm-llm-harness resume   plans/plan-...  [--goal G]
fsm-llm-harness status   plans/plan-...
fsm-llm-harness validate plans/plan-...  [--workspace .]
fsm-llm-harness close    plans/plan-...  [--workspace .] [--apply]
```

Exit codes: `0` pass, `1` a negative answer (audit ERRORs, a failed run, a usage
fault), `2` **RESERVED** for a HARD `pre_step_gate` refusal. Because `2` is
reserved, argparse usage errors exit `1` rather than argparse's conventional `2`.
Model resolution: `--model` > `$LLM_MODEL` > package default. `close` is DRY-RUN
without `--apply` and refuses to compress a directory with audit ERRORs.

## Eval (`fsm_llm.eval`)

Two evaluation kinds behind one CLI, `fsm-llm-eval` (also `python -m fsm_llm.eval`;
run `examples` from the repository root or pass `--examples-dir`). Extra
`eval` has no third-party dependencies. Import names directly
(`from fsm_llm.eval import wilson_ci`) or as `from fsm_llm import eval as fsm_eval`,
so the builtin `eval` is not shadowed.

### CLI

```bash
fsm-llm-eval examples [--model M] [--workers N] [--timeout S] [--category C] [--filter S]
                      [--output-dir D] [--examples-dir P] [--python EXE]
                      [--config FILE] [--fail-under PCT] [--list]
fsm-llm-eval run DATASET [--model M] [--trials N] [--temperature T] [--max-tokens N]
                         [--workers N] [--output-dir D] [--config FILE]
                         [--fail-under PCT] [--list]
```

Exit codes: `0` success, also for low scores; `1` usage error, bad config or dataset
(including a file that is not UTF-8 or not JSON), unwritable output, a missing
examples directory, or nothing matched; `2` only when `--fail-under PCT` is given and
the health score (`examples`) or overall pass rate (`run`) is below PCT (compared
exactly: 57 of 100 meets 57); `130` interrupted by Ctrl-C. Usage errors exit `1`, not
argparse's `2`, so CI can tell a regression from a typo. On Ctrl-C, work not yet
started is cancelled and the report files are written for what finished, with
`"interrupted": true`.

### Configuration (`EvalConfig`)

Settings precedence: built-in defaults < a dataset's embedded `config` (`run` only) <
`--config FILE` (one JSON object) < explicit flags. Unknown keys and bad values raise
`EvalConfigError` (CLI exit 1).

| Field | Default | Used by | Meaning |
|-------|---------|---------|---------|
| `model` | `None` | both | `None` = `$LLM_MODEL`, else `DEFAULT_LLM_MODEL`, read when the run starts |
| `workers` | `4` | both | Parallel threads (>= 1) |
| `timeout` | `120` | examples | Seconds for an example no timeout table names (>= 1) |
| `output_root` | `"evaluation"` | both | Parent of auto-named run directories |
| `output_dir` | `None` | both | Exact run directory; must be new or empty |
| `fail_under` | `None` | both | Threshold in percent (0-100) for exit 2 |
| `examples_dir` | `"examples"` | examples | Tree scanned for `<category>/<name>/run.py` and `run_manual.py` |
| `python` | `None` | examples | Interpreter for example scripts; `None` = the running one |
| `category` | `None` | examples | Only this category |
| `name_filter` | `None` | examples | Only names containing this substring (`--filter`) |
| `example_inputs` | `{}` | examples | Stdin script per example name, merged over the built-in table |
| `example_timeouts` | `{}` | examples | Seconds per example name, merged over the built-in table |
| `category_timeouts` | `{}` | examples | Seconds per category, merged over the built-in table |
| `trials` | `3` | run | Trials per case (>= 1) |
| `temperature` | `None` | run | LLM temperature (>= 0); `None` = framework default |
| `max_tokens` | `None` | run | LLM max tokens per call (>= 1); `None` = framework default |
| `llm_kwargs` | `{}` | run | Extra `API(...)` keyword arguments (e.g. `api_base`, `api_key`, `max_history_size`), merged key by key; may not set `model`, `temperature`, `max_tokens`, `llm_interface`, `fsm_definition`, `definition`, `path`, or the non-JSON `handlers`, `transition_config`, `session_store`; values are not written to `results.json` |

The model is the first set of `--model`, `--config` file, dataset `config`,
`$LLM_MODEL`, `DEFAULT_LLM_MODEL`, so a dataset that pins `model` beats an exported
`LLM_MODEL`. A case's `fsm` path is relative to the dataset file; `output_root` and
`output_dir` are relative to the current directory. The LLM settings do not apply when
an `llm_interface_factory` is given.

Example timeout precedence: per-example table > category table > `timeout`.

### Python API

```python
from fsm_llm.eval import (
    EvalConfig, load_config, merge_config, resolve_model,
    discover_examples, run_examples,
    load_cases, run_cases, run_case_trial, run_dataset, check_expectations,
    open_run_dir, wilson_ci, fisher_exact_two_sided, pass_rate,
)

# One call, like `fsm-llm-eval run`: config layer (file path, dict or EvalConfig),
# then keyword overrides; `checks` are callables(trial) -> list of failure messages
report = run_dataset("cases.json", config="eval.json", trials=5,
                     checks=[lambda t: [] if t.responses else ["no reply"]])

config = merge_config({"workers": 2}, load_config("eval.json"))  # later layers win
model = resolve_model(config)

# Examples: subprocess per script, heuristic score 0-4 per example
targets = discover_examples(config)
run_dir = open_run_dir(config.output_dir, config.output_root, model)
report = run_examples(targets, config, model, run_dir)   # ExampleReport
report.health, report.wall_time, report.results          # results: list[ExampleResult]

# Conversation cases: fresh API per trial, expectations checked per trial
cases, embedded = load_cases("cases.json")               # embedded: config layer
config = merge_config(embedded, {"trials": 5})
report = run_cases(cases, config, open_run_dir(None, config.output_root, model),
                   llm_interface_factory=None,           # or a callable returning an LLMInterface
                   dataset="cases.json")                 # CaseReport
report.overall   # {"k": 12, "n": 15, "rate": 0.8, "wilson_ci": [0.548, 0.930]}

wilson_ci(33, 40)                      # (lo, hi), 95% by default
fisher_exact_two_sided(2, 40, 40, 40)  # two-sided p for two k/n arms
pass_rate(0, 0)                        # rate 0.0, wilson_ci [0.0, 1.0]
```

`run_case_trial(case, config, trial, llm_interface_factory=None, *, checks=())` never
raises: an exception (a raising check included) becomes a failed trial with `error`
set. `run_cases(..., checks=())` passes `checks` to every trial. Both runners return a
report whose `interrupted` is `True` after Ctrl-C (results then hold only finished
work).
Records helpers: `append_row`, `read_rows` (JSONL), `write_json` (indent 2, sorted
keys), `utc_now`, `git_commit` (raises on failure), `git_short_hash` (`"unknown"` on
failure), `model_slug`, `make_run_dir` (adds `_2`, `_3`, ... on a name collision).

### Dataset schema

A JSON list of cases, a JSON object `{"config": {...}, "cases": [...]}`, or a `.jsonl`
file with one case per line (blank lines skipped).

| Case key | Required | Meaning |
|----------|----------|---------|
| `id` | yes | Unique, non-empty |
| `fsm` | yes | Path relative to the dataset file, or an inline FSM definition object |
| `turns` | yes | User messages sent in order (at least one); sending stops once the conversation ends, and unsent turns fail the trial unless `expect` declares `ended` |
| `expect` | yes | At least one check (below) |
| `initial_context` | no | Context passed to `start_conversation` |
| `description` | no | Shown in `--list` and the report |

| `expect` key | Passes when |
|--------------|-------------|
| `final_state` | The state after the last turn sent equals it |
| `visited_states` | Each listed state was current after start or after some turn (any order) |
| `context` | Each key in `get_data` equals the given value |
| `context_keys` | Each key is present in `get_data` and not null |
| `responses_contain` | Each substring occurs, case-insensitively, in at least one response (greeting included) |
| `ended` | `has_conversation_ended` equals it |

### Output

Each run gets `<output_root>/<YYYY-MM-DD_HH-MM>_<git-short-hash>_<model-slug>/`
(`_2`, `_3`, ... when the name exists).

- `examples`: `scorecard.md`, `results.json` (`date, git_commit, model, health_score,
  total_examples, distribution, results[{name, category, score, failures, duration,
  exit_code, timed_out}], wall_time_s, workers, default_timeout, evaluator,
  interrupted`),
  `logs/<category>/<category>_<name>.log`.
- `run`: `rows.jsonl` (one row per trial, appended as it finishes: `case_id, trial,
  passed, failures, error, final_state, visited_states, responses, context, ended,
  turns_sent, duration`), `results.json` (`date, git_commit, model, dataset, evaluator,
  config, overall, cases[{id, description, k, n, rate, wilson_ci, first_failure,
  failure_counts}], wall_time_s, interrupted`; `config.llm_kwargs` values are replaced
  by `"<not recorded>"`), `summary.md`.

## Monitor (`fsm_llm.monitor`)

```python
from fsm_llm.monitor import InstanceManager, configure, app

manager = InstanceManager()
manager.attach_api(api)           # show an API you created: its events and conversations
configure(manager=manager)        # keyword-only; installs the manager in the web server

import uvicorn
uvicorn.run(app, host="127.0.0.1", port=8420)
# Or just use the CLI: fsm-llm-monitor (it also enables library logging;
# when embedding, call fsm_llm.setup_logging() to fill the Logs page)

# OTEL export is available via OTELExporter (requires fsm-llm[otel])
```

`attach_api` registers the monitor handlers on the `API` and lists its conversations next to the launched ones; one API is attached at a time (attaching another switches the previous one's handlers off), and a failed registration raises `MonitorConnectionError` with the previous API kept. The FSM visualizer routes (`POST /api/fsm/visualize`, `GET /api/fsm/visualize/preset/{id}`) draw the nodes and edges of core `build_fsm_graph`, the same graph data as `to_mermaid` and `to_dot`; a definition core cannot graph (unknown initial state, a transition to a missing state) answers 400 `failed to parse FSM definition: <reason>`. Agent and workflow graphs come from the hand-written `static/flows.json`.

## Exception Hierarchy

```
FSMError
├── ConversationBusyError
├── FSMDefinitionNotFoundError (also a ValueError: id is not a file path, no FSM registry)
├── StateNotFoundError
├── InvalidTransitionError
├── LLMResponseError
├── TransitionEvaluationError
├── ClassificationError (-> ClassificationResponseError)
├── RunBudgetExceededError (run_until_terminal spent max_steps or max_seconds)
├── BuildError (also a ValueError: a builder's build() refused; .errors, cause chained)
├── HandlerSystemError (-> HandlerExecutionError)
├── ReasoningEngineError (-> ReasoningExecutionError, ReasoningClassificationError)
├── WorkflowError (-> Definition, Step, Instance, Timeout, Validation, State, Event, Resource)
├── HarnessError (-> HarnessArtifactError, HarnessOwnershipError, HarnessReentrancyError, HarnessConfinementError)
├── EvalError (-> EvalConfigError, EvalDatasetError)
└── AgentError (-> ToolExecution, ToolNotFound, BudgetExhausted, ApprovalDenied, AgentTimeout, Evaluation)
    └── MetaBuilderError (-> Builder, MetaValidation, Output)

Exception
└── MonitorError (-> MonitorInitialization, MetricCollection, MonitorConnection)
```

## Constants

```python
from fsm_llm.constants import (
    DEFAULT_LLM_MODEL,       # "ollama_chat/qwen3.5:4b"
    DEFAULT_TEMPERATURE,     # 0.5, the API and LiteLLMInterface default
    DEFAULT_MAX_HISTORY_SIZE, # 5
    DEFAULT_MAX_MESSAGE_LENGTH, # 1000
    DEFAULT_MAX_STACK_DEPTH,  # 10
)
```
