# Architecture Deep Dive

> Covers FSM-LLM v0.13.0

Technical overview of the FSM-LLM architecture and how components work together.

## System Overview

FSM-LLM uses a **2-pass architecture** that separates data extraction, transition evaluation, and response generation:

1. **Pass 1 -- Analysis & Transition**: Data extraction from user input, transition evaluation via JsonLogic rules (with LLM classification fallback for ambiguous cases), state transition.
2. **Pass 2 -- Response Generation**: Response generated from the new state's context, ensuring the response always reflects the correct state.

```
┌──────────────────────────────────────────────────────┐
│                    User Application                   │
├──────────────────────────────────────────────────────┤
│                     API Layer                         │
├──────────────────────────────────────────────────────┤
│   FSM Manager  │  Handler System  │  LLM Interface   │
├──────────────────────────────────────────────────────┤
│              MessagePipeline (2-pass)                 │
├──────────────────────────────────────────────────────┤
│  Prompt Builders  │  Context Manager  │  Evaluator   │
├──────────────────────────────────────────────────────┤
│   Storage: In-Memory Dicts (+ optional SessionStore)  │
└──────────────────────────────────────────────────────┘
```

Conversation state lives in in-memory dicts by default. Attaching a `SessionStore`
(e.g. `FileSessionStore`) adds durable, cross-restart persistence -- state is saved after each
`converse()` and restored via `restore_session()`. A snapshot holds the root frame only; with
the opt-in event log (`API(event_log=)`), `API.fork` and `API.resume` rebuild a whole pushed FSM stack
from the log (see the `runtime` row of Package Layout and the Event Log section of `docs/api_reference.md`).

## Core Components

### API (`engine/api/`)

User-facing entry point. Manages FSM definition processing, component wiring, conversation lifecycle, and FSM stacking. Every turn entry has an async twin (`aconverse`, `aadvance`, `astart_conversation`, `aconverse_stream`, `aadvance_stream`, `arun_until_terminal`, `arun_until_terminal_stream`): the turn is one `async def` body in the engine and the sync names (`converse`, `advance`, ...) are thin `run_sync` or `iterate_sync` wrappers over it, so a sync caller needs no event loop (`engine/api/turn_entries.py`, `engine/api/run.py`). `fork` and `resume` (`engine/api/forks.py`) rebuild a conversation tree from the event log; `max_steps` and `max_seconds` give the root and each pushed frame a lifetime turn budget.

### FSMManager (`engine/fsm/`) + MessagePipeline (`engine/turn/`)

FSMManager handles FSM instance lifecycle with a per-conversation `TurnGate` (`engine/fsm/gate.py`) and an LRU FSM cache. The gate is owned by a ticket kept in a ContextVar, not by a thread, so it also holds between tasks of one event loop: a second turn on a conversation whose turn is in flight is refused with `FSMError` and never waits, and a reader called from a sibling task during an async turn raises `ConversationBusyError` at once. It also owns the optional lifetime turn budget (`engine/fsm/budget.py`): one private record per manager conversation id (the root and every pushed frame), charged on the turn entry before any model call, outside the context, the metadata and the rollback snapshot, so no handler, extraction or rollback can reset it; a spent frame raises `RunBudgetExceededError` on its next turn and the engine never pops it (one exception: the frame a `call` state pushed, which `settle` pops at its child's end or budget stop). MessagePipeline implements the 2-pass processing:

```
Pass 1: PRE_PROCESSING handlers → LLM extract data → CONTEXT_UPDATE handlers
      → Evaluate transitions → PRE_TRANSITION → State change → POST_TRANSITION

Pass 2: POST_PROCESSING handlers → LLM generate response → Return
```

### Prompt Builders (`context/layout/`)

The prompt builders produce structured XML-like prompts, plus the Pass-1 builder slot:
- `DataExtractionPromptBuilder` -- the Pass-1 slot of `FSMManager`/`MessagePipeline`; the pipeline builds the bulk extraction prompt inline and uses this builder only for the shared sanitizer
- `ResponseGenerationPromptBuilder` -- Pass 2: user-facing response generation
- `FieldExtractionPromptBuilder` -- Targeted single-field extraction

Security: XML tag sanitization, CDATA wrapping for JSON, token budget management.

### Handler System (`handlers/`)

Plugin architecture with a timing point for each stage of a turn. Handlers self-determine execution via state/context conditions, execute in priority order, with configurable error isolation (`"continue"` or `"raise"`). Critical handlers always raise on error.

### LLM Interface (`llm/`)

`LLMInterface` ABC with the methods:
- `generate_response(request)` -- Generate user-facing response (Pass 2)
- `extract_field(request)` -- Extract a targeted field from input (Pass 1)
- `generate_response_stream(request)` -- Stream the Pass-2 response token-by-token
- `extract_bulk_data(request)` -- Bulk extraction for states with `extraction_instructions`
- `complete(request: CompletionRequest) -> CompletionResponse` -- The one primitive for tool calling, plain and structured completion (classifier, completion states, the agents' LLM judge). Not abstract: the base method raises `NotImplementedError`

Every method has an async twin named with an `a` prefix (`acomplete`, `agenerate_response`, `agenerate_response_stream`, `aextract_field`, `aextract_bulk_data`), none abstract. The default runs the sync method on a worker thread, so a sync-only interface works from async code; `LiteLLMInterface` implements the twins natively over `acompletion` and its sync names are thin `run_sync` wrappers over them (`fsm_llm.aio`, a kernel module), so a sync caller needs no event loop and a patch of `completion` still intercepts the sync names.

`LiteLLMInterface` provides the built-in implementation supporting every provider litellm supports. It is the one place a provider request is built (`_build_call_params`) and sent (one send path that counts every call on the instance: `usage()`), for every call kind; `LiteLLMEmbedder` is the one embedding path. `litellm` is imported in exactly one module, `llm/providers/litellm.py`; nothing else in the package (agents, reasoning, meta-builder) calls a provider by itself. The classifier of a conversation sends its `complete` requests through the conversation's own `llm_interface`. Request models carry `user_message: str | None`; the request builder sends `None` (no user message) as the fixed `NEUTRAL_USER_TURN` and an empty or whitespace message as `EMPTY_USER_MESSAGE_TURN`, so the provider never sees an empty user turn and never reads an empty message as an instruction.

### TransitionEvaluator (`core/transition_evaluator.py`)

Three-outcome evaluation:
- **DETERMINISTIC** -- One passing transition, or a unique lowest `priority` value among several passing ones, proceed automatically
- **AMBIGUOUS** -- Two or more passing transitions tied at the lowest priority; only the tied group is delegated to LLM classification
- **BLOCKED** -- No matches, stay in current state

Uses JsonLogic conditions (`core/expressions/`) and required context key checks.

### Executables, results, names and patches (`core/`)

These core modules are imported by name. `import fsm_llm` loads only `core/names.py` and, through `fsm_llm.core.definitions`, the patch applier:

- `core/definitions/executable.py` is the executable envelope (a header around an FSM body, two identities, the variable rules) and `core/verify/executable.py` is its load sequence, `load_executable`. `fsm-llm-validate` imports the loader inside a function, only for a file whose header says it is an executable. See [Executables](executables.md).
- `core/results.py` is the `RunResult` and `Identity` envelope; `core/codes.py` holds the closed vocabularies (`Status`, `Reason`, `Lifecycle`, `MessageSource`, `CompletionKind`).
- `core/names.py` is `NameRegistry`. The engine builds its state-kind table from it at import, which is why it is loaded with the package.
- `core/definitions/patch.py` is the JSON patch applier, re-exported by `fsm_llm.core.definitions`.

See [Runs, results and names](results_and_names.md) for the rules of each.

### Executables run (`engine/`)

`engine/context.py` and `engine/executables/` sit above the `API` and are imported by name. The engine packages below them (`engine.api`, `engine.fsm`, `engine.turn`, `engine.kinds` and the others) import neither, an import contract in `.importlinter`, so `import fsm_llm` never loads them and a turn never depends on a registry.

- `engine/context.py` is `RunContext` and `CancelToken`: the frozen value a runner takes for one run (interface or factory, event log, limits, cancel token, the host objects a code-backed executable needs, the asking run).
- `engine/executables/` holds the `Executable` protocol, `FsmBodyExecutable` (one adapter for the kinds `fsm`, `strategy` and `procedure`: it owns a fresh `API` per run and drives the body with the message-free step loop), `CodeBackedExecutable` (the base of an executable whose body is code), `ExecutableRegistry` (serve by id, reload atomically, retain templates only, weighted aliases) and the two namespaces `executable_kinds` and `executable_sources`.
- The engine imports no extension package. A package declares its built-ins by registering a lazy source in `executable_sources` (`fsm_llm.agents` and `fsm_llm.reasoning` each register one when they are imported), and the registry calls the sources it is asked to load.
- `fsm_llm.workflows` reads the executable side through `engine.executables` and `engine.context`: `ExecutableStep` runs an executable by id inside a workflow.

See [Executables](executables.md) for the behaviour and its limits.

## Data Flow

### Message Processing

```
User Message → API.converse()
  → FSMManager.process_message() [acquires per-conversation lock]
    → MessagePipeline.process()
      → PRE_PROCESSING handlers
      → LLM: extract data from user input
      → Update context (merge extracted data)
      → CONTEXT_UPDATE handlers (if keys changed)
      → TransitionEvaluator: evaluate transitions
      → If transitioning: PRE_TRANSITION → state change → POST_TRANSITION
      → POST_PROCESSING handlers
      → LLM: generate response from NEW state
    → Return response
```

### Steps Without a User Message (`API.advance`, `API.run_until_terminal`)

```
API.advance(conversation_id)
  → FSMManager.advance() [same per-conversation lock and turn guard]
    → MessagePipeline.advance() → the same turn body as process(), user_message=None
      → no user exchange appended to history
      → Pass 1 prompts in their no-message wording (bulk prompt built from the
        scoped, security-filtered context; fills unset keys only)
      → transition evaluation and Pass 2 exactly as for a message
    → Return AdvanceResult(state_before, state_after, transition_outcome, response, ended)

API.run_until_terminal(conversation_id, *, max_steps, max_seconds=None, before_step=None,
                       seconds_exempt_states=())
  → per round: ended? → seconds budget → steps budget → before_step(n) → advance()
  → an ended PUSHED FSM on top: a call state's child is popped by settle first; any other:
    before_step(n) once (not a step) so the hook can pop it;
    the run then continues on the parent, or returns if nothing was popped
  → RunBudgetExceededError when a budget is spent; a conversation closed mid-run ends it normally
```

This is how the agents, the reasoning engine and any program that drives an FSM by itself run a conversation: no synthetic user message is ever sent. The reasoning engine runs one orchestrator run whose `execute_reasoning` state is a `call` state: the engine pushes the chosen strategy FSM as a frame and pops it when it ends or spends its 30 steps. `advance_stream` and `run_until_terminal_stream` are the streaming forms (reply text only).

### Completion States (tool calling and structured output)

```
State with "completion": {tools | response_format_key, tool_choice, instructions,
                          messages_key, result_key}
  Pass 1 → one LLMInterface.complete() over [system(instructions)] + context[messages_key]
         → context[result_key] = {kind: calls | final | malformed, text, calls: [{id, name, arguments}]}
  transitions read <result_key>.kind; a handler runs the calls, appends
  tool_exchange(content, calls, results) to the transcript and clears result_key
```

The transcript is a context list the consumer owns (an internal key), never the conversation history. Core runs no tool, adds no prompt text and no history, and refuses an unpaired transcript with `LLMResponseError` before sending (the turn rolls back). `NativeFunctionCallingReactAgent` (tool loop, forced final tool, repair turn) and the meta-builder's build call are completion states.

### Synthesize States (procedures written on demand)

```
State with "synthesize": {goal, inputs, return_keys, on_failure, result_key, min_lifecycle}
  Pass 1 → no extraction; the pipeline makes result_key handler-only
  ProcedureRunner (before_step hook of run_until_terminal, or around converse):
    start   → write {status: running} → obtain(goal) → library hit, or compile + gates G1 to G7
            → push_fsm(child, inherit_context=False) with only the declared inputs
    tick    → a child over policy.max_turns steps gets procedure_abort set (priority 0 abort edge)
    finish  → pop_fsm(context_to_return=return_keys) only on success
            → context[result_key] = {status: ok | failed | aborted, procedure_id, version, reason}
  transitions read <result_key>.status
```

Core only loads the field and checks the host FSM with the real evaluator (every edge reads the result, none fires before the runner has written one, the failure edge passes for `failed` and `aborted`). The runner is a consumer of the public step driver, as the reasoning engine is: it adds no seam to core. A bare `API.converse` on a synthesize state stays in it. See [synthesis.md](./synthesis.md).

### State Kinds and Executors

A state has exactly one kind, derived from the config it carries: `conversation` (the default: field, bulk and classification extraction, then a reply), `completion` (a `completion` config) or `synthesize` (a `synthesize` config). `State.kind` returns the name. It is a plain read-only property, not a field: it is absent from `model_dump`, `model_fields` and the JSON schema, so a definition cannot declare it and no `fsm_id` or library digest moves.

The extension point has two halves, and a new kind needs both:

- **Declaration** (core). A frozen `KindSpec` in `fsm_llm.core.state_kinds` names the kind, the `State` field that carries its config (`config_field`) and the rules the rest of core reads: `result_key`, `inherits_result_on_push`, `forbidden_declarations`, `check_state`, `check_definition` and `sample_results` (the loader), `dryrun_judged`, `mentioned_keys` and `typed_keys` (gate G6a), `minted_keys` and `produced_keys` (the pipeline, G3 and G4), `lint_raw` and `raw_result_key` (the validator, which reads raw dicts), `declared_tools`, `raw_declared_tools` and `supported_in_procedure` (synthesis). The registry (`KINDS`, `kind_of`, `get_kind`) is static and process-wide; `fsm_llm.core.state_kinds` is a leaf that imports only the standard library.
- **Execution** (engine). A `StateExecutor` in `fsm_llm.engine.kinds` says what the kind does in Pass 1: `extract(...)` produces the turn's extraction result, `cleans_extracted` says whether the data goes through `clean_context_keys`, and `after_transition(...)` runs when a transition lands in a state of the kind. The pipeline picks the executor with `executor_for(state, executors)` at the one Pass-1 seam that the sync and the stream path share. The executor table is per instance: `API(..., executors={kind_name: executor})`, passed down to `FSMManager` and `MessagePipeline`, is validated, copied read-only and visible to that `API` only. A kind with no executor (`synthesize`, which is passive, or a runtime kind nobody registered) runs as conversation, resolved through that `API`'s own `conversation` entry first and the module's second, so an `executors={"conversation": X}` override covers it too (a bare `converse` on a synthesize state stays in it, as before).

What the engine guarantees to an executor: `process()` takes the one snapshot of the turn and restores it on a failure (see Turn atomicity in `src/fsm_llm/AGENTS.md`), so an executor takes none of its own; handler-only keys (a kind's `result_key`) are never minted as extraction targets; Pass 2 stays shared and kind-agnostic. What it does not: the conversation Pass-1 methods (`_extract_after_transition`, `_execute_data_extraction` and their field, bulk and classification helpers) still live on `MessagePipeline` and the conversation executor calls them; they move with the module layout split (roadmap 0.4), they were not deleted. A test-only kind registered at runtime runs a full conversation with no edit in core (`tests/test_fsm_llm/test_executor_map.py`).

The module-level table of registered executors is a `NameRegistry` named `state_kinds` (`engine/kinds/registry.py`, see [Runs, results and names](results_and_names.md#name-registries)): `register_executor` replaces an entry on purpose, one executor instance cannot serve two kind names, and the registry declares no entry-point group yet.

Nothing here is public yet: `fsm_llm/__init__.py` exports neither module, `executors=` is not on the builders, and the entry-point group for third-party kinds arrives with the plugin work (roadmap C10, row 5.5). Until then both packages (`core.state_kinds` and `engine.kinds`) are internal. The `executors` keyword of `API`, `FSMManager` and `MessagePipeline` shows in the API snapshot fixture (`tests/fixtures/api_snapshot.txt`) as `Mapping[str, StateExecutor] | None`: `StateExecutor` is imported from `fsm_llm.engine.kinds` (not from `fsm_llm`) and is internal until the kind registry is exported, so the annotation names a class the package does not export, on purpose.

### Conversation Start

```
API.start_conversation(initial_context)
  → FSMManager: create FSM instance
  → START_CONVERSATION handlers
  → MessagePipeline: generate initial response from initial state
  → Return (conversation_id, response)
```

### Streaming (`API.converse_stream`)

```
API.converse_stream() → MessagePipeline.process_stream()
  → Pass 1 runs fully (extract → transition), identical to the synchronous path
  → Pass 2 streams the response token-by-token via LLMInterface.generate_response_stream()
  → yields str chunks to the caller
```

States with an empty `response_instructions` skip Pass 2 entirely: no LLM call, the empty
reply (`""` from `start_conversation`/`converse`, no chunk from a stream, `response=None`
from `advance`) and nothing added to the history. This is the common shape for
intermediate agent/tool-dispatch states.

## Context Handling

```python
class FSMContext:
    data: dict           # User-defined context (extracted data, handler outputs)
    conversation: Conversation  # Message history with max_history_size
    metadata: dict       # System metadata
    working_memory: WorkingMemory  # Named buffers (core, scratch, environment, reasoning)
```

`get_user_visible_data()` (which merges the non-hidden buffers into `data`, minus internal keys and secret-looking entries; the one key policy lives in `security/`, so a key named exactly `system` is an ordinary visible key, unlike `system_*`) enters the Pass-1 per-field extraction prompt (`_execute_field_extractions`, when a field's `context_keys` is left at its default) and both classifier call sites. The Pass-2 sites (`_execute_response_generation_pass`, `_stream_response_generation_pass`, and `generate_initial_response` when a greeting is generated) build the `<current_context>` block from `get_merged_data()`, scoped by `read_keys`; the prompt is the only context channel, since `ResponseGenerationRequest` carries no context field. `update_context`/handlers write `data` only; a value that must reach the shipped response prompt belongs in `data`. `working_memory` is also `exclude=True`, so `FSMContext.model_dump()` omits it (sessions persist it through `WorkingMemory.to_dict()`).

Special keys (prefixed with `_`): `_conversation_id`, `_conversation_start`, `_fsm_id`, `_timestamp`, and, once a transition has happened, `_current_state`, `_previous_state`, `_transition_timestamp`. ERROR handlers additionally see `_error` and `_traceback`. Formats: `_conversation_start` is naive local-time ISO text (`datetime.now().isoformat()`), `_timestamp` and `_transition_timestamp` are epoch seconds (floats), and none of the three is persisted by `save_session` (UTC-aware values are carried, not built). The raw user message is not in the context, so handlers see extracted keys only.

### Context in Stacked FSMs

On `push_fsm`: parent context optionally inherited/passed to child.
On `pop_fsm`: only what was declared is merged back into the parent, the child's other keys are dropped: the child's values for `shared_context_keys`, plus the `return_context` given to `push_fsm` and the `context_to_return` given to `pop_fsm`. The merge strategy only decides collisions on those keys:
- **UPDATE** -- Returned value overwrites the parent's
- **PRESERVE** -- Only keys the parent does not have yet are added

## Handler Execution Pipeline

```
START_CONVERSATION → [per message: PRE_PROCESSING → extract → CONTEXT_UPDATE
→ evaluate → PRE_TRANSITION → transition → POST_TRANSITION → POST_PROCESSING
→ generate response] → END_CONVERSATION. ERROR on an exception raised while processing a message (`converse` / `converse_stream`; not for a failure inside `start_conversation`).
```

Priority ordering (lower first). Error modes: `"continue"` (log + skip) or `"raise"` (propagate). Critical handlers always raise.

## Security

- **Prompt injection prevention**: XML tag sanitization on all user input
- **Context isolation**: Per-conversation isolated state, no cross-conversation access
- **Input validation**: Length limits, sanitization
- **Forbidden patterns**: Regex filtering for passwords, secrets, API keys, tokens
- **Internal key prefixes**: `_`, `system_`, `internal_`, `__` stripped from user-facing context

## Performance

- **Conversation history limits**: `max_history_size` (default 5) is the number of exchanges stored per conversation and also the number of messages the extraction and response prompts show (the last N, passed unchanged to the default prompt builders, so the unit of the shown window is messages); the history token budget (default 3000) can cut the window below N, so at most N messages are shown. `max_message_length` (default 1000)
- **FSM definition caching**: LRU cache (max 64) for loaded definitions
- **Handler pre-filtering**: Only check handlers matching current timing
- **Concurrency**: Per-conversation `TurnGate`s in FSMManager (one turn per conversation, per thread and per task; a second turn is refused). One loop serves many conversations: while a model call is suspended other tasks run. Handlers run inline on the loop. A cancelled turn is rolled back whole (Pass 1 included) and runs no ERROR handler. With the event log on, the log writes and their fsyncs run inline on the loop (about 76 ms per logged turn on the measured NVMe): a known limit

## Package Layout

The library is one import package, `fsm_llm`, in `src/fsm_llm/`. Beneath the extension subpackages sit layer packages that ship with every install, and a layer imports only the layers below it. The rule is enforced by `.importlinter` (`make lint-imports`, and the test `tests/test_fsm_llm_regression/test_import_contract.py`), and no module is over 800 lines.

| Layer (top to bottom) | Packages | What it holds |
|---|---|---|
| Subpackages | `monitor`, `eval`; `agents`, `synthesis`, `workflows`, `reasoning`; `testing` | One extra each, imported by name, never by `import fsm_llm` |
| `engine` | `fsm_llm.engine` | `API` (`engine.api`), `FSMManager` (`engine.fsm`), the 2-pass `MessagePipeline` (`engine.turn`), the state-kind executors (`engine.kinds`), `Classifier`, the builders, the session store, the plain-FSM OKF handler (`engine.okf_bridge`), and, above the API, the run context (`engine.context`) and the executables (`engine.executables`), both loaded by name only |
| `handlers` | `fsm_llm.handlers` | The handler system, `HandlerBuilder`, `create_handler` |
| `policy` | `fsm_llm.policy` | Approval types and the human-in-the-loop manager; imports `tools` and none of `context`, `okf`, `skills` (the contract `policy-imports-no-sibling-layer`: the row sits above the siblings, so only the contract keeps it off them) |
| Independent siblings | `context`, `okf`, `skills`, `tools` | Context compaction and prompt builders (`context.layout`); Open Knowledge Format; Agent Skills; the tool registry and tool models |
| `runtime` | `fsm_llm.runtime` | The opt-in durable event log: `EventLog` (`runtime/log.py`), the JSONL store (`runtime/stores/jsonl.py`), `BlobStore` (`runtime/blobs.py`), the format (`runtime/events.py`), the turn scope (`runtime/scope.py`), the session and stack projections (`runtime/stores/session.py`, `runtime/stores/stack.py`: the folds `restore_session`, `API.fork` and `API.resume` read), privacy levels and the redaction hook (`runtime/privacy.py`), explicit retention (`runtime/retention.py`) and the usage fold (`runtime/usage.py`); imports `core`, `llm` and the kernel only; the log directory is as private as a session directory |
| `llm` | `fsm_llm.llm` | `LLMInterface`, `LiteLLMInterface` (the one litellm import), the embedder, reply parsing |
| `core` | `fsm_llm.core` | Pydantic FSM models (`core.definitions`), JsonLogic (`core.expressions`), `core.transition_evaluator`, `core.state_kinds`, `core.verify` (validator, data-flow and liveness analyses, the executable load sequence), `core.visualize`, `core.results`, `core.names`, `core.codes`, the executable envelope and the JSON patch applier in `core.definitions` |
| Kernel | `constants`, `logging`, `security`, `text`, `env`, `frontmatter`, `memory`, `cli_support` | Small modules any layer may import; they import no layer |

`import fsm_llm` loads the `core`, `engine`, `handlers`, `context`, `llm`, `runtime` and `security` packages and a few kernel modules, and nothing else: no subpackage and none of `okf`, `skills`, `tools` or `policy` (pinned by `tests/test_fsm_llm_regression/test_import_laziness.py`). A few edges break the order on purpose and are named in `.importlinter`: `agents` imports `reasoning` (for `ReasoningReactAgent`), and `synthesis` imports `agents.semantic_tools` and `eval` (the simulator reuses the scripted-trial code).

## Extension Integration

```
fsm_llm (core, includes classification)
├── fsm_llm.reasoning  -- Uses API run_until_terminal; its execute_reasoning call state pushes/pops strategy FSMs
├── fsm_llm.workflows  -- Uses API (via ConversationStep); lifecycle hooks via add_hook
├── fsm_llm.agents     -- Uses API (auto-generates FSMs, drives them with run_until_terminal)
│                          + handlers for tool execution; native_fc and the meta-builder use completion states
├── fsm_llm.monitor    -- Uses API + handlers (observer callbacks at priority 9999)
├── fsm_llm.eval       -- Runs example scripts as subprocesses; drives scripted conversations
│                          through API (one instance per trial) and checks the outcome
├── fsm_llm.synthesis  -- Uses API (push_fsm/pop_fsm, run_until_terminal's before_step hook), agents
│                          (tools, human approval) and eval (scripted trials for simulation); core
│                          loads its `synthesize` state field and runs the data-flow and liveness
│                          analyses (`core/verify/flow.py`)
└── fsm_llm.testing    -- Public test doubles, scripted user and assertions; imports core only
```

The extensions are subpackages of `fsm_llm`. Core never imports them at package
import time; each one imports core (`from fsm_llm import API`). Their place among the
layers is in Package Layout above.

| Package | Integration | Key Mechanism |
|---------|------------|---------------|
| Classification | Built into core | Structured `complete` call through the conversation's `llm_interface` |
| Reasoning | One message-free run + a `call` state | `run_until_terminal` on the orchestrator; the `execute_reasoning` call state (`choose` on the handler-written `classified_problem_type`, fallback `reasoning.analytical`) pushes a strategy FSM as a frame and pops it when it ends; a strategy over its 30 steps ends `aborted/budget_spent` and the solve still answers; a spent 170-step budget raises `ReasoningExecutionError` |
| Workflows | Async engine + ConversationStep | ConversationStep creates API instance for FSM conversations |
| Agents | Auto-generated FSMs + handlers, driven by core `run_until_terminal` | `build_react_fsm()` generates FSM; handlers execute tools on state entry; core owns the step and time budgets; no synthetic message is sent. `native_fc` runs on core completion states (native tool calls); the meta-builder is an FSM driven with `converse`/`advance`. Also covers multi-agent graph/swarm orchestration, MCP tools, A2A remote agents, and semantic tool retrieval |
| Monitor | Observer handlers + loguru sink + core graph data | Registers observer handlers (priority 9999; not POST_TRANSITION), never modifies state; FSM graphs come from `build_fsm_graph` |
| Eval | Subprocesses + public API only | `fsm-llm-eval examples` scores example output with a 0-4 heuristic; `fsm-llm-eval run` sends each case's turns through a fresh `API`, reads `get_current_state`/`get_data`/`has_conversation_ended`, and checks declared expectations over N trials. No handlers, no core changes |
| Synthesis | A silent `synthesize` state + a runner on the core step driver | A model compiles a goal into an FSM, gates G1 to G7 judge it, a content-addressed library stores it, and `ProcedureRunner` pushes it as a child (no inherited context), runs it on the host model and merges back only the declared return keys. Core adds the `synthesize` state field, the loader checks of its edges and `core/verify/flow.py`; no handler and no second engine |

## Extension Points

### Custom LLM Interface

```python
from fsm_llm.constants import NEUTRAL_USER_TURN

class CustomLLM(LLMInterface):
    def generate_response(self, request: ResponseGenerationRequest) -> ResponseGenerationResponse:
        # user_message is None when there is no user message (a message-free
        # step, a greeting); "" or any string is what the user sent. A silent
        # state (empty response_instructions) never reaches this method.
        user_turn = request.user_message if request.user_message is not None else NEUTRAL_USER_TURN
        response = your_api(request.system_prompt, user_turn)
        return ResponseGenerationResponse(message=response)

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        ...

    def extract_bulk_data(self, request: BulkExtractionRequest) -> DataExtractionResponse:
        ...  # needed by states with extraction_instructions; the base class raises NotImplementedError

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        ...  # needed by classification and completion states; API refuses a definition with
             # classification_extractions when this is not implemented
```

### Custom Handlers

```python
class DatabaseHandler(BaseHandler):
    def should_execute(self, timing, current_state, target_state, context, updated_keys):
        return timing == HandlerTiming.POST_TRANSITION

    def execute(self, context):
        database.save(context["_conversation_id"], context)
        return {"saved_to_db": True}
```

### State Kinds (internal)

A new state kind is a `KindSpec` in `fsm_llm.core.state_kinds` plus a `StateExecutor` supplied with `executors={kind_name: executor}` on `API`, `FSMManager` or `MessagePipeline`; see State Kinds and Executors above. It is not a public extension point yet: no export from `fsm_llm`, no builder setter, no entry-point group before roadmap row 5.5.

## Next steps

- [Concepts](concepts.md) is the shorter tour of the same ideas.
- [Handler Development](handlers.md) covers the extension points of a turn.
- [Runtime FSM synthesis](synthesis.md) covers the `synthesize` state kind.
- [Executables](executables.md) and [Runs, results and names](results_and_names.md) cover the core contracts above.
- The [API Reference](api_reference.md) lists the public names of every package.
