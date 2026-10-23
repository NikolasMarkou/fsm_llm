# Architecture Deep Dive

> Covers FSM-LLM v0.12.0

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
`converse()` and restored via `restore_session()`.

## Core Components

### API (`api.py`)

User-facing entry point. Manages FSM definition processing, component wiring, conversation lifecycle, and FSM stacking.

### FSMManager (`fsm.py`) + MessagePipeline (`pipeline.py`)

FSMManager handles FSM instance lifecycle with per-conversation thread locks and LRU FSM cache. MessagePipeline implements the 2-pass processing:

```
Pass 1: PRE_PROCESSING handlers → LLM extract data → CONTEXT_UPDATE handlers
      → Evaluate transitions → PRE_TRANSITION → State change → POST_TRANSITION

Pass 2: POST_PROCESSING handlers → LLM generate response → Return
```

### Prompt Builders (`prompts.py`)

Two builders produce structured XML-like prompts, plus the Pass-1 builder slot:
- `DataExtractionPromptBuilder` -- the Pass-1 slot of `FSMManager`/`MessagePipeline`; the pipeline builds the bulk extraction prompt inline and uses this builder only for the shared sanitizer
- `ResponseGenerationPromptBuilder` -- Pass 2: user-facing response generation
- `FieldExtractionPromptBuilder` -- Targeted single-field extraction

Security: XML tag sanitization, CDATA wrapping for JSON, token budget management.

### Handler System (`handlers.py`)

Plugin architecture with 8 timing points. Handlers self-determine execution via state/context conditions, execute in priority order, with configurable error isolation (`"continue"` or `"raise"`). Critical handlers always raise on error.

### LLM Interface (`llm.py`)

`LLMInterface` ABC with the methods:
- `generate_response(request)` -- Generate user-facing response (Pass 2)
- `extract_field(request)` -- Extract a targeted field from input (Pass 1)
- `generate_response_stream(request)` -- Stream the Pass-2 response token-by-token
- `extract_bulk_data(request)` -- Bulk extraction for states with `extraction_instructions`
- `complete(request: CompletionRequest) -> CompletionResponse` -- The one primitive for tool calling, plain and structured completion (classifier, completion states, the agents' LLM judge). Not abstract: the base method raises `NotImplementedError`

`LiteLLMInterface` provides the built-in implementation supporting 100+ providers. It is the one place a provider request is built (`_build_call_params`) and sent (one send path that counts every call on the instance: `usage()`), for every call kind; `LiteLLMEmbedder` is the one embedding path. `litellm` is imported only in `llm.py`; nothing else in the package (agents, reasoning, meta-builder) calls a provider by itself. The classifier of a conversation sends its `complete` requests through the conversation's own `llm_interface`. Request models carry `user_message: str | None`; the request builder sends `None` (no user message) as the fixed `NEUTRAL_USER_TURN` and an empty or whitespace message as `EMPTY_USER_MESSAGE_TURN`, so the provider never sees an empty user turn and never reads an empty message as an instruction.

### TransitionEvaluator (`transition_evaluator.py`)

Three-outcome evaluation:
- **DETERMINISTIC** -- One passing transition, or a unique lowest `priority` value among several passing ones, proceed automatically
- **AMBIGUOUS** -- Two or more passing transitions tied at the lowest priority; only the tied group is delegated to LLM classification
- **BLOCKED** -- No matches, stay in current state

Uses JsonLogic conditions (`expressions.py`) and required context key checks.

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
  → an ended PUSHED FSM on top: before_step(n) once (not a step) so the hook can pop it;
    the run then continues on the parent, or returns if nothing was popped
  → RunBudgetExceededError when a budget is spent; a conversation closed mid-run ends it normally
```

This is how the agents, the reasoning engine and any program that drives an FSM by itself run a conversation: no synthetic user message is ever sent. The reasoning engine runs one orchestrator run whose `before_step` hook pushes and pops the strategy FSMs. `advance_stream` and `run_until_terminal_stream` are the streaming forms (reply text only).

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

`get_user_visible_data()` (which merges the non-hidden buffers into `data`, minus internal keys) enters the Pass-1 per-field extraction prompt (`_execute_field_extractions`, when a field's `context_keys` is left at its default) and both classifier call sites. The three Pass-2 sites (`_execute_response_generation_pass`, `_stream_response_generation_pass`, and `generate_initial_response` when a greeting is generated) build the `<current_context>` block from `get_merged_data()`, scoped by `read_keys`; the prompt is the only context channel, since `ResponseGenerationRequest` carries no context field. `update_context`/handlers write `data` only; a value that must reach the shipped response prompt belongs in `data`. `working_memory` is also `exclude=True`, so `FSMContext.model_dump()` omits it (sessions persist it through `WorkingMemory.to_dict()`).

Special keys (prefixed with `_`): `_conversation_id`, `_conversation_start`, `_fsm_id`, `_timestamp`, and, once a transition has happened, `_current_state`, `_previous_state`, `_transition_timestamp`. ERROR handlers additionally see `_error` and `_traceback`. The raw user message is not in the context, so handlers see extracted keys only.

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

- **Conversation history limits**: `max_history_size` (default 5), `max_message_length` (default 1000)
- **FSM definition caching**: LRU cache (max 64) for loaded definitions
- **Handler pre-filtering**: Only check handlers matching current timing
- **Thread safety**: Per-conversation RLocks in FSMManager

## Extension Integration

```
fsm_llm (core, includes classification)
├── fsm_llm.reasoning  -- Uses API run_until_terminal; its before_step hook pushes/pops strategy FSMs
├── fsm_llm.workflows  -- Uses API (via ConversationStep); lifecycle hooks via add_hook
├── fsm_llm.agents     -- Uses API (auto-generates FSMs, drives them with run_until_terminal)
│                          + handlers for tool execution; native_fc and the meta-builder use completion states
├── fsm_llm.monitor    -- Uses API + handlers (observer callbacks at priority 9999)
└── fsm_llm.eval       -- Runs example scripts as subprocesses; drives scripted conversations
                           through API (one instance per trial) and checks the outcome
```

The five extensions are subpackages of `fsm_llm`. Core never imports them at package
import time; each one imports core (`from fsm_llm import API`).

| Package | Integration | Key Mechanism |
|---------|------------|---------------|
| Classification | Built into core | Structured `complete` call through the conversation's `llm_interface` |
| Reasoning | One message-free run + FSM stacking | `run_until_terminal` on the orchestrator; the `before_step` hook pushes a strategy FSM by type and pops it when it ends (or after 30 steps); a spent 170-step budget raises `ReasoningExecutionError` |
| Workflows | Async engine + ConversationStep | ConversationStep creates API instance for FSM conversations |
| Agents | Auto-generated FSMs + handlers, driven by core `run_until_terminal` | `build_react_fsm()` generates FSM; handlers execute tools on state entry; core owns the step and time budgets; no synthetic message is sent. `native_fc` runs on core completion states (native tool calls); the meta-builder is an FSM driven with `converse`/`advance`. Also covers multi-agent graph/swarm orchestration, MCP tools, A2A remote agents, and semantic tool retrieval |
| Monitor | Observer handlers + loguru sink + core graph data | Registers observer handlers (priority 9999; not POST_TRANSITION), never modifies state; FSM graphs come from `build_fsm_graph` |
| Eval | Subprocesses + public API only | `fsm-llm-eval examples` scores example output with a 0-4 heuristic; `fsm-llm-eval run` sends each case's turns through a fresh `API`, reads `get_current_state`/`get_data`/`has_conversation_ended`, and checks declared expectations over N trials. No handlers, no core changes |

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
