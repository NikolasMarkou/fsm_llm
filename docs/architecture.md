# Architecture Deep Dive

> Covers FSM-LLM v0.5.0

Technical overview of the FSM-LLM architecture and how components work together.

## System Overview

FSM-LLM uses an **improved 2-pass architecture** that separates data extraction, transition evaluation, and response generation:

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

Three builders produce structured XML-like prompts:
- `DataExtractionPromptBuilder` -- Pass 1: data extraction from user input
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

`LiteLLMInterface` provides the built-in implementation supporting 100+ providers.

### TransitionEvaluator (`transition_evaluator.py`)

Three-outcome evaluation:
- **DETERMINISTIC** -- Exactly one transition matches, proceed automatically
- **AMBIGUOUS** -- Multiple matches, delegate to LLM classification
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

States with an empty `response_instructions` skip Pass 2 entirely (no response LLM call),
which is the common shape for intermediate agent/tool-dispatch states.

## Context Handling

```python
class FSMContext:
    data: dict           # User-defined context (extracted data, handler outputs)
    conversation: Conversation  # Message history with max_history_size
    metadata: dict       # System metadata
    working_memory: WorkingMemory  # Named buffers (core, scratch, environment, reasoning)
```

Special keys (prefixed with `_`): `_conversation_id`, `_current_state`, `_previous_state`, `_timestamp`, `_user_input`.

### Context in Stacked FSMs

On `push_fsm`: parent context optionally inherited/passed to child.
On `pop_fsm`: child context merged back via strategy:
- **UPDATE** -- Child context overwrites parent
- **PRESERVE** -- Only new keys added to parent

## Handler Execution Pipeline

```
START_CONVERSATION → [per message: PRE_PROCESSING → extract → CONTEXT_UPDATE
→ evaluate → PRE_TRANSITION → transition → POST_TRANSITION → POST_PROCESSING
→ generate response] → END_CONVERSATION. ERROR on any exception.
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
├── fsm_llm_reasoning  — Uses API (push/pop FSM stacking) + classification
├── fsm_llm_workflows  — Uses HandlerSystem + API (via ConversationStep)
├── fsm_llm_agents     — Uses API (auto-generates FSMs) + handlers for tool execution
├── fsm_llm_monitor    — Uses API + handlers (observer callbacks at priority 9999)
└── fsm_llm_harness    — Uses API (hand-written FSM) + handlers at state entry; dispatches
                          fsm_llm_agents workers as protocol roles
```

| Package | Integration | Key Mechanism |
|---------|------------|---------------|
| Classification | Built into core | LLM-backed via litellm |
| Reasoning | FSM stacking via push/pop | Orchestrator pushes strategy FSMs onto stack |
| Workflows | Async engine + ConversationStep | ConversationStep creates API instance for FSM conversations |
| Agents | Auto-generated FSMs + handlers | `build_react_fsm()` generates FSM; handlers execute tools at POST_TRANSITION. Also covers multi-agent graph/swarm orchestration, MCP tools, A2A remote agents, and semantic tool retrieval |
| Monitor | Observer handlers + loguru sink | Registers at all 8 timing points (priority 9999), never modifies state |
| Harness | Hand-written FSM + state-entry handlers | `build_harness_fsm()` returns a 6-state definition whose gates are JsonLogic conditions; a handler per state entry dispatches one agent worker, and the gate values it writes are derived from the filesystem |

## The Harness: a Protocol on Top of the 2-Pass Core (`fsm_llm_harness`)

The harness runs a 6-state planning protocol -- EXPLORE, PLAN, EXECUTE, REFLECT,
PIVOT, CLOSE -- as an ordinary FSM-LLM conversation. It adds no machinery to the
core: it is a definition, a set of handlers, and a filesystem layer.

### Where it sits in the 2-pass flow

Each protocol turn is one `converse()` call, so it runs both passes. Two
deliberate choices reduce that to exactly ONE core LLM call per turn -- Pass 2's
response generation -- with Pass 1 issuing none at all:

- **No state carries `extraction_instructions`.** Pass 1's additive
  bulk-extraction call fires on the presence of that string alone, so the harness
  states omit it entirely. Measured live: 2.000 -> 1.000 core LLM calls per turn.
- **Every driver-owned context key is seeded before turn 1** -- the nine gate
  flags plus the counters and rollups, sixteen in all. Core mints a required
  extraction config for every key a transition condition names in
  `requires_context_keys`, and skips a field already present in context. Seeding
  them falsy means the model is never asked to invent a gate value, and it also
  removes the per-turn extraction call each of those keys would otherwise cost.

The second point is a security property, not an optimisation. Before the seeds
existed, an LLM emitting `{"plan_approved": true, "close_confirmed": true}` in
Pass 1 drove a full traverse to CLOSE while every worker dispatch failed and a
DENYING approval callback was never consulted once.

### Driver and worker: two different actors

```
HarnessAgent (the DRIVER)              worker (a ROLE)
├── owns the FSM conversation          ├── an fsm_llm_agents agent
├── owns all nine gate flags           ├── receives a RoleRequest (context SNAPSHOT)
├── writes exactly one artifact        ├── holds only the tools its ownership
│     (state.md), on transitions       │     entry grants it
└── dispatches one worker per          └── returns an AgentResult; its keys pass
      state entry                            an exact-type allowlist to reach context
```

A worker cannot reach the driver: `run`, `api` and `conversation_id` raise
`HarnessReentrancyError` while a dispatch is in flight, and the re-dispatch and
leash counters are driver run state rather than context keys -- so no worker can
refill its own budget from inside its own callback.

### Gates read the disk, not the reply

This is the design commitment the rest follows from.

- `findings_count` is a count of non-empty `findings/*.md` files, read back
  through the same role-scoped `PlanMemory` the write tools wrote through. A
  worker-supplied integer is popped from the payload and treated as advisory.
- A dispatch that HOLDS a write tool must show a write-tool call in its trace
  whose target now carries BYTES. Tool-name presence alone is not enough: a
  `write_plan_file` the ownership layer refused appears in the trace and leaves
  nothing.
- A failed observation never fabricates a zero. No plan directory, a directory
  that is not there yet, a path that is a file, or an I/O error all leave the gate
  value UNCHANGED -- absence of evidence is not evidence of absence.

These mechanisms exist because measurement demanded them, not because they were
elegant: a 4B model returned "implemented retry-with-backoff in uploader.py"
5 runs out of 5 having written zero bytes, and a payload claiming
`findings_count: 3` over an empty `findings/` directory. Three successive attempts
to fix this with prompt WORDING moved nothing; the mechanical check moved it
immediately.

### Filesystem as memory

A plan directory holds 15 artifact kinds with strict Markdown grammars
(`plan.md`'s 11 sections in exact order, `decisions.md`'s
`## D-NNN | PHASE | YYYY-MM-DD` header, `changelog.md`'s 8 pipe-delimited
fields). Two roots are confined independently -- `Workspace` over the source tree,
`PlanMemory` over the plan directory -- through one `resolve()` chokepoint that
resolves first and compares second. `PlanMemory` adds an ownership check from a
single `OWNERSHIP` table that also derives each role's tool scope and prompt text,
so what a role is TOLD it may write and what it CAN write are one fact.

Writes are atomic (`mkstemp` in the target's own directory, then `os.replace`),
because a torn `state.md` still parses and a truncated Fix-Attempts section reads
as a leash that reset itself.

### Safety bounds

| Bound | Value | Enforced by |
|---|---|---|
| Autonomy leash | 2 fix attempts per plan step | HARD JsonLogic gate + the pre-step gate |
| Leash grants | 2 human continues per plan step | driver counter; executor dispatches per step are bounded by `attempts x (1 + grants)` for any approval sequence |
| Iteration cap | 6 | HARD gate on PLAN -> EXECUTE |
| EXPLORE re-dispatches | 9 extra per run | driver run state, reset only per run |
| Human gates | 4 (approve plan, confirm close, continue after leash, revert) | `HumanInTheLoop`; the default callback DENIES |

The `leash-cap` revert is COMPUTED and scoped -- never touching the plan
directory -- and reported, but executed only by an explicitly supplied callback
that the approval gate has already granted. `git` is deliberately absent from the
command allowlist, so the driver never shells out to it.

## Extension Points

### Custom LLM Interface

```python
class CustomLLM(LLMInterface):
    def generate_response(self, request: ResponseGenerationRequest) -> ResponseGenerationResponse:
        response = your_api(request.system_prompt, request.user_message)
        return ResponseGenerationResponse(message=response)

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        ...
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
