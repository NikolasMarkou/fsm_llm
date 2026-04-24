# Architecture Deep Dive

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
│              Storage (In-Memory Dicts)                │
└──────────────────────────────────────────────────────┘
```

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

`LLMInterface` ABC with two methods:
- `generate_response(request)` -- Generate user-facing response
- `extract_field(request)` -- Extract a targeted field from input

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
    → MessagePipeline.process_compiled()
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
└── fsm_llm_monitor    — Uses API + handlers (observer callbacks at priority 9999)
```

| Package | Integration | Key Mechanism |
|---------|------------|---------------|
| Classification | Built into core | LLM-backed via litellm |
| Reasoning | FSM stacking via push/pop | Orchestrator pushes strategy FSMs onto stack |
| Workflows | Async engine + ConversationStep | ConversationStep creates API instance for FSM conversations |
| Agents | Auto-generated FSMs + handlers | `build_react_fsm()` generates FSM; handlers execute tools at POST_TRANSITION |
| Monitor | Observer handlers + loguru sink | Registers at all 8 timing points (priority 9999), never modifies state |

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
