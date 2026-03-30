# Strands Feature Adaptation -- Phase 1 (Implemented)

**Status**: Complete (commit `a7e3d88`)
**Tests**: 33 new tests, 2232 total passing

Phase 1 delivers 4 core infrastructure features adapted from the Strands Agents SDK. These were selected for highest value at lowest risk, and they establish the foundation that Phase 2 features build on.

---

## 1. Schema-Enforced Output (Strands Feature #12)

**What it does**: When `AgentConfig.output_schema` is set and the LLM provider supports it, FSM-LLM passes a JSON schema to the provider via `response_format` for constrained decoding. This guarantees valid structured output instead of relying on post-hoc parsing.

**How it works**:
- `ResponseGenerationRequest` gained a `response_format` field
- `LiteLLMInterface.generate_response()` threads it to `_make_llm_call()`
- `_make_llm_call()` applies it when the provider supports `response_format`
- Agents auto-set it: `BaseAgent._init_context()` stores the Pydantic model's JSON schema in `_output_response_format` context key
- The pipeline reads it when constructing the Pass 2 request
- Falls back gracefully to the existing 3-tier validation (context keys -> JSON parsing -> observation scan) for providers that don't support constrained output

**Files changed**:
- `src/fsm_llm/definitions.py` -- `response_format` field on `ResponseGenerationRequest`
- `src/fsm_llm/llm.py` -- `response_format` param on `_make_llm_call()`, applied in `generate_response()`
- `src/fsm_llm/pipeline.py` -- Reads `_output_response_format` from context
- `src/fsm_llm_agents/base.py` -- Sets `_output_response_format` from `output_schema`

**Usage**:
```python
from pydantic import BaseModel
from fsm_llm_agents import ReactAgent, AgentConfig

class Analysis(BaseModel):
    summary: str
    risk_level: str
    confidence: float

agent = ReactAgent(
    config=AgentConfig(output_schema=Analysis),
    tools=[...],
)
result = agent.run("Analyze this report...")
result.structured_output  # Analysis instance (enforced by LLM, not just parsed)
```

---

## 2. Invocation State -- Hidden Metadata Buffer (Strands Feature #11)

**What it does**: Adds a `metadata` buffer to `WorkingMemory` that is excluded from all LLM-visible aggregate views. Orchestration metadata (user permissions, billing tier, retry counts, routing decisions) flows through multi-agent pipelines without polluting LLM context.

**How it works**:
- New `BUFFER_METADATA = "metadata"` constant
- `WorkingMemory` gains a `_hidden_buffers` frozenset (defaults to `{"metadata"}`)
- `get_all_data()` skips hidden buffers during merge
- `to_scoped_view()` skips hidden buffers (uses `get_all_data()`)
- `search()` skips hidden buffers
- Direct access via `get()`/`set()` still works
- `to_dict()` includes hidden buffers (serialization preserves all data)
- `from_dict()` accepts `hidden_buffers` parameter

**Files changed**:
- `src/fsm_llm/memory.py` -- Hidden buffer concept, `BUFFER_METADATA`, `DEFAULT_HIDDEN_BUFFERS`
- `src/fsm_llm/__init__.py` -- Export `BUFFER_METADATA`

**Usage**:
```python
from fsm_llm import WorkingMemory, BUFFER_METADATA

memory = WorkingMemory(buffers=("core", "scratch", "metadata"))

# Set orchestration metadata (hidden from LLM)
memory.set("metadata", "user_tier", "premium")
memory.set("metadata", "billing_id", "B-123")
memory.set("metadata", "retry_count", 0)

# Set LLM-visible data
memory.set("core", "user_name", "Alice")

# LLM only sees core/scratch data
all_data = memory.get_all_data()  # {"user_name": "Alice"} -- no metadata!

# Direct access still works for application code
tier = memory.get("metadata", "user_tier")  # "premium"
```

---

## 3. Token-Level Response Streaming (Strands Feature #2)

**What it does**: Adds streaming support for Pass 2 (response generation). Pass 1 (extraction + transitions) always runs to completion. Pass 2 yields response tokens as they arrive from the LLM.

**How it works**:
- `LLMInterface` gained `generate_response_stream()` (default: falls back to `generate_response()`)
- `LiteLLMInterface.generate_response_stream()` uses `completion(stream=True)` and yields `chunk.choices[0].delta.content`
- `MessagePipeline.process_stream()` runs Pass 1 fully, then streams Pass 2
- `FSMManager.process_message_stream()` handles locking and delegates to pipeline
- `API.converse_stream()` is the user-facing entry point

**Files changed**:
- `src/fsm_llm/llm.py` -- `generate_response_stream()` on both `LLMInterface` and `LiteLLMInterface`
- `src/fsm_llm/pipeline.py` -- `process_stream()` and `_stream_response_generation_pass()`
- `src/fsm_llm/fsm.py` -- `process_message_stream()`
- `src/fsm_llm/api.py` -- `converse_stream()`

**Usage**:
```python
from fsm_llm import API

api = API.from_file("chatbot.json", model="gpt-4o")
conv_id, greeting = api.start_conversation()

# Stream response tokens
for chunk in api.converse_stream("Tell me about Python", conv_id):
    print(chunk, end="", flush=True)
print()  # newline after stream completes
```

---

## 4. Session Persistence (Strands Feature #9)

**What it does**: Adds a `SessionStore` interface for saving and restoring conversation state across process restarts. Ships with `FileSessionStore` (JSON files). Auto-saves after each `converse()` call when configured.

**How it works**:
- `SessionState` Pydantic model captures: conversation_id, fsm_id, current_state, context_data, conversation_history, stack_depth, timestamp
- `SessionStore` ABC defines: `save()`, `load()`, `delete()`, `list_sessions()`, `exists()`
- `FileSessionStore` writes atomic JSON files (temp + rename) with path traversal protection
- `API` gains `session_store=` constructor parameter
- `converse()` auto-saves after each call (failure to save logs warning, doesn't block)
- `save_session()` / `load_session()` / `restore_session()` for explicit control

**Files changed**:
- `src/fsm_llm/session.py` -- NEW: `SessionStore`, `FileSessionStore`, `SessionState`
- `src/fsm_llm/api.py` -- `session_store` param, auto-save, save/load/restore methods
- `src/fsm_llm/__init__.py` -- Export `SessionStore`, `FileSessionStore`, `SessionState`

**Usage**:
```python
from fsm_llm import API, FileSessionStore

# Configure with session persistence
store = FileSessionStore("./sessions")
api = API.from_file("bot.json", model="gpt-4o", session_store=store)

# Start conversation (auto-saves after each converse)
conv_id, greeting = api.start_conversation()
response = api.converse("Hello!", conv_id)  # auto-saved

# Later, in a new process:
api = API.from_file("bot.json", model="gpt-4o", session_store=store)
result = api.restore_session(conv_id)
if result:
    restored_conv_id, state = result
    # Continue conversation with restored context
    response = api.converse("Where were we?", restored_conv_id)

# List all saved sessions
sessions = store.list_sessions()

# Explicit save/load
api.save_session(conv_id)
state = api.load_session(conv_id)
```

---

## Summary

| Feature | Files | LOC | Tests |
|---------|-------|-----|-------|
| Schema-Enforced Output | 4 modified | ~60 | 6 |
| Invocation State | 2 modified | ~40 | 8 |
| Response Streaming | 4 modified | ~180 | 7 |
| Session Persistence | 3 modified + 1 new | ~180 | 12 |
| **Total** | **9 files + 1 new** | **~460** | **33** |

All features are opt-in, backward-compatible, and independently usable.
