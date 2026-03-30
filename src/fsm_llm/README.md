# FSM-LLM Core

> Stateful conversational AI through Finite State Machines and LLMs with a 2-pass architecture.

---

## Overview

`fsm_llm` is the core framework package that combines Large Language Models with Finite State Machines to build structured, stateful conversations. It uses a **2-pass architecture**:

- **Pass 1 (Analysis)**: Extracts data from user input, evaluates transition conditions, resolves ambiguity via classification, and executes state transitions
- **Pass 2 (Generation)**: Generates the final user-facing response from the new state's context

This separation ensures the LLM focuses on one task at a time — understanding the user first, then crafting the response.

## Installation

```bash
# Core only
pip install fsm-llm

# With all extensions
pip install fsm-llm[all]

# Development
pip install fsm-llm[dev]
```

**Requirements**: Python 3.10+ | Dependencies: loguru, litellm (1.82+), pydantic 2.0+, python-dotenv

## Quick Start

**1. Define your FSM** (JSON):

```json
{
  "name": "Greeter",
  "initial_state": "greeting",
  "persona": "A friendly assistant",
  "states": {
    "greeting": {
      "id": "greeting",
      "purpose": "Welcome and ask their name",
      "extraction_instructions": "Extract the user's name if provided",
      "response_instructions": "Greet warmly, ask for name if not given",
      "transitions": [
        {
          "target_state": "farewell",
          "description": "User has given their name",
          "conditions": [{"description": "Name is available", "requires_context_keys": ["name"], "logic": {"has_context": "name"}}]
        }
      ]
    },
    "farewell": {
      "id": "farewell",
      "purpose": "Thank the user and end conversation",
      "response_instructions": "Say a personalized goodbye using their name",
      "transitions": []
    }
  }
}
```

**2. Run a conversation** (Python):

```python
from fsm_llm import API

api = API.from_file("greeter.json", model="gpt-4o-mini")
conv_id, greeting = api.start_conversation()
print(greeting)

response = api.converse("My name is Alice", conv_id)
print(response)

api.end_conversation(conv_id)
api.close()
```

**3. Or run from the CLI**:

```bash
export OPENAI_API_KEY=your-key
fsm-llm --fsm greeter.json
```

## Architecture

```
User Input → Pass 1: Data Extraction → Context Update → Transition Evaluation
           → Classify (if ambiguous) → State Transition
           → Pass 2: Response Generation → User Output
```

### Key Components

| Component | Module | Purpose |
|-----------|--------|---------|
| `API` | `api.py` | User-facing entry point — factory methods, conversation lifecycle, FSM stacking |
| `FSMManager` | `fsm.py` | Core orchestration with per-conversation thread locks |
| `MessagePipeline` | `pipeline.py` | 2-pass processing engine (extraction → transition → response) |
| `HandlerSystem` | `handlers.py` | Event-driven hooks at 8 lifecycle points |
| `Classifier` | `classification.py` | LLM-backed intent classification (single, multi, hierarchical) |
| `TransitionEvaluator` | `transition_evaluator.py` | Rule-based transition evaluation with JsonLogic |
| `LiteLLMInterface` | `llm.py` | LLM communication via litellm (100+ providers) |
| `WorkingMemory` | `memory.py` | Structured named buffers (core, scratch, environment, reasoning) |
| `SessionStore` / `FileSessionStore` | `session.py` | Session persistence with atomic file writes |

## Key API Reference

### API Class

```python
from fsm_llm import API

api = API.from_file("path/to/fsm.json", model="gpt-4o-mini")
api = API.from_definition(fsm_dict, model="gpt-4o-mini")

conv_id, greeting = api.start_conversation(initial_context={"key": "value"})
response = api.converse("user message", conv_id)
api.end_conversation(conv_id)

# FSM stacking (sub-conversations)
sub_conv_id = api.push_fsm(conv_id, sub_fsm_definition)
response = api.pop_fsm(sub_conv_id, merge_strategy=ContextMergeStrategy.UPDATE)

# State queries
state = api.get_current_state(conv_id)
data = api.get_data(conv_id)
history = api.get_conversation_history(conv_id)
```

### Handlers

Eight lifecycle hook points via `HandlerTiming`:

| Timing | When |
|--------|------|
| `START_CONVERSATION` | Conversation initialized |
| `PRE_PROCESSING` | Before message processing |
| `POST_PROCESSING` | After message processing |
| `PRE_TRANSITION` | Before state transition |
| `POST_TRANSITION` | After state transition |
| `CONTEXT_UPDATE` | During context updates |
| `END_CONVERSATION` | Conversation terminated |
| `ERROR` | Error handling |

```python
handler = api.create_handler("logger") \
    .at(HandlerTiming.POST_TRANSITION) \
    .on_state("checkout") \
    .do(lambda ctx: print(f"Entered checkout: {ctx}") or {})
api.register_handler(handler)
```

### Classification

```python
from fsm_llm import Classifier, ClassificationSchema, IntentDefinition

schema = ClassificationSchema(
    intents=[
        IntentDefinition(name="buy", description="User wants to purchase"),
        IntentDefinition(name="browse", description="User is browsing"),
    ],
    fallback_intent="browse",
)
classifier = Classifier(schema, model="gpt-4o-mini")
result = classifier.classify("I'd like to buy the red shoes")
```

### Transition Conditions (JsonLogic)

```json
{
  "and": [
    {"has_context": "email"},
    {">=": [{"var": "age"}, 18]},
    {"in": [{"var": "country"}, ["US", "CA", "UK"]]}
  ]
}
```

Supported operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `and`, `or`, `!`, `in`, `has_context`, `context_length`, `var`, `if`, `cat`, `min`, `max`, arithmetic (`+`, `-`, `*`, `/`, `%`).

### LLM Interface

```python
from fsm_llm import LiteLLMInterface, LLMInterface

llm = LiteLLMInterface(model="gpt-4o-mini", temperature=0.7)

# Streaming (Pass 2 only)
for chunk in llm.generate_response_stream(request):
    print(chunk, end="")

# Schema enforcement (structured JSON output)
# Set response_format on ResponseGenerationRequest for constrained decoding

# Or implement your own
class CustomLLM(LLMInterface):
    def generate_response(self, request): ...
    def extract_field(self, request): ...
    def generate_response_stream(self, request): ...  # Optional
```

### Session Persistence

```python
from fsm_llm import API, FileSessionStore

store = FileSessionStore("./sessions")
api = API.from_file("bot.json", model="gpt-4o-mini", session_store=store)

# State is auto-saved after each converse() call
conv_id, greeting = api.start_conversation()
response = api.converse("Hello!", conv_id)

# Explicit save/load
api.save_session(conv_id)
api.load_session(conv_id)
```

## CLI Tools

| Command | Description |
|---------|-------------|
| `fsm-llm --fsm <path>` | Run interactive conversation |
| `fsm-llm-validate --fsm <path>` | Validate FSM definition |
| `fsm-llm-visualize --fsm <path>` | ASCII FSM visualization |

## FSM Definition Format (v4.1)

States support:
- `extraction_instructions` / `response_instructions` — LLM prompts for each pass
- `required_context_keys` — keys that must exist before leaving the state
- `field_extractions` — targeted single-field extraction with validation rules
- `classification_extractions` — intent classification with confidence thresholds
- `transitions` — conditions with JsonLogic, priority ordering, and LLM descriptions
- `context_scope` — read/write key filtering per state

## Exception Hierarchy

```
FSMError
├── StateNotFoundError
├── InvalidTransitionError
├── LLMResponseError
├── TransitionEvaluationError
├── ClassificationError
│   ├── SchemaValidationError
│   └── ClassificationResponseError
└── HandlerSystemError
    └── HandlerExecutionError
```

## License

GPL-3.0-or-later. See [LICENSE](../../LICENSE) for details.
