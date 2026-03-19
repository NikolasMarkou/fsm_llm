# FSM-LLM Core Package

The core framework for building stateful conversational AI by combining Large Language Models with Finite State Machines. Uses a **2-pass architecture** that separates data extraction from response generation for optimal contextual accuracy.

## Features

- **2-Pass Architecture**: Pass 1 extracts data and evaluates transitions; Pass 2 generates responses from the final state
- **JSON-Defined FSMs**: Declarative conversation flow design with states, transitions, conditions (JsonLogic), and priorities
- **Handler System**: Fluent builder API for extending FSM behavior at 8 lifecycle timing points
- **FSM Stacking**: Push/pop FSMs for modular, hierarchical conversations with context merging
- **Multi-Provider LLM**: 100+ LLM providers via LiteLLM (OpenAI, Anthropic, Ollama, etc.)
- **Thread-Safe**: Per-conversation locks for concurrent conversation processing
- **Security**: Context isolation, internal key filtering, prompt injection prevention

## Installation

```bash
pip install fsm-llm
```

## Quick Start

### Define an FSM (my_bot.json)

```json
{
  "name": "MyBot",
  "description": "A simple greeting bot",
  "initial_state": "ask_name",
  "persona": "A friendly assistant",
  "states": {
    "ask_name": {
      "id": "ask_name",
      "purpose": "Ask the user for their name",
      "response_instructions": "Politely ask the user for their name",
      "transitions": [
        {
          "target_state": "greet",
          "description": "User has provided their name"
        }
      ]
    },
    "greet": {
      "id": "greet",
      "purpose": "Greet the user by name",
      "required_context_keys": ["name"],
      "response_instructions": "Greet the user warmly using their name",
      "transitions": []
    }
  }
}
```

### Run It

```python
from fsm_llm import API

api = API.from_file("my_bot.json")
conversation_id, response = api.start_conversation()
print(f"Bot: {response}")

user_input = input("You: ")
response = api.converse(user_input, conversation_id)
print(f"Bot: {response}")

print(f"Data: {api.get_data(conversation_id)}")
```

### Add Handlers

```python
from fsm_llm import HandlerTiming

api.register_handler(
    api.create_handler("NameLogger")
       .at(HandlerTiming.POST_PROCESSING)
       .on_state("ask_name")
       .when_context_has("name")
       .do(lambda ctx: print(f"Got name: {ctx.get('name')}") or {})
)
```

### FSM Stacking

```python
# Push a sub-FSM for specialized handling
api.push_fsm(conversation_id, sub_fsm_definition, inherit_context=True)

# Pop back to parent when done
api.pop_fsm(conversation_id, context_to_return={"result": "value"})
```

## File Map

| File | Purpose |
|------|---------|
| `api.py` | **API** class — primary user-facing entry point, FSM stacking, conversation management |
| `fsm.py` | **FSMManager** — core 2-pass orchestration with per-conversation thread locks |
| `definitions.py` | Pydantic models: State, Transition, FSMDefinition, FSMContext, FSMInstance + exception hierarchy |
| `handlers.py` | **HandlerSystem**, **HandlerBuilder** (fluent API), BaseHandler, **HandlerTiming** enum (8 values) |
| `prompts.py` | Prompt builders: DataExtractionPromptBuilder, ResponseGenerationPromptBuilder, TransitionPromptBuilder |
| `llm.py` | **LLMInterface** ABC + **LiteLLMInterface** — three methods: extract_data, generate_response, decide_transition |
| `transition_evaluator.py` | **TransitionEvaluator** — rule-based evaluation producing DETERMINISTIC, AMBIGUOUS, or BLOCKED |
| `expressions.py` | JsonLogic evaluator — operators: var, ==, !=, <, >, and, or, in, has_context, context_length |
| `context.py` | **clean_context_keys()** — stateless context cleaning (strips None values, internal prefixes, forbidden patterns) |
| `runner.py` | Interactive CLI conversation runner (used by `__main__`) |
| `validator.py` | **FSMValidator** — structural validation of FSM definitions |
| `visualizer.py` | ASCII FSM diagram generation |
| `utilities.py` | JSON extraction (4 fallback strategies), FSM loading helpers |
| `constants.py` | Defaults (DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE), security patterns, internal key prefixes |
| `logging.py` | Loguru setup with conversation context — `from fsm_llm.logging import logger` |
| `__main__.py` | CLI entry point: run, validate, visualize modes |
| `__version__.py` | Package version string |
| `__init__.py` | Public API exports — single `__all__` list, extension check functions |

## API Reference

### API Class

| Method | Description |
|--------|-------------|
| `API.from_file(path, **kwargs)` | Create API from FSM JSON file |
| `API.from_definition(fsm_def, **kwargs)` | Create API from dict/FSMDefinition |
| `start_conversation(initial_context=None)` | Start conversation → `(conversation_id, response)` |
| `converse(message, conversation_id)` | Process message → `str` response |
| `push_fsm(conv_id, fsm_def, ...)` | Push sub-FSM onto stack |
| `pop_fsm(conv_id, context_to_return, ...)` | Pop back to parent FSM |
| `register_handler(handler)` | Register event handler |
| `create_handler(name)` | Create HandlerBuilder |
| `get_data(conversation_id)` | Get conversation context data |
| `has_conversation_ended(conv_id)` | Check if in terminal state |
| `end_conversation(conv_id)` | End and clean up |
| `quick_start(fsm_file, model=None)` | Quick helper returning configured API |

### HandlerTiming Points

| Timing | When |
|--------|------|
| `START_CONVERSATION` | Conversation initialized |
| `PRE_PROCESSING` | Before data extraction |
| `POST_PROCESSING` | After data extraction |
| `PRE_TRANSITION` | Before state transition |
| `POST_TRANSITION` | After state transition |
| `CONTEXT_UPDATE` | When context keys are updated |
| `END_CONVERSATION` | Conversation ending |
| `ERROR` | On error |

### HandlerBuilder Methods

`.at(*timings)` · `.on_state(*states)` · `.not_on_state(*states)` · `.on_target_state(*states)` · `.when_context_has(*keys)` · `.when_keys_updated(*keys)` · `.on_state_entry(*states)` · `.on_state_exit(*states)` · `.do(fn)`

### FSM State Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | State identifier (must match dict key) |
| `description` | str | Brief description |
| `purpose` | str | What should be accomplished |
| `extraction_instructions` | str? | What data to extract from user input |
| `response_instructions` | str? | How to respond to the user |
| `required_context_keys` | list? | Keys required to enter this state |
| `transitions` | list | Available transitions from this state |

### Exception Hierarchy

- `FSMError` → `StateNotFoundError`, `InvalidTransitionError`, `LLMResponseError`, `TransitionEvaluationError`
- `HandlerSystemError` → `HandlerExecutionError`

### Extension Checks

| Function | Description |
|----------|-------------|
| `has_workflows()` / `get_workflows()` | Check/get workflows extension |
| `has_reasoning()` / `get_reasoning()` | Check/get reasoning extension |
| `has_classification()` / `get_classification()` | Check/get classification extension |
| `get_version_info()` | Detailed version and feature info |
| `enable_debug_logging()` | Enable debug logging for development |

## CLI Tools

```bash
fsm-llm --fsm <path.json>            # Run FSM interactively
fsm-llm-visualize --fsm <path.json>  # ASCII visualization
fsm-llm-validate --fsm <path.json>   # Validate FSM definition
```

## Development

```bash
pytest tests/test_fsm_llm/            # 336 core tests
pytest tests/test_fsm_llm_regression/ # 218 regression tests
make test                              # Full test suite (810+)
make lint                              # Ruff linting
make format                            # Ruff formatting
```

## Documentation

- [Quick Start Guide](../../docs/quickstart.md)
- [API Reference](../../docs/api_reference.md)
- [Architecture Deep Dive](../../docs/architecture.md)
- [FSM Design Guide](../../docs/fsm_design.md)
- [Handler Development](../../docs/handlers.md)
- [LLM Interaction Guide](../../LLM.md)

## License

GNU General Public License v3.0. See [LICENSE](../../LICENSE).
