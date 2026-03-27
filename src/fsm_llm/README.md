# fsm_llm

Core framework for building stateful conversational AI by combining Large Language Models with Finite State Machines. Implements a 2-pass architecture where Pass 1 extracts data and evaluates transitions, and Pass 2 generates the response from the resolved state.

## Features

- **2-pass architecture** -- Pass 1 extracts data and evaluates transitions; Pass 2 generates responses from the final state
- **Handler system** -- 8 lifecycle timing points (`START_CONVERSATION`, `PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `END_CONVERSATION`, `ERROR`) with a fluent builder API
- **JsonLogic transitions** -- rule-based conditional transitions with operators like `var`, `==`, `in`, `has_context`, `and`, `or`, plus LLM-assisted fallback for ambiguous cases
- **FSM stacking** -- push/pop FSM definitions mid-conversation for modular dialog flows with configurable context merging
- **100+ LLM providers** -- pluggable via LiteLLM (OpenAI, Anthropic, Ollama, Azure, Bedrock, etc.)
- **Context management** -- automatic cleaning of internal keys, forbidden security patterns, and None values; `ContextCompactor` for transient key removal and state-entry pruning
- **CLI tools** -- run conversations interactively, validate FSM definitions, and generate ASCII visualizations
- **Thread-safe** -- per-conversation locks for concurrent conversation processing

## Installation

```bash
pip install fsm-llm
```

For development with all extras:

```bash
pip install fsm-llm[dev,workflows,classification,reasoning,agents,monitor]
```

## Quick Start

```python
from fsm_llm import API

# Load FSM definition and create API instance
api = API.from_file("my_fsm.json", model="gpt-4o-mini")

# Start a conversation (returns conversation_id and initial response)
conversation_id, initial_response = api.start_conversation()
print(initial_response)

# Send user messages
response = api.converse("Hello, I need help.", conversation_id)
print(response)

# Retrieve collected data
data = api.get_data(conversation_id)
```

## Architecture

```
User Input
    |
    v
[Pass 1: Data Extraction (LLM)]
    |
    v
Context Update
    |
    v
Transition Evaluation (JsonLogic rules / LLM fallback)
    |
    v
State Transition
    |
    v
[Pass 2: Response Generation (LLM)]
    |
    v
User Output
```

**Pass 1 -- Analysis and Transition.** The `MessagePipeline` sends the user message to the LLM with extraction-focused prompts built by `DataExtractionPromptBuilder`. Extracted data is merged into the conversation context. The `TransitionEvaluator` then evaluates all outgoing transitions using JsonLogic conditions. If the result is `DETERMINISTIC`, the FSM transitions immediately. If `AMBIGUOUS`, the LLM is consulted to decide. If `BLOCKED`, the FSM stays in the current state.

**Pass 2 -- Response Generation.** After the state transition (or stay), `ResponseGenerationPromptBuilder` constructs a prompt for the new state. The LLM generates a contextually appropriate response based on the final state's `response_instructions` and the updated context.

## API Reference

All methods are on the `fsm_llm.API` class.

| Method | Signature | Description |
|--------|-----------|-------------|
| `from_file` | `(cls, path: Path \| str, **kwargs) -> API` | Class method. Create API from a JSON FSM definition file |
| `__init__` | `(fsm_definition, llm_interface=None, model=None, api_key=None, temperature=None, max_tokens=None, max_history_size=5, max_message_length=1000, handlers=None, handler_error_mode="continue", transition_config=None, **llm_kwargs)` | Initialize with FSMDefinition, dict, or file path |
| `start_conversation` | `(initial_context=None) -> tuple[str, str]` | Start a new conversation. Returns `(conversation_id, initial_response)` |
| `converse` | `(user_message: str, conversation_id: str) -> str` | Process a user message through the 2-pass pipeline. Returns the response string |
| `push_fsm` | `(conversation_id, new_fsm_definition, context_to_pass=None, return_context=None, shared_context_keys=None, preserve_history=False) -> str` | Push a new FSM onto the conversation stack. Returns initial response |
| `pop_fsm` | `(conversation_id, context_to_return=None, merge_strategy="update") -> str` | Pop the current FSM and resume the parent. Returns resume response |
| `register_handler` | `(handler: FSMHandler) -> None` | Register an FSMHandler with the handler system |
| `register_handlers` | `(handlers: list[FSMHandler]) -> None` | Register multiple handlers at once |
| `create_handler` | `(name: str = "CustomHandler") -> HandlerBuilder` | Create a handler using the fluent builder API |
| `get_data` | `(conversation_id: str) -> dict[str, Any]` | Get all collected context data from the current FSM |
| `get_current_state` | `(conversation_id: str) -> str` | Get the current state ID of the active FSM |
| `get_conversation_history` | `(conversation_id: str) -> list[dict[str, str]]` | Get the conversation message history |
| `get_stack_depth` | `(conversation_id: str) -> int` | Get the FSM stack depth (1 = no stacking) |
| `get_sub_conversation_id` | `(conversation_id: str) -> str` | Get the internal sub-FSM conversation ID |
| `get_llm_interface` | `() -> LLMInterface` | Get the underlying LLM interface |

Module-level helper: `quick_start(fsm_file, model=None) -> API` -- shorthand for `API.from_file()`.

## Handler System

Handlers execute custom logic at specific points in the FSM lifecycle. Use the fluent `HandlerBuilder` API:

```python
from fsm_llm import API, HandlerTiming, create_handler

api = API.from_file("my_fsm.json", model="gpt-4o-mini")

# Create a handler that runs after transitioning to "confirmed"
handler = (
    create_handler("SaveOrder")
    .at(HandlerTiming.POST_TRANSITION)
    .on_state("confirmed")
    .do(lambda ctx: save_to_database(ctx))
)

api.register_handler(handler)
```

### HandlerBuilder Methods

| Method | Description |
|--------|-------------|
| `.at(timing)` | Set the timing point (required) |
| `.on_state(state_id)` | Only execute when current state matches |
| `.not_on_state(state_id)` | Exclude a specific state |
| `.on_target_state(state_id)` | Only execute when transitioning to this state |
| `.when_context_has(key)` | Only execute when context contains key |
| `.when_keys_updated(*keys)` | Only execute when specific keys were updated |
| `.on_state_entry(*states)` | Shorthand for POST_TRANSITION on specific states |
| `.on_state_exit(*states)` | Shorthand for PRE_TRANSITION on specific states |
| `.on_context_update()` | Shorthand for CONTEXT_UPDATE timing |
| `.when(condition_fn)` | Custom condition function `(timing, state, target, ctx, keys) -> bool` |
| `.do(execution_fn)` | Set the execution function `(ctx) -> dict` (required). Returns the built handler |

### HandlerTiming Values

| Value | When It Fires |
|-------|---------------|
| `START_CONVERSATION` | When a new conversation begins |
| `PRE_PROCESSING` | Before Pass 1 data extraction |
| `POST_PROCESSING` | After Pass 1 data extraction |
| `PRE_TRANSITION` | Before a state transition |
| `POST_TRANSITION` | After a state transition |
| `CONTEXT_UPDATE` | When context data is updated |
| `END_CONVERSATION` | When the conversation ends (terminal state) |
| `ERROR` | When an error occurs during processing |

## Transitions

Transitions use JsonLogic conditions evaluated by `TransitionEvaluator`. Each transition has a `priority` (higher fires first) and a list of `conditions`:

```json
{
  "target_state": "next_state",
  "description": "User provided their name",
  "priority": 100,
  "conditions": [
    {
      "description": "Name is present",
      "requires_context_keys": ["user_name"],
      "logic": {"!=": [{"var": "user_name"}, ""]},
      "evaluation_priority": 0
    }
  ]
}
```

### Evaluation Results

| Result | Meaning |
|--------|---------|
| `DETERMINISTIC` | Exactly one transition's conditions are satisfied. Transition fires automatically |
| `AMBIGUOUS` | Multiple transitions are valid. The LLM is consulted to choose |
| `BLOCKED` | No transitions have satisfied conditions. The FSM stays in the current state |

### Supported JsonLogic Operators

`var`, `==`, `!=`, `===`, `!==`, `>`, `>=`, `<`, `<=`, `and`, `or`, `!`, `!!`, `if`, `in`, `contains`, `has_context`, `context_length`, `missing`, `missing_some`, `+`, `-`, `*`, `/`, `%`, `min`, `max`, `cat`

## FSM Definition Format

FSM definitions use JSON v4.1 format:

```json
{
  "name": "MyBot",
  "description": "What this FSM does",
  "initial_state": "greeting",
  "persona": "A friendly assistant",
  "states": {
    "greeting": {
      "id": "greeting",
      "description": "Initial greeting state",
      "purpose": "Welcome the user and understand their need",
      "extraction_instructions": "Extract the user's name and intent",
      "response_instructions": "Greet the user warmly and ask how you can help",
      "required_context_keys": [],
      "transitions": [
        {
          "target_state": "collecting_info",
          "description": "User stated their need",
          "priority": 100,
          "conditions": [
            {
              "description": "Intent is identified",
              "requires_context_keys": ["user_intent"],
              "logic": {"!=": [{"var": "user_intent"}, null]}
            }
          ]
        }
      ]
    }
  }
}
```

### Key State Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | State identifier (must match dict key) |
| `description` | `str` | Brief description of the state |
| `purpose` | `str` | What should be accomplished in this state |
| `extraction_instructions` | `str?` | What data the LLM should extract from user input (Pass 1) |
| `response_instructions` | `str?` | How the LLM should respond to the user (Pass 2) |
| `required_context_keys` | `list[str]?` | Keys that must exist in context before entering this state |
| `transitions` | `list[Transition]` | Available transitions from this state |

**Warning**: The bare `instructions` field is silently ignored by Pydantic. Always use `extraction_instructions` and `response_instructions`.

## File Map

| File | Purpose |
|------|---------|
| `api.py` | `API` class -- primary user-facing entry point, factory methods, FSM stacking, conversation management |
| `fsm.py` | `FSMManager` -- conversation lifecycle orchestration, per-conversation thread locks, FSM definition caching |
| `pipeline.py` | `MessagePipeline` -- 2-pass message processing engine (stateless with respect to conversation instances) |
| `definitions.py` | Pydantic models: `State`, `Transition`, `TransitionCondition`, `FSMDefinition`, `FSMInstance`, `FSMContext`, `Conversation`, plus request/response models and exception classes |
| `handlers.py` | `HandlerSystem`, `HandlerBuilder` (fluent API), `BaseHandler`, `LambdaHandler`, `HandlerTiming` enum (8 values) |
| `prompts.py` | Prompt builders: `DataExtractionPromptBuilder`, `ResponseGenerationPromptBuilder`, `TransitionPromptBuilder` |
| `llm.py` | `LLMInterface` ABC + `LiteLLMInterface` implementation (two active methods: `generate_response`, `extract_field`) |
| `ollama.py` | Ollama-specific helpers: model detection, JSON schema format, thinking disable |
| `transition_evaluator.py` | `TransitionEvaluator` -- rule-based evaluation producing `DETERMINISTIC`, `AMBIGUOUS`, or `BLOCKED` |
| `expressions.py` | JsonLogic evaluator with custom operators (`has_context`, `context_length`) |
| `context.py` | `clean_context_keys()` for security filtering; `ContextCompactor` for transient key removal and state-entry pruning |
| `constants.py` | Defaults (`DEFAULT_LLM_MODEL`, `DEFAULT_TEMPERATURE`), security patterns (pre-compiled), internal key prefixes, environment variable keys |
| `validator.py` | `FSMValidator` -- structural validation of FSM definitions |
| `visualizer.py` | ASCII FSM diagram generation |
| `utilities.py` | JSON extraction with 4 fallback strategies, FSM loading helpers (`load_fsm_definition`, `load_fsm_from_file`) |
| `runner.py` | Interactive CLI conversation runner (used by `__main__`) |
| `logging.py` | Loguru configuration with conversation context binding. Use `from fsm_llm.logging import logger` |
| `__main__.py` | CLI entry point for run, validate, and visualize modes |
| `__version__.py` | Package version string |
| `__init__.py` | Public API exports (single `__all__` list with 67 entries), extension check functions |

## CLI Tools

```bash
# Run an interactive conversation
fsm-llm --fsm path/to/definition.json

# Validate an FSM definition
fsm-llm-validate --fsm path/to/definition.json

# Generate ASCII visualization
fsm-llm-visualize --fsm path/to/definition.json
```

## Examples

Example FSM applications are in the `examples/` directory at the repository root:

- **basic/** -- `simple_greeting`, `form_filling`, `story_time`
- **intermediate/** -- `book_recommendation`, `product_recommendation`, `adaptive_quiz`
- **advanced/** -- `yoga_instructions` (JsonLogic conditions), `e_commerce` (FSM stacking with push/pop), `support_pipeline`

Run any example with:

```bash
python examples/<category>/<name>/run.py
```

## Development

506 tests across 20 test files in `tests/test_fsm_llm/`.

```bash
# Run core package tests
pytest tests/test_fsm_llm/ -v

# Run regression tests
pytest tests/test_fsm_llm_regression/ -v

# Lint and format
make lint
make format

# Type checking
make type-check
```

Key test fixtures (defined in `tests/conftest.py`):

- `sample_fsm_definition` -- v3.0 format FSM
- `sample_fsm_definition_v2` -- v4.1 format FSM
- `mock_llm_interface` -- single-pass mock LLM
- `mock_llm2_interface` -- 2-pass architecture mock LLM
