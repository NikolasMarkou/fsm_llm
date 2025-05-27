# API Reference - Complete API Documentation

This document provides comprehensive documentation for the LLM-FSM API, including all classes, methods, and parameters.

## Table of Contents

1. [API Class](#api-class)
2. [FSM Definition Classes](#fsm-definition-classes)
3. [Handler System](#handler-system)
4. [LLM Interfaces](#llm-interfaces)
5. [Utility Functions](#utility-functions)
6. [Exceptions](#exceptions)
7. [Constants](#constants)

---

## API Class

The main interface for working with LLM-FSM.

### Constructor

```python
API(
    fsm_definition: Union[FSMDefinition, Dict[str, Any], str],
    llm_interface: Optional[LLMInterface] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_history_size: int = 5,
    max_message_length: int = 1000,
    handlers: Optional[List[FSMHandler]] = None,
    handler_error_mode: str = "continue",
    **llm_kwargs
)
```

**Parameters:**
- `fsm_definition`: FSM definition as an object, dictionary, or path to file
- `llm_interface`: Optional custom LLM interface instance
- `model`: LLM model to use (e.g., "gpt-4o", "claude-3-opus")
- `api_key`: Optional API key (uses environment variables if not provided)
- `temperature`: LLM temperature parameter (0.0-1.0)
- `max_tokens`: Maximum tokens for LLM responses
- `max_history_size`: Maximum conversation exchanges to keep in history
- `max_message_length`: Maximum length of messages in characters
- `handlers`: Optional list of handlers to register
- `handler_error_mode`: How to handle handler errors ("continue", "raise", "skip")
- `**llm_kwargs`: Additional keyword arguments for LLM interface

### Class Methods

#### `from_file()`

```python
@classmethod
def from_file(
    cls,
    path: Union[Path, str],
    **kwargs
) -> 'API'
```

Create an API instance from a JSON file.

**Example:**
```python
api = API.from_file("customer_service.json", model="gpt-4o-mini")
```

#### `from_definition()`

```python
@classmethod
def from_definition(
    cls,
    fsm_definition: Union[FSMDefinition, Dict[str, Any]],
    **kwargs
) -> 'API'
```

Create an API instance from an FSM definition object or dictionary.

**Example:**
```python
fsm_def = {
    "name": "simple_bot",
    "initial_state": "start",
    "states": {...}
}
api = API.from_definition(fsm_def, model="gpt-4o")
```

### Core Methods

#### `start_conversation()`

```python
def start_conversation(
    self,
    initial_context: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]
```

Start a new conversation with the FSM.

**Parameters:**
- `initial_context`: Optional initial context data

**Returns:**
- Tuple of (conversation_id, initial_response)

**Example:**
```python
conv_id, response = api.start_conversation({"user_name": "Alice"})
print(f"Bot: {response}")
```

#### `converse()`

```python
def converse(
    self,
    user_message: str,
    conversation_id: str
) -> str
```

Process a user message and return the response.

**Parameters:**
- `user_message`: The user's message
- `conversation_id`: ID for an existing conversation

**Returns:**
- The bot's response

**Example:**
```python
response = api.converse("My name is Bob", conv_id)
```

### FSM Stacking Methods

#### `push_fsm()`

```python
def push_fsm(
    self,
    conversation_id: str,
    new_fsm_definition: Union[FSMDefinition, Dict[str, Any], str],
    context_to_pass: Optional[Dict[str, Any]] = None,
    return_context: Optional[Dict[str, Any]] = None,
    entry_point: Optional[str] = None,
    shared_context_keys: Optional[List[str]] = None,
    preserve_history: bool = False,
    inherit_context: bool = True
) -> str
```

Push a new FSM onto the conversation stack.

**Parameters:**
- `conversation_id`: The main conversation ID
- `new_fsm_definition`: FSM definition for the new FSM
- `context_to_pass`: Explicit context to pass to the new FSM
- `return_context`: Context to store for when returning
- `entry_point`: Optional state to resume when returning
- `shared_context_keys`: Keys that should be automatically synced
- `preserve_history`: Whether to copy conversation history
- `inherit_context`: Whether to inherit all context

**Example:**
```python
response = api.push_fsm(
    conv_id,
    "detailed_form.json",
    context_to_pass={"step": "details"},
    shared_context_keys=["user_id", "email"],
    preserve_history=True
)
```

#### `pop_fsm()`

```python
def pop_fsm(
    self,
    conversation_id: str,
    context_to_return: Optional[Dict[str, Any]] = None,
    merge_strategy: Union[str, ContextMergeStrategy] = ContextMergeStrategy.UPDATE
) -> str
```

Pop the current FSM from the stack and return to the previous one.

**Parameters:**
- `conversation_id`: The main conversation ID
- `context_to_return`: Context to merge back into the previous FSM
- `merge_strategy`: How to merge contexts:
  - `"update"`: Update parent context with returned context
  - `"preserve"`: Only add new keys, don't overwrite
  - `"selective"`: Only merge shared_context_keys

**Example:**
```python
response = api.pop_fsm(
    conv_id,
    context_to_return={"form_completed": True},
    merge_strategy="update"
)
```

### Handler Methods

#### `register_handler()`

```python
def register_handler(
    self,
    handler: FSMHandler
) -> None
```

Register a handler with the FSM system.

**Example:**
```python
api.register_handler(my_handler)
```

#### `register_handlers()`

```python
def register_handlers(
    self,
    handlers: List[FSMHandler]
) -> None
```

Register multiple handlers at once.

#### `create_handler()`

```python
def create_handler(
    self,
    name: str = "CustomHandler"
) -> HandlerBuilder
```

Create a new handler using the fluent builder interface.

**Example:**
```python
handler = api.create_handler("EmailValidator") \
    .at(HandlerTiming.CONTEXT_UPDATE) \
    .when_keys_updated("email") \
    .do(validate_email)
```

#### Convenience Handler Methods

```python
def add_logging_handler(
    self,
    log_timings: Optional[List[HandlerTiming]] = None,
    log_states: Optional[List[str]] = None,
    priority: int = 10
) -> None
```

Add a logging handler for debugging.

```python
def add_context_validator_handler(
    self,
    required_keys: List[str],
    timing: HandlerTiming = HandlerTiming.PRE_PROCESSING,
    priority: int = 5
) -> None
```

Add a handler that validates required context keys.

```python
def add_state_entry_handler(
    self,
    state: str,
    handler_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    priority: int = 50
) -> None
```

Add a handler for specific state entry.

### Data Access Methods

#### `get_data()`

```python
def get_data(
    self,
    conversation_id: str
) -> Dict[str, Any]
```

Get collected data from the current active FSM.

**Example:**
```python
data = api.get_data(conv_id)
print(f"User name: {data.get('name')}")
```

#### `get_all_stack_data()`

```python
def get_all_stack_data(
    self,
    conversation_id: str
) -> List[Dict[str, Any]]
```

Get data from all FSMs in the conversation stack.

#### `get_conversation_history()`

```python
def get_conversation_history(
    self,
    conversation_id: str
) -> List[Dict[str, str]]
```

Get the conversation history for the current FSM.

### State Information Methods

#### `has_conversation_ended()`

```python
def has_conversation_ended(
    self,
    conversation_id: str
) -> bool
```

Check if the current FSM has reached a terminal state.

#### `get_current_state()`

```python
def get_current_state(
    self,
    conversation_id: str
) -> str
```

Get the current state ID of the active FSM.

#### `get_stack_depth()`

```python
def get_stack_depth(
    self,
    conversation_id: str
) -> int
```

Get the current FSM stack depth.

#### `get_context_flow()`

```python
def get_context_flow(
    self,
    conversation_id: str
) -> Dict[str, Any]
```

Get a summary of context flow between stacked FSMs.

### Conversation Management

#### `end_conversation()`

```python
def end_conversation(
    self,
    conversation_id: str
) -> None
```

End a conversation but retain data.

#### `delete_conversation()`

```python
def delete_conversation(
    self,
    conversation_id: str
) -> None
```

Completely delete a conversation and all data.

#### `save_conversation()`

```python
def save_conversation(
    self,
    conversation_id: str,
    path: str
) -> None
```

Save conversation state to a file.

#### `list_active_conversations()`

```python
def list_active_conversations(
    self
) -> List[str]
```

Get list of all active conversation IDs.

---

## FSM Definition Classes

### FSMDefinition

```python
class FSMDefinition(BaseModel):
    name: str
    description: str
    states: Dict[str, State]
    initial_state: str
    version: str = "3.0"
    persona: Optional[str] = None
```

**Fields:**
- `name`: Name of the FSM
- `description`: Human-readable description
- `states`: Dictionary of state definitions
- `initial_state`: Starting state identifier
- `version`: FSM definition version
- `persona`: Optional persona description for responses

### State

```python
class State(BaseModel):
    id: str
    description: str
    purpose: str
    transitions: List[Transition] = []
    required_context_keys: Optional[List[str]] = None
    instructions: Optional[str] = None
    example_dialogue: Optional[List[Dict[str, str]]] = None
```

**Fields:**
- `id`: Unique state identifier
- `description`: Human-readable description
- `purpose`: What this state should accomplish
- `transitions`: Available transitions from this state
- `required_context_keys`: Keys to collect in this state
- `instructions`: Additional instructions for the LLM
- `example_dialogue`: Example conversations for this state

### Transition

```python
class Transition(BaseModel):
    target_state: str
    description: str
    conditions: Optional[List[TransitionCondition]] = None
    priority: int = 100
```

**Fields:**
- `target_state`: State to transition to
- `description`: When this transition should occur
- `conditions`: Optional conditions that must be met
- `priority`: Priority for transition selection

### TransitionCondition

```python
class TransitionCondition(BaseModel):
    description: str
    requires_context_keys: Optional[List[str]] = None
    logic: Optional[Dict[str, Any]] = None
```

**Fields:**
- `description`: Human-readable condition description
- `requires_context_keys`: Keys that must be present
- `logic`: JsonLogic expression for evaluation

---

## Handler System

### HandlerTiming

```python
class HandlerTiming(Enum):
    START_CONVERSATION = auto()
    PRE_PROCESSING = auto()
    POST_PROCESSING = auto()
    PRE_TRANSITION = auto()
    POST_TRANSITION = auto()
    CONTEXT_UPDATE = auto()
    END_CONVERSATION = auto()
    ERROR = auto()
```

### HandlerBuilder

Fluent interface for building handlers.

#### Methods

```python
def with_priority(self, priority: int) -> 'HandlerBuilder'
def when(self, condition: ConditionLambda) -> 'HandlerBuilder'
def at(self, *timings: HandlerTiming) -> 'HandlerBuilder'
def on_state(self, *states: str) -> 'HandlerBuilder'
def on_target_state(self, *states: str) -> 'HandlerBuilder'
def when_context_has(self, *keys: str) -> 'HandlerBuilder'
def when_keys_updated(self, *keys: str) -> 'HandlerBuilder'
def on_state_entry(self, *states: str) -> 'HandlerBuilder'
def on_state_exit(self, *states: str) -> 'HandlerBuilder'
def do(self, execution: Callable) -> BaseHandler
```

**Example:**
```python
handler = create_handler("MyHandler") \
    .at(HandlerTiming.POST_PROCESSING) \
    .on_state("checkout") \
    .when_context_has("cart_items") \
    .with_priority(10) \
    .do(process_checkout)
```

---

## LLM Interfaces

### LLMInterface (Protocol)

```python
class LLMInterface(Protocol):
    def send_request(self, request: LLMRequest) -> LLMResponse:
        ...
```

Base protocol for LLM interfaces.

### LiteLLMInterface

```python
class LiteLLMInterface(LLMInterface):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 1000,
        **kwargs
    )
```

Default implementation using LiteLLM.

**Supported Models:**
- OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- Anthropic: `claude-3-opus`, `claude-3-sonnet`
- Local: `ollama/llama2`, `ollama/mistral`

---

## Utility Functions

### `load_fsm_from_file()`

```python
def load_fsm_from_file(file_path: str) -> FSMDefinition
```

Load an FSM definition from a JSON file.

### `validate_fsm_from_file()`

```python
def validate_fsm_from_file(json_file: str) -> FSMValidationResult
```

Validate an FSM definition file.

### `visualize_fsm_from_file()`

```python
def visualize_fsm_from_file(
    json_file: str,
    style: str = "full"
) -> str
```

Generate ASCII visualization of an FSM.

**Styles:**
- `"full"`: Complete visualization with all details
- `"compact"`: Condensed view
- `"minimal"`: Basic structure only

---

## Exceptions

### FSMError

```python
class FSMError(Exception):
    """Base exception for FSM errors."""
```

### Specific Exceptions

- `StateNotFoundError`: State not found in FSM
- `InvalidTransitionError`: Invalid state transition
- `LLMResponseError`: Error in LLM response processing
- `WorkflowError`: Base workflow exception
- `WorkflowDefinitionError`: Invalid workflow definition
- `WorkflowStepError`: Error in workflow step execution
- `WorkflowInstanceError`: Workflow instance management error
- `WorkflowTimeoutError`: Operation timeout
- `WorkflowValidationError`: Validation failure

---

## Constants

### Environment Variables

```python
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_LLM_MODEL = "LLM_MODEL"
ENV_LLM_TEMPERATURE = "LLM_TEMPERATURE"
ENV_LLM_MAX_TOKENS = "LLM_MAX_TOKENS"
ENV_FSM_PATH = "FSM_PATH"
```

### Default Values

```python
DEFAULT_MAX_HISTORY_SIZE = 5
DEFAULT_MAX_MESSAGE_LENGTH = 1000
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 1000
```

---

## Complete Example

```python
from llm_fsm import API
from llm_fsm.handlers import HandlerTiming
import os

# Create API instance
api = API.from_file(
    "customer_service.json",
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
    max_history_size=10
)

# Add handlers
def log_interaction(context):
    print(f"State: {context.get('_current_state')}")
    return {}

api.register_handler(
    api.create_handler("Logger")
        .at(HandlerTiming.POST_TRANSITION)
        .do(log_interaction)
)

# Start conversation
conv_id, response = api.start_conversation({
    "user_id": "123",
    "timestamp": "2024-01-01"
})

# Have conversation
while not api.has_conversation_ended(conv_id):
    print(f"Bot: {response}")
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        break
        
    response = api.converse(user_input, conv_id)

# Get results
data = api.get_data(conv_id)
print(f"Collected data: {data}")

# Clean up
api.end_conversation(conv_id)
```
