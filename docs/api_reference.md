# API Reference - Complete API Documentation

This document provides comprehensive documentation for the FSM-LLM API, including all classes, methods, and parameters.

## Table of Contents

1. [API Class](#api-class)
2. [FSM Definition Classes](#fsm-definition-classes)
3. [Handler System](#handler-system)
4. [LLM Interfaces](#llm-interfaces)
5. [Utility Functions](#utility-functions)
6. [Exceptions](#exceptions)
7. [Classification (`fsm_llm_classification`)](#classification-fsm_llm_classification)
8. [ReasoningEngine (`fsm_llm_reasoning`)](#reasoningengine-fsm_llm_reasoning)
9. [Agentic Patterns (`fsm_llm_agents`)](#agentic-patterns-fsm_llm_agents)
10. [Workflow Engine (`fsm_llm_workflows`)](#workflow-engine-fsm_llm_workflows)
11. [Monitor Dashboard (`fsm_llm_monitor`)](#monitor-dashboard-fsm_llm_monitor)
12. [Constants](#constants)

---

## API Class

The main interface for working with FSM-LLM.

### Constructor

```python
API(
    fsm_definition: FSMDefinition | dict[str, Any] | str,
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
    **llm_kwargs
)
```

**Parameters:**
- `fsm_definition`: FSM definition as an object, dictionary, or path to file
- `llm_interface`: Optional custom LLM interface instance
- `model`: LLM model to use (e.g., "gpt-4o", "gpt-4o-mini")
- `api_key`: Optional API key (uses environment variables if not provided)
- `temperature`: LLM temperature parameter (0.0-1.0)
- `max_tokens`: Maximum tokens for LLM responses
- `max_history_size`: Maximum conversation exchanges to keep in history
- `max_message_length`: Maximum length of messages in characters
- `handlers`: Optional list of handlers to register
- `handler_error_mode`: How to handle handler errors (`"continue"` or `"raise"`)
- `transition_config`: Optional `TransitionEvaluatorConfig` for configuring transition evaluation behavior
- `**llm_kwargs`: Additional keyword arguments for LLM interface

### Class Methods

#### `from_file()`

```python
@classmethod
def from_file(
    cls,
    path: Path | str,
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
    fsm_definition: FSMDefinition | dict[str, Any],
    **kwargs
) -> 'API'
```

Create an API instance from an FSM definition object or dictionary.

**Example:**
```python
fsm_def = {
    "name": "simple_bot",
    "description": "A simple bot",
    "initial_state": "start",
    "states": {...}
}
api = API.from_definition(fsm_def, model="gpt-4o-mini")
```

### Core Methods

#### `start_conversation()`

```python
def start_conversation(
    self,
    initial_context: dict[str, Any] | None = None
) -> tuple[str, str]
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
- The bot's response as a string

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
    new_fsm_definition: FSMDefinition | dict[str, Any] | str,
    context_to_pass: dict[str, Any] | None = None,
    return_context: dict[str, Any] | None = None,
    shared_context_keys: list[str] | None = None,
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
    context_to_return: dict[str, Any] | None = None,
    merge_strategy: str | ContextMergeStrategy = ContextMergeStrategy.UPDATE
) -> str
```

Pop the current FSM from the stack and return to the previous one.

**Parameters:**
- `conversation_id`: The main conversation ID
- `context_to_return`: Context to merge back into the previous FSM
- `merge_strategy`: How to merge contexts:
  - `"update"` / `ContextMergeStrategy.UPDATE`: Update parent context with returned context
  - `"preserve"` / `ContextMergeStrategy.PRESERVE`: Only add new keys, don't overwrite

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
    handlers: list[FSMHandler]
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

### Data Access Methods

#### `get_data()`

```python
def get_data(
    self,
    conversation_id: str
) -> dict[str, Any]
```

Get collected data from the current active FSM.

**Example:**
```python
data = api.get_data(conv_id)
print(f"User name: {data.get('name')}")
```

#### `get_conversation_history()`

```python
def get_conversation_history(
    self,
    conversation_id: str
) -> list[dict[str, str]]
```

Get the conversation history for the current FSM.

#### `update_context()`

```python
def update_context(
    self,
    conversation_id: str,
    context_update: dict[str, Any]
) -> None
```

Update context data for the current FSM in a conversation.

**Example:**
```python
api.update_context(conv_id, {"priority": "high", "category": "billing"})
```

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

#### `get_sub_conversation_id()`

```python
def get_sub_conversation_id(
    self,
    conversation_id: str
) -> str
```

Get the internal conversation ID of the current (top-of-stack) sub-FSM. Useful for extensions that need to track sub-FSM identity across push/pop operations.

### Conversation Management

#### `end_conversation()`

```python
def end_conversation(
    self,
    conversation_id: str
) -> None
```

End a conversation but retain data.

#### `list_active_conversations()`

```python
def list_active_conversations(
    self
) -> list[str]
```

Get list of all active conversation IDs.

#### `close()`

```python
def close(self) -> None
```

Clean up all active conversations. The API also supports context manager usage:

```python
with API.from_file("bot.json") as api:
    conv_id, response = api.start_conversation()
    # ... conversations are automatically cleaned up on exit
```

#### `get_llm_interface()`

```python
def get_llm_interface(self) -> LLMInterface
```

Get the current LLM interface instance.

---

## FSM Definition Classes

### FSMDefinition

```python
class FSMDefinition(BaseModel):
    name: str
    description: str
    states: dict[str, State]
    initial_state: str
    version: str = "4.1"
    persona: str | None = None
```

**Fields:**
- `name`: Name of the FSM
- `description`: Human-readable description
- `states`: Dictionary of state definitions
- `initial_state`: Starting state identifier
- `version`: FSM definition version (default: "4.1")
- `persona`: Optional persona description for responses

### State

```python
class State(BaseModel):
    id: str
    description: str
    purpose: str
    transitions: list[Transition] = []
    required_context_keys: list[str] | None = None
    extraction_instructions: str | None = None
    response_instructions: str | None = None
```

**Fields:**
- `id`: Unique state identifier
- `description`: Human-readable state description
- `purpose`: What should be accomplished in this state
- `transitions`: Available transitions from this state
- `required_context_keys`: Keys to collect in this state
- `extraction_instructions`: Instructions for the LLM data extraction pass
- `response_instructions`: Instructions for the LLM response generation pass

> **Important**: The bare `instructions` field is silently ignored by Pydantic. Always use `extraction_instructions` and/or `response_instructions`.

### Transition

```python
class Transition(BaseModel):
    target_state: str
    description: str
    conditions: list[TransitionCondition] | None = None
    priority: int = 100
    llm_description: str | None = None
```

**Fields:**
- `target_state`: State to transition to
- `description`: When this transition should occur
- `conditions`: Optional conditions that must be met
- `priority`: Priority for transition selection (lower = higher priority, 0-1000)
- `llm_description`: Optional description for LLM when choosing between transitions (max 300 chars)

### TransitionCondition

```python
class TransitionCondition(BaseModel):
    description: str
    requires_context_keys: list[str] | None = None
    logic: dict[str, Any] | None = None
    evaluation_priority: int = 100
```

**Fields:**
- `description`: Human-readable condition description
- `requires_context_keys`: Keys that must be present
- `logic`: JsonLogic expression for evaluation
- `evaluation_priority`: Priority for condition evaluation (lower = earlier, 0-1000)

---

## Handler System

### HandlerTiming

```python
class HandlerTiming(str, Enum):
    START_CONVERSATION = "start_conversation"
    PRE_PROCESSING = "pre_processing"
    POST_PROCESSING = "post_processing"
    PRE_TRANSITION = "pre_transition"
    POST_TRANSITION = "post_transition"
    CONTEXT_UPDATE = "context_update"
    END_CONVERSATION = "end_conversation"
    ERROR = "error"
```

### HandlerBuilder

Fluent interface for building handlers.

#### Methods

```python
def with_priority(self, priority: int) -> 'HandlerBuilder'
def when(self, condition: ConditionLambda) -> 'HandlerBuilder'
def at(self, *timings: HandlerTiming) -> 'HandlerBuilder'
def on_state(self, *states: str) -> 'HandlerBuilder'
def not_on_state(self, *states: str) -> 'HandlerBuilder'
def on_target_state(self, *states: str) -> 'HandlerBuilder'
def not_on_target_state(self, *states: str) -> 'HandlerBuilder'
def when_context_has(self, *keys: str) -> 'HandlerBuilder'
def when_keys_updated(self, *keys: str) -> 'HandlerBuilder'
def on_state_entry(self, *states: str) -> 'HandlerBuilder'
def on_state_exit(self, *states: str) -> 'HandlerBuilder'
def on_context_update(self, *keys: str) -> 'HandlerBuilder'
def do(self, execution: Callable) -> BaseHandler
```

### BaseHandler

The `critical` parameter on `BaseHandler` causes errors to always raise regardless of `handler_error_mode`:

```python
handler = BaseHandler(
    name="CriticalValidation",
    timing=[HandlerTiming.PRE_PROCESSING],
    execution=validate_input,
    critical=True  # Always raises on error, ignoring error_mode
)
```

**Example:**
```python
handler = api.create_handler("MyHandler") \
    .at(HandlerTiming.POST_PROCESSING) \
    .on_state("checkout") \
    .when_context_has("cart_items") \
    .with_priority(10) \
    .do(process_checkout)
```

---

## LLM Interfaces

### LLMInterface (ABC)

```python
class LLMInterface(abc.ABC):
    """Abstract interface for LLM communication supporting the 2-pass architecture."""

    @abc.abstractmethod
    def extract_data(self, request: DataExtractionRequest) -> DataExtractionResponse:
        """Extract data from user input without generating user-facing content."""
        ...

    @abc.abstractmethod
    def generate_response(self, request: ResponseGenerationRequest) -> ResponseGenerationResponse:
        """Generate user-facing response based on final state context."""
        ...

    @abc.abstractmethod
    def decide_transition(self, request: TransitionDecisionRequest) -> TransitionDecisionResponse:
        """Decide between multiple valid transition options (used when rule-based evaluation is ambiguous)."""
        ...
```

### LiteLLMInterface

```python
class LiteLLMInterface(LLMInterface):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1000,
        **kwargs
    )
```

Default implementation using LiteLLM.

**Supported Models (via LiteLLM):**
- OpenAI: `gpt-4o`, `gpt-4o-mini`
- Anthropic: `claude-sonnet-4-6`, `claude-haiku-4-5`
- Local: `ollama_chat/llama3`, `ollama_chat/qwen3.5:4b`
- And 100+ other providers

**Ollama-Specific Behavior:**
- Thinking mode automatically disabled via `reasoning_effort="none"` (prevents `<think>` traces from corrupting JSON)
- Structured calls (data extraction, transition decision) use `json_schema` response format and force `temperature=0`
- Response generation calls preserve user-configured temperature
- Requires litellm >=1.82.0 for proper Ollama `think` parameter forwarding

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

### Core Exceptions

```python
class FSMError(Exception):
    """Base exception for FSM errors."""

class StateNotFoundError(FSMError): ...
class InvalidTransitionError(FSMError): ...
class LLMResponseError(FSMError): ...
class TransitionEvaluationError(FSMError): ...
```

### Handler Exceptions

```python
class HandlerSystemError(Exception): ...  # Note: NOT caught by `except FSMError`
class HandlerExecutionError(HandlerSystemError): ...
```

### Workflow Exceptions

```python
class WorkflowError(FSMError): ...
class WorkflowDefinitionError(WorkflowError): ...
class WorkflowStepError(WorkflowError): ...
class WorkflowInstanceError(WorkflowError): ...
class WorkflowTimeoutError(WorkflowError): ...
class WorkflowValidationError(WorkflowError): ...
class WorkflowStateError(WorkflowError): ...
class WorkflowEventError(WorkflowError): ...
class WorkflowResourceError(WorkflowError): ...
```

### Classification Exceptions

```python
class ClassificationError(FSMError): ...
class SchemaValidationError(ClassificationError): ...
class ClassificationResponseError(ClassificationError): ...
```

### Reasoning Exceptions

```python
class ReasoningEngineError(FSMError): ...
class ReasoningExecutionError(ReasoningEngineError): ...
class ReasoningClassificationError(ReasoningEngineError): ...
```

### Agent Exceptions

```python
class AgentError(FSMError): ...
class ToolExecutionError(AgentError): ...
class ToolNotFoundError(AgentError): ...
class ToolValidationError(AgentError): ...
class BudgetExhaustedError(AgentError): ...
class ApprovalDeniedError(AgentError): ...
class AgentTimeoutError(AgentError): ...
class EvaluationError(AgentError): ...
class DecompositionError(AgentError): ...
```

### Monitor Exceptions

```python
class MonitorError(Exception): ...
class MonitorInitializationError(MonitorError): ...
class MetricCollectionError(MonitorError): ...
class MonitorConnectionError(MonitorError): ...
```

---

## Classification (`fsm_llm_classification`)

LLM-backed intent classification with single-intent, multi-intent, and hierarchical (two-stage) support.

### `Classifier`

```python
from fsm_llm_classification import Classifier, ClassificationSchema, IntentDefinition

schema = ClassificationSchema(
    intents=[
        IntentDefinition(name="order_status", description="User asks about an order"),
        IntentDefinition(name="product_info", description="User asks about a product"),
        IntentDefinition(name="general_support", description="Anything else"),
    ],
    fallback_intent="general_support",
)

classifier = Classifier(schema, model="gpt-4o-mini")
result = classifier.classify("Where is my order #12345?")
print(result.intent)       # "order_status"
print(result.confidence)   # 0.95
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `schema` | `ClassificationSchema` | *required* | Intent schema defining available classes |
| `model` | `str` | `DEFAULT_LLM_MODEL` | LLM model identifier |
| `api_key` | `str \| None` | `None` | Optional API key |
| `config` | `ClassificationPromptConfig \| None` | `None` | Prompt generation config |
| `**litellm_kwargs` | | | Additional kwargs passed to litellm |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `classify(user_message)` | `ClassificationResult` | Classify into a single intent |
| `classify_multi(user_message)` | `MultiClassificationResult` | Classify into multiple intents |
| `is_low_confidence(result)` | `bool` | Check if result is below schema's confidence threshold |

### `HierarchicalClassifier`

Two-stage classifier for large intent sets (>15 classes). Stage 1 classifies the domain, stage 2 classifies the intent within that domain.

```python
from fsm_llm_classification import HierarchicalClassifier, HierarchicalSchema

h_classifier = HierarchicalClassifier(schema=hierarchical_schema, model="gpt-4o-mini")
result = h_classifier.classify("I need to return my order")
print(result.domain.intent)   # "orders"
print(result.intent.intent)   # "return_request"
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `schema` | `HierarchicalSchema` | *required* | Hierarchical schema with domain + intent schemas |
| `model` | `str` | `DEFAULT_LLM_MODEL` | LLM model identifier |
| `api_key` | `str \| None` | `None` | Optional API key |
| `config` | `ClassificationPromptConfig \| None` | `None` | Prompt generation config |
| `**litellm_kwargs` | | | Additional kwargs passed to litellm |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `classify(user_message)` | `HierarchicalResult` | Two-stage classification (domain then intent) |

### Schema Models

#### `ClassificationSchema`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `intents` | `list[IntentDefinition]` | *required* (min 2) | List of intent definitions |
| `fallback_intent` | `str` | *required* | Name of fallback intent (must be in intents) |
| `confidence_threshold` | `float` | `0.6` | Below this threshold, signals low confidence |

**Property:** `intent_names -> list[str]` — list of intent name strings.

#### `IntentDefinition`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Snake_case identifier (alphanumeric + underscores only) |
| `description` | `str` | Human-readable description shown to the LLM |

#### `HierarchicalSchema`

| Field | Type | Description |
|-------|------|-------------|
| `domain_schema` | `ClassificationSchema` | Stage 1: domain-level classification |
| `intent_schemas` | `dict[str, ClassificationSchema]` | Stage 2: domain → intent schema mapping |

### Result Models

#### `ClassificationResult`

| Field | Type | Description |
|-------|------|-------------|
| `intent` | `str` | The classified intent name |
| `confidence` | `float` | Model confidence (0.0–1.0) |
| `reasoning` | `str` | Chain-of-thought explanation |
| `entities` | `dict[str, str]` | Extracted entities |

**Property:** `is_low_confidence -> bool` — check against default threshold (0.6).

#### `MultiClassificationResult`

| Field | Type | Description |
|-------|------|-------------|
| `reasoning` | `str` | Chain-of-thought explanation |
| `intents` | `list[IntentScore]` | Ranked list of detected intents (1–5) |

**Property:** `primary -> IntentScore` — the highest-ranked intent.

#### `IntentScore`

| Field | Type | Description |
|-------|------|-------------|
| `intent` | `str` | Intent name |
| `confidence` | `float` | Model confidence (0.0–1.0) |
| `entities` | `dict[str, str]` | Extracted entities |

#### `HierarchicalResult`

| Field | Type | Description |
|-------|------|-------------|
| `domain` | `ClassificationResult` | Stage 1 domain classification |
| `intent` | `ClassificationResult` | Stage 2 intent classification |

### `IntentRouter`

Maps classified intents to handler functions with low-confidence fallback.

```python
from fsm_llm_classification import IntentRouter

router = IntentRouter(schema, clarification_handler=my_clarify_fn)
router.register("order_status", handle_order_status)
router.register("product_info", handle_product_info)

result = classifier.classify(user_message)
response = router.route(user_message, result)
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `schema` | `ClassificationSchema` | *required* | Schema for intent validation |
| `clarification_handler` | `HandlerFn \| None` | `None` | Called on low-confidence results |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `register(intent, handler)` | `IntentRouter` | Register a handler for an intent (chainable) |
| `register_many(mapping)` | `IntentRouter` | Register multiple handlers at once |
| `route(user_message, result)` | `Any` | Route a single-intent result to its handler |
| `route_multi(user_message, result)` | `list[Any]` | Route each intent in a multi-intent result |
| `validate()` | `list[str]` | List intent names that lack handlers |

### `ClassificationPromptConfig`

Dataclass controlling prompt generation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `include_reasoning` | `bool` | `True` | Include chain-of-thought in output |
| `max_tokens` | `int` | `512` | Max response tokens |
| `temperature` | `float` | `0.0` | LLM temperature |
| `include_entities` | `bool` | `True` | Include entity extraction |
| `multi_intent` | `bool` | `False` | Multi-intent mode |
| `max_intents` | `int` | `3` | Max intents in multi-intent mode |

---

## ReasoningEngine (`fsm_llm_reasoning`)

Structured reasoning engine that orchestrates 9 strategies via hierarchical FSMs.

### `ReasoningEngine`

```python
from fsm_llm_reasoning import ReasoningEngine

engine = ReasoningEngine(model="gpt-4o-mini")
solution, trace_info = engine.solve_problem("What is 15% of 240?")
print(solution)
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `DEFAULT_LLM_MODEL` | LLM model identifier |
| `**kwargs` | | | Additional kwargs passed to API |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `solve_problem(problem, initial_context=None)` | `tuple[str, dict]` | Solve a problem using automatic strategy selection. Returns (solution_string, trace_info). |

### `ReasoningType`

Enum with 9 strategies:

| Value | Description |
|-------|-------------|
| `SIMPLE_CALCULATOR` | Direct arithmetic calculations |
| `ANALYTICAL` | Break down complex systems into components |
| `DEDUCTIVE` | Derive specific conclusions from general premises |
| `INDUCTIVE` | Find patterns from specific observations |
| `ABDUCTIVE` | Find the best explanation for observations |
| `ANALOGICAL` | Transfer insights via analogies |
| `CREATIVE` | Generate novel solutions |
| `CRITICAL` | Evaluate arguments and evidence |
| `HYBRID` | Combine multiple reasoning approaches |

### Architecture

```
solve_problem(problem)
  -> Orchestrator FSM (6 states)
       -> Classifier FSM (auto-selects strategy)
       -> push Specialized FSM (e.g., analytical, deductive)
       -> pop results back to orchestrator
       -> validate + synthesize
  -> (solution, trace)
```

---

## Agentic Patterns (`fsm_llm_agents`)

12 agent patterns with a consistent API: `agent.run(task) -> AgentResult`.

### `ToolRegistry`

```python
from fsm_llm_agents import ToolRegistry, tool

registry = ToolRegistry()

# Register via function
registry.register_function(my_fn, name="search", description="Search the web")

# Register via decorator
@tool(description="Calculate math", requires_approval=False)
def calculate(expr: str) -> str:
    return str(eval(expr))
registry.register(calculate._tool_definition)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `register(tool_def)` | `ToolRegistry` | Register a ToolDefinition (chainable) |
| `register_function(fn, name, description, ...)` | `ToolRegistry` | Register a callable as a tool (chainable) |
| `get(name)` | `ToolDefinition` | Retrieve tool by name |
| `list_tools()` | `list[ToolDefinition]` | All registered tools |
| `execute(tool_call)` | `ToolResult` | Execute a tool with timing and error handling |
| `to_prompt_description()` | `str` | Generate LLM-friendly tool listing |

### `HumanInTheLoop`

```python
from fsm_llm_agents import HumanInTheLoop

hitl = HumanInTheLoop(
    approval_policy=lambda call, ctx: call.tool_name in ["send_email"],
    approval_callback=lambda req: input(f"Approve {req.tool_name}? ") == "y",
    confidence_threshold=0.3,
    on_escalation=lambda reason, ctx: print(f"ESCALATION: {reason}"),
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `approval_policy` | `Callable[[ToolCall, dict], bool]` | Decides if a tool call needs approval |
| `approval_callback` | `Callable[[ApprovalRequest], bool]` | Requests approval from human |
| `on_escalation` | `Callable[[str, dict], None]` | Called when confidence is too low |
| `confidence_threshold` | `float` | Auto-escalate below this confidence (default: 0.3) |

### Agent Classes

All agents share these common parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `AgentConfig` | `AgentConfig()` | Model, max_iterations, timeout, temperature |

All agents return `AgentResult` with: `answer` (str), `success` (bool), `trace` (AgentTrace), `final_context` (dict), `iterations_used` (int), `tools_used` (set).

#### `ReactAgent(tools, config=None, hitl=None)`

ReAct loop: think -> act -> observe -> conclude. Requires tools.

#### `ReflexionAgent(tools, config=None, evaluation_fn=None, max_reflections=3, hitl=None)`

ReAct + evaluation gate + verbal self-critique with episodic memory. If evaluation fails, reflects and retries.

#### `PlanExecuteAgent(tools=None, config=None, max_replans=2)`

Separates planning from execution. Creates a full plan upfront, executes steps sequentially, replans on failure. Tools optional.

#### `REWOOAgent(tools, config=None)`

Plans ALL tool calls in a single LLM pass with `#E1`, `#E2` evidence references, then executes sequentially. More token-efficient than ReAct.

#### `SelfConsistencyAgent(config=None, num_samples=5, aggregation_fn=None)`

Generates multiple independent answers at varying temperatures, aggregates via majority vote. No tools needed.

#### `DebateAgent(config=None, num_rounds=3, proposer_persona="", critic_persona="", judge_persona="")`

Multi-round debate: proposer argues, critic challenges, judge evaluates. Produces nuanced analysis of controversial topics. No tools needed.

#### `EvaluatorOptimizerAgent(evaluation_fn, config=None, max_refinements=3)`

Generate -> evaluate -> refine loop. External `evaluation_fn(output, context) -> EvaluationResult` drives the refinement cycle. No tools needed.

#### `MakerCheckerAgent(maker_instructions, checker_instructions, config=None, max_revisions=3, quality_threshold=0.7)`

Two-persona quality loop: maker generates content, checker evaluates against criteria. Continues until checker approves or max revisions reached. No tools needed.

#### `PromptChainAgent(chain, config=None)`

Linear pipeline of `ChainStep` objects. Each step's output feeds into the next. Useful for multi-stage generation (research -> draft -> polish).

#### `OrchestratorAgent(worker_factory=None, tools=None, config=None, max_workers=5)`

Decomposes tasks into subtasks, delegates to worker agents (via `worker_factory`), collects results, synthesizes. Tools optional.

#### `ADaPTAgent(tools=None, config=None, max_depth=3)`

Adaptive: tries direct solution first, assesses quality, decomposes recursively if needed. Tools optional.

#### `ReasoningReactAgent(tools, config=None, hitl=None, reasoning_model=None)`

ReAct agent with integrated structured reasoning via FSM stacking. Auto-registers a `reason` pseudo-tool. When the LLM selects `reason`, the agent pushes a reasoning FSM (from `fsm_llm_reasoning`) onto the stack to apply structured reasoning strategies. Requires the `fsm_llm_reasoning` package to be installed.

### Key Data Models

#### `AgentConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `DEFAULT_LLM_MODEL` | LLM model identifier |
| `max_iterations` | `int` | `10` | Maximum agent iterations |
| `timeout_seconds` | `float` | `120.0` | Total timeout for agent execution |
| `temperature` | `float` | `0.5` | LLM temperature |
| `max_tokens` | `int` | `2000` | Max LLM response tokens |

#### `AgentResult`

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `str` | Final answer text |
| `success` | `bool` | Whether the agent completed successfully |
| `trace` | `AgentTrace` | Execution trace with tool calls |
| `final_context` | `dict` | Final conversation context |
| `iterations_used` | `int` | Number of iterations consumed |
| `tools_used` | `set[str]` | Names of tools invoked |

#### `ToolDefinition`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *required* | Tool name (alphanumeric + underscores/hyphens) |
| `description` | `str` | *required* | Tool description shown to the LLM |
| `parameter_schema` | `dict` | `{}` | JSON Schema for tool parameters |
| `requires_approval` | `bool` | `False` | Whether tool requires HITL approval |

#### `ApprovalRequest`

| Field | Type | Description |
|-------|------|-------------|
| `tool_call` | `ToolCall` | The tool call awaiting approval |
| `context` | `dict` | Current agent context |
| `reason` | `str` | Why approval is needed |

---

## Workflow Engine (`fsm_llm_workflows`)

Async event-driven workflow orchestration with 8 step types, a Python DSL, and FSM integration via `ConversationStep`.

### `WorkflowEngine`

```python
from fsm_llm_workflows import WorkflowEngine, create_workflow, auto_step
from fsm_llm.handlers import HandlerSystem

engine = WorkflowEngine(handler_system=HandlerSystem())
engine.register_workflow(my_workflow)
instance_id = await engine.start_workflow("my_workflow", initial_context={"key": "value"})
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `handler_system` | `HandlerSystem \| None` | `None` | Handler system for registration. Creates new if omitted |
| `max_concurrent_workflows` | `int` | `100` | Maximum concurrent workflow instances |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `register_workflow(workflow)` | `None` | Register a workflow definition |
| `start_workflow(workflow_id, initial_context, instance_id, workflow_timeout)` | `str` | Start a workflow instance, returns instance ID |
| `advance_workflow(instance_id, user_input)` | `bool` | Advance a workflow instance |
| `cancel_workflow(instance_id, reason)` | `bool` | Cancel a running workflow instance |
| `process_event(event)` | `list[str]` | Process an external event, returns affected instance IDs |
| `register_event_listener(instance_id, event_type, ...)` | `None` | Register a workflow to listen for an event |
| `schedule_timer(instance_id, delay_seconds, next_state)` | `None` | Schedule a timer for a workflow instance |
| `get_workflow_instance(instance_id)` | `WorkflowInstance \| None` | Get a workflow instance by ID |
| `get_workflow_definition(workflow_id)` | `WorkflowDefinition \| None` | Get a workflow definition by ID |
| `get_workflow_status(instance_id)` | `WorkflowStatus \| None` | Get the status of a workflow instance |
| `get_workflow_context(instance_id)` | `dict \| None` | Get the context of a workflow instance |
| `get_active_workflows()` | `list[str]` | Get list of active workflow instance IDs |
| `get_statistics()` | `dict` | Get engine statistics |

### `WorkflowDefinition`

```python
class WorkflowDefinition(BaseModel):
    workflow_id: str
    name: str
    description: str = ""
    steps: dict[str, WorkflowStep] = {}
    initial_step_id: str | None = None
    metadata: dict[str, Any] = {}
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `with_step(step, is_initial=False)` | `WorkflowDefinition` | Add a step (chainable) |
| `with_initial_step(step)` | `WorkflowDefinition` | Add the initial step (chainable) |
| `validate()` | `None` | Validate definition (raises `WorkflowValidationError`) |
| `has_cycles()` | `bool` | Check for cycles in the workflow graph |
| `get_terminal_states()` | `set[str]` | Get states with no outgoing transitions |
| `serialize()` | `dict` | Serialize to dictionary |

### Step Types (8 types)

All steps inherit from `WorkflowStep` and implement `async execute(context) -> WorkflowStepResult`.

| Step Class | Purpose | Key Parameters |
|------------|---------|----------------|
| `AutoTransitionStep` | Execute action, auto-transition | `next_state`, `action` (callable) |
| `APICallStep` | External API integration | `api_function`, `success_state`, `failure_state`, `input_mapping`, `output_mapping` |
| `ConditionStep` | Branching logic | `condition` (callable), `true_state`, `false_state` |
| `LLMProcessingStep` | LLM-based processing | `llm_interface`, `prompt_template`, `context_mapping`, `output_mapping`, `next_state` |
| `WaitForEventStep` | Wait for external event | `config` (WaitEventConfig: `event_type`, `success_state`, `timeout_seconds`, `timeout_state`) |
| `TimerStep` | Wait for duration | `delay_seconds`, `next_state` |
| `ParallelStep` | Parallel execution | `steps` (list), `next_state`, `aggregation_function` |
| `ConversationStep` | Run FSM conversation | `fsm_file` or `fsm_definition`, `model`, `auto_messages`, `context_mapping`, `max_turns` |

### DSL Functions

```python
from fsm_llm_workflows import (
    create_workflow, auto_step, api_step, llm_step, condition_step,
    wait_event_step, timer_step, parallel_step, conversation_step,
    workflow_builder, linear_workflow, conditional_workflow, event_driven_workflow,
)
```

| Function | Returns | Description |
|----------|---------|-------------|
| `create_workflow(workflow_id, name, description)` | `WorkflowDefinition` | Create a new workflow definition |
| `auto_step(step_id, name, next_state, action, description)` | `AutoTransitionStep` | Create an auto-transition step |
| `api_step(step_id, name, api_function, success_state, failure_state, ...)` | `APICallStep` | Create an API call step |
| `condition_step(step_id, name, condition, true_state, false_state, ...)` | `ConditionStep` | Create a condition step |
| `llm_step(step_id, name, llm_interface, prompt_template, ...)` | `LLMProcessingStep` | Create an LLM processing step |
| `wait_event_step(step_id, name, event_type, success_state, ...)` | `WaitForEventStep` | Create a wait-for-event step |
| `timer_step(step_id, name, delay_seconds, next_state, ...)` | `TimerStep` | Create a timer step |
| `parallel_step(step_id, name, steps, next_state, ...)` | `ParallelStep` | Create a parallel execution step |
| `conversation_step(step_id, name, success_state, ...)` | `ConversationStep` | Create a step that runs an FSM conversation |
| `workflow_builder(workflow_id, name, description)` | `WorkflowBuilder` | Create a fluent workflow builder |
| `linear_workflow(workflow_id, name, steps, description)` | `WorkflowDefinition` | Create a linear sequential workflow |
| `conditional_workflow(workflow_id, name, initial_step, condition_step, ...)` | `WorkflowDefinition` | Create a conditional branching workflow |
| `event_driven_workflow(workflow_id, name, setup_steps, event_step, ...)` | `WorkflowDefinition` | Create an event-driven workflow |

### `WorkflowStatus`

```python
class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"       # waiting for events or timers
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

### `WorkflowEvent`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `event_type` | `str` | *required* | Event type identifier |
| `payload` | `dict[str, Any]` | `{}` | Event payload data |
| `timestamp` | `datetime` | auto | Event timestamp (UTC) |
| `event_id` | `str` | auto (UUID) | Unique event identifier |

### `WorkflowStepResult`

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the step succeeded |
| `data` | `dict[str, Any]` | Data to merge into workflow context |
| `next_state` | `str \| None` | State to transition to |
| `message` | `str \| None` | Status message |
| `error` | `str \| None` | Error message if failed |

**Class methods:** `success_result(data, next_state, message)`, `failure_result(error, next_state, message)`.

### `WorkflowInstance`

| Field | Type | Description |
|-------|------|-------------|
| `instance_id` | `str` | Unique instance identifier |
| `workflow_id` | `str` | Associated workflow definition ID |
| `current_step_id` | `str` | Current step being executed |
| `context` | `dict[str, Any]` | Workflow context data |
| `status` | `WorkflowStatus` | Current status |
| `created_at` | `datetime` | Creation timestamp |
| `deadline` | `datetime \| None` | Workflow timeout deadline |
| `history` | `list[WorkflowHistoryEntry]` | Execution history |

**Methods:** `is_active() -> bool`, `is_terminal() -> bool`.

---

## Monitor Dashboard (`fsm_llm_monitor`)

Web-based real-time monitoring dashboard for FSM conversations, agents, and workflows. Streams events and metrics via WebSocket.

### Quick Start

**CLI:**
```bash
pip install fsm-llm[monitor]
fsm-llm-monitor                  # Launch on port 8420, auto-opens browser
fsm-llm-monitor --port 9000      # Custom port
fsm-llm-monitor --no-browser     # Don't auto-open browser
```

**Programmatic:**
```python
from fsm_llm import API
from fsm_llm_monitor import MonitorBridge, configure
import uvicorn

api = API.from_file("my_bot.json", model="gpt-4o-mini")
bridge = MonitorBridge(api)
configure(bridge)

uvicorn.run("fsm_llm_monitor.server:app", host="0.0.0.0", port=8420)
```

### `MonitorBridge`

Connects an `EventCollector` to a live `API` instance and provides a unified query interface.

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api` | `API \| None` | `None` | FSM API instance to monitor |
| `config` | `MonitorConfig \| None` | `None` | Monitor configuration |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `connect(api)` | `None` | Connect to an API instance and register handlers |
| `disconnect()` | `None` | Disconnect from the API |
| `get_metrics()` | `MetricSnapshot` | Get current system metrics |
| `get_active_conversations()` | `list[str]` | List active conversation IDs |
| `get_conversation_snapshot(conversation_id)` | `ConversationSnapshot \| None` | Get snapshot of a conversation |
| `get_all_conversation_snapshots()` | `list[ConversationSnapshot]` | Get snapshots for all active conversations |
| `get_recent_events(limit)` | `list[MonitorEvent]` | Get recent events |
| `load_fsm_from_file(path)` | `FSMSnapshot \| None` | Load FSM definition for visualization |
| `load_fsm_from_dict(data)` | `FSMSnapshot \| None` | Convert FSM dict to snapshot |

**Properties:** `connected -> bool`, `collector -> EventCollector`, `config -> MonitorConfig`.

### `EventCollector`

Thread-safe event collector using bounded deques. Captures FSM lifecycle events via handler callbacks and log records via a loguru sink.

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_events` | `int` | `1000` | Maximum events to retain |
| `max_log_lines` | `int` | `5000` | Maximum log lines to retain |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `record_event(event)` | `None` | Record a monitor event |
| `record_log(record)` | `None` | Record a log entry |
| `get_events(limit)` | `list[MonitorEvent]` | Get recent events, newest first |
| `get_logs(limit, level)` | `list[LogRecord]` | Get logs, optionally filtered by level |
| `get_metrics()` | `MetricSnapshot` | Get current metric snapshot |
| `get_events_by_conversation(conversation_id, limit)` | `list[MonitorEvent]` | Get events for a specific conversation |
| `create_handler_callbacks()` | `dict[str, Any]` | Create callback functions for all 8 handler timing points |
| `create_loguru_sink()` | `Callable` | Create a loguru sink function |
| `clear()` | `None` | Clear all collected data |
| `cleanup()` | `None` | Remove the loguru sink |

### `MonitorConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `refresh_interval` | `float` | `1.0` | WebSocket refresh interval in seconds |
| `max_events` | `int` | `1000` | Maximum events to retain |
| `max_log_lines` | `int` | `5000` | Maximum log lines to retain |
| `log_level` | `str` | `"DEBUG"` | Minimum log level to capture |
| `show_internal_keys` | `bool` | `False` | Show internal context keys (prefixed with `_`) |
| `auto_scroll_logs` | `bool` | `True` | Auto-scroll log display |

### Key Models

#### `MonitorEvent`

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | `str` | Event type (e.g., `"conversation_start"`, `"state_transition"`, `"error"`) |
| `timestamp` | `datetime` | Event timestamp |
| `conversation_id` | `str \| None` | Associated conversation |
| `source_state` | `str \| None` | Source state (for transitions) |
| `target_state` | `str \| None` | Target state (for transitions) |
| `data` | `dict` | Additional event data |
| `level` | `str` | Event severity level |
| `message` | `str` | Human-readable description |

#### `MetricSnapshot`

| Field | Type | Description |
|-------|------|-------------|
| `active_conversations` | `int` | Currently active conversations |
| `total_events` | `int` | Total events captured |
| `total_errors` | `int` | Total errors recorded |
| `total_transitions` | `int` | Total state transitions |
| `events_per_type` | `dict[str, int]` | Event count by type |
| `states_visited` | `dict[str, int]` | Visit count by state |

#### `ConversationSnapshot`

| Field | Type | Description |
|-------|------|-------------|
| `conversation_id` | `str` | Conversation identifier |
| `current_state` | `str` | Current FSM state |
| `state_description` | `str` | State description |
| `is_terminal` | `bool` | Whether in terminal state |
| `context_data` | `dict` | Collected context data |
| `message_history` | `list[dict]` | Conversation messages |
| `stack_depth` | `int` | FSM stack depth |

### REST API Summary

The monitor dashboard exposes 35+ REST endpoints organized into groups:

| Group | Endpoints | Purpose |
|-------|-----------|---------|
| **Core** | `GET /api/metrics`, `GET /api/events`, `GET /api/logs`, `GET /api/config`, `POST /api/config`, `GET /api/info`, `GET /api/capabilities` | Metrics, events, logs, configuration |
| **WebSocket** | `WS /ws` | Real-time metrics + event streaming (1s poll) |
| **Instances** | `GET /api/instances`, `GET /api/instances/{id}`, `DELETE /api/instances/{id}` | Instance lifecycle management |
| **FSM** | `POST /api/fsm/launch`, `POST /api/fsm/{id}/start`, `POST /api/fsm/{id}/converse`, `POST /api/fsm/{id}/end`, `GET /api/fsm/{id}/conversations` | Launch and control FSM instances |
| **Agents** | `POST /api/agent/launch`, `GET /api/agent/{id}/status`, `GET /api/agent/{id}/result`, `POST /api/agent/{id}/cancel` | Launch and monitor agent instances |
| **Workflows** | `POST /api/workflow/launch`, `POST /api/workflow/{id}/advance`, `POST /api/workflow/{id}/cancel`, `GET /api/workflow/{id}/status` | Launch and control workflow instances |
| **Presets** | `GET /api/presets`, `GET /api/preset/fsm/{id}` | Browse example FSM presets |
| **Visualizer** | `POST /api/fsm/visualize`, `GET /api/fsm/visualize/preset/{id}`, `GET /api/agent/visualize`, `GET /api/workflow/visualize` | FSM, agent, and workflow visualization |

**Interactive docs:** Available at `http://localhost:8420/api/docs` (Swagger UI).

---

## Constants

### Environment Variables

```python
ENV_LLM_MODEL = "LLM_MODEL"
ENV_LLM_TEMPERATURE = "LLM_TEMPERATURE"
ENV_LLM_MAX_TOKENS = "LLM_MAX_TOKENS"
ENV_FSM_PATH = "FSM_PATH"
```

### Default Values

```python
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_HISTORY_SIZE = 5
DEFAULT_MAX_MESSAGE_LENGTH = 1000
DEFAULT_HANDLER_TIMEOUT = 30.0
DEFAULT_STEP_TIMEOUT = 120.0
```

---

## Complete Example

```python
from fsm_llm import API, HandlerTiming
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
