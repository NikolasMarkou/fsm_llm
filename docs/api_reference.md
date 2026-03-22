# API Reference - Complete API Documentation

This document provides comprehensive documentation for the FSM-LLM API, including all classes, methods, and parameters.

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

11 agent patterns with a consistent API: `agent.run(task) -> AgentResult`.

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
