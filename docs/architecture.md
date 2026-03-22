# Architecture Deep Dive - How It All Works

This document provides a comprehensive technical overview of the FSM-LLM architecture, explaining how all components work together to create stateful conversations with Large Language Models.

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [The Prompt Engineering Layer](#the-prompt-engineering-layer)
5. [State Management](#state-management)
6. [Context Handling](#context-handling)
7. [Handler Execution Pipeline](#handler-execution-pipeline)
8. [FSM Stacking Architecture](#fsm-stacking-architecture)
9. [Security Considerations](#security-considerations)
10. [Performance Architecture](#performance-architecture)
11. [Extension Points](#extension-points)

---

## System Overview

FSM-LLM is built on an **improved 2-pass architecture** that separates data extraction, transition evaluation, and response generation for better conversation quality:

1. **Pass 1 — Analysis & Transition**: Data extraction from user input, transition evaluation using configurable logic, state management and context updates.
2. **Pass 2 — Response Generation**: Response generation based on the new state and updated context, producing consistent, contextually-appropriate responses.

The system uses a layered architecture that separates concerns and provides clear extension points:

```
┌─────────────────────────────────────────────────────────────┐
│                      User Application                        │
├─────────────────────────────────────────────────────────────┤
│                         API Layer                            │
│                    (Simplified Interface)                    │
├─────────────────────────────────────────────────────────────┤
│     FSM Manager     │    Handler System    │   LLM Bridge   │
├─────────────────────────────────────────────────────────────┤
│                    Core FSM Engine                           │
│              (State Machine Logic)                           │
├─────────────────────────────────────────────────────────────┤
│   Prompt Builder   │  Context Manager  │  History Tracker   │
├─────────────────────────────────────────────────────────────┤
│                    Storage Layer                             │
│                  (In-Memory Dicts)                            │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Extensibility**: Clear interfaces for adding new functionality
3. **Testability**: Components can be tested in isolation
4. **Performance**: Efficient memory usage and minimal overhead
5. **Security**: Input sanitization and prompt injection prevention

---

## Core Components

### 1. API Class (`api.py`)

The API class serves as the main entry point, providing a simplified interface that hides internal complexity.

```python
class API:
    def __init__(self, fsm_definition, llm_interface=None, **kwargs):
        # Process FSM definition
        self.fsm_definition, self.fsm_id = self.process_fsm_definition(fsm_definition)

        # Initialize components
        self.llm_interface = llm_interface or LiteLLMInterface(**kwargs)
        self.handler_system = HandlerSystem()
        self.fsm_manager = FSMManager(
            fsm_loader=custom_fsm_loader,
            llm_interface=self.llm_interface,
            handler_system=self.handler_system
        )

        # Conversation tracking
        self.active_conversations = {}
        self.conversation_stacks = {}
```

**Key Responsibilities:**
- FSM definition processing and validation
- Component initialization and wiring
- Conversation lifecycle management
- FSM stacking orchestration

### 2. FSM Manager (`fsm.py`) and Message Pipeline (`pipeline.py`)

The FSM Manager handles FSM instance lifecycle and delegates message processing to `MessagePipeline`.

```python
class FSMManager:
    def __init__(self, fsm_loader, llm_interface, handler_system,
                 data_extraction_prompt_builder, response_generation_prompt_builder,
                 transition_prompt_builder, transition_evaluator,
                 max_history_size, max_message_length, handler_error_mode,
                 max_fsm_cache_size):
        self._pipeline = MessagePipeline(...)  # Delegates 2-pass processing
        self.instances = {}  # conversation_id -> FSMInstance
        self._conversation_locks = {}  # Per-conversation thread locks
```

The `MessagePipeline` (`pipeline.py`) implements the actual 2-pass processing:

```python
class MessagePipeline:
    def process(self, instance, user_input, conversation_id):
        # === PASS 1: Data Extraction & Transition ===
        # 1. PRE_PROCESSING handlers
        # 2. Build extraction prompt → LLM extract_data() call
        # 3. CONTEXT_UPDATE handlers (if keys changed)
        # 4. Evaluate transitions (deterministic first, LLM fallback if ambiguous)
        # 5. PRE_TRANSITION handlers
        # 6. Perform state transition
        # 7. POST_TRANSITION handlers (with rollback on failure)

        # === PASS 2: Response Generation ===
        # 8. POST_PROCESSING handlers
        # 9. Build response prompt for NEW state → LLM generate_response() call
        # 10. Return generated response
```

**Key Features:**
- Per-conversation thread locks for thread safety
- FSM definition caching with LRU eviction
- Handler integration at 8 timing points
- Automatic context updates with security filtering

### 3. Prompt Builders (`prompts.py`)

Three specialized prompt builders handle the different LLM roles, all inheriting from `BasePromptBuilder`:

```python
class DataExtractionPromptBuilder(BasePromptBuilder):
    """Builds prompts for Pass 1: data extraction from user input."""

class ResponseGenerationPromptBuilder(BasePromptBuilder):
    """Builds prompts for Pass 2: user-facing response generation."""

class TransitionPromptBuilder(BasePromptBuilder):
    """Builds prompts for LLM-assisted transition decisions (when rule-based evaluation is ambiguous)."""
```

Each builder produces structured XML-like prompts with task-specific sections, guidelines, and response format requirements.

**Security Features:**
- XML tag sanitization to prevent prompt injection
- CDATA protection for JSON data
- Token budget management
- Input validation

### 4. Handler System (`handlers.py`)

The Handler System implements a flexible plugin architecture for extending FSM behavior.

```python
class HandlerSystem:
    def execute_handlers(self, timing, current_state, target_state, context, updated_keys=None):
        # 1. Filter applicable handlers
        applicable = [h for h in self.handlers
                     if h.should_execute(timing, current_state, target_state, context)]

        # 2. Sort by priority
        applicable.sort(key=lambda h: h.priority)

        # 3. Execute in order
        updates = {}
        for handler in applicable:
            try:
                result = handler.execute(context)
                updates.update(result)
            except Exception as e:
                self._handle_error(e, handler)

        return updates
```

**Handler Architecture:**
- Self-determining execution (handlers decide when to run)
- Priority-based ordering
- Error isolation with configurable error modes
- Critical handler flag (always raises regardless of error mode)
- Configurable execution timeout

---

## Data Flow

### 1. Conversation Start

```mermaid
sequenceDiagram
    participant User
    participant API
    participant FSMManager
    participant Pipeline
    participant LLM
    participant Handlers

    User->>API: start_conversation(initial_context)
    API->>FSMManager: create FSM instance
    FSMManager->>Handlers: START_CONVERSATION
    FSMManager->>Pipeline: generate initial response
    Pipeline->>LLM: generate_response(prompt)
    LLM->>Pipeline: initial response
    Pipeline->>FSMManager: response
    FSMManager->>API: (conversation_id, response)
    API->>User: (conversation_id, response)
```

### 2. Message Processing

```mermaid
sequenceDiagram
    participant User
    participant API
    participant FSMManager
    participant Handlers
    participant LLM
    participant Context

    User->>API: converse(message, conv_id)
    API->>FSMManager: process_message
    FSMManager->>Pipeline: process(instance, message)
    Pipeline->>Handlers: PRE_PROCESSING
    Pipeline->>LLM: extract_data(prompt, message)
    LLM->>Pipeline: extracted data
    Pipeline->>Context: update context
    Pipeline->>Handlers: CONTEXT_UPDATE
    Pipeline->>Pipeline: evaluate transitions
    Pipeline->>Handlers: PRE_TRANSITION
    Pipeline->>Pipeline: change state
    Pipeline->>Handlers: POST_TRANSITION
    Pipeline->>Handlers: POST_PROCESSING
    Pipeline->>LLM: generate_response(prompt)
    LLM->>Pipeline: user-facing response
    Pipeline->>FSMManager: response
    FSMManager->>API: response
    API->>User: response
```

---

## The Prompt Engineering Layer

### Prompt Structure

The system uses a carefully designed XML-like structure for prompts:

```xml
<task>
    You are the Natural Language Understanding component in an FSM system.
    Process user input based on current state and determine transitions.
</task>

<fsm>
    <persona>Friendly customer service representative</persona>

    <current_state>
        <id>collect_email</id>
        <description>Collecting user email</description>
        <purpose>Get valid email address from user</purpose>
        <required_context_keys>email</required_context_keys>
    </current_state>

    <current_context><![CDATA[
    {
        "name": "John Doe",
        "timestamp": "2024-01-01T10:00:00"
    }
    ]]></current_context>

    <conversation_history><![CDATA[
    [
        {"user": "Hi, I need help"},
        {"system": "Hello! I'd be happy to help. What's your name?"},
        {"user": "I'm John Doe"}
    ]
    ]]></conversation_history>

    <transitions><![CDATA[
    [
        {
            "to": "collect_phone",
            "desc": "Valid email provided",
            "priority": 100
        }
    ]
    ]]></transitions>

    <response>
        Return JSON with structure:
        {
            "transition": {
                "target_state": "state_id",
                "context_update": {}
            },
            "message": "Your response",
            "reasoning": "Optional reasoning"
        }
    </response>
</fsm>
```

### Security Measures

1. **Tag Sanitization**: All user input is sanitized to prevent XML injection
2. **CDATA Wrapping**: JSON data is wrapped in CDATA to prevent parsing issues
3. **Token Management**: History is trimmed to stay within token limits

---

## State Management

### State Lifecycle

```
┌─────────────┐
│   Created   │ ──────┐
└─────────────┘       │
                      ▼
┌─────────────┐  ┌─────────────┐
│   Running   │◄─│   Waiting   │
└─────────────┘  └─────────────┘
       │                │
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│  Terminal   │  │   Failed    │
└─────────────┘  └─────────────┘
```

### Transition Evaluation

The `TransitionEvaluator` (`transition_evaluator.py`) evaluates transitions using a three-outcome model:

1. **DETERMINISTIC** — Exactly one transition matches. Proceed automatically.
2. **AMBIGUOUS** — Multiple transitions match. Delegates to LLM via `decide_transition()`.
3. **BLOCKED** — No transitions match conditions. Stay in current state.

Evaluation uses JsonLogic conditions (`expressions.py`) and required context key checks.

---

## Context Handling

### Context Structure

```python
class FSMContext:
    data: Dict[str, Any]           # User-defined context
    conversation: Conversation      # Conversation history
    metadata: Dict[str, Any]       # System metadata

    # Special keys (prefixed with _)
    # _conversation_id: Current conversation ID
    # _current_state: Current state ID
    # _previous_state: Previous state ID
    # _timestamp: Last update timestamp
    # _user_input: Most recent user input
```

### Context Flow in Stacked FSMs

```
Main FSM Context          Sub-FSM Context
┌─────────────────┐      ┌─────────────────┐
│ user_id: 123    │      │ user_id: 123    │ (inherited)
│ cart: [...]     │ ───> │ cart: [...]     │ (passed)
│ email: a@b.com  │      │ step: "details" │ (new)
└─────────────────┘      └─────────────────┘
                              │
                              ▼ (on pop)
                         ┌─────────────────┐
                         │ form_complete: T │
                         │ collected: {...} │
                         └─────────────────┘
```

### Merge Strategies

1. **UPDATE** (`ContextMergeStrategy.UPDATE`): Child context overwrites parent
2. **PRESERVE** (`ContextMergeStrategy.PRESERVE`): Only new keys added to parent

---

## Handler Execution Pipeline

### Execution Order

```
START_CONVERSATION
    │
    ▼
For each message:
    PRE_PROCESSING
        │
        ▼
    [LLM extract_data]
        │
        ▼
    CONTEXT_UPDATE (if keys changed)
        │
        ▼
    [Transition evaluation]
        │
        ▼
    PRE_TRANSITION (if transitioning)
        │
        ▼
    [State Change]
        │
        ▼
    POST_TRANSITION (if transitioned)
        │
        ▼
    POST_PROCESSING
        │
        ▼
    [LLM generate_response]
        │
        ▼
END_CONVERSATION (when complete)

ERROR (on any exception)
```

### Handler Priority System

```python
handlers = [
    Handler(priority=10),   # Executes first
    Handler(priority=50),   # Executes second
    Handler(priority=100),  # Executes third
]
```

### Error Handling Modes

1. **continue**: Log error, continue with other handlers (default)
2. **raise**: Stop execution, propagate exception

Handlers marked as `critical=True` always raise on error regardless of error mode.

---

## FSM Stacking Architecture

### Stack Structure

```
┌─────────────────────────┐
│   Stack for conv_123    │
├─────────────────────────┤
│ Frame 2: checkout_fsm   │ ← Current (top)
│   - conv_id: abc789     │
│   - shared: [user_id]   │
├─────────────────────────┤
│ Frame 1: detail_form    │
│   - conv_id: def456     │
│   - shared: [email]     │
├─────────────────────────┤
│ Frame 0: main_fsm       │ ← Root
│   - conv_id: conv_123   │
│   - shared: []          │
└─────────────────────────┘
```

### Push Operation

```python
def push_fsm(self, conversation_id, new_fsm_definition, **kwargs):
    # 1. Get current context
    current_context = self.get_data(conversation_id)

    # 2. Create new FSM instance
    new_instance = self.create_instance(new_fsm_definition)

    # 3. Transfer context
    if kwargs.get('inherit_context'):
        new_instance.context.update(current_context)
    if kwargs.get('context_to_pass'):
        new_instance.context.update(kwargs['context_to_pass'])

    # 4. Add to stack
    stack_frame = FSMStackFrame(
        fsm_definition=new_fsm_definition,
        conversation_id=new_instance.id,
        shared_context_keys=kwargs.get('shared_context_keys', [])
    )
    self.conversation_stacks[conversation_id].append(stack_frame)
```

### Pop Operation

```python
def pop_fsm(self, conversation_id, context_to_return=None, merge_strategy="update"):
    # 1. Get current and previous frames
    current = self.conversation_stacks[conversation_id].pop()
    previous = self.conversation_stacks[conversation_id][-1]

    # 2. Get final context from current FSM
    final_context = self.get_data(current.conversation_id)

    # 3. Merge contexts based on strategy
    if merge_strategy == "update":
        previous_context.update(final_context)
    elif merge_strategy == "preserve":
        for key, value in final_context.items():
            if key not in previous_context:
                previous_context[key] = value
```

---

## Security Considerations

### 1. Prompt Injection Prevention

```python
def _sanitize_text_for_prompt(self, text: str) -> str:
    # Escape XML-like tags that could break prompt structure
    critical_tags = ["fsm", "task", "response", "transitions"]

    for tag in critical_tags:
        # Handle opening tags with attributes
        text = re.sub(f'<{tag}[^>]*>', lambda m: html.escape(m.group(0)), text)
        # Handle closing tags
        text = re.sub(f'</{tag}>', lambda m: html.escape(m.group(0)), text)

    return text
```

### 2. Context Isolation

Each conversation has isolated context:
- No cross-conversation data access
- Separate memory allocation
- Independent state machines

### 3. Input Validation

```python
def validate_user_input(self, user_input: str) -> str:
    # Length check
    if len(user_input) > self.max_message_length:
        user_input = user_input[:self.max_message_length]

    # Sanitization
    user_input = self._sanitize_text_for_prompt(user_input)

    return user_input
```

---

## Performance Architecture

### Memory Management

1. **Conversation History Limits**
   ```python
   class Conversation:
       max_history_size: int = 5  # Keep last 5 exchanges
       max_message_length: int = 1000  # Truncate long messages
   ```

2. **Lazy Loading**
   ```python
   def get_fsm_definition(self, fsm_id):
       if fsm_id not in self.fsm_cache:
           self.fsm_cache[fsm_id] = self.fsm_loader(fsm_id)
       return self.fsm_cache[fsm_id]
   ```

3. **Resource Cleanup**
   ```python
   def end_conversation(self, conversation_id):
       # Clean up resources
       del self.instances[conversation_id]
       if conversation_id in self.conversation_stacks:
           del self.conversation_stacks[conversation_id]
   ```

### Optimization Strategies

1. **Handler Pre-filtering**
   ```python
   # Only check handlers that could potentially execute
   potential_handlers = [h for h in self.handlers
                        if timing in h.timings or not h.timings]
   ```

2. **Prompt Caching**
   - Cache static prompt sections
   - Only rebuild dynamic parts

3. **FSM Definition Caching**
   - LRU cache for loaded FSM definitions
   - Configurable cache size via `max_fsm_cache_size`

---

## Extension Points

### 1. Custom LLM Interface

Implement the three abstract methods of `LLMInterface`:

```python
class CustomLLMInterface(LLMInterface):
    def extract_data(self, request: DataExtractionRequest) -> DataExtractionResponse:
        """Extract structured data from user input."""
        response = your_llm_api(request.system_prompt, request.user_message)
        return DataExtractionResponse(
            extracted_data=parse_json(response),
            confidence=0.9
        )

    def generate_response(self, request: ResponseGenerationRequest) -> ResponseGenerationResponse:
        """Generate user-facing response."""
        response = your_llm_api(request.system_prompt, request.user_message)
        return ResponseGenerationResponse(message=response)

    def decide_transition(self, request: TransitionDecisionRequest) -> TransitionDecisionResponse:
        """Select the best transition when rule-based evaluation is ambiguous."""
        response = your_llm_api(request.system_prompt, request.user_message)
        return TransitionDecisionResponse(target_state=parse_target(response))
```

### 2. Custom FSM Loader

```python
def custom_loader(fsm_id: str) -> FSMDefinition:
    # Load from database
    fsm_data = database.get_fsm(fsm_id)
    return FSMDefinition(**fsm_data)

fsm_manager = FSMManager(fsm_loader=custom_loader)
```

### 3. Custom Handlers

```python
class DatabaseHandler(BaseHandler):
    def should_execute(self, timing, current_state, target_state, context, updated_keys):
        return timing == HandlerTiming.POST_TRANSITION

    def execute(self, context):
        # Save to database
        database.save_conversation_state(
            conversation_id=context["_conversation_id"],
            state=context["_current_state"],
            data=context
        )
        return {"saved_to_db": True}
```

### 4. Structured Classification

```python
from fsm_llm_classification import (
    Classifier, ClassificationSchema, IntentDefinition, IntentRouter
)

# Define intent classes
schema = ClassificationSchema(
    intents=[
        IntentDefinition(name="order_status", description="User asks about an order"),
        IntentDefinition(name="product_info", description="User asks about a product"),
        IntentDefinition(name="general_support", description="Anything else"),
    ],
    fallback_intent="general_support",
)

# Classify user input
classifier = Classifier(schema, model="gpt-4o-mini")
result = classifier.classify("Where is my order #12345?")
# result.intent == "order_status", result.confidence == 0.95

# Route to handlers
router = IntentRouter(schema)
router.register("order_status", handle_order_status)
router.register("general_support", handle_general)
response = router.route(user_message, result)
```

### 5. Workflow Integration

```python
from fsm_llm_workflows import WorkflowEngine, AutoTransitionStep

# Create workflow with FSM steps
workflow = WorkflowDefinition(
    workflow_id="onboarding",
    steps={
        "collect_info": AutoTransitionStep(
            next_state="verify_info",
            action=lambda ctx: api.converse("start", ctx["conv_id"])
        )
    }
)
```

---

## Summary

The FSM-LLM architecture achieves its goals through:

1. **Clear Separation**: Each component has a single responsibility
2. **Extensibility**: Multiple extension points for customization
3. **Security**: Built-in protections against common attacks
4. **Performance**: Efficient resource usage and optimization
5. **Flexibility**: Supports simple to complex conversation flows

The system provides a robust foundation for building stateful conversational AI applications while maintaining the flexibility to adapt to various use cases and requirements.
