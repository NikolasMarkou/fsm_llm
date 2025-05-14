# LLM-FSM Technical Reference

LLM-FSM is a framework that implements Finite State Machines for Large Language Models to address their inherent statelessness in conversational applications. The framework functions by maintaining explicit state representation in the form of JSON structures embedded in system prompts, enabling deterministic conversation flows while leveraging LLMs' natural language capabilities. LLM-FSM provides a comprehensive solution for structured information extraction, persistent context management, and conditional transition logic through a JsonLogic implementation. The architecture separates responsibilities between the FSM (handling state management and transition rules), the LLM (handling natural language understanding and generation), and the Python framework (orchestrating the interaction between FSM and LLM). This implementation enables robust, predictable conversational experiences that maintain context persistence while supporting complex branching logic, provider-agnostic LLM integration, and explicit validation of state transitions.

## Schema Definitions

### Core Data Structures

#### FSMDefinition

```typescript
interface FSMDefinition {
  name: string;                   // FSM identifier
  description: string;            // Human-readable description
  initial_state: string;          // Starting state ID
  version: string;                // Schema version, default "3.0"
  persona?: string;               // Optional response persona
  states: {[id: string]: State};  // Map of all state definitions
}
```

#### State

```typescript
interface State {
  id: string;                             // Unique state identifier
  description: string;                    // State description
  purpose: string;                        // Functional purpose of state
  transitions: Transition[];              // Available transitions
  required_context_keys?: string[];       // Required context keys
  instructions?: string;                  // LLM instructions for this state
  example_dialogue?: {[role: string]: string}[]; // Optional example conversations
}
```

#### Transition

```typescript
interface Transition {
  target_state: string;            // Target state ID
  description: string;             // Transition description
  conditions?: TransitionCondition[]; // Optional conditions
  priority: number;                // Priority (lower = higher), default 100
}
```

#### TransitionCondition

```typescript
interface TransitionCondition {
  description: string;               // Human-readable description
  requires_context_keys?: string[];  // Required context keys
  logic?: JsonLogicExpression;       // Optional logic expression
}
```

#### FSMInstance

```typescript
interface FSMInstance {
  fsm_id: string;            // Reference to FSM definition
  current_state: string;     // Current state ID
  context: FSMContext;       // Runtime context
  persona?: string;          // Optional persona
}
```

#### FSMContext

```typescript
interface FSMContext {
  data: {[key: string]: any};           // Context data
  conversation: Conversation;           // Conversation history
  metadata: {[key: string]: any};       // Additional metadata
}
```

#### Conversation

```typescript
interface Conversation {
  exchanges: {[role: string]: string}[];  // Conversation history
  max_history_size: number;               // Maximum exchanges to keep
  max_message_length: number;             // Maximum message length
}
```

#### LLMRequest

```typescript
interface LLMRequest {
  system_prompt: string;       // System prompt
  user_message: string;        // User input
  context?: {[key: string]: any}; // Optional additional context
}
```

#### LLMResponse

```typescript
interface LLMResponse {
  transition: StateTransition;  // State transition
  message: string;              // User-facing message
  reasoning?: string;           // Optional reasoning
}
```

#### StateTransition

```typescript
interface StateTransition {
  target_state: string;           // Target state ID
  context_update: {[key: string]: any}; // Context updates
}
```

## API Specification

### FSMManager

Primary interface for managing FSM-based conversations.

#### Constructor

```python
def __init__(
    self,
    fsm_loader: Callable[[str], FSMDefinition] = load_fsm_definition,
    llm_interface: LLMInterface = None,
    prompt_builder: Optional[PromptBuilder] = None,
    max_history_size: int = DEFAULT_MAX_HISTORY_SIZE,
    max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH
) -> None
```

Parameters:
- `fsm_loader`: Function that loads an FSM definition by ID
- `llm_interface`: Interface for communicating with LLMs
- `prompt_builder`: Builder for creating prompts (optional)
- `max_history_size`: Maximum conversation exchanges to keep
- `max_message_length`: Maximum message length in characters

#### Methods

##### start_conversation

```python
def start_conversation(
    self,
    fsm_id: str,
    initial_context: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]
```

Parameters:
- `fsm_id`: The ID of the FSM definition or path to FSM file
- `initial_context`: Optional initial context data

Returns:
- Tuple of (conversation_id, initial_response)

Errors:
- `ValueError`: If the FSM definition cannot be loaded
- `FSMError`: If the FSM definition is invalid

##### process_message

```python
def process_message(
    self,
    conversation_id: str,
    message: str
) -> str
```

Parameters:
- `conversation_id`: The conversation ID
- `message`: The user's message

Returns:
- The system's response

Errors:
- `ValueError`: If the conversation ID is not found
- `LLMResponseError`: If there's an error processing the LLM response
- `InvalidTransitionError`: If the proposed transition is invalid

##### is_conversation_ended

```python
def is_conversation_ended(
    self,
    conversation_id: str
) -> bool
```

Parameters:
- `conversation_id`: The conversation ID

Returns:
- True if the conversation has ended, False otherwise

Errors:
- `ValueError`: If the conversation ID is not found

##### get_conversation_data

```python
def get_conversation_data(
    self,
    conversation_id: str
) -> Dict[str, Any]
```

Parameters:
- `conversation_id`: The conversation ID

Returns:
- The context data collected during the conversation

Errors:
- `ValueError`: If the conversation ID is not found

##### validate_transition

```python
def validate_transition(
    self,
    instance: FSMInstance,
    target_state: str,
    conversation_id: Optional[str] = None
) -> Tuple[bool, Optional[str]]
```

Parameters:
- `instance`: The FSM instance
- `target_state`: The target state
- `conversation_id`: Optional conversation ID for logging

Returns:
- Tuple of (is_valid, error_message)

##### end_conversation

```python
def end_conversation(
    self,
    conversation_id: str
) -> None
```

Parameters:
- `conversation_id`: The conversation ID

Errors:
- `ValueError`: If the conversation ID is not found

### LLMInterface

Interface for communicating with LLMs.

#### Methods

##### send_request

```python
def send_request(
    self,
    request: LLMRequest
) -> LLMResponse
```

Parameters:
- `request`: The LLM request

Returns:
- The LLM's response

Errors:
- `LLMResponseError`: If there's an error processing the LLM response

### LiteLLMInterface

Implementation of LLMInterface using LiteLLM for provider-agnostic operation.

#### Constructor

```python
def __init__(
    self,
    model: str,
    api_key: Optional[str] = None,
    enable_json_validation: bool = True,
    **kwargs
) -> None
```

Parameters:
- `model`: The model to use (e.g., "gpt-4", "claude-3-opus")
- `api_key`: Optional API key (will use environment variables if not provided)
- `enable_json_validation`: Whether to enable JSON schema validation
- `**kwargs`: Additional arguments to pass to LiteLLM

## JsonLogic Expression System

### Expression Types

#### Comparison Operators

- `==`: Soft equality with type coercion
- `===`: Strict equality (value and type)
- `!=`: Soft inequality
- `!==`: Strict inequality
- `>`: Greater than
- `>=`: Greater than or equal
- `<`: Less than
- `<=`: Less than or equal

#### Logical Operators

- `!`: Logical NOT
- `!!`: Boolean cast
- `and`: Logical AND (all values must be truthy)
- `or`: Logical OR (at least one value must be truthy)

#### Conditional Operator

- `if`: Conditional logic with if/else branches

#### Access Operators

- `var`: Retrieve value from context using dot notation
- `missing`: Check for missing required fields
- `missing_some`: Check if at least N of M fields are present

#### Membership Operators

- `in`: Check if a value is in a collection
- `contains`: Check if a collection contains a value

#### Arithmetic Operators

- `+`: Addition (sum)
- `-`: Subtraction/negation
- `*`: Multiplication (product)
- `/`: Division
- `%`: Modulo

#### String Operators

- `cat`: String concatenation

### Expression Structure

JsonLogic expressions are structured as JSON objects where:
- The key is the operator name
- The value is an array of arguments (or a single argument)

Example:
```json
{
  "and": [
    {"==": [{"var": "customer.status"}, "vip"]},
    {">": [{"var": "customer.lifetime_value"}, 5000]}
  ]
}
```

### Expression Evaluation

```python
def evaluate_logic(
    logic: JsonLogicExpression,
    data: Dict[str, Any] = None
) -> Any
```

Parameters:
- `logic`: The JsonLogic expression to evaluate
- `data`: The data object to evaluate against

Returns:
- The result of evaluating the expression

### Typical Expression Patterns

#### Required Field Check

```json
{
  "conditions": [
    {
      "description": "Name has been provided",
      "requires_context_keys": ["name"]
    }
  ]
}
```

#### Logical Combination

```json
{
  "conditions": [
    {
      "description": "Customer is premium member",
      "logic": {
        "or": [
          {"==": [{"var": "customer.tier"}, "premium"]},
          {">": [{"var": "customer.lifetime_value"}, 5000]}
        ]
      }
    }
  ]
}
```

#### Complex Decision Tree

```json
{
  "logic": {
    "if": [
      {"==": [{"var": "issue.resolved"}, true]},
      {"var": "issue.resolution_time"},
      {"var": ["agent.estimated_time", 30]}
    ]
  }
}
```

## Prompt Construction

### System Prompt Structure

System prompts are structured as follows:

```
<task>
[Task description and overall instructions]
</task>

<fsm>
<current_state>[state_id]</current_state>
<current_state_description>[state_description]</current_state_description>
<current_purpose>[state_purpose]</current_purpose>

[Optional persona section if defined]
<persona>[persona_description]</persona>

[State-specific instructions]
<state_instructions>[instructions]</state_instructions>

[Information collection directives if required_context_keys present]
<information_to_collect>[required_keys]</information_to_collect>
<information_extraction_instructions>[extraction_instructions]</information_extraction_instructions>

[Available transitions with conditions and priorities]
<available_state_transitions>
[transitions_json]
</available_state_transitions>

[Transition rules]
<transition_rules>[rules]</transition_rules>

[Current context data]
<current_context>
[context_json]
</current_context>

[Conversation history if available]
<conversation_history>
[history_json]
</conversation_history>

[Response format instructions]
<response>
[response_instructions]
<response_format>
[response_json_schema]
</response_format>
</response>

[Important guidelines]
<instructions>[guidelines]</instructions>
</fsm>
```

### PromptBuilder Class

```python
class PromptBuilder:
    def __init__(self, max_history_size: int = DEFAULT_MAX_HISTORY_SIZE) -> None
    
    def build_system_prompt(self, instance: FSMInstance, state: State) -> str
```

## Error Handling

### Exception Hierarchy

```
FSMError
  ├── StateNotFoundError
  ├── InvalidTransitionError
  └── LLMResponseError
```

### Error Scenarios

1. **State Not Found**: When a referenced state doesn't exist in the FSM definition
2. **Invalid Transition**: When a transition fails validation
3. **LLM Response Error**: When the LLM response cannot be parsed or is invalid
4. **Missing Required Fields**: When required context keys are missing

## Conversation Lifecycle

1. **Initialization**:
   - Create FSM instance from definition
   - Set initial state
   - Initialize empty context or with provided initial_context

2. **Message Processing**:
   - Add user message to conversation history
   - Get current state definition
   - Generate system prompt
   - Send request to LLM
   - Parse LLM response
   - Update context with extracted data
   - Validate proposed transition
   - Update state if transition is valid
   - Add system response to conversation history

3. **Termination**:
   - Conversation ends when reaching a state with no outgoing transitions
   - All data collected during conversation is available via get_conversation_data()

## Implementation Details

### Context Persistence

Context data persists across state transitions because it's maintained at the FSM instance level, not within individual states. This ensures:

1. Information collected in any state remains available later
2. No explicit context copying between states is needed
3. The entire conversation history is preserved

### Transition Validation

Transition validation includes multiple checks:

1. Target state exists in the FSM definition
2. Current state has a transition to the target state
3. All required context keys are present in the context
4. Any JsonLogic conditions evaluate to true

### LLM Response Formatting

The LLM must return a structured response with:

1. A transition decision (where to go next)
2. Context updates (what information was extracted)
3. User-facing message (what to tell the user)
4. Optional reasoning (for debugging)

## Example Implementation

```python
from llm_fsm.llm import LiteLLMInterface
from llm_fsm.fsm import FSMManager
from llm_fsm.utilities import load_fsm_definition

# Initialize the LLM interface
llm_interface = LiteLLMInterface(
    model="gpt-4o",
    api_key="your-api-key",
    temperature=0.5
)

# Create an FSM manager
fsm_manager = FSMManager(
    fsm_loader=load_fsm_definition,
    llm_interface=llm_interface
)

# Start a conversation
conversation_id, response = fsm_manager.start_conversation("personal_information_collection.json")

# Process messages until conversation ends
while not fsm_manager.is_conversation_ended(conversation_id):
    user_input = input("User: ")
    response = fsm_manager.process_message(conversation_id, user_input)
    print(f"System: {response}")

# Get collected data
collected_data = fsm_manager.get_conversation_data(conversation_id)
```

## FSM Definition Example

```json
{
  "name": "Customer Support Router",
  "description": "Routes customer support requests based on customer status and issue type",
  "initial_state": "greeting",
  "version": "3.0",
  "states": {
    "greeting": {
      "id": "greeting",
      "description": "Initial greeting",
      "purpose": "Welcome the customer and identify their status",
      "transitions": [
        {
          "target_state": "premium_support",
          "description": "Route to premium support",
          "conditions": [
            {
              "description": "Customer is premium member",
              "logic": {
                "or": [
                  {"==": [{"var": "customer.tier"}, "premium"]},
                  {">": [{"var": "customer.lifetime_value"}, 5000]}
                ]
              }
            }
          ]
        },
        {
          "target_state": "standard_support",
          "description": "Route to standard support",
          "priority": 10
        }
      ]
    },
    "premium_support": {
      "id": "premium_support",
      "description": "Premium support handling",
      "purpose": "Handle premium customer issues with high priority",
      "required_context_keys": ["issue.description"],
      "transitions": [
        {
          "target_state": "billing_issues",
          "description": "Route billing issues to specialized team",
          "conditions": [
            {
              "description": "Issue relates to billing",
              "logic": {
                "or": [
                  {"==": [{"var": "issue.category"}, "billing"]},
                  {"in": ["bill", {"var": "issue.description"}]},
                  {"in": ["payment", {"var": "issue.description"}]},
                  {"in": ["charge", {"var": "issue.description"}]}
                ]
              }
            }
          ],
          "priority": 1
        },
        {
          "target_state": "general_resolution",
          "description": "Handle non-billing issues",
          "priority": 2
        }
      ]
    },
    "standard_support": {
      "id": "standard_support",
      "description": "Standard support handling",
      "purpose": "Handle regular customer issues",
      "required_context_keys": ["issue.description"],
      "transitions": [
        {
          "target_state": "general_resolution",
          "description": "Route to general resolution",
          "priority": 1
        }
      ]
    },
    "billing_issues": {
      "id": "billing_issues",
      "description": "Billing issues handling",
      "purpose": "Resolve customer billing problems",
      "transitions": [
        {
          "target_state": "resolution_confirmation",
          "description": "Proceed to confirmation after handling billing",
          "priority": 1
        }
      ]
    },
    "general_resolution": {
      "id": "general_resolution",
      "description": "General issue resolution",
      "purpose": "Resolve general customer issues",
      "transitions": [
        {
          "target_state": "resolution_confirmation",
          "description": "Proceed to confirmation after handling issue",
          "priority": 1
        }
      ]
    },
    "resolution_confirmation": {
      "id": "resolution_confirmation",
      "description": "Confirm resolution",
      "purpose": "Verify the customer's issue has been resolved",
      "required_context_keys": ["issue.resolved"],
      "transitions": [
        {
          "target_state": "feedback",
          "description": "Issue resolved, request feedback",
          "conditions": [
            {
              "description": "Issue is marked as resolved",
              "logic": {
                "==": [{"var": "issue.resolved"}, true]
              }
            }
          ],
          "priority": 1
        },
        {
          "target_state": "escalation",
          "description": "Issue not resolved, escalate",
          "priority": 2
        }
      ]
    },
    "feedback": {
      "id": "feedback",
      "description": "Collect feedback",
      "purpose": "Get customer feedback on support experience",
      "required_context_keys": ["feedback.rating"],
      "transitions": [
        {
          "target_state": "end",
          "description": "End conversation after feedback",
          "priority": 1
        }
      ]
    },
    "escalation": {
      "id": "escalation",
      "description": "Escalate unresolved issue",
      "purpose": "Escalate issue to higher support tier",
      "transitions": [
        {
          "target_state": "end",
          "description": "End conversation after escalation",
          "priority": 1
        }
      ]
    },
    "end": {
      "id": "end",
      "description": "End of conversation",
      "purpose": "Thank customer and conclude interaction",
      "transitions": []
    }
  }
}
```

## Error Detection and Recovery

```python
try:
    # Start a conversation
    conversation_id, response = fsm_manager.start_conversation("customer_support.json")
    
    # Process first message
    user_input = "I'm having a problem with my subscription"
    response = fsm_manager.process_message(conversation_id, user_input)
    
except StateNotFoundError as e:
    logging.error(f"State not found: {str(e)}")
    # Recover by creating a new conversation with a different FSM
    conversation_id, response = fsm_manager.start_conversation("fallback_support.json")
    
except InvalidTransitionError as e:
    logging.error(f"Invalid transition: {str(e)}")
    # Force transition to a valid state
    fsm_manager.instances[conversation_id].current_state = "standard_support"
    
except LLMResponseError as e:
    logging.error(f"LLM response error: {str(e)}")
    # Provide fallback response
    response = "I'm sorry, I'm having trouble processing your request. Could you please rephrase?"
```

## Performance Considerations

1. **Context Window Management**:
   - `max_history_size` limits conversation history
   - `max_message_length` truncates long messages

2. **LLM Provider Selection**:
   - Anthropic models (e.g., claude-3-opus) provide more reliable structured outputs
   - OpenAI models may offer better latency for simpler use cases
   - Use temperature=0 for most consistent responses

3. **Transition Logic Optimization**:
   - Use priority values to order transitions efficiently
   - Place most commonly used transitions with highest priority (lowest numeric value)
   - Use JsonLogic sparingly for complex conditions

4. **Cache Management**:
   - FSM definitions are cached automatically
   - Invalidate cache if definitions change at runtime

## Thread Safety

The FSMManager is not thread-safe by default. When using in a multi-threaded environment:

1. Use conversation_id as a unique key
2. Implement synchronization around FSMManager method calls
3. Consider using separate FSMManager instances per thread

## Edge Cases

1. **Ambiguous User Input**:
   - LLM extracts information based on best interpretation
   - Stay in current state if required information cannot be extracted

2. **Invalid Target States**:
   - Automatically reverts to staying in current state
   - Logs error but does not crash

3. **Context Update Conflicts**:
   - Last update wins for same key
   - Consider domain-specific merge logic for complex cases

4. **Unreachable States**:
   - Validation detects and fails for orphaned states
   - Ensures all states are reachable from initial state

5. **Terminal State Detection**:
   - A state is terminal if transitions list is empty
   - Self-loops do not make a state terminal