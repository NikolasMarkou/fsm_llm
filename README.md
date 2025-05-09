# LLM-FSM: Adding State to the Stateless

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## The Problem: Stateless LLMs

Large Language Models are revolutionary for natural language understanding and generation. However, they have a fundamental limitation: **they are inherently stateless**. Each interaction is processed independently with only the context provided in the prompt. This makes building robust multi-turn conversations challenging:

- How do you maintain consistent state across interactions?
- How do you implement structured flows with clear transitions?
- How do you reliably collect and validate information over multiple turns?

## The Solution: Finite State Machines + LLMs

LLM-FSM combines the structured flow of Finite State Machines with the powerful language capabilities of Large Language Models:

> We keep the state as a JSON structure inside the system prompt of an LLM, describing transition nodes and conditions, while using the LLM's reasoning capabilities to determine when to transition and what information to extract.

This hybrid approach gives you the best of both worlds:
- ✅ **Predictable conversation flows** with clear rules and transitions
- ✅ **Natural language understanding** powered by state-of-the-art LLMs
- ✅ **Persistent context** across the entire conversation
- ✅ **Dynamic adaptation** to user inputs

## Example: A Simple FSM

Here's a visual representation of a basic FSM for collecting user information:

![FSM Diagram](./images/fsm-diagram-example.png "User Information Collection Flow")

## Sample Conversation

```
System: Hello! I'm an assistant that will be collecting some information. Could you tell me your name?

User: I'm Alex Smith

System: Nice to meet you, Alex Smith! Could you please provide your email address?

User: alex.smith@example.com

System: Thank you! And what's your age?

User: I'm 32 years old

System: Great! Let me summarize what I've collected:
- Name: Alex Smith
- Email: alex.smith@example.com
- Age: 32
Is this information correct?

User: Actually, my email is alex.j.smith@example.com

System: I'll correct that for you. Your email is now alex.j.smith@example.com. 
Let me update the summary:
- Name: Alex Smith
- Email: alex.j.smith@example.com
- Age: 32
Is this information correct now?

User: Yes, that's correct

System: Thank you for confirming your information! Your profile has been saved.
```

## How It Works

At its core, LLM-FSM defines states, transitions, and conditions in a structured JSON format that controls the conversation flow:

```json
{
  "name": "User Information Collection",
  "description": "A conversation flow to collect user information",
  "initial_state": "welcome",
  "states": {
    "welcome": {
      "id": "welcome",
      "description": "Initial welcome state",
      "purpose": "Welcome the user and explain purpose",
      "transitions": [
        {
          "target_state": "collect_name",
          "description": "Always transition to collecting name",
          "priority": 0
        }
      ]
    },
    "collect_name": {
      "id": "collect_name",
      "description": "Collect user's name",
      "purpose": "Ask for and record the user's name",
      "required_context_keys": ["name"],
      "transitions": [
        {
          "target_state": "collect_email",
          "description": "Transition once name is obtained",
          "conditions": [
            {
              "description": "Name has been provided",
              "requires_context_keys": ["name"]
            }
          ]
        }
      ]
    }
    // Additional states omitted for brevity
  }
}
```

## Key Features

- 🚦 **Structured Conversation Flows**: Define states, transitions, and conditions
- 🧠 **LLM-Powered Entity Extraction**: Let the LLM do what it does best
- 📝 **Persistent Context**: Maintain user information throughout the conversation
- 🔄 **Provider-Agnostic**: Works with OpenAI, Anthropic, and other LLM providers
- 📊 **Visualization**: See your FSM structure with a built-in ASCII visualizer
- 📝 **Validation & Error Handling**: Catch and handle issues gracefully
- 🧪 **Test-Friendly**: Easy to unit test and verify behavior

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-fsm.git
cd llm-fsm

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Quick Start

```python
# Import the necessary components
from llm_fsm.llm import LiteLLMInterface
from llm_fsm.fsm_manager import FSMManager
from llm_fsm.utilities import load_fsm_definition

# Step 1: Initialize the LLM interface
# This handles communication with the language model
llm_interface = LiteLLMInterface(
    model="gpt-4o",  # Use any model supported by LiteLLM
    api_key="your-api-key",
    temperature=0.5  # Adjust for more/less creativity
)

# Step 2: Create an FSM manager
# This manages the state transitions and context
fsm_manager = FSMManager(
    fsm_loader=load_fsm_definition,
    llm_interface=llm_interface
)

# Step 3: Start a conversation with your FSM
# This creates a new instance and processes initial input
conversation_id, response = fsm_manager.start_conversation("examples/personal_information_collection.json")
print(f"System: {response}")

# Step 4: Process user input
# This handles state transitions and information extraction
user_input = "My name is Alex Smith"
response = fsm_manager.process_message(conversation_id, user_input)
print(f"System: {response}")

# Step 5: Continue the conversation until completion
# The FSM defines when the conversation is done
while not fsm_manager.is_conversation_ended(conversation_id):
    user_input = input("You: ")
    response = fsm_manager.process_message(conversation_id, user_input)
    print(f"System: {response}")

# Step 6: Get the collected data when done
user_data = fsm_manager.get_conversation_data(conversation_id)
print(f"Collected data: {user_data}")
```

### Command Line Interface

You can also run conversations directly from the command line:

```bash
python -m llm_fsm.main --fsm examples/personal_information_collection.json
```

## How LLM-FSM Works Under the Hood

### 1. The Core Architecture

The framework consists of several key components:

```python
# The FSM definition that structures your conversation
class FSMDefinition(BaseModel):
    """Complete definition of a Finite State Machine."""
    name: str                      # Name of the FSM
    description: str               # Human-readable description
    states: Dict[str, State]       # All states in the FSM
    initial_state: str             # The starting state
    version: str = "3.0"           # Version of the FSM definition
```

```python
# A state within the FSM that represents a conversation step
class State(BaseModel):
    """Defines a state in the FSM."""
    id: str                        # Unique identifier for the state
    description: str               # Human-readable description
    purpose: str                   # The purpose/goal of this state
    transitions: List[Transition]  # Available transitions from this state
    required_context_keys: Optional[List[str]]  # Information to collect
    instructions: Optional[str]    # Instructions for the LLM
    example_dialogue: Optional[List[Dict[str, str]]]  # Example conversations
```

```python
# A transition between states
class Transition(BaseModel):
    """Defines a transition between states."""
    target_state: str              # The state to transition to
    description: str               # When this transition should occur
    conditions: Optional[List[TransitionCondition]]  # Required conditions
    priority: int = 100            # Priority (lower = higher)
```

```python
# The runtime context that maintains information across states
class FSMContext(BaseModel):
    """Runtime context for an FSM instance."""
    data: Dict[str, Any]           # Context data (name, email, etc.)
    conversation: Conversation     # Conversation history
    metadata: Dict[str, Any]       # Additional metadata
```

### 2. The Execution Flow

When a user sends a message:

1. **Prompt Construction**: We build a system prompt containing:
   - Current state information
   - Available transitions
   - Current context values
   - Recent conversation history
   - Response format instructions

2. **LLM Reasoning**: The LLM:
   - Extracts relevant information from the user's message
   - Determines which state to transition to
   - Generates a natural language response
   - Returns a structured JSON response

3. **State Management**: The FSM Manager:
   - Updates the context with extracted information
   - Validates the proposed state transition
   - Updates the current state
   - Stores the conversation history

### 3. Example Prompt

Here's an example of the system prompt sent to the LLM:

```
# Personal Information Collection
## Current State: collect_name
Description: Collect user's name
Purpose: Ask for and record the user's full name

## Instructions:
Ask the user for their full name. If they only provide first name, ask for their full name.
Extract and store their full name in the 'name' context variable.

## Information to collect:
name

## EXTRACTION INSTRUCTIONS:
When the user provides their name, you MUST:
1. Extract the name explicitly mentioned (e.g., 'My name is John' → 'John')
2. Extract implicit name mentions (e.g., 'Call me John' → 'John')
3. Store the extracted name in the context_update field as: {"name": "ExtractedName"}
4. Only transition to the next state if you have successfully extracted and stored the name

## Available Transitions:
1. To 'collect_email': Transition to email collection once name is obtained

## IMPORTANT TRANSITION RULES:
1. You MUST ONLY choose from the following valid target states:
   'collect_email'
2. Do NOT invent or create new states that are not in the above list.
3. If you're unsure which state to transition to, stay in the current state.
4. The current state is 'collect_name' - you can choose to stay here if needed.

## Current Context: None (empty)

## Response Format:
Respond with a JSON object with the following structure:
```json
{
  "transition": {
    "target_state": "state_id",
    "context_update": {"key1": "value1", "key2": "value2"}
  },
  "message": "Your message to the user",
  "reasoning": "Your reasoning for this decision"
}
```

Important:
1. Collect all required information from the user's message
2. Only transition to a new state if all required information is collected
3. Your message should be conversational and natural
4. Don't mention states, transitions, or context keys to the user
5. Remember, you can ONLY choose from these valid target states: 'collect_email'
```

## Architecture Design Decisions

### 1. LLM as the NLU Engine

Rather than implementing brittle extraction rules in Python, we leverage the LLM's inherent understanding of language:

```python
# Old approach (v1): Hardcoded extraction in Python
def extract_name(user_message):
    name_patterns = [
        r"My name is ([\w\s]+)", 
        r"I am ([\w\s]+)",
        # Many more patterns needed
    ]
    for pattern in name_patterns:
        match = re.search(pattern, user_message)
        if match:
            return match.group(1)
    return None

# New approach (v3): Let the LLM handle extraction
# The LLM receives instructions like:
"""
When the user provides their name, extract it and store it in the 'name' context variable.
Consider explicit mentions ('My name is John') and implicit mentions ('Call me John').
"""
# The LLM does the extraction and returns:
{
  "transition": {
    "target_state": "collect_email",
    "context_update": {"name": "John Smith"}
  },
  "message": "Nice to meet you, John Smith! Could you please provide your email address?",
  "reasoning": "The user provided their full name 'John Smith', so I've stored it and am transitioning to collect their email."
}
```

### 2. Context Outside the State

We keep context separate from states, allowing information to persist across transitions:

```python
# Bad approach: Context within states
{
  "state": "collect_name",
  "context": {"name": "John"}
}
# When transitioning, we'd need to manually copy context

# Good approach: Context outside states
{
  "current_state": "collect_name",
  "context": {"name": "John", "email": "john@example.com"}
}
# Context naturally persists across state transitions
```

### 3. Structured LLM Responses

We use a structured response format that separates user-facing content from system-facing content:

```json
{
  "transition": {
    "target_state": "collect_email",
    "context_update": {"name": "John Smith"}
  },
  "message": "Nice to meet you, John! Could you please provide your email address?",
  "reasoning": "The user provided their name, so I'm transitioning to collect their email."
}
```

This clear separation ensures:
- Users only see the `message` field
- The system uses the `transition` information for state management
- Developers can see the `reasoning` for debugging

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Areas for contribution:
- Additional FSM examples
- Integration with more LLM providers
- Enhanced visualization tools
- Web-based UI for FSM creation and testing
- Improved documentation and tutorials

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.