# LLM-FSM: Finite State Machines for Large Language Models

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## What is LLM-FSM?

LLM-FSM is a Python framework that enables building robust conversational systems by combining Finite State Machines with Large Language Models. This approach creates structured, predictable conversations while leveraging the powerful language understanding capabilities of modern LLMs.

## The Problem: LLMs Are Stateless

Large Language Models excel at natural language understanding and generation but lack the inherent ability to maintain state across interactions. This limitation makes implementing structured, multi-turn conversations challenging and prone to unpredictability.

**LLM-FSM** solves this problem by combining the best of both worlds:
- The structured, predictable flow of Finite State Machines
- The powerful language understanding capabilities of modern LLMs

> We keep the state as a JSON structure inside the system prompt of an LLM, describing transition nodes and conditions, while using the LLM's reasoning capabilities to determine when to transition and what information to extract.

## Examples of FSM Conversations

### Example 1: Personal Information Collection

This FSM gathers user information step-by-step, with correction handling:

```json
{
  "name": "Personal Information Collection",
  "description": "A conversation flow to collect user's personal information with confirmation",
  "initial_state": "welcome",
  "states": {
    "welcome": {
      "id": "welcome",
      "description": "Initial welcome state",
      "purpose": "Welcome the user and explain the purpose of the conversation",
      "transitions": [
        {
          "target_state": "collect_name",
          "description": "Always transition to collecting name after welcome",
          "priority": 0
        }
      ]
    },
    "collect_name": {
      "id": "collect_name",
      "description": "Collect user's name",
      "purpose": "Ask for and record the user's full name",
      "required_context_keys": ["name"],
      "transitions": [
        {
          "target_state": "collect_email",
          "description": "Transition to email collection once name is obtained",
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

**Sample conversation flow:**
1. System greets user and explains purpose
2. System asks for user's name
3. User provides name
4. System asks for email
5. ... and so on, with validation and correction handling

### Example 2: Product Recommendation System

This FSM helps users find technology products based on their preferences:

```json
{
  "name": "Product Recommendation System",
  "description": "A tree-structured conversation to recommend technology products",
  "initial_state": "welcome",
  "states": {
    "welcome": {
      "id": "welcome",
      "description": "Initial welcome state",
      "purpose": "Welcome the user and introduce the product recommendation service",
      "transitions": [
        {
          "target_state": "device_type_selection",
          "description": "Transition to device type selection after welcome",
          "priority": 0
        }
      ]
    },
    "device_type_selection": {
      "id": "device_type_selection",
      "description": "Determine user's preferred device type",
      "purpose": "Ask whether the user is interested in a smartphone or a laptop",
      "required_context_keys": ["device_type"],
      "transitions": [
        {
          "target_state": "smartphone_budget",
          "description": "User wants a smartphone",
          "conditions": [
            {
              "description": "User has indicated they want a smartphone",
              "requires_context_keys": ["device_type"]
            }
          ],
          "priority": 0
        },
        {
          "target_state": "laptop_budget",
          "description": "User wants a laptop",
          "conditions": [
            {
              "description": "User has indicated they want a laptop",
              "requires_context_keys": ["device_type"]
            }
          ],
          "priority": 1
        }
      ]
    }
    // Additional states omitted for brevity
  }
}
```

**Sample conversation flow:**
1. System welcomes user
2. System asks whether user wants smartphone or laptop
3. Based on choice, system asks about budget
4. System provides recommendations based on device type and budget

## Key Features

- 🔄 **State-based conversation management**: Define conversational flows as states with clear transitions
- 🧠 **LLM-powered entity extraction**: Leverage the LLM's understanding rather than rigid rules
- 📝 **Persistent context**: Maintain information across the entire conversation
- 🔄 **Provider-agnostic**: Works with OpenAI, Anthropic, and other LLM providers via LiteLLM
- 📊 **Visualization tools**: ASCII-based visualization of your FSM structure
- 📝 **Detailed logging**: Comprehensive logging for debugging and monitoring
- 🧪 **Easy to extend**: Modular design makes it easy to adapt to your needs

## Installation

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

## Quick Start

### 1. Define your FSM in a JSON file

```json
{
  "name": "Simple Greeting",
  "description": "A simple FSM that greets the user and collects their name",
  "initial_state": "start",
  "states": {
    "start": {
      "id": "start",
      "description": "Initial state",
      "purpose": "Begin the conversation",
      "transitions": [
        {
          "target_state": "greeting",
          "description": "Always transition to greeting",
          "priority": 0
        }
      ]
    },
    "greeting": {
      "id": "greeting",
      "description": "Greet the user",
      "purpose": "Welcome the user and ask for their name",
      "transitions": [
        {
          "target_state": "collect_name",
          "description": "After greeting, collect the user's name",
          "priority": 0
        }
      ]
    },
    "collect_name": {
      "id": "collect_name",
      "description": "Collect the user's name",
      "purpose": "Get the user's name",
      "required_context_keys": ["name"],
      "transitions": [
        {
          "target_state": "end",
          "description": "When name is collected, end the conversation",
          "conditions": [
            {
              "description": "Name has been provided",
              "requires_context_keys": ["name"]
            }
          ]
        }
      ]
    },
    "end": {
      "id": "end",
      "description": "End of conversation",
      "purpose": "End the conversation",
      "transitions": []
    }
  }
}
```

### 2. Run a conversation in Python

```python
from src.llm import LiteLLMInterface
from src.fsm_manager import FSMManager
from src.utilities import load_fsm_definition

# Initialize the LLM interface
llm_interface = LiteLLMInterface(
    model="gpt-4o",  # Or any other model supported by LiteLLM
    api_key="your-api-key",
    temperature=0.5
)

# Create an FSM manager
fsm_manager = FSMManager(
    fsm_loader=load_fsm_definition,
    llm_interface=llm_interface
)

# Create an FSM instance from your JSON file
instance = fsm_manager.create_instance("path/to/your/fsm.json")

# Start the conversation
instance, response = fsm_manager.process_user_input(instance, "")
print(f"System: {response}")

# Process user input
user_input = "Hello there!"
instance, response = fsm_manager.process_user_input(instance, user_input)
print(f"System: {response}")
```

### 3. Or use the CLI

```bash
python -m src.main --fsm examples/personal_information_collection.json
```

## Architecture

LLM-FSM consists of several key components:

### 1. FSM Definition

The FSM is defined using a structured set of Python classes:

```python
class FSMDefinition(BaseModel):
    """Complete definition of a Finite State Machine."""
    name: str
    description: str
    states: Dict[str, State]
    initial_state: str
    version: str = "3.0"
```

### 2. States and Transitions

```python
class State(BaseModel):
    """Defines a state in the FSM."""
    id: str
    description: str
    purpose: str
    transitions: List[Transition]
    required_context_keys: Optional[List[str]]
    instructions: Optional[str]
    example_dialogue: Optional[List[Dict[str, str]]]
```

```python
class Transition(BaseModel):
    """Defines a transition between states."""
    target_state: str
    description: str
    conditions: Optional[List[TransitionCondition]]
    priority: int = 100
```

### 3. Context Management

```python
class FSMContext(BaseModel):
    """Runtime context for an FSM instance."""
    data: Dict[str, Any]
    conversation: Conversation
    metadata: Dict[str, Any]
```

### 4. LLM Integration

```python
class LLMInterface:
    """Interface for communicating with LLMs."""
    def send_request(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("Subclasses must implement send_request")
```

## Design Decisions

### LLM as the NLU Engine

Rather than implementing brittle extraction rules in Python, we leverage the LLM's inherent understanding of language. This makes the system more robust against variations in user input and reduces the amount of code needed.

### Context Outside the State

By keeping context separate from states, we allow information to persist across transitions and create a cleaner separation of concerns. States define behavior, while context stores data.

### Structured LLM Responses

The LLM responses follow a structured format that separates user-facing content from system-facing content:

```json
{
  "transition": {
    "target_state": "next_state_id",
    "context_update": {"extracted_key": "extracted_value"}
  },
  "message": "The message to display to the user",
  "reasoning": "Internal reasoning about why this transition was chosen"
}
```

## Advanced Features

### Visualization

Visualize your FSM structure using the built-in ASCII visualizer:

```bash
python -m src.visualizer --fsm examples/conversational_loop.json
```

### Dynamic Entity Extraction

The LLM dynamically extracts entities from user messages, storing them in the context. For example, when collecting a user's name:

```json
{
  "id": "collect_name",
  "description": "Collect the user's name",
  "purpose": "Get the user's name",
  "required_context_keys": ["name"],
  "instructions": "Ask the user for their full name. Extract and store their name in the 'name' context variable."
}
```

## Creating Your Own FSM

### Step 1: Define Your Conversation Flow

Before coding, sketch out your conversation flow:
- What states will your conversation have?
- What information do you need to collect?
- What transitions are possible between states?
- What are the conditions for each transition?

### Step 2: Create Your FSM Definition JSON

Create a JSON file defining your FSM structure (see examples above).

### Step 3: Test and Refine

Run your FSM using the CLI tool and iterate based on testing:
- Are transitions working as expected?
- Is the LLM extracting information correctly?
- Is the conversation flow natural and intuitive?

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Areas for contribution:
- Additional examples
- Integration with more LLM providers
- Enhanced visualization tools
- Web-based UI for FSM creation and testing
- Improved documentation and tutorials

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.