# LLM-FSM: Finite State Machines for Large Language Models

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## The Problem: LLMs Are Stateless

Large Language Models excel at natural language understanding and generation but lack the inherent ability to maintain state across interactions. This limitation makes implementing structured, multi-turn conversations challenging and prone to unpredictability.

**LLM-FSM** solves this problem by combining the best of both worlds:
- The structured, predictable flow of Finite State Machines
- The powerful language understanding capabilities of modern LLMs

## What is LLM-FSM?

LLM-FSM is a Python framework that enables you to build robust conversational systems using a Finite State Machine approach powered by Large Language Models. The core idea is elegantly simple:

> We keep the state as a JSON structure inside the system prompt of an LLM, describing transition nodes and conditions, while using the LLM's reasoning capabilities to determine when to transition and what information to extract.

This approach brings several key advantages:
- **Structured conversations** with predictable flows
- **Clear separation of concerns** between state management and language understanding
- **Incremental information collection** over multiple turns
- **Robust error handling** and graceful fallbacks

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
git clone https://github.com/yourusername/fsm_llm.git
cd fsm_llm

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

1. Define your FSM in a JSON file:

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
          ],
          "priority": 0
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

2. Run a conversation:

```python
from src.fsm import LiteLLMInterface, FSMManager
from src.loader import load_fsm_from_file

# Initialize the LLM interface
llm_interface = LiteLLMInterface(
    model="gpt-4o",  # Or any other model supported by LiteLLM
    api_key="your-api-key",
    temperature=0.5
)

# Create an FSM manager
fsm_manager = FSMManager(
    fsm_loader=load_fsm_from_file,
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

3. Or use the CLI:

```bash
python -m src.main --fsm examples/personal_information_collection.json
```

## Included Examples

- **Simple Greeting**: A basic example of collecting a user's name
- **Book Recommendation System**: A conversational system that recommends books and checks user engagement
- **Personal Information Collection**: A flow to collect and validate personal information
- **Product Recommendation**: A tree-structured conversation for recommending technology products

## Architecture

The LLM-FSM framework consists of several key components:

### 1. FSM Definition

```
FSMDefinition
 ├── name: str
 ├── description: str
 ├── states: Dict[str, State]
 ├── initial_state: str
 └── version: str
```

### 2. States and Transitions

```
State
 ├── id: str
 ├── description: str
 ├── purpose: str
 ├── transitions: List[Transition]
 ├── required_context_keys: Optional[List[str]]
 ├── instructions: Optional[str]
 └── example_dialogue: Optional[List[Dict[str, str]]]

Transition
 ├── target_state: str
 ├── description: str
 ├── conditions: Optional[List[TransitionCondition]]
 └── priority: int
```

### 3. Context Management

```
FSMContext
 ├── data: Dict[str, Any]
 ├── conversation: Conversation
 └── metadata: Dict[str, Any]
```

### 4. LLM Integration

```
LLMInterface
 └── send_request(request: LLMRequest) -> LLMResponse

LiteLLMInterface  # Implementation using LiteLLM
```

## Design Decisions

### LLM as the NLU Engine

Rather than implementing brittle extraction rules in Python, we leverage the LLM's inherent understanding of language. This makes the system more robust against variations in user input and reduces the amount of code needed.

### Context Outside the State

By keeping context separate from states, we allow information to persist across transitions and create a cleaner separation of concerns. States define behavior, while context stores data.

### Structured LLM Responses

The LLM responses follow a structured format that separates user-facing content from system-facing content, creating a clean boundary between UI and logic.

## Advanced Features

### Visualization

Visualize your FSM structure using the built-in ASCII visualizer:

```bash
python -m src.visualizer --fsm examples/conversational_loop.json
```

### Robust Error Handling

The framework includes comprehensive validation and error handling to ensure that:
- All transitions are valid
- Required context keys are present before transitions
- LLM responses conform to the expected format

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.