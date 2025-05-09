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

### Environment Variables

The following environment variables can be configured in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `LLM_MODEL`: The model to use (e.g., "gpt-4o", "claude-3-opus")
- `LLM_TEMPERATURE`: Temperature parameter for generation (default: 0.5)
- `LLM_MAX_TOKENS`: Maximum tokens for generation (default: 1000)

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

The repository comes with several example FSM definitions:

### 1. Simple Greeting

A basic example of collecting a user's name and providing a personalized greeting.

- States: start → greeting → collect_name → personalized_greeting → conversation → end
- Features: Basic user interaction, name extraction

### 2. Book Recommendation System (conversational_loop.json)

A conversational system that recommends books and checks user engagement.

- States: welcome → genre_selection → recommend_book → check_engagement → (loop back or end)
- Features: User preference collection, engagement detection, looping conversation

### 3. Personal Information Collection (personal_information_collection.json)

A flow to collect and validate personal information with correction handling.

- States: welcome → collect_name → collect_email → collect_birthdate → collect_occupation → summary → (correction loop or end)
- Features: Incremental data collection, validation, correction handling

### 4. Product Recommendation (tree_conversation_with_4_endings.json)

A tree-structured conversation for recommending technology products based on user preferences.

- States: welcome → device_type_selection → (smartphone or laptop paths) → budget determination → recommendations → end
- Features: Branching paths, decision tree navigation, contextual recommendations

## Architecture

The LLM-FSM framework consists of several key components:

### 1. FSM Definition

```
FSMDefinition
 ├── name: str                 # Name of the FSM
 ├── description: str          # Human-readable description
 ├── states: Dict[str, State]  # Dictionary of all states
 ├── initial_state: str        # The starting state identifier
 └── version: str              # Version of the FSM definition (default: "3.0")
```

### 2. States and Transitions

```
State
 ├── id: str                                      # Unique identifier for the state
 ├── description: str                             # Human-readable description
 ├── purpose: str                                 # The purpose of this state
 ├── transitions: List[Transition]                # Available transitions from this state
 ├── required_context_keys: Optional[List[str]]   # Context keys that should be collected
 ├── instructions: Optional[str]                  # Instructions for the LLM
 └── example_dialogue: Optional[List[Dict[str, str]]]  # Example dialogue for the state

Transition
 ├── target_state: str                                # The state to transition to
 ├── description: str                                 # When this transition should occur
 ├── conditions: Optional[List[TransitionCondition]]  # Conditions for transition
 └── priority: int                                    # Priority (lower = higher priority)

TransitionCondition
 ├── description: str                               # Description of the condition
 └── requires_context_keys: Optional[List[str]]     # Context keys that must be present
```

### 3. Context Management

```
FSMContext
 ├── data: Dict[str, Any]                # Context data collected during conversation
 ├── conversation: Conversation          # Conversation history
 └── metadata: Dict[str, Any]            # Additional metadata

Conversation
 └── exchanges: List[Dict[str, str]]     # List of conversation exchanges
```

### 4. LLM Integration

```
LLMInterface
 └── send_request(request: LLMRequest) -> LLMResponse   # Abstract interface method

LiteLLMInterface  # Implementation using LiteLLM
 ├── model: str                          # LLM model to use (e.g., "gpt-4o")
 ├── api_key: Optional[str]              # API key for the provider
 ├── enable_json_validation: bool        # Enable JSON schema validation
 └── kwargs: Dict[str, Any]              # Additional arguments to pass to LiteLLM
```

## Design Decisions

### LLM as the NLU Engine

Rather than implementing brittle extraction rules in Python, we leverage the LLM's inherent understanding of language. This makes the system more robust against variations in user input and reduces the amount of code needed.

### Context Outside the State

By keeping context separate from states, we allow information to persist across transitions and create a cleaner separation of concerns. States define behavior, while context stores data.

### Structured LLM Responses

The LLM responses follow a structured format that separates user-facing content from system-facing content, creating a clean boundary between UI and logic. The response format looks like this:

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

### Priority-Based Transitions

Each transition can have a priority value (lower number = higher priority). This allows for defining fallback paths and handling special cases, such as safety concerns or error states.

### Validation and Error Handling

The framework includes comprehensive validation to ensure:
- All transitions target valid states
- Required context keys are present before transitions
- No orphaned states exist in the FSM
- LLM responses conform to the expected format

## Advanced Features

### Visualization

Visualize your FSM structure using the built-in ASCII visualizer:

```bash
python -m src.visualizer --fsm examples/conversational_loop.json
```

The visualizer generates an ASCII representation of your FSM, showing:
- A list of all states with their descriptions
- All transitions between states
- Required context keys for transitions
- A simple diagram showing the flow of the conversation
- Information about loops and special transitions

You can save the visualization to a file using the `--output` flag:

```bash
python -m src.visualizer --fsm examples/conversational_loop.json --output fsm_diagram.txt
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

The LLM will extract the name regardless of how the user provides it:
- "My name is John" → `{"name": "John"}`
- "Call me Sarah" → `{"name": "Sarah"}`
- "I'm Robert Smith" → `{"name": "Robert Smith"}`

### Detailed Logging

The framework includes comprehensive logging for debugging and monitoring:
- Console logging with color-coded levels
- File-based logging with rotation and compression
- Detailed logs of state transitions, context updates, and LLM requests/responses

## Creating Your Own FSM

### Step 1: Define Your Conversation Flow

Before coding, sketch out your conversation flow:
- What states will your conversation have?
- What information do you need to collect?
- What transitions are possible between states?
- What are the conditions for each transition?

### Step 2: Create Your FSM Definition JSON

Create a JSON file defining your FSM structure using the following template:

```json
{
  "name": "Your FSM Name",
  "description": "Description of your FSM",
  "initial_state": "start_state_id",
  "version": "3.0",
  "states": {
    "start_state_id": {
      "id": "start_state_id",
      "description": "Description of this state",
      "purpose": "Purpose of this state",
      "transitions": [
        {
          "target_state": "next_state_id",
          "description": "When to transition",
          "priority": 0
        }
      ],
      "instructions": "Instructions for the LLM in this state"
    },
    "next_state_id": {
      "id": "next_state_id",
      "description": "Description of next state",
      "purpose": "Purpose of next state",
      "required_context_keys": ["key_to_collect"],
      "transitions": [
        {
          "target_state": "end",
          "description": "When to transition to end",
          "conditions": [
            {
              "description": "Condition description",
              "requires_context_keys": ["key_to_collect"]
            }
          ],
          "priority": 0
        }
      ],
      "instructions": "Ask the user for specific information. Store it in the 'key_to_collect' context variable."
    },
    "end": {
      "id": "end",
      "description": "End of conversation",
      "purpose": "End the conversation",
      "transitions": [],
      "instructions": "Thank the user and conclude the conversation."
    }
  }
}
```

### Step 3: Test and Refine

Run your FSM using the CLI tool:

```bash
python -m src.main --fsm path/to/your/fsm.json
```

Iterate and refine your FSM based on testing:
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