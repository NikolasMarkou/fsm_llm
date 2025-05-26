# LLM-FSM: Adding State to the Stateless

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/llm-fsm.svg)](https://badge.fury.io/py/llm-fsm)

![logo](./images/fsm-llm-logo-1.png)

## The Problem

Large Language Models are **stateless**. Each interaction is processed independently, making it challenging to build structured, multi-turn conversations that maintain context and follow predictable flows.

## The Solution

LLM-FSM combines **Finite State Machines** with **Large Language Models**:
- The FSM provides structure and state management
- The LLM handles natural language understanding and generation
- Python orchestrates the interaction between them

## Installation

```bash
pip install llm-fsm
```

For development:
```bash
pip install llm-fsm[all]
```

## Getting Started: From Simple to Complex

### Level 1: Inline FSM (Simplest)

Start with a minimal example using an FSM defined directly in Python:

```python
from llm_fsm import API

# Define FSM structure directly in code
simple_fsm = {
    "name": "greeting_bot",
    "initial_state": "start",
    "states": {
        "start": {
            "id": "start",
            "description": "Initial greeting state",
            "purpose": "Greet the user and ask their name",
            "transitions": [
                {
                    "target_state": "goodbye", 
                    "description": "User provided their name"
                }
            ]
        },
        "goodbye": {
            "id": "goodbye", 
            "description": "Final goodbye state",
            "purpose": "Say goodbye using the user's name",
            "transitions": []
        }
    }
}

# Create and use the FSM
api = API.from_definition(simple_fsm, model="gpt-4o-mini")

# Start conversation
conv_id, response = api.start_conversation()
print(f"Bot: {response}")

# Chat
user_input = input("You: ")
response = api.converse(user_input, conv_id)
print(f"Bot: {response}")

# Get collected data
data = api.get_data(conv_id)
print(f"Collected: {data}")
```

### Level 2: JSON FSM Files (Medium Complexity)

Move to external JSON files for more complex, reusable FSMs:

**Create `greeting.json`:**
```json
{
  "name": "greeting_system",
  "description": "A friendly greeting system that collects user information",
  "initial_state": "welcome",
  "states": {
    "welcome": {
      "id": "welcome",
      "description": "Welcome the user",
      "purpose": "Greet user and ask for their name",
      "required_context_keys": ["name"],
      "transitions": [
        {
          "target_state": "ask_age",
          "description": "User provided their name"
        }
      ]
    },
    "ask_age": {
      "id": "ask_age", 
      "description": "Ask for user's age",
      "purpose": "Ask for the user's age",
      "required_context_keys": ["age"],
      "transitions": [
        {
          "target_state": "farewell",
          "description": "User provided their age"
        }
      ]
    },
    "farewell": {
      "id": "farewell",
      "description": "Say goodbye with personalized message",
      "purpose": "Thank user and say goodbye using their information",
      "transitions": []
    }
  }
}
```

**Use the JSON FSM:**
```python
from llm_fsm import API

# Load FSM from file
api = API.from_file("greeting.json", model="gpt-4o-mini")

# Run full conversation
conv_id, response = api.start_conversation()
print(f"Bot: {response}")

while not api.has_conversation_ended(conv_id):
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = api.converse(user_input, conv_id)
    print(f"Bot: {response}")

# See what was collected
data = api.get_data(conv_id)
print(f"Final data: {data}")
api.end_conversation(conv_id)
```

### Level 3: Advanced Features (High Complexity)

Add custom handlers, FSM stacking, and complex workflows:

```python
from llm_fsm import API
from llm_fsm.handlers import HandlerTiming
import time
import json

# Create API with handlers
api = API.from_file("greeting.json", model="gpt-4o-mini")

# Add custom handlers for advanced functionality
def log_interactions(context):
    """Log all user interactions"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return {"last_interaction_time": timestamp}

def validate_age(context):
    """Validate age input"""
    age = context.get("age")
    if age and str(age).isdigit():
        age_num = int(age)
        if age_num < 13:
            return {"age_category": "child", "special_handling": True}
        elif age_num < 65:
            return {"age_category": "adult"}
        else:
            return {"age_category": "senior", "special_handling": True}
    return {}

# Register handlers using fluent interface
interaction_logger = api.create_handler("InteractionLogger") \
    .at(HandlerTiming.POST_PROCESSING) \
    .do(log_interactions)

age_validator = api.create_handler("AgeValidator") \
    .at(HandlerTiming.CONTEXT_UPDATE) \
    .when_keys_updated("age") \
    .do(validate_age)

api.register_handlers([interaction_logger, age_validator])

# Add convenience handlers
api.add_logging_handler()  # Automatic debug logging
api.add_context_validator_handler(["name"])  # Ensure name is collected

# Start enhanced conversation
conv_id, response = api.start_conversation()
print(f"Bot: {response}")

# Example of FSM stacking - push a detailed form mid-conversation
conversation_active = True
while conversation_active and not api.has_conversation_ended(conv_id):
    user_input = input("You: ")
    
    if user_input.lower() == "detailed form":
        # Push a sub-FSM for detailed information gathering
        print("Switching to detailed form...")
        response = api.push_fsm(
            conv_id,
            "examples/basic/form_filling/fsm.json",  # More detailed form FSM
            context_to_pass={"initiated_by": "user_request"},
            shared_context_keys=["name", "age"],
            preserve_history=True
        )
        print(f"Bot: {response}")
        continue
    
    if user_input.lower() == "quit":
        conversation_active = False
        continue
    
    response = api.converse(user_input, conv_id)
    print(f"Bot: {response}")
    
    # Check if we're in a stacked FSM and it completed
    if api.get_stack_depth(conv_id) > 1:
        current_state = api.get_current_state(conv_id)
        if current_state == "complete":  # Assuming sub-FSM has a 'complete' state
            print("Detailed form completed, returning to main conversation...")
            response = api.pop_fsm(
                conv_id, 
                context_to_return={"detailed_form_completed": True},
                merge_strategy="update"
            )
            print(f"Bot: {response}")

# Advanced introspection
print(f"\nConversation Summary:")
print(f"Stack depth: {api.get_stack_depth(conv_id)}")
print(f"Registered handlers: {api.get_registered_handlers()}")
print(f"Context flow: {json.dumps(api.get_context_flow(conv_id), indent=2)}")
print(f"Final data: {json.dumps(api.get_data(conv_id), indent=2)}")

# Save conversation for later analysis
api.save_conversation(conv_id, "conversation_log.json")
api.end_conversation(conv_id)
```

## Key Features

Based on the examples above, LLM-FSM provides:

- **🚀 Progressive Complexity**: Start simple, add features as needed
- **🔗 FSM Stacking**: Stack multiple FSMs for complex workflows
- **🎣 Custom Handlers**: Add your own logic at any point in the conversation
- **📝 Context Management**: Automatic context collection and handover
- **🔄 Multiple LLM Providers**: OpenAI, Anthropic, and others via LiteLLM
- **📊 Rich Introspection**: Debug and analyze conversation flows
- **💾 Conversation Persistence**: Save and resume conversations

## Command Line Tools

```bash
# Run any FSM file
llm-fsm --fsm your_fsm.json

# Visualize FSM structure
llm-fsm-visualize --fsm your_fsm.json

# Validate FSM definitions
llm-fsm-validate --fsm your_fsm.json
```

## Project Structure

```
llm-fsm/
├── examples/
│   ├── basic/              # Simple FSMs to start with
│   ├── intermediate/       # More complex examples
│   ├── advanced/          # Full-featured applications
│   ├── tutorials/         # Step-by-step learning
│   └── use_cases/         # Real-world applications
├── src/llm_fsm/          # Core library
└── docs/                 # Documentation
```

## Examples by Complexity

**Basic (`examples/basic/`):**
- `simple_greeting/` - Minimal hello/goodbye FSM
- `form_filling/` - Basic information collection
- `story_time/` - Interactive storytelling

**Intermediate (`examples/intermediate/`):**
- `book_recommendation/` - Recommendation system with preferences
- `product_recommendation/` - Decision-tree product suggestions

**Advanced (`examples/advanced/`):**
- `yoga_instructions/` - Adaptive instructions with engagement tracking
- `e_commerce/` - Full shopping workflow with multiple FSMs

**Tutorials (`examples/tutorials/`):**
- `01_hello_world/` - Your first FSM
- `02_adding_context/` - Working with data
- `03_conditional_transitions/` - Logic-based transitions
- `04_custom_handlers/` - Adding custom functionality
- `05_complex_workflows/` - Multi-FSM applications

## Advanced Concepts

### FSM Stacking
Stack FSMs to create complex, hierarchical conversations:
```python
# Push specialized FSM for detailed tasks
api.push_fsm(conv_id, "detailed_task.json", 
             shared_context_keys=["user_id"],
             preserve_history=True)

# Pop back to main FSM when done
api.pop_fsm(conv_id, context_to_return={"task_completed": True})
```

### Custom Handlers
Add logic at specific points in the conversation:
```python
# Execute custom code on state transitions
handler = api.create_handler("CustomLogic") \
    .on_state_entry("checkout") \
    .do(lambda ctx: send_notification(ctx["user_email"]))
api.register_handler(handler)
```

### Context Strategies
Control how data flows between FSMs:
```python
api.pop_fsm(conv_id, 
           context_to_return={"form_data": collected_data},
           merge_strategy="update")  # or "preserve" or "selective"
```

## Documentation

- **[LLM Guide](./LLM.md)**: How LLMs interact with the framework
- **[Handler Guide](./docs/fsm_handler_integration_guide.md)**: Advanced handler usage

## Development

```bash
# Clone and setup
git clone https://github.com/nikolasmarkou/llm-fsm.git
cd llm-fsm
pip install -e .[dev,workflows]

# Run tests
make test

# Build
make build
```

## License

GPL v3.0 - see [LICENSE](./LICENSE) for details.