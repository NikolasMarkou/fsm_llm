# LLM-FSM: Adding State to the Stateless

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/llm-fsm.svg)](https://badge.fury.io/py/llm-fsm)

![logo](./images/fsm-llm-logo-1.png)

## Table of Contents
- [The Problem: Stateless LLMs in Structured Conversations](#the-problem-stateless-llms-in-structured-conversations)
- [The Solution: Finite State Machines + LLMs](#the-solution-finite-state-machines--llms)
  - [Key Features](#key-features)
- [Installation](#installation)
  - [Core Library](#core-library)
  - [With Workflows Extension](#with-workflows-extension)
  - [For Development](#for-development)
- [Quick Start](#quick-start)
  - [Environment Setup](#environment-setup)
  - [Simple Conversation](#simple-conversation)
  - [FSM Stacking Example](#fsm-stacking-example)
  - [Custom Handlers Example](#custom-handlers-example)
  - [Command Line Interface](#command-line-interface)
- [Core Components](#core-components)
  - [LLM-FSM Core (`llm_fsm`)](#llm-fsm-core-llm_fsm)
  - [FSM Stacking & Context Handover](#fsm-stacking--context-handover)
  - [Handler System](#handler-system)
  - [Workflows Extension (`llm_fsm_workflows`)](#workflows-extension-llm_fsm_workflows)
  - [JsonLogic Expressions](#jsonlogic-expressions)
- [Examples & Tutorials](#examples--tutorials)
  - [Basic Examples](#basic-examples)
  - [Intermediate Examples](#intermediate-examples)
  - [Advanced Examples](#advanced-examples)
  - [Tutorials](#tutorials)
  - [Use Cases](#use-cases)
- [Documentation](#documentation)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## The Problem: Stateless LLMs in Structured Conversations

Large Language Models have revolutionized natural language processing with their remarkable generation capabilities.
However, they have a fundamental limitation: **they are inherently stateless**.
Each interaction is processed independently with only the context provided in the prompt.

This statelessness creates significant challenges for building robust conversational applications:

- **State Fragility**: Without explicit tracking, conversations easily lose their place
- **Context Limitations**: As conversations grow, context windows fill up quickly
- **Transition Ambiguity**: Determining when to move to different conversation stages is difficult
- **Information Extraction Inconsistency**: Extracting structured data from free-form text is unreliable
- **Validation Challenges**: Ensuring required information is collected before proceeding is complex

Consider a flight booking scenario:

```
User: I'd like to book a flight
System: Where would you like to fly to?
User: I'm thinking maybe Hawaii
System: Great choice! And where will you be departing from?
User: Actually, I'd prefer Bali instead of Hawaii
```

Without explicit state tracking, the system might miss the change in destination or maintain inconsistent information.

## The Solution: Finite State Machines + LLMs

LLM-FSM elegantly combines classical Finite State Machines with modern Large Language Models:

> "We keep the state as a JSON structure inside the system prompt of an LLM, describing transition nodes and conditions for that specific state, along with any emittance of symbols that the LLM might do."

The state and transitions are handled by Python and language ambiguities are handled by the LLM.

This hybrid approach gives you the best of both worlds:
- ✅ **Predictable conversation flows** with clear rules and transitions
- ✅ **Natural language understanding** powered by state-of-the-art LLMs
- ✅ **Persistent context** across the entire conversation
- ✅ **Dynamic adaptation** to user inputs
- ✅ **FSM Stacking** for complex, multi-level conversational workflows
- ✅ **Comprehensive Context Handover** between stacked FSMs
- ✅ **Expressive Logic** for complex transitional decision-making using JsonLogic
- ✅ **Extensible Workflows** for building complex, multi-step automated processes
- ✅ **Customizable Handlers** for integrating external logic and side effects

### Key Features

- 🚦 **Structured Conversation Flows**: Define states, transitions, and conditions in JSON.
- 🧠 **LLM-Powered NLU**: Leverage LLMs for understanding, entity extraction, and response generation.
- 🔗 **FSM Stacking**: Stack multiple FSMs for complex, hierarchical conversations with seamless context handover.
- 🎣 **Advanced Handler System**: Integrate custom Python functions at various lifecycle points of FSM execution.
- 👤 **Persona Support**: Define a consistent tone and style for LLM responses.
- 📝 **Persistent Context Management**: Maintain information throughout conversations with flexible merge strategies.
- 🔄 **Provider-Agnostic**: Works with OpenAI, Anthropic, and other LLM providers via LiteLLM.
- 📊 **Visualization & Validation**: Built-in CLI tools to visualize FSMs as ASCII art and validate definitions.
- 🪵 **Comprehensive Logging**: Detailed logs via Loguru for debugging and monitoring.
- 🧪 **Test-Friendly**: Designed for easy unit testing and behavior verification.
- 🧮 **JsonLogic Expressions**: Powerful conditional logic for logic-based FSM transitions.
- 🧩 **Workflow Engine**: Build complex, automated processes on top of FSMs with the `llm_fsm_workflows` extension.

## Installation

### Core Library

To install the core LLM-FSM library:
```bash
pip install llm-fsm
```

### With Workflows Extension

To include the workflows extension (for building automated multi-step processes):
```bash
pip install llm-fsm[workflows]
```

To install all optional features:
```bash
pip install llm-fsm[all]
```

### For Development

1.  Clone the repository:
    ```bash
    git clone https://github.com/nikolasmarkou/llm-fsm.git
    cd llm-fsm
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    pip install -e .[dev,workflows]  # Install in editable mode with dev and workflows extras
    ```
3.  Set up environment variables:
    ```bash
    cp .env.example .env
    # Edit .env with your API keys (e.g., OPENAI_API_KEY) and default model
    ```

## Quick Start

### Environment Setup

Ensure you have your LLM API key set as an environment variable. For OpenAI, it's `OPENAI_API_KEY`.
You can also specify the default `LLM_MODEL` in your `.env` file (e.g., `LLM_MODEL=gpt-4o-mini`).

### Simple Conversation

The `API` class provides a high-level interface for basic conversations:

```python
from llm_fsm import API

# Create the LLM-FSM instance from an FSM definition file
api = API.from_file(
    path="examples/basic/simple_greeting/fsm.json",
    model="gpt-4o-mini"  # Or your preferred model from .env
)

# Start a conversation
conversation_id, response = api.start_conversation()
print(f"System: {response}")

# Continue conversation
while not api.has_conversation_ended(conversation_id):
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = api.converse(user_input, conversation_id)
    print(f"System: {response}")

# Get collected data
data = api.get_data(conversation_id)
print(f"Collected data: {data}")

# Clean up
api.end_conversation(conversation_id)
```

### FSM Stacking Example

Stack multiple FSMs for complex workflows with automatic context handover:

```python
from llm_fsm import API

# Create main conversation FSM
api = API.from_file("examples/intermediate/product_recommendation/fsm.json")
conv_id, response = api.start_conversation({"user_id": "12345"})

# During conversation, push a detailed form-filling FSM
response = api.push_fsm(
    conv_id,
    "examples/basic/form_filling/fsm.json",
    context_to_pass={"form_type": "preferences"},
    shared_context_keys=["user_id", "preferences"],
    preserve_history=True,
    inherit_context=True
)

# User interacts with the sub-FSM
response = api.converse("I need to update my profile", conv_id)

# When sub-FSM completes, pop back to main FSM with collected data
response = api.pop_fsm(
    conv_id, 
    context_to_return={"profile_completed": True},
    merge_strategy="update"
)

# Continue with main FSM which now has the form data
response = api.converse("What products do you recommend?", conv_id)

# Check stack depth and context flow
print(f"Stack depth: {api.get_stack_depth(conv_id)}")
print(f"Context flow: {api.get_context_flow(conv_id)}")
```

### Custom Handlers Example

Add custom logic at various points in the conversation:

```python
from llm_fsm import API
from llm_fsm.handlers import HandlerTiming
import time

# Create API instance
api = API.from_file("examples/basic/simple_greeting/fsm.json")

# Add a logging handler
api.add_logging_handler(
    log_timings=[HandlerTiming.PRE_PROCESSING, HandlerTiming.POST_PROCESSING]
)

# Add a custom handler using the fluent interface
handler = api.create_handler("TimestampHandler") \
    .at(HandlerTiming.POST_PROCESSING) \
    .do(lambda ctx: {"last_interaction": time.time()})
api.register_handler(handler)

# Add state-specific handler
api.add_state_entry_handler(
    state="greeting",
    handler_func=lambda ctx: {"greeting_time": time.time()}
)

# Start conversation with handlers active
conv_id, response = api.start_conversation()
response = api.converse("Hello!", conv_id)

# Handlers automatically execute during conversation flow
data = api.get_data(conv_id)
print(f"Handler data: {data}")
```

### Command Line Interface

LLM-FSM provides several command-line tools:

-   **Run a conversation:**
    ```bash
    llm-fsm --fsm path/to/your/fsm.json
    ```
-   **Visualize an FSM definition:**
    ```bash
    llm-fsm-visualize --fsm path/to/your/fsm.json
    # For different styles:
    llm-fsm-visualize --fsm path/to/your/fsm.json --style compact
    llm-fsm-visualize --fsm path/to/your/fsm.json --style minimal
    ```
-   **Validate an FSM definition:**
    ```bash
    llm-fsm-validate --fsm path/to/your/fsm.json
    ```

## Core Components

### LLM-FSM Core (`llm_fsm`)

Located in `src/llm_fsm/`, this is the heart of the library:
-   **`API` (`api.py`):** High-level interface with FSM stacking and handler support.
-   **`FSMDefinition` (`definitions.py`):** Pydantic models for defining FSMs (states, transitions, conditions).
-   **`FSMManager` (`fsm.py`):** Manages FSM instances, state transitions, and context.
-   **`LLMInterface` (`llm.py`):** Interface for LLM communication, with `LiteLLMInterface` for broad provider support.
-   **`PromptBuilder` (`prompts.py`):** Constructs structured system prompts for the LLM.
-   **`Handler System` (`handlers.py`):** Advanced system for custom logic integration.
-   **`JsonLogic Expressions` (`expressions.py`):** Evaluates complex conditions for transitions.
-   **`Validator` (`validator.py`):** Validates FSM definition files.
-   **`Visualizer` (`visualizer.py`):** Generates ASCII art for FSMs.

### FSM Stacking & Context Handover

The framework supports sophisticated FSM stacking with comprehensive context management:

**Stack Operations:**
- `push_fsm()`: Add a new FSM to the conversation stack
- `pop_fsm()`: Return to the previous FSM with context handover
- `get_stack_depth()`: Check current stack depth
- `get_context_flow()`: Inspect context flow between FSMs

**Context Handover Features:**
- **Forward Context**: Pass context from parent to child FSM
- **Backward Context**: Return context from child to parent FSM
- **Shared Context Keys**: Automatically sync specified keys across the stack
- **Merge Strategies**: Control how contexts are merged (UPDATE, PRESERVE, SELECTIVE)
- **History Preservation**: Optionally maintain conversation history across FSM transitions

### Handler System

The advanced handler system allows custom Python logic integration:

**Handler Types:**
- **Custom Classes**: Implement the `FSMHandler` protocol
- **Lambda Handlers**: Use the fluent `HandlerBuilder` interface
- **Convenience Handlers**: Pre-built handlers for common use cases

**Execution Points:**
- `START_CONVERSATION`: When conversation begins
- `PRE_PROCESSING`: Before LLM processes user input
- `POST_PROCESSING`: After LLM responds
- `PRE_TRANSITION`: Before state changes
- `POST_TRANSITION`: After state changes
- `CONTEXT_UPDATE`: When context is updated
- `END_CONVERSATION`: When conversation ends
- `ERROR`: When errors occur

### Workflows Extension (`llm_fsm_workflows`)

Located in `src/llm_fsm_workflows/`, this extension builds upon the core FSM to enable complex, automated processes:
-   **`WorkflowDefinition` (`definitions.py`):** Defines sequences of steps.
-   **`WorkflowStep` (`steps.py`):** Various step types (API calls, conditions, LLM processing, FSM conversations).
-   **`WorkflowEngine` (`engine.py`):** Executes workflow instances and manages state.
-   **`DSL` (`dsl.py`):** Fluent API for programmatically creating workflow definitions.

### JsonLogic Expressions

A powerful, JSON-based way to define complex conditions for state transitions. These expressions are evaluated against the current conversation context, enabling sophisticated conditional logic.

## Examples & Tutorials

The repository includes comprehensive examples and tutorials:

### Basic Examples (`examples/basic/`)
-   **`simple_greeting`**: Minimal FSM with greeting and farewell
-   **`form_filling`**: Step-by-step information collection
-   **`story_time`**: Interactive storytelling FSM

### Intermediate Examples (`examples/intermediate/`)
-   **`book_recommendation`**: Conversational recommendation system
-   **`product_recommendation`**: Decision-tree conversation for tech products

### Advanced Examples (`examples/advanced/`)
-   **`yoga_instructions`**: Adaptive yoga instruction based on user engagement
-   **`e_commerce`**: Complex e-commerce workflow with multiple FSMs

### Tutorials (`examples/tutorials/`)
-   **`01_hello_world`**: Your first FSM
-   **`02_adding_context`**: Working with conversation context
-   **`03_conditional_transitions`**: Using JsonLogic for complex conditions
-   **`04_custom_handlers`**: Integrating custom logic
-   **`05_complex_workflows`**: Building multi-FSM applications

### Use Cases (`examples/use_cases/`)
Real-world application examples:
-   **`customer_service`**: Customer support workflows
-   **`e_commerce`**: Shopping cart and checkout processes
-   **`education`**: Interactive learning experiences
-   **`entertainment`**: Games and interactive stories
-   **`healthcare`**: Patient intake and triage systems
-   **`personal_assistant`**: AI assistant workflows

Each example includes:
-   `fsm.json`: The FSM definition
-   `run.py`: Python script to run the example
-   `README.md`: Detailed explanation and usage

## Documentation

-   **[LLM Reference (LLM.md)](./LLM.md):** Detailed guide for LLMs to understand the framework's architecture, system prompt structure, and expected response formats.
-   **[Handler Integration Guide](./docs/fsm_handler_integration_guide.md):** Learn how to use the handler system to extend FSM functionality.

## Development

-   **Testing:** Run tests using `tox` or `make test` (uses `pytest`):
    ```bash
    tox
    # or
    make test
    ```
-   **Linting & Formatting:** Uses `flake8` and `black`. Configured in `tox.ini`:
    ```bash
    tox -e lint
    ```
-   **Building:** Use `make build` to create wheel and sdist packages:
    ```bash
    make build
    ```
-   **Cleaning:** Use `make clean` to remove build artifacts.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](./LICENSE) file for details.