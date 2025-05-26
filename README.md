# LLM-FSM: Adding State to the Stateless

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/llm-fsm.svg)](https://badge.fury.io/py/llm-fsm)

![logo](./images/fsm-llm-logo-1.png)

## Table of Contents
- [The Problem: Stateless LLMs in Structured Conversations](#the-problem-stateless-llms-in-structured-conversations)
- [The Solution: Finite State Machines + LLMs](#the-solution-finite-state-machines--llms)
  - [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Core Library](#core-library)
  - [With Workflows Extension](#with-workflows-extension)
  - [For Development](#for-development)
- [Quick Start](#quick-start)
  - [Environment Setup](#environment-setup)
  - [Simplified Python API](#simplified-python-api)
  - [Command Line Interface](#command-line-interface)
- [Core Components](#core-components)
  - [LLM-FSM Core (`llm_fsm`)](#llm-fsm-core-llm_fsm)
  - [Workflows Extension (`llm_fsm_workflows`)](#workflows-extension-llm_fsm_workflows)
  - [Handler System](#handler-system)
  - [JsonLogic Expressions](#jsonlogic-expressions)
- [Examples](#examples)
  - [Basic Examples](#basic-examples)
  - [Intermediate Examples](#intermediate-examples)
  - [Advanced Examples](#advanced-examples)
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

The state and transitions are handled by python and language ambiguities are handled by the LLM.

This hybrid approach gives you the best of both worlds:
- ✅ **Predictable conversation flows** with clear rules and transitions
- ✅ **Natural language understanding** powered by state-of-the-art LLMs
- ✅ **Persistent context** across the entire conversation
- ✅ **Dynamic adaptation** to user inputs
- ✅ **Expressive Logic** for complex transitional decision-making using JsonLogic
- ✅ **Extensible Workflows** for building complex, multi-step automated processes
- ✅ **Customizable Handlers** for integrating external logic and side effects

### Key Features

- 🚦 **Structured Conversation Flows**: Define states, transitions, and conditions in JSON.
- 🧠 **LLM-Powered NLU**: Leverage LLMs for understanding, entity extraction, and response generation.
- 🎣 **Handler System**: Integrate custom Python functions at various lifecycle points of FSM execution.
- 👤 **Persona Support**: Define a consistent tone and style for LLM responses.
- 📝 **Persistent Context Management**: Maintain information throughout the conversation.
- 🔄 **Provider-Agnostic**: Works with OpenAI, Anthropic, and other LLM providers via LiteLLM.
- 📊 **Visualization & Validation**: Built-in CLI tools to visualize FSMs as ASCII art and validate definitions.
- 🪵 **Comprehensive Logging**: Detailed logs via Loguru for debugging and monitoring.
- 🧪 **Test-Friendly**: Designed for easy unit testing and behavior verification.
- 🧮 **JsonLogic Expressions**: Powerful conditional logic for logic based FSM transitions.
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

### Simplified Python API

The `LLM_FSM` class provides a high-level interface:

```python
from llm_fsm import API
import os

# Ensure OPENAI_API_KEY is set in your environment
# export OPENAI_API_KEY='your-key-here'

# Create the LLM-FSM instance from an FSM definition file
fsm = API.from_file(
    path="examples/basic/simple_greeting/fsm.json",
    model="gpt-4o-mini",  # Or your preferred model from .env
    # api_key="your-api-key" # Can be omitted if OPENAI_API_KEY is set
)

# Start a conversation (empty message for initial greeting)
conversation_id, response = fsm.converse("")
print(f"System: {response}")

# Continue conversation
while not fsm.has_conversation_ended(conversation_id):
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    _, response = fsm.converse(user_input, conversation_id)
    print(f"System: {response}")

# Get collected data (if any)
data = fsm.get_data(conversation_id)
print(f"Collected data: {data}")

# End the conversation
fsm.end_conversation(conversation_id)
print("Conversation ended.")
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
-   **Run Workflows (if `workflows` extra is installed):**
    *(Details for workflow CLI to be added as the feature matures)*
    ```bash
    # Example: llm-fsm-workflow --workflow path/to/workflow.json --run
    ```

## Core Components

### LLM-FSM Core (`llm_fsm`)

Located in `src/llm_fsm/`, this is the heart of the library.
-   **`FSMDefinition` (`definitions.py`):** Pydantic models for defining FSMs (states, transitions, conditions).
-   **`FSMManager` (`fsm.py`):** Manages FSM instances, state transitions, and context.
-   **`LLMInterface` (`llm.py`):** Interface for LLM communication, with `LiteLLMInterface` for broad provider support.
-   **`PromptBuilder` (`prompts.py`):** Constructs structured system prompts for the LLM.
-   **`JsonLogic Expressions` (`expressions.py`):** Evaluates complex conditions for transitions.
-   **`Validator` (`validator.py`):** Validates FSM definition files.
-   **`Visualizer` (`visualizer.py`):** Generates ASCII art for FSMs.

### Handler System (`llm_fsm.handler_system`)

Provides a way to inject custom Python logic at various points in the FSM execution lifecycle (e.g., before/after processing, on context update, on state transition). See the [FSM Handler Integration Guide](./docs/fsm_handler_integration_guide.md).

### JsonLogic Expressions (`llm_fsm.expressions`)

A powerful, JSON-based way to define complex conditions for state transitions. These expressions are evaluated against the current conversation context.

### Workflows Extension (`llm_fsm_workflows`)

Located in `src/llm_fsm_workflows/`, this extension builds upon the core FSM to enable more complex, automated processes.
-   **`WorkflowDefinition` (`definitions.py`):** Defines a sequence of steps.
-   **`WorkflowStep` (`steps.py`):** Abstract base class for various step types (API calls, conditions, LLM processing, etc.).
-   **`WorkflowEngine` (`engine.py`):** Executes workflow instances, manages state, and handles events.
-   **`DSL` (`dsl.py`):** A fluent API for programmatically creating workflow definitions.

## Examples

The `examples/` directory contains various FSM definitions and run scripts:

### Basic Examples
-   **`simple_greeting`**: A minimal FSM with greeting and farewell.
-   **`form_filling`**: A step-by-step form for information collection.
-   **`book_recommendation`**: A conversational loop for recommending books.
-   **`story_time`**: An interactive storytelling FSM.
-   **`dialog_persona`**: Demonstrates using a detailed persona for the LLM.

### Intermediate Examples
-   **`product_recommendation_system`**: A decision-tree conversation for tech product recommendations.

### Advanced Examples
-   **`yoga_instructions`**: An FSM that adapts yoga instruction based on user engagement.

Each example typically includes:
-   `fsm.json`: The FSM definition.
-   `run.py`: A Python script to run the example.
-   `README.md`: Explanation of the example.

## Development

-   **Testing:** Run tests using `tox` or `make test` (which uses `pytest`).
    ```bash
    tox
    # or
    make test
    ```
-   **Linting & Formatting:** Uses `flake8` and `black`. Configured in `tox.ini` and `.pre-commit-config.yaml`.
    ```bash
    tox -e lint
    ```
-   **Building:** Use `make build` to create wheel and sdist packages.
    ```bash
    make build
    ```
-   **Cleaning:** Use `make clean` to remove build artifacts.

## Documentation

-   **[LLM Reference (LLM.md)](./LLM.md):** A detailed guide designed for LLMs to understand the framework's architecture, system prompt structure, and expected response formats.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.
(A more formal `CONTRIBUTING.md` can be added later).

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](./LICENSE) file for details.
