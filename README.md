# FSM-LLM: Adding State to the Stateless

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![PyPI version](https://badge.fury.io/py/fsm-llm.svg)](https://badge.fury.io/py/fsm-llm)

<p align="center">
  <img src="./images/fsm-llm-logo-1.png" alt="FSM-LLM Logo" width="500"/>
</p>

**FSM-LLM is a Python framework for building robust, stateful conversational AI applications by combining the power of Large Language Models (LLMs) with the predictability of Finite State Machines (FSMs).**

---

## Why FSM-LLM?

Large Language Models (LLMs) are phenomenal at generating human-like text. However, their inherent statelessness makes it challenging to build complex, multi-turn conversations that require remembering context, following structured flows, and making consistent decisions.

**FSM-LLM bridges this gap by:**

*   **Leveraging LLMs:** For natural language understanding, intent recognition, information extraction, and dynamic response generation.
*   **Employing Finite State Machines:** To provide a clear, testable, and predictable structure for conversation flows.
*   **Managing State & Context:** The Python framework handles state transitions, context persistence, and business logic, allowing the LLM to focus on what it does best: language.

The result? You can build sophisticated conversational agents that:
*   Follow well-defined, predictable paths.
*   Remember information across multiple turns.
*   Handle complex branching logic with ease.
*   Integrate with external systems and custom logic seamlessly.
*   Feel natural and intelligent to the end-user.

---

## Quick Installation

Get started with FSM-LLM in seconds:

```bash
pip install fsm-llm
```

For LLM-backed intent classification and routing, install the optional `classification` extension:
```bash
pip install fsm-llm[classification]
```

For advanced workflow orchestration capabilities (event-driven flows, timers, parallel execution), install the optional `workflows` extension:
```bash
pip install fsm-llm[workflows]
```

---

## Configuration

Before you run your first bot, you'll need to configure your LLM provider. FSM-LLM uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood, giving you access to 100+ LLM providers (OpenAI, Anthropic, Cohere, local models via Ollama, etc.).

Create a `.env` file in your project root (or set environment variables):
```bash
# .env file
OPENAI_API_KEY=sk-your-openai-key-here
LLM_MODEL=gpt-4o-mini # Or any LiteLLM supported model string
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000
```
*(See `.env.example` for a template)*

---

## Your First Stateful Bot (2 Minutes!)

Let's create a simple bot that asks for your name and then greets you personally.

**1. Define your FSM (e.g., `my_first_bot.json`):**
```json
{
  "name": "MyFirstBot",
  "description": "A simple bot to ask for a name and greet.",
  "initial_state": "ask_name",
  "persona": "A friendly and curious assistant.",
  "states": {
    "ask_name": {
      "id": "ask_name",
      "purpose": "Politely ask the user for their name.",
      "transitions": [{
        "target_state": "greet_user",
        "description": "User has provided their name."
      }]
    },
    "greet_user": {
      "id": "greet_user",
      "purpose": "Greet the user personally using their name.",
      "required_context_keys": ["name"],
      "transitions": []
    }
  }
}
```

**2. Write your Python script (e.g., `run_bot.py`):**
```python
from fsm_llm import API
import os

# Ensure your API key is set via environment variable or .env file
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found. Please set it in your environment or .env file.")
    exit()

# Initialize the API with your FSM definition
# (Assumes my_first_bot.json and .env are in the same directory)
api = API.from_file("my_first_bot.json") # Model and API key are picked from .env

# Start the conversation
conversation_id, response = api.start_conversation()
print(f"Bot: {response}")

# Interact with the bot
user_name = input("You: ")
response = api.converse(user_name, conversation_id)
print(f"Bot: {response}")

# Check the collected data
collected_data = api.get_data(conversation_id)
print(f"\nData collected by bot: {collected_data}")

print("Conversation ended." if api.has_conversation_ended(conversation_id) else "Conversation ongoing.")
```

**3. Run it!**
```bash
python run_bot.py
```

You've just created a stateful conversation! The bot remembered the name you provided because FSM-LLM managed the state and context.

---

## Core Features

*   **2-Pass Architecture**: A key design that separates **Data Extraction** from **Response Generation**. This ensures transitions happen *before* a response is crafted, leading to more consistent and contextually-aware conversations.

*   **JSON-Defined FSMs:** Design your conversation flows declaratively.
    *   Define states, their purposes, and specific instructions for the LLM.
    *   Specify transitions, conditions (using JsonLogic), and priorities.
    *   Set a global `persona` for consistent bot behavior.
    *   *(See `docs/fsm_design.md` for best practices)*

*   **Intelligent Prompt Engineering:**
    *   Automatic generation of structured XML-like prompts for the LLM.
    *   Includes current state, context, history, valid transitions, and response format requirements.
    *   Secure: Sanitizes inputs to prevent prompt injection.
    *   *(See `LLM.md` for the detailed prompt structure given to the LLM)*

*   **Powerful Handler System:**
    *   Extend FSM behavior with custom Python functions at various `HandlerTiming` points (e.g., `PRE_PROCESSING`, `POST_TRANSITION`, `CONTEXT_UPDATE`).
    *   Fluent `HandlerBuilder` for easy creation:
        ```python
        def my_custom_logic(context):
            # ... do something ...
            return {"new_data": "value"}

        api.register_handler(
            api.create_handler("MyLogicHandler")
               .at(HandlerTiming.POST_PROCESSING)
               .on_state("some_state")
               .when_context_has("specific_key")
               .do(my_custom_logic)
        )
        ```
    *   *(See `docs/handlers.md` for details)*

*   **FSM Stacking:**
    *   Build complex, modular applications by stacking FSMs.
    *   `api.push_fsm(...)` to delegate to a sub-FSM for specialized tasks.
    *   `api.pop_fsm(...)` to return to the parent FSM with merged context.
    *   *(Explore `examples/advanced/e_commerce/run.py`)*

*   **Structured Reasoning Engine:**
    *   Utilize a dedicated FSM-based engine for decomposing and solving complex problems.
    *   Comes with pre-built FSMs for various reasoning types: Analytical, Deductive, Inductive, Creative, Critical, and a Hybrid orchestrator.
    *   Includes an intelligent FSM-based classifier to select the most appropriate reasoning strategy.
    *   *(See `src/fsm_llm_reasoning/` for implementation details and FSM definitions).*

*   **Expression Evaluation:**
    *   Use [JsonLogic](https://jsonlogic.com/) for defining complex `conditions` in your FSM transitions.
    *   *(See `src/fsm_llm/expressions.py` and `tests/test_fsm_llm/test_expressions.py`)*

*   **Command-Line Tools:**
    *   `fsm-llm --fsm <path_to_fsm.json>`: Run any FSM interactively.
    *   `fsm-llm-visualize --fsm <path_to_fsm.json>`: Generate an ASCII visualization.
    *   `fsm-llm-validate --fsm <path_to_fsm.json>`: Validate your FSM definition.

*   **(Optional) Structured Classification:**
    *   Map free-form user input to predefined intent classes with validated JSON output.
    *   Supports single-intent, multi-intent, and hierarchical (two-stage) classification.
    *   Includes `IntentRouter` for mapping classified intents to handler functions.
    *   *(See `src/fsm_llm_classification/` for implementation)*

*   **(Optional) Workflow Engine:**
    *   If `fsm-llm[workflows]` is installed, orchestrate FSMs with event-driven steps, timers, and parallel execution.
    *   Define workflows using a Python DSL.
    *   *(See `src/fsm_llm_workflows/` for implementation)*

---

## Project Structure

```
├── .github/workflows/        # CI/CD (GitHub Actions)
├── docs/                     # Detailed documentation
│   ├── architecture.md
│   ├── api_reference.md
│   ├── fsm_design.md
│   ├── handlers.md
│   └── quickstart.md
├── examples/                 # Practical examples
│   ├── basic/
│   ├── intermediate/
│   └── advanced/
├── src/
│   ├── fsm_llm/              # Core FSM-LLM library
│   │   ├── api.py            # Primary user-facing API class
│   │   ├── definitions.py    # Pydantic models for FSM structure
│   │   ├── fsm.py            # FSMManager, core state logic
│   │   ├── handlers.py       # Handler system and builder
│   │   ├── llm.py            # LLM interface (LiteLLM)
│   │   ├── prompts.py        # Prompt engineering
│   │   ├── transition_evaluator.py # Deterministic transition logic
│   │   ├── expressions.py    # JsonLogic evaluator
│   │   └── ...               # Other utilities, constants, logging
│   ├── fsm_llm_reasoning/    # Structured reasoning engine
│   │   ├── engine.py         # Core reasoning logic
│   │   ├── reasoning_modes.py# FSM definitions for reasoning strategies
│   │   ├── handlers.py       # Custom handlers for reasoning processes
│   │   ├── definitions.py    # Pydantic models for reasoning traces
│   │   └── ...               # Other utilities and constants
│   ├── fsm_llm_classification/ # Optional structured classification extension
│   │   ├── classifier.py     # Classifier and HierarchicalClassifier
│   │   ├── definitions.py    # Pydantic models for schemas and results
│   │   ├── prompts.py        # Prompt and JSON schema builders
│   │   └── router.py         # Intent-to-handler routing
│   └── fsm_llm_workflows/    # Optional workflow engine extension
│       ├── engine.py         # Core workflow execution engine
│       ├── dsl.py            # Python DSL for defining workflows
│       └── ...               # Other utilities, steps, exceptions
├── tests/                    # Unit and integration tests
├── .env.example              # Example environment variables
├── LLM.md                    # Guide for how LLMs should interpret prompts
├── pyproject.toml            # Project metadata and dependencies
└── README.md                 # This file
```

---

## Learn More

*   **[Quick Start Guide](./docs/quickstart.md)**: Your first steps with FSM-LLM.
*   **[FSM Design Guide](./docs/fsm_design.md)**: Best practices for crafting effective FSMs.
*   **[Handler Development](./docs/handlers.md)**: Adding custom logic and integrations.
*   **[API Reference](./docs/api_reference.md)**: Detailed documentation of the `API` class and its methods.
*   **[Architecture Deep Dive](./docs/architecture.md)**: Understand the internals of FSM-LLM.
*   **[LLM Interaction Guide](./LLM.md)**: How FSM-LLM structures prompts for LLMs and expects responses.

---

## Development & Contributing

We welcome contributions! Whether it's bug fixes, new features, examples, or documentation improvements, your help is valued.

**Setup your development environment:**
```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/fsm_llm.git
cd fsm_llm

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

# Install in editable mode with development dependencies
pip install -e ".[dev,workflows,classification]"

# Set up pre-commit hooks
pre-commit install
```

**Running Tests:**
```bash
# Run all tests
pytest

# Or use the Makefile
make test
```

**Contribution Guidelines:**
1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Write tests for your changes.
4.  Ensure all tests pass (`pytest`).
5.  Format your code and check linting.
6.  Update documentation if necessary.
7.  Submit a pull request!

---

## Use Cases

FSM-LLM is ideal for building a wide range of stateful conversational applications, including:

*   **Chatbots & Virtual Assistants:** Customer service, personal assistants, technical support.
*   **Information Collection:** Smart forms, surveys, user onboarding.
*   **Workflow Automation:** Guiding users through multi-step processes.
*   **Interactive Storytelling:** Choose-your-own-adventure games, educational narratives.
*   **E-commerce:** Personalized shopping assistants, product recommenders.
*   **Tutoring Systems:** Adaptive learning paths, interactive quizzes.
*   **Complex Problem Solving:** Decomposing and solving intricate problems using structured reasoning strategies.
*   **Intent Classification:** Mapping natural language input to predefined classes for routing and automation.

---

## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](./LICENSE) file for details.

---

<p align="center">
  <b>Give your LLM the memory and structure it deserves.</b><br>
  Build reliable, stateful conversational AI with FSM-LLM.
  <br><br>
  <a href="https://pypi.org/project/fsm-llm/">Install on PyPI</a> |
  <a href="./examples/">Explore Examples</a> |
  <a href="https://github.com/NikolasMarkou/fsm_llm/discussions">Join Discussions</a> |
  <a href="https://github.com/NikolasMarkou/fsm_llm/issues">Report Issues</a>
</p>
