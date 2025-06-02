# LLM-FSM: Adding State to the Stateless 🧠🔄💾

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org)
[![PyPI version](https://badge.fury.io/py/llm-fsm.svg)](https://badge.fury.io/py/llm-fsm)

<p align="center">
  <img src="./images/fsm-llm-logo-1.png" alt="LLM-FSM Logo" width="500"/>
</p>

**LLM-FSM is a Python framework for building robust, stateful conversational AI applications by combining the power of Large Language Models (LLMs) with the predictability of Finite State Machines (FSMs).**

---

## 🎯 Why LLM-FSM?

Large Language Models (LLMs) are phenomenal at understanding and generating human-like text. However, their inherent statelessness makes it challenging to build complex, multi-turn conversations that require remembering context, following structured flows, and making consistent decisions.

**LLM-FSM bridges this gap by:**

*   🧠 **Leveraging LLMs:** For natural language understanding, intent recognition, information extraction, and dynamic response generation.
*   🔄 **Employing Finite State Machines:** To provide a clear, testable, and predictable structure for conversation flows.
*   💾 **Managing State & Context:** Python handles the state transitions, context persistence, and business logic, allowing the LLM to focus on language tasks.

The result? You can build sophisticated conversational agents that:
*   Follow well-defined paths.
*   Remember information across multiple turns.
*   Handle complex branching logic.
*   Integrate with external systems seamlessly.
*   Feel natural and intelligent to the user.

---

## 🚀 Quick Installation

Get started with LLM-FSM in seconds:

```bash
pip install llm-fsm
```

For advanced workflow orchestration capabilities (event-driven flows, timers, parallel execution), install the optional `workflows` extension:
```bash
pip install llm-fsm[workflows]
```

---

## ⚙️ Configuration

Before you run your first bot, you'll need to configure your LLM provider. LLM-FSM uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood, giving you access to 100+ LLM providers (OpenAI, Anthropic, Cohere, local models via Ollama, etc.).

Create a `.env` file in your project root (or set environment variables):
```bash
# .env.example
OPENAI_API_KEY=sk-your-openai-key-here
LLM_MODEL=gpt-4o-mini # Or any LiteLLM supported model string
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000
```
*(See `.env.example` for a template)*

---

## 💡 Your First Stateful Bot (2 Minutes!)

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
      "required_context_keys": ["user_name"],
      "transitions": []
    }
  }
}
```

**2. Write your Python script (e.g., `run_bot.py`):**
```python
from llm_fsm import API
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
print(f"Bot: {response}") # Expected: "Hello! I'm a friendly assistant. What's your name?"

# Interact with the bot
user_name = input("You: ")
response = api.converse(user_name, conversation_id)
print(f"Bot: {response}") # Expected: "Nice to meet you, [User's Name]! ..."

# Check the collected data
collected_data = api.get_data(conversation_id)
print(f"Data collected by bot: {collected_data}") # Expected: {'user_name': '[User's Name]', ...}

print("Conversation ended." if api.has_conversation_ended(conversation_id) else "Conversation ongoing.")
```

**3. Run it!**
```bash
python run_bot.py
```

You've just created a stateful conversation! The bot remembered the name you provided because LLM-FSM managed the state and context.

---

## 🔥 Core Features

*   **JSON-Defined FSMs:** Design your conversation flows declaratively.
    *   Define states, their purposes, and instructions for the LLM.
    *   Specify transitions, conditions (using JsonLogic), and priorities.
    *   Set a global `persona` for consistent bot behavior.
    *   *(See `docs/fsm_design.md` for best practices)*

*   **Intelligent Prompt Engineering:**
    *   Automatic generation of structured XML-like prompts for the LLM.
    *   Includes current state, context, history, valid transitions, and response format.
    *   Secure: Sanitizes inputs to prevent prompt injection.
    *   *(See `LLM.md` for the detailed prompt structure given to the LLM)*

*   **Context Management:**
    *   Automatically collects data specified in `required_context_keys`.
    *   LLM can store additional relevant info in an `_extra` field.
    *   Conversation history is managed and truncated to fit token limits.

*   **Flexible LLM Integration:**
    *   Powered by `LiteLLM` for broad provider compatibility (OpenAI, Anthropic, Cohere, Ollama, etc.).
    *   Easily configure `model`, `temperature`, `max_tokens`.
    *   Plug in your custom `LLMInterface` for specialized needs.

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

*   **FSM Stacking & Workflows:**
    *   Build complex, modular applications by stacking FSMs.
    *   `api.push_fsm(...)` to delegate to a sub-FSM.
    *   `api.pop_fsm(...)` to return to the parent FSM.
    *   Control context inheritance, sharing, and merge strategies.
    *   Preserve conversation history across the stack.
    *   *(Explore `examples/advanced/e_commerce/run.py`)*

*   **Expression Evaluation:**
    *   Use JsonLogic for defining complex `conditions` in your FSM transitions.
    *   *(See `src/llm_fsm/expressions.py` and `tests/test_llm_fsm/test_expressions.py`)*

*   **Comprehensive Observability:**
    *   Detailed logging with `Loguru`, including per-conversation context.
    *   `api.get_data()`, `api.get_current_state()`, `api.get_conversation_history()`.
    *   `api.save_conversation()` to persist state.

*   **Command-Line Tools:**
    *   `llm-fsm --fsm <path_to_fsm.json>`: Run any FSM interactively.
    *   `llm-fsm-visualize --fsm <path_to_fsm.json>`: Generate an ASCII visualization.
    *   `llm-fsm-validate --fsm <path_to_fsm.json>`: Validate your FSM definition.

*   **Structured Reasoning Engine:**
    *   Utilize a dedicated FSM-based engine for decomposing and solving complex problems.
    *   Comes with pre-built FSMs for various reasoning types: Analytical, Deductive, Inductive, Creative, Critical, and a Hybrid orchestrator.
    *   Includes an intelligent FSM-based classifier to select the most appropriate reasoning strategy for a given problem.
    *   Provides full traceability of the reasoning process, detailing each step and context change.
    *   Easily extensible with custom reasoning patterns and domain-specific classifiers.
    *   *(See `src/llm_fsm_reasoning/` for implementation details and `src/llm_fsm_reasoning/fsms/` for the various reasoning FSM definitions).*

*   **(Optional) Workflow Engine:**
    *   If `llm-fsm[workflows]` is installed, orchestrate FSMs with event-driven steps, timers, and parallel execution.
    *   Define workflows using a Python DSL.
    *   *(See `src/llm_fsm_workflows/` for implementation)*

---

## 📂 Project Structure

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
│   ├── llm_fsm/              # Core LLM-FSM library
│   │   ├── __init__.py       # Main package exports
│   │   ├── api.py            # Primary user-facing API class
│   │   ├── definitions.py    # Pydantic models for FSM structure
│   │   ├── fsm.py            # FSMManager, core state logic
│   │   ├── handlers.py       # Handler system and builder
│   │   ├── llm.py            # LLM interface (LiteLLM)
│   │   ├── prompts.py        # Prompt engineering
│   │   ├── expressions.py    # JsonLogic evaluator
│   │   ├── validator.py      # FSM validation logic
│   │   ├── visualizer.py     # ASCII FSM visualizer
│   │   ├── runner.py         # CLI runner logic
│   │   ├── __main__.py       # CLI entry point
│   │   └── ...               # Other utilities, constants, logging
│   ├── llm_fsm_reasoning/    # Structured reasoning engine
│   │   ├── __init__.py       # Reasoning engine exports
│   │   ├── engine.py         # Core reasoning logic
│   │   ├── handlers.py       # Custom handlers for reasoning processes
│   │   ├── models.py         # Pydantic models for reasoning traces and results
│   │   ├── constants.py      # Reasoning types and context keys
│   │   ├── fsms/             # Pre-defined FSMs for reasoning strategies
│   │   │   ├── analytical.json
│   │   │   └── ...           # (deductive, inductive, creative, critical, hybrid, classifier, orchestrator)
│   │   └── ...               # Other utilities
│   └── llm_fsm_workflows/    # Optional workflow engine extension
│       ├── __init__.py       # Workflow engine exports
│       ├── engine.py         # Core workflow execution engine
│       ├── dsl.py            # Python DSL for defining workflows
│       ├── steps.py          # Various workflow step implementations
│       ├── definitions.py    # Pydantic models for workflow structure
│       └── ...               # Other utilities, handlers, exceptions
├── tests/                    # Unit and integration tests
│   ├── fixtures/
│   ├── test_llm_fsm/
│   └── ...
├── .env.example              # Example environment variables
├── LLM.md                    # Guide for how LLMs should interpret prompts
├── Makefile                  # Build/test automation
├── pyproject.toml            # Project metadata and dependencies
├── requirements.txt          # Core dependencies
└── README.md                 # This file
```

---

## 📚 Learn More

*   **[Quick Start Guide](./docs/quickstart.md)**: Your first steps with LLM-FSM.
*   **[FSM Design Guide](./docs/fsm_design.md)**: Best practices for crafting effective FSMs.
*   **[Handler Development](./docs/handlers.md)**: Adding custom logic and integrations.
*   **[API Reference](./docs/api_reference.md)**: Detailed documentation of the `API` class and its methods.
*   **[Architecture Deep Dive](./docs/architecture.md)**: Understand the internals of LLM-FSM.
*   **[LLM Interaction Guide](./LLM.md)**: How LLM-FSM structures prompts for LLMs and expects responses.

---

## 🛠️ Development & Contributing

We welcome contributions! Whether it's bug fixes, new features, examples, or documentation improvements, your help is valued.

**Setup your development environment:**
```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/llm-fsm.git
cd llm-fsm

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

# Install in editable mode with development dependencies
pip install -e ".[dev,workflows]" # Install with all optional extras

# Set up pre-commit hooks
pre-commit install
```

**Running Tests:**
```bash
pytest  # Run all tests
# Or use Makefile:
make test
```

**Contribution Guidelines:**
1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Write tests for your changes.
4.  Ensure all tests pass (`pytest`).
5.  Format your code with `black .` and check linting with `flake8 .`.
6.  Update documentation if necessary.
7.  Submit a pull request!

---

## 🌟 Use Cases

LLM-FSM is ideal for building a wide range of stateful conversational applications, including:

*   🤖 **Chatbots & Virtual Assistants:** Customer service, personal assistants, technical support.
*   📝 **Information Collection:** Smart forms, surveys, user onboarding.
*   ⚙️ **Workflow Automation:** Guiding users through multi-step processes.
*   🎮 **Interactive Storytelling:** Choose-your-own-adventure games, educational narratives.
*   🛍️ **E-commerce:** Personalized shopping assistants, product recommenders.
*   🎓 **Tutoring Systems:** Adaptive learning paths, interactive quizzes.
*   💡 **Complex Problem Solving:** Decomposing and solving intricate problems using structured reasoning strategies.

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](./LICENSE) file for details.

---

<p align="center">
  <b>Give your LLM the memory and structure it deserves.</b><br>
  Build reliable, stateful conversational AI with LLM-FSM.
  <br><br>
  <a href="https://pypi.org/project/llm-fsm/">📦 Install on PyPI</a> |
  <a href="./examples/">🚀 Explore Examples</a> |
  <a href="https://github.com/nikolasmarkou/llm-fsm/discussions">💬 Join Discussions</a> |
  <a href="https://github.com/nikolasmarkou/llm-fsm/issues">🐛 Report Issues</a>
</p>