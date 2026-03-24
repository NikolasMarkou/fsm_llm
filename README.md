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
# Core framework
pip install fsm-llm

# With intent classification
pip install fsm-llm[classification]

# With structured reasoning engine
pip install fsm-llm[reasoning]

# With workflow orchestration (event-driven flows, timers, parallel execution)
pip install fsm-llm[workflows]

# With agentic patterns (ReAct, Human-in-the-Loop)
pip install fsm-llm[agents]

# With real-time monitoring dashboard
pip install fsm-llm[monitor]

# Everything
pip install fsm-llm[classification,reasoning,workflows,agents,monitor]
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
      "description": "Ask for the user's name",
      "purpose": "Politely ask the user for their name.",
      "extraction_instructions": "Extract the user's name from their message.",
      "transitions": [{
        "target_state": "greet_user",
        "description": "User has provided their name."
      }]
    },
    "greet_user": {
      "id": "greet_user",
      "description": "Greet the user by name",
      "purpose": "Greet the user personally using their name.",
      "response_instructions": "Use the extracted name to greet the user warmly.",
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

*   **Powerful Handler System:**
    *   Extend FSM behavior with custom Python functions at 8 `HandlerTiming` points (e.g., `PRE_PROCESSING`, `POST_TRANSITION`, `CONTEXT_UPDATE`).
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

*   **Expression Evaluation:**
    *   Use [JsonLogic](https://jsonlogic.com/) for defining complex `conditions` in your FSM transitions.
    *   *(See `src/fsm_llm/expressions.py` and `tests/test_fsm_llm/test_expressions.py`)*

*   **Command-Line Tools:**
    *   `fsm-llm --fsm <path_to_fsm.json>`: Run any FSM interactively.
    *   `fsm-llm-visualize --fsm <path_to_fsm.json>`: Generate an ASCII visualization.
    *   `fsm-llm-validate --fsm <path_to_fsm.json>`: Validate your FSM definition.

---

## Classification Extension (`fsm_llm_classification`)

LLM-backed structured intent classification that maps free-form user input to predefined intent classes with validated JSON output.

```bash
pip install fsm-llm[classification]
```

### Key Classes

*   **`Classifier`** — Single-intent and multi-intent classification against a schema of up to ~15 intents.
*   **`HierarchicalClassifier`** — Two-stage classification (domain → intent) for larger intent taxonomies.
*   **`IntentRouter`** — Maps classified intents to handler functions with low-confidence fallback.

### Example

```python
from fsm_llm_classification import (
    Classifier, ClassificationSchema, IntentDefinition, IntentRouter
)

# Define your intent schema
schema = ClassificationSchema(
    intents=[
        IntentDefinition(name="greeting", description="User says hello"),
        IntentDefinition(name="farewell", description="User says goodbye"),
        IntentDefinition(name="help", description="User asks for help"),
    ],
    fallback_intent="help",
    confidence_threshold=0.6,
)

# Classify user input
classifier = Classifier(schema=schema, model="gpt-4o-mini")
result = classifier.classify("Hey there!")
print(f"Intent: {result.intent}, Confidence: {result.confidence}")

# Route intents to handlers
router = IntentRouter()
router.register("greeting", lambda msg, res: "Hello! How can I help?")
router.register("farewell", lambda msg, res: "Goodbye!")
response = router.route("Hey there!", result)
```

### Features

*   **Structured output** — Uses JSON schema enforcement when the LLM provider supports it, with prompt-based fallback.
*   **Multi-intent classification** — `classifier.classify_multi(text)` returns ranked intent scores.
*   **Hierarchical classification** — Two-stage domain→intent classification for large taxonomies.
*   **Reasoning field** — Schema design ensures the LLM reasons before classifying, reducing constrained-decoding distortion.

---

## Reasoning Engine (`fsm_llm_reasoning`)

A structured reasoning engine that decomposes and solves complex problems using 9 specialized reasoning strategies, each implemented as its own FSM.

```bash
pip install fsm-llm[reasoning]
```

### Key Classes

*   **`ReasoningEngine`** — Main entry point. Orchestrates problem analysis, strategy selection, and reasoning execution.
*   **`ReasoningType`** — Enum of 9 strategies: `ANALYTICAL`, `DEDUCTIVE`, `INDUCTIVE`, `ABDUCTIVE`, `ANALOGICAL`, `CREATIVE`, `CRITICAL`, `HYBRID`, `SIMPLE_CALCULATOR`.
*   **`ReasoningTrace`** / **`SolutionResult`** — Structured output models for reasoning steps and final solutions.

### Example

```python
from fsm_llm_reasoning import ReasoningEngine, ReasoningType

engine = ReasoningEngine(model="gpt-4o-mini")

# Let the engine auto-select the best strategy
solution, trace = engine.solve_problem(
    "What are the implications of rising interest rates on housing markets?"
)
print(solution)

# Or specify a strategy explicitly
solution, trace = engine.solve_problem(
    "All mammals are warm-blooded. Whales are mammals. Therefore..."
)
```

### CLI

```bash
# Interactive reasoning from the command line
python -m fsm_llm_reasoning "What causes inflation?" --type analytical
python -m fsm_llm_reasoning "problem statement" --output json --save results.json
```

### Architecture

The engine uses a hierarchical FSM approach:

1. **Orchestrator FSM** — Manages the overall flow: problem analysis → strategy selection → execution → synthesis → validation.
2. **Classifier FSM** — Automatically selects the best reasoning strategy during strategy selection.
3. **Specialized FSMs** — One per reasoning type, pushed onto the FSM stack for execution.

Built-in loop prevention (retry limiting) and automatic context pruning ensure stable execution.

---

## Workflow Engine (`fsm_llm_workflows`)

An event-driven workflow orchestration engine built on FSM-LLM. Enables automated state transitions, external API integration, timers, parallel execution, and embedded FSM conversations.

```bash
pip install fsm-llm[workflows]
```

### Key Classes

*   **`WorkflowEngine`** — Core async execution engine with timer management and workflow lifecycle.
*   **`WorkflowBuilder`** — Fluent API for building workflow definitions.
*   **8 Step Types:**
    *   `AutoTransitionStep` — Automatic state transitions with custom actions
    *   `APICallStep` — External API integration
    *   `ConditionStep` — Branching based on context evaluation
    *   `LLMProcessingStep` — LLM-powered processing within workflows
    *   `WaitForEventStep` — Event-driven triggers
    *   `TimerStep` — Time-based transitions
    *   `ParallelStep` — Concurrent step execution
    *   `ConversationStep` — Embeds a full FSM conversation within a workflow step
*   **DSL Functions** — `create_workflow()`, `auto_step()`, `api_step()`, `llm_step()`, `condition_step()`, `conversation_step()`, etc.

### Example

```python
from fsm_llm_workflows import create_workflow, auto_step, llm_step, conversation_step

# Build a workflow with the DSL
wf = create_workflow("onboarding", initial_state="welcome")
wf.with_step(auto_step("welcome", "collect_info", action=send_welcome_email))
wf.with_step(conversation_step(
    "collect_info", "Collect User Info",
    fsm_file="onboarding_form.json",
    model="gpt-4o-mini",
    context_mapping={"user_name": "name", "user_email": "email"},
    success_state="process",
))
wf.with_step(llm_step("process", "done", prompt_template="Summarize: {user_name}"))

definition = wf.build()
```

### Factory Functions

*   `linear_workflow(name, steps)` — Sequential step execution.
*   `conditional_workflow(name, ...)` — Branching based on conditions.
*   `event_driven_workflow(name, ...)` — Event-based triggers.

### ConversationStep

The `ConversationStep` embeds a complete FSM-LLM conversation inside a workflow step. It runs an FSM to completion, maps context between the workflow and conversation, and supports auto-messages for non-interactive flows.

```python
step = conversation_step(
    "collect_data", "Data Collection",
    fsm_file="form.json",
    initial_context={"user_name": "name"},   # workflow_key → conversation_key
    context_mapping={"collected_name": "name"},  # conversation_key → workflow_key
    auto_messages=["My name is Alice"],
    success_state="next_step",
)
```

---

## Agentic Patterns (`fsm_llm_agents`)

12 agentic design patterns built on top of the core FSM engine. Each pattern auto-generates FSM definitions at runtime — no manual JSON authoring required.

```bash
pip install fsm-llm[agents]
```

### Pattern Selection Guide

| Pattern | Best For | Tools? | Key Strength |
|---------|----------|--------|-------------|
| **ReactAgent** | General tool-use tasks | Required | Adaptive reasoning + acting loop |
| **ReflexionAgent** | Tasks needing self-improvement | Required | Episodic memory + verbal self-critique |
| **PlanExecuteAgent** | Multi-step research/analysis | Optional | Upfront planning + sequential execution |
| **REWOOAgent** | Token-efficient tool-use | Required | Plans ALL tools in 1 LLM call |
| **SelfConsistencyAgent** | Factual/reasoning questions | No | Majority vote over multiple samples |
| **DebateAgent** | Nuanced/controversial topics | No | Multi-perspective argumentation |
| **EvaluatorOptimizerAgent** | Quality-constrained generation | No | External evaluation + iterative refinement |
| **MakerCheckerAgent** | Content with quality gates | No | Two-persona quality loop |
| **PromptChainAgent** | Multi-stage generation | No | Linear pipeline of LLM steps |
| **OrchestratorAgent** | Complex decomposable tasks | Optional | Subtask delegation to workers |
| **ADaPTAgent** | Unknown complexity tasks | Optional | Try direct, decompose if needed |
| **ReasoningReactAgent** | Tasks needing structured reasoning | Required | ReAct + FSM-stacked reasoning strategies |

### Quick Example (ReactAgent)

```python
from fsm_llm_agents import ReactAgent, ToolRegistry, AgentConfig, tool

registry = ToolRegistry()

@tool(description="Search the web for information")
def search(query: str) -> str:
    return f"Results for: {query}"

registry.register(search._tool_definition)

agent = ReactAgent(
    tools=registry,
    config=AgentConfig(model="gpt-4o-mini", max_iterations=5),
)
result = agent.run("What is the population of France divided by 2?")
print(result.answer)
```

### Tool-Free Patterns

```python
from fsm_llm_agents import DebateAgent, SelfConsistencyAgent, AgentConfig

# Debate: multi-perspective analysis
debate = DebateAgent(num_rounds=2, proposer_persona="...", critic_persona="...")
result = debate.run("Should AI be used in hiring decisions?")

# Self-Consistency: reliable answers via majority vote
sc = SelfConsistencyAgent(num_samples=5)
result = sc.run("What is the capital of Australia?")
```

### Human-in-the-Loop

```python
from fsm_llm_agents import ReactAgent, ToolRegistry, HumanInTheLoop

hitl = HumanInTheLoop(
    approval_policy=lambda call, ctx: call.tool_name in ["send_email", "delete"],
    approval_callback=lambda req: input(f"Approve {req.tool_name}? (y/n): ") == "y",
    confidence_threshold=0.3,
)

agent = ReactAgent(tools=registry, hitl=hitl)
result = agent.run("Send a summary email to the team")
```

### Architecture

All agents follow the same API: `agent.run(task) -> AgentResult` with answer, success, trace, and final context. Each agent auto-generates an FSM definition tailored to its pattern, registers handlers for tool execution and budget enforcement, and runs the core FSM engine underneath.

*   **Auto-generated FSMs** — Each pattern builds FSM definitions at runtime from its configuration.
*   **Tool execution via handlers** — Tools run as POST_TRANSITION handlers in execution states.
*   **Budget enforcement** — Max iterations, timeout, and pattern-specific limits prevent runaway agents.
*   **Observation accumulation** — Tool results are automatically accumulated and pruned.
*   **Approval gates** — HITL approval before critical tool executions with configurable policies.
*   **Structured tracing** — Every agent returns an `AgentTrace` with steps, tool calls, and iteration counts.

---

## Monitor Dashboard (`fsm_llm_monitor`)

Web-based real-time monitoring dashboard for FSM-LLM conversations, agents, and workflows. Features a Grafana-inspired dark dashboard theme with live event streaming.

```bash
pip install fsm-llm[monitor]
```

### Key Classes

*   **`MonitorBridge`** — Connects an `EventCollector` to a live `API` instance. Registers observer handlers at all 8 timing points.
*   **`EventCollector`** — Thread-safe event and log capture with bounded deques. Computes metrics.
*   **`MonitorConfig`** — Configuration model (refresh interval, buffer sizes, log level, display options).

### Example

```python
import threading
import uvicorn
from fsm_llm import API
from fsm_llm_monitor import MonitorBridge
from fsm_llm_monitor.server import app, configure

api = API.from_file("my_bot.json")
bridge = MonitorBridge(api=api)
configure(bridge)

# Run monitor in background
threading.Thread(
    target=uvicorn.run,
    kwargs={"app": app, "host": "127.0.0.1", "port": 8420, "log_level": "warning"},
    daemon=True,
).start()

# Use the API as normal — events are captured automatically
conversation_id, response = api.start_conversation()
```

### Features

*   **5-page SPA** — Dashboard (metrics + events), Control Center (unified instance management), Visualizer (FSM/agent/workflow graphs), Logs (level-filtered stream), Settings (runtime config).
*   **36 REST + WebSocket endpoints** — Metrics, instances, events, logs, FSM/agent/workflow lifecycle, visualization, presets.
*   **WebSocket streaming** — Real-time metrics and event push at 1-second intervals.
*   **CLI standalone** — `fsm-llm-monitor` launches the dashboard with auto-open browser and preset FSM visualizer.
*   **Loguru integration** — Capture log records via a custom loguru sink.

*→ [API Reference](./docs/api_reference.md#monitor-dashboard-fsm_llm_monitor)*

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
│   ├── basic/               # simple_greeting, form_filling, story_time
│   ├── intermediate/        # book_recommendation, product_recommendation, adaptive_quiz
│   ├── advanced/            # yoga_instructions, e_commerce, support_pipeline
│   ├── classification/      # intent_routing, smart_helpdesk
│   ├── reasoning/           # math_tutor
│   ├── workflows/           # order_processing
│   └── agents/              # react_search, hitl_approval, debate, plan_execute, reflexion, + 13 more
├── src/
│   ├── fsm_llm/              # Core framework (~10,000 LOC)
│   │   ├── api.py            # API class — primary user-facing entry point
│   │   ├── fsm.py            # FSMManager — state machine orchestration
│   │   ├── definitions.py    # Pydantic models for FSM structure + exception hierarchy
│   │   ├── handlers.py       # Handler system, builder, and timing enum
│   │   ├── llm.py            # LLM interface (LiteLLM, 100+ providers)
│   │   ├── pipeline.py       # MessagePipeline — 2-pass message processing engine
│   │   ├── prompts.py        # Prompt engineering for extraction + response generation
│   │   ├── transition_evaluator.py # Rule-based transition evaluation with JsonLogic
│   │   ├── expressions.py    # JsonLogic evaluator
│   │   ├── context.py        # Context cleaning utilities
│   │   ├── runner.py         # Interactive CLI conversation runner
│   │   ├── validator.py      # FSM structure validation
│   │   ├── visualizer.py     # ASCII FSM diagrams
│   │   ├── utilities.py      # JSON extraction with fallback strategies
│   │   ├── constants.py      # Defaults, security patterns, internal key prefixes
│   │   ├── logging.py        # Loguru setup with conversation context
│   │   ├── __main__.py       # CLI entry point (run, validate, visualize)
│   │   ├── __version__.py    # Package version
│   │   └── __init__.py       # Public API exports
│   │
│   ├── fsm_llm_classification/  # Intent classification extension (~1,030 LOC)
│   │   ├── classifier.py     # Classifier + HierarchicalClassifier
│   │   ├── definitions.py    # Classification schemas and result models
│   │   ├── prompts.py        # Prompt and JSON schema builders
│   │   ├── router.py         # IntentRouter — intent-to-handler routing
│   │   ├── __version__.py    # Package version
│   │   └── __init__.py       # Public exports
│   │
│   ├── fsm_llm_reasoning/      # Structured reasoning engine (~4,300 LOC)
│   │   ├── engine.py         # ReasoningEngine — orchestrates 9 reasoning strategies
│   │   ├── reasoning_modes.py # FSM definitions for each strategy (as Python dicts)
│   │   ├── handlers.py       # Validation, tracing, context pruning, retry limiting
│   │   ├── definitions.py    # ReasoningStep, ReasoningTrace, SolutionResult
│   │   ├── constants.py      # ReasoningType enum, ContextKeys, OrchestratorStates
│   │   ├── utilities.py      # FSM loading, type mapping helpers
│   │   ├── exceptions.py     # ReasoningEngineError hierarchy
│   │   ├── __main__.py       # CLI: python -m fsm_llm_reasoning
│   │   ├── __version__.py    # Package version
│   │   └── __init__.py       # Public exports
│   │
│   ├── fsm_llm_workflows/      # Workflow orchestration engine (~2,600 LOC)
│   │   ├── engine.py          # WorkflowEngine — async execution engine
│   │   ├── dsl.py             # Python DSL and factory functions
│   │   ├── steps.py           # 8 step types including ConversationStep
│   │   ├── definitions.py     # WorkflowDefinition with validation
│   │   ├── models.py          # WorkflowStatus, WorkflowEvent, WorkflowInstance
│   │   ├── handlers.py        # Handler integration (engine manages operations directly)
│   │   ├── exceptions.py      # WorkflowError hierarchy (8 error types)
│   │   ├── __version__.py     # Package version
│   │   └── __init__.py        # Public exports
│   │
│   ├── fsm_llm_agents/         # Agentic patterns (~7,400 LOC)
│   │   ├── react.py            # ReactAgent — ReAct loop with tool dispatch
│   │   ├── tools.py            # ToolRegistry + @tool decorator
│   │   ├── hitl.py             # HumanInTheLoop — approval, escalation, override
│   │   ├── handlers.py         # Tool executor, iteration limiter, approval checker
│   │   ├── fsm_definitions.py  # Auto-generated FSM definitions for agent patterns
│   │   ├── prompts.py          # Tool-aware prompt builders
│   │   ├── definitions.py      # ToolDefinition, ToolCall, ToolResult, AgentTrace, AgentResult
│   │   ├── constants.py        # AgentStates, ContextKeys, Defaults
│   │   ├── exceptions.py       # AgentError hierarchy (7 error types)
│   │   ├── __main__.py         # CLI: python -m fsm_llm_agents --info
│   │   ├── __version__.py      # Package version
│   │   └── __init__.py         # Public exports
│   │
│   └── fsm_llm_monitor/        # Monitoring dashboard (~3,000 LOC)
│       ├── server.py            # FastAPI app — REST + WebSocket APIs
│       ├── bridge.py            # MonitorBridge — connects collector to API
│       ├── collector.py         # EventCollector — event capture + loguru sink
│       ├── instance_manager.py  # InstanceManager — lifecycle for FSMs, agents, workflows
│       ├── definitions.py       # Pydantic models (events, metrics, config, snapshots)
│       ├── constants.py         # Theme colors, event types, defaults
│       ├── exceptions.py        # MonitorError hierarchy (3 error types)
│       ├── __main__.py          # CLI: fsm-llm-monitor
│       ├── __version__.py       # Package version
│       ├── static/              # Frontend assets (13 JS modules, style.css, flows.json)
│       ├── templates/           # Jinja2 templates (index.html)
│       └── __init__.py          # Public exports
│
├── tests/                    # 1701 tests across 77 test files
├── .env.example              # Example environment variables
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
*   **[Monitor Dashboard](./docs/api_reference.md#monitor-dashboard-fsm_llm_monitor)**: Real-time monitoring with REST + WebSocket APIs.

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

# Install in editable mode with all development dependencies
pip install -e ".[dev,workflows,classification,reasoning,agents,monitor]"

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
*   **Complex Problem Solving:** Decomposing and solving intricate problems using 9 structured reasoning strategies.
*   **Intent Classification:** Mapping natural language input to predefined classes for routing and automation.
*   **Agentic Workflows:** ReAct agents with tool use and human-in-the-loop approval gates.
*   **Real-Time Monitoring:** Live dashboards for conversation metrics, event streams, and FSM visualization.

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
