# FSM-LLM: Adding State to the Stateless

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![PyPI version](https://badge.fury.io/py/fsm-llm.svg)](https://badge.fury.io/py/fsm-llm)
[![Tests](https://github.com/NikolasMarkou/fsm_llm/actions/workflows/ci.yml/badge.svg)](https://github.com/NikolasMarkou/fsm_llm/actions)

<p align="center">
  <img src="./images/fsm-llm-logo-1.png" alt="FSM-LLM Logo" width="500"/>
</p>

**A Python framework for building robust, stateful conversational AI by combining Large Language Models with Finite State Machines.**

---

## Why FSM-LLM?

Large Language Models generate remarkable text, but they are stateless. Building multi-turn conversations that remember context, follow structured flows, and make consistent decisions requires something more.

FSM-LLM bridges this gap:

- **LLMs handle language** -- understanding, extraction, and response generation.
- **Finite State Machines provide structure** -- predictable flows, testable transitions, and clear business logic.
- **The framework manages state** -- context persistence, transition evaluation, and handler orchestration.

The result is conversational agents that follow well-defined paths, remember information across turns, handle complex branching, and integrate with external systems -- while feeling natural to the end user.

## Key Features

- **2-pass architecture** -- Pass 1 extracts data and evaluates transitions; Pass 2 generates the response from the correct state.
- **Handler system** -- 8 hook points (pre/post processing, pre/post transition, context update, etc.) with a fluent builder API.
- **JsonLogic transitions** -- Deterministic rule-based transitions with operators like `==`, `in`, `has_context`, `and`, `or`.
- **FSM stacking** -- Push/pop nested FSMs with context merging for complex multi-flow scenarios.
- **100+ LLM providers** -- OpenAI, Anthropic, Ollama, Azure, AWS Bedrock, and more via litellm.
- **7 extension packages** -- Classification, reasoning, workflows, agents (12 patterns), monitoring dashboard, and meta-agent builder.
- **Security built in** -- Internal key prefixes, forbidden context patterns (passwords, secrets, tokens), XML tag sanitization.

## Installation

```bash
pip install fsm-llm
```

Install with extension packages:

```bash
pip install fsm-llm[all]  # Everything
```

Or pick what you need:

| Extra | Command | Additional Dependencies |
|-------|---------|------------------------|
| `classification` | `pip install fsm-llm[classification]` | None (included in core) |
| `reasoning` | `pip install fsm-llm[reasoning]` | None (included in core) |
| `agents` | `pip install fsm-llm[agents]` | None (included in core) |
| `meta` | `pip install fsm-llm[meta]` | None (included in core) |
| `workflows` | `pip install fsm-llm[workflows]` | None (included in core) |
| `monitor` | `pip install fsm-llm[monitor]` | fastapi, uvicorn, jinja2 |
| `all` | `pip install fsm-llm[all]` | All of the above |

## Quick Start

**1. Define an FSM** (`greeting.json`):

```json
{
  "name": "GreetingBot",
  "initial_state": "greeting",
  "persona": "A friendly assistant",
  "states": {
    "greeting": {
      "id": "greeting",
      "purpose": "Greet the user and ask their name",
      "extraction_instructions": "Extract the user's name if provided",
      "response_instructions": "Greet the user warmly and ask for their name if not yet known",
      "transitions": [
        {
          "target_state": "farewell",
          "description": "User wants to end the conversation",
          "conditions": [
            {
              "description": "User said goodbye",
              "logic": {"has_context": "wants_to_leave"}
            }
          ]
        }
      ]
    },
    "farewell": {
      "id": "farewell",
      "purpose": "Say goodbye",
      "extraction_instructions": "No extraction needed",
      "response_instructions": "Say a warm goodbye using the user's name if known"
    }
  }
}
```

**2. Run a conversation**:

```python
from fsm_llm import API

api = API.from_file("greeting.json", model="openai/gpt-4o-mini")
conversation_id, initial_response = api.start_conversation()

response = api.converse("Hi there! I'm Alice.", conversation_id)
print(response)

response = api.converse("Goodbye!", conversation_id)
print(response)
```

**3. Or use the CLI**:

```bash
export OPENAI_API_KEY="your-key-here"
fsm-llm --fsm greeting.json
```

## 2-Pass Architecture

```
User Input
    |
    v
+---------------------------+
|  Pass 1: Data Extraction  |  LLM extracts structured data from user input
+---------------------------+
    |
    v
+---------------------------+
|    Context Update         |  Extracted data merged into conversation context
+---------------------------+
    |
    v
+---------------------------+
|  Transition Evaluation    |  JsonLogic rules or LLM-assisted decision
+---------------------------+
    |
    v
+---------------------------+
|    State Transition       |  Move to target state (or stay)
+---------------------------+
    |
    v
+---------------------------+
|  Pass 2: Response Gen     |  LLM generates response from the FINAL state
+---------------------------+
    |
    v
User Output
```

Pass 2 runs **after** the transition, so the response always reflects the correct state. This eliminates the stale-response problem found in single-pass architectures.

## Extension Packages

### Classification

LLM-backed intent classification with structured output.

```python
from fsm_llm_classification import Classifier, ClassificationSchema, IntentDefinition

schema = ClassificationSchema(intents=[
    IntentDefinition(name="billing", description="Billing and payment questions"),
    IntentDefinition(name="technical", description="Technical support issues"),
])

classifier = Classifier(schema=schema, model="openai/gpt-4o-mini")
result = classifier.classify("I can't log in to my account")
print(result.intent, result.confidence)
```

Features: single-intent, multi-intent, hierarchical two-stage classification, `IntentRouter` for mapping intents to handlers.

### Reasoning

9 structured reasoning strategies implemented as FSMs.

```python
from fsm_llm_reasoning import ReasoningEngine

engine = ReasoningEngine(model="openai/gpt-4o-mini")
solution, trace = engine.solve_problem(
    "What is the probability of rolling two sixes?",
)
print(solution)
```

Strategies: analytical, deductive, inductive, abductive, analogical, causal, critical, creative, hybrid.

### Workflows

Async event-driven workflow orchestration with 8 step types.

```python
from fsm_llm_workflows import create_workflow, auto_step, llm_step, conversation_step

workflow = create_workflow("order_pipeline") \
    .add(auto_step("validate", action=validate_order)) \
    .add(llm_step("summarize", prompt="Summarize: {order}")) \
    .add(conversation_step("support", fsm_file="support.json")) \
    .build()
```

Step types: AutoTransition, APICall, Condition, LLMProcessing, WaitForEvent, Timer, Parallel, ConversationStep.

### Agents

12 agentic patterns with tool support, human-in-the-loop, and structured output.

```python
from fsm_llm_agents import create_agent, tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

agent = create_agent(tools=[search])
result = agent("What is the capital of France?")
print(result.answer)
```

All agents inherit from `BaseAgent` and support `__call__` syntax, structured output via `output_schema`, and agents-as-tools composition. The `@tool` decorator auto-infers parameter schemas from type hints.

Agent patterns: ReactAgent, REWOOAgent, PlanExecuteAgent, ReflexionAgent, DebateAgent, SelfConsistencyAgent, PromptChainAgent, EvaluatorOptimizerAgent, MakerCheckerAgent, OrchestratorAgent, ADaPTAgent, ReasoningReactAgent.

### Monitor

Web-based dashboard for real-time FSM, agent, and workflow monitoring.

```bash
fsm-llm-monitor
# Opens at http://localhost:8420
```

```python
from fsm_llm_monitor import MonitorBridge, configure, app
import uvicorn

bridge = MonitorBridge(api=api)
configure(bridge)
uvicorn.run(app, host="127.0.0.1", port=8420)
```

Pages: Dashboard (metrics, instance grid, events), Control Center (instance management), Visualizer (graph rendering), Logs (level-filtered stream), Settings.

### Meta

Interactive artifact builder -- constructs FSM definitions, workflow definitions, and agent configurations through conversation.

```python
from fsm_llm_meta import MetaAgent

agent = MetaAgent(model="openai/gpt-4o-mini")
response = agent.start()

while not agent.is_complete():
    user_input = input("> ")
    response = agent.send(user_input)
    print(response)

result = agent.get_result()
print(result.artifact_json)
```

```bash
fsm-llm-meta  # Interactive CLI
```

## CLI Tools

| Command | Description |
|---------|-------------|
| `fsm-llm --fsm <path.json>` | Run an FSM interactively in the terminal |
| `fsm-llm-visualize --fsm <path.json>` | Generate ASCII visualization of an FSM |
| `fsm-llm-validate --fsm <path.json>` | Validate an FSM definition file |
| `fsm-llm-monitor` | Launch the web monitoring dashboard |
| `fsm-llm-meta` | Interactive artifact builder |

## Examples

33 examples across 8 categories, organized by complexity:

| Category | Count | Highlights |
|----------|-------|------------|
| Basic | 3 | simple_greeting, form_filling, story_time |
| Intermediate | 3 | book_recommendation, product_recommendation, adaptive_quiz |
| Advanced | 3 | yoga_instructions, e_commerce (FSM stacking), support_pipeline |
| Classification | 3 | intent_routing, smart_helpdesk, classified_transitions |
| Reasoning | 1 | math_tutor |
| Workflows | 1 | order_processing |
| Agents | 18 | react_search, plan_execute, reflexion, debate, rewoo, full_pipeline, and more |
| Meta | 1 | build_fsm |

See [examples/README.md](examples/README.md) for the full catalog, sub-package usage matrix, and suggested learning path.

## Project Structure

```
src/
├── fsm_llm/                        # Core framework
│   ├── api.py                      # API class -- primary entry point
│   ├── fsm.py                      # FSMManager -- state machine orchestration
│   ├── pipeline.py                 # MessagePipeline -- 2-pass processing engine
│   ├── definitions.py              # Pydantic models: State, Transition, FSMDefinition, FSMContext
│   ├── handlers.py                 # HandlerSystem, HandlerBuilder, HandlerTiming (8 hook points)
│   ├── prompts.py                  # Prompt builders for extraction, response, transition
│   ├── llm.py                      # LLMInterface ABC + LiteLLMInterface
│   ├── ollama.py                   # Ollama-specific structured output helpers
│   ├── transition_evaluator.py     # Rule-based transition evaluation with JsonLogic
│   ├── expressions.py              # JsonLogic evaluator (var, ==, in, has_context, etc.)
│   ├── context.py                  # Context cleaning utilities
│   ├── constants.py                # Defaults, security patterns, internal key prefixes
│   ├── validator.py                # FSM structure validation
│   ├── visualizer.py               # ASCII FSM diagrams
│   ├── utilities.py                # JSON extraction with fallback strategies
│   ├── runner.py                   # Interactive CLI conversation runner
│   └── logging.py                  # Loguru setup with conversation context
│
├── fsm_llm_classification/         # Intent classification
│   ├── classifier.py               # Classifier, HierarchicalClassifier
│   ├── definitions.py              # ClassificationSchema, IntentDefinition, ClassificationResult
│   ├── prompts.py                  # Prompt and JSON schema builders
│   └── router.py                   # IntentRouter -- maps intents to handlers
│
├── fsm_llm_reasoning/              # Structured reasoning engine
│   ├── engine.py                   # ReasoningEngine -- 9 reasoning strategies via FSMs
│   ├── reasoning_modes.py          # FSM definitions for each strategy
│   ├── handlers.py                 # Validation, tracing, context pruning, retry limiting
│   └── definitions.py              # ReasoningStep, ReasoningTrace, SolutionResult
│
├── fsm_llm_workflows/              # Workflow orchestration
│   ├── engine.py                   # WorkflowEngine -- async event-driven execution
│   ├── dsl.py                      # Python DSL: create_workflow(), auto_step(), llm_step()
│   ├── steps.py                    # 8 step types: AutoTransition, APICall, Condition, etc.
│   └── definitions.py              # WorkflowDefinition with reachability/cycle validation
│
├── fsm_llm_agents/                 # Agentic patterns
│   ├── base.py                     # BaseAgent -- ABC with shared loop, budgets, __call__
│   ├── react.py                    # ReactAgent -- ReAct loop with tool dispatch
│   ├── tools.py                    # ToolRegistry + @tool decorator (auto-schema inference)
│   ├── hitl.py                     # HumanInTheLoop -- approval gates, escalation
│   ├── plan_execute.py             # PlanExecuteAgent
│   ├── reflexion.py                # ReflexionAgent -- self-reflection with memory
│   ├── debate.py                   # DebateAgent -- multi-perspective with judge
│   ├── self_consistency.py         # SelfConsistencyAgent -- multiple samples + voting
│   ├── rewoo.py                    # REWOOAgent -- planning-first execution
│   ├── prompt_chain.py             # PromptChainAgent -- sequential pipeline
│   ├── evaluator_optimizer.py      # EvaluatorOptimizerAgent -- iterative refinement
│   ├── maker_checker.py            # MakerCheckerAgent -- draft-review loop
│   ├── orchestrator.py             # OrchestratorAgent -- worker delegation
│   ├── adapt.py                    # ADaPTAgent -- adaptive complexity
│   └── reasoning_react.py          # ReasoningReactAgent -- ReAct + reasoning
│
├── fsm_llm_monitor/                # Web monitoring dashboard
│   ├── server.py                   # FastAPI server -- REST + WebSocket APIs
│   ├── bridge.py                   # MonitorBridge -- connects collector to API
│   ├── collector.py                # EventCollector -- handler-based event capture
│   ├── instance_manager.py         # Instance lifecycle management
│   ├── static/                     # Frontend: 15 JS modules + CSS
│   └── templates/index.html        # Single-page dashboard template
│
└── fsm_llm_meta/                   # Interactive artifact builder
    ├── agent.py                    # MetaAgent -- conversational builder orchestration
    ├── builders.py                 # FSMBuilder, WorkflowBuilder, AgentBuilder
    ├── handlers.py                 # Build-phase handlers
    ├── output.py                   # Artifact formatting and saving
    └── prompts.py                  # Builder-specific prompt generation
```

## Development

```bash
# Setup
make install-dev    # Install in dev mode with all extras + pre-commit hooks

# Testing
make test           # Run full test suite (1,970 tests)
make coverage       # Tests with coverage report

# Code quality
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make type-check     # mypy across all packages

# Build
make build          # python -m build (wheel + sdist)
make clean          # Remove build artifacts and caches

# Security
make audit          # Audit site-packages for suspicious .pth files
```

## Documentation

- [Quick Start Guide](docs/quickstart.md)
- [API Reference](docs/api_reference.md)
- [Architecture](docs/architecture.md) -- 2-pass flow, security model, performance
- [FSM Design Patterns](docs/fsm_design.md) -- patterns, anti-patterns, real-world examples
- [Handler Development](docs/handlers.md) -- 8 timing points, builder API, error handling

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Run `make install-dev` to set up the development environment
4. Make your changes with tests
5. Ensure `make lint` and `make test` pass
6. Submit a pull request

## License

This project is licensed under the GNU General Public License v3.0 or later. See [LICENSE](LICENSE) for details.
