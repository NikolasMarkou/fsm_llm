# FSM-LLM: Adding State to the Stateless

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![PyPI version](https://badge.fury.io/py/fsm-llm.svg)](https://badge.fury.io/py/fsm-llm)
[![Tests](https://github.com/NikolasMarkou/fsm_llm/actions/workflows/python-package.yml/badge.svg)](https://github.com/NikolasMarkou/fsm_llm/actions)

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

## Key Features

- **2-pass architecture** -- Pass 1 extracts data and evaluates transitions; Pass 2 generates the response from the correct state.
- **Handler system** -- 8 hook points (pre/post processing, pre/post transition, context update, etc.) with a fluent builder API.
- **JsonLogic transitions** -- Deterministic rule-based transitions with operators like `==`, `in`, `has_context`, `and`, `or`.
- **FSM stacking** -- Push/pop nested FSMs with context merging for complex multi-flow scenarios.
- **100+ LLM providers** -- OpenAI, Anthropic, Ollama, Azure, AWS Bedrock, and more via litellm.
- **4 extension packages** -- Reasoning, workflows, agents (12 patterns + meta builder), and monitoring dashboard.
- **Security built in** -- Internal key prefixes, forbidden context patterns, XML tag sanitization.

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
| `reasoning` | `pip install fsm-llm[reasoning]` | None |
| `agents` | `pip install fsm-llm[agents]` | None |
| `workflows` | `pip install fsm-llm[workflows]` | None |
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
          "conditions": [{"description": "User said goodbye", "logic": {"has_context": "wants_to_leave"}}]
        }
      ]
    },
    "farewell": {
      "id": "farewell",
      "purpose": "Say goodbye",
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
User Input → [Pass 1: Data Extraction (LLM)] → Context Update
           → Transition Evaluation (JsonLogic rules or LLM classification)
           → State Transition
           → [Pass 2: Response Generation (LLM)] → User Output
```

Pass 2 runs **after** the transition, so the response always reflects the correct state.

## Extension Packages

### Classification (built into core)

LLM-backed intent classification: single-intent, multi-intent, hierarchical two-stage, and `IntentRouter`.

```python
from fsm_llm import Classifier, ClassificationSchema, IntentDefinition

schema = ClassificationSchema(intents=[
    IntentDefinition(name="billing", description="Billing and payment questions"),
    IntentDefinition(name="technical", description="Technical support issues"),
])
classifier = Classifier(schema=schema, model="openai/gpt-4o-mini")
result = classifier.classify("I can't log in to my account")
```

### Reasoning -- 9 strategies as FSMs

```python
from fsm_llm_reasoning import ReasoningEngine
engine = ReasoningEngine(model="openai/gpt-4o-mini")
solution, trace = engine.solve_problem("What is the probability of rolling two sixes?")
```

### Workflows -- async event-driven, 11 step types

```python
from fsm_llm_workflows import create_workflow, auto_step, llm_step, conversation_step
workflow = create_workflow("order_pipeline") \
    .add(auto_step("validate", action=validate_order)) \
    .add(llm_step("summarize", prompt="Summarize: {order}")) \
    .add(conversation_step("support", fsm_file="support.json")) \
    .build()
```

### Agents -- 12 patterns with tool use

```python
from fsm_llm_agents import create_agent, tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

agent = create_agent(tools=[search])
result = agent("What is the capital of France?")
```

Patterns: ReAct, REWOO, Reflexion, Plan-Execute, Prompt Chain, Self-Consistency, Debate, Orchestrator, ADaPT, Eval-Optimize, Maker-Checker, Reasoning-ReAct.

### Monitor -- web dashboard

```bash
fsm-llm-monitor  # Opens at http://localhost:8420
```

Pages: Dashboard, Control Center, Visualizer, Conversations, Logs, Builder, Settings.

### Meta Builder -- interactive artifact creation

```bash
fsm-llm-meta  # Interactive CLI for building FSMs, workflows, agents
```

## CLI Tools

| Command | Description |
|---------|-------------|
| `fsm-llm --fsm <path.json>` | Run an FSM interactively |
| `fsm-llm-visualize --fsm <path.json>` | ASCII visualization |
| `fsm-llm-validate --fsm <path.json>` | Validate FSM definition |
| `fsm-llm-monitor` | Launch web monitoring dashboard |
| `fsm-llm-meta` | Interactive artifact builder |

## Examples

82 examples across 8 categories:

| Category | Count | Highlights |
|----------|-------|------------|
| Basic | 5 | simple_greeting, form_filling, story_time, multi_turn_extraction |
| Intermediate | 3 | book_recommendation, product_recommendation, adaptive_quiz |
| Advanced | 7 | e_commerce (FSM stacking), support_pipeline, handler_hooks, concurrent_conversations |
| Classification | 5 | intent_routing, smart_helpdesk, classified_transitions, multi_intent |
| Reasoning | 1 | math_tutor |
| Workflows | 8 | order_processing, parallel_steps, conditional_branching, loan_processing |
| Agents | 48 | react_search, plan_execute, reflexion, debate, orchestrator, adapt, and more |
| Meta | 5 | build_fsm, build_workflow, build_agent, meta_review_loop, meta_from_spec |

Run with: `python examples/<category>/<name>/run.py`. See `EVALUATE.md` for evaluation results.

## Development

```bash
make install-dev    # Install in dev mode with all extras + pre-commit hooks
make test           # Run full test suite (2,206 tests)
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make type-check     # mypy across all packages
make build          # python -m build (wheel + sdist)
make coverage       # Tests with coverage report
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
