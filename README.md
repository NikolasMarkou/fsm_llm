<div align="center">

<img src="./images/fsm-llm-logo-1.png" alt="FSM-LLM" width="420"/>

# FSM-LLM

**Stateful LLM apps with predictable cost.**

Build chatbots, agents, multi-step pipelines, and long-context workflows from a single Python runtime.

[![PyPI](https://badge.fury.io/py/fsm-llm.svg)](https://pypi.org/project/fsm-llm/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![Tests](https://github.com/NikolasMarkou/fsm_llm/actions/workflows/python-package.yml/badge.svg)](https://github.com/NikolasMarkou/fsm_llm/actions)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[Quick start](#quick-start) · [Examples](#examples) · [Documentation](#documentation) · [Contributing](#contributing)

</div>

---

## Why FSM-LLM?

Most LLM frameworks are great for one-shot prompts and start to wobble when you need real program structure: turn-by-turn dialog state, multi-step pipelines, retries with budgets, or chunked recursion over a long document.

FSM-LLM is built around two ideas:

1. **Pick the right surface for the shape of your program.** Use a JSON state machine when you need a chatbot with persistent state. Write a few-line Python expression when you need a stateless pipeline. Both compile down to the same engine.
2. **Know the cost before you run it.** Every program has a planner that tells you the exact number of LLM calls in advance — so you can size your budget, set retry limits, and catch regressions in CI.

Works with **100+ LLM providers** through [litellm](https://github.com/BerriAI/litellm) — OpenAI, Anthropic, Ollama, Azure, Bedrock, Vertex AI, and so on.

## Install

```bash
pip install fsm-llm
```

Optional extras: `[monitor]` (web dashboard), `[mcp]`, `[otel]`, `[a2a]`, `[oolong]`, or `[all]`. See [extras table](#optional-extras).

Requires Python 3.10+.

## Quick start

Three flavours of "real" LLM program — pick whichever matches your use case.

### 1. A chatbot with state

Describe states + transitions in JSON. The framework keeps track of where the user is and what's been collected so far.

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
      "response_instructions": "Greet warmly; ask for the name if not yet known",
      "transitions": [{
        "target_state": "farewell",
        "description": "User is leaving",
        "conditions": [{"description": "User said goodbye", "logic": {"has_context": "wants_to_leave"}}]
      }]
    },
    "farewell": {
      "id": "farewell",
      "purpose": "Say goodbye",
      "response_instructions": "Say a warm goodbye, using the user's name if known"
    }
  }
}
```

```python
from fsm_llm import API

api = API.from_file("greeting.json", model="openai/gpt-4o-mini")
conv_id, hello = api.start_conversation()
print(hello)
print(api.converse("Hi! I'm Alice.", conv_id))
print(api.converse("Goodbye!", conv_id))
```

Or just run it from the terminal:

```bash
export OPENAI_API_KEY=sk-...
fsm-llm run greeting.json
```

### 2. A multi-step pipeline

For stateless flows — `extract → reason → answer`, ReAct, REWOO, debate, and so on — write the steps directly:

```python
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.lam import Executor, LiteLLMOracle, leaf, let_
from pydantic import BaseModel

class Topic(BaseModel):
    topic: str

term = let_(
    "topic", leaf(prompt="Extract the topic in one word: {q}", schema=Topic, input_var="q"),
    leaf(prompt="Write a one-paragraph article about {topic}.", input_var="topic"),
)

ex = Executor(oracle=LiteLLMOracle(LiteLLMInterface(model="openai/gpt-4o-mini")))
print(ex.run(term, env={"q": "What is photosynthesis?"}))
assert ex.oracle_calls == 2   # known ahead of time
```

### 3. Long-context recursion with known cost

Chunk and recurse over a document that's too big for one prompt. The planner tells you the call count in closed form, and a hard assert catches drift:

```python
from fsm_llm.lam import Executor, LiteLLMOracle, plan, PlanInputs
from fsm_llm.stdlib.long_context import niah
from fsm_llm.llm import LiteLLMInterface

term = niah(question="Where is the needle hidden?", tau=256, k=2)
ex = Executor(oracle=LiteLLMOracle(LiteLLMInterface(model="openai/gpt-4o-mini")))
ex.run(term, env={"document": four_kb_document})

predicted = plan(PlanInputs(n=4096, tau=256, k=2)).predicted_calls
assert ex.oracle_calls == predicted   # 16 calls, exactly
```

## Features

- **Two surfaces, one runtime.** Author chatbots in declarative JSON, or write pipelines as small composable Python expressions. Both compile to the same typed AST.
- **Predictable cost.** A planner computes the exact LLM-call count of any program before it runs. Long-context demos ship with hard equality assertions; cost regressions fail CI.
- **100+ providers** via [litellm](https://github.com/BerriAI/litellm). Swap models with one string.
- **Schema-typed outputs.** Pass a Pydantic class to any prompt step; the framework enforces the schema at the LLM boundary.
- **JsonLogic transitions.** Deterministic rule-based transitions (`==`, `in`, `has_context`, `and`, `or`, …) — no surprises from a classifier.
- **Standard library** — agent patterns (ReAct, REWOO, Reflexion, Plan-Execute, debate, swarm, MCP, A2A, …), reasoning strategies, workflow orchestration (linear / branch / switch / parallel / retry), long-context primitives (NIAH, aggregate, pairwise, multi-hop).
- **Lifecycle hooks.** Eight typed hook points (pre/post processing, pre/post transition, context update, start/end, error) with a fluent builder API.
- **Persistent sessions.** Atomic file-based session store; resume conversations across process restarts.
- **Observability.** Web dashboard (`fsm-llm monitor`) and an OpenTelemetry exporter that ships per-AST-node spans.
- **Security defaults.** Internal context-key prefixes, forbidden patterns for secrets/tokens, XML-tag sanitisation in prompts.

## Standard library

| Subpackage | What it gives you |
|------------|-------------------|
| `fsm_llm.stdlib.agents` | `create_agent`, `tool` decorator, 12 agent patterns (ReAct, REWOO, Reflexion, …), MCP and A2A integrations, meta builder |
| `fsm_llm.stdlib.reasoning` | `ReasoningEngine` + 11 named strategy factories (analytical, abductive, deductive, …) |
| `fsm_llm.stdlib.workflows` | Async event-driven engine, 11 step types, 5 λ-term factories (linear/branch/switch/parallel/retry) |
| `fsm_llm.stdlib.long_context` | `niah`, `aggregate`, `pairwise`, `multi_hop`, `niah_padded` — closed-form cost per `plan(...)` |

A minimal agent:

```python
from fsm_llm.stdlib.agents import create_agent, tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

agent = create_agent(tools=[search])
print(agent("What is the capital of France?"))
```

## How it works

Both surfaces compile to a typed λ-AST. One `Executor` runs everything.

```
   FSM JSON  (chatbots)              Python DSL  (pipelines, agents, long context)
        │                                       │
        ▼  compile_fsm                          ▼  dsl builders
   ┌──────────────────────────────────────────────────────┐
   │                λ-AST  (typed Term)                   │
   └──────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────┐
   │ Executor + Planner + Oracle + Session + Cost tracker │
   └──────────────────────────────────────────────────────┘
```

The planner is the load-bearing piece: for any program, it computes a closed-form `(τ, k, depth, predicted_calls)` from the AST shape, so cost is a property of the program rather than something you discover at runtime.

The full architectural thesis (typed kernel, FSM-as-surface, Theorem 1–5) lives in [`docs/lambda.md`](docs/lambda.md).

## Command-line tools

The unified `fsm-llm` binary dispatches to subcommands:

| Command | Description |
|---------|-------------|
| `fsm-llm run <target>` | Run an FSM JSON file or a Python factory (`pkg.mod:fn`) interactively |
| `fsm-llm explain <target>` | Print AST shape, leaf schemas, and the planner output |
| `fsm-llm validate --fsm <path.json>` | Validate an FSM definition |
| `fsm-llm visualize --fsm <path.json>` | ASCII visualization of the state graph |
| `fsm-llm monitor` | Launch the web dashboard at `http://localhost:8420` |
| `fsm-llm meta` | Interactive builder for FSMs, workflows, and agents |

Legacy aliases (`fsm-llm-validate`, `fsm-llm-visualize`, `fsm-llm-monitor`, `fsm-llm-meta`, `fsm-llm --fsm <path>`) still work.

## Optional extras

```bash
pip install fsm-llm[all]          # everything
pip install fsm-llm[monitor]      # web dashboard
pip install fsm-llm[otel]         # OpenTelemetry exporter
pip install fsm-llm[mcp]          # MCP integration for agents
pip install fsm-llm[a2a]          # agent-to-agent transport
pip install fsm-llm[oolong]       # OOLONG long-context benchmark loader
pip install fsm-llm[dev]          # tests, lint, typecheck
```

The `reasoning`, `agents`, and `workflows` extras are pure-Python stdlib subpackages and are included in the core install — the extras exist only for explicit dependency declaration.

## Examples

The repo ships **152 runnable examples** across 10 trees:

- `examples/basic/` (14) and `examples/intermediate/` (3) — small chatbots, FSM-by-example
- `examples/advanced/` (17) — multi-state flows, classification, FSM stacking
- `examples/agents/` (48) — every agent pattern with mock and real LLMs
- `examples/pipeline/` (47) — λ-DSL twins of the agent patterns
- `examples/long_context/` (5) — NIAH, aggregate, pairwise, multi-hop, padded NIAH (with hard cost asserts)
- `examples/meta/` (5) — meta-builder examples
- `examples/reasoning/` (1), `examples/workflows/` (8), `examples/classification/` (4)

Run any of them with:

```bash
python examples/<tree>/<name>/run.py
```

See [`EVALUATE.md`](EVALUATE.md) for the evaluation harness and recent scorecards.

## Documentation

- [`docs/quickstart.md`](docs/quickstart.md) — getting started
- [`docs/api_reference.md`](docs/api_reference.md) — full API
- [`docs/architecture.md`](docs/architecture.md) — execution model, security, performance
- [`docs/fsm_design.md`](docs/fsm_design.md) — FSM design patterns and anti-patterns
- [`docs/handlers.md`](docs/handlers.md) — handler development
- [`docs/lambda.md`](docs/lambda.md) — the architectural thesis (Theorem 1–5)

## Contributing

```bash
git clone https://github.com/NikolasMarkou/fsm_llm
cd fsm_llm
make install-dev      # editable install + all extras + pre-commit hooks
make test             # run the suite (currently 2,899 tests)
make lint             # ruff check
make type-check       # mypy
```

Then open a feature branch, make your changes, and submit a PR. Make sure `make lint` and `make test` pass.

## License

GNU General Public License v3.0 or later — see [LICENSE](LICENSE).
