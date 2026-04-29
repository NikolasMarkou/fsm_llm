<div align="center">

<img src="./images/fsm-llm-logo-1.png" alt="FSM-LLM" width="420"/>

# FSM-LLM

**Stateful LLM programs with predictable cost.**

One typed runtime. Two surface syntaxes. One verb.

[![PyPI](https://badge.fury.io/py/fsm-llm.svg)](https://pypi.org/project/fsm-llm/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![Tests](https://github.com/NikolasMarkou/fsm_llm/actions/workflows/python-package.yml/badge.svg)](https://github.com/NikolasMarkou/fsm_llm/actions)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[Quick start](#quick-start) · [Examples](#examples) · [Architecture](#how-it-works) · [Documentation](#documentation)

</div>

---

## Why FSM-LLM?

Most LLM frameworks are great for one-shot prompts and start to wobble when you need real program structure: turn-by-turn dialog state, multi-step pipelines, retries with budgets, or chunked recursion over a long document.

FSM-LLM is built around two ideas:

1. **Pick the right surface for the shape of your program.**
   - A **chatbot** with persistent state across user turns → write a JSON state machine.
   - A **pipeline** that runs one shot through a few LLM steps (extract → reason → answer; ReAct; debate; …) → compose a Python λ-term.
   - A **long-context task** that needs to chunk and recurse over a big document → use a long-context primitive.

   All three compile to the same typed AST and run on the same executor.

2. **Know the cost before you run it.** Every program has a planner that returns the exact number of LLM calls in advance — so you can size your budget, set retry limits, and catch regressions in CI.

Works with **100+ LLM providers** through [litellm](https://github.com/BerriAI/litellm) — OpenAI, Anthropic, Ollama, Azure, Bedrock, Vertex AI, and so on.

## Install

```bash
pip install fsm-llm
```

Optional extras: `[monitor]`, `[mcp]`, `[otel]`, `[a2a]`, `[oolong]`, `[all]`. See [optional extras](#optional-extras).

Requires Python 3.10+.

## Quick start

The unified entry point is `Program`. Three constructors cover three program shapes.

### 1. A chatbot with state — `Program.from_fsm`

Describe states + transitions in JSON. The framework keeps track of where the user is and what's been collected.

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
        "conditions": [{"description": "User said goodbye",
                        "logic": {"has_context": "wants_to_leave"}}]
      }]
    },
    "farewell": {
      "id": "farewell",
      "purpose": "Say goodbye",
      "response_instructions": "Say goodbye warmly, using the user's name if known"
    }
  }
}
```

```python
from fsm_llm import Program

program = Program.from_fsm("greeting.json", model="openai/gpt-4o-mini")

result = program.invoke(message="Hi! I'm Alice.")
print(result.value)                      # the assistant reply
print(result.conversation_id)            # use this for the next turn

result = program.invoke(message="Goodbye!", conversation_id=result.conversation_id)
print(result.value)
```

Or run it from the terminal:

```bash
export OPENAI_API_KEY=sk-...
fsm-llm run greeting.json
```

### 2. A multi-step pipeline — `Program.from_term`

For stateless flows — `extract → reason → answer`, ReAct, REWOO, debate, and so on — author a λ-term directly:

```python
from fsm_llm import Program, leaf, let_
from pydantic import BaseModel

class Topic(BaseModel):
    topic: str

term = let_(
    "topic", leaf(prompt="Extract the topic in one word: {q}", schema=Topic, input_var="q"),
    leaf(prompt="Write a one-paragraph article about {topic}.", input_var="topic"),
)

program = Program.from_term(term, model="openai/gpt-4o-mini")
result = program.invoke(inputs={"q": "What is photosynthesis?"})
print(result.value)
assert result.oracle_calls == 2          # known ahead of time
```

### 3. Long-context recursion with a hard cost gate — `Program.from_factory`

Chunk and recurse over a document that's too big for one prompt. The planner gives a closed-form call count and a hard equality assertion catches drift:

```python
from fsm_llm import Program
from fsm_llm.stdlib.long_context import niah

program = Program.from_factory(niah, factory_kwargs={
    "question": "Where is the needle hidden?", "tau": 256, "k": 2,
})
result = program.invoke(inputs={"document": four_kb_document})
print(result.value)
assert result.oracle_calls == result.plan.predicted_calls    # 16, exactly
```

## Features

- **Two surfaces, one runtime.** Author chatbots in declarative FSM JSON, or write λ-terms directly. Both compile to the same typed AST and run on the same `Executor`.
- **Predictable cost.** A planner computes the exact LLM-call count of any program before it runs. Long-context demos ship with hard equality assertions; cost regressions fail CI.
- **One verb (`Program.invoke`).** Returns a uniform `Result` in every mode — value, conversation id, planner output, leaf/oracle counts.
- **One Oracle.** Every LLM call across both surfaces flows through the Oracle owned by the `Program`, which makes telemetry, structured output, and provider routing universal.
- **100+ providers** via [litellm](https://github.com/BerriAI/litellm). Swap models with one string.
- **Schema-typed outputs.** Pass a Pydantic class to any prompt step; the framework enforces the schema at the LLM boundary.
- **JsonLogic transitions.** Deterministic rule-based transitions (`==`, `in`, `has_context`, `and`, `or`, …) — no surprises from a classifier.
- **Standard library** — agent patterns (ReAct, REWOO, Reflexion, Plan-Execute, debate, swarm, MCP, A2A, …), reasoning strategies, workflow orchestration, long-context primitives.
- **Lifecycle hooks.** Eight typed timing points (pre/post processing, pre/post transition, context update, start/end, error) with a fluent builder API.
- **Persistent sessions.** Atomic file-based session store; resume conversations across process restarts.
- **Observability.** Web dashboard (`fsm-llm monitor`) and OpenTelemetry exporter that ships per-AST-node spans.
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

Both surfaces compile to the same typed program tree (a `Term`), and one runtime evaluates it.

```
   FSM JSON  (chatbots)              Python λ-DSL  (pipelines, agents, long context)
        │                                       │
        ▼  compile_fsm                          ▼  dsl builders (leaf, fix, let_, case_, …)
   ┌──────────────────────────────────────────────────────┐
   │            L3  AUTHOR — typed Term                    │
   ├──────────────────────────────────────────────────────┤
   │            L2  COMPOSE — handler + transform passes   │
   ├──────────────────────────────────────────────────────┤
   │            L1  REDUCE — Executor · Planner · Oracle   │
   ├──────────────────────────────────────────────────────┤
   │            L4  INVOKE — Program.invoke → Result       │
   └──────────────────────────────────────────────────────┘
```

The planner is the load-bearing piece: for any program, it computes a closed-form `(τ, k, depth, predicted_calls)` from the AST shape, so cost is a property of the program rather than something you discover at runtime.

**Want the deep dive?** The merge contract — invariants, falsification gates, and the `Program` API specification — lives in [`docs/lambda_fsm_merge.md`](docs/lambda_fsm_merge.md). The architectural thesis (Theorems 1–5) lives in [`docs/lambda.md`](docs/lambda.md).

## Command-line tools

The unified `fsm-llm` binary dispatches to subcommands:

| Command | Description |
|---------|-------------|
| `fsm-llm run <target>` | Run an FSM JSON file or a Python factory (`pkg.mod:fn`) interactively |
| `fsm-llm explain <target>` | Print AST shape, leaf schemas, and planner output |
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

The repo ships **172 runnable examples** across 10 trees:

- `examples/basic/`, `examples/intermediate/`, `examples/advanced/` — small chatbots, multi-state flows, FSM stacking
- `examples/agents/` — every agent pattern with mock and real LLMs
- `examples/pipeline/` — λ-DSL twins of the agent patterns (Theorem-2 hard-asserted)
- `examples/long_context/` — NIAH, aggregate, pairwise, multi-hop, padded NIAH (with hard cost asserts)
- `examples/meta/` — meta-builder examples
- `examples/reasoning/`, `examples/workflows/`, `examples/classification/`

Run any of them with:

```bash
python examples/<tree>/<name>/run.py
```

See [`EVALUATE.md`](EVALUATE.md) for the evaluation harness and recent scorecards.

## Migration from 0.4.x

The legacy `API` class still works in 0.5.x with no warnings. To migrate to `Program`:

```python
# Before
from fsm_llm import API
api = API.from_file("bot.json", model="gpt-4o-mini")
conv_id, hello = api.start_conversation()
reply = api.converse("hi", conv_id)

# After
from fsm_llm import Program
program = Program.from_fsm("bot.json", model="gpt-4o-mini")
result = program.invoke(message="hi")
print(result.value, result.conversation_id)
```

`Program` works for the other two surfaces too — see Quick Start above.

## Documentation

- [`docs/quickstart.md`](docs/quickstart.md) — getting started
- [`docs/api_reference.md`](docs/api_reference.md) — full API: `Program`, `Result`, `Oracle`, `Executor`, λ-DSL, legacy `API`
- [`docs/architecture.md`](docs/architecture.md) — execution model, layered architecture, Theorem-2 cost model
- [`docs/fsm_design.md`](docs/fsm_design.md) — FSM design patterns and anti-patterns
- [`docs/handlers.md`](docs/handlers.md) — handler development
- [`docs/lambda.md`](docs/lambda.md) — the architectural thesis
- [`docs/lambda_fsm_merge.md`](docs/lambda_fsm_merge.md) — the merge contract (canonical)

## Contributing

```bash
git clone https://github.com/NikolasMarkou/fsm_llm
cd fsm_llm
make install-dev      # editable install + all extras + pre-commit hooks
make test             # run the suite
make lint             # ruff check
make type-check       # mypy
```

Then open a feature branch, make your changes, and submit a PR. Make sure `make lint` and `make test` pass.

## License

GNU General Public License v3.0 or later — see [LICENSE](LICENSE).
