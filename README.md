# FSM-LLM

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![PyPI version](https://badge.fury.io/py/fsm-llm.svg)](https://badge.fury.io/py/fsm-llm)
[![Tests](https://github.com/NikolasMarkou/fsm_llm/actions/workflows/python-package.yml/badge.svg)](https://github.com/NikolasMarkou/fsm_llm/actions)

<p align="center">
  <img src="./images/fsm-llm-logo-1.png" alt="FSM-LLM Logo" width="500"/>
</p>

**Build stateful LLM applications — chatbots, agents, multi-step pipelines, and long-context workflows — with predictable cost and a single Python runtime.**

---

## What is this for?

LLMs are stateless. Real programs are not. fsm-llm gives you a clean way to build:

- **Chatbots that remember turn-to-turn state** — describe the conversation as states + transitions in a JSON file, and let the framework keep track of where the user is.
- **Multi-step pipelines** — `extract → reason → answer`, ReAct, REWOO, Reflexion, plan-execute, debate, and more, written as small composable Python expressions.
- **Long-context workflows** — recursively summarise / search / aggregate over documents that are too big for one prompt, with the cost known in advance.

It works with **100+ LLM providers** through [litellm](https://github.com/BerriAI/litellm) — OpenAI, Anthropic, Ollama, Azure, Bedrock, and the rest.

## Installation

```bash
pip install fsm-llm
```

That's it for the core. Optional extras (web dashboard, MCP, OpenTelemetry, OOLONG benchmark loader) are listed [further down](#optional-extras).

## Quick Start

### A chatbot from a JSON file

Save this as `greeting.json`:

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
        "description": "User wants to end the conversation",
        "conditions": [{"description": "User said goodbye", "logic": {"has_context": "wants_to_leave"}}]
      }]
    },
    "farewell": {
      "id": "farewell",
      "purpose": "Say goodbye",
      "response_instructions": "Say a warm goodbye using the user's name if known"
    }
  }
}
```

Then either run it from Python:

```python
from fsm_llm import API

api = API.from_file("greeting.json", model="openai/gpt-4o-mini")
conversation_id, greeting = api.start_conversation()
print(greeting)
print(api.converse("Hi! I'm Alice.", conversation_id))
print(api.converse("Goodbye!", conversation_id))
```

…or interact with it from the command line:

```bash
export OPENAI_API_KEY="your-key-here"
fsm-llm run greeting.json
```

### A multi-step pipeline in 5 lines

For stateless flows where you just need "do this, then do that", write the steps directly:

```python
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.lam import Executor, LiteLLMOracle, leaf, let_
from pydantic import BaseModel

class Topic(BaseModel): topic: str

term = let_(
    "topic", leaf(prompt="Extract the topic in one word: {q}", schema=Topic, input_var="q"),
    leaf(prompt="Write a one-paragraph article about {topic}.", input_var="topic"),
)
ex = Executor(oracle=LiteLLMOracle(LiteLLMInterface(model="openai/gpt-4o-mini")))
print(ex.run(term, env={"q": "What is photosynthesis?"}))
```

This makes exactly 2 LLM calls — the framework knows ahead of time, and you can [verify the cost from a planner](#under-the-hood).

### Long-context recursion with known cost

For programs that need to chunk and recurse over a long document:

```python
from fsm_llm.lam import Executor, LiteLLMOracle, plan, PlanInputs
from fsm_llm.stdlib.long_context import niah
from fsm_llm.llm import LiteLLMInterface

term = niah(question="Where is the needle hidden?", tau=256, k=2)
ex = Executor(oracle=LiteLLMOracle(LiteLLMInterface(model="ollama_chat/qwen3.5:4b")))
ex.run(term, env={"document": long_doc_4k_chars})

predicted = plan(PlanInputs(n=4096, tau=256, k=2)).predicted_calls
assert ex.oracle_calls == predicted   # 16 LLM calls, exactly as predicted
```

## Under the hood

fsm-llm is built on **one runtime, two surfaces**: a typed λ-calculus kernel that both the FSM JSON format and the Python pipeline DSL compile to. Whichever surface you pick, the same executor runs it.

```
   FSM JSON  (chatbots)              λ-DSL  (pipelines, agents, long context)
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

The planner gives you the LLM-call count of any program **before you run it**, in closed form (Theorem 2 in `docs/lambda.md`). Long-context demos ship with a hard `oracle_calls == plan(...).predicted_calls` gate — if cost ever drifts from the prediction, tests fail.

The runtime under the hood was renamed from `lam/` to `runtime/` in v0.3 (the old import path still works). The internals are documented in [`docs/lambda.md`](docs/lambda.md) — Theorem 1–5 and the architectural thesis.

## Key features

- **FSM JSON for chatbots** — declarative state machine with deterministic [JsonLogic](https://jsonlogic.com/) transitions (`==`, `in`, `has_context`, `and`, `or`, …) and a 2-pass per-turn flow (extract data → evaluate transition → generate response).
- **λ-DSL for pipelines, agents, and long context** — small Python factory functions returning composable terms. No frameworks, no callbacks.
- **Predictable cost** — the planner computes oracle-call counts ahead of time. Long-context bench cells (`evaluation/bench_long_context_*.json`) verify it on every run.
- **Stdlib subpackages** — agents (12 patterns + swarm, agent graph, MCP, A2A, SOPs, semantic tools, meta builder), reasoning (11 strategies), workflows (linear / branch / switch / parallel / retry), long context (`niah`, `aggregate`, `pairwise`, `multi_hop`, `niah_padded`).
- **Handler system** — 8 hook points (`PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `START_CONVERSATION`, `END_CONVERSATION`, `ERROR`) with a fluent builder API.
- **FSM stacking** — `push_fsm()` / `pop_fsm()` with context-merge strategies for nested dialog flows.
- **100+ LLM providers** through litellm (OpenAI, Anthropic, Ollama, Azure, Bedrock, …).
- **Web dashboard + OpenTelemetry** — `fsm-llm monitor` opens a dashboard at `http://localhost:8420`; OTEL exporter ships per-AST-node spans.
- **Security built in** — internal context-key prefixes, forbidden patterns for secrets/tokens, XML tag sanitisation in prompts.

## Optional extras

```bash
pip install fsm-llm[all]
```

| Extra | Command | Additional Dependencies |
|-------|---------|------------------------|
| `reasoning` | `pip install fsm-llm[reasoning]` | None (stdlib subpackage) |
| `agents` | `pip install fsm-llm[agents]` | None (stdlib subpackage) |
| `workflows` | `pip install fsm-llm[workflows]` | None (stdlib subpackage) |
| `monitor` | `pip install fsm-llm[monitor]` | fastapi, uvicorn, jinja2 |
| `mcp` | `pip install fsm-llm[mcp]` | mcp (>=1.0.0) |
| `otel` | `pip install fsm-llm[otel]` | opentelemetry-api, opentelemetry-sdk |
| `a2a` | `pip install fsm-llm[a2a]` | httpx (>=0.24.0) |
| `oolong` | `pip install fsm-llm[oolong]` | datasets (>=3.0.0) — OOLONG benchmark loader |

## Stdlib Subpackages

### Reasoning — 9 strategies + classifier + orchestrator

Class-based engine still works; M3 slice 2 also exposes 11 named λ-term factories.

```python
from fsm_llm.stdlib.reasoning import ReasoningEngine
engine = ReasoningEngine(model="openai/gpt-4o-mini")
solution, trace = engine.solve_problem("What is the probability of rolling two sixes?")
```

Or directly as a λ-term:

```python
from fsm_llm.stdlib.reasoning import analytical_term
term = analytical_term(problem_var="p", classify_prompt=..., decompose_prompt=..., synthesise_prompt=...)
```

### Workflows — async event-driven, 11 step types + 5 λ factories

```python
from fsm_llm.stdlib.workflows import create_workflow, auto_step, llm_step, conversation_step
workflow = create_workflow("order_pipeline") \
    .add(auto_step("validate", action=validate_order)) \
    .add(llm_step("summarize", prompt="Summarize: {order}")) \
    .add(conversation_step("support", fsm_file="support.json")) \
    .build()
```

Or as a λ-term factory: `linear_term`, `branch_term`, `switch_term`, `parallel_term`, `retry_term`.

### Agents — 12 patterns + 4 canonical λ-shapes

```python
from fsm_llm.stdlib.agents import create_agent, tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

agent = create_agent(tools=[search])
result = agent("What is the capital of France?")
```

Patterns: ReAct, REWOO, Reflexion, Plan-Execute, Prompt Chain, Self-Consistency, Debate, Orchestrator, ADaPT, Eval-Optimize, Maker-Checker, Reasoning-ReAct. Plus Swarm, Agent Graph, MCP, A2A, SOPs, semantic tool retrieval, meta builder.

M3 slice 1 ships 4 named λ-shape factories (each closes over no Python state):
- `react_term` — 2 oracle calls (decide → tool → synthesise)
- `rewoo_term` — 2 oracle calls (plan → execute → synthesise)
- `reflexion_term` — 4 oracle calls (solve → eval → reflect → re-solve)
- `memory_term` — 2 oracle calls (context → answer)

### Long-context — recurse over big documents

```python
from fsm_llm.stdlib.long_context import niah, aggregate, pairwise, multi_hop, niah_padded
```

Each factory returns a λ-term whose oracle cost is closed-form per `plan(...)`. See `examples/long_context/*_demo/` for runnable demos with hard Theorem-2 gates.

### Monitor — web dashboard + OTEL

```bash
fsm-llm monitor   # Opens at http://localhost:8420
```

Dashboard, Control Center, Visualizer, Conversations, Logs, Builder, Settings. OTEL span exporter ships λ-AST node spans (per-`Fix` / per-`Leaf` / per-`Combinator`).

### Meta builder

```bash
fsm-llm meta   # Interactive CLI for building FSMs, workflows, agents
```

## Sibling Packages — Back-Compat Shims

`fsm_llm_reasoning`, `fsm_llm_workflows`, and `fsm_llm_agents` are silent `sys.modules` shims that resolve to `fsm_llm.stdlib.<pkg>`. Existing imports keep working:

```python
from fsm_llm_agents import ReactAgent       # Still works
from fsm_llm.stdlib.agents import ReactAgent  # Same object — preferred for new code
```

## CLI Tools

The unified `fsm-llm` binary dispatches to subcommands:

| Command | Description |
|---------|-------------|
| `fsm-llm run <target>` | Run an FSM JSON file or a Python factory (`pkg.mod:fn`) interactively |
| `fsm-llm explain <target>` | Print AST shape, leaf schemas, and per-Fix planner output |
| `fsm-llm validate --fsm <path.json>` | Validate an FSM definition |
| `fsm-llm visualize --fsm <path.json>` | ASCII visualization of the state graph |
| `fsm-llm monitor` | Launch the web monitoring dashboard |
| `fsm-llm meta` | Interactive artifact builder (FSMs, workflows, agents) |

The legacy aliases (`fsm-llm-validate`, `fsm-llm-visualize`, `fsm-llm-monitor`, `fsm-llm-meta`) and the legacy `fsm-llm --fsm <path.json>` flag still work and route through the same entry point.

## Examples — 152 across 10 trees

**Category A (FSM, read-only baselines)** — 95 examples in `basic/` (14), `intermediate/` (3), `advanced/` (17), `classification/` (4), `agents/` (48), `reasoning/` (1), `workflows/` (8).

**Category B/C + meta** — 57 examples:
- `pipeline/` (47) — λ-DSL twins of `agents/*` (M4). 3-file shape: `__init__.py`, `schemas.py`, `run.py` with inline λ-term.
- `long_context/` (5) — `niah_demo`, `niah_padded_demo`, `aggregate_demo`, `pairwise_demo`, `multi_hop_demo` (M5).
- `meta/` (5) — `build_fsm`, `build_workflow`, `build_agent`, `meta_review_loop`, `meta_from_spec`.

Run with: `python examples/<tree>/<name>/run.py`. See `EVALUATE.md` for evaluation methodology and recent scorecards.

## Development

```bash
make install-dev    # Install in dev mode with all extras + pre-commit hooks
make test           # Run full test suite (currently 2,899 tests)
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make type-check     # mypy across all packages
make build          # python -m build (wheel + sdist)
make coverage       # Tests with coverage report
```

## Documentation

- [`docs/lambda.md`](docs/lambda.md) — **Architectural thesis**: λ-substrate, FSM as surface, Theorem 1-5.
- [Quick Start Guide](docs/quickstart.md)
- [API Reference](docs/api_reference.md)
- [Architecture](docs/architecture.md) — execution model, security, performance
- [FSM Design Patterns](docs/fsm_design.md)
- [Handler Development](docs/handlers.md)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. `make install-dev` to set up
4. Make changes with tests
5. Ensure `make lint` and `make test` pass
6. Submit a pull request

## License

GNU General Public License v3.0 or later. See [LICENSE](LICENSE).
