# FSM-LLM: One Runtime, Two Surfaces

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![PyPI version](https://badge.fury.io/py/fsm-llm.svg)](https://badge.fury.io/py/fsm-llm)
[![Tests](https://github.com/NikolasMarkou/fsm_llm/actions/workflows/python-package.yml/badge.svg)](https://github.com/NikolasMarkou/fsm_llm/actions)

<p align="center">
  <img src="./images/fsm-llm-logo-1.png" alt="FSM-LLM Logo" width="500"/>
</p>

**A Python framework for building stateful LLM programs on a typed λ-calculus runtime.** Two surface syntaxes share one executor: FSM JSON for dialog programs (Category A); a λ-DSL for pipelines, agents, and long-context recursion (Category B/C).

---

## Why FSM-LLM?

LLMs are stateless. Real programs are not. fsm-llm gives you **one runtime that compiles both shapes**:

- **FSMs (Category A)** when you need persistent per-turn dialog state with non-linear transitions.
- **λ-DSL (Category B)** when you have a stateless pipeline (extract → reason → respond, ReAct, REWOO, Reflexion, debate, plan-execute, …).
- **λ-DSL (Category C)** when you need bounded recursion over long inputs with closed-form cost (NIAH, aggregate, pairwise, multi-hop).

The substrate is a typed λ-AST (`src/fsm_llm/lam/`). FSM JSON is compiled to a λ-term at load time and executed by the same engine that runs hand-written λ-programs. There is no second runtime — and per `docs/lambda.md` Theorem 2, the planner pre-computes oracle-call cost in closed form for every `Fix` node.

## Key Features

- **Typed λ-kernel** (M1) — `Var · Abs · App · Let · Case · Combinator · Fix · Leaf` AST + Executor + Planner + Oracle. Pure (no FSM imports).
- **FSM → λ compiler** (M2) — every existing FSM JSON program runs unchanged on the new substrate. Single-path runtime; legacy `MessagePipeline.process` retired.
- **Stdlib factory layers** (M3) — named λ-term factories for agents (`react_term`, `rewoo_term`, `reflexion_term`, `memory_term`), reasoning (11 strategies), workflows (linear / branch / switch / parallel / retry), and long-context (`niah`, `aggregate`, `pairwise`, `multi_hop`, `niah_padded`).
- **Theorem-2 cost equality** — every long-context demo and bench cell ships with a hard `oracle_calls == plan(...).predicted_calls` gate. See `evaluation/bench_long_context_*.json`.
- **OOLONG benchmark loader** (M5 slice 7) — HuggingFace streaming-mode ingestion of the real OOLONG benchmark (arXiv 2511.02817) via `pip install fsm-llm[oolong]`.
- **2-pass FSM body** — Pass 1 extracts data + evaluates transitions; Pass 2 generates response from the post-transition state. Compiled into the body of every Category-A λ-term.
- **Handler system** — 8 hook points (`PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `START_CONVERSATION`, `END_CONVERSATION`, `ERROR`) with a fluent builder API. Hooks compose into the compiled λ-term.
- **JsonLogic transitions** — Deterministic rule-based transitions (`==`, `in`, `has_context`, `and`, `or`, …) at the `Case` node.
- **FSM stacking** — `push_fsm()` / `pop_fsm()` with context-merge strategies for nested dialog flows.
- **100+ LLM providers** via litellm (OpenAI, Anthropic, Ollama, Azure, Bedrock, …).
- **Stdlib subpackages** — reasoning, workflows, agents (12 patterns + swarm, agent graph, MCP, A2A, SOPs, semantic tools, meta builder), monitor (web dashboard + OTEL exporter).
- **Security built in** — Internal context-key prefixes, forbidden patterns for secrets/tokens, XML tag sanitisation in prompts.

## Installation

```bash
pip install fsm-llm
```

With all extras:

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

## Quick Start

### 1A. λ-DSL — a 5-line Category-B pipeline

A linear extract → answer chain. No FSM ceremony, no JSON file:

```python
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.lam import Executor, LiteLLMOracle, leaf, let_, var
from pydantic import BaseModel

class Topic(BaseModel): topic: str

term = let_(
    "topic", leaf(prompt="Extract the topic in one word: {q}", schema=Topic, input_var="q"),
    leaf(prompt="Write a one-paragraph article about {topic}.", input_var="topic"),
)
ex = Executor(oracle=LiteLLMOracle(LiteLLMInterface(model="openai/gpt-4o-mini")))
print(ex.run(term, env={"q": "What is photosynthesis?"}))
# 2 oracle calls, deterministic shape, planner-bounded.
```

### 1B. λ-DSL — a Category-C long-context program

Bounded recursion over a 4 KB document; closed-form cost; Theorem-2 verified:

```python
from fsm_llm.lam import Executor, LiteLLMOracle, plan, PlanInputs
from fsm_llm.stdlib.long_context import niah
from fsm_llm.llm import LiteLLMInterface

term = niah(question="Where is the needle hidden?", tau=256, k=2)
ex = Executor(oracle=LiteLLMOracle(LiteLLMInterface(model="ollama_chat/qwen3.5:4b")))
ex.run(term, env={"document": long_doc_4k_chars})

predicted = plan(PlanInputs(n=4096, tau=256, k=2)).predicted_calls
assert ex.oracle_calls == predicted   # Theorem-2 holds: 16 leaf calls, k^d = 2^4
```

### 2. FSM JSON — a Category-A dialog (`greeting.json`)

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

```python
from fsm_llm import API

api = API.from_file("greeting.json", model="openai/gpt-4o-mini")
conversation_id, initial_response = api.start_conversation()
print(api.converse("Hi! I'm Alice.", conversation_id))
print(api.converse("Goodbye!", conversation_id))
```

The FSM is compiled to a λ-term at `from_file()` time; every `converse()` is one β-reduction step on the same executor that runs the λ-DSL examples above.

### 3. Or use the CLI

```bash
export OPENAI_API_KEY="your-key-here"
fsm-llm --fsm greeting.json
```

## Architecture

```
   FSM JSON  (Category A)              λ-DSL  (Category B / C)
        │                                       │
        ▼  fsm_llm.lam.fsm_compile              ▼  fsm_llm.lam.dsl
   ┌──────────────────────────────────────────────────────┐
   │                λ-AST  (typed Term)                   │
   └──────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────┐
   │ Executor + Planner + Oracle + Session + CostAccum.   │
   └──────────────────────────────────────────────────────┘
```

For Category A: per-turn `step : (state, input, context) → (state', output, context')` is a λ-term with a top-level `Case` on `state_id`. Pass 1 / transition / Pass 2 are three `Leaf` nodes (oracle calls) with a pure `Case` between them. For Category B/C: write the λ-term directly; the planner gives you predicted cost, the executor delivers it.

See `docs/lambda.md` for the full thesis and Theorem 1-5 statements.

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

### Long-context — Category-C native

```python
from fsm_llm.stdlib.long_context import niah, aggregate, pairwise, multi_hop, niah_padded
```

Each factory returns a λ-term whose oracle cost is closed-form per `plan(...)`. See `examples/long_context/*_demo/` for runnable demos with hard Theorem-2 gates.

### Monitor — web dashboard + OTEL

```bash
fsm-llm-monitor   # Opens at http://localhost:8420
```

Dashboard, Control Center, Visualizer, Conversations, Logs, Builder, Settings. OTEL span exporter ships λ-AST node spans (per-`Fix` / per-`Leaf` / per-`Combinator`).

### Meta builder

```bash
fsm-llm-meta   # Interactive CLI for building FSMs, workflows, agents
```

## Sibling Packages — Back-Compat Shims

`fsm_llm_reasoning`, `fsm_llm_workflows`, and `fsm_llm_agents` are silent `sys.modules` shims that resolve to `fsm_llm.stdlib.<pkg>`. Existing imports keep working:

```python
from fsm_llm_agents import ReactAgent       # Still works
from fsm_llm.stdlib.agents import ReactAgent  # Same object — preferred for new code
```

## CLI Tools

| Command | Description |
|---------|-------------|
| `fsm-llm --fsm <path.json>` | Run an FSM interactively (compiled λ-path) |
| `fsm-llm-visualize --fsm <path.json>` | ASCII visualization |
| `fsm-llm-validate --fsm <path.json>` | Validate FSM definition |
| `fsm-llm-monitor` | Launch web monitoring dashboard |
| `fsm-llm-meta` | Interactive artifact builder |

## Examples — 152 across 10 trees

**Category A (FSM, read-only baselines)** — 95 examples in `basic/` (14), `intermediate/` (3), `advanced/` (17), `classification/` (4), `agents/` (48), `reasoning/` (1), `workflows/` (8).

**Category B/C + meta** — 57 examples:
- `pipeline/` (47) — λ-DSL twins of `agents/*` (M4). 3-file shape: `__init__.py`, `schemas.py`, `run.py` with inline λ-term.
- `long_context/` (5) — `niah_demo`, `niah_padded_demo`, `aggregate_demo`, `pairwise_demo`, `multi_hop_demo` (M5).
- `meta/` (5) — `build_fsm`, `build_workflow`, `build_agent`, `meta_review_loop`, `meta_from_spec`.

Run with: `python examples/<tree>/<name>/run.py`. See `EVALUATE.md` for evaluation results (last published: 90.8% health on `ollama_chat/qwen3.5:4b`, Run 004 — fresh run pending).

## Development

```bash
make install-dev    # Install in dev mode with all extras + pre-commit hooks
make test           # Run full test suite (currently 2,728 tests)
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
