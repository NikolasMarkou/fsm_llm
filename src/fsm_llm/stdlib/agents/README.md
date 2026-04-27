# fsm_llm.stdlib.agents — Agentic Patterns

> Agentic patterns on a typed λ-runtime: four canonical λ-term factories (M3 slice 1) **and** 12+ class-based patterns with tool use, HITL, structured output, multi-agent coordination, MCP/A2A, SOPs, semantic tools, and a meta-builder.

---

## Two Layers, One Subpackage

`fsm_llm.stdlib.agents` exposes two ways to build agents:

1. **λ-term factories** (M3 slice 1) — `react_term`, `rewoo_term`, `reflexion_term`, `memory_term`. Pure factories returning `Term`. Use when composing into larger λ-programs or when you want planner-bounded oracle costs.
2. **Class-based patterns** — `ReactAgent`, `REWOOAgent`, etc. Use when you need lifecycle, budgets, HITL, swarm coordination, or interactive CLIs out of the box.

Both layers coexist; neither is deprecated. The legacy `fsm_llm_agents` import path resolves here via `sys.modules` shim.

## Installation

```bash
pip install fsm-llm[agents]
```

**Requirements**: Python 3.10+ | No additional dependencies beyond core `fsm-llm`.

## Quick Start

### A — λ-term factory (Category-B style)

```python
from fsm_llm.lam import Executor, LiteLLMOracle
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.stdlib.agents import react_term
from pydantic import BaseModel

class ToolDecision(BaseModel):
    tool: str
    args: dict

def tool_runner(decision: ToolDecision) -> str:
    if decision.tool == "search": return "Population of France: 67M"
    return f"Unknown tool: {decision.tool}"

term = react_term(
    decide_prompt="Decide a tool call for: {question}",
    synth_prompt="Use the observation to answer: {observation}",
    decision_schema=ToolDecision,
)
ex = Executor(oracle=LiteLLMOracle(LiteLLMInterface(model="openai/gpt-4o-mini")))
print(ex.run(term, env={"question": "What is 2× the population of France?",
                        "tool_dispatch": tool_runner}))
assert ex.oracle_calls == 2   # Strict equality (let-chain shape)
```

### B — Class-based agent (interactive style)

```python
from fsm_llm.stdlib.agents import ReactAgent, tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

agent = ReactAgent(model="openai/gpt-4o-mini", tools=[search], max_iterations=10)
result = agent.run("What is the population of France times 2?")
print(result.answer)
```

The factory `create_agent("react", model=..., tools=...)` returns the same shape:

```python
from fsm_llm.stdlib.agents import create_agent
agent = create_agent("react", model="openai/gpt-4o-mini", tools=[search])
result = agent("Your task here")  # callable shorthand
```

## λ-term Factories (M3 slice 1)

| Factory | Oracle calls | Body sketch |
|---------|-------------|-------------|
| `react_term` | 2 | `let_("decision", decide_leaf, let_("observation", app(var("tool_dispatch"), var("decision")), synth_leaf))` |
| `rewoo_term` | 2 | `let_("plan", plan_leaf, let_("evidence", app(var("plan_exec"), var("plan")), synth_leaf))` |
| `reflexion_term` | 4 | `let_("attempt1", solve, let_("evaluation", evaluate, let_("reflection", reflect, re_solve)))` (depth-1 retry flatten) |
| `memory_term` | 2 | `let_("context", ctx_leaf, ans_leaf)` |

Caller binds host callables (`tool_dispatch`, `plan_exec`) in env. See `examples/pipeline/_helpers.py` for runtime glue (`run_pipeline`, `make_tool_dispatcher`, `make_plan_executor`) and `examples/pipeline/` for 47 working M4 references.

## Class-based Patterns

| Pattern | Class | Description |
|---------|-------|-------------|
| **ReAct** | `ReactAgent` | Reasoning + Acting loop with tool dispatch |
| **REWOO** | `REWOOAgent` | Planning-first: plan all tool calls, then execute |
| **Reflexion** | `ReflexionAgent` | Self-reflection with memory of past attempts |
| **Plan-Execute** | `PlanExecuteAgent` | Decompose task into plan, execute steps sequentially |
| **Prompt Chain** | `PromptChainAgent` | Sequential prompt pipeline with quality gates |
| **Self-Consistency** | `SelfConsistencyAgent` | Multiple samples with majority voting |
| **Debate** | `DebateAgent` | Multi-perspective debate with judge synthesis |
| **Orchestrator** | `OrchestratorAgent` | Delegate subtasks to worker agents |
| **ADaPT** | `ADaPTAgent` | Adaptive complexity with task decomposition |
| **Eval-Optimize** | `EvaluatorOptimizerAgent` | Iterative evaluation and optimization loop |
| **Maker-Checker** | `MakerCheckerAgent` | Draft-review verification pattern |
| **Reasoning-ReAct** | `ReasoningReactAgent` | ReAct with structured reasoning (requires `reasoning` extra) |

### Pattern selection guide

| Task type | Recommended pattern |
|-----------|--------------------|
| Tool use + reasoning | ReAct |
| Known tool sequence | REWOO |
| Multi-step with learning | Reflexion |
| Complex decomposable tasks | Plan-Execute |
| Sequential data transformation | Prompt Chain |
| High-stakes decisions | Self-Consistency or Debate |
| Multi-agent delegation | Orchestrator |
| Variable complexity | ADaPT |
| Quality-critical output | Eval-Optimize or Maker-Checker |

## Tool System

### `@tool` decorator

```python
from fsm_llm.stdlib.agents import tool

@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city.

    Args:
        city: City name to look up
        units: Temperature units (celsius or fahrenheit)
    """
    return f"Weather in {city}: 22{units[0].upper()}"
```

Auto-generates JSON schema from type hints and docstrings.

### `ToolRegistry`

```python
from fsm_llm.stdlib.agents import ToolRegistry
registry = ToolRegistry()
registry.register(get_weather)
result = registry.execute("get_weather", {"city": "Paris"})
```

### Agents as tools

```python
from fsm_llm.stdlib.agents import register_agent
researcher = ReactAgent(model="openai/gpt-4o-mini", tools=[search])
register_agent(registry, researcher, name="research", description="Deep research")
```

## Human-in-the-Loop

```python
from fsm_llm.stdlib.agents import HumanInTheLoop

hitl = HumanInTheLoop(
    approval_callback=lambda req: input(f"Approve {req.tool_name}? (y/n): ") == "y",
    require_approval_for=["send_email", "delete_file"],
    confidence_threshold=0.8,
    timeout=300,
)
agent = ReactAgent(model="openai/gpt-4o-mini", tools=[send_email], hitl=hitl)
```

## Structured Output

```python
from pydantic import BaseModel
from fsm_llm.stdlib.agents import ReactAgent, AgentConfig

class Analysis(BaseModel):
    summary: str
    sentiment: str
    confidence: float

agent = ReactAgent(model="openai/gpt-4o-mini", tools=[search],
                   config=AgentConfig(output_schema=Analysis))
result = agent.run("Analyze sentiment of recent Tesla news")
print(result.structured_output.summary)
```

## Working Memory & Skills

```python
from fsm_llm.stdlib.agents import ReactAgent, create_memory_tools, SkillLoader
from fsm_llm import WorkingMemory

memory_tools = create_memory_tools(WorkingMemory())
agent = ReactAgent(model="openai/gpt-4o-mini", tools=[search, *memory_tools])

skills = SkillLoader().load_directory("./skills/")
tools = [skill.to_tool() for skill in skills]
```

## Multi-Agent Coordination

### Swarm (dynamic handoffs)

```python
from fsm_llm.stdlib.agents import SwarmAgent

swarm = SwarmAgent(
    agents={"triage": triage_agent, "billing": billing_agent, "support": support_agent},
    entry_agent="triage",
    max_handoffs=5,
)
result = swarm.run("I need help with my bill")
```

Agents hand off by setting `next_agent` and `handoff_message` in `final_context`.

### Agent Graph (DAG orchestration)

```python
from fsm_llm.stdlib.agents import AgentGraphBuilder

graph = (
    AgentGraphBuilder()
    .add_node("classifier", classifier_agent)
    .add_node("billing", billing_agent)
    .add_node("support", support_agent)
    .add_edge("classifier", "billing", condition=lambda ctx: ctx.get("intent") == "billing")
    .add_edge("classifier", "support", condition=lambda ctx: ctx.get("intent") == "support")
    .set_entry("classifier")
    .build()
)
result = graph.run("I need help with my invoice")
```

### MCP tool integration

```python
from fsm_llm.stdlib.agents import MCPToolProvider, ToolRegistry

provider = MCPToolProvider.from_stdio("npx", ["-y", "@modelcontextprotocol/server-everything"])
tools = await provider.discover_tools()
registry = ToolRegistry()
provider.register_tools(registry)
```

Requires: `pip install fsm-llm[mcp]`

### A2A remote agents

```python
from fsm_llm.stdlib.agents import AgentServer, RemoteAgentTool

server = AgentServer(agent=my_agent, port=8500)
server.run()

remote = RemoteAgentTool(url="http://localhost:8500", name="remote_agent", description="Remote helper")
registry.register(remote.to_tool_definition())
```

Requires: `pip install fsm-llm[a2a]`

### SOPs (Standard Operating Procedures)

```python
from fsm_llm.stdlib.agents import SOPRegistry, load_builtin_sops

registry = load_builtin_sops()  # code-review, summarize, data-extraction
sop = registry.get("code-review")
task = sop.render_task(code="def foo(): pass", language="python")
```

## Meta-Builder Agent

Interactively builds FSMs, workflows, and agents:

```python
from fsm_llm.stdlib.agents import MetaBuilderAgent
builder = MetaBuilderAgent(model="openai/gpt-4o-mini")
result = builder.run("Build an FSM for a pizza ordering chatbot")
```

Or via CLI: `fsm-llm-meta`

## Key API Reference

### `BaseAgent` (shared by all class patterns)

```python
agent = ReactAgent(
    model="openai/gpt-4o-mini", tools=[...], system_prompt="...",
    max_iterations=10, timeout=300, hitl=hitl, config=AgentConfig(...),
)
result = agent.run("task")     # or agent("task")

result.answer              # str — final answer
result.success             # bool — completed successfully
result.trace               # AgentTrace — execution trace
result.structured_output   # Pydantic model (if output_schema set)
```

## Exception Hierarchy

```
FSMError
└── AgentError
    ├── ToolExecutionError
    ├── ToolNotFoundError
    ├── ToolValidationError
    ├── BudgetExhaustedError
    ├── ApprovalDeniedError
    ├── AgentTimeoutError
    ├── EvaluationError
    ├── DecompositionError
    └── MetaBuilderError
        ├── BuilderError
        ├── MetaValidationError
        └── OutputError
```

## License

GPL-3.0-or-later. See [LICENSE](../../../../LICENSE).
