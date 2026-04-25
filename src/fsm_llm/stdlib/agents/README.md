# FSM-LLM Agents

> 12+ agentic patterns with tool use, human-in-the-loop, structured output, and a unified interface.

---

## Overview

`fsm_llm_agents` brings agentic AI patterns to FSM-LLM. Each agent pattern is implemented as an auto-generated FSM, giving you the reliability of state machines with the flexibility of LLM-driven tool use and reasoning.

Key capabilities:
- **12+ agent patterns** from simple ReAct to multi-agent orchestration
- **Swarm coordination** -- agents hand off to each other dynamically
- **Agent Graph** -- DAG-based orchestration with conditional edges
- **MCP integration** -- connect MCP servers for tool discovery (`pip install fsm-llm[mcp]`)
- **A2A protocol** -- expose agents as HTTP endpoints, call remote agents as tools (`pip install fsm-llm[a2a]`)
- **Semantic tool retrieval** -- embedding-based tool selection for large tool registries
- **SOPs** -- reusable agent configurations from YAML/JSON templates
- **Tool system** with `@tool` decorator and auto-schema from type hints
- **Human-in-the-loop** approval gates, escalation, and confidence thresholds
- **Structured output** via Pydantic model validation
- **Budget enforcement** with iteration limits and timeouts
- **Working memory** tools (remember, recall, forget, list)
- **Skill loading** from directories with auto-discovery
- **Meta-builder** agent that builds FSMs, workflows, and agents interactively

## Installation

```bash
pip install fsm-llm[agents]
```

**Requirements**: Python 3.10+ | No additional dependencies beyond core `fsm-llm`.

## Quick Start

```python
from fsm_llm_agents import ReactAgent, tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

agent = ReactAgent(model="gpt-4o-mini", tools=[search], max_iterations=10)
result = agent.run("What is the population of France times 2?")
print(result.answer)
```

Use the factory for any pattern:

```python
from fsm_llm_agents import create_agent
agent = create_agent("react", model="gpt-4o-mini", tools=[search])
result = agent("Your task here")  # callable shorthand
```

## Agent Patterns

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
| **Reasoning-ReAct** | `ReasoningReactAgent` | ReAct with structured reasoning (requires `fsm_llm_reasoning`) |

### Pattern Selection Guide

| Task Type | Recommended Pattern |
|-----------|-------------------|
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

### @tool Decorator

```python
from fsm_llm_agents import tool

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

### ToolRegistry

```python
from fsm_llm_agents import ToolRegistry
registry = ToolRegistry()
registry.register(get_weather)
result = registry.execute("get_weather", {"city": "Paris"})
```

### Agents as Tools

```python
from fsm_llm_agents import register_agent
researcher = ReactAgent(model="gpt-4o-mini", tools=[search])
register_agent(registry, researcher, name="research", description="Deep research")
```

## Human-in-the-Loop

```python
from fsm_llm_agents import HumanInTheLoop

hitl = HumanInTheLoop(
    approval_callback=lambda req: input(f"Approve {req.tool_name}? (y/n): ") == "y",
    require_approval_for=["send_email", "delete_file"],
    confidence_threshold=0.8,
    timeout=300,
)
agent = ReactAgent(model="gpt-4o-mini", tools=[send_email], hitl=hitl)
```

## Structured Output

```python
from pydantic import BaseModel
from fsm_llm_agents import ReactAgent, AgentConfig

class Analysis(BaseModel):
    summary: str
    sentiment: str
    confidence: float

agent = ReactAgent(model="gpt-4o-mini", tools=[search],
                   config=AgentConfig(output_schema=Analysis))
result = agent.run("Analyze sentiment of recent Tesla news")
print(result.structured_output.summary)
```

## Working Memory & Skills

```python
from fsm_llm_agents import ReactAgent, create_memory_tools, SkillLoader
from fsm_llm import WorkingMemory

# Memory tools
memory_tools = create_memory_tools(WorkingMemory())
agent = ReactAgent(model="gpt-4o-mini", tools=[search, *memory_tools])

# Load skills from directory
skills = SkillLoader().load_directory("./skills/")
tools = [skill.to_tool() for skill in skills]
```

## Key API Reference

### BaseAgent (shared by all patterns)

```python
agent = ReactAgent(
    model="gpt-4o-mini", tools=[...], system_prompt="...",
    max_iterations=10, timeout=300, hitl=hitl, config=AgentConfig(...)
)
result = agent.run("task")  # or agent("task")

result.answer             # str — final answer
result.success            # bool — completed successfully
result.trace              # AgentTrace — execution trace
result.structured_output  # Pydantic model (if output_schema set)
```

## Multi-Agent Coordination

### Swarm (Dynamic Handoffs)

```python
from fsm_llm_agents import SwarmAgent

swarm = SwarmAgent(
    agents={"triage": triage_agent, "billing": billing_agent, "support": support_agent},
    entry_agent="triage",
    max_handoffs=5,
)
result = swarm.run("I need help with my bill")
```

Agents hand off by setting `next_agent` and `handoff_message` in their `final_context`.

### Agent Graph (DAG Orchestration)

```python
from fsm_llm_agents import AgentGraphBuilder

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

### MCP Tool Integration

```python
from fsm_llm_agents import MCPToolProvider, ToolRegistry

provider = MCPToolProvider.from_stdio("npx", ["-y", "@modelcontextprotocol/server-everything"])
tools = await provider.discover_tools()
registry = ToolRegistry()
provider.register_tools(registry)
```

Requires: `pip install fsm-llm[mcp]`

### A2A Remote Agents

```python
from fsm_llm_agents import AgentServer, RemoteAgentTool

# Serve an agent over HTTP
server = AgentServer(agent=my_agent, port=8500)
server.run()

# Call remote agent as a tool
remote = RemoteAgentTool(url="http://localhost:8500", name="remote_agent", description="Remote helper")
registry.register(remote.to_tool_definition())
```

Requires: `pip install fsm-llm[a2a]`

### SOPs (Standard Operating Procedures)

```python
from fsm_llm_agents import SOPRegistry, load_builtin_sops

registry = load_builtin_sops()  # code-review, summarize, data-extraction
sop = registry.get("code-review")
task = sop.render_task(code="def foo(): pass", language="python")
```

## Meta-Builder Agent

Interactively builds FSMs, workflows, and agents:

```python
from fsm_llm_agents import MetaBuilderAgent
builder = MetaBuilderAgent(model="gpt-4o-mini")
result = builder.run("Build an FSM for a pizza ordering chatbot")
```

Or use the CLI: `fsm-llm-meta`

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

GPL-3.0-or-later. See [LICENSE](../../LICENSE) for details.
