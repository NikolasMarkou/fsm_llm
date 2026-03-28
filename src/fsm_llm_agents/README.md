# FSM-LLM Agents

> 12+ agentic patterns with tool use, human-in-the-loop, structured output, and a unified interface.

---

## Overview

`fsm_llm_agents` brings agentic AI patterns to FSM-LLM. Each agent pattern is implemented as an auto-generated FSM, giving you the reliability of state machines with the flexibility of LLM-driven tool use and reasoning.

Key capabilities:
- **12+ agent patterns** from simple ReAct to multi-agent orchestration
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

**1. Create a ReAct agent with tools**:

```python
from fsm_llm_agents import ReactAgent, tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> float:
    """Evaluate a math expression."""
    return eval(expression)

agent = ReactAgent(
    model="gpt-4o-mini",
    tools=[search, calculate],
    system_prompt="You are a helpful research assistant.",
    max_iterations=10,
)

result = agent.run("What is the population of France times 2?")
print(result.answer)
print(f"Steps taken: {len(result.trace.steps)}")
```

**2. Use the factory for any pattern**:

```python
from fsm_llm_agents import create_agent

agent = create_agent(
    "react",
    model="gpt-4o-mini",
    tools=[search, calculate],
)
result = agent.run("Your task here")
```

**3. Callable interface**:

```python
# Agents are callable — same as .run()
result = agent("Your task here")
```

**4. CLI info**:

```bash
python -m fsm_llm_agents --info
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
| **Evaluator-Optimizer** | `EvaluatorOptimizerAgent` | Iterative evaluation and optimization loop |
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
| Quality-critical output | Evaluator-Optimizer or Maker-Checker |

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

The `@tool` decorator automatically generates a JSON schema from type hints and docstrings.

### ToolRegistry

```python
from fsm_llm_agents import ToolRegistry

registry = ToolRegistry()
registry.register(get_weather)
registry.register(search)

# List tools
for t in registry.list_tools():
    print(f"{t.name}: {t.description}")

# Execute by name
result = registry.execute("get_weather", {"city": "Paris"})
```

### Register Another Agent as a Tool

```python
from fsm_llm_agents import register_agent

researcher = ReactAgent(model="gpt-4o-mini", tools=[search])
register_agent(registry, researcher, name="research", description="Deep research")
```

## Human-in-the-Loop

```python
from fsm_llm_agents import ReactAgent, HumanInTheLoop

def approval_callback(request):
    """Called when agent needs human approval."""
    print(f"Agent wants to call: {request.tool_name}({request.arguments})")
    return input("Approve? (y/n): ").lower() == "y"

hitl = HumanInTheLoop(
    approval_callback=approval_callback,
    require_approval_for=["send_email", "delete_file"],
    confidence_threshold=0.8,  # auto-approve above this
    timeout=300,               # 5 min approval timeout
)

agent = ReactAgent(
    model="gpt-4o-mini",
    tools=[send_email, delete_file, search],
    hitl=hitl,
)
```

## Structured Output

```python
from pydantic import BaseModel
from fsm_llm_agents import ReactAgent, AgentConfig

class Analysis(BaseModel):
    summary: str
    sentiment: str
    confidence: float

agent = ReactAgent(
    model="gpt-4o-mini",
    tools=[search],
    config=AgentConfig(output_schema=Analysis),
)

result = agent.run("Analyze the sentiment of recent Tesla news")
analysis = result.structured_output  # Analysis instance
print(analysis.summary, analysis.sentiment, analysis.confidence)
```

## Working Memory Tools

```python
from fsm_llm_agents import ReactAgent, create_memory_tools
from fsm_llm import WorkingMemory

memory = WorkingMemory()
memory_tools = create_memory_tools(memory)

agent = ReactAgent(
    model="gpt-4o-mini",
    tools=[search, *memory_tools],  # adds remember, recall, forget, list_memories
)
```

## Skills

```python
from fsm_llm_agents import SkillLoader

# Load skills from a directory of Python files
loader = SkillLoader()
skills = loader.load_directory("./skills/")

# Convert to tools for agent use
tools = [skill.to_tool() for skill in skills]
agent = ReactAgent(model="gpt-4o-mini", tools=tools)
```

## Key API Reference

### BaseAgent (shared by all patterns)

```python
# Constructor (common parameters)
agent = ReactAgent(
    model="gpt-4o-mini",          # LLM model
    tools=[...],                   # Tool list
    system_prompt="...",           # System prompt
    max_iterations=10,             # Iteration budget (hard limit = 1.5x)
    timeout=300,                   # Timeout in seconds
    hitl=hitl,                     # Human-in-the-loop config
    config=AgentConfig(...),       # Advanced config (output_schema, etc.)
)

# Run
result = agent.run("task description")
result = agent("task description")  # callable shorthand

# Result
result.answer       # str — final answer
result.success      # bool — completed successfully
result.trace        # AgentTrace — execution trace
result.structured_output  # Pydantic model (if output_schema set)
```

### AgentResult

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `str` | Final answer text |
| `success` | `bool` | Whether task completed |
| `trace` | `AgentTrace` | Full execution trace |
| `structured_output` | `BaseModel \| None` | Validated output (if schema set) |

### AgentTrace

| Field | Type | Description |
|-------|------|-------------|
| `steps` | `list[AgentStep]` | All execution steps |
| `tool_calls` | `list[ToolCall]` | All tool invocations |
| `total_iterations` | `int` | Loop count |
| `execution_time` | `float` | Total seconds |

## Meta-Builder Agent

The `MetaBuilderAgent` interactively builds FSMs, workflows, and agents:

```python
from fsm_llm_agents import MetaBuilderAgent

builder = MetaBuilderAgent(model="gpt-4o-mini")
result = builder.run("Build an FSM for a pizza ordering chatbot")
print(result.answer)  # JSON FSM definition
```

Or use the CLI:

```bash
fsm-llm-meta
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

GPL-3.0-or-later. See [LICENSE](../../LICENSE) for details.
