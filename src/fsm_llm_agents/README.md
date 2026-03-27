# fsm_llm_agents

Agentic patterns for FSM-LLM. Provides 12 agent architectures -- from basic ReAct tool-use loops to multi-agent debate and adaptive decomposition -- all built on FSM-LLM's 2-pass state machine engine. Includes a tool registry with `@tool` decorator and Human-in-the-Loop (HITL) approval gates.

Part of [FSM-LLM](https://github.com/NikolasMarkou/fsm_llm) v0.3.0. License: GPL-3.0-or-later.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Agent Patterns](#agent-patterns)
- [Pattern Selection Guide](#pattern-selection-guide)
- [Tool System](#tool-system)
- [Human-in-the-Loop](#human-in-the-loop)
- [API Reference](#api-reference)
- [File Map](#file-map)
- [Examples](#examples)
- [Integration](#integration)
- [Development](#development)

---

## Features

- **12 agent patterns** covering tool use, planning, reflection, debate, and more
- **BaseAgent ABC** -- all agents share a common conversation loop, budget enforcement, `__call__` syntax, and structured output
- **`@tool` decorator** with auto-schema inference from type hints and `typing.Annotated` descriptions
- **Structured output** -- `output_schema=PydanticModel` validates agent answers against typed schemas
- **`create_agent()`** factory -- create agents in one line from tools and pattern name
- **Agents-as-tools** -- `ToolRegistry.register_agent()` enables supervisor/orchestrator patterns
- **ToolRegistry** with parameter validation, prompt generation, and classification schema generation
- **Human-in-the-Loop** with approval policies, confidence-based escalation, and timeout
- **Auto-generated FSMs** -- agent patterns build their own FSM definitions at runtime from tool registries and configuration; no JSON authoring required
- **Budget and timeout control** -- max iterations, wall-clock timeout, per-pattern limits (reflections, revisions, debate rounds, decomposition depth)
- **Full tracing** -- every agent run produces an `AgentTrace` with tool calls, iterations, and timing

## Installation

```bash
pip install fsm-llm[agents]
```

Or for development:

```bash
pip install -e ".[dev,agents]"
```

## Quick Start

### Simplest (1-line agent creation)

```python
from fsm_llm_agents import create_agent, tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

agent = create_agent(tools=[search])
result = agent("What is the capital of France?")
print(result)  # "The capital of France is Paris."
```

### Full control

```python
from fsm_llm_agents import ReactAgent, ToolRegistry, AgentConfig, tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

registry = ToolRegistry()
registry.register(search._tool_definition)

agent = ReactAgent(
    tools=registry,
    config=AgentConfig(model="gpt-4o-mini", max_iterations=5),
)
result = agent.run("What is the capital of France?")

print(result.answer)           # "The capital of France is Paris."
print(result.success)          # True
print(result.tools_used)       # ["search"]
print(result.iterations_used)  # 3
```

### Structured output

```python
from pydantic import BaseModel
from fsm_llm_agents import create_agent, tool, AgentConfig

class CityInfo(BaseModel):
    city: str
    country: str
    population: int

@tool
def lookup(query: str) -> str:
    """Look up city information."""
    return '{"city": "Paris", "country": "France", "population": 2161000}'

agent = create_agent(
    tools=[lookup],
    config=AgentConfig(output_schema=CityInfo),
)
result = agent("Tell me about Paris")
print(result.structured_output)  # CityInfo(city='Paris', country='France', population=2161000)
```

---

## Agent Patterns

| Pattern | Strategy | Best For |
|---------|----------|----------|
| [ReactAgent](#1-reactagent) | Think-act-observe loop | General tool use, Q&A with tools |
| [REWOOAgent](#2-rewooagent) | Plan all tools upfront, execute, synthesize | Minimizing LLM calls (exactly 2) |
| [PlanExecuteAgent](#3-planexecuteagent) | Create plan, execute steps, replan on failure | Multi-step structured work |
| [ReflexionAgent](#4-reflexionagent) | ReAct + self-evaluation + episodic memory | Tasks requiring iterative improvement |
| [PromptChainAgent](#5-promptchainagent) | Sequential prompt pipeline with gates | Fixed multi-stage processing |
| [SelfConsistencyAgent](#6-selfconsistencyagent) | N samples + majority vote | Factual questions, reducing variance |
| [DebateAgent](#7-debateagent) | Proposer-critic-judge debate rounds | Nuanced analysis, opinion questions |
| [OrchestratorAgent](#8-orchestratoragent) | Decompose, delegate to workers, synthesize | Parallelizable complex tasks |
| [ADaPTAgent](#9-adaptagent) | Attempt first, decompose recursively on failure | Tasks with unknown complexity |
| [EvaluatorOptimizerAgent](#10-evaluatoroptimizeragent) | Generate-evaluate-refine loop | Code generation, constrained output |
| [MakerCheckerAgent](#11-makercheckeragent) | Draft-review verification cycle | Content quality assurance |
| [ReasoningReactAgent](#12-reasoningreactagent) | ReAct + structured reasoning (FSM stacking) | Math, logic, analytical problems |

### 1. ReactAgent

The classic ReAct (Reasoning + Acting) loop. The agent thinks about what to do, acts by calling a tool, observes the result, and repeats until it can conclude.

**FSM flow:** `think -> act -> think -> ... -> conclude`

```python
ReactAgent(
    tools: ToolRegistry,                      # Required -- registered tools
    config: AgentConfig | None = None,        # Model, iterations, timeout
    hitl: HumanInTheLoop | None = None,       # Optional approval gates
    **api_kwargs: Any,                        # Passed to fsm_llm.API
)
```

**When to use:** General-purpose tool-use. Start here if unsure which pattern to pick.

**When to avoid:** If you know the exact tool sequence upfront (use REWOOAgent instead) or if you need self-improvement (use ReflexionAgent).

---

### 2. REWOOAgent

REWOO (Reasoning Without Observation) makes exactly **2 LLM calls**: one to plan all tool calls upfront with `#E1`, `#E2` variable references, and one to synthesize the final answer from collected evidence.

**FSM flow:** `plan_all -> execute_plans (no LLM) -> solve`

```python
REWOOAgent(
    tools: ToolRegistry,                      # Required
    config: AgentConfig | None = None,
    **api_kwargs: Any,
)
```

**Key feature:** Evidence substitution. The plan uses `#E1`, `#E2` references that get replaced with actual tool results before subsequent tools execute, enabling chained dependencies without additional LLM calls.

**When to use:** High-volume automation where LLM cost/latency matters. Best when the tool sequence is predictable.

---

### 3. PlanExecuteAgent

Separates strategic planning from tactical execution. First creates a complete plan, then executes each step sequentially. If a step fails, the agent can revise the remaining plan.

**FSM flow:** `plan -> execute_step -> check_result -> ... -> synthesize`

```python
PlanExecuteAgent(
    tools: ToolRegistry | None = None,        # Optional -- LLM-only mode supported
    config: AgentConfig | None = None,
    max_replans: int = 2,                     # Max replan cycles on failure
    **api_kwargs: Any,
)
```

**When to use:** Multi-step tasks where planning once then executing is more efficient than ReAct's per-iteration thinking. Works without tools for pure-LLM reasoning.

---

### 4. ReflexionAgent

Extends ReAct with an evaluation gate and episodic memory. After the ReAct loop produces an answer, it evaluates the result. If evaluation fails, the agent reflects, stores lessons in memory, and retries with an improved strategy.

**FSM flow:** `think -> act -> evaluate -> (reflect -> think | conclude)`

```python
ReflexionAgent(
    tools: ToolRegistry,                      # Required
    config: AgentConfig | None = None,
    evaluation_fn: Callable[[dict], EvaluationResult] | None = None,
        # External evaluator. If None, the LLM self-evaluates.
    max_reflections: int = 3,                 # Max reflect-think cycles
    hitl: HumanInTheLoop | None = None,       # Optional approval gates
    **api_kwargs: Any,
)
```

**Key feature:** Episodic memory. Each reflection cycle stores a `ReflexionMemory` entry with the episode number, what went wrong, and lessons learned. These accumulate in context so the LLM can learn from past failures within the same run.

---

### 5. PromptChainAgent

A linear pipeline of user-defined LLM steps. Each step has extraction and response instructions, and an optional validation gate. If a gate fails, the chain terminates early.

**FSM flow:** `step_0 -> step_1 -> ... -> step_N -> output`

```python
PromptChainAgent(
    chain: list[ChainStep],                   # Required -- ordered pipeline steps
    config: AgentConfig | None = None,
    **api_kwargs: Any,
)
```

**ChainStep fields:** `step_id` (str), `name` (str), `extraction_instructions` (str), `response_instructions` (str), `validation_fn` (optional callable returning bool).

**When to use:** Content pipelines (outline -> draft -> edit -> polish), data transformation chains, any sequential workflow where each step builds on the previous.

---

### 6. SelfConsistencyAgent

Generates N independent answers to the same question at varying temperatures, then aggregates via majority vote (or a custom aggregation function). No tools needed.

**FSM flow:** `[generate x N samples at different temperatures] -> aggregate -> answer`

```python
SelfConsistencyAgent(
    config: AgentConfig | None = None,
    num_samples: int = 5,                     # Independent generations
    aggregation_fn: Callable[[list[str]], str] | None = None,
        # Default: majority vote. Custom: e.g., longest answer.
    **api_kwargs: Any,
)
```

**When to use:** Factual questions, classification tasks, math problems -- anywhere statistical reliability beats single-shot generation.

---

### 7. DebateAgent

Three personas (proposer, critic, judge) engage in structured multi-round debate. Continues until the judge declares consensus or the maximum number of rounds is reached.

**FSM flow:** `propose -> critique -> counter -> judge -> (propose | conclude)`

```python
DebateAgent(
    config: AgentConfig | None = None,
    num_rounds: int = 3,                      # Max debate rounds
    proposer_persona: str = "",               # Default: constructive advocate
    critic_persona: str = "",                 # Default: rigorous critic
    judge_persona: str = "",                  # Default: impartial evaluator
    **api_kwargs: Any,
)
```

**When to use:** Controversial topics, policy questions, any task where structured argumentation produces better answers than a single perspective.

---

### 8. OrchestratorAgent

Decomposes tasks into subtasks, delegates each to a worker (via `worker_factory`), collects results, and synthesizes. If no worker_factory is provided, the LLM handles subtasks inline.

**FSM flow:** `orchestrate -> delegate -> collect -> (synthesize | orchestrate)`

```python
OrchestratorAgent(
    worker_factory: Callable[[str], AgentResult] | None = None,
        # Subtask solver. If None, LLM handles inline.
    tools: ToolRegistry | None = None,
    config: AgentConfig | None = None,
    max_workers: int = 5,                     # Max subtask delegations per round
    **api_kwargs: Any,
)
```

**When to use:** Hierarchical multi-agent systems, tasks that decompose into independent subtasks that can be solved by specialized agents.

---

### 9. ADaPTAgent

Adaptive Decomposition and Planning for Tasks. Tries the task directly first. If the direct attempt fails, decomposes into subtasks and solves them recursively (up to `max_depth` levels).

**FSM flow:** `attempt -> assess -> (combine | decompose -> [recursive] -> combine)`

```python
ADaPTAgent(
    tools: ToolRegistry | None = None,        # Optional -- can run LLM-only
    config: AgentConfig | None = None,
    max_depth: int = 3,                       # Recursion depth limit
    **api_kwargs: Any,
)
```

**Key feature:** Subtasks can use `AND` (all must succeed) or `OR` (first success wins) operators. Each recursive `run()` creates a fresh FSM and handler set for proper isolation.

**When to use:** Tasks with unknown complexity. The agent adapts its strategy based on whether direct solving works.

---

### 10. EvaluatorOptimizerAgent

Generate-evaluate-refine loop driven by an **external evaluation function** (not LLM self-evaluation). The agent generates output, your evaluation function scores it, and if it fails, the agent refines based on feedback.

**FSM flow:** `generate -> evaluate -> (output | refine -> evaluate)`

```python
EvaluatorOptimizerAgent(
    evaluation_fn: Callable[[str, dict], EvaluationResult],
        # Required -- (generated_output, context) -> EvaluationResult
    config: AgentConfig | None = None,
    max_refinements: int = 3,                 # Max refine-evaluate cycles
    **api_kwargs: Any,
)
```

**When to use:** Code generation + linting, content creation + schema validation, any task where you can write a programmatic quality check.

---

### 11. MakerCheckerAgent

Two-persona quality loop. A "maker" persona generates or revises content, then a "checker" persona evaluates it against quality criteria. Continues until quality threshold is met or max revisions are reached.

**FSM flow:** `make -> check -> (output | revise -> make)`

```python
MakerCheckerAgent(
    maker_instructions: str,                  # Required -- what to produce
    checker_instructions: str,                # Required -- what to evaluate
    config: AgentConfig | None = None,
    max_revisions: int = 3,
    quality_threshold: float = 0.7,           # 0.0-1.0, auto-pass above this
    **api_kwargs: Any,
)
```

**When to use:** Email/document review, compliance checking, any content that needs structured quality assurance before delivery.

---

### 12. ReasoningReactAgent

Extends ReactAgent with a `reason` pseudo-tool backed by `fsm_llm_reasoning.ReasoningEngine`. When the LLM selects reasoning, it pushes a reasoning FSM onto the stack via FSM stacking. Requires `fsm_llm_reasoning` to be installed.

**Requires:** `pip install fsm-llm[reasoning]`

**FSM flow:** `think -> act (tool or reason) -> think -> ... -> conclude`

```python
ReasoningReactAgent(
    tools: ToolRegistry,                      # Required -- reason tool auto-registered
    config: AgentConfig | None = None,
    hitl: HumanInTheLoop | None = None,
    reasoning_model: str | None = None,       # Defaults to config.model
    **api_kwargs: Any,
)
```

**Key feature:** The `reason` pseudo-tool is auto-registered in the tool registry. When selected, the agent pushes a reasoning FSM onto the stack, executes it, and pops results back into context under namespaced keys from `ReasoningIntegrationKeys`. Gives access to all 9 reasoning strategies without explicit configuration.

**When to use:** Tasks that benefit from structured analytical, deductive, or critical reasoning alongside tool use.

---

## Pattern Selection Guide

**Decision tree:**

```
Need tools?
+-- Yes
|   +-- Need structured reasoning? --> ReasoningReactAgent
|   +-- Need self-improvement? --> ReflexionAgent
|   +-- Need upfront planning? --> PlanExecuteAgent
|   +-- Need minimal LLM calls? --> REWOOAgent
|   +-- General purpose --> ReactAgent
+-- No
    +-- Need external evaluation? --> EvaluatorOptimizerAgent
    +-- Need two-persona review? --> MakerCheckerAgent
    +-- Need multiple perspectives?
    |   +-- Structured debate --> DebateAgent
    |   +-- Statistical reliability --> SelfConsistencyAgent
    +-- Need sequential pipeline? --> PromptChainAgent
    +-- Need multi-agent delegation? --> OrchestratorAgent
    +-- Unknown complexity? --> ADaPTAgent
```

| Pattern | Tools | LLM Calls | Best For |
|---------|:-----:|:---------:|----------|
| ReactAgent | Required | 2-3 per cycle | General tool-use tasks |
| REWOOAgent | Required | Exactly 2 | Token-efficient tool-use |
| PlanExecuteAgent | Optional | 1 plan + 1 per step | Multi-step structured work |
| ReflexionAgent | Required | 2-3 per cycle + eval | Tasks needing self-improvement |
| PromptChainAgent | No | 1 per step | Multi-stage pipelines |
| SelfConsistencyAgent | No | N samples | Factual/reasoning accuracy |
| DebateAgent | No | 4 per round | Nuanced/controversial topics |
| OrchestratorAgent | Optional | 1 per round + workers | Complex decomposable tasks |
| ADaPTAgent | Optional | 2+ (recursive) | Unknown complexity tasks |
| EvaluatorOptimizerAgent | No | 1 per refine cycle | Quality-constrained generation |
| MakerCheckerAgent | No | 2 per revision | Content with quality gates |
| ReasoningReactAgent | Required | 2-3 per cycle + reasoning | Tasks needing structured reasoning |

---

## Tool System

### ToolRegistry

Central registry for managing tools available to agents. Handles registration, lookup, prompt generation, and execution with timing and error handling.

```python
from fsm_llm_agents import ToolRegistry, tool, ToolDefinition

registry = ToolRegistry()

# Method 1: register_function
registry.register_function(
    fn=my_search_function,
    name="search",
    description="Search the web for information",
    parameter_schema={
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    },
    requires_approval=False,
)

# Method 2: @tool decorator
@tool(description="Send an email", requires_approval=True)
def send_email(to: str, subject: str, body: str) -> str:
    return f"Sent to {to}"

registry.register(send_email._tool_definition)

# Method 3: ToolDefinition directly
registry.register(ToolDefinition(
    name="calculate",
    description="Evaluate a math expression",
    parameter_schema={"properties": {"expression": {"type": "string"}}},
    execute_fn=lambda p: str(eval(p["expression"])),
))
```

**Key methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `register(tool_def)` | `ToolRegistry` | Register a ToolDefinition (chainable) |
| `register_function(fn, name, ...)` | `ToolRegistry` | Register a callable (chainable) |
| `register_agent(agent, name, desc)` | `ToolRegistry` | Register an agent as a tool (chainable) |
| `get(name)` | `ToolDefinition` | Retrieve by name (raises `ToolNotFoundError`) |
| `list_tools()` | `list[ToolDefinition]` | All registered tools |
| `tool_names` | `list[str]` | All tool names (property) |
| `execute(tool_call)` | `ToolResult` | Execute with timing and error handling |
| `to_prompt_description()` | `str` | LLM-friendly tool listing |
| `to_classification_schema()` | `dict` | ClassificationSchema-compatible dict for tool selection |
| `len(registry)` | `int` | Number of tools |
| `"name" in registry` | `bool` | Check if tool exists |

### @tool Decorator

Marks a function as an agent tool with auto-schema inference from type hints.

```python
# Simplest: bare @tool — infers everything from function
@tool
def search(query: str) -> str:
    """Search the web for information."""
    return search_engine.search(query)

# With overrides
@tool(name="web_search", description="Search the web", requires_approval=True)
def web_search(query: str) -> str:
    return search_engine.search(query)

# With Annotated descriptions
from typing import Annotated

@tool
def search(
    query: Annotated[str, "The search query"],
    limit: Annotated[int, "Max results to return"] = 10,
) -> str:
    """Search the web."""
    return search_engine.search(query, limit)

# Explicit schema (backward compatible)
@tool(parameter_schema={"properties": {"query": {"type": "string"}}})
def search(params: dict) -> str:
    return search_engine.search(params["query"])

# Access the attached definition
registry.register(search._tool_definition)
```

Auto-schema type mapping: `str→string`, `int→integer`, `float→number`, `bool→boolean`, `list→array`, `dict→object`. Legacy `params: dict` single-parameter functions are auto-detected and skip inference.

### Agents-as-Tools

Register agents as tools to build supervisor/orchestrator patterns:

```python
research_agent = ReactAgent(tools=research_registry)
math_agent = ReactAgent(tools=math_registry)

supervisor_registry = ToolRegistry()
supervisor_registry.register_agent(research_agent, name="researcher", description="Research questions")
supervisor_registry.register_agent(math_agent, name="mathematician", description="Math problems")

supervisor = ReactAgent(tools=supervisor_registry)
result = supervisor("What is the GDP of France and what is it divided by population?")
```

### Tool Execution

The registry handles parameter validation, function signature inspection, timing, and error wrapping.

```python
result = registry.execute(ToolCall(tool_name="search", parameters={"query": "test"}))
# -> ToolResult(tool_name="search", success=True, result=..., execution_time_ms=...)

# On failure:
# -> ToolResult(tool_name="search", success=False, error="...", execution_time_ms=...)
```

Functions are called based on their signature and schema: zero-arg functions are called with no arguments, single-arg functions without a schema receive the full parameters dict (legacy), and functions with schema-inferred or explicit parameters receive `**kwargs` filtered to known properties.

---

## Human-in-the-Loop

### HumanInTheLoop

Gates tool executions behind human approval and provides confidence-based escalation.

```python
from fsm_llm_agents import HumanInTheLoop, ReactAgent

hitl = HumanInTheLoop(
    # Which tool calls need approval (receives ToolCall + context, returns bool)
    approval_policy=lambda call, ctx: call.tool_name in ["send_email", "delete"],

    # How to request approval (receives ApprovalRequest, returns bool)
    approval_callback=my_approval_handler,

    # Escalate when agent confidence drops below this
    confidence_threshold=0.3,

    # Callback when escalation happens
    on_escalation=lambda reason, ctx: notify_human(reason),

    # Max seconds to wait for approval (None = no limit)
    approval_timeout=60.0,
)

agent = ReactAgent(tools=registry, hitl=hitl)
result = agent.run("Send an email to the team")
# Agent pauses and requests approval before calling send_email
```

### How It Works

1. A `CONTEXT_UPDATE` handler checks if the selected tool requires approval
2. When `approval_required=True` appears in context, the agent enters the `await_approval` FSM state (auto-included when HITL is active and approval tools exist)
3. The agent calls `hitl.request_approval()` with an `ApprovalRequest` containing tool name, parameters, reasoning, and context summary
4. If approved, execution proceeds normally; if denied, the tool selection is cleared and the agent continues thinking
5. If confidence falls below `confidence_threshold`, the `on_escalation` callback fires

### Type Aliases

```python
ApprovalPolicy = Callable[[ToolCall, dict[str, Any]], bool]
ApprovalCallback = Callable[[ApprovalRequest], bool]
EscalationCallback = Callable[[str, dict[str, Any]], None]
```

---

## API Reference

### Models

| Class | Description |
|-------|-------------|
| `BaseAgent` | ABC: `__call__`, `_standard_run()`, `_check_budgets()`, `_extract_answer()`, `_build_trace()`, `_filter_context()`, `_try_parse_structured_output()` |
| `AgentConfig` | Configuration: `model`, `max_iterations` (default 10), `timeout_seconds` (300), `temperature` (0.5), `max_tokens` (1000), `output_schema` (optional Pydantic model class) |
| `AgentResult` | Result: `answer`, `success`, `trace`, `final_context`, `structured_output`. Properties: `iterations_used`, `tools_used`. `__str__` returns structured_output if available |
| `AgentTrace` | Trace: `tool_calls`, `total_iterations`. Property: `tools_used` |
| `AgentStep` | Single step: `iteration`, `thought`, `action`, `observation`, `timestamp` |
| `ToolDefinition` | Tool: `name`, `description`, `parameter_schema`, `requires_approval`, `execute_fn` (runtime-only) |
| `ToolCall` | Invocation: `tool_name`, `parameters`, `reasoning` |
| `ToolResult` | Result: `tool_name`, `success`, `result`, `error`, `execution_time_ms`. Property: `summary` |
| `ApprovalRequest` | HITL: `tool_name`, `parameters`, `reasoning`, `context_summary` |
| `ChainStep` | Prompt chain: `step_id`, `name`, `extraction_instructions`, `response_instructions`, `validation_fn` |
| `PlanStep` | Plan: `step_id`, `description`, `dependencies`, `status`, `result` |
| `EvaluationResult` | Evaluation: `passed`, `score` (0-1), `feedback`, `criteria_met` |
| `ReflexionMemory` | Memory: `episode`, `task_summary`, `outcome`, `reflection`, `lessons`, `timestamp` |
| `DebateRound` | Debate: `round_num`, `proposition`, `critique`, `counter_argument`, `judge_verdict` |
| `DecompositionResult` | ADaPT: `subtasks`, `operator` (AND/OR), `depth` |

### Exceptions

All inherit from `AgentError(FSMError)` and include an optional `details` dict.

```
AgentError
+-- ToolExecutionError          .tool_name
+-- ToolNotFoundError           .tool_name
+-- ToolValidationError         .tool_name
+-- BudgetExhaustedError        .budget_type, .limit
+-- AgentTimeoutError           .timeout_seconds
+-- ApprovalDeniedError         .action_description
+-- EvaluationError             .evaluator
+-- DecompositionError          .depth
```

### Constants

| Class | Key Values |
|-------|------------|
| `AgentStates` | `THINK`, `ACT`, `CONCLUDE`, `AWAIT_APPROVAL` |
| `ContextKeys` | `TASK`, `TOOL_NAME`, `TOOL_INPUT`, `SHOULD_TERMINATE`, `FINAL_ANSWER`, `CONFIDENCE`, plus pattern-specific keys |
| `Defaults` | `MAX_ITERATIONS=10`, `TIMEOUT_SECONDS=300`, `TEMPERATURE=0.5`, `MAX_TOKENS=1000`, `CONFIDENCE_THRESHOLD=0.3`, `FSM_BUDGET_MULTIPLIER=3` |
| `HandlerNames` | `TOOL_EXECUTOR`, `ITERATION_LIMITER`, `OBSERVATION_TRACKER`, `HITL_GATE`, plus pattern-specific names |

---

## File Map

| File | Purpose |
|------|---------|
| `base.py` | `BaseAgent` -- ABC with shared conversation loop, budgets, `__call__`, structured output |
| `react.py` | `ReactAgent(BaseAgent)` -- ReAct loop with tool dispatch |
| `rewoo.py` | `REWOOAgent(BaseAgent)` -- planning-first tool execution (2 LLM calls) |
| `plan_execute.py` | `PlanExecuteAgent(BaseAgent)` -- plan decomposition and sequential execution |
| `reflexion.py` | `ReflexionAgent(BaseAgent)` -- self-reflection with episodic memory |
| `prompt_chain.py` | `PromptChainAgent(BaseAgent)` -- sequential prompt pipeline with gates |
| `self_consistency.py` | `SelfConsistencyAgent(BaseAgent)` -- multiple samples with majority vote (full run override) |
| `debate.py` | `DebateAgent(BaseAgent)` -- multi-perspective debate with judge |
| `orchestrator.py` | `OrchestratorAgent(BaseAgent)` -- worker delegation and synthesis |
| `adapt.py` | `ADaPTAgent(BaseAgent)` -- adaptive complexity with recursive decomposition |
| `evaluator_optimizer.py` | `EvaluatorOptimizerAgent(BaseAgent)` -- iterative evaluation and optimization |
| `maker_checker.py` | `MakerCheckerAgent(BaseAgent)` -- draft-review verification loop |
| `reasoning_react.py` | `ReasoningReactAgent(BaseAgent)` -- ReAct + structured reasoning (requires `fsm_llm_reasoning`) |
| `tools.py` | `ToolRegistry` + `@tool` decorator (auto-schema) + `register_agent()` |
| `hitl.py` | `HumanInTheLoop` -- approval policies, escalation, callbacks |
| `handlers.py` | `AgentHandlers` -- tool executor, iteration limiter, observation tracker, HITL gate |
| `fsm_definitions.py` | FSM builders: `build_react_fsm()`, `build_reflexion_fsm()`, `build_plan_execute_fsm()`, etc. |
| `prompts.py` | Prompt builders for think/act/conclude/approval and all pattern-specific states |
| `definitions.py` | Pydantic models (14 model classes) |
| `constants.py` | State enums (11 classes), context keys, defaults, error/log messages |
| `exceptions.py` | Exception hierarchy (9 exception classes) |
| `__main__.py` | CLI: `python -m fsm_llm_agents --info` |
| `__init__.py` | Public API exports (44 symbols) + `create_agent()` factory |
| `__version__.py` | Version (from `fsm_llm.__version__`) |

---

## Examples

All examples are in `examples/agents/` and support OpenAI (default) and Ollama fallback.

```bash
export OPENAI_API_KEY=your-key-here
python examples/agents/react_search/run.py

# Or with Ollama:
export LLM_MODEL=ollama_chat/qwen3.5:9b
python examples/agents/react_search/run.py
```

| Example | Pattern | Description |
|---------|---------|-------------|
| `react_search/` | ReactAgent | Search + calculate + lookup tools |
| `hitl_approval/` | ReactAgent + HITL | Approval gates for sensitive tools |
| `react_hitl_combined/` | ReactAgent + HITL | @tool decorator + approval workflow |
| `reflexion/` | ReflexionAgent | External evaluation with retry |
| `plan_execute/` | PlanExecuteAgent | Multi-step research planning |
| `rewoo/` | REWOOAgent | Plan-first tool-use with evidence substitution |
| `self_consistency/` | SelfConsistencyAgent | Majority vote over 5 samples |
| `debate/` | DebateAgent | AI hiring debate with custom personas |
| `evaluator_optimizer/` | EvaluatorOptimizerAgent | Haiku generation with structure validation |
| `maker_checker/` | MakerCheckerAgent | Professional email with quality review |
| `prompt_chain/` | PromptChainAgent | Research -> Draft -> Polish pipeline |
| `classified_dispatch/` | Classifier + agent dispatch | Classification-driven agent selection |
| `classified_tools/` | ReactAgent + Classifier | Classification for tool selection |
| `full_pipeline/` | Classifier + ReactAgent | End-to-end classify -> agent -> tools |
| `hierarchical_tools/` | ReactAgent | Nested tool registries, tool grouping |
| `reasoning_stacking/` | ReasoningReactAgent | Push/pop reasoning FSMs via agent |
| `reasoning_tool/` | ReactAgent + ReasoningEngine | Reasoning engine wrapped as @tool |
| `workflow_agent/` | ReactAgent + WorkflowEngine | Agent integrated with workflow orchestration |

---

## Integration

This package integrates with other FSM-LLM sub-packages:

- **fsm_llm (core)** -- All agents use `fsm_llm.API` for FSM execution, `HandlerTiming` for handler registration, and `LLMInterface` for LLM communication. FSM definitions are generated at runtime and passed through `API.from_definition()`.
- **fsm_llm (classification)** -- `ToolRegistry.to_classification_schema()` generates classification schemas for intent-based tool selection. See `classified_dispatch` and `classified_tools` examples.
- **fsm_llm_reasoning** -- `ReasoningReactAgent` auto-registers a `reason` pseudo-tool backed by `ReasoningEngine` via FSM stacking. Conditionally available only when `fsm_llm_reasoning` is installed.
- **fsm_llm_workflows** -- Agents can be used as workflow steps. See `workflow_agent` example.

---

## Development

```bash
# Run agent tests (596 tests across 25 files)
pytest tests/test_fsm_llm_agents/ -v

# Lint and format
ruff check src/fsm_llm_agents/
ruff format src/fsm_llm_agents/

# Type check
mypy src/fsm_llm_agents/
```

Test files cover each agent pattern individually (`test_react.py`, `test_reflexion.py`, `test_debate.py`, etc.) plus integration tests (`test_integration_methods.py`), bug fix regressions (`test_bug_fixes.py`), and shared infrastructure (`test_handlers.py`, `test_tools.py`, `test_hitl.py`, `test_definitions.py`, `test_constants.py`, `test_exceptions.py`, `test_fsm_definitions.py`, `test_prompts.py`).

---

## License

GPL-3.0-or-later. See the [main project](https://github.com/NikolasMarkou/fsm_llm) for details.
