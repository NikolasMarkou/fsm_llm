# fsm_llm_agents — Agentic Design Patterns for FSM-LLM

11 production-ready agentic patterns built on top of [FSM-LLM](https://github.com/NikolasMarkou/fsm_llm)'s 2-pass state machine engine. Each pattern auto-generates FSM definitions at runtime — no manual JSON authoring required.

```bash
pip install fsm-llm[agents]
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Pattern Selection Guide](#pattern-selection-guide)
- [Tool-Based Patterns](#tool-based-patterns)
  - [ReactAgent](#1-reactagent)
  - [ReflexionAgent](#2-reflexionagent)
  - [PlanExecuteAgent](#3-planexecuteagent)
  - [REWOOAgent](#4-rewooagent)
- [Evaluation & Refinement Patterns](#evaluation--refinement-patterns)
  - [EvaluatorOptimizerAgent](#5-evaluatoroptimizeragent)
  - [MakerCheckerAgent](#6-makercheckeragent)
- [Multi-Perspective Patterns](#multi-perspective-patterns)
  - [SelfConsistencyAgent](#7-selfconsistencyagent)
  - [DebateAgent](#8-debateagent)
- [Composition Patterns](#composition-patterns)
  - [PromptChainAgent](#9-promptchainagent)
  - [OrchestratorAgent](#10-orchestratoragent)
  - [ADaPTAgent](#11-adaptagent)
- [Core Infrastructure](#core-infrastructure)
  - [ToolRegistry & @tool Decorator](#toolregistry--tool-decorator)
  - [HumanInTheLoop (HITL)](#humanintheloop-hitl)
  - [AgentConfig & AgentResult](#agentconfig--agentresult)
- [Exception Hierarchy](#exception-hierarchy)
- [Architecture](#architecture)
- [Examples](#examples)

---

## Quick Start

```python
from fsm_llm_agents import ReactAgent, ToolRegistry, AgentConfig, tool

# Define tools
@tool(description="Search the web for information")
def search(query: str) -> str:
    return f"Results for: {query}"

# Register tools
registry = ToolRegistry()
registry.register(search._tool_definition)

# Run agent
agent = ReactAgent(
    tools=registry,
    config=AgentConfig(model="gpt-4o-mini", max_iterations=5),
)
result = agent.run("What is the capital of France?")

print(result.answer)       # "The capital of France is Paris."
print(result.success)      # True
print(result.tools_used)   # {"search"}
print(result.iterations_used)  # 3
```

---

## Pattern Selection Guide

| Pattern | Tools | LLM Calls | Best For |
|---------|:-----:|:---------:|----------|
| [ReactAgent](#1-reactagent) | Required | 2-3 per cycle | General tool-use tasks |
| [ReflexionAgent](#2-reflexionagent) | Required | 2-3 per cycle + eval | Tasks needing self-improvement |
| [PlanExecuteAgent](#3-planexecuteagent) | Optional | 1 plan + 1 per step | Multi-step structured work |
| [REWOOAgent](#4-rewooagent) | Required | Exactly 2 | Token-efficient tool-use |
| [EvaluatorOptimizerAgent](#5-evaluatoroptimizeragent) | No | 1 per refine cycle | Quality-constrained generation |
| [MakerCheckerAgent](#6-makercheckeragent) | No | 2 per revision | Content with quality gates |
| [SelfConsistencyAgent](#7-selfconsistencyagent) | No | N samples | Factual/reasoning accuracy |
| [DebateAgent](#8-debateagent) | No | 4 per round | Nuanced/controversial topics |
| [PromptChainAgent](#9-promptchainagent) | No | 1 per step | Multi-stage pipelines |
| [OrchestratorAgent](#10-orchestratoragent) | Optional | 1 per round + workers | Complex decomposable tasks |
| [ADaPTAgent](#11-adaptagent) | Optional | 2+ (recursive) | Unknown complexity tasks |

**Decision tree:**

```
Need tools?
├── Yes
│   ├── Need self-improvement? → ReflexionAgent
│   ├── Need upfront planning? → PlanExecuteAgent
│   ├── Need minimal LLM calls? → REWOOAgent
│   └── General purpose → ReactAgent
└── No
    ├── Need external evaluation? → EvaluatorOptimizerAgent
    ├── Need two-persona review? → MakerCheckerAgent
    ├── Need multiple perspectives?
    │   ├── Structured debate → DebateAgent
    │   └── Statistical reliability → SelfConsistencyAgent
    ├── Need sequential pipeline? → PromptChainAgent
    ├── Need multi-agent delegation? → OrchestratorAgent
    └── Unknown complexity? → ADaPTAgent
```

---

## Tool-Based Patterns

### 1. ReactAgent

The classic ReAct (Reasoning + Acting) loop. The agent thinks about what to do, acts by calling a tool, observes the result, and repeats until it can conclude.

**FSM flow:**

```
think → act → think → ... → conclude
```

**Constructor:**

```python
ReactAgent(
    tools: ToolRegistry,                      # Required — registered tools
    config: AgentConfig | None = None,        # Model, iterations, timeout
    hitl: HumanInTheLoop | None = None,       # Optional approval gates
    **api_kwargs: Any,                        # Passed to fsm_llm.API
)
```

**Usage:**

```python
from fsm_llm_agents import ReactAgent, ToolRegistry, AgentConfig

registry = ToolRegistry()
registry.register_function(
    lambda p: f"Population: {p.get('country', 'unknown')}: 67M",
    name="lookup",
    description="Look up population data",
    parameter_schema={"country": "country name"},
)

agent = ReactAgent(
    tools=registry,
    config=AgentConfig(model="gpt-4o-mini", max_iterations=8),
)
result = agent.run("What is the population of France?")
```

**When to use:** General-purpose tool-use. Start here if unsure which pattern to pick.

**When to avoid:** If you know the exact tool sequence upfront (use REWOOAgent instead) or if you need self-improvement (use ReflexionAgent).

---

### 2. ReflexionAgent

Extends ReactAgent with an evaluation gate, verbal self-critique, and episodic memory. After the ReAct loop produces an answer, it evaluates the result. If the evaluation fails, the agent reflects on what went wrong, stores lessons in memory, and retries with an improved strategy.

**FSM flow:**

```
think → act → evaluate ──passed──→ conclude
                │
                └──failed──→ reflect → think (retry with lessons)
```

**Constructor:**

```python
ReflexionAgent(
    tools: ToolRegistry,                      # Required
    config: AgentConfig | None = None,
    evaluation_fn: Callable[[dict], EvaluationResult] | None = None,
        # External evaluator. If None, the LLM self-evaluates.
    max_reflections: int = 3,                 # Max reflect→think cycles
    hitl: HumanInTheLoop | None = None,       # Optional approval gates
    **api_kwargs: Any,
)
```

**Usage:**

```python
from fsm_llm_agents import ReflexionAgent, ToolRegistry, EvaluationResult

def check_answer(context):
    answer = context.get("final_answer", "")
    has_number = any(c.isdigit() for c in answer)
    return EvaluationResult(
        passed=has_number,
        score=1.0 if has_number else 0.0,
        feedback="Answer must include a numeric value" if not has_number else "OK",
    )

agent = ReflexionAgent(
    tools=registry,
    evaluation_fn=check_answer,
    max_reflections=3,
)
result = agent.run("What is the population density of Japan?")
# Agent will retry up to 3 times if answer lacks a number
```

**Key feature:** Episodic memory. Each reflection cycle stores a `ReflexionMemory` entry with the episode number, what went wrong, and lessons learned. These accumulate in context so the LLM can learn from past failures within the same run.

---

### 3. PlanExecuteAgent

Separates strategic planning from tactical execution. First creates a complete plan, then executes each step sequentially. If a step fails, the agent can revise the remaining plan (up to `max_replans` times).

**FSM flow:**

```
plan → execute_step → check_result ──all done──→ synthesize
                          │
                          ├──next step──→ execute_step
                          │
                          └──failed──→ replan → execute_step
```

**Constructor:**

```python
PlanExecuteAgent(
    tools: ToolRegistry | None = None,        # Optional — LLM-only mode supported
    config: AgentConfig | None = None,
    max_replans: int = 2,                     # Max replan cycles
    **api_kwargs: Any,
)
```

**Usage:**

```python
from fsm_llm_agents import PlanExecuteAgent, ToolRegistry

agent = PlanExecuteAgent(
    tools=registry,
    max_replans=2,
)
result = agent.run(
    "Compare Django, Flask, and FastAPI for building a REST API. "
    "Search for each, then summarize the comparison."
)
# The agent creates a plan, then executes step-by-step
```

**When to use:** Multi-step tasks where planning once then executing is more efficient than ReAct's per-iteration planning. Works without tools for pure-LLM reasoning.

---

### 4. REWOOAgent

REWOO (Reasoning Without Observation) makes exactly **2 LLM calls**: one to plan all tool calls upfront with `#E1`, `#E2` variable references, and one to synthesize the final answer from collected evidence.

**FSM flow:**

```
plan_all → execute_plans (no LLM) → solve
```

**Constructor:**

```python
REWOOAgent(
    tools: ToolRegistry,                      # Required
    config: AgentConfig | None = None,
    **api_kwargs: Any,
)
```

**Usage:**

```python
from fsm_llm_agents import REWOOAgent, ToolRegistry

agent = REWOOAgent(tools=registry)
result = agent.run(
    "Find the population of France and Germany, "
    "calculate the difference."
)
# Plan: #E1 = search("France population"), #E2 = search("Germany population"),
#        #E3 = calculate("#E1 - #E2")
# Execute: all 3 tools run, evidence substituted
# Solve: LLM synthesizes from evidence dict
```

**Key feature:** Evidence substitution. The plan uses `#E1`, `#E2` references that get replaced with actual tool results before subsequent tools execute. This allows chained dependencies without additional LLM calls.

**When to use:** High-volume automation where LLM cost/latency matters. Best when the tool sequence is predictable.

---

## Evaluation & Refinement Patterns

### 5. EvaluatorOptimizerAgent

Generate-evaluate-refine loop driven by an **external evaluation function** (not LLM self-evaluation). The agent generates output, your evaluation function scores it, and if it fails, the agent refines based on feedback.

**FSM flow:**

```
generate → evaluate ──passed──→ output
               │
               └──failed──→ refine → evaluate (loop)
```

**Constructor:**

```python
EvaluatorOptimizerAgent(
    evaluation_fn: Callable[[str, dict], EvaluationResult],
        # Required — (generated_output, context) → EvaluationResult
    config: AgentConfig | None = None,
    max_refinements: int = 3,                 # Max refine→evaluate cycles
    **api_kwargs: Any,
)
```

**Usage:**

```python
from fsm_llm_agents import EvaluatorOptimizerAgent, EvaluationResult

def check_haiku(output: str, context: dict) -> EvaluationResult:
    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    passed = len(lines) == 3
    return EvaluationResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        feedback="Must have exactly 3 lines" if not passed else "Valid haiku",
    )

agent = EvaluatorOptimizerAgent(
    evaluation_fn=check_haiku,
    max_refinements=3,
)
result = agent.run("Write a haiku about autumn")
```

**When to use:** Code generation + linting, content creation + schema validation, any task where you can write a programmatic quality check.

---

### 6. MakerCheckerAgent

Two-persona quality loop. A "maker" persona generates or revises content, then a "checker" persona evaluates it against quality criteria. The loop continues until the checker approves (quality score meets threshold) or maximum revisions are reached.

**FSM flow:**

```
make → check ──quality >= threshold──→ output
          │
          └──needs revision──→ revise → make (loop)
```

**Constructor:**

```python
MakerCheckerAgent(
    maker_instructions: str,                  # Required — what the maker should produce
    checker_instructions: str,                # Required — what the checker evaluates
    config: AgentConfig | None = None,
    max_revisions: int = 3,
    quality_threshold: float = 0.7,           # 0.0-1.0, auto-pass above this
    **api_kwargs: Any,
)
```

**Usage:**

```python
from fsm_llm_agents import MakerCheckerAgent

agent = MakerCheckerAgent(
    maker_instructions=(
        "Write a professional email: concise, warm tone, "
        "clear call to action, under 150 words."
    ),
    checker_instructions=(
        "Score 0.0-1.0 on: tone, conciseness, clarity, grammar. "
        "Set checker_passed=true only if quality_score >= 0.7."
    ),
    max_revisions=3,
    quality_threshold=0.7,
)
result = agent.run("Apologize to a client for a 2-day project delay")
```

**When to use:** Email/document review, compliance checking, any content that needs structured quality assurance before delivery.

---

## Multi-Perspective Patterns

### 7. SelfConsistencyAgent

Generates N independent answers to the same question at varying temperatures, then aggregates via majority vote (or a custom aggregation function). No tools needed.

**FSM flow:**

```
[generate × N samples at different temperatures] → aggregate → answer
```

**Constructor:**

```python
SelfConsistencyAgent(
    config: AgentConfig | None = None,
    num_samples: int = 5,                     # Independent generations
    aggregation_fn: Callable[[list[str]], str] | None = None,
        # Default: majority vote. Custom: e.g., longest answer.
    **api_kwargs: Any,
)
```

**Usage:**

```python
from fsm_llm_agents import SelfConsistencyAgent

agent = SelfConsistencyAgent(num_samples=7)
result = agent.run("What is the capital of Australia?")
# Runs 7 times at temperatures 0.5→1.0, majority vote picks "Canberra"

# Custom aggregation: pick the longest answer
agent = SelfConsistencyAgent(
    num_samples=5,
    aggregation_fn=lambda samples: max(samples, key=len),
)
```

**Key feature:** Fault-tolerant. If some samples fail (e.g., API errors), the agent continues with the remaining samples. Majority vote still works with partial results.

**When to use:** Factual questions, classification tasks, math problems — anywhere statistical reliability beats single-shot generation.

---

### 8. DebateAgent

Three personas engage in structured multi-round debate: a proposer advocates, a critic challenges, and a judge evaluates. Continues until the judge declares consensus or the maximum number of rounds is reached.

**FSM flow:**

```
propose → critique → counter → judge ──consensus──→ conclude
                                  │
                                  └──no consensus──→ propose (next round)
```

**Constructor:**

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

**Usage:**

```python
from fsm_llm_agents import DebateAgent

agent = DebateAgent(
    num_rounds=2,
    proposer_persona="You are a technology optimist with concrete examples.",
    critic_persona="You are a technology skeptic focusing on risks.",
    judge_persona="You are a balanced policy analyst.",
)
result = agent.run("Should AI be used in hiring decisions?")

# Access debate rounds
for round_data in result.final_context.get("debate_rounds", []):
    print(f"Round {round_data['round_num']}: {round_data['judge_verdict'][:80]}")
```

**When to use:** Controversial topics, policy questions, any task where structured argumentation produces better answers than a single perspective.

---

## Composition Patterns

### 9. PromptChainAgent

A linear pipeline of user-defined LLM steps. Each step has extraction and response instructions, and an optional validation gate. If a gate fails, the chain terminates early.

**FSM flow:**

```
step_0 → step_1 → ... → step_N → output
           │
           └── gate fails → terminate early
```

**Constructor:**

```python
PromptChainAgent(
    chain: list[ChainStep],                   # Required — ordered pipeline steps
    config: AgentConfig | None = None,
    **api_kwargs: Any,
)
```

**ChainStep fields:**

```python
ChainStep(
    step_id: str,                             # Unique identifier
    name: str,                                # Human-readable name
    extraction_instructions: str,             # What to extract (JSON schema)
    response_instructions: str,               # How to respond
    validation_fn: Callable[[dict], bool] | None = None,
        # Optional gate — return False to terminate chain
)
```

**Usage:**

```python
from fsm_llm_agents import PromptChainAgent, ChainStep

chain = [
    ChainStep(
        step_id="research",
        name="Research",
        extraction_instructions='Extract: "key_points" (list), "main_thesis" (str)',
        response_instructions="Research the topic thoroughly. Present key facts.",
    ),
    ChainStep(
        step_id="draft",
        name="Draft",
        extraction_instructions='Extract: "draft_text" (str), "word_count" (int)',
        response_instructions="Write a structured draft from the research.",
        validation_fn=lambda ctx: len(str(ctx.get("chain_step_result", ""))) > 50,
    ),
    ChainStep(
        step_id="polish",
        name="Polish",
        extraction_instructions='Extract: "final_text" (str)',
        response_instructions="Refine the draft. Improve clarity and flow.",
    ),
]

agent = PromptChainAgent(chain=chain)
result = agent.run("Write a short essay on why learning to code is valuable.")
```

**When to use:** Content pipelines (outline -> draft -> edit -> polish), data transformation chains, any sequential workflow where each step builds on the previous.

---

### 10. OrchestratorAgent

Decomposes tasks into subtasks, delegates each to a worker (via `worker_factory`), collects results, and synthesizes. If no worker_factory is provided, the LLM handles subtasks inline.

**FSM flow:**

```
orchestrate → delegate → collect ──all done──→ synthesize
                             │
                             └──more work──→ orchestrate (loop)
```

**Constructor:**

```python
OrchestratorAgent(
    worker_factory: Callable[[str], AgentResult] | None = None,
        # Subtask solver. If None, LLM handles inline.
    tools: ToolRegistry | None = None,
        # Reserved for worker_factory implementations (not used directly)
    config: AgentConfig | None = None,
    max_workers: int = 5,                     # Max subtask delegations per round
    **api_kwargs: Any,
)
```

**Usage:**

```python
from fsm_llm_agents import OrchestratorAgent, ReactAgent, AgentResult

# Create a worker that uses ReactAgent for each subtask
def react_worker(subtask: str) -> AgentResult:
    worker = ReactAgent(tools=registry)
    return worker.run(subtask)

agent = OrchestratorAgent(
    worker_factory=react_worker,
    max_workers=3,
)
result = agent.run("Analyze the pros, cons, and market trends of electric vehicles")
# Orchestrator decomposes into 3 subtasks, delegates to react_worker,
# collects results, synthesizes final analysis
```

**When to use:** Hierarchical multi-agent systems, tasks that decompose into independent subtasks that can be solved by specialized agents.

---

### 11. ADaPTAgent

Adaptive Decomposition and Planning for Tasks. Tries the task directly first. If the direct attempt fails, decomposes into subtasks and solves them recursively (up to `max_depth` levels).

**FSM flow:**

```
attempt → assess ──success──→ combine (final answer)
             │
             └──failure──→ decompose → [recursive run() per subtask] → combine
```

**Constructor:**

```python
ADaPTAgent(
    tools: ToolRegistry | None = None,        # Optional
    config: AgentConfig | None = None,
    max_depth: int = 3,                       # Recursion depth limit
    **api_kwargs: Any,
)
```

**Usage:**

```python
from fsm_llm_agents import ADaPTAgent

agent = ADaPTAgent(max_depth=2)
result = agent.run(
    "Explain how gradient descent works in neural networks, "
    "including the math behind backpropagation."
)
# May solve directly, or decompose into:
#   1. "Explain gradient descent"
#   2. "Explain backpropagation math"
# Then combine results
```

**Key feature:** The `_depth` parameter tracks recursion depth. Each recursive `self.run()` creates a fresh FSM and handler set, ensuring proper isolation between subtask executions.

**Operators:** Subtasks can use `AND` (all must succeed) or `OR` (first success wins).

**When to use:** Tasks with unknown complexity. The agent adapts its strategy based on whether direct solving works, avoiding unnecessary decomposition.

---

## Core Infrastructure

### ToolRegistry & @tool Decorator

The `ToolRegistry` manages tool definitions, generates LLM-aware prompts, and handles execution with timing and error handling.

**Registration:**

```python
from fsm_llm_agents import ToolRegistry, tool, ToolDefinition

registry = ToolRegistry()

# Method 1: register_function
registry.register_function(
    fn=my_search_function,
    name="search",
    description="Search the web for information",
    parameter_schema={"query": "search query string"},
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
    parameter_schema={"expression": "math expression"},
    execute_fn=lambda p: str(eval(p["expression"])),
))
```

**Key methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `register(tool_def)` | `ToolRegistry` | Register a ToolDefinition (chainable) |
| `register_function(fn, name, ...)` | `ToolRegistry` | Register a callable (chainable) |
| `get(name)` | `ToolDefinition` | Retrieve by name (raises `ToolNotFoundError`) |
| `list_tools()` | `list[ToolDefinition]` | All registered tools |
| `tool_names` | `list[str]` | All tool names (property) |
| `execute(tool_call)` | `ToolResult` | Execute with timing and error handling |
| `to_prompt_description()` | `str` | LLM-friendly tool listing |
| `len(registry)` | `int` | Number of tools |
| `"name" in registry` | `bool` | Check if tool exists |

---

### HumanInTheLoop (HITL)

Configurable approval gates for sensitive tool calls. Integrates with `ReactAgent` and `ReflexionAgent`.

```python
from fsm_llm_agents import HumanInTheLoop, ReactAgent

hitl = HumanInTheLoop(
    # Which tools need approval?
    approval_policy=lambda tool_call, context: tool_call.tool_name in [
        "send_email", "delete_record", "publish"
    ],
    # How to request approval (interactive prompt, Slack bot, etc.)
    approval_callback=lambda req: input(
        f"Approve {req.tool_name}({req.parameters})? [y/n]: "
    ).lower() == "y",
    # Auto-escalate when confidence drops below threshold
    confidence_threshold=0.3,
    # Called when escalation triggers
    on_escalation=lambda reason, ctx: print(f"ESCALATION: {reason}"),
)

agent = ReactAgent(tools=registry, hitl=hitl)
result = agent.run("Send a project update email to the team")
# Agent will pause and ask for approval before calling send_email
```

**How it works:**

1. A `CONTEXT_UPDATE` handler flags tools marked `requires_approval=True`
2. The agent's main loop detects `approval_required=True` in context
3. It calls `hitl.request_approval()` with tool details
4. If denied, the agent clears the tool selection and continues thinking

---

### AgentConfig & AgentResult

**AgentConfig** — shared configuration for all agents:

```python
from fsm_llm_agents import AgentConfig

config = AgentConfig(
    model="gpt-4o-mini",        # LLM model identifier
    max_iterations=10,           # Max agent cycles (default: 10)
    timeout_seconds=300.0,       # Wall-clock timeout (default: 300s)
    temperature=0.5,             # LLM temperature (default: 0.5)
    max_tokens=1000,             # Max output tokens (default: 1000)
)
```

**AgentResult** — returned by all agents:

```python
result = agent.run("task")

result.answer           # str — the final answer
result.success          # bool — whether the agent succeeded
result.trace            # AgentTrace — execution trace
result.final_context    # dict — all context data (internal keys stripped)
result.iterations_used  # int — total FSM iterations
result.tools_used       # set[str] — unique tool names called
```

**AgentTrace:**

```python
result.trace.tool_calls     # list[ToolCall] — all tool invocations
result.trace.total_iterations  # int — iteration count
result.trace.tools_used     # set[str] — unique tool names
```

---

## Exception Hierarchy

All exceptions inherit from `AgentError(FSMError)` and include an optional `details` dict.

```
AgentError
├── ToolExecutionError          # Tool raised an exception
│     .tool_name: str
├── ToolNotFoundError           # Tool not in registry
│     .tool_name: str
├── ToolValidationError         # Parameter validation failed
│     .tool_name: str
├── BudgetExhaustedError        # Exceeded iterations/tokens/time
│     .budget_type: str
│     .limit: int | float
├── AgentTimeoutError           # Wall-clock timeout exceeded
│     .timeout_seconds: float
├── ApprovalDeniedError         # Human denied tool approval
│     .action_description: str
├── EvaluationError             # Evaluation function failed
│     .evaluator: str
└── DecompositionError          # Task decomposition failed
      .depth: int
```

**Usage:**

```python
from fsm_llm_agents import AgentError, BudgetExhaustedError

try:
    result = agent.run("complex task")
except BudgetExhaustedError as e:
    print(f"Ran out of {e.budget_type} (limit: {e.limit})")
except AgentError as e:
    print(f"Agent failed: {e}")
    print(f"Details: {e.details}")
```

---

## Architecture

All 11 agents share the same execution model:

```
agent.run(task)
  ├── 1. Build FSM definition (auto-generated from pattern + config)
  ├── 2. Create fsm_llm.API instance
  ├── 3. Register handlers (tool execution, budget limits, pattern-specific)
  ├── 4. Start conversation with initial context
  ├── 5. Loop: converse → check budget → pattern-specific logic
  ├── 6. Extract answer from final context
  └── 7. Return AgentResult(answer, success, trace, final_context)
```

**FSM auto-generation:** Each pattern has a `build_*_fsm()` function in `fsm_definitions.py` that generates a valid `FSMDefinition` dict at runtime. No JSON files needed.

**Handler system:** Agents use fsm_llm's handler framework for:
- **Tool execution** — `POST_TRANSITION` on act/execute states
- **Budget enforcement** — `PRE_TRANSITION` on every state change
- **HITL approval** — `CONTEXT_UPDATE` when `tool_name` changes
- **Pattern logic** — e.g., reflection handler, debate judge, revision tracker

**Budget enforcement:** All agents use `Defaults.FSM_BUDGET_MULTIPLIER` (default: 3) to compute the hard iteration ceiling: `max_iterations * FSM_BUDGET_MULTIPLIER`. This prevents runaway FSMs while giving each cycle enough transitions to complete.

**Context flow:** All data passes through a context dict with standardized keys from `ContextKeys`. Each handler reads and writes specific keys, and the final context is returned (with internal `_`-prefixed keys stripped).

---

## Examples

All examples are in `examples/agents/` and support OpenAI (default `gpt-4o-mini`) and Ollama fallback:

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

---

## License

GPL-3.0-or-later. See the [main project](https://github.com/NikolasMarkou/fsm_llm) for details.
