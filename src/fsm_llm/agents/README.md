# fsm_llm.agents

Ready-made AI agent patterns for FSM-LLM, found at `src/fsm_llm/agents`: agents that call tools, plan, reflect, debate, vote, or hand work to other agents. It also includes a "meta-builder" that turns a plain description into an FSM, workflow, or agent definition.

## What it is for

An agent is an LLM that works on a task over several steps, often calling tools (your Python functions) along the way. Each pattern here is a known way to organise those steps. ReAct is the main one: think, call a tool, look at the result, repeat. Most patterns are built as an FSM-LLM finite state machine (FSM): a fixed set of states such as `think`, `act` and `conclude`, with rules for moving between them. That keeps the loop structured and bounded, which matters most with small local models. Every pattern has the same `agent.run(task)` call, which returns an answer, a success flag, a trace of tool calls, and the final context.

## How it works

```mermaid
flowchart TD
    T[agent.run task] --> B[Build an FSM for this pattern]
    B --> A[Create an FSM-LLM API and register handlers]
    A --> L{Loop: send 'Continue.'}
    L --> TH[think: LLM picks a tool and its input]
    TH --> AC[act: handler runs the tool, stores the observation]
    AC --> L
    TH -- done, and a tool has run --> C[conclude: LLM writes the final answer]
    C --> R[AgentResult: answer, success, trace, final_context]
```

- The agent sends the FSM the message `Continue.` again and again until it reaches a final state.
- Tools run inside handlers (hooks that FSM-LLM calls at fixed points), not inside the LLM. A tool failure is shown to the model as an observation marked `[TOOL FAILED]` instead of crashing the run.
- Limits stop runaway loops: `max_iterations` (default 10; the hard ceiling is three times that in FSM turns, which raises `BudgetExhaustedError`) and `timeout_seconds` (default 300, raises `AgentTimeoutError`).
- The ReAct family cannot conclude before a tool has run unless the loop was forced to stop, so a small model cannot answer from memory on turn one.
- `success` is `True` only when the run produced a real answer or ran at least one tool. Planner patterns (orchestrator, ReWOO, plan and execute) must also show real executed work.

## Patterns

| Pattern | Class | Idea |
| --- | --- | --- |
| ReAct | `ReactAgent` | Think, call a tool, observe, repeat, then answer |
| ReWOO | `REWOOAgent` | Plan every tool call first (`#E1`, `#E2` refer to earlier results), run them, then answer |
| Reflexion | `ReflexionAgent` | ReAct plus self-evaluation, reflection and retry |
| Plan and execute | `PlanExecuteAgent` | Break the task into steps, run each, replan on failure |
| Prompt chain | `PromptChainAgent` | A fixed series of prompts, with optional validation gates |
| Self-consistency | `SelfConsistencyAgent` | Answer several times at different temperatures, take the majority |
| Debate | `DebateAgent` | Proposer and critic argue, a judge decides |
| Orchestrator | `OrchestratorAgent` | Split into subtasks and hand them to workers |
| ADaPT | `ADaPTAgent` | Try directly; if that fails, break the task down and recurse |
| Evaluator-optimizer | `EvaluatorOptimizerAgent` | Generate, score with your function, refine |
| Maker-checker | `MakerCheckerAgent` | One role drafts, another reviews, revise until good |
| Reasoning ReAct | `ReasoningReactAgent` | ReAct plus a built-in `reason` tool that runs the structured reasoning engine (needs `fsm_llm.reasoning`) |
| Parallel ReAct | `ParallelReactAgent` | Several tool calls per step, run in a thread pool |
| Verified ReAct | `VerifiedReactAgent` | Check the answer with your function and retry |
| Auto-memory ReAct | `AutoMemoryReactAgent` | Recall related memories before a run, save the exchange after |
| Native function calling | `NativeFunctionCallingReactAgent` | Uses the provider's own tool-calling API instead of an FSM |
| Swarm | `SwarmAgent` | Agents pass the task to each other |
| Agent graph | `AgentGraph` | Agents wired as a graph with conditional edges, no cycles |

## Files

- `base.py` - `BaseAgent`: the shared loop, limits, answer extraction, trace, structured output.
- `react.py`, `rewoo.py`, `reflexion.py`, `plan_execute.py`, `prompt_chain.py`, `self_consistency.py`, `debate.py`, `orchestrator.py`, `adapt.py`, `evaluator_optimizer.py`, `maker_checker.py`, `reasoning_react.py`, `parallel_react.py`, `verified_react.py`, `auto_memory.py`, `native_fc.py`, `swarm.py`, `agent_graph.py` - one pattern each.
- `fsm_definitions.py` - builds the FSM for each pattern. `prompts.py` - the instructions each state gives the LLM.
- `tools.py` - `ToolRegistry` and the `@tool` decorator. `tool_registries.py` - caching and retrying registries. `semantic_tools.py` - picks relevant tools by embedding similarity.
- `handlers.py` - runs tools, enforces the iteration limit, and checks human approval. `hitl.py` - human approval before sensitive tools.
- `memory_tools.py`, `semantic_memory.py`, `memory_persistence.py`, `summarization.py`, `truncation.py` - working-memory tools, long-term embedding memory, saving memory to disk, condensing old observations, shortening long tool output.
- `composition.py` - use ReAct agents as orchestrator workers; an LLM judge for the evaluator-optimizer.
- `skills.py`, `sop.py` - load tools from folders of Python files; reusable task templates (three built in).
- `mcp.py` - load tools from an MCP (Model Context Protocol) server. `remote.py` - serve an agent over HTTP, or call a remote one as a tool.
- `meta_builder.py`, `meta_builders.py`, `meta_tools.py`, `meta_prompts.py`, `meta_output.py`, `meta_cli.py`, `meta_fsm.py` - the meta-builder, its builders, its `fsm-llm-meta` command, and a legacy FSM stub.
- `definitions.py`, `constants.py`, `exceptions.py` - data models, names and defaults, errors.
- `__init__.py`, `__main__.py`, `__version__.py` - exports and `create_agent()`, `python -m fsm_llm.agents --info`, version.

## How to use it

```python
from fsm_llm.agents import AgentConfig, ReactAgent, ToolRegistry, tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

registry = ToolRegistry()
registry.register(add._tool_definition)

agent = ReactAgent(tools=registry, config=AgentConfig(model="ollama_chat/qwen3.5:4b"))
result = agent.run("What is 17 + 25?")
print(result.answer, result.success, [c.tool_name for c in result.trace.tool_calls])
```

Or with the factory, which accepts a list of `@tool` functions:

```python
from fsm_llm.agents import create_agent
agent = create_agent(tools=[add], pattern="react")
print(agent("What is 2 + 3?").answer)
```

Build a new FSM by chatting (say "build it" when ready):

```bash
fsm-llm-meta --model ollama_chat/qwen3.5:4b --output my_bot.json
```

## Things to know

- `pip install "fsm-llm[agents]"` adds no dependencies. `mcp.py` needs `fsm-llm[mcp]`; `RemoteAgentTool` needs `fsm-llm[a2a]` (httpx); `AgentServer` needs fastapi (for example via `fsm-llm[monitor]`). `ReasoningReactAgent` needs `fsm_llm.reasoning`.
- `ReactAgent`, `REWOOAgent`, `ReflexionAgent`, `ParallelReactAgent` and `NativeFunctionCallingReactAgent` refuse an empty tool registry.
- Each `run()` makes many LLM calls. Small models can hit the iteration limit and fail with `BudgetExhaustedError`.
- If the model names a tool that does not exist, it is told which tools exist and the turn counts as a turn without a tool call, not as evidence of work. The bad name is cleared so the next turn can pick a real tool. After three turns in a row with no tool, the run is forced to conclude.
- Human approval works in `ReactAgent`, `ReflexionAgent` and `ReasoningReactAgent` with `HumanInTheLoop(approval_policy=..., approval_callback=...)`. A tool the policy gates runs only after the callback approved that exact call, and each approval covers one call. The model cannot approve a call itself: the grant is stored under an internal key the model cannot write. If a tool needs approval and no callback is set, the run fails with `ApprovalDeniedError` (reported as `AgentError`). Known gaps: the per-tool `requires_approval` flag does nothing without an `approval_policy`; `ParallelReactAgent`, `REWOOAgent`, `PlanExecuteAgent` and `NativeFunctionCallingReactAgent` have no approval at all; the policy sees the call before an empty input is filled from the task text.
- `AgentServer` is unauthenticated by default. Pass `api_key=` to require `Authorization: Bearer <key>` or `X-API-Key` on `/invoke` and `/stream`; `RemoteAgentTool(api_key=...)` sends it. Input over `max_input_chars` (default 100,000) gets 413. At most `max_concurrent` runs (default 8) are in flight; more get 503. A failing run returns a generic error with an `error_id`; the exception is only logged. There is no per-client rate limiting. `RemoteAgentTool` raises `ToolExecutionError` when the server reports `success: false`. `/stream` sends one event after the run finishes; it does not stream tokens.
- Keys in the returned `final_context` that look internal (starting with `_`, `system_`, `internal_` or `__`) are removed by the FSM-based patterns. `SwarmAgent` and `AgentGraph` add their own `_swarm_*` and `_graph_*` keys on top of the last agent's result.
- `EvaluatorOptimizerAgent` and `MakerCheckerAgent` need arguments `create_agent` cannot guess (an evaluation function; maker and checker instructions). Pass them yourself.
- Maker and checker instructions and the three `DebateAgent` personas go into the prompts as written, with no sanitizing. Write them yourself; put user input in the task.
- `MakerCheckerAgent` and `EvaluatorOptimizerAgent` judge every new draft. While the model rewrites, the old draft is kept under `previous_draft` / `previous_output`, which later prompts can see but which is dropped from `final_context`. When the pass is forced (the quality score reaches the threshold, the revision limit is hit, or the iteration budget runs out), the draft that was just judged is the one returned.
- `OrchestratorAgent` without a `worker_factory` only records placeholder results, so it always reports `success=False`. `PlanExecuteAgent` without tools cannot mark a step as executed, so it also reports `success=False`.
- `SemanticToolRegistry` returns every tool when fewer than 10 are registered.
- MCP servers get a 30-second timeout per tool discovery and per tool call by default (`timeout=None` turns it off). Each call starts a new connection to the server, and that start counts toward the timeout.
- The meta-builder makes one LLM call to extract the whole design, then assembles it in Python. For workflows and agents, `is_valid` means the spec is complete, not that it loads as a runnable object; you still wire in the Python functions.
- `save_artifact` writes wherever it is told; check paths that come from untrusted input.
