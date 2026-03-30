# fsm_llm_agents -- Agentic Patterns

12+ agent patterns with tool use, human-in-the-loop, structured output, working memory, and a meta-builder. Each pattern is backed by an auto-generated FSM.

- **Version**: 0.3.0 (synced from fsm_llm)
- **Extra deps**: None beyond core fsm_llm
- **Install**: `pip install fsm-llm[agents]`

## File Map

```
fsm_llm_agents/
├── base.py                 # BaseAgent ABC -- shared conversation loop, budgets, __call__, structured output
├── react.py                # ReactAgent -- ReAct loop with auto-generated FSM and tool dispatch
├── tools.py                # ToolRegistry + @tool decorator (auto-schema from type hints) + register_agent()
├── skills.py               # SkillDefinition + SkillLoader -- directory-based plugin loading
├── memory_tools.py         # create_memory_tools(WorkingMemory) -- remember, recall, forget, list_memories
├── hitl.py                 # HumanInTheLoop -- approval gates, escalation, confidence thresholds, timeouts
├── handlers.py             # AgentHandlers -- execute_tool(), check_iteration_limit(), check_approval()
├── fsm_definitions.py      # build_react_fsm() + 10 pattern-specific FSM builders
├── prompts.py              # Prompt builders for think/act/conclude/approval states
├── definitions.py          # ToolDefinition, ToolCall, ToolResult, AgentStep, AgentTrace, AgentConfig, AgentResult, ApprovalRequest
├── constants.py            # AgentStates, ContextKeys (30+), HandlerNames, HandlerPriorities, Defaults
├── exceptions.py           # AgentError hierarchy + MetaBuilderError hierarchy
├── __main__.py             # CLI: python -m fsm_llm_agents --info
├── __version__.py          # Imports from fsm_llm.__version__
├── __init__.py             # Public exports + create_agent() factory
│
│ -- Agent pattern implementations --
├── adapt.py                # ADaPTAgent -- adaptive complexity with decomposition
├── debate.py               # DebateAgent -- multi-perspective debate with judge
├── evaluator_optimizer.py  # EvaluatorOptimizerAgent -- iterative eval/optimize loop
├── maker_checker.py        # MakerCheckerAgent -- draft-review verification
├── orchestrator.py         # OrchestratorAgent -- worker delegation and synthesis
├── plan_execute.py         # PlanExecuteAgent -- plan decomposition + sequential execution
├── prompt_chain.py         # PromptChainAgent -- sequential prompt pipeline with gates
├── reasoning_react.py      # ReasoningReactAgent -- ReAct + structured reasoning (requires fsm_llm_reasoning)
├── reflexion.py            # ReflexionAgent -- self-reflection with memory
├── rewoo.py                # REWOOAgent -- planning-first tool execution
├── self_consistency.py     # SelfConsistencyAgent -- multiple samples with voting
│
│
│ -- Multi-agent coordination, integrations, SOPs --
├── swarm.py                # SwarmAgent -- emergent coordination with dynamic agent handoffs
├── agent_graph.py          # AgentGraph + AgentGraphBuilder -- DAG-based agent orchestration with conditional edges
├── mcp.py                  # MCPToolProvider -- MCP server integration for tool discovery (requires fsm-llm[mcp])
├── remote.py               # AgentServer + RemoteAgentTool -- A2A HTTP protocol (requires fsm-llm[a2a])
├── semantic_tools.py       # SemanticToolRegistry -- embedding-based tool retrieval via litellm
├── sop.py                  # SOPDefinition + SOPRegistry + load_builtin_sops() -- YAML/JSON SOP management
│
│ -- Meta-builder (interactive artifact creation) --
├── meta_builder.py         # MetaBuilderAgent -- routes to FSM/workflow/agent builders
├── meta_builders.py        # FSMBuilder, WorkflowBuilder, AgentBuilder -- automated generation
├── meta_cli.py             # CLI entry for fsm-llm-meta command
├── meta_tools.py           # create_fsm_tools(), create_workflow_tools(), create_agent_tools()
├── meta_fsm.py             # 3-state FSM (INTAKE → REVIEW → OUTPUT) with classification_extractions
├── meta_prompts.py         # Intake, build spec, review, revision prompts
└── meta_output.py          # format_artifact_json(), format_summary(), save_artifact()
```

## Key Classes

### BaseAgent (`base.py`) -- ABC for all agents

- Constructor: `__init__(model, tools, system_prompt, max_iterations, timeout, hitl, config)`
- `run(task, context=None)` → AgentResult
- `__call__(task)` → AgentResult (shorthand)
- Budget enforcement: hard limit = `max_iterations * 1.5`, timeout via threading
- Structured output: `config.output_schema` (Pydantic BaseModel) validated at conclusion

### Agent Patterns

| Pattern | Class | Key Mechanism |
|---------|-------|---------------|
| ReAct | `ReactAgent` | Think → Act (tool) → Observe → loop or Conclude |
| REWOO | `REWOOAgent` | Plan all tools upfront → Execute sequentially |
| Reflexion | `ReflexionAgent` | Attempt → Reflect on failure → Retry with memory |
| Plan-Execute | `PlanExecuteAgent` | Decompose → Execute steps → Replan if needed |
| Prompt Chain | `PromptChainAgent` | Sequential prompts with quality gates between |
| Self-Consistency | `SelfConsistencyAgent` | N samples → Majority vote |
| Debate | `DebateAgent` | Proposer vs Critic → Judge synthesizes |
| Orchestrator | `OrchestratorAgent` | Delegate subtasks to worker agents |
| ADaPT | `ADaPTAgent` | Estimate complexity → Decompose if too complex |
| Eval-Optimize | `EvaluatorOptimizerAgent` | Generate → Evaluate → Optimize loop |
| Maker-Checker | `MakerCheckerAgent` | Draft → Review → Revise if needed |
| Reasoning-ReAct | `ReasoningReactAgent` | ReAct + structured reasoning engine |
| Swarm | `SwarmAgent` | Emergent coordination -- agents hand off to each other dynamically |
| Agent Graph | `AgentGraph` | DAG-based orchestration with conditional edges (built via `AgentGraphBuilder`) |

### Multi-Agent Coordination & Integrations

- **SwarmAgent** (`swarm.py`) -- Agents hand off to each other by setting `next_agent` in `final_context`. Constructor: `SwarmAgent(agents={name: agent}, entry_agent="name", max_handoffs=10)`. Properties: `agents`, `entry_agent`. Method: `add_agent(name, agent)`
- **AgentGraph** (`agent_graph.py`) -- DAG execution with BFS traversal. Built via `AgentGraphBuilder().add_node(name, agent).add_edge(src, dst, condition=fn).set_entry(name).build()`. Validates no cycles, rejects with error suggesting SwarmAgent for cyclic patterns. Properties: `nodes`, `entry`. Methods: `get_edges(node)`, `get_terminal_nodes()`
- **MCPToolProvider** (`mcp.py`) -- Connects to MCP servers, discovers tools, converts to ToolDefinitions. Factory: `from_stdio(command, args)`, `from_url(url)`. Methods: `discover_tools()` (async), `register_tools(registry)`. Static: `create_mock_tool()` for testing. Requires `pip install fsm-llm[mcp]`
- **AgentServer** (`remote.py`) -- Wraps any agent as HTTP endpoint with `/invoke`, `/stream` (SSE), `/health`, `/info`. Constructor: `AgentServer(agent, host, port, timeout)`. Property: `app` (FastAPI). Method: `run()`. Requires `fastapi`
- **RemoteAgentTool** (`remote.py`) -- Wraps remote agent URL as local tool. Methods: `invoke(task)`, `ainvoke(task)` (async), `to_tool_definition()`, `health_check()`. Requires `pip install fsm-llm[a2a]`
- **SemanticToolRegistry** (`semantic_tools.py`) -- Extends `ToolRegistry` with embedding-based retrieval. Constructor: `SemanticToolRegistry(embedding_model, top_k, auto_embed)`. Methods: `retrieve(query, top_k)`, `rebuild_embeddings()`, `to_prompt_description(query)`. Falls back to full list for <20 tools. Uses litellm embeddings
- **SOPDefinition** (`sop.py`) -- Reusable agent config template: name, agent_pattern, task_template, required_tools, output_schema, config_overrides. Methods: `render_task(**vars)`, `to_agent_config()`, `to_dict()`, `from_dict()`
- **SOPRegistry** (`sop.py`) -- Registry for SOPs. Methods: `register(sop)`, `register_from_file(path)`, `register_directory(path)`, `get(name)`, `list_sops()`, `list_names()`, `has(name)`, `remove(name)`
- **load_builtin_sops()** (`sop.py`) -- Returns SOPRegistry with 3 built-in SOPs: code-review, summarize, data-extraction

### create_agent() Factory (`__init__.py`)

`create_agent(pattern, model, tools, **kwargs)` → BaseAgent. Pattern names: "react", "rewoo", "reflexion", "plan_execute", "prompt_chain", "self_consistency", "debate", "orchestrator", "adapt", "evaluator_optimizer", "maker_checker", "reasoning_react"

### ToolRegistry (`tools.py`)

- `register(tool_or_func)` -- Register @tool-decorated function or ToolDefinition
- `execute(name, arguments)` → ToolResult
- `list_tools()` → list[ToolDefinition]
- `get_tool(name)` → ToolDefinition
- `get_json_schemas()` → list[dict] (for LLM function calling)
- `register_agent(registry, agent, name, description)` -- Register another agent as a tool

### @tool Decorator (`tools.py`)

Auto-generates JSON schema from type hints and docstrings. Supports: str, int, float, bool, list, dict, Optional, Literal, Enum.

### HumanInTheLoop (`hitl.py`)

- Constructor: `__init__(approval_callback, require_approval_for, confidence_threshold, timeout, escalation_callback)`
- `request_approval(ApprovalRequest)` → bool
- Confidence-based auto-approve: above threshold → skip human
- Timeout: raises ApprovalDeniedError after N seconds

### SkillLoader (`skills.py`)

- `load_directory(path)` → list[SkillDefinition]
- `load_function(func)` → SkillDefinition
- Skills convert to tools via `skill.to_tool()`

## Data Models (`definitions.py`)

- **AgentConfig**: output_schema (Pydantic BaseModel), custom settings
- **AgentResult**: answer (str), success (bool), trace (AgentTrace), structured_output (BaseModel | None)
- **AgentTrace**: steps list, tool_calls list, total_iterations, execution_time
- **AgentStep**: step_type, content, tool_call (optional), observation (optional)
- **ToolDefinition**: name, description, parameters (JSON schema), function (callable)
- **ToolCall**: tool_name, arguments dict
- **ToolResult**: tool_name, result (str), success (bool), error (optional)
- **ApprovalRequest**: tool_name, arguments, context, confidence

## Constants (`constants.py`)

- **AgentStates**: THINK, ACT, OBSERVE, CONCLUDE, APPROVE, ERROR (+ pattern-specific states)
- **ContextKeys**: 30+ keys (TASK, TOOLS, THOUGHTS, ACTION, OBSERVATION, ANSWER, ITERATION_COUNT, etc.)
- **Defaults**: MAX_ITERATIONS=10, TIMEOUT=300, HARD_LIMIT_MULTIPLIER=1.5

## Handlers (`handlers.py`)

- `execute_tool(context)` -- Dispatches tool call from context, accumulates observations
- `check_iteration_limit(context)` -- Enforces budget, forces conclude at limit
- `check_approval(context)` -- Routes to HITL approval if required

## FSM Builders (`fsm_definitions.py`)

- `build_react_fsm(tools, hitl)` -- THINK → ACT → OBSERVE → CONCLUDE (+ APPROVE if HITL)
- Pattern-specific: `build_rewoo_fsm()`, `build_reflexion_fsm()`, `build_plan_execute_fsm()`, etc.
- Each generates a complete FSMDefinition dict from ToolRegistry

## Testing

```bash
pytest tests/test_fsm_llm_agents/  # 706 tests
pytest tests/test_fsm_llm_meta/    # 205 tests, 11 files
```

## Exceptions

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
