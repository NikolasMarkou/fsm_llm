# fsm_llm.stdlib.agents — Agentic Patterns

The agents subpackage. Two coexisting layers:

- **λ-term factories** (M3 slice 1): four canonical shapes proven by 46 M4-migrated examples — `react_term`, `rewoo_term`, `reflexion_term`, `memory_term`. Pure; close over no Python state.
- **Class-based agent patterns**: 12+ patterns (`ReactAgent`, `REWOOAgent`, …) plus multi-agent coordination (Swarm, Agent Graph), integrations (MCP, A2A), SOPs, semantic tools, meta builder.

The legacy `fsm_llm_agents` import path resolves here via `sys.modules` shim.

- **Version**: 0.3.0 (synced from `fsm_llm`)
- **Extra deps**: None beyond core `fsm_llm`
- **Install**: `pip install fsm-llm[agents]`

## Layer 1 — λ-term Factories (`lam_factories.py`)

Each factory takes prompt strings + env-var names; builds `Leaf` nodes internally; returns a closed `Term`. Imports only from `fsm_llm.lam` (purity invariant; AST-walk test enforces).

| Factory | Shape | Oracle calls | Body |
|---------|-------|---|------|
| `react_term` | `let_("decision", decide_leaf, let_("observation", app(var(tool_dispatch_var), var("decision")), synth_leaf))` | 2 | decide → tool → synthesise |
| `rewoo_term` | `let_("plan", plan_leaf, let_("evidence", app(var(plan_exec_var), var("plan")), synth_leaf))` | 2 | plan → execute → synthesise |
| `reflexion_term` | 4-leaf let-chain | 4 | solve → eval → reflect → re-solve (depth-1 retry flatten) |
| `memory_term` | `let_("context", ctx_leaf, ans_leaf)` | 2 | context retrieval → answer |

Caller binds host callables (e.g. `tool_dispatch`) in env at execution time. See `examples/pipeline/_helpers.py` for runtime glue (`run_pipeline`, `make_tool_dispatcher`, `make_plan_executor`).

```python
from fsm_llm.stdlib.agents import react_term

term = react_term(
    decide_prompt="Decide which tool to call for: {question}",
    synth_prompt="Synthesise an answer using: {observation}",
    decision_schema=ToolDecision,
)
ex.run(term, env={"question": "...", "tool_dispatch": tool_runner})
assert ex.oracle_calls == 2
```

## Layer 2 — Class-based agents (file map)

```
agents/
├── lam_factories.py       # M3 slice 1 — 4 named term factories
├── base.py                # BaseAgent ABC — shared conversation loop, budgets, __call__, structured output
├── react.py               # ReactAgent — ReAct loop with auto-generated FSM and tool dispatch
├── tools.py               # ToolRegistry + @tool decorator (auto-schema from type hints) + register_agent()
├── skills.py              # SkillDefinition + SkillLoader — directory-based plugin loading
├── memory_tools.py        # create_memory_tools(WorkingMemory) — remember, recall, forget, list_memories
├── hitl.py                # HumanInTheLoop — approval gates, escalation, confidence thresholds, timeouts
├── handlers.py            # AgentHandlers — execute_tool(), check_iteration_limit(), check_approval()
├── fsm_definitions.py     # build_react_fsm() + 10 pattern-specific FSM builders
├── prompts.py             # Prompt builders for think/act/conclude/approval states
├── definitions.py         # ToolDefinition, ToolCall, ToolResult, AgentStep, AgentTrace, AgentConfig, AgentResult, ApprovalRequest
├── constants.py           # AgentStates, ContextKeys (30+), HandlerNames, HandlerPriorities, Defaults
├── exceptions.py          # AgentError hierarchy + MetaBuilderError hierarchy
├── __main__.py            # CLI: python -m fsm_llm.stdlib.agents --info
├── __version__.py         # Imports from fsm_llm.__version__
├── __init__.py            # Public exports + create_agent() factory
│
│ -- Agent pattern implementations --
├── adapt.py               # ADaPTAgent — adaptive complexity with decomposition
├── debate.py              # DebateAgent — multi-perspective debate with judge
├── evaluator_optimizer.py # EvaluatorOptimizerAgent — iterative eval/optimize loop
├── maker_checker.py       # MakerCheckerAgent — draft-review verification
├── orchestrator.py        # OrchestratorAgent — worker delegation and synthesis
├── plan_execute.py        # PlanExecuteAgent — plan decomposition + sequential execution
├── prompt_chain.py        # PromptChainAgent — sequential prompt pipeline with gates
├── reasoning_react.py     # ReasoningReactAgent — ReAct + structured reasoning (requires reasoning extra)
├── reflexion.py           # ReflexionAgent — self-reflection with memory
├── rewoo.py               # REWOOAgent — planning-first tool execution
├── self_consistency.py    # SelfConsistencyAgent — multiple samples with voting
│
│ -- Multi-agent coordination, integrations, SOPs --
├── swarm.py               # SwarmAgent — emergent coordination with dynamic agent handoffs
├── agent_graph.py         # AgentGraph + AgentGraphBuilder — DAG-based orchestration with conditional edges
├── mcp.py                 # MCPToolProvider — MCP server integration (requires fsm-llm[mcp])
├── remote.py              # AgentServer + RemoteAgentTool — A2A HTTP protocol (requires fsm-llm[a2a])
├── semantic_tools.py      # SemanticToolRegistry — embedding-based tool retrieval via litellm
├── sop.py                 # SOPDefinition + SOPRegistry + load_builtin_sops() — YAML/JSON SOP management
│
│ -- Meta-builder (interactive artifact creation) --
├── meta_builder.py        # MetaBuilderAgent — routes to FSM/workflow/agent builders
├── meta_builders.py       # FSMBuilder, WorkflowBuilder, AgentBuilder — automated generation
├── meta_cli.py            # CLI entry for fsm-llm-meta command
├── meta_tools.py          # create_fsm_tools(), create_workflow_tools(), create_agent_tools()
├── meta_fsm.py            # 3-state FSM (INTAKE → REVIEW → OUTPUT) with classification_extractions
├── meta_prompts.py        # Intake, build spec, review, revision prompts
└── meta_output.py         # format_artifact_json(), format_summary(), save_artifact()
```

## Key Classes

- **Tooling**: `@tool` decorator (schema inferred from type hints + docstring), `ToolRegistry`, `register_agent()` (treat any agent as a tool).
- **`BaseAgent`**: `run(task) → AgentResult`, `__call__(task)`, structured-output via Pydantic, budget enforcement.
- **`HumanInTheLoop`**: `request_approval(action, context, timeout)`, `should_escalate(confidence)`.
- **`SwarmAgent`**: `add_agent(name, agent)`, `run(task)` — dynamic handoffs via `transfer_to(target)`.
- **`AgentGraph`** + **`AgentGraphBuilder`**: `add_node`, `add_edge(condition=...)`, `add_human_loop`, parallel execution.
- **`SkillLoader`**: `load_skills(directory)`, `register_with_agent(agent)`.
- **`SOPRegistry`**: load YAML/JSON SOP templates, instantiate as configured agents.
- **Multi-agent integrations**: `MCPToolProvider` (MCP), `AgentServer` + `RemoteAgentTool` (A2A HTTP).
- **Meta-builder**: `MetaBuilderAgent` (interactive), `FSMBuilder` / `WorkflowBuilder` / `AgentBuilder` (programmatic).

## Pattern-class catalog

| Pattern | Class | Description |
|---------|-------|-------------|
| ReAct | `ReactAgent` | Think → Act → Observe loop |
| ADaPT | `ADaPTAgent` | Adaptive decomposition by complexity |
| Plan-Execute | `PlanExecuteAgent` | Plan → execute steps |
| REWOO | `REWOOAgent` | Reasoning Without Observation (plan first) |
| Reflexion | `ReflexionAgent` | Self-reflection with episodic memory |
| Self-Consistency | `SelfConsistencyAgent` | Sample N times, vote |
| Debate | `DebateAgent` | Multi-perspective + judge |
| Orchestrator | `OrchestratorAgent` | Delegate to workers + synthesise |
| Eval-Optimize | `EvaluatorOptimizerAgent` | Iterative refinement |
| Maker-Checker | `MakerCheckerAgent` | Draft + review |
| Prompt-Chain | `PromptChainAgent` | Sequential prompts with gates |
| Reasoning-ReAct | `ReasoningReactAgent` | ReAct + structured reasoning steps |

`create_agent(pattern, model, tools, ...)` is the factory.

## Constants (`constants.py`)

- `AgentStates`: `THINK`, `ACT`, `OBSERVE`, `CONCLUDE`, `APPROVE`, `REFLECT`, ...
- `ContextKeys`: 30+ keys for tools, observations, decisions, etc.
- `Defaults`: `MAX_ITERATIONS = 10`, `TOOL_TIMEOUT = 30s`, `APPROVAL_TIMEOUT = 300s`, etc.

## Exceptions

```
AgentError
├── ToolExecutionError(tool_name, original_error)
├── ToolNotFoundError(tool_name)
├── ToolValidationError(field_name)
├── BudgetExhaustedError(budget_type, limit)
├── ApprovalDeniedError(action, reason)
├── AgentTimeoutError(timeout_seconds)
├── EvaluationError
├── DecompositionError(task)
└── MetaBuilderError
    ├── BuilderError(builder_type)
    ├── MetaValidationError
    └── OutputError
```

## Testing

```bash
pytest tests/test_fsm_llm_agents/                           # 723 tests (class-based + factories)
pytest tests/test_fsm_llm_agents/test_lam_factories.py      # M3 slice 1 unit tests (factory shape + purity)
pytest tests/test_fsm_llm_meta/                             # Meta-builder tests
```

- Test classes: `class Test<Feature>`. Mock LLMs via `Mock(spec=LLMInterface)`.
- Live smoke for factories gated by `TEST_REAL_LLM=1` on `ollama_chat/qwen3.5:4b`. Asserts `ex.oracle_calls == M4_baseline` (2/2/4/2 for the four shapes).

## Code Conventions

- **Stdlib purity**: `lam_factories.py` imports only from `fsm_llm.lam`.
- **Tool dispatch**: bind a callable in env (`tool_dispatch`) — never close over it inside the factory.
- **Structured output**: pass a Pydantic `schema` to `leaf()`. The `LiteLLMOracle` handles the `temperature=0` + `required` synth automatically (LESSONS.md M4 D-011).
- **Class-based vs factory**: choose by need. Class-based gives lifecycle + auto-FSM with budgets and HITL; factory gives a pure λ-term you can compose into bigger structures (M3 stdlib trinity pattern).
