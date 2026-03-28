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
│ -- Meta-builder (interactive artifact creation) --
├── meta_builder.py         # MetaBuilderAgent -- routes to FSM/workflow/agent builders
├── meta_builders.py        # FSMBuilder, WorkflowBuilder, AgentBuilder -- automated generation
├── meta_cli.py             # CLI entry for fsm-llm-meta command
├── meta_tools.py           # create_fsm_tools(), create_workflow_tools(), create_agent_tools()
├── meta_fsm.py             # FSM definitions for meta-agent classification routing
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
pytest tests/test_fsm_llm_agents/  # 645 tests, 27 files
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
