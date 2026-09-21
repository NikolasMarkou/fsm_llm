# fsm_llm_agents

Path: `src/fsm_llm_agents`
Purpose: Agent patterns (ReAct and 17 others) built as generated FSM-LLM FSMs driven by a shared loop, plus tools, HITL, memory, multi-agent coordination, MCP/A2A integration, and the meta-builder.

## Scope

All agent code for the `fsm-llm` distribution (extra `agents`, no third-party deps; optional `mcp`, `a2a`=httpx, fastapi for `AgentServer`). Version from `fsm_llm.__version__`. Consumers: `fsm_llm_monitor` (launches 7 agent types, Builder page uses `MetaBuilderAgent`), `fsm_llm_harness` (uses `NativeFunctionCallingReactAgent` and the tool registry), examples under `examples/agents/` and `examples/meta/`. Console script `fsm-llm-meta = fsm_llm_agents.meta_cli:main_cli`.

## Architecture

```mermaid
sequenceDiagram
    participant U as caller
    participant A as XAgent.run
    participant B as BaseAgent._standard_run
    participant API as fsm_llm.API
    U->>A: run(task, initial_context)
    A->>A: fsm_def = build_x_fsm(...); ctx = _init_context(task, ..., extra)
    A->>B: _standard_run(task, fsm_def, ctx, agent_type, handlers=?)
    B->>API: _create_api(fsm_def) (model/temperature/max_tokens from AgentConfig)
    B->>API: _register_handlers + _register_lifecycle_handlers
    loop until has_conversation_ended
        B->>B: _check_budgets (timeout, iterations > max*3)
        B->>B: _on_loop_iteration (HITL)
        B->>API: converse("Continue.")
    end
    B->>API: get_data, end_conversation (finally)
    B-->>U: AgentResult(answer, success, trace, final_context filtered, structured_output)
```

ReAct FSM (`build_react_fsm`): `think` -> `act` (tool) -> `think` ... -> `conclude`; optional `await_approval` when HITL has a policy. Tools execute in a handler on `act` state entry (`AgentHandlers.execute_tool`); `check_iteration_limit` runs at PRE_TRANSITION. `think`/`act` have empty `response_instructions`, so Pass 2 is skipped for them. The pipeline treats a context carrying `agent_trace` as agent-managed (no post-transition extraction, bulk overwrite rules differ).

## Key files

| File | Role | Notes |
| --- | --- | --- |
| `base.py` | `BaseAgent` ABC | `_standard_run`, `_standard_run_stream`, `_run_conversation_loop`, `_extract_answer`, `_completion_is_real`, `_build_trace`, `_try_parse_structured_output`, `_filter_context`, `_create_api` |
| `fsm_definitions.py` | FSM builders | `build_react_fsm`, `build_rewoo_fsm`, `build_reflexion_fsm`, `build_plan_execute_fsm`, `build_prompt_chain_fsm`, `build_self_consistency_fsm`, `build_debate_fsm`, `build_orchestrator_fsm`, `build_adapt_fsm`, `build_evalopt_fsm`, `build_maker_checker_fsm` |
| `prompts.py` | Per-state extraction/response instruction builders | |
| `handlers.py` | `AgentHandlers(registry)` | `execute_tool`, `check_iteration_limit`, `classification_tool_override`, `reset` |
| `tools.py` | `ToolRegistry`, `tool`, `register_agent`, `normalize_tool_input` | thread-safe (`_tools_lock`) |
| `hitl.py` | `HumanInTheLoop`, `make_hitl_checker` | |
| `definitions.py` | Models | `AgentConfig`, `AgentResult`, etc. |
| `constants.py` | States per pattern, `ContextKeys`, `HandlerNames`, `HandlerPriorities`, `Defaults`, `MetaDefaults`, messages | |
| `native_fc.py` | `NativeFunctionCallingReactAgent` | own litellm loop with `tools=`, does NOT use the FSM pipeline |
| `meta_builder.py` | `MetaBuilderAgent` | classify-next-tool then extract-params loop over an `ArtifactBuilder` |
| `meta_builders.py` | `ArtifactBuilder`, `FSMBuilder`, `WorkflowBuilder`, `AgentBuilder` | |

## Public interface

- `BaseAgent(config=None, **api_kwargs)`: `run(task, initial_context=None) -> AgentResult` (abstract), `__call__(task, **kw)`. Subclasses implement `_register_handlers(api)` (React/ParallelReact widen with a required call-local `handlers`).
- Constructors:
  - `ReactAgent(tools, config=None, hitl=None, use_classification=False, **api_kwargs)`; also `run_stream(task) -> Iterator[str]`.
  - `REWOOAgent(tools, config)`, `ReflexionAgent(tools, config, evaluation_fn=None, max_reflections=3, hitl=None)`, `ParallelReactAgent(tools, config, max_parallel=4)`, `ReasoningReactAgent(tools, config, hitl, reasoning_model=None)` (adds a `reason` pseudo-tool; `AgentError` if `fsm_llm_reasoning` missing).
  - `PlanExecuteAgent(tools=None, config, max_replans=2)`, `ADaPTAgent(tools=None, config, max_depth=3)`, `OrchestratorAgent(worker_factory=None, tools=None, config, max_workers=5)`.
  - `PromptChainAgent(chain: list[ChainStep], config)`, `SelfConsistencyAgent(config, num_samples=5, aggregation_fn=None, max_workers=1)`, `DebateAgent(config, num_rounds=3, proposer_persona="", critic_persona="", judge_persona="")`.
  - `EvaluatorOptimizerAgent(evaluation_fn, config, max_refinements=3)`, `MakerCheckerAgent(maker_instructions, checker_instructions, config, max_revisions=3, quality_threshold=0.7)`.
  - `VerifiedReactAgent(*args, max_verify_retries=1, **kw)` (uses `config.verification_fn`, `reflect_every_n`), `AutoMemoryReactAgent(*args, memory=None, recall_k=3, recall_min_score=0.25, auto_remember=True, remember_only_on_success=False, enable_respond=True, **kw)`, `with_auto_memory`, `augment_task_with_memories`, `remember_interaction`.
  - `NativeFunctionCallingReactAgent(tools, config, complete_fn=None, system_policy=None, *, seed=None)`; `complete_fn(model, messages, tool_schemas) -> {"content", "tool_calls": [{"id","name","arguments"}]}`.
  - `SwarmAgent(agents: dict[str, BaseAgent], entry_agent, max_handoffs=10, memory=None, config)`: hands off via `final_context["next_agent"]`; `add_agent`, `agents`, `entry_agent`.
  - `AgentGraphBuilder().add_node(name, agent).add_edge(src, dst, condition=None).set_entry(name).build() -> AgentGraph` (rejects cycles); `AgentGraph.run`, `nodes`, `entry`, `get_edges`, `get_terminal_nodes`.
- `create_agent(system_prompt="You are a helpful assistant.", tools=None (list of @tool fns or ToolRegistry), pattern="react", **kwargs)`; patterns: `react, rewoo, debate, plan_execute, prompt_chain, self_consistency, orchestrator, adapt, evaluator_optimizer, maker_checker, reflexion, meta_builder, swarm, parallel_react, native_fc, verified_react, auto_memory` (+ `reasoning_react` if installed). Unknown -> `ValueError`.
- `ToolRegistry()`: `register(ToolDefinition) -> self`, `register_function(fn, name=None, description=None, parameter_schema=None, requires_approval=False) -> self`, `register_skill`, `register_agent(agent, name, description)`, `get(name)` (only method that raises `ToolNotFoundError`), `list_tools`, `tool_names` (property), `execute(ToolCall) -> ToolResult` (never raises), `to_prompt_description`, `get_json_schemas`, `to_classification_schema`, `__len__`, `__contains__`. `CachingToolRegistry(max_entries=256)`, `RetryingToolRegistry(max_retries=2, backoff_seconds=0.0)`, `SemanticToolRegistry(embedding_model, top_k, auto_embed)` (`retrieve`, `rebuild_embeddings`; full list under 20 tools).
- `@tool` / `@tool(name=, description=, parameter_schema=, requires_approval=)`: attaches `fn._tool_definition`; schema inferred from type hints (`Annotated[str, "desc"]` for descriptions).
- `HumanInTheLoop(approval_policy=None, approval_callback=None, on_escalation=None, confidence_threshold=0.3, approval_timeout=None)`: `requires_approval(call, ctx)`, `request_approval(call, ctx) -> bool` (raises `ApprovalDeniedError` without a callback; timeout = denied), `escalate`, `should_escalate_on_confidence`, `has_approval_policy`, `has_approval_callback`.
- Memory: `create_memory_tools(WorkingMemory)` (remember, recall, forget, list_memories); `SemanticMemoryStore(...)` (`add`, `search`, `forget`, `clear`, `save`, `load`, `to_dict`, `from_dict`), `create_semantic_memory_tools`, `MemoryEntry`; `MemorySessionStore` (a `fsm_llm.SessionStore`), `save_working_memory(mem, path)`, `load_working_memory(path)`; `make_observation_summarizer(n)`; `smart_truncate(text, max_length=2000)`.
- Skills/SOPs: `SkillDefinition` (`to_tool_definition`), `SkillLoader.from_directory`, `.from_functions`, `.to_tool_registry`, `.by_category`; `SOPDefinition` (`render_task`, `to_agent_config`, `to_dict`, `from_dict`), `SOPRegistry` (`register`, `register_from_dict`, `register_from_file`, `register_directory`, `get`, `list_sops`, `list_names`, `has`, `remove`), `load_builtin_sops()` (code-review, summarize, data-extraction).
- Integrations: `MCPToolProvider.from_stdio(command, args)` / `.from_url(url)`, async `discover_tools()`, `register_tools(registry) -> int`, `tools`, `get_tool_names`, `create_mock_tool`. `AgentServer(agent, host, port, timeout)` (`app`, `run`; routes `/invoke`, `/stream` SSE, `/health`, `/info`). `RemoteAgentTool(url, ...)` (`invoke`, async `ainvoke`, `to_tool_definition`, `health_check`, `url`).
- Composition: `react_worker_factory(...)` (ReAct workers for `OrchestratorAgent`), `default_llm_judge(...)`.
- Meta-builder: `MetaBuilderAgent(config: MetaBuilderConfig | None)`: `run(task) -> MetaBuilderResult`, turn-by-turn `start(initial_message="") -> str`, `send(msg) -> str`, `is_complete()`, `get_result()`, `get_internal_state()`, `run_interactive()`. Tool factories `create_fsm_tools`, `create_workflow_tools`, `create_agent_tools`, `create_builder_tools`; output helpers `format_artifact_json`, `format_summary`, `save_artifact`.
- CLIs: `fsm-llm-meta [--model] [--output/-o] [--temperature] [--max-turns]`; `python -m fsm_llm_agents [--info] [--version]` (no `--meta` flag despite a docstring in `meta_cli.py`).

## Data shapes

- `AgentConfig{model, max_iterations=10, timeout_seconds=300.0, temperature=0.5, max_tokens=1000, output_schema (pydantic class, excluded), transition_config (excluded), max_history_size=None, enable_prompt_cache=False (passes litellm caching=True), reflect_every_n, auto_summarize_after, verification_fn (excluded), force_final_tool}`.
- `AgentResult{answer, success, trace: AgentTrace, final_context, structured_output}`; `AgentTrace{tool_calls: list[ToolCall], total_iterations, ...}`; `ToolCall{tool_name, parameters, reasoning}`; `ToolResult{tool_name, success, result, error, execution_time_ms}`; `ToolDefinition{name, description, parameter_schema, requires_approval, execute_fn (excluded)}`; `ApprovalRequest{tool_name, parameters, reasoning, context_summary}`; `AgentStep{iteration, thought, action, observation, timestamp}`.
- Pattern models: `PlanStep`, `EvaluationResult`, `ReflexionMemory`, `DebateRound`, `ChainStep`, `DecompositionResult`; meta: `ArtifactType`, `BuildProgress{total_required, completed, missing, warnings}`, `MetaBuilderConfig(AgentConfig){max_turns, build_max_iterations, build_timeout_seconds, build_temperature, output_path}`, `MetaBuilderResult(AgentResult){artifact, artifact_json, artifact_type, is_valid, validation_errors, ...}`.
- Core context keys: `task`, `tool_name`, `tool_input`, `reasoning`, `should_terminate`, `tool_result`, `tool_error`, `tool_status`, `observations`, `final_answer`, `confidence`, `iteration_count`, `max_iterations_reached`, `approval_required`, `approval_granted`, `agent_trace`; `NO_TOOL = "none"`. Pattern keys (plan, evidence, draft/checker, chain, samples, subtasks/worker_results, debate, attempt) live in `ContextKeys`.

## Invariants and constraints

- `_init_context` always sets `task`, `agent_trace=[]`, `iteration_count=0` (warns and overwrites if the caller passed them) and `_output_response_format` when `output_schema` is set.
- Budgets: `timeout_seconds` -> `AgentTimeoutError`; loop iterations > `max_iterations * FSM_BUDGET_MULTIPLIER (3)` -> `BudgetExhaustedError`. Both propagate unwrapped; any other failure becomes `AgentError("<Type> execution failed: ...")`.
- Answer: `final_answer` -> pattern `extra_answer_keys` -> last non-empty response -> "Agent could not determine an answer.".
- `success`: planner patterns (orchestrator, rewoo, plan_execute) pass `execution_evidence_keys` and need real executed work; others need an answer key or at least one tool call. Otherwise `success=False` with a WARNING.
- `final_context` is filtered with `fsm_llm.constants.has_internal_prefix` (top level only). Never re-inline `startswith("_")`.
- Tool-using agents raise `AgentError` on an empty registry (React, ReWOO, Reflexion, ParallelReact, NativeFC).
- `AgentHandlers` is created per `run()` and passed explicitly to `_register_handlers`; never stash it on `self` (data race).
- `ToolRegistry.execute` never raises: missing tool, bad params, or tool exception return `ToolResult(success=False)`.
- `execute_tool` refuses to conclude before any tool call on the first iteration (premature-terminate guard).
- `_output_response_format` has exactly two call sites (`_init_context`, `native_fc`); keep it that way.
- Lifecycle handlers registered on every API: END_CONVERSATION marker, ERROR logger, `ContextCompactor` clearing `tool_result/tool_status/tool_error` at PRE_PROCESSING, optional observation summarizer.

## Dependencies

- `fsm_llm`: `API`, `HandlerTiming`, `ContextCompactor`, `WorkingMemory`, `SessionStore`, `Classifier`, `ClassificationSchema`, `IntentDefinition`, `utilities.extract_json_from_text`, `constants.has_internal_prefix`, `DEFAULT_LLM_MODEL`.
- `litellm` directly in `native_fc.py`, `meta_builder.py`, `semantic_tools.py`, `semantic_memory.py`, `composition.py`.
- Optional: `fsm_llm_reasoning` (ReasoningReactAgent), `mcp`, `httpx`, `fastapi`, `fsm_llm_workflows` (WorkflowBuilder artifacts are validated structurally).

## Failure modes

- `AgentError(message, details)` (base: `FSMError`) -> `ToolExecutionError(message, tool_name)`, `ToolNotFoundError(tool_name)`, `ToolValidationError(tool_name, reason)` (no call site), `BudgetExhaustedError(budget_type, limit)`, `ApprovalDeniedError(action_description)`, `AgentTimeoutError(timeout_seconds)`, `EvaluationError(message, evaluator)`, `DecompositionError(message, depth)`, `MetaBuilderError` -> `BuilderError(message, action)`, `MetaValidationError`, `OutputError(message, path)`.
- Structured output parse failure returns `structured_output=None` (context keys, then JSON in answer, then JSON in observations) with a WARNING.
- Known open issue F-LIVE-02: a post-tool stall in the agents loop on live small models (reproduced by the live Ollama suite, not fixed).

## Working here

- New pattern: add `build_<x>_fsm` in `fsm_definitions.py`, prompt builders in `prompts.py`, a states class and context keys in `constants.py`, the agent class subclassing `BaseAgent` (use `_standard_run`), register it in `create_agent._PATTERNS` and `__all__`, and a graph in `src/fsm_llm_monitor/static/flows.json` if the monitor should draw it.
- Keep intermediate states' `response_instructions` empty so Pass 2 is skipped; only the final state should produce prose.
- Tests: `pytest tests/test_fsm_llm_agents/` (45 files, e.g. `test_react.py`, `test_base_agent.py`, `test_completion_guard.py`, `test_native_fc.py`) and `pytest tests/test_fsm_llm_meta/` (meta-builder). Examples: `examples/agents/*`, `examples/meta/*` (do not modify examples unless asked; they are evaluation baselines for `scripts/eval.py`).
