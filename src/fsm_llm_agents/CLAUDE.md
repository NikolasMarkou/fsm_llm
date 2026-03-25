# fsm_llm_agents -- Claude Code Instructions

## What This Package Does

12 agentic patterns with tool execution and Human-in-the-Loop support, built on fsm_llm's 2-pass core. Each pattern auto-generates FSM definitions at runtime from tool registries and configuration -- no manual JSON authoring. The largest sub-package in FSM-LLM.

Patterns: ReactAgent, REWOOAgent, PlanExecuteAgent, ReflexionAgent, PromptChainAgent, SelfConsistencyAgent, DebateAgent, OrchestratorAgent, ADaPTAgent, EvaluatorOptimizerAgent, MakerCheckerAgent, ReasoningReactAgent (conditional on fsm_llm_reasoning).

## File Map

| File | Purpose |
|------|---------|
| `react.py` | **ReactAgent** -- ReAct loop: think -> act -> observe -> conclude. Auto-generates FSM from ToolRegistry, registers handlers, manages tool dispatch |
| `rewoo.py` | **REWOOAgent** -- planning-first execution. Plans all tool calls with #E1/#E2 variable references, executes, synthesizes. Exactly 2 LLM calls |
| `plan_execute.py` | **PlanExecuteAgent** -- plan decomposition, sequential step execution, replan on failure, synthesize |
| `reflexion.py` | **ReflexionAgent** -- ReAct + evaluation gate + episodic memory. Reflects on failures and retries with lessons |
| `prompt_chain.py` | **PromptChainAgent** -- fixed sequential prompt pipeline with optional validation gates between steps |
| `self_consistency.py` | **SelfConsistencyAgent** -- N independent samples at varying temperatures, majority vote or custom aggregation |
| `debate.py` | **DebateAgent** -- 3 personas (proposer, critic, judge), multi-round structured debate |
| `orchestrator.py` | **OrchestratorAgent** -- task decomposition, worker delegation via factory callable, result collection, synthesis |
| `adapt.py` | **ADaPTAgent** -- attempt direct solve, assess, recursively decompose on failure (AND/OR operators) |
| `evaluator_optimizer.py` | **EvaluatorOptimizerAgent** -- generate-evaluate-refine loop with external evaluation function |
| `maker_checker.py` | **MakerCheckerAgent** -- maker persona drafts, checker persona evaluates against quality criteria |
| `reasoning_react.py` | **ReasoningReactAgent** -- ReactAgent + auto-registered `reason` pseudo-tool backed by ReasoningEngine (FSM stacking) |
| `tools.py` | **ToolRegistry** + `@tool` decorator -- tool management, prompt generation, classification schema generation, execution with timing |
| `hitl.py` | **HumanInTheLoop** -- approval policies, approval callbacks, confidence-based escalation, timeout support |
| `handlers.py` | **AgentHandlers** -- stateful handler collection: tool executor (POST_TRANSITION on act), iteration limiter (PRE_TRANSITION), observation tracker, HITL gate (CONTEXT_UPDATE) |
| `fsm_definitions.py` | FSM builders: `build_react_fsm()`, `build_reflexion_fsm()`, `build_plan_execute_fsm()`, `build_rewoo_fsm()`, `build_eval_opt_fsm()`, `build_maker_checker_fsm()`, `build_chain_fsm()`, `build_self_consistency_fsm()`, `build_orchestrator_fsm()`, `build_debate_fsm()`, `build_adapt_fsm()` |
| `prompts.py` | Prompt builders for think/act/conclude/approval states and all pattern-specific states |
| `definitions.py` | Pydantic models: ToolDefinition, ToolCall, ToolResult, AgentStep, AgentTrace, AgentConfig, AgentResult, ApprovalRequest, PlanStep, EvaluationResult, ReflexionMemory, DebateRound, ChainStep, DecompositionResult |
| `constants.py` | 11 state enum classes (AgentStates, ReflexionStates, PlanExecuteStates, etc.), ContextKeys (60+ keys), HandlerNames, Defaults, ErrorMessages, LogMessages, ReasoningIntegrationKeys |
| `exceptions.py` | AgentError hierarchy (9 exception types) |
| `__main__.py` | CLI: `python -m fsm_llm_agents --info` / `--version` |
| `__init__.py` | Public API exports -- 36 symbols. ReasoningReactAgent conditionally exported via try/except ImportError |
| `__version__.py` | Version imported from `fsm_llm.__version__` |

## Key Patterns

### ReactAgent Pipeline
1. User calls `agent.run(task)` with a task string and optional `initial_context`
2. Agent builds FSM via `build_react_fsm(tools, task_description, include_approval_state)` -- generates states: think, act, conclude (+ await_approval if HITL)
3. Creates `fsm_llm.API.from_definition()` instance with model/temperature/max_tokens from AgentConfig
4. Registers handlers: tool executor (POST_TRANSITION on `act`), iteration limiter (PRE_TRANSITION), observation tracker, HITL gate (if hitl provided)
5. Starts conversation with task in context, loops `api.converse("Continue.")` until terminal state or budget exhausted
6. Extracts answer from `final_answer` context key, returns `AgentResult`

### ToolRegistry
- `register(ToolDefinition)` / `register_function(fn, name, description)` -- both return self for chaining
- `@tool(description=..., requires_approval=...)` decorator attaches `_tool_definition` attribute to decorated function
- `to_prompt_description()` -- generates multi-line "Available tools:" listing with parameters
- `to_classification_schema()` -- generates dict compatible with `fsm_llm_classification.ClassificationSchema`, includes automatic "none" fallback intent
- `execute(ToolCall)` -- inspects function signature to determine calling convention (0-arg, 1-arg dict, or **kwargs), validates required/unknown parameters against schema, wraps result in `ToolResult` with timing

### HumanInTheLoop
- `approval_policy: Callable[[ToolCall, dict], bool]` -- decides if a tool call needs approval
- `approval_callback: Callable[[ApprovalRequest], bool]` -- requests approval from human (raises AgentError if None when needed)
- `confidence_threshold: float` (default 0.3) -- escalate below this level
- `on_escalation: Callable[[str, dict], None]` -- callback on escalation
- `approval_timeout: float | None` -- max wait seconds, denial on timeout
- Thread-safe via `threading.Event` for timeout implementation

### Auto-Generated FSMs
Each pattern has a `build_*_fsm()` function that returns a `dict[str, Any]` (valid FSM definition dict). The dict is passed to `API.from_definition()` which validates it against `FSMDefinition`. FSMs include proper extraction_instructions, response_instructions, transitions with conditions, and persona strings. No JSON files on disk.

### AgentHandlers
Stateful class instantiated per agent. Tracks `_current_iteration` counter. Must call `reset()` before each `run()`. Methods:
- `execute_tool(context)` -- reads `tool_name`/`tool_input` from context, normalizes input, calls registry.execute(), returns context update dict
- `check_iteration_limit(context)` -- increments counter, sets `max_iterations_reached=True` if exceeded, enforces FSM budget ceiling
- `track_observation(context)` -- appends tool results to observations list in context
- `check_approval(context)` -- flags `approval_required=True` if tool needs HITL approval

### Budget Enforcement
`Defaults.FSM_BUDGET_MULTIPLIER = 3` -- hard ceiling is `max_iterations * 3`. Each agent cycle uses multiple FSM transitions (think + act = 2 transitions minimum), so the multiplier gives headroom. The PRE_TRANSITION handler checks both iteration count and wall-clock timeout.

## Dependencies on Core

- `fsm_llm.API` -- `from_definition()`, `start_conversation()`, `converse()`, `get_data()`
- `fsm_llm.handlers.HandlerTiming` -- `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`
- `fsm_llm.constants.DEFAULT_LLM_MODEL` -- used in `Defaults.MODEL`
- `fsm_llm.logging.logger` -- loguru logging throughout
- `fsm_llm.definitions.FSMError` -- base class for `AgentError`
- `fsm_llm.definitions.FSMDefinition` -- validates generated FSM dicts (implicitly via API.from_definition)

## Exception Hierarchy

```
AgentError (FSMError)
+-- ToolExecutionError          # Tool function raised an exception (.tool_name)
+-- ToolNotFoundError           # Tool not in registry (.tool_name)
+-- ToolValidationError         # Parameter validation failed (.tool_name)
+-- BudgetExhaustedError        # Max iterations/tokens/time exceeded (.budget_type, .limit)
+-- AgentTimeoutError           # Wall-clock timeout exceeded (.timeout_seconds)
+-- ApprovalDeniedError         # Human denied tool approval (.action_description)
+-- EvaluationError             # Evaluation function failed (.evaluator)
+-- DecompositionError          # Task decomposition failed (.depth)
```

All accept `message: str` and optional `details: dict[str, Any]` (inherited from FSMError).

## Testing

```bash
pytest tests/test_fsm_llm_agents/ -v    # 547 tests across 24 files
```

Test files: test_react.py, test_rewoo.py, test_plan_execute.py, test_reflexion.py, test_prompt_chain.py, test_self_consistency.py, test_debate.py, test_orchestrator.py, test_adapt.py, test_evaluator_optimizer.py, test_maker_checker.py, test_reasoning_react.py, test_tools.py, test_hitl.py, test_handlers.py, test_definitions.py, test_constants.py, test_exceptions.py, test_fsm_definitions.py, test_prompts.py, test_integration_methods.py, test_bug_fixes.py.

Tests use `MockLLMWithResponses` and `MockLLM2Interface` from conftest for deterministic agent runs without real LLM calls.

## Gotchas

- **ReactAgent.run() needs a real LLM or mock** -- it creates an API instance internally with `API.from_definition()`. Tests use MockLLM fixtures.
- **FSM definitions are generated at runtime** -- not loaded from JSON files. Each `run()` call builds a fresh FSM definition dict.
- **ToolDefinition requires execute_fn** -- the `register()` method raises ValueError if `execute_fn` is None. Schema-only (no-execution) tools are not supported.
- **AgentHandlers is stateful** -- tracks iteration count. The `reset()` method must be called before each `run()`, which all agents do internally.
- **ReasoningReactAgent is conditionally available** -- only exported in `__init__.py` when `fsm_llm_reasoning` can be imported. The try/except block in `__init__.py` handles the missing dependency gracefully.
- **Tool name validation** -- ToolDefinition validates that `name` is alphanumeric with underscores or hyphens only. Special characters will raise `ValueError`.
- **ToolResult.summary truncates** -- results longer than `Defaults.MAX_OBSERVATION_LENGTH` (2000 chars) are truncated with `...[truncated]` suffix.
- **Approval callback required** -- `HumanInTheLoop.request_approval()` raises AgentError if `approval_callback` is None. Setting `approval_policy` without `approval_callback` will fail at runtime when approval is needed.
- **Continue message** -- all agents advance the FSM loop by sending `Defaults.CONTINUE_MESSAGE` ("Continue.") via `api.converse()`. The LLM sees this as user input.
