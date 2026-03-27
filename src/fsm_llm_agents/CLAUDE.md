# fsm_llm_agents -- Claude Code Instructions

## What This Package Does

12 agentic patterns with tool execution and Human-in-the-Loop support, built on fsm_llm's 2-pass core. All agents inherit from `BaseAgent` ABC, which provides the shared conversation loop, budget enforcement, answer extraction, trace building, `__call__` syntax, and structured output validation. Each pattern auto-generates FSM definitions at runtime from tool registries and configuration -- no manual JSON authoring. The largest sub-package in FSM-LLM.

Patterns: ReactAgent, REWOOAgent, PlanExecuteAgent, ReflexionAgent, PromptChainAgent, SelfConsistencyAgent, DebateAgent, OrchestratorAgent, ADaPTAgent, EvaluatorOptimizerAgent, MakerCheckerAgent, ReasoningReactAgent (conditional on fsm_llm_reasoning).

## File Map

| File | Purpose |
|------|---------|
| `base.py` | **BaseAgent** -- ABC with shared conversation loop (`_run_conversation_loop`, `_standard_run`), budget enforcement (`_check_budgets`), answer extraction (`_extract_answer`), trace building (`_build_trace`), context filtering (`_filter_context`), API factory (`_create_api`), structured output (`_try_parse_structured_output`), `__call__` support |
| `react.py` | **ReactAgent(BaseAgent)** -- ReAct loop: think -> act -> observe -> conclude. HITL via `_on_loop_iteration()` hook |
| `rewoo.py` | **REWOOAgent(BaseAgent)** -- planning-first execution. Plans all tool calls with #E1/#E2 variable references, executes, synthesizes. Custom `_build_trace()` |
| `plan_execute.py` | **PlanExecuteAgent(BaseAgent)** -- plan decomposition, sequential step execution, replan on failure, synthesize |
| `reflexion.py` | **ReflexionAgent(BaseAgent)** -- ReAct + evaluation gate + episodic memory. HITL via `_on_loop_iteration()` hook |
| `prompt_chain.py` | **PromptChainAgent(BaseAgent)** -- fixed sequential prompt pipeline with optional validation gates between steps. Custom `_extract_answer()` |
| `self_consistency.py` | **SelfConsistencyAgent(BaseAgent)** -- N independent samples at varying temperatures, majority vote or custom aggregation. Full `run()` override (unique multi-sample architecture) |
| `debate.py` | **DebateAgent(BaseAgent)** -- 3 personas (proposer, critic, judge), multi-round structured debate |
| `orchestrator.py` | **OrchestratorAgent(BaseAgent)** -- task decomposition, worker delegation via factory callable, result collection, synthesis |
| `adapt.py` | **ADaPTAgent(BaseAgent)** -- attempt direct solve, assess, recursively decompose on failure (AND/OR operators). Custom recursive `run()` |
| `evaluator_optimizer.py` | **EvaluatorOptimizerAgent(BaseAgent)** -- generate-evaluate-refine loop with external evaluation function. Custom `_extract_answer()` |
| `maker_checker.py` | **MakerCheckerAgent(BaseAgent)** -- maker persona drafts, checker persona evaluates against quality criteria |
| `reasoning_react.py` | **ReasoningReactAgent(BaseAgent)** -- ReactAgent + auto-registered `reason` pseudo-tool backed by ReasoningEngine (FSM stacking) |
| `tools.py` | **ToolRegistry** + `@tool` decorator (supports bare `@tool` with auto-schema inference from type hints, `Annotated` descriptions) + `register_agent()` for agents-as-tools. Classification schema generation, execution with timing |
| `hitl.py` | **HumanInTheLoop** -- approval policies, approval callbacks, confidence-based escalation, timeout support |
| `handlers.py` | **AgentHandlers** -- stateful handler collection: tool executor (POST_TRANSITION on act), iteration limiter (PRE_TRANSITION), observation tracker, HITL gate (CONTEXT_UPDATE) |
| `fsm_definitions.py` | FSM builders: `build_react_fsm()`, `build_reflexion_fsm()`, `build_plan_execute_fsm()`, `build_rewoo_fsm()`, `build_eval_opt_fsm()`, `build_maker_checker_fsm()`, `build_chain_fsm()`, `build_self_consistency_fsm()`, `build_orchestrator_fsm()`, `build_debate_fsm()`, `build_adapt_fsm()` |
| `prompts.py` | Prompt builders for think/act/conclude/approval states and all pattern-specific states |
| `definitions.py` | Pydantic models: ToolDefinition, ToolCall, ToolResult, AgentStep, AgentTrace, AgentConfig (with `output_schema`), AgentResult (with `structured_output`, `__str__`), ApprovalRequest, PlanStep, EvaluationResult, ReflexionMemory, DebateRound, ChainStep, DecompositionResult |
| `constants.py` | 11 state enum classes (AgentStates, ReflexionStates, PlanExecuteStates, etc.), ContextKeys (60+ keys), HandlerNames, Defaults, ErrorMessages, LogMessages, ReasoningIntegrationKeys |
| `exceptions.py` | AgentError hierarchy (9 exception types) |
| `__main__.py` | CLI: `python -m fsm_llm_agents --info` / `--version` |
| `__init__.py` | Public API exports -- 44 symbols + `create_agent()` factory. ReasoningReactAgent conditionally exported via try/except ImportError |
| `__version__.py` | Version imported from `fsm_llm.__version__` |

## Key Patterns

### BaseAgent (all agents inherit from this)
- `__init__(config, **api_kwargs)` — stores config and API kwargs
- `run(task, initial_context)` — abstract, implemented by each agent
- `__call__(task, **kwargs)` — delegates to `run()` for Strands-style `agent("task")` syntax
- `_standard_run(task, fsm_def, context, agent_type, max_iterations, extra_answer_keys)` — common run implementation: creates API, registers handlers, runs conversation loop, extracts answer, builds trace, returns AgentResult
- `_run_conversation_loop(api, context, start_time, agent_type, max_iterations)` — the while-not-ended loop with budget checks
- `_on_loop_iteration(api, conv_id, iteration)` — hook for HITL gates (overridden by ReactAgent, ReflexionAgent)
- `_check_budgets(start_time, iteration, max_iterations)` — raises AgentTimeoutError or BudgetExhaustedError
- `_extract_answer(final_context, responses, extra_keys)` — fallback chain: final_answer → extra_keys → last response
- `_build_trace(final_context, iteration)` — builds AgentTrace from AGENT_TRACE context
- `_filter_context(context)` — removes `_`-prefixed internal keys
- `_create_api(fsm_def)` — factory for `API.from_definition()`
- `_try_parse_structured_output(answer)` — validates answer against `config.output_schema` Pydantic model

### ReactAgent Pipeline
1. User calls `agent.run(task)` or `agent("task")` with a task string and optional `initial_context`
2. Agent builds FSM via `build_react_fsm(tools, task_description, include_approval_state)` -- generates states: think, act, conclude (+ await_approval if HITL)
3. Delegates to `_standard_run()` which creates API, registers handlers, runs conversation loop
4. HITL approval handled via `_on_loop_iteration()` hook — checks context for approval_required flag
5. Extracts answer from `final_answer` context key, returns `AgentResult` (with `structured_output` if `output_schema` set)

### @tool Decorator
Three usage forms:
- `@tool` — bare decorator, auto-infers name from function name, description from docstring, parameter schema from type hints
- `@tool(description="...", requires_approval=True)` — with keyword overrides
- `@tool(parameter_schema={...})` — explicit schema (backward compatible)

Auto-schema inference maps: `str→string`, `int→integer`, `float→number`, `bool→boolean`, `list→array`, `dict→object`. Supports `typing.Annotated[str, "description"]` for per-parameter descriptions.

Legacy `params: dict` single-parameter functions are auto-detected and skip inference.

### ToolRegistry
- `register(ToolDefinition)` / `register_function(fn, name, description)` -- both return self for chaining
- `register_agent(agent, name, description)` -- wraps agent.run() as a tool for supervisor/orchestrator patterns
- `to_prompt_description()` -- generates multi-line "Available tools:" listing with parameters
- `to_classification_schema()` -- generates dict compatible with `fsm_llm.ClassificationSchema`, includes automatic "none" fallback intent
- `execute(ToolCall)` -- inspects function signature to determine calling convention (0-arg, 1-arg legacy dict, or **kwargs with schema filtering), validates parameters, wraps result in `ToolResult` with timing

### create_agent() Factory
```python
from fsm_llm_agents import create_agent, tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return "results"

agent = create_agent(tools=[search], pattern="react")
result = agent("What is the capital of France?")
```
Supports all 11 agent patterns. Auto-builds ToolRegistry from list of @tool-decorated functions.

### Structured Output
```python
from pydantic import BaseModel
from fsm_llm_agents import ReactAgent, AgentConfig

class Report(BaseModel):
    title: str
    findings: list[str]
    confidence: float

agent = ReactAgent(tools=registry, config=AgentConfig(output_schema=Report))
result = agent.run("Analyze X")
result.structured_output  # Report instance or None on validation failure
```
Graceful fallback: if JSON extraction or Pydantic validation fails, `structured_output` is None and `answer` remains the raw string.

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
`Defaults.FSM_BUDGET_MULTIPLIER = 3` -- hard ceiling is `max_iterations * 3`. Each agent cycle uses multiple FSM transitions (think + act = 2 transitions minimum), so the multiplier gives headroom. Budget checks are in `BaseAgent._check_budgets()`, called from `_run_conversation_loop()`.

## Dependencies on Core

- `fsm_llm.API` -- `from_definition()`, `start_conversation()`, `converse()`, `get_data()`
- `fsm_llm.handlers.HandlerTiming` -- `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`
- `fsm_llm.constants.DEFAULT_LLM_MODEL` -- used in `Defaults.MODEL`
- `fsm_llm.logging.logger` -- loguru logging throughout
- `fsm_llm.definitions.FSMError` -- base class for `AgentError`
- `fsm_llm.definitions.FSMDefinition` -- validates generated FSM dicts (implicitly via API.from_definition)
- `fsm_llm.utilities.extract_json_from_text` -- used by structured output parsing

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
pytest tests/test_fsm_llm_agents/ -v    # 596 tests across 25 files
```

Test files: test_react.py, test_rewoo.py, test_plan_execute.py, test_reflexion.py, test_prompt_chain.py, test_self_consistency.py, test_debate.py, test_orchestrator.py, test_adapt.py, test_evaluator_optimizer.py, test_maker_checker.py, test_reasoning_react.py, test_tools.py, test_hitl.py, test_handlers.py, test_definitions.py, test_constants.py, test_exceptions.py, test_fsm_definitions.py, test_prompts.py, test_integration_methods.py, test_bug_fixes.py, test_base_agent.py, test_structured_output.py.

Tests use `Mock(spec=LLMInterface)` and `MockLLM2Interface` from conftest for deterministic agent runs without real LLM calls.

## Gotchas

- **All agents inherit from BaseAgent** -- shared conversation loop, budget enforcement, answer extraction, and `__call__` are in `base.py`. Agent-specific logic is in each agent's `run()`, `_register_handlers()`, and optional method overrides.
- **SelfConsistencyAgent fully overrides `run()`** -- its multi-sample architecture doesn't use the standard conversation loop. It still uses `_check_budgets()` and `_filter_context()` from BaseAgent.
- **ADaPTAgent has recursive `run()`** -- `_depth` parameter tracks decomposition depth. Uses `_create_api()` and `_run_conversation_loop()` from base but builds results manually.
- **@tool supports bare decorator** -- `@tool` (no parens) auto-infers schema from type hints. `@tool(...)` with explicit params still works. Legacy `params: dict` pattern is auto-detected.
- **register_agent() enables agents-as-tools** -- wraps agent.run() in a tool function for supervisor patterns.
- **output_schema is on AgentConfig, not agent constructor** -- `AgentConfig(output_schema=MyModel)` enables structured output validation.
- **ReactAgent.run() needs a real LLM or mock** -- it creates an API instance internally with `API.from_definition()`. Tests use MockLLM fixtures.
- **FSM definitions are generated at runtime** -- not loaded from JSON files. Each `run()` call builds a fresh FSM definition dict.
- **ToolDefinition requires execute_fn** -- the `register()` method raises ValueError if `execute_fn` is None. Schema-only (no-execution) tools are not supported.
- **AgentHandlers is stateful** -- tracks iteration count. The `reset()` method must be called before each `run()`, which all agents do internally.
- **ReasoningReactAgent is conditionally available** -- only exported in `__init__.py` when `fsm_llm_reasoning` can be imported.
- **Continue message** -- all agents advance the FSM loop by sending `Defaults.CONTINUE_MESSAGE` ("Continue.") via `api.converse()`. The LLM sees this as user input.
