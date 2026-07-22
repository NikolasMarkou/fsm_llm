# API Reference

> Covers FSM-LLM v0.5.0

Complete API documentation for FSM-LLM and its extension packages.

## API Class (`fsm_llm.API`)

Main interface for working with FSM-LLM.

### Constructor

```python
API(
    fsm_definition: FSMDefinition | dict | str,
    llm_interface: LLMInterface | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_history_size: int = 5,
    max_message_length: int = 1000,
    handlers: list[FSMHandler] | None = None,
    handler_error_mode: str = "continue",
    transition_config: TransitionEvaluatorConfig | None = None,
    session_store: SessionStore | None = None,
    **llm_kwargs,
)
```

### Factory Methods

```python
api = API.from_file("path/to/fsm.json", model="gpt-4o-mini")
api = API.from_definition(fsm_dict, model="gpt-4o-mini")
```

### Conversation Lifecycle

```python
conv_id, response = api.start_conversation(initial_context={"key": "val"})
response = api.converse("user message", conv_id)
for chunk in api.converse_stream("user message", conv_id):  # -> Iterator[str]
    print(chunk, end="", flush=True)
api.end_conversation(conv_id)
api.has_conversation_ended(conv_id)  # -> bool
api.close()                          # cleanup all resources
```

### Session Persistence

Pass a `session_store` to persist conversations across process restarts (state auto-saves after each `converse`):

```python
from fsm_llm import API, FileSessionStore

api = API.from_file("bot.json", model="gpt-4o-mini",
                    session_store=FileSessionStore("./sessions"))
api.save_session(conv_id)                              # -> None
state = api.load_session(session_id)                   # -> SessionState | None  (inspect only)
new_conv_id, state = api.restore_session(session_id)   # -> (conv_id, SessionState) | None  (resume)
```

### State & Data Queries

```python
api.get_current_state(conv_id)           # -> str
api.get_data(conv_id)                    # -> dict
api.get_conversation_history(conv_id)    # -> list[dict]
api.list_active_conversations()          # -> list[str]
api.update_context(conv_id, {"k": "v"})
api.cleanup_stale_conversations(max_idle_seconds=3600)  # -> list[str]
api.get_llm_interface()                  # -> LLMInterface
```

### FSM Stacking

```python
response = api.push_fsm(conv_id, new_fsm,
    context_to_pass={"step": "details"},
    shared_context_keys=["user_id"],
    preserve_history=True, inherit_context=True)
response = api.pop_fsm(conv_id,
    context_to_return={"complete": True},
    merge_strategy=ContextMergeStrategy.UPDATE)  # or "preserve"
api.get_stack_depth(conv_id)           # -> int
api.get_sub_conversation_id(conv_id)   # -> str
```

### Handler Registration

```python
api.register_handler(handler)
api.register_handlers([handler1, handler2])
builder = api.create_handler("MyHandler")
```

## HandlerBuilder

Fluent API returned by `api.create_handler()`:

| Method | Description |
|--------|-------------|
| `.at(*timings)` | Specify HandlerTiming values |
| `.on_state(*states)` | Execute only in these states |
| `.not_on_state(*states)` | Exclude these states |
| `.on_target_state(*states)` | Execute only when transitioning TO these states |
| `.not_on_target_state(*states)` | Exclude transitions TO these states |
| `.when_context_has(*keys)` | Require these context keys |
| `.when_keys_updated(*keys)` | Execute when these keys change |
| `.on_state_entry(*states)` | Shorthand: `.at(POST_TRANSITION).on_target_state()` |
| `.on_state_exit(*states)` | Shorthand: `.at(PRE_TRANSITION).on_state()` |
| `.on_context_update(*keys)` | Shorthand: `.at(CONTEXT_UPDATE).when_keys_updated()` |
| `.when(condition)` | Custom condition lambda |
| `.with_priority(n)` | Execution priority (lower runs first, default 100) |
| `.do(fn)` | Set handler function and build |

## HandlerTiming Enum

`START_CONVERSATION`, `PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `END_CONVERSATION`, `ERROR`

## LLMInterface (`fsm_llm.llm`)

```python
class LLMInterface(ABC):
    @abstractmethod
    def generate_response(self, request: ResponseGenerationRequest) -> ResponseGenerationResponse: ...

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse: ...

    def generate_response_stream(self, request: ResponseGenerationRequest) -> Iterator[str]: ...
```

`LiteLLMInterface` is the built-in implementation supporting 100+ providers via litellm; it implements `generate_response_stream` (the Pass-2 streaming path behind `API.converse_stream`).

## WorkingMemory (`fsm_llm`)

```python
from fsm_llm import WorkingMemory, BUFFER_METADATA

wm = WorkingMemory()
wm.set("core", "goal", "book a flight")   # set(buffer, key, value)
wm.get("core", "goal")                    # get(buffer, key, default=None)
wm.get_all_data()                         # flattened view across buffers
```

Named buffers: `core`, `scratch`, `environment`, `reasoning`. The `BUFFER_METADATA` (`"metadata"`) buffer is hidden by default and excluded from the LLM-visible context.

## TransitionEvaluatorConfig

```python
@dataclass
class TransitionEvaluatorConfig:
    ambiguity_threshold: float = 0.1
    minimum_confidence: float = 0.5
    strict_condition_matching: bool = True
    evidence_conditions_normalizer: float = 5.0
    detailed_logging: bool = False
```

## Classification (`fsm_llm`)

Built into core -- no separate install needed.

```python
from fsm_llm import Classifier, ClassificationSchema, IntentDefinition, IntentRouter

schema = ClassificationSchema(
    intents=[IntentDefinition(name="billing", description="Billing questions")],
    fallback_intent="general",
)
classifier = Classifier(schema, model="gpt-4o-mini")
result = classifier.classify("Where is my invoice?")
# result.intent, result.confidence, result.is_low_confidence

# Multi-intent
result = classifier.classify_multi("Check order and update billing")

# Hierarchical (two-stage domain -> intent, for >15 intents)
from fsm_llm import HierarchicalClassifier
h_classifier = HierarchicalClassifier(domains=[...], model="gpt-4o-mini")

# Intent routing
router = IntentRouter(schema)
router.register("billing", handle_billing)
response = router.route(message, classification_result)
```

## ReasoningEngine (`fsm_llm_reasoning`)

```python
from fsm_llm_reasoning import ReasoningEngine, ReasoningType

engine = ReasoningEngine(model="gpt-4o-mini")
solution, trace = engine.solve_problem("problem text", initial_context={})
# solution: str, trace: dict (run metadata: steps, reasoning_types_used, final_confidence, execution_time_seconds)
```

9 strategies: `SIMPLE_CALCULATOR`, `ANALYTICAL`, `DEDUCTIVE`, `INDUCTIVE`, `CREATIVE`, `CRITICAL`, `HYBRID`, `ABDUCTIVE`, `ANALOGICAL`.

## Agents (`fsm_llm_agents`)

```python
from fsm_llm_agents import create_agent, ReactAgent, tool, ToolRegistry, HumanInTheLoop, AgentConfig

# @tool decorator auto-generates JSON schema from type hints
@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

# Create agent
agent = create_agent("react", model="gpt-4o-mini", tools=[search])
result = agent("task")  # or agent.run("task")
# result.answer, result.success, result.trace, result.structured_output

# Structured output
agent = ReactAgent(model="gpt-4o-mini", tools=[search],
                   config=AgentConfig(output_schema=MyPydanticModel))

# Human-in-the-loop
hitl = HumanInTheLoop(approval_callback=fn, require_approval_for=["tool_name"])
agent = ReactAgent(model="gpt-4o-mini", tools=[search], hitl=hitl)
```

13 `create_agent()` patterns: `react`, `rewoo`, `debate`, `plan_execute`, `prompt_chain`, `self_consistency`, `orchestrator`, `adapt`, `evaluator_optimizer`, `maker_checker`, `reflexion`, `meta_builder`, `swarm` (plus `reasoning_react` when `fsm_llm_reasoning` is installed).

Multi-agent coordination and integrations (constructed directly, not via the factory): `SwarmAgent`, `AgentGraph` / `AgentGraphBuilder` (DAG orchestration), `MCPToolProvider` (MCP tools), `AgentServer` / `RemoteAgentTool` (A2A), `SemanticToolRegistry` (embedding-based tool retrieval), `SOPRegistry` / `load_builtin_sops` (reusable agent templates).

## WorkflowEngine (`fsm_llm_workflows`)

```python
from fsm_llm_workflows import WorkflowEngine, create_workflow, auto_step, condition_step

workflow = create_workflow("my_workflow", "My Workflow")
workflow.with_initial_step(auto_step("start", "Start", next_state="check"))
workflow.with_step(condition_step("check", "Check", condition=fn, true_state="ok", false_state="fail"))

engine = WorkflowEngine(max_concurrent_workflows=100)
engine.register_workflow(workflow)
instance_id = await engine.start_workflow("my_workflow", initial_context={})
await engine.advance_workflow(instance_id)
status = engine.get_workflow_status(instance_id)
await engine.shutdown()
```

11 step types: `auto_step`, `api_step`, `condition_step`, `llm_step`, `wait_event_step`, `timer_step`, `parallel_step`, `conversation_step`, `agent_step`, `retry_step`, `switch_step`.

## Harness (`fsm_llm_harness`)

The iterative-planner protocol as a 6-state FSM over a plan directory. Requires
`pip install fsm-llm[harness]`. 118 public names in one literal `__all__`; the
load-bearing ones are below.

### HarnessAgent -- the driver

```python
from fsm_llm_harness import HarnessAgent, Workspace, build_default_worker_factory, ContextKeys

workspace = Workspace("./src")                       # confined source-tree root
agent = HarnessAgent(
    worker_factory=build_default_worker_factory(workspace),
    approval_callback=None,      # None => a callback that DENIES every human gate
    revert_callback=None,        # None => compute the leash revert, execute nothing
    findings_threshold=3,        # EXPLORE -> PLAN
    max_fix_attempts=2,          # the autonomy leash
    max_leash_grants=2,          # human leash-continues honoured per plan step
    iteration_hard_cap=6,        # PLAN -> EXECUTE
    max_explore_redispatches=9,  # extra EXPLORE dispatches per run while blocked
)
result = agent.run(
    "add a retry to the uploader",
    initial_context={
        ContextKeys.PLAN_DIR: "plans/plan-2026-07-22T101500-1a2b3c4d",
        ContextKeys.WORKSPACE_ROOT: "./src",
    },
)
agent.presentations   # list[Presentation] -- the Presentation Contracts emitted
agent.reverts         # list[RevertDirective] -- computed, not executed
agent.audit_issues    # list[Issue] | None -- the CLOSE audit, if CLOSE was reached
```

Executor dispatches on one plan step are bounded by
`max_fix_attempts * (1 + max_leash_grants)` for **any** sequence of approvals --
an approving callback cannot raise it. `worker_factory=None` is a diagnostic mode:
the FSM turns but no gate opens.

Worker seam: `WorkerFactory = Callable[[RoleRequest], AgentResult]`. `RoleSpec`
(one per state) carries `tool_scope`, `plan_tool_scope`, `owned_artifacts`,
`output_schema`, `expected_keys`, `writable_keys`, `max_iterations`;
`ROLE_SPECS` / `get_role_spec(state)` expose them.

### Plan directory, gate and audit

```python
from fsm_llm_harness import PlanDirectory, Role, ArtifactNames, StateDoc, pre_step_gate, audit

directory = PlanDirectory.create("plans", role=Role.ORCHESTRATOR)   # mints plan-<ts>-<hex8>
directory.write_text(ArtifactNames.STATE, StateDoc(state="explore").to_markdown())  # atomic
run_state = directory.load_run_state()          # RunState(plan_id, doc); .state == "explore"

gate = pre_step_gate(directory.path)            # GateResult(passed, slug, detail, exit_code)
issues = audit(directory.path, workspace_root=".")     # list[Issue]; [] means clean
```

`directory.path` is the plan directory; `directory.root` is its **parent** (the
confinement root `PlanMemory` was given), so pass `path` to anything that expects
a plan directory. `mint_plan_id()` is exported separately for callers that want
the id without a directory.

`pre_step_gate` evaluates 4 slugs in `GateSlug.ORDER` (`no-plan`, `wrong-state`,
`leash-cap`, `iteration-cap`), short-circuits on the first failure, and never
raises -- an unreadable `state.md` **is** the `no-plan` answer. Every failure is
HARD and carries `exit_code == 2`. `audit` runs the 30 checks in `CHECKS` and
reports a check that raises as an ERROR rather than letting it suppress the rest.

CLOSE-phase size policies, all pure and non-writing: `evict_lessons(doc)` and
`apply_sliding_window(doc)` return `(trimmed_doc, report)`, while
`check_system_cap(doc)` returns a `CapReport` only -- `SYSTEM.md` is measured and
REFUSED over its cap, never trimmed, because it carries no eviction order and all
six of its sections are required. `PlanDirectory.enforce_lessons_cap` /
`enforce_system_cap` / `apply_sliding_window` are the wrappers that write.

### Confined tools

```python
from fsm_llm_harness import Workspace, PlanMemory, build_workspace_tools, build_plan_tools, Role

workspace = Workspace("./src")                          # confinement only
memory = PlanMemory("plans/plan-...", role=Role.EXPLORER)  # confinement + ownership
registry = build_workspace_tools(workspace, allowed=("read_file", "grep_files"))
build_plan_tools(memory, allowed=("write_plan_file",), registry=registry)
```

Tool name groups: `READ_ONLY_TOOLS`, `WRITE_TOOLS`, `SHELL_TOOLS`,
`PLAN_READ_TOOLS`, `PLAN_WRITE_TOOLS`. `run_command` is off unless
`Workspace(..., allow_shell=True)`; `COMMAND_ALLOWLIST` is
`cat grep head ls tail wc` (`git` is deliberately excluded), and
`VERIFICATION_COMMANDS` (`git make mypy pytest ruff`) is an opt-in set.
An escape raises `HarnessConfinementError`; an ownership violation raises
`HarnessOwnershipError`.

### Artifacts and hardening

`ARTIFACT_MODELS` maps each of the 15 artifact names to its pydantic model -- 14
classes, since `FINDINGS.md` and `DECISIONS.md` share `ConsolidatedDoc`:
`StateDoc`, `PlanDoc`, `DecisionsDoc`, `FindingsIndexDoc`, `FindingsTopicDoc`,
`ProgressDoc`, `VerificationDoc`, `ChangelogDoc`, `CheckpointDoc`, `SummaryDoc`,
`ConsolidatedDoc`, `LessonsDoc`, `SystemAtlasDoc`, `IndexDoc`. Every one carries
`from_markdown()` / `to_markdown()`. Contract tables: `DECISION_ENTRY_SCHEMAS`,
`PRESENTATION_CONTRACTS`, `MANDATORY_ADDITIONAL_CHECKS`, `VERDICT_BULLETS`,
`VERDICT_RECOMMENDATIONS`, `REJECTED_EVIDENCE`.

Small-model reply recovery (`hardening`): `strip_model_noise`,
`parse_json_payload`, `parse_role_output` -> `RoleOutput`, `coerce_worker_output`,
`type_matches`, `as_int`, `retry`. All fail CLOSED -- a garbled reply is never
retried into a pass.

### CLI

```bash
fsm-llm-harness new "add a retry to the uploader" [--create-only] [--model M] [--workspace .]
fsm-llm-harness resume   plans/plan-...  [--goal G]
fsm-llm-harness status   plans/plan-...
fsm-llm-harness validate plans/plan-...  [--workspace .]
fsm-llm-harness close    plans/plan-...  [--workspace .] [--apply]
```

Exit codes: `0` pass, `1` a negative answer (audit ERRORs, a failed run, a usage
fault), `2` **RESERVED** for a HARD `pre_step_gate` refusal. Because `2` is
reserved, argparse usage errors exit `1` rather than argparse's conventional `2`.
Model resolution: `--model` > `$LLM_MODEL` > package default. `close` is DRY-RUN
without `--apply` and refuses to compress a directory with audit ERRORs.

## Monitor (`fsm_llm_monitor`)

```python
from fsm_llm_monitor import MonitorBridge, configure, app

# MonitorBridge creates and wires its own EventCollector internally
bridge = MonitorBridge(api=api)   # or: bridge = MonitorBridge(); bridge.connect(api)
configure(bridge)                 # registers the bridge with the global web server

import uvicorn
uvicorn.run(app, host="127.0.0.1", port=8420)
# Or just use the CLI: fsm-llm-monitor

# OTEL export is available via OTELExporter (requires fsm-llm[otel])
```

## Exception Hierarchy

```
FSMError
├── StateNotFoundError
├── InvalidTransitionError
├── LLMResponseError
├── TransitionEvaluationError
├── ClassificationError (-> SchemaValidationError, ClassificationResponseError)
├── HandlerSystemError (-> HandlerExecutionError)
├── ReasoningEngineError (-> ReasoningExecutionError, ReasoningClassificationError)
├── WorkflowError (-> Definition, Step, Instance, Timeout, Validation, State, Event, Resource)
├── HarnessError (-> HarnessArtifactError, HarnessOwnershipError, HarnessReentrancyError, HarnessConfinementError)
└── AgentError (-> ToolExecution, ToolNotFound, ToolValidation, Budget, Approval, Timeout, Evaluation, Decomposition)
    └── MetaBuilderError (-> Builder, MetaValidation, Output)

Exception
└── MonitorError (-> MonitorInitialization, MetricCollection, MonitorConnection)
```

## Constants

```python
from fsm_llm.constants import (
    DEFAULT_LLM_MODEL,       # "ollama_chat/qwen3.5:4b"
    DEFAULT_TEMPERATURE,     # 0.5
    DEFAULT_MAX_HISTORY_SIZE, # 5
    DEFAULT_MAX_MESSAGE_LENGTH, # 1000
    DEFAULT_MAX_STACK_DEPTH,  # 10
)
```
