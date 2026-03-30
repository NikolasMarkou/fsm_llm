# API Reference

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
api.end_conversation(conv_id)
api.has_conversation_ended(conv_id)  # -> bool
api.close()                          # cleanup all resources
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
```

`LiteLLMInterface` is the built-in implementation supporting 100+ providers via litellm.

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
# solution: str, trace: dict with steps, reasoning_types_used, final_confidence
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

12 patterns: `react`, `rewoo`, `reflexion`, `plan_execute`, `prompt_chain`, `self_consistency`, `debate`, `orchestrator`, `adapt`, `evaluator_optimizer`, `maker_checker`, `reasoning_react`.

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

## Monitor (`fsm_llm_monitor`)

```python
from fsm_llm_monitor import MonitorBridge, EventCollector, InstanceManager, create_server

collector = EventCollector(max_events=1000, max_logs=5000)
bridge = MonitorBridge()
bridge.connect(api, collector)
app = create_server(bridge, collector)
# Run with: uvicorn.run(app, host="127.0.0.1", port=8420)
# Or CLI: fsm-llm-monitor
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
