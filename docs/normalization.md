# Naming normalization

> **Proposal, nothing applied.** This is a report with suggestions. No name in `src/`, `tests/`, `docs/` or `examples/` has been changed. It is a planning record, not a guide: it is not covered by the snippet tests and does not carry a version header.

Goal: one vocabulary across the core and the extension subpackages, so the framework reads as one product. The same concept should carry the same name wherever it appears.

Method: three read-only audits (the `workflows` subpackage against core, the conventions of core itself, and a cross-subpackage comparison), followed by spot checks of the cited lines. Line numbers are those found at the time of the audit and will drift. Blast-radius counts come from `grep -rl` over `src/`, `tests/`, `docs/` and `examples/` and count files that contain the token. They include unrelated matches (local variables, prose), so read them as an upper bound.

## How to read this report

Stages run from the clearest outlier outward. Each stage can be reviewed and applied on its own.

| Stage | Scope | Why this order |
| --- | --- | --- |
| 0 | Target vocabulary | Every later stage is measured against it |
| 1 | `workflows` against core | Largest deviation, and the starting point |
| 2 | `monitor` seam to workflows and agents | It wraps workflow names, so it moves with stage 1 |
| 3 | `agents` | Largest package, mostly aligned with core already |
| 4 | `reasoning`, `synthesis`, `eval` | Small surface each |
| 5 | Core's own splits | Last, because it touches every package |

Severity per item: **A** the name differs across packages for the same concept, **B** an inconsistency inside one package or against core, **C** cosmetic or layout.

## Stage 0: target vocabulary

Core (`src/*.py`) is the reference, since every extension imports it. Where core is itself split, the rule below picks one form. Evidence counts are grep counts over core.

| # | Rule | Evidence in core | Today's outliers |
| --- | --- | --- | --- |
| 1 | Ids are `<thing>_id`, spelled out: `conversation_id`, `fsm_id`, `state_id`, `session_id`. Never `conv_id`. | `conversation_id` 431 uses in 53 signatures | `conv_id` in reasoning, agents, synthesis |
| 2 | Time limits are in seconds with the unit in the name. `timeout_seconds` is how long one call may take, `max_seconds` is a run budget. Floats. | `max_seconds` 52 uses, `max_idle_seconds`, `*_TIMEOUT_SECONDS` constants | `timeout`, `workflow_timeout`, `conversation_timeout`, `timeout_s`, `handler_timeout` (no unit) |
| 3 | A step budget is `max_steps`. A budget of something that is not a step keeps its own noun (`max_reflections`, `max_replans`). | `max_steps` in `run_until_terminal` | `max_iterations`, `max_steps_per_run` |
| 4 | `create_<thing>` returns an object. `build_<thing>` returns data or text. A builder uses `add_*` and `set_*` and ends in one `build()`. `with_*` stays on `HandlerBuilder` option chaining. | `create_handler`, `create_okf_handler`; `APIBuilder.set_*`; `build_classification_json_schema` | `create_*` used for text fragments in `visualizer.py`, `with_step` beside `add_step` |
| 5 | Exceptions: root is `<Pkg>Error(FSMError)`. Message first, ids as optional keyword arguments. Nest by specificity. | `StateNotFoundError(message, state_id=None)`, `HandlerSystemError` then `HandlerExecutionError` | workflows put the id first and are flat, `MonitorError(Exception)`, `ReasoningEngineError` |
| 6 | Constants: `DEFAULT_*` for a parameter default, `MAX_*` for a hard cap, `CONTEXT_KEY_*` for a context key, unit suffix (`_SECONDS`, `_BYTES`). A subsystem takes its prefix first (`OKF_`, `SKILL_`). | `DEFAULT_*` 15, `CONTEXT_KEY_*` 5 | `KEY_*`, bare `MAX_STEPS_PER_RUN`, `Defaults` classes |
| 7 | A closed vocabulary is a `str, Enum`: UPPER members, snake_case values. | 5 enums in core | status strings and `STATUS_*`/`REASON_*` constants, `error_mode` as a raw string, `TrustTier` kebab-case values |
| 8 | Lifecycle verbs: `start` begins, `advance` takes one step, `cancel` aborts one run, `end_*` finishes one conversation or run, `close` stops an engine or manager. | `start_conversation`, `end_conversation`, `API.close` | `shutdown`, `destroy_instance`, `remove_instance`, `launch_*` |
| 9 | One parameter name per concept: `llm_interface`, `path`, `model`, `definition`. | `llm_interface` in 6 signatures, `path` 286 uses | `llm=`, `file_path`, `fsm_file`, `fsm_id_or_path` |
| 10 | Method names do not repeat the owning class: `engine.advance(...)`, not `engine.advance_workflow(...)`. | `API.advance`, `API.get_data`, `FSMManager` methods | `WorkflowEngine.*_workflow` |

## Stage 1: `workflows` against core

Files: `definitions.py`, `models.py`, `engine.py`, `steps.py`, `dsl.py`, `constants.py`, `exceptions.py`, `dependency_resolver.py`, `__init__.py`. The package exports 56 names.

Blast radius legend: files containing the token in `src` / `tests` / `docs` / `examples`.

### 1A. Severity A

| Current | Proposed | Where | Reason (rule) | src / tests / docs / examples |
| --- | --- | --- | --- | --- |
| `workflow_timeout` (`start_workflow`, `WorkflowInstance`) | `max_seconds` (a run budget) | `engine.py:333`, `models.py:148` | Core calls a run budget `max_seconds` (2, 3) | 4 / 5 / 0 / 0 |
| `timeout` on `WorkflowStep` and the DSL | `timeout_seconds` | `steps.py:93`, `dsl.py:65` and every step factory | Unit in the name (2). `timeout_seconds` is already used by `WaitEventConfig`, `register_event_listener` and `WorkflowTimeoutError`, so `timeout` is the outlier | see next row |
| `conversation_timeout` | `timeout_seconds` | `steps.py:549` | Same | 3 / 0 / 0 / 0 |
| `delay_seconds` | keep | `steps.py:491` | Not a deviation: it is a delay, not a limit. Listed so it is not reopened | 4 / 5 / 0 / 0 |
| `max_steps_per_run`, `MAX_STEPS_PER_RUN` | `max_steps`, `DEFAULT_MAX_STEPS` | `constants.py:46`, `engine.py` ctor | Core step budget is `max_steps` (3). The constant is a default, so it takes `DEFAULT_` (6) | 5 / 2 / 1 / 0 and 4 / 1 / 0 / 0 |
| `KEY_WAITING_INFO`, `KEY_TIMER_INFO`, `KEY_WORKFLOW_INFO`, `KEY_TIMER_EXPIRED`, `KEY_LAST_EVENT`, `KEY_USER_INPUT`, `KEY_CANCELLATION_REASON` | `CONTEXT_KEY_*` | `constants.py:16-32` | Core context keys are `CONTEXT_KEY_*` (6) | 3 / 2 / 0 / 0 |
| `KEY_TIMEOUT` | `CONTEXT_KEY_EVENT_TIMEOUT` | `constants.py` | It stores timeout info, not a limit. The bare name reads as a limit | same row |
| `next_state`, `success_state`, `error_state`, `failure_state`, `timeout_state`, `default_state` | `next_step_id`, `success_step_id`, ... | `steps.py`, `dsl.py`, `definitions.py` | Core `state_id` means an FSM state. Here the value is a step id, and `initial_step_id` / `current_step_id` already say "step". A third meaning of "state" is `WorkflowStateError.current_state` (a status). Optional, see open decision 1 | 7 / 16 / 1 / 9 |
| `start_workflow`, `advance_workflow`, `cancel_workflow` | `start`, `advance`, `cancel` | `engine.py:329, 1356, 1386` | Method names do not repeat the class (10); `advance` matches `API.advance` (8) | 7 / 13 / 1 / 9, 8 / 5 / 1 / 0, 6 / 4 / 1 / 0 |
| `get_workflow_instance`, `get_workflow_definition`, `get_workflow_status`, `get_workflow_context` | `get_instance`, `get_definition`, `get_status`, `get_context` | `engine.py:1525-1538` | Same (10). Core: `get_data`, `get_current_state` | 6 / 9 / 1 / 9 |
| `get_active_workflows` (returns a list of instance ids) | `list_active_instances` | `engine.py:1543` | It returns instance ids, not workflows. Core uses `list_` for this shape (`list_active_conversations`, `api.py:1993`) | 2 / 0 / 0 / 0 |
| `shutdown` | `close` | `engine.py:288` | Core `API.close()` (8). Monitor's `InstanceManager` also has a `shutdown`, see stage 2 | 5 / 8 / 1 / 0 |
| `register_workflow` | keep | `engine.py:311` | Matches core `register_handler`. The class name is not repeated, it is the registered thing | |
| `remove_instance` | `end` or keep | `engine.py:1475` | Rule 8: a finished run is ended, not removed. Needs a look at what it does (it frees a stored instance); `remove` may be the correct verb for that. Decide in stage 1 review | |

### 1B. Severity B

| Current | Proposed | Where | Reason | Note |
| --- | --- | --- | --- | --- |
| Four step classes whose factory differs from the class: `auto_step` makes `AutoTransitionStep`, `api_step` makes `APICallStep`, `llm_step` makes `LLMProcessingStep`, `wait_event_step` makes `WaitForEventStep` | Rename the classes: `AutoStep`, `APIStep`, `LLMStep`, `WaitEventStep` | `steps.py`, `dsl.py` | The factory is `<kind>_step` for every step, the class should be `<Kind>Step` | 4 / 10 / 0 / 0, 6 / 6 / 2 / 1, 6 / 5 / 2 / 0, 6 / 9 / 0 / 0 |
| Factories with no verb: `auto_step` ... `switch_step`, `workflow_builder`, `linear_workflow`, `conditional_workflow`, `event_driven_workflow` | Keep `<kind>_step` as a deliberate short DSL form. Prefix the rest: `create_workflow_builder`, `create_linear_workflow`, `create_conditional_workflow`, `create_event_driven_workflow` | `dsl.py:519, 540, 577, 623` | Core factories use `create_*` (4). The step helpers are a DSL read in long chains, where the verb costs more than it gives; see open decision 2 | |
| `create_workflow` returns a `WorkflowDefinition` | Returns a builder; the definition comes from `.build()` | `dsl.py:36` | Core `create_handler` returns a `HandlerBuilder` (4). Today `create_workflow` and `workflow_builder` are two entry points to one DSL, which returns different things | check callers before deciding, examples use it |
| `WorkflowDefinition.with_step`, `with_initial_step` | `add_step`, `set_initial_step` | `definitions.py:94, 113` | `WorkflowBuilder` already has `add_step` and `set_initial_step`. Core reserves `with_*` for `HandlerBuilder` chaining (4) | 6 / 9 / 1 / 9 |
| `WorkflowDefinition.validate()` | `validate_structure` | `definitions.py:125` | It shadows pydantic's `validate` and needs `# type: ignore[override]`. Core uses `validate_<what>` (`validate_fsm_structure`) | |
| `WorkflowDefinition.serialize()` | `model_dump`, if equivalent | `definitions.py:327` | Core has no `serialize`; pydantic `model_dump` is used | verify equivalence first |
| `WorkflowStepResult.success_result`, `failure_result` | keep, or drop for plain construction | `models.py:101, 113` | Core `AdvanceResult` and `ClassificationResult` have no such classmethods. Low value either way | |
| Exception constructors: `WorkflowDefinitionError(workflow_id, message, details)`, `WorkflowInstanceError(instance_id, message, details)`, `WorkflowStepError(step_id, message, cause, details)`, `WorkflowTimeoutError(operation, timeout_seconds, details, instance_id)`, `WorkflowValidationError(validation_errors, details)`, `WorkflowStateError(current_state, operation, message, details)`, `WorkflowEventError(event_type, message, details)`, `WorkflowResourceError(resource_type, resource_id, message, details)` | `(message, *, <id>=None, details=None)` for all | `exceptions.py` | Core puts the message first and the id as an optional keyword (5). Eight different positional shapes | |
| Flat tree under `WorkflowError` | Nest by specificity, for example timeout and resource errors under an instance error | `exceptions.py` | Core nests (`HandlerSystemError` then `HandlerExecutionError`) (5). Low priority; nest only where a caller would catch the parent | |
| `add_hook`, `remove_hook`, `LifecycleHook`, string event names (`"status_changed"`, `"step_completed"`) | `register_hook`, `unregister_hook`, an event enum | `engine.py:181, 192, 58` | Core has `register_handler` and the `HandlerTiming` enum (7) | 6 / 3 / 2 / 0 |
| `EventListener`, `Timer`, `DependencyResolver` without the `Workflow` prefix, while the other model classes carry it | One rule: prefix only where the bare name would collide with a core name | `models.py`, `engine.py:78`, `dependency_resolver.py` | Mixed today. `Timer` and `EventListener` are engine-internal; `DependencyResolver` is exported | |
| `WorkflowStateError` | rename to avoid "state" for a status, for example `WorkflowStatusError` | `exceptions.py` | Its name collides with core `StateNotFoundError` / `InvalidTransitionError` vocabulary but concerns run status | |

### 1C. Severity C

| Item | Note |
| --- | --- |
| `definitions.py` holds one class, `models.py` the rest | Core keeps models and exceptions in one `definitions.py`. Either merge, or state the split rule (definition vs runtime state) |
| `WaitEventConfig` is in `models.py` but used only by `steps.py`; `Timer` is in `engine.py` while `EventListener` is in `models.py` | Move next to the single user |
| `engine.py:55` is the only submodule with an `__all__` (it re-exports `MAX_STEPS_PER_RUN`, `Timer`, `WorkflowEngine`) | Core modules do not define one; `__init__.py` is the export point. Drop it |
| `STEP_EXECUTOR` (`steps.py:43`) has a public name but is internal | `_STEP_EXECUTOR` |
| `LifecycleHook` (`engine.py:58`) is public-looking but not exported | Export it or underscore it |
| `get_statistics` | No core analogue. Keep, but its keys (`total_workflows`, `active_workflows`, ...) should follow the renames above |
| `PAUSING_STEP_TYPES` holds class names as strings | Breaks silently when the four step classes are renamed. Derive it from the classes |
| `dsl.py` ordering: `agent_step`, `retry_step`, `switch_step` sit after the pattern helpers | Group all step factories together |

### 1D. Checked and not deviations

`instance_id` and `workflow_id` (consistent, and `instance_id` has no core twin), `event_id`, `DependencyResolver.from_dict`, the `Workflow*Error` prefix, `WaitEventConfig` as a `*Config` name, `delay_seconds`.

## Stage 2: `monitor` seam

The monitor wraps the workflow engine and the agents, so most of its names follow stage 1. Names are as found in the audit.

| Item | Current | Proposed | Severity | Note |
| --- | --- | --- | --- | --- |
| Root exception | `MonitorError(Exception)` (`monitor/exceptions.py:8`) | `MonitorError(FSMError)` | A | Only root outside the `FSMError` tree. AGENTS.md records this as a fact, so changing it is a documented change |
| Workflow run id in requests | `workflow_instance_id` (`monitor/definitions.py:306, 313, 326`) | `instance_id` | A | The engine uses `instance_id`; 6 / 4 / 0 / 0 |
| `start_workflow_instance(instance_id, workflow_id, ...)` (`instance_manager.py:1352`) | The `instance_id` here is the monitor's managed-instance id, a different id from the engine's workflow instance | Rename one side so a reader can tell them apart, for example `managed_id` | A | Same word, two ids |
| Creation verbs | `launch_fsm`, `launch_workflow`, `launch_agent` beside `start_conversation`, `start_workflow_instance` | One verb. `launch_*` creates a managed instance; keep it and give it one meaning | B | Rule 8 |
| Stop verbs | `cancel_agent`, `cancel_workflow`, `destroy_instance`, `end_conversation`, `shutdown` (`instance_manager.py:746`) | `cancel` for a run, `end_*` for a conversation, `close` for the manager | B | Rule 8 |
| Sync and async mixed | `launch_workflow` sync (`:1230`), `start_workflow_instance`, `advance_workflow`, `send_workflow_event`, `cancel_workflow` async (`:1352, 1410, 1423, 1458`) | State the rule; at least do not name a sync and an async call for the same stage with different verbs | B | See open decision 5 |
| Default agent timeout | `DEFAULT_AGENT_TIMEOUT_SECONDS = 120.0` (`monitor/constants.py:15`) against agents `Defaults.TIMEOUT_SECONDS = 300.0` | One default, or document why they differ | B | Same parameter, two values |
| Agent budget | `max_iterations` (`monitor/definitions.py:281`), `DEFAULT_AGENT_MAX_ITERATIONS = 10` | `max_steps` with stage 3 | A | Follows rule 3 |
| Snapshot and info types | `*Snapshot`, `InstanceInfo`, `ActivityItem` | keep | | Monitor-only vocabulary, no core twin |

## Stage 3: `agents`

| Item | Current | Proposed | Severity | Blast radius src / tests / docs / examples |
| --- | --- | --- | --- | --- |
| Step budget | `AgentConfig.max_iterations` (`agents/definitions.py:237`), `Defaults.MAX_ITERATIONS = 10` | `max_steps`. Pattern-specific budgets (`MAX_REFLECTIONS`, `MAX_PLAN_STEPS`, `MAX_REPLANS`, `MAX_REFINEMENTS`, `MAX_REVISIONS`, `MAX_DEBATE_ROUNDS`) keep their nouns | A | 28 / 66 / 2 / 52. Large; examples use it. See open decision 3 |
| Per-tool timeout | `timeout_s` | `timeout_seconds` | A | 7 / 7 / 2 / 0 |
| Run timeout | `AgentConfig.timeout_seconds: float` | keep | | Already the target form |
| Conversation id | `conv_id` in `agents/base.py:677` and internals | `conversation_id` | A | Mostly local variables; check which are parameters |
| Two roots | `AgentError` and `MetaBuilderError(AgentError)` with `BuilderError(MetaBuilderError, BuildError)` | One root. Two bases on `BuilderError` is acceptable because `BuildError` is the cross-package builder error | B | |
| `build()` returns | `ConfiguredAgentBuilder.build() -> Any`, `AgentGraphBuilder.build() -> AgentGraph`, `ArtifactBuilder.build() -> dict` | Type `ConfiguredAgentBuilder.build` precisely. Rename `ArtifactBuilder` (it is tool driven, not fluent), for example `ArtifactAssembler` | B | |
| Constants layout | `Defaults`, `ContextKeys`, `HandlerNames`, `ErrorMessages` classes | Flat `DEFAULT_*`, `CONTEXT_KEY_*` as in core, or declare class-grouped constants the extension style and apply it in reasoning too | B | Core is flat; reasoning and agents are class-grouped. One of the two has to move, see open decision 4 |
| Async naming | `RemoteAgentTool.invoke` and `ainvoke` (`agents/remote.py:421, 442`) | Decide a rule (open decision 5) | C | The only `a`-prefixed name in the extensions |
| Run verb | `run(task, initial_context=None) -> AgentResult` | keep | | See the verb table in stage 4 |

## Stage 4: `reasoning`, `synthesis`, `eval`

| Package | Item | Current | Proposed | Severity |
| --- | --- | --- | --- | --- |
| reasoning | Root exception | `ReasoningEngineError` (`reasoning/exceptions.py:10`) | `ReasoningError`. Every other root is `<Pkg>Error` | A. 4 / 3 / 1 / 0 |
| reasoning | Result type | `solve_problem` returns `tuple[str, dict]` (`reasoning/engine.py:402`) while `SolutionResult` (`reasoning/definitions.py:498`) is exported and unused by the engine | Return `SolutionResult` (core and agents return result objects: `AdvanceResult`, `AgentResult`) | A |
| reasoning | Config and factory | None; `**kwargs` and a `Defaults` class | A `ReasoningConfig` if budgets are to be set by callers, and `create_reasoning_engine` only if the others keep a factory. Optional | B |
| reasoning | Conversation id | `conv_id` (`reasoning/engine.py:496`) next to `conversation_id` (`:606, 642`) in one file | `conversation_id` | A |
| reasoning | Budgets | `Defaults.MAX_SOLVE_STEPS`, `MAX_TOTAL_ITERATIONS`, `MAX_SUB_FSM_ITERATIONS` | `MAX_STEPS` family. Review which are steps | B |
| synthesis | Config | `SynthesisPolicy` (`synthesis/policy.py:117`) | Either keep and state "policy" is a distinct kind of object (limits plus rules), or rename to `SynthesisConfig`. Not decided here | B |
| synthesis | Conversation id | `conv_id` in `ProcedureRunner.run`, `converse`, `end_conversation` (`runner.py:232, 241, 252`) | `conversation_id` | A. These mirror public core methods whose parameter is `conversation_id` |
| synthesis | Time budgets | `max_synthesis_seconds`, `max_simulation_seconds` (ints) and `max_seconds` on `run` | Floats, and one spelling of the budget | B |
| synthesis | Status vocabulary | `STATUS_OK`, `STATUS_FAILED`, `REASON_*`, `GATE_*` string constants | A `str, Enum` per closed vocabulary (7) | B |
| synthesis | Run verbs | `compile`, `obtain`, `run`, `converse` | Fine as domain verbs; make sure `run` matches `run_until_terminal` semantics (`max_steps`, `max_seconds`) | C |
| eval | Timeout | `EvalConfig.timeout: int`, `DEFAULT_TIMEOUT = 120` (`eval/config.py`, `eval/constants.py:41`) | `timeout_seconds: float`; `example_timeouts`, `category_timeouts` follow | A |
| eval | Directory helpers | `create_output_dir` (`eval/examples.py:280`) and `create_run_dir` (`eval/cases.py:490`) | One name for the job | B |
| eval | Entry point | Functions only (`run_examples`, `run_cases`, `run_dataset`) | Keep. State that eval is function-first | C |

Run-verb summary across the packages, for reference: `solve_problem` (reasoning), `run` (agents, synthesis), `start_workflow` then `advance_workflow` (workflows), `launch_*` (monitor), `run_examples` / `run_cases` (eval), `converse` and `advance` (core). The proposal does not force one verb, because the domains differ, but it asks that a call that runs a bounded loop is `run` and takes `max_steps` and `max_seconds`, as core `run_until_terminal` does.

## Stage 5: core's own splits

Core is the reference, but it is not uniform. These items change names that every package imports, so they go last.

| Item | Current | Proposed | Severity |
| --- | --- | --- | --- |
| Classifier LLM parameter | `Classifier(llm=...)`, `HierarchicalClassifier(llm=...)` (`classification.py:123, 502`) against `llm_interface=` on `API`, `FSMManager`, `MessagePipeline` | `llm_interface=` everywhere (and type `ContextCompactor.llm_interface` as `LLMInterface`, not `Any`, `context.py:115`) | A |
| Session ids | `SessionStore` and `API.load_session` use `session_id`; `API.save_session` takes `conversation_id` (`api.py:2040, 2114`) | One spelling for the save and load pair | B |
| Error mode | `API(handler_error_mode=...)`, `HandlerSystem(error_mode=...)` (`api.py:620`, `handlers.py:277`), validated as a raw string | An enum `HandlerErrorMode` and one parameter name | B |
| Path parameters | `load_fsm_from_file(file_path)` (`utilities.py:1177`), `quick_start(fsm_file)` (`__init__.py:461`), `load_fsm_definition(fsm_id_or_path)` (`utilities.py:1224`) against `API.from_file(path)` | `path` | B |
| `create_*` for text | `create_fancy_header`, `create_legend`, `create_states_section` (`visualizer.py:493-1299`) | `build_*` (4). Consistent with `build_graph_representation` in the same file | B |
| Prompt builders | Four classes named `*Builder` in `prompts.py` that render text, next to the fluent builders in `builders.py` and `handlers.py` | Rename the prompt classes (`*Renderer` or `*Composer`) so `Builder` means a fluent builder with `build()` | B |
| Builder grammar | `APIBuilder` uses `set_*`; `HandlerBuilder` uses `with_*`, `on_*`, `when_*` | Documented as two kinds: a configuration builder and a rule builder. No change, but state it | C |
| `TrustTier` values | `"machine-confirmed"`, `"human-reviewed"` (`okf.py:120-121`) | snake_case like the other four enums. Check whether the values are part of the OKF file format first. If they are, keep them and document the exception | C |
| `ContextFilterWorkError` | `utilities.py:174`, a `ValueError` subclass outside `FSMError` | `ContextFilterError(FSMError)` | B |
| `_FrontmatterTooLarge` | `okf.py:1077`, no `Error` suffix | `_FrontmatterTooLargeError` | C |
| Exports | `OKFError`, `SkillError` live in `definitions.py` but are not in `fsm_llm.__all__` | Export them with the other exceptions | B |
| Raw `ValueError` | 129 raises against 24 `FSMError` raises | Not a rename. Out of scope; noted as the largest gap against rule 5 | C |
| `__all__` style | `__init__.py` is grouped by subsystem, `okf.py` and `agent_skills.py` alphabetised (UPPER, Classes, functions) | One ordering | C |
| Class-suffix roles | `Interface` (2), `Manager` (1), `System` (1) with no stated rule | Document when each is used: `Interface` for a replaceable provider, `Manager` for a lifecycle owner, `System` for a dispatch table | C |
| `error_mode` and `ContextMergeStrategy` suffix mix | `Timing`, `Strategy`, `Tier`, `Result` on enums | Not worth a rename | C |

## Cross-package matrix, before and after

| Concept | Today | Proposed |
| --- | --- | --- |
| Conversation id | `conversation_id` (core, monitor), `conv_id` (reasoning, agents, synthesis) | `conversation_id` |
| Workflow run id | `instance_id`, `workflow_instance_id` | `instance_id` |
| Time limit of one call | `timeout` (eval, workflow steps), `timeout_seconds` (agents, monitor, wait step), `timeout_s` (tools), `workflow_timeout` | `timeout_seconds` |
| Run time budget | `max_seconds` (core, synthesis run), `max_synthesis_seconds`, `workflow_timeout` | `max_seconds` |
| Step budget | `max_steps` (core, synthesis), `max_iterations` (agents, monitor), `max_steps_per_run` (workflows), `MAX_SOLVE_STEPS` (reasoning) | `max_steps` |
| Root exception | `FSMError`, `AgentError`, `WorkflowError`, `EvalError`, `SynthesisError`, `ReasoningEngineError`, `MonitorError(Exception)` | `<Pkg>Error(FSMError)` for all |
| Context key constants | `CONTEXT_KEY_*` (core), `ContextKeys` (reasoning, agents), `KEY_*` (workflows), `*_CONTEXT_KEY` (synthesis) | `CONTEXT_KEY_*` |
| Config type | `*Config` (agents, monitor, eval), `SynthesisPolicy`, constructor kwargs (workflows, reasoning) | `*Config` |
| Stop verb | `close`, `end_conversation`, `shutdown`, `destroy_instance`, `remove_instance`, `cancel_*` | `close`, `end_*`, `cancel` |
| Status vocabulary | `str, Enum` (core, workflows), string constants (synthesis, monitor), `StopReason` (agents) | `str, Enum` |

## Application notes

- `examples/` are evaluation baselines and are not to be edited unless asked. Several renames reach them: the `*_state` routing parameters and the `with_step` family touch 9 example files, `max_iterations` 52, `start_workflow` and the `get_workflow_*` getters 9. Every stage that touches these names is blocked until that permission is given, or the example is left on the old name, which cannot work without a shim.
- No aliases or deprecated shims. Every rename is a clean break, with an `Unreleased` entry in `CHANGELOG.md` for each public removal or rename.
- A rename of a name listed in `docs/api_reference.md`, `docs/architecture.md` or the package READMEs and AGENTS.md files requires updating them. The snippet tests (`tests/test_fsm_llm/test_docs_snippets.py`) and the export checks in `tests/test_packaging.py` and `tests/test_fsm_llm_regression/test_regression_cli_and_exports.py` are the guard rails.
- Apply stages in order, one stage per change, each ending with the repo's lint, type check and the suites of the touched packages. Within stage 1, the two sets with the lowest risk and highest vocabulary value are first: time-limit names (rule 2) and context-key constants (rule 6). The method renames (rule 10) and the `*_step_id` rename come last, since they touch every example.
- A behaviour change needs a test that fails on the parent commit. A pure rename does not change behaviour, so the existing tests should fail on the old name and pass on the new one; no new test is needed unless a rename merges two meanings (for instance the monitor's two `instance_id`s).
- `PAUSING_STEP_TYPES` (class names as strings) and `referenced_states()` read names as text. Grep for the string forms of every renamed class and parameter, not only for the identifier.

## Open decisions

1. Rename the routing parameters `next_state`, `success_state`, ... to `*_step_id`? It removes a third meaning of "state" and matches `initial_step_id`, but it is the largest rename and breaks every workflow JSON-like definition and 9 examples. Recommendation: yes, last, and only with example permission.
2. Keep the `<kind>_step` DSL short form, or prefix every factory with `create_`? Recommendation: keep the short form, rename the four classes to match, and apply `create_` to the builder and pattern factories only.
3. `max_iterations` to `max_steps` in agents and monitor (52 examples)? The word "iteration" is accurate for a ReAct loop, and `max_steps` already means "one `advance` call" in core. Recommendation: rename for one vocabulary; if the example cost is too high, rename only the workflows and synthesis spellings and record `max_iterations` as the agents term for a reasoning loop.
4. Constants as flat module names (core, monitor, eval, workflows, synthesis) or class-grouped (`Defaults`, `ContextKeys` in agents and reasoning)? Recommendation: flat, because five of the seven places already use it and `from fsm_llm.constants import ...` is the documented path.
5. Async marking: workflows are async with bare names, monitor mixes, agents have `ainvoke`. Recommendation: no prefix or suffix for a package that is async by design (workflows, monitor), and `_async` or `a` only where a sync twin of the same name exists in the same class.
6. `SynthesisPolicy`: keep as a distinct concept or rename to `SynthesisConfig`.
7. `remove_instance` on the workflow engine: rename to an `end`/`close` verb, or keep `remove` because it frees a stored object.
