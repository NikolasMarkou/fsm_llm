# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Two sets of changes. The entries marked "Step driver" come from plan `07ad3f8c`
(2026-09-30 to 2026-10-01): core gained a message-free step (`API.advance`) and a
bounded run loop (`API.run_until_terminal`), every agent pattern and the harness now
run on them with no synthetic "Continue." message, a silent state says nothing, and
the legacy names listed under Removed are gone with no deprecation period. The other
agents entries come from the 2026-09-29 audit of `fsm_llm.agents` (Track A); its
record, with each finding's status and the deferred Track B work, is
`docs/agents_roadmap.md`.

The entries marked "One LLM layer" come from plan `944e2692` (2026-10-01 to
2026-10-02), the second half of the step driver work. Core gained one request
primitive for tool calling, plain and structured completion
(`LLMInterface.complete`), per-instance usage counters, an embedder
(`LiteLLMEmbedder`) and a general tool-calling and structured-output state (the
optional `completion` state field). Every caller that used to talk to litellm by
itself now goes through that layer: the classifier (on the conversation's own
interface), the LLM judge, semantic memory and semantic tool retrieval, the
meta-builder and `native_fc`; `litellm` is imported only in `src/fsm_llm/llm.py`.
`NativeFunctionCallingReactAgent` and `MetaBuilderAgent` are FSM definitions run by
core, ToolSpec (tool annotations, exact schemas, an enforced timeout) is ported, and
the reasoning engine runs on core's message-free run loop with no synthetic message.

Measured for One LLM layer on `ollama_chat/qwen3.5:4b` (Ollama digest `2a654d98`,
2026-10-01 and 2026-10-02, pre-registered fixed-n blocks): agent bench block
`agents-react/B2`, arm `fsm_toolcall` (`native_fc` on core), 37/38 first-trial
pass@1 at 2.39 LLM calls per task with 0 envelope leaks, against B0 `native_fc`
37/38 at 2.39 (counted by a different meter; every difference is listed in
`scripts/bench_data/README.md`); harness block `l4-execute-write/B2`, arm
`native_fsm`, 40/40 verified writes against B1 `native` 40/40. Reasoning probe (12
pre-registered problems, 2 trials each): the `e1f63a9` baseline passed 22/24 at
63.17 LLM calls per solve with 2 null solutions; the engine on core's run loop that
ships passed 24/24 at 49.71 calls with none; a further variant with typed per-key
fields and silent states passed 18/24 at 28.50 calls, failed its pre-registered rule
(logic problems answered with a bare "True"/"False", an open-ended answer with one
item) and was reverted. Full examples eval (101 examples, 4 workers, same day, Ollama
shared with another loaded model): 386/404 (95.5%) against 367/404 (90.8%) for
`e1f63a9`, 0 F-CODE, 0 envelope leaks; the one example that dropped
(`agents/reasoning_stacking`, a 180 s timeout) scores 4 in about 45 s run alone at both
commits (`EVALUATE.md` Run 009).

Measured on `ollama_chat/qwen3.5:4b` (2026-10-01, `docs/agents_roadmap.md`,
`EVALUATE.md` Runs 007 and 008): agent bench block `agents-react/B1` 32/38 first-trial
pass@1 at 10.97 LLM calls per task against B0's 28/38 at 11.5, 0 envelope leaks,
0 "Continue." in answers; full examples evaluation 388/404 against 391/404 for
`d4b1626` on the same day; agents category 177/192 after the PlanExecute planner fix.

### Added

- Builder standard (plan `89b03f61`): `BuildError(FSMError, ValueError)` in core,
  exported from `fsm_llm`, with `.errors` (the problem strings) and the domain or
  constructor error chained as `__cause__`. A `build()` failure that comes from the
  builder's own rules or from the constructor or validation it calls is a
  `BuildError` chained from the cause; a programming error such as `None` where an
  object is required may still raise its own error.
- Builder standard: `APIBuilder` and `FSMManagerBuilder` (`fsm_llm`),
  `ConfiguredAgentBuilder` (`fsm_llm.agents`, any `create_agent` pattern) and
  `HarnessAgentBuilder` (`fsm_llm.harness`). Each has `set_`/`add_` mutators that
  return the builder and only record, and a `build()` that passes only the values that
  were set to the real constructor, copies what the builder owns and raises
  `BuildError`. Open-ended input has its own setter (`set_llm_option`, `set_option`,
  `set_api_option`) that passes names through unfiltered, except that `build()`
  refuses a name that repeats a named constructor parameter (read from the
  signature) and says which typed setter to use. `ConfiguredAgentBuilder` and
  `HarnessAgentBuilder` copy the `AgentConfig` shallowly: callables inside it stay
  shared. `ConfiguredAgentBuilder.set_option` also refuses `pattern`, `tools`,
  `config` and `system_prompt`. The convention is written once in
  `src/fsm_llm/CLAUDE.md` (Invariants) and `docs/api_reference.md`.
- One LLM layer, core: `LLMInterface.complete(request: CompletionRequest) ->
  CompletionResponse`, the one request primitive for tool calling, plain completion
  and structured output. It is not abstract: the base method raises
  `NotImplementedError`, so existing `LLMInterface` subclasses keep working;
  `LiteLLMInterface.complete` builds the request with the same builder as every other
  call and sends it through the one send path. Every failure is `LLMResponseError`
  (provider error chained, empty reply, unreadable reply shape); a provider
  "malformed tool call" error on a request with tools is data, not an error.
- One LLM layer, core: `CompletionRequest{messages, tools, tool_choice,
  response_format, temperature, max_tokens, call_type}` (frozen, unknown fields
  refused; `tools` and `response_format` never together, `tool_choice` only with
  `tools`), `CompletionResponse{kind, text, calls}` (frozen; `kind` is `"calls"`,
  `"final"` or `"malformed"`) and `ModelToolCall{id, name, arguments: dict}`, all
  exported from `fsm_llm`. A reply whose tool call has no id, no name, or arguments
  that are not a JSON object is `kind="malformed"` with no calls, so none of that
  turn's calls runs; a reply with text and calls is `kind="calls"` with the text kept.
- One LLM layer, core: tool-transcript helpers in `fsm_llm.llm`: `tool_exchange(content,
  calls, results)` (exported from `fsm_llm`; the paired assistant tool-call message
  and one `tool` message per call), `check_tool_transcript(messages)` (refuses an
  unpaired or malformed transcript), `is_malformed_tool_call_error`,
  `decode_tool_arguments`, `implements_complete`, `interface_model`, and
  `constants.MALFORMED_TOOL_CALL_MARKERS`.
- One LLM layer, core: usage counters. `LiteLLMInterface.usage() -> LLMUsage` and
  `reset_usage()` (returns the snapshot it cleared). Every provider call of an instance
  is counted once on that instance, per kind (`generate`, `extract`, `classify`,
  `stream`, `complete`; `constants.USAGE_KIND_*` and `USAGE_KIND_BY_CALL_TYPE`): calls,
  errors, replies without usage, and
  prompt, completion and total tokens. A call that raises counts as a call and an
  error; a streamed call counts with usage missing (also when it fails mid-way).
  `LLMUsage(LLMCallCounts)` adds `by_kind`; both are frozen and exported from
  `fsm_llm`. There is no process-wide counter: a reader meters the interface it owns
  (inject it with `llm_interface=`).
- One LLM layer, core: `LiteLLMEmbedder(model, *, api_key=None, timeout=120.0,
  retries=0, **kwargs)` with `embed(texts) -> list[list[float]]` (one provider request
  for all texts, `[]` and no request for no texts), `usage()` and `reset_usage()`
  (kind `embed`), exported from `fsm_llm`. It has its own model and connection
  settings, never the chat model's. A reply with a missing, uneven, empty or
  non-finite vector raises `LLMResponseError`. Reserved call kwargs
  (`constants.RESERVED_EMBEDDING_CALL_KWARGS`) are ignored with a WARNING.
- One LLM layer, core: the optional `completion` state field
  (`CompletionStateConfig{tools, tool_choice, response_format_key, instructions,
  messages_key="_completion_messages", result_key="completion_result"}`, exported from
  `fsm_llm`; FSM definition format stays v4.1). A completion state's Pass 1 is one
  `complete` call over `[system(instructions)] + context[messages_key]`: a tool-calling
  turn (`tools`) or a structured turn (`response_format_key` names the internal context
  key holding the response format), never both. No prompt builder text, no history and
  no neutral user turn are added. The result `{kind, text, calls}` is written to the
  public `result_key` (handler-only for extraction) and transitions read
  `<result_key>.kind`; the call is made only while that key is unset. The transcript
  is a consumer-owned internal context list, never the conversation history; core runs
  no tool. A transcript that `check_tool_transcript` refuses, or a tool-calling turn
  whose transcript holds no user message, raises `LLMResponseError` before anything is
  sent and the turn rolls back. A completion state may not also declare extraction
  fields, `classification_extractions`, `required_context_keys` or
  `extraction_instructions`, nor be terminal; `fsm-llm-validate` warns when its
  transitions do not cover every result kind. New constants
  `DEFAULT_COMPLETION_MESSAGES_KEY`, `DEFAULT_COMPLETION_RESULT_KEY`,
  `COMPLETION_TOOL_CHOICE_KEYWORDS`.
- One LLM layer, core: `Classifier(schema, model=None, *, llm=None, api_key=None,
  config=None, **llm_kwargs)` and `HierarchicalClassifier(..., llm=None)`: with `llm`
  every classifier request goes to that interface (an `api_key`, extra kwargs, a
  different `model`, or an object that is not an `LLMInterface` implementing
  `complete` raise `ValueError`); without it the classifier builds its own interface
  as before.
- One LLM layer, core: `API.run_until_terminal(..., seconds_exempt_states=())` and the
  stream form: a round whose current state is in the set runs even when the seconds
  budget is spent (the steps budget still applies). Ids are checked against the
  running definition (`ValueError` for an unknown id or a bare `str`).
- One LLM layer, core: `ResponseGenerationRequest.temperature: float | None` (a
  per-call Pass-2 temperature, set by the pipeline from the internal context key
  `constants.CONTEXT_KEY_RESPONSE_TEMPERATURE`, `"_response_temperature"`; a custom
  interface should honour it) and `fsm_llm.api.llm_settings_for(api_kwargs,
  **settings)` (a subpackage's own LLM settings, only when no interface is injected).
- One LLM layer, core: `typed_field_extraction(field_name, field_type, instructions,
  *, context_keys, required=True)` and `clear_keys_on_entry(keys, *, state, name=None,
  priority=100)`, exported from `fsm_llm` (moved from agents); also
  `definitions.TypedFieldType`, `handlers.clear_keys_delta(keys, context)`,
  `definitions.checked_key_names(keys, *, argument)` (refuses `None`, a bare `str` and
  non-`str` members), `constants.EXTRACTION_ENVELOPE_KEYS` and
  `constants.FIELD_PROMPT_CONTEXT_LABEL`.
- One LLM layer, agents: ToolSpec. `ToolAnnotations{read_only, destructive,
  idempotent, open_world}` (exported; `retry_safe` is True for an explicit
  `idempotent` or `read_only`), `ToolDefinition.annotations`, `.timeout_s` (finite, up
  to `Defaults.MAX_TOOL_TIMEOUT_S`) and `.args_model` (the pydantic model of the
  function's parameters, giving exact JSON schemas); `@tool(annotations=,
  timeout_s=)` and `register_function(..., *, annotations=, timeout_s=)`; MCP tool
  annotations are mapped onto `ToolAnnotations`. `tools.tool_parameters_schema(tool)`
  is the one schema source for native tool schemas and prompt descriptions;
  `tools.schema_types(prop, *, root=None)` renders `anyOf`, type lists and local
  `$ref`s (an `Optional[int]` parameter reads `integer or null`, a `str` Enum
  `string`; an unresolvable reference reads `any`).
- One LLM layer, agents: `ToolResult.timed_out`, `ToolResult.status` and
  `ToolResult.observation` (the one place the model-facing text is built:
  `[TOOL FAILED]`, `[TOOL OUTCOME UNKNOWN]` for a timed-out call, or the summary);
  `constants.ToolRunStatus` (`success`, `failed`, `unknown`); every executor trace
  entry carries `tool_status` (AgentHandlers, ParallelReact, `native_fc`, REWOO; REWOO
  `evidence_status` entries too); `tools.refuse_execute_without_gated(registry)`;
  `ErrorMessages.PROMPT_CACHE_WITH_INTERFACE`.
- One LLM layer, agents: `SemanticToolRegistry(..., *, embed_fn=None)`, the same
  per-text callable seam `SemanticMemoryStore` has.
- One LLM layer, agents: the FSM definitions behind the two rebuilt agents and their
  constants: `fsm_definitions.build_native_fc_fsm`, `NativeFCStates`,
  `NativeFCContextKeys`, `NativeFCHandlerNames`; `fsm_definitions.build_meta_builder_fsm`,
  `MetaBuilderStates`, `MetaContextKeys`, `MetaBuildOutcome`, `MetaHandlerNames` and the
  `META_*` constants (`META_BUILD_PROMPT`, the keyword tables, intents and key lists).
  `BaseAgent._step_ceiling(max_iterations)` is the overridable step-ceiling hook.
- One LLM layer, reasoning: `ContextKeys.REASONING_PUSH_PENDING`, `NEEDS_REFINEMENT`,
  `HYBRID_LOOP_COUNT`; `Defaults.MAX_SOLVE_STEPS` (170) and `MAX_HYBRID_LOOPS` (2);
  `HandlerNames.RETRY_KEY_CLEARER` and `HYBRID_LOOP_COUNTER`;
  `ReasoningHandlers.count_hybrid_loop`; constants `RETRY_CLEARED_KEYS`,
  `ORCHESTRATOR_HANDLER_ONLY_KEYS`, `HYBRID_EVALUATION_STATE`, `SOLVE_DRIVER_KEYS`.
- One LLM layer, bench: `scripts/agents_bench.py` meter `wrapper_version` "2" (reads
  the injected interface's `usage()`; "1" stays for recounting B0/B1), arm
  `fsm_toolcall`, manifest disclosures (`llm_request`, `agent_class`, `run_cap`,
  `tool_schemas_sha256`, `first_request`) and the recorded block `agents-react/B2`.
  `scripts/harness_bench.py`: `register`, arm `native_fsm`, `report --blocks` and
  `--pair BLOCK/ARM:BLOCK/ARM`, the shared request-capture and digest helpers, and the
  recorded block `l4-execute-write/B2`.
- Step driver, core: `API.advance(conversation_id) -> AdvanceResult` runs one turn of
  the current state with no user message: the same turn body as `converse` (one turn
  at a time per conversation, rollback on failure, every handler timing, extraction,
  transition evaluation, Pass 2 from the post-transition state, session auto-save),
  resolved on the top of the FSM stack. Nothing is added to the history for a user.
  It raises `FSMError` on a terminal state. Also `FSMManager.advance` and
  `MessagePipeline.advance`.
- Step driver, core: `API.advance_stream(conversation_id) -> Iterator[str]`, the
  streamed form (reply text only; read the outcome afterwards with
  `get_current_state` / `has_conversation_ended`). Also `FSMManager.advance_stream`
  and `MessagePipeline.advance_stream`.
- Step driver, core: `API.run_until_terminal(conversation_id, *, max_steps,
  max_seconds=None, before_step=None) -> tuple[AdvanceResult, ...]` and
  `API.run_until_terminal_stream(...) -> Iterator[str]`. Each round asks whether the
  conversation (top of the stack) has ended, checks the seconds and steps budgets,
  calls `before_step(n)`, asks again whether the conversation ended and re-checks the
  seconds budget (the hook may end it or use up the time), then runs one step. Budgets are checked only between steps. `max_steps` must be an int of at
  least 1 and `max_seconds` `None` or a positive number (`ValueError` for a bool, a
  non-number, NaN or a value out of range).
- Step driver, core: `AdvanceResult{state_before, state_after, transition_outcome,
  response, ended}` (frozen; `response` is `None` for a silent state), exported from
  `fsm_llm`.
- Step driver, core: `RunBudgetExceededError(FSMError)` with `budget` (`"steps"` or
  `"seconds"`), `limit` and `steps_done`, exported from `fsm_llm`. The steps already
  run are kept; the error does not carry their results (read the history and the
  current state).
- Step driver, core: FSM graph data and export, exported from `fsm_llm`:
  `build_fsm_graph(fsm) -> FSMGraph{name, initial_state, nodes, edges}` (frozen
  `FSMGraphNode`, `FSMGraphEdge`; `ValueError` for a definition whose initial state or
  a transition target is missing or whose shape is malformed), `to_mermaid(graph)`
  (Mermaid `stateDiagram-v2`) and `to_dot(graph)` (Graphviz DOT).
  `fsm-llm-visualize --format ascii|mermaid|dot` (default `ascii`; `--style` applies
  to ASCII only) and `visualize_fsm_from_file(..., *, output_format="ascii")`. The
  monitor's FSM visualizer reads the same graph data.
- Step driver, core: `constants.NEUTRAL_USER_TURN` ("Proceed according to the
  instructions above.") is the provider user turn sent when there is no user message
  (a step, the greeting, a context-only classifier call); `constants.EMPTY_USER_MESSAGE_TURN`
  ("(empty message)") is sent when the user's message is empty or whitespace, so an
  empty message is never read as an instruction to proceed. Both are filled in one
  place, the request builder of `LiteLLMInterface`.
- Step driver, core: `LiteLLMInterface.complete_structured(system_prompt,
  user_message, *, json_schema, schema_name)`, the structured call the `Classifier`
  sent through (removed again in this cycle: One LLM layer replaced it with
  `LLMInterface.complete`, see Removed); `DataExtractionPromptBuilder.build_extraction_source_sections(
  instance, scoped_context)`, the context and history sections of the no-message bulk
  extraction prompt.
- Step driver, agents: `StopReason.ENDED` (`"ended"`, a forced stop): the run's
  conversation was closed from outside before a terminal state; the run returns
  `success=False` with its last output.
- Step driver, agents: `ContextKeys.REFUSED_ACTIONS` (`refused_actions`): one sentence
  per gated call the human approver refused and that did not run ("`<tool>(<redacted
  params>)`: was refused by the human approver and was not performed."). It is a run
  output (caller context cannot set it), framework-only (no extraction writes it),
  shown to the conclude prompt of approval-gated FSMs and returned in
  `final_context`. `build_conclude_response_instructions(*, refused_actions=False)`
  adds the sentence that tells the model to report those actions as not performed.
- Step driver, agents: `handlers.call_label(tool_name, tool_input)` (the one
  `<tool>(<redacted normalised params>)` label shared by the trace `action`, the
  refusal record and the ran-check), `handlers.call_ran(trace, tool_name,
  tool_input)` and `handlers.refusal_record(tool_name, tool_input)`.
- Step driver, agents: `ContextKeys.OPERATOR` (`operator`), the ADaPT `decompose`
  typed field (AND/OR; null or other text is AND), and `ReflexionStates.AWAIT_APPROVAL`
  (the shared approval state). Every `*States` class now names every state of its
  FSM and the FSM builders read the constants.
- Step driver, monitor: `InstanceManager.attach_api(api)` shows an `API` you created
  (events and conversations). One API is attached at a time; attaching another
  switches the previous one's handlers off; a failed handler registration raises
  `MonitorConnectionError` and keeps the previous API attached.
- Step driver, bench: `scripts/agents_bench.py` (`register`, `run`, `report`,
  `list-tasks`; 38 deterministic tool tasks in 7 categories with non-LLM graders) and
  `tests/test_agents_bench.py`. Recorded blocks under `scripts/bench_data/agents-react/`:
  `B0` (arms `legacy` and `native_fc`, the agents code of `d4b1626`, copied byte for
  byte) and `B1` (arm `fsm_advance`, the code of this release, pre-registered pass
  rule met). The B0 arm labels are retired: new rows cannot carry them.
- Agents: `AgentResult.stop_reason` (default `None`) and the `StopReason` constants
  (exported from `fsm_llm.agents`): `answered`, `evidence`, `max_iterations`,
  `forced_pass`, `stalled`, `verification_failed`, `no_result`, `gate_failed`.
  `AgentServer` `/invoke` and `/stream` responses include `stop_reason`.
- Agents: `AgentServer(max_concurrent=8)`. A request that finds every slot taken gets
  503; a slot is held until the agent thread ends, even after a 504.
- Agents: context keys `agent_feedback` (executor warnings and HITL denials the next
  think turn reads) and `forced_stop_reason` (written only by a forcing handler).
  Both are run-owned: caller context cannot set them.
- Agents: `Defaults.ADAPT_MAX_SUBTASKS` (8) caps one ADaPT decomposition.
- Tests: `tests.conftest.PromptGroundedLLM`, a fake LLM that answers a field only when
  the prompt contains its evidence, and `block_network`, autoused by the agents and
  meta suites so they cannot open a TCP connection.
- `docs/agents_roadmap.md`: the agents audit record and deferred roadmap.

### Changed

- Install hints: runtime install messages (lazy extension imports, monitor, OTEL,
  MCP, A2A, harness CLI) and the extras docstrings now name a clone install
  (`pip install -e ".[extra]"`), because the `fsm-llm` PyPI name belongs to another
  project.

- Builder standard: `fsm_llm.workflows.WorkflowBuilder.build()` has no `validate`
  argument (removed, not deprecated): it always validates, returns a fresh,
  isolated `WorkflowDefinition` (a later builder call no longer changes it) and
  raises `BuildError` chained from `WorkflowValidationError` or
  `WorkflowDefinitionError`. A caller that caught `WorkflowValidationError` from
  `build()` now catches `BuildError` and reads `__cause__`. `set_initial_step`
  keeps its step in call order (the last call decides the initial step, earlier
  ones stay as steps) and a malformed step or definition field raises `BuildError`
  at `build()`.
- Builder standard: `AgentGraphBuilder.add_node` with a name already used is now a
  `build()` error instead of a silent overwrite; `HandlerBuilder.build()` refuses a
  non-callable execution function. The graph, handler, workflow and meta builders
  raise `BuildError` (still a `ValueError`, so existing `except ValueError` keeps
  working). `HandlerBuilder` keeps its `at`/`on_state`/`when`/`do` vocabulary.
- Builder standard: the meta artifact builders' mutators return the builder, so
  calls chain, and the warnings they used to return as a `list[str]` are read with
  `take_warnings()` (returns and clears). A new `build()` runs `validate_complete()`
  and raises `BuildError`, else returns a deep copy of `to_dict()`. The call-time
  `BuilderError` refusals (empty id, unknown type, missing source) stay, as does
  `update_state`. `BuilderError` is now also a `BuildError`.
- Agents (fix): the meta artifact tools (`create_fsm_tools` and the workflow and
  agent siblings) serialise each call per builder, so parallel tool calls no longer
  steal each other's warnings, and warnings left by direct mutator calls no longer
  leak into the next tool reply.

- One LLM layer, core: the classifier of an AMBIGUOUS transition and of a
  `classification_extractions` entry sends its request to the conversation's own
  `LLMInterface` (`complete`), so a custom `llm_interface` given to `API` is used and
  its timeout, retries and kwargs apply (the 120 s classifier bound now comes from the
  interface's timeout). A private interface is built only when the entry names a
  different `model`, and it inherits no connection kwargs. The classifier cache key is
  the content plus the interface's identity. The pipeline's "no model available"
  skip and the `InvalidTransitionError` at a tie are gone.
- One LLM layer, core: classifier failures. Every failure of the classifier's call is
  `ClassificationError` (an unreadable reply was `ClassificationResponseError`, a dict
  `choices` leaked `KeyError`); a reply text that does not parse stays
  `ClassificationResponseError`. The turn soft-fails (stays, WARNING) only on
  `ClassificationError`: any other exception raised by a custom interface's
  `complete`, and a classifier that cannot be built (a bad `prompt_config`), now fail
  the turn instead of a silent stay. `API` (and `push_fsm`) raise `ValueError` when a
  definition's `classification_extractions` classify through an interface that does
  not implement `complete`; a tie with such an interface still soft-fails each turn.
- One LLM layer, core: `API(llm_interface=...)` raises `ValueError` when it is given any
  other LLM setting beside the interface (`api_key`, a non-None `temperature` or
  `max_tokens`, a `model` other than the interface's own, `seed`, `caching` or any
  other kwarg); they used to be dropped silently.
- One LLM layer, core: request building. `tools`, `tool_choice`, `functions`,
  `function_call` and `n` join `constants.RESERVED_LLM_CALL_KWARGS` (ignored with a
  WARNING when given to a constructor). The empty-turn filler touches only `user`
  messages, so an assistant tool-call message keeps `content: None`. On Ollama a
  `complete` call that sends a `response_format` runs at temperature 0 unless its call
  type is `response_generation`. Every Pass-2 request carries the conversation's
  temperature (`ResponseGenerationRequest.temperature`).
- One LLM layer, core: `fsm_id` hashes `model_dump(exclude_defaults=True)`, so adding an
  optional field to the models no longer changes every id and an explicit default
  hashes like its omission. Every FSM's id changes once against 0.11.0 (restoring a
  session saved under 0.11.0 logs the existing `fsm_id` mismatch WARNING).
- One LLM layer, core: `run_until_terminal` and its stream form call `before_step` once
  when the top of the stack is an ended pushed FSM (stack depth above 1), so the hook
  can pop it and the run continues on the parent; if it does not pop, the run returns
  as before. That call is not a step and is not gated by the budgets; a hook tells it
  from a normal call with `has_conversation_ended`. A pushed child FSM does not
  inherit a completion result key.
- One LLM layer, agents: the LLM judge (`default_llm_judge`) sends through a core
  `LiteLLMInterface` (temperature 0): on Ollama thinking is off; requests carry a
  120 s timeout and `max_tokens` 1000; a provider failure is still `EvaluationError`,
  now chained from core's `LLMResponseError`. `complete_fn=` stays its extension point.
- One LLM layer, agents: `SemanticMemoryStore` and `SemanticToolRegistry` embed through
  a `LiteLLMEmbedder` by default; `rebuild_embeddings` makes one batch request
  (all-or-nothing on failure, one WARNING); a registry with an empty
  `embedding_model` and no `embed_fn` raises `ValueError` at construction.
- One LLM layer, agents: `execute(tool_call, *, gated=False)` on `ToolRegistry`,
  `CachingToolRegistry`, `RetryingToolRegistry` and ReasoningReact's registry view;
  the HITL executor passes `gated=True` for a granted call. A third-party `execute`
  override must accept `gated`: the ReAct family, Reflexion and PlanExecute refuse one
  without it at the start of `run()`, ReasoningReact at construction (`AgentError`);
  REWOO, ParallelReact and `native_fc` never pass it. `RetryingToolRegistry` retries
  only a `retry_safe` tool (annotated `idempotent` or `read_only`) and never a gated
  call or a `requires_approval` tool, so unannotated tools are no longer retried.
  `CachingToolRegistry` neither serves nor stores a gated call.
- One LLM layer, agents: `ToolDefinition.timeout_s` is enforced by `ToolRegistry.execute`
  (a worker thread; the registry lock is never held across the call). A timed-out call
  returns a failed `ToolResult` with `timed_out=True` and `tool_status` `unknown`; the
  Python tool keeps running and its late result is discarded with a WARNING. PlanExecute
  sends an unknown-outcome step to synthesis (no replan, no next step) and reports
  `success=False, stop_reason="no_result"`; a ParallelReact batch with a timed-out call
  is `unknown`.
- One LLM layer, agents: prompt-mode tool descriptions read the same exact schema as
  native tool schemas (`Optional[int]` is `integer or null`, an Enum its value type);
  tools with a `dict` parameter gain `additionalProperties: true`, and harness tools
  with a defaulted path gain `"default": "."` in their schemas.
- One LLM layer, agents: `enable_prompt_cache=True` with an injected `llm_interface`
  raises `AgentError` at construction. `AgentConfig.model`, `temperature` and
  `max_tokens` configure the interface core builds and are not applied to an injected
  one. SelfConsistency sets each sample's temperature on the request, so it works with
  an injected interface.
- One LLM layer, agents: `NativeFunctionCallingReactAgent` runs as an FSM on core
  (`build_native_fc_fsm`: `call_model` and the optional `force_final` and `repair`
  completion states, `run_tools`, `conclude`) through `BaseAgent._standard_run`. The
  model-visible requests are byte-identical to the old loop on 24 recorded scenarios
  (transport keys aside). Changes: each request carries the interface timeout (120 s
  by default, none before); `initial_context` goes through `_init_context` (run-output
  keys and the approval grant stripped; the model still sees only the system message
  and transcript) and `final_context` is filtered like the other patterns; other
  `api_kwargs` reach `API`; `iterations_used` counts loop model turns; a
  whitespace-only answer is `no_result`; the third positional constructor argument is
  `system_policy` (it was `complete_fn`), and `system_policy` must be a `str` or `None`
  (`TypeError`); a forced turn with an unrunnable call runs none of its calls; an empty
  task is sent as `EMPTY_USER_MESSAGE_TURN`; a call with an empty tool name makes the
  turn malformed (it ran as an unknown tool); a provider outage is `AgentError` chained
  from core's `LLMResponseError`. The wall clock is checked only before each loop model
  turn, so the forced and repair turns still run after the deadline, as before. The
  test and caller seam is `llm_interface=`.
- One LLM layer, agents: `MetaBuilderAgent` runs an FSM through core (`classify`,
  `collect`, `build`, `build_failed`, `done`): the artifact type is a
  `classification_extractions` field (intents fsm, workflow, agent and a fallback
  `unknown`), collect replies are core Pass 2, and the build is a structured
  completion state (temperature 0 and core's Ollama preparation). Every model call
  goes to the conversation interface, so `llm_interface=` reaches all of them. Public
  API unchanged; behaviour shifts: the reply of `start(message)` is written by the
  model (canned only for `start("")`), and every collect reply ends with "Say 'build
  it' when you're ready." (appended to the returned reply when the model drops it;
  the stored history keeps the model's text); the agent pattern comes from an
  `agent_type` enum in the build schema (no second classifier; keyword fallback); a
  low-confidence or failed reclassification keeps the previous type; `start` never
  builds; keyword hints match whole words, a negation that governs the build phrase
  ("don't build it yet") blocks it, and a switch word plus a build phrase reclassifies
  before building. A malformed build reply (a JSON schema echo, or a field of the wrong
  type) is a failed build with one validation error per field in `send` and raises
  `MetaValidationError(errors=...)` from `run`; a build-call outage raises
  `BuilderError` (chained from `LLMResponseError`) from `run` and keeps the session
  open with the failed-build reply in `send`. `MetaBuilderConfig.timeout_seconds` is
  the per-request timeout; misplaced constructor kwargs (`model=`, `temperature=`,
  `hitl=`, ...) raise `TypeError`. `fsm-llm-meta` exits 130 on Ctrl-C (it exited 1).
  Prompt tokens per session rose about 43% (core's Pass-2 prompt).
- One LLM layer, reasoning: the engine sends no user message (no "Continue reasoning"
  turns). The classifier FSM runs on core `run_until_terminal` (10 steps; a spent
  budget is `ReasoningClassificationError` chained from `RunBudgetExceededError`). One
  orchestrator run (170 steps, `Defaults.MAX_SOLVE_STEPS`) drives every strategy FSM
  through its `before_step` hook: pushed by type (no FSM dict in context, so none in a
  prompt), popped when it ends or after 30 steps; a failing pop stops the solve.
  `execute_reasoning` entry clears `proposed_solution`, `key_insights` and
  `validation_result`, so a retry produces and validates a new answer. The validation
  verdict, counters, chosen strategy and confidence are handler-only (the model cannot
  open the validation gate); an attempt with no solution counts as a failed one; the
  hybrid back edge runs at most twice. The classifier, strategy executor, validator,
  retry limiter and hybrid counter are critical handlers: their failure stops the
  solve. A spent budget, and any other failed solve, raises `ReasoningExecutionError`
  with `details={conversation_id, responses_so_far, partial_context}` (it used to
  return the fallback string as the solution); ReasoningReact's `reason` tool reports
  a failed call. Malformed classifier values are normalised (a bad recommended type
  becomes `analytical` with a WARNING); a `0` or `False` answer is a solution; the
  driver keys `reasoning_push_pending`, `reasoning_type_selected` and
  `classified_problem_type` are dropped from `initial_context` with a WARNING;
  `all_responses` leaves out silent steps. Prompts reworded where they asked the model
  for handler-owned keys.
- One LLM layer, bench: `scripts/harness_bench.py probe-seed` sends through core's
  LLM layer (Ollama preparation now applies, so a new seed-probe record is not
  byte-comparable with an old one).
- Step driver, core: a silent state (empty `response_instructions`) says nothing on
  every entry. It makes no Pass-2 LLM call and no `LLMInterface` call at all;
  `start_conversation` and `converse` return `""` (they returned a `[<state>]`
  marker), `converse_stream`, `advance_stream` and the stream run loop yield nothing,
  `advance` returns `response=None`, and nothing is added to the history (no marker;
  a `converse` turn still records the user message). `MessagePipeline.process`
  returns `""` for it instead of raising. A custom `LLMInterface` is no longer called
  for silent states.
- Step driver, core: conversation history holds no synthetic entries: no `[state]`
  marker, and a message-free step adds no user exchange.
- Step driver, core: the LLM request models `ResponseGenerationRequest`,
  `FieldExtractionRequest` and `BulkExtractionRequest` carry `user_message: str |
  None` and refuse unknown fields (`extra="forbid"`). `None` means there is no user
  message; a string, `""` included, is what the user sent. A custom `LLMInterface`
  must accept `None` (it gets `None` for message-free steps, the greeting and
  context-only classifier calls).
- Step driver, core: the provider user turn is never empty. No user message is sent
  as `NEUTRAL_USER_TURN`; an empty or whitespace user message as
  `EMPTY_USER_MESSAGE_TURN` (`"(empty message)"`), which a model does not read as
  "proceed" (live: an empty message no longer fires a payment transition or leaves the
  classifier's fallback). This changes the user turn of every greeting (it was
  empty) and of every `converse("")`.
- Step driver, core: prompts have a no-message wording selected by `user_message=None`
  (a string keeps the old prompt byte for byte). Per-field extraction: "Determine the
  value of the field ..." from the instructions, context and conversation, composed
  when the instructions ask for something to be written or decided, null only when
  they cannot be followed. Bulk extraction: built from the state's `read_keys`-scoped,
  security-filtered context and history; on a message-free step it fills only unset
  keys and reports no rejected correction. Classification and ambiguity resolution:
  a context-only system prompt (`build_classification_system_prompt` and
  `build_classification_context_block` take keyword-only `user_message`, default
  `""`; `Classifier.classify`, `classify_multi` and `HierarchicalClassifier.classify`
  accept `None`). Pass 2: `build_response_prompt(user_message: str | None)`; with
  `None` the reply is the state's output, opens with its content (no greeting,
  thanks or remark about the step) and the `<transition_info>` block is dropped.
- Step driver, core: `Classifier` sends its request through its own
  `LiteLLMInterface.complete_structured` (the one request builder; `classification.py`
  no longer imports litellm). Its constructor and `classify` signatures are
  unchanged, but `retries=N` now means the SDK's `max_retries`, a model without
  `response_format` support logs a WARNING on every classifier call, whitespace-only
  user content counts as empty, and the extra keyword arguments are `**llm_kwargs`.
  Tests that patched `fsm_llm.classification.completion` must patch
  `fsm_llm.llm.completion`. (Superseded in this cycle by One LLM layer: the classifier
  sends `LLMInterface.complete` requests, inside a conversation through the
  conversation's own interface.)
- Step driver, core: `API.from_definition(fsm_definition=None, *, definition=None,
  **kwargs)`: the definition is given positionally or under either keyword, exactly
  once.
- Step driver, core: the transition-classification record of an AMBIGUOUS turn lives
  only in `context.metadata["transition_classification"]` (read it with
  `fsm_manager.get_complete_conversation(conv_id)["metadata"]`); handlers no longer
  see it in their context dict.
- Step driver, core: `fsm-llm-validate` reports a `handler_only_keys` entry that no
  state references as INFO, not WARNING (so exported agent FSMs no longer raise a
  false alarm). The `fsm-llm` runner prints no `System:` line for a silent reply.
- Step driver, core: error and log wording on the turn path: "Processing turn in
  state ...", "Failed to process message: ...", and for a step "Failed to advance
  conversation: ..." (manager) and "Failed to advance without a message: ..." (API).
- Step driver, agents: every FSM pattern runs on core's `run_until_terminal` /
  `run_until_terminal_stream` (`BaseAgent._run_conversation_loop` and
  `_standard_run_stream` are thin callers). No agent sends a synthetic "Continue."
  message, counts steps or filters markers. Core holds both budgets: `max_steps =
  max_iterations * FSM_BUDGET_MULTIPLIER` and `max_seconds` = what is left of
  `timeout_seconds`; the agents map core's `RunBudgetExceededError` to
  `BudgetExhaustedError` / `AgentTimeoutError` with the same messages, and the agent
  error's `__cause__` is now the core error. `_check_budgets(start_time)` keeps only
  the wall-clock check (SelfConsistency samples; `native_fc` used it too until One
  LLM layer moved it onto core's run loop).
- Step driver, agents: the HITL driver runs in the run loop's `before_step` hook and
  the seconds budget is checked again after it, so a slow approver uses up the
  timeout before an approved call runs (`AgentTimeoutError`; before, the call ran).
- Step driver, agents: a run whose conversation is closed from outside (by a hook or
  another thread) ends normally and reports `success=False, stop_reason="ended"` with
  its last output, instead of failing with `AgentError` "Conversation ... not found".
  Core's run loops end normally when the conversation is closed mid-run.
- Step driver, agents: a refused gated call is reported as not performed. On a denial
  the driver appends `refusal_record(tool, input)` to `refused_actions`, unless the
  same call (`call_label`) already ran in this run; `spend_grant` removes the matching
  entry when the same call is later approved and runs, and the key is dropped when no
  entry is left. The conclude prompt of approval-gated FSMs tells the model to say
  those actions were not performed (live: 8 of 8 denied runs that reached the gated
  tool answered so). A re-asked identical call is still asked again.
- Step driver, agents: `await_approval` extracts nothing (one LLM call less per visit,
  and no model channel in the state that guards the approval).
  `plan_all`, `orchestrate`, `collect` and ADaPT `attempt`, `assess`, `decompose` have
  no state-level bulk extraction (one call less per visit; their typed fields are
  every key the run reads; ADaPT `operator` is a typed `decompose` field).
  `delegation_plan`, ADaPT `confidence`, `reasoning` and `evaluation_feedback` no
  longer appear in the final context. Ten terminal states lost their dead
  `extraction_instructions`, and four (`conclude` of React, Reflexion and
  ParallelReact, ADaPT `combine`) their `required_context_keys: ["final_answer"]`, so
  the conclude prompt no longer asks to collect a "Final answer".
- Step driver, agents: `final_context["final_answer"]` is no longer an answer source.
  `BaseAgent._extract_answer`, `_completion_is_real` and the ADaPT, EvalOpt and
  PromptChain overrides read the pattern's own answer keys and the last spoken reply;
  a model-written `final_answer` is inert. `final_answer` stays a run output that
  `initial_context` cannot set.
- Step driver, agents: prompt wording without loop-signal text. The conclude
  instructions describe the reply (the run's last output from the observations, no
  further tool, no progress report, a plain statement of what could not be
  determined); the typed-field prompts drop "The user message is only a loop signal".
  PlanExecute's `plan_steps` and replan instructions add no step that needs no tool
  unless the task asks for one, and no step that confirms, waits for or asks for
  anything.
- Step driver, agents: ADaPT `operator` is stripped from caller context like the
  other run outputs. `run_stream` yields only model text from speaking states.
  `ReactAgent.run_stream` on a conversation ended from outside ends without an error
  (Known open).
- Step driver, monitor: `configure(*, manager=None, cors_origins=None, api_key=None,
  trusted_hosts=None)` is keyword-only (a positional call is a `TypeError`). The FSM
  visualize routes draw core's `build_fsm_graph` and answer 400 `failed to parse FSM
  definition: <reason>` for a definition core cannot graph (an empty `{}` or a
  dangling target used to draw a partial graph); every valid definition gives the
  same payload as before. The control page no longer labels `final_answer`.
- Step driver, workflows: `WorkflowEngine(*, max_concurrent_workflows=100,
  max_completed_instances=1000, max_steps_per_run=1000, executor=None)` takes keyword
  arguments only. `ConversationStep` publishes `last_response` / `final_answer` as the
  last reply that was actually spoken and adds neither key when nothing was spoken
  (it published `""` when the last turn ended on a silent state).
- Step driver, harness: the harness runs on core's run loop through `BaseAgent`; its
  step ceiling is unchanged (`MAX_TURNS` 60 x 3 = 180 core steps).
- Core: `FSMDefinition.persona` and `FSMInstance.persona` accept up to 4,000
  characters (was 500), one constant `fsm_llm.constants.MAX_PERSONA_LENGTH`. Persona
  is still rendered only in the Pass-2 response prompt, sanitized; a longer persona
  makes every reply call's prompt larger.
- Agents: `create_agent(pattern="react", tools=None, *, config=None, system_prompt=None,
  **kwargs)` takes the pattern first, so `create_agent("debate")` builds a
  `DebateAgent`. Pattern names are matched after `strip().lower()` (`"Debate "` is the
  debate pattern); any other first argument raises `ValueError` listing the patterns
  (the positional system prompt is removed, see Removed). `system_prompt` counts
  against the prompt limits: over 2,000 characters raises `ValidationError`, and one
  that makes an FSM instruction slot exceed core's 5,000-character limit together with
  the tool catalogue raises `AgentError` at `run()` (naming the slot, the
  instructions length and the tool count). A third positional argument is a
  `TypeError`.
- Agents: `system_prompt` (new `AgentConfig.instructions`, at most 2,000 characters)
  now reaches the model. FSM patterns prefix it to every non-empty state and per-field
  instruction (replies and per-field extractions); it does not reach
  `classification_extractions` (the `use_classification=True` think path), core's
  AMBIGUOUS transition classifier or ReasoningReact's reasoning engine.
  `NativeFunctionCallingReactAgent` uses it as its default `system_policy`. Swarm and
  meta_builder reject it in `create_agent`.
- Agents: `AgentStep.thought`, `ToolCall.reasoning` and `ApprovalRequest.reasoning`
  are now empty in the ReAct family unless `use_classification=True`: the think state
  no longer extracts a `reasoning` field (it collided with the extraction envelope's
  own key), so an approval UI no longer gets the model's stated rationale. Known open
  (LOOP-16, `docs/agents_roadmap.md`).
- Agents: `AgentConfig` and `MetaBuilderConfig` reject unknown fields
  (`extra="forbid"`). A misspelt key, the removed `MetaBuilderConfig.output_path`, or a
  typo in an SOP's `config_overrides` now raises instead of being ignored.
- Agents: `AgentConfig.model` defaults to env `LLM_MODEL` (read when the config is
  built), then `DEFAULT_LLM_MODEL`; `default_llm_judge(model=None)` resolves the same
  way. An explicit model always wins.
- Agents: `success` has one meaning, "the run reached its goal". A run that was forced
  to stop (`max_iterations_reached`, three turns with no tool, a failing
  EvaluatorOptimizer or MakerChecker verdict overridden at its revision or budget
  limit, a Reflexion run that hit `max_reflections` without a passing evaluation, a
  Debate consensus forced by `num_rounds` or the budget, a failed PromptChain gate)
  still returns its last answer but reports `success=False` with the matching
  `stop_reason`. A genuine pass, verdict or conclusion on the budget's last round
  counts as success. SelfConsistency is no longer always `True` (it needs a sample
  with text), a Debate needs a proposition (the last round that had one), and REWOO
  needs at least one tool call that succeeded. A `SwarmAgent` handoff to an unknown
  agent reports `success=False, no_result`. Monitor runs and workflow `AgentStep`s
  show these runs as failed.
- Agents: every agent FSM lists `max_iterations_reached`, `forced_stop_reason`,
  `iteration_count` and `observation_count` in core `handler_only_keys`, so no model
  extraction can write them.
- Agents: in the ReAct family (`ReactAgent`, `ParallelReactAgent`,
  `ReasoningReactAgent` and the ReAct subclasses) `max_iterations=N` now counts think
  turns: for N >= 2 a run that never concludes gets N think turns and N - 1 tool
  calls, and N = 1 behaves like N = 2 (2 think turns, 1 tool call). Before, it
  counted every FSM transition (about half as many tool calls). `ReflexionAgent`
  counts the same think turns but closes a cycle on the act exit, so its forced
  conclusion comes from `evaluate` (N = 1, 2, 3, 4 gave 1, 1, 2, 3 think turns and
  1, 1, 2, 2 tool calls). A run that never concludes by itself takes about twice as
  long before its forced stop. No tool runs after the forced-stop flag is set, and
  the model is still asked on the last think turn: its own conclusion there, backed
  by a tool result, reports `success=True`.
- Agents: `@tool(requires_approval=True)` now gates the tool when `HumanInTheLoop` has
  an approval callback and no policy (the flag is the default policy). With a policy,
  the policy alone decides, as before. `ReactAgent`, `ReflexionAgent` and
  `ReasoningReactAgent` raise `AgentError` when a flagged tool exists and nobody can
  decide on it (`hitl=None`, or a `HumanInTheLoop` with neither a callback nor a
  policy), at construction and again at the start of `run()`/`run_stream()`; before,
  the tool ran unasked after a logged WARNING. A policy without a callback still
  constructs (it logs a WARNING): the policy decides, and a call it gates raises
  `ApprovalDeniedError`.
- Agents: `REWOOAgent`, `PlanExecuteAgent`, `ParallelReactAgent` and
  `NativeFunctionCallingReactAgent` have no approval step, so they now raise
  `AgentError` when their registry holds a `requires_approval` tool (at construction
  and again at the start of `run()`).
- Agents: constructors raise `TypeError` for `hitl=`, `tools=`, `evaluation_fn=` or
  any `HumanInTheLoop` argument (`approval_policy=`, `approval_callback=`,
  `on_escalation=`, `confidence_threshold=`, `approval_timeout=`) on a pattern that
  does not take them, and for `model=`,
  `temperature=`, `max_tokens=` (set them on `AgentConfig`). They used to be forwarded
  to `litellm.completion`, so HITL was silently ignored. Other keyword arguments
  (`seed`, `timeout`, `handlers`, `llm_interface`, ...) still pass through.
  `create_agent` passes `tools` only to patterns whose constructor takes them and
  raises `TypeError` for the others.
- Agents: `initial_context` can no longer set run-owned keys (`final_answer`,
  `should_terminate`, `observation_count`, tool selection and results, approval keys)
  or the driver grant `_approval_granted`; they are dropped with a WARNING and
  `observation_count` is seeded 0. `AgentServer` also drops every internal-prefix key
  from the request context. Swarm hand-offs and AgentGraph edges strip the same keys.
  Each pattern also drops its own run outputs (drafts, verdicts, answers and progress
  keys, e.g. MakerChecker `draft_output`/`checker_passed`, EvalOpt
  `generated_output`, Debate `proposition`/`consensus_reached`, PlanExecute
  `plan_steps`) from `initial_context`, and an AgentGraph node or Swarm hand-off
  target never receives its own run outputs from a predecessor: a forged or inherited
  draft used to ship as a successful answer with no work done.
- Agents: loop values (tool selection, thoughts that route, drafts, critiques,
  verdicts, plans, reflections, step results) are extracted as typed per-field values
  with the task and results in the prompt, cleared before each round, instead of from
  a context-free bulk prompt; intermediate states no longer write a Pass-2 reply.
  `agent_trace` is kept out of these prompts. Each typed field is one LLM call.
  PromptChain no longer bulk-extracts the keys a step's `extraction_instructions`
  name; each step extracts one `chain_step_result`.
- Agents: `ReactAgent.run_stream` yields only model text and wraps errors as
  `AgentError` like `run()`.
  `VerifiedReactAgent` and `AutoMemoryReactAgent` stream by running `run()` and
  yielding the answer once, so verification and memory are kept.
- Agents: the Debate answer is the conclusion written after the last round, not the
  first round's judge verdict. SelfConsistency votes on each sample's last `Answer:`
  line (casefolded) and `confidence` is the share of samples that agree.
- Agents: `AgentGraph` runs nodes in topological order, each once, after all its
  predecessors; the answer comes from the last executed sink. A node that raises or
  returns `success=False` takes no outgoing edge. A cycle raises `ValueError` also for
  a directly constructed `AgentGraph`.
- Agents: `SwarmAgent` gives every member the original task (the hand-off message
  travels in context) and `max_handoffs=N` allows exactly N handoffs.
- Agents: `PlanExecuteAgent` replans when a step's tool fails, `max_replans=N` allows
  exactly N replans, and a new plan replaces the old one. `plan_steps` is a typed list
  capped at 10 steps.
- Agents: `VerifiedReactAgent` periodic reflection notes go to `agent_feedback`, not
  `observations` (they no longer count as evidence).
- Agents: tool calls bind their arguments against the function signature first. A
  `TypeError` raised inside a tool is a failed call, never a retry, and a named
  optional argument is never moved into a missing required one. `register_function`
  infers the parameter schema from type hints like `@tool`, also for
  `functools.partial` objects, callable instances and async callables. A
  positional-only parameter is bound by position from its named value.
- Agents: `RetryingToolRegistry` never retries a `requires_approval` tool: one approval
  covers one execution. A tool gated only by an approval policy is still retried
  (closed later in this cycle by One LLM layer: a granted call is never retried, and
  only `retry_safe` tools are).
- Agents: `ReasoningReactAgent`'s tool view follows the caller's registry live, so a
  tool re-registered there with `requires_approval=True` after construction is gated.
- Core: in an envelope-shaped single-field reply (one carrying `value` or
  `field_name`), `llm._field_value` no longer falls back to `data[field_name]` for a
  field named `reasoning`, `confidence` or `field_name`: there that key is the
  envelope's own explanation, score or echoed name, so with a null `value` the
  fallback returned it as the field's value. A flat reply without `value` or
  `field_name` (`{"confidence": 0.8}` for a field named `confidence`) keeps its value.

### Fixed

- One LLM layer, core: a custom `llm_interface` given to `API` was bypassed by the
  classifier (AMBIGUOUS transitions and `classification_extractions` ran on a private
  litellm interface).
- One LLM layer, agents (TOOL-04): typed tool parameters had coarse schemas
  (`Optional[int]` was `string`); schemas now come from the function's own model.
- One LLM layer, agents (TOOL-07): `RetryingToolRegistry` could re-run a call an
  approver granted when the tool was gated only by an approval policy.
- One LLM layer, agents: the default LLM judge ran with thinking on for `ollama_chat`
  models (live: 8.3 s per verdict, 0.43 s now, same 8/8 verdicts).
- One LLM layer, reasoning: a solve whose validation kept failing never reached its
  retry limit (the rejected solution stayed set and was never re-extracted; the
  baseline ran until core's prompt cap), the model could write the validation verdict
  and retry counters itself, the hybrid strategy's loop counter never moved, the whole
  strategy FSM dict reached a reply prompt through context, a spent budget returned a
  placeholder sentence as the solution, and a failing pop was logged and swallowed.
- Core: a single-field extraction reply that opens with this field's
  `{"field_name": ..., "value": ` envelope but does not parse now yields the
  envelope's `value` on the `str`/`any` rung, never the envelope text. Prose that
  merely contains JSON is still kept verbatim. A complete value is kept whole, also
  when it holds an escape JSON does not define (`\d` in a regex, `C:\Users`), which is
  kept literally. A value cut off by `max_tokens` keeps its decoded prefix (never
  ending in half a surrogate pair), is logged as a WARNING naming the field, and
  returns at confidence 0.3 (`constants.TRUNCATED_SALVAGE_CONFIDENCE`, below the 0.5 of
  other unstructured coercions), so a field `confidence_threshold` of 0.5 rejects it.
  On Ollama this salvage, not the field type, is what keeps an envelope out of an agent
  answer: the Ollama `any` grammar has no object branch.
- Agents: EvalOpt `generated_output`, MakerChecker `draft_output` and PromptChain
  `chain_step_result` are typed `any` again, their type before this release (step 20
  had made them `str`). On a provider without a grammar the model can return a JSON
  deliverable as a native object; a dict/list value reaches `evaluation_fn` and
  `AgentResult.answer` as indented JSON text, so `output_schema` validation parses
  it. `False`, `0`, `{}` and `[]` in an answer key count as no answer (they rendered
  as `"False"`, `"0"`, `"{}"`, `"[]"` and made an unjudged run report success).
- Reasoning: `ReasoningTrace` now dumps `reasoning_types_used` as a sorted list, so
  `python -m fsm_llm.reasoning --output json` and `--save` JSON files list the types
  instead of writing `"<redacted:set>"`. `model_dump()` returns a list too, so the
  `ReasoningReactAgent` observation shows `type=['analytical']` instead of a set repr.
- Monitor UI: the launch dialog's Max Iterations input now allows 1 to 100 like the
  server, the Settings log-level select lists SUCCESS, and SUCCESS and TRACE log lines
  get their own CSS classes. A test ties the HTML to `LOG_LEVELS` and
  `MAX_AGENT_ITERATIONS`.
- `setup_cli_logging` docstring names its three call sites (harness and eval `run()`,
  reasoning `--verbose`).
- Agents (SEC-01, LOOP-12): a caller could forge the HITL driver grant or a finished
  run through `initial_context`, directly or via `AgentServer`, SelfConsistency, Swarm
  or AgentGraph, and run a gated tool without asking.
- Agents (SEC-05, SEC-06): memory tools listed, wrote and deleted the hidden
  `metadata` buffer and printed secrets. Secret-looking tool arguments appeared in
  observations, traces, logs and the approval `context_summary`; they are now
  `<redacted>` there (the tool and `ApprovalRequest.parameters` keep the real values).
- Agents (SEC-09): `AgentServer` 500 responses and `/stream` error events no longer
  contain the exception text; they carry a generic message and an `error_id`.
- Agents (TOOL-01, TOOL-02): a tool whose body raised `TypeError` ran twice; a nested
  `tool_input` dropped its sibling arguments.
- Agents (TOOL-14): `register_agent` tools and `RemoteAgentTool` reported a failed
  sub-agent run (`success: false`) as a successful call.
- Agents (REACT-11): `NativeFunctionCallingReactAgent` ran a tool with `{}` when its
  arguments were not a JSON object; such a turn now ends the loop without running it.
- Agents (MEM-01 to MEM-04): `SemanticMemoryStore(persist_path=...)` never loaded the
  file, so the next save overwrote stored memories; entries without an embedding were
  unreachable once any had one; `max_entries` was not saved; a missing parent directory
  made saving fail; saves were not fsynced; vectors of different length were zipped.
  A corrupt store file now raises `ValueError` instead of being overwritten.
- Agents (LOOP-14): with `handler_timeout`, an approved call whose handler timed out
  could run again on the same approval.
- Agents (PAT-04, PAT-10): `BudgetExhaustedError` and `AgentTimeoutError` from an
  ADaPT subtask or an Orchestrator worker were swallowed; they now end the run. ADaPT
  answered with the failed first attempt and counted its decomposition as a tool call;
  Orchestrator silently dropped subtasks beyond `max_workers` (now listed in
  `skipped_subtasks`, kept out of `worker_results` so the collect step does not
  re-delegate them). Orchestrator, ADaPT, REWOO and the Debate judge extract their
  planning and verdict fields through typed prompts that no longer carry `agent_trace`.
- Agents (REACT-01, REACT-02): Reflexion stored an empty reflection for episode 1 and
  lagged one episode behind; `evaluation_fn` was skipped when the self-evaluation came
  back empty.
- Agents (PAT-01, PAT-02): PlanExecute never replanned, and a string plan was iterated
  per character.
- Agents (PAT-03, PAT-05, PAT-06): Debate rounds reused round 1's text; SelfConsistency
  compared whole replies, so the first sample always won; PromptChain validation gates
  stopped nothing and `chain_results` stayed empty.
- Agents (PAT-09): REWOO reported success when every tool failed, and plan id `"E1"`
  became `EE1` so `#E1` never resolved.
- Agents (PAT-11): MakerChecker drafts and revisions did not see the task or the
  checker's feedback.
- Agents (REACT-04, REACT-05): a VerifiedReact answer rejected on the last attempt kept
  its success flag and a raising verifier counted as a pass; ReasoningReact sent the
  reasoning engine `"{}"` instead of the task, shadowed a user tool named `reason`, and
  dropped the behaviour of Caching/Retrying registries.
- Agents (LOOP-06, LOOP-16, LOOP-17): executor warnings and HITL denials never reached
  the next think turn; observation step numbers repeated after 20 observations;
  `BudgetExhaustedError` cited `max_iterations` instead of the loop ceiling. An early
  `should_terminate` no longer drops an approved call.
- Agents (META-06): the meta-builder's FSM few-shot example targeted an undeclared
  `end` state.
- Agents: `ToolRegistry.register_function` raised `AttributeError` for a
  `functools.partial` or a callable object (a regression from schema inference in
  this release). The two argument-fallback DEBUG log lines printed raw tool input;
  they are redacted like the other tool-input log lines.
- Docs: `docs/api_reference.md` agent snippets passed `model=` to constructors, which
  crashed at run time.

### Renamed

- Agents (meta-builder): `FSMBuilder`, `WorkflowBuilder` and `AgentBuilder` in
  `fsm_llm.agents` and `fsm_llm.agents.meta_builders` are now `FSMArtifactBuilder`,
  `WorkflowArtifactBuilder` and `AgentArtifactBuilder`, with no alias. The old
  `WorkflowBuilder` name clashed with `fsm_llm.workflows.WorkflowBuilder`, a different
  class that keeps its name. `ArtifactBuilder` is unchanged. Removed names:
  `fsm_llm.agents.FSMBuilder`, `fsm_llm.agents.WorkflowBuilder`,
  `fsm_llm.agents.AgentBuilder` and the same three in
  `fsm_llm.agents.meta_builders`. `fsm_llm.harness.AgentBuilder` (a callable alias)
  and `fsm_llm.workflows.WorkflowBuilder` are different objects and keep their names.
- Builder standard: `fsm_llm.workflows.WorkflowBuilder.build(validate=...)` lost its
  `validate` argument (see Changed).

### Removed

- One LLM layer, core: `LiteLLMInterface.complete_structured`; use
  `LiteLLMInterface.complete(CompletionRequest(..., response_format=...))`.
- One LLM layer, core: the private classifier connection-kwargs path of the pipeline
  (`MessagePipeline._classifier_connection_kwargs`, `pipeline._CLASSIFIER_BOUND_NAMES`).
  `Classifier._extract_response` now takes the `CompletionResponse` alone (it took
  `(content, response)`).
- One LLM layer, agents: the `complete_fn` constructor parameter of
  `NativeFunctionCallingReactAgent` and the `CompleteFn` type alias in
  `fsm_llm.agents.native_fc`; inject an `LLMInterface` with `llm_interface=`.
- One LLM layer, agents: the private `native_fc` loop and its helpers
  (`NativeFunctionCallingReactAgent._litellm_complete`, `_complete`,
  `_assistant_message`, module `_degrades_turn`, `_is_malformed_tool_call`,
  `_call_arguments`, `_MALFORMED_TOOL_CALL_MARKERS`; use core
  `llm.is_malformed_tool_call_error`, `llm.decode_tool_arguments`,
  `constants.MALFORMED_TOOL_CALL_MARKERS`), and the `details["malformed_tool_call"]`
  label of its `AgentError` (a malformed tool turn is now data, not an error).
- One LLM layer, agents: `MetaBuilderConfig.build_max_iterations`,
  `MetaBuilderConfig.build_timeout_seconds` and `MetaBuilderConfig.build_temperature`
  (passing one now raises `ValidationError`); use `timeout_seconds` for the request
  timeout.
- One LLM layer, agents: the `max_iterations` default override of `MetaBuilderConfig`
  (25); the field is the inherited `AgentConfig.max_iterations` (default 10), accepted
  and not read by the meta-builder.
- One LLM layer, agents: `MetaDefaults.BUILD_MAX_ITERATIONS`,
  `MetaDefaults.BUILD_TIMEOUT_SECONDS`, `MetaDefaults.BUILD_TEMPERATURE`.
- One LLM layer, agents: the meta-builder's own LLM path and Python state machine
  (`MetaBuilderAgent._llm_call`, `_get_type_classifier`, `_get_agent_type_classifier`,
  `_generate_collect_response`, `_detect_type`, `_execute_build`,
  `_run_deterministic_pipeline`, `_preseed_agent_type`, `_build_result`, the
  `_FSM_SCHEMA`/`_WORKFLOW_SCHEMA`/`_AGENT_SCHEMA`/`_ARTIFACT_SCHEMAS` class attributes
  (the schemas are `meta_prompts.artifact_schema`), and the private fields `_builder`,
  `_artifact_type`, `_complete`, `_build_error`, `_started`, `_messages`).
- One LLM layer, agents: `fsm_definitions._typed_field_extraction`,
  `fsm_definitions._EXTRACTION_ENVELOPE_KEYS` and `fsm_definitions.TypedFieldType`;
  import `typed_field_extraction` and `TypedFieldType` from core (`fsm_llm`,
  `fsm_llm.definitions`) and `EXTRACTION_ENVELOPE_KEYS` from `fsm_llm.constants`. No
  re-export from agents.
- One LLM layer, reasoning: `ContextKeys.REASONING_FSM_TO_PUSH`
  (`"reasoning_fsm_to_push"`); the strategy FSM is pushed by type
  (`ContextKeys.REASONING_PUSH_PENDING`).
- Step driver, core: `FSMManager.cleanup_stale_conversations()`; use
  `FSMManager.prune_orphaned_locks()` (drops orphaned locks) or
  `API.cleanup_stale_conversations()` (ends idle conversations; kept).
- Step driver, core: `ClassificationResult.is_low_confidence`; use
  `is_below_default_threshold` (fixed 0.6) or `Classifier.is_low_confidence(result)`
  (schema threshold; kept).
- Step driver, core: `SchemaValidationError` (nothing raised it); catch
  `ClassificationError`.
- Step driver, core: the no-op `TransitionEvaluatorConfig` fields
  `ambiguity_threshold`, `minimum_confidence` and `evidence_conditions_normalizer`
  (passing one is now a `TypeError`); transitions are ranked by `priority` alone.
- Step driver, core: `HandlerSystem.close()` (it did nothing).
- Step driver, core: the `IntentDefinition` re-export from `fsm_llm.classification`;
  import it from `fsm_llm` (or `fsm_llm.definitions`).
- Step driver, core: `constants.CONTEXT_KEY_CLASSIFICATION_RESULT` and the
  `context.data["_transition_classification_result"]` copy; read
  `context.metadata["transition_classification"]`.
- Step driver, core: the underscore-private `security` names re-exported from
  `fsm_llm.constants`, and the private `pipeline._PROVENANCE_KEY`. The public security
  names stay importable from `constants`.
- Step driver, core: `ResponseGenerationRequest.skip_generation` and the `"."`
  system-prompt sentinel: a silent state makes no interface call (passing
  `skip_generation=` is now a validation error).
- Step driver, core: the `[<state>]` marker that `start_conversation`, `converse` and
  streams returned and recorded for a silent state; they return `""` and record
  nothing.
- Step driver, core: the "Continue." de-anchor branch of the per-field extraction
  prompt (it told the model to ignore a "Continue." message); message-free steps use
  the no-message prompt instead.
- Step driver, agents: `Defaults.CONTINUE_MESSAGE` and the synthetic "Continue."
  turns of the agent loops; the private `BaseAgent._skip_marker_states`,
  `_drop_skip_marker` and `_start_conversation`. Use core's run loop.
- Step driver, agents: the positional system prompt of `create_agent`
  (`create_agent("You are ...", tools)`, deprecated in this release by Track A); any
  first argument that names no pattern raises `ValueError`. Use
  `create_agent(pattern, tools, system_prompt=...)`.
- Step driver, agents: `DecompositionError` and `ToolValidationError` (never raised).
- Step driver, agents: the module `fsm_llm.agents.meta_fsm` (`build_meta_builder_fsm`,
  an unused placeholder; the meta-builder never called it). One LLM layer later added
  a real `build_meta_builder_fsm` in `fsm_llm.agents.fsm_definitions`, which the
  meta-builder runs.
- Step driver, agents: `PromptChainStates.GATE_PREFIX` and
  `SelfConsistencyStates.AGGREGATE` (no state of that name), `Defaults.EVALUATION_THRESHOLD`,
  `ErrorMessages.BUDGET_EXHAUSTED`, `ContextKeys.DELEGATION_PLAN`.
- Step driver, agents: prompt builders whose text no run sends any more:
  `build_approval_extraction_instructions`, `build_orchestrate_extraction_instructions`,
  `build_collect_extraction_instructions`, `build_attempt_extraction_instructions`,
  `build_assess_extraction_instructions`, `build_decompose_extraction_instructions`,
  and the terminal-state builders `build_conclude_extraction_instructions`,
  `build_synthesize_extraction_instructions`, `build_rewoo_solve_extraction_instructions`,
  `build_evalopt_output_extraction_instructions`,
  `build_maker_checker_output_extraction_instructions`,
  `build_orchestrator_synthesize_extraction_instructions`,
  `build_combine_extraction_instructions`, `build_chain_output_extraction_instructions`.
- Step driver, monitor: `MonitorBridge` and the module `fsm_llm.monitor.bridge`;
  `configure(bridge=)`; `server.get_bridge` (use `server.get_manager`);
  `InstanceManager.connect_bridge` (use `InstanceManager.attach_api`). To show your own
  `API`: `manager = InstanceManager(); manager.attach_api(api);
  configure(manager=manager)`.
- Step driver, monitor: `THEME_NAME` and the `COLOR_*` constants (`COLOR_PRIMARY`,
  `COLOR_SECONDARY`, `COLOR_BACKGROUND`, `COLOR_SURFACE`, `COLOR_FOREGROUND`,
  `COLOR_ACCENT`, `COLOR_WARNING`, `COLOR_ERROR`, `COLOR_SUCCESS`, `COLOR_MUTED`,
  `COLOR_BORDER`; the colors live in `static/style.css`), `EVENT_LOG` (no event of
  that type is emitted), the no-op POST_TRANSITION entry of
  `EventCollector.create_handler_callbacks()` (7 callbacks now), and the unused
  front-end accessors `isLogPaused` and `isAuthRequired`.
- Step driver, workflows: `MAX_STEP_DEPTH` (use `MAX_STEPS_PER_RUN`),
  `WorkflowEngine(handler_system=)` and its `handler_system` attribute (never called;
  use `add_hook`), and the private `_KEY_*` / `_STEP_INTERNAL_WHITELIST` aliases in
  `engine.py`.
- Step driver, reasoning: `Defaults.MAX_CONTEXT_SIZE`, `ErrorMessages.CONTEXT_TOO_LARGE`,
  `ErrorMessages.VALIDATION_FAILED`, `ErrorMessages.CALCULATION_ERROR` (no reader).
- Step driver, harness: `Defaults.CONTINUE_MESSAGE`, `Defaults.DECISIONS_COMPRESS_LINES`,
  `CHANGELOG_COMPRESS_LINES`, `LESSONS_IMPORTANCE_MIN`, `LESSONS_IMPORTANCE_MAX`;
  `fsm_llm.harness.storage.PLAN_ID_RE` (use `fsm_llm.harness.PLAN_ID_RE`, defined once
  in `artifacts`); the private `storage._atomic_write_text` alias (use
  `fsm_llm.harness._atomic.atomic_write_text`); the read path for legacy
  `plan_YYYY-MM-DD_<hex8>` plan ids (a cross-plan section headed by one is no longer
  counted as a plan section).
- Step driver, build: `make clean` no longer deletes the pre-2026-09-29
  `src/fsm_llm_<sub>/` directories, and `tests/test_packaging.py` no longer checks for
  them; delete them by hand in an old clone.
- `fsm_llm.harness.constants.HandlerPriorities` and `HandlerNames` no longer define
  `PRE_STEP_GATE`, `END_CONVERSATION` or `ERROR`. Nothing registered them: the pre-step
  gate runs inside the EXECUTE dispatch, and end and error handlers come from
  `BaseAgent`.
- `fsm_llm.harness.PRESENTATION_CONTRACTS` no longer defines `PC-PIVOT`; the driver
  never emitted it and no artifact supplies its candidate-directions or
  ghost-constraints fields.
- Agents: `AgentHandlers.classification_tool_override` and its registration. It read
  a context key nothing wrote, so it never did anything (`AgentHandlers` is not
  exported).
- Agents: unused constants in `fsm_llm.agents.constants`: the classes
  `MetaBuilderStates` (back, with new states, for the meta FSM of One LLM layer) and
  `DecisionWords`; `MetaDefaults.BUILD_MAX_TOKENS`;
  `MetaLogMessages.ARTIFACT_CLASSIFIED`, `BUILD_COMPLETE`, `REVIEW_STARTED`,
  `REVISION_STARTED`; `MetaErrorMessages.BUILDER_NOT_INITIALIZED`,
  `INVALID_ARTIFACT_TYPE`; `ErrorMessages.APPROVAL_DENIED`, `TIMEOUT`, `NO_TOOLS`,
  `MAX_REFLECTIONS`, `MAX_REFINEMENTS`, `MAX_REVISIONS`, `MAX_DEPTH`.
- Agents: `fsm_llm.agents.__all__` is one static list; `ReasoningReactAgent` is always
  exported and `create_agent("reasoning_react")` always available.

### Known open

From One LLM layer (plan `944e2692`; review leftovers and work kept out of scope):

- Out of scope: the ReAct family still extracts `tool_name`/`tool_input` as typed
  fields (about 11 LLM calls per task against 2.4 for native tool calling) and is not
  on the completion state; the reasoning classifier FSM is not a
  `classification_extractions` field; no `parallel_tool_calls` control, no streamed
  tool calls, no per-conversation usage counters; nothing was checked live on a
  non-Ollama provider.
- Reasoning: typed per-key reasoning fields with silent intermediate states cut calls
  by a further 43% but lost answers on the pre-registered probe (18/24 against 24/24:
  logic problems answered with a bare "True"/"False", an open-ended answer with one
  item), so they were reverted; a new attempt needs its own pre-registered probe. A
  final step that fails (for example over core's 30,000-character prompt cap) loses
  that turn's `final_solution` (`partial_context` keeps the validated
  `proposed_solution`). `ReasoningClassificationError` reaches the caller only as the
  `__cause__` chain of `ReasoningExecutionError`. A `final_context` passed back as
  `initial_context` still carries the previous problem's analysis (`problem_type`,
  `reasoning_strategy`). A malformed recommended type falls back to `analytical`
  (63 scripted calls against 45 for the model's own strategy; the cost was not
  measured live). The classifier's unused final reply can approach the prompt cap with
  very long fields (live peak 8.8k of 30k characters). A stuck orchestrator state costs
  up to the 170-step budget in calls. `responses_so_far` in the error details leaves
  out step replies.
- Core run loop: the ended-frame `before_step` call repeats the next step number
  (tell the two calls apart with `has_conversation_ended`), and `seconds_exempt_states`
  ids are matched per round against the current top of the stack, so a pushed child
  state with an exempt id is exempt too. `clear_keys_on_entry` fires only on a
  transition into the state, never on start, push, pop or restore.
  `typed_field_extraction` accepts underscore field names and its preamble says "from
  the task" (the agents wording, kept byte for byte).
- Core classification: a tie between transitions with a custom interface that has no
  `complete` soft-fails every turn, so two unconditional transitions at one priority
  never resolve; a direct `Classifier(llm=...)` refuses such an interface with
  `ValueError` while the pipeline reports `ClassificationError`. Classifier calls to an
  entry's own `model` are not counted on the conversation interface's `usage()`. A
  classifier reply carrying a huge integer fails the turn (`ValueError` in parsing)
  instead of soft-failing.
- Core LLM layer: a custom `LLMInterface` may ignore
  `ResponseGenerationRequest.temperature` (SelfConsistency samples then share one
  temperature); `_response_temperature` is not saved in sessions and not passed to a
  pushed child; the apology retry's temperature is untested; `deepcopy` or pickling of
  a `LiteLLMInterface` after its first call raises `TypeError` (its meter holds a
  lock); constructor kwargs such as `parallel_tool_calls` or `logprobs` reach every
  call, classification included; list-shaped message content is sent to
  Ollama as a Python repr; a provider that omits tool-call ids makes every such turn
  `malformed` (Ollama sends ids); the validator's result-kind coverage warning misreads
  `!=` and variable comparisons.
- Core JsonLogic: `==` against `true` also passes for `"true"`, `"True"`, `1` and `1.0`
  (documented; behaviour unchanged; the meta-builder's gates use `===`).
- Core prompts: the conversational Pass-2 prompt tells the model to acknowledge a
  transition, so replies can name internal phases (meta-builder: "moving into the
  collection phase").
- Tools: a pydantic-model parameter still receives a plain dict (the schema is exact,
  the binder is unchanged); a tool that keeps hanging past its timeout keeps its thread
  (a Python thread cannot be killed), so repeated timeouts pile up threads; the MCP
  annotation mapping is untested against the real `mcp` package, MCP `$defs` are
  dropped (an Enum parameter reads `any`) and an MCP timeout reads `failed`, not
  `unknown`; `allOf`-wrapped and pathological hand-written `$ref`s read `any` or recurse
  deeply in `schema_types`; an `IntEnum` think-state example shows `0`; REWOO,
  ParallelReact and `native_fc` do not check that a registry's `execute` accepts
  `gated`; PlanExecute's `no_result` for a timed-out step does not name the timeout.
- `native_fc`: with an interface built with `timeout=None` the forced and repair turns
  after the deadline are unbounded; an agent given `seed=` beside `llm_interface=` is
  refused only when `run()` builds its `API`; `reasoning_model=` of
  `ReasoningReactAgent` is dropped beside an injected interface. The harness anchor in
  `roles.py` still names the old `native_fc` block (harness source unchanged here).
- Meta-builder: an explicit `null` in a build reply field is a malformed build; a
  one-word trigger with punctuation ("go.") is not a trigger; the negation guard lets
  some negated requests build ("don't you dare build it"); the reply returned to the
  user and the reply stored in core's history differ by the appended build sentence;
  prompt tokens per session are about 43% higher than before.
- Bench: `agents-react/B2` calls (meter "2") and B0 calls (meter "1") come from
  different meters, bridged by an offline and a live parity check; only the first
  request of each task is digested in a manifest; harness rows carry no LLM-call
  count; the L4 react control arm makes few tool calls (as in B0/B1);
  `agents_bench register`/`run` still import litellm (cost-map fetch).

From the step driver work (plan `07ad3f8c`), open until its iteration 2 or later:

- Closed by One LLM layer: every LLM caller is on core's LLM layer (`native_fc` and
  the meta-builder are FSMs on core); the classifier uses the conversation's
  `llm_interface`; the reasoning engine sends no synthetic message; ToolSpec is ported
  (TOOL-04 and TOOL-07 fixed); usage counters exist and the agent bench reads them.
- A message-free step of a non-agent FSM can repeat the same extraction prompt on
  every step.
- `RunBudgetExceededError` does not carry the results of the steps already run (read
  the history and the current state).
- `agents/plan_execute_recovery` times out at 180 s under 4 eval workers (score 1;
  4 at `d4b1626`): the planner now writes a complete 6-step plan (one fetch per
  category per tool) that costs 20 LLM calls against 16 for `d4b1626`'s copied 4-step
  plan, which collected one category only. Run alone it succeeds in about 28 s. The
  pre-registered full-evaluation rule D-044 (b) is recorded as FAILED.
  `hierarchical_orchestrator` timed out once at 300 s in the agents-only re-run (also
  a timeout in the 2026-09-29 `d4b1626` baseline; one run, load or noise).
- PlanExecute costs more calls per task than at `d4b1626` (16.7 vs 15.3 in the live
  probe), a run that completes every plan step can still report `success=False,
  max_iterations` at small limits (also REWOO; pre-existing), and a plan step that
  needs no tool costs a wasted retry call.
- React-family success gap: a run can answer that it did something it never
  attempted and report `success=True` (`hitl_search_publish` answers "published"
  though the gated tool was never selected); a refused run that then answers through
  an ungated tool reports `(True, "answered")`. Forced Reflexion answers can cite
  numbers that are in no observation (also at `d4b1626`).
- Refusal record (`refused_actions`) limits; enforcement is unaffected, only the
  conclude wording can be wrong: the same call is matched by its redacted label text,
  so dict key order or `5` vs `5.0` makes an approved-then-denied repeat look new
  (also for the removal in `spend_grant`), and two calls that differ only in a
  secret-looking value share one record; ReasoningReact's built-in `reason` tool
  builds its own trace label, so the ran-check does not cover it; a failed earlier
  call counts as "ran", so a refused retry gets no record; a call whose empty input
  the executor filled from the task is traced under the filled label. Repeated asks
  of a denied call are not reduced (3 per denied React run).
- An agent `run_stream` whose conversation is ended from outside ends silently (it
  raised before), and no test pins an outside end that lands after the last step.
- Secrets inside free-text string values are not redacted (only secret-looking keys
  and value shapes are).
- The no-message Pass-2 prompt: 4 of 94 live replies were not a clean JSON envelope
  (2 salvaged, 2 accepted as plain text). Greetings of speaking initial states keep
  the conversational Pass-2 prompt. ADaPT `attempt`, `assess` and `decompose` still
  make a Pass-2 call whose text is only a last-resort answer. With the bulk calls
  gone, a typed field that comes back null in REWOO `plan_all` or the orchestrator
  has no second chance (the state takes its fallback edge). `evaluator_optimizer`
  strict runs scored lower in one live probe (5/6 vs 6/6).
- `build_fsm_graph` raises `TypeError`, not `ValueError`, for an unhashable
  `initial_state`.
- The examples binding guard (`tests/test_examples/test_example_calls_bind.py`) does
  not check calls on names whose type it cannot infer.
- The message-free `classification_extractions` path has offline test coverage only.
- Harness live L3 runs took 19.9 to 26.5 s against 15.1 to 17.9 s at `d4b1626`.

From the agents audit reviews (`docs/agents_roadmap.md`, Known open items):

- A long generated artifact (EvalOpt, MakerChecker, PromptChain) can still be cut off
  by the per-field `max_tokens` budget. It ships as a truncated answer, marked only by
  its WARNING and confidence 0.3; PromptChain has no judge, so such a run can report
  `success=True`. Fixing it needs an output-budget change (Track B).
- Orchestrator: subtasks dropped over `max_workers` (`skipped_subtasks`) are not shown
  to the collect judge and do not affect `success`; the answer may be incomplete.
- AgentGraph: a node with `success=False` (a forced pass, a Debate without agreement,
  a budget stop) takes no outgoing edge and no edge can route on failure, so a
  fallback branch cannot be built.
- The `TypeError` denylist of agent constructors includes every `HumanInTheLoop`
  constructor parameter, derived from its signature: a new HITL parameter with a
  generic name would start rejecting a litellm passthrough kwarg of that name.
- Core: validating `FSMDefinition`s on first use from several threads at once can race
  (pre-existing at `c632893`; seen with concurrent Debate panels).

## [0.11.0] - 2026-09-29

### Added (`fsm_llm.eval`, 2026-09-29)

- New subpackage `fsm_llm.eval` (extra `eval`, no third-party dependencies, included in
  `all`) and console script `fsm-llm-eval` (also `python -m fsm_llm.eval`). It gathers
  the evaluation code that lived in `scripts/eval.py` and the generic statistics and
  row helpers of `scripts/harness_bench.py`. Docs: `src/fsm_llm/eval/README.md`, `EVALUATE.md`,
  `docs/api_reference.md`.
- `fsm-llm-eval examples`: the examples evaluator with the same flags, 0-4 scores,
  example names and output layout as `scripts/eval.py`, plus `--examples-dir`,
  `--python`, `--config FILE` and `--fail-under PCT`. A golden table built from the
  old scorer at `0facf56` pins identical scores.
- `fsm-llm-eval run DATASET`: conversation evaluations. A dataset (JSON list, JSON
  object with `config` and `cases`, or JSONL) lists cases with an FSM (path relative to
  the dataset, or inline), optional `initial_context`, user `turns` and `expect`
  checks (`final_state`, `visited_states`, `context`, `context_keys`,
  `responses_contain`, `ended`). Each case runs `trials` times (default 3) through
  `fsm_llm.API`; the run writes `rows.jsonl` (one row per trial, flushed as it
  finishes), `results.json` (per-case and overall pass rates with Wilson 95% intervals)
  and `summary.md`. Once the conversation ends, remaining turns are not sent, and that
  fails the trial unless the case declares `ended`. Sample dataset:
  `evaluation/datasets/simple_greeting_cases.json`: three plumbing cases over
  `examples/basic/simple_greeting` and `name_is_extracted` over
  `evaluation/datasets/name_capture_fsm.json`, which fails for a model that extracts
  nothing.
- `EvalConfig`: every setting has a default and can be set, in increasing precedence,
  by a dataset's embedded `config`, a `--config` JSON file, or a flag. Unknown keys are
  an error. The per-example stdin and timeout tables can be extended or overridden from
  the config file. LLM settings for `run`: `temperature`, `max_tokens` (flags
  `--temperature`, `--max-tokens`) and `llm_kwargs`, extra `fsm_llm.API` arguments
  such as `api_base` (recorded in `results.json` by key only, never by value).
- Exit codes for both commands: `0` finished, `1` usage or input error (a file that is
  not UTF-8 or not JSON included), `2` only when `--fail-under PCT` is given and the
  score is below it (compared exactly in integers, so 57 of 100 meets 57), `130`
  interrupted. Ctrl-C cancels the work not yet started and writes the report files for
  what finished, marked `"interrupted": true`.
- Python API: `run_dataset(path, *, config=None, llm_interface_factory=None,
  checks=(), **overrides)` runs a dataset in one call with the CLI's settings layers;
  `checks` are callables that get each finished trial and return failure messages
  (also on `run_cases`). Also `run_examples`, `run_cases`, `load_cases`, `check_expectations`,
  `wilson_ci`, `fisher_exact_two_sided`, `pass_rate`, `append_row`/`read_rows`,
  `open_run_dir` and more; `run_cases(..., llm_interface_factory=...)` runs offline
  with a fake LLM. Exceptions `EvalError(FSMError)` -> `EvalConfigError`,
  `EvalDatasetError`.

### Changed

- `fsm-llm-eval examples` takes the same flags as the removed `scripts/eval.py`, with
  two behaviour changes: an `--output-dir` that already holds files is now refused
  (exit 1) instead of reused, and a usage error (unknown flag, bad value) exits 1
  instead of argparse's 2, because 2 now means "below `--fail-under`".
- `scripts/harness_bench.py` is unchanged and stays stdlib-only and offline: its
  `wilson_ci`, `fisher_exact_two_sided`, `append_row`, `read_rows`, `_write_json`,
  `_utc_now` and `_git_commit` are copies of the `fsm_llm.eval` helpers, kept equal
  by `tests/test_fsm_llm_eval/test_bench_parity.py`. Delegating them was tried and
  reverted: importing `fsm_llm` pulls litellm, which opens a network socket.
- Example scorecards: "Total wall time" is now the real elapsed time of the run; the
  old value (sum of example durations) is reported as "Total example time". Compare
  old and new scorecards on "Total example time".
- Example `results.json` gains `wall_time_s`, `workers`, `default_timeout`,
  `evaluator` and `interrupted`; every old key keeps its meaning.
- Example runs use a thread pool driving one subprocess per example instead of a
  process pool.

### Fixed

- The examples evaluator ran examples with a hardcoded `.venv/bin/python`; it now uses
  the running interpreter, overridable with `--python`.
- Two runs in the same minute on the same commit and model wrote into the same run
  directory; a new run now gets a `_2`, `_3`, ... suffix and never overwrites one.
- A worker that crashed (runner bug, OS error) was dropped from the scorecard; it is
  now recorded as a score-0 result with the error text.
- A run with no results no longer crashes the scorecard timing section.
- The default model is read from `$LLM_MODEL` when the run starts, not at import.

### Removed

- `scripts/eval.py`: use `fsm-llm-eval examples` or `python -m fsm_llm.eval examples`
  (same flags) from the repository root, or pass `--examples-dir`.
- `scripts/agent_chat_harness.py`, a one-off multi-turn agent diagnostic.
- `evaluation/datasets/oolong_synth_real_subset.jsonl`, which nothing read.

## [0.10.0] - 2026-09-29

### Changed (BREAKING: `fsm_llm` is the single top-level package, 2026-09-29)

The five extension packages now live under `fsm_llm` as subpackages. `src/` holds one
package, `src/fsm_llm/`. There are no compatibility shims: the old top-level names
raise `ModuleNotFoundError`, so update your imports.

| Old import | New import |
|------------|------------|
| `import fsm_llm_agents` / `from fsm_llm_agents import X` | `from fsm_llm import agents` / `from fsm_llm.agents import X` |
| `import fsm_llm_reasoning` / `from fsm_llm_reasoning import X` | `from fsm_llm import reasoning` / `from fsm_llm.reasoning import X` |
| `import fsm_llm_workflows` / `from fsm_llm_workflows import X` | `from fsm_llm import workflows` / `from fsm_llm.workflows import X` |
| `import fsm_llm_monitor` / `from fsm_llm_monitor import X` | `from fsm_llm import monitor` / `from fsm_llm.monitor import X` |
| `import fsm_llm_harness` / `from fsm_llm_harness import X` | `from fsm_llm import harness` / `from fsm_llm.harness import X` |

Submodules move the same way, for example `fsm_llm_agents.meta_cli` is now
`fsm_llm.agents.meta_cli` and `fsm_llm_monitor.server:app` is now
`fsm_llm.monitor.server:app`. Update `mock.patch` targets and `sys.modules` keys too.

**Existing clones:** after pulling this change, run `make clean` (or delete the old
`src/fsm_llm_<sub>/` directories by hand), then reinstall with `pip install -e .`. Git
leaves those directories behind because they still hold ignored `__pycache__/`, and an
empty leftover imports as a namespace package, so `import fsm_llm_agents` would succeed
instead of raising `ModuleNotFoundError`.

- `import fsm_llm` still loads no subpackage. `from fsm_llm import agents` works through
  the normal submodule import; `fsm_llm.agents` is not an attribute of a bare
  `import fsm_llm` until something imports it.
- Console scripts keep their names and point at the new modules:
  `fsm-llm-monitor = fsm_llm.monitor.__main__:main_cli`,
  `fsm-llm-meta = fsm_llm.agents.meta_cli:main_cli`,
  `fsm-llm-harness = fsm_llm.harness.__main__:run`. Reinstall (`pip install -e .`)
  to refresh editable wrappers.
- Module entry points: `python -m fsm_llm.reasoning`, `python -m fsm_llm.agents`,
  `python -m fsm_llm.monitor`, `python -m fsm_llm.harness`.
- **(behavior)** Logging: the single `logger.disable("fsm_llm")` now covers every
  subpackage, so agents, reasoning, workflows, monitor and harness logs are silent until
  `setup_logging()` or `enable_debug_logging()` runs (agents, reasoning, harness and
  monitor logs used to reach loguru's default stderr handler). The workflows
  self-disable is gone and `LIBRARY_LOGGER_NAMES` is `("fsm_llm",)`. The
  `fsm-llm-monitor` CLI enables library logging so its Logs page keeps working; if you
  run `uvicorn fsm_llm.monitor.server:app` yourself, call `setup_logging()` first.
  `fsm-llm-harness` and `python -m fsm_llm.harness` (the new `run()` wrapper) show
  library records at WARNING and above, so worker failures, halts and audit errors
  still reach the terminal; `FSM_LLM_LOG_LEVEL` overrides the level.
  `python -m fsm_llm.reasoning --verbose` shows them at INFO. `fsm-llm-meta` and the
  other `python -m` entry points stay silent by default.
- **(behavior)** Evaluation: `scripts/eval.py` classifies failures partly from each
  example's stderr, and the examples never enable logging, so agent examples no longer
  print extension warnings and tracebacks there. Eval scores from before and after this
  change are not comparable, for agent examples especially; re-baseline before
  comparing.
- **(behavior)** `disable_warnings()` still filters `fsm_llm` and its submodules, which
  now include the five subpackages, so their warnings are silenced too (before, they
  stayed visible). A lookalike sibling such as `fsm_llm_contrib` is still not matched.
- `has_agents()`, `has_reasoning()` and `has_workflows()` keep their names and now probe
  `fsm_llm.agents` etc. The subpackages ship in every install, so they return `True`
  whenever `fsm_llm` is installed. `get_agents()` and friends still raise the install
  hint only when the subpackage itself is missing; a missing third-party dependency
  propagates.
- The extras keep their names (`agents`, `reasoning`, `workflows`, `monitor`,
  `harness`) and now only add third-party dependencies; the code ships with the core.
- Packaging: the wheel's `top_level.txt` is just `fsm_llm`. The monitor frontend's
  `static/pages/`, `static/services/` and `static/utils/` subdirectories were missing
  from the wheel and sdist (package-data was `static/*`); `static/**/*` now ships all
  of them.
- The five sub-package `py.typed` markers are removed; `fsm_llm/py.typed` covers every
  subpackage under PEP 561.
- mypy and coverage run on one root: `mypy src/fsm_llm/`, `--cov=fsm_llm`.
- Test directories keep their names (`tests/test_fsm_llm_agents/` and so on).
- Version stays 0.9.0; this change ships in the next release.

### Fixed (monitor audit, 2026-09-28)

Breaking behavior changes are marked **(behavior)**.

- **(behavior)** Security: state-changing requests from a foreign `Origin` get 403, a
  foreign `Host` gets 400 (DNS rebinding), bodies over 1 MB get 413, and every response
  carries CSP / nosniff / frame-ancestors headers. Trusted hosts and CORS origins come
  from `FSM_LLM_MONITOR_TRUSTED_HOSTS` / `FSM_LLM_MONITOR_CORS_ORIGINS`; the permissive
  CORS origin regex is gone.
- **(behavior)** With an API key set, sensitive reads (conversations, activity, events,
  logs, agent status/result, builder result) and `/ws` require it. `/ws` authenticates
  with a first message `{type: "auth", api_key}` and closes 4401 (bad key) or 4403
  (foreign origin). `FSM_LLM_MONITOR_API_KEY` applies without calling `configure()`;
  new `GET /api/auth` and `--api-key` CLI flag.
- **(behavior)** Error mapping: unknown ids 404 (were 500), busy conversation 409,
  capacity 429 (`MonitorCapacityError`), missing extension 501, bad input 400; 500
  details are generic.
- Secret-looking context entries were shown by the snapshot, workflow status, agent
  result and tool parameters; one redaction path (`collector.redact_context`) now covers
  all of them, and non-JSON values are no longer `str()`-ed.
- Monitor handlers recorded the wrong states (the core never passes `_target_state`);
  they now capture current/target state in `should_execute`. Registration is idempotent,
  handlers are unregistered on destroy, and the POST_TRANSITION duplicate is dropped (7
  handlers per FSM, was 8).
- WebSocket streaming used timestamps and list lengths and dropped or repeated events
  after bursts or ring-buffer eviction; it now uses sequence cursors.
- Workflow runs: status, completion, failure and cancellation come from engine hooks;
  every run of an instance is tracked; unknown runs are 404; launch has a timeout and
  cleans up orphans; engines are shut down on destroy.
- Agents: `max_running_agents` cap, finished instances evicted at `max_instances`, a
  late cancel keeps the result, no double cancelled event, `last_tool` read from the
  conversation log.
- Conversations: terminal conversations are ended after caching their final state;
  `end_conversation` on an unknown id raises; a completed FSM instance reopens on start.
- Config: bounds on refresh interval and buffer sizes, `log_level` validated and applied
  to the log sink, buffers resized live. Request bounds on message, task, iterations,
  timeout and tools.
- Builder: sessions capped at 50 (429), TTL counts from last use, a timed-out send keeps
  the session busy until its thread finishes, delete while busy is 409.
- Dashboard config accepts the builder's unwrapped output; malformed input is 400.
- `get_logs` level filter is ordered and case-insensitive; blocking snapshot routes run
  off the event loop.
- OTEL: bounded open conversation spans, fixed span names, lifecycle spans carry ids only,
  `disable()` restores only its own wrapper.
- Frontend: API key prompt and retry, WebSocket auth, server-side internal-key filtering
  (client `startsWith('_')` removed), workflow "Send event" and "End conversation"
  controls, readable 422 errors, and many UI fixes (stale timers, log dedup/pause,
  drawer re-render, escaping, accessibility).

### Fixed (workflows audit, 2026-09-27)

Breaking behavior changes are marked **(behavior)**.

- `process_event`: one instance's failure (deadline passed, cancelled meanwhile, bad
  state) no longer aborts delivery to the other waiting instances, whose listeners had
  already been consumed. Delivery re-checks WAITING under the instance lock and no longer
  mutates the context of a non-waiting instance.
- A follow-up wait on the same event type kept its timeout: the old code cancelled the new
  wait's timeout after the transition.
- A wait with `timeout_seconds` and no `timeout_state` now FAILS the instance on timeout
  instead of staying WAITING forever; `success_state=""` completes the instance when the
  event arrives. `timeout_state` without `timeout_seconds` is rejected.
- `LLMProcessingStep` works with a core `LLMInterface` (`generate_response`) and with a
  sync or async `generate(prompt)`; before, every core interface failed with
  `AttributeError`.
- **(behavior)** Failures never fall back to the success route: LLM, conversation,
  parallel and agent steps without `error_state` now FAIL the instance, and an agent
  reporting `success=False` fails its step.
- `ParallelStep` counts every failed child (an exception with an empty message was
  aggregated as success), drops internal keys before the `step_<i>_` prefix, rejects
  wait/timer children, keeps the successful children's data on failure, and falls back to
  a shallow copy for contexts that cannot be deep-copied.
- User callables that return a coroutine (a lambda wrapping an async function, an object
  with `async __call__`) are awaited: a `ConditionStep` no longer always takes
  `true_state` and an `APICallStep` no longer maps nothing.
- Cancelling instance `order` no longer cancels the timers of `order_2`; a custom
  `instance_id` still held by the engine is rejected instead of overwriting it.
- The workflow deadline is enforced while WAITING; `WorkflowTimeoutError.instance_id`
  names the instance when `start_workflow` raises.
- **(behavior)** Only synchronous cycles are rejected at registration: loops through a
  timer or event wait are allowed. The recursive driver with a 20-step depth cap is
  replaced by a loop with `max_steps_per_run` (default 1000).
- Events can be targeted (`WorkflowEvent.instance_id`, buffered until the instance
  waits) and correlated (`WaitEventConfig.correlation_key`).
- **(behavior)** `switch_step` defaults to `default_state=None` like `SwitchStep` (an
  unmatched value FAILS instead of completing); its outputs are now
  `switch_<id>_matched`/`_target` (the old `_switch_*` keys never reached the context).
- `AgentStep` maps `"answer"` even with an empty `final_context`, passes `input_mapping`
  as `initial_context`, accepts a plain string result, and adds per-step
  `agent_<id>_answer`/`_success`.
- `ConversationStep` honours the base `timeout`, reports `conversation_<id>_ended`, can
  `require_completion` and `use_user_input`, accepts an `FSMDefinition` object, and
  rejects a missing or doubled FSM source at construction.
- Resources: fired event timeouts leave no timer entry; timers, listeners and buffered
  events are released on every terminal status; failed instances are purged too;
  **(behavior)** `max_completed_instances` defaults to 1000 and history to 1000 entries
  per instance.
- Running instances keep the definition they started with; the engine stores a copy on
  registration.
- `fsm_llm_workflows` logging is off until `setup_logging()`/`enable_debug_logging()`,
  like the core package (new `fsm_llm.logging.enable_library_logging`).
- Smaller: `error_state` on `AutoTransitionStep`/`ConditionStep`; float timer and wait
  durations; every DSL factory exposes `timeout`; history records step errors without
  redundant "running" entries; `DependencyResolver.has_cycles` no longer reports an
  unknown dependency as a cycle; exception constructors copy `details`;
  `register_event_listener` raises `WorkflowEventError` for an empty event type;
  duplicate step ids raise `WorkflowDefinitionError`; `APICallStep` output paths
  (`"a.b"`, `""`); `serialize()` strips custom callables; `shutdown()` awaits its tasks
  and refuses new starts; `start_workflow(wait=False)`; `get_workflow_context` returns a
  copy; lifecycle hooks (`add_hook`); injectable `executor`; `WorkflowHistoryEntry`
  exported; new `constants.py`.
- Event delivery only wakes an instance still at the wait the listener was registered
  for; leaving a step drops its listeners and wait timers; a cancelled `process_event`
  restores the listeners it had not delivered; a background start never runs a step of
  an instance that was cancelled or driven first; long prompts fit the core
  `ResponseGenerationRequest` limits.
- Monitor: workflow status redacts secret-shaped context and history entries; new
  `send_workflow_event` / `POST /api/workflow/{id}/event` (API-key gated).


### Changed

- **Default model is `ollama_chat/qwen3.5:4b` again.** 0.9.0 changed `DEFAULT_LLM_MODEL`
  (and the `scripts/eval.py`, `scripts/agent_chat_harness.py` and
  `react_worker_factory` defaults) to `ollama_chat/qwen3.5:9b-q8_0`; this reverts all of
  them and the docs that state the default. Set `LLM_MODEL` or pass `model=` to use 9b.
  The 0.9.0 live results below were measured on 9b and stay as recorded.

### Live results for 0.9.0 (measured after the release)

Measured on `ollama_chat/qwen3.5:9b-q8_0` (the new default) after v0.9.0. Raw records
were kept outside the repository; numbers are reported as measured.

- `scripts/eval.py`, all 101 examples, `--workers 4`, N=1: 80.9% (327/404). Almost every
  loss is an agent example hitting its eval timeout (the per-example timeouts were tuned
  on the faster 4b model). This is not comparable to the old 95.3% 4b baseline.
- Controlled A/B on the 48 agent examples, same model, workers and timeouts, run back to
  back: pre-plan code (c6e8461) 73.4% (141/192, 17 timeouts); 0.9.0 77.1% (148/192,
  14 timeouts). Six examples improved (both debate examples, pipeline_review,
  react_hitl_combined, reasoning_stacking, concurrent_react) and three went from pass to
  timeout (plan_execute, agent_as_tool, hierarchical_orchestrator). The logs show those
  three progressing, not looping: plan_execute now runs every step's tool (it did not
  before), so a 5-step plan needs more than the 180 s example timeout.
- F-LIVE-02: `TestLiveMemoryAgent` passed 3 of 3 runs (81 s, 68 s, 83 s), up from 1 of 2
  and then 0 of 2 before the follow-up. Runs were sequential with nothing else on the
  GPU.
- FB-05 probe (one orchestrator and one debate run per side): before, the orchestrator
  never extracted `all_collected` (null) and debate stayed at `current_round=1` after 12
  iterations; after, the orchestrator extracts `all_collected=True` and debate runs its
  2 rounds to `current_round=3` in 8 iterations.

## [0.9.0] - 2026-09-24

### Agents follow-up 2026-09-24

Follow-up to the agents audit below (`plans/plan-2026-09-24T091842-c1d5bfbc`, one
commit per step or completion fix, three adversarial review passes). It closes the
audit's HITL security gaps and most of its Known open list, makes CI green on
Python 3.10, 3.11 and 3.12, and changes the default model. Every behaviour change has a
test that fails on the pre-fix code. Full suite: 7,158 tests collected (was 6,940). Core `src/fsm_llm` changed only in
`DEFAULT_LLM_MODEL` and one comment. Reviews and decisions: the plan's
`findings/review-iter-1*.md` and `decisions.md` (D-001 to D-030).

Live results (ollama_chat/qwen3.5:9b-q8_0): measured after the release, see
"Live results for 0.9.0" above.

### Security -- agents follow-up 2026-09-24

- **Approval is a driver-only grant bound to one call (D-004, D-023).** In
  `ReactAgent`, `ReflexionAgent` and `ReasoningReactAgent` with an approval policy, a
  tool the policy gates runs only when the approval driver wrote a grant for that exact
  call: the internal key `_approval_granted` (`ContextKeys.DRIVER_APPROVAL`), whose
  value is `{"tool_name", "parameters"}` of the call the human approved. Core drops
  internal-prefixed keys from every model extraction, so the model cannot write it.
  `AgentHandlers(registry, requires_approval=...)` checks it before a registered tool
  runs (`approval_refusal`): without a matching grant the tool does not run, no
  observation is recorded (a refused call is never conclude evidence), the selection
  is kept, `approval_required` is set, a model-written `approval_granted` is deleted,
  and `tool_status` is `awaiting_approval`, so the driver asks on the next loop
  iteration. The grant is spent by the call it names, so one approval still covers one
  call (D-015). A call changed after the approval is refused: pre-fix `ReactAgent` ran
  `danger(x="EVIL")` after the human approved `danger` with an empty input and the
  model filled the input on the `await_approval` turn. The public
  `approval_granted`/`approval_required` keys still route the FSM and stay
  model-writable; the refusal, not the route, is the boundary.
- **Reflexion asks before a gated tool (D-005).** `build_reflexion_fsm` takes
  `include_approval_state` and builds the same `await_approval` state as
  `build_react_fsm` (shared builders `_await_approval_state` and
  `_approval_think_transition`). Before, a gated tool ran first and the callback was
  asked afterwards, so a denial did not stop it.
- **ReasoningReact has an approval driver (D-005).** `ReasoningReactAgent` now calls
  `_handle_hitl_approval` before each turn, as `ReactAgent` does. Before, the callback
  was never asked and the model opened the gate by extracting `approval_granted`.
- **One approval predicate for all three agents (D-019).** `_hitl_active` and
  `_approval_predicate` moved to `BaseAgent`. The same predicate builds
  `await_approval`, registers the gate and feeds the refusal, so they cannot disagree.
- **The approval is a strict bool (D-023).** The driver asks unless `approval_granted`
  is exactly `True` and stores the callback's answer as `True` or `False`. A
  model-written `"yes"` or a callback returning `None` used to park the run in
  `await_approval` until `BudgetExhaustedError`.
- **The driver asks only about a real gated tool, and sees the full context (D-005).**
  A model that writes `approval_required=True` with tool `none`, an unknown tool or an
  ungated tool is no longer asked about; the driver routes the run back to `think`.
  The policy is evaluated on the same full context (internal keys included) the
  refusal sees, so a policy that reads a `_`-prefixed key asks and runs as expected.
  The approval callback still receives the filtered `get_data` view.
- **`AgentServer` opt-in API key and input limit (D-006).**
  `AgentServer(..., api_key=None, max_input_chars=100_000)`. With `api_key` set,
  `/invoke` and `/stream` need `Authorization: Bearer <key>` or `X-API-Key: <key>`
  (401 otherwise; `hmac.compare_digest` on UTF-8 `surrogateescape` bytes, a copy of the
  monitor's compare). `/health` and `/info` stay open. A request whose
  `len(task) + len(json.dumps(context, ensure_ascii=False))` exceeds `max_input_chars`
  gets 413 before the agent runs; `None` disables the check. The key is checked before
  the size and before body validation. `RemoteAgentTool(..., api_key=None)` sends the
  key as a bearer token on `invoke` and `ainvoke`. An empty or whitespace-only
  `api_key` raises `ValueError` in both classes: `compare_digest(b"", b"")` is true, so
  `api_key=os.getenv("KEY", "")` used to give an open server that looked protected.
  The server is still unauthenticated by default.
- **Monitor `configure(api_key="")` raises `ValueError` (D-006).** The same empty-key
  hole existed in the monitor's mutating routes. An empty or whitespace-only `api_key`
  now raises before any global changes, so a key already configured stays in force.
  The env var `FSM_LLM_MONITOR_API_KEY=""` keeps its documented meaning (no key).

### Behaviour changes to know about -- agents follow-up 2026-09-24

- **Default model is `ollama_chat/qwen3.5:9b-q8_0` (D-003).** `DEFAULT_LLM_MODEL`, the
  harness default (which follows it), `react_worker_factory`, `scripts/eval.py`,
  `scripts/agent_chat_harness.py` and the docs changed from `ollama_chat/qwen3.5:4b`.
  New users who rely on the default download a larger model (about 10 GB). Bench
  records and "measured on 4b" notes are unchanged.
- **Pins: `litellm==1.102.1` and `mcp==2.2.0` in `constraints.txt` (D-002, D-025).**
  litellm 1.100.1 and 1.100.2 import `typing.NotRequired` (3.11+) and fail to import on
  Python 3.10; 1.101.0 is the first fixed release. One pin for every Python version,
  no marker split. mcp is pinned to the version CI ran green, since mcp 2.x already
  renamed `Tool.inputSchema` once.
- **Reflexion and ParallelReact conclude only on tool evidence (D-008).** Their
  `think->conclude` and `act->conclude` edges now carry ReAct's guard
  (`should_terminate` and (`observation_count > 0` or `max_iterations_reached`)), built
  by `_conclude_on_evidence_logic()`. Reflexion's `evaluate->conclude` edge needs
  `evaluation_passed` plus the same evidence, or a forced stop
  (`max_iterations_reached` alone). A turn-1 answer from memory now goes through `act`;
  a Reflexion run that never calls a tool ends at the limiter's forced stop instead of
  concluding on turn 3. Reflexion at `max_iterations=1` no longer raises
  `BudgetExhaustedError`. ParallelReact's empty-batch branch does not clear
  `should_terminate`, so a model that insists on stopping with no batch loops until the
  forced stop (bounded, `max_iterations + 2`).
- **maker_checker forces a pass only in `check` (D-007, D-021).** The limiter still
  counts every turn, but the forced `checker_passed=True` is written only on the
  `check` state (new handler `MakerCheckerForcePass`), and `check->output` also passes
  on `max_iterations_reached == True`. Every shipped draft is one the checker judged;
  before, odd budgets could ship the draft written on the forced `revise` turn
  unjudged. Measured with an always-reject checker: budgets 2 to 10 take the same
  number of turns as before the plan; budget 1 ships a judged draft in 2 turns
  (before the plan it shipped an unjudged one).
- **evaluator_optimizer ends at every budget (D-022).** The limiter's forced
  `evaluation_passed=True` was overwritten by the next evaluation, so with an
  always-failing evaluator the run could raise `BudgetExhaustedError` (budgets 1, 2,
  3 and 10 with the defaults, 2 to 7 with a high `max_refinements`). `_run_evaluation` now
  treats `max_iterations_reached` like the refinement cap and ships the last evaluated
  output.
- **A forced stop reports a verdict the judge did not give.** maker_checker's forced
  pass and evaluator_optimizer's budget stop return `success=True` with
  `checker_passed`/`evaluation_passed=True` in `final_context`. Read
  `max_iterations_reached` to tell a forced stop from a real pass.
- **Orchestrator and debate decisions are typed bools (D-009).** The orchestrator
  `collect` state and the debate judge declare `all_collected` and
  `consensus_reached` as `field_type: "bool"` extractions (one extra extraction call per
  such turn). `all_collected` is cleared on entry to `orchestrate` and
  `consensus_reached` on entry to `propose`, so each round is decided again; before,
  the first round's False stuck and later values were never extracted. The debate no
  longer seeds `consensus_reached=False`, so the model's verdict is extracted: a
  round-1 consensus now ends after 1 judge (it ran 3 or 5 before). The judge->conclude
  edge reads `consensus_reached == True` or `current_round > max_rounds`, so the round
  cap holds in every round (it only worked in round 1).
- **meta_builder `step_type` is an enum (D-010).** The extraction schema sets
  `"enum": sorted(WorkflowBuilder.VALID_STEP_TYPES)`, so a live model cannot invent a
  step type. The agentic `meta_tools.add_step` parameter is still a free string.
- **List tool input stays a list (D-011).** `normalize_tool_input` turns a list into
  `{"input": list}` instead of a string repr. A list reaches a parameter whose schema
  type is `array` as a list; a string-typed or untyped parameter still gets
  `str(list)`, the pre-fix value.
- **`@tool` infers an array schema for list-like annotations (D-011).** Bare `list`,
  `list[X]`, `typing.List[X]`, `tuple[...]`, `set[...]`, `frozenset[...]`,
  `Sequence[X]`, and any of these inside `Optional`/`X | None`, give
  `{"type": "array"}`, with `items: {"type": ...}` for a parametrized generic (OpenAI
  native function calling rejects an array without `items`). They used to be
  `"string"`. The prompt now tells the model to send an array for these parameters.
  Other `Optional` types are unchanged. An empty `tool_input` on a single list-typed
  parameter is filled with `[task]` (D-030); it used to get the bare task string.
- **A `**kwargs`-only tool is called with keywords (D-024).** A tool whose one
  parameter is `**kwargs` used to receive the parameters dict positionally and fail on
  every call. This covers zero-argument MCP tools and the monitor's launched-agent stub
  tools.
- **A single `params` dict tool typed `dict[...]` still gets its dict (D-024).** Only a
  bare `dict` annotation was recognised; `dict[str, Any]`, `typing.Dict[...]` or a
  string annotation (`from __future__ import annotations`) made every call fail. Such
  a parameter is the legacy dict form only when the schema does not name it.
- **`previous_draft` and `previous_output` are dropped from `final_context` (D-012).**
  Checker and refiner prompts still see them during the run
  (`RESULT_DROPPED_CONTEXT_KEYS`).
- **`max_iterations_reached` is always present and cannot be forged (D-007).**
  `_init_context` seeds it `False`, so it appears in every prompt context and in
  `final_context` of the agents that use `_init_context`. Core bulk extraction fills
  only unset keys, so a model that writes `max_iterations_reached: true` no longer
  passes every evidence guard (it concluded React and Reflexion on turn 1 and made
  maker_checker ship its first draft). Only the limiter and stall handlers write True.
- **ReasoningReact builds `await_approval` for any approval policy (D-019).** It used
  to need a policy and at least one tool flagged `requires_approval`. A policy can gate
  any tool, including unflagged ones and the `reason` pseudo-tool.
- **A gated call with no approval callback raises in Reflexion and ReasoningReact
  too.** With a policy and no `approval_callback`, the driver's
  `request_approval` raises `ApprovalDeniedError`, which ends the run as
  `AgentError`, as `ReactAgent` already did. Before, Reflexion ran the tool and
  ReasoningReact let the model approve.

### Fixed -- agents follow-up 2026-09-24

- `AgentServer` returned 422 for every valid `/invoke` and `/stream` body: the request
  models were defined inside `_create_app`, and under `from __future__ import
  annotations` FastAPI read `request` as a query parameter. They are module-level now
  (D-020). No test had ever called the server.
- plan_execute ran each step's tool on `execute_step` entry, before the step's
  `tool_name`/`tool_input` were extracted, so a 1-step plan never called its tool and
  step N ran step N-1's input; every step also recorded `success: False` because the
  compactor had cleared `tool_status`. The executor now runs on `check_result` entry,
  and the checker runs after it in the same cascade (D-024).
- mcp 2.x renamed `Tool.inputSchema` to `input_schema`, so every discovered tool had an
  empty schema and every call failed. `MCPToolProvider` reads either name (D-025).
- An MCP tool error result (`isError`/`is_error`) is a failed tool call. It used to be
  reported as `success=True` and counted as evidence (D-026).
- `SkillLoader` loads a `@tool` reachable under two attribute names once. The first
  attribute in `dir()` order wins; a different function claiming a loaded name is
  skipped with a warning (D-012).
- CI's Python 3.10 leg was red since v0.8.0 (litellm import error, see the pin above).
  Two latent mypy errors in `mcp.py` and four OTEL tests that ran without the
  OpenTelemetry SDK surfaced once CI installed the `mcp` extra; both fixed.

### Tests -- agents follow-up 2026-09-24

- `tests/test_fsm_llm_agents/test_hitl_security.py`: forged `approval_granted` and
  `_approval_granted` with a deny-always callback (tool never runs, callback asked), a
  swapped or filled-in call after approval, Reflexion asking before the tool,
  ReasoningReact asking its callback, the driver asking only about a real gated tool,
  and a policy that reads an internal key.
- `tests/test_fsm_llm_agents/test_remote.py`: the 401/200/413 matrix on both routes,
  bearer and `X-API-Key`, non-ASCII and empty keys, at-limit and over-limit input, and
  `RemoteAgentTool` sending the key.
- `tests/test_fsm_llm_agents/test_forced_stop_flag.py`: a model-written
  `max_iterations_reached` changes nothing in React, Reflexion and maker_checker.
- `tests/test_fsm_llm_agents/test_mcp_stdio.py` with the fixture
  `mcp_fixture_server.py`: real stdio discovery and call, a slow call timing out with
  the child process gone within 5 s, a hung server, and an erroring tool. They run in
  CI, which now installs the `mcp` extra; mcp 1.x was checked only in a local Python
  3.10 venv.
- `tests/test_fsm_llm/test_turn_guard_deterministic.py`: a deterministic
  same-conversation turn-guard test (an Event-blocked mock LLM; exactly one success and
  one "already being processed"), shown to fail with the guard disabled. The live
  `TestThreadSafety` counts only guard outcomes and skips on live-model errors.
- Extended: maker_checker and Reflexion budget sweeps (1 to 10), evaluator_optimizer
  budget stops, `early=True` limiter boundaries for orchestrator, debate and
  prompt_chain, the unknown-then-real tool run asserting success, the harness
  non-native `ReactAgent` path, list and dict parameter dispatch, the SkillLoader alias
  case, the meta_builder enum, and the monitor's empty-key and stub-tool cases.

### Known open -- agents follow-up 2026-09-24

- An approved call can be dropped: the model selects a gated tool with
  `should_terminate=True`, the human approves, and `await_approval->conclude` fires
  before `act`. The tool never runs and the run reports `success=False` (fails closed,
  but the human was asked about an action that never happened).
- The per-tool `requires_approval` flag does nothing without an `approval_policy`.
  Agents with no `hitl` parameter have no approval at all: ParallelReact, REWOO,
  `NativeFunctionCallingReactAgent` and PlanExecute.
- The approval policy sees the call before empty-input recovery, so a
  parameter-sensitive policy can pass `{}` while the tool then runs with the task text
  filled in. The policy gets a shallow copy of the live context, read outside the
  conversation lock; write policies as pure predicates. The driver now calls the policy
  on every turn that selects a registered tool.
- The react-family success rule ("an answer key or at least one tool call") counts
  failed tool calls and forced stops as success: a model that never calls a tool
  reports `success=True` once the budget runs out.
- plan_execute's recorded step `result` is the model's text from before the tool ran
  (only `success` reflects the real call), and a failed tool does not set
  `step_failed`, so it does not trigger a replan.
- ParallelReact: an empty batch with `should_terminate` loops to the limiter (bounded,
  no stall detector).
- Debate: `proposition`, `critique`, `counter_argument` and `judge_verdict` stick
  across rounds (core fills only unset keys), so rounds 2+ judge round-1 text; a judge
  turn whose consensus extraction is null is not counted as a round.
- `@tool` item-type inference: `list[Any]` and nested lists give
  `items: {"type": "string"}`, and `tuple[int, str]` gives `items: {"type": "integer"}`.
  Not worse than the old `"string"` schema.
- A string (or a JSON-array string) sent to a list-typed parameter reaches the tool as
  a `str`, so a type-correct tool iterates its characters (pre-existing).
- `AgentServer` has no rate limiting (use a reverse proxy), and the HTTP body is parsed
  before the size check.
- MCP reconnects on every call, so the 30 s per-call timeout includes starting the
  server. mcp 1.x was tested only locally. `constraints.txt` does not pin pydantic
  (fresh installs resolve 2.13.5).
- Out of scope, recorded: core PRE_TRANSITION handlers do not run on a BLOCKED turn;
  `validate-plan` reports pre-existing errors (`pyproject.toml:33`,
  `fsm_llm_monitor/server.py:1382`) and orphaned decision anchors; `src` grew by about
  +570 net lines in this plan (5 pre-existing bugs found on the way), and
  `fsm_definitions.py` keeps growing.
- F-LIVE-02 live re-check on `ollama_chat/qwen3.5:9b-q8_0`: 3 of 3 passed after the
  release, see "Live results for 0.9.0" above.

### Agents audit 2026-09-24

Audit of `src/fsm_llm_agents` dated 2026-09-24 (`plans/plan-2026-09-24T045559-3e4eb3e5`,
13 steps and 9 completion fixes after review, one commit each). Finding ids below are
the audit's own. Every behaviour change has a test that fails on the pre-fix code,
except the logging and docstring items. Full suite: 6,940 tests collected (was 6,873). `ruff` and `mypy` clean across
all 6 packages. The ReAct-family routing change (first item) changes how a live model's
turns are routed, so the already stale eval baseline was not re-measured.

### Behaviour changes to know about -- agents audit 2026-09-24

- **`think` can no longer stall on a bad tool name (CR-01, the F-LIVE-02 mechanism).**
  In `build_react_fsm` (and so `ReasoningReactAgent`), `build_reflexion_fsm` and
  `build_parallel_react_fsm`, the `think->act` edge used to require a valid tool
  selection. A null or unknown tool name left `think` BLOCKED, and on a BLOCKED turn
  neither the `act` entry handler nor any PRE_TRANSITION handler runs, so the run
  burned the 3x loop ceiling and raised `BudgetExhaustedError`. `think->act` is now the
  unconditional lowest-priority (300) fallback, so the existing no-tool feedback,
  stall detector and iteration limiter run and the agent concludes. The `conclude` and
  `await_approval` edges are unchanged. Every non-concluding `think` turn now goes
  through `act`, which gives corrective feedback where the turn used to sit in `think`
  silently.
- **An unknown tool name is a no-tool turn, not evidence (D-012).** In
  `AgentHandlers.execute_tool`, a tool name that is not `none` and not registered now
  takes the no-tool path: the model gets `Unknown tool '<name>'.` plus the list of
  available tools, the turn counts toward the 3-turn stall detector, and no observation
  is recorded. Before, it recorded a `[TOOL FAILED]` observation that satisfied the
  `think->conclude` evidence guard, so a hallucinated tool plus `should_terminate=True`
  ended with `success=True`. The turn's `tool_status` is now `skipped`, not
  `failed`. Unknown-name and no-tool turns (including the literal `none`) now clear
  `tool_name` and `tool_input` as a real tool call does, and the plain warning turn
  also clears `should_terminate`. Core extracts a key only while it is unset, so a
  kept name used to block every later real tool call. `ParallelReactAgent` is
  unchanged.
- **Fallback edges out of more loop states (FB-01, D-002).** A turn where the routing
  key was never extracted used to BLOCK the state until `BudgetExhaustedError`, because
  no PRE_TRANSITION limiter runs on a BLOCKED turn. Unconditional priority-900 edges now
  continue the loop: maker_checker `check->revise`, ADaPT `assess->combine` and
  `decompose->combine`, evaluator_optimizer `generate->evaluate`. Every agent loop state
  now has an unconditional fallback except plan_execute `plan` (its key is seeded) and
  `await_approval`. In ReactAgent the approval driver always writes a bool before
  `await_approval` routes. ReasoningReactAgent uses the same state with no driver, so
  it can BLOCK there until the 3x ceiling (see Known open).
- **maker_checker and evaluator_optimizer re-judge every round (D-013).** Core extracts
  a key only while it is unset, so maker_checker judged `checker_passed` and
  `quality_score` once per run and `revise` never replaced `draft_output`; the
  evaluator_optimizer `refine` state never replaced `generated_output`, so the
  evaluation function scored the first draft on every round. Entering `revise`/`refine`
  now moves the draft to `previous_draft`/`previous_output` and clears the draft (and,
  in maker_checker, a False verdict), so the revised draft really replaces the previous
  one and is judged again. Leaving `revise` clears the consumed `checker_feedback`, so
  the next check writes fresh feedback; if no new draft was produced, the previous one
  is restored. When `checker_passed` is already True (a forced pass from the quality
  auto-pass or `max_revisions`), entering `revise` changes nothing, so the draft the
  checker judged is the one that ships; the forced pass still costs one extra
  `revise` turn. One exception (see Known open): when the iteration limiter forces
  the pass on a `revise` turn, the new draft skips `check` and ships unjudged; the run
  reports `max_iterations_reached=True`. `previous_draft`/`previous_output` stay in `final_context` and in
  later prompts (the checker sees both drafts); answer extraction never reads them.
  `max_revisions` now ends the maker_checker loop (always-False checker at
  `max_iterations=10`: 12 turns with 1 verdict before, 8 turns with 3 verdicts now).
  Two prompt lines changed to point at the new keys (`Your previous output is in
  'previous_output'.`, `Your previous draft is in 'previous_draft'.`); this was not
  measured on a live model. New handler factory `make_redraft_handlers`.
- **Iteration limiters share one factory (PT-05/PT-06/FB-06).** The 8 hand-rolled
  PRE_TRANSITION limiters (plan_execute, rewoo, evaluator_optimizer, maker_checker,
  prompt_chain, debate, orchestrator, adapt) now use
  `fsm_llm_agents.handlers.make_iteration_limiter(max, forced, *, context_max_key=None,
  early=True)`. With `early=True` it forces its keys at `count >= max_iterations - 1`
  (only adapt did this before); maker_checker passes `early=False` and triggers at
  `max_iterations` as before, so the checker judges at every budget (D-014). The early
  trigger changes routing only where a transition reads the forced keys: orchestrator,
  debate and prompt_chain now stop one iteration earlier (adapt already did). For plan_execute, rewoo
  and evaluator_optimizer the forced keys do not change routing (no transition reads
  them, the loop is linear, or `_run_evaluation` overwrites them). `DebateAgent.run()`
  no longer stores `_max_fsm_iterations` on the instance (PT-02).
- **HITL approvals are single-use (D-015).** `approval_granted` was never reset, so
  after one granted approval every later approval-required call skipped the callback
  and ran. This approval bypass predates the audit. `execute_tool` now deletes
  `approval_granted` at `act` entry once no approval is pending, so each gated call is
  asked for exactly once. This holds for ReactAgent only. It does not make approval a
  security boundary: ReflexionAgent, ReasoningReactAgent and model-written approvals
  have open gaps (see Known open).
- **Empty `tool_input` recovery checks the parameter type (CR-02).** When a tool with a
  single required parameter is called with no input, the task string is used as that
  parameter only when its schema type is absent, `"string"`, or a list containing
  `"string"`. A property schema that is not a dict (a JSON-Schema `true`, a description
  string) counts as untyped. An integer parameter used to receive prose and fail with a
  `TypeError`; the call now fails with `Tool requires parameters: [...]`, naming what
  the model must supply.
- **MCP calls time out (MM-06).** `MCPToolProvider`, `from_stdio` and `from_url` take
  `timeout` (default `Defaults.MCP_TIMEOUT_SECONDS = 30.0`; `None` means no limit). A
  discovery that runs past it raises `AgentTimeoutError`; a hung tool call raises
  `ToolExecutionError`, which `ToolRegistry.execute` turns into a failed `ToolResult`.
  Before, both could hang forever. The executor reconnects on every call, so the limit
  is per call and includes starting the server and `initialize`: a long-running tool
  that worked before can now fail by default (pass a larger `timeout` or `None`).
  Teardown of a real stdio server under cancellation is untested; on Python 3.10/3.11
  a slow teardown can run past the timeout.
- **Unknown workflow `step_type` is a validation error (MM-01/MM-02).**
  `WorkflowBuilder.validate_complete` reports a step whose `step_type` is not in
  `VALID_STEP_TYPES` as an error (`add_step` still only warns). The meta-builder's
  extraction schema types `step_type` as a free string, so a live model that invents a
  step type now gets `MetaBuilderResult.is_valid == False` where it used to get True
  (visible as `artifact_valid` in `examples/meta/build_workflow`). `is_valid` for
  workflow and agent artifacts means a structurally complete spec, not a directly
  loadable object; the docstrings now say so.
- **Duplicate tool registration warns (FB-07).** `ToolRegistry.register` logs a warning
  when a name is already registered. The last registration still wins.
- **`SkillLoader` loads a skill once (PT-07).** A name already provided through a
  module's `SKILLS` list is no longer registered a second time by the `@tool` scan.
- **HITL logs a late callback error (CR-05).** An approval callback that raises after
  the approval timeout (the request was already denied) is now logged instead of
  dropped. Two small timing changes come with it: a callback that returns between the
  join timeout and the internal lock is now honoured (it used to be denied), and a
  callback that raises a `BaseException` (not an `Exception`) is logged as a timeout
  and denied.

### Removed -- agents audit 2026-09-24

- `MetaBuilderConfig.output_path` (MM-07): declared and documented but never read. A
  caller that still passes it is ignored, as before (pydantic `extra="ignore"`).
- `fsm_llm_agents.prompts.build_think_response_instructions`,
  `build_act_response_instructions`, `build_evalopt_evaluate_response_instructions` and
  `build_checker_response_instructions` (FB-04): no caller anywhere in the repository.
- The `".." in Path(path).parts` check in `meta_output.save_artifact` (MM-03): it ran
  after `.resolve()` and so never fired. `save_artifact` trusts its caller and writes to
  the resolved path; the docstring says so. There was never any path confinement.

### Fixed -- agents audit 2026-09-24

- `test_reasoning_react.py` handler-reset tests no longer make a live LLM call when
  Ollama is reachable (RB-05). New smoke tests for `fsm-llm-meta` and
  `python -m fsm_llm_agents` (MM-09); their docstrings drop options that do not exist
  (CR-04).
- Internal cleanup with identical FSM output: `ContextKeys` constants replace 56 context
  key literals in `fsm_definitions.py` and one `_finalize_fsm` helper applies the
  description truncation rule (FB-02/FB-03); shared persona and meta-tool factories
  (FB-08/MM-08); unreachable branches removed in `native_fc.py` and `hitl.py` (CR-03).

### Known open -- agents audit 2026-09-24

Closed by the agents follow-up above: `AgentServer` auth and size limit, FB-05 typing
(the live check is still pending), CR-06, the three HITL gaps, the maker_checker
unjudged draft, the Reflexion/ParallelReact evidence guard and the `SkillLoader`
alias. Each closed item below is marked; the text is kept as it was.

- F-LIVE-02 live check, raw outcome, not a fix claim: `TestLiveMemoryAgent` on `ollama_chat/qwen3.5:9b-q8_0` ran twice on the final
  commit. Run 1 failed with `AgentTimeoutError` (120 s) after the first `remember`
  call (a cold model, with test collection running on the same machine); run 2 passed
  in 85 s. Both agents in run 2 reached `Iteration 5/5` before concluding. 1 of 2 is
  not enough to call the issue closed, and the base commit was not re-run.
- **Closed by the follow-up above (D-006).** `AgentServer` (A2A) has no authentication and no request size limit (MM-05). Fine for
  local use; it needs its own design before it is exposed.
- **Closed by the follow-up above (D-009; live check pending).** FB-05: `all_collected` and `consensus_reached` are read by transitions but not
  declared for extraction; confirming whether this bites needs a live extraction check.
- **Closed by the follow-up above (D-011).** CR-06: a list `tool_input` value is stringified (the known trade-off of the `any`
  union type).
- Core: PRE_TRANSITION handlers do not run on a BLOCKED turn. This fix works around it
  in the agents package; changing it would change handler timing for every FSM.
- **Closed by the follow-up above (D-004, D-005; remaining gaps listed there).** **HITL approval gaps (security, pre-existing, not fixed).** Do not rely on
  approval to guard a dangerous tool yet.
  (a) ReflexionAgent has no `await_approval` state: an approval-required tool runs
  first and the callback is asked afterwards, so a denial does not stop it.
  (b) ReasoningReactAgent builds `await_approval` (a policy plus a
  `requires_approval` tool) but has no approval driver: the callback is never asked,
  and the model opens the gate by extracting `approval_granted` (the state's
  extraction prompt asks for it). If the model never extracts it, the state BLOCKS
  until the 3x ceiling.
  (c) `approval_granted` is an ordinary context key the model can extract, so a bulk
  extraction in any state (for example `think`) can approve a call itself, in
  ReactAgent too; a callback that always denies is then never asked.
  Fix direction, for a follow-up plan: `execute_tool` refuses to run a real tool while
  `approval_required` is set and the grant did not come from the driver, and the
  grant moves to a driver-only internal key (for example `_approval_granted`).
- **Closed by the follow-up above (D-007).** maker_checker can ship an unjudged draft when the iteration limiter ends the run
  (D-017/D-018). If the limiter forces `checker_passed=True` on a `revise` turn, the
  draft written on that turn skips `check` and ships without a verdict; the run
  reports `max_iterations_reached=True`. It depends on the budget: with an always-False
  checker, odd `max_iterations` values ship an unjudged draft and even ones ship the
  judged draft. Fix direction, for a follow-up plan: force the pass only while the
  current state is `check`, and add a RED test at `max_iterations=3` asserting the
  answer is the first (judged) draft.
- **Closed by the follow-up above (D-008), except ReWOO.** Reflexion's and ParallelReact's `think->conclude` have no evidence guard (D-004), so
  they can conclude without a tool call (pre-existing). ReWOO has no stall guard (still open).
- **Closed by the follow-up above (D-012).** `SkillLoader`'s `@tool` scan dedupes only against `SKILLS` names: a `@tool` function
  reachable under two attribute names (an alias or a re-import) still loads twice.
- `scripts/eval.py` was not re-run; the baseline in `CLAUDE.md` is staler after the
  routing change above. The follow-up above re-runs it on the new default model.

### Core audit 2026-09-22

Core audit of `src/fsm_llm` dated 2026-09-22 (`plans/plan-2026-09-22T080837-8b258a25`,
16 steps in 26 commits, one commit per step or substep, the last five being fixes from
the iteration review). Every behaviour change has a
test in `tests/test_fsm_llm/test_audit_2026_09_22.py` that fails on the parent commit;
four cross-cutting sweeps live in `tests/test_fsm_llm/test_audit_sweeps.py`. Full suite:
6,862 tests collected (was 6,564). `ruff` and `mypy` clean across all 6 packages. No
prompt text changed, so the stale eval baseline was not re-measured.

### Behaviour changes to know about -- core audit 2026-09-22

- **Load-time validation of `prompt_config` and `transition_classification`.** A
  `classification_extractions[].prompt_config` with an unknown key or an out-of-range
  value (`max_tokens < 1`, `temperature` outside 0.0-2.0, `max_intents` outside
  1-`MAX_MULTI_INTENTS`) now fails when the FSM loads, with the same rules
  `ClassificationPromptConfig` applies at extraction time (it used to fail, softly,
  mid-conversation). `State.transition_classification` must be None or a dict: the
  reserved `confidence_threshold` key must be a non-bool number in [0, 1], and every
  other key must map to a dict whose only key is `description` (str or None). `{}` is
  accepted. `fsm-llm-validate` gives the same verdict. All shipped examples still load.
- **Session files hold placeholders for non-JSON objects.** `FileSessionStore.save`
  no longer calls `str()` on an arbitrary value. An exact `datetime`, `date`, `time`,
  `timedelta`, `Decimal` or `UUID` is still written as `str(value)` (byte-identical to
  before); anything else (`set`, `bytes`, `frozenset`, a subclass of those scalars,
  `pathlib.Path` (`"<redacted:PosixPath>"`), an `Enum` member, a numpy scalar such as
  `numpy.int64`, a custom object) is written as `"<redacted:TypeName>"`, so a restore
  reads the placeholder string. `fsm_llm_agents.memory_persistence.save_working_memory`
  (and so `MemorySessionStore`) writes WorkingMemory with the same hook, now public as
  `fsm_llm.session.session_json_default`.
- **`context_snapshot` is filtered.** `get_complete_conversation(...)["metadata"]
  ["classification_results"][field]["context_snapshot"]` drops every secret-shaped
  entry (`is_forbidden_context_entry`) at any depth, including inside lists. The verdict
  is taken on the stored JSON form (tuples as lists, non-str keys as strings), and a
  value nested deeper than `MAX_CONTEXT_FILTER_DEPTH` is dropped. A cyclic value is
  still omitted, as before.
- **`FSMStackFrame.fsm_definition` is typed `FSMDefinition`.** A frame built from an id
  string, or from anything else that does not validate as an `FSMDefinition`, is now a
  validation error.
- **Typed "unknown FSM id" error.** `load_fsm_definition("<id>")` for an id that is not
  a file path raises `FSMDefinitionNotFoundError`, which is both an `FSMError` and a
  `ValueError`; the message keeps the `Unknown FSM ID` prefix and names the missing
  registry.
- **`API(handler_timeout=..., max_fsm_cache_size=...)` are consumed.** Both used to fall
  into `**llm_kwargs` and reach the LLM interface; they now configure `HandlerSystem`
  and `FSMManager`. Defaults (`None`, 64) behave as before. `max_fsm_cache_size` below
  1 raises `ValueError` when the `API` or `FSMManager` is constructed (0 used to make
  every `start_conversation` fail with `dictionary is empty`). The
  `MAX_TIMED_HANDLER_STRAGGLERS` cap is shared by every conversation of one `API`
  (documented, unchanged).
- **Bulk extraction ignores a sanitiser override.** Bulk Pass-1 extraction sanitises the
  user message through `prompts.sanitize_text_for_prompt` (the shared default
  sanitiser), so a custom `data_extraction_prompt_builder` subclass that overrides
  `_sanitize_text_for_prompt` no longer affects that call.
- **Restore into a missing instance raises.** `restore_session` seeds summary, history,
  provenance and working memory in one `FSMManager.seed_restored_conversation` call. If
  the FSM instance is gone it raises a typed `FSMError` (and the half-restored
  conversation is torn down) instead of logging a warning and skipping the history
  replay.
- **A failed FSM load no longer evicts a cache entry.** `get_fsm_definition` evicts only
  when it inserts a loaded definition (same `len >= max_fsm_cache_size` rule). On a
  concurrent miss of the same id the loader may run twice; the first-inserted
  definition is returned to both callers.
- **Getters on ended conversations agree.** `get_data`, `get_current_state`,
  `has_conversation_ended` and `get_conversation_history` re-raise
  `ConversationBusyError`. On any other error they answer from the ended-conversation
  cache only when the conversation is gone (its stack or its FSM instance was torn
  down); on a live conversation the original error re-raises, so a failing state
  lookup is no longer reported as "not ended". A cache miss re-raises too, except that
  `has_conversation_ended` returns False for an id it does not know.
  `get_conversation_history` now returns the cached history after `end_conversation`
  (it used to raise). An id that was never started still raises `ValueError`.
- **`max_history_size=0` keeps a summary.** Exchanges are digested into
  `Conversation.summary` (same 2,000-character cap) before they are cleared.
- **`get_version_info()["architecture"]`** is `"2-pass"` (was `"improved-2-pass"`).

### Fixed -- core audit 2026-09-22

- P0-1: the ended-conversation cache (`API._ended_conversations`) is written, evicted
  and read only under `API._stack_lock`. A reader during a concurrent end no longer
  sees an evicted-but-not-inserted entry, and concurrent evictions no longer fail with
  `KeyError`.
- P0-3: `restore_session` no longer takes `FSMManager._lock` while holding a
  conversation lock (4 sites, C-NEW-007). The seeds run in the canonical order
  (`_lock` for the lookup only, then the conversation lock). `api.py` has no remaining
  access to `fsm_manager._lock`, `._conversation_locks` or `.instances`.
- P0-4: `handler_timeout` and `max_fsm_cache_size` are reachable through `API` (see
  above).
- P1-1: the Pass-1 memo name is bound on every path (defensive; no `NameError` was
  reachable).
- P1-2: `max_history_size=0` summarises before clearing (see above).
- P1-4: `get_conversation_history` falls back to the ended cache (see above).
- P1-5: `get_data` falls back to the ended cache when the conversation is gone and
  propagates `ConversationBusyError` (see above).
- P1-6: `SessionState.stack_depth` is documented as advisory (written by
  `save_session`, never read by restore).
- P1-7: the Pass-2 apology retry is counted (`LiteLLMInterface.apology_retry_count`);
  the retry itself is unchanged.
- Security: secret-shaped entries no longer reach the `context_snapshot` metadata (see
  above), and a value's `__str__` no longer reaches a saved session file or a saved
  WorkingMemory file (see above).
- `runner._redact_context` (the `fsm-llm` CLI's debug context dump) terminates on a
  self-cycle (the cycle becomes `"<redacted:cycle>"`) and runs in linear time on aliased
  acyclic input: a 3-way-aliased tower 16 levels deep went from more than 25 s to
  under 1 ms. Input that aliases every level and also cycles is not memoised and can
  still be slow; the runner only redacts `get_data()` output, which is always acyclic.
  Output on acyclic input is unchanged.
- `DEFAULT_TEMPERATURE` is the single source of the 0.5 default in `API.__init__` and
  `LiteLLMInterface.__init__` (it was defined but unused next to two literals).
- Stale comments in `fsm.py` said `get_fsm_definition` re-takes `_lock`; they now give
  the real reason (the loader runs on a cache miss).

### Added -- core audit 2026-09-22

- `src/fsm_llm/security.py`: the credential/forbidden-context filter and
  `has_internal_prefix` moved there verbatim (`constants.py` 1,978 -> 299 lines).
  `fsm_llm.constants` re-exports every public name and every private name used in
  `src/` or `tests/`; no filter verdict changed.
- `ResponseGenerationRequest.skip_generation` (default False), set by the greeting and
  synchronous Pass-2 skip sites and honoured by `LiteLLMInterface.generate_response`
  and its streaming path. The streaming skip site still makes no interface call.
- `FSMDefinitionNotFoundError` (exported, pickles with `fsm_id` and details).
- `FSMManager.seed_restored_conversation`, `FSMManager.has_instance`,
  `FSMManager.copy_raw_context_data`, `FSMManager.prune_orphaned_locks`.
- `ClassificationResult.is_below_default_threshold` (property).
- `WorkingMemory.hidden_buffers` (read-only property, returns the instance's own
  `frozenset`).
- `fsm_llm.logging.reset_handlers()` and `fsm_llm.logging.register_stream_handler()`;
  `enable_debug_logging` uses them and touches no logging private.
- `fsm_llm.prompts.sanitize_text_for_prompt(text)`, the public entry to the shared
  prompt sanitiser (output identical).
- Constants: `DEFAULT_MAX_FSM_CACHE_SIZE` (64, shared by `API` and `FSMManager`),
  `CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT`, `PROVENANCE_METADATA_KEY` (`pipeline._PROVENANCE_KEY`
  stays as an alias), `TRANSITION_CLASSIFICATION_THRESHOLD_KEY`.
- `LiteLLMInterface.apology_retry_count`.
- `API.__init__` parameters `handler_timeout` and `max_fsm_cache_size`.
- Tests: `tests/test_fsm_llm/test_audit_2026_09_22.py` and
  `tests/test_fsm_llm/test_audit_sweeps.py` (JsonLogic operator partition and dispatch,
  agreement of the four context filters, reachability of every `HandlerSystem` and
  `FSMManager` option from `API`, an exhaustive small-alphabet prompt-sanitiser sweep
  with 255/256/257/300 padding cases). The sweeps found no defect.
- Docs: `CONTRIBUTING.md` (setup, commands, count pins, frozen examples, the `plans/`
  convention, DECISION anchor format, the 6-line rule, `[STALE]` versus
  `[SUPERSEDED BY D-nnn]`); a README "Behaviour details" section (LLM calls per turn,
  `handler_timeout` and the straggler cap, what the extractor can write, JsonLogic
  differences, `extraction_retries` on Ollama).
- `expressions.py` checks at import time that no short-circuit operator (`and`, `or`,
  `if`) re-enters the eager `operations` table.

### Deprecated -- core audit 2026-09-22 (removal in 1.0)

- `FSMManager.cleanup_stale_conversations`: use `prune_orphaned_locks` (warns).
  `API.cleanup_stale_conversations` is a different method and is not deprecated.
- `ClassificationResult.is_low_confidence`: use `is_below_default_threshold` (warns).
  `Classifier.is_low_confidence` is not deprecated.
- `TransitionEvaluatorConfig.ambiguity_threshold`, `minimum_confidence` and
  `evidence_conditions_normalizer`: no effect since priority-only ranking; setting any
  of them to a non-default value emits `DeprecationWarning`. Defaults stay silent.
- The `system_prompt="."` Pass-2 skip sentinel: still sent and still honoured by
  `LiteLLMInterface`; custom interfaces should read `skip_generation` instead (no
  runtime warning).

### Removed -- core audit 2026-09-22 (breaking)

- `fsm_llm.DomainSchema`, `fsm_llm.LLMRequestType` and
  `fsm_llm.validate_json_structure` (and their `__all__` entries).
- Constants: `LOG_FIELD_TIMESTAMP`, `LOG_FIELD_LEVEL`, `LOG_FIELD_MESSAGE`,
  `LOG_FIELD_MODULE`, `LOG_FIELD_FUNCTION`, `LOG_FIELD_LINE`,
  `LOG_FIELD_CONVERSATION_ID`, `LOG_FIELD_PACKAGE`, `LOG_MESSAGE_PREVIEW_LENGTH`,
  `LOG_RESPONSE_PREVIEW_LENGTH`, `DEFAULT_STEP_TIMEOUT`, `DEFAULT_HANDLER_TIMEOUT`,
  `MIN_BASE_CONFIDENCE`, `PRIORITY_SCALING_DIVISOR`, `CONDITION_SUCCESS_RATE_BOOST`.
- `fsm_llm.constants` no longer exposes the credential filter's unreferenced private
  helpers (import them from `fsm_llm.security`): `_ACRONYM_BOUNDARY`,
  `_AMBIGUOUS_CREDENTIAL_ABBREVIATIONS`, `_CAMEL_BOUNDARY`,
  `_CREDENTIAL_MATERIAL_HEADS_SHARED`, `_CREDENTIAL_NAME_ANY_RE`,
  `_CREDENTIAL_NAME_HEAD`, `_CREDENTIAL_NAME_TERMS`, `_CRYPTO_AT_WORD`,
  `_CRYPTO_GAP_REACH`, `_ISO_DATE_RE`, `_KEY_TRIGGER`, `_MIN_CREDENTIAL_VALUE_ENTROPY`,
  `_NAME_TOKEN_SPLIT_RE`, `_PATH_SEGMENT_TRIGGER`, `_PEM_PREFIX`,
  `_POLICY_TAIL_COUNT_END_RE`, `_POLICY_TAIL_DURATION_END_RE`, `_POLICY_TAIL_MAX_COUNT`,
  `_POLICY_TAIL_RE`, `_POLICY_TAIL_STATE_WORDS`, `_POLICY_TAIL_WORD_SPLIT`,
  `_POLICY_TAIL_WORD_VALUE_MAX_CHARS`, `_PURE_HEX_RE`, `_TOKEN_ONLY_MATERIAL_HEADS`,
  `_TOKEN_TRIGGER`, `_TRIGGER_PLURAL`, `_ULID_VALUE_RE`, `_UUID_VALUE_RE`,
  `_VALUE_SCAN_NAME_RE`, `_WORD_END`, `_is_path_shaped`,
  `_policy_tail_scalar_is_credential`, `_policy_tail_value_is_credential`; nor the
  stdlib names `re`, `math`, `unquote` and `Iterable`.
- `TransitionEvaluation.confidence`, the evaluator's `"confidence"` score key and
  `"confidence_factor"` condition-result key, and the confidence computation behind
  them (an old `confidence=` kwarg is ignored).
- `ResponseGenerationRequest.context`, `.extracted_data` and `.previous_state`, and
  their per-turn computation (a caller still passing them is ignored, `extra="ignore"`).
- `DataExtractionResponse.additional_info_needed` (and the required-names scan that
  fed it).
- `ContextCompactor(summarize_on_trim=...)` and the attribute.
- `BasePromptBuilder._estimate_token_count(is_json=...)` and
  `_build_response_format(field_heading=...)` parameters (the heading is always
  `"Where:"`).
- `expressions.operations["and"]`, `["or"]`, `["if"]` (unreachable eager fallbacks) and
  `expressions.if_condition`. `_SHORT_CIRCUIT_OPERATORS` is now a `frozenset`.
- `ollama.TRANSITION_JSON_SCHEMA` and the `"transition_decision"` entry of
  `_CALL_TYPE_SCHEMAS`; `build_ollama_response_format("transition_decision")` returns
  the unknown-call-type fallback, `None`.
- Visualizer: `ICONS["note"]`, `ARROW_STYLES["down_arrow"]`, `["right_arrow"]`,
  `["diamond"]`, `BOX_STYLES["section"]`.
- `API._replay_history` (private; folded into `seed_restored_conversation`).

### Performance -- core audit 2026-09-22

- `LiteLLMInterface` calls `litellm.get_supported_openai_params` once per instance and
  model string (a raised lookup is not memoised).
- The FSM definition cache has its own leaf lock, so turn-path definition lookups no
  longer take `FSMManager._lock`; the loader runs outside the lock.
- `utilities.filter_context_tree` skips the cycle pre-scan for a plain `dict` root whose
  values are all leaves (output identical).
- Runner redaction is linear on aliased acyclic input, and its debug context dumps are lazy
  (no redaction when debug logging is off).
- The removed `ResponseGenerationRequest` fields are no longer computed on every turn.

### Fixed -- follow-up 2026-09-22

- `json.dumps(default=str)` is gone from every writer that emits context or trace values
  out of the process, closing the same `__str__`-leak class the session fix closed on
  disk. An object whose text carries a secret is now written as
  `"<redacted:TypeName>"`, and its `__str__` is never called:
  - `fsm_llm_monitor`'s dashboard websocket push, which broadcast to every connected
    browser;
  - the two `fsm_llm_reasoning` engine sites that render context into an LLM prompt;
  - the `fsm_llm_reasoning` CLI's JSON output and its saved results file.
- The hook itself moved to `fsm_llm.utilities.redacting_json_default`, since it now
  serves disk, prompt and socket writers. `fsm_llm.session.session_json_default` is the
  same object under its on-disk name, so existing imports keep working. An exact
  `datetime`/`date`/`time`/`timedelta`/`Decimal`/`UUID` is still written as its `str()`.

### Not changed (considered and declined) -- core audit 2026-09-22

- P0-2, Ollama null memo across retries: kept. Structured Ollama calls run at
  temperature 0, so a retry of the same prompt is pure waste; documented in the README
  instead (D-021).
- Unifying the 4 tie-break rules of `extract_json_from_text`: all four or none, no
  failing case, and it radiates to every package (D-022).
- One shared context walker with a budget parameter: the prompt walker truncates, the
  data walker must never truncate, the runner redacts instead of dropping (D-023).
- Splitting `pipeline.py` into three modules: high churn, no behaviour value (D-024).
- Per-FSM `secret_context_keys`/`public_context_keys`: needs plumbing into all four
  filter seams; its own plan (D-025).
- `FSMDefinition.agent_managed` instead of sniffing `agent_trace`: agent FSMs are built
  at many sites; deferred (D-026).
- Splitting `is_forbidden_context_entry` by arity: every production caller passes the
  value, and one decision point stays one name (D-027).
- `setup_logging`'s `-1` sentinel: test-pinned intended behaviour (D-028).
- `handle_conversation_errors` union parameter: local convention, no defect (D-028).
- Renaming `HandlerBuilder.critical`: not a real name collision (D-028).
- `_CLASSIFICATION_CONTEXT_BUILDER` visibility: module-private, no reach-in (D-028).
- Transition-snapshot reuse: would touch the rollback contracts (D-028).
- A classifier digest cache: minor gain (D-028).
- Session `strip_forbidden_keys` opt-in: needs a walker on the save path; the `str()`
  leak is fixed instead (D-028).
- Relocating the 281 existing DECISION anchor bodies: mass churn; the convention now
  applies to new anchors (D-028).
- Per-conversation straggler keying: would thread `conversation_id` into
  `execute_handlers`; the shared cap is documented instead (D-028).
- P1-8, one unit for `max_history_messages` (fetch `ceil(n / 2)` exchanges): changes
  the rendered history under the `TOKEN_BUDGET` strategy, so not output-identical; the
  fetch still passes an exchange count (D-038).
- P1-3, a read-only `should_execute` probe: landed in step 4 as a
  `types.MappingProxyType` view, then reverted in the same release. It broke conditions
  that only read the context but serialise, copy or type-check it (`json.dumps`,
  `copy.deepcopy`, `pickle`, `isinstance(ctx, dict)`), and under the default
  `error_mode="continue"` such a handler was silently skipped. The probe again gets the
  live context; a condition must not mutate it (the documented pure-predicate contract).
- `json.dumps(default=str)` in the three `fsm_llm_reasoning` size measurements
  (`handlers.py`): the serialized text is only measured with `len()` and never emitted,
  so no `__str__` reaches a prompt, a file or a socket. The emitting writers were fixed
  (see "Fixed", below).
- `context_snapshot` keeps internal-prefixed keys named in `context_keys`
  (pre-existing): whether `context_keys` may name internal keys needs its own decision
  (D-039).

### Core audit 2026-09-21

Core audit of `src/fsm_llm` dated 2026-09-21 (`plans/plan-2026-09-21T203800-8a03483a`,
15 fix steps, one commit per step). 45 audit ids: 42 fixed, 2 partly fixed (D9, D12),
1 skipped (D7). Every fixed id is pinned by a `test_<id>_*` regression test in
`tests/test_fsm_llm/test_audit_2026_09_21.py` that fails on the pre-fix code. Full
suite: 6,564 tests collected (was 6,177). `ruff` and `mypy` clean across all 6 packages.

### Behaviour changes to know about

- **A2: transition priority now decides.** When several transitions pass, the one
  with the unique lowest `priority` wins deterministically, whatever the gap between
  priorities or the number of conditions. Only a tie at the lowest priority is
  AMBIGUOUS and goes to the classifier (tied group only). `minimum_confidence` and
  `ambiguity_threshold` are kept as deprecated no-ops. 10 shipped example FSMs now
  resolve some formerly ambiguous turns deterministically (87 state/passing-set
  combinations). `examples/basic/simple_greeting` has two unconditioned transitions
  (farewell p0, conversation p1) and now goes to `farewell` on every turn. Authors who
  want the classifier to choose between intent-routed transitions must give them
  equal priority. The eval baseline in CLAUDE.md is stale until re-run.
- **B1-B4: one None/missing rule for JsonLogic.** Ordering (`<`, `<=`, `>`, `>=`) is
  False when any operand is `None`. Arithmetic (`+ - * / % min max`) on a `None`
  operand yields an internal "undefined" value that no comparison can satisfy:
  `==`, `!=`, `===`, `!==`, ordering, `in` and `contains` are all False on it, and
  `!`, `!!`, `and`, `or`, `if` treat it as false. So
  `{"<": [{"-": [total, discount]}, 100]}` and `{"!=": [{"-": [balance, paid]}, 0]}`
  with the second operand unset are False (the first used to compare `False` as 0
  and pass), and a bare arithmetic condition on an unset operand does not fire
  (`evaluate_logic` returns `None` for it). `-` is unary only with
  exactly one operand. `missing`, `missing_some` and `requires_context_keys`
  treat absent, `None` and `""` as missing, and an extracted `None` no longer
  overwrites a stored value during transition evaluation. `==`/`!=` coerce
  numerically only for a mixed number/string pair whose string is a plain decimal or
  scientific literal in ASCII digits (`1.0 == "1"` is True; `"1_000" == 1000`,
  `" 1 " == 1` and `"١٠٠٠" == 1000` are False); two strings are never coerced (`"01" == "1"` stays False), bools are never
  coerced, and `null == null` stays True.
- **B5, B6, B10: new load-time errors.** A JsonLogic operator object with more than one
  key, or nesting deeper than `MAX_JSONLOGIC_DEPTH`, fails to load (B5).
  `context_scope` is now a `ContextScope` model (`read_keys`/`write_keys` lists,
  unknown keys rejected); a string instead of a list or a misspelled key is an error
  (B6). A `field_name` declared twice in `field_extractions`, or twice in
  `classification_extractions`, is an error, and so is a blank or internal-prefixed
  `required_context_keys` entry (B10). `fsm-llm-validate` reports the same errors.
- **C3: framework-reserved context keys.** Handler deltas can no longer set or delete
  the keys in `constants.RESERVED_CONTEXT_KEYS` (`_conversation_id`,
  `_current_state`, `_fsm_id`, `_previous_state`, and 8 more framework-seeded keys).
  An attempted change logs a WARNING; an equal echo is ignored silently.
  Handler-owned internal keys (for example agents' `_replan_count`) still merge.
- **C8: CLI exit codes and output streams.** `fsm-llm`, `fsm-llm-validate` and
  `fsm-llm-visualize` exit 0 on success, 1 on failure (a turn error used to exit 255)
  and 130 on Ctrl-C (mid-turn Ctrl-C used to print a traceback). The validation
  report and the diagram now go to stdout; diagnostics stay on stderr.
- **C12: `end_conversation` raises on lock timeout.** If the conversation lock cannot
  be taken within `END_CONVERSATION_LOCK_TIMEOUT_SECONDS` (30 s),
  `FSMManager.end_conversation` raises `ConversationBusyError` (a new exported
  `FSMError` subclass) and changes nothing, instead of tearing the conversation down
  while a turn may still be running. Retry after the turn ends. `API.end_conversation` propagates that refusal (it used to log it and
  return) and keeps the conversation active with its stack, idle tracking and data;
  frames above the refusing one that already ended are removed from the stack. Its
  data/state/history cache read is bounded by the same timeout (it used to wait
  forever); any other failure of that read is logged and the end proceeds without
  a cache entry. `cleanup_stale_conversations` and `close()` log a refusal and continue
  with the other conversations.
- **D5: more credential names are stripped from prompts.** `passwd`, `pwd`, `pass`,
  `passcode`, `passphrase`, `pin`, `otp`, `mfa_code`, `cvv`, `ssn`, `credit_card`,
  `card_number`, `cookie`, `jwt`, `bearer`, `authorization`, `auth_header` and
  `recovery_code` (whole name segments, optional plural) are now filtered out of
  prompt context. Policy-style tails (`password_min_length`) and `bool` values are
  kept. Names such as `pass_id`, `pin_code` or `cookie_consent` are now stripped too.
  `cvc`, digit-suffixed terms (`cvv2`, `pin2`) and camelCase/acronym forms
  (`pinCode`, `PINCode`) match as well. A policy-suffix name (`pin_attempts`,
  `authorization_status`) keeps only a value that cannot be a credential: None, a
  `bool`, a dict (its keys are then filtered by their own names), a number under a
  count suffix (`pin_attempts: 3`, below 1,000) or a duration suffix
  (`cookie_max_age: 86400`), an ISO date, or a string made only of state words
  (`verified`, `locked`, `enabled`, ...). `pin_enabled: "1234"`, `cvv_status: 737`,
  `pin_status: "hunter"` and `authorization_status: "Bearer ..."` are stripped.
  `ttl`, `timeout` and `limit` join the password policy suffixes.
- **D6: no reasoning fallback.** A Pass-2 reply with an empty or missing `message`
  no longer shows the model's internal `reasoning` to the user; it produces the
  generic apology and the pipeline's one retry.
- **D10: removed dead Pass-1 prompt API.** `DataExtractionPromptBuilder.build_extraction_prompt`
  and `build_refinement_prompt` had no caller and are deleted, together with the five
  `DataExtractionPromptConfig` fields only they read (`include_context_data`,
  `include_state_instructions`, `enable_detailed_guidelines`, `enable_format_rules`,
  `enable_extraction_guidance`). Passing those fields now raises `TypeError`.
- **New export:** `fsm_llm.ContextScope`.

### Fixed

A. Decision-making (classification / memory)

- A1: the transition classifier's `confidence_threshold` is enforced; a result below it stays in the current state (WARNING, record flagged `low_confidence`).
- A2: priority spread alone no longer bypasses the classifier inconsistently; the unique lowest priority wins and only ties are ambiguous (see above).
- A3: `Classifier.classify`/`classify_multi` accept a context (last 3 exchanges with each line capped at 150 characters, state purpose, scoped visible data), rendered sanitized and security-filtered; the pipeline feeds it at both call sites. An extraction's `context_keys` only narrows the state's `context_scope.read_keys`; it can no longer expose a key `read_keys` hides.
- A4: every classification-extraction result is kept in `context.metadata["classification_results"]` and the ambiguous-transition record in `context.metadata["transition_classification"]`, readable via `get_complete_conversation`.
- A5: after a transition, the new state's unset `classification_extractions` run on the same message.
- A6: `Conversation.summary` is rendered as `<conversation_summary>` in prompts and persisted in sessions (`SessionState.conversation_summary`).
- A7: `WorkingMemory` buffers reach the transition evaluator and the Pass-2 prompt context (`FSMContext.get_merged_data`).
- A8: `_transition_classification_result` is cleared at the start of every turn.

B. Rules engine (expressions / evaluator / definitions / validator)

- B1: `-` with a `None` second operand no longer becomes unary negation.
- B2: `<=`/`>=` are False when both operands are unset.
- B3: `==` agrees with `<=`/`>=` for mixed number/string operands.
- B4: `missing` and `requires_context_keys` treat `None`/`""` as missing; an extracted `None` does not mask a stored value.
- B5: multi-key operator objects and over-deep logic are rejected at load time.
- B6: `context_scope` is validated (`ContextScope` model).
- B7: the validator detects trap cycles per strongly connected component, so interlocking cycles with no exit are reported.
- B8: `missing_some` with a non-integer minimum is a clean error; an integer needle `in` a string haystack is matched as text; `%` sign semantics are documented.
- B9: the load-time logic walk no longer recurses into data lists, and its depth is bounded.
- B10: duplicate `field_name` within one extraction list and blank or internal-prefixed `required_context_keys` are load errors.
- B11: dead "unreachable terminal" warning branch removed.
- B12: the validator's structure pass no longer crashes on type-invalid input; a non-validation loader exception is an ERROR, not a warning with `is_valid=True`; key references are read from the logic tree, including dotted `var` paths.
- B13: `strict_condition_matching` is documented as diagnostics only.

C. Handlers / sessions / logging / CLI

- C1: `register_handler` is copy-on-write under a lock; concurrent `execute_handlers` never sees an empty handler list.
- C2: a timed handler runs on its own copy of the context in a daemon thread; a timed-out handler's writes never reach later handlers and never block exit. A timed handler that finishes in time has its in-place writes adopted, so later handlers see the same context with or without `handler_timeout`. At most `MAX_TIMED_HANDLER_STRAGGLERS` (4) timed-out threads per `HandlerSystem` may still be running; past that a timed call fails at once as a timeout (WARNING) instead of starting another thread.
- C3: handler deltas cannot overwrite framework-reserved context keys (see above).
- C4: `FileSessionStore` ids use a full match; `load`/`exists`/`delete` return None/False on `OSError`; `list_sessions` returns only loadable ids; `save` fsyncs before replace.
- C5: file logging is marked initialized only after `logger.add` succeeds; the JSON sink no longer stashes the rendered line in the shared record.
- C6: non-dict handler returns log a WARNING; `priority` is validated at registration; condition errors are wrapped once; `HandlerExecutionError` pickles.
- C7: JSON log lines never `str()` unknown values (`<non-serializable: TypeName>`); the session store's lossy coercions are documented.
- C8: consistent CLI exit codes and stdout payloads (see above).
- C9: the visualizer handles `"transitions": null` and multi-line FSM names.
- C10: `get_workflows()` and siblings re-raise inner `ImportError`s; `disable_warnings()` only silences `fsm_llm` warnings.
- C11: `_sub_conversation_summary.fsm_type` is the definition name.
- C12: `end_conversation` refuses to tear down on lock timeout (see above).

D. LLM interface / prompts / context filters

- D1: the Pass-2 embedded-JSON fallback strips `<think>` blocks before scanning.
- D2: context filters redact non-JSON-native leaves as `<redacted:TypeName>` (stdlib date/time, `Decimal` and `UUID` values are kept).
- D3: context filters drop reference cycles. The prompt filter caps work at `MAX_CONTEXT_FILTER_NODES` (100,000), failing closed. `get_data`, `save_session` and the extracted-data commit never truncate: every container that cannot reach a cycle is memoised, even next to an unrelated cycle, and only a cycle that is itself aliased past the work ceiling raises `utilities.ContextFilterWorkError` instead of returning a partial value.
- D4: a flat bulk-extraction reply drops top-level `confidence`/`reasoning` instead of merging them into context.
- D5: 18 more credential names are stripped from prompts (see above).
- D6: internal `reasoning` is never shown as the reply (see above).
- D8: a non-string `reasoning` from the classifier model no longer raises a pydantic error; intent names match case-insensitively when the match is unique.
- D9 (partly): the prompt sanitizer escapes tag names starting with `_`, `!` or `?` (`<_task>`, `<!--`, `<![CDATA[`, `<?xml`) and unterminated closers at the end of text.
- D10: dead Pass-1 builders and their unread config fields removed (see above).
- D11: `{"value": null, "<field_name>": ...}` falls back to the field-name key.
- D12 (partly): `stream` and `response_format` are reserved LLM call kwargs (`constants.RESERVED_LLM_CALL_KWARGS`), ignored with a WARNING in `LiteLLMInterface` and `Classifier`.

### Not fixed (with reasons)

- D7 skipped: the `...key` value-scan names (`monkey`, `primary_key`, `cache_key`) strip only for a bare high-entropy hex value, the same verdict `passkey` gets; bare `token` is a corpus must-strip row and `secret_santa` strips by design. Changing it would loosen the shared secret filter.
- D9, attribute part skipped: escaping `<b onclick>` also escaped benign prose such as `a < b and c > d` that is pinned as safe, and attributes are inert to an LLM. The audit's `<Think>` claim was false (already escaped).
- D12, `/nothink` gating skipped: every Ollama call already sets `reasoning_effort="none"`, the harness depends on the prefix, and the effect is live-model behaviour offline tests cannot measure.
- D12, `<original_input>` truncation skipped: truncating at `max_message_length` (1,000) would cut long agent tasks and reasoning problems, and the same raw message also reaches Pass 1 and the classifier.
- B10, cross-list duplicates kept legal: a `field_name` in both `field_extractions` and `classification_extractions` is a supported fallback pattern (the explicit extractor fills the key when the classifier is below threshold), so only same-list duplicates are rejected.

## [0.8.0] - 2026-09-21

Licence change release: FSM-LLM is now distributed under the Apache License 2.0
(previously GPL-3.0-or-later). Releases up to and including 0.7.0 remain available
under GPL-3.0-or-later. No code or behaviour changes.

### Changed

- Relicensed from GPL-3.0-or-later to Apache-2.0 (`LICENSE`, `pyproject.toml` `license`, `fsm_llm.__license__`, README, CLAUDE.md).

## [0.7.0] - 2026-09-22

Second-layer core-engine audit (`plans/plan-2026-09-20T114608-a8e47b88`, 4 audit-fix
iterations, each with its own regression tests and adversarial review, building on the
0.6.0 audit release below). 9 items re-verified from that release's own "Known
limitations" list still reproduced; 1 genuinely new correctness bug and 8 smaller gaps
were found by an independent fresh sweep of `src/fsm_llm/`. Every one of the 10 items
shipped with a regression test that reproduces the defect pre-fix and closes it
post-fix (or, for one mechanical refactor, an equivalence proof both ways). Adversarial
review ran at the end of every iteration and found real, reproduced problems each time
(2 findings iteration 1, 5 iteration 2, 5 iteration 3, plus one the verifier itself
found in iteration 3) -- every one caught and closed inside that same iteration's
completion-fix, per this repo's own established audit practice. A live-Ollama
regression pass (`ollama_chat/qwen3.5:9b-q8_0`, 27 calls across 2 of the 5
pre-registered scenarios) found no regression in the sanitizer/`rejected_corrections`
code paths this plan does not touch. Full suite: 6,091 tests collected (was 6,035),
zero regressions; `ruff`/`mypy` clean across all 6 packages throughout.

### Fixed

- **`save_session` torn snapshot (D-005, D-009; `f9166f8`, `6c22714`).**
  `FSMManager.get_conversation_snapshot` now reads state, internal-key-stripped data,
  history, the working-memory dict and provenance metadata inside ONE
  `_read_under_lock` hold; `api.py`'s `save_session` calls it instead of composing from
  4 separately-locked reads. A concurrent turn landing in the gap between those reads
  could previously persist pre-transition state paired with post-transition data.
  `get_stack_depth` deliberately stays outside the snapshot (a separate, non-atomic
  call): it is structural and `restore_session` ignores a session file's persisted
  `stack_depth` entirely, so a torn pairing there is never read back.
- **Visualizer truncation-ellipsis sweep completed (D-002, D-007; `ae83494`,
  `cd26435`).** `create_state_boxes`'s STATE DIAGRAM row and `create_states_section`'s
  icon-carrying row -- the sibling call site an earlier decision's own comment named
  but never migrated -- now both route through the existing `_fit()` helper. Two
  states sharing a long common prefix previously rendered byte-identical rows with no
  truncation marker in `--style full` output; they now render distinguishably with a
  middle-ellipsis marker, matching every other bordered row in the module.
- **`restore_session`'s `fsm_id` mismatch is now observable, and `fsm_id` is
  content-derived (D-011, D-016; `44ad994`, `e5a1415`).** `restore_session` logs a
  `logger.warning(...)` (does not hard-fail, since `fsm_id` legitimately drifts across
  additive schema upgrades) when `state.fsm_id != self.fsm_id`. `fsm_id` is now one
  content hash of the parsed FSM definition (`fsm_{name}_{hash(model_dump())}`),
  computed once after the dict / `FSMDefinition` / file construction paths converge --
  replacing three separate per-branch hash inputs, one of which (the file path) carried
  no content hash at all. This closes both the silent-restore-under-a-different-FSM gap
  on `API.from_file`/the CLI path (the documented primary entry point -- exactly the
  gap the 0.6.0 "Known limitations" list below named as `restore_session does not
  compare fsm_id`) and a false-positive-mismatch-noise bug where the identical FSM
  produced different ids depending on how it was constructed or loaded.
- **`runner.py`'s CLI JSON logging survives a non-JSON-native context value (D-014,
  D-015; `0e95801`, `c9f920d`).** Both `json.dumps(...)` call sites now pass
  `default=_json_default`, a callable that emits a `<non-serializable: TypeName>`
  placeholder and logs one WARNING -- never `str(obj)`/`repr(obj)` -- mirroring
  `_redact_mapping`'s existing non-str-key WARNING pattern. A handler storing a
  `datetime` (or any non-JSON-native value) no longer crashes the CLI's debug/dump
  logging path; a secret-bearing object under a benign key no longer leaks its repr
  into persisted logs on the very redaction path meant to protect it.
- **`enable_debug_logging()` no longer duplicates log lines (D-013, D-015; `0eb3be7`,
  `07b58f1`).** It now registers its stderr handler in `logging.py`'s existing
  `_stream_handler_ids` dict, mirroring `setup_logging()`'s own registration, so a
  later `setup_logging(sink="stderr", format="human")` call no longer adds a second
  handler and every subsequent log line no longer prints twice. `src/fsm_llm/CLAUDE.md`'s
  file map corrected (`enable_debug_logging`/`disable_warnings` live in `__init__.py`,
  not `logging.py`). Three scope edges are now documented: the `fsm_llm` logger is
  disabled by default, a `level=` request is silently dropped on the short-circuit
  path, and the dedup covers only the exact `(stderr, human, context=False)` triple.
- **A deleted context key's provenance digest no longer outlives its plaintext (D-018,
  D-024; `8d4c05e`, `8bf5f18`).** `MessagePipeline`'s `merge_delta` -- the one place a
  handler's `None`-delta convention actually deletes a context key -- now also pops the
  matching entry from `context.metadata`'s provenance map, for any handler at any
  non-ERROR timing (not just `ContextCompactor.compact`/`prune`, which never had a path
  to `context.metadata` themselves). A completion-fix closed a gap this same fix
  introduced: `_execute_state_transition`'s POST_TRANSITION rollback now snapshots and
  restores `context.metadata` alongside `context.data`, so a handler-chain failure
  after a provenance-clearing deletion no longer leaves the plaintext restored with its
  digest permanently gone.
- **`create_fancy_header`'s box width now matches every sibling section box (D-019;
  `ab155a6`).** Capped to a fixed 60 columns (was `max(60, len(name) + 10)`), with the
  FSM name routed through `_fit()`. An FSM name over roughly 50 characters previously
  rendered a header box strictly wider than every box below it, with no truncation
  guard of any kind.
- **`execute_handlers` defers its context deep-copy (D-020, D-025; `7dfc60c`,
  `2154af2`).** `HandlerSystem.execute_handlers` now deep-copies `context` only once
  the first handler whose `should_execute()` passes is found, instead of
  unconditionally before the loop -- zero deep copies when no registered handler at a
  timing will run. A completion-fix hoisted the copy back above the per-handler
  execution `try:` (a regression the deferral itself introduced): a non-deep-copyable
  context value now raises a loud, correctly-attributed `TypeError` again in both error
  modes, instead of being silently swallowed in the default `error_mode="continue"` or
  misattributed to a handler that never actually ran in `error_mode="raise"`.
- **`WorkingMemory.to_dict()`/`from_dict()` round-trip `_hidden_buffers` symmetrically
  (D-021, D-026; `1a1c2f6`, `5eff8f5`, `47c081b`).** `to_dict()` now folds
  `_hidden_buffers` into its own flat return; `from_dict()` reads it from there by
  default, with an explicit kwarg still overriding. This closes a real gap in
  `fsm_llm_agents/memory_persistence.py`'s `save_working_memory`/`load_working_memory`,
  which call neither method with a `hidden_buffers` kwarg, so a custom hidden-buffer
  set previously reset to the default on every file-backed reload. **Fix, not a new
  behavior change to work around:** the reserved name `"_hidden_buffers"` is now
  rejected with `ValueError` at every buffer-creation entry point (`set`,
  `create_buffer`, `update_buffer`/`import_flat_data`, the constructor); previously a
  buffer literally named `_hidden_buffers` -- reachable from LLM-chosen input via
  `fsm_llm_agents/memory_tools.py`'s `remember(buffer=...)` tool -- had its entire
  contents silently destroyed by `to_dict()`'s unconditional overwrite of that key.
- **`llm.py`'s structural-presence `hasattr()` checks converted to `getattr()` (D-003,
  D-022; `80417cd`, `0881317`).** 3 of the 4 flagged sites (the streaming loop's
  `chunk.choices`; `_make_llm_call`'s `response.choices` and `choice.message`)
  converted to `getattr(obj, "x", None)`-style checks, proven behaviorally identical to
  their pre-conversion form for a genuinely missing (not just `None`-valued) attribute.
  The 4th (`choice.message.content`) is deliberately kept as `hasattr()` -- a
  documented exception, not an oversight: it must distinguish "attribute absent" from
  "attribute present with value `None`" (the H3 Ollama reasoning-only-reply recovery
  path needs exactly that distinction, and `getattr(x, "content", None) is not None`
  cannot make it), confirmed empirically by temporarily forcing the conversion and
  watching a real litellm-object-driven pre-existing test fail.

### Known limitations

- **7 items re-confirmed WONTFIX this plan, unchanged from the "Known limitations"
  list under 0.6.0 below** (`decisions.md` D-004 of `plan-2026-09-20T114608-a8e47b88`
  documents the reasoning for each): `handler_only_keys` stays opt-in (LV2-04); the
  Pass-1 half-committed-turn design (CF-08) and the CONTEXT_UPDATE-provenance decline
  (RB-04) still need their own superseding design, not a mechanical audit fix; the
  history-cap recall (LV2-10) and the developer-text newline-flattening (LS-10) are
  prompt-content changes gated on the still-stale `scripts/eval.py` baseline
  (re-running it is out of this plan's scope); RB-08's per-transition extra-call cost
  stays a documented-not-capped trade-off; and `EvaluatorOptimizerAgent.success` after
  the refinement cap is out of this package's scope (`fsm_llm_agents`).
- **The `scripts/eval.py` 95.3% baseline (N=3 median, Run 006) is now stale against
  five audit iterations of prompt-adjacent and core-engine changes**, not just the four
  the 0.6.0 entry below named -- this plan's own `context.py`/`pipeline.py` provenance
  fix touches the same handler-execution path a prompt-content change would. Still not
  re-measured; still a release gate before the baseline can be trusted.

Third-layer audit of `classification.py` / `memory.py` (`plans/plan-2026-09-20T165703-0d9c218e`,
iteration 1, "classification.py / memory.py audit, loop 1 of 5"). Two deep audits (16 memory
findings, 9 classification findings) drove 8 code steps, one documentation-truth step and one
NEW gated live suite; every code item ships with a regression test in `tests/test_fsm_llm/`.
Finding ids below are the audit's own (`memory #N`, `classification #N`). Full suite: 6,158
tests collected (was 6,091); `ruff`/`mypy src/fsm_llm/` clean after every code step.

### Fixed -- `classification.py` / `memory.py` audit loop 1 (plan-2026-09-20-0d9c218e, iteration 1)

- **`fsm_llm.memory.__doc__` was `None` (memory #1; `6122510`).** The module docstring sat
  BELOW `from __future__ import annotations`, so Python discarded it. Moved above the import;
  regression test asserts a non-empty `str`. The executor's sweep then found the SAME defect in
  81 modules repo-wide (D-003: `fsm_llm`, `fsm_llm_workflows`, `fsm_llm_agents`,
  `fsm_llm_monitor`); all swapped in `4aadd01`, with a filesystem-derived
  `tests/test_packaging.py` test that fails on any future module hiding its docstring behind
  the future import.
- **`WorkingMemory.from_dict` accepted malformed session data silently or with a raw stdlib
  error (memory #10, #11; `0a02716`).** A non-dict buffer body (`None`, a string, a list) now
  raises `ValueError` naming the buffer (`buffer 'core' must be a dict, got NoneType`); a
  `_hidden_buffers` value that is not a `list`/`tuple`/`set`/`frozenset` of `str` (a bare
  `str` was previously iterated character by character) raises `ValueError` naming
  `_hidden_buffers`. `API.restore_session` on a hand-corrupted session file raises with the
  buffer named and leaves no half-registered conversation.
- **`WorkingMemory.__init__` edges (memory #8, #9, #12, #13; `a311bb7`).** `buffers=[]` /
  `buffers=()` now means NO buffers (was: silently replaced by the four defaults through a
  falsy check); `initial_data` passed without a `core` buffer logs a WARNING naming the
  dropped key count instead of vanishing; `get_all_data`'s "last non-core buffer wins" shadow
  order is documented and pinned by a 3-buffer collision test; `to_dict` is documented as a
  one-level-shallow copy (nested containers are shared with the caller) at the source, not
  one file away. **Behavior change**: `WorkingMemory(buffers=[])` and
  `WorkingMemory.from_dict({})` now mean ZERO buffers (previously both silently produced
  the four default buffers); pass `buffers=None` (the default) to get the defaults.
- **`IntentRouter.validate()` dead branch and duplicated dispatch (classification #1, #7,
  #8; `345c28c`).** The unreachable "append fallback" branch is deleted with a one-line
  invariant comment (`ClassificationSchema.validate_schema` guarantees `fallback_intent in
  intent_names`); `route` and `route_multi` share one private `_resolve_handler(intent)` so
  both raise identically when handler and fallback are absent; `Classifier._kwargs` is typed
  `dict[str, Any]`.
- **`ClassificationPromptConfig` had no bounds (classification #3, #9; `ed40bea`).**
  `__post_init__` now enforces `1 <= max_intents <= MAX_MULTI_INTENTS` (new `constants.py`
  value, 5), `max_tokens >= 1` and `0 <= temperature <= 2`, each `ValueError` naming the
  field; a `prompt_config={"max_intents": 0}` on a `classification_extractions` entry is
  caught by the extraction site's existing narrow tuple, logged, and leaves the key unset.
  The field doc states that Ollama models force `temperature` to 0 under structured output
  (`apply_ollama_params`), and `ClassificationExtractionConfig.prompt_config` lists all six
  accepted keys. **Behavior change**: the bounds are enforced at construction, so
  `ClassificationPromptConfig(max_intents=6)` (previously accepted, any value above
  `MAX_MULTI_INTENTS`), `max_tokens=0` or `temperature=3.0` now raise `ValueError` instead
  of being carried into the prompt.
- **`_resolve_ambiguous_transition` caught `Exception` (classification #2; `d4acaac`).** The
  ambiguous-transition classifier site now catches the same D-004 soft-fail tuple the
  extraction site already used, hoisted to one module-level
  `_CLASSIFICATION_SOFT_FAIL_EXCEPTIONS` in `pipeline.py` (`ClassificationError`,
  `ValueError`, `TypeError`, `KeyError`, `RuntimeError`, `OSError`). Those still degrade to
  "stay in state" with `_classification_result.fallback = True`; programming errors
  (`AttributeError`, `ZeroDivisionError`) and `KeyboardInterrupt` now propagate through
  `API.converse` instead of being swallowed as a stay. **Behavior change**: a
  non-soft-fail exception (`AttributeError`, `ZeroDivisionError`, `IndexError`, ...)
  raised by `Classifier.classify` at the ambiguous-transition site now escapes
  `API.converse` as `FSMError` (`converse` wraps every non-`FSMError` exception);
  previously the turn completed as a "stay" and the caller never saw it.

### Changed

- **Classifier instances are cached per pipeline (classification #5; `7d2e81c`).** Both
  pipeline classification sites go through `MessagePipeline._get_classifier`, keyed by a
  content hash of schema + model + prompt config + connection kwargs (NOT
  `(state_id, field_name)`), bounded at `MAX_CLASSIFIER_CACHE_SIZE` (new `constants.py`
  value, 64) with oldest-entry eviction. One `Classifier` (pre-built prompts and schema, no
  per-call `self` writes) now serves every turn and conversation with the same key; a
  different schema still constructs a new instance. `patch("fsm_llm.pipeline.Classifier")`
  keeps working because construction goes through the module-level symbol.

### Added

- **Buffer-name exports (memory #15; `4aadd01`).** `BUFFER_CORE`, `BUFFER_SCRATCH`,
  `BUFFER_ENVIRONMENT`, `BUFFER_REASONING`, `DEFAULT_BUFFERS` and `DEFAULT_HIDDEN_BUFFERS`
  are importable from `fsm_llm` and listed in `__all__`.
- **`tests/test_fsm_llm/test_live_classification_memory.py` (`c462594`).** A 7-test live
  suite gated exactly like `tests/test_integration_ollama.py` (`integration` + `real_llm` +
  `slow` markers, `conftest.ollama_available()` skip), retaining raw litellm
  request/response pairs and printing them on failure: single- and multi-intent
  classification, `<intent>` injection resistance, entity extraction, a two-stage
  `HierarchicalClassifier`, a classified FSM transition with a working-memory session
  round trip, and a `ReactAgent` + `create_memory_tools` round trip. Iteration-1 result on `ollama_chat/qwen3.5:9b-q8_0` (run together with
  `test_integration_ollama.py`): **18 passed / 1 failed in 122.72s**. The one FAIL is
  F-LIVE-01 below. The 5 example runs at 9b all exited 0; `examples/classification/multi_intent`
  scoring 2/4 is an example-design artefact, not a classifier miss (OBS-LIVE-01:
  `primary_intent` is a `classification_extractions` field on the `listen` state only, so
  turns arriving in `handle_purchase`/`handle_question` keep the stale value; both turns that
  reached the classifier were classified correctly). Examples untouched.

### Docs

- **WorkingMemory's real prompt reach (memory #2-#4, #7, #14; `4aadd01`).** The
  `WorkingMemory` class docstring, `FSMContext` in `definitions.py`, `docs/architecture.md`
  and `docs/fsm_design.md` now state that working-memory data reaches ONLY the Pass-1
  per-field extraction prompt (through the default `context_keys`) and NOT the Pass-2
  response prompt, which is built from raw `context.data`; `to_scoped_view` drops its
  "scoped view for LLM prompts" claim (method kept); the `exclude=True` consequence for
  `model_dump()` is noted at the field.
- **`docs/api_reference.md` `is_low_confidence` example (classification #4; `4aadd01`).**
  The example now calls `classifier.is_low_confidence(result)` (the property on the result is
  a fixed 0.6 threshold, not the classifier's configured one).

### Known limitations -- still open for later loops of this plan

- **WorkingMemory is not wired into Pass 2 (memory #2, #5, #6).** Documented truthfully this
  loop, not changed: the response prompt never sees buffer contents, and the
  scoped-view/reasoning-buffer machinery has no consumer in the pipeline. Wiring is a
  design decision for a later loop, not a mechanical fix.
- **The classifier receives no conversation history or FSM context (classification #6).**
  `Classifier._call_llm` builds exactly two messages (static system prompt + raw user
  message); `ClassificationExtractionConfig.context_keys` only feeds a post-hoc debug
  snapshot. A bare "yes" reply whose intent is decidable only from prior turns cannot be
  classified correctly. A design gap, deferred: it changes prompt content and is gated on
  the still-stale `scripts/eval.py` baseline.
- **`WorkingMemory.search` and `fsm_llm_agents` `MemoryBackend` have different shapes
  (memory #16).** `MemoryBackend.search(query, k) -> list[(text, score, meta)]` vs
  `WorkingMemory.search(query, limit) -> list[(buffer, key, value)]`; passing a
  `WorkingMemory` to `augment_task_with_memories` raises `TypeError` inside that function's
  broad catch and silently degrades to "no recall" with a WARNING. One of three
  incompatible memory shapes in the repo; consolidation is out of this core-package loop's
  scope.
- **F-LIVE-01 -- `ReactAgent` + `create_memory_tools` on 9b never calls `remember`
  (`fsm_llm_agents`, MEDIUM).** `TestLiveMemoryAgent::test_remember_then_recall` fails
  (`remember did not write 'teal' into WorkingMemory: {}`) and the untouched
  `examples/agents/memory_agent` reproduces it (`Tools used: []`, budget exhausted, empty
  memory). Mechanism read from the 14 raw calls: the ReAct loop's per-field extraction
  (`tool_name`, `tool_input`) driven by the synthetic `Continue.` message returns `null` on
  every iteration ("contains no new information"), then `"none"` once the Pass-2 prose
  "I've noted that your favorite color is teal." has leaked into the history. A model-side
  under-call on the ReAct extraction path, the same family as the fixed 4b under-call, now
  on 9b with a different prompt; not a `classification.py`/`memory.py` defect (neither was
  invoked). The test is kept strict; the agents package is out of this plan's scope.
  (Superseded by loop 2 below: the root cause turned out to be in core `pipeline.py`, and
  is fixed.)

### Fixed -- `classification.py` / `memory.py` audit loop 2 (plan-2026-09-20-0d9c218e, iteration 2, final loop)

Iteration 2 re-audited iteration 1's own diffs (adversarial review W1-W7, N8-N12) and
root-caused F-LIVE-01 from the retained raw calls. Every code item ships with a regression
test in `tests/test_fsm_llm/`. Full suite: 6,174 tests collected (was 6,158);
`ruff`/`mypy src/fsm_llm/` clean after every code step.

- **F-LIVE-01 root cause was in core, not the agent: the greeting Pass-2 call ignored an
  empty `response_instructions` (D-006; `ead5535`).** `MessagePipeline.process_message`
  skips Pass 2 when the state's `response_instructions` is empty, but
  `generate_initial_response` (the `start_conversation` greeting) did not, so every ReAct
  agent run opened with a full prose greeting built from the `think` state's purpose; that
  prose then sat in the history and poisoned the `tool_name` extraction on the first real
  turn. `generate_initial_response` now mirrors the sync site exactly: on an empty
  `response_instructions` it makes the `"."` sentinel request (no litellm call in
  `LiteLLMInterface`), records the synthetic `[<state_id>]` marker as the first assistant
  entry and returns it. RED test written against the old code first; a
  `response_instructions=None` control still builds the full prompt.
- **W2: classifier-cache race and unguarded construction (`4047ccd`).** The check /
  construct / evict / insert sequence in `_get_classifier` is one critical section under a
  new `_classifier_cache_lock` (review reproduced `RuntimeError: dictionary changed size
  during iteration` on the unlocked code); the `_get_classifier` call at the
  ambiguous-transition site moved INSIDE the soft-fail `try`, so a classifier construction
  failure (`ValueError` from a bad `prompt_config`) degrades to "stay" at both sites
  instead of escaping only at one.
- **W1 / N11: cache key digested non-JSON-native connection kwargs via `default=str`
  (`40204c8`).** A `SecretStr` (whose `str()` is a fixed mask) or any object whose `str()`
  does not identify its value made two different credentials hash to the SAME key, so a
  cached instance built for one could be served for the other. `_get_classifier` now BYPASSES the cache for a
  non-JSON-native connection kwarg (fresh construct, no insert, no lock); `default=str` is
  gone. Documented at the D-001 anchor: cached `Classifier` instances retain `api_key` and
  other connection credentials for the pipeline's lifetime, up to
  `MAX_CLASSIFIER_CACHE_SIZE` copies, by design.
- **W4: cache-key components pinned (`1d0ad06`).** Tests assert that a changed
  `prompt_config` and a changed connection kwarg each produce a distinct cache entry, and
  that the same inputs hit the same instance.
- **N10: `Classifier._call_llm` malformed-shape wrap (`118aff1`).** The post-call parsing
  (`.choices[0].message.content` chase and `_extract_response`) is wrapped in
  `except (AttributeError, IndexError, TypeError)` re-raised as
  `ClassificationResponseError("Malformed LLM response shape: ...")`, so a choice without
  `.message` now lands in the existing `ClassificationError` family instead of leaking a
  raw `AttributeError`; the existing `"Empty response from LLM"` raise is untouched.
  Caveat: defensive, not reachable via the installed litellm's real response objects
  (those always carry `.message`); the RED test uses a `SimpleNamespace` shape.
- **N8 / W7: `restore_session` collapsed an explicit empty buffers map into the defaults
  (`3b9deac`).** `API.restore_session` now maps a missing / `None` `working_memory.buffers`
  to the default buffers and, when no `hidden_buffers` key is present, the default
  hidden set (`{"metadata"}`; an explicit list, `[]` included, is honoured as-is --
  loop-2 completion fix `iter-2/step-6.1`), and an explicit `{}` to
  `WorkingMemory.from_dict({})` (zero buffers), matching loop 1's `from_dict({})` contract;
  a `# DECISION plan-2026-09-20T165703-0d9c218e/D-001` anchor in `memory.py` pins
  `buffers if buffers is not None else DEFAULT_BUFFERS` against the tempting
  `buffers or DEFAULT_BUFFERS` simplification.
- **W3: prompt-reach docs corrected (`cc71820`).** `FSMContext` (class docstring and the
  `working_memory` Field) and `docs/architecture.md` now say where the buffer data goes:
  `get_user_visible_data()` feeds the Pass-1 per-field prompt AND the public
  `ResponseGenerationRequest.context` at the three Pass-2 sites when a response is
  generated; the shipped `LiteLLMInterface` never reads `request.context`, so no buffer
  data reaches the Pass-2 PROMPT, but a custom `LLMInterface` that reads it will see it.
  The earlier "only Pass-1" wording was wrong about the request object.
- **W5: live suite hardened (`8b33388`).** The `<intent>` injection test asserts the
  injected message still classifies as `check_balance` (it previously only asserted "no
  crash"); the raw-call recorder writes `rec.dump()` for EVERY test (pass or fail) to
  `$FSM_LLM_LIVE_RAW_DIR/<nodeid>.txt`, so passing tests can be audited from their raw
  calls; the memory-agent test mirrors the example's `temperature=0.7` /
  `max_iterations=5`.

Live evidence (iteration 2, `ollama_chat/qwen3.5:9b-q8_0`, n=1, no commit): the live suite
plus `test_integration_ollama.py` ran **18 passed / 1 failed in 246.83s**. The F-LIVE-01
greeting mechanism is GONE: `remember` fires at iteration 1 with the correct
`{key, value, buffer}` payload and the first `agent.run` reaches `conclude`. The remaining
FAIL is a NEW known limitation, **F-LIVE-02** (`fsm_llm_agents`, not fixed, out of this
plan's scope): after a tool result the ReAct loop can sit in `think` with every transition
BLOCKED until the budget is exhausted. Three framework-side links: (a) a `null`
`tool_input` falls back to the whole task string, so `recall` searches for
`"Use the recall tool: what is the user's favourite colour?"` and `WorkingMemory.search`
(substring match) finds nothing; (b) the per-field extraction prompt's closing IMPORTANT
block ("extract from the current message") overrides its own NOTE about the `Continue.`
signal, so `tool_name` comes back `null` on every continuation turn while
`should_terminate=false` stays pinned in `Already extracted:` and is never re-asked; (c)
the 3-consecutive-no-tool stall detector (`AgentToolExecutor`, POST_TRANSITION) and the
iteration limiter (PRE_TRANSITION) never run while transitions are blocked, so the net
written for exactly this case is unreachable and the run ends only on the outer
`BudgetExhaustedError`. The same mechanism drives `examples/agents/memory_agent` (1/5;
task 1 additionally sends `remember` with the whole fact stuffed into `key` and no
`value`, rejected by the tool; the example's `memory_populated` check counts the four
default buffers, so it is `True` on an empty memory). Two further raw-read observations,
recorded not fixed: the multi-intent ranking follows the order intents are mentioned in
the message rather than any salience, and a terminal reply may claim a side effect
("I've saved that") that no tool performed. `examples/classification/multi_intent` 2/4 is
unchanged from iteration 1 (OBS-LIVE-01, example design). Examples untouched.

These changes are not yet released; no version bump is cut by this plan (D-004).

## [0.6.0] - 2026-09-20

Core-engine audit release (4 audit-fix loops over `src/fsm_llm`, each verified live on
`ollama_chat/qwen3.5:9b-q8_0` and adversarially reviewed). **Read before upgrading:**
several behaviours changed on purpose (see "Changed -- public contract" below) --
notably the ERROR-handler and re-entrancy contract, provenance-gated bulk corrections,
`extract_json_from_text` returning `dict | None`, the `fsm_id` hash of
`API.from_definition(FSMDefinition(...))` (now carries `handler_only_keys: []`), extra
LLM calls on back-edge revisits and on an Ollama null-extraction memo hit/miss, and a
wider prompt sanitizer. The `scripts/eval.py` 95.3% health baseline was **not re-measured**
against these prompt-content changes and is stale. See "Known limitations" for what is
not fixed.

### Fixed -- core (`fsm_llm`) audit remediation (plan-2026-09-19-21cd7f8e, iteration 1)

Each fix was reproduced RED first and pinned in `tests/test_fsm_llm/test_audit_iter1_seam.py`
(plus updated tests where a superseded contract was encoded). Decision ids refer to
that plan's `decisions.md`.

- **LV-01, typed field-extraction schema (D-001, D-002).** On Ollama models the
  per-field extraction `response_format` now types `value` per `field_type`
  (str `[string,null]`, int/float `[number,string,null]`, bool `[boolean,string,null]`,
  list/dict/any analogous) instead of `"value": {}`, which made the model return
  dict-wrapped junk. `_coerce_str` and `_coerce_bool` now raise on `dict`/`list`
  input, so a str/bool field no longer stores the repr of a container as a valid
  value (it fails extraction instead). Supersedes the D-018 "never raise" clause of
  plan-2026-07-18T051819-80b0bd4d for those two coercers. A `confidence == 0.0`
  extraction is no longer stored. **Behavior change**: `field_type="any"` on the
  Ollama path can no longer yield an object; declare `field_type="dict"` for that.
- **LV-04, plain-text streaming prompt (D-003).** `converse_stream` builds the
  Pass-2 prompt with a plain-text response-format section
  (`build_response_prompt(..., plain_text_response=True)`), so streamed tokens and
  stored history no longer carry the `{"message","reasoning"}` envelope. The sync
  path and terminal states with an output format keep the JSON prompt. **Behavior
  change**: sync and stream Pass-2 prompts now differ.
- **LV-03, bulk correction overwrite (D-004).** A later-turn correction returned by
  the bulk extraction now overwrites an already-set key when the key is covered by
  one of the state's own field configs, was not extracted this turn, is non-null and
  differs, and the FSM is not agent-managed. Instruction-only keys and agent FSMs
  keep skip-if-set. **Behavior change**: on non-agent FSMs a handler-set value for a
  config-covered key can be overwritten by a bulk LLM value.
- **CF-05, ERROR handlers (D-009).** `HandlerTiming.ERROR` handlers now fire on
  `FSMError` (including `LLMResponseError`, the LLM-outage case) and on the streaming
  path, via one `FSMManager._fire_error_handlers`. The `FSMError` is still re-raised
  unwrapped; `KeyboardInterrupt`/`SystemExit`/`GeneratorExit` still run no handlers.
  **Behavior change**: a critical ERROR handler failure now replaces the `FSMError`
  as the raised exception.
- **CF-01, context scope enforced in prompts (D-005).** `context_scope.read_keys` is
  now applied to the context shown in the Pass-2 prompt on the turn, stream and
  greeting paths (`build_response_prompt(..., context=...)`); it previously only
  filtered `request.context`, which the LLM never read. Unscoped states are
  byte-identical.
- **CF-02, classification-owned keys (D-006).** Keys owned by a
  `classification_extractions` entry are no longer also auto-minted as plain
  `requires_context_keys` extraction configs, so a below-threshold classification can
  no longer be bypassed by the plain extractor and one LLM call per turn is saved.
- **CF-04, classifier stay is not a transition (D-007).** A classifier error or the
  fallback intent no longer counts as a transition to the current state
  (`_resolve_ambiguous_transition` returns `None`): no PRE/POST_TRANSITION handlers,
  no post-transition re-extraction. Declared self-loops are unchanged.
- **CF-03, classifier connection (D-008).** The `Classifier` now inherits
  `api_key`, `api_base`, other litellm kwargs and `timeout` from the
  `LiteLLMInterface` (`_classifier_connection_kwargs`), so proxy and self-hosted
  users' classification goes to their endpoint. A per-config `model` on a different
  model does not inherit the key.
- **DH-01 / DH-19, linear-time parsing (D-010).** `extract_json_from_text` scans
  code fences with `str.find` (was a cubic regex: 5000 spaces took about 80 s) and
  `strip_think_and_fences` strips `<think>` blocks in one linear pass (was quadratic
  on unclosed tags).
- **DH-08 / LS-05, dict-or-None JSON (D-010).** `extract_json_from_text` now honours
  its `dict | None` annotation: valid non-object JSON (`42`, `[1,2]`, `true`, `"hi"`)
  returns `None` instead of the raw value. `Classifier.classify` raises
  `ClassificationResponseError` (was `AttributeError`) on non-dict JSON, and
  `LiteLLMInterface.extract_bulk_data` returns an empty response on non-dict JSON and
  tries one `extract_json_from_text` recovery parse when `json.loads` fails.
  **Behavior change**: a caller that relied on a list being returned now gets `None`.
- **DH-02 / DH-03, null-safe validator and visualizer (D-011).** `validator.py` and
  `visualizer.py` use `.get(k) or []` for `conditions`, `requires_context_keys` and
  `required_context_keys`, so a `model_dump()`-ed FSM (explicit `null` Optionals)
  validates and renders. Sibling sites in `fsm_llm_monitor/bridge.py` and
  `fsm_llm_agents/meta_builders.py` got the same treatment.

### Fixed -- core (`fsm_llm`) and agents audit remediation (plan-2026-09-19-21cd7f8e, iteration 2)

Iteration 2 fixes the defects the iteration-1 review and the re-audit found in
iteration 1's own changes, plus the live-proven and small localised backlog items.
Each fix was reproduced RED first and pinned in
`tests/test_fsm_llm/test_audit_iter2_seam.py` (plus updated tests where a superseded
contract was encoded). Decision ids refer to that plan's `decisions.md` (D-014..D-025).

- **LS-18 / RA-05, Ollama detection by prefix (D-021, e768178).** `is_ollama_model`
  matches only the `ollama/` and `ollama_chat/` prefixes (case-insensitive). A model
  such as `azure/gpt-4o-not-ollama` no longer gets the Ollama `json_schema` grammar
  and forced temperature 0. The classifier's connection kwargs now drop `schema`,
  `model` and `config`, so a `config` kwarg no longer raises `TypeError` in both
  classification paths.
- **Bulk pass provenance and coercion (D-015, 8e178fd).** The bulk pass overwrites a
  stored key only when the stored value is still exactly what the pipeline extracted
  (a sha256 digest of the value is recorded in `context.metadata` at the two Pass-1
  commit sites), so a handler-seeded gate value is never flipped (reviewer repro5:
  `is_verified` False -> True). Bulk values for config-covered keys go through the
  same coercion and validation as per-field values, so a dict no longer lands in a
  `str` key and `24` stays an int instead of becoming `'24'` (RA-02).
- **Re-entrancy guard and ERROR handler merge (D-018, b3431f3).** A same-conversation
  `converse` or `converse_stream` called from inside a handler while a turn is in
  flight raises `FSMError` instead of nesting (reviewer repro1: depth 99 and 99
  provider calls for one user turn, now depth 1). An ERROR handler's returned dict is
  no longer merged into the conversation whose turn was just rolled back (RA-07).
- **RA-01, kwargs tool called with `input=` (D-017, 3f96567).**
  `normalize_tool_input` json-decodes a string `tool_input` that is a JSON object, so
  a keyword-argument tool is called with its arguments instead of `input=`.
- **RA-01b, ReAct tool never ran on the Ollama grammar (D-024, d22a0f2).** The
  react and reflexion think states declare `tool_name` (`str`) and `tool_input`
  (`dict`) as explicit `field_extractions`. Live on `ollama_chat/qwen3.5:9b-q8_0`
  (n=3): tool ran 3/3, versus 0/3 under the auto-minted `any` grammar.
- **RA-04 / RA-06, JSON extraction (d4ec278).** `extract_json_from_text` resumes the
  brace scan after a fenced non-object, so a fenced array's interior object is no
  longer returned, and Strategy 1 and 2 tolerate `RecursionError` on deeply nested
  input instead of raising.
- **LS-01 / LS-06, sanitizer and forbidden names (b674384).** The prompt sanitizer no
  longer passes an unterminated safe tag that swallowed a closing tag (`<b </task>`).
  `is_forbidden_context_entry` also tests the snake-cased form of camelCase key names
  (`apiKey`, `authToken`, ...). Benign prompts are byte-identical; the change for
  non-benign input is not measured by `scripts/eval.py`.
- **LS-02, `extracted_data` prompt section (5d964ec).** The Pass-2 `extracted_data`
  section goes through the security filter, so a declared secret-named field is not
  echoed, and a `datetime` value no longer drops the whole section. Benign data
  renders byte-identically.
- **CF-06, bulk-pass prompt and result filtering (D-019, b5e06b5).** The bulk
  extraction prompt sanitizes user text, and the bulk result drops the `agent_trace`
  marker and forbidden-name keys on both call sites. The sanitization of non-benign
  text is not measured by `scripts/eval.py`.
- **RA-03, classification-owned keys on the bulk pass (D-019, da4eea4).** The
  additive bulk pass no longer fills a key owned by a `classification_extractions`
  entry on a non-agent FSM, so a below-threshold classifier cannot be bypassed by
  bulk extraction. Agent FSMs keep the bulk fill.
- **LV2-01, structured terminal reply (D-020, 3d440ec).** A schema-valid JSON reply to
  a requested `response_format` that has no `message` key is the reply (verbatim JSON
  text in the response and history) instead of the generic apology. Replies over the
  5000-character cap still degrade; behaviour without a schema is unchanged.
- **LV2-02, empty greeting (D-020, 3edf412).** A Pass-2 reply that would be the
  generic apology (for example an empty `message`) is retried once and the retry
  result is returned; an error on the retry keeps the first apology. Normal replies
  make one call.
- **CF-07, stacked `save_session` (D-022, 6b924f7).** `save_session` on a stacked
  conversation saves the root frame's state, data, history and working memory, so
  `restore_session` no longer meets a sub-FSM state that does not exist in the root
  definition. Unstacked saves are unchanged.
- **DH-07 / EF-03, zero-handler turns (D-022, 470ba14).** `execute_handlers` returns
  immediately when no handler subscribes to the timing (new
  `HandlerSystem.handlers_at`), so a zero-handler advance turn drops from 14 to 4
  context deep-copies and a non-copyable context value no longer crashes handler
  timings. The pre-turn rollback snapshots are untouched.
- **LS-05 remainder, bulk confidence (D-023, 8099e54).** `extract_bulk_data` keeps the
  extracted data when the model's confidence cannot be coerced (`"high"`, `null`, an
  object); the confidence falls back to 1.0 instead of failing the whole bulk pass.
- **DH-09, `<=` and `>=` on numeric strings (D-023, d3d2b2a).** `1 <= "1.0"` and
  `"2.0" >= 2` are true (they previously fell back to string ordering).
  `==` and soft equals are unchanged.
- **DH-10, `validation_rules` typing (D-023, 896fc13).** `min_length` and
  `max_length` must be int, `allowed_values` a list, `pattern` a compilable regex
  string; a bad rule fails at FSM load and in `fsm-llm-validate` instead of as a
  `TypeError` on the first extraction turn.
- **LV2-05, anchors and pin (D-011, D-016, 0bab5a5).** Qualified DECISION anchors for the
  exact-0.0 confidence rejection (D-016) and the null-safe validator reads (D-011),
  and a seam test pinning the accepted below-threshold classifier stall. Comments and
  tests only.

### Fixed -- core (`fsm_llm`) and agents audit remediation (plan-2026-09-19-21cd7f8e, iteration 3)

Fix-of-a-fix items from the iteration-2 review and a fresh re-audit (RB-01..RB-12),
two agent-pattern fixes with live proof, and first-touch docs that load. Each fix was
reproduced RED first and pinned in `tests/test_fsm_llm/test_audit_iter3_seam.py` (or
the agent test files). Decision ids refer to that plan's `decisions.md`.

- **RB-01, uncoercible confidence (D-028, 0a5c59b).** A per-field extraction whose
  `confidence` cannot be coerced (`"high"`, `null`, an object, `"95%"`) keeps the
  returned value at confidence 0.5 at both field-extraction rungs, instead of falling
  to the unstructured rung and storing the raw JSON text as the field value.
- **RB-02, structured reply without `message` (D-028, a9dd46f).** A reply to a
  requested `response_format` whose object has no `message` key but has a `reasoning`
  key reaches the user as the JSON text, not as the reasoning alone.
- **RB-10, JSON before a fenced example (D-029, 05744ec).** `extract_json_from_text`
  skips a fenced non-object by blanking its span in place (length-preserving), so an
  object that appears BEFORE a fenced example is found again, the fenced interior is
  never recovered, and Strategy 3 and Strategy 4 read the same string.
- **RB-06, quadratic tag sanitizer (D-029, eb63863).** The prompt tag sanitizer's
  attribute tail excludes `<`, so `<a<a<a...` is linear (10k characters: 2.5 s to
  under 0.5 s) and an unterminated `<b ` can no longer swallow text; its closing tags
  stay escaped. (Iteration 4 found this change also stopped escaping a closing tag with
  a nested `<`, `</original_input <b>`; restored under D-047, see the iteration 4
  section.)
- **RB-11, duck-typed handler system (D-029, 44f89ae).** `handlers_at` is an optional
  fast-path hook, so a `handler_system` that only implements `execute_handlers` starts
  and converses again.
- **LS-09, inline fences kept (D-030, b126340).** `strip_think_and_fences` strips a
  code fence only at the START of the reply, so an inline fence in prose keeps its
  markers. JSON in a mid-text fence is still recovered by `extract_json_from_text`.
- **LS-03 / LS-04 / RB-03, plain-text rung (D-030, b1eac34).** The plain-text rung
  strips `<think>` blocks first and replaces brace-shaped text by the message or
  apology only when it parses as JSON, so `{name}, welcome! ...` and `{1, 2, 3}` reach
  the user after one provider call and a think-prefixed reply is no longer shown raw.
- **RB-05, provenance survives a restart (D-031, b2e2144).** `save_session` persists
  the provenance digests in `SessionState.metadata["pipeline_extracted"]` and
  `restore_session` re-seeds them, so a correction lands after a restart while a
  handler-seeded or `update_context` value is still never overwritten. An old session
  file restores an empty map.
- **RB-12a, refused corrections are reported (D-032, 96a42ac).** A bulk correction the
  provenance rule refuses and the user's message contains is carried on
  `DataExtractionResponse.rejected_corrections` (default `{}`); the stored value is
  unchanged. Ungrounded, landed and agent-managed cases report nothing.
- **RB-12b, reply is told (D-032, 8ee02e3).** `build_response_prompt` takes an optional
  last `rejected_corrections` argument and both Pass-2 call sites pass it, so the reply
  is told the requested value was NOT applied. A turn without a rejection builds a
  byte-identical prompt (hash pinned).
- **RB-07, opt-in `handler_only_keys` (D-033, 0086e60).** New `FSMDefinition` field
  (default `[]`, behaviour unchanged): a listed gate key is dropped from the bulk
  return, the per-field configs and the post-transition configs, so user text cannot
  write it. A handler write, `update_context` and `initial_context` still work; a
  stacked child uses its own list.
- **RB-08, back-edge correction (D-034, D-042, 765dda1).** A transition into a
  different state whose own config-covered key is already set with provenance re-runs
  the target state's Pass-1 extraction, so a same-message correction lands
  ("wait, change my name, it's Bob Jones"). A bare "go back", a self-loop, an
  agent-managed FSM, a handler-seeded key, a forward hop into an empty state and a
  state that owns `classification_extractions` make no extra call and keep the stored
  value.
- **RB-09, memoised identical null extractions (D-035, 1332ac9).** On an Ollama model
  (exact `ollama/` or `ollama_chat/` prefix, temperature 0) an identical null per-field
  extraction is memoised within one extraction call, keyed on field name plus the built
  prompt and message: 3 null keys at `extraction_retries=3` cost 3 provider calls
  instead of 12. Non-Ollama providers, exceptions and successful results are never
  memoised.
- **LV4-04, evaluator_optimizer success (D-036, 2571213).** `EvaluatorOptimizerAgent.run`
  passes `generated_output` as an extra answer key, so a run whose final context holds
  a non-empty `generated_output` and no `final_answer` reports `success=True` instead
  of reading as a prose fallback. An empty output still fails.
- **LV4-01, plan_execute plans (D-036, 4b93236; the seed removal was REVERSED in
  iteration 4, D-046).** `PlanExecuteAgent` no longer seeds
  `plan_steps` with an empty list, so the pipeline's skip-if-set filter stops reading
  `[]` as already set and the plan is extracted (live A/B on `qwen3.5:9b-q8_0`: a
  non-empty plan 3/3 under both the seed-removal and an empty-as-unset variant; the
  smaller change shipped). `success` stays False on those runs, see Known limitations.
- **LV2-05, below-threshold classification is visible (D-037, b44d223).** A
  classification result discarded for being below its `confidence_threshold` is a
  WARNING naming field, intent, confidence and threshold instead of a debug line.
  Behaviour is unchanged.
- **First-touch docs load (D-037, 6a349c7).** The FSM snippets in `README.md`,
  `docs/quickstart.md`, `src/fsm_llm/README.md` and `CLAUDE.md` now pass the loader
  (`description` is required at FSM, state and condition level; classification
  `schema` shape), and `tests/test_fsm_llm/test_docs_snippets.py` loads each one. False
  claims corrected: `required_context_keys` never blocks a transition, handlers do not
  see `_user_input`, `push_fsm` returns the greeting, `pop_fsm` takes the root id and
  merges only declared keys, ERROR handlers skip `start_conversation`, `write_keys` is
  advisory, a custom `LLMInterface` needs `extract_bulk_data`, the CLI needs
  `LLM_MODEL`.
- **Anchors (D-038, D-039, D-040, 636d7e9).** The inline LS-01, LS-02 and LS-06
  comments in `prompts.py` and `constants.py` became qualified DECISION anchors naming
  the constraint and the rejected alternative. Comment-only.

Not done in iteration 3:

- **RB-04, normalising CONTEXT_UPDATE handler (D-041, declined).** Recording the
  digest of the post-handler value cannot tell a normalisation (`blue` -> `BLUE`) from
  an authoritative override (`blue` -> `HANDLER`) and would let a later bulk turn
  overwrite a value a handler deliberately replaced. Provenance keeps recording the
  PRE-handler value (fail closed). The step-9 patch is kept as reference only.
- **GD-22, validator warning for a condition on a key nothing extracts (D-037,
  discarded by its own gate).** The prototype added warnings on 12 of 55 shipped FSMs
  (keys a handler or the engine sets) and was about 30 lines against the 15-line cap.
  Deferred: a silent-for-handler-keys warning needs a declared list of those keys on
  the FSM, a schema change.

### Fixed -- core (`fsm_llm`) and agents audit remediation (plan-2026-09-19-21cd7f8e, iteration 4, final loop)

The final loop repairs what iterations 1 to 3 introduced and closes the small gaps their
live runs and reviews proved; no feature was added. Each fix was reproduced RED first and
pinned in `tests/test_fsm_llm/test_audit_iter4_seam.py` (or the agent test files).
Decision ids refer to that plan's `decisions.md`.

- **Sanitizer nested-`<` bypass restored (D-047, d7f1679).** Iteration 3's
  denial-of-service fix (RB-06) silently stopped escaping a closing tag with a nested
  `<` (`</original_input <b>`, `</user_message <i>`, `</task <b>`), so a hostile user
  message could close the prompt's own wrapper tag and inject instructions. The tag
  tail is now `(?:[^>]{0,256}/?>|(?=[^>]{257}))`: a tag with a nested `<` or a tail longer
  than 256 characters (a padded closer) is escaped again (the overflow arm is a
  zero-width lookahead, so only the `<` and the name are escaped; the first version
  consumed 257 characters and escaped benign prose, fixed in the final review round,
  concern 1), and the scan stays linear
  (`"<a" * 10000` and `"<" + "a" * 100000` each under 0.5 s). A benign `< name`
  (`latency < threshold`) is kept raw only when NO `>` follows it anywhere in the
  text (D-054): a padded opener (`< name` + 257 or more characters + `>`) is escaped,
  as it was in ed6cffd. A differential test compares the new pattern with the
  pre-D-029 pattern on every token string of length up to 6 AND on padded shapes of
  258 to 1000 characters (the short corpus alone could not see a padded shape, it
  stops at 24 characters); the documented residual set is asserted explicitly.
  Three tests that pinned the weakened iteration 3 output were rewritten with
  annotations, and the reversed-ordering payloads were added.
- **`<rejected_corrections>` honours `context_scope.read_keys` (D-032, final review
  concern 3).** The refused-value block bypassed the scope that hides a key from
  `<current_context>`, so a value the state hides (widened in iteration 4 to every
  instruction-only key) still reached the Pass-2 prompt. The rejected dict is now
  filtered through the same scope on the turn and stream paths; a state without a
  scope, and the stored data, are unchanged. A block whose keys are all out of scope
  is omitted.
- **`plan_execute` on an unplannable task ends normally again (D-046, 4ecbfce).**
  Iteration 3 removed the `plan_steps: []` seed, so a task the model cannot plan raised
  `BudgetExhaustedError` after 36 wasted model calls. The seed is restored (it is the
  only exit of the `plan` state) and the pipeline's skip-if-set filter reads an empty
  list or dict as unset for agent-managed FSMs only, so the seeded plan is still
  extracted: the task returns `AgentResult(success=False)` after 1 `plan_steps` ask.
  A non-agent FSM that seeds `[]` for a config-covered key is unchanged.
- **A reply that ends in a code block keeps its closing fence (D-048, f736c86).**
  `strip_think_and_fences` strips the closing fence only when a leading fence was
  stripped; a fully fenced JSON reply is still unwrapped, and `extract_field` and
  `extract_bulk_data` still recover an object followed by a stray closing fence. One
  annotated `_STRIP_CORPUS` row was rewritten (a lone stray closing fence is now kept
  by the helper).
- **The "not applied" note needs a whole-token match (D-049, 007a405).** The grounding
  test for `rejected_corrections` was a raw substring, so `bored now` grounded `red`
  and `1500 items` grounded `500`. It now requires a whole token of at least 3
  characters; `make it red` and `make it red.` still ground.
- **A failed bulk extraction is surfaced to Pass 2 (D-050, 07afae1; LV5-01).** When the
  bulk extraction call raises, the reply used to claim an update the store never made.
  `DataExtractionResponse` gains `extraction_failed: bool = False` and
  `build_response_prompt` gains a last optional parameter `extraction_failed`, which
  adds one plain line saying a restated value may not have been stored. A turn without a
  failure builds a byte-identical prompt. One annotated iteration 3 test that pinned the
  exact signature tail was rewritten.
- **`fsm-llm-validate` warns about a `handler_only_keys` entry that protects nothing
  (D-051, ed39590).** One WARNING for a listed key no state references (a likely typo)
  and one for a listed key that is a `classification_extractions` field name (the
  classification channel is not covered). Warnings only, silent for an empty list, zero
  new warnings on every shipped example and agent builder.
- **An instruction-only key is reported when a correction is refused (D-052, 1b53d7c;
  LV5-03).** A bulk value for a key with no field config that already holds a different
  value is now listed in `rejected_corrections` under the same grounding test, for
  non-agent FSMs only; the stored value is still never overwritten. The change is net
  minus one source line, inside the plan's 10-line gate.
- **Docs-snippet coverage extended (D-052, ea49e53, test only).** The docs-snippet test
  also scans `docs/api_reference.md`, `docs/architecture.md`, `docs/fsm_design.md` and
  `docs/handlers.md`; none carries a full FSM (`"initial_state"`) today, so they add no
  case and the guard still requires the four first-touch files to contribute. `docs/fsm_design.md`
  carries no full FSM (no `"initial_state"` block), only fragments, and fragments are
  NOT loaded by the test: a wrong fragment there is not caught.
- **LV6-01, an applied correction listed as rejected (D-052, 2da48c4).** Introduced by
  the LV5-03 change above (1b53d7c) and found by the live audit, fixed inside this run:
  on a back-edge turn the confirm state's refusal was merged into
  `rejected_corrections` and never pruned, although the target state's re-extraction
  then applied the value, so Pass 2 was told the opposite of the store (3/3 live looks;
  0/3 in iteration 3). After the re-extraction an entry is dropped when the stored
  value now equals it under the same trimmed, case-insensitive comparison as the merge
  point; a value a handler edited or that never landed stays listed. One live look after
  the fix showed no block (n=1, an observation, not a rate).

Regressions found and fixed inside this run (stated plainly):

- **Iteration 3 introduced two regressions, both found by the iteration 3 adversarial
  review and fixed in iteration 4.** (1) The sanitizer nested-`<` bypass: the RB-06 fix
  stopped escaping a closing tag with a nested `<` (introduced in iteration 3 at
  eb63863, fixed at d7f1679). (2) The `plan_execute` exception: the LV4-01 seed removal
  made an unplannable task raise `BudgetExhaustedError` after 36 model calls instead of
  returning `success=False` (introduced at 4b93236, fixed at 4ecbfce).
- **Iteration 4 introduced one regression, LV6-01, found by the live audit and fixed
  inside the same iteration.** Introduced by the LV5-03 change (1b53d7c), fixed at
  2da48c4. The live audit found it by reading the raw prompts of the back-edge turn; the
  check booleans (5/5 in all three looks) did not see it. The offline suite did not
  either, because no test built a state whose key is config-covered only in the target
  state.

### Changed -- public contract (iterations 1 to 4)

- **Bulk overwrite is provenance-gated (D-015).** Supersedes iteration 1's D-004
  overwrite rule. A later-turn correction returned by the bulk pass replaces a stored
  key only when the key is config-covered, the FSM is not agent-managed and the
  stored value is still exactly the value the pipeline extracted. A handler-set or
  `update_context`-written value for a config-covered key is never overwritten by the
  bulk pass (iteration 1 allowed it). Iteration 3 (D-031) persists the
  provenance across `save_session`/`restore_session`; a session file from before that
  has none, so corrections fall back to skip-if-set until the pipeline re-extracts the
  key. A correction that changes only letter case or
  whitespace counts as no change.
- **Exact-0.0 confidence is rejected (D-016).** A per-field extraction the model
  reports at `confidence` exactly `0.0` is not stored, whatever
  `confidence_threshold` is. There is no knob to accept a 0.0-confidence value.
- **Re-entrant `converse` raises (D-018).** Calling `converse` or `converse_stream`
  on a conversation from inside a handler (or between `next()` calls of an open
  stream) while a turn is in flight raises `FSMError`. `update_context` stays
  allowed.
- **ERROR handler return value is not merged (D-018).** The dict an ERROR-timing
  handler returns is dropped (debug-logged). `update_context` is the supported write
  path for an ERROR handler.
- **Forbidden-name keys are not captured by the bulk pass (D-019).** An
  instruction-only field whose name matches a forbidden pattern (`password`,
  `api_key`, ...) is no longer captured by the bulk pass; declare it in
  `field_extractions` or `required_context_keys` to capture it.
- **Ollama detection by prefix (D-021).** `is_ollama_model` is true only for
  `ollama/...` and `ollama_chat/...`; a model that merely contains "ollama" in its
  name (a proxy route) now takes the non-Ollama path.
- **Bad `validation_rules` are rejected at load (D-023).** An FSM definition whose
  `validation_rules` carries a wrongly-typed value (`min_length: "abc"`,
  `allowed_values: 5`, `pattern: "["`) now raises `ValueError` when loaded, where it
  previously loaded and failed on the first extraction turn. `<=` and `>=` now agree
  with `==` on numeric strings.
- **`extract_json_from_text` returns `dict | None`.** It is exported in
  `fsm_llm.__all__`; valid non-object JSON (`42`, `[1,2]`, `true`, `"hi"`) now returns
  `None` where it returned the raw value (iteration 1, D-010).
- **`_build_response_format_section` takes a positional parameter.** The method on
  `ResponseGenerationPromptBuilder` gained `plain_text` (iteration 1, D-003). A
  subclass that overrides it with the old zero-argument signature now raises
  `TypeError` when the base calls it.
- **LV-01 re-extraction cost.** A required field that stays unset (null,
  zero-confidence or a rejected container value) is re-asked every turn, and
  `extraction_retries` makes each such field cost up to `1 + retries` extra provider
  round-trips per turn (wording corrected in iteration 3, D-027). One
  agent test went from 34 to 152 provider calls. Retries against Ollama are
  byte-identical (temperature is forced to 0).
- **React and reflexion extraction contract (D-024).** The think state extracts
  `tool_name` and `tool_input` through explicit typed configs; extraction order
  changed (`should_terminate` now precedes them). `tool_input` can still come back as
  an empty `{}`; the tool layer fills required parameters from the task text.
- **Structured terminal reply and greeting retry (D-020).** A terminal state with an
  output schema returns the schema JSON as the reply; an apology-producing Pass-2
  reply is retried once, so one turn can make at most one extra LLM call, and only
  on a turn that would otherwise have shown the apology.
- **Stacked `save_session` and zero-handler turns (D-022).** `save_session` on a
  stacked conversation saves the root frame (a session saved while stacked used to
  record the sub-FSM state). Turns with no handler for a timing skip the handler
  deep-copies.

- **`handlers_at` is an optional hook (D-029).** `execute_handlers` reads it with a
  guarded `getattr`; a duck-typed `handler_system` without it works and keeps the
  handler deep-copies (it just does not get the zero-handler shortcut).
- **`extract_json_from_text` fence handling (D-029, D-030).** A fenced non-object is
  skipped by blanking its span in place (length-preserving), not by truncating the
  text before it; `strip_think_and_fences` strips a fence only at the start of the
  reply. Six iteration-1 corpus rows in `test_audit_iter1_seam.py` were rewritten to
  the D-030 semantics.
- **Uncoercible confidence keeps the value (D-028).** A per-field `confidence` that
  cannot be coerced now means "value kept at 0.5", not "failed extraction".
- **Brace-shaped prose is no longer replaced (D-030).** The plain-text rung decides by
  parseability, not text shape. Trade-off: a malformed brace envelope such as
  `{"message": hi}` (no recoverable JSON) is now shown as text instead of the generic
  apology. `<think>` in STREAMED replies still passes through raw (buffering would
  break time-to-first-token).
- **Provenance is persisted (D-031).** Supersedes the D-015 clause "not persisted":
  the digest map is stored in `SessionState.metadata["pipeline_extracted"]`, exposed as
  `_pipeline_extracted` in `FSMManager.get_complete_conversation()['metadata']` and
  written to the session file (digests, not values; same trust domain as
  `context_data`). A session file without the key restores an empty map and fails
  closed.
- **`handler_only_keys` is a new `FSMDefinition` field (D-033).** `model_dump()` emits
  `handler_only_keys: []` for every FSM, so a snapshot that compares a dumped
  definition exactly must expect the new key.
- **Rejected corrections reach Pass 2 (D-032).** `DataExtractionResponse` gained
  `rejected_corrections` (default `{}`) and `build_response_prompt` a last optional
  argument; a turn with no rejection is byte-identical. A caller that subclasses or
  wraps `build_response_prompt` positionally is unaffected.
- **Retry-cost wording corrected (D-027, review note 11).** A required field that
  stays unset costs up to `1 + extraction_retries` extra provider round-trips per
  field per turn (the "LV-01 re-extraction cost" bullet above is corrected in place);
  on Ollama the identical retries
  are now collapsed by the null memo (D-035).
- **Extra calls (D-034, D-042, D-035).** A transition into a different, already-filled
  state costs +1 bulk call (+1 retry per still-null required key; measured 4 -> 5
  provider calls on the back-edge turn, 5 -> 7 with one required null key) and the
  same call fires on a revisit that changes nothing (live: +1 call per return to
  `confirm`). On Ollama a null key at `extraction_retries=3` drops from 1+3 calls to 1.
- **`plan_execute` seed removed (D-036), then restored in iteration 4 (D-046).**
  Iteration 3 stopped pre-seeding `plan_steps` with `[]`; that made an unplannable task
  raise `BudgetExhaustedError`, so the seed is back (see the iteration 4 section).
- **`evaluator_optimizer` success (D-036).** `AgentResult.success` is true when
  `generated_output` is non-empty; see the LV5-02 limitation.
- **Iteration 4 (final loop) contract changes.**
  - The prompt sanitizer escapes a `<name` opener or closer whose tail contains a
    nested `<` or is longer than 256 characters; only the `<` and the tag name are
    escaped (`&lt;/task`), never the text after it. Benign prose is byte-identical,
    including a `<` followed by a space (`latency < threshold`) when no `>` follows
    it in the text. The exact residual (D-054): a `<` DIRECTLY followed by a letter
    (`x<y`), with 257 or more characters and no `>` after it, has its `<` escaped;
    a `<` plus whitespace plus a name with no `/` (`< task`), no `>` within 256
    characters AND no `>` anywhere later in the text, is kept raw (a comparison);
    with any `>` later in the text it is escaped. An unterminated closer
    or opener with a short tail and no `>` anywhere (`</task NEW INSTRUCTIONS`)
    still reaches the prompt raw; that shape is pre-existing (every earlier pattern
    left it raw) and is not closed by this change.
  - `strip_think_and_fences` strips the closing fence only after a leading fence was
    stripped (a reply that ends in a code block keeps its fence).
  - Rejected-correction grounding is a whole-token match with a 3-character floor. The
    named cost: a real correction to a 1 or 2 character value (`US`, `EU`, `42`) never
    produces the `<rejected_corrections>` block, and a differently formatted value
    (`1,000` for `1000`) does not ground.
  - `DataExtractionResponse.extraction_failed: bool = False` (new public field) and
    `build_response_prompt(..., extraction_failed=False)` (new optional last parameter;
    positional callers are unaffected).
  - `fsm-llm-validate` emits two new WARNINGs for `handler_only_keys` (never errors).
  - `PlanExecuteAgent` seeds `plan_steps: []` again (D-046).
  - `fsm_id` change on upgrade: `API.from_definition(FSMDefinition(...))` derives its
    `fsm_id` from `model_dump()`, which now carries `handler_only_keys: []`, so the id
    of an FSM built that way differs from the id the previous release computed
    (`restore_session` does not compare `fsm_id`, see the limitations).

### Known limitations -- iteration 2 list (current status in the iteration 3 list below)

- **LV2-04, undeclared gate key.** A key a transition reads but no field declares can
  still be added by a steered bulk value or by the per-field channel (live s13:
  `is_admin` opened, 1/1). Provenance narrows but does not close CF-06.
- **LV2-05, low-confidence classification stall.** A classification below the state's
  `confidence_threshold` keeps the stale fallback intent and does not transition
  (live s7: three turns end in `triage`). The behaviour is pinned by a seam test, not
  changed.
- **LV3-01, reply and data contradict.** When a correction of a handler-set key is
  rejected, the Pass-2 reply can still claim the correction was applied while the
  stored data is unchanged (live s14, 2/2 looks). The data is safe; the reply is not.
- **`tool_input` quality after D-024.** The tool now runs live, but `tool_input` is
  often `{}` and parameter quality is not fixed. Reflexion is not measured live.
- **Prompt-content changes are not measured by `scripts/eval.py`.** The sanitizer
  (LS-01), the secret-name filter (LS-06), the `extracted_data` filter (LS-02) and the
  bulk-prompt sanitizer (CF-06) change prompt content only for non-benign input;
  benign prompts are byte-identical (hash test). The 95.3% eval baseline stays stale
  and must be re-run before it is trusted.
- **Live evidence is n=1 to n=3 on one model** (`ollama_chat/qwen3.5:9b-q8_0`, 106
  calls). The LV2-02 retry was not exercised live (0 empty greetings in 18 starts);
  it is proven offline only. Non-Ollama providers and non-`any` union types are not
  measured.
- **Deferred to iteration 3:** the efficiency batch (identical retries, filter
  caches), LV2-03 (form back-edge correction), provenance persistence across
  `restore_session`, and LV3-02/LV3-03 (`<information_still_needed>` listing a filled
  key; `agent_trace` visible in per-field extraction prompts).

### Known limitations -- after iteration 4 (final loop; supersedes the iteration 3 list where it differs)

Live evidence: `ollama_chat/qwen3.5:9b-q8_0` only, 1 to 4 looks per scenario, 173 calls
against a target of about 150 and a hard stop of 180, box idle-checked before the run.
Every rate is a look count, not a significance claim. Resolved since the iteration 3
list: LV5-01 (D-050) and LV5-03 (D-052) are fixed offline; the sanitizer bypass, the
`plan_execute` exception and LV6-01 are fixed (see the regression list above).

- **The `scripts/eval.py` baseline (95.3%, N=3 median, Run 006) was NOT re-measured
  against four iterations of prompt-content changes and is STALE. This is a release
  gate.** The prompt-content changes since that baseline are: the tag sanitizer
  (LS-01, restored in D-047), the bulk-prompt sanitising (CF-06), the `extracted_data`
  filter (LS-02, nested-dict security filtering), the plain-text stream prompt, and
  the `<rejected_corrections>` block and the extraction-failed line (each only on a turn
  with a rejection or a failure). Benign prompts are byte-identical (hash tests), but no
  gate available in this run observes prompt effects at eval scale (the fast gates mock
  the LLM). Re-run the eval suite before trusting the baseline.
- **The live audit found LV6-01 that the check booleans missed.** All three back-edge
  looks scored 5/5 while the raw Pass-2 prompt carried a wrong `<rejected_corrections>`
  block. Read raw `calls[]`, not `summary.checks`.
- **`EvaluatorOptimizerAgent` `success` after the refinement cap (LV5-02, concern 6,
  D-052).** `success=True` even though the evaluator never passed the output;
  `max_iterations_reached` in the final context is the only signal. It needs an owner
  decision on a public contract and was not changed in a final loop.
- **`<extracted_data>` is not scoped by `context_scope.read_keys` (D-054, final
  review concern 3).** `read_keys` scopes `<current_context>` and
  `<rejected_corrections>` only. `<extracted_data>` (the keys extracted this turn from
  the user's own message) is shown unscoped, so a key a state hides from
  `<current_context>` still reaches Pass 2 on the turn it is extracted. It is
  pre-existing since D-005 and was deliberately not changed in a final loop.
- **RB-04 declined (D-041).** A normalising CONTEXT_UPDATE handler on an extracted key
  still disables later LLM corrections of that key.
- **`restore_session` does not compare `fsm_id`** (concern 11): a session file can be
  restored onto a different FSM definition without error.
- **RB-08 fires on any transition into a filled state (concern 9)**, not only on back
  edges: +1 bulk call per return to a filled state (live: `confirm`), even when nothing
  changes. The name and cost are accepted for now.
- **`handler_only_keys` is opt-in.** LV2-04 stays the default for FSMs that do not
  declare it; only listed keys are covered and a classification-owned key is not (the
  validator now warns about that overlap). The bulk-output filter on a listed key was
  never exercised live (the model did not return the key in a bulk pass).
- **The grounding floor is 3 characters (D-049).** A real correction to a value of 1 or
  2 characters never produces the block, so the reply may still claim the change was
  made; the false-positive rate on real forms is unmeasured. The intended live
  true-positive of the instruction-only report (D-052) was not observed: the s14 value
  `US` is under the floor, so live s14 does not exercise it.
- **Not observed live:** the D-050 extraction-failed line (no bulk call errored while
  Pass 2 could still run), the D-046 unplannable-task branch (the live task was
  plannable), the RB-08 cost on real forms, non-Ollama providers.
- **plan_execute plans but does not execute (LV5-07) is unchanged:** the plan was
  extracted 4/4 live runs and the model narrated instead of calling the tool 4/4;
  `success` was never True, and the honest `completed with no execution evidence`
  WARNING was logged. Three of the four runs were truncated by the harness call cap.
- **LV6-02 / LV6-03 (low, info):** the bulk pass can store a model-invented key
  (`intent: jailbreak_attempt`); the low-confidence classification WARNING count depends
  on the classifier, so "exactly two WARNINGs" is not a property of the code.
- **The unchanged items of the iteration 3 list stay open:** LV3-01 (reply/data
  contradiction, measured 0/10 cumulative on s14, not significant), LV5-04, LV2-05,
  LV2-10/LV4-08, LV3-02/LV3-03, rewoo/maker_checker/debate at base, LV5-05, LV5-08,
  `tool_input` quality, and the accepted items of D-027 and D-052 (the `0.5` unscored
  confidence literal duplicated in `llm.py`, the 303-line `_execute_data_extraction`).

### Known limitations -- after iteration 3

Live evidence: `ollama_chat/qwen3.5:9b-q8_0` only, 1 to 3 looks per scenario at
temperature 0.3 to 0.5, about 282 calls against a 250-call budget; 15 of 32 records
ran at 10 to 20 times normal latency from unrelated load on the shared Ollama (the
scripts' own 280/300 s alarms fired inside in-flight calls, not provider timeouts).
No rate carries a significance claim.

- **`handler_only_keys` is opt-in (LV2-04 stays the default).** Without the list a
  key a transition reads can still be set by a steered value (live control arm:
  `is_admin` opened). With it, the per-field closure held 2/2 looks; the bulk-output
  filter was not exercised live (the model never returned `is_admin` in the bulk
  pass). Only listed keys are covered: an unlisted gate key stays writable and a
  classification-owned key is not covered. Say "closable per FSM for the listed
  keys", not "LV2-04 closed".
- **A normalising CONTEXT_UPDATE handler disables corrections on that key (D-041).**
  Provenance records the pre-handler value, so a same-timing handler edit of an
  extracted key blocks later LLM overwrites of it; the reply is told the change was
  not applied.
- **LV5-01, a failed back-edge re-extraction is silent to Pass 2 (FIXED in iteration 4,
  D-050; the text below is the iteration 3 state).** If the RB-08
  re-extraction call errors, `_bulk_extract_from_instructions` returns `{}` (a failed
  bulk is indistinguishable from "nothing to extract"), the store keeps the old value
  and the reply can still say it was updated (live: store `Jane Doe`, reply "I've
  updated your name to Janet Doe"). Live 2 of 3 looks landed the correction; the
  third failed on a script alarm and its re-run crashed. Same class as LV3-01 on a
  different path.
- **LV3-01, reply and data contradict: measured 0/8 after the rejected-corrections
  block (was 2/4).** Not significant at this n (Wilson upper bound about 32%); only 4
  of 8 replies say the value was not changed, the rest merely restate it.
- **LV5-03, `<rejected_corrections>` lists config-covered keys only (FIXED in
  iteration 4, D-052; the text below is the iteration 3 state).** An
  instruction-only key that already holds a value (`region` in the live fixture) is
  refused by skip-if-set and never reported, so the reply is not told (0/8 replies
  happened to claim it).
- **LV5-04, a bare "Yes" after the back edge does not flip `user_confirmation`** while
  the reply says confirmed: a stale set key is only correctable by the bulk path,
  which returns `{}` for a bare affirmation. Pre-existing shape of D-015.
- **LV2-05, low-confidence classification stall** is unchanged (three turns end in
  `triage`); it is now a visible WARNING (live: one line per stalled turn).
- **LV2-10 / LV4-08, history summary is never injected** into the prompt (the history
  cap is not measured against `scripts/eval.py`); CF-08 turn atomicity and LS-10 are
  deferred with it. **LV3-02 / LV3-03**: `<information_still_needed>` can list a
  filled key and `agent_trace` can be visible in per-field extraction prompts.
- **rewoo, maker_checker and debate fail at base too** (LV4-02/03/05): the extractor
  is fed the literal message "Continue." and the model reports it does not contain the
  plan. The typed-config recipe of D-024 does not transfer; a fix needs the plan
  produced in a call whose input is the task. Not run in the iteration-3 live block.
- **plan_execute plans but does not execute (LV5-07, carried forward from LV4-01).** The plan is extracted (6/6 live
  runs, non-empty `plan_steps`), but the model narrates "I've completed the search"
  and never calls the search tool (no tool output in any prompt, no tool-execution
  log), so `success` is False. Final `success` is UNMEASURED in this block: 6/6 runs
  ended on the 250 s agent timeout or the script alarm at 25 to 60 s per call.
- **LV5-02, `EvaluatorOptimizerAgent` `success=True` after the refinement cap even
  though the evaluator never passed the output.** `evaluator_optimizer.py:207-208`
  forces `evaluation_passed=True` at the cap (pre-existing design); since
  `generated_output` became an answer key (LV4-04) that reads as `success=True`.
  `success` means "produced a non-empty output"; `max_iterations_reached` in the
  context is the only signal. Live run 1: evaluator `passed False`, `refinement_count`
  2, `success True`. Needs an owner decision.
- **LV5-05, `evaluation_result` (a dict set by a handler) appears in the evaluator
  agent's Pass-2 `<current_context>`.** Handler-set, not extracted, so the no-dict-in-
  extraction rule is not violated; it is a dict-valued key in a prompt block.
- **LV5-08, literal backslash-n in extracted multi-line values.** Field
  extraction of a multi-line value can store a literal backslash-n (live n=1: the
  evaluator counted one line and the loop spent both refinements; not reproduced in the
  other two runs). The parse does not unescape.
- **`tool_input` quality (D-024).** The tool runs live (3/3, no crash), but `tool_input`
  is `{}` 3/3 and the tool receives the whole task sentence as `query`; parameter
  quality is not fixed. Reflexion is not measured live.
- **Accepted, not fixed (D-027).** `<=`/`>=` agree with numeric strings while `==` does
  not (review note 12); a classification-owned key above threshold blocks correction
  (the classifier is the owner, note 9); bulk values validate at a fixed confidence 1.0
  (note 8); a truncated or malformed brace envelope shows as text (D-030). Also
  deferred: object-typed `output_schema` fields, reasoning-mode mapping keys, GD-05
  code half (`find_dotenv(usecwd=True)`, default model), GD-16 (CLI output via the
  logger), GD-13..GD-25 beyond the doc items fixed in step 20.
- **Prompt-content changes are not measured by `scripts/eval.py`, and its 95.3%
  baseline is stale.** The sanitizer (LS-01), the secret-name filter (LS-06), the
  `extracted_data` filter (LS-02), the bulk-prompt sanitizer (CF-06), the
  `<rejected_corrections>` block (only for turns with a rejection), the plain-text
  stream prompt and the RB-06 tag pattern change prompt content; benign prompts are
  byte-identical (hash tests). Re-run the eval suite before trusting the baseline.
- **Not measured live.** RB-01 non-numeric confidence, RB-03 `<think>` prefix, RB-10
  fences, RB-05 restore-then-correct and RB-06 had no live exposure (0 of 247 raw calls
  carried `<think>`, a fence or a non-numeric confidence). The memo (RB-09) fired twice,
  both null-to-null, so a stale null masking a non-null resample is not observed, not
  excluded. Non-Ollama providers, non-`any`/`str` union types and the false-positive
  rate of the rejected-correction grounding test at scale are not measured.

### Added — new package: `fsm_llm_harness` (extra: `pip install fsm-llm[harness]`)

An FSM-LLM-native emulation of the iterative-planner protocol: a 6-state
EXPLORE / PLAN / EXECUTE / REFLECT / PIVOT / CLOSE machine with mechanically
enforced gates, filesystem-as-memory artifacts, per-role file ownership, a
2-attempt autonomy leash, and a small-model hardening layer. Additive and
backward-compatible: **no Python file under `src/fsm_llm/` was modified**, and the
2-pass core contract is unchanged. The only other packages touched are
`fsm_llm_agents` (two files, listed under *Changed* below) and packaging/CI.

- **`HarnessAgent`** — the protocol driver. Builds the harness FSM, registers a
  handler per state entry, dispatches one worker role per entry, and owns all nine
  gate flags. Executor dispatches on one plan step are bounded by
  `max_fix_attempts * (1 + max_leash_grants)` for **any** sequence of approvals;
  the approval callback cannot raise it. The default approval callback DENIES, so
  an unattended run cannot approve its own plan or close itself.
- **`build_harness_fsm()`** — 6 states, 9 transitions. Every gate is a JsonLogic
  `TransitionCondition`, so a gated edge is DETERMINISTIC or BLOCKED, never an LLM
  judgement call, and every condition declares `requires_context_keys` so a
  garbled worker reply leaves the edge blocked rather than accidentally satisfied.
- **`artifacts`** — pydantic models and Markdown (de)serializers for 15 artifact
  kinds, the 9 decision entry-type schemas and the 6 Presentation Contracts, with
  strict grammars (`plan.md`'s 11 ordered sections, `decisions.md`'s
  `## D-NNN | PHASE | YYYY-MM-DD` header, `changelog.md`'s 8 pipe-delimited fields).
- **`storage.PlanDirectory`** — plan-id minting, atomic artifact writes
  (`mkstemp` in the target's own directory + `os.replace`), LESSONS `[I:N]`
  eviction, the SYSTEM line cap, the 4-plan cross-plan sliding window, and
  resumable run state read back from `state.md` itself.
- **`plan_validator`** — `pre_step_gate()` (4 slugs, ordered, short-circuit, all
  HARD) and `audit()` (30 structural checks; a check that raises is reported as an
  ERROR rather than suppressing the rest).
- **`tools`** — `Workspace` (confined source tree) and `PlanMemory` (confined
  **and** ownership-scoped plan directory) sharing one `resolve()` chokepoint,
  plus 13 agent-facing tools. `run_command` is off by default and `git` is
  deliberately absent from the command allowlist.
- **`roles`** — six frozen `RoleSpec`s derived from a single `OWNERSHIP` table, so
  a role's tool scope, its prompt text and its owned artifacts are one fact read
  three times. `build_default_worker_factory()` builds the stock worker, backed by
  `NativeFunctionCallingReactAgent`.
- **`hardening`** — `strip_model_noise`, `parse_json_payload`,
  `parse_role_output`, `coerce_worker_output`, `retry`. All fail CLOSED: a garbled
  reply is never retried into a pass.
- **`fsm-llm-harness` CLI** — `new` / `resume` / `status` / `validate` / `close`,
  with exactly three exit codes (`0` pass, `1` negative answer, `2` RESERVED for a
  HARD gate refusal). `close` is dry-run unless `--apply` and refuses to compress
  a directory with audit ERRORs.

**Gates read the filesystem, not the model's report.** `findings_count` is a count
of non-empty `findings/*.md` files; a dispatch that holds a write tool and claims a
write must show a tool call whose target now carries bytes; and a failed
observation leaves a gate value unchanged rather than writing a zero. This is the
package's central design commitment, and it is a response to measurement: a small
local model asserted completed code changes over an untouched workspace, and
claimed three findings over an empty directory. Prompt wording did not fix it.

**Status, stated as measured.** Offline the package is green (1,793 tests, `ruff`
clean, `mypy` 0 errors). Live on a local 4B model (`ollama_chat/qwen3.5:4b`), the
harness-level criteria pass — a full EXPLORE→CLOSE traverse whose plan directory
audits with zero ERRORs (3/3), the leash halting at exactly 2 attempts and not
resettable by an approving callback (6/6), a REFLECT→PIVOT→PLAN loop-back (3/3) —
and after the driver-assigned EXECUTE target fix (next section) the single-state
model-level criteria are MET at the untouched bars for the first time: write tool
issued 5/5 and workspace bytes 5/5 (bar >=4/5), strict sha256 content-hash match
4/5 (vs >=4/5), findings 5/5 (bar >=4/5). The new graded END-TO-END criterion on
real workers (L6, n=3) measured **0/3 against its floor and is NOT met**: two runs
halted honestly at the EXPLORE redispatch cap, one reached PLAN and stalled
sluglessly after an empty plan-writer reply (verified writes 3/3; no crashes).
This is recorded rather than rounded up: the package is not claimed to be
production-ready, and a small model is not claimed to drive it unattended to a
useful result.

### Added — harness measurement iteration: durable bench, driver-assigned EXECUTE targets, e2e criterion

- **`scripts/harness_bench.py` + `scripts/bench_data/`** — a durable, powered
  bench for harness capability claims: pre-registered fixed-n blocks (n=40/arm),
  6-field manifests (prompt-bytes hash, tool surface, fixture hash, model digest,
  arm, git commit), append-only raw jsonl rows, and a `report` subcommand that
  recomputes every k plus Wilson CI and Fisher exact (stdlib-only math) from the
  committed rows. Blocks are committed under `scripts/bench_data/` (tracked), so
  future numbers can be diffed — earlier benches lived in a gitignored scratch
  directory and no longer exist.
- **Seed determinism dispositioned by probe** — ollama honors `seed` for `:4b`
  (same seed byte-identical at temperature 0.7, different seed diverges; raw
  probe committed at `scripts/bench_data/seed-probe/probe.json`). `seed` is
  plumbed as an optional keyword-only parameter through
  `build_default_worker_factory` → `NativeFunctionCallingReactAgent`'s
  `litellm.completion` call site (default `None` = key absent, byte-identical
  call shape to before); per-row seeds are recorded in every bench row.
- **Driver-assigned EXECUTE write target** — baseline block B0 measured the
  wrong-ROOT defect: native EXECUTE dispatches content-matched the requested
  edit **2/40**. The fix extends the driver-assigned-target pattern to EXECUTE:
  `derive_execute_target` reads plan.md's Files To Modify and the dispatch names
  the exact target path + tool; an unparseable plan falls back to the previous
  prompt byte-identically. Post block B1, same manifest: **40/40** (Fisher
  p=1.6e-20). The ReAct control arm measured 0/40 in both blocks. The armed
  standing-bar classes were then re-run ONCE: L4 MET for the first time (write
  tool 5/5, bytes 5/5, strict content-hash 4/5 vs >=4/5), L5 MET 5/5;
  `MODEL_BAR=4` / `RUNS_MODEL=5` unchanged.
- **`TestL6EndToEndRealWorkers`** — the first graded end-to-end criterion on
  REAL role workers (n=3, disk-derived rubric vectors committed under
  `scripts/bench_data/l6-e2e/`, DENY-default disk-bound approval stub). Floor
  **NOT MET, 0/3** (two honest explore-cap halts at EXPLORE, one slugless PLAN
  stall; verified writes 3/3). Two structural findings recorded: EXPLORE over an
  empty plan directory clears the 3-findings gate ~1/3 of the time vs 5/5 on a
  seeded corpus, and PLAN has no redispatch budget, so one empty reply becomes a
  stall.
- **Adversarial audit executed, not just read** — 5/5 load-bearing guard
  mutations (leash-cap boundary, writable-key allowlist, empty-file gate
  counting, ownership deny branch, live-gate short-circuit) each flipped tests
  red in a scratch copy (93 red total); `test_cli.py`'s exit-code 0/1/2 contract
  close-read verdict: CLEAN.
- **Count-pinning tests** (`tests/test_packaging.py`) — documented test-count
  literals are checked against one `pytest --collect-only` subprocess, so doc
  drift now fails the suite.
- **Anchor hygiene** — 18 dead plan-ids retired via the skill's `retire` tool;
  plan-validator `[anchor-unknown-plan]` errors 155 → 0.

### Added — packaging and CI

- `harness` extra (pulls `fsm-llm[agents]`; no third-party dependencies of its
  own), included in `all`, in `make install-dev`, in `tox`'s `extras`, and in the
  CI install list.
- `fsm_llm_harness` added to the mypy target list, the coverage target list, the
  ruff `known-first-party` list, `package-data`, and `MANIFEST.in`.
- **`tests/test_packaging.py`** — derives the package list from the filesystem
  (`src/*/__init__.py`) and asserts every package appears in all 14 build/CI slots,
  so a future package that misses a slot fails loudly instead of silently. The one
  pre-existing gap (`fsm_llm_monitor` in `MANIFEST.in`) is a named, ratcheted
  exception rather than a weakened assertion.

### Changed — `fsm_llm_agents`

Both changes are gated so they are provably inert off Ollama; all pre-existing
`test_native_fc.py` tests pass unmodified in substance.

- **`NativeFunctionCallingReactAgent`** now applies `apply_ollama_params` /
  `prepare_ollama_messages` behind an `is_ollama_model` gate, recovers content from
  the reasoning trace only when there are no `tool_calls`, absorbs a malformed
  tool-call turn as a failed TURN (the loop breaks; the trace, the bytes already
  written and any answer survive) instead of losing the whole run, and gained an
  optional `system_policy` appended to its system prompt.
- **`AgentResult.success` is honest for `native_fc`** — it now requires a final,
  tool-call-free answer AND a loop that did not exhaust `max_iterations`. It
  previously returned `True` for a run that called tools, produced nothing and
  concluded nothing, which made it useless as a caller's failure signal.
- **`NativeFunctionCallingReactAgent` structured output** — when
  `AgentConfig.output_schema` is set and the free-text answer does not validate,
  exactly ONE additional completion is made carrying `response_format=` and **no**
  `tools=`. The two are never stacked in one call. Previously `output_schema` was
  silently inert on this agent's `run()` path.
- **`base._output_response_format(schema)`** extracted from `_init_context` and
  shared with the above, so there is one response-format envelope builder rather
  than two that can drift.

## [0.5.0] - 2026-07-21

### Added — agent layer additive improvements (`fsm_llm_agents`)
All of the following are **additive and backward-compatible**: existing agents,
examples, signatures, and the 2-pass core contract are unchanged. New optional
`AgentConfig` fields default to prior behavior.

- **`ToolRegistry.get_json_schemas()`** — OpenAI-compatible function-calling
  tool schemas (closes a documented-but-missing method gap).
- **`CachingToolRegistry` / `RetryingToolRegistry`** — drop-in `ToolRegistry`
  subclasses adding result memoization and retry-on-failure.
- **`AgentConfig`** new optional fields: `max_history_size`, `enable_prompt_cache`
  (litellm response caching), `reflect_every_n`, `auto_summarize_after`,
  `verification_fn`.
- **`SelfConsistencyAgent(max_workers=...)`** — opt-in parallel sampling
  (default 1 = unchanged serial; results assembled in order, deterministic).
- **`SemanticMemoryStore` + `create_semantic_memory_tools`** — embedding-backed
  long-term memory with cosine recall, JSON persistence across sessions, and an
  offline substring fallback.
- **`AutoMemoryReactAgent`** (+ `augment_task_with_memories`,
  `remember_interaction`) — automatic recall-before / remember-after at the
  `run()` boundary, removing the model-must-call-the-tool dependency.
- **`MemorySessionStore` + `save_working_memory` / `load_working_memory`** —
  persist `WorkingMemory` alongside FSM session state.
- **`BaseAgent._standard_run_stream` + `ReactAgent.run_stream`** — stream the
  final answer token by token via `API.converse_stream`.
- **`ParallelReactAgent`** — ReAct variant that extracts and dispatches multiple
  tool calls per step concurrently.
- **`VerifiedReactAgent`** — verify-and-retry via `config.verification_fn` plus
  periodic self-reflection via `config.reflect_every_n`.
- **`make_observation_summarizer`** — condenses old observations instead of
  hard-dropping them (wired in when `config.auto_summarize_after` is set).
- **`react_worker_factory` + `default_llm_judge`** — composition helpers
  (Orchestrator+ReAct worker; built-in LLM-as-judge `evaluation_fn`).
- **`NativeFunctionCallingReactAgent`** — self-contained ReAct loop using
  provider-native `tools=`/`tool_calls` (litellm) instead of JSON-in-prompt.

## [0.4.0] - 2026-05-29

### Security
- **litellm supply chain compromise**: litellm versions 1.82.7 and 1.82.8 were compromised
  with credential-stealing malware via `.pth` file injection. These versions are now
  explicitly excluded from the dependency specification (`!=1.82.7,!=1.82.8`).
  - **Impact**: Any Python invocation in an environment with the compromised versions would
    exfiltrate environment variables, SSH keys, AWS credentials, Kubernetes configs, and git
    credentials to an attacker-controlled server. No import of litellm was required.
  - **Action for users**: If you installed litellm 1.82.7 or 1.82.8 at any time, treat all
    credentials in that environment as compromised and rotate them immediately.
  - **Current status**: PyPI has quarantined the entire litellm package. Existing installs of
    safe versions (<=1.82.6) continue to work.
- Added `.pth` file audit in CI pipeline and local `make audit` / `scripts/audit_pth.py`
- Added `constraints.txt` for dependency version locking in dev/CI builds

### Changed
- **Skip Pass 2 for intermediate agent states** — States with `response_instructions=""` now skip
  the response generation LLM call entirely. The pipeline sends a minimal sentinel to the LLM
  interface (for cycle tracking) and the real LLM returns immediately without an API call. This
  halves the number of LLM calls for agent iterations, cutting wall time ~50% and eliminating
  all F-LOOP timeout failures. Applied to: think/act (ReAct, Reflexion), evaluate (EvalOpt),
  check (MakerChecker).
- **Stall detection threshold** reduced from 3 to 2 consecutive no-tool iterations before
  forced termination, saving ~20s per stall event.

### Fixed
- **MakerChecker quality_score extraction** — When the LLM embeds quality_score inside the
  checker_feedback dict instead of as a separate context field, `_track_revisions` now recovers
  it from the dict. Previously quality_score defaulted to 0.0, forcing max revisions.
- Evaluation health score improved from 95.7% to **100%** (70/70 PASS) on `ollama_chat/qwen3.5:4b`.
- **Comprehensive code-review hardening** — four deep static-review passes across all five packages
  fixed ~75 issues: 2-pass / locking / streaming-rollback bugs in core; handler-timing and
  budget/iteration-limiter bugs across the 12 agent patterns; async workflow-engine event/timeout/retry
  races; meta-builder, reasoning, and monitor fixes; sibling-class propagation of budget/timeout re-raise
  guards; and recursion-safety (recursive→iterative DFS) in workflow, agent-graph, and FSM-validator
  cycle detection. Tracked via the `*-NEW-*`, `AI3-*`, `RW3-*`, `AG-*`, `RWM-*`, and `FA-*` issue IDs in
  the git history.

### Added
- **BaseAgent ABC** for all 12 agent implementations — shared conversation loop, budget enforcement,
  answer extraction, trace building, context filtering, and `__call__` syntax (`agent("task")`)
- **Enhanced `@tool` decorator** — supports bare `@tool` (no parentheses) with auto-schema inference
  from type hints (`str→string`, `int→integer`, `float→number`, `bool→boolean`, `list→array`, `dict→object`).
  Supports `typing.Annotated[T, "description"]` for per-parameter descriptions. Backward compatible
  with explicit `parameter_schema` overrides.
- **Structured output** — `AgentConfig(output_schema=PydanticModel)` validates agent answers against
  Pydantic models. Parsed result stored in `AgentResult.structured_output`. Graceful fallback on
  validation failure.
- **`create_agent()` factory** — create agents in one line: `create_agent(tools=[search], pattern="react")`
- **`ToolRegistry.register_agent()`** — register agents as tools for supervisor/orchestrator patterns
- **`AgentResult.__str__`** — returns structured_output if available, else raw answer
- `ollama.py` module — centralized Ollama helpers for structured output compatibility
  - `is_ollama_model()` — model detection
  - `apply_ollama_params()` — disables thinking via `reasoning_effort="none"`, forces `temperature=0` for structured calls
  - `build_ollama_response_format()` — builds `json_schema` response format with extraction/transition schemas
  - `EXTRACTION_JSON_SCHEMA`, `TRANSITION_JSON_SCHEMA` — JSON Schema constants for structured output
- `fsm_llm_agents` extension package for ReAct and Human-in-the-Loop agentic patterns
  - `ReactAgent` — ReAct loop agent with auto-generated FSM from tool registry (think → act → observe → conclude)
  - `ToolRegistry` — tool management with schema descriptions, prompt generation, and execution
  - `HumanInTheLoop` — configurable approval gates, confidence-based escalation, and human override
  - `@tool` decorator for simple tool registration
  - Pydantic models: `ToolDefinition`, `ToolCall`, `ToolResult`, `AgentStep`, `AgentTrace`, `AgentConfig`, `AgentResult`, `ApprovalRequest`
  - `AgentError` exception hierarchy (7 error types)
  - 109 unit tests across 8 test files
- `has_agents()` / `get_agents()` extension checks in `fsm_llm`
- `MessagePipeline` class extracted from FSMManager — encapsulates all 2-pass message processing
- `context.py` module extracted from FSMManager — stateless context cleaning utilities
- `ConversationStep` added to workflows — embeds full FSM conversations within workflow steps
- Handler execution timeout support (`DEFAULT_HANDLER_TIMEOUT = 30s`)
- Workflow step async timeout support (`DEFAULT_STEP_TIMEOUT = 120s`)
- Workflow-level timeout, conversation timeout, and event listener expiration
- `critical` flag on `BaseHandler` — errors always raise regardless of error_mode
- `FORBIDDEN_CONTEXT_PATTERNS` enforcement for password/secret/token key filtering
- 5 new examples combining sub-packages (reasoning, workflows, classification)
- 20 new complex examples (70 total) focused on agentic patterns and meta builders:
  - **Agents (14)**: debate_with_tools (evidence-based debate), reflexion_code_gen (self-improving code
    generation with test runner), orchestrator_specialist (multi-specialist ReactAgents), pipeline_review
    (PromptChain + MakerChecker QA), adapt_with_memory (ADaPT + WorkingMemory), rewoo_multi_step (complex
    multi-dependency planning), eval_opt_structured (EvaluatorOptimizer + Pydantic validation),
    plan_execute_recovery (replanning on tool failure), consistency_with_tools (SelfConsistency for
    multi-step reasoning), maker_checker_code (code review pattern), hierarchical_orchestrator (nested
    multi-level delegation), agent_memory_chain (multi-task continuity via WorkingMemory),
    react_structured_pipeline (ReAct → structured output → PromptChain), multi_debate_panel (parallel
    debates with synthesis)
  - **Meta (4)**: build_workflow (interactive workflow builder), build_agent (interactive agent builder),
    meta_review_loop (FSMBuilder + MakerChecker quality review), meta_from_spec (programmatic
    FSM/workflow/agent from text specs)
  - **Workflows (2)**: conditional_branching (condition-based routing), workflow_agent_loop (quality-gated
    agent execution with retry)
- Automated evaluation baseline: 95.7% health score (70 examples, ollama_chat/qwen3.5:4b)
- Tests for MessagePipeline, handler timeout, step timeout, context, logging, runner, LiteLLMInterface
- Audit verification tests across all packages

### Changed
- Ollama structured output uses `json_schema` response format instead of `json_object` for grammar-constrained output
- Ollama thinking mode disabled via `reasoning_effort="none"` (litellm >=1.82 maps this to Ollama's `think: false`)
- Ollama structured calls (data extraction, transition decision) force `temperature=0` for deterministic output
- Classification `Classifier._call_llm()` now applies Ollama params via shared `fsm_llm.ollama` helpers
- Minimum litellm version bumped from 1.68.1 to 1.82.0 (required for proper Ollama `think` parameter forwarding)
- FSMManager delegates message processing to MessagePipeline
- `push_fsm`/`pop_fsm` decomposed into focused sub-methods
- `evaluate_logic()` refactored with dispatch pattern
- Runner refactored to use API; workflows drops phantom FSMManager dependency
- Exception handling standardized across codebase (chaining with `from e`)
- Regex patterns pre-compiled for performance
- Test fixtures deduplicated across test suites
- mypy enforcement enabled in CI with pydantic plugin
- All 118 mypy errors fixed

### Removed
- `fsm_llm_classification` deprecation shim package (use `from fsm_llm import Classifier` directly)
- `LLMInterface.decide_transition()` deprecated method
- `LLMInterface.extract_data()` deprecated method
- `FSMManager` `transition_prompt_builder` parameter
- `WorkflowEngine` `fsm_manager` and `llm_interface` parameters
- `DataExtractionRequest` class
- `State._coerce_and_warn()` boolean coercion for `transition_classification`
- `State` `instructions` field deprecation warning
- `has_classification()` and `get_classification()` helper functions
- Empty `fsm_llm_workflows.handlers` compatibility shim
- 7 forwarding methods from FSMManager (moved to MessagePipeline)
- Dead workflow handler code (AutoTransitionHandler, EventHandler, TimerHandler)
- Dead code and empty extras across multiple packages

### Fixed
- Ollama/Qwen3 thinking mode corrupting structured JSON output (ollama/ollama#10538)
- Integration test `test_pre_processing_handler_fires` using wrong HandlerBuilder API (`.on_timing()` → `.at()`, `.execute().build()` → `.do()`)
- Race condition in conversation lock retrieval
- Conversation lock leak with cleanup methods
- Event listener race condition in workflows
- Confidence collapse with additive boost in classification
- MockLLM2Interface crash on empty transitions
- Classifier thinking hacks and multi-intent prompt mismatch
- Classification confidence handling and dead code
- Workflow step error paths and type safety
- Workflow engine safety issues
- Security gaps in handlers, context, and prompts
- JSON regex fallback validation (requires meaningful keys)
- Multi-key JsonLogic expression error
- Reasoning engine bugs and magic number extraction
- Algorithm and logic issues across codebase

### Security
- Safety limits and validation guards added
- Security gaps fixed in handlers, context, and prompts
- Context key security filtering (internal prefixes, forbidden patterns)

## [0.3.0] - 2026-03-19

### Added
- `fsm_llm_classification` extension package for LLM-backed structured classification
  - `Classifier` for single-intent and multi-intent classification
  - `HierarchicalClassifier` for two-stage domain-then-intent classification (>15 classes)
  - `IntentRouter` for mapping classified intents to handler functions with low-confidence fallback
  - Pydantic models: `ClassificationSchema`, `IntentDefinition`, `ClassificationResult`, `MultiClassificationResult`, `HierarchicalSchema`
  - Prompt and JSON schema builders with reasoning-first field ordering (mitigates constrained-decoding distortion)
  - Structured output support via `response_format` when the LLM provider supports it
- `has_classification()` / `get_classification()` extension checks in `fsm_llm`
- 39 unit tests for classification package
- Classification extension documentation (README, examples, architecture docs)
- `timeout` parameter on `LiteLLMInterface` (default 120s) to prevent indefinite hangs on network issues
- `pytest-mock` added to dev extras in pyproject.toml
- `[tool.ruff.lint]` configuration in pyproject.toml to suppress false E402 from `__future__` annotations
- 21 regression tests for codebase review fixes (`test_regression_review.py`)
- 15 new `ContextKeys` constants for reasoning sub-FSM result keys (deductive, inductive, abductive, analogical, critical, hybrid)

### Fixed
- Version number aligned to 0.3.0 across `pyproject.toml` and `__version__.py` (was still 0.2.1)
- Context pruning log now reports actual new size instead of repeating the original size
- Hard-coded context keys in `merge_reasoning_results` replaced with `ContextKeys` constants (prevents silent `None` on key mismatch)
- Duplicate `import re` removed from `llm.py` `_make_llm_call()` (leftover from Qwen3.5 workaround)
- Extraction parse failure now returns `confidence=0.0` instead of `0.5` (callers can distinguish failure from low-confidence extraction)
- `requirements.txt` aligned with `pyproject.toml` core deps (removed dev deps, fixed `python-dotenv` version pin)

### Removed
- Unused async handler support from `handlers.py` (asyncio import, `AsyncExecutionLambda` type, `is_async` detection, ThreadPoolExecutor fallback) — no async handlers existed in the codebase
- `MergeStrategy` alias from `reasoning/constants.py` — engine now imports `ContextMergeStrategy` directly
- Dynamic `__all__.extend()` / `__all__.append()` calls from `__init__.py` — consolidated into single `__all__` definition
- Dead `[testenv:docs]` sphinx environment from `tox.ini`

## [0.2.1] - 2026-03-19

### Added
- `[tool.pytest.ini_options]` in pyproject.toml
- `[tool.mypy]` configuration in pyproject.toml
- Python 3.12 support in CI and tox
- CHANGELOG.md (this file)
- examples/README.md with example index and learning path

### Changed
- Python minimum version updated to 3.10 (was 3.8)
- Package-data now includes `fsm_llm_reasoning`
- Pre-commit hooks replaced: pytest-on-commit removed, ruff + standard hooks added
- Makefile expanded from 3 to 8 targets (added help, lint, format, type-check, install-dev)
- CI workflow installs from pyproject.toml instead of requirements.txt
- tox.ini aligned with CI (consistent flake8 config, added mypy env)

### Fixed
- CLI entry point now correctly resolves `fsm-llm` command
- Exception chaining (`from e`) added to all catch-and-reraise blocks for proper traceback preservation
- `__main__.py` docstring placement (was after imports, not recognized by Python)
- Workflows package version now imported from main package instead of hardcoded
- LLM interface log levels demoted from INFO to DEBUG (less noisy)
- Input validation added to `LiteLLMInterface` (model, temperature, max_tokens)

## [0.2.0] - 2026-03-18

### Changed
- Project renamed from `llm-fsm` to `fsm-llm` across all packages, tests, docs, and examples

## [0.1.0] - 2026-03-07

### Added
- Initial release with 2-pass architecture
- FSM stacking with push/pop operations
- Handler system with builder pattern
- JsonLogic expression evaluator
- LiteLLM multi-provider support
- CLI tools: fsm-llm, fsm-llm-visualize, fsm-llm-validate
- 7 examples (basic, intermediate, advanced)
- Comprehensive documentation

[0.11.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/NikolasMarkou/fsm_llm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NikolasMarkou/fsm_llm/releases/tag/v0.1.0
