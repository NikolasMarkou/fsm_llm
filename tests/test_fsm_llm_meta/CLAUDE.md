# test_fsm_llm_meta

Path: `tests/test_fsm_llm_meta`
Purpose: Pytest suite (218 tests) for the meta-builder in `fsm_llm.agents`: builders, builder tools, prompts, definitions, output helpers, and `MetaBuilderAgent`.

## Scope

Covers these source modules (all under `src/fsm_llm/agents/`):

- `meta_builder.py` (`MetaBuilderAgent`)
- `meta_builders.py` (`FSMBuilder`, `WorkflowBuilder`, `AgentBuilder`)
- `meta_tools.py` (`create_fsm_tools`, `create_workflow_tools`, `create_agent_tools`, `create_builder_tools`)
- `meta_prompts.py` (`build_welcome_message`, `build_followup_message`, `build_review_presentation`, `build_output_message`)
- `meta_output.py` (`format_artifact_json`, `format_summary`, `save_artifact`)
- `meta_fsm.py` (`build_meta_builder_fsm`, legacy FSM dict)
- `definitions.py` (`ArtifactType`, `BuildProgress`, `MetaBuilderConfig`, `MetaBuilderResult`, `ToolCall`)
- `exceptions.py` (`MetaBuilderError`, `BuilderError`, `MetaValidationError`, `OutputError`, `AgentError`)
- `constants.py` (`MetaDefaults`)

Not covered here: `meta_cli.py` (the `fsm-llm-meta` console script), other agent patterns, core FSM runtime (only `fsm_llm.definitions.FSMDefinition` is used to check that `FSMBuilder.to_dict()` output is valid).

## Architecture

Test layers, from pure to LLM-adjacent:

1. Pure builder tests (`test_builders.py`, `test_builders_elaborate.py`): call builder methods, assert on `states`/`steps`/`tools`, returned warning lists, raised `BuilderError`, `to_dict()`, `validate_complete()`, `get_summary(level)`.
2. Tool tests (`test_tools.py`, part of `test_integration.py`): build a `ToolRegistry` over a builder, call `registry.execute(ToolCall(...))`, assert the builder was mutated in place.
3. Agent tests (`test_agent.py`, part of `test_integration.py`): construct `MetaBuilderAgent`, often set private fields directly (`_started`, `_complete`, `_turn_count`, `_artifact_type`, `_builder`, `_messages`) and call private methods.

LLM isolation methods used:

| Method | Where |
| --- | --- |
| `monkeypatch.setattr("litellm.completion", ...)` raising `RuntimeError` or returning a fake response | `TestLlmCallProviderFailure`, `TestWorkflowStepTypeEnum` |
| `agent._llm_call = lambda *a, **k: ...` | `TestSchemaEchoRejection.test_pipeline_raises_on_schema_echo` |
| `offline_llm` fixture (conftest): `litellm.completion` and `fsm_llm.classification.completion` raise `RuntimeError("offline")`, so the fallback path is asserted with no network call | `TestTypeDetection`, `TestStartSendFlow` (`_detect_type` falls back to keyword matching; `_generate_collect_response` falls back to canned text) |

## Key files

| File | Role | Notes |
| --- | --- | --- |
| `conftest.py` | Fixtures | `fsm_builder`, `workflow_builder`, `agent_builder`, `populated_fsm_builder`, `meta_config`, `offline_llm` |
| `test_agent.py` | `MetaBuilderAgent` behavior | 44 tests; carries DECISION references |
| `test_builders.py` | Builder core behavior | 71 tests |
| `test_builders_elaborate.py` | Builder edge cases, exceptions, config validators | 44 tests |
| `test_tools.py` | Tool registries | 26 tests; `_make_call(tool_name, **kwargs)` helper at file bottom |
| `test_integration.py` | Cross-piece checks | 11 tests |
| `test_definitions.py` | Pydantic models and enum | 12 tests |
| `test_prompts.py` | Prompt text builders | 8 tests |
| `test_handlers.py`, `test_handlers_elaborate.py` | Import smoke test | 1 test each; handlers module was removed, only the `fsm_llm.agents.meta_builder` import is checked |
| `__init__.py` | Package marker | empty |

## Public interface

A test package exports no runtime API. What other tests here reuse:

Fixtures in `conftest.py` (all function-scoped):

| Fixture | Returns |
| --- | --- |
| `fsm_builder` | `FSMBuilder()` (empty) |
| `workflow_builder` | `WorkflowBuilder()` (empty) |
| `agent_builder` | `AgentBuilder()` (empty) |
| `populated_fsm_builder` | `FSMBuilder` with 3 states and 2 transitions (shape below) |
| `meta_config` | `MetaBuilderConfig` with test values (shape below) |
| `offline_llm` | `None`; monkeypatches `litellm.completion` and the classifier's import-bound `fsm_llm.classification.completion` to raise at once. Apply with `@pytest.mark.usefixtures("offline_llm")` |

Helper in `test_tools.py`:

- `_make_call(tool_name: str, **kwargs) -> ToolCall`: returns `ToolCall(tool_name=tool_name, parameters=kwargs)` for `registry.execute(...)`.

## Data shapes

- `populated_fsm_builder`: `set_overview("GreetingBot", "A simple greeting bot", persona="Friendly assistant")`; states added as `add_state(state_id, description, purpose, ...)`: `greeting`, `ask_name` (with `extraction_instructions` and `response_instructions`), `farewell`; transitions `greeting -> ask_name` and `ask_name -> farewell`, the second with `conditions=[{"description": "Name is set", "logic": {"has_context": "user_name"}}]`. `validate_complete()` returns `[]`. Many tests assert exact strings from it ("GreetingBot", "States (3)", "-> ask_name", "Persona:").
- `meta_config`: `MetaBuilderConfig(model="gpt-4o-mini", temperature=0.5, max_tokens=1000, max_turns=20)`. No test in the directory requests this fixture at present (`agent.meta_config` in `test_agent.py` is the agent attribute, not the fixture).
- `ToolCall` (from `fsm_llm.agents.definitions`): `tool_name` plus a `parameters` dict. Tool results are read through `result.success` and `result.result`.

## Invariants and constraints

Behavior pinned by the tests:


Builders:
- First `add_state`/`add_step` auto-sets the initial state/step and returns an "Auto-set" warning (FSM). Duplicate state id overwrites with an "already exists" warning. Empty id raises `BuilderError` matching "empty".
- `remove_state` clears `initial_state` if it was removed and strips transitions pointing at it; returns `False` for unknown ids. `remove_step` also strips transitions.
- `add_transition` raises `BuilderError` "not found" if source or target is missing; default priority is `MetaDefaults.DEFAULT_PRIORITY` (100). Self-transitions allowed.
- `update_state`: unknown fields give an "Ignoring" warning; `None` is rejected with a warning and the old value kept; non-strings are converted with a warning (`42` becomes `"42"`); unknown state raises "not found".
- `FSMBuilder.to_dict()` sets `version == "4.1"` and must load as `FSMDefinition(**d)`.
- `validate_complete()` reports missing name, unreachable (orphan) states, and transitions to nonexistent states.
- `get_summary(level)` levels: `minimal` < `standard` <= `full`; default equals `full`; only `full` includes `extraction:`; `minimal` on an empty FSM builder contains "none yet". Headers: "FSM Builder Status", "Workflow Builder Status", "Agent Builder Status".
- `WorkflowBuilder.VALID_STEP_TYPES` is a class-level set of exactly 8: `auto_transition`, `api_call`, `condition`, `llm_processing`, `wait_for_event`, `timer`, `parallel`, `conversation`. Unknown type warns on `add_step` ("Unknown step type") and is an error in `validate_complete()`.
- `AgentBuilder.set_agent_type` normalizes (`"  REACT  "` -> `"react"`), raises "Unknown agent type" and leaves `agent_type` as `None` on failure. `set_config` rejects wrong types for `max_iterations`/`temperature` with warnings, accepts int temperature, warns on unknown fields. Defaults come from `MetaDefaults.AGENT_*`. Duplicate `add_tool` replaces the tool with a warning.

Tools:
- Tool errors are returned as text: `result.success` stays `True` and `"Error"` is in `result.result` (bad transition, invalid agent type).
- `create_builder_tools(builder, artifact_type)` raises `TypeError` on a builder/type mismatch.

Definitions and exceptions:
- `MetaBuilderConfig` defaults: `temperature 0.7`, `max_tokens 4096`, `max_turns 50`, `max_iterations 25`. `max_turns=0` raises "max_turns must be at least 1". Temperature must be in 0.0-2.0, `max_tokens >= 1`. `output_path` is not a field and is silently ignored.
- `BuildProgress.is_complete` is false when `total_required == 0`.
- `BuilderError.action`, `MetaValidationError.errors` (default `[]`), `OutputError.path`; all three subclass `MetaBuilderError`.

Agent:
- `send` before `start` raises "not been started"; second `start` raises "already been started"; `send` after completion raises "already completed"; reaching `max_turns` raises "Maximum turns"; `get_result` before completion raises "not complete".
- `_build_type_aliases()` orders longer aliases first ("finite state machine" before "state machine", "data pipeline" before "pipeline"). Unknown text defaults to `ArtifactType.FSM`.
- `_is_build_trigger` accepts "build it", "go", "done", "approve", "yes", "lgtm".
- `_is_schema_echo` flags JSON-schema-shaped specs; `_run_deterministic_pipeline` raises `MetaValidationError` matching "JSON schema" on an echo (DECISION plan_2026-05-30_26c9510a/D-001, marked STALE).
- `_llm_call` wraps provider exceptions in `BuilderError` chained via `__cause__`; an empty model answer still returns `""` (DECISION plan-2026-07-20T040150-876e7164/D-006, marked STALE). `_generate_collect_response` still returns text containing "Say 'build it' when ready" on provider failure.
- Workflow extraction schema: `MetaBuilderAgent._ARTIFACT_SCHEMAS["workflow"]` constrains `step_type` to `sorted(WorkflowBuilder.VALID_STEP_TYPES)`, and the enum survives into `response_format` and litellm's Ollama `format` (DECISION plan-2026-09-24T091842-c1d5bfbc/D-010).
- `_build_result()` with no builder gives `success is False`, `is_valid is False`, `final_context == {}`; with a builder, `final_context` has `artifact_json` (dict) and `artifact_type` ("fsm").
- `save_artifact` writes to the resolved path (`..` segments collapsed); it does not confine the path.
- `fsm_llm.agents` exports `MetaBuilderAgent`, the three builders, `ArtifactType`, `MetaBuilderConfig`, `MetaBuilderResult`, `MetaBuilderError`, `create_builder_tools`, `create_fsm_tools`, `__version__`.

## Dependencies

- Internal: `fsm_llm.agents` (modules listed in Scope), `fsm_llm.definitions.FSMDefinition`.
- External: `pytest`; `litellm` (monkeypatched `litellm.completion`; `litellm.llms.ollama.chat.transformation.OllamaChatConfig` imported in `TestWorkflowStepTypeEnum`).

## Failure modes

- A litellm upgrade that moves `OllamaChatConfig` or changes `map_openai_params` breaks `test_enum_reaches_response_format_and_ollama_format`.
- The type classifier calls `fsm_llm.classification.completion`, a name bound at import, so patching only `litellm.completion` leaves `_detect_type` reaching a real model (slow, and a reachable provider changes the answer). `offline_llm` patches both.
- Changing `populated_fsm_builder` breaks many exact-string assertions.

## Working here

- Run: `.venv/bin/python -m pytest tests/test_fsm_llm_meta/ -q` from the repo root.
- Use the conftest fixtures instead of building builders inline; for new tool tests use `_make_call` in `test_tools.py`.
- Mock the LLM at `litellm.completion` or `agent._llm_call`; do not add tests that need a live provider.
- Read the DECISION notes in `TestSchemaEchoRejection`, `TestLlmCallProviderFailure`, `TestWorkflowStepTypeEnum` before changing the behavior they pin.
- Adding or removing tests changes the suite count (218) stated in this file and the README here. Re-measure with `.venv/bin/python -m pytest tests/test_fsm_llm_meta --collect-only -q | tail -1`.
