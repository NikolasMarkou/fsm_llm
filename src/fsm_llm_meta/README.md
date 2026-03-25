# fsm_llm_meta

Interactive conversational agent that helps users build FSMs, workflows, and agent configurations through guided dialogue. The meta-agent asks adaptive questions, incrementally validates the artifact, and outputs a ready-to-use JSON definition. Part of the [fsm-llm](https://github.com/NikolasMarkou/fsm_llm) framework (v0.3.0).

## Features

- **3 Artifact Types** -- Build FSM definitions, workflow configurations, or agent setups through a single unified conversation
- **Guided 7-State Conversation** -- A purpose-built FSM drives the dialogue from welcome through classification, overview gathering, structure design, connection wiring, review, and final output
- **Incremental Validation** -- Each builder validates progressively; partial validation surfaces warnings during construction, complete validation catches errors before output
- **Adaptive Detail Levels** -- Builder summaries injected into LLM context vary by phase: minimal during design, standard during connections, full during review
- **CLI and Programmatic API** -- Interactive terminal mode via `fsm-llm-meta`, or turn-by-turn control with `start()` / `send()` / `is_complete()` / `get_result()`

## Installation

```bash
pip install fsm-llm[meta]
```

## Quick Start

### Interactive CLI

```bash
fsm-llm-meta
fsm-llm-meta --model gpt-4o-mini --output my_fsm.json
```

### Programmatic (Turn-by-Turn)

```python
from fsm_llm_meta import MetaAgent

agent = MetaAgent()
response = agent.start()
print(response)

while not agent.is_complete():
    user_input = input("> ")
    response = agent.send(user_input)
    print(response)

result = agent.get_result()
print(result.artifact_json)    # Valid JSON definition
print(result.is_valid)         # True if validation passed
print(result.artifact_type)    # "fsm", "workflow", or "agent"
```

### Interactive REPL

```python
from fsm_llm_meta import MetaAgent

agent = MetaAgent()
result = agent.run_interactive()  # Full REPL loop with stdin prompts
print(result.artifact_json)
```

### With Custom Configuration

```python
from fsm_llm_meta import MetaAgent, MetaAgentConfig

config = MetaAgentConfig(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=2000,
    max_turns=50,
)
agent = MetaAgent(config=config)
result = agent.run_interactive()
```

## How It Works

The meta-agent is itself an FSM with 7 states:

```
WELCOME --> CLASSIFY --> GATHER_OVERVIEW --> DESIGN_STRUCTURE --> DEFINE_CONNECTIONS --> REVIEW --> OUTPUT
                                                   ^                                     |
                                                   |_______________ (revise) ____________|
```

### State Purposes

| State | Purpose |
|-------|---------|
| **WELCOME** | Greets the user, explains the three artifact types |
| **CLASSIFY** | Extracts which artifact type the user wants (FSM, Workflow, or Agent) |
| **GATHER_OVERVIEW** | Collects name, description, and persona (FSM only) |
| **DESIGN_STRUCTURE** | Iteratively adds components -- states for FSMs, steps for workflows, tools for agents |
| **DEFINE_CONNECTIONS** | Wires components together -- transitions, step links, or tool configurations |
| **REVIEW** | Runs full validation, presents summary with errors/warnings, accepts approval or revision |
| **OUTPUT** | Serializes the validated artifact to JSON (terminal state) |

Static states (WELCOME, CLASSIFY, OUTPUT) use fixed prompt instructions. Dynamic states (GATHER_OVERVIEW, DESIGN_STRUCTURE, DEFINE_CONNECTIONS, REVIEW) receive context through builder-state injection via handlers.

## Builders

Three concrete builders inherit from `ArtifactBuilder`. Each provides mutation methods, incremental validation, progress tracking, and summary generation.

### FSMBuilder

Builds FSM definitions compatible with `FSMDefinition` (v4.1).

| Method | Description |
|--------|-------------|
| `set_overview(name, description, persona=None)` | Set FSM metadata |
| `add_state(state_id, description, purpose, extraction_instructions=None, response_instructions=None)` | Add a state; auto-sets initial state on first call |
| `remove_state(state_id)` | Remove a state and clean up referencing transitions |
| `update_state(state_id, **fields)` | Update allowed fields: description, purpose, extraction/response instructions |
| `add_transition(from_state, target_state, description, priority=100, conditions=None)` | Add a transition between existing states |
| `remove_transition(from_state, target_state)` | Remove a transition |
| `set_initial_state(state_id)` | Set which state starts the FSM |

### WorkflowBuilder

Builds workflow definitions with typed steps.

| Method | Description |
|--------|-------------|
| `set_overview(workflow_id, name, description)` | Set workflow metadata |
| `add_step(step_id, step_type, name, description="", config=None)` | Add a step; valid types: auto_transition, api_call, condition, llm_processing, wait_for_event, timer, parallel, conversation |
| `remove_step(step_id)` | Remove a step and clean up transitions |
| `set_step_transition(from_step, to_step, condition=None)` | Connect two steps |
| `set_initial_step(step_id)` | Set the starting step |

### AgentBuilder

Builds agent configurations with tools.

| Method | Description |
|--------|-------------|
| `set_overview(name, description)` | Set agent metadata |
| `set_agent_type(agent_type)` | Set the pattern type; valid: react, plan_execute, reflexion, rewoo, evaluator_optimizer, maker_checker, prompt_chain, self_consistency, debate, orchestrator, adapt |
| `set_config(**kwargs)` | Update config fields: model, max_iterations, timeout_seconds, temperature, max_tokens |
| `add_tool(name, description, parameter_schema=None)` | Add a tool definition |
| `remove_tool(name)` | Remove a tool |

### Common ArtifactBuilder Interface

All builders implement these abstract methods:

| Method | Description |
|--------|-------------|
| `to_dict()` | Serialize to dict |
| `validate_partial()` | Return warnings about current partial state |
| `validate_complete()` | Return errors that would prevent a valid artifact |
| `get_progress()` | Return `BuildProgress` with completion percentage and missing fields |
| `get_missing_fields()` | Return list of required things still missing |
| `get_summary(detail_level="full")` | Human-readable summary; levels: "minimal", "standard", "full" |

## Action System

During the DESIGN_STRUCTURE and DEFINE_CONNECTIONS states, the LLM extracts actions from user input as JSON:

```json
{"action": "add_state", "action_params": {"state_id": "greeting", "description": "...", "purpose": "..."}}
```

The `MetaHandlers.dispatch_action()` handler (registered at POST_PROCESSING) routes each action to the appropriate builder method. Available actions depend on the artifact type:

| Artifact | Structure Actions | Connection Actions |
|----------|------------------|--------------------|
| FSM | `add_state`, `remove_state`, `update_state`, `set_initial_state` | `add_transition`, `remove_transition` |
| Workflow | `add_step`, `remove_step`, `set_initial_step` | `set_step_transition` |
| Agent | `set_agent_type`, `set_config`, `add_tool`, `remove_tool` | -- |

The `done` action signals the end of a phase, setting `structure_done` or `connections_done` in context to trigger the transition to the next state.

## Configuration

`MetaAgentConfig` is a Pydantic model with these fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `DEFAULT_LLM_MODEL` | LLM model identifier (any litellm-compatible model) |
| `temperature` | `float` | `0.7` | LLM temperature |
| `max_tokens` | `int` | `2000` | Max tokens per LLM response |
| `max_turns` | `int` | `50` | Maximum conversation turns before raising `MetaAgentError` |

## CLI Reference

```bash
fsm-llm-meta [OPTIONS]
python -m fsm_llm_meta [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | LLM model to use | From config |
| `--output`, `-o` | Output file path for artifact JSON | Print to stdout |
| `--temperature` | LLM temperature | 0.7 |
| `--max-turns` | Maximum conversation turns | 50 |

## API Reference

### MetaAgent

| Method | Description |
|--------|-------------|
| `MetaAgent(config=None, **api_kwargs)` | Initialize; `api_kwargs` are forwarded to `API.from_definition()` |
| `start(initial_message="")` | Start conversation, returns first agent response |
| `send(message)` | Send user message, returns agent response; raises `MetaAgentError` if max turns exceeded |
| `is_complete()` | Returns `True` when the artifact has been fully built and approved |
| `get_result()` | Returns `MetaAgentResult`; raises `MetaAgentError` if not complete |
| `run_interactive()` | Full interactive REPL loop reading from stdin; returns `MetaAgentResult` |

### Models

| Class | Key Fields |
|-------|------------|
| `ArtifactType` | Enum: `FSM`, `WORKFLOW`, `AGENT` |
| `BuildProgress` | `total_required`, `completed`, `missing`, `warnings`; properties: `percentage`, `is_complete` |
| `MetaAgentConfig` | `model`, `temperature`, `max_tokens`, `max_turns` |
| `MetaAgentResult` | `artifact_type`, `artifact`, `artifact_json`, `is_valid`, `validation_errors`, `conversation_turns` |

## Output Utilities

| Function | Description |
|----------|-------------|
| `format_artifact_json(artifact)` | Pretty-print an artifact dict as indented JSON |
| `format_summary(result)` | Human-readable summary of a `MetaAgentResult` (type, name, validity, turn count, errors) |
| `save_artifact(artifact, path)` | Write artifact dict to a JSON file; creates parent dirs; returns resolved `Path`; raises `OutputError` on failure |

## File Map

| File | Purpose |
|------|---------|
| `agent.py` | `MetaAgent` -- conversation management, handler registration, result building |
| `builders.py` | `ArtifactBuilder` (abstract), `FSMBuilder`, `WorkflowBuilder`, `AgentBuilder` |
| `fsm_definitions.py` | `build_meta_fsm()` -- generates the 7-state meta-agent FSM definition dict |
| `handlers.py` | `MetaHandlers` -- builder injection, action dispatch, overview handling, validation, finalization |
| `prompts.py` | Prompt builders for static states (welcome, classify, output) |
| `definitions.py` | Pydantic models: `ArtifactType`, `BuildProgress`, `MetaAgentConfig`, `MetaAgentResult` |
| `constants.py` | `MetaStates`, `ContextKeys`, `Actions`, `HandlerNames`, `Defaults`, `ErrorMessages`, `LogMessages` |
| `exceptions.py` | `MetaAgentError`, `BuilderError`, `MetaValidationError`, `OutputError` |
| `output.py` | `format_artifact_json()`, `format_summary()`, `save_artifact()` |
| `__main__.py` | CLI entry point: `fsm-llm-meta` / `python -m fsm_llm_meta` |
| `__version__.py` | Package version (synced from `fsm_llm.__version__`) |
| `__init__.py` | Public API exports -- single `__all__` list (16 items) |

## Examples

See [`examples/meta/build_fsm/`](../../examples/meta/build_fsm/) for a complete runnable example.

## Development

```bash
pytest tests/test_fsm_llm_meta/  # 129 tests across 5 test files
```

Test files: `test_builders.py`, `test_handlers.py`, `test_definitions.py`, `test_fsm_definitions.py`, `test_agent.py`.

## License

GNU General Public License v3.0. See [LICENSE](../../LICENSE).
