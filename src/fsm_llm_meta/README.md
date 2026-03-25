# FSM-LLM Meta-Agent — Interactive Artifact Builder

An interactive conversational agent that builds FSMs, Workflows, and Agents through adaptive questioning. Asks the user questions until the artifact is fully specified, validated, and ready to use. Part of the `fsm-llm` package.

```bash
pip install fsm-llm[meta]
```

---

## Features

- **3 Artifact Types**: Build FSM definitions, workflow definitions, or agent configurations interactively
- **Adaptive Conversation**: The agent asks targeted questions based on what's already been specified and what's still missing
- **Incremental Validation**: Validates the artifact at each step, surfacing errors and warnings as you build
- **Progress Tracking**: `BuildProgress` model tracks completion percentage and missing fields
- **Turn-by-Turn API**: Programmatic `start()` / `send()` / `is_complete()` / `get_result()` interface
- **Interactive CLI**: `fsm-llm-meta` command for terminal-based artifact building
- **Output Utilities**: Format, summarize, and save generated artifacts

## Quick Start

### Turn-by-Turn API

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

### Interactive CLI

```python
from fsm_llm_meta import MetaAgent

agent = MetaAgent()
result = agent.run_interactive()  # Full REPL loop with prompts
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

## CLI

```bash
# Interactive artifact builder
fsm-llm-meta

# Or with Python module syntax
python -m fsm_llm_meta

# With options
fsm-llm-meta --model gpt-4o-mini --temperature 0.5 --max-turns 30

# Save output to file
fsm-llm-meta --output my_fsm.json
```

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | LLM model to use | From config |
| `--output`, `-o` | Output file path for artifact JSON | Print to stdout |
| `--temperature` | LLM temperature | 0.7 |
| `--max-turns` | Maximum conversation turns | 50 |

## Architecture

The meta-agent uses a 7-state FSM to drive the conversation:

```
WELCOME → CLASSIFY → GATHER_OVERVIEW → DESIGN_STRUCTURE → DEFINE_CONNECTIONS → REVIEW → OUTPUT
```

1. **WELCOME** — Greets the user, explains what can be built
2. **CLASSIFY** — Determines which artifact type (FSM, Workflow, or Agent) to build
3. **GATHER_OVERVIEW** — Collects name, description, and persona
4. **DESIGN_STRUCTURE** — Iteratively adds states/steps/tools based on artifact type
5. **DEFINE_CONNECTIONS** — Adds transitions, step links, or tool configurations
6. **REVIEW** — Validates the artifact, shows errors/warnings, allows revisions
7. **OUTPUT** — Produces the final validated JSON definition

Handlers manage the build loop: `BuilderInjector` injects builder state into prompts, `ActionDispatcher` processes extracted actions (add_state, add_transition, etc.), `ProgressTracker` monitors completion, and `Finalizer` produces the output.

## File Map

| File | Purpose |
|------|---------|
| `agent.py` | **MetaAgent** — main entry point, conversation management, `start()` / `send()` / `run_interactive()` |
| `builders.py` | **FSMBuilder**, **WorkflowBuilder**, **AgentBuilder** — incremental artifact construction with validation |
| `fsm_definitions.py` | `build_meta_fsm()` — generates the 7-state meta-agent FSM definition |
| `handlers.py` | **MetaHandlers** — builder injection, action dispatch, progress tracking, finalization |
| `prompts.py` | Prompt templates for adaptive questioning at each conversation stage |
| `definitions.py` | Pydantic models: ArtifactType, BuildProgress, MetaAgentConfig, MetaAgentResult |
| `constants.py` | MetaStates, ContextKeys, Actions, HandlerNames, Defaults |
| `exceptions.py` | Exception hierarchy: MetaAgentError, BuilderError, MetaValidationError, OutputError |
| `output.py` | `format_artifact_json()`, `format_summary()`, `save_artifact()` |
| `__main__.py` | CLI: `fsm-llm-meta` / `python -m fsm_llm_meta` |
| `__version__.py` | Package version (synced from fsm_llm) |
| `__init__.py` | Public API exports — single `__all__` list (16 items) |

## API Reference

### MetaAgent

| Method | Description |
|--------|-------------|
| `MetaAgent(config=None, **api_kwargs)` | Initialize the meta-agent |
| `start(initial_message="")` | Start conversation → first agent response |
| `send(user_input)` | Send user input → agent response |
| `is_complete()` | Check if artifact building is finished |
| `get_result()` | Get `MetaAgentResult` with artifact and validation status |
| `run_interactive()` | Full interactive REPL loop → `MetaAgentResult` |

### Builders

| Class | Artifact | Key Methods |
|-------|----------|-------------|
| `FSMBuilder` | FSM definition | `set_overview()`, `add_state()`, `update_state()`, `remove_state()`, `add_transition()`, `remove_transition()`, `set_initial_state()` |
| `WorkflowBuilder` | Workflow definition | `set_overview()`, `add_step()`, `remove_step()`, `set_step_transition()`, `set_initial_step()` |
| `AgentBuilder` | Agent configuration | `set_overview()`, `set_agent_type()`, `set_config()`, `add_tool()`, `remove_tool()` |

All builders inherit from `ArtifactBuilder` which provides: `validate_partial()`, `validate_complete()`, `get_progress()`, `get_summary()`, `build()`.

### Models

| Class | Key Fields |
|-------|------------|
| `ArtifactType` | Enum: `FSM`, `WORKFLOW`, `AGENT` |
| `BuildProgress` | `total_required`, `completed`, `missing`, `warnings`, `percentage`, `is_complete` |
| `MetaAgentConfig` | `model`, `temperature`, `max_tokens`, `max_turns` |
| `MetaAgentResult` | `artifact_type`, `artifact`, `artifact_json`, `is_valid`, `validation_errors`, `conversation_turns` |

### Output Utilities

| Function | Description |
|----------|-------------|
| `format_artifact_json(artifact)` | Format artifact dict as pretty-printed JSON |
| `format_summary(result)` | Human-readable build summary |
| `save_artifact(artifact, path)` | Save artifact dict to JSON file → file path |

## Exception Hierarchy

```
MetaAgentError (extends FSMError)
├── BuilderError           # Builder state mutation failures
├── MetaValidationError    # Artifact validation failures
└── OutputError            # JSON formatting or file save failures
```

## Examples

See [`examples/meta/build_fsm/`](../../examples/meta/build_fsm/) for a complete runnable example.

## Development

```bash
pytest tests/test_fsm_llm_meta/  # 115 tests across 5 test files
```

## License

GNU General Public License v3.0. See [LICENSE](../../LICENSE).
