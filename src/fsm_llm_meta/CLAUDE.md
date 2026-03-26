# fsm_llm_meta -- Claude Code Instructions

## What This Package Does

Interactive meta-agent that builds FSMs, workflows, and agent configurations through conversation. The user describes what they want; the agent extracts requirements, uses a ReactAgent with builder tools to autonomously construct the artifact, validates it, and outputs a JSON definition. The build phase leverages `fsm_llm_agents.ReactAgent` for autonomous multi-tool execution.

## File Map

| File | What It Contains |
|------|-----------------|
| `agent.py` | `MetaAgent` class -- `start()`, `send()`, `is_complete()`, `get_result()`, `run_interactive()`; 3-phase architecture: intake (LLM extraction), build (ReactAgent), review (approve/revise) |
| `tools.py` | Builder tool factories: `create_fsm_tools()`, `create_workflow_tools()`, `create_agent_tools()`, `create_builder_tools()`; wraps builder methods as `@tool`-decorated closures for ToolRegistry |
| `builders.py` | `ArtifactBuilder` (ABC) + `FSMBuilder`, `WorkflowBuilder`, `AgentBuilder`; each has mutation methods, `validate_partial()`, `validate_complete()`, `get_progress()`, `get_summary(detail_level)`, `to_dict()` |
| `prompts.py` | Prompt builders: `INTAKE_SYSTEM_PROMPT`, `build_intake_user_message()`, `build_task_prompt()`, `build_revision_prompt()`, `build_review_presentation()`, `build_welcome_message()`, `build_followup_message()`, `build_output_message()` |
| `definitions.py` | Pydantic models: `ArtifactType` (FSM/WORKFLOW/AGENT enum), `BuildProgress`, `MetaAgentConfig`, `MetaAgentResult` |
| `constants.py` | `MetaPhases` (INTAKE/BUILD/REVIEW/DONE), `Defaults`, `DecisionWords` (approve/revise detection), `ErrorMessages`, `LogMessages` |
| `handlers.py` | Minimal module kept for backward compatibility (handler logic moved to ReactAgent) |
| `fsm_definitions.py` | Empty module kept for backward compatibility (ReactAgent auto-generates FSMs) |
| `output.py` | `format_artifact_json()`, `format_summary()`, `save_artifact()` |
| `exceptions.py` | `MetaAgentError` (extends `FSMError`) -> `BuilderError`, `MetaValidationError`, `OutputError` |
| `__main__.py` | CLI entry point: argparse with --model, --output, --temperature, --max-turns |
| `__init__.py` | 20-item `__all__` list (includes tool factories) |
| `__version__.py` | Imports version from `fsm_llm.__version__` |

## Key Patterns

### 3-Phase Architecture

```
INTAKE (1-2 turns) -> BUILD (autonomous) -> REVIEW (1+ turns) -> DONE
                                               |
                                               |-- (revise) --> BUILD
```

**Phase 1: INTAKE** -- Extract requirements via `LiteLLMInterface.extract_data()`:
- Extracts artifact_type, name, description, persona, component hints
- Transitions to BUILD when type + name + description are all present
- Asks at most one follow-up question for missing fields

**Phase 2: BUILD** -- Autonomous construction via `ReactAgent`:
- Creates appropriate builder (FSMBuilder/WorkflowBuilder/AgentBuilder)
- Creates ToolRegistry with builder tools via `create_builder_tools()`
- Launches `ReactAgent(tools=registry).run(task=prompt)` which autonomously loops think→act→observe until complete
- All builder mutations happen through tool calls
- ReactAgent calls `validate()` tool to self-correct errors before concluding

**Phase 3: REVIEW** -- User approval or revision:
- Presents builder summary + validation results
- User approves → finalize and output
- User requests changes → re-run ReactAgent with revision task prompt

### Builder Tools (`tools.py`)

Each factory creates a `ToolRegistry` with closures over a builder instance:

**FSM tools**: `set_overview`, `add_state`, `update_state`, `remove_state`, `add_transition`, `remove_transition`, `set_initial_state`, `validate`, `get_summary`

**Workflow tools**: `set_overview`, `add_step`, `remove_step`, `set_step_transition`, `set_initial_step`, `validate`, `get_summary`

**Agent tools**: `set_overview`, `set_agent_type`, `add_tool`, `remove_tool`, `set_config`, `validate`, `get_summary`

Tools return string results. Errors are returned as strings (not exceptions) so the ReactAgent can self-correct.

### Builder Pattern (unchanged)

`ArtifactBuilder` is the abstract base. Three concrete builders:

- `FSMBuilder` -- states dict keyed by state_id, each with transitions list; validates via `FSMDefinition(**self.to_dict())`
- `WorkflowBuilder` -- steps dict keyed by step_id with step_type and transitions; 8 valid step types
- `AgentBuilder` -- agent_type (11 valid types), config dict, tools list; defaults: model=gpt-4o-mini, max_iterations=10, timeout_seconds=300, temperature=0.5, max_tokens=1000

### MetaAgentConfig Defaults

- model: `DEFAULT_LLM_MODEL` (from `fsm_llm.constants`)
- temperature: 0.7
- max_tokens: 2000
- max_turns: 50

### ReactAgent Config (for build phase)

- max_iterations: 25 (higher than default to handle complex artifacts)
- timeout_seconds: 120.0
- temperature: 0.3 (low for reliable tool calling)
- max_tokens: 1000

## Dependencies

### On Core (`fsm_llm`)
- `fsm_llm.llm.LiteLLMInterface` -- used for intake extraction
- `fsm_llm.definitions.DataExtractionRequest` -- request model for extract_data()
- `fsm_llm.definitions.FSMDefinition` -- used by FSMBuilder.validate_complete()
- `fsm_llm.definitions.FSMError` -- base class for MetaAgentError
- `fsm_llm.logging.logger` -- loguru logger
- `fsm_llm.constants.DEFAULT_LLM_MODEL` -- default model value

### On Agents (`fsm_llm_agents`)
- `fsm_llm_agents.ReactAgent` -- autonomous build phase
- `fsm_llm_agents.definitions.AgentConfig` -- ReactAgent configuration
- `fsm_llm_agents.tools.ToolRegistry` -- tool registration
- `fsm_llm_agents.tools.tool` -- `@tool` decorator for builder tools

## Exception Hierarchy

```
FSMError (from fsm_llm.definitions)
  └── MetaAgentError              # Base for all meta-agent errors
        ├── BuilderError          # Invalid builder operations; has .action attribute
        ├── MetaValidationError   # Validation failures; has .errors list
        └── OutputError           # File save / serialization failures; has .path attribute
```

## Testing

203 tests across 9 files in `tests/test_fsm_llm_meta/`:

| File | What It Tests |
|------|---------------|
| `test_builders.py` | FSMBuilder, WorkflowBuilder, AgentBuilder mutation, validation, progress, summaries |
| `test_builders_elaborate.py` | Edge cases: type coercion, to_dict fidelity, multi-transition scenarios |
| `test_tools.py` | Tool factories: tool registration, execution, error handling for all three builder types |
| `test_agent.py` | MetaAgent lifecycle: intake, build (mocked ReactAgent), review, approval, revision, error paths |
| `test_prompts.py` | Prompt builders: intake, task, revision, review, welcome, follow-up, output |
| `test_definitions.py` | Pydantic models: ArtifactType, BuildProgress, MetaAgentConfig, MetaAgentResult |
| `test_handlers.py` | Module importability (handler logic moved to ReactAgent) |
| `test_handlers_elaborate.py` | Module importability |
| `test_fsm_definitions.py` | Module importability (FSM generation moved to ReactAgent) |

Run: `pytest tests/test_fsm_llm_meta/ -v`

## Gotchas

- **MetaAgent requires a real LLM for full flow** -- `start()` creates a `LiteLLMInterface` for intake extraction, and `_do_build()` creates a `ReactAgent` that makes LLM calls. Unit tests mock both.
- **Builders validate incrementally** -- `validate_partial()` returns warnings (non-blocking), `validate_complete()` returns errors (blocking).
- **Tool errors don't crash the agent** -- builder tools catch `BuilderError` and return error strings. The ReactAgent sees these as observations and self-corrects.
- **First state/step auto-sets initial** -- when adding the first state/step, the initial state/step is automatically set.
- **AgentBuilder has default config** -- unlike FSMBuilder and WorkflowBuilder, AgentBuilder starts with non-empty config.
- **Build phase is autonomous** -- the ReactAgent loops internally (think→act→observe→think→...) without user interaction. Multiple tools are called per `_do_build()` invocation.
- **Review classification is heuristic** -- `_classify_decision()` uses word matching against `DecisionWords.APPROVE` and `DecisionWords.REVISE` sets. Ambiguous messages default to revision.
- **Revision preserves builder state** -- the existing builder is reused, not recreated. The ReactAgent modifies the existing artifact.
