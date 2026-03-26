# fsm_llm_meta -- Claude Code Instructions

## What This Package Does

Interactive meta-agent that builds FSMs, workflows, and agent configurations through guided conversation. The user describes what they want; the agent asks clarifying questions, incrementally constructs the artifact using a builder, validates it, and outputs a JSON definition. The meta-agent is itself driven by a 7-state FSM processed by the core `fsm_llm` engine.

## File Map

| File | What It Contains |
|------|-----------------|
| `agent.py` | `MetaAgent` class -- `start()`, `send()`, `is_complete()`, `get_result()`, `run_interactive()`; registers handlers, manages conversation lifecycle, builds `MetaAgentResult` |
| `builders.py` | `ArtifactBuilder` (ABC) + `FSMBuilder`, `WorkflowBuilder`, `AgentBuilder`; each has mutation methods, `validate_partial()`, `validate_complete()`, `get_progress()`, `get_summary(detail_level)`, `to_dict()` |
| `definitions.py` | Pydantic models: `ArtifactType` (FSM/WORKFLOW/AGENT enum), `BuildProgress`, `MetaAgentConfig`, `MetaAgentResult` |
| `constants.py` | `MetaStates` (7 states), `ContextKeys`, `Actions`, `HandlerNames`, `Defaults` (MODEL, TEMPERATURE=0.7, MAX_TOKENS=2000, MAX_TURNS=50), `ErrorMessages`, `LogMessages` |
| `handlers.py` | `MetaHandlers` -- `inject_builder_state()` (PRE_PROCESSING), `handle_overview()` (POST_PROCESSING on gather_overview), `dispatch_action()` (POST_PROCESSING on design_structure/define_connections), `run_validation()` (POST_TRANSITION on review entry), `finalize()` (POST_TRANSITION on output entry) |
| `prompts.py` | Static prompt builders for welcome, classify, and output states only; dynamic states get context through handler injection |
| `fsm_definitions.py` | `build_meta_fsm()` -- returns a dict (not JSON) defining the 7-state conversation FSM with transitions and conditions |
| `output.py` | `format_artifact_json()`, `format_summary()`, `save_artifact()` |
| `exceptions.py` | `MetaAgentError` (extends `FSMError`) -> `BuilderError`, `MetaValidationError`, `OutputError` |
| `__main__.py` | CLI entry point: argparse with --model, --output, --temperature, --max-turns |
| `__init__.py` | 16-item `__all__` list |
| `__version__.py` | Imports version from `fsm_llm.__version__` |

## Key Patterns

### 7-State FSM Flow (MetaStates)

```
WELCOME -> CLASSIFY -> GATHER_OVERVIEW -> DESIGN_STRUCTURE -> DEFINE_CONNECTIONS -> REVIEW -> OUTPUT
                                                ^                                     |
                                                |_____________ (revise) ______________|
```

- WELCOME auto-transitions to CLASSIFY after the first response
- CLASSIFY transitions to GATHER_OVERVIEW when `artifact_type` is in context
- GATHER_OVERVIEW transitions when both `artifact_name` and `artifact_description` are set
- DESIGN_STRUCTURE loops until `structure_done == True`
- DEFINE_CONNECTIONS loops until `connections_done == True`
- REVIEW transitions to OUTPUT on `user_decision == "approve"` or back to DESIGN_STRUCTURE on `"revise"`
- OUTPUT is terminal (no transitions)

### Builder Pattern

`ArtifactBuilder` is the abstract base. Three concrete builders:

- `FSMBuilder` -- states dict keyed by state_id, each with transitions list; validates via `FSMDefinition(**self.to_dict())`
- `WorkflowBuilder` -- steps dict keyed by step_id with step_type and transitions; 8 valid step types
- `AgentBuilder` -- agent_type (11 valid types), config dict, tools list; defaults: model=gpt-4o-mini, max_iterations=10, timeout_seconds=300, temperature=0.5, max_tokens=1000

Builders track progress with `get_progress()` returning `BuildProgress` (percentage, missing fields, warnings).

### Action Dispatch System

1. LLM extracts JSON: `{"action": "add_state", "action_params": {"state_id": "...", ...}}`
2. `MetaHandlers.dispatch_action()` runs at POST_PROCESSING on design_structure and define_connections
3. Routes to `_execute_fsm_action()`, `_execute_workflow_action()`, or `_execute_agent_action()` based on builder type
4. Builder method called with params; warnings/errors captured
5. Action and params cleared from context after dispatch to prevent re-firing

### Context Compaction

Uses `ContextCompactor` from `fsm_llm.context`:
- **Transient keys** cleared every turn: `action_result`, `action_errors`, `action`, `action_params`
- **Prune on entry** clears phase-specific keys when entering certain states (e.g., entering DESIGN_STRUCTURE clears overview fields; entering REVIEW clears `connections_done`)
- Compaction runs at PRE_PROCESSING (before builder injection); pruning runs at POST_TRANSITION

### Detail Levels for Builder Summaries

`MetaHandlers._DETAIL_LEVEL_MAP` controls how much builder state is injected into LLM context:
- `design_structure` -> "minimal" (names + counts)
- `define_connections` -> "standard" (names + short descriptions + targets)
- `review` -> "full" (everything including instructions)
- All other states -> "minimal" (fallback)

### MetaAgentConfig Defaults

- model: `DEFAULT_LLM_MODEL` (from `fsm_llm.constants`)
- temperature: 0.7
- max_tokens: 2000
- max_turns: 50

## Dependencies on Core

- `fsm_llm.API` -- used by MetaAgent to create and drive conversation
- `fsm_llm.context.ContextCompactor` -- transient key cleanup and state-entry pruning
- `fsm_llm.handlers.HandlerTiming` -- for registering handlers at correct timing points
- `fsm_llm.definitions.FSMDefinition` -- used by FSMBuilder.validate_complete() to validate output
- `fsm_llm.definitions.FSMError` -- base class for MetaAgentError
- `fsm_llm.logging.logger` -- loguru logger
- `fsm_llm.constants.DEFAULT_LLM_MODEL` -- default model value

## Exception Hierarchy

```
FSMError (from fsm_llm.definitions)
  └── MetaAgentError              # Base for all meta-agent errors
        ├── BuilderError          # Invalid builder operations; has .action attribute
        ├── MetaValidationError   # Validation failures; has .errors list
        └── OutputError           # File save / serialization failures; has .path attribute
```

## Testing

129 tests across 5 files in `tests/test_fsm_llm_meta/`:

| File | What It Tests |
|------|---------------|
| `test_builders.py` | FSMBuilder, WorkflowBuilder, AgentBuilder mutation, validation, progress, summaries |
| `test_handlers.py` | MetaHandlers: builder injection, action dispatch, overview handling, validation, finalization |
| `test_definitions.py` | Pydantic models: ArtifactType, BuildProgress, MetaAgentConfig, MetaAgentResult |
| `test_fsm_definitions.py` | build_meta_fsm() structure and state validity |
| `test_agent.py` | MetaAgent lifecycle, start/send/is_complete/get_result, error handling |

Run: `pytest tests/test_fsm_llm_meta/ -v`

## Gotchas

- **MetaAgent requires a real LLM for full flow** -- the `start()` and `send()` methods call `API.from_definition()` which needs a working LLM. Unit tests mock the API layer.
- **Builders validate incrementally** -- `validate_partial()` returns warnings (non-blocking), `validate_complete()` returns errors (blocking). An artifact can have partial warnings and still be valid.
- **Action dispatch happens at POST_PROCESSING** -- not POST_TRANSITION. The action is extracted by the LLM in Pass 1, then the handler dispatches it before Pass 2 generates the response.
- **Transient context keys are auto-cleaned** -- `action`, `action_params`, `action_result`, `action_errors` are cleared at the start of each turn via ContextCompactor. Do not rely on them persisting across turns.
- **`build_meta_fsm()` returns a dict, not JSON** -- it returns a Python dict suitable for passing to `API.from_definition()` or `FSMDefinition(**d)`.
- **`FSMBuilder.add_state()` creates a minimal state** -- only id, description, purpose, and an empty transitions list. extraction_instructions and response_instructions are optional and only added if provided.
- **First state/step auto-sets initial** -- when adding the first state to FSMBuilder or first step to WorkflowBuilder, the initial_state/initial_step_id is automatically set.
- **AgentBuilder has default config** -- unlike FSMBuilder and WorkflowBuilder, AgentBuilder starts with non-empty config (model, max_iterations, timeout_seconds, temperature, max_tokens).
- **17 exports in `__init__.py`** -- MetaAgent, ArtifactBuilder, 3 specialized builders, 4 models, 4 exceptions, 3 output functions, __version__.
- **Handler registration order matters** -- context compaction must run before builder injection (both at PRE_PROCESSING); pruning runs at POST_TRANSITION.
