# fsm_llm_agents — Agentic Patterns Extension

## What This Package Does

ReAct (Reasoning + Acting) and Human-in-the-Loop agent patterns built on fsm_llm's core, classification, and workflow packages. Auto-generates FSM definitions from tool registries so users don't need to author FSM JSON for agent patterns.

## File Map

| File | Purpose |
|------|---------|
| `react.py` | **ReactAgent** — main entry point. Auto-generates FSM from ToolRegistry, registers handlers, manages think→act→observe→conclude loop |
| `tools.py` | **ToolRegistry** + `@tool` decorator — tool management, prompt generation, execution, classification schema generation |
| `hitl.py` | **HumanInTheLoop** — approval policies, approval callbacks, confidence-based escalation, human override |
| `handlers.py` | **AgentHandlers** — tool executor (POST_TRANSITION on act), iteration limiter (PRE_TRANSITION), approval checker (CONTEXT_UPDATE) |
| `fsm_definitions.py` | `build_react_fsm()` — generates valid FSMDefinition dict from ToolRegistry with optional approval state |
| `prompts.py` | Prompt builders for think/act/conclude/approval states with tool awareness |
| `definitions.py` | Pydantic models: ToolDefinition, ToolCall, ToolResult, AgentStep, AgentTrace, AgentConfig, AgentResult, ApprovalRequest |
| `constants.py` | AgentStates, ContextKeys, HandlerNames, Defaults, ErrorMessages, LogMessages |
| `exceptions.py` | AgentError hierarchy (7 types) |
| `__main__.py` | CLI: `python -m fsm_llm_agents --info` |
| `__init__.py` | Public API exports |
| `__version__.py` | Version import from fsm_llm |

## Key Patterns

### ReactAgent Architecture
1. User calls `agent.run(task)` with a task string
2. Agent builds FSM from ToolRegistry via `build_react_fsm()`
3. FSM states: **think** (extract tool_name/tool_input/should_terminate) → **act** (execute tool via handler) → **think** (loop) or **conclude** (terminal)
4. Tool execution happens in POST_TRANSITION handler on `act` state
5. Observations accumulate in context automatically
6. Budget enforcement via PRE_TRANSITION handler (max iterations, timeout)

### ToolRegistry
- `register_function(fn, name, description)` — register any callable
- `@tool(description="...")` decorator — marks functions as tools
- `to_prompt_description()` — generates LLM-friendly tool listing
- `to_classification_schema()` — generates ClassificationSchema for tool selection
- `execute(tool_call)` — executes with timing, error handling, result wrapping

### HumanInTheLoop
- `approval_policy` — function deciding if a tool call needs approval
- `approval_callback` — function that requests approval from human
- `confidence_threshold` — auto-escalate below this confidence
- Integrates via CONTEXT_UPDATE handler + optional `await_approval` FSM state

## Dependencies on Core
- `fsm_llm.API` — FSM execution (from_definition, converse, start_conversation, etc.)
- `fsm_llm.handlers.HandlerTiming` — handler timing points
- `fsm_llm.constants.DEFAULT_LLM_MODEL` — default model
- `fsm_llm.logging.logger` — loguru logging
- `fsm_llm.definitions.FSMError` — base exception class
- `fsm_llm.definitions.FSMDefinition` — validates generated FSM dicts

## Testing
```bash
pytest tests/test_fsm_llm_agents/
```

## Gotchas
- `ReactAgent.run()` needs a real LLM or mock — it creates an API instance internally
- FSM definitions are generated at runtime, not loaded from JSON files
- ToolDefinition requires `execute_fn` to be set before registering (no schema-only tools)
- AgentHandlers is stateful (tracks iteration count) — create fresh per `run()` call
