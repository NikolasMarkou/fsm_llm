# FSM-LLM -- Claude Code Instructions

## Project Overview

FSM-LLM (v0.3.0) is a Python framework for building stateful conversational AI by combining LLMs with Finite State Machines. It uses a **2-pass architecture**: Pass 1 extracts data + evaluates transitions, Pass 2 generates the response from the final state.

- **License**: GPL-3.0-or-later
- **Python**: 3.10, 3.11, 3.12
- **Core deps**: loguru, litellm (>=1.82,<2.0, excluding 1.82.7/1.82.8), pydantic (>=2.0), python-dotenv
- **Virtual environment**: Always use `.venv` -- run commands with `.venv/bin/python` or activate first

## Quick Commands

```bash
make test           # pytest -v (2,206 tests)
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make type-check     # mypy across all 5 packages
make build          # python -m build (wheel + sdist)
make clean          # remove build artifacts and caches
make coverage       # pytest with coverage report
make install-dev    # pip install -c constraints.txt -e ".[dev,workflows,reasoning,agents,monitor]" + pre-commit install
make audit          # audit site-packages for suspicious .pth files

# CLI tools
fsm-llm --fsm <path.json>            # Run FSM interactively
fsm-llm-visualize --fsm <path.json>  # ASCII visualization
fsm-llm-validate --fsm <path.json>   # Validate FSM definition
fsm-llm-monitor                      # Launch web monitoring dashboard
fsm-llm-meta                         # Interactive artifact builder (routes to fsm_llm_agents.meta_cli)
```

## Architecture -- 2-Pass Flow

```
User Input -> [Pass 1: Data Extraction (LLM)] -> Context Update -> Classification Extractions
           -> Transition Evaluation (rules) -> If AMBIGUOUS: Classification -> State Transition
           -> [Pass 2: Response Generation (LLM)] -> User Output
```

Key classes in `src/fsm_llm/`:
- **API** (`api.py`) -- User-facing entry point. Factory: `from_file()`, `from_definition()`. Conversation: `start_conversation()`, `converse()`, `end_conversation()`, `has_conversation_ended()`. Queries: `get_data()`, `get_current_state()`, `get_conversation_history()`, `list_active_conversations()`. FSM stacking: `push_fsm()`, `pop_fsm()`, `get_stack_depth()`, `get_sub_conversation_id()`. Handlers: `register_handler()`, `register_handlers()`, `create_handler()`. Management: `update_context()`, `cleanup_stale_conversations()`, `get_llm_interface()`, `close()`
- **FSMManager** (`fsm.py`) -- Core orchestration. Implements 2-pass processing with per-conversation thread locks
- **MessagePipeline** (`pipeline.py`) -- 2-pass message processing engine, extracted from FSMManager
- **HandlerBuilder** (`handlers.py`) -- Fluent builder: `.at()`, `.on_state()`, `.when_context_has()`, `.do()`
- **HandlerTiming** (`handlers.py`) -- 8 hook points: `START_CONVERSATION`, `PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `END_CONVERSATION`, `ERROR`
- **TransitionEvaluator** (`transition_evaluator.py`) -- Evaluates transitions: DETERMINISTIC, AMBIGUOUS, or BLOCKED
- **Classifier** (`classification.py`) -- LLM-backed structured intent classification (single, multi, hierarchical). Resolves ambiguous transitions
- **LiteLLMInterface** (`llm.py`) -- LLM communication via litellm (100+ providers). Two active methods: `generate_response`, `extract_field`
- **clean_context_keys** (`context.py`) -- Stateless context cleaning (strips None values, internal key prefixes, forbidden patterns)
- **WorkingMemory** (`memory.py`) -- Structured working memory with named buffers (core, scratch, environment, reasoning) for organizing agent context

## Package Map

```
src/
├── fsm_llm/                         # Core framework
│   ├── api.py                       # API class -- primary user-facing entry point
│   ├── fsm.py                       # FSMManager -- state machine orchestration
│   ├── pipeline.py                  # MessagePipeline -- 2-pass message processing engine
│   ├── classification.py            # Classifier, HierarchicalClassifier, IntentRouter, HandlerFn
│   ├── definitions.py               # Pydantic models: State, Transition, FSMDefinition, FSMContext, FSMInstance, Conversation, ClassificationSchema, IntentDefinition, ClassificationResult, ClassificationExtractionConfig + exception hierarchy
│   ├── handlers.py                  # HandlerSystem, HandlerBuilder, BaseHandler, LambdaHandler, HandlerTiming enum
│   ├── prompts.py                   # Prompt builders for extraction, response gen, classification (ClassificationPromptConfig, build_classification_json_schema, build_classification_system_prompt)
│   ├── llm.py                       # LLMInterface ABC + LiteLLMInterface implementation
│   ├── ollama.py                    # Ollama-specific helpers (thinking disable, json_schema format)
│   ├── transition_evaluator.py      # Rule-based transition evaluation with JsonLogic
│   ├── expressions.py               # JsonLogic evaluator (var, and, or, ==, in, has_context, context_length, etc.)
│   ├── context.py                   # Context cleaning utilities -- clean_context_keys(), ContextCompactor
│   ├── memory.py                    # WorkingMemory -- structured named buffers (core, scratch, environment, reasoning)
│   ├── runner.py                    # Interactive CLI conversation runner (used by __main__)
│   ├── validator.py                 # FSM structure validation
│   ├── visualizer.py                # ASCII FSM diagrams
│   ├── utilities.py                 # JSON extraction with multiple fallback strategies
│   ├── constants.py                 # Defaults, security patterns, internal key prefixes
│   ├── logging.py                   # Loguru setup with conversation context, enable_debug_logging()
│   ├── __main__.py                  # CLI entry point (run, validate, visualize modes)
│   ├── __version__.py               # Package version string
│   └── __init__.py                  # Public API exports (single __all__ list)
│
├── fsm_llm_reasoning/               # Structured reasoning engine
│   ├── engine.py                    # ReasoningEngine -- orchestrates 9 reasoning strategies via FSMs
│   ├── reasoning_modes.py           # FSM definitions for each strategy (analytical, deductive, etc.)
│   ├── handlers.py                  # Reasoning-specific handlers (validation, tracing, context pruning, retry limiting)
│   ├── definitions.py               # ReasoningStep, ReasoningTrace, SolutionResult, ProblemContext
│   ├── constants.py                 # ReasoningType enum, ContextKeys, OrchestratorStates, ClassifierStates
│   ├── utilities.py                 # load_fsm_definition(), map_reasoning_type(), get_available_reasoning_types()
│   ├── exceptions.py                # ReasoningEngineError -> ReasoningExecutionError, ReasoningClassificationError
│   ├── __main__.py                  # CLI: python -m fsm_llm_reasoning
│   ├── __version__.py               # Package version string
│   └── __init__.py                  # Public API exports
│
├── fsm_llm_workflows/               # Workflow orchestration engine
│   ├── engine.py                    # WorkflowEngine -- async event-driven execution
│   ├── dsl.py                       # Python DSL: create_workflow(), auto_step(), llm_step(), conversation_step(), agent_step(), retry_step(), switch_step(), etc.
│   ├── steps.py                     # 11 step types: AutoTransition, APICall, Condition, LLMProcessing, WaitForEvent, Timer, Parallel, Conversation, Agent, Retry, Switch
│   ├── definitions.py               # WorkflowDefinition with validation (reachability, cycles)
│   ├── models.py                    # WorkflowStatus, WorkflowEvent, WorkflowInstance
│   ├── exceptions.py                # WorkflowError -> Definition, Step, Instance, Timeout, Validation, State, Event, Resource errors
│   ├── __version__.py               # Package version string
│   └── __init__.py                  # Public API exports
│
├── fsm_llm_agents/                  # Agentic patterns (12 patterns + meta builder)
│   ├── base.py                      # BaseAgent -- ABC with shared conversation loop, budgets, __call__, structured output
│   ├── react.py                     # ReactAgent -- ReAct loop with auto-generated FSM and tool dispatch
│   ├── tools.py                     # ToolRegistry + @tool decorator (auto-schema from type hints) + register_agent()
│   ├── skills.py                    # SkillDefinition + SkillLoader -- external skill/plugin loading system
│   ├── memory_tools.py              # create_memory_tools() -- remember, recall, forget, list_memories tools for WorkingMemory
│   ├── hitl.py                      # HumanInTheLoop -- approval gates, escalation, confidence thresholds
│   ├── handlers.py                  # AgentHandlers -- tool executor, iteration limiter, approval checker
│   ├── fsm_definitions.py           # build_react_fsm() and 10 more -- auto-generates FSM from ToolRegistry
│   ├── prompts.py                   # Tool-aware prompt builders for think/act/conclude states
│   ├── definitions.py               # ToolDefinition, ToolCall, ToolResult, AgentStep, AgentTrace, AgentConfig (output_schema), AgentResult (structured_output), ApprovalRequest
│   ├── constants.py                 # AgentStates, ContextKeys, HandlerNames, Defaults
│   ├── exceptions.py                # AgentError + MetaBuilderError hierarchies
│   ├── __main__.py                  # CLI: python -m fsm_llm_agents --info
│   ├── __version__.py               # Package version string
│   ├── adapt.py                     # ADaPTAgent -- adaptive complexity with decomposition
│   ├── debate.py                    # DebateAgent -- multi-perspective debate with judge
│   ├── evaluator_optimizer.py       # EvaluatorOptimizerAgent -- iterative evaluation and optimization
│   ├── maker_checker.py             # MakerCheckerAgent -- draft-review verification loop
│   ├── orchestrator.py              # OrchestratorAgent -- worker delegation and synthesis
│   ├── plan_execute.py              # PlanExecuteAgent -- plan decomposition and sequential execution
│   ├── prompt_chain.py              # PromptChainAgent -- sequential prompt pipeline with gates
│   ├── reasoning_react.py           # ReasoningReactAgent -- ReAct with structured reasoning (requires fsm_llm_reasoning)
│   ├── reflexion.py                 # ReflexionAgent -- self-reflection with memory
│   ├── rewoo.py                     # REWOOAgent -- planning-first tool execution
│   ├── self_consistency.py          # SelfConsistencyAgent -- multiple samples with voting
│   ├── meta_builder.py              # MetaBuilderAgent -- interactive artifact builder (FSMs, workflows, agents)
│   ├── meta_builders.py             # FSMBuilder, WorkflowBuilder, AgentBuilder -- automated artifact generation
│   ├── meta_cli.py                  # CLI entry point for fsm-llm-meta command
│   ├── meta_tools.py                # Builder tool factories: create_fsm_tools(), create_workflow_tools(), create_agent_tools()
│   ├── meta_fsm.py                  # FSM definitions for meta-agent classification-driven routing
│   ├── meta_prompts.py              # Intake, build spec, review, revision prompt builders
│   ├── meta_output.py               # format_artifact_json(), format_summary(), save_artifact()
│   └── __init__.py                  # Public API exports + create_agent() factory
│
├── fsm_llm_monitor/                 # Web-based monitoring dashboard
│   ├── server.py                    # FastAPI web server -- REST + WebSocket APIs
│   ├── bridge.py                    # MonitorBridge -- connects EventCollector to API instance
│   ├── collector.py                 # EventCollector -- handler-based event capture + log sink
│   ├── instance_manager.py          # InstanceManager -- lifecycle management for FSMs, agents, workflows
│   ├── definitions.py               # MonitorEvent, MetricSnapshot, MonitorConfig, FSMSnapshot, etc.
│   ├── constants.py                 # Theme colors, defaults, event types
│   ├── exceptions.py                # MonitorError -> MonitorInitializationError, MetricCollectionError, MonitorConnectionError
│   ├── __main__.py                  # CLI: python -m fsm_llm_monitor / fsm-llm-monitor
│   ├── __version__.py               # Package version string
│   ├── static/                      # Frontend assets
│   │   ├── app.js                   # Main application module
│   │   ├── style.css                # Grafana-inspired dark dashboard theme
│   │   ├── flows.json               # Agent/workflow pattern flow definitions
│   │   ├── pages/                   # Page components
│   │   │   ├── builder.js           # Builder page module
│   │   │   ├── control.js           # Control Center -- unified instance table with drawer
│   │   │   ├── conversations.js     # Conversation detail view and chat interface
│   │   │   ├── dashboard.js         # Dashboard page -- metric cards, instance grid, events
│   │   │   ├── launch.js            # Launch modal for FSMs, agents, workflows
│   │   │   ├── logs.js              # Logs page -- level-filtered stream with live/pause
│   │   │   ├── settings.js          # Settings page -- runtime config and system info
│   │   │   └── visualizer.js        # Visualizer page -- tabbed graph viewer with presets
│   │   ├── services/                # Service layer
│   │   │   ├── api.js               # REST API client
│   │   │   ├── state.js             # Global state management
│   │   │   └── ws.js                # WebSocket communication and message dispatch
│   │   └── utils/                   # Utility modules
│   │       ├── dom.js               # DOM manipulation helpers
│   │       ├── format.js            # Data formatting utilities
│   │       ├── graph.js             # FSM/agent/workflow graph rendering
│   │       └── markdown.js          # Markdown rendering utilities
│   ├── templates/
│   │   └── index.html               # Single-page template
│   └── __init__.py                  # Public API exports
```

## Code Conventions

- **Linting/Formatting**: ruff (target Python 3.10, line-length 88). Ignored rules: E402, E501, RUF013, RUF001, RUF022
- **Type hints**: Used throughout. mypy configured with `disallow_untyped_defs=false`, pydantic plugin enabled
- **Models**: Pydantic v2 `BaseModel` with `model_validator` for complex validation
- **Logging**: loguru via `from fsm_llm.logging import logger`
- **Exports**: Single `__all__` list in `__init__.py` -- no dynamic extend/append
- **Exceptions**:
  - Core: `FSMError` -> `StateNotFoundError`, `InvalidTransitionError`, `LLMResponseError`, `TransitionEvaluationError`, `ClassificationError` -> `SchemaValidationError`, `ClassificationResponseError`
  - Handlers: `HandlerSystemError(FSMError)` -> `HandlerExecutionError`
  - Reasoning: `ReasoningEngineError` -> `ReasoningExecutionError`, `ReasoningClassificationError`
  - Workflows: `WorkflowError` -> `WorkflowDefinitionError`, `WorkflowStepError`, `WorkflowInstanceError`, `WorkflowTimeoutError`, `WorkflowValidationError`, `WorkflowStateError`, `WorkflowEventError`, `WorkflowResourceError`
  - Agents: `AgentError` -> `ToolExecutionError`, `ToolNotFoundError`, `ToolValidationError`, `BudgetExhaustedError`, `ApprovalDeniedError`, `AgentTimeoutError`, `EvaluationError`, `DecompositionError`
  - Agents (Meta): `MetaBuilderError(AgentError)` -> `BuilderError`, `MetaValidationError`, `OutputError`
  - Monitor: `MonitorError(Exception)` -> `MonitorInitializationError`, `MetricCollectionError`, `MonitorConnectionError` (inherits from `Exception`, not `FSMError`)
- **Constants**: Centralized in `constants.py` per package. Reasoning uses `ContextKeys` class with class-level string constants
- **Security**: Internal context key prefixes (`_`, `system_`, `internal_`, `__`). Forbidden patterns for passwords/secrets/tokens. XML tag sanitization in prompts

## FSM Definition Format (JSON, v4.1)

```json
{
  "name": "MyBot",
  "description": "What this FSM does",
  "initial_state": "start",
  "persona": "A friendly assistant",
  "states": {
    "start": {
      "id": "start",
      "description": "Brief state description",
      "purpose": "What should be accomplished in this state",
      "extraction_instructions": "What data to extract from user input",
      "response_instructions": "How to respond to the user",
      "required_context_keys": ["key1"],
      "classification_extractions": [
        {
          "field_name": "user_intent",
          "schema": {
            "intents": [
              {"name": "buy", "description": "User wants to purchase"},
              {"name": "browse", "description": "User is browsing"}
            ],
            "fallback_intent": "browse"
          },
          "confidence_threshold": 0.7,
          "required": false
        }
      ],
      "transitions": [
        {
          "target_state": "next",
          "description": "When this transition should fire",
          "priority": 100,
          "conditions": [
            {
              "description": "Human-readable condition",
              "requires_context_keys": ["key1"],
              "logic": {"==": [{"var": "key1"}, "expected_value"]},
              "evaluation_priority": 0
            }
          ]
        }
      ]
    }
  }
}
```

## Testing

```bash
pytest                                 # Run all tests (2,206)
pytest tests/test_fsm_llm/            # Core package tests (582 tests, 25 files)
pytest tests/test_fsm_llm_reasoning/  # Reasoning tests (112 tests, 7 files)
pytest tests/test_fsm_llm_workflows/  # Workflows tests (136 tests, 7 files)
pytest tests/test_fsm_llm_agents/     # Agents tests (645 tests, 27 files)
pytest tests/test_fsm_llm_monitor/    # Monitor tests (171 tests, 6 files)
pytest tests/test_fsm_llm_meta/       # Meta tests (205 tests, 11 files)
pytest tests/test_fsm_llm_regression/ # Regression tests (273 tests, 15 files)
pytest tests/test_examples/           # Example validation tests (4 tests, 2 files)
pytest -m "not slow"                  # Skip slow tests
pytest -m integration                 # Integration tests only
```

**Conventions**:
- Test files: `test_<module>.py` and `test_<module>_elaborate.py` for extended scenarios
- Test classes: `class Test<Feature>`
- Helper functions: prefixed with `_` (e.g., `_make_state()`, `_minimal_fsm_dict()`)
- **Markers**: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.examples`, `@pytest.mark.real_llm`
- **Mock LLMs**: `Mock(spec=LLMInterface)` (simple mock) and `MockLLM2Interface` (2-pass architecture) in `conftest.py`
- **Fixtures**: `sample_fsm_definition` (v3.0), `sample_fsm_definition_v2` (v4.1), `mock_llm_interface`, `mock_llm2_interface`
- **Environment**: `SKIP_SLOW_TESTS`, `TEST_REAL_LLM`, `TEST_LLM_MODEL`, `OPENAI_API_KEY`
- Workflows tests auto-skip if extension not installed

## Examples

33 examples across 8 categories:

- **basic/**: simple_greeting, form_filling, story_time
- **intermediate/**: book_recommendation, product_recommendation, adaptive_quiz
- **advanced/**: yoga_instructions (JsonLogic conditions), e_commerce (FSM stacking with push/pop), support_pipeline
- **classification/**: intent_routing, smart_helpdesk, classified_transitions
- **reasoning/**: math_tutor
- **workflows/**: order_processing
- **agents/**: react_search, hitl_approval, react_hitl_combined, plan_execute, reflexion, debate, self_consistency, rewoo, prompt_chain, evaluator_optimizer, maker_checker, classified_dispatch, classified_tools, full_pipeline, hierarchical_tools, reasoning_stacking, reasoning_tool, workflow_agent
- **meta/**: build_fsm

All examples support OpenAI and Ollama fallback. Run with: `python examples/<category>/<name>/run.py`

## Documentation

- `README.md` -- Project overview, quick start, feature summary
- `docs/quickstart.md` -- Getting started guide
- `docs/api_reference.md` -- Complete API class documentation
- `docs/architecture.md` -- System design, 2-pass flow, security, performance
- `docs/fsm_design.md` -- FSM design patterns, anti-patterns, real-world examples
- `docs/handlers.md` -- Handler development guide with 8 timing points
- `CHANGELOG.md` -- Version history (current: 0.3.0)

## Pre-commit & CI

- **Pre-commit**: trailing whitespace, EOF fixer, YAML/JSON validation, ruff (with --fix), pytest pre-push
- **CI**: GitHub Actions on push/PR to main -- tests on Python 3.10, 3.11, 3.12
- **Tox**: Multi-version testing + lint + mypy environments
