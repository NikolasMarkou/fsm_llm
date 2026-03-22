# FSM-LLM — Claude Code Instructions

## Project Overview

FSM-LLM (v0.3.0) is a Python framework for building stateful conversational AI by combining LLMs with Finite State Machines. It uses a **2-pass architecture**: Pass 1 extracts data + evaluates transitions, Pass 2 generates the response from the final state.

- **License**: GPL-3.0-or-later
- **Python**: 3.10, 3.11, 3.12
- **Core deps**: loguru, litellm (<2.0), pydantic (>=2.0), python-dotenv

## Quick Commands

```bash
make test           # pytest -v
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make type-check     # mypy across all 5 packages
make build          # python -m build (wheel + sdist)
make clean          # remove build artifacts and caches
make install-dev    # pip install -e ".[dev,workflows,classification,reasoning,agents]" + pre-commit install

# CLI tools
fsm-llm --fsm <path.json>            # Run FSM interactively
fsm-llm-visualize --fsm <path.json>  # ASCII visualization
fsm-llm-validate --fsm <path.json>   # Validate FSM definition
```

## Architecture — 2-Pass Flow

```
User Input → [Pass 1: Data Extraction (LLM)] → Context Update → Transition Evaluation (rules/LLM)
           → State Transition → [Pass 2: Response Generation (LLM)] → User Output
```

Key classes in `src/fsm_llm/`:
- **API** (`api.py`) — User-facing entry point. Methods: `from_file()`, `start_conversation()`, `converse()`, `push_fsm()`, `pop_fsm()`, `register_handler()`, `create_handler()`, `get_data()`
- **FSMManager** (`fsm.py`) — Core orchestration. Implements 2-pass processing with per-conversation thread locks
- **HandlerBuilder** (`handlers.py`) — Fluent builder: `.at()`, `.on_state()`, `.when_context_has()`, `.do()`
- **HandlerTiming** (`handlers.py`) — 8 hook points: `START_CONVERSATION`, `PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `END_CONVERSATION`, `ERROR`
- **TransitionEvaluator** (`transition_evaluator.py`) — Evaluates transitions → DETERMINISTIC, AMBIGUOUS, or BLOCKED
- **LiteLLMInterface** (`llm.py`) — LLM communication via litellm (100+ providers)
- **clean_context_keys** (`context.py`) — Stateless context cleaning (strips None values, internal key prefixes, forbidden patterns)

## Package Map

```
src/
├── fsm_llm/                     # Core framework (~8,900 LOC)
│   ├── api.py                   # API class — primary user interface
│   ├── fsm.py                   # FSMManager — state machine orchestration
│   ├── pipeline.py               # MessagePipeline — 2-pass message processing engine
│   ├── definitions.py           # Pydantic models: State, Transition, FSMDefinition, FSMContext, FSMInstance + exception hierarchy
│   ├── handlers.py              # HandlerSystem, HandlerBuilder, BaseHandler, HandlerTiming enum
│   ├── prompts.py               # Prompt builders for extraction, response gen, transition decision
│   ├── llm.py                   # LLMInterface ABC + LiteLLMInterface
│   ├── transition_evaluator.py  # Rule-based transition evaluation with JsonLogic
│   ├── expressions.py           # JsonLogic evaluator (var, and, or, ==, in, has_context, etc.)
│   ├── context.py               # Context cleaning utilities — clean_context_keys()
│   ├── runner.py                # Interactive CLI conversation runner (used by __main__)
│   ├── validator.py             # FSM structure validation
│   ├── visualizer.py            # ASCII FSM diagrams
│   ├── utilities.py             # JSON extraction with multiple fallback strategies
│   ├── constants.py             # Defaults, security patterns, internal key prefixes
│   ├── logging.py               # Loguru setup with conversation context, enable_debug_logging()
│   ├── __main__.py              # CLI entry point (run, validate, visualize modes)
│   ├── __version__.py           # Package version string
│   └── __init__.py              # Public API exports (single __all__ list)
│
├── fsm_llm_classification/      # Intent classification extension (~990 LOC)
│   ├── classifier.py            # Classifier, HierarchicalClassifier (two-stage)
│   ├── definitions.py           # ClassificationSchema, IntentDefinition, ClassificationResult
│   ├── prompts.py               # System prompt + JSON schema builders
│   ├── router.py                # IntentRouter — maps intents to handler functions
│   ├── __version__.py           # Package version string
│   └── __init__.py              # Public API exports
│
├── fsm_llm_reasoning/           # Structured reasoning engine (~4,000 LOC)
│   ├── engine.py                # ReasoningEngine — orchestrates 9 reasoning strategies via FSMs
│   ├── reasoning_modes.py       # FSM definitions for each strategy (analytical, deductive, etc.)
│   ├── handlers.py              # Reasoning-specific handlers (validation, tracing, context pruning, retry limiting)
│   ├── definitions.py           # ReasoningStep, ReasoningTrace, SolutionResult, ProblemContext
│   ├── constants.py             # ReasoningType enum, ContextKeys, OrchestratorStates, ClassifierStates
│   ├── utilities.py             # load_fsm_definition(), map_reasoning_type(), get_available_reasoning_types()
│   ├── exceptions.py            # ReasoningEngineError → ReasoningExecutionError, ReasoningClassificationError
│   ├── __main__.py              # CLI: python -m fsm_llm_reasoning
│   ├── __version__.py           # Package version string
│   └── __init__.py              # Public API exports
│
├── fsm_llm_workflows/           # Workflow orchestration engine (~2,300 LOC)
│   ├── engine.py                # WorkflowEngine — async event-driven execution
│   ├── dsl.py                   # Python DSL: create_workflow(), auto_step(), llm_step(), conversation_step(), etc.
│   ├── steps.py                 # 8 step types: AutoTransition, APICall, Condition, LLMProcessing, WaitForEvent, Timer, Parallel, ConversationStep
│   ├── definitions.py           # WorkflowDefinition with validation (reachability, cycles)
│   ├── models.py                # WorkflowStatus, WorkflowEvent, WorkflowInstance
│   ├── handlers.py              # Handler integration (engine manages operations directly)
│   ├── exceptions.py            # WorkflowError → Definition, Step, Instance, Timeout, Validation, State, Event, Resource errors
│   ├── __version__.py           # Package version string
│   └── __init__.py              # Public API exports
│
└── fsm_llm_agents/              # Agentic patterns — ReAct + HITL (~7,200 LOC)
    ├── react.py                 # ReactAgent — ReAct loop with auto-generated FSM and tool dispatch
    ├── tools.py                 # ToolRegistry + @tool decorator — tool management, prompt gen, execution
    ├── hitl.py                  # HumanInTheLoop — approval gates, escalation, confidence thresholds
    ├── handlers.py              # AgentHandlers — tool executor, iteration limiter, approval checker
    ├── fsm_definitions.py       # build_react_fsm() — auto-generates FSM from ToolRegistry
    ├── prompts.py               # Tool-aware prompt builders for think/act/conclude states
    ├── definitions.py           # ToolDefinition, ToolCall, ToolResult, AgentStep, AgentTrace, AgentConfig, AgentResult, ApprovalRequest
    ├── constants.py             # AgentStates, ContextKeys, HandlerNames, Defaults
    ├── exceptions.py            # AgentError → ToolExecutionError, ToolNotFoundError, BudgetExhaustedError, ApprovalDeniedError, AgentTimeoutError
    ├── __main__.py              # CLI: python -m fsm_llm_agents --info
    ├── __version__.py           # Package version string
    ├── adapt.py                 # ADaPTAgent — adaptive complexity with decomposition
    ├── debate.py                # DebateAgent — multi-perspective debate with judge
    ├── evaluator_optimizer.py   # EvaluatorOptimizerAgent — iterative evaluation and optimization
    ├── maker_checker.py         # MakerCheckerAgent — draft-review verification loop
    ├── orchestrator.py          # OrchestratorAgent — worker delegation and synthesis
    ├── plan_execute.py          # PlanExecuteAgent — plan decomposition and sequential execution
    ├── prompt_chain.py          # PromptChainAgent — sequential prompt pipeline with gates
    ├── reasoning_react.py       # ReasoningReactAgent — ReAct with structured reasoning (requires fsm_llm_reasoning)
    ├── reflexion.py             # ReflexionAgent — self-reflection with memory
    ├── rewoo.py                 # REWOOAgent — planning-first tool execution
    ├── self_consistency.py      # SelfConsistencyAgent — multiple samples with voting
    └── __init__.py              # Public API exports
```

## Code Conventions

- **Linting/Formatting**: ruff (E402 suppressed for `__future__` annotations)
- **Type hints**: Used throughout. mypy configured with `disallow_untyped_defs=false`
- **Models**: Pydantic v2 `BaseModel` with `model_validator` for complex validation
- **Logging**: loguru via `from fsm_llm.logging import logger`
- **Exports**: Single `__all__` list in `__init__.py` — no dynamic extend/append
- **Exceptions**:
  - Core: `FSMError` → `StateNotFoundError`, `InvalidTransitionError`, `LLMResponseError`, `TransitionEvaluationError`
  - Handlers: `HandlerSystemError` → `HandlerExecutionError`
  - Classification: `ClassificationError` → `SchemaValidationError`, `ClassificationResponseError`
  - Reasoning: `ReasoningEngineError` → `ReasoningExecutionError`, `ReasoningClassificationError`
  - Workflows: `WorkflowError` → `WorkflowDefinitionError`, `WorkflowStepError`, `WorkflowInstanceError`, `WorkflowTimeoutError`, `WorkflowValidationError`, `WorkflowStateError`, `WorkflowEventError`, `WorkflowResourceError`
  - Agents: `AgentError` → `ToolExecutionError`, `ToolNotFoundError`, `ToolValidationError`, `BudgetExhaustedError`, `ApprovalDeniedError`, `AgentTimeoutError`, `EvaluationError`, `DecompositionError`
- **Constants**: Centralized in `constants.py`. Reasoning uses `ContextKeys` class with class-level string constants
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

**Important**: State instructions use `extraction_instructions` and `response_instructions` (NOT `instructions`). The bare `instructions` field is silently ignored by Pydantic.

## Testing

```bash
pytest                              # Run all tests (1571)
pytest tests/test_fsm_llm/         # Core package tests only
pytest tests/test_fsm_llm_regression/  # Regression tests
pytest tests/test_fsm_llm_agents/  # Agents package tests
pytest -m "not slow"               # Skip slow tests
pytest -m integration              # Integration tests only
```

**Conventions**:
- Test files: `test_<module>.py` and `test_<module>_elaborate.py` for extended scenarios
- Test classes: `class Test<Feature>`
- Helper functions: prefixed with `_` (e.g., `_make_state()`, `_minimal_fsm_dict()`)
- **Markers**: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.examples`, `@pytest.mark.real_llm`
- **Mock LLMs**: `MockLLMWithResponses` (pattern-based) and `MockLLM2Interface` (2-pass architecture) in `conftest.py`
- **Fixtures**: `sample_fsm_definition` (v3.0), `sample_fsm_definition_v2` (v4.1), `mock_llm_interface`, `mock_llm2_interface`
- **Environment**: `SKIP_SLOW_TESTS`, `TEST_REAL_LLM`, `TEST_LLM_MODEL`, `OPENAI_API_KEY`
- Workflows tests auto-skip if extension not installed

## Examples

Located in `examples/` organized by difficulty:
- **basic/**: simple_greeting, form_filling, story_time
- **intermediate/**: book_recommendation, product_recommendation, adaptive_quiz
- **advanced/**: yoga_instructions (JsonLogic conditions), e_commerce (FSM stacking with push/pop), support_pipeline
- **classification/**: intent_routing, smart_helpdesk
- **reasoning/**: math_tutor
- **workflows/**: order_processing
- **agents/**: react_search, hitl_approval, react_hitl_combined, plan_execute, reflexion, debate, self_consistency, rewoo, prompt_chain, evaluator_optimizer, maker_checker, classified_dispatch, classified_tools, full_pipeline, hierarchical_tools, reasoning_stacking, reasoning_tool, workflow_agent

All examples support OpenAI and Ollama fallback. Run with: `python examples/<category>/<name>/run.py`

## Documentation

- `README.md` — Project overview, quick start, feature summary
- `LLM.md` — How the framework structures prompts for LLMs (unique to this project)
- `docs/quickstart.md` — Getting started guide
- `docs/api_reference.md` — Complete API class documentation
- `docs/architecture.md` — System design, 2-pass flow, security, performance
- `docs/fsm_design.md` — FSM design patterns, anti-patterns, real-world examples
- `docs/handlers.md` — Handler development guide with 8 timing points
- `CHANGELOG.md` — Version history (current: 0.3.0)

## Pre-commit & CI

- **Pre-commit**: trailing whitespace, EOF fixer, YAML/JSON validation, ruff (with --fix), pytest pre-push
- **CI**: GitHub Actions on push/PR to main — tests on Python 3.10, 3.11, 3.12
- **Tox**: Multi-version testing + lint + mypy environments
