# FSM-LLM -- Claude Code Instructions

## Project Overview

FSM-LLM (v0.3.0) is a Python framework for building stateful conversational AI by combining LLMs with Finite State Machines. It uses a **2-pass architecture**: Pass 1 extracts data + evaluates transitions, Pass 2 generates the response from the final state.

- **License**: GPL-3.0-or-later
- **Python**: 3.10, 3.11, 3.12
- **Core deps**: loguru, litellm (>=1.82,<2.0, excluding 1.82.7/1.82.8), pydantic (>=2.0), python-dotenv
- **Virtual environment**: Always use `.venv` -- run commands with `.venv/bin/python` or activate first

## Quick Commands

```bash
make test           # pytest -v (2,303 tests)
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
- **LiteLLMInterface** (`llm.py`) -- LLM communication via litellm (100+ providers). Methods: `generate_response`, `extract_field`, `generate_response_stream`. Supports schema enforcement via `response_format`
- **clean_context_keys** (`context.py`) -- Stateless context cleaning (strips None values, internal key prefixes, forbidden patterns)
- **WorkingMemory** (`memory.py`) -- Structured working memory with named buffers (core, scratch, environment, reasoning) for organizing agent context
- **SessionStore** / **FileSessionStore** (`session.py`) -- Abstract session persistence interface + file-based implementation with atomic writes. Enables conversation state save/restore across process restarts

## Package Map

```
src/
├── fsm_llm/              # Core framework (20 files)
├── fsm_llm_reasoning/    # Structured reasoning engine (8 files)
├── fsm_llm_workflows/    # Workflow orchestration engine (8 files) + dependency resolver
├── fsm_llm_agents/       # Agentic patterns -- 12 patterns + swarm, graph, MCP, A2A, SOPs, semantic tools + meta builder (42 files)
└── fsm_llm_monitor/      # Web-based monitoring dashboard (9 files + OTEL exporter + static/)
```

**Optional extras** (beyond core):

| Extra | Command | Dependencies |
|-------|---------|-------------|
| `reasoning` | `pip install fsm-llm[reasoning]` | None |
| `agents` | `pip install fsm-llm[agents]` | None |
| `workflows` | `pip install fsm-llm[workflows]` | None |
| `monitor` | `pip install fsm-llm[monitor]` | fastapi, uvicorn, jinja2 |
| `mcp` | `pip install fsm-llm[mcp]` | mcp (>=1.0.0) |
| `otel` | `pip install fsm-llm[otel]` | opentelemetry-api, opentelemetry-sdk (>=1.20.0) |
| `a2a` | `pip install fsm-llm[a2a]` | httpx (>=0.24.0) |
| `all` | `pip install fsm-llm[all]` | All of the above |

Each sub-package has its own `CLAUDE.md` with detailed file maps, key classes, and API reference.

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
  "initial_state": "start",
  "persona": "A friendly assistant",
  "states": {
    "start": {
      "id": "start",
      "description": "Brief state description",
      "purpose": "What should be accomplished",
      "extraction_instructions": "What data to extract",
      "response_instructions": "How to respond",
      "required_context_keys": ["key1"],
      "classification_extractions": [
        {
          "field_name": "user_intent",
          "schema": {
            "intents": [{"name": "buy", "description": "User wants to purchase"}],
            "fallback_intent": "browse"
          },
          "confidence_threshold": 0.7
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
              "logic": {"==": [{"var": "key1"}, "expected_value"]}
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
pytest                                 # Run all tests (2,349)
pytest tests/test_fsm_llm/            # Core package tests (643 tests)
pytest tests/test_fsm_llm_reasoning/  # Reasoning tests (112 tests)
pytest tests/test_fsm_llm_workflows/  # Workflows tests (136 tests)
pytest tests/test_fsm_llm_agents/     # Agents tests (706 tests)
pytest tests/test_fsm_llm_monitor/    # Monitor tests (217 tests)
pytest tests/test_fsm_llm_meta/       # Meta tests (205 tests)
pytest tests/test_fsm_llm_regression/ # Regression tests (275 tests)
pytest tests/test_examples/           # Example validation tests (43 tests)
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

80 examples across 8 categories:

- **basic/** (4): simple_greeting, form_filling, story_time, multi_turn_extraction
- **intermediate/** (3): book_recommendation, product_recommendation, adaptive_quiz
- **advanced/** (7): yoga_instructions, e_commerce (FSM stacking), support_pipeline, handler_hooks, concurrent_conversations, context_compactor, multi_level_stack
- **classification/** (4): intent_routing, smart_helpdesk, classified_transitions, multi_intent
- **reasoning/** (1): math_tutor
- **workflows/** (8): order_processing, agent_workflow_chain, parallel_steps, conditional_branching, workflow_agent_loop, loan_processing, release_management, customer_onboarding
- **agents/** (48): react_search, hitl_approval, react_hitl_combined, plan_execute, reflexion, debate, self_consistency, rewoo, prompt_chain, evaluator_optimizer, maker_checker, classified_dispatch, classified_tools, full_pipeline, hierarchical_tools, reasoning_stacking, reasoning_tool, workflow_agent, adapt, agent_as_tool, concurrent_react, memory_agent, multi_tool_recovery, orchestrator, skill_loader, structured_output, tool_decorator, debate_with_tools, reflexion_code_gen, orchestrator_specialist, pipeline_review, adapt_with_memory, rewoo_multi_step, eval_opt_structured, plan_execute_recovery, consistency_with_tools, maker_checker_code, hierarchical_orchestrator, agent_memory_chain, react_structured_pipeline, multi_debate_panel, legal_document_review, investment_portfolio, security_audit, medical_literature, architecture_review, supply_chain_optimizer, regulatory_compliance
- **meta/** (5): build_fsm, build_workflow, build_agent, meta_review_loop, meta_from_spec

All examples support OpenAI and Ollama fallback. Run with: `python examples/<category>/<name>/run.py`

**IMPORTANT**: Do NOT modify existing examples unless explicitly asked by the user. Examples serve as stable evaluation baselines.

### Evaluation

Automated evaluation via `scripts/eval.py` runs all examples in parallel and produces scorecards. Current baseline: **98.8% health score** on `ollama_chat/qwen3.5:4b`. See `EVALUATE.md` for methodology and results history.

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
