# FSM-LLM -- Claude Code Instructions

## Project Overview

FSM-LLM (v0.5.0) is a Python framework for building stateful conversational AI by combining LLMs with Finite State Machines. It uses a **2-pass architecture**: Pass 1 extracts data + evaluates transitions, Pass 2 generates the response from the final state.

- **License**: GPL-3.0-or-later
- **Python**: 3.10, 3.11, 3.12
- **Core deps**: loguru, litellm (>=1.82,<2.0, excluding 1.82.7/1.82.8), pydantic (>=2.0), python-dotenv
- **Virtual environment**: Always use `.venv` -- run commands with `.venv/bin/python` or activate first

## Quick Commands

```bash
make test           # pytest -v (5,240 tests)
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make type-check     # mypy across all 6 packages
make build          # python -m build (wheel + sdist)
make clean          # remove build artifacts and caches
make coverage       # pytest with coverage report
make install-dev    # pip install -c constraints.txt -e ".[dev,workflows,reasoning,agents,monitor,harness]" + pre-commit install
make audit          # audit site-packages for suspicious .pth files

# CLI tools
fsm-llm --fsm <path.json>            # Run FSM interactively
fsm-llm-visualize --fsm <path.json>  # ASCII visualization
fsm-llm-validate --fsm <path.json>   # Validate FSM definition
fsm-llm-monitor                      # Launch web monitoring dashboard
fsm-llm-meta                         # Interactive artifact builder (routes to fsm_llm_agents.meta_cli)
fsm-llm-harness <new|resume|status|validate|close>  # Iterative-planner protocol harness
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
├── fsm_llm/              # Core framework (23 files)
├── fsm_llm_reasoning/    # Structured reasoning engine (10 files)
├── fsm_llm_workflows/    # Workflow orchestration engine (9 files, incl. dependency resolver)
├── fsm_llm_agents/       # Agentic patterns -- 12 patterns + swarm, graph, MCP, A2A, SOPs, semantic tools + meta builder (49 files)
├── fsm_llm_monitor/      # Web-based monitoring dashboard (11 files, incl. OTEL exporter + static/)
└── fsm_llm_harness/      # Iterative-planner protocol harness -- 6-state FSM, disk-derived gates, autonomy leash (14 files)
```

**Optional extras** (beyond core):

| Extra | Command | Dependencies |
|-------|---------|-------------|
| `reasoning` | `pip install fsm-llm[reasoning]` | None |
| `agents` | `pip install fsm-llm[agents]` | None |
| `workflows` | `pip install fsm-llm[workflows]` | None |
| `harness` | `pip install fsm-llm[harness]` | `fsm-llm[agents]` (no third-party deps of its own) |
| `monitor` | `pip install fsm-llm[monitor]` | fastapi, uvicorn, jinja2 |
| `mcp` | `pip install fsm-llm[mcp]` | mcp (>=1.0.0) |
| `otel` | `pip install fsm-llm[otel]` | opentelemetry-api, opentelemetry-sdk (>=1.20.0) |
| `a2a` | `pip install fsm-llm[a2a]` | httpx (>=0.24.0) |
| `all` | `pip install fsm-llm[all]` | All of the above |

Each sub-package has its own `CLAUDE.md` with detailed file maps, key classes, and API reference.

**`fsm_llm_harness` in one paragraph.** It runs the iterative-planner protocol as
a real FSM: 6 states (EXPLORE / PLAN / EXECUTE / REFLECT / PIVOT / CLOSE), 9
transitions, and hard gates that are JsonLogic `TransitionCondition` terms -- so a
gated edge is DETERMINISTIC or BLOCKED, never an LLM judgement call. Its memory is
a directory of Markdown artifacts (`state.md`, `plan.md`, `decisions.md`,
`findings/`, ...) with an ownership table saying which role may write which file,
and its gate values are **derived from the filesystem**, not read from the model's
report: `findings_count` counts non-empty `findings/*.md`, and a dispatch claiming
a write must show a tool call whose target now carries bytes. The autonomy leash
halts at exactly 2 fix attempts and cannot be reset from inside an approving
callback. Measured on `ollama_chat/qwen3.5:4b`: the single-state model bars are
MET for the first time after a bench-measured structural fix (driver-assigned
EXECUTE write targets: B0 content-match 2/40 -> B1 **40/40**, Fisher p=1.6e-20,
n=40/arm blocks committed under `scripts/bench_data/`; armed standing bar L4
write tool 5/5, bytes 5/5, strict content-hash 4/5 vs >=4/5; L5 5/5) -- but the
first graded end-to-end criterion on REAL workers (L6, n=3) measured **0/3 NOT
MET** (two honest explore-cap halts, one slugless PLAN stall; verified writes
3/3). The harness is not production-ready and a 4B model is not claimed to
drive it unattended to a useful result. See `src/fsm_llm_harness/CLAUDE.md` for
the full reference, including what is measured and what is not.

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
  - Harness: `HarnessError(FSMError)` -> `HarnessArtifactError`, `HarnessOwnershipError`, `HarnessReentrancyError`, `HarnessConfinementError`
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
pytest                                 # Run all tests (5,240 collected)
pytest tests/test_fsm_llm/            # Core package tests (1,305 tests)
pytest tests/test_fsm_llm_reasoning/  # Reasoning tests (112 tests)
pytest tests/test_fsm_llm_workflows/  # Workflows tests (137 tests)
pytest tests/test_fsm_llm_agents/     # Agents tests (968 tests)
pytest tests/test_fsm_llm_monitor/    # Monitor tests (270 tests)
pytest tests/test_fsm_llm_meta/       # Meta tests (213 tests)
pytest tests/test_fsm_llm_harness/    # Harness tests (1,840 tests)
pytest tests/test_fsm_llm_regression/ # Regression tests (282 tests)
pytest tests/test_examples/           # Example validation tests (43 tests)
# The 9 suites above sum to 5,170. The remaining 70 are three root-level files:
#   tests/test_integration_ollama.py (12), tests/test_packaging.py (24)
#   and tests/test_harness_bench.py (34)
pytest -m "not slow"                  # Skip slow tests
pytest -m integration                 # Integration tests only
```

Counts measured with `pytest --collect-only` at commit `89f3b9c`; the full run is
5,164 passed / 26 skipped / 2 xfailed in ~247s.

**Conventions**:
- Test files: `test_<module>.py` and `test_<module>_elaborate.py` for extended scenarios
- Test classes: `class Test<Feature>`
- Helper functions: prefixed with `_` (e.g., `_make_state()`, `_minimal_fsm_dict()`)
- **Markers**: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.examples`, `@pytest.mark.real_llm`
- **Mock LLMs**: `Mock(spec=LLMInterface)` (simple mock) and `MockLLM2Interface` (2-pass architecture) in `conftest.py`
- **Fixtures**: `sample_fsm_definition` (v3.0), `sample_fsm_definition_v2` (v4.1), `mock_llm_interface`, `mock_llm2_interface`
- **Environment**: `SKIP_SLOW_TESTS`, `TEST_REAL_LLM`, `TEST_LLM_MODEL`, `OPENAI_API_KEY`, `FSM_LLM_HARNESS_LIVE`
- Workflows tests auto-skip if extension not installed
- `tests/test_packaging.py` derives the package list from the filesystem
  (`src/*/__init__.py`) and asserts every package appears in all 14 build/CI
  slots, so a seventh package that misses a slot fails loudly instead of silently
- Harness live tests (`test_live_ollama.py`) are double-gated on
  `FSM_LLM_HARNESS_LIVE=1` **and** a reachable Ollama, env term checked first so
  `make test` pays no socket timeout at collection

## Examples

100 examples across 8 categories:

- **basic/** (14): simple_greeting, form_filling, story_time, multi_turn_extraction, event_registration, insurance_claim, job_application, medical_intake, pet_adoption, rental_application, restaurant_reservation, scholarship_application, tech_support_intake, travel_booking
- **intermediate/** (3): book_recommendation, product_recommendation, adaptive_quiz
- **advanced/** (17): yoga_instructions, e_commerce (FSM stacking), support_pipeline, handler_hooks, concurrent_conversations, context_compactor, multi_level_stack, budget_review, compliance_audit, customer_feedback_pipeline, employee_onboarding, incident_response, loan_assessment, medical_triage, project_planning, quality_inspection, vendor_evaluation
- **classification/** (4): intent_routing, smart_helpdesk, classified_transitions, multi_intent
- **reasoning/** (1): math_tutor
- **workflows/** (8): order_processing, agent_workflow_chain, parallel_steps, conditional_branching, workflow_agent_loop, loan_processing, release_management, customer_onboarding
- **agents/** (48): react_search, hitl_approval, react_hitl_combined, plan_execute, reflexion, debate, self_consistency, rewoo, prompt_chain, evaluator_optimizer, maker_checker, classified_dispatch, classified_tools, full_pipeline, hierarchical_tools, reasoning_stacking, reasoning_tool, workflow_agent, adapt, agent_as_tool, concurrent_react, memory_agent, multi_tool_recovery, orchestrator, skill_loader, structured_output, tool_decorator, debate_with_tools, reflexion_code_gen, orchestrator_specialist, pipeline_review, adapt_with_memory, rewoo_multi_step, eval_opt_structured, plan_execute_recovery, consistency_with_tools, maker_checker_code, hierarchical_orchestrator, agent_memory_chain, react_structured_pipeline, multi_debate_panel, legal_document_review, investment_portfolio, security_audit, medical_literature, architecture_review, supply_chain_optimizer, regulatory_compliance
- **meta/** (5): build_fsm, build_workflow, build_agent, meta_review_loop, meta_from_spec

All examples support OpenAI and Ollama fallback. Run with: `python examples/<category>/<name>/run.py`

**IMPORTANT**: Do NOT modify existing examples unless explicitly asked by the user. Examples serve as stable evaluation baselines.

### Evaluation

Automated evaluation via `scripts/eval.py` runs all examples in parallel and produces scorecards. Current baseline: **95.3% health score** (N=3 median, 101 examples) on `ollama_chat/qwen3.5:4b` — Run 006, commit `2df048f`. **This baseline is STALE and must be re-run before it is trusted.** The F-01..F-24 remediation plan (`plans/plan-2026-07-19T191147-4b664252`) changed prompt CONTENT in `prompts.py`, `context.py` and `constants.py` — history capping in `FieldExtractionPromptBuilder`, nested-dict security filtering, and the forbidden-key regexes — and none of the gates available to that plan can observe prompt effects (the eval suite was not run; the fast gate mocks the LLM). See `EVALUATE.md` for methodology and results history. (The heuristic overstates ~15pp; pair with manual log inspection. The ~5 agent score-1s per run are non-deterministic `--workers 4` timeouts, not regressions.)

Harness capability benches run via `scripts/harness_bench.py` (pre-registered fixed-n blocks, 6-field manifests, append-only raw jsonl, Wilson CI + Fisher exact, per-row seeds); committed blocks live under `scripts/bench_data/` (e.g. `l4-execute-write/{B0,B1}`, n=40/arm).

## Documentation

- `README.md` -- Project overview, quick start, feature summary
- `docs/quickstart.md` -- Getting started guide
- `docs/api_reference.md` -- Complete API class documentation
- `docs/architecture.md` -- System design, 2-pass flow, security, performance
- `docs/fsm_design.md` -- FSM design patterns, anti-patterns, real-world examples
- `docs/handlers.md` -- Handler development guide with 8 timing points
- `CHANGELOG.md` -- Version history (current: 0.5.0)

## Pre-commit & CI

- **Pre-commit**: trailing whitespace, EOF fixer, YAML/JSON validation, ruff (with --fix), pytest pre-push
- **CI**: GitHub Actions on push/PR to main -- tests on Python 3.10, 3.11, 3.12
- **Tox**: Multi-version testing + lint + mypy environments
