# fsm_llm.stdlib.workflows — Workflow Orchestration

The workflows subpackage. Two coexisting layers:

- **λ-term factories** (M3 slice 3): 5 composition combinators — `linear_term`, `branch_term`, `switch_term`, `parallel_term`, `retry_term`. Pure factories returning `Term`.
- **`WorkflowEngine` + Python DSL** (legacy class layer): async event-driven engine with 11 step types (`WorkflowBuilder`, `auto_step`, `api_step`, `llm_step`, `parallel_step`, `retry_step`, …). Both paths coexist; neither is deprecated.

The legacy `fsm_llm_workflows` import path resolves here via `sys.modules` shim.

- **Version**: 0.6.0 (synced from `fsm_llm`)
- **Extra deps**: None beyond core
- **Install**: `pip install fsm-llm[workflows]`

## Layer 1 — λ-term Factories (`lam_factories.py`)

Each factory takes sub-`Term`s and returns a `Term`. Imports only from `fsm_llm.runtime` (purity invariant; AST-walk test enforces).

| Factory | Body | Theorem-2 form |
|---------|------|----------------|
| `linear_term(*pairs)` | Folds `[(name, term), …]` into nested `let_` chain | strict — `oracle_calls == sum(leaves)` |
| `branch_term(predicate, then_term, else_term)` | `case_(predicate(...), {"true": then_term}, default=else_term)` | runtime arm-only — `oracle_calls == leaves(taken_arm)` |
| `switch_term(classifier, branches: dict[str, Term], default)` | `case_(classifier_call, branches, default)` | runtime arm-only |
| `parallel_term(*branches)` | `let_(...) → reduce_(fmap(identity, branch_outputs))` (fan-out + reduce) | strict — sum across all branches |
| `retry_term(body, max_attempts, success_predicate)` | `fix(λself. λx. let_("attempt", body(x), case_(success(attempt), {"true": attempt}, default=self(x))))` | bounded recursion; `0` if body is host-callable |

This is the **first stdlib slice exercising `case_` (Case node) + `fix` (bounded recursion) + `fmap`/`reduce_` (parallel idiom)** outside `long_context`. Private `_chain(*pairs) -> Term` helper duplicated from slice 2 by design — purity is per-module.

```python
from fsm_llm.stdlib.workflows import linear_term, switch_term

# Linear pipeline
pipeline = linear_term(
    ("validated", validate_leaf),
    ("scored", score_leaf),
    ("answered", answer_leaf),
)

# Switch on a runtime classifier
router = switch_term(
    classifier=classifier_term(...),
    branches={"billing": billing_pipeline, "support": support_pipeline},
    default=fallback_pipeline,
)
```

## Layer 2 — `WorkflowEngine` + Python DSL

```python
from fsm_llm.stdlib.workflows import (
    create_workflow, auto_step, llm_step, conversation_step,
    api_step, condition_step, parallel_step, retry_step,
)

workflow = (
    create_workflow("order_pipeline")
    .add(auto_step("validate", action=validate_order))
    .add(llm_step("summarize", prompt="Summarize: {order}"))
    .add(conversation_step("support", fsm_file="support.json"))
    .build()
)
```

Async event-driven engine. Status lifecycle: `PENDING → RUNNING → COMPLETED | FAILED`.

## File Map

```
workflows/
├── lam_factories.py        # M3 slice 3 — 5 named term factories + private _chain helper
├── engine.py               # WorkflowEngine — async event-driven (Timer, instance state, retries)
├── dsl.py                  # WorkflowBuilder + 11 step factories + 3 pattern helpers
├── steps.py                # 11 WorkflowStep subclasses (async execute)
│                          #   AutoTransitionStep, AgentStep, APICallStep, ConditionStep,
│                          #   LLMProcessingStep, WaitForEventStep, TimerStep, ParallelStep,
│                          #   RetryStep, SwitchStep, ConversationStep
├── definitions.py          # WorkflowDefinition + WorkflowValidator
├── dependency_resolver.py  # DependencyResolver — Kahn topological sort
├── models.py               # WorkflowInstance, WorkflowStatus, WorkflowEvent, WorkflowStepResult, EventListener, WaitEventConfig
├── exceptions.py           # WorkflowError + 8 subclasses
├── __version__.py
└── __init__.py             # 48 + 5 (λ factories) public exports
```

## Public Exports (highlights)

```python
# Class layer — engine + DSL
WorkflowEngine, WorkflowDefinition, WorkflowValidator, WorkflowBuilder,
create_workflow, workflow_builder, linear_workflow, conditional_workflow, event_driven_workflow,
auto_step, api_step, condition_step, llm_step, wait_event_step, timer_step,
parallel_step, conversation_step, agent_step, retry_step, switch_step,

# Steps + models
WorkflowStep, AutoTransitionStep, APICallStep, ConditionStep, LLMProcessingStep,
WaitForEventStep, TimerStep, ParallelStep, RetryStep, SwitchStep, ConversationStep, AgentStep,
WorkflowInstance, WorkflowStatus, WorkflowEvent, WorkflowStepResult, EventListener, WaitEventConfig,
DependencyResolver, Timer,

# λ-factories (M3 slice 3)
linear_term, branch_term, switch_term, parallel_term, retry_term,
```

## Step Types (`steps.py`) — for the class layer

| Step type | Purpose |
|-----------|---------|
| `AutoTransitionStep` | Synchronous Python callable |
| `AgentStep` | Wraps an agent (e.g. ReactAgent) |
| `APICallStep` | HTTP request with retries |
| `ConditionStep` | Boolean branch on context |
| `LLMProcessingStep` | LLM call with templated prompt |
| `WaitForEventStep` | Pause until external event |
| `TimerStep` | Delay |
| `ParallelStep` | Fan-out, await all |
| `RetryStep` | Retry inner step on failure |
| `SwitchStep` | N-way branch on context value |
| `ConversationStep` | Embed a sub-FSM |

## Theorem-2 Forms (per factory shape)

| Factory | Form | Notes |
|---------|------|-------|
| `linear_term` | strict | `oracle_calls == sum(leaves)` |
| `branch_term`, `switch_term` | runtime arm-only | bench records both `actual_oracle_calls` and `static_leaf_upper_bound` + `actual_le_upper_bound` flag |
| `parallel_term` | strict | Sum across all parallel branches |
| `retry_term` | bounded recursion | `0` when body is host-callable (no leaves) |

5-cell bench (`evaluation/m3_slice3_workflow_scorecard.json`) confirms 5/5 `theorem2_holds=true` in 15s wall on `ollama_chat/qwen3.5:4b`.

## Exceptions

```
WorkflowError
├── WorkflowDefinitionError
├── WorkflowStepError
├── WorkflowInstanceError
├── WorkflowTimeoutError
├── WorkflowValidationError
├── WorkflowStateError
├── WorkflowEventError
└── WorkflowResourceError
```

## Testing

```bash
pytest tests/test_fsm_llm_workflows/                  # 155 tests (engine + factories)
pytest tests/test_fsm_llm_workflows/test_lam_factories.py  # M3 slice 3 unit tests + AST-walk purity
TEST_REAL_LLM=1 pytest -m real_llm tests/test_fsm_llm_workflows/  # Live smokes
```

Bench: `python scripts/bench_workflow_factories.py` → `evaluation/m3_slice3_workflow_scorecard.json`.

## Code Conventions

- **Stdlib purity**: `lam_factories.py` imports only from `fsm_llm.runtime`.
- **Async patterns**: engine uses `asyncio.gather` for parallel steps and `loop.run_in_executor` for sync callables.
- **Choosing layers**: factories compose into bigger λ-programs and have planner-bounded cost; engine + DSL provides async event handling, retries, and external integrations out of the box.
