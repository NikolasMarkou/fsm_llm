# fsm_llm.stdlib.reasoning — Structured Reasoning

The reasoning subpackage. Two coexisting layers:

- **λ-term factories** (M3 slice 2): 11 named factories — 9 reasoning strategies + classifier + outer orchestrator.
- **`ReasoningEngine`** (legacy class): FSM-orchestrated reasoning over the same 9 strategies. Both paths live in this subpackage; neither is deprecated.

The pre-0.7.0 `fsm_llm_reasoning` sibling shim package was deleted at 0.7.0 (I5 epoch closure). The only supported path is `from fsm_llm.stdlib.reasoning import …`.

- **Version**: 0.8.0 (synced from `fsm_llm`)
- **Extra deps**: None beyond core
- **Install**: `pip install fsm-llm[reasoning]`

**0.8.0**: reasoning factory parameters renamed from generic positional names (`prompt_a` / `prompt_b` / `prompt_c`, plus the matching `*_input_vars` / `*_schema_ref` kwargs) to descriptive names matching each factory's bind_names. E.g. `analytical_term(decomposition_prompt, analysis_prompt, integration_prompt)` instead of `analytical_term(prompt_a, prompt_b, prompt_c)`. Every reasoning factory was migrated. See `docs/migration_0.7_to_0.8.md` for the per-factory mapping.

## Layer 1 — λ-term Factories (`lam_factories.py`)

Each factory takes prompt strings + env-var names; builds `Leaf` nodes internally; returns a closed `Term`. Imports only from `fsm_llm.runtime` (purity invariant; AST-walk test enforces).

| Factory | Leaves | Shape |
|---------|-------:|-------|
| `analytical_term` | 3 | classify → decompose → synthesise |
| `deductive_term` | 3 | premise → derive → conclude |
| `inductive_term` | 3 | observe → pattern → generalise |
| `abductive_term` | 3 | observe → hypothesise → infer |
| `analogical_term` | 3 | source → mapping → target |
| `creative_term` | 3 | brainstorm → evaluate → refine |
| `critical_term` | 3 | claim → evidence → judge |
| `hybrid_term` | 4 | classify → strategy A → strategy B → synthesise |
| `calculator_term` | 2 | parse → compute |
| `classifier_term` | 4 | domain → structure → needs → recommend |
| `solve_term` | (orchestrator) | Host-callable wrapping `classifier_term` + dispatch + validation + final synthesis |

A private `_chain(*pairs) -> Term` helper folds `[(name, leaf), …]` into a nested let-chain (D-S2-003). Per-subpackage by design — purity is per-module.

```python
from fsm_llm.stdlib.reasoning import analytical_term

term = analytical_term(
    classify_prompt="Classify the problem domain: {problem}",
    decompose_prompt="Break down: {problem}",
    synthesise_prompt="Synthesise an answer using the decomposition.",
    problem_var="problem",
)
ex.run(term, env={"problem": "What forces act on a falling apple?"})
assert ex.oracle_calls == 3   # let-chain strict equality
```

The `solve_term` orchestrator picks a strategy at runtime via `classifier_term` and dispatches:

```python
from fsm_llm.stdlib.reasoning import solve_term

solver = solve_term(...)   # See module docstring for full kwargs
ex.run(solver, env={"problem": "...", "dispatch": strategy_dispatcher})
```

## Layer 2 — `ReasoningEngine` (legacy class)

```python
from fsm_llm.stdlib.reasoning import ReasoningEngine, ReasoningType

engine = ReasoningEngine(model="openai/gpt-4o-mini")
solution, trace = engine.solve_problem("What is the probability of rolling two sixes?")
print(solution)
for step in trace.steps:
    print(step.reasoning_type, step.output)
```

`ReasoningType` enum (`constants.py`):
- `SIMPLE_CALCULATOR` — direct arithmetic
- `ANALYTICAL` — break down complex systems
- `DEDUCTIVE` — derive specific from general
- `INDUCTIVE` — find patterns from examples
- `ABDUCTIVE` — best-explanation inference
- `ANALOGICAL` — analogy-based mapping
- `CREATIVE` — generate novel solutions
- `CRITICAL` — evaluate arguments
- `HYBRID` — combine approaches

## File Map

```
reasoning/
├── lam_factories.py      # M3 slice 2 — 11 named term factories + private _chain helper
├── engine.py             # ReasoningEngine — orchestrator over 9 strategies (FSM-orchestrated)
├── reasoning_modes.py    # ALL_REASONING_FSMS dict (one FSM per ReasoningType)
├── definitions.py        # ProblemContext, ReasoningStep, ReasoningTrace, ReasoningClassificationResult, SolutionResult, ValidationResult
├── handlers.py           # Validate, trace, prune handlers
├── prompts.py            # Strategy-specific prompt templates
├── utilities.py          # get_available_reasoning_types() + helpers
├── constants.py          # ReasoningType enum, ContextKeys, defaults (MAX_RETRIES=3, MAX_TOTAL_ITERATIONS=50)
├── exceptions.py         # ReasoningEngineError, ReasoningExecutionError, ReasoningClassificationError
├── __version__.py
└── __init__.py
```

## Public Exports (`__init__.py`)

```python
# Class layer
ReasoningEngine, ReasoningType,
ReasoningStep, ReasoningTrace, ValidationResult, ReasoningClassificationResult,
ProblemContext, SolutionResult, get_available_reasoning_types,

# Exceptions
ReasoningEngineError, ReasoningExecutionError, ReasoningClassificationError,

# λ-factories (M3 slice 2)
analytical_term, deductive_term, inductive_term, abductive_term, analogical_term,
creative_term, critical_term, hybrid_term, calculator_term,
classifier_term, solve_term,
```

## Theorem-2 Form

All 11 factories are **let-chain strict** (no Fix, no Case): `oracle_calls == sum(leaves)`. Per LESSONS.md verified across 17 shape-equivalence unit tests + 5-cell live smoke + 10-cell bench scorecard (`evaluation/m3_slice2_reasoning_scorecard.json`, all `theorem2_holds=true`).

## Constants (`constants.py`)

- `MAX_RETRIES = 3`, `MAX_TOTAL_ITERATIONS = 50` (engine defaults)
- `ContextKeys` class with 220+ keys for problem/strategy/output threading

## Testing

```bash
pytest tests/test_fsm_llm_reasoning/                  # 134 tests (engine + factories)
pytest tests/test_fsm_llm_reasoning/test_lam_factories.py  # M3 slice 2 unit tests
TEST_REAL_LLM=1 pytest -m real_llm tests/test_fsm_llm_reasoning/  # Live smokes
```

Bench: `python scripts/bench_reasoning_factories.py` produces `evaluation/m3_slice2_reasoning_scorecard.json` with per-cell `oracle_calls == leaves` evidence.

## Code Conventions

- **Stdlib purity**: `lam_factories.py` imports only from `fsm_llm.runtime`.
- Class-based engine uses `LiteLLMInterface` directly; factories use `Leaf` + `LiteLLMOracle`.
- Pick the layer that matches your need: factories compose into bigger λ-programs and have closed-form cost; the class engine handles trace/validate/prune lifecycle.
