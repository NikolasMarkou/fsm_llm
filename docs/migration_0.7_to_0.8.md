# Migration guide — fsm-llm 0.7.x → 0.8.0

`0.8.0` is the deep-cleanup release. Eight back-compat surfaces are hard-removed (no warn cycle), four `0.7.0`-deferred refactors land, the `Program` facade is tightened, and every `*PromptBuilder` shares dedupe'd helpers. Per the project's no-back-compat directive on this release, callers must update; the canonical paths below have been valid since `0.7.0` (or earlier) and are unchanged in `0.8.0`.

## TL;DR

| If you used | At 0.8.0 raises | Use instead |
|---|---|---|
| `from fsm_llm import Handler` | `ImportError` | `from fsm_llm import FSMHandler` |
| `from fsm_llm import LLMInterface` | `ImportError` | `from fsm_llm.runtime._litellm import LLMInterface` |
| `from fsm_llm import BUILTIN_OPS` | `ImportError` | `from fsm_llm.runtime import BUILTIN_OPS` |
| `fsm_llm.has_workflows()` (and 5 siblings) | `AttributeError` | drop the call — stdlib subpackages always ship since `0.7.0` |
| `from fsm_llm.dialog.definitions import FSMError` | `ImportError` | `from fsm_llm.types import FSMError` (and 14 sibling names) |
| `state._emit_response_leaf_for_non_cohort = False` | `AttributeError` | drop the line — non-cohort states always emit a Leaf now |
| `Program(term=t, _api=…)` | `TypeError` | use `Program.from_fsm(definition, …)` |
| `Program.from_fsm(d, **api_kwargs)` | mostly works; unsupported kwargs raise | pass kwargs by their explicit names (see below) |
| `analytical_term(prompt_a, prompt_b, prompt_c)` | `TypeError` | `analytical_term(decomposition_prompt, analysis_prompt, integration_prompt)` (and 10 sibling factories) |

The migration guide for `0.6.x → 0.7.0` is archived at [`docs/archive/migration_0.6_to_0.7.md`](archive/migration_0.6_to_0.7.md).

---

## 1. `Handler` alias removed

`Handler` was a back-compat alias for the `FSMHandler` Protocol added during the R11 surface promotion. `FSMHandler` is now the only protocol name.

**Before (0.7.x):**
```python
from fsm_llm import Handler

def my_handler(ctx) -> dict:
    return {}
```

**After (0.8.0):**
```python
from fsm_llm import FSMHandler  # the Protocol — same object as the old alias
```

Most user code never references the Protocol directly; you build handlers via `HandlerBuilder` / `create_handler` and that surface is unchanged.

---

## 2. Top-level `LLMInterface` re-export removed

The `LLMInterface` ABC lives in the private `runtime._litellm` adapter (D-009 closure formalised at `0.7.0`). The convenience top-level re-export was removed at `0.8.0`.

**Before:**
```python
from fsm_llm import LLMInterface
```

**After:**
```python
from fsm_llm.runtime._litellm import LLMInterface
```

Most callers wire LLMs via `LiteLLMOracle(litellm_iface)` or `Program.from_fsm(model="…")`; direct `LLMInterface` references are rare.

---

## 3. Top-level `BUILTIN_OPS` re-export removed

`BUILTIN_OPS` is the closed kernel registry of `Combinator` operators. It was never meant to be a user-extension point — new operators bind through the executor env. The top-level re-export was removed at `0.8.0`; the canonical `runtime` path is unchanged.

**Before:**
```python
from fsm_llm import BUILTIN_OPS
```

**After:**
```python
from fsm_llm.runtime import BUILTIN_OPS  # if you really need it
```

---

## 4. Extension-check helpers removed

Pre-`0.7.0` the helpers `has_workflows`, `has_reasoning`, `has_agents` and their `get_*` siblings gated on whether the optional sibling-shim packages (`fsm_llm_{workflows,reasoning,agents}`) were installed. The shim packages were deleted at `0.7.0` (I5 epoch closure), making the helpers no-ops. At `0.8.0` they're gone.

**Before:**
```python
from fsm_llm import has_workflows, get_workflows

if has_workflows():
    workflows = get_workflows()
    workflows.linear_term(...)
```

**After:**
```python
from fsm_llm.stdlib import workflows  # always works — ships with core
workflows.linear_term(...)
# or:
from fsm_llm import linear_term  # top-level convenience export
```

`get_version_info()['features']` keeps the per-feature flags (`workflows`, `reasoning`, `agents`) as hard-coded `True` for back-compat with callers that branch on the dict.

---

## 5. `dialog/definitions.py` type re-exports removed

At `0.7.0` the `FSMError` hierarchy and the runtime-touching Pydantic models moved to a neutral `fsm_llm.types` layer. `dialog/definitions.py` kept a back-compat re-export block for byte-equivalent identity. At `0.8.0` the re-export block is gone — direct imports are required.

**Before:**
```python
from fsm_llm.dialog.definitions import FSMError, ResponseGenerationResponse
```

**After:**
```python
from fsm_llm.types import FSMError, ResponseGenerationResponse
```

The 15 names that moved:

* **Exceptions**: `FSMError`, `StateNotFoundError`, `InvalidTransitionError`, `LLMResponseError`, `TransitionEvaluationError`, `ClassificationError`, `SchemaValidationError`, `ClassificationResponseError`.
* **Models**: `DataExtractionResponse`, `ResponseGenerationRequest`, `ResponseGenerationResponse`, `FieldExtractionRequest`, `FieldExtractionResponse`.
* **Enums**: `LLMRequestType`, `TransitionEvaluationResult`.

Everything else in `dialog/definitions.py` (`State`, `Transition`, `FSMDefinition`, `FSMContext`, `FSMInstance`, `Conversation`, classification/extraction config models, `TransitionOption`, `TransitionEvaluation`) is dialog-domain and stays exactly where it was. Top-level imports `from fsm_llm import FSMError`, `from fsm_llm import State`, etc. continue to work via the `fsm_llm/__init__.py` re-exports.

---

## 6. `State._emit_response_leaf_for_non_cohort` gate removed

The private gate field on `State` controlled the M3a/M3b/M3c rollout of the response-Leaf emission for non-cohort FSM states. It defaulted to `True` at `0.7.0` (the rollout's flip-the-default milestone). At `0.8.0` the field is gone; non-cohort states *always* emit a Leaf for the response position. Theorem-2 strict equality (`Executor.oracle_calls == plan(...).predicted_calls`) holds universally for non-terminal FSM programs.

**Before:**
```python
state._emit_response_leaf_for_non_cohort = False  # restore legacy App(CB_RESPOND, instance)
```

**After:**

Drop the line. There's no opt-out — Leaf emission is unconditional.

If you genuinely need a 0-oracle-call host-callable path (e.g. for a deterministic synthetic response), set `response_instructions=""` (empty string, not `None`) on the State. The compile_fsm dispatcher routes empty-instructions states to the `D1` synthetic path which emits `App(CB_RESPOND_SYNTHETIC, instance)` and returns `f"[{state.id}]"` with zero oracle calls.

The `output_schema_ref: Pydantic | str | None` opt-in for terminal-state structured Leaves is unchanged. Setting it routes to the `D5` structured-Leaf path; leaving it `None` (the default) routes to the `D3` legacy `App(CB_RESPOND, instance)` fallback (because runtime-injected Pydantic schemas via `instance.context.data["_output_response_format"]` are not statically determinable at compile time).

---

## 7. `Program` facade tightening

### 7a. Hidden `_api` and `_profile` constructor kwargs

The `Program.__init__` method previously accepted `_api: API | None` and `_profile: HarnessProfile | None` as kw-only args (underscore-prefixed but still in the public signature). At `0.8.0` they're removed. The public `__init__` is term-mode only.

**Before:**
```python
Program(term=None, _api=api_instance)         # FSM-mode (private)
Program(term=t, _profile=profile)             # term-mode + profile
```

**After:**
```python
Program.from_fsm(definition, …)               # FSM-mode (only path)
Program.from_term(t, profile=profile)         # term-mode + profile (kwarg, not _profile)
Program(term=t)                               # term-mode (simple)
```

If you were doing direct FSM-mode construction via `Program(_api=…)` for tests, switch to `Program.from_fsm`. If you genuinely need to inject a pre-built `API`, the test surface in `tests/test_fsm_llm/test_program.py` still exercises every shape you'll need.

### 7b. Explicit kwargs on `Program.from_fsm` (replaces `**api_kwargs`)

The `**api_kwargs` catch-all at `0.7.0` was a leaky abstraction — users couldn't tell what the function accepted. At `0.8.0` every supported API kwarg is in the signature.

**Before:**
```python
Program.from_fsm(
    definition,
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=1024,
    transition_config=cfg,
    max_history_size=10,
    handler_error_mode="raise",
    top_p=0.9,                # forwarded to LiteLLMInterface
)
```

**After:**

The same call works unchanged. The signature is now:

```python
Program.from_fsm(
    fsm_definition,
    *,
    oracle=None,
    session=None,
    handlers=None,
    profile=None,
    # API constructor passthroughs (explicit since 0.8.0):
    model=None,
    api_key=None,
    temperature=None,
    max_tokens=None,
    max_history_size=5,
    max_message_length=1000,
    handler_error_mode="continue",
    transition_config=None,
    # Additional LiteLLM kwargs (top_p, presence_penalty, …):
    **llm_kwargs,
)
```

If you were passing some other API kwarg via `**api_kwargs` (e.g. a kwarg that isn't enumerated above), it still flows through — `**llm_kwargs` is the new catch-all and forwards to the `LiteLLMInterface` constructor. If you hit a kwarg that no longer routes correctly, file an issue: that kwarg should probably be promoted to an explicit parameter.

---

## 8. Reasoning factory parameter rename

Eleven reasoning factories had generic positional parameter names (`prompt_a`, `prompt_b`, `prompt_c`) that didn't communicate their semantics. At `0.8.0` the names match each factory's `bind_names` tuple.

The internal `_three_leaf` helper keeps its generic names — it's private.

### Per-factory mapping

| Factory | Old | New |
|---|---|---|
| `analytical_term` | `prompt_a, prompt_b, prompt_c` | `decomposition_prompt, analysis_prompt, integration_prompt` |
| `deductive_term` | `prompt_a, prompt_b, prompt_c` | `premises_prompt, inference_prompt, conclusion_prompt` |
| `inductive_term` | `prompt_a, prompt_b, prompt_c` | `examples_prompt, pattern_prompt, generalization_prompt` |
| `abductive_term` | `prompt_a, prompt_b, prompt_c` | `observation_prompt, hypothesis_prompt, selection_prompt` |
| `analogical_term` | `prompt_a, prompt_b, prompt_c` | `source_prompt, mapping_prompt, target_inference_prompt` |
| `creative_term` | `prompt_a, prompt_b, prompt_c` | `divergence_prompt, combination_prompt, refinement_prompt` |
| `critical_term` | `prompt_a, prompt_b, prompt_c` | `examination_prompt, evaluation_prompt, verdict_prompt` |
| `hybrid_term` (4 leaves) | `…, execute_prompt, integrate_prompt` | `…, execution_prompt, integration_prompt` |
| `calculator_term` | already `parse_prompt, compute_prompt` | unchanged |
| `classifier_term` | `…, recommend_prompt` | `…, recommendation_prompt` |
| `solve_term` | `validate_prompt, validate_input_vars, validate_schema_ref` | `validation_prompt, validation_input_vars, validation_schema_ref` |

The same rename pattern applies to the matching `*_input_vars` and `*_schema_ref` kwargs on each factory.

**Before:**
```python
from fsm_llm.stdlib.reasoning import analytical_term

term = analytical_term(
    prompt_a="Decompose: {problem}",
    prompt_b="Analyze: {problem}, {decomposition}",
    prompt_c="Integrate: {problem}, {analysis}",
    schema_ref_c="my_module.AnswerSchema",
)
```

**After:**
```python
from fsm_llm.stdlib.reasoning import analytical_term

term = analytical_term(
    decomposition_prompt="Decompose: {problem}",
    analysis_prompt="Analyze: {problem}, {decomposition}",
    integration_prompt="Integrate: {problem}, {analysis}",
    integration_schema_ref="my_module.AnswerSchema",
)
```

---

## 9. Internal module relocations (no caller-side changes)

These are noted for context — public imports continue to resolve unchanged.

* **`dialog/turn.py` → `dialog/extraction.py`.** The Pass-1 extraction cluster (8 methods + helpers) was extracted from the 2,295-LOC `turn.py` into a new private `dialog/extraction.py` (~976 LOC). `MessagePipeline` constructs an `ExtractionEngine` once and delegates. Tests that patch `pipeline._<method>` directly need to switch to `pipeline._extraction._<method>` if they were doing internal patching; the dozens of tests that call `pipeline._<method>(...)` directly still work via thin delegation wrappers on `MessagePipeline`.
* **`handlers.py` → `runtime/_handlers_ast.py`.** The `compose` function and the eight `_splice_<timing>` AST splicers moved out of `handlers.py` into a new private `runtime/_handlers_ast.py` (457 LOC). `handlers.py` is leaner (1,426 → 1,093 LOC). Public imports — `from fsm_llm import compose` and `from fsm_llm.handlers import compose` — work unchanged. Tests that imported underscore-prefixed splicer helpers (`_splice_pre_processing` etc.) directly need to update those imports to `from fsm_llm.runtime._handlers_ast import _splice_pre_processing`; no public test surface broke.

---

## 10. Verifying your migration

After updating callsites, run your test suite and grep for stale references:

```bash
grep -rn "from fsm_llm import Handler\b" .
grep -rn "from fsm_llm import LLMInterface\b" .
grep -rn "from fsm_llm import BUILTIN_OPS\b" .
grep -rn "has_workflows\|has_reasoning\|has_agents\|get_workflows\|get_reasoning\|get_agents" .
grep -rn "from fsm_llm.dialog.definitions import FSMError\|from fsm_llm.dialog.definitions import .*Error" .
grep -rn "_emit_response_leaf_for_non_cohort" .
grep -rn "Program(.*_api=\|Program(.*_profile=" .
grep -rn "prompt_a=\|prompt_b=\|prompt_c=\|input_vars_a=\|input_vars_b=\|input_vars_c=" .
```

All should return zero hits in user code (they may appear in your own changelog / migration notes).

For library-side regression coverage, the deprecation calendar's `TestZ8EpochHardRemovedAt080` class at `tests/test_fsm_llm/test_deprecation_calendar.py` documents every removed surface as an executable contract. Reading that test class is the fastest way to see "what is no longer supported, with one example per row."
