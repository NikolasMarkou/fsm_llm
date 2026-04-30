# Migration guide: 0.8.0 → 0.9.0

The 0.9.0 cleanup release reshapes the public surface. **Every change is a
hard break — no aliases, no deprecation warnings.** Update your imports.

This guide is the practical companion to the
[0.9.0 CHANGELOG entry](../CHANGELOG.md#090--2026-04-30).

## TL;DR

```python
# 0.8.0
from fsm_llm import (
    Term, Var, Abs, App, Let, Case, Leaf, Fix,
    leaf, var, abs_, app, let_, case_, fix,
    split, fmap, ffilter, reduce_, concat, cross,
    react_term, analytical_term, linear_term, niah_term,
    FSMError, LambdaError, ASTConstructionError, TerminationError,
    enable_debug_logging, disable_warnings, BUFFER_METADATA,
    register_harness_profile, get_harness_profile,
    HandlerSystem, compile_fsm_cached, get_version_info,
)

# 0.9.0
from fsm_llm import Program, Result, compile_fsm, Executor, FSMHandler
from fsm_llm.ast import Term, Var, Abs, App, Let, Case, Leaf, Fix
from fsm_llm.dsl import leaf, var, abs_, app, let, case_, fix
from fsm_llm.combinators import split, fmap, ffilter, reduce, concat, cross
from fsm_llm.factories import react_term, analytical_term, linear_term, niah_term
from fsm_llm.errors import FSMError, LambdaError, ASTConstructionError, TerminationError
from fsm_llm.debug import enable_debug_logging, disable_warnings, BUFFER_METADATA
from fsm_llm import profile_registry  # singleton — no module-level functions
from fsm_llm.handlers import HandlerSystem
from fsm_llm.dialog.compile_fsm import compile_fsm_cached
# get_version_info is deleted — use fsm_llm.__version__ directly
```

## Rename table

| Old (0.8.0) | New (0.9.0) | Reason |
|---|---|---|
| `from fsm_llm import Term, Var, Abs, ...` | `from fsm_llm.ast import ...` | sub-namespace |
| `from fsm_llm import leaf, var, abs_, ...` | `from fsm_llm.dsl import ...` | sub-namespace |
| `let_(...)` | `let(...)` | no Python keyword collision |
| `reduce_(...)` | `reduce(...)` | no Python keyword collision (caveat: shadows `functools.reduce`) |
| `from fsm_llm import split, fmap, ...` | `from fsm_llm.combinators import ...` | sub-namespace |
| `from fsm_llm import react_term, ...` | `from fsm_llm.factories import ...` (or per-domain) | sub-namespace |
| `from fsm_llm import ASTConstructionError, ...` | `from fsm_llm.errors import ...` | sub-namespace (only roots `FSMError`/`LambdaError` stay top-level) |
| `from fsm_llm import enable_debug_logging, disable_warnings, BUFFER_METADATA` | `from fsm_llm.debug import ...` | sub-namespace |
| `register_harness_profile("openai", p)` | `profile_registry.register("openai", p, kind="harness")` | collapsed registry |
| `register_provider_profile("ollama_chat", p)` | `profile_registry.register("ollama_chat", p, kind="provider")` | collapsed registry |
| `get_harness_profile("openai")` | `profile_registry.get("openai", kind="harness")` | collapsed registry |
| `get_provider_profile("ollama_chat")` | `profile_registry.get("ollama_chat", kind="provider")` | collapsed registry |
| `unregister_harness_profile("openai")` | `profile_registry.unregister("openai", kind="harness")` | collapsed registry |
| `unregister_provider_profile("ollama_chat")` | `profile_registry.unregister("ollama_chat", kind="provider")` | collapsed registry |
| `HandlerBuilder().on_state("s")` | `HandlerBuilder().when_state("s")` | `.when_*` prefix for all conditions |
| `HandlerBuilder().not_on_state("s")` | `HandlerBuilder().not_when_state("s")` | (same) |
| `HandlerBuilder().on_state_entry("s")` | `HandlerBuilder().when_state_entry("s")` | (same) |
| `HandlerBuilder().on_state_exit("s")` | `HandlerBuilder().when_state_exit("s")` | (same) |
| `system.execute_handlers(...)` | `system._execute_handlers(...)` | privatized (internal-only post-R5) |
| `from fsm_llm import HandlerSystem` | `from fsm_llm.handlers import HandlerSystem` | dropped from top-level |
| `from fsm_llm import compile_fsm_cached` | `from fsm_llm.dialog.compile_fsm import compile_fsm_cached` | perf-tier; `compile_fsm` stays at top |
| `get_version_info()` | `fsm_llm.__version__` | function deleted (info dict was hardcoded `True` for all features) |
| `from fsm_llm.types import FSMError` | `from fsm_llm.errors import FSMError` | `types.py` split into `_models.py` + `errors.py` |

## Find-and-replace patterns

For a typical codebase, these `sed` patterns cover the bulk of the migration:

```bash
# DSL renames
sed -i 's/\blet_\b/let/g; s/\breduce_\b/reduce/g' $(find . -name '*.py')

# HandlerBuilder method renames
sed -i \
  -e 's/\.on_state_entry\b/.when_state_entry/g' \
  -e 's/\.on_state_exit\b/.when_state_exit/g' \
  -e 's/\.not_on_state\b/.not_when_state/g' \
  -e 's/\.on_state\b/.when_state/g' \
  $(find . -name '*.py')

# types module split
sed -i 's/fsm_llm\.types/fsm_llm.errors/g' $(grep -rl 'fsm_llm.types' --include='*.py' .)
```

Top-level → sub-namespace migrations cannot be done blindly with `sed`
(some names appear in `__all__` lists, comments, and docstrings without
being imports). Use your editor's "find references" against each removed
top-level name.

## ProfileRegistry migration in detail

The 0.8.0 profile registry was six parallel module-level functions:

```python
# 0.8.0
register_harness_profile("openai", HarnessProfile(...))
register_provider_profile("ollama_chat", ProviderProfile(...))
prof = get_harness_profile("openai:gpt-4o")
provider = get_provider_profile("ollama_chat/qwen3.5:4b")
unregister_harness_profile("openai")
unregister_provider_profile("ollama_chat")
```

0.9.0 collapses these into a single `ProfileRegistry` class with a
module-level `profile_registry` singleton:

```python
# 0.9.0
from fsm_llm import profile_registry, HarnessProfile, ProviderProfile

profile_registry.register("openai", HarnessProfile(...), kind="harness")
profile_registry.register("ollama_chat", ProviderProfile(...), kind="provider")

prof = profile_registry.get("openai:gpt-4o", kind="harness")
provider = profile_registry.get("ollama_chat/qwen3.5:4b", kind="provider")

profile_registry.unregister("openai", kind="harness")
profile_registry.unregister("ollama_chat", kind="provider")

# New: list registered names
profile_registry.list(kind="harness")  # ["openai"]
profile_registry.list()                # {"harness": [...], "provider": [...]}
```

Lookup conventions are unchanged: harness lookup falls back from
`"provider:model"` to bare `"provider"`; provider lookup falls back
from `"provider/model"` to bare `"provider"`.

The kind kwarg can be omitted on `register()` when the profile instance
is one of `HarnessProfile` or `ProviderProfile` — it's inferred from the
type. Other registry methods always require `kind=`.

## Program constructor symmetry

The `from_term` and `from_factory` constructors now accept the same
explicit LLM kwargs as `from_fsm`:

```python
# 0.8.0 — only oracle= worked here
program = Program.from_term(my_term, oracle=LiteLLMOracle(LiteLLMInterface(model="gpt-4o")))

# 0.9.0 — explicit kwargs match from_fsm
program = Program.from_term(my_term, model="gpt-4o", temperature=0)

# Mixing oracle= with LLM kwargs raises ValueError
program = Program.from_term(my_term, oracle=my_oracle, model="gpt-4o")  # ValueError
```

This mirrors `Program.from_fsm(...)` and removes the silent "first
invoke makes a network call" surprise — passing a model up-front makes
intent explicit; the lazy default still runs when no kwargs are supplied.

## What stayed the same

- Top-level `Program`, `Result`, `ExplainOutput`, `ProgramModeError`.
- Top-level `compile_fsm`, `Executor`, `Plan`, `PlanInputs`, `plan`,
  `Oracle`, `LiteLLMOracle`, `CostAccumulator`.
- Top-level handler surface: `compose`, `HandlerTiming`,
  `HandlerBuilder`, `FSMHandler`, `BaseHandler`, `create_handler`.
- Top-level profile types: `HarnessProfile`, `ProviderProfile` (the
  registry surface changed; the model classes did not).
- The dialog tier (FSM JSON authoring + classification + sessions +
  validators) still imports from the top level — only the "Legacy"
  label retired.
- Theorem-2 contract — runtime semantics unchanged.

## Verification

After migrating, run:

```bash
.venv/bin/python -m pytest -q                 # full test suite
.venv/bin/python -c "import fsm_llm; print(fsm_llm.__version__)"  # 0.9.0
```

Audit greps that should be empty in your codebase:

```bash
grep -rn 'let_\|reduce_' --include='*.py' .              # if any, you missed a rename
grep -rn 'from fsm_llm.types' --include='*.py' .         # types.py is gone
grep -rn 'register_harness_profile\|get_provider_profile' --include='*.py' .  # 6 fn surface gone
grep -rn '\.on_state\b\|\.execute_handlers\b' --include='*.py' .  # handler renames
```
