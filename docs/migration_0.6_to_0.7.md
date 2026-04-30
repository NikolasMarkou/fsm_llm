# Migration Guide: 0.6.x → 0.7.0

`fsm_llm 0.7.0` closes the I5 deprecation epoch announced in 0.6.0. The
five deprecation surfaces are now hard removals (`AttributeError` /
`ImportError`), the `LiteLLMInterface` top-level re-export is formalised
as private (D-009), the `quick_start()` helper is gone, and a neutral
`fsm_llm.types` module breaks the runtime↔dialog Pydantic-model coupling.

This guide walks each change with before/after code and notes when no
caller change is required.

## TL;DR

If your 0.6.x code did **any** of the following, you have to update it:

| Pattern | Action |
|---|---|
| `program.run(**env)` | `program.invoke(inputs=env).value` |
| `program.converse(msg, conv_id)` | `program.invoke(message=msg, conversation_id=conv_id).value` |
| `program.register_handler(h)` | pass `handlers=[h]` at `Program.from_fsm/from_term/from_factory` |
| `from fsm_llm import API` | `from fsm_llm.dialog.api import API` (or use `Program.from_fsm`) |
| `import fsm_llm_reasoning` | `from fsm_llm.stdlib import reasoning` |
| `import fsm_llm_workflows` | `from fsm_llm.stdlib import workflows` |
| `import fsm_llm_agents` | `from fsm_llm.stdlib import agents` |
| `niah(...)` (or any of 5 other bare names) | `niah_term(...)` (and the other `*_term` names) |
| `from fsm_llm import LiteLLMInterface` | `from fsm_llm.runtime._litellm import LiteLLMInterface` |
| `quick_start("bot.json")` | `Program.from_fsm("bot.json")` |

Everything else — `Program.invoke`, `Program.from_fsm/from_term/from_factory`,
`compile_fsm`, the stdlib factory surface, all CLI scripts — continues
to work unchanged.

## Removed surfaces

### `Program.run` / `.converse` / `.register_handler`

These three methods were thin aliases over `Program.invoke(...)`; they
warned in 0.6.x and are gone in 0.7.0.

**Before (0.6.x):**

```python
prog = Program.from_term(my_term)
value = prog.run(question="hello")                      # warns

prog = Program.from_fsm("bot.json")
reply = prog.converse("hi", conversation_id="conv-1")   # warns

handler = HandlerBuilder("audit").at(...).do(...).build()
prog.register_handler(handler)                          # warns
```

**After (0.7.0):**

```python
prog = Program.from_term(my_term)
value = prog.invoke(inputs={"question": "hello"}).value

prog = Program.from_fsm("bot.json")
reply = prog.invoke(message="hi", conversation_id="conv-1").value

handler = HandlerBuilder("audit").at(...).do(...).build()
prog = Program.from_fsm("bot.json", handlers=[handler])    # constructor kwarg
```

`Program.invoke(...)` returns a `Result` value object in every mode.
The `.value` field carries what the legacy aliases returned directly;
`.conversation_id`, `.plan`, `.leaf_calls`, `.oracle_calls`, and
`.explain` are populated per mode. See the L4 INVOKE section of
`docs/api_reference.md` for the field contract.

### Top-level `from fsm_llm import API`

The convenience re-export was a 0.6.x deprecation (`__getattr__` shim
emitting `DeprecationWarning`); at 0.7.0 the shim is gone.

**Before:**

```python
from fsm_llm import API   # warns

api = API.from_file("bot.json")
api.start_conversation()
```

**After (preferred — Program facade):**

```python
from fsm_llm import Program

prog = Program.from_fsm("bot.json")
result = prog.invoke(message="hello")
```

**After (if you really need the class):**

```python
from fsm_llm.dialog.api import API   # canonical home

api = API.from_file("bot.json")
api.start_conversation()
```

The `API` class itself is unchanged — only the top-level re-export was
removed.

### Sibling shim packages: `fsm_llm_reasoning`, `fsm_llm_workflows`, `fsm_llm_agents`

These three were `sys.modules`-redirect shim packages introduced as a
migration aid when the stdlib subpackages moved into the main namespace.
The whole `src/fsm_llm_{reasoning,workflows,agents}/` directory tree is
deleted at 0.7.0.

**Before:**

```python
from fsm_llm_reasoning import ReasoningEngine, ReasoningType   # warns
from fsm_llm_workflows import WorkflowBuilder, linear_term     # warns
from fsm_llm_agents import ReactAgent, react_term              # warns
```

**After:**

```python
from fsm_llm.stdlib.reasoning import ReasoningEngine, ReasoningType
from fsm_llm.stdlib.workflows import WorkflowBuilder, linear_term
from fsm_llm.stdlib.agents import ReactAgent, react_term
```

The submodules also follow the canonical path:

```python
# 0.6.x
from fsm_llm_agents.constants import ContextKeys, Defaults
# 0.7.0
from fsm_llm.stdlib.agents.constants import ContextKeys, Defaults

# Same pattern for fsm_llm_reasoning.* and fsm_llm_workflows.*.
```

The factory exports at the **top** level (`from fsm_llm import react_term,
linear_term, niah_term, ...`) are unchanged — only the *package-level*
shim went away.

### Long-context bare names

When the long-context factories were renamed to the `*_term` convention
in 0.6.0 for stdlib consistency, a `__getattr__` shim kept the bare
names (`niah`, `aggregate`, `pairwise`, `multi_hop`, `multi_hop_dynamic`,
`niah_padded`) reachable with a deprecation warning. At 0.7.0 the shim
is gone.

**Before:**

```python
from fsm_llm.stdlib.long_context import niah, aggregate, pairwise   # warns
term = niah(question="Where?", tau=256, k=2)
```

**After:**

```python
from fsm_llm.stdlib.long_context import niah_term, aggregate_term, pairwise_term
term = niah_term(question="Where?", tau=256, k=2)
```

The `*_term` names have always worked since 0.6.0. Helper functions
(`make_size_bucket`, `best_answer_op`, `aggregate_op`, etc.) are
unchanged.

### Top-level `LiteLLMInterface` re-export (D-009 formalisation)

`LiteLLMInterface` is the private adapter behind the Oracle layer. It
was importable as `from fsm_llm import LiteLLMInterface` for back-compat
through 0.6.x but was never in `__all__`. At 0.7.0 the convenience
re-export is removed (D-009 formalised).

**Before:**

```python
from fsm_llm import LiteLLMInterface           # gone in 0.7.0

llm = LiteLLMInterface(model="openai/gpt-4o-mini")
```

**After (preferred — compose through Oracle):**

```python
from fsm_llm import LiteLLMOracle, Program
from fsm_llm.runtime._litellm import LiteLLMInterface

# Or just let Program.from_fsm pick the oracle:
prog = Program.from_fsm("bot.json", model="openai/gpt-4o-mini")
```

**After (direct construction still supported):**

```python
from fsm_llm.runtime._litellm import LiteLLMInterface

llm = LiteLLMInterface(model="openai/gpt-4o-mini")
```

The class itself is unchanged. Implementing the `Oracle` Protocol
directly remains the preferred extension path for callers who need
custom provider logic — see the D-008 caveat at
`runtime/oracle.py:449-572`.

### `quick_start()` helper

The undocumented `quick_start(fsm_file, model=None)` helper at
`fsm_llm/__init__.py` returned the deprecated `_API_INTERNAL`. It had
no callers in the test suite, examples, or documentation. Replace with
the canonical `Program.from_fsm(path)`.

**Before:**

```python
from fsm_llm import quick_start            # gone in 0.7.0
api = quick_start("bot.json", model="openai/gpt-4o-mini")
```

**After:**

```python
from fsm_llm import Program
prog = Program.from_fsm("bot.json", model="openai/gpt-4o-mini")
```

## Refactored — no caller change required

### `fsm_llm.types` neutral models layer

The `FSMError` hierarchy and the runtime-touching request/response
Pydantic models (`FieldExtractionRequest/Response`,
`ResponseGenerationRequest/Response`, `DataExtractionResponse`) plus the
two enums (`LLMRequestType`, `TransitionEvaluationResult`) moved to a
new neutral `fsm_llm.types` module so that the runtime kernel and the
stdlib subpackages can share them without reaching across the dialog
boundary.

`fsm_llm.dialog.definitions` re-exports the same names — every existing
caller that did `from fsm_llm.dialog.definitions import FSMError` (or
any other moved name) **continues to work byte-equivalently**. Pydantic
`isinstance` identity is preserved.

For new code, the canonical home is `fsm_llm.types`:

```python
# Equivalent — both resolve to the same class object:
from fsm_llm.types import FSMError
from fsm_llm.dialog.definitions import FSMError
```

### Internal moves

- `_DISCARD_COUNTER` mutable global → `itertools.count(1)` in
  `handlers.py`. Thread-safe; semantics unchanged.
- Eight broad-`except Exception:` clauses narrowed to specific
  exception subclasses; one site (end-of-conversation cache snapshot
  in `dialog/api.py:962`) gained `logger.warning`. Behaviour identical
  for the success path; failures now surface in the log.

## Verification checklist

After upgrading, run:

```bash
pytest --collect-only -q | tail -3
# expect: ~3187 tests collected (vs ~3214 at 0.6.0)
```

Audit your codebase for the removed surfaces:

```bash
grep -rn "Program\.run\b\|Program\.converse\b\|\.register_handler\b" src/
grep -rn "from fsm_llm import API\|from fsm_llm import.*\bAPI\b" src/
grep -rn "import fsm_llm_reasoning\|import fsm_llm_workflows\|import fsm_llm_agents" src/
grep -rn "from fsm_llm import LiteLLMInterface" src/
grep -rn "fsm_llm\.stdlib\.long_context\.\(niah\|aggregate\|pairwise\|multi_hop\)\b" src/
```

All five greps should return zero hits in your code.

## FAQ

**Q: Will my pickled / serialised state still load?**
Yes. `fsm_llm.types.FSMError` and the moved Pydantic models are
identity-preserved through the back-compat re-export from
`fsm_llm.dialog.definitions`. Existing pickles deserialise unchanged.

**Q: I subclassed `LiteLLMInterface`. Does that still work?**
Yes — the class is unchanged, only the top-level re-export is gone.
Note the standing D-008 caveat: `LiteLLMOracle._invoke_structured`
bypasses your subclass's `generate_response` for structured-Leaf calls
(any `Leaf` with `schema_ref != None`). If you need provider-side logic
on every structured call, implement the `Oracle` Protocol directly
instead of subclassing `LiteLLMInterface`.

**Q: When are the `_emit_response_leaf_for_non_cohort` and
`output_schema_ref` State fields going away?**
Both are tracked for re-evaluation at 0.8.0 (see the inline `# Removal
target` comments in `dialog/definitions.py`). They are forward-compat
gates for the universal Theorem-2 strict-equality rollout; removal
follows adoption.

**Q: Was anything else considered for removal at 0.7.0 but deferred?**
Yes — the `dialog/turn.py` extraction-engine extraction, the
`prompts.py` shared-section dedup, the `runtime/_handlers_ast.py`
splicer move, and the reasoning factory parameter renames are all
tracked for 0.8.0. They are pure refactors with regression risk and
were intentionally kept out of the deprecation-removal release.

## See also

- [`CHANGELOG.md`](../CHANGELOG.md) — the structured release notes for 0.7.0.
- [`docs/api_reference.md`](api_reference.md) — the full L1–L4 surface.
- [`docs/lambda_fsm_merge.md`](lambda_fsm_merge.md) — the canonical merge
  contract; deprecation calendar §3.
- `tests/test_fsm_llm/test_deprecation_calendar.py` — the executable
  audit of the deprecation calendar; auto-flips assertions per
  `__version__`.
