# Quickstart

A five-minute tour of `fsm-llm` `0.7.0`. By the end you'll have an FSM dialog program, a λ-term pipeline, a handler wired in, and a provider switched via a profile.

## 1. Install

```bash
pip install fsm-llm
```

Set a provider key — `OPENAI_API_KEY` for OpenAI, or any LiteLLM-supported provider:

```bash
export OPENAI_API_KEY=sk-...
# or for local Ollama:
export OLLAMA_BASE_URL=http://127.0.0.1:11434
```

A clean install gives you the dialog surface, the runtime, and the standard library. Install extras as needed (`fsm-llm[monitor]` for the dashboard, `fsm-llm[mcp]` for MCP, etc.). See the README install section for the full list.

## 2. The first program — FSM JSON

A minimal greet-then-quit FSM. Save as `greet.json`:

```json
{
  "name": "greet",
  "version": "4.1",
  "initial_state": "hello",
  "states": {
    "hello": {
      "id": "hello",
      "description": "say hi",
      "purpose": "greet the user warmly",
      "transitions": [
        {"target_state": "bye", "description": "always", "priority": 100, "conditions": []}
      ]
    },
    "bye": {
      "id": "bye",
      "description": "say bye",
      "purpose": "wish the user well",
      "transitions": []
    }
  }
}
```

Run it:

```python
from fsm_llm import Program

prog = Program.from_fsm("greet.json")
result = prog.invoke(message="hi")
print(result.value)            # "Hello! ..."
print(result.conversation_id)  # auto-started; pass it back next turn
```

`Program.from_fsm(...)` accepts a path, a dict, or a parsed `FSMDefinition`. The same `.invoke(message=...)` works for every turn; pass `conversation_id=` on subsequent turns to continue the same session.

For a richer example, browse [`examples/basic`](../examples/basic), [`examples/intermediate`](../examples/intermediate), or [`examples/advanced`](../examples/advanced).

## 3. The second program — a λ-term pipeline

No FSM required for stateless work. Build a term in the DSL:

```python
from fsm_llm import Program, leaf, let_

term = let_(
    "summary", leaf(template="Summarise: {doc}", input_vars=("doc",)),
    leaf(template="Translate to French: {summary}", input_vars=("summary",)),
)

prog = Program.from_term(term)
result = prog.invoke(inputs={"doc": "Long article text..."})
print(result.value)            # the French summary
print(result.oracle_calls)     # 2 — the planner predicts exactly this
```

`Result` carries the same fields in every mode: `value`, `conversation_id` (None for term mode), `plan`, `leaf_calls`, `oracle_calls`, `explain`.

## 4. A stdlib factory — ReAct agent

The standard library ships ready-made factory functions. Every name ends in `*_term`:

```python
from fsm_llm import Program, react_term

term = react_term(
    decide_prompt="Question: {question}\nDecide on a tool call as JSON.",
    synth_prompt="Tool returned {observation}.\nFinal answer:",
)

def my_tools(decision):
    # User-supplied tool dispatcher
    return f"42 (mock answer for {decision!r})"

prog = Program.from_term(term)
result = prog.invoke(inputs={
    "question": "What is the capital of France?",
    "tool_dispatch": my_tools,
})
print(result.value)            # the final synthesised answer
```

Stdlib factories ship in four slices, all reachable from the top level:

```python
# Agents
from fsm_llm import react_term, rewoo_term, reflexion_term, memory_term

# Reasoning
from fsm_llm import (
    analytical_term, deductive_term, inductive_term, abductive_term,
    analogical_term, creative_term, critical_term, hybrid_term,
    calculator_term, classifier_term, solve_term,
)

# Workflows
from fsm_llm import (
    linear_term, branch_term, switch_term, parallel_term, retry_term,
)

# Long-context
from fsm_llm import (
    niah_term, aggregate_term, pairwise_term,
    multi_hop_term, multi_hop_dynamic_term, niah_padded_term,
)
```

## 5. Adding a handler

Handlers fire at one of 8 timing points around an FSM turn or a term reduction. Build one with `HandlerBuilder` and pass it at construction time:

```python
from fsm_llm import HandlerBuilder, HandlerTiming, Program

events = []

audit = (
    HandlerBuilder("audit")
    .at(HandlerTiming.PRE_PROCESSING)
    .do(lambda **kw: events.append(("pre", kw.get("current_state"))))
    .build()
)

prog = Program.from_fsm("greet.json", handlers=[audit])
prog.invoke(message="hi")
print(events)                  # [("pre", "hello"), ("pre", "bye"), ...]
```

`PRE_PROCESSING` and `POST_PROCESSING` splice into the AST via `compose`. The other six timings (`START_CONVERSATION`, `END_CONVERSATION`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `ERROR`) dispatch host-side. See [`docs/handlers.md`](handlers.md) for the full timing reference.

## 6. Switching providers via profiles

`HarnessProfile` and `ProviderProfile` bundle prompt fragments and provider kwargs that apply once at construction:

```python
from fsm_llm import (
    HarnessProfile, ProviderProfile, Program,
    register_harness_profile, register_provider_profile,
)

# Provider-side: map a model name to extra litellm kwargs.
register_provider_profile(
    "ollama:qwen3.5:4b",
    ProviderProfile(extra_kwargs={"api_base": "http://127.0.0.1:11434"}),
)

# Harness-side: bundle a system-prompt prefix and per-Leaf overrides.
register_harness_profile(
    "ollama:qwen3.5:4b",
    HarnessProfile(
        system_prompt_base="You are a precise, terse assistant.",
        provider_profile_name="ollama:qwen3.5:4b",
    ),
)

prog = Program.from_term(my_term, profile="ollama:qwen3.5:4b")
```

Profiles are pure: they rewrite `Leaf.template` strings via `Term.model_copy` and never add or remove AST nodes, so **Theorem-2 strict equality is preserved** — `result.oracle_calls` still matches the planner.

## 7. Inspecting cost before running — `Program.explain`

Call `.explain(n=N, K=K)` for a planner-derived prediction without invoking the LLM:

```python
from fsm_llm import Program
from fsm_llm.stdlib.long_context import niah_term

prog = Program.from_factory(niah_term, factory_kwargs={
    "question": "Where is the artefact?", "tau": 256, "k": 2,
})
explanation = prog.explain(n=10000, K=8192)

for p in explanation.plans:
    print(p.predicted_calls)   # closed-form k^d for each Fix subtree

print(explanation.ast_shape)   # indented multi-line term skeleton
print(explanation.leaf_schemas)# {leaf_id: schema or None}
```

After running, `result.oracle_calls` will equal `plan.predicted_calls`. That's Theorem 2.

## 8. Running from the CLI

The package ships five console scripts:

```bash
# Run, validate, or visualize an FSM JSON definition.
fsm-llm --mode run --fsm greet.json
fsm-llm --mode validate --fsm greet.json
fsm-llm --mode visualize --fsm greet.json

# Single-purpose aliases — same code, simpler signatures.
fsm-llm-validate --fsm greet.json
fsm-llm-visualize --fsm greet.json

# Interactive artifact builder.
fsm-llm-meta

# Web dashboard (requires fsm-llm[monitor]).
fsm-llm-monitor
```

## What next

- [`api_reference.md`](api_reference.md) — every public name with signatures.
- [`architecture.md`](architecture.md) — how `Program.invoke` reaches the executor.
- [`handlers.md`](handlers.md) — the full handler lifecycle.
- [`fsm_design.md`](fsm_design.md) — FSM JSON patterns and anti-patterns.
- [`lambda.md`](lambda.md) and [`lambda_fsm_merge.md`](lambda_fsm_merge.md) — the architectural thesis and merge contract.
