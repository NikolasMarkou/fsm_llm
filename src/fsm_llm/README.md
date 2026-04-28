# fsm_llm — Core Package

> A typed λ-calculus runtime for stateful LLM programs. Two surface syntaxes share one executor: FSM JSON for dialog (Category A); a λ-DSL for pipelines and long-context recursion (Category B/C).

---

## Overview

`fsm_llm` is the core package. It contains:

- **`runtime/`** — the λ-kernel: typed AST, β-reduction Executor, closed-form Planner, Oracle adapter. (Renamed from `lam/` in R4 — `from fsm_llm.lam import ...` keeps working via shim; deprecation 0.5.0, removal 0.6.0.)
- **`dialog/`** — the FSM dialog surface (R4): `API`, `FSMManager`, `MessagePipeline`, handlers, classification, prompts, the FSM→λ compiler (`compile_fsm`), definitions, and sessions.
- **`stdlib/`** — named λ-term factories (agents, reasoning, workflows, long-context).
- **`program.py`** — the unified `Program` facade (R1; mode-bifurcated — see caveat below).

Per `docs/lambda.md` §1: **every fsm_llm program is already a λ-term.** The 2-pass FSM body (`extract → evaluate transition → respond`) is the body of the compiled λ-term for Category A; Category B/C programs are written directly as λ-terms.

`docs/lambda_integration.md` (Refactoring Report v2.0) tracks what shipped narrowed (L1–L8 residual gaps) and what's planned for v0.5.0 (R8–R13).

### What changed in M1-M5

- **M1** (kernel) — `runtime/` is the substrate.
- **M2** (compiler) — every FSM JSON is compiled to a λ-term at load time. Single execution path. `MessagePipeline.process` / `process_stream` retired (S11). `FSMManager.use_compiled` flag removed.
- **M3** (stdlib) — agents/reasoning/workflows/long_context expose named λ-term factories under `stdlib/<pkg>/lam_factories.py`.
- **M4** — Category-B examples migrated to `examples/pipeline/`.
- **M5** — Category-C long-context library + bench harness + OOLONG ingestion.

### `Program` facade — current caveat (L1)

`Program` is the unified entry over `(term, oracle, optional_session, optional_handlers)` with three constructors (`from_fsm`, `from_term`, `from_factory`). At HEAD it is **mode-bifurcated**: `Program.from_fsm(d).run(...)` and `Program.from_term(t).converse(...)` raise `NotImplementedError`; `Program.from_term(t).register_handler(h)` raises; `.explain()` returns `plans=[]` for FSM mode. R8 closes this. Until then, prefer `API` for FSM dialogs and `Executor` for term programs.

## Installation

```bash
pip install fsm-llm                  # Core only
pip install fsm-llm[all]             # All extras
pip install fsm-llm[dev]             # Development
```

**Requirements**: Python 3.10+ | Deps: `loguru`, `litellm` (>=1.82,<2.0), `pydantic` (>=2.0), `python-dotenv`

## Quick Start

### A — λ-DSL (Category B): a 2-call extract-then-answer chain

```python
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.lam import Executor, LiteLLMOracle, leaf, let_, var
from pydantic import BaseModel

class Topic(BaseModel): topic: str

term = let_(
    "topic", leaf(prompt="Extract the topic in one word: {q}", schema=Topic, input_var="q"),
    leaf(prompt="Write a one-paragraph article about {topic}.", input_var="topic"),
)
ex = Executor(oracle=LiteLLMOracle(LiteLLMInterface(model="openai/gpt-4o-mini")))
print(ex.run(term, env={"q": "What is photosynthesis?"}))
assert ex.oracle_calls == 2
```

### B — Long-context (Category C): NIAH with Theorem-2 gate

```python
from fsm_llm.lam import Executor, LiteLLMOracle, plan, PlanInputs
from fsm_llm.stdlib.long_context import niah
from fsm_llm.llm import LiteLLMInterface

term = niah(question="Where is the needle hidden?", tau=256, k=2)
ex = Executor(oracle=LiteLLMOracle(LiteLLMInterface(model="ollama_chat/qwen3.5:4b")))
ex.run(term, env={"document": four_kb_document})

predicted = plan(PlanInputs(n=4096, tau=256, k=2)).predicted_calls
assert ex.oracle_calls == predicted   # Theorem-2 holds
```

### C — FSM JSON (Category A): a dialog program

```json
{
  "name": "Greeter",
  "initial_state": "greeting",
  "persona": "A friendly assistant",
  "states": {
    "greeting": {
      "id": "greeting",
      "purpose": "Welcome and ask their name",
      "extraction_instructions": "Extract the user's name if provided",
      "response_instructions": "Greet warmly, ask for name if not given",
      "transitions": [{
        "target_state": "farewell",
        "description": "User has given their name",
        "conditions": [{"description": "Name is available", "requires_context_keys": ["name"], "logic": {"has_context": "name"}}]
      }]
    },
    "farewell": {
      "id": "farewell",
      "purpose": "Thank the user and end conversation",
      "response_instructions": "Say a personalized goodbye using their name",
      "transitions": []
    }
  }
}
```

```python
from fsm_llm import API

api = API.from_file("greeter.json", model="openai/gpt-4o-mini")
conv_id, greeting = api.start_conversation()
print(greeting)

response = api.converse("My name is Alice", conv_id)
print(response)

api.end_conversation(conv_id)
api.close()
```

The FSM is compiled to a λ-term at `from_file()` time; `converse()` is one β-reduction step on the executor.

### Or run from the CLI

```bash
export OPENAI_API_KEY=your-key
fsm-llm run greeter.json            # Unified subcommand (preferred)
fsm-llm --fsm greeter.json          # Legacy flag — still works
```

## Architecture

```
   FSM JSON  →  fsm_compile()  →┐
                                │
                                ▼
                              λ-AST  →  Executor.run(env)  →  result
                                ▲
   λ-DSL    →  dsl builders   →┘
```

For Category A: the per-turn `step : (state, input, context) → (state', output, context')` becomes a λ-term with a top-level `Case` on `state_id`. Pass 1 / transition / Pass 2 are three `Leaf` nodes around a pure `Case`. Per-Fix nodes use `push_fsm` stacking; the planner is queried at each Fix.

### Key Components

| Component | Module | Purpose |
|-----------|--------|---------|
| `Executor` | `runtime/executor.py` | β-reduction, depth-bounded, per-Leaf cost |
| `Planner` (`plan`, `PlanInputs`, `Plan`) | `runtime/planner.py` | Closed-form (k*, τ*, d, predicted_calls) per Fix node |
| `Oracle`, `LiteLLMOracle` | `runtime/oracle.py` | Schema-typed LLM adapter |
| `compile_fsm` | `dialog/compile_fsm.py` | FSMDefinition → Term (also reachable via `from fsm_llm.lam import compile_fsm` shim) |
| λ-DSL builders | `runtime/dsl.py` | `var`, `abs_`, `app`, `let_`, `case_`, `fix`, `leaf`, `split`, `peek`, `fmap`, `ffilter`, `reduce_`, `concat`, `cross` |
| `Program` | `program.py` | Unified facade over `(term, oracle, session, handlers)` — mode-bifurcated at HEAD; see L1 caveat above |
| `API` | `dialog/api.py` | FSM-dialog entry point — factory, conversation lifecycle, FSM stacking |
| `FSMManager` | `dialog/fsm.py` | Per-conversation thread locks; compiled-term cache lives in `dialog/compile_fsm.py` |
| `MessagePipeline` | `dialog/turn.py` | Compiled-path 2-pass body (file renamed `pipeline.py`→`turn.py` in R13; old import shim still works). **Internal post-M2 S11.** |
| `HandlerSystem` | `handlers.py` | 8 lifecycle hook points; composes into the compiled term |
| `Classifier` | `dialog/classification.py` | LLM intent classification (single, multi, hierarchical) |
| `TransitionEvaluator` | `dialog/transition_evaluator.py` | JsonLogic evaluation: DETERMINISTIC / AMBIGUOUS / BLOCKED |
| `LiteLLMInterface` | `runtime/_litellm.py` | LLM communication via litellm (100+ providers); old import path `fsm_llm.llm` still works |
| `WorkingMemory` | `memory.py` | 4 named buffers (core, scratch, environment, reasoning) |
| `SessionStore` / `FileSessionStore` | `session.py` | Persistence with atomic file writes |

## Key API Reference

### `API` class

```python
from fsm_llm import API

api = API.from_file("path/to/fsm.json", model="openai/gpt-4o-mini")
api = API.from_definition(fsm_dict, model="openai/gpt-4o-mini")

conv_id, greeting = api.start_conversation(initial_context={"key": "value"})
response = api.converse("user message", conv_id)
api.end_conversation(conv_id)

# FSM stacking (sub-conversations) — compiles to bounded `Fix`
sub_conv_id = api.push_fsm(conv_id, sub_fsm_definition)
response = api.pop_fsm(sub_conv_id, merge_strategy=ContextMergeStrategy.UPDATE)

# State queries
state = api.get_current_state(conv_id)
data = api.get_data(conv_id)
history = api.get_conversation_history(conv_id)
```

### λ-Kernel

```python
from fsm_llm.lam import (
    Executor, LiteLLMOracle, plan, PlanInputs, Plan,
    var, abs_, app, let_, case_, fix, leaf,
    split, peek, fmap, ffilter, reduce_, concat, cross,
    BUILTIN_OPS, ReduceOp,
    Var, Abs, App, Let, Case, Combinator, Fix, Leaf, Term,
)
```

The kernel (`fsm_llm.runtime` / `fsm_llm.lam` shim) imports nothing from `fsm_llm.dialog` — it is pure substrate. The LiteLLM adapter lives in `runtime/oracle.py`; the FSM→λ compiler lives in `dialog/compile_fsm.py` and is re-exported into the kernel namespace for back-compat (per `runtime/CLAUDE.md` D-001).

### Handlers

Eight lifecycle hook points via `HandlerTiming`:

| Timing | When |
|--------|------|
| `START_CONVERSATION` | Conversation initialised |
| `PRE_PROCESSING` | Before message processing |
| `POST_PROCESSING` | After message processing |
| `PRE_TRANSITION` | Before state transition |
| `POST_TRANSITION` | After state transition |
| `CONTEXT_UPDATE` | During context updates |
| `END_CONVERSATION` | Conversation terminated |
| `ERROR` | Error handling |

```python
handler = api.create_handler("logger") \
    .at(HandlerTiming.POST_TRANSITION) \
    .on_state("checkout") \
    .do(lambda ctx: print(f"Entered checkout: {ctx}") or {})
api.register_handler(handler)
```

Hooks compose into the compiled λ-term per `docs/lambda.md` §6.3.

### Classification

```python
from fsm_llm import Classifier, ClassificationSchema, IntentDefinition

schema = ClassificationSchema(
    intents=[
        IntentDefinition(name="buy", description="User wants to purchase"),
        IntentDefinition(name="browse", description="User is browsing"),
    ],
    fallback_intent="browse",
)
classifier = Classifier(schema, model="openai/gpt-4o-mini")
result = classifier.classify("I'd like to buy the red shoes")
```

### Transition Conditions (JsonLogic)

```json
{
  "and": [
    {"has_context": "email"},
    {">=": [{"var": "age"}, 18]},
    {"in": [{"var": "country"}, ["US", "CA", "UK"]]}
  ]
}
```

Operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `and`, `or`, `!`, `in`, `has_context`, `context_length`, `var`, `if`, `cat`, `min`, `max`, arithmetic (`+`, `-`, `*`, `/`, `%`).

### LLM Interface

```python
from fsm_llm import LiteLLMInterface, LLMInterface

llm = LiteLLMInterface(model="openai/gpt-4o-mini", temperature=0.7)

# Streaming (Pass 2 only)
for chunk in llm.generate_response_stream(request):
    print(chunk, end="")

# Schema enforcement: set response_format on ResponseGenerationRequest for
# Pydantic-derived constrained decoding. Recommended for small Ollama models —
# see plans/LESSONS.md "LiteLLMInterface Reuse Pattern".

class CustomLLM(LLMInterface):
    def generate_response(self, request): ...
    def extract_field(self, request): ...
    def generate_response_stream(self, request): ...  # Optional
```

### Session Persistence

```python
from fsm_llm import API, FileSessionStore

store = FileSessionStore("./sessions")
api = API.from_file("bot.json", model="openai/gpt-4o-mini", session_store=store)

# State is auto-saved after each converse() call
conv_id, greeting = api.start_conversation()
response = api.converse("Hello!", conv_id)

# Explicit save/load
api.save_session(conv_id)
api.load_session(conv_id)
```

## CLI Tools

| Command | Description |
|---------|-------------|
| `fsm-llm run <path>` | Run interactive conversation (compiled λ-path); legacy `fsm-llm --fsm <path>` still works |
| `fsm-llm explain <target>` | Print AST shape, leaf schemas, per-Fix planner output |
| `fsm-llm validate --fsm <path>` | Validate FSM definition (alias: `fsm-llm-validate`) |
| `fsm-llm visualize --fsm <path>` | ASCII FSM visualization (alias: `fsm-llm-visualize`) |
| `fsm-llm monitor` | Web monitoring dashboard (alias: `fsm-llm-monitor`) |
| `fsm-llm meta` | Interactive artifact builder (alias: `fsm-llm-meta`) |

## FSM Definition Format (v4.1)

States support:
- `extraction_instructions` / `response_instructions` — LLM prompts for each pass
- `required_context_keys` — keys that must exist before leaving the state
- `field_extractions` — targeted single-field extraction with validation rules
- `classification_extractions` — intent classification with confidence thresholds
- `transitions` — JsonLogic conditions, priority ordering, descriptions
- `context_scope` — read/write key filtering per state

## Exception Hierarchy

```
FSMError                                      LambdaError (kernel)
├── StateNotFoundError                        ├── ASTConstructionError
├── InvalidTransitionError                    ├── TerminationError
├── LLMResponseError                          ├── PlanningError
├── TransitionEvaluationError                 └── OracleError
├── ClassificationError
│   ├── SchemaValidationError
│   └── ClassificationResponseError
└── HandlerSystemError
    └── HandlerExecutionError
```

## License

GPL-3.0-or-later. See [LICENSE](../../LICENSE).
