<div align="center">

<img src="./images/fsm-llm-logo-1.png" alt="FSM-LLM" width="420"/>

# FSM-LLM

**Stateful LLM programs on a typed λ-calculus runtime. One executor, two surfaces, one verb.**

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.6.0-orange)](CHANGELOG.md)

</div>

`fsm-llm` is a Python framework for building **stateful LLM programs** — dialog bots, agents, reasoning chains, workflows, and long-context pipelines — that all compile to and execute on the same typed λ-calculus runtime. You author programs in whichever surface fits the problem; one verb (`Program.invoke`) runs all of them.

```python
from fsm_llm import Program

# Surface A — FSM JSON: dialog with persistent per-turn state.
prog = Program.from_fsm("my_bot.json")
result = prog.invoke(message="Hi, I'd like to book a flight")
print(result.value)            # "Sure — where to?"
print(result.conversation_id)  # auto-started session id

# Surface B — λ-DSL: a one-shot pipeline, agent, or recursion.
from fsm_llm import react_term
prog = Program.from_term(react_term(decide_prompt=..., synth_prompt=...))
result = prog.invoke(inputs={"question": "What is 17 * 23?"})
print(result.value)            # the agent's final answer
print(result.oracle_calls)     # 2  — exactly what the planner predicted
```

The second guarantee — `oracle_calls` matches the planner's static prediction — is **Theorem 2** of the design. Every `Fix` subtree carries a closed-form cost; the executor honors it.

## Why fsm-llm

- **One runtime.** FSM JSON dialogs and λ-DSL pipelines compile to the same AST. There is no separate "agents engine" plus "workflows engine" plus "FSM engine" — there is one β-reduction interpreter, and everything is a λ-term.
- **Theorem-2 cost prediction.** For any program with planner-bounded recursion, the executor's oracle-call count equals the planner's prediction. Budget LLM calls before running.
- **Provider-agnostic.** Built on [LiteLLM](https://github.com/BerriAI/litellm) — 100+ providers (OpenAI, Anthropic, Ollama, Google, Bedrock, Together, …) behind one interface. Switch with a string.
- **Layered API.** Four documented layers: L1 substrate, L2 composition, L3 authoring, L4 invoke. Use only what you need.
- **Typed throughout.** Pydantic v2 models for AST, definitions, results. Frozen, JSON-roundtrippable.

## Install

```bash
pip install fsm-llm                      # core: dialog, runtime, stdlib
pip install fsm-llm[reasoning]           # reasoning engine (no extra deps)
pip install fsm-llm[agents]              # agents (no extra deps)
pip install fsm-llm[workflows]           # workflows (no extra deps)
pip install fsm-llm[monitor]             # web dashboard (fastapi, uvicorn)
pip install fsm-llm[mcp]                 # MCP tool provider
pip install fsm-llm[otel]                # OpenTelemetry exporter
pip install fsm-llm[oolong]              # OOLONG long-context bench loader
pip install fsm-llm[all]                 # everything
```

Python 3.10–3.12. Set `OPENAI_API_KEY` (or any provider key) in `.env` or your shell.

## Three surfaces, one verb

`Program` is the unified entry point. Three constructors fix the mode at construction time; the same `.invoke(...)` returns a `Result` in every mode.

### 1. FSM JSON — dialogs with state

Author a state machine as JSON, compile to a λ-term, run turn by turn:

```python
from fsm_llm import Program

prog = Program.from_fsm("intake_bot.json")            # path or dict or FSMDefinition
result = prog.invoke(message="hello", conversation_id=None)
# result.value             — the response string
# result.conversation_id   — auto-started or echoed back
```

See [`examples/basic`](examples/basic), [`examples/intermediate`](examples/intermediate), and [`examples/advanced`](examples/advanced) for runnable FSMs.

### 2. λ-term — pipelines, agents, reasoning, recursion

Author a term directly in the DSL:

```python
from fsm_llm import Program, leaf, let_, var

term = let_(
    "summary", leaf(template="Summarise: {doc}", input_vars=("doc",)),
    leaf(template="Translate to French: {summary}", input_vars=("summary",)),
)
prog = Program.from_term(term)
result = prog.invoke(inputs={"doc": "..."})
```

Or use a stdlib factory:

```python
from fsm_llm import Program, react_term

term = react_term(
    decide_prompt="Given {question}, propose a tool call as JSON.",
    synth_prompt="Tool returned {observation}. Final answer:",
)
prog = Program.from_term(term)
result = prog.invoke(inputs={"question": "Capital of France?", "tool_dispatch": my_tools})
```

### 3. Factory — late-bound term construction

`Program.from_factory` calls a factory at construction time with explicit args:

```python
from fsm_llm import Program
from fsm_llm.stdlib.long_context import niah_term

prog = Program.from_factory(
    niah_term,
    factory_kwargs={"question": "Where is the artefact stored?", "tau": 256, "k": 2},
)
result = prog.invoke(inputs={"document": long_doc})
```

## Provider switching with `HarnessProfile` / `ProviderProfile`

Bundle prompt prefixes, leaf overrides, and provider kwargs at construction:

```python
from fsm_llm import HarnessProfile, ProviderProfile, Program, register_harness_profile

register_harness_profile(
    "ollama:qwen3.5:4b",
    HarnessProfile(
        system_prompt_base="You are a precise, terse assistant.",
        leaf_template_overrides={"leaf_001_summarise": "Be brief: {doc}"},
        provider_profile_name="ollama:qwen3.5:4b",
    ),
)

prog = Program.from_term(my_term, profile="ollama:qwen3.5:4b")
```

Profiles apply once at construction; **Theorem-2 strict equality is preserved**.

## Handlers

Hook into 8 timing points across an FSM turn or a term reduction. Two timings (`PRE_PROCESSING`, `POST_PROCESSING`) splice into the AST via `compose`; the other six dispatch host-side.

```python
from fsm_llm import HandlerBuilder, HandlerTiming, Program

audit = (
    HandlerBuilder("audit")
    .at(HandlerTiming.PRE_PROCESSING)
    .do(lambda **kw: log_event(kw))
    .build()
)

prog = Program.from_fsm("bot.json", handlers=[audit])
```

See [`docs/handlers.md`](docs/handlers.md).

## CLI

The package ships five console scripts:

```bash
fsm-llm --mode run --fsm path/to/fsm.json              # interactive run
fsm-llm --mode validate --fsm path/to/fsm.json         # schema check
fsm-llm --mode visualize --fsm path/to/fsm.json        # ASCII state graph

# Single-purpose subcommand aliases — same code, simpler signatures.
fsm-llm-validate  --fsm path/to/fsm.json
fsm-llm-visualize --fsm path/to/fsm.json
fsm-llm-monitor                                        # web dashboard (requires fsm-llm[monitor])
fsm-llm-meta                                           # interactive artifact builder
```

## Architecture at a glance

```
        FSM JSON (Category A)              λ-DSL (Category B / C)
              │                                    │
              ▼  fsm_llm.dialog.compile_fsm        ▼  fsm_llm.runtime.dsl
        ┌─────────────────────────────────────────────────────┐
        │                  λ-AST (typed Term)                 │
        │  Var · Abs · App · Let · Case · Combinator · Fix    │
        │                       · Leaf                        │
        └─────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────┐
        │ Executor (β-reduction, depth-bounded)    │
        │ Planner  (closed-form k*, τ*, d, calls)  │
        │ Oracle   (one per Program — uniform)     │
        │ Session  (per-conversation persistence)  │
        │ Cost     (per-leaf accumulator)          │
        └──────────────────────────────────────────┘
                                │
                                ▼
                       Program.invoke(...)  →  Result
```

The kernel (`runtime/`) is closed against the dialog surface — no upward edges. The dialog surface (`dialog/`) is the FSM-JSON compiler and orchestrator. The standard library (`stdlib/`) is named factories built on the kernel. `Program` (in `program.py`) is the L4 facade.

See [`docs/architecture.md`](docs/architecture.md) for the full picture.

## Documentation

| Doc | What it covers |
|---|---|
| [`docs/quickstart.md`](docs/quickstart.md) | Five-minute tour: install, FSM hello-world, λ-term hello-world, handlers, profiles |
| [`docs/api_reference.md`](docs/api_reference.md) | Every public name across L1–L4 with signatures and examples |
| [`docs/architecture.md`](docs/architecture.md) | The runtime, the layers, Theorem 2, the M3c default-flip |
| [`docs/handlers.md`](docs/handlers.md) | All 8 timing points; AST-side vs host-side; `HandlerBuilder` cookbook |
| [`docs/fsm_design.md`](docs/fsm_design.md) | Patterns and anti-patterns for authoring FSM JSON |
| [`docs/lambda.md`](docs/lambda.md) | The architectural thesis — why λ-calculus is the substrate |
| [`docs/lambda_fsm_merge.md`](docs/lambda_fsm_merge.md) | Canonical merge contract — invariants I1–I6, falsification gates G1–G5, deprecation calendar |
| [`docs/threat_model.md`](docs/threat_model.md) | Trust boundaries, T-01..T-11, dismissed proposals |
| [`CHANGELOG.md`](CHANGELOG.md) | Release notes |

## Examples

172 runnable examples across 10 trees. Run with:

```bash
python examples/basic/echo_bot/run.py
python examples/pipeline/react/run.py
python examples/long_context/niah_demo/run.py
```

All examples support OpenAI and Ollama out of the box. See `examples/README.md` for the index.

## Migrating from earlier versions

`0.6.0` is the post-cleanup release.

- **Removed (R13 epoch).** The shim modules `fsm_llm.api`, `fsm_llm.fsm`, `fsm_llm.pipeline`, `fsm_llm.prompts`, `fsm_llm.definitions`, `fsm_llm.llm`, `fsm_llm.session`, `fsm_llm.classification`, `fsm_llm.transition_evaluator`, and the `fsm_llm.lam` package are gone. Use canonical paths under `fsm_llm.dialog.*` and `fsm_llm.runtime.*` (or top-level `fsm_llm` for `compile_fsm`, `Executor`, `Term`, `leaf`, …).
- **Warning (I5 epoch).** `Program.run`, `Program.converse`, `Program.register_handler`, `from fsm_llm import API`, and `import fsm_llm_{reasoning,workflows,agents}` now emit `DeprecationWarning(removal="0.7.0")`. Migrate to `Program.invoke(...)`, the `handlers=` constructor kwarg, `Program.from_fsm(...)`, and `from fsm_llm.stdlib.<x>` respectively.
- **Renamed.** Long-context factories `niah`, `aggregate`, `pairwise`, `multi_hop`, `multi_hop_dynamic`, `niah_padded` are now `*_term` for consistency with every other stdlib slice. Bare names still resolve via `__getattr__` and warn until `0.7.0`.

See [`CHANGELOG.md`](CHANGELOG.md) for the full diff.

## Contributing

```bash
make install-dev      # editable install with all extras + pre-commit
make test             # pytest -v
make lint format      # ruff
make type-check       # mypy
```

`make test` should report ~3200 tests passing on a clean checkout.

## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE).
