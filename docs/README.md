# docs

The `docs/` folder at the root of the FSM-LLM repository holds the long-form user guides for the framework. FSM-LLM is a Python library that runs conversations as JSON-defined finite state machines (FSMs), with an LLM doing data extraction and response writing. It contains Markdown files only, no code: nine content files plus this index and a maintainer-notes file (`CLAUDE.md`, for coding agents).

## What it is for

The package READMEs say what each part is. These guides go further: they teach you to build a bot, design states and transitions, write handlers, and they list the public API. Five files are current guides for version 0.11.0, and each one starts with the line `> Covers FSM-LLM v0.11.0`. The other four are dated design records, kept to explain design choices, not to teach usage:

- Three come from the "Strands" initiative, which adapted features from the Strands Agents SDK. All of it shipped in v0.4.0, and their counts and paths are from the time of writing.
- `agents_roadmap.md` is the audit record and roadmap for `fsm_llm.agents` (dated 2026-09-29, with measured baselines added later). It has no `Covers` header, and parts of it describe unreleased work on top of 0.11.0.

If you only want to use the library, read the five guides. Per-package READMEs under `src/fsm_llm/` and the repo root `README.md` hold the rest.

## How it works

Read the current guides in this order:

```mermaid
flowchart LR
    Q[quickstart.md] --> D[fsm_design.md] --> H[handlers.md]
    Q --> A[architecture.md]
    H --> R[api_reference.md]
    A --> R
```

- Start with `quickstart.md` to install the package and run a first bot.
- `fsm_design.md` covers how to shape states and transitions. It ends with a link to `handlers.md`.
- `architecture.md` explains how a message moves through the system.
- `api_reference.md` is the lookup table for classes, methods, CLIs and exceptions.

Some tests read these files. `tests/test_fsm_llm/test_docs_snippets.py` finds every fenced `json` or `python` block that contains `"initial_state"` in `quickstart.md`, `api_reference.md`, `architecture.md`, `fsm_design.md` and `handlers.md` (and in the root `README.md`, root `CLAUDE.md` and `src/fsm_llm/README.md`), and loads it through `FSMDefinition`. A broken full FSM example fails the test suite. `tests/test_fsm_llm_regression/test_regression_cli_and_exports.py` checks that `quickstart.md` does not mention `examples/basic/quiz`, `examples/intermediate/customer_service` or `python main.py`.

## Files

- `quickstart.md` - install, API key and environment variables (`LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `FSM_PATH`), a first bot (`friendly_greeter`), a first handler, streaming and session persistence, CLI commands, troubleshooting.
- `fsm_design.md` - design guide: one purpose per state, JsonLogic transitions, patterns (Gatekeeper, Collector, Router, Confirmer), priority and ambiguity, `evaluation_priority`, `llm_description`, context scope, `handler_only_keys`, how corrections work, FSM stacking, classification routing, `completion` states (tool calling and structured turns), designing for agents, anti-patterns, testing and validation.
- `handlers.md` - the 8 handler timings, the `HandlerBuilder` methods, `error_mode` (`"continue"` or `"raise"`), critical handlers, what ERROR handlers can and cannot do, common patterns, testing, how each extension uses handlers.
- `architecture.md` - the 2-pass design, core components (`api.py`, `fsm.py`, `pipeline.py`, `prompts.py`, `handlers.py`, `llm.py`, `transition_evaluator.py`), message, start and streaming flows, context handling, stacked-FSM merge rules, security, performance, how the six extension subpackages plug in, a detailed section on the harness, extension points.
- `api_reference.md` - `API` constructor and methods (including steps without a user message), `HandlerBuilder`, the other builders and `BuildError`, `HandlerTiming`, `LLMInterface`, `WorkingMemory`, `TransitionEvaluatorConfig`, classification, reasoning, agents, workflows, harness, eval (CLI, `EvalConfig`, dataset schema, output files), monitor, exception tree, constants.
- `strands_features.md` - historical: the report that picked 12 Strands features to adapt, plus the features it rejected.
- `strands_features_phase_1.md` - historical: delivery note for schema-enforced output, the hidden `metadata` memory buffer, response streaming and session persistence (commit `a7e3d88`).
- `strands_features_phase_2.md` - historical: plan for the other 8 features (OTEL, swarm, agent graph, MCP, semantic tool retrieval, dependency-based workflow parallelism, SOPs, A2A).
- `agents_roadmap.md` - design record, not a guide: the `fsm_llm.agents` audit (verdict per audit id, commits, known open items, deferred Track B) and recorded baselines (G3 examples run, agent bench blocks B0 to B2). Current agents contracts live in `src/fsm_llm/agents/README.md` and the Agents section of `api_reference.md`. Source code and `scripts/agents_bench.py` point to it by name.
- `CLAUDE.md` - maintainer notes for coding agents; not part of the user guides.

## How to use it

Open the files in any Markdown viewer. Snippets assume the package is installed, and a model is configured (`LLM_MODEL`, see `quickstart.md`). Run the examples from a repository clone:

```bash
# from a clone of the repository (the examples are not in the package)
pip install -e .
python examples/basic/form_filling/run.py
```

After editing a guide, run the snippet tests:

```bash
.venv/bin/python -m pytest tests/test_fsm_llm/test_docs_snippets.py tests/test_fsm_llm_regression/test_regression_cli_and_exports.py
```

## Things to know

- Extensions are documented as subpackages of `fsm_llm`: `fsm_llm.reasoning`, `fsm_llm.workflows`, `fsm_llm.agents`, `fsm_llm.monitor`, `fsm_llm.harness`, `fsm_llm.eval`. This matches the code under `src/fsm_llm/`.
- The Strands files quote counts, file paths and "not yet implemented" lists as they were when written. Some of those paths use the subpackage layout. Their counts are not current. `agents_roadmap.md` is likewise a dated record: its status lines and baselines are as recorded on their dates.
- Monitor documentation belongs in `api_reference.md` (the Monitor section). Do not create a separate `docs/monitor.md`.
- Pre-registered benchmark data is not documented here; see `scripts/bench_data/README.md`.
- `required_context_keys` never blocks a transition. The guides gate transitions with a condition (`requires_context_keys` plus `logic`). Keep new examples consistent with this.
- Source code points to these files by name: `src/fsm_llm/fsm.py` mentions `docs/architecture.md` and `docs/handlers.md`. Renaming a file breaks those pointers and the tests above.
