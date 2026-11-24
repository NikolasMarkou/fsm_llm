# Command line

> Covers FSM-LLM v0.13.0

The distribution installs one console script per tool, all prefixed `fsm-llm`, and the reasoning
engine has one more entry that is run as a module. Each section gives the entry point (the
`module:function` of `[project.scripts]` in `pyproject.toml`), what it is for and its usage.
`--help` on any of them prints every option.

## Common rules

**Model.** The model is never built in. The CLIs that call one (`fsm-llm`, `fsm-llm-eval`,
`fsm-llm-meta`, `python -m fsm_llm.reasoning`) take `--model` (`-m` on `fsm-llm` and the
reasoning CLI), then the process environment variable `LLM_MODEL`, then the file named by
`--env-file`. With none of the three they exit 1 with one message on stderr
(`ModelNotConfiguredError`), `fsm-llm-eval` included. `fsm-llm-monitor` starts without a model
and answers a launch request that names none, with no `LLM_MODEL`, with HTTP 400. A machine
that wants a default keeps `LLM_MODEL` in its own `.env` (gitignored; `make monitor` and
`make eval-examples` pass `--env-file .env`), never in code.

**`--env-file PATH`.** Importing `fsm_llm` reads no `.env` file. The five model CLIs
(`fsm-llm`, `fsm-llm-eval`, `fsm-llm-meta`, `fsm-llm-monitor`, `python -m fsm_llm.reasoning`)
take one flag, `--env-file PATH`, that loads the file into the environment first. A variable the
process already has wins; a blank one counts as unset; a missing file is one stderr line and
exit 1. The CLIs that call no model (`fsm-llm-validate`, `fsm-llm-visualize`, `fsm-llm-okf`,
`fsm-llm-skills`) do not take it.

**Exit codes.** 0 is success, 1 is a failure, 130 is Ctrl-C. A usage error exits 2 (argparse),
except in `fsm-llm-eval`, which exits 1 for it. `fsm-llm-eval` exits 2 only when the score is
below `--fail-under PCT`.

## fsm-llm

Entry point: `fsm_llm.__main__:main_cli`.

Runs an FSM definition as an interactive conversation, or validates or draws it with
`--mode validate` and `--mode visualize`. Running needs a model.

```bash
fsm-llm --fsm <path.json> [--model M] [--env-file PATH]
```

Other options: `--mode {run,validate,visualize}`, `--style {full,compact,minimal}`,
`--history-size N` (exchanges kept, default 5), `--message-length N` (default 1000) and
`--version`.

## fsm-llm-validate

Entry point: `fsm_llm.core.verify.cli:main_cli`.

Validates an FSM definition file, or an [executable](executables.md) document: the command reads
the file and routes on its `schema` header (`fsm-llm/executable`), never on the extension. The
report is JSON on stdout and the exit code is 0 for a valid file and 1 otherwise. For an
executable the report is the whole load sequence at once, each line tagged by its step, and no
names are resolved. `--upgrade` validates first and then rewrites a definition that declares
version 4.0 up to below the current one to the current version; it exits 1 when the upgrade is
refused or fails, and it never rewrites an executable (it prints the report, warns and exits 1).
Calls no model.

```bash
fsm-llm-validate --fsm <path.json> [--upgrade]
```

## fsm-llm-visualize

Entry point: `fsm_llm.core.visualize.cli:main_cli`.

Draws an FSM definition as an ASCII, Mermaid or DOT diagram. `--style` applies to ASCII only.
Calls no model.

```bash
fsm-llm-visualize --fsm <path.json> [--format ascii|mermaid|dot] [--style full|compact|minimal]
```

## fsm-llm-monitor

Entry point: `fsm_llm.monitor.__main__:main_cli`.

Starts the web dashboard (REST, WebSocket and a single-page app) at `http://127.0.0.1:8420`.
Needs the `monitor` extra. `--api-key` (or the `FSM_LLM_MONITOR_API_KEY` variable) guards the
mutating and sensitive routes; `--otel` exports monitor events as OpenTelemetry spans.

```bash
fsm-llm-monitor [--host HOST] [--port PORT] [--no-browser] [--api-key KEY] [--otel] [--env-file PATH]
```

Also `--info` and `--version`, which print and exit.

## fsm-llm-meta

Entry point: `fsm_llm.agents.meta_cli:main_cli`.

An interactive builder: a conversation that produces an FSM, workflow or agent definition as a
JSON artifact. Needs a model.

```bash
fsm-llm-meta [--model M] [--env-file PATH] [--output FILE] [--temperature T] [--max-turns N]
             [--executables DIR]
```

`--executables DIR` reads the pattern documents (`*.json`, any case of the suffix, a BOM accepted)
of a directory, and the `id` of each becomes a valid `agent_type` beside the built-in ones: it is
listed in the enum of the build schema the model answers, and a reply naming an id that is neither
built in nor in the directory is rejected as before. A file that cannot be read or is not a valid
document is one `warning:` line on stderr and is skipped; a file whose id is a built-in one is
refused the same way and never replaces it; a directory that does not exist is an `error:` line
and exit 1. The files are read as operator input, not as model output: the checks for a document a
model wrote (V7) do not apply to them, so the directory is as trusted as the command line.

## fsm-llm-eval

Entry point: `fsm_llm.eval.__main__:run`.

Two commands. `examples` runs the repository examples and scores each 0 to 4; `run` plays
scripted conversations against an FSM, N trials each, and reports pass rates with Wilson
confidence intervals. Both need a model and write their results under `evaluation/`. Settings
precedence is defaults, then the dataset `config`, then `--config FILE`, then flags. With one
model server slot use `--workers 1`.

```bash
fsm-llm-eval examples [--category C] [--fail-under PCT] [--env-file PATH]
fsm-llm-eval run <dataset.json|.jsonl> [--trials N] [--env-file PATH]
```

Exit codes differ from the common rule: a usage error exits 1, and exit 2 means the score fell
below `--fail-under PCT`.

## fsm-llm-okf

Entry point: `fsm_llm.okf.cli:main_cli`.

Works with Open Knowledge Format bundles. Calls no model.

```bash
fsm-llm-okf <validate|list|show|search|export> ...
```

`validate` checks a bundle (`--strict`), `list` lists its concepts, `show` prints one concept,
`search` searches titles, tags and bodies, and `export` writes a semantic-memory JSON file as a
bundle. Every command except `export` takes `--max-seconds`.

## fsm-llm-skills

Entry point: `fsm_llm.skills.cli:main_cli`.

Reads Agent Skills folders (`SKILL.md`). It is read-only and runs nothing from a skill. Calls
no model.

```bash
fsm-llm-skills <validate|list|show> PATH...
```

`validate --strict` is the strict check; `show NAME PATH...` prints a skill body, or one bundled
file with `--file RELPATH`.

## python -m fsm_llm.reasoning

Entry point: `fsm_llm.reasoning.__main__` (run as a module: the reasoning CLI has no console
script).

Solves a problem with one of the reasoning strategies and prints the answer with a trace. Needs
a model.

```bash
python -m fsm_llm.reasoning "problem" [--type T] [--executables DIR] [--model M] [--env-file PATH]
```

`--executables DIR` adds the strategy documents (`*.json`) of a directory to the nine built-in
strategies. It is read first: `--type` takes the name of any eligible strategy of that registry
(a name, never an alias) and anything else is a usage error (exit 2); `--list-types` prints the
same registry; the classifier can choose them too. A file that cannot be read, or whose id is a
built-in one (it never replaces it), is one `warning:` line on stderr and is skipped, and a
directory that does not exist is an `error:` line and exit 1. A document that is not an eligible
strategy (not `deprecated`, a name that is an ASCII identifier, contract inputs within the four
problem keys) is skipped as well; run with `--verbose` to see why. Ctrl-C ends the run with the
interrupt exit code of the common rules above and closes the engine.

Other options: `--context JSON`, `--output text|json|detailed`, `--save FILE`, `--verbose`,
`--quiet`, `--list-types` and `--version`.

## Next steps

- The [Quick Start](quickstart.md) runs `fsm-llm` on a first definition.
- The [API Reference](api_reference.md) covers the same tools from Python.
- [Evaluation](evaluate.md) describes the scorer behind `fsm-llm-eval`.
