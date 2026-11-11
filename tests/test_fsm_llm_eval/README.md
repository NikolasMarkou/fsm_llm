# test_fsm_llm_eval

The pytest suite for `fsm_llm.eval`, the evaluation package of the FSM-LLM repository, and for its command line `fsm-llm-eval`. It lives at `tests/test_fsm_llm_eval/` and holds 430 tests in 10 files.

## What it is for

`fsm_llm.eval` measures how well a model drives FSM-LLM. It has two modes. `fsm-llm-eval examples` runs every example script under `examples/` in a subprocess and gives each a score from 0 (CRASH) to 4 (PASS). `fsm-llm-eval run <dataset>` plays scripted conversations ("cases") against an FSM several times ("trials") and reports a pass rate with a confidence interval. These tests check both modes, the settings system, the statistics, the result files, and the command line exit codes. No test calls a real LLM.

## How it works

- Conversation cases run against `MockLLM2Interface` from `tests/conftest.py`, a fake 2-pass LLM. A small test FSM moves from `start` to `done` only when the fake LLM "extracts" a `name`, so the same case can be made to pass or fail.
- Example runs use a tiny examples tree written to a temporary directory. Each fake `run.py` runs under the current Python, so no real example and no model is touched.
- The scoring rubric is pinned by a golden table copied from the original `scripts/eval.py` at commit `0facf56`. A mismatch means the rubric changed.
- `scripts/bench_common.py` is the one stdlib source of the statistics and record helpers that `agents_bench.py` and `skills_bench.py` share, so they run without network access. Parity tests check it gives the same results as `fsm_llm.eval`.

## Files

- `__init__.py` - empty package marker.
- `test_bench_parity.py` - checks `scripts/bench_common.py` helpers match `fsm_llm.eval.stats` and `fsm_llm.eval.records`, and that the bench statistics run with sockets disabled.
- `test_cases.py` - dataset loading, expectation checks, single trials, whole runs, `run_dataset`, the `run` subcommand, interrupts and `--fail-under`.
- `test_cli.py` - the `examples` subcommand: usage errors, `--list`, config errors, runs, `--fail-under`, Ctrl-C.
- `test_config.py` - `EvalConfig` defaults, layered merging, validation, model resolution, LLM settings.
- `test_examples.py` - example discovery, timeout precedence, subprocess runs, log files, `results.json` and `scorecard.md`, output directories, interrupts (Ctrl-C and SIGTERM, including one before the first example finishes, which still leaves `logs/`).
- `test_example_inputs.py` - every stdin script in `EXAMPLE_INPUTS` checked against the example it feeds (no script for an example that never reads stdin, no trailing `quit` for the meta builders, a task naming the gated `publish_report` tool for `react_hitl_combined`).
- `test_records.py` - JSONL rows, JSON writing, UTC time, git hashes, model slugs, run directory names, exception hierarchy.
- `test_scoring.py` - golden table for `classify_result`, the score labels, and the rule that a line-anchored `Error:` or `Success: False`, or an extraction rate of at most 50%, cannot score 4.
- `test_timeout_table.py` - reads the examples' source with `ast` and checks that no harness timeout ties or is below an agent's own `timeout_seconds` budget.
- `test_stats.py` - Wilson interval, Fisher exact test, `pass_rate`, and exact `below_percent` threshold checks.

## How to use it

Run from the repository root with the project virtualenv:

```bash
.venv/bin/python -m pytest tests/test_fsm_llm_eval/
.venv/bin/python -m pytest tests/test_fsm_llm_eval/test_scoring.py -v
```

## Things to know

- Run from the repository root. `test_cases.py` imports `tests.conftest`, and several tests read real repository files: `evaluation/datasets/simple_greeting_cases.json`, `scripts/bench_common.py`, and the `examples/` tree.
- Git tests compare against `git rev-parse HEAD`, so the checkout must be a git repository.
- Do not regenerate the golden table in `test_scoring.py` from the current code. It pins the original rubric for every input the 2026-11-06 change (`Error:` and `Success: False` lines, extraction at most 50%) does not touch; `TestReportedFailure` pins the change.
- `test_examples.py` starts real subprocesses and one of them waits for a 1 second timeout. One slow-marked test starts a real `fsm-llm-eval examples` process, sends it SIGTERM and checks the partial report and exit 130. `TestSigtermEndsRunningExamples` runs an example that starts its own subprocess, sends SIGTERM and checks, by pid through `/proc`, that both are gone at once (in-process and with a real CLI process, which must exit promptly), and that a timeout ends the example's subprocess too.
- The repository root `AGENTS.md` states this suite's test count (420). `tests/test_packaging.py` checks that number against `pytest --collect-only`, so update it when adding or removing tests.
