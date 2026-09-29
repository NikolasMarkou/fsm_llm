# test_fsm_llm_eval

The pytest suite for `fsm_llm.eval`, the evaluation package of the FSM-LLM repository, and for its command line `fsm-llm-eval`. It lives at `tests/test_fsm_llm_eval/` and holds 261 tests in 8 files.

## What it is for

`fsm_llm.eval` measures how well a model drives FSM-LLM. It has two modes. `fsm-llm-eval examples` runs every example script under `examples/` in a subprocess and gives each a score from 0 (CRASH) to 4 (PASS). `fsm-llm-eval run <dataset>` plays scripted conversations ("cases") against an FSM several times ("trials") and reports a pass rate with a confidence interval. These tests check both modes, the settings system, the statistics, the result files, and the command line exit codes. No test calls a real LLM.

## How it works

- Conversation cases run against `MockLLM2Interface` from `tests/conftest.py`, a fake 2-pass LLM. A small test FSM moves from `start` to `done` only when the fake LLM "extracts" a `name`, so the same case can be made to pass or fail.
- Example runs use a tiny examples tree written to a temporary directory. Each fake `run.py` runs under the current Python, so no real example and no model is touched.
- The scoring rubric is pinned by a golden table copied from the original `scripts/eval.py` at commit `0facf56`. A mismatch means the rubric changed.
- `scripts/harness_bench.py` keeps its own stdlib copies of the statistics and record helpers so it can run without network access. Parity tests check both copies give identical results.

## Files

- `__init__.py` - empty package marker.
- `test_bench_parity.py` - checks `scripts/harness_bench.py` helpers match `fsm_llm.eval.stats` and `fsm_llm.eval.records`, and that the bench statistics run with sockets disabled.
- `test_cases.py` - dataset loading, expectation checks, single trials, whole runs, `run_dataset`, the `run` subcommand, interrupts and `--fail-under`.
- `test_cli.py` - the `examples` subcommand: usage errors, `--list`, config errors, runs, `--fail-under`, Ctrl-C.
- `test_config.py` - `EvalConfig` defaults, layered merging, validation, model resolution, LLM settings.
- `test_examples.py` - example discovery, timeout precedence, subprocess runs, log files, `results.json` and `scorecard.md`, output directories, interrupts.
- `test_records.py` - JSONL rows, JSON writing, UTC time, git hashes, model slugs, run directory names, exception hierarchy.
- `test_scoring.py` - golden table for `classify_result` and the score labels.
- `test_stats.py` - Wilson interval, Fisher exact test, `pass_rate`, and exact `below_percent` threshold checks.

## How to use it

Run from the repository root with the project virtualenv:

```bash
.venv/bin/python -m pytest tests/test_fsm_llm_eval/
.venv/bin/python -m pytest tests/test_fsm_llm_eval/test_scoring.py -v
```

## Things to know

- Run from the repository root. `test_cases.py` imports `tests.conftest`, and several tests read real repository files: `evaluation/datasets/simple_greeting_cases.json`, `evaluation/2026-04-01_23-35_4025a78_qwen3.5-4b/results.json`, `scripts/harness_bench.py`, and the `examples/` tree.
- Git tests compare against `git rev-parse HEAD`, so the checkout must be a git repository.
- Do not regenerate the golden table in `test_scoring.py` from the current code. It exists to keep new scores comparable with old runs.
- `test_examples.py` starts real subprocesses and one of them waits for a 1 second timeout.
- The repository root `CLAUDE.md` states this suite's test count (261). `tests/test_packaging.py` checks that number against `pytest --collect-only`, so update it when adding or removing tests.
