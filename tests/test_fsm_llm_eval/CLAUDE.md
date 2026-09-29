# test_fsm_llm_eval

Path: `tests/test_fsm_llm_eval`
Purpose: Offline pytest suite (261 tests, 8 files) for `fsm_llm.eval` and the `fsm-llm-eval` CLI (`fsm_llm.eval.__main__`), plus parity with `scripts/harness_bench.py`.

## Scope

Covers `fsm_llm.eval`: `config`, `records`, `stats`, `scoring`, `examples`, `cases`, `__main__`, and the exception hierarchy. Two eval modes are tested:

- `examples` mode: run each `examples/<category>/<name>/run.py` (and `run_manual.py`) as a subprocess, score 0 (CRASH) to 4 (PASS) with `classify_result`, write `results.json`, `scorecard.md`, and per-example logs.
- `run` mode (cases): replay JSON/JSONL conversation cases N trials each against an FSM, check expectations, write `rows.jsonl`, `results.json`, `summary.md` with Wilson intervals.

No test calls a real LLM or network. Not covered here: the rest of `fsm_llm` core, `scripts/harness_bench.py` beyond the seven mirrored helpers.

## Architecture

Test doubles and fixtures:

| Double | Where | What it does |
| --- | --- | --- |
| `MockLLM2Interface(extraction_data=...)` | `tests/conftest.py` | Fake 2-pass LLM; `_passing()` extracts `{"name": "Ada"}`, `_failing()` extracts `{}` |
| `_name_fsm()` | `test_cases.py` | FSM `start -> done`, gated on `{"!!": [{"var": "name"}]}` |
| `offline_cli` fixture | `test_cases.py` | Monkeypatches `cli.run_cases` with `functools.partial(run_cases, llm_interface_factory=factory)` |
| `_CapturingAPI` | `test_cases.py` | Replaces `cases_mod.API`; records kwargs passed to `from_definition`/`from_file`, then raises `RuntimeError("captured")` |
| `examples_dir` fixture | `test_examples.py` | tmp tree: `basic/ok/run.py`, `basic/ok/run_manual.py`, `basic/boom/run.py`, `agents/slow/run.py` (sleeps 30 s), `toplevel/run.py` (2 path parts, skipped) |
| `examples_dir` fixture | `test_cli.py` | tmp tree: `basic/ok/run.py` (prints two states), `basic/bad/run.py` (exit 3) |
| `_config(examples_dir, **overrides)` | `test_examples.py` | `merge_config` layer with `category_timeouts={"agents": 1}` so the slow example times out in 1 s |
| `hb` fixture | `test_bench_parity.py` | Loads `scripts/harness_bench.py` via `importlib.util.spec_from_file_location` |
| `_GOLDEN` | `test_scoring.py` | 32 rows `(id, exit_code, stdout, stderr, duration, timed_out, score, failures)` from the original `scripts/eval.py` at commit `0facf56` |

## Key files

| File | Tests | Covers |
| --- | --- | --- |
| `test_bench_parity.py` | 28 | `wilson_ci`, `fisher_exact_two_sided`, `append_row`/`read_rows`, `_write_json`, `_utc_now`, `_git_commit` equal between `harness_bench` and `fsm_llm.eval`; bench stats run in a subprocess with `socket.socket` patched to raise, and `fsm_llm` must not be in `sys.modules` |
| `test_cases.py` | 65 | `load_cases`, `check_expectations`, `run_case_trial`, `run_cases`, `run_dataset`, `run` subcommand, checks hook, interrupt, `--fail-under` boundary |
| `test_cli.py` | 23 | `main_cli`/`build_parser` for `examples`: exit codes, `--list`, config errors, runs, `--fail-under`, Ctrl-C |
| `test_config.py` | 34 | `EvalConfig`, `merge_config`, `load_config`, `resolve_model`, reserved `llm_kwargs` |
| `test_examples.py` | 24 | `discover_examples`, `get_timeout`, `run_examples`, `run_example`, `create_output_dir`, logs, `results.json`, `scorecard.md` |
| `test_records.py` | 21 | `append_row`, `read_rows`, `write_json`, `utc_now`, `git_commit`, `git_short_hash`, `model_slug`, `make_run_dir`, exceptions |
| `test_scoring.py` | 36 | `classify_result` golden parity, `SCORE_LABELS`, `MAX_SCORE` |
| `test_stats.py` | 30 | `wilson_ci`, `fisher_exact_two_sided`, `pass_rate`, `below_percent` |
| `__init__.py` | 0 | Empty package marker |

## Public interface under test

CLI exit codes (`main_cli(argv) -> int`):

| Situation | Code |
| --- | --- |
| Success, including a low score with no `--fail-under` | 0 |
| No subcommand, unknown flag (argparse would give 2), bad config key/value, missing or non-UTF-8 config, missing examples dir, no matching examples, non-empty `--output-dir`, bad or missing dataset | 1 |
| Score below `--fail-under` (stderr contains `below --fail-under`) | 2 |
| Ctrl-C (partial report still written, `interrupted: true`) | 130 |
| `--help` | `SystemExit(0)` |

`--version` prints `fsm-llm-eval <version>`. `examples --help` must list `--model --workers --timeout --category --filter --output-dir --list --examples-dir --python --config --fail-under`. `run --help` must list `--model --trials --workers --output-dir --config --fail-under --list`; `run` also takes `--temperature` and `--max-tokens`. Parser defaults for `model`, `workers`, `timeout`, `output_dir`, `fail_under` are `None` so config layers apply.

## Data shapes

`EvalConfig()` defaults: `model=None`, `workers=4`, `timeout=120`, `output_root="evaluation"`, `output_dir=None`, `fail_under=None`, `examples_dir="examples"`, `python=None`, `example_timeouts={}`, `trials=3`, `temperature=None`, `max_tokens=None`, `llm_kwargs={}`. Other keys used in tests: `category`, `name_filter`, `example_inputs`, `category_timeouts`.

Case (`ConversationCase`): `id`, `fsm` (inline dict or path string), `turns` (non-empty), `expect`, optional `initial_context`. Unknown keys rejected. `Expectations` keys: `final_state`, `visited_states`, `context`, `context_keys`, `responses_contain`, `ended`; at least one required.

Dataset: a JSON list, a JSON object `{"config": {...}, "cases": [...]}`, or JSONL. `load_cases(path) -> (cases, config_dict)`.

`run` `results.json`: `cases[]` with `id`, `k`, `n`, `wilson_ci`, `first_failure`, `failure_counts`; `overall` (`pass_rate` shape `{"k","n","rate","wilson_ci"}`); `config`; `dataset`; `interrupted`. `llm_kwargs` values are written as `"<not recorded>"`.

`examples` `results.json`: every key of the historical `evaluation/2026-04-01_23-35_4025a78_qwen3.5-4b/results.json` plus exactly `wall_time_s`, `workers`, `default_timeout`, `evaluator`, `interrupted`. `distribution` keys `"0".."4"`. Logs at `logs/<category>/<category>_<name>.log`.

Run dir name: `YYYY-MM-DD_HH-MM_<short hash or "unknown">_<model_slug>`, collisions get `_2`, `_3`. `model_slug("ollama_chat/qwen3.5:4b") == "qwen3.5-4b"`.

## Invariants and constraints

- Golden table in `test_scoring.py`: never regenerate from the new `classify_result`. A mismatch means the rubric changed. `test_table_reaches_every_score` requires scores 0 to `MAX_SCORE` all appear; ids must be unique.
- `classify_result` must ignore a precomputed `result.score`/`result.failures`.
- Parity pairs must stay byte-identical (JSON output, JSONL rows) and value-identical (stats, rejections of impossible counts via `ValueError`).
- The bench script must not import `fsm_llm` (it pulls litellm, which opens a socket; D-008 of plan 581c2634).
- `--fail-under` compares with integers via `below_percent`: 57/100 meets 57, 116/200 meets 58.
- Timeout precedence: per-example table > category table > default, even when the default is lower. Config tables override built-ins without mutating `EXAMPLE_TIMEOUTS`.
- Discovery order: all `run.py` sorted by path, then `run_manual.py` (named `<name>_manual`, `interactive=True`); paths with 2 parts are skipped.
- A case whose FSM ends early with turns unsent fails unless `expect.ended` is declared.
- Setting precedence for `run`: defaults (3) < dataset config < `--config` file < flags.
- `main_cli` must not call `fsm_llm.logging.setup_cli_logging`.
- An explicit non-empty output dir is refused; default run dirs are never reused.

## Dependencies

- `fsm_llm.eval` and its submodules (`__main__`, `cases`, `examples`, `stats`, `records`, `constants`).
- `fsm_llm.constants` (`DEFAULT_LLM_MODEL`, `ENV_LLM_MODEL`), `fsm_llm.definitions.FSMError` (root of `EvalError`), `fsm_llm.logging`.
- `tests.conftest.MockLLM2Interface`.
- Repository files: `evaluation/datasets/simple_greeting_cases.json` (config `{"trials": 3}`, includes case `name_is_extracted`, passes fully with extraction `{"user_name": "Alex"}`), the historical `results.json` above, `scripts/harness_bench.py`, `examples/` (listing must include `basic/simple_greeting` and `classification/classified_transitions_manual`), and a git checkout.
- pytest only; no network.

## Failure modes

- Run outside the repo root: `from tests.conftest import ...` and repo-file tests fail.
- Not a git checkout: `git_commit`/`git_short_hash`/bench `_git_commit` tests fail.
- Editing the sample dataset, the historical results file, or example names breaks tests in `test_cases.py`, `test_examples.py`, `test_cli.py`.
- `test_examples.py` spawns subprocesses and sleeps; slow machines may stretch wall-time assertions (`wall_time < sum(durations) + 1.0`).

## Working here

- Name tests after the defect they guard; many docstrings start `Defect guarded:` with a review id (W1, W2, W6, NOTE 10, NOTE 15).
- Keep every case test able to both pass and fail (use `_passing()` and `_failing()`), so a checker that checks nothing is caught.
- A new helper mirrored in `scripts/harness_bench.py` needs a parity test in `test_bench_parity.py`.
- After adding or removing tests, re-measure with `.venv/bin/python -m pytest tests/test_fsm_llm_eval --collect-only -q | tail -1` and update the `pytest tests/test_fsm_llm_eval/ ... (261 tests)` line and totals in the root `CLAUDE.md`; `tests/test_packaging.py` checks those literals against collection.
- Run: `.venv/bin/python -m pytest tests/test_fsm_llm_eval/`.
