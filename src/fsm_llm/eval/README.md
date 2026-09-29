# fsm_llm.eval

Evaluation tools for FSM-LLM. It runs the repository examples and scores them, and it runs scripted conversations against any FSM and checks the results, reporting pass rates with confidence intervals.

## What it is for

LLM behaviour varies from run to run, so "it worked once" is not a measurement. This package gives two ways to measure:

- **Examples evaluation** (`fsm-llm-eval examples`): runs every `examples/<category>/<name>/run.py` in its own process, feeds interactive ones a fixed script of user input, scores each 0 (crash) to 4 (pass) from its output, and writes a scorecard. This is the evaluation described in `EVALUATE.md`; `scripts/eval.py` still works and runs the same thing.
- **Conversation evaluation** (`fsm-llm-eval run`): you write a small dataset of conversations: which FSM, what the user says, and what should be true at the end (final state, states visited, extracted values, words in the replies, whether it ended). Each conversation runs several times, and the report gives the pass rate per case with a 95% Wilson confidence interval.

## How it works

```mermaid
flowchart TD
    C[defaults < dataset config < --config file < flags] --> E[fsm-llm-eval examples]
    C --> R[fsm-llm-eval run dataset]
    E --> P[one subprocess per example, N in parallel]
    P --> S[heuristic score 0-4 per example]
    S --> O1[scorecard.md, results.json, logs/]
    R --> T[each case x trials, fresh fsm_llm.API per trial]
    T --> X[check expectations: pass or fail]
    X --> O2[rows.jsonl, results.json, summary.md]
```

Every run writes into a new directory under `evaluation/`, named `<date>_<time>_<git-hash>_<model>`. If that name already exists, `_2`, `_3`, ... is added; a run never overwrites another. An explicit `--output-dir` must be new or empty.

## Files

- `__main__.py` - the `fsm-llm-eval` command and its two subcommands.
- `config.py` - `EvalConfig` (every setting with its default), loading JSON config files, merging settings layers.
- `examples.py` - finds, runs and reports example scripts.
- `scoring.py` - the 0-4 heuristic that reads an example's output.
- `cases.py` - conversation datasets, one trial of one case, whole runs and their reports.
- `records.py` - append-only JSONL rows, JSON output, run-directory names.
- `stats.py` - Wilson confidence interval, Fisher exact test, `pass_rate`, the exact `--fail-under` comparison.
- `_pool.py` - the thread pool both runners use, with Ctrl-C handling.
- `constants.py` - defaults, exit codes, score labels, and the per-example input and timeout tables.
- `exceptions.py` - error classes.
- `__init__.py`, `__version__.py` - public exports, version.

## How to use it

Examples:

```bash
fsm-llm-eval examples --list                      # what would run
fsm-llm-eval examples --model ollama_chat/qwen3.5:4b --workers 4
fsm-llm-eval examples --category agents --fail-under 80
```

A conversation dataset (`cases.json`), with the FSM path relative to the dataset file:

```json
{
  "config": {"trials": 3},
  "cases": [
    {
      "id": "reaches_farewell",
      "fsm": "bots/greeting.json",
      "turns": ["Hi!", "That's all, bye!"],
      "expect": {"final_state": "farewell", "ended": true}
    },
    {
      "id": "remembers_name",
      "fsm": "bots/greeting.json",
      "turns": ["My name is Ada"],
      "expect": {"context": {"name": "Ada"}, "responses_contain": ["Ada"]}
    }
  ]
}
```

```bash
fsm-llm-eval run cases.json --list        # case ids
fsm-llm-eval run cases.json --trials 5 --model gpt-4o-mini
fsm-llm-eval run cases.json --temperature 0 --max-tokens 512
```

LLM settings beyond model, temperature and max tokens go in `llm_kwargs` in a config file or the dataset's `config`; they are passed to `fsm_llm.API` (for example `{"llm_kwargs": {"api_base": "http://localhost:11434", "max_history_size": 10}}`). `results.json` records their keys, never their values.

From Python, one call does what `fsm-llm-eval run` does:

```python
from fsm_llm.eval import TrialResult, run_dataset

def polite(trial: TrialResult) -> list[str]:      # a custom check
    return [] if "thank" in trial.responses[-1].lower() else ["no thanks said"]

report = run_dataset("cases.json", trials=2, checks=[polite])
print(report.overall)   # {"k": ..., "n": ..., "rate": ..., "wilson_ci": [lo, hi]}
```

`run_dataset(path, *, config=None, llm_interface_factory=None, checks=(), progress=None, **overrides)` layers settings like the CLI: `config` (a JSON file, a dict or an `EvalConfig`) sits where `--config` does and `overrides` where the flags do. A check gets each finished trial and returns one message per failure. Pass `llm_interface_factory=lambda: MyFakeLLM()` to test offline without a model. The building blocks (`load_cases`, `merge_config`, `open_run_dir`, `run_cases`) are public too.

## Things to know

- Settings come in layers, later ones winning: built-in defaults, the `config` object inside a dataset, a `--config FILE` (JSON), then flags. An unknown key or a bad value is an error (exit 1), not a silently ignored setting.
- The model is the first set of: `--model`, `model` in the `--config` file, `model` in the dataset's `config`, `$LLM_MODEL`, the framework default (`ollama_chat/qwen3.5:4b`). A dataset that sets `model` therefore beats an exported `LLM_MODEL`.
- An `fsm` path in a case is relative to the dataset file; `output_root` and `output_dir` are relative to the current directory, wherever they are set.
- Exit codes: `0` finished (even with low scores), `1` usage or input error (including a file that is not UTF-8 or not JSON, and a missing examples directory), `2` only when `--fail-under PCT` is given and the score is below it, `130` interrupted. Use `--fail-under` in CI; the comparison is exact, so 57 of 100 passes `--fail-under 57`.
- Ctrl-C stops a run: work not yet started is cancelled, the report files are written for what finished (marked "Interrupted"), and the command exits 130.
- A trial passes only when every expectation it declares holds (and every custom check returns no failures). A case must declare at least one expectation, so an empty `expect` cannot pass by default.
- Once the conversation ends, remaining turns are not sent. That counts as a failure unless the case declares `ended`, so an FSM that ends too early cannot pass a `final_state` check.
- A trial that raises (model down, bad FSM) counts as a failed trial with the error recorded; the run continues.
- Conversation trials run in threads inside one process. A hung model call cannot be stopped early; it ends when litellm's own request timeout fires.
- Example timeouts: a per-example table value wins over the category value, which wins over `--timeout`. To change a tabled example's timeout, set `example_timeouts` in a `--config` file.
- Import it as `from fsm_llm import eval as fsm_eval`, or import names directly (`from fsm_llm.eval import wilson_ci`), so the builtin `eval` is not shadowed.
- A sample dataset is in `evaluation/datasets/simple_greeting_cases.json`. Its three `simple_greeting` cases check plumbing only (their transitions are unconditional, so any model passes); `name_is_extracted`, over `evaluation/datasets/name_capture_fsm.json`, passes only when the model extracts the user's name. Write your cases like the last one: make the pass depend on what the model does.
