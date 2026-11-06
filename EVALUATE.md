# FSM-LLM Evaluation Process

This document defines the evaluation methodology for testing FSM-LLM examples against LLM backends. It provides a repeatable, model-agnostic process for measuring framework quality over time.

---

## Quick Start

```bash
# Automated parallel evaluation (recommended when the server can serve requests in parallel)
fsm-llm-eval examples --model ollama_chat/bonsai-2:27b --workers 4

# One worker: use it when the server has ONE slot (for example OLLAMA_NUM_PARALLEL=1)
fsm-llm-eval examples --workers 1

# Only a specific category
fsm-llm-eval examples --category agents

# Filter by name
fsm-llm-eval examples --filter react

# List discovered examples without running
fsm-llm-eval examples --list

# Fail a CI job (exit 2) when the health score is below 80%
fsm-llm-eval examples --fail-under 80

# Scripted conversation evaluation of any FSM (see "Conversation evaluations")
fsm-llm-eval run evaluation/datasets/simple_greeting_cases.json --trials 3
```

`fsm-llm-eval` is installed with the package (`fsm_llm.eval`, extra `eval`);
`python -m fsm_llm.eval` is the same program. These are the only entry points. Run
`examples` from the repository root, or pass `--examples-dir`, since `examples/` is
resolved from the current directory. Two behaviours differ from the removed
`scripts/eval.py`: an `--output-dir` that already holds files is refused (exit 1)
instead of reused, and usage errors exit 1 instead of argparse's 2 (exit 2 now means
below `--fail-under`).

Output goes to a new `evaluation/<timestamp>_<hash>_<model>/` (`_2`, `_3`, ... appended if that name exists) containing:
- `scorecard.md` -- human-readable results with scores, timing, and category breakdown
- `results.json` -- machine-readable results for scripting and diff
- `logs/<category>/<name>.log` -- full stdout+stderr per example

---

## 1. Evaluation Scope

### What We Test

Every `run.py` under `examples/` is an evaluation target. The suite auto-discovers examples:

```bash
find examples/ -name "run.py" | sort
```

The count will grow over time. The scoring system is ratio-based (percentages), so adding examples doesn't break historical comparisons.

**IMPORTANT**: Do NOT modify existing examples unless explicitly asked by the user. Examples serve as stable evaluation baselines.

### Categories

| Category | Tests | Focus |
|----------|-------|-------|
| `basic/` | Core FSM conversations | Extraction, transitions, response quality |
| `intermediate/` | Multi-state FSMs | Complex transitions, context accumulation |
| `advanced/` | Stacking, handlers, concurrency | FSM stacking, handler hooks, context isolation |
| `classification/` | Intent classification | Classification accuracy, routing correctness |
| `agents/` | Agent patterns | Tool use, iteration control, agent composition |
| `reasoning/` | Structured reasoning | Reasoning engine, validation, FSM stacking |
| `workflows/` | Workflow orchestration | Step execution, context passing, async |
| `meta/` | Meta-builder | Artifact generation, interactive building |

---

## 2. Running an Evaluation

### Automated (recommended)

Use `fsm-llm-eval examples` to run all examples in parallel:

```bash
fsm-llm-eval examples [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `$LLM_MODEL` or `ollama_chat/bonsai-2:27b` | LLM model identifier |
| `--workers` | 4 | Parallel workers, one example subprocess each. More workers only help when the model server runs requests in parallel. On a server with one slot (for example `OLLAMA_NUM_PARALLEL=1`) they queue behind each other, and the per-example timeout counts the wait, so examples time out that finish alone: use `--workers 1`, or raise the server's parallelism |
| `--timeout` | 120 | Default timeout per example (seconds) |
| `--category` | all | Filter by category (basic, agents, etc.) |
| `--filter` | all | Substring match on example name |
| `--output-dir` | auto-generated | Exact output directory; must be new or empty |
| `--examples-dir` | `examples` | Examples tree to scan |
| `--python` | the running interpreter | Interpreter for the example scripts |
| `--config` | none | JSON file of settings (flags override it) |
| `--fail-under` | none | Exit 2 when the health score is below this percentage |
| `--list` | -- | List examples and exit (dry run) |

The runner auto-discovers examples, classifies them as interactive or automated, pipes pre-configured stdin for interactive ones, applies per-category and per-example timeout overrides, and runs them in parallel on a thread pool, one subprocess per example.

**Timeout overrides** (built-in tables in `fsm_llm/eval/constants.py`):
- basic/intermediate: 120s (default)
- agents/workflows/advanced: 180s
- reasoning: 300s
- Known slow examples (e_commerce, reflexion, orchestrator, etc.): 240-360s. The 23 examples whose agents run on the default `AgentConfig.timeout_seconds` (300 s) are at 360 s (budget + 60 s), so the agent's own timeout (an `Error: Agent timed out after 300.0 seconds` line) always comes before a harness kill; `tests/test_fsm_llm_eval/test_timeout_table.py` scans `examples/` and fails when a harness timeout ties or is below an example's own agent budget.
- The timeout clock also runs while a request waits for a busy server, so too many workers for the server's slots show up as timeouts (score 1), not as slowness: see `--workers` above.

Stdin scripts (`EXAMPLE_INPUTS` in the same file) are checked against the examples by `tests/test_fsm_llm_eval/test_example_inputs.py`: an example that never reads stdin has none, the `meta/build_*` scripts end without `quit` (the builder reads to EOF and would take `quit` as a message), and `agents/react_hitl_combined` is sent a task that calls a gated tool so its `y` answers a real approval prompt (a model that skips the gated tool still turns the `y` into a second task).

A per-example value wins over the category value, which wins over `--timeout`, so `--timeout` cannot shorten a tabled example. To change a tabled value, pass a config file, for example `{"example_timeouts": {"agents/debate": 400}, "category_timeouts": {"agents": 240}}`; table entries in the file are merged over the built-in ones. Settings precedence is built-in defaults < `--config FILE` < flags; unknown keys are an error.

**Exit codes**: `0` the run finished (also with a low score), `1` usage error, bad config (including a file that is not UTF-8 or not JSON), unwritable output, a missing examples directory, or no example matched, `2` only when `--fail-under PCT` is given and the health score is below it (compared exactly: 116/200 meets 58), `130` interrupted. Ctrl-C cancels the examples not yet started and still writes `scorecard.md` and `results.json` for the finished ones, marked interrupted (`"interrupted": true`). SIGTERM (what `timeout(1)` and service managers send) is read the same way as Ctrl-C, for `examples` and `run`: the partial report is written and the exit code is 130. Before that change a `timeout`-killed run left a directory holding `logs/` only. The handler lives for the run only, and is not installed off the main thread or over an ignored SIGTERM.

### Prerequisites

```bash
# Ensure the model is available
ollama list  # for Ollama models
# or set OPENAI_API_KEY for cloud models
```

**Which model runs.** The model is `--model`, then the process environment `LLM_MODEL`, then the built-in default `ollama_chat/bonsai-2:27b`. A `.env` file is loaded when `fsm_llm` is imported (litellm does it) and by the `fsm-llm` CLI, and it fills `LLM_MODEL` when the process environment leaves it unset, before the built-in default is consulted, so a stale `.env` silently beats the default. python-dotenv searches upward from the loading module's own directory, not from the current directory: in a checkout with `.venv` inside it, that is the repository root. The scorecard and `results.json` name the model that really ran, so read them. `LITELLM_MODE=PRODUCTION` stops litellm's load; `env -u LLM_MODEL` alone does not.

### Manual (single example)

For debugging or manual scoring of individual examples:

**Interactive examples** (have `input()` calls):
```bash
echo -e "input line 1\ninput line 2\n..." | LLM_MODEL=$LLM_MODEL .venv/bin/python examples/<category>/<name>/run.py 2>&1
```

**Automated examples** (run to completion):
```bash
LLM_MODEL=$LLM_MODEL .venv/bin/python examples/<category>/<name>/run.py 2>&1
```

### What to Observe

For each example, evaluate these dimensions:

1. **Startup** -- Does it initialize without errors?
2. **Core Function** -- Does it perform its primary purpose?
3. **Transitions** -- Do FSM state transitions fire correctly?
4. **Extraction** -- Are context fields extracted from user input?
5. **Tool Use** -- (agents only) Are tools called and do they execute?
6. **Completion** -- Does it finish without crashing or infinite loops?
7. **Output Quality** -- Is the LLM output coherent and on-topic?

### Conversation evaluations (`fsm-llm-eval run`)

The example scores above come from a heuristic over stdout. To test a specific FSM's behaviour, write a dataset of scripted conversations with expected outcomes and run it several times:

```bash
fsm-llm-eval run DATASET [--model M] [--trials N] [--temperature T] [--max-tokens N] [--workers N] [--output-dir D] [--config FILE] [--fail-under PCT] [--list]
```

A dataset is a JSON list of cases, a JSON object `{"config": {...}, "cases": [...]}`, or a `.jsonl` file with one case per line. Each case has an `id`, an `fsm` (path relative to the dataset file, or an inline definition), optional `initial_context`, the user `turns`, and `expect` with at least one of `final_state`, `visited_states`, `context` (exact values), `context_keys` (present and not null), `responses_contain` (case-insensitive substrings) and `ended`. A trial passes when every declared check holds and every turn was sent (an FSM that ends before the last turn fails unless the case declares `ended`); a trial that raises (model down, bad FSM) is a failed trial with the error recorded. The sample `evaluation/datasets/simple_greeting_cases.json` has three plumbing cases over `examples/basic/simple_greeting` (unconditional transitions: any model passes) and `name_is_extracted` over `evaluation/datasets/name_capture_fsm.json`, which passes only when the model extracts the user's name. A useful case makes its pass depend on the model.

Each case runs `trials` times (default 3), in-process through `fsm_llm.API`, one fresh `API` per trial. Settings precedence: defaults < the dataset's `config` < `--config FILE` < flags, so a dataset that sets `model` beats `$LLM_MODEL`. LLM settings: `temperature`, `max_tokens`, and `llm_kwargs` (extra `fsm_llm.API` arguments such as `api_base`; recorded by key only). From Python, `fsm_llm.eval.run_dataset(path, trials=5, checks=[...])` does the same in one call; a check is a function of the finished trial returning failure messages. The run directory (same naming as example runs) holds `rows.jsonl` (one row per trial, written as each finishes), `results.json` (per-case and overall `k`, `n`, `rate` and Wilson 95% `wilson_ci`, plus the config used) and `summary.md`. Report a pass rate with its interval: with 3 trials, 3/3 still has a Wilson lower bound of about 44%. `--fail-under PCT` compares the overall pass rate exactly (57/100 meets 57). Exit codes and Ctrl-C behaviour are the same as for `examples`.

---

## 3. Scoring Rubric

Each example receives a single score from 0-4:

| Score | Label | Criteria |
|-------|-------|----------|
| **4** | **PASS** | Fully functional. All transitions fire, tools work, output is correct and coherent. No errors. |
| **3** | **MOSTLY** | Works with minor issues. Core function succeeds but has cosmetic problems (wrong counter, extra loop iteration, slightly off output). |
| **2** | **PARTIAL** | Partially functional. Some features work but key behaviors fail (transitions don't fire, tools not used, gets stuck). Requires workarounds. |
| **1** | **BROKEN** | Starts but fails at core function. Crashes mid-run, hits iteration limits, produces wrong results, infinite loops. |
| **0** | **CRASH** | Cannot start. Import errors, validation failures, crashes before any useful output. |

**Rules the automatic scorer applies on top of the table** (since 2026-11-06, plan `0068b055`): a run that exits 0 but prints a line starting with `Error:` (or `Error during ...`; indentation allowed, stdout only) or a line starting with `Success: False` is capped at 2 (failure codes `F-CODE` and `F-MODEL`), and an extraction rate of at most 50% (`Extraction rate: 1/2 (50%)`) is a partial failure (exactly 50% used to pass). `Error:` quoted mid-line in prose, `Error Rate:` metric lines, `ValueError: ...` and stderr are not read. **Scores before this change (Runs 001 to 010) were given by the older rule, so they are not comparable with scores after it**: an example that said it failed but exited cleanly could score 4. The heuristic still overstates.

### Distinguishing Scores

- **4 vs 3**: Does everything work, or is there a minor glitch that doesn't affect the main outcome?
- **3 vs 2**: Does the example achieve its purpose (even if imperfectly), or does it fail at its primary goal?
- **2 vs 1**: Do some components work independently, or does the whole thing fall apart?
- **1 vs 0**: Does it at least start and produce some output, or does it crash immediately?

---

## 4. Failure Classification

When an example scores below 4, classify the failure:

| Code | Category | Description |
|------|----------|-------------|
| `F-CODE` | Code Bug | Framework bug, not model-related. Fails with any model. |
| `F-MODEL` | Model Limitation | Small model can't handle the task (bad classification, no tool use). |
| `F-TRANS` | Transition Failure | FSM transitions don't fire despite correct context. |
| `F-EXTRACT` | Extraction Failure | Pass 1 doesn't extract expected fields. |
| `F-TOOL` | Tool Failure | Tool execution errors (calling convention, params). |
| `F-LOOP` | Loop/Budget | Agent exceeds iteration limit or gets stuck in a loop. |
| `F-PARSE` | Parse Failure | Can't parse LLM output (scores, JSON, structured data). |
| `F-SCHEMA` | Schema Error | Pydantic validation, FSM definition errors. |

An example can have multiple failure codes. Record all that apply.

---

## 5. Scorecard Format

After running all examples, record results in this format:

```
# Evaluation Scorecard
# Date: YYYY-MM-DD
# Model: <model identifier>
# Evaluator: <name>
# Example count: <N>

## Scores

| # | Example | Score | Failures | Notes |
|---|---------|-------|----------|-------|
| 1 | basic/simple_greeting | 4 | | Clean pass |
| 2 | basic/form_filling | 2 | F-TRANS, F-EXTRACT | Never reaches confirmation state |
| ... | ... | ... | ... | ... |

## Summary

- Total examples: N
- Score distribution: X at 4, Y at 3, Z at 2, W at 1, V at 0
- Overall score: <sum> / <max possible> = <percentage>%
- Category breakdown:
  - basic: X/Y (Z%)
  - intermediate: ...
  - advanced: ...
  - classification: ...
  - agents: ...
  - reasoning: ...
  - workflows: ...
  - meta: ...
- Top failure codes: F-TOOL (N), F-TRANS (N), F-MODEL (N), ...
```

---

## 6. Aggregate Metrics

### Overall Health Score

```
Health = (sum of all scores) / (4 * number of examples) * 100
```

A perfect run scores 100%. This is the primary metric for tracking progress.

### Category Health

Same formula applied per-category. Identifies which subsystem needs the most work.

### Failure Distribution

Count how many examples have each failure code. This drives prioritization:
- High `F-CODE` count = framework bugs to fix (highest priority)
- High `F-MODEL` count = model compatibility issues (tune prompts or set minimum model size)
- High `F-TOOL` count = tool system bugs (single root cause, high leverage fix)

### Regression Detection

Compare current run's per-example scores against the previous run. Flag:
- **Regressions**: Score decreased (something broke)
- **Improvements**: Score increased (a fix worked)
- **New examples**: First-time evaluation (no comparison)

**Comparability break (2026-09-29 restructure).** Since the extensions moved under
`fsm_llm`, one library-wide `logger.disable("fsm_llm")` silences agents, reasoning,
workflows, monitor and harness logs, and no example enables logging. Before, their
warnings and tracebacks reached stderr, which `fsm_llm.eval` (`scoring.py`) scans for failure signals and
which the per-example logs keep. Runs from before and after that change are not
comparable, for agent examples especially: re-baseline before flagging a regression or
improvement, and expect thinner per-example logs.

---

## 7. Model Compatibility Matrix

Track results across models to understand minimum viable model size:

```
| Example | qwen3.5:4b | qwen3.5:9b | gpt-4o-mini | Notes |
|---------|------------|------------|-------------|-------|
| basic/simple_greeting | 4 | 4 | 4 | Works on all |
| classification/intent_routing | 1 | 3 | 4 | Needs 9B+ for classification |
| ... | ... | ... | ... | ... |
```

---

## 8. Storing Results

All evaluation results live in the `evaluation/` directory. `fsm-llm-eval` generates output automatically.

### Output Structure

Each run creates a timestamped directory (the names below are an illustration of the layout, not directories that exist in the repository):

```
evaluation/
├── 2026-03-29_14-38_0aec60a_qwen3.5-4b/      # Auto-generated by fsm-llm-eval
│   ├── scorecard.md                            # Human-readable results
│   ├── results.json                            # Machine-readable results
│   └── logs/                                   # Per-example logs
│       ├── basic/
│       │   ├── basic_simple_greeting.log
│       │   └── basic_form_filling.log
│       ├── agents/
│       │   ├── agents_debate.log
│       │   └── ...
│       └── ...
├── 2026-03-28:10-15_36aed00_qwen3.5-4b.md     # Legacy manual runs
└── ...
```

### What Gets Generated

`fsm-llm-eval examples` produces three outputs per run:

1. **`scorecard.md`** -- date, git commit, model, scores table (per-example with score/duration/failures), summary (health score, distribution, category breakdown, top failure codes), timing stats. "Total wall time" is the real elapsed time of the run; "Total example time" is the sum of the example durations (sequential equivalent). Scorecards written by `scripts/eval.py` before the `fsm_llm.eval` move labelled that sum "Total wall time", so compare old and new runs on "Total example time"
2. **`results.json`** -- same data in machine-readable format for scripting, diffing, and trend analysis. The keys of the old format are unchanged; newer runs add `wall_time_s`, `workers`, `default_timeout` and `evaluator`
3. **`logs/<category>/<name>.log`** -- full stdout+stderr capture per example, with metadata header (exit code, duration, timeout status). The example child runs with `PYTHONUNBUFFERED=1`, so the log of an example killed by its timeout holds the output it printed before the kill (earlier runs left such a log empty, which made a slow model look like a loop)

### Custom Output Directory

```bash
# Override the auto-generated path (must be new or empty)
fsm-llm-eval examples --output-dir evaluation/my_run
```

A run never overwrites another: when the auto-generated name already exists (two runs in the same minute on the same commit and model), `_2`, `_3`, ... is appended. Before the `fsm_llm.eval` move, such a rerun overwrote the first run's files.

### Comparing Runs

```bash
# Compare JSON results between runs
diff <(jq '.results[] | {name, score}' evaluation/run_a/results.json) \
     <(jq '.results[] | {name, score}' evaluation/run_b/results.json)

# Filter by model
ls evaluation/*bonsai*

# Check a specific example's log
cat evaluation/<run-dir>/logs/agents/agents_debate.log
```

---

## 9. Evaluation Log Index

Quick reference for all evaluation runs. The raw run directories (`scorecard.md`, `results.json`, `logs/`) of Runs 001 to 009 were removed from `evaluation/` on 2026-11-06 together with the other qwen3.5 run directories; the numbers in each entry below are the record. The scorecard text of Run 001, the last of these directories that git tracked, is recoverable from git history with `git show 72388e6^:evaluation/2026-04-01_23-35_4025a78_qwen3.5-4b/scorecard.md` (and `results.json` the same way). Entries that give a path name the directory the run wrote; it no longer exists.

---

### Run 001 -- 2026-04-01 (100 Examples, Honest Scoring Baseline, 75.0%)

- **File**: `evaluation/2026-04-01_23-35_4025a78_qwen3.5-4b/scorecard.md` (directory removed 2026-11-06; recover it with `git show 72388e6^:evaluation/2026-04-01_23-35_4025a78_qwen3.5-4b/scorecard.md`)
- **Model**: `ollama_chat/qwen3.5:4b`
- **Commit**: `4025a78`
- **Examples**: 100 (all with standardized verification output)
- **Health Score**: 75.0% (300/400)
- **Score distribution**: 63x4, 2x3, 7x2, 28x1, 0x0
- **Category breakdown**: workflows 100% (8/8), classification 100% (4/4), reasoning 100% (1/1), agents 95.8% (46/48 pass), meta 80% (2 pass, 3 partial), basic 7% (1/14 pass — only multi_turn_extraction), advanced 12% (1/17 pass — only yoga_instructions), intermediate 44% (adaptive_quiz partial, book_recommendation partial, product_recommendation broken)
- **Top failure codes**: F-EXTRACT (35), F-TRANS (28), F-LOOP (2), F-CODE (1)
- **Root cause**: FSM field extraction on 4B model fails across most basic/intermediate/advanced examples. The 2-pass architecture's Pass 1 (data extraction) does not reliably extract named fields — the model generates good conversational responses but returns no structured extraction data. Agent, workflow, classification, and meta examples work because they use different execution patterns (tool calling, step execution, intent classification) that don't rely on FSM field extraction.
- **Note**: This is the first honest baseline. All prior runs (001-012) used eval scoring that only checked exit codes, allowing examples with 0% extraction to score PASS. Those runs have been invalidated and removed.

---

### Run 002 -- 2026-04-02 (100 Examples, Extraction Pipeline Improvements, 88.0%)

- **File**: `evaluation/2026-04-02_08-28_830692b_qwen3.5-4b/scorecard.md` (directory removed 2026-11-06)
- **Model**: `ollama_chat/qwen3.5:4b`
- **Commit**: `830692b`
- **Examples**: 100
- **Health Score**: 88.0% (352/400) -- **+13.0pp from Run 001**
- **Score distribution**: 82x4, 0x3, 6x2, 12x1, 0x0
- **Category breakdown**: advanced 96% (+84pp), basic 86% (+79pp), agents 86%, intermediate 67% (+23pp), classification 100%, workflows 100%, reasoning 100%, meta 70%
- **Top failure codes**: F-EXTRACT (9), F-LOOP (9), F-TRANS (3)
- **Changes made** (all in `src/`, no example modifications):
  1. Simplified field extraction prompt: replaced verbose XML+CDATA with concise plain text optimized for 4B models
  2. Post-transition extraction: after a state transition, re-extract in the new state from the same user message (skipped for agent FSMs)
  3. Transition condition scanning: extract fields from transition `requires_context_keys`, not just state `required_context_keys`
  4. Bulk extraction fallback: for states with `extraction_instructions` but no explicit field configs
  5. Partial/relative value acceptance: prompts now accept "next Saturday" for dates, "around 7pm" for times
  6. Date context: today's date included in extraction prompts
  7. JSON parsing improvements: markdown fence stripping, `extracted_data` wrapper fallback
  8. Conversation data cache: `get_data()`/`get_current_state()` work after `end_conversation()`
- **Remaining failures**: Agent timeouts (9, model limitation), 2 basic examples without `required_context_keys`, 3 meta builder stdin issues
- **Note**: This is the new official baseline. Agent score varies between runs due to non-deterministic LLM output (agent patterns are sensitive to exact model responses). The framework changes do not cause regressions — agent FSMs are explicitly excluded from post-transition extraction.

---

### Run 003 -- 2026-04-02 (100 Examples, Example Fixes + Eval Input Improvements, 90.2%)

- **File**: `evaluation/iter8/scorecard.md` (directory removed 2026-11-06)
- **Model**: `ollama_chat/qwen3.5:4b`
- **Commit**: `e6cf19d`
- **Examples**: 100
- **Health Score**: 90.2% (361/400) -- **+2.2pp from Run 002, +15.2pp from Run 001**
- **Score distribution**: 84x4, 1x3, 7x2, 8x1, 0x0
- **Category breakdown**: advanced 97%, basic 96% (+10pp), agents 86%, intermediate 75% (+8pp), classification 100%, workflows 100%, reasoning 100%, meta 70%
- **Top failure codes**: F-LOOP (10), F-EXTRACT (7)
- **Changes made**:
  - Added `required_context_keys` to: simple_greeting (mood/intent), adaptive_quiz (4 states), book_recommendation (recommended_book), support_pipeline (3 states)
  - Improved eval inputs: form_filling (one field per message), story_time (actual opinions), adaptive_quiz (player name + feedback), support_pipeline (customer name)
  - Increased agent timeouts: debate 180→300s, evaluator_optimizer 180→300s, orchestrator 240→300s
- **Remaining failures**: Agent timeouts (8, model limitation), 3 meta builder stdin issues, ~5 non-deterministic partial extractions
---

### Run 004 -- 2026-04-02 (100 Examples, Debate Bug Fix, 90.8%)

- **File**: `evaluation/2026-04-02_10-24_7fa6633_qwen3.5-4b/scorecard.md` (directory removed 2026-11-06)
- **Model**: `ollama_chat/qwen3.5:4b`
- **Commit**: `7fa6633`
- **Examples**: 100
- **Health Score**: 90.8% (363/400) -- **+0.6pp from Run 003, +15.8pp from Run 001**
- **Score distribution**: 85x4, 1x3, 6x2, 8x1, 0x0
- **Category breakdown**: advanced 93%, basic 96%, agents 89% (+3pp), intermediate 75%, classification 100%, workflows 100%, reasoning 100%, meta 70%
- **Top failure codes**: F-LOOP (8), F-EXTRACT (7), F-TRANS (1)
- **Changes made**: Fixed bug where bulk extraction fallback overwrote handler-set context values (consensus_reached). This caused the debate agent to loop past max_rounds.
- **Note**: This is the new official baseline.

---

### Run 005 -- 2026-04-02 (100 Examples, Iterative Improvements, 95.8%)

- **File**: `evaluation/2026-04-02_16-11_0b2b7f8_qwen3.5-4b/scorecard.md` (directory removed 2026-11-06)
- **Model**: `ollama_chat/qwen3.5:4b`
- **Commit**: `0b2b7f8`
- **Examples**: 100
- **Health Score**: 95.8% (383/400) -- **+5.0pp from Run 004, +20.8pp from Run 001**
- **Score distribution**: 94x4, 0x3, 1x2, 5x1, 0x0
- **Category breakdown**: advanced 97%, basic 100% (+4pp), agents 94% (+5pp), intermediate 100% (+25pp), classification 81%, workflows 100%, reasoning 100%, meta 100% (+30pp)
- **Top failure codes**: F-LOOP (5), F-EXTRACT (1)
- **Changes made** (all in `src/` and `scripts/`, no example modifications):
  1. Orchestrator/EvalOpt FSM deadlock fix: added fallback unconditional transitions (priority 900) to orchestrator `collect` and evaluator-optimizer `evaluate` states, preventing deadlock when LLM fails to extract transition-triggering fields
  2. Type validation: `FSMManager.start_conversation()` now raises `FSMError` instead of `AttributeError` when `initial_context` is not a dict
  3. Extraction prompt improvement: field extraction now explicitly guides LLMs to check conversation history when value is not in current message
  4. Default extraction retries: changed from 0 to 1, giving each required field one retry pass
  5. Structured output terminal-only: `response_format` (JSON schema enforcement) now only applied on terminal FSM states, preventing small models from hanging on intermediate states
  6. Field name echo rejection: extraction validation now rejects values that match the field name (model confusion artifact)
  7. Eval script fixes: stdin piping for meta builders, timeout overrides for slow examples, improved story_time inputs
- **Remaining failures**: Agent timeouts (5, model contention during parallel eval — pass when run individually), support_pipeline extraction (FSM stacking complexity)
- **Note**: This is the new official baseline. Agent timeout failures are non-deterministic and depend on Ollama load during parallel evaluation. Classification dip (81% vs 100%) is a non-deterministic flap.

### Run 006 -- 2026-05-31 (101 Examples, ReAct under-call + memory-coherence fixes, N=3 median 95.3%)

- **Model**: `ollama_chat/qwen3.5:4b` | **Workers**: 4 | **Commit**: `2df048f` (tag `agent-coherence-2026-05-30`)
- **Examples**: 101
- **Health Score**: **N=3 median 95.3%** (385/404). Runs: 95.3% / 95.3% / 92.3%. **Flat vs Run 005 (95.8%, 100 ex) — no regression.** (Note: CLAUDE.md's "90.8%" figure was stale; the real prior baseline was Run 005's 95.8%.)
- **Category median**: basic/intermediate/classification/reasoning/workflows/meta 100%; advanced 94.1%; **agents 90.6%**.
- **Changes evaluated** (plan_2026-05-30_5598b755, all additive in `src/fsm_llm_agents/` + `scripts/`, no example edits):
  1. **B4 under-call fix** — gate `think→conclude`/`act→conclude` on `should_terminate AND (observation_count>0 OR max_iterations_reached)`; stall-detector sets `max_iterations_reached` (re-merge-immune; tool turns now invoke tools instead of answering mentally).
  2. **Memory coherence** — conclude EXTRACTION prompt answers from task/context+recalled memory (killed "I don't have the previous context" filler); `respond` action for AutoMemory; `recall_min_score=0.25` stops stale-memory bleed.
  3. **Harness** — `scripts/agent_chat_harness.py` (long multi-turn diagnostic).
- **Agent score-1s are non-deterministic timeouts, NOT regressions** — all 7 examples that varied across the 3 runs are in `agents/` and toggle 4↔1 (full-pass ↔ 300s/240s timeout), never a partial/wrong answer: `agent_as_tool [1,1,4]`, `react_hitl_combined [1,4,1]`, `reflexion [4,1,1]`, `classified_dispatch/hitl_approval/multi_tool_recovery/pipeline_review [4,4,1]`. Log inspection confirmed bounded iterations (≤6) and zero `BudgetExhaustedError` — `--workers 4` contention, pass solo at `--workers 1`.
- **Note**: This re-baseline supersedes the stale 90.8% figure. Validate behavior fixes with the harness + manual log read (the heuristic overstates ~15pp and flags correct tool-free answers as `success=False`).

---

### Run 007 -- 2026-10-01 (101 Examples, core step driver A/B against `d4b1626` on the same day, 96.0% vs 96.8%)

- **Files** (directories removed 2026-11-06): `evaluation/2026-10-01_09-23_0f0789c_qwen3.5-4b/scorecard.md` (HEAD) and `evaluation/2026-10-01_09-54_d4b1626_qwen3.5-4b_same-day-baseline/scorecard.md` (baseline, run from an isolated `d4b1626` worktree with `PYTHONPATH=<worktree>/src`; `results.json` records `git_commit` `d4b1626`)
- **Model**: `ollama_chat/qwen3.5:4b` (digest `2a654d98e6fb`) | **Workers**: 4 | **Timeouts**: default 120 s plus the built-in overrides (eval code and `examples/` unchanged since `d4b1626`) | **N**: 1 per commit, run one after the other
- **Commits**: `0f0789c` (plan `07ad3f8c` iteration 1: agent loops on core `advance`/`run_until_terminal`, no synthetic "Continue." turn) vs `d4b1626`
- **Examples**: 101
- **Health Score**: `0f0789c` 96.0% (388/404) vs `d4b1626` 96.8% (391/404), -3 points
- **Score distribution**: `0f0789c` 95x4, 0x3, 2x2, 4x1, 0x0; `d4b1626` 96x4, 0x3, 2x2, 3x1, 0x0
- **Category breakdown**: identical at both commits except agents (180/192 vs 183/192): advanced 64/68, basic 56/56, classification 20/20, intermediate 12/12, meta 20/20, reasoning 4/4, workflows 32/32
- **Top failure codes**: `0f0789c` F-LOOP (4, all timeouts), F-EXTRACT (2); `d4b1626` F-LOOP (3, all timeouts), F-EXTRACT (2); F-CODE 0 at both
- **Only changed score**: `agents/plan_execute_recovery` 4 -> 1 (timeout at 180 s; 136.6 s at `d4b1626`). Not model noise: run alone 3 times per commit, `0f0789c` builds a 7-step plan (15 iterations, 24 LLM calls) every time, `d4b1626` a 4-step plan (9 iterations, 16 calls); both succeed alone.
- **Raw logs**: 0 envelope leaks (`"field_name"`, `"extracted_data"`), 0 `Continue.`, 0 tracebacks at both commits. Agents `Success: True`/`False` lines: 22/3 vs 22/4. Every `Success: False` example still scores 4.
- **Note**: the agents category alone (180/192) is 2 points above the recorded G3 baseline at `d4b1626` (178/192, 2026-09-29). Bench block `agents-react/B1` from the same day is in `docs/agents_roadmap.md`.

### Run 008 -- 2026-10-01 (48 agents Examples, after the PlanExecute planner fix, 92.2%)

- **File**: `evaluation/2026-10-01_10-50_e9ef064_qwen3.5-4b/scorecard.md` (directory removed 2026-11-06)
- **Model**: `ollama_chat/qwen3.5:4b` (digest `2a654d98e6fb`) | **Workers**: 4 | **Timeouts**: default 120 s plus the built-in overrides (same settings as Run 007) | **N**: 1 | **Category**: agents only (`--category agents`), started 07:50 UTC, nothing else on Ollama
- **Commit**: `e9ef064` (plan `07ad3f8c` step 24.1: the PlanExecute `plan_steps`/replan field adds no tool-less step unless the task asks for one and no confirm/wait step, D-055)
- **Health Score**: agents 177/192 = 92.2% (Run 007 same day: `0f0789c` 180/192, `d4b1626` 183/192; G3 `d4b1626` 2026-09-29: 178/192)
- **Score distribution**: 43x4, 0x3, 0x2, 5x1, 0x0; wall time 1,363 s
- **Top failure codes**: F-LOOP 5 (all timeouts); F-CODE 0
- **Changed scores**: `agents/hierarchical_orchestrator` 4 -> 1 (timeout at 300 s; 114.4 s at `0f0789c`, 164.1 s at `d4b1626`; OrchestratorAgent, code unchanged by the fix, also a timeout in the G3 run at `d4b1626`: load, not this change). `agents/plan_execute_recovery` stays 1 (timeout at 180 s; 4 at `d4b1626`): run alone it now takes 20 LLM calls and 28 s (`0f0789c` 24 calls, `d4b1626` 16), still over the timeout under 4 workers. `plan_execute`, `orchestrator_specialist`, `supply_chain_optimizer` time out at all three points.
- **Raw logs**: 0 envelope leaks, 0 `Continue.`, 0 tracebacks; `Success: True`/`False` lines 21/3 (the same three `Success: False` examples as `0f0789c`, all scored 4); the 5 timeouts print nothing.

### Run 009 -- 2026-10-01 (101 Examples, one LLM layer A/B against `e1f63a9` on the same day, 95.5% vs 90.8%)

- **Files** (directories removed 2026-11-06): `evaluation/2026-10-01_22-19_7c823be_qwen3.5-4b/scorecard.md` (HEAD) and `evaluation/2026-10-01_21-32_e1f63a9_qwen3.5-4b_same-day-baseline/scorecard.md` (baseline); both run from isolated worktrees with `PYTHONPATH=<worktree>/src`, one after the other (21:32 and 22:19 UTC)
- **Model**: `ollama_chat/qwen3.5:4b` (digest `2a654d98e6fb`) | **Workers**: 4 | **Timeouts**: default 120 s plus the built-in overrides | **N**: 1 per commit | **Load**: Ollama was shared with an external `qwen3.5:9b-q8_0` server under load during both runs, so timeouts are higher than in Run 007
- **Commits**: `7c823be` (plan `944e2692`: one LLM layer, core completion state, native_fc and the meta-builder as FSMs on core, reasoning engine on core's run loop; the last reasoning fix `6496430` landed during the run and its examples were re-run, below) vs `e1f63a9`
- **Health Score**: `7c823be` 95.5% (386/404) vs `e1f63a9` 90.8% (367/404), +19 points
- **Top failure codes**: `7c823be` F-LOOP 4 (timeouts), F-EXTRACT 3; `e1f63a9` F-LOOP 11 (timeouts), F-EXTRACT 2; F-CODE 0 at both
- **Changed scores**: up at HEAD (baseline timeouts): `advanced/vendor_evaluation`, `agents/agent_memory_chain`, `agents/hierarchical_tools`, `agents/pipeline_review`, `agents/plan_execute_recovery`, `agents/react_hitl_combined`, `agents/react_structured_pipeline` 1 -> 4, `agents/concurrent_react` 1 -> 2. Down: `agents/reasoning_stacking` 4 -> 1 (timeout at 180 s; 168.9 s at the baseline). Run alone twice per commit it scores 4 at both (`e1f63a9` 45.4 s, 45.5 s; `6496430` 45.2 s, 45.6 s): load, not code
- **Rebuilt paths**: the 5 meta examples, `reasoning/math_tutor` and `agents/reasoning_tool` score 4 at HEAD; at `6496430` `reasoning/math_tutor` 4 (8.1 s) and `agents/reasoning_tool` 4 (11.8 s) run alone
- **Raw logs**: 0 envelope leaks (`"field_name"`, `"extracted_data"`), 0 `Continue.`, 0 tracebacks at HEAD; agents `Success: True`/`False` lines 22/4
- **Rule** (pre-registered, plan `944e2692` D-016): total within 8 points of the baseline, no code-caused 4 -> 1 drop, 0 F-CODE, rebuilt examples within 1 point, 0 leaks: met (the `reasoning_stacking` clause through the solo re-runs, D-063)

### Run 010 -- 2026-11-06 (plan `0068b055`, first live evidence on `ollama_chat/bonsai-2:27b`; the agents category was completed in a later session)

- **Model**: `ollama_chat/bonsai-2:27b` (Q2_0, 26.9B, one GPU, context 32768) with the embedder `ollama/embeddinggemma-2:740m` | **Commit**: `8da28be` (the F-AR-05 comparison arms are `8a62490` and `8da28be`) | **N**: 1 per category, **3** per arm in the F-AR-05 comparison | **Workers**: 2 (4 for `fsm-llm-eval run`) | **Timeouts**: default 120 s plus the built-in overrides
- **Not comparable with Runs 001 to 009.** Every earlier run was on `ollama_chat/qwen3.5:4b`. The default model changed in this plan (D-014), the model is larger and slower per call (about 10 to 35 s a call here, with 2 workers sharing one GPU), and the 4b figures (Run 006 and Run 007) are not a baseline for it. Nothing below was compared with them. The 0-4 score is a heuristic that overstates (see the Run 006 note), and the sample is one run per category.
- **Run environment**: the repository `.env` (untracked) still named `ollama_chat/qwen3.5:4b` and, through litellm, beat the built-in default, so every run used `LITELLM_MODE=PRODUCTION` with `LLM_MODEL`, `LLM_TEMPERATURE` and `LLM_MAX_TOKENS` unset to make the default apply. See the note under "Prerequisites".
- **Smoke**: a greeting FSM with no model argument ran on `bonsai-2:27b` (6 calls, 0 errors); `SemanticMemoryStore` with the real embedder returned 768-dimensional vectors and ranked all three probes correctly (relevant scores 0.73 to 0.84 against 0.52 to 0.59); a ReAct agent called a function tool and a keyword-only `dict` tool in one batch and answered correctly.
- **`fsm-llm-eval run`** on `evaluation/datasets/simple_greeting_cases.json`, `--trials 3`: 12/12 = 100.0% (Wilson 95% CI [75.7%, 100.0%]), 4 cases of 3 trials, `name_is_extracted` 3/3.
- **`fsm-llm-eval examples`, health per category** (score sum over 4 per example):

| Category | Examples | Health | Distribution |
|---|---|---|---|
| basic | 14 | 53/56 = 94.6% | 13 x 4, 1 x 1 (`story_time`, timeout at 120 s) |
| classification | 5 | 20/20 = 100.0% | 5 x 4 |
| intermediate | 3 | 12/12 = 100.0% | 3 x 4 |
| advanced | 17 | 68/68 = 100.0% | 17 x 4 |
| workflows | 8 | 32/32 = 100.0% | 8 x 4 |
| reasoning | 1 | 4/4 = 100.0% | 1 x 4 |
| meta | 5 | 20/20 = 100.0% | 5 x 4 |
| agents | 48 | 141/192 = 73.4% as the harness scored them at 2 workers (first session: 109/124 for 31 examples, parsed from stdout; second session: the other 17); 183/192 = 95.3% with the 14 timeouts re-run alone (old scorer, see below) | 31 x 4 and 17 x 1 at 2 workers |

- **Agents category** (two sessions, scores below are from the scorer that existed at the time, before the 2026-11-06 rubric change in section 3). First session: the run was cut at 2100 s (SIGTERM from `timeout`, exit 124) before it wrote a scorecard, so 17 of 48 examples were not run and 109/124 = 87.9% was a stdout parse of the 31 that finished (26 x 4, 5 x 1; the five scored 1 were `legal_document_review`, `orchestrator_specialist`, `pipeline_review`, `plan_execute` and `plan_execute_recovery`). Second session (HEAD `6368411` for the first batches; a docs-only commit landed during it, the code under test did not change): all 48 examples have now run. The 16 of the 17 not-run examples that went through the batch (`--workers 2`, table timeouts) scored 5 x 4 and 11 x 1 (timeouts at 180 to 300 s); the 17th, `supply_chain_optimizer`, scored 1 in every run of the first session. Counting the 31 earlier ones, the harness scored the category 31 x 4 and 17 x 1 = 141/192 = 73.4%.
- **The timeouts were queueing, not the examples.** The server ran with `OLLAMA_NUM_PARALLEL:1` on one GPU, so two workers interleave: every request of one waits for the other's current generation, and the example timeout counts the wait. At 2 workers 14 examples timed out (the 11 of the batch plus `legal_document_review`, `pipeline_review` and `supply_chain_optimizer`) and every one of them finished when run alone (`--workers 1`, a 600 or 900 s limit set through `--config` because `--timeout` cannot lift a tabled value): 11 s to 307 s, steady progress, no loop. The server log shows the clients killed at the harness timeout while queued (HTTP 499 after 3m50s, 4m20s and 5m0s) and 8 `Load failed ... client connection closed` lines. The host GPU was also used by another project during the run, so outside contention cannot be excluded. With the 14 re-run alone the category reads 45 x 4 and 3 x 1 (`orchestrator_specialist`, `plan_execute`, `plan_execute_recovery`, not re-run in this session; the first two were shown slow, not looping, in the first one) = 183/192 = 95.3% by the old scorer. `supply_chain_optimizer` run alone took 306.6 s: the agent's own 300 s budget fired (`Error: Agent timed out after 300.0 seconds`), the run exited cleanly and the old scorer gave it 4. `basic/story_time` alone with `--timeout 600` took 100.6 s (score 4); it needs more than the 120 s default.
- **What the old scorer let through, and the rescoring.** Several of those 4s are failed runs: `Error: Agent timed out` (`supply_chain_optimizer`), `Success: False` with no tool run (`rewoo_multi_step`), `Success: False` with a quality score of 0.55 (`regulatory_compliance`), `Success: False` after three reflections (`reflexion`). The scorer was changed for this (CHANGELOG, step 12.19). The step that changed it rescored 116 logs of that session offline and found nine that change, each from 4 to 2: `classification/multi_intent` and `advanced/multi_level_stack` (an extraction rate of exactly 50%), `agents/architecture_review` and `agents/eval_opt_structured` (`Success: False` and 2 of 4), `agents/reflexion`, `agents/regulatory_compliance` and `agents/rewoo_multi_step` (`Success: False`), `agents/skill_loader` (50% extraction) and `agents/supply_chain_optimizer` (`Error: Agent timed out`); the other 107 were identical and no legitimately successful run dropped. A later rescoring of the logs still on disk (127, including the F-AR-05 before-arm directories) also flags `agents/medical_literature` (one before-arm log). No health figure for the agents category under the fixed scorer is claimed, and the table above is not rescored: **the per-category numbers above are not comparable with scores from the fixed scorer.**
- **Other live findings of that session, all handled in this plan** (see the CHANGELOG Unreleased entries): a reply cut by `max_tokens` was shipped as the answer (`legal_document_review`, `security_audit`; 4 of the `regulatory_compliance` calls, 7 of 12 in `pipeline_review`) and `rewoo_multi_step` lost its plan to a cut `plan_blueprint` field (fixed as far as detection and a named cause, step 12.23; the carry into the agent result is a Known open row); litellm's provider-error banner printed to stdout in 9 of the 16 batch logs and none of the 15 solo ones (suppressed at import, step 12.20); `react_hitl_combined` was sent a task that asked for no approval, so its `y` ran as a second task (step 12.22); 23 examples had a harness timeout equal to the agent's own 300 s budget (now 360 s, step 12.21). The other 27 examples that run on the default 300 s agent budget have a harness timeout below it (120 to 240 s) and keep it: there the harness is the hang detector and fires first.
- **F-AR-05-LIVE** (the silent Orchestrator and ADaPT intermediate states, commit `c73dab7`): the 7 agents examples that use the pattern (`adapt`, `adapt_with_memory`, `medical_literature`, `orchestrator`, `orchestrator_specialist`, `hierarchical_orchestrator`, `supply_chain_optimizer`), same model, same day, runs interleaved, N=3 per arm, before arm `8a62490` (the parent of `c73dab7`) in a scratch worktree, after arm `8da28be`. Health over the 7: before 67.9 / 67.9 / 67.9 (median 67.9, spread 0.0), after 78.6 / 67.9 / 78.6 (median 78.6, spread 10.7). The whole difference is `hierarchical_orchestrator` (timeout at 300 s in all three before runs, finished in 204 s and 218 s in two of three after runs). `orchestrator_specialist` and `supply_chain_optimizer` time out in both arms, so their quality is unmeasured. Wall time of the examples that finish fell by about half (`orchestrator` 93 s to 38 s, `adapt` 65 s to 32 s, medians). Verdict by the rule of plan decision D-017: do not revert `c73dab7`. Caveat: the scorer saturates (6 of 7 examples score 4 in both arms), so this shows the change did not lower the score, and it does not show that the answers are as good; the after arm also holds the later 12.x fixes, none of which touches these patterns.

---

_New evaluation runs should be appended above this line._
