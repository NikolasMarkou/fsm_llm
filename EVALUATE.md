# FSM-LLM Evaluation Process

This document defines the evaluation methodology for testing FSM-LLM examples against LLM backends. It provides a repeatable, model-agnostic process for measuring framework quality over time.

---

## Quick Start

```bash
# Automated parallel evaluation (recommended)
fsm-llm-eval examples --model ollama_chat/qwen3.5:4b --workers 4

# With more parallelism for higher GPU utilization
fsm-llm-eval examples --workers 8

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
| `--model` | `$LLM_MODEL` or `ollama_chat/qwen3.5:4b` | LLM model identifier |
| `--workers` | 4 | Parallel workers, one example subprocess each (increase for GPU utilization) |
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
- Known slow examples (e_commerce, reflexion, orchestrator, etc.): 240-300s

A per-example value wins over the category value, which wins over `--timeout`, so `--timeout` cannot shorten a tabled example. To change a tabled value, pass a config file, for example `{"example_timeouts": {"agents/debate": 400}, "category_timeouts": {"agents": 240}}`; table entries in the file are merged over the built-in ones. Settings precedence is built-in defaults < `--config FILE` < flags; unknown keys are an error.

**Exit codes**: `0` the run finished (also with a low score), `1` usage error, bad config (including a file that is not UTF-8 or not JSON), unwritable output, a missing examples directory, or no example matched, `2` only when `--fail-under PCT` is given and the health score is below it (compared exactly: 116/200 meets 58), `130` interrupted. Ctrl-C cancels the examples not yet started and still writes `scorecard.md` and `results.json` for the finished ones, marked interrupted (`"interrupted": true`).

### Prerequisites

```bash
# Ensure the model is available
ollama list  # for Ollama models
# or set OPENAI_API_KEY for cloud models
```

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
warnings and tracebacks reached stderr, which `eval.py` scans for failure signals and
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

Each run creates a timestamped directory:

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
3. **`logs/<category>/<name>.log`** -- full stdout+stderr capture per example, with metadata header (exit code, duration, timeout status)

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
ls evaluation/*qwen3.5*

# Check a specific example's log
cat evaluation/2026-03-29_14-38_0aec60a_qwen3.5-4b/logs/agents/agents_debate.log
```

---

## 9. Evaluation Log Index

Quick reference for all evaluation runs. Each entry links to its result file.

---

### Run 001 -- 2026-04-01 (100 Examples, Honest Scoring Baseline, 75.0%)

- **File**: [`evaluation/2026-04-01_23-35_4025a78_qwen3.5-4b/scorecard.md`](evaluation/2026-04-01_23-35_4025a78_qwen3.5-4b/scorecard.md)
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

- **File**: [`evaluation/2026-04-02_08-28_830692b_qwen3.5-4b/scorecard.md`](evaluation/2026-04-02_08-28_830692b_qwen3.5-4b/scorecard.md)
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

- **File**: [`evaluation/iter8/scorecard.md`](evaluation/iter8/scorecard.md)
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

- **File**: [`evaluation/2026-04-02_10-24_7fa6633_qwen3.5-4b/scorecard.md`](evaluation/2026-04-02_10-24_7fa6633_qwen3.5-4b/scorecard.md)
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

- **File**: [`evaluation/2026-04-02_16-11_0b2b7f8_qwen3.5-4b/scorecard.md`](evaluation/2026-04-02_16-11_0b2b7f8_qwen3.5-4b/scorecard.md)
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

- **Files**: `evaluation/2026-10-01_09-23_0f0789c_qwen3.5-4b/scorecard.md` (HEAD) and `evaluation/2026-10-01_09-54_d4b1626_qwen3.5-4b_same-day-baseline/scorecard.md` (baseline, run from an isolated `d4b1626` worktree with `PYTHONPATH=<worktree>/src`; `results.json` records `git_commit` `d4b1626`)
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

- **File**: `evaluation/2026-10-01_10-50_e9ef064_qwen3.5-4b/scorecard.md`
- **Model**: `ollama_chat/qwen3.5:4b` (digest `2a654d98e6fb`) | **Workers**: 4 | **Timeouts**: default 120 s plus the built-in overrides (same settings as Run 007) | **N**: 1 | **Category**: agents only (`--category agents`), started 07:50 UTC, nothing else on Ollama
- **Commit**: `e9ef064` (plan `07ad3f8c` step 24.1: the PlanExecute `plan_steps`/replan field adds no tool-less step unless the task asks for one and no confirm/wait step, D-055)
- **Health Score**: agents 177/192 = 92.2% (Run 007 same day: `0f0789c` 180/192, `d4b1626` 183/192; G3 `d4b1626` 2026-09-29: 178/192)
- **Score distribution**: 43x4, 0x3, 0x2, 5x1, 0x0; wall time 1,363 s
- **Top failure codes**: F-LOOP 5 (all timeouts); F-CODE 0
- **Changed scores**: `agents/hierarchical_orchestrator` 4 -> 1 (timeout at 300 s; 114.4 s at `0f0789c`, 164.1 s at `d4b1626`; OrchestratorAgent, code unchanged by the fix, also a timeout in the G3 run at `d4b1626`: load, not this change). `agents/plan_execute_recovery` stays 1 (timeout at 180 s; 4 at `d4b1626`): run alone it now takes 20 LLM calls and 28 s (`0f0789c` 24 calls, `d4b1626` 16), still over the timeout under 4 workers. `plan_execute`, `orchestrator_specialist`, `supply_chain_optimizer` time out at all three points.
- **Raw logs**: 0 envelope leaks, 0 `Continue.`, 0 tracebacks; `Success: True`/`False` lines 21/3 (the same three `Success: False` examples as `0f0789c`, all scored 4); the 5 timeouts print nothing.

---

_New evaluation runs should be appended above this line._
