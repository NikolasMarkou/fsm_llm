# FSM-LLM Evaluation Process

This document defines the evaluation methodology for testing FSM-LLM examples against LLM backends. It provides a repeatable, model-agnostic process for measuring framework quality over time.

---

## Quick Start

```bash
# Automated parallel evaluation (recommended)
.venv/bin/python scripts/eval.py --model ollama_chat/qwen3.5:4b --workers 4

# With more parallelism for higher GPU utilization
.venv/bin/python scripts/eval.py --workers 8

# Only a specific category
.venv/bin/python scripts/eval.py --category agents

# Filter by name
.venv/bin/python scripts/eval.py --filter react

# List discovered examples without running
.venv/bin/python scripts/eval.py --list
```

Output goes to `evaluation/<timestamp>_<hash>_<model>/` containing:
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

Use `scripts/eval.py` to run all examples in parallel:

```bash
.venv/bin/python scripts/eval.py [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `$LLM_MODEL` or `ollama_chat/qwen3.5:4b` | LLM model identifier |
| `--workers` | 4 | Parallel worker processes (increase for GPU utilization) |
| `--timeout` | 120 | Default timeout per example (seconds) |
| `--category` | all | Filter by category (basic, agents, etc.) |
| `--filter` | all | Substring match on example name |
| `--output-dir` | auto-generated | Override output directory |
| `--list` | — | List examples and exit (dry run) |

The script auto-discovers examples, classifies them as interactive or automated, pipes pre-configured stdin for interactive ones, applies per-category and per-example timeout overrides, and runs them in parallel via `ProcessPoolExecutor`.

**Timeout overrides** (built into the script):
- basic/intermediate: 120s (default)
- agents/workflows/advanced: 180s
- reasoning: 300s
- Known slow examples (e_commerce, reflexion, orchestrator, etc.): 240-300s

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

All evaluation results live in the `evaluation/` directory. `scripts/eval.py` generates output automatically.

### Output Structure

Each run creates a timestamped directory:

```
evaluation/
├── 2026-03-29_14-38_0aec60a_qwen3.5-4b/      # Auto-generated by scripts/eval.py
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

`scripts/eval.py` produces three outputs per run:

1. **`scorecard.md`** -- date, git commit, model, scores table (per-example with score/duration/failures), summary (health score, distribution, category breakdown, top failure codes), timing stats
2. **`results.json`** -- same data in machine-readable format for scripting, diffing, and trend analysis
3. **`logs/<category>/<name>.log`** -- full stdout+stderr capture per example, with metadata header (exit code, duration, timeout status)

### Custom Output Directory

```bash
# Override the auto-generated path
.venv/bin/python scripts/eval.py --output-dir evaluation/my_run
```

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

### Run 010 -- 2026-03-30 (80 Examples: +10 High-Complexity, 98.8%)

- **File**: [`evaluation/2026-03-29_23-40_416c3b8_qwen3.5-4b/scorecard.md`](evaluation/2026-03-29_23-40_416c3b8_qwen3.5-4b/scorecard.md)
- **Model**: `ollama_chat/qwen3.5:4b`
- **Commit**: `416c3b8` + 10 new examples
- **Examples**: 80 (+10 new high-complexity: 7 agents, 3 workflows)
- **Health Score**: 98.8% (316/320)
- **Score distribution**: 78x4, 1x3, 0x2, 1x1, 0x0
- **Category breakdown**: advanced 100%, basic 100%, classification 100%, intermediate 100%, reasoning 100%, meta 100%, workflows 100%, agents 97.9%
- **Top failure codes**: F-LOOP (2) — non-deterministic model timeouts on existing examples
- **New examples (all PASS)**: legal_document_review (ReactAgent, 4 tools, 2.5k task), investment_portfolio (PlanExecute, 5 tools, 2k task), security_audit (ReactAgent, 5 tools, 2.5k task), medical_literature (ADaPT, 3 tools, 2k task), architecture_review (EvalOpt+structured, 2k task), supply_chain_optimizer (Orchestrator, 6 tools, 2.5k task), regulatory_compliance (MakerChecker, 2k task), loan_processing (6-step workflow), release_management (7-step workflow), customer_onboarding (5-step workflow+agent)
- **Changes**: Added 10 complex examples with 2k-3k char tasks and multi-step pipelines. Fixed ResponseGenerationResponse min_length validation for Pass 2 skip sentinel.

---

### Run 009 -- 2026-03-29 (100% Pass Rate: Skip Pass 2 Optimization)

- **File**: [`evaluation/2026-03-29_21-43_7d54bad_qwen3.5-4b/scorecard.md`](evaluation/2026-03-29_21-43_7d54bad_qwen3.5-4b/scorecard.md)
- **Model**: `ollama_chat/qwen3.5:4b`
- **Commit**: `7d54bad` + Pass 2 skip optimization
- **Examples**: 70
- **Health Score**: 100.0% (280/280)
- **Score distribution**: 70x4, 0x3, 0x2, 0x1, 0x0
- **Category breakdown**: All categories 100%
- **Wall time**: 645s (down from 1343s in Run 008 — 52% faster)
- **Top failure codes**: None
- **Changes from Run 008**: Skip Pass 2 (response generation) for intermediate agent states (`response_instructions=""`). This halves LLM calls for agent iterations, eliminating all F-LOOP timeouts. Also: parse quality_score from checker_feedback dict (fixes maker_checker F-EXTRACT), reduce stall detection threshold 3→2.
- **Files changed**: `pipeline.py`, `llm.py`, `fsm_definitions.py`, `handlers.py`, `maker_checker.py`

---

### Run 008 -- 2026-03-29 (Final Baseline: 70 Examples, 95.7%)

- **File**: [`evaluation/2026-03-29_19-27_4eb7e3c_qwen3.5-4b/scorecard.md`](evaluation/2026-03-29_19-27_4eb7e3c_qwen3.5-4b/scorecard.md)
- **Model**: `ollama_chat/qwen3.5:4b`
- **Commit**: `4eb7e3c`
- **Examples**: 70
- **Health Score**: 95.7% (268/280)
- **Score distribution**: 66x4, 0x3, 0x2, 4x1, 0x0
- **Category breakdown**: advanced 100%, basic 100%, classification 100%, intermediate 100%, reasoning 100%, workflows 100%, meta 100%, agents 93%
- **Top failure codes**: F-LOOP (4) — all are model-dependent timeout issues with 4B model
- **Changes from Run 007**: Fixed eval_opt_structured (timeout + reduced iterations/refinements), agent_memory_chain (simplified prompts for 4B), removed N/A output strings that triggered F-EXTRACT false positives in maker_checker_code, pipeline_review, meta_review_loop. Eliminated all F-EXTRACT and F-CODE failures.

---

### Run 007 -- 2026-03-29 (70 Examples Added)

- **File**: [`evaluation/2026-03-29_18-47_4eb7e3c_qwen3.5-4b/scorecard.md`](evaluation/2026-03-29_18-47_4eb7e3c_qwen3.5-4b/scorecard.md)
- **Model**: `ollama_chat/qwen3.5:4b`
- **Commit**: `4eb7e3c`
- **Examples**: 70 (+20 new complex examples)
- **Health Score**: 92.1% (258/280)
- **Score distribution**: 60x4, 4x3, 0x2, 6x1, 0x0
- **Top failure codes**: F-LOOP (6), F-EXTRACT (4)
- **Changes**: Added 20 new complex examples (14 agents, 4 meta, 2 workflows)

---

### Run 006 -- 2026-03-28

- **File**: [`evaluation/2026-03-28:20-05_7928630_qwen3.5-4b.md`](evaluation/2026-03-28:20-05_7928630_qwen3.5-4b.md)
- **Model**: `ollama_chat/qwen3.5:4b`
- **Commit**: `7928630`
- **Examples**: 50
- **Health Score**: 61.5%
- **Top failures**: F-MODEL (14), F-TRANS (9), F-LOOP (7), F-EXTRACT (6), F-TOOL (3)
- **Changes**: ADaPT iteration limit enforced, entity type coercion, tool_input unwrap, legacy dict-param dispatch

---

### Run 005 -- 2026-03-28

- **File**: [`evaluation/2026-03-28:18-30_6e22fc2_qwen3.5-4b.md`](evaluation/2026-03-28:18-30_6e22fc2_qwen3.5-4b.md)
- **Model**: `ollama_chat/qwen3.5:4b`
- **Commit**: `6e22fc2`
- **Examples**: 50
- **Health Score**: 59.0%
- **Top failures**: F-MODEL (14), F-TRANS (9), F-LOOP (8), F-EXTRACT (6), F-TOOL (5)

---

### Run 001 -- 2026-03-28

- **File**: [`evaluation/2026-03-28:10-00_36aed00_qwen3.5-4b.md`](evaluation/2026-03-28:10-00_36aed00_qwen3.5-4b.md)
- **Model**: `ollama_chat/qwen3.5:4b`
- **Commit**: `36aed00`
- **Examples**: 33
- **Health Score**: 55.3%
- **Top failures**: F-CODE (6), F-TOOL (5), F-MODEL (5), F-TRANS (4), F-LOOP (4)

---

_New evaluation runs should be appended above this line._
