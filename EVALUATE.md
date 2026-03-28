# FSM-LLM Evaluation Process

This document defines the evaluation methodology for testing FSM-LLM examples against LLM backends. It provides a repeatable, model-agnostic process for measuring framework quality over time.

---

## Quick Start

```bash
# 1. Pick a model
export LLM_MODEL=ollama_chat/qwen3.5:4b

# 2. Run examples one by one, record results in the scorecard
# 3. Calculate scores using the rubric below
# 4. Save the scorecard to evaluation/ (see Section 8)
```

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

### Prerequisites

```bash
# Activate virtual environment
source .venv/bin/activate

# Ensure the model is available
ollama list  # for Ollama models
# or set OPENAI_API_KEY for cloud models

# Set the model
export LLM_MODEL=ollama_chat/qwen3.5:4b
```

### Per-Example Procedure

For each example, run it interactively and observe behavior:

**Interactive examples** (have `input()` calls):
```bash
echo -e "input line 1\ninput line 2\n..." | LLM_MODEL=$LLM_MODEL .venv/bin/python examples/<category>/<name>/run.py 2>&1
```

**Automated examples** (run to completion):
```bash
LLM_MODEL=$LLM_MODEL .venv/bin/python examples/<category>/<name>/run.py 2>&1
```

**Timeouts**: Use 120s for basic/intermediate, 300s for agent/reasoning/workflow examples.

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

All evaluation results live in the `evaluation/` directory. Each run produces a timestamped markdown file that captures the exact state of the codebase.

### File Naming

```
evaluation/<YYYY-MM-DD:HH-mm>_<short-hash>_<model-slug>.md
```

Examples:
```
evaluation/2026-03-28:10-15_36aed00_qwen3.5-4b.md
evaluation/2026-04-02:14-30_a1b2c3d_gpt-4o-mini.md
```

### Generating a Result File

```bash
# Get the values
DATETIME=$(date +%Y-%m-%d:%H-%M)
HASH=$(git rev-parse --short HEAD)
MODEL_SLUG=$(echo "$LLM_MODEL" | sed 's|.*/||; s/:/-/g')

# Create the results file
mkdir -p evaluation
RESULT_FILE="evaluation/${DATETIME}_${HASH}_${MODEL_SLUG}.md"

cat > "$RESULT_FILE" <<EOF
# Evaluation: ${DATETIME}

- **Date**: ${DATETIME}
- **Git commit**: $(git rev-parse HEAD)
- **Git short hash**: ${HASH}
- **Branch**: $(git branch --show-current)
- **Model**: ${LLM_MODEL}
- **Example count**: $(find examples/ -name "run.py" | wc -l)
- **Evaluator**: <name>

## Scores

| # | Example | Score | Failures | Notes |
|---|---------|-------|----------|-------|

## Summary

- Total examples:
- Score distribution:
- **Health Score: / = %**
- Category breakdown:
- Top failure codes:

## Changes Since Last Run

<list code changes, fixes applied, new examples added>
EOF

echo "Created: $RESULT_FILE"
```

### What Goes in Each File

Every result file must contain:

1. **Header** -- date, full git commit hash, short hash, branch, model, example count
2. **Scores table** -- every example with score, failure codes, and notes
3. **Summary** -- health score, distribution, category breakdown, top failures
4. **Changes since last run** -- what was fixed/added since the previous evaluation

### Directory Structure

```
evaluation/
├── 2026-03-28:10-15_36aed00_qwen3.5-4b.md    # Run 001
├── 2026-04-02:14-30_a1b2c3d_qwen3.5-4b.md    # Run 002 (after fixes)
├── 2026-04-02:15-00_a1b2c3d_gpt-4o-mini.md   # Run 002 (different model)
└── ...
```

This structure enables:
- **`git log evaluation/`** to see evaluation history
- **`ls evaluation/*qwen3.5*`** to filter by model
- **`diff evaluation/run001.md evaluation/run002.md`** to see progress
- Sorting by date naturally via filename prefix

---

## 9. Evaluation Log Index

Quick reference for all evaluation runs. Each entry links to its result file.

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
