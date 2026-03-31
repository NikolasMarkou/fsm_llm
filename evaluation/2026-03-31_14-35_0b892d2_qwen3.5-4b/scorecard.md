# Evaluation: 2026-03-31 14:37

- **Date**: 2026-03-31 14:37
- **Git commit**: 0b892d2
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 5
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | meta/build_agent | 4 (PASS) | 4.4s |  | 4.4s |
| 2 | meta/build_fsm | 4 (PASS) | 4.5s |  | 4.5s |
| 3 | meta/build_workflow | 4 (PASS) | 3.8s |  | 3.8s |
| 4 | meta/meta_from_spec | 4 (PASS) | 40.1s |  | 40.1s |
| 5 | meta/meta_review_loop | 4 (PASS) | 62.2s |  | 62.2s |

## Summary

- **Total examples**: 5
- **Score distribution**: 5x4, 0x3, 0x2, 0x1, 0x0
- **Health Score**: 20/20 = **100.0%**
- **Category breakdown**:
  - meta: 20/20 (100%)

## Timing

- **Total wall time**: 115.1s (sequential equivalent)
- **Fastest**: 3.8s
- **Slowest**: 62.2s
- **Mean**: 23.0s
