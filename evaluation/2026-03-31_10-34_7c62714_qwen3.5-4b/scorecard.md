# Evaluation: 2026-03-31 10:36

- **Date**: 2026-03-31 10:36
- **Git commit**: 7c62714
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 5
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | meta/build_agent | 4 (PASS) | 10.5s |  | 10.5s |
| 2 | meta/build_fsm | 4 (PASS) | 8.0s |  | 8.0s |
| 3 | meta/build_workflow | 4 (PASS) | 9.2s |  | 9.2s |
| 4 | meta/meta_from_spec | 4 (PASS) | 156.0s |  | 156.0s |
| 5 | meta/meta_review_loop | 4 (PASS) | 108.9s |  | 108.9s |

## Summary

- **Total examples**: 5
- **Score distribution**: 5x4, 0x3, 0x2, 0x1, 0x0
- **Health Score**: 20/20 = **100.0%**
- **Category breakdown**:
  - meta: 20/20 (100%)

## Timing

- **Total wall time**: 292.6s (sequential equivalent)
- **Fastest**: 8.0s
- **Slowest**: 156.0s
- **Mean**: 58.5s
