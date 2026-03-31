# Evaluation: 2026-03-31 11:22

- **Date**: 2026-03-31 11:22
- **Git commit**: b7ce881
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 5
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | meta/build_agent | 4 (PASS) | 4.9s |  | 4.9s |
| 2 | meta/build_fsm | 4 (PASS) | 5.2s |  | 5.2s |
| 3 | meta/build_workflow | 4 (PASS) | 4.6s |  | 4.6s |
| 4 | meta/meta_from_spec | 1 (BROKEN) | 240.1s | F-LOOP | Timeout (240s) |
| 5 | meta/meta_review_loop | 4 (PASS) | 238.9s |  | 238.9s |

## Summary

- **Total examples**: 5
- **Score distribution**: 4x4, 0x3, 0x2, 1x1, 0x0
- **Health Score**: 17/20 = **85.0%**
- **Category breakdown**:
  - meta: 17/20 (85%)
- **Top failure codes**: F-LOOP (1)

## Timing

- **Total wall time**: 493.7s (sequential equivalent)
- **Fastest**: 4.6s
- **Slowest**: 240.1s
- **Mean**: 98.7s
