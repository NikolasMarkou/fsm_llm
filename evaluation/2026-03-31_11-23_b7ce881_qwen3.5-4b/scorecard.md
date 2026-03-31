# Evaluation: 2026-03-31 11:27

- **Date**: 2026-03-31 11:27
- **Git commit**: b7ce881
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 5
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | meta/build_agent | 4 (PASS) | 4.9s |  | 4.9s |
| 2 | meta/build_fsm | 4 (PASS) | 5.0s |  | 5.0s |
| 3 | meta/build_workflow | 4 (PASS) | 4.9s |  | 4.9s |
| 4 | meta/meta_from_spec | 3 (MOSTLY) | 162.3s | F-LOOP | 162.3s |
| 5 | meta/meta_review_loop | 1 (BROKEN) | 240.0s | F-LOOP | Timeout (240s) |

## Summary

- **Total examples**: 5
- **Score distribution**: 3x4, 1x3, 0x2, 1x1, 0x0
- **Health Score**: 16/20 = **80.0%**
- **Category breakdown**:
  - meta: 16/20 (80%)
- **Top failure codes**: F-LOOP (2)

## Timing

- **Total wall time**: 417.1s (sequential equivalent)
- **Fastest**: 4.9s
- **Slowest**: 240.0s
- **Mean**: 83.4s
