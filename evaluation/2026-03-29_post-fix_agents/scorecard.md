# Evaluation: 2026-03-29 15:30

- **Date**: 2026-03-29 15:30
- **Git commit**: 453e22f
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 27
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | agents/adapt | 4 (PASS) | 43.8s |  | 43.8s |
| 2 | agents/agent_as_tool | 1 (BROKEN) | 240.0s | F-LOOP | Timeout (240s) |
| 3 | agents/classified_dispatch | 4 (PASS) | 137.6s |  | 137.6s |
| 4 | agents/classified_tools | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 5 | agents/concurrent_react | 1 (BROKEN) | 240.0s | F-LOOP | Timeout (240s) |
| 6 | agents/debate | 4 (PASS) | 94.4s |  | 94.4s |
| 7 | agents/evaluator_optimizer | 4 (PASS) | 91.5s |  | 91.5s |
| 8 | agents/full_pipeline | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 9 | agents/hierarchical_tools | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 10 | agents/hitl_approval | 4 (PASS) | 48.4s |  | 48.4s |
| 11 | agents/maker_checker | 4 (PASS) | 76.6s |  | 76.6s |
| 12 | agents/memory_agent | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 13 | agents/multi_tool_recovery | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 14 | agents/orchestrator | 4 (PASS) | 13.3s |  | 13.3s |
| 15 | agents/plan_execute | 4 (PASS) | 34.8s |  | 34.8s |
| 16 | agents/prompt_chain | 4 (PASS) | 36.2s |  | 36.2s |
| 17 | agents/react_hitl_combined | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 18 | agents/react_search | 3 (MOSTLY) | 155.5s | F-LOOP | 155.5s |
| 19 | agents/reasoning_stacking | 4 (PASS) | 30.1s |  | 30.1s |
| 20 | agents/reasoning_tool | 4 (PASS) | 30.1s |  | 30.1s |
| 21 | agents/reflexion | 4 (PASS) | 75.3s |  | 75.3s |
| 22 | agents/rewoo | 4 (PASS) | 19.5s |  | 19.5s |
| 23 | agents/self_consistency | 4 (PASS) | 24.8s |  | 24.8s |
| 24 | agents/skill_loader | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 25 | agents/structured_output | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 26 | agents/tool_decorator | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 27 | agents/workflow_agent | 4 (PASS) | 8.8s |  | 8.8s |

## Summary

- **Total examples**: 27
- **Score distribution**: 15x4, 1x3, 0x2, 11x1, 0x0
- **Health Score**: 74/108 = **68.5%**
- **Category breakdown**:
  - agents: 74/108 (69%)
- **Top failure codes**: F-LOOP (12)

## Timing

- **Total wall time**: 3021.1s (sequential equivalent)
- **Fastest**: 8.8s
- **Slowest**: 240.0s
- **Mean**: 111.9s
