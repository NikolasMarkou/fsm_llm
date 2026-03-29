# Evaluation: 2026-03-29 17:20

- **Date**: 2026-03-29 17:20
- **Git commit**: 232ce8a
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 50
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 25.4s |  | 25.4s |
| 2 | advanced/context_compactor | 4 (PASS) | 27.1s |  | 27.1s |
| 3 | advanced/e_commerce | 4 (PASS) | 139.4s |  | 139.4s |
| 4 | advanced/handler_hooks | 4 (PASS) | 19.2s |  | 19.2s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 24.8s |  | 24.8s |
| 6 | advanced/support_pipeline | 4 (PASS) | 40.7s |  | 40.7s |
| 7 | advanced/yoga_instructions | 4 (PASS) | 22.3s |  | 22.3s |
| 8 | agents/adapt | 4 (PASS) | 117.8s |  | 117.8s |
| 9 | agents/agent_as_tool | 4 (PASS) | 235.2s |  | 235.2s |
| 10 | agents/classified_dispatch | 4 (PASS) | 108.7s |  | 108.7s |
| 11 | agents/classified_tools | 4 (PASS) | 75.6s |  | 75.6s |
| 12 | agents/concurrent_react | 4 (PASS) | 179.0s |  | 179.0s |
| 13 | agents/debate | 4 (PASS) | 99.1s |  | 99.1s |
| 14 | agents/evaluator_optimizer | 4 (PASS) | 113.1s |  | 113.1s |
| 15 | agents/full_pipeline | 4 (PASS) | 62.9s |  | 62.9s |
| 16 | agents/hierarchical_tools | 4 (PASS) | 103.1s |  | 103.1s |
| 17 | agents/hitl_approval | 4 (PASS) | 84.4s |  | 84.4s |
| 18 | agents/maker_checker | 3 (MOSTLY) | 73.3s | F-EXTRACT | 73.3s |
| 19 | agents/memory_agent | 4 (PASS) | 209.3s |  | 209.3s |
| 20 | agents/multi_tool_recovery | 4 (PASS) | 80.9s |  | 80.9s |
| 21 | agents/orchestrator | 4 (PASS) | 14.2s |  | 14.2s |
| 22 | agents/plan_execute | 4 (PASS) | 30.8s |  | 30.8s |
| 23 | agents/prompt_chain | 4 (PASS) | 34.4s |  | 34.4s |
| 24 | agents/react_hitl_combined | 4 (PASS) | 211.7s |  | 211.7s |
| 25 | agents/react_search | 4 (PASS) | 115.1s |  | 115.1s |
| 26 | agents/reasoning_stacking | 4 (PASS) | 62.8s |  | 62.8s |
| 27 | agents/reasoning_tool | 3 (MOSTLY) | 175.7s | F-LOOP | 175.7s |
| 28 | agents/reflexion | 4 (PASS) | 169.7s |  | 169.7s |
| 29 | agents/rewoo | 4 (PASS) | 19.1s |  | 19.1s |
| 30 | agents/self_consistency | 4 (PASS) | 24.6s |  | 24.6s |
| 31 | agents/skill_loader | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 32 | agents/structured_output | 3 (MOSTLY) | 147.7s | F-LOOP | 147.7s |
| 33 | agents/tool_decorator | 4 (PASS) | 159.8s |  | 159.8s |
| 34 | agents/workflow_agent | 4 (PASS) | 8.5s |  | 8.5s |
| 35 | basic/form_filling | 4 (PASS) | 23.1s |  | 23.1s |
| 36 | basic/multi_turn_extraction | 4 (PASS) | 41.7s |  | 41.7s |
| 37 | basic/simple_greeting | 4 (PASS) | 19.3s |  | 19.3s |
| 38 | basic/story_time | 4 (PASS) | 31.7s |  | 31.7s |
| 39 | classification/classified_transitions | 4 (PASS) | 28.0s |  | 28.0s |
| 40 | classification/intent_routing | 4 (PASS) | 13.7s |  | 13.7s |
| 41 | classification/multi_intent | 4 (PASS) | 34.0s |  | 34.0s |
| 42 | classification/smart_helpdesk | 4 (PASS) | 23.3s |  | 23.3s |
| 43 | intermediate/adaptive_quiz | 4 (PASS) | 26.8s |  | 26.8s |
| 44 | intermediate/book_recommendation | 4 (PASS) | 20.9s |  | 20.9s |
| 45 | intermediate/product_recommendation | 4 (PASS) | 18.8s |  | 18.8s |
| 46 | meta/build_fsm | 4 (PASS) | 8.3s |  | 8.3s |
| 47 | reasoning/math_tutor | 4 (PASS) | 11.6s |  | 11.6s |
| 48 | workflows/agent_workflow_chain | 4 (PASS) | 11.1s |  | 11.1s |
| 49 | workflows/order_processing | 4 (PASS) | 23.4s |  | 23.4s |
| 50 | workflows/parallel_steps | 4 (PASS) | 5.1s |  | 5.1s |

## Summary

- **Total examples**: 50
- **Score distribution**: 46x4, 3x3, 0x2, 1x1, 0x0
- **Health Score**: 194/200 = **97.0%**
- **Category breakdown**:
  - advanced: 28/28 (100%)
  - agents: 102/108 (94%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 4/4 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 12/12 (100%)
- **Top failure codes**: F-LOOP (3), F-EXTRACT (1)

## Timing

- **Total wall time**: 3536.3s (sequential equivalent)
- **Fastest**: 5.1s
- **Slowest**: 235.2s
- **Mean**: 70.7s
