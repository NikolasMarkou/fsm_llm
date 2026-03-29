# Evaluation: 2026-03-29 17:36

- **Date**: 2026-03-29 17:36
- **Git commit**: 232ce8a
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 50
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 24.0s |  | 24.0s |
| 2 | advanced/context_compactor | 4 (PASS) | 23.0s |  | 23.0s |
| 3 | advanced/e_commerce | 4 (PASS) | 153.8s |  | 153.8s |
| 4 | advanced/handler_hooks | 4 (PASS) | 18.5s |  | 18.5s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 25.4s |  | 25.4s |
| 6 | advanced/support_pipeline | 4 (PASS) | 44.1s |  | 44.1s |
| 7 | advanced/yoga_instructions | 4 (PASS) | 21.4s |  | 21.4s |
| 8 | agents/adapt | 4 (PASS) | 121.2s |  | 121.2s |
| 9 | agents/agent_as_tool | 4 (PASS) | 204.8s |  | 204.8s |
| 10 | agents/classified_dispatch | 4 (PASS) | 119.2s |  | 119.2s |
| 11 | agents/classified_tools | 4 (PASS) | 142.2s |  | 142.2s |
| 12 | agents/concurrent_react | 4 (PASS) | 137.5s |  | 137.5s |
| 13 | agents/debate | 4 (PASS) | 79.7s |  | 79.7s |
| 14 | agents/evaluator_optimizer | 4 (PASS) | 62.2s |  | 62.2s |
| 15 | agents/full_pipeline | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 16 | agents/hierarchical_tools | 4 (PASS) | 142.6s |  | 142.6s |
| 17 | agents/hitl_approval | 4 (PASS) | 167.9s |  | 167.9s |
| 18 | agents/maker_checker | 3 (MOSTLY) | 86.5s | F-EXTRACT | 86.5s |
| 19 | agents/memory_agent | 4 (PASS) | 272.1s |  | 272.1s |
| 20 | agents/multi_tool_recovery | 4 (PASS) | 105.7s |  | 105.7s |
| 21 | agents/orchestrator | 4 (PASS) | 16.5s |  | 16.5s |
| 22 | agents/plan_execute | 4 (PASS) | 28.6s |  | 28.6s |
| 23 | agents/prompt_chain | 4 (PASS) | 31.9s |  | 31.9s |
| 24 | agents/react_hitl_combined | 4 (PASS) | 237.8s |  | 237.8s |
| 25 | agents/react_search | 4 (PASS) | 157.8s |  | 157.8s |
| 26 | agents/reasoning_stacking | 4 (PASS) | 107.0s |  | 107.0s |
| 27 | agents/reasoning_tool | 4 (PASS) | 47.6s |  | 47.6s |
| 28 | agents/reflexion | 4 (PASS) | 174.2s |  | 174.2s |
| 29 | agents/rewoo | 4 (PASS) | 19.0s |  | 19.0s |
| 30 | agents/self_consistency | 4 (PASS) | 23.6s |  | 23.6s |
| 31 | agents/skill_loader | 4 (PASS) | 156.5s |  | 156.5s |
| 32 | agents/structured_output | 4 (PASS) | 156.2s |  | 156.2s |
| 33 | agents/tool_decorator | 4 (PASS) | 120.7s |  | 120.7s |
| 34 | agents/workflow_agent | 4 (PASS) | 8.4s |  | 8.4s |
| 35 | basic/form_filling | 4 (PASS) | 19.9s |  | 19.9s |
| 36 | basic/multi_turn_extraction | 4 (PASS) | 37.0s |  | 37.0s |
| 37 | basic/simple_greeting | 4 (PASS) | 21.5s |  | 21.5s |
| 38 | basic/story_time | 4 (PASS) | 30.6s |  | 30.6s |
| 39 | classification/classified_transitions | 4 (PASS) | 25.7s |  | 25.7s |
| 40 | classification/intent_routing | 4 (PASS) | 13.3s |  | 13.3s |
| 41 | classification/multi_intent | 4 (PASS) | 30.8s |  | 30.8s |
| 42 | classification/smart_helpdesk | 4 (PASS) | 21.7s |  | 21.7s |
| 43 | intermediate/adaptive_quiz | 4 (PASS) | 22.5s |  | 22.5s |
| 44 | intermediate/book_recommendation | 4 (PASS) | 23.5s |  | 23.5s |
| 45 | intermediate/product_recommendation | 4 (PASS) | 16.7s |  | 16.7s |
| 46 | meta/build_fsm | 4 (PASS) | 6.2s |  | 6.2s |
| 47 | reasoning/math_tutor | 4 (PASS) | 9.5s |  | 9.5s |
| 48 | workflows/agent_workflow_chain | 4 (PASS) | 8.7s |  | 8.7s |
| 49 | workflows/order_processing | 4 (PASS) | 23.8s |  | 23.8s |
| 50 | workflows/parallel_steps | 4 (PASS) | 5.0s |  | 5.0s |

## Summary

- **Total examples**: 50
- **Score distribution**: 48x4, 1x3, 0x2, 1x1, 0x0
- **Health Score**: 196/200 = **98.0%**
- **Category breakdown**:
  - advanced: 28/28 (100%)
  - agents: 104/108 (96%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 4/4 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 12/12 (100%)
- **Top failure codes**: F-LOOP (1), F-EXTRACT (1)

## Timing

- **Total wall time**: 3734.2s (sequential equivalent)
- **Fastest**: 5.0s
- **Slowest**: 272.1s
- **Mean**: 74.7s
