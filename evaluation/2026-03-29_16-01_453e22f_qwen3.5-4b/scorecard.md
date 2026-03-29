# Evaluation: 2026-03-29 16:14

- **Date**: 2026-03-29 16:14
- **Git commit**: 453e22f
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 50
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 27.0s |  | 27.0s |
| 2 | advanced/context_compactor | 4 (PASS) | 24.8s |  | 24.8s |
| 3 | advanced/e_commerce | 4 (PASS) | 161.4s |  | 161.4s |
| 4 | advanced/handler_hooks | 4 (PASS) | 18.7s |  | 18.7s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 21.3s |  | 21.3s |
| 6 | advanced/support_pipeline | 2 (PARTIAL) | 41.1s | F-CODE | Exit 1 |
| 7 | advanced/yoga_instructions | 4 (PASS) | 20.3s |  | 20.3s |
| 8 | agents/adapt | 4 (PASS) | 130.9s |  | 130.9s |
| 9 | agents/agent_as_tool | 1 (BROKEN) | 240.1s | F-LOOP | Timeout (240s) |
| 10 | agents/classified_dispatch | 4 (PASS) | 129.0s |  | 129.0s |
| 11 | agents/classified_tools | 4 (PASS) | 48.7s |  | 48.7s |
| 12 | agents/concurrent_react | 4 (PASS) | 128.4s |  | 128.4s |
| 13 | agents/debate | 4 (PASS) | 93.1s |  | 93.1s |
| 14 | agents/evaluator_optimizer | 4 (PASS) | 90.3s |  | 90.3s |
| 15 | agents/full_pipeline | 4 (PASS) | 102.1s |  | 102.1s |
| 16 | agents/hierarchical_tools | 3 (MOSTLY) | 81.8s | F-PARSE | 81.8s |
| 17 | agents/hitl_approval | 4 (PASS) | 59.8s |  | 59.8s |
| 18 | agents/maker_checker | 3 (MOSTLY) | 77.6s | F-EXTRACT | 77.6s |
| 19 | agents/memory_agent | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 20 | agents/multi_tool_recovery | 4 (PASS) | 101.8s |  | 101.8s |
| 21 | agents/orchestrator | 4 (PASS) | 13.1s |  | 13.1s |
| 22 | agents/plan_execute | 4 (PASS) | 35.2s |  | 35.2s |
| 23 | agents/prompt_chain | 4 (PASS) | 39.4s |  | 39.4s |
| 24 | agents/react_hitl_combined | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 25 | agents/react_search | 4 (PASS) | 43.5s |  | 43.5s |
| 26 | agents/reasoning_stacking | 4 (PASS) | 177.7s |  | 177.7s |
| 27 | agents/reasoning_tool | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 28 | agents/reflexion | 4 (PASS) | 122.1s |  | 122.1s |
| 29 | agents/rewoo | 4 (PASS) | 19.1s |  | 19.1s |
| 30 | agents/self_consistency | 4 (PASS) | 28.4s |  | 28.4s |
| 31 | agents/skill_loader | 4 (PASS) | 67.6s |  | 67.6s |
| 32 | agents/structured_output | 3 (MOSTLY) | 71.7s | F-PARSE | 71.7s |
| 33 | agents/tool_decorator | 4 (PASS) | 66.4s |  | 66.4s |
| 34 | agents/workflow_agent | 4 (PASS) | 10.0s |  | 10.0s |
| 35 | basic/form_filling | 4 (PASS) | 21.7s |  | 21.7s |
| 36 | basic/multi_turn_extraction | 4 (PASS) | 42.0s |  | 42.0s |
| 37 | basic/simple_greeting | 4 (PASS) | 21.3s |  | 21.3s |
| 38 | basic/story_time | 4 (PASS) | 28.6s |  | 28.6s |
| 39 | classification/classified_transitions | 4 (PASS) | 26.3s |  | 26.3s |
| 40 | classification/intent_routing | 4 (PASS) | 12.1s |  | 12.1s |
| 41 | classification/multi_intent | 4 (PASS) | 31.1s |  | 31.1s |
| 42 | classification/smart_helpdesk | 4 (PASS) | 20.8s |  | 20.8s |
| 43 | intermediate/adaptive_quiz | 4 (PASS) | 22.8s |  | 22.8s |
| 44 | intermediate/book_recommendation | 3 (MOSTLY) | 23.2s | F-PARSE | 23.2s |
| 45 | intermediate/product_recommendation | 4 (PASS) | 16.1s |  | 16.1s |
| 46 | meta/build_fsm | 4 (PASS) | 7.5s |  | 7.5s |
| 47 | reasoning/math_tutor | 4 (PASS) | 9.7s |  | 9.7s |
| 48 | workflows/agent_workflow_chain | 3 (MOSTLY) | 9.1s | F-EXTRACT | 9.1s |
| 49 | workflows/order_processing | 4 (PASS) | 24.4s |  | 24.4s |
| 50 | workflows/parallel_steps | 4 (PASS) | 5.1s |  | 5.1s |

## Summary

- **Total examples**: 50
- **Score distribution**: 40x4, 5x3, 1x2, 4x1, 0x0
- **Health Score**: 181/200 = **90.5%**
- **Category breakdown**:
  - advanced: 26/28 (93%)
  - agents: 93/108 (86%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 11/12 (92%)
  - meta: 4/4 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 11/12 (92%)
- **Top failure codes**: F-LOOP (4), F-PARSE (3), F-EXTRACT (2), F-CODE (1)

## Timing

- **Total wall time**: 3154.5s (sequential equivalent)
- **Fastest**: 5.1s
- **Slowest**: 240.1s
- **Mean**: 63.1s
