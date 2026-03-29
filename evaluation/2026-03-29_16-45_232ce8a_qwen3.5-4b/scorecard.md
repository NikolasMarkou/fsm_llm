# Evaluation: 2026-03-29 16:58

- **Date**: 2026-03-29 16:58
- **Git commit**: 232ce8a
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 50
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 27.8s |  | 27.8s |
| 2 | advanced/context_compactor | 4 (PASS) | 25.5s |  | 25.5s |
| 3 | advanced/e_commerce | 4 (PASS) | 151.6s |  | 151.6s |
| 4 | advanced/handler_hooks | 4 (PASS) | 15.0s |  | 15.0s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 24.7s |  | 24.7s |
| 6 | advanced/support_pipeline | 2 (PARTIAL) | 40.1s | F-CODE | Exit 1 |
| 7 | advanced/yoga_instructions | 4 (PASS) | 19.7s |  | 19.7s |
| 8 | agents/adapt | 4 (PASS) | 147.6s |  | 147.6s |
| 9 | agents/agent_as_tool | 1 (BROKEN) | 240.1s | F-LOOP | Timeout (240s) |
| 10 | agents/classified_dispatch | 4 (PASS) | 120.0s |  | 120.0s |
| 11 | agents/classified_tools | 4 (PASS) | 36.7s |  | 36.7s |
| 12 | agents/concurrent_react | 4 (PASS) | 133.1s |  | 133.1s |
| 13 | agents/debate | 4 (PASS) | 78.4s |  | 78.4s |
| 14 | agents/evaluator_optimizer | 4 (PASS) | 113.0s |  | 113.0s |
| 15 | agents/full_pipeline | 4 (PASS) | 71.9s |  | 71.9s |
| 16 | agents/hierarchical_tools | 3 (MOSTLY) | 87.1s | F-PARSE | 87.1s |
| 17 | agents/hitl_approval | 4 (PASS) | 111.3s |  | 111.3s |
| 18 | agents/maker_checker | 4 (PASS) | 50.7s |  | 50.7s |
| 19 | agents/memory_agent | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 20 | agents/multi_tool_recovery | 4 (PASS) | 58.8s |  | 58.8s |
| 21 | agents/orchestrator | 4 (PASS) | 13.6s |  | 13.6s |
| 22 | agents/plan_execute | 4 (PASS) | 26.2s |  | 26.2s |
| 23 | agents/prompt_chain | 4 (PASS) | 27.6s |  | 27.6s |
| 24 | agents/react_hitl_combined | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 25 | agents/react_search | 4 (PASS) | 40.6s |  | 40.6s |
| 26 | agents/reasoning_stacking | 4 (PASS) | 24.4s |  | 24.4s |
| 27 | agents/reasoning_tool | 4 (PASS) | 51.6s |  | 51.6s |
| 28 | agents/reflexion | 4 (PASS) | 120.7s |  | 120.7s |
| 29 | agents/rewoo | 4 (PASS) | 14.9s |  | 14.9s |
| 30 | agents/self_consistency | 4 (PASS) | 24.4s |  | 24.4s |
| 31 | agents/skill_loader | 4 (PASS) | 147.9s |  | 147.9s |
| 32 | agents/structured_output | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 33 | agents/tool_decorator | 4 (PASS) | 85.8s |  | 85.8s |
| 34 | agents/workflow_agent | 4 (PASS) | 8.2s |  | 8.2s |
| 35 | basic/form_filling | 4 (PASS) | 26.3s |  | 26.3s |
| 36 | basic/multi_turn_extraction | 4 (PASS) | 42.2s |  | 42.2s |
| 37 | basic/simple_greeting | 4 (PASS) | 24.2s |  | 24.2s |
| 38 | basic/story_time | 4 (PASS) | 33.7s |  | 33.7s |
| 39 | classification/classified_transitions | 4 (PASS) | 32.6s |  | 32.6s |
| 40 | classification/intent_routing | 4 (PASS) | 14.7s |  | 14.7s |
| 41 | classification/multi_intent | 4 (PASS) | 29.6s |  | 29.6s |
| 42 | classification/smart_helpdesk | 4 (PASS) | 22.0s |  | 22.0s |
| 43 | intermediate/adaptive_quiz | 4 (PASS) | 24.1s |  | 24.1s |
| 44 | intermediate/book_recommendation | 3 (MOSTLY) | 23.6s | F-PARSE | 23.6s |
| 45 | intermediate/product_recommendation | 4 (PASS) | 18.9s |  | 18.9s |
| 46 | meta/build_fsm | 4 (PASS) | 6.8s |  | 6.8s |
| 47 | reasoning/math_tutor | 4 (PASS) | 11.1s |  | 11.1s |
| 48 | workflows/agent_workflow_chain | 3 (MOSTLY) | 9.5s | F-EXTRACT | 9.5s |
| 49 | workflows/order_processing | 4 (PASS) | 23.7s |  | 23.7s |
| 50 | workflows/parallel_steps | 4 (PASS) | 5.2s |  | 5.2s |

## Summary

- **Total examples**: 50
- **Score distribution**: 42x4, 3x3, 1x2, 4x1, 0x0
- **Health Score**: 183/200 = **91.5%**
- **Category breakdown**:
  - advanced: 26/28 (93%)
  - agents: 95/108 (88%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 11/12 (92%)
  - meta: 4/4 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 11/12 (92%)
- **Top failure codes**: F-LOOP (4), F-PARSE (2), F-CODE (1), F-EXTRACT (1)

## Timing

- **Total wall time**: 3027.6s (sequential equivalent)
- **Fastest**: 5.2s
- **Slowest**: 240.1s
- **Mean**: 60.6s
