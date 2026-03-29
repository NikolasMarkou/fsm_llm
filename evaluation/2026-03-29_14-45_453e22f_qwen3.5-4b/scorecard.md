# Evaluation: 2026-03-29 14:59

- **Date**: 2026-03-29 14:59
- **Git commit**: 453e22f
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 50
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 29.4s |  | 29.4s |
| 2 | advanced/context_compactor | 4 (PASS) | 25.6s |  | 25.6s |
| 3 | advanced/e_commerce | 4 (PASS) | 138.9s |  | 138.9s |
| 4 | advanced/handler_hooks | 4 (PASS) | 19.8s |  | 19.8s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 27.9s |  | 27.9s |
| 6 | advanced/support_pipeline | 2 (PARTIAL) | 35.1s | F-CODE | Exit 1 |
| 7 | advanced/yoga_instructions | 4 (PASS) | 20.9s |  | 20.9s |
| 8 | agents/adapt | 4 (PASS) | 135.5s |  | 135.5s |
| 9 | agents/agent_as_tool | 4 (PASS) | 29.7s |  | 29.7s |
| 10 | agents/classified_dispatch | 4 (PASS) | 136.0s |  | 136.0s |
| 11 | agents/classified_tools | 4 (PASS) | 135.0s |  | 135.0s |
| 12 | agents/concurrent_react | 4 (PASS) | 114.3s |  | 114.3s |
| 13 | agents/debate | 4 (PASS) | 78.2s |  | 78.2s |
| 14 | agents/evaluator_optimizer | 4 (PASS) | 101.2s |  | 101.2s |
| 15 | agents/full_pipeline | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 16 | agents/hierarchical_tools | 3 (MOSTLY) | 56.6s | F-PARSE | 56.6s |
| 17 | agents/hitl_approval | 4 (PASS) | 128.9s |  | 128.9s |
| 18 | agents/maker_checker | 4 (PASS) | 56.6s |  | 56.6s |
| 19 | agents/memory_agent | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 20 | agents/multi_tool_recovery | 4 (PASS) | 32.9s |  | 32.9s |
| 21 | agents/orchestrator | 4 (PASS) | 12.4s |  | 12.4s |
| 22 | agents/plan_execute | 4 (PASS) | 32.8s |  | 32.8s |
| 23 | agents/prompt_chain | 4 (PASS) | 42.2s |  | 42.2s |
| 24 | agents/react_hitl_combined | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 25 | agents/react_search | 4 (PASS) | 114.7s |  | 114.7s |
| 26 | agents/reasoning_stacking | 4 (PASS) | 26.5s |  | 26.5s |
| 27 | agents/reasoning_tool | 4 (PASS) | 80.4s |  | 80.4s |
| 28 | agents/reflexion | 4 (PASS) | 161.5s |  | 161.5s |
| 29 | agents/rewoo | 4 (PASS) | 19.8s |  | 19.8s |
| 30 | agents/self_consistency | 4 (PASS) | 23.7s |  | 23.7s |
| 31 | agents/skill_loader | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 32 | agents/structured_output | 3 (MOSTLY) | 139.6s | F-PARSE | 139.6s |
| 33 | agents/tool_decorator | 4 (PASS) | 146.3s |  | 146.3s |
| 34 | agents/workflow_agent | 4 (PASS) | 8.8s |  | 8.8s |
| 35 | basic/form_filling | 4 (PASS) | 21.2s |  | 21.2s |
| 36 | basic/multi_turn_extraction | 4 (PASS) | 40.8s |  | 40.8s |
| 37 | basic/simple_greeting | 4 (PASS) | 19.4s |  | 19.4s |
| 38 | basic/story_time | 4 (PASS) | 27.4s |  | 27.4s |
| 39 | classification/classified_transitions | 4 (PASS) | 25.3s |  | 25.3s |
| 40 | classification/intent_routing | 4 (PASS) | 12.9s |  | 12.9s |
| 41 | classification/multi_intent | 4 (PASS) | 30.4s |  | 30.4s |
| 42 | classification/smart_helpdesk | 4 (PASS) | 20.9s |  | 20.9s |
| 43 | intermediate/adaptive_quiz | 4 (PASS) | 21.7s |  | 21.7s |
| 44 | intermediate/book_recommendation | 3 (MOSTLY) | 23.5s | F-PARSE | 23.5s |
| 45 | intermediate/product_recommendation | 4 (PASS) | 19.5s |  | 19.5s |
| 46 | meta/build_fsm | 4 (PASS) | 8.4s |  | 8.4s |
| 47 | reasoning/math_tutor | 4 (PASS) | 11.3s |  | 11.3s |
| 48 | workflows/agent_workflow_chain | 3 (MOSTLY) | 9.7s | F-EXTRACT | 9.7s |
| 49 | workflows/order_processing | 4 (PASS) | 22.5s |  | 22.5s |
| 50 | workflows/parallel_steps | 4 (PASS) | 5.1s |  | 5.1s |

## Summary

- **Total examples**: 50
- **Score distribution**: 41x4, 4x3, 1x2, 4x1, 0x0
- **Health Score**: 182/200 = **91.0%**
- **Category breakdown**:
  - advanced: 26/28 (93%)
  - agents: 94/108 (87%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 11/12 (92%)
  - meta: 4/4 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 11/12 (92%)
- **Top failure codes**: F-LOOP (4), F-PARSE (3), F-CODE (1), F-EXTRACT (1)

## Timing

- **Total wall time**: 3151.2s (sequential equivalent)
- **Fastest**: 5.1s
- **Slowest**: 180.1s
- **Mean**: 63.0s
