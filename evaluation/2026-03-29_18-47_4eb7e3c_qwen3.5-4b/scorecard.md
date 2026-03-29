# Evaluation: 2026-03-29 19:08

- **Date**: 2026-03-29 19:08
- **Git commit**: 4eb7e3c
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 70
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 25.3s |  | 25.3s |
| 2 | advanced/context_compactor | 4 (PASS) | 24.3s |  | 24.3s |
| 3 | advanced/e_commerce | 4 (PASS) | 151.1s |  | 151.1s |
| 4 | advanced/handler_hooks | 4 (PASS) | 15.7s |  | 15.7s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 25.7s |  | 25.7s |
| 6 | advanced/support_pipeline | 4 (PASS) | 40.2s |  | 40.2s |
| 7 | advanced/yoga_instructions | 4 (PASS) | 20.8s |  | 20.8s |
| 8 | agents/adapt | 4 (PASS) | 128.5s |  | 128.5s |
| 9 | agents/adapt_with_memory | 4 (PASS) | 33.0s |  | 33.0s |
| 10 | agents/agent_as_tool | 4 (PASS) | 260.1s |  | 260.1s |
| 11 | agents/agent_memory_chain | 1 (BROKEN) | 300.0s | F-LOOP | Timeout (300s) |
| 12 | agents/classified_dispatch | 4 (PASS) | 118.2s |  | 118.2s |
| 13 | agents/classified_tools | 4 (PASS) | 73.0s |  | 73.0s |
| 14 | agents/concurrent_react | 4 (PASS) | 153.2s |  | 153.2s |
| 15 | agents/consistency_with_tools | 4 (PASS) | 34.7s |  | 34.7s |
| 16 | agents/debate | 4 (PASS) | 115.4s |  | 115.4s |
| 17 | agents/debate_with_tools | 4 (PASS) | 3.6s |  | 3.6s |
| 18 | agents/eval_opt_structured | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 19 | agents/evaluator_optimizer | 4 (PASS) | 135.4s |  | 135.4s |
| 20 | agents/full_pipeline | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 21 | agents/hierarchical_orchestrator | 4 (PASS) | 21.5s |  | 21.5s |
| 22 | agents/hierarchical_tools | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 23 | agents/hitl_approval | 4 (PASS) | 55.4s |  | 55.4s |
| 24 | agents/maker_checker | 3 (MOSTLY) | 89.1s | F-EXTRACT | 89.1s |
| 25 | agents/maker_checker_code | 3 (MOSTLY) | 143.2s | F-EXTRACT | 143.2s |
| 26 | agents/memory_agent | 1 (BROKEN) | 300.1s | F-LOOP | Timeout (300s) |
| 27 | agents/multi_debate_panel | 4 (PASS) | 81.6s |  | 81.6s |
| 28 | agents/multi_tool_recovery | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 29 | agents/orchestrator | 4 (PASS) | 16.0s |  | 16.0s |
| 30 | agents/orchestrator_specialist | 4 (PASS) | 17.5s |  | 17.5s |
| 31 | agents/pipeline_review | 3 (MOSTLY) | 101.3s | F-EXTRACT | 101.3s |
| 32 | agents/plan_execute | 4 (PASS) | 41.5s |  | 41.5s |
| 33 | agents/plan_execute_recovery | 4 (PASS) | 26.3s |  | 26.3s |
| 34 | agents/prompt_chain | 4 (PASS) | 34.1s |  | 34.1s |
| 35 | agents/react_hitl_combined | 4 (PASS) | 232.8s |  | 232.8s |
| 36 | agents/react_search | 4 (PASS) | 28.9s |  | 28.9s |
| 37 | agents/react_structured_pipeline | 4 (PASS) | 170.8s |  | 170.8s |
| 38 | agents/reasoning_stacking | 4 (PASS) | 51.1s |  | 51.1s |
| 39 | agents/reasoning_tool | 4 (PASS) | 75.8s |  | 75.8s |
| 40 | agents/reflexion | 4 (PASS) | 185.3s |  | 185.3s |
| 41 | agents/reflexion_code_gen | 4 (PASS) | 80.3s |  | 80.3s |
| 42 | agents/rewoo | 4 (PASS) | 22.8s |  | 22.8s |
| 43 | agents/rewoo_multi_step | 4 (PASS) | 24.1s |  | 24.1s |
| 44 | agents/self_consistency | 4 (PASS) | 22.9s |  | 22.9s |
| 45 | agents/skill_loader | 4 (PASS) | 97.0s |  | 97.0s |
| 46 | agents/structured_output | 4 (PASS) | 57.5s |  | 57.5s |
| 47 | agents/tool_decorator | 4 (PASS) | 55.6s |  | 55.6s |
| 48 | agents/workflow_agent | 4 (PASS) | 8.2s |  | 8.2s |
| 49 | basic/form_filling | 4 (PASS) | 19.7s |  | 19.7s |
| 50 | basic/multi_turn_extraction | 4 (PASS) | 39.8s |  | 39.8s |
| 51 | basic/simple_greeting | 4 (PASS) | 18.8s |  | 18.8s |
| 52 | basic/story_time | 4 (PASS) | 30.5s |  | 30.5s |
| 53 | classification/classified_transitions | 4 (PASS) | 26.7s |  | 26.7s |
| 54 | classification/intent_routing | 4 (PASS) | 11.9s |  | 11.9s |
| 55 | classification/multi_intent | 4 (PASS) | 32.0s |  | 32.0s |
| 56 | classification/smart_helpdesk | 4 (PASS) | 21.6s |  | 21.6s |
| 57 | intermediate/adaptive_quiz | 4 (PASS) | 23.4s |  | 23.4s |
| 58 | intermediate/book_recommendation | 4 (PASS) | 23.0s |  | 23.0s |
| 59 | intermediate/product_recommendation | 4 (PASS) | 17.5s |  | 17.5s |
| 60 | meta/build_agent | 4 (PASS) | 6.9s |  | 6.9s |
| 61 | meta/build_fsm | 4 (PASS) | 6.5s |  | 6.5s |
| 62 | meta/build_workflow | 4 (PASS) | 7.1s |  | 7.1s |
| 63 | meta/meta_from_spec | 4 (PASS) | 3.5s |  | 3.5s |
| 64 | meta/meta_review_loop | 3 (MOSTLY) | 46.9s | F-EXTRACT | 46.9s |
| 65 | reasoning/math_tutor | 4 (PASS) | 12.7s |  | 12.7s |
| 66 | workflows/agent_workflow_chain | 4 (PASS) | 13.2s |  | 13.2s |
| 67 | workflows/conditional_branching | 4 (PASS) | 3.8s |  | 3.8s |
| 68 | workflows/order_processing | 4 (PASS) | 45.2s |  | 45.2s |
| 69 | workflows/parallel_steps | 4 (PASS) | 5.1s |  | 5.1s |
| 70 | workflows/workflow_agent_loop | 4 (PASS) | 70.8s |  | 70.8s |

## Summary

- **Total examples**: 70
- **Score distribution**: 60x4, 4x3, 0x2, 6x1, 0x0
- **Health Score**: 258/280 = **92.1%**
- **Category breakdown**:
  - advanced: 28/28 (100%)
  - agents: 143/164 (87%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 19/20 (95%)
  - reasoning: 4/4 (100%)
  - workflows: 20/20 (100%)
- **Top failure codes**: F-LOOP (6), F-EXTRACT (4)

## Timing

- **Total wall time**: 4905.8s (sequential equivalent)
- **Fastest**: 3.5s
- **Slowest**: 300.1s
- **Mean**: 70.1s
