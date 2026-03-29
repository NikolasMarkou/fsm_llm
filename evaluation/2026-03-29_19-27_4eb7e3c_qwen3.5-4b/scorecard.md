# Evaluation: 2026-03-29 19:48

- **Date**: 2026-03-29 19:48
- **Git commit**: 4eb7e3c
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 70
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 22.8s |  | 22.8s |
| 2 | advanced/context_compactor | 4 (PASS) | 25.4s |  | 25.4s |
| 3 | advanced/e_commerce | 4 (PASS) | 149.5s |  | 149.5s |
| 4 | advanced/handler_hooks | 4 (PASS) | 16.7s |  | 16.7s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 21.9s |  | 21.9s |
| 6 | advanced/support_pipeline | 4 (PASS) | 45.4s |  | 45.4s |
| 7 | advanced/yoga_instructions | 4 (PASS) | 20.3s |  | 20.3s |
| 8 | agents/adapt | 4 (PASS) | 125.7s |  | 125.7s |
| 9 | agents/adapt_with_memory | 4 (PASS) | 185.6s |  | 185.6s |
| 10 | agents/agent_as_tool | 4 (PASS) | 270.4s |  | 270.4s |
| 11 | agents/agent_memory_chain | 1 (BROKEN) | 300.0s | F-LOOP | Timeout (300s) |
| 12 | agents/classified_dispatch | 4 (PASS) | 94.8s |  | 94.8s |
| 13 | agents/classified_tools | 4 (PASS) | 43.4s |  | 43.4s |
| 14 | agents/concurrent_react | 4 (PASS) | 152.7s |  | 152.7s |
| 15 | agents/consistency_with_tools | 4 (PASS) | 31.5s |  | 31.5s |
| 16 | agents/debate | 4 (PASS) | 104.0s |  | 104.0s |
| 17 | agents/debate_with_tools | 4 (PASS) | 3.7s |  | 3.7s |
| 18 | agents/eval_opt_structured | 4 (PASS) | 194.3s |  | 194.3s |
| 19 | agents/evaluator_optimizer | 4 (PASS) | 129.1s |  | 129.1s |
| 20 | agents/full_pipeline | 4 (PASS) | 153.2s |  | 153.2s |
| 21 | agents/hierarchical_orchestrator | 4 (PASS) | 28.4s |  | 28.4s |
| 22 | agents/hierarchical_tools | 4 (PASS) | 150.7s |  | 150.7s |
| 23 | agents/hitl_approval | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 24 | agents/maker_checker | 4 (PASS) | 78.1s |  | 78.1s |
| 25 | agents/maker_checker_code | 4 (PASS) | 129.8s |  | 129.8s |
| 26 | agents/memory_agent | 1 (BROKEN) | 300.0s | F-LOOP | Timeout (300s) |
| 27 | agents/multi_debate_panel | 4 (PASS) | 76.8s |  | 76.8s |
| 28 | agents/multi_tool_recovery | 4 (PASS) | 36.5s |  | 36.5s |
| 29 | agents/orchestrator | 4 (PASS) | 13.4s |  | 13.4s |
| 30 | agents/orchestrator_specialist | 4 (PASS) | 13.0s |  | 13.0s |
| 31 | agents/pipeline_review | 4 (PASS) | 107.2s |  | 107.2s |
| 32 | agents/plan_execute | 4 (PASS) | 36.5s |  | 36.5s |
| 33 | agents/plan_execute_recovery | 4 (PASS) | 38.2s |  | 38.2s |
| 34 | agents/prompt_chain | 4 (PASS) | 43.0s |  | 43.0s |
| 35 | agents/react_hitl_combined | 4 (PASS) | 214.0s |  | 214.0s |
| 36 | agents/react_search | 4 (PASS) | 28.3s |  | 28.3s |
| 37 | agents/react_structured_pipeline | 1 (BROKEN) | 300.1s | F-LOOP | Timeout (300s) |
| 38 | agents/reasoning_stacking | 4 (PASS) | 47.8s |  | 47.8s |
| 39 | agents/reasoning_tool | 4 (PASS) | 28.1s |  | 28.1s |
| 40 | agents/reflexion | 4 (PASS) | 141.2s |  | 141.2s |
| 41 | agents/reflexion_code_gen | 4 (PASS) | 34.8s |  | 34.8s |
| 42 | agents/rewoo | 4 (PASS) | 26.8s |  | 26.8s |
| 43 | agents/rewoo_multi_step | 4 (PASS) | 27.5s |  | 27.5s |
| 44 | agents/self_consistency | 4 (PASS) | 35.8s |  | 35.8s |
| 45 | agents/skill_loader | 4 (PASS) | 82.1s |  | 82.1s |
| 46 | agents/structured_output | 4 (PASS) | 80.6s |  | 80.6s |
| 47 | agents/tool_decorator | 4 (PASS) | 48.3s |  | 48.3s |
| 48 | agents/workflow_agent | 4 (PASS) | 8.6s |  | 8.6s |
| 49 | basic/form_filling | 4 (PASS) | 20.8s |  | 20.8s |
| 50 | basic/multi_turn_extraction | 4 (PASS) | 44.5s |  | 44.5s |
| 51 | basic/simple_greeting | 4 (PASS) | 20.6s |  | 20.6s |
| 52 | basic/story_time | 4 (PASS) | 31.6s |  | 31.6s |
| 53 | classification/classified_transitions | 4 (PASS) | 28.5s |  | 28.5s |
| 54 | classification/intent_routing | 4 (PASS) | 13.1s |  | 13.1s |
| 55 | classification/multi_intent | 4 (PASS) | 31.3s |  | 31.3s |
| 56 | classification/smart_helpdesk | 4 (PASS) | 21.1s |  | 21.1s |
| 57 | intermediate/adaptive_quiz | 4 (PASS) | 25.2s |  | 25.2s |
| 58 | intermediate/book_recommendation | 4 (PASS) | 24.4s |  | 24.4s |
| 59 | intermediate/product_recommendation | 4 (PASS) | 17.2s |  | 17.2s |
| 60 | meta/build_agent | 4 (PASS) | 8.2s |  | 8.2s |
| 61 | meta/build_fsm | 4 (PASS) | 6.4s |  | 6.4s |
| 62 | meta/build_workflow | 4 (PASS) | 7.5s |  | 7.5s |
| 63 | meta/meta_from_spec | 4 (PASS) | 3.6s |  | 3.6s |
| 64 | meta/meta_review_loop | 4 (PASS) | 37.2s |  | 37.2s |
| 65 | reasoning/math_tutor | 4 (PASS) | 11.0s |  | 11.0s |
| 66 | workflows/agent_workflow_chain | 4 (PASS) | 12.3s |  | 12.3s |
| 67 | workflows/conditional_branching | 4 (PASS) | 3.5s |  | 3.5s |
| 68 | workflows/order_processing | 4 (PASS) | 37.1s |  | 37.1s |
| 69 | workflows/parallel_steps | 4 (PASS) | 5.2s |  | 5.2s |
| 70 | workflows/workflow_agent_loop | 4 (PASS) | 66.3s |  | 66.3s |

## Summary

- **Total examples**: 70
- **Score distribution**: 66x4, 0x3, 0x2, 4x1, 0x0
- **Health Score**: 268/280 = **95.7%**
- **Category breakdown**:
  - advanced: 28/28 (100%)
  - agents: 152/164 (93%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 20/20 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 20/20 (100%)
- **Top failure codes**: F-LOOP (4)

## Timing

- **Total wall time**: 4902.7s (sequential equivalent)
- **Fastest**: 3.5s
- **Slowest**: 300.1s
- **Mean**: 70.0s
