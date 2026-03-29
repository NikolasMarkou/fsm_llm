# Evaluation: 2026-03-29 18:36

- **Date**: 2026-03-29 18:36
- **Git commit**: 4eb7e3c
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 70
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 27.1s |  | 27.1s |
| 2 | advanced/context_compactor | 4 (PASS) | 25.5s |  | 25.5s |
| 3 | advanced/e_commerce | 4 (PASS) | 163.3s |  | 163.3s |
| 4 | advanced/handler_hooks | 4 (PASS) | 17.7s |  | 17.7s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 22.9s |  | 22.9s |
| 6 | advanced/support_pipeline | 4 (PASS) | 47.7s |  | 47.7s |
| 7 | advanced/yoga_instructions | 4 (PASS) | 20.9s |  | 20.9s |
| 8 | agents/adapt | 4 (PASS) | 131.8s |  | 131.8s |
| 9 | agents/adapt_with_memory | 4 (PASS) | 152.1s |  | 152.1s |
| 10 | agents/agent_as_tool | 4 (PASS) | 238.2s |  | 238.2s |
| 11 | agents/agent_memory_chain | 1 (BROKEN) | 300.0s | F-LOOP | Timeout (300s) |
| 12 | agents/classified_dispatch | 4 (PASS) | 81.7s |  | 81.7s |
| 13 | agents/classified_tools | 4 (PASS) | 46.8s |  | 46.8s |
| 14 | agents/concurrent_react | 4 (PASS) | 127.6s |  | 127.6s |
| 15 | agents/consistency_with_tools | 3 (MOSTLY) | 3.6s | F-CODE | 3.6s |
| 16 | agents/debate | 4 (PASS) | 85.2s |  | 85.2s |
| 17 | agents/debate_with_tools | 3 (MOSTLY) | 3.6s | F-CODE | 3.6s |
| 18 | agents/eval_opt_structured | 4 (PASS) | 56.4s |  | 56.4s |
| 19 | agents/evaluator_optimizer | 4 (PASS) | 87.2s |  | 87.2s |
| 20 | agents/full_pipeline | 4 (PASS) | 128.5s |  | 128.5s |
| 21 | agents/hierarchical_orchestrator | 4 (PASS) | 13.9s |  | 13.9s |
| 22 | agents/hierarchical_tools | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 23 | agents/hitl_approval | 4 (PASS) | 61.1s |  | 61.1s |
| 24 | agents/maker_checker | 4 (PASS) | 42.6s |  | 42.6s |
| 25 | agents/maker_checker_code | 3 (MOSTLY) | 125.0s | F-EXTRACT | 125.0s |
| 26 | agents/memory_agent | 1 (BROKEN) | 300.1s | F-LOOP | Timeout (300s) |
| 27 | agents/multi_debate_panel | 4 (PASS) | 69.1s |  | 69.1s |
| 28 | agents/multi_tool_recovery | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 29 | agents/orchestrator | 4 (PASS) | 12.1s |  | 12.1s |
| 30 | agents/orchestrator_specialist | 4 (PASS) | 19.0s |  | 19.0s |
| 31 | agents/pipeline_review | 3 (MOSTLY) | 116.3s | F-EXTRACT | 116.3s |
| 32 | agents/plan_execute | 4 (PASS) | 52.2s |  | 52.2s |
| 33 | agents/plan_execute_recovery | 4 (PASS) | 25.3s |  | 25.3s |
| 34 | agents/prompt_chain | 4 (PASS) | 36.1s |  | 36.1s |
| 35 | agents/react_hitl_combined | 4 (PASS) | 225.2s |  | 225.2s |
| 36 | agents/react_search | 4 (PASS) | 71.9s |  | 71.9s |
| 37 | agents/react_structured_pipeline | 2 (PARTIAL) | 127.0s | F-CODE | Exit 1 |
| 38 | agents/reasoning_stacking | 4 (PASS) | 25.0s |  | 25.0s |
| 39 | agents/reasoning_tool | 4 (PASS) | 51.7s |  | 51.7s |
| 40 | agents/reflexion | 4 (PASS) | 124.5s |  | 124.5s |
| 41 | agents/reflexion_code_gen | 4 (PASS) | 177.7s |  | 177.7s |
| 42 | agents/rewoo | 4 (PASS) | 18.9s |  | 18.9s |
| 43 | agents/rewoo_multi_step | 4 (PASS) | 23.8s |  | 23.8s |
| 44 | agents/self_consistency | 4 (PASS) | 29.8s |  | 29.8s |
| 45 | agents/skill_loader | 4 (PASS) | 76.3s |  | 76.3s |
| 46 | agents/structured_output | 4 (PASS) | 140.1s |  | 140.1s |
| 47 | agents/tool_decorator | 4 (PASS) | 72.9s |  | 72.9s |
| 48 | agents/workflow_agent | 4 (PASS) | 7.1s |  | 7.1s |
| 49 | basic/form_filling | 4 (PASS) | 20.0s |  | 20.0s |
| 50 | basic/multi_turn_extraction | 4 (PASS) | 37.1s |  | 37.1s |
| 51 | basic/simple_greeting | 4 (PASS) | 18.3s |  | 18.3s |
| 52 | basic/story_time | 4 (PASS) | 31.2s |  | 31.2s |
| 53 | classification/classified_transitions | 4 (PASS) | 30.5s |  | 30.5s |
| 54 | classification/intent_routing | 4 (PASS) | 14.7s |  | 14.7s |
| 55 | classification/multi_intent | 4 (PASS) | 30.6s |  | 30.6s |
| 56 | classification/smart_helpdesk | 4 (PASS) | 21.7s |  | 21.7s |
| 57 | intermediate/adaptive_quiz | 4 (PASS) | 23.2s |  | 23.2s |
| 58 | intermediate/book_recommendation | 4 (PASS) | 21.7s |  | 21.7s |
| 59 | intermediate/product_recommendation | 4 (PASS) | 15.6s |  | 15.6s |
| 60 | meta/build_agent | 4 (PASS) | 6.8s |  | 6.8s |
| 61 | meta/build_fsm | 4 (PASS) | 8.0s |  | 8.0s |
| 62 | meta/build_workflow | 4 (PASS) | 6.3s |  | 6.3s |
| 63 | meta/meta_from_spec | 4 (PASS) | 3.4s |  | 3.4s |
| 64 | meta/meta_review_loop | 3 (MOSTLY) | 33.0s | F-EXTRACT | 33.0s |
| 65 | reasoning/math_tutor | 4 (PASS) | 11.2s |  | 11.2s |
| 66 | workflows/agent_workflow_chain | 4 (PASS) | 10.9s |  | 10.9s |
| 67 | workflows/conditional_branching | 4 (PASS) | 3.6s |  | 3.6s |
| 68 | workflows/order_processing | 4 (PASS) | 30.3s |  | 30.3s |
| 69 | workflows/parallel_steps | 4 (PASS) | 5.2s |  | 5.2s |
| 70 | workflows/workflow_agent_loop | 1 (BROKEN) | 3.6s | F-CODE | Exit 1 |

## Summary

- **Total examples**: 70
- **Score distribution**: 59x4, 5x3, 1x2, 5x1, 0x0
- **Health Score**: 258/280 = **92.1%**
- **Category breakdown**:
  - advanced: 28/28 (100%)
  - agents: 146/164 (89%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 19/20 (95%)
  - reasoning: 4/4 (100%)
  - workflows: 17/20 (85%)
- **Top failure codes**: F-LOOP (4), F-CODE (4), F-EXTRACT (3)

## Timing

- **Total wall time**: 4559.5s (sequential equivalent)
- **Fastest**: 3.4s
- **Slowest**: 300.1s
- **Mean**: 65.1s
