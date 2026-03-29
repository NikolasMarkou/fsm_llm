# Evaluation: 2026-03-29 21:54

- **Date**: 2026-03-29 21:54
- **Git commit**: 7d54bad
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 70
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 29.0s |  | 29.0s |
| 2 | advanced/context_compactor | 4 (PASS) | 27.0s |  | 27.0s |
| 3 | advanced/e_commerce | 4 (PASS) | 158.6s |  | 158.6s |
| 4 | advanced/handler_hooks | 4 (PASS) | 19.1s |  | 19.1s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 21.8s |  | 21.8s |
| 6 | advanced/support_pipeline | 4 (PASS) | 43.3s |  | 43.3s |
| 7 | advanced/yoga_instructions | 4 (PASS) | 18.6s |  | 18.6s |
| 8 | agents/adapt | 4 (PASS) | 142.0s |  | 142.0s |
| 9 | agents/adapt_with_memory | 4 (PASS) | 173.4s |  | 173.4s |
| 10 | agents/agent_as_tool | 4 (PASS) | 54.8s |  | 54.8s |
| 11 | agents/agent_memory_chain | 4 (PASS) | 68.5s |  | 68.5s |
| 12 | agents/classified_dispatch | 4 (PASS) | 27.8s |  | 27.8s |
| 13 | agents/classified_tools | 4 (PASS) | 34.6s |  | 34.6s |
| 14 | agents/concurrent_react | 4 (PASS) | 29.6s |  | 29.6s |
| 15 | agents/consistency_with_tools | 4 (PASS) | 25.8s |  | 25.8s |
| 16 | agents/debate | 4 (PASS) | 80.4s |  | 80.4s |
| 17 | agents/debate_with_tools | 4 (PASS) | 3.6s |  | 3.6s |
| 18 | agents/eval_opt_structured | 4 (PASS) | 72.6s |  | 72.6s |
| 19 | agents/evaluator_optimizer | 4 (PASS) | 55.8s |  | 55.8s |
| 20 | agents/full_pipeline | 4 (PASS) | 48.0s |  | 48.0s |
| 21 | agents/hierarchical_orchestrator | 4 (PASS) | 24.0s |  | 24.0s |
| 22 | agents/hierarchical_tools | 4 (PASS) | 38.4s |  | 38.4s |
| 23 | agents/hitl_approval | 4 (PASS) | 34.1s |  | 34.1s |
| 24 | agents/maker_checker | 4 (PASS) | 9.3s |  | 9.3s |
| 25 | agents/maker_checker_code | 4 (PASS) | 13.1s |  | 13.1s |
| 26 | agents/memory_agent | 4 (PASS) | 153.9s |  | 153.9s |
| 27 | agents/multi_debate_panel | 4 (PASS) | 79.2s |  | 79.2s |
| 28 | agents/multi_tool_recovery | 4 (PASS) | 67.1s |  | 67.1s |
| 29 | agents/orchestrator | 4 (PASS) | 34.3s |  | 34.3s |
| 30 | agents/orchestrator_specialist | 4 (PASS) | 33.3s |  | 33.3s |
| 31 | agents/pipeline_review | 4 (PASS) | 72.9s |  | 72.9s |
| 32 | agents/plan_execute | 4 (PASS) | 62.8s |  | 62.8s |
| 33 | agents/plan_execute_recovery | 4 (PASS) | 59.4s |  | 59.4s |
| 34 | agents/prompt_chain | 4 (PASS) | 26.9s |  | 26.9s |
| 35 | agents/react_hitl_combined | 4 (PASS) | 41.9s |  | 41.9s |
| 36 | agents/react_search | 4 (PASS) | 27.6s |  | 27.6s |
| 37 | agents/react_structured_pipeline | 4 (PASS) | 28.5s |  | 28.5s |
| 38 | agents/reasoning_stacking | 4 (PASS) | 17.7s |  | 17.7s |
| 39 | agents/reasoning_tool | 4 (PASS) | 18.0s |  | 18.0s |
| 40 | agents/reflexion | 4 (PASS) | 18.6s |  | 18.6s |
| 41 | agents/reflexion_code_gen | 4 (PASS) | 20.5s |  | 20.5s |
| 42 | agents/rewoo | 4 (PASS) | 18.4s |  | 18.4s |
| 43 | agents/rewoo_multi_step | 4 (PASS) | 16.7s |  | 16.7s |
| 44 | agents/self_consistency | 4 (PASS) | 19.5s |  | 19.5s |
| 45 | agents/skill_loader | 4 (PASS) | 17.6s |  | 17.6s |
| 46 | agents/structured_output | 4 (PASS) | 17.7s |  | 17.7s |
| 47 | agents/tool_decorator | 4 (PASS) | 17.1s |  | 17.1s |
| 48 | agents/workflow_agent | 4 (PASS) | 7.6s |  | 7.6s |
| 49 | basic/form_filling | 4 (PASS) | 18.7s |  | 18.7s |
| 50 | basic/multi_turn_extraction | 4 (PASS) | 36.4s |  | 36.4s |
| 51 | basic/simple_greeting | 4 (PASS) | 19.8s |  | 19.8s |
| 52 | basic/story_time | 4 (PASS) | 30.0s |  | 30.0s |
| 53 | classification/classified_transitions | 4 (PASS) | 24.2s |  | 24.2s |
| 54 | classification/intent_routing | 4 (PASS) | 13.1s |  | 13.1s |
| 55 | classification/multi_intent | 4 (PASS) | 30.8s |  | 30.8s |
| 56 | classification/smart_helpdesk | 4 (PASS) | 18.7s |  | 18.7s |
| 57 | intermediate/adaptive_quiz | 4 (PASS) | 23.4s |  | 23.4s |
| 58 | intermediate/book_recommendation | 4 (PASS) | 23.8s |  | 23.8s |
| 59 | intermediate/product_recommendation | 4 (PASS) | 18.7s |  | 18.7s |
| 60 | meta/build_agent | 4 (PASS) | 8.3s |  | 8.3s |
| 61 | meta/build_fsm | 4 (PASS) | 6.7s |  | 6.7s |
| 62 | meta/build_workflow | 4 (PASS) | 7.4s |  | 7.4s |
| 63 | meta/meta_from_spec | 4 (PASS) | 3.4s |  | 3.4s |
| 64 | meta/meta_review_loop | 4 (PASS) | 6.3s |  | 6.3s |
| 65 | reasoning/math_tutor | 4 (PASS) | 9.1s |  | 9.1s |
| 66 | workflows/agent_workflow_chain | 4 (PASS) | 9.8s |  | 9.8s |
| 67 | workflows/conditional_branching | 4 (PASS) | 3.5s |  | 3.5s |
| 68 | workflows/order_processing | 4 (PASS) | 26.7s |  | 26.7s |
| 69 | workflows/parallel_steps | 4 (PASS) | 5.2s |  | 5.2s |
| 70 | workflows/workflow_agent_loop | 4 (PASS) | 32.6s |  | 32.6s |

## Summary

- **Total examples**: 70
- **Score distribution**: 70x4, 0x3, 0x2, 0x1, 0x0
- **Health Score**: 280/280 = **100.0%**
- **Category breakdown**:
  - advanced: 28/28 (100%)
  - agents: 164/164 (100%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 20/20 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 20/20 (100%)

## Timing

- **Total wall time**: 2506.9s (sequential equivalent)
- **Fastest**: 3.4s
- **Slowest**: 173.4s
- **Mean**: 35.8s
