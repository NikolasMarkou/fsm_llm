# Evaluation: 2026-03-29 20:41

- **Date**: 2026-03-29 20:41
- **Git commit**: 7d54bad
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 70
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 26.2s |  | 26.2s |
| 2 | advanced/context_compactor | 4 (PASS) | 27.4s |  | 27.4s |
| 3 | advanced/e_commerce | 4 (PASS) | 165.4s |  | 165.4s |
| 4 | advanced/handler_hooks | 4 (PASS) | 16.8s |  | 16.8s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 25.4s |  | 25.4s |
| 6 | advanced/support_pipeline | 4 (PASS) | 46.7s |  | 46.7s |
| 7 | advanced/yoga_instructions | 4 (PASS) | 19.3s |  | 19.3s |
| 8 | agents/adapt | 4 (PASS) | 144.7s |  | 144.7s |
| 9 | agents/adapt_with_memory | 4 (PASS) | 152.7s |  | 152.7s |
| 10 | agents/agent_as_tool | 4 (PASS) | 227.4s |  | 227.4s |
| 11 | agents/agent_memory_chain | 1 (BROKEN) | 300.0s | F-LOOP | Timeout (300s) |
| 12 | agents/classified_dispatch | 4 (PASS) | 133.8s |  | 133.8s |
| 13 | agents/classified_tools | 4 (PASS) | 59.6s |  | 59.6s |
| 14 | agents/concurrent_react | 4 (PASS) | 153.0s |  | 153.0s |
| 15 | agents/consistency_with_tools | 4 (PASS) | 29.2s |  | 29.2s |
| 16 | agents/debate | 4 (PASS) | 103.2s |  | 103.2s |
| 17 | agents/debate_with_tools | 4 (PASS) | 3.3s |  | 3.3s |
| 18 | agents/eval_opt_structured | 1 (BROKEN) | 300.1s | F-LOOP | Timeout (300s) |
| 19 | agents/evaluator_optimizer | 4 (PASS) | 82.9s |  | 82.9s |
| 20 | agents/full_pipeline | 4 (PASS) | 146.3s |  | 146.3s |
| 21 | agents/hierarchical_orchestrator | 4 (PASS) | 23.4s |  | 23.4s |
| 22 | agents/hierarchical_tools | 4 (PASS) | 119.7s |  | 119.7s |
| 23 | agents/hitl_approval | 4 (PASS) | 114.8s |  | 114.8s |
| 24 | agents/maker_checker | 3 (MOSTLY) | 144.0s | F-EXTRACT | 144.0s |
| 25 | agents/maker_checker_code | 4 (PASS) | 125.6s |  | 125.6s |
| 26 | agents/memory_agent | 4 (PASS) | 289.1s |  | 289.1s |
| 27 | agents/multi_debate_panel | 4 (PASS) | 76.1s |  | 76.1s |
| 28 | agents/multi_tool_recovery | 4 (PASS) | 129.2s |  | 129.2s |
| 29 | agents/orchestrator | 4 (PASS) | 15.3s |  | 15.3s |
| 30 | agents/orchestrator_specialist | 4 (PASS) | 16.5s |  | 16.5s |
| 31 | agents/pipeline_review | 4 (PASS) | 99.4s |  | 99.4s |
| 32 | agents/plan_execute | 4 (PASS) | 41.1s |  | 41.1s |
| 33 | agents/plan_execute_recovery | 4 (PASS) | 32.4s |  | 32.4s |
| 34 | agents/prompt_chain | 4 (PASS) | 33.0s |  | 33.0s |
| 35 | agents/react_hitl_combined | 1 (BROKEN) | 240.1s | F-LOOP | Timeout (240s) |
| 36 | agents/react_search | 4 (PASS) | 131.2s |  | 131.2s |
| 37 | agents/react_structured_pipeline | 1 (BROKEN) | 300.1s | F-LOOP | Timeout (300s) |
| 38 | agents/reasoning_stacking | 4 (PASS) | 74.1s |  | 74.1s |
| 39 | agents/reasoning_tool | 4 (PASS) | 29.4s |  | 29.4s |
| 40 | agents/reflexion | 4 (PASS) | 80.2s |  | 80.2s |
| 41 | agents/reflexion_code_gen | 4 (PASS) | 58.5s |  | 58.5s |
| 42 | agents/rewoo | 4 (PASS) | 23.4s |  | 23.4s |
| 43 | agents/rewoo_multi_step | 4 (PASS) | 22.2s |  | 22.2s |
| 44 | agents/self_consistency | 4 (PASS) | 23.9s |  | 23.9s |
| 45 | agents/skill_loader | 4 (PASS) | 116.1s |  | 116.1s |
| 46 | agents/structured_output | 4 (PASS) | 80.8s |  | 80.8s |
| 47 | agents/tool_decorator | 4 (PASS) | 121.2s |  | 121.2s |
| 48 | agents/workflow_agent | 4 (PASS) | 9.4s |  | 9.4s |
| 49 | basic/form_filling | 4 (PASS) | 19.0s |  | 19.0s |
| 50 | basic/multi_turn_extraction | 4 (PASS) | 34.8s |  | 34.8s |
| 51 | basic/simple_greeting | 4 (PASS) | 20.0s |  | 20.0s |
| 52 | basic/story_time | 4 (PASS) | 33.3s |  | 33.3s |
| 53 | classification/classified_transitions | 4 (PASS) | 26.8s |  | 26.8s |
| 54 | classification/intent_routing | 4 (PASS) | 15.1s |  | 15.1s |
| 55 | classification/multi_intent | 4 (PASS) | 33.8s |  | 33.8s |
| 56 | classification/smart_helpdesk | 4 (PASS) | 24.2s |  | 24.2s |
| 57 | intermediate/adaptive_quiz | 4 (PASS) | 22.9s |  | 22.9s |
| 58 | intermediate/book_recommendation | 4 (PASS) | 23.9s |  | 23.9s |
| 59 | intermediate/product_recommendation | 4 (PASS) | 15.7s |  | 15.7s |
| 60 | meta/build_agent | 4 (PASS) | 6.2s |  | 6.2s |
| 61 | meta/build_fsm | 4 (PASS) | 6.8s |  | 6.8s |
| 62 | meta/build_workflow | 4 (PASS) | 6.0s |  | 6.0s |
| 63 | meta/meta_from_spec | 4 (PASS) | 3.5s |  | 3.5s |
| 64 | meta/meta_review_loop | 4 (PASS) | 31.2s |  | 31.2s |
| 65 | reasoning/math_tutor | 4 (PASS) | 10.3s |  | 10.3s |
| 66 | workflows/agent_workflow_chain | 4 (PASS) | 10.8s |  | 10.8s |
| 67 | workflows/conditional_branching | 4 (PASS) | 3.6s |  | 3.6s |
| 68 | workflows/order_processing | 4 (PASS) | 33.7s |  | 33.7s |
| 69 | workflows/parallel_steps | 4 (PASS) | 5.2s |  | 5.2s |
| 70 | workflows/workflow_agent_loop | 4 (PASS) | 76.0s |  | 76.0s |

## Summary

- **Total examples**: 70
- **Score distribution**: 65x4, 1x3, 0x2, 4x1, 0x0
- **Health Score**: 267/280 = **95.4%**
- **Category breakdown**:
  - advanced: 28/28 (100%)
  - agents: 151/164 (92%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 20/20 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 20/20 (100%)
- **Top failure codes**: F-LOOP (4), F-EXTRACT (1)

## Timing

- **Total wall time**: 5196.3s (sequential equivalent)
- **Fastest**: 3.3s
- **Slowest**: 300.1s
- **Mean**: 74.2s
