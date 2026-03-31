# Evaluation: 2026-03-30 15:25

- **Date**: 2026-03-30 15:25
- **Git commit**: 7a69714
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 80
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 24.6s |  | 24.6s |
| 2 | advanced/context_compactor | 4 (PASS) | 26.5s |  | 26.5s |
| 3 | advanced/e_commerce | 4 (PASS) | 159.1s |  | 159.1s |
| 4 | advanced/handler_hooks | 4 (PASS) | 17.3s |  | 17.3s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 26.8s |  | 26.8s |
| 6 | advanced/support_pipeline | 4 (PASS) | 45.4s |  | 45.4s |
| 7 | advanced/yoga_instructions | 4 (PASS) | 23.1s |  | 23.1s |
| 8 | agents/adapt | 4 (PASS) | 136.6s |  | 136.6s |
| 9 | agents/adapt_with_memory | 4 (PASS) | 216.6s |  | 216.6s |
| 10 | agents/agent_as_tool | 4 (PASS) | 169.5s |  | 169.5s |
| 11 | agents/agent_memory_chain | 4 (PASS) | 282.4s |  | 282.4s |
| 12 | agents/architecture_review | 4 (PASS) | 67.7s |  | 67.7s |
| 13 | agents/classified_dispatch | 4 (PASS) | 45.4s |  | 45.4s |
| 14 | agents/classified_tools | 4 (PASS) | 41.0s |  | 41.0s |
| 15 | agents/concurrent_react | 4 (PASS) | 84.6s |  | 84.6s |
| 16 | agents/consistency_with_tools | 4 (PASS) | 29.1s |  | 29.1s |
| 17 | agents/debate | 4 (PASS) | 92.9s |  | 92.9s |
| 18 | agents/debate_with_tools | 4 (PASS) | 102.6s |  | 102.6s |
| 19 | agents/eval_opt_structured | 4 (PASS) | 79.1s |  | 79.1s |
| 20 | agents/evaluator_optimizer | 4 (PASS) | 108.8s |  | 108.8s |
| 21 | agents/full_pipeline | 4 (PASS) | 49.1s |  | 49.1s |
| 22 | agents/hierarchical_orchestrator | 4 (PASS) | 15.9s |  | 15.9s |
| 23 | agents/hierarchical_tools | 4 (PASS) | 46.3s |  | 46.3s |
| 24 | agents/hitl_approval | 4 (PASS) | 66.0s |  | 66.0s |
| 25 | agents/investment_portfolio | 4 (PASS) | 35.6s |  | 35.6s |
| 26 | agents/legal_document_review | 4 (PASS) | 255.2s |  | 255.2s |
| 27 | agents/maker_checker | 4 (PASS) | 79.3s |  | 79.3s |
| 28 | agents/maker_checker_code | 4 (PASS) | 129.5s |  | 129.5s |
| 29 | agents/medical_literature | 4 (PASS) | 213.1s |  | 213.1s |
| 30 | agents/memory_agent | 1 (BROKEN) | 300.0s | F-LOOP | Timeout (300s) |
| 31 | agents/multi_debate_panel | 4 (PASS) | 88.6s |  | 88.6s |
| 32 | agents/multi_tool_recovery | 3 (MOSTLY) | 126.1s | F-LOOP | 126.1s |
| 33 | agents/orchestrator | 4 (PASS) | 23.8s |  | 23.8s |
| 34 | agents/orchestrator_specialist | 4 (PASS) | 20.0s |  | 20.0s |
| 35 | agents/pipeline_review | 4 (PASS) | 109.2s |  | 109.2s |
| 36 | agents/plan_execute | 4 (PASS) | 58.8s |  | 58.8s |
| 37 | agents/plan_execute_recovery | 4 (PASS) | 27.3s |  | 27.3s |
| 38 | agents/prompt_chain | 4 (PASS) | 39.6s |  | 39.6s |
| 39 | agents/react_hitl_combined | 4 (PASS) | 117.5s |  | 117.5s |
| 40 | agents/react_search | 4 (PASS) | 66.5s |  | 66.5s |
| 41 | agents/react_structured_pipeline | 4 (PASS) | 117.1s |  | 117.1s |
| 42 | agents/reasoning_stacking | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 43 | agents/reasoning_tool | 4 (PASS) | 64.9s |  | 64.9s |
| 44 | agents/reflexion | 4 (PASS) | 179.3s |  | 179.3s |
| 45 | agents/reflexion_code_gen | 4 (PASS) | 68.4s |  | 68.4s |
| 46 | agents/regulatory_compliance | 4 (PASS) | 84.8s |  | 84.8s |
| 47 | agents/rewoo | 4 (PASS) | 24.3s |  | 24.3s |
| 48 | agents/rewoo_multi_step | 4 (PASS) | 23.3s |  | 23.3s |
| 49 | agents/security_audit | 4 (PASS) | 109.5s |  | 109.5s |
| 50 | agents/self_consistency | 4 (PASS) | 26.5s |  | 26.5s |
| 51 | agents/skill_loader | 4 (PASS) | 110.6s |  | 110.6s |
| 52 | agents/structured_output | 3 (MOSTLY) | 146.4s | F-LOOP | 146.4s |
| 53 | agents/supply_chain_optimizer | 4 (PASS) | 23.5s |  | 23.5s |
| 54 | agents/tool_decorator | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 55 | agents/workflow_agent | 4 (PASS) | 8.0s |  | 8.0s |
| 56 | basic/form_filling | 4 (PASS) | 20.5s |  | 20.5s |
| 57 | basic/multi_turn_extraction | 4 (PASS) | 36.7s |  | 36.7s |
| 58 | basic/simple_greeting | 4 (PASS) | 20.9s |  | 20.9s |
| 59 | basic/story_time | 4 (PASS) | 36.2s |  | 36.2s |
| 60 | classification/classified_transitions | 4 (PASS) | 29.3s |  | 29.3s |
| 61 | classification/intent_routing | 4 (PASS) | 13.4s |  | 13.4s |
| 62 | classification/multi_intent | 4 (PASS) | 32.9s |  | 32.9s |
| 63 | classification/smart_helpdesk | 4 (PASS) | 25.2s |  | 25.2s |
| 64 | intermediate/adaptive_quiz | 4 (PASS) | 26.1s |  | 26.1s |
| 65 | intermediate/book_recommendation | 4 (PASS) | 20.7s |  | 20.7s |
| 66 | intermediate/product_recommendation | 4 (PASS) | 19.7s |  | 19.7s |
| 67 | meta/build_agent | 4 (PASS) | 8.6s |  | 8.6s |
| 68 | meta/build_fsm | 4 (PASS) | 10.1s |  | 10.1s |
| 69 | meta/build_workflow | 4 (PASS) | 7.6s |  | 7.6s |
| 70 | meta/meta_from_spec | 4 (PASS) | 207.2s |  | 207.2s |
| 71 | meta/meta_review_loop | 4 (PASS) | 166.1s |  | 166.1s |
| 72 | reasoning/math_tutor | 4 (PASS) | 37.6s |  | 37.6s |
| 73 | workflows/agent_workflow_chain | 4 (PASS) | 37.3s |  | 37.3s |
| 74 | workflows/conditional_branching | 4 (PASS) | 3.9s |  | 3.9s |
| 75 | workflows/customer_onboarding | 4 (PASS) | 82.8s |  | 82.8s |
| 76 | workflows/loan_processing | 4 (PASS) | 48.9s |  | 48.9s |
| 77 | workflows/order_processing | 4 (PASS) | 91.4s |  | 91.4s |
| 78 | workflows/parallel_steps | 4 (PASS) | 5.3s |  | 5.3s |
| 79 | workflows/release_management | 4 (PASS) | 87.5s |  | 87.5s |
| 80 | workflows/workflow_agent_loop | 4 (PASS) | 66.2s |  | 66.2s |

## Summary

- **Total examples**: 80
- **Score distribution**: 75x4, 2x3, 0x2, 3x1, 0x0
- **Health Score**: 309/320 = **96.6%**
- **Category breakdown**:
  - advanced: 28/28 (100%)
  - agents: 181/192 (94%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 20/20 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 32/32 (100%)
- **Top failure codes**: F-LOOP (5)

## Timing

- **Total wall time**: 6135.7s (sequential equivalent)
- **Fastest**: 3.9s
- **Slowest**: 300.0s
- **Mean**: 76.7s
