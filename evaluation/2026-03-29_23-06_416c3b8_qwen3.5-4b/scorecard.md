# Evaluation: 2026-03-29 23:29

- **Date**: 2026-03-29 23:29
- **Git commit**: 416c3b8
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 80
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 23.8s |  | 23.8s |
| 2 | advanced/context_compactor | 4 (PASS) | 22.9s |  | 22.9s |
| 3 | advanced/e_commerce | 4 (PASS) | 153.5s |  | 153.5s |
| 4 | advanced/handler_hooks | 4 (PASS) | 18.4s |  | 18.4s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 22.4s |  | 22.4s |
| 6 | advanced/support_pipeline | 4 (PASS) | 41.0s |  | 41.0s |
| 7 | advanced/yoga_instructions | 4 (PASS) | 18.6s |  | 18.6s |
| 8 | agents/adapt | 4 (PASS) | 165.1s |  | 165.1s |
| 9 | agents/adapt_with_memory | 4 (PASS) | 202.7s |  | 202.7s |
| 10 | agents/agent_as_tool | 4 (PASS) | 102.2s |  | 102.2s |
| 11 | agents/agent_memory_chain | 3 (MOSTLY) | 168.3s | F-LOOP | 168.3s |
| 12 | agents/architecture_review | 4 (PASS) | 76.4s |  | 76.4s |
| 13 | agents/classified_dispatch | 4 (PASS) | 87.4s |  | 87.4s |
| 14 | agents/classified_tools | 4 (PASS) | 75.8s |  | 75.8s |
| 15 | agents/concurrent_react | 3 (MOSTLY) | 149.1s | F-LOOP | 149.1s |
| 16 | agents/consistency_with_tools | 4 (PASS) | 24.0s |  | 24.0s |
| 17 | agents/debate | 4 (PASS) | 73.6s |  | 73.6s |
| 18 | agents/debate_with_tools | 4 (PASS) | 3.7s |  | 3.7s |
| 19 | agents/eval_opt_structured | 4 (PASS) | 153.4s |  | 153.4s |
| 20 | agents/evaluator_optimizer | 4 (PASS) | 86.7s |  | 86.7s |
| 21 | agents/full_pipeline | 4 (PASS) | 93.9s |  | 93.9s |
| 22 | agents/hierarchical_orchestrator | 4 (PASS) | 18.1s |  | 18.1s |
| 23 | agents/hierarchical_tools | 4 (PASS) | 50.9s |  | 50.9s |
| 24 | agents/hitl_approval | 4 (PASS) | 48.2s |  | 48.2s |
| 25 | agents/investment_portfolio | 4 (PASS) | 51.1s |  | 51.1s |
| 26 | agents/legal_document_review | 4 (PASS) | 236.6s |  | 236.6s |
| 27 | agents/maker_checker | 4 (PASS) | 111.5s |  | 111.5s |
| 28 | agents/maker_checker_code | 4 (PASS) | 96.6s |  | 96.6s |
| 29 | agents/medical_literature | 1 (BROKEN) | 300.0s | F-LOOP | Timeout (300s) |
| 30 | agents/memory_agent | 4 (PASS) | 291.9s |  | 291.9s |
| 31 | agents/multi_debate_panel | 4 (PASS) | 41.7s |  | 41.7s |
| 32 | agents/multi_tool_recovery | 4 (PASS) | 114.8s |  | 114.8s |
| 33 | agents/orchestrator | 4 (PASS) | 13.6s |  | 13.6s |
| 34 | agents/orchestrator_specialist | 4 (PASS) | 19.2s |  | 19.2s |
| 35 | agents/pipeline_review | 4 (PASS) | 116.9s |  | 116.9s |
| 36 | agents/plan_execute | 4 (PASS) | 62.4s |  | 62.4s |
| 37 | agents/plan_execute_recovery | 4 (PASS) | 38.4s |  | 38.4s |
| 38 | agents/prompt_chain | 4 (PASS) | 41.0s |  | 41.0s |
| 39 | agents/react_hitl_combined | 4 (PASS) | 185.7s |  | 185.7s |
| 40 | agents/react_search | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 41 | agents/react_structured_pipeline | 4 (PASS) | 146.8s |  | 146.8s |
| 42 | agents/reasoning_stacking | 4 (PASS) | 25.3s |  | 25.3s |
| 43 | agents/reasoning_tool | 4 (PASS) | 107.4s |  | 107.4s |
| 44 | agents/reflexion | 4 (PASS) | 180.3s |  | 180.3s |
| 45 | agents/reflexion_code_gen | 4 (PASS) | 43.6s |  | 43.6s |
| 46 | agents/regulatory_compliance | 4 (PASS) | 86.7s |  | 86.7s |
| 47 | agents/rewoo | 4 (PASS) | 22.5s |  | 22.5s |
| 48 | agents/rewoo_multi_step | 4 (PASS) | 35.2s |  | 35.2s |
| 49 | agents/security_audit | 4 (PASS) | 166.1s |  | 166.1s |
| 50 | agents/self_consistency | 4 (PASS) | 46.3s |  | 46.3s |
| 51 | agents/skill_loader | 4 (PASS) | 130.0s |  | 130.0s |
| 52 | agents/structured_output | 4 (PASS) | 59.2s |  | 59.2s |
| 53 | agents/supply_chain_optimizer | 4 (PASS) | 28.7s |  | 28.7s |
| 54 | agents/tool_decorator | 4 (PASS) | 43.6s |  | 43.6s |
| 55 | agents/workflow_agent | 4 (PASS) | 13.5s |  | 13.5s |
| 56 | basic/form_filling | 4 (PASS) | 17.1s |  | 17.1s |
| 57 | basic/multi_turn_extraction | 4 (PASS) | 33.5s |  | 33.5s |
| 58 | basic/simple_greeting | 4 (PASS) | 18.9s |  | 18.9s |
| 59 | basic/story_time | 4 (PASS) | 31.6s |  | 31.6s |
| 60 | classification/classified_transitions | 4 (PASS) | 25.8s |  | 25.8s |
| 61 | classification/intent_routing | 4 (PASS) | 11.6s |  | 11.6s |
| 62 | classification/multi_intent | 4 (PASS) | 27.0s |  | 27.0s |
| 63 | classification/smart_helpdesk | 4 (PASS) | 20.7s |  | 20.7s |
| 64 | intermediate/adaptive_quiz | 4 (PASS) | 21.0s |  | 21.0s |
| 65 | intermediate/book_recommendation | 4 (PASS) | 21.4s |  | 21.4s |
| 66 | intermediate/product_recommendation | 4 (PASS) | 16.7s |  | 16.7s |
| 67 | meta/build_agent | 4 (PASS) | 7.0s |  | 7.0s |
| 68 | meta/build_fsm | 4 (PASS) | 6.9s |  | 6.9s |
| 69 | meta/build_workflow | 4 (PASS) | 6.4s |  | 6.4s |
| 70 | meta/meta_from_spec | 4 (PASS) | 3.5s |  | 3.5s |
| 71 | meta/meta_review_loop | 4 (PASS) | 53.2s |  | 53.2s |
| 72 | reasoning/math_tutor | 4 (PASS) | 11.6s |  | 11.6s |
| 73 | workflows/agent_workflow_chain | 4 (PASS) | 11.5s |  | 11.5s |
| 74 | workflows/conditional_branching | 4 (PASS) | 3.6s |  | 3.6s |
| 75 | workflows/customer_onboarding | 4 (PASS) | 61.3s |  | 61.3s |
| 76 | workflows/loan_processing | 4 (PASS) | 3.5s |  | 3.5s |
| 77 | workflows/order_processing | 4 (PASS) | 59.9s |  | 59.9s |
| 78 | workflows/parallel_steps | 4 (PASS) | 5.2s |  | 5.2s |
| 79 | workflows/release_management | 4 (PASS) | 3.5s |  | 3.5s |
| 80 | workflows/workflow_agent_loop | 3 (MOSTLY) | 108.5s | F-LOOP | 108.5s |

## Summary

- **Total examples**: 80
- **Score distribution**: 75x4, 3x3, 0x2, 2x1, 0x0
- **Health Score**: 311/320 = **97.2%**
- **Category breakdown**:
  - advanced: 28/28 (100%)
  - agents: 184/192 (96%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 20/20 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 31/32 (97%)
- **Top failure codes**: F-LOOP (5)

## Timing

- **Total wall time**: 5507.3s (sequential equivalent)
- **Fastest**: 3.5s
- **Slowest**: 300.0s
- **Mean**: 68.8s
