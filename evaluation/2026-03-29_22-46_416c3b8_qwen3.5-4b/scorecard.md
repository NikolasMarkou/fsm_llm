# Evaluation: 2026-03-29 23:00

- **Date**: 2026-03-29 23:00
- **Git commit**: 416c3b8
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 80
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 26.3s |  | 26.3s |
| 2 | advanced/context_compactor | 4 (PASS) | 24.0s |  | 24.0s |
| 3 | advanced/e_commerce | 4 (PASS) | 155.2s |  | 155.2s |
| 4 | advanced/handler_hooks | 4 (PASS) | 18.4s |  | 18.4s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 20.7s |  | 20.7s |
| 6 | advanced/support_pipeline | 4 (PASS) | 47.0s |  | 47.0s |
| 7 | advanced/yoga_instructions | 4 (PASS) | 21.2s |  | 21.2s |
| 8 | agents/adapt | 4 (PASS) | 140.9s |  | 140.9s |
| 9 | agents/adapt_with_memory | 4 (PASS) | 170.3s |  | 170.3s |
| 10 | agents/agent_as_tool | 4 (PASS) | 49.0s |  | 49.0s |
| 11 | agents/agent_memory_chain | 4 (PASS) | 94.1s |  | 94.1s |
| 12 | agents/architecture_review | 4 (PASS) | 51.7s |  | 51.7s |
| 13 | agents/classified_dispatch | 4 (PASS) | 46.0s |  | 46.0s |
| 14 | agents/classified_tools | 4 (PASS) | 35.9s |  | 35.9s |
| 15 | agents/concurrent_react | 4 (PASS) | 35.9s |  | 35.9s |
| 16 | agents/consistency_with_tools | 4 (PASS) | 27.6s |  | 27.6s |
| 17 | agents/debate | 4 (PASS) | 83.8s |  | 83.8s |
| 18 | agents/debate_with_tools | 4 (PASS) | 3.4s |  | 3.4s |
| 19 | agents/eval_opt_structured | 4 (PASS) | 114.2s |  | 114.2s |
| 20 | agents/evaluator_optimizer | 4 (PASS) | 41.1s |  | 41.1s |
| 21 | agents/full_pipeline | 4 (PASS) | 48.3s |  | 48.3s |
| 22 | agents/hierarchical_orchestrator | 4 (PASS) | 25.8s |  | 25.8s |
| 23 | agents/hierarchical_tools | 4 (PASS) | 38.5s |  | 38.5s |
| 24 | agents/hitl_approval | 4 (PASS) | 37.9s |  | 37.9s |
| 25 | agents/investment_portfolio | 4 (PASS) | 28.3s |  | 28.3s |
| 26 | agents/legal_document_review | 4 (PASS) | 77.8s |  | 77.8s |
| 27 | agents/maker_checker | 4 (PASS) | 13.0s |  | 13.0s |
| 28 | agents/maker_checker_code | 4 (PASS) | 16.7s |  | 16.7s |
| 29 | agents/medical_literature | 4 (PASS) | 294.0s |  | 294.0s |
| 30 | agents/memory_agent | 4 (PASS) | 153.4s |  | 153.4s |
| 31 | agents/multi_debate_panel | 4 (PASS) | 90.4s |  | 90.4s |
| 32 | agents/multi_tool_recovery | 4 (PASS) | 52.5s |  | 52.5s |
| 33 | agents/orchestrator | 4 (PASS) | 21.0s |  | 21.0s |
| 34 | agents/orchestrator_specialist | 4 (PASS) | 16.5s |  | 16.5s |
| 35 | agents/pipeline_review | 4 (PASS) | 43.6s |  | 43.6s |
| 36 | agents/plan_execute | 4 (PASS) | 34.1s |  | 34.1s |
| 37 | agents/plan_execute_recovery | 4 (PASS) | 40.9s |  | 40.9s |
| 38 | agents/prompt_chain | 4 (PASS) | 40.9s |  | 40.9s |
| 39 | agents/react_hitl_combined | 4 (PASS) | 69.1s |  | 69.1s |
| 40 | agents/react_search | 4 (PASS) | 36.1s |  | 36.1s |
| 41 | agents/react_structured_pipeline | 4 (PASS) | 24.8s |  | 24.8s |
| 42 | agents/reasoning_stacking | 4 (PASS) | 26.6s |  | 26.6s |
| 43 | agents/reasoning_tool | 4 (PASS) | 24.8s |  | 24.8s |
| 44 | agents/reflexion | 4 (PASS) | 24.5s |  | 24.5s |
| 45 | agents/reflexion_code_gen | 4 (PASS) | 42.5s |  | 42.5s |
| 46 | agents/regulatory_compliance | 4 (PASS) | 17.4s |  | 17.4s |
| 47 | agents/rewoo | 4 (PASS) | 35.9s |  | 35.9s |
| 48 | agents/rewoo_multi_step | 4 (PASS) | 23.8s |  | 23.8s |
| 49 | agents/security_audit | 4 (PASS) | 20.5s |  | 20.5s |
| 50 | agents/self_consistency | 4 (PASS) | 21.5s |  | 21.5s |
| 51 | agents/skill_loader | 4 (PASS) | 18.9s |  | 18.9s |
| 52 | agents/structured_output | 4 (PASS) | 18.1s |  | 18.1s |
| 53 | agents/supply_chain_optimizer | 4 (PASS) | 18.1s |  | 18.1s |
| 54 | agents/tool_decorator | 4 (PASS) | 20.8s |  | 20.8s |
| 55 | agents/workflow_agent | 4 (PASS) | 12.9s |  | 12.9s |
| 56 | basic/form_filling | 4 (PASS) | 22.2s |  | 22.2s |
| 57 | basic/multi_turn_extraction | 4 (PASS) | 38.9s |  | 38.9s |
| 58 | basic/simple_greeting | 4 (PASS) | 14.4s |  | 14.4s |
| 59 | basic/story_time | 4 (PASS) | 34.9s |  | 34.9s |
| 60 | classification/classified_transitions | 4 (PASS) | 30.3s |  | 30.3s |
| 61 | classification/intent_routing | 4 (PASS) | 12.4s |  | 12.4s |
| 62 | classification/multi_intent | 4 (PASS) | 34.7s |  | 34.7s |
| 63 | classification/smart_helpdesk | 4 (PASS) | 22.9s |  | 22.9s |
| 64 | intermediate/adaptive_quiz | 4 (PASS) | 25.6s |  | 25.6s |
| 65 | intermediate/book_recommendation | 4 (PASS) | 25.3s |  | 25.3s |
| 66 | intermediate/product_recommendation | 4 (PASS) | 16.3s |  | 16.3s |
| 67 | meta/build_agent | 4 (PASS) | 7.0s |  | 7.0s |
| 68 | meta/build_fsm | 4 (PASS) | 7.5s |  | 7.5s |
| 69 | meta/build_workflow | 4 (PASS) | 7.3s |  | 7.3s |
| 70 | meta/meta_from_spec | 4 (PASS) | 3.5s |  | 3.5s |
| 71 | meta/meta_review_loop | 4 (PASS) | 7.5s |  | 7.5s |
| 72 | reasoning/math_tutor | 4 (PASS) | 10.5s |  | 10.5s |
| 73 | workflows/agent_workflow_chain | 4 (PASS) | 10.5s |  | 10.5s |
| 74 | workflows/conditional_branching | 4 (PASS) | 3.5s |  | 3.5s |
| 75 | workflows/customer_onboarding | 3 (MOSTLY) | 13.4s | F-EXTRACT | 13.4s |
| 76 | workflows/loan_processing | 4 (PASS) | 3.6s |  | 3.6s |
| 77 | workflows/order_processing | 4 (PASS) | 27.7s |  | 27.7s |
| 78 | workflows/parallel_steps | 4 (PASS) | 5.1s |  | 5.1s |
| 79 | workflows/release_management | 4 (PASS) | 3.6s |  | 3.6s |
| 80 | workflows/workflow_agent_loop | 4 (PASS) | 33.5s |  | 33.5s |

## Summary

- **Total examples**: 80
- **Score distribution**: 79x4, 1x3, 0x2, 0x1, 0x0
- **Health Score**: 319/320 = **99.7%**
- **Category breakdown**:
  - advanced: 28/28 (100%)
  - agents: 192/192 (100%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 20/20 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 31/32 (97%)
- **Top failure codes**: F-EXTRACT (1)

## Timing

- **Total wall time**: 3158.7s (sequential equivalent)
- **Fastest**: 3.4s
- **Slowest**: 294.0s
- **Mean**: 39.5s
