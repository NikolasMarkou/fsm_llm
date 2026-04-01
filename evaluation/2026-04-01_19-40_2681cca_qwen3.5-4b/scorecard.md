# Evaluation: 2026-04-01 20:03

- **Date**: 2026-04-01 20:03
- **Git commit**: 2681cca
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 100
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/budget_review | 1 (BROKEN) | 44.4s | F-EXTRACT, F-TRANS | 44.4s |
| 2 | advanced/compliance_audit | 1 (BROKEN) | 40.4s | F-EXTRACT, F-TRANS | 40.4s |
| 3 | advanced/concurrent_conversations | 4 (PASS) | 46.6s |  | 46.6s |
| 4 | advanced/context_compactor | 4 (PASS) | 50.3s |  | 50.3s |
| 5 | advanced/customer_feedback_pipeline | 1 (BROKEN) | 29.6s | F-EXTRACT, F-TRANS | 29.6s |
| 6 | advanced/e_commerce | 4 (PASS) | 114.1s |  | 114.1s |
| 7 | advanced/employee_onboarding | 1 (BROKEN) | 33.3s | F-EXTRACT, F-TRANS | 33.3s |
| 8 | advanced/handler_hooks | 4 (PASS) | 25.4s |  | 25.4s |
| 9 | advanced/incident_response | 1 (BROKEN) | 45.3s | F-EXTRACT, F-TRANS | 45.3s |
| 10 | advanced/loan_assessment | 1 (BROKEN) | 48.4s | F-EXTRACT, F-TRANS | 48.4s |
| 11 | advanced/medical_triage | 1 (BROKEN) | 50.4s | F-EXTRACT, F-TRANS | 50.4s |
| 12 | advanced/multi_level_stack | 4 (PASS) | 36.6s |  | 36.6s |
| 13 | advanced/project_planning | 1 (BROKEN) | 41.5s | F-EXTRACT, F-TRANS | 41.5s |
| 14 | advanced/quality_inspection | 1 (BROKEN) | 37.6s | F-EXTRACT, F-TRANS | 37.6s |
| 15 | advanced/support_pipeline | 4 (PASS) | 59.9s |  | 59.9s |
| 16 | advanced/vendor_evaluation | 1 (BROKEN) | 40.0s | F-EXTRACT, F-TRANS | 40.0s |
| 17 | advanced/yoga_instructions | 4 (PASS) | 21.4s |  | 21.4s |
| 18 | agents/adapt | 4 (PASS) | 115.5s |  | 115.5s |
| 19 | agents/adapt_with_memory | 4 (PASS) | 126.7s |  | 126.7s |
| 20 | agents/agent_as_tool | 3 (MOSTLY) | 33.6s | F-LOOP | 33.6s |
| 21 | agents/agent_memory_chain | 4 (PASS) | 156.1s |  | 156.1s |
| 22 | agents/architecture_review | 4 (PASS) | 53.9s |  | 53.9s |
| 23 | agents/classified_dispatch | 4 (PASS) | 40.5s |  | 40.5s |
| 24 | agents/classified_tools | 4 (PASS) | 42.5s |  | 42.5s |
| 25 | agents/concurrent_react | 4 (PASS) | 102.2s |  | 102.2s |
| 26 | agents/consistency_with_tools | 4 (PASS) | 33.4s |  | 33.4s |
| 27 | agents/debate | 4 (PASS) | 133.2s |  | 133.2s |
| 28 | agents/debate_with_tools | 4 (PASS) | 126.6s |  | 126.6s |
| 29 | agents/eval_opt_structured | 4 (PASS) | 82.9s |  | 82.9s |
| 30 | agents/evaluator_optimizer | 4 (PASS) | 80.0s |  | 80.0s |
| 31 | agents/full_pipeline | 4 (PASS) | 63.2s |  | 63.2s |
| 32 | agents/hierarchical_orchestrator | 4 (PASS) | 19.8s |  | 19.8s |
| 33 | agents/hierarchical_tools | 4 (PASS) | 34.2s |  | 34.2s |
| 34 | agents/hitl_approval | 4 (PASS) | 70.7s |  | 70.7s |
| 35 | agents/investment_portfolio | 4 (PASS) | 30.9s |  | 30.9s |
| 36 | agents/legal_document_review | 4 (PASS) | 151.8s |  | 151.8s |
| 37 | agents/maker_checker | 4 (PASS) | 55.6s |  | 55.6s |
| 38 | agents/maker_checker_code | 4 (PASS) | 79.6s |  | 79.6s |
| 39 | agents/medical_literature | 4 (PASS) | 119.6s |  | 119.6s |
| 40 | agents/memory_agent | 4 (PASS) | 147.1s |  | 147.1s |
| 41 | agents/multi_debate_panel | 4 (PASS) | 44.3s |  | 44.3s |
| 42 | agents/multi_tool_recovery | 4 (PASS) | 123.5s |  | 123.5s |
| 43 | agents/orchestrator | 4 (PASS) | 19.8s |  | 19.8s |
| 44 | agents/orchestrator_specialist | 4 (PASS) | 20.1s |  | 20.1s |
| 45 | agents/pipeline_review | 4 (PASS) | 152.8s |  | 152.8s |
| 46 | agents/plan_execute | 4 (PASS) | 65.9s |  | 65.9s |
| 47 | agents/plan_execute_recovery | 4 (PASS) | 63.2s |  | 63.2s |
| 48 | agents/prompt_chain | 4 (PASS) | 44.2s |  | 44.2s |
| 49 | agents/react_hitl_combined | 4 (PASS) | 223.4s |  | 223.4s |
| 50 | agents/react_search | 4 (PASS) | 43.8s |  | 43.8s |
| 51 | agents/react_structured_pipeline | 4 (PASS) | 139.0s |  | 139.0s |
| 52 | agents/reasoning_stacking | 4 (PASS) | 32.9s |  | 32.9s |
| 53 | agents/reasoning_tool | 4 (PASS) | 24.9s |  | 24.9s |
| 54 | agents/reflexion | 4 (PASS) | 149.0s |  | 149.0s |
| 55 | agents/reflexion_code_gen | 4 (PASS) | 53.6s |  | 53.6s |
| 56 | agents/regulatory_compliance | 4 (PASS) | 74.8s |  | 74.8s |
| 57 | agents/rewoo | 4 (PASS) | 26.1s |  | 26.1s |
| 58 | agents/rewoo_multi_step | 4 (PASS) | 29.5s |  | 29.5s |
| 59 | agents/security_audit | 4 (PASS) | 85.8s |  | 85.8s |
| 60 | agents/self_consistency | 4 (PASS) | 23.5s |  | 23.5s |
| 61 | agents/skill_loader | 4 (PASS) | 135.3s |  | 135.3s |
| 62 | agents/structured_output | 4 (PASS) | 59.4s |  | 59.4s |
| 63 | agents/supply_chain_optimizer | 4 (PASS) | 24.4s |  | 24.4s |
| 64 | agents/tool_decorator | 4 (PASS) | 46.9s |  | 46.9s |
| 65 | agents/workflow_agent | 4 (PASS) | 13.5s |  | 13.5s |
| 66 | basic/event_registration | 1 (BROKEN) | 22.2s | F-EXTRACT, F-TRANS | 22.2s |
| 67 | basic/form_filling | 4 (PASS) | 24.4s |  | 24.4s |
| 68 | basic/insurance_claim | 1 (BROKEN) | 26.9s | F-EXTRACT, F-TRANS | 26.9s |
| 69 | basic/job_application | 1 (BROKEN) | 29.5s | F-EXTRACT, F-TRANS | 29.5s |
| 70 | basic/medical_intake | 1 (BROKEN) | 28.1s | F-EXTRACT, F-TRANS | 28.1s |
| 71 | basic/multi_turn_extraction | 4 (PASS) | 49.2s |  | 49.2s |
| 72 | basic/pet_adoption | 1 (BROKEN) | 25.2s | F-EXTRACT, F-TRANS | 25.2s |
| 73 | basic/rental_application | 1 (BROKEN) | 28.6s | F-EXTRACT, F-TRANS | 28.6s |
| 74 | basic/restaurant_reservation | 1 (BROKEN) | 30.4s | F-EXTRACT, F-TRANS | 30.4s |
| 75 | basic/scholarship_application | 1 (BROKEN) | 30.2s | F-EXTRACT, F-TRANS | 30.2s |
| 76 | basic/simple_greeting | 4 (PASS) | 27.5s |  | 27.5s |
| 77 | basic/story_time | 4 (PASS) | 41.9s |  | 41.9s |
| 78 | basic/tech_support_intake | 1 (BROKEN) | 29.2s | F-EXTRACT, F-TRANS | 29.2s |
| 79 | basic/travel_booking | 1 (BROKEN) | 32.3s | F-EXTRACT, F-TRANS | 32.3s |
| 80 | classification/classified_transitions | 4 (PASS) | 33.3s |  | 33.3s |
| 81 | classification/intent_routing | 4 (PASS) | 15.7s |  | 15.7s |
| 82 | classification/multi_intent | 4 (PASS) | 33.6s |  | 33.6s |
| 83 | classification/smart_helpdesk | 4 (PASS) | 25.0s |  | 25.0s |
| 84 | intermediate/adaptive_quiz | 4 (PASS) | 25.3s |  | 25.3s |
| 85 | intermediate/book_recommendation | 4 (PASS) | 24.7s |  | 24.7s |
| 86 | intermediate/product_recommendation | 4 (PASS) | 15.6s |  | 15.6s |
| 87 | meta/build_agent | 4 (PASS) | 4.0s |  | 4.0s |
| 88 | meta/build_fsm | 4 (PASS) | 3.7s |  | 3.7s |
| 89 | meta/build_workflow | 4 (PASS) | 3.7s |  | 3.7s |
| 90 | meta/meta_from_spec | 4 (PASS) | 64.1s |  | 64.1s |
| 91 | meta/meta_review_loop | 4 (PASS) | 66.6s |  | 66.6s |
| 92 | reasoning/math_tutor | 4 (PASS) | 21.6s |  | 21.6s |
| 93 | workflows/agent_workflow_chain | 4 (PASS) | 27.7s |  | 27.7s |
| 94 | workflows/conditional_branching | 4 (PASS) | 3.6s |  | 3.6s |
| 95 | workflows/customer_onboarding | 4 (PASS) | 52.1s |  | 52.1s |
| 96 | workflows/loan_processing | 4 (PASS) | 33.5s |  | 33.5s |
| 97 | workflows/order_processing | 4 (PASS) | 39.0s |  | 39.0s |
| 98 | workflows/parallel_steps | 4 (PASS) | 5.2s |  | 5.2s |
| 99 | workflows/release_management | 4 (PASS) | 36.4s |  | 36.4s |
| 100 | workflows/workflow_agent_loop | 4 (PASS) | 44.0s |  | 44.0s |

## Summary

- **Total examples**: 100
- **Score distribution**: 79x4, 1x3, 0x2, 20x1, 0x0
- **Health Score**: 339/400 = **84.8%**
- **Category breakdown**:
  - advanced: 38/68 (56%)
  - agents: 191/192 (99%)
  - basic: 26/56 (46%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 20/20 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 32/32 (100%)
- **Top failure codes**: F-EXTRACT (20), F-TRANS (20), F-LOOP (1)

## Timing

- **Total wall time**: 5388.5s (sequential equivalent)
- **Fastest**: 3.6s
- **Slowest**: 223.4s
- **Mean**: 53.9s
