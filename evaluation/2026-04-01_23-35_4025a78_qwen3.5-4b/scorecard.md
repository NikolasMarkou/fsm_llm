# Evaluation: 2026-04-01 23:58

- **Date**: 2026-04-01 23:58
- **Git commit**: 4025a78
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 100
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/budget_review | 1 (BROKEN) | 46.3s | F-EXTRACT, F-TRANS | 46.3s |
| 2 | advanced/compliance_audit | 1 (BROKEN) | 49.5s | F-EXTRACT, F-TRANS | 49.5s |
| 3 | advanced/concurrent_conversations | 1 (BROKEN) | 54.4s | F-EXTRACT, F-TRANS | 54.4s |
| 4 | advanced/context_compactor | 1 (BROKEN) | 55.7s | F-EXTRACT, F-TRANS | 55.7s |
| 5 | advanced/customer_feedback_pipeline | 1 (BROKEN) | 28.5s | F-EXTRACT, F-TRANS | 28.5s |
| 6 | advanced/e_commerce | 2 (PARTIAL) | 107.2s | F-EXTRACT | 107.2s |
| 7 | advanced/employee_onboarding | 1 (BROKEN) | 36.3s | F-EXTRACT, F-TRANS | 36.3s |
| 8 | advanced/handler_hooks | 2 (PARTIAL) | 32.1s | F-CODE | Exit 1 |
| 9 | advanced/incident_response | 1 (BROKEN) | 41.2s | F-EXTRACT, F-TRANS | 41.2s |
| 10 | advanced/loan_assessment | 1 (BROKEN) | 42.6s | F-EXTRACT, F-TRANS | 42.6s |
| 11 | advanced/medical_triage | 1 (BROKEN) | 45.0s | F-EXTRACT, F-TRANS | 45.0s |
| 12 | advanced/multi_level_stack | 1 (BROKEN) | 33.5s | F-EXTRACT, F-TRANS | 33.5s |
| 13 | advanced/project_planning | 1 (BROKEN) | 39.9s | F-EXTRACT, F-TRANS | 39.9s |
| 14 | advanced/quality_inspection | 1 (BROKEN) | 37.2s | F-EXTRACT, F-TRANS | 37.2s |
| 15 | advanced/support_pipeline | 1 (BROKEN) | 66.0s | F-EXTRACT, F-TRANS | 66.0s |
| 16 | advanced/vendor_evaluation | 1 (BROKEN) | 46.0s | F-EXTRACT, F-TRANS | 46.0s |
| 17 | advanced/yoga_instructions | 4 (PASS) | 28.7s |  | 28.7s |
| 18 | agents/adapt | 4 (PASS) | 41.5s |  | 41.5s |
| 19 | agents/adapt_with_memory | 4 (PASS) | 125.3s |  | 125.3s |
| 20 | agents/agent_as_tool | 4 (PASS) | 104.1s |  | 104.1s |
| 21 | agents/agent_memory_chain | 4 (PASS) | 147.8s |  | 147.8s |
| 22 | agents/architecture_review | 4 (PASS) | 52.0s |  | 52.0s |
| 23 | agents/classified_dispatch | 4 (PASS) | 71.7s |  | 71.7s |
| 24 | agents/classified_tools | 4 (PASS) | 38.2s |  | 38.2s |
| 25 | agents/concurrent_react | 4 (PASS) | 113.3s |  | 113.3s |
| 26 | agents/consistency_with_tools | 4 (PASS) | 28.6s |  | 28.6s |
| 27 | agents/debate | 4 (PASS) | 139.5s |  | 139.5s |
| 28 | agents/debate_with_tools | 4 (PASS) | 147.7s |  | 147.7s |
| 29 | agents/eval_opt_structured | 4 (PASS) | 98.4s |  | 98.4s |
| 30 | agents/evaluator_optimizer | 4 (PASS) | 78.7s |  | 78.7s |
| 31 | agents/full_pipeline | 4 (PASS) | 81.5s |  | 81.5s |
| 32 | agents/hierarchical_orchestrator | 4 (PASS) | 21.0s |  | 21.0s |
| 33 | agents/hierarchical_tools | 4 (PASS) | 44.1s |  | 44.1s |
| 34 | agents/hitl_approval | 4 (PASS) | 75.3s |  | 75.3s |
| 35 | agents/investment_portfolio | 4 (PASS) | 41.7s |  | 41.7s |
| 36 | agents/legal_document_review | 4 (PASS) | 223.2s |  | 223.2s |
| 37 | agents/maker_checker | 4 (PASS) | 85.3s |  | 85.3s |
| 38 | agents/maker_checker_code | 4 (PASS) | 80.7s |  | 80.7s |
| 39 | agents/medical_literature | 4 (PASS) | 168.8s |  | 168.8s |
| 40 | agents/memory_agent | 4 (PASS) | 148.5s |  | 148.5s |
| 41 | agents/multi_debate_panel | 4 (PASS) | 89.6s |  | 89.6s |
| 42 | agents/multi_tool_recovery | 3 (MOSTLY) | 30.6s | F-LOOP | 30.6s |
| 43 | agents/orchestrator | 4 (PASS) | 19.9s |  | 19.9s |
| 44 | agents/orchestrator_specialist | 4 (PASS) | 18.3s |  | 18.3s |
| 45 | agents/pipeline_review | 4 (PASS) | 125.2s |  | 125.2s |
| 46 | agents/plan_execute | 4 (PASS) | 67.0s |  | 67.0s |
| 47 | agents/plan_execute_recovery | 4 (PASS) | 71.0s |  | 71.0s |
| 48 | agents/prompt_chain | 4 (PASS) | 74.9s |  | 74.9s |
| 49 | agents/react_hitl_combined | 4 (PASS) | 83.3s |  | 83.3s |
| 50 | agents/react_search | 4 (PASS) | 36.4s |  | 36.4s |
| 51 | agents/react_structured_pipeline | 3 (MOSTLY) | 39.3s | F-LOOP | 39.3s |
| 52 | agents/reasoning_stacking | 4 (PASS) | 29.3s |  | 29.3s |
| 53 | agents/reasoning_tool | 4 (PASS) | 60.8s |  | 60.8s |
| 54 | agents/reflexion | 4 (PASS) | 121.2s |  | 121.2s |
| 55 | agents/reflexion_code_gen | 4 (PASS) | 100.3s |  | 100.3s |
| 56 | agents/regulatory_compliance | 4 (PASS) | 87.7s |  | 87.7s |
| 57 | agents/rewoo | 4 (PASS) | 31.7s |  | 31.7s |
| 58 | agents/rewoo_multi_step | 4 (PASS) | 28.5s |  | 28.5s |
| 59 | agents/security_audit | 4 (PASS) | 81.5s |  | 81.5s |
| 60 | agents/self_consistency | 4 (PASS) | 25.3s |  | 25.3s |
| 61 | agents/skill_loader | 4 (PASS) | 53.5s |  | 53.5s |
| 62 | agents/structured_output | 4 (PASS) | 48.0s |  | 48.0s |
| 63 | agents/supply_chain_optimizer | 4 (PASS) | 21.7s |  | 21.7s |
| 64 | agents/tool_decorator | 4 (PASS) | 33.1s |  | 33.1s |
| 65 | agents/workflow_agent | 4 (PASS) | 7.5s |  | 7.5s |
| 66 | basic/event_registration | 1 (BROKEN) | 33.6s | F-EXTRACT, F-TRANS | 33.6s |
| 67 | basic/form_filling | 1 (BROKEN) | 33.7s | F-EXTRACT, F-TRANS | 33.7s |
| 68 | basic/insurance_claim | 1 (BROKEN) | 28.5s | F-EXTRACT, F-TRANS | 28.5s |
| 69 | basic/job_application | 1 (BROKEN) | 30.1s | F-EXTRACT, F-TRANS | 30.1s |
| 70 | basic/medical_intake | 1 (BROKEN) | 26.5s | F-EXTRACT, F-TRANS | 26.5s |
| 71 | basic/multi_turn_extraction | 4 (PASS) | 52.2s |  | 52.2s |
| 72 | basic/pet_adoption | 1 (BROKEN) | 30.5s | F-EXTRACT, F-TRANS | 30.5s |
| 73 | basic/rental_application | 1 (BROKEN) | 34.8s | F-EXTRACT, F-TRANS | 34.8s |
| 74 | basic/restaurant_reservation | 1 (BROKEN) | 33.8s | F-EXTRACT, F-TRANS | 33.8s |
| 75 | basic/scholarship_application | 1 (BROKEN) | 27.8s | F-EXTRACT, F-TRANS | 27.8s |
| 76 | basic/simple_greeting | 1 (BROKEN) | 24.1s | F-EXTRACT, F-TRANS | 24.1s |
| 77 | basic/story_time | 1 (BROKEN) | 40.1s | F-EXTRACT, F-TRANS | 40.1s |
| 78 | basic/tech_support_intake | 1 (BROKEN) | 27.5s | F-EXTRACT, F-TRANS | 27.5s |
| 79 | basic/travel_booking | 1 (BROKEN) | 30.2s | F-EXTRACT, F-TRANS | 30.2s |
| 80 | classification/classified_transitions | 4 (PASS) | 74.2s |  | 74.2s |
| 81 | classification/intent_routing | 4 (PASS) | 23.6s |  | 23.6s |
| 82 | classification/multi_intent | 4 (PASS) | 32.5s |  | 32.5s |
| 83 | classification/smart_helpdesk | 4 (PASS) | 23.9s |  | 23.9s |
| 84 | intermediate/adaptive_quiz | 2 (PARTIAL) | 33.0s | F-EXTRACT | 33.0s |
| 85 | intermediate/book_recommendation | 2 (PARTIAL) | 27.2s | F-EXTRACT | 27.2s |
| 86 | intermediate/product_recommendation | 1 (BROKEN) | 27.6s | F-EXTRACT, F-TRANS | 27.6s |
| 87 | meta/build_agent | 2 (PARTIAL) | 3.6s | F-EXTRACT | 3.6s |
| 88 | meta/build_fsm | 2 (PARTIAL) | 3.3s | F-EXTRACT | 3.3s |
| 89 | meta/build_workflow | 2 (PARTIAL) | 3.5s | F-EXTRACT | 3.5s |
| 90 | meta/meta_from_spec | 4 (PASS) | 63.7s |  | 63.7s |
| 91 | meta/meta_review_loop | 4 (PASS) | 66.7s |  | 66.7s |
| 92 | reasoning/math_tutor | 4 (PASS) | 21.3s |  | 21.3s |
| 93 | workflows/agent_workflow_chain | 4 (PASS) | 23.0s |  | 23.0s |
| 94 | workflows/conditional_branching | 4 (PASS) | 3.8s |  | 3.8s |
| 95 | workflows/customer_onboarding | 4 (PASS) | 52.5s |  | 52.5s |
| 96 | workflows/loan_processing | 4 (PASS) | 38.8s |  | 38.8s |
| 97 | workflows/order_processing | 4 (PASS) | 37.8s |  | 37.8s |
| 98 | workflows/parallel_steps | 4 (PASS) | 5.3s |  | 5.3s |
| 99 | workflows/release_management | 4 (PASS) | 45.7s |  | 45.7s |
| 100 | workflows/workflow_agent_loop | 4 (PASS) | 35.9s |  | 35.9s |

## Summary

- **Total examples**: 100
- **Score distribution**: 63x4, 2x3, 7x2, 28x1, 0x0
- **Health Score**: 300/400 = **75.0%**
- **Category breakdown**:
  - advanced: 22/68 (32%)
  - agents: 190/192 (99%)
  - basic: 17/56 (30%)
  - classification: 16/16 (100%)
  - intermediate: 5/12 (42%)
  - meta: 14/20 (70%)
  - reasoning: 4/4 (100%)
  - workflows: 32/32 (100%)
- **Top failure codes**: F-EXTRACT (34), F-TRANS (28), F-LOOP (2), F-CODE (1)

## Timing

- **Total wall time**: 5433.3s (sequential equivalent)
- **Fastest**: 3.3s
- **Slowest**: 223.2s
- **Mean**: 54.3s
