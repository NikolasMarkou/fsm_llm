# Evaluation: 2026-03-30 11:50

- **Date**: 2026-03-30 11:50
- **Git commit**: ae21348
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 80
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 142.5s |  | 142.5s |
| 2 | advanced/context_compactor | 4 (PASS) | 133.1s |  | 133.1s |
| 3 | advanced/e_commerce | 1 (BROKEN) | 300.1s | F-LOOP | Timeout (300s) |
| 4 | advanced/handler_hooks | 4 (PASS) | 102.1s |  | 102.1s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 77.5s |  | 77.5s |
| 6 | advanced/support_pipeline | 4 (PASS) | 98.8s |  | 98.8s |
| 7 | advanced/yoga_instructions | 4 (PASS) | 38.9s |  | 38.9s |
| 8 | agents/adapt | 1 (BROKEN) | 240.0s | F-LOOP | Timeout (240s) |
| 9 | agents/adapt_with_memory | 1 (BROKEN) | 300.0s | F-LOOP | Timeout (300s) |
| 10 | agents/agent_as_tool | 1 (BROKEN) | 300.0s | F-LOOP | Timeout (300s) |
| 11 | agents/agent_memory_chain | 1 (BROKEN) | 300.0s | F-LOOP | Timeout (300s) |
| 12 | agents/architecture_review | 4 (PASS) | 126.2s |  | 126.2s |
| 13 | agents/classified_dispatch | 4 (PASS) | 157.2s |  | 157.2s |
| 14 | agents/classified_tools | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 15 | agents/concurrent_react | 4 (PASS) | 88.4s |  | 88.4s |
| 16 | agents/consistency_with_tools | 4 (PASS) | 43.3s |  | 43.3s |
| 17 | agents/debate | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 18 | agents/debate_with_tools | 4 (PASS) | 212.1s |  | 212.1s |
| 19 | agents/eval_opt_structured | 1 (BROKEN) | 300.1s | F-LOOP | Timeout (300s) |
| 20 | agents/evaluator_optimizer | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 21 | agents/full_pipeline | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 22 | agents/hierarchical_orchestrator | 4 (PASS) | 23.8s |  | 23.8s |
| 23 | agents/hierarchical_tools | 4 (PASS) | 70.7s |  | 70.7s |
| 24 | agents/hitl_approval | 4 (PASS) | 131.3s |  | 131.3s |
| 25 | agents/investment_portfolio | 4 (PASS) | 48.5s |  | 48.5s |
| 26 | agents/legal_document_review | 4 (PASS) | 69.0s |  | 69.0s |
| 27 | agents/maker_checker | 4 (PASS) | 97.4s |  | 97.4s |
| 28 | agents/maker_checker_code | 4 (PASS) | 102.1s |  | 102.1s |
| 29 | agents/medical_literature | 4 (PASS) | 157.8s |  | 157.8s |
| 30 | agents/memory_agent | 4 (PASS) | 270.5s |  | 270.5s |
| 31 | agents/multi_debate_panel | 4 (PASS) | 52.2s |  | 52.2s |
| 32 | agents/multi_tool_recovery | 4 (PASS) | 109.1s |  | 109.1s |
| 33 | agents/orchestrator | 4 (PASS) | 19.0s |  | 19.0s |
| 34 | agents/orchestrator_specialist | 4 (PASS) | 22.0s |  | 22.0s |
| 35 | agents/pipeline_review | 4 (PASS) | 105.2s |  | 105.2s |
| 36 | agents/plan_execute | 4 (PASS) | 35.3s |  | 35.3s |
| 37 | agents/plan_execute_recovery | 4 (PASS) | 42.9s |  | 42.9s |
| 38 | agents/prompt_chain | 4 (PASS) | 44.7s |  | 44.7s |
| 39 | agents/react_hitl_combined | 4 (PASS) | 156.3s |  | 156.3s |
| 40 | agents/react_search | 4 (PASS) | 72.7s |  | 72.7s |
| 41 | agents/react_structured_pipeline | 1 (BROKEN) | 300.1s | F-LOOP | Timeout (300s) |
| 42 | agents/reasoning_stacking | 4 (PASS) | 28.4s |  | 28.4s |
| 43 | agents/reasoning_tool | 4 (PASS) | 30.1s |  | 30.1s |
| 44 | agents/reflexion | 1 (BROKEN) | 240.1s | F-LOOP | Timeout (240s) |
| 45 | agents/reflexion_code_gen | 4 (PASS) | 219.0s |  | 219.0s |
| 46 | agents/regulatory_compliance | 4 (PASS) | 93.3s |  | 93.3s |
| 47 | agents/rewoo | 4 (PASS) | 23.0s |  | 23.0s |
| 48 | agents/rewoo_multi_step | 4 (PASS) | 27.6s |  | 27.6s |
| 49 | agents/security_audit | 4 (PASS) | 85.6s |  | 85.6s |
| 50 | agents/self_consistency | 4 (PASS) | 21.6s |  | 21.6s |
| 51 | agents/skill_loader | 4 (PASS) | 96.9s |  | 96.9s |
| 52 | agents/structured_output | 4 (PASS) | 76.7s |  | 76.7s |
| 53 | agents/supply_chain_optimizer | 4 (PASS) | 19.9s |  | 19.9s |
| 54 | agents/tool_decorator | 4 (PASS) | 53.9s |  | 53.9s |
| 55 | agents/workflow_agent | 4 (PASS) | 15.9s |  | 15.9s |
| 56 | basic/form_filling | 4 (PASS) | 20.9s |  | 20.9s |
| 57 | basic/multi_turn_extraction | 4 (PASS) | 36.6s |  | 36.6s |
| 58 | basic/simple_greeting | 4 (PASS) | 21.5s |  | 21.5s |
| 59 | basic/story_time | 4 (PASS) | 32.9s |  | 32.9s |
| 60 | classification/classified_transitions | 4 (PASS) | 27.9s |  | 27.9s |
| 61 | classification/intent_routing | 4 (PASS) | 13.3s |  | 13.3s |
| 62 | classification/multi_intent | 4 (PASS) | 33.0s |  | 33.0s |
| 63 | classification/smart_helpdesk | 4 (PASS) | 21.9s |  | 21.9s |
| 64 | intermediate/adaptive_quiz | 4 (PASS) | 24.9s |  | 24.9s |
| 65 | intermediate/book_recommendation | 4 (PASS) | 24.6s |  | 24.6s |
| 66 | intermediate/product_recommendation | 4 (PASS) | 15.9s |  | 15.9s |
| 67 | meta/build_agent | 4 (PASS) | 8.3s |  | 8.3s |
| 68 | meta/build_fsm | 4 (PASS) | 7.9s |  | 7.9s |
| 69 | meta/build_workflow | 4 (PASS) | 8.5s |  | 8.5s |
| 70 | meta/meta_from_spec | 4 (PASS) | 202.1s |  | 202.1s |
| 71 | meta/meta_review_loop | 4 (PASS) | 172.3s |  | 172.3s |
| 72 | reasoning/math_tutor | 4 (PASS) | 37.5s |  | 37.5s |
| 73 | workflows/agent_workflow_chain | 4 (PASS) | 62.6s |  | 62.6s |
| 74 | workflows/conditional_branching | 4 (PASS) | 4.0s |  | 4.0s |
| 75 | workflows/customer_onboarding | 4 (PASS) | 115.7s |  | 115.7s |
| 76 | workflows/loan_processing | 3 (MOSTLY) | 30.0s | F-EXTRACT | 30.0s |
| 77 | workflows/order_processing | 4 (PASS) | 98.8s |  | 98.8s |
| 78 | workflows/parallel_steps | 4 (PASS) | 5.5s |  | 5.5s |
| 79 | workflows/release_management | 4 (PASS) | 71.3s |  | 71.3s |
| 80 | workflows/workflow_agent_loop | 3 (MOSTLY) | 167.7s | F-LOOP | 167.7s |

## Summary

- **Total examples**: 80
- **Score distribution**: 66x4, 2x3, 0x2, 12x1, 0x0
- **Health Score**: 282/320 = **88.1%**
- **Category breakdown**:
  - advanced: 25/28 (89%)
  - agents: 159/192 (83%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 20/20 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 30/32 (94%)
- **Top failure codes**: F-LOOP (13), F-EXTRACT (1)

## Timing

- **Total wall time**: 7909.2s (sequential equivalent)
- **Fastest**: 4.0s
- **Slowest**: 300.1s
- **Mean**: 98.9s
