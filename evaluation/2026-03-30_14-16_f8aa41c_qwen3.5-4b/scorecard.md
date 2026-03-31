# Evaluation: 2026-03-30 14:44

- **Date**: 2026-03-30 14:44
- **Git commit**: f8aa41c
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 80
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 25.4s |  | 25.4s |
| 2 | advanced/context_compactor | 4 (PASS) | 27.4s |  | 27.4s |
| 3 | advanced/e_commerce | 4 (PASS) | 165.0s |  | 165.0s |
| 4 | advanced/handler_hooks | 4 (PASS) | 16.2s |  | 16.2s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 25.1s |  | 25.1s |
| 6 | advanced/support_pipeline | 4 (PASS) | 43.9s |  | 43.9s |
| 7 | advanced/yoga_instructions | 4 (PASS) | 22.1s |  | 22.1s |
| 8 | agents/adapt | 4 (PASS) | 200.2s |  | 200.2s |
| 9 | agents/adapt_with_memory | 4 (PASS) | 253.2s |  | 253.2s |
| 10 | agents/agent_as_tool | 4 (PASS) | 90.6s |  | 90.6s |
| 11 | agents/agent_memory_chain | 1 (BROKEN) | 300.0s | F-LOOP | Timeout (300s) |
| 12 | agents/architecture_review | 4 (PASS) | 86.0s |  | 86.0s |
| 13 | agents/classified_dispatch | 4 (PASS) | 63.6s |  | 63.6s |
| 14 | agents/classified_tools | 1 (BROKEN) | 180.0s | F-LOOP | Timeout (180s) |
| 15 | agents/concurrent_react | 4 (PASS) | 130.5s |  | 130.5s |
| 16 | agents/consistency_with_tools | 4 (PASS) | 30.0s |  | 30.0s |
| 17 | agents/debate | 4 (PASS) | 102.1s |  | 102.1s |
| 18 | agents/debate_with_tools | 4 (PASS) | 87.9s |  | 87.9s |
| 19 | agents/eval_opt_structured | 1 (BROKEN) | 300.1s | F-LOOP | Timeout (300s) |
| 20 | agents/evaluator_optimizer | 4 (PASS) | 98.6s |  | 98.6s |
| 21 | agents/full_pipeline | 4 (PASS) | 82.6s |  | 82.6s |
| 22 | agents/hierarchical_orchestrator | 4 (PASS) | 20.0s |  | 20.0s |
| 23 | agents/hierarchical_tools | 4 (PASS) | 105.7s |  | 105.7s |
| 24 | agents/hitl_approval | 4 (PASS) | 90.5s |  | 90.5s |
| 25 | agents/investment_portfolio | 4 (PASS) | 53.8s |  | 53.8s |
| 26 | agents/legal_document_review | 4 (PASS) | 196.9s |  | 196.9s |
| 27 | agents/maker_checker | 4 (PASS) | 81.8s |  | 81.8s |
| 28 | agents/maker_checker_code | 4 (PASS) | 107.7s |  | 107.7s |
| 29 | agents/medical_literature | 4 (PASS) | 166.6s |  | 166.6s |
| 30 | agents/memory_agent | 4 (PASS) | 225.6s |  | 225.6s |
| 31 | agents/multi_debate_panel | 4 (PASS) | 78.4s |  | 78.4s |
| 32 | agents/multi_tool_recovery | 4 (PASS) | 87.6s |  | 87.6s |
| 33 | agents/orchestrator | 4 (PASS) | 17.6s |  | 17.6s |
| 34 | agents/orchestrator_specialist | 4 (PASS) | 18.3s |  | 18.3s |
| 35 | agents/pipeline_review | 4 (PASS) | 87.5s |  | 87.5s |
| 36 | agents/plan_execute | 4 (PASS) | 38.3s |  | 38.3s |
| 37 | agents/plan_execute_recovery | 4 (PASS) | 42.4s |  | 42.4s |
| 38 | agents/prompt_chain | 4 (PASS) | 33.0s |  | 33.0s |
| 39 | agents/react_hitl_combined | 4 (PASS) | 212.9s |  | 212.9s |
| 40 | agents/react_search | 4 (PASS) | 82.3s |  | 82.3s |
| 41 | agents/react_structured_pipeline | 1 (BROKEN) | 300.1s | F-LOOP | Timeout (300s) |
| 42 | agents/reasoning_stacking | 4 (PASS) | 57.4s |  | 57.4s |
| 43 | agents/reasoning_tool | 4 (PASS) | 28.5s |  | 28.5s |
| 44 | agents/reflexion | 4 (PASS) | 117.9s |  | 117.9s |
| 45 | agents/reflexion_code_gen | 4 (PASS) | 175.4s |  | 175.4s |
| 46 | agents/regulatory_compliance | 4 (PASS) | 101.4s |  | 101.4s |
| 47 | agents/rewoo | 4 (PASS) | 25.0s |  | 25.0s |
| 48 | agents/rewoo_multi_step | 4 (PASS) | 36.6s |  | 36.6s |
| 49 | agents/security_audit | 4 (PASS) | 129.8s |  | 129.8s |
| 50 | agents/self_consistency | 4 (PASS) | 35.0s |  | 35.0s |
| 51 | agents/skill_loader | 4 (PASS) | 91.7s |  | 91.7s |
| 52 | agents/structured_output | 4 (PASS) | 80.4s |  | 80.4s |
| 53 | agents/supply_chain_optimizer | 4 (PASS) | 24.6s |  | 24.6s |
| 54 | agents/tool_decorator | 4 (PASS) | 40.9s |  | 40.9s |
| 55 | agents/workflow_agent | 4 (PASS) | 10.0s |  | 10.0s |
| 56 | basic/form_filling | 4 (PASS) | 22.0s |  | 22.0s |
| 57 | basic/multi_turn_extraction | 4 (PASS) | 42.5s |  | 42.5s |
| 58 | basic/simple_greeting | 4 (PASS) | 24.3s |  | 24.3s |
| 59 | basic/story_time | 4 (PASS) | 31.0s |  | 31.0s |
| 60 | classification/classified_transitions | 4 (PASS) | 25.9s |  | 25.9s |
| 61 | classification/intent_routing | 4 (PASS) | 13.9s |  | 13.9s |
| 62 | classification/multi_intent | 4 (PASS) | 30.3s |  | 30.3s |
| 63 | classification/smart_helpdesk | 4 (PASS) | 21.5s |  | 21.5s |
| 64 | intermediate/adaptive_quiz | 4 (PASS) | 24.1s |  | 24.1s |
| 65 | intermediate/book_recommendation | 4 (PASS) | 25.7s |  | 25.7s |
| 66 | intermediate/product_recommendation | 4 (PASS) | 19.4s |  | 19.4s |
| 67 | meta/build_agent | 4 (PASS) | 7.5s |  | 7.5s |
| 68 | meta/build_fsm | 4 (PASS) | 8.2s |  | 8.2s |
| 69 | meta/build_workflow | 4 (PASS) | 8.4s |  | 8.4s |
| 70 | meta/meta_from_spec | 4 (PASS) | 220.0s |  | 220.0s |
| 71 | meta/meta_review_loop | 4 (PASS) | 173.4s |  | 173.4s |
| 72 | reasoning/math_tutor | 4 (PASS) | 36.1s |  | 36.1s |
| 73 | workflows/agent_workflow_chain | 4 (PASS) | 56.5s |  | 56.5s |
| 74 | workflows/conditional_branching | 4 (PASS) | 3.7s |  | 3.7s |
| 75 | workflows/customer_onboarding | 4 (PASS) | 89.3s |  | 89.3s |
| 76 | workflows/loan_processing | 3 (MOSTLY) | 34.6s | F-EXTRACT | 34.6s |
| 77 | workflows/order_processing | 4 (PASS) | 116.4s |  | 116.4s |
| 78 | workflows/parallel_steps | 4 (PASS) | 5.7s |  | 5.7s |
| 79 | workflows/release_management | 4 (PASS) | 105.8s |  | 105.8s |
| 80 | workflows/workflow_agent_loop | 4 (PASS) | 58.7s |  | 58.7s |

## Summary

- **Total examples**: 80
- **Score distribution**: 75x4, 1x3, 0x2, 4x1, 0x0
- **Health Score**: 307/320 = **95.9%**
- **Category breakdown**:
  - advanced: 28/28 (100%)
  - agents: 180/192 (94%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 20/20 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 31/32 (97%)
- **Top failure codes**: F-LOOP (4), F-EXTRACT (1)

## Timing

- **Total wall time**: 6537.5s (sequential equivalent)
- **Fastest**: 3.7s
- **Slowest**: 300.1s
- **Mean**: 81.7s
