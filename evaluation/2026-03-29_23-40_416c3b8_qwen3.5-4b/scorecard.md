# Evaluation: 2026-03-30 00:01

- **Date**: 2026-03-30 00:01
- **Git commit**: 416c3b8
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 80
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 24.3s |  | 24.3s |
| 2 | advanced/context_compactor | 4 (PASS) | 27.8s |  | 27.8s |
| 3 | advanced/e_commerce | 4 (PASS) | 157.7s |  | 157.7s |
| 4 | advanced/handler_hooks | 4 (PASS) | 15.3s |  | 15.3s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 24.3s |  | 24.3s |
| 6 | advanced/support_pipeline | 4 (PASS) | 45.2s |  | 45.2s |
| 7 | advanced/yoga_instructions | 4 (PASS) | 24.1s |  | 24.1s |
| 8 | agents/adapt | 4 (PASS) | 136.8s |  | 136.8s |
| 9 | agents/adapt_with_memory | 4 (PASS) | 198.8s |  | 198.8s |
| 10 | agents/agent_as_tool | 4 (PASS) | 198.5s |  | 198.5s |
| 11 | agents/agent_memory_chain | 3 (MOSTLY) | 165.7s | F-LOOP | 165.7s |
| 12 | agents/architecture_review | 4 (PASS) | 72.6s |  | 72.6s |
| 13 | agents/classified_dispatch | 4 (PASS) | 63.1s |  | 63.1s |
| 14 | agents/classified_tools | 4 (PASS) | 159.7s |  | 159.7s |
| 15 | agents/concurrent_react | 4 (PASS) | 88.1s |  | 88.1s |
| 16 | agents/consistency_with_tools | 4 (PASS) | 23.4s |  | 23.4s |
| 17 | agents/debate | 4 (PASS) | 85.7s |  | 85.7s |
| 18 | agents/debate_with_tools | 4 (PASS) | 3.7s |  | 3.7s |
| 19 | agents/eval_opt_structured | 4 (PASS) | 115.2s |  | 115.2s |
| 20 | agents/evaluator_optimizer | 4 (PASS) | 80.1s |  | 80.1s |
| 21 | agents/full_pipeline | 4 (PASS) | 97.9s |  | 97.9s |
| 22 | agents/hierarchical_orchestrator | 4 (PASS) | 18.9s |  | 18.9s |
| 23 | agents/hierarchical_tools | 4 (PASS) | 132.6s |  | 132.6s |
| 24 | agents/hitl_approval | 4 (PASS) | 99.2s |  | 99.2s |
| 25 | agents/investment_portfolio | 4 (PASS) | 33.4s |  | 33.4s |
| 26 | agents/legal_document_review | 4 (PASS) | 57.4s |  | 57.4s |
| 27 | agents/maker_checker | 4 (PASS) | 82.7s |  | 82.7s |
| 28 | agents/maker_checker_code | 4 (PASS) | 139.6s |  | 139.6s |
| 29 | agents/medical_literature | 4 (PASS) | 170.4s |  | 170.4s |
| 30 | agents/memory_agent | 4 (PASS) | 245.0s |  | 245.0s |
| 31 | agents/multi_debate_panel | 4 (PASS) | 83.9s |  | 83.9s |
| 32 | agents/multi_tool_recovery | 4 (PASS) | 98.3s |  | 98.3s |
| 33 | agents/orchestrator | 4 (PASS) | 19.4s |  | 19.4s |
| 34 | agents/orchestrator_specialist | 4 (PASS) | 16.8s |  | 16.8s |
| 35 | agents/pipeline_review | 4 (PASS) | 106.8s |  | 106.8s |
| 36 | agents/plan_execute | 4 (PASS) | 46.9s |  | 46.9s |
| 37 | agents/plan_execute_recovery | 4 (PASS) | 27.9s |  | 27.9s |
| 38 | agents/prompt_chain | 4 (PASS) | 33.8s |  | 33.8s |
| 39 | agents/react_hitl_combined | 4 (PASS) | 157.9s |  | 157.9s |
| 40 | agents/react_search | 4 (PASS) | 69.0s |  | 69.0s |
| 41 | agents/react_structured_pipeline | 4 (PASS) | 247.4s |  | 247.4s |
| 42 | agents/reasoning_stacking | 4 (PASS) | 23.6s |  | 23.6s |
| 43 | agents/reasoning_tool | 4 (PASS) | 54.5s |  | 54.5s |
| 44 | agents/reflexion | 1 (BROKEN) | 240.1s | F-LOOP | Timeout (240s) |
| 45 | agents/reflexion_code_gen | 4 (PASS) | 171.0s |  | 171.0s |
| 46 | agents/regulatory_compliance | 4 (PASS) | 88.5s |  | 88.5s |
| 47 | agents/rewoo | 4 (PASS) | 21.4s |  | 21.4s |
| 48 | agents/rewoo_multi_step | 4 (PASS) | 20.4s |  | 20.4s |
| 49 | agents/security_audit | 4 (PASS) | 35.1s |  | 35.1s |
| 50 | agents/self_consistency | 4 (PASS) | 26.8s |  | 26.8s |
| 51 | agents/skill_loader | 4 (PASS) | 64.5s |  | 64.5s |
| 52 | agents/structured_output | 4 (PASS) | 49.1s |  | 49.1s |
| 53 | agents/supply_chain_optimizer | 4 (PASS) | 20.6s |  | 20.6s |
| 54 | agents/tool_decorator | 4 (PASS) | 31.9s |  | 31.9s |
| 55 | agents/workflow_agent | 4 (PASS) | 16.2s |  | 16.2s |
| 56 | basic/form_filling | 4 (PASS) | 19.8s |  | 19.8s |
| 57 | basic/multi_turn_extraction | 4 (PASS) | 34.8s |  | 34.8s |
| 58 | basic/simple_greeting | 4 (PASS) | 18.6s |  | 18.6s |
| 59 | basic/story_time | 4 (PASS) | 32.6s |  | 32.6s |
| 60 | classification/classified_transitions | 4 (PASS) | 26.0s |  | 26.0s |
| 61 | classification/intent_routing | 4 (PASS) | 14.3s |  | 14.3s |
| 62 | classification/multi_intent | 4 (PASS) | 28.4s |  | 28.4s |
| 63 | classification/smart_helpdesk | 4 (PASS) | 20.0s |  | 20.0s |
| 64 | intermediate/adaptive_quiz | 4 (PASS) | 20.9s |  | 20.9s |
| 65 | intermediate/book_recommendation | 4 (PASS) | 21.8s |  | 21.8s |
| 66 | intermediate/product_recommendation | 4 (PASS) | 16.8s |  | 16.8s |
| 67 | meta/build_agent | 4 (PASS) | 7.8s |  | 7.8s |
| 68 | meta/build_fsm | 4 (PASS) | 7.3s |  | 7.3s |
| 69 | meta/build_workflow | 4 (PASS) | 6.6s |  | 6.6s |
| 70 | meta/meta_from_spec | 4 (PASS) | 3.5s |  | 3.5s |
| 71 | meta/meta_review_loop | 4 (PASS) | 29.1s |  | 29.1s |
| 72 | reasoning/math_tutor | 4 (PASS) | 11.4s |  | 11.4s |
| 73 | workflows/agent_workflow_chain | 4 (PASS) | 12.4s |  | 12.4s |
| 74 | workflows/conditional_branching | 4 (PASS) | 3.4s |  | 3.4s |
| 75 | workflows/customer_onboarding | 4 (PASS) | 34.2s |  | 34.2s |
| 76 | workflows/loan_processing | 4 (PASS) | 3.8s |  | 3.8s |
| 77 | workflows/order_processing | 4 (PASS) | 34.8s |  | 34.8s |
| 78 | workflows/parallel_steps | 4 (PASS) | 5.0s |  | 5.0s |
| 79 | workflows/release_management | 4 (PASS) | 3.7s |  | 3.7s |
| 80 | workflows/workflow_agent_loop | 4 (PASS) | 40.3s |  | 40.3s |

## Summary

- **Total examples**: 80
- **Score distribution**: 78x4, 1x3, 0x2, 1x1, 0x0
- **Health Score**: 316/320 = **98.8%**
- **Category breakdown**:
  - advanced: 28/28 (100%)
  - agents: 188/192 (98%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 20/20 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 32/32 (100%)
- **Top failure codes**: F-LOOP (2)

## Timing

- **Total wall time**: 5016.5s (sequential equivalent)
- **Fastest**: 3.4s
- **Slowest**: 247.4s
- **Mean**: 62.7s
