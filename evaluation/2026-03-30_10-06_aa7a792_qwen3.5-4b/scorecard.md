# Evaluation: 2026-03-30 10:33

- **Date**: 2026-03-30 10:33
- **Git commit**: aa7a792
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 80
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 25.4s |  | 25.4s |
| 2 | advanced/context_compactor | 4 (PASS) | 24.7s |  | 24.7s |
| 3 | advanced/e_commerce | 4 (PASS) | 182.3s |  | 182.3s |
| 4 | advanced/handler_hooks | 4 (PASS) | 19.6s |  | 19.6s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 25.4s |  | 25.4s |
| 6 | advanced/support_pipeline | 4 (PASS) | 49.5s |  | 49.5s |
| 7 | advanced/yoga_instructions | 4 (PASS) | 21.2s |  | 21.2s |
| 8 | agents/adapt | 1 (BROKEN) | 240.0s | F-LOOP | Timeout (240s) |
| 9 | agents/adapt_with_memory | 4 (PASS) | 211.5s |  | 211.5s |
| 10 | agents/agent_as_tool | 4 (PASS) | 49.8s |  | 49.8s |
| 11 | agents/agent_memory_chain | 1 (BROKEN) | 300.1s | F-LOOP | Timeout (300s) |
| 12 | agents/architecture_review | 4 (PASS) | 239.8s |  | 239.8s |
| 13 | agents/classified_dispatch | 4 (PASS) | 93.7s |  | 93.7s |
| 14 | agents/classified_tools | 4 (PASS) | 62.8s |  | 62.8s |
| 15 | agents/concurrent_react | 4 (PASS) | 173.9s |  | 173.9s |
| 16 | agents/consistency_with_tools | 4 (PASS) | 49.9s |  | 49.9s |
| 17 | agents/debate | 4 (PASS) | 135.8s |  | 135.8s |
| 18 | agents/debate_with_tools | 4 (PASS) | 128.2s |  | 128.2s |
| 19 | agents/eval_opt_structured | 1 (BROKEN) | 300.1s | F-LOOP | Timeout (300s) |
| 20 | agents/evaluator_optimizer | 4 (PASS) | 92.1s |  | 92.1s |
| 21 | agents/full_pipeline | 4 (PASS) | 148.2s |  | 148.2s |
| 22 | agents/hierarchical_orchestrator | 4 (PASS) | 17.5s |  | 17.5s |
| 23 | agents/hierarchical_tools | 4 (PASS) | 58.8s |  | 58.8s |
| 24 | agents/hitl_approval | 4 (PASS) | 83.5s |  | 83.5s |
| 25 | agents/investment_portfolio | 4 (PASS) | 43.6s |  | 43.6s |
| 26 | agents/legal_document_review | 3 (MOSTLY) | 35.1s | F-LOOP | 35.1s |
| 27 | agents/maker_checker | 4 (PASS) | 77.7s |  | 77.7s |
| 28 | agents/maker_checker_code | 4 (PASS) | 102.0s |  | 102.0s |
| 29 | agents/medical_literature | 4 (PASS) | 202.7s |  | 202.7s |
| 30 | agents/memory_agent | 1 (BROKEN) | 300.0s | F-LOOP | Timeout (300s) |
| 31 | agents/multi_debate_panel | 4 (PASS) | 103.0s |  | 103.0s |
| 32 | agents/multi_tool_recovery | 4 (PASS) | 96.3s |  | 96.3s |
| 33 | agents/orchestrator | 4 (PASS) | 17.1s |  | 17.1s |
| 34 | agents/orchestrator_specialist | 4 (PASS) | 17.2s |  | 17.2s |
| 35 | agents/pipeline_review | 4 (PASS) | 110.3s |  | 110.3s |
| 36 | agents/plan_execute | 4 (PASS) | 57.1s |  | 57.1s |
| 37 | agents/plan_execute_recovery | 4 (PASS) | 51.4s |  | 51.4s |
| 38 | agents/prompt_chain | 4 (PASS) | 30.5s |  | 30.5s |
| 39 | agents/react_hitl_combined | 4 (PASS) | 180.5s |  | 180.5s |
| 40 | agents/react_search | 1 (BROKEN) | 180.1s | F-LOOP | Timeout (180s) |
| 41 | agents/react_structured_pipeline | 4 (PASS) | 287.1s |  | 287.1s |
| 42 | agents/reasoning_stacking | 4 (PASS) | 39.4s |  | 39.4s |
| 43 | agents/reasoning_tool | 4 (PASS) | 29.0s |  | 29.0s |
| 44 | agents/reflexion | 4 (PASS) | 159.4s |  | 159.4s |
| 45 | agents/reflexion_code_gen | 4 (PASS) | 51.1s |  | 51.1s |
| 46 | agents/regulatory_compliance | 4 (PASS) | 61.3s |  | 61.3s |
| 47 | agents/rewoo | 4 (PASS) | 18.1s |  | 18.1s |
| 48 | agents/rewoo_multi_step | 4 (PASS) | 24.4s |  | 24.4s |
| 49 | agents/security_audit | 4 (PASS) | 102.4s |  | 102.4s |
| 50 | agents/self_consistency | 4 (PASS) | 33.6s |  | 33.6s |
| 51 | agents/skill_loader | 4 (PASS) | 106.2s |  | 106.2s |
| 52 | agents/structured_output | 3 (MOSTLY) | 67.7s | F-LOOP | 67.7s |
| 53 | agents/supply_chain_optimizer | 4 (PASS) | 21.5s |  | 21.5s |
| 54 | agents/tool_decorator | 4 (PASS) | 29.9s |  | 29.9s |
| 55 | agents/workflow_agent | 4 (PASS) | 7.8s |  | 7.8s |
| 56 | basic/form_filling | 4 (PASS) | 18.0s |  | 18.0s |
| 57 | basic/multi_turn_extraction | 4 (PASS) | 32.1s |  | 32.1s |
| 58 | basic/simple_greeting | 4 (PASS) | 17.9s |  | 17.9s |
| 59 | basic/story_time | 4 (PASS) | 28.5s |  | 28.5s |
| 60 | classification/classified_transitions | 4 (PASS) | 24.5s |  | 24.5s |
| 61 | classification/intent_routing | 4 (PASS) | 14.5s |  | 14.5s |
| 62 | classification/multi_intent | 4 (PASS) | 28.5s |  | 28.5s |
| 63 | classification/smart_helpdesk | 4 (PASS) | 22.2s |  | 22.2s |
| 64 | intermediate/adaptive_quiz | 4 (PASS) | 21.4s |  | 21.4s |
| 65 | intermediate/book_recommendation | 4 (PASS) | 21.9s |  | 21.9s |
| 66 | intermediate/product_recommendation | 4 (PASS) | 16.9s |  | 16.9s |
| 67 | meta/build_agent | 4 (PASS) | 7.5s |  | 7.5s |
| 68 | meta/build_fsm | 4 (PASS) | 8.2s |  | 8.2s |
| 69 | meta/build_workflow | 4 (PASS) | 7.6s |  | 7.6s |
| 70 | meta/meta_from_spec | 4 (PASS) | 203.9s |  | 203.9s |
| 71 | meta/meta_review_loop | 4 (PASS) | 188.8s |  | 188.8s |
| 72 | reasoning/math_tutor | 4 (PASS) | 37.4s |  | 37.4s |
| 73 | workflows/agent_workflow_chain | 4 (PASS) | 37.7s |  | 37.7s |
| 74 | workflows/conditional_branching | 4 (PASS) | 3.9s |  | 3.9s |
| 75 | workflows/customer_onboarding | 4 (PASS) | 160.7s |  | 160.7s |
| 76 | workflows/loan_processing | 3 (MOSTLY) | 44.8s | F-EXTRACT | 44.8s |
| 77 | workflows/order_processing | 4 (PASS) | 110.3s |  | 110.3s |
| 78 | workflows/parallel_steps | 4 (PASS) | 5.2s |  | 5.2s |
| 79 | workflows/release_management | 4 (PASS) | 37.8s |  | 37.8s |
| 80 | workflows/workflow_agent_loop | 4 (PASS) | 37.9s |  | 37.9s |

## Summary

- **Total examples**: 80
- **Score distribution**: 72x4, 3x3, 0x2, 5x1, 0x0
- **Health Score**: 302/320 = **94.4%**
- **Category breakdown**:
  - advanced: 28/28 (100%)
  - agents: 175/192 (91%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 20/20 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 31/32 (97%)
- **Top failure codes**: F-LOOP (7), F-EXTRACT (1)

## Timing

- **Total wall time**: 6516.0s (sequential equivalent)
- **Fastest**: 3.9s
- **Slowest**: 300.1s
- **Mean**: 81.5s
