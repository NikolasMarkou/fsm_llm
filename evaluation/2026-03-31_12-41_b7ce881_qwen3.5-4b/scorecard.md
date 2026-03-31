# Evaluation: 2026-03-31 13:02

- **Date**: 2026-03-31 13:02
- **Git commit**: b7ce881
- **Model**: ollama_chat/qwen3.5:4b
- **Example count**: 80
- **Workers**: parallel execution
- **Evaluator**: scripts/eval.py (automated)

## Scores

| # | Example | Score | Duration | Failures | Notes |
|---|---------|-------|----------|----------|-------|
| 1 | advanced/concurrent_conversations | 4 (PASS) | 28.5s |  | 28.5s |
| 2 | advanced/context_compactor | 4 (PASS) | 26.4s |  | 26.4s |
| 3 | advanced/e_commerce | 4 (PASS) | 158.2s |  | 158.2s |
| 4 | advanced/handler_hooks | 4 (PASS) | 16.3s |  | 16.3s |
| 5 | advanced/multi_level_stack | 4 (PASS) | 23.9s |  | 23.9s |
| 6 | advanced/support_pipeline | 4 (PASS) | 51.7s |  | 51.7s |
| 7 | advanced/yoga_instructions | 4 (PASS) | 22.9s |  | 22.9s |
| 8 | agents/adapt | 4 (PASS) | 144.6s |  | 144.6s |
| 9 | agents/adapt_with_memory | 4 (PASS) | 183.0s |  | 183.0s |
| 10 | agents/agent_as_tool | 4 (PASS) | 112.4s |  | 112.4s |
| 11 | agents/agent_memory_chain | 4 (PASS) | 258.1s |  | 258.1s |
| 12 | agents/architecture_review | 4 (PASS) | 71.9s |  | 71.9s |
| 13 | agents/classified_dispatch | 4 (PASS) | 91.8s |  | 91.8s |
| 14 | agents/classified_tools | 4 (PASS) | 49.0s |  | 49.0s |
| 15 | agents/concurrent_react | 4 (PASS) | 52.9s |  | 52.9s |
| 16 | agents/consistency_with_tools | 4 (PASS) | 30.0s |  | 30.0s |
| 17 | agents/debate | 4 (PASS) | 117.9s |  | 117.9s |
| 18 | agents/debate_with_tools | 4 (PASS) | 123.5s |  | 123.5s |
| 19 | agents/eval_opt_structured | 4 (PASS) | 128.1s |  | 128.1s |
| 20 | agents/evaluator_optimizer | 4 (PASS) | 57.3s |  | 57.3s |
| 21 | agents/full_pipeline | 4 (PASS) | 110.7s |  | 110.7s |
| 22 | agents/hierarchical_orchestrator | 4 (PASS) | 20.2s |  | 20.2s |
| 23 | agents/hierarchical_tools | 4 (PASS) | 42.7s |  | 42.7s |
| 24 | agents/hitl_approval | 4 (PASS) | 86.5s |  | 86.5s |
| 25 | agents/investment_portfolio | 4 (PASS) | 34.0s |  | 34.0s |
| 26 | agents/legal_document_review | 3 (MOSTLY) | 48.3s | F-LOOP | 48.3s |
| 27 | agents/maker_checker | 4 (PASS) | 62.1s |  | 62.1s |
| 28 | agents/maker_checker_code | 4 (PASS) | 105.6s |  | 105.6s |
| 29 | agents/medical_literature | 4 (PASS) | 198.0s |  | 198.0s |
| 30 | agents/memory_agent | 4 (PASS) | 269.9s |  | 269.9s |
| 31 | agents/multi_debate_panel | 4 (PASS) | 88.8s |  | 88.8s |
| 32 | agents/multi_tool_recovery | 4 (PASS) | 58.3s |  | 58.3s |
| 33 | agents/orchestrator | 4 (PASS) | 17.2s |  | 17.2s |
| 34 | agents/orchestrator_specialist | 4 (PASS) | 15.3s |  | 15.3s |
| 35 | agents/pipeline_review | 4 (PASS) | 147.8s |  | 147.8s |
| 36 | agents/plan_execute | 4 (PASS) | 65.9s |  | 65.9s |
| 37 | agents/plan_execute_recovery | 4 (PASS) | 43.0s |  | 43.0s |
| 38 | agents/prompt_chain | 4 (PASS) | 51.2s |  | 51.2s |
| 39 | agents/react_hitl_combined | 4 (PASS) | 105.6s |  | 105.6s |
| 40 | agents/react_search | 4 (PASS) | 68.3s |  | 68.3s |
| 41 | agents/react_structured_pipeline | 4 (PASS) | 146.4s |  | 146.4s |
| 42 | agents/reasoning_stacking | 4 (PASS) | 24.1s |  | 24.1s |
| 43 | agents/reasoning_tool | 4 (PASS) | 31.5s |  | 31.5s |
| 44 | agents/reflexion | 4 (PASS) | 92.6s |  | 92.6s |
| 45 | agents/reflexion_code_gen | 4 (PASS) | 36.1s |  | 36.1s |
| 46 | agents/regulatory_compliance | 4 (PASS) | 72.8s |  | 72.8s |
| 47 | agents/rewoo | 4 (PASS) | 44.3s |  | 44.3s |
| 48 | agents/rewoo_multi_step | 4 (PASS) | 20.3s |  | 20.3s |
| 49 | agents/security_audit | 4 (PASS) | 83.0s |  | 83.0s |
| 50 | agents/self_consistency | 4 (PASS) | 24.4s |  | 24.4s |
| 51 | agents/skill_loader | 4 (PASS) | 62.1s |  | 62.1s |
| 52 | agents/structured_output | 4 (PASS) | 78.9s |  | 78.9s |
| 53 | agents/supply_chain_optimizer | 4 (PASS) | 27.0s |  | 27.0s |
| 54 | agents/tool_decorator | 4 (PASS) | 59.1s |  | 59.1s |
| 55 | agents/workflow_agent | 4 (PASS) | 10.5s |  | 10.5s |
| 56 | basic/form_filling | 4 (PASS) | 21.6s |  | 21.6s |
| 57 | basic/multi_turn_extraction | 4 (PASS) | 37.5s |  | 37.5s |
| 58 | basic/simple_greeting | 4 (PASS) | 18.4s |  | 18.4s |
| 59 | basic/story_time | 4 (PASS) | 32.0s |  | 32.0s |
| 60 | classification/classified_transitions | 4 (PASS) | 26.4s |  | 26.4s |
| 61 | classification/intent_routing | 4 (PASS) | 15.8s |  | 15.8s |
| 62 | classification/multi_intent | 4 (PASS) | 34.3s |  | 34.3s |
| 63 | classification/smart_helpdesk | 4 (PASS) | 22.6s |  | 22.6s |
| 64 | intermediate/adaptive_quiz | 4 (PASS) | 23.4s |  | 23.4s |
| 65 | intermediate/book_recommendation | 4 (PASS) | 27.8s |  | 27.8s |
| 66 | intermediate/product_recommendation | 4 (PASS) | 20.7s |  | 20.7s |
| 67 | meta/build_agent | 4 (PASS) | 3.7s |  | 3.7s |
| 68 | meta/build_fsm | 4 (PASS) | 4.0s |  | 4.0s |
| 69 | meta/build_workflow | 4 (PASS) | 3.9s |  | 3.9s |
| 70 | meta/meta_from_spec | 4 (PASS) | 154.5s |  | 154.5s |
| 71 | meta/meta_review_loop | 4 (PASS) | 112.7s |  | 112.7s |
| 72 | reasoning/math_tutor | 4 (PASS) | 9.6s |  | 9.6s |
| 73 | workflows/agent_workflow_chain | 4 (PASS) | 7.7s |  | 7.7s |
| 74 | workflows/conditional_branching | 4 (PASS) | 3.6s |  | 3.6s |
| 75 | workflows/customer_onboarding | 4 (PASS) | 35.5s |  | 35.5s |
| 76 | workflows/loan_processing | 4 (PASS) | 22.0s |  | 22.0s |
| 77 | workflows/order_processing | 4 (PASS) | 44.4s |  | 44.4s |
| 78 | workflows/parallel_steps | 4 (PASS) | 5.2s |  | 5.2s |
| 79 | workflows/release_management | 4 (PASS) | 76.1s |  | 76.1s |
| 80 | workflows/workflow_agent_loop | 4 (PASS) | 53.5s |  | 53.5s |

## Summary

- **Total examples**: 80
- **Score distribution**: 79x4, 1x3, 0x2, 0x1, 0x0
- **Health Score**: 319/320 = **99.7%**
- **Category breakdown**:
  - advanced: 28/28 (100%)
  - agents: 191/192 (99%)
  - basic: 16/16 (100%)
  - classification: 16/16 (100%)
  - intermediate: 12/12 (100%)
  - meta: 20/20 (100%)
  - reasoning: 4/4 (100%)
  - workflows: 32/32 (100%)
- **Top failure codes**: F-LOOP (1)

## Timing

- **Total wall time**: 5017.8s (sequential equivalent)
- **Fastest**: 3.6s
- **Slowest**: 269.9s
- **Mean**: 62.7s
