"""ORCHESTRATOR FSM - Main control flow with retry management.

Part of the reasoning FSM definitions (see `reasoning_modes/__init__.py`).
"""

from fsm_llm.reasoning.constants import (
    ORCHESTRATOR_HANDLER_ONLY_KEYS,
    ContextKeys,
    Defaults,
    OrchestratorStates,
)

from ._phrases import _IMPORTANT_NO_QUESTIONS, _PRESENT_FINAL, _PRESENT_FINDINGS

orchestrator_fsm = {
    "name": "reasoning_orchestrator",
    "description": "Orchestrates various reasoning strategies with retry limits and loop prevention.",
    "initial_state": OrchestratorStates.PROBLEM_ANALYSIS,
    "persona": "You are a reasoning guide helping to solve problems step by step. Be clear, logical, and thorough.",
    # DECISION plan-2026-10-01T093600-944e2692/D-054: the validation verdict,
    # retry counters, confidence and strategy choice are handler-owned. Do NOT
    # take a key out of this list to let the model "help" (a bulk reply with
    # validation_result: true would open the gate) and do NOT list them in a
    # state's required_context_keys (never extracted, the validator warns).
    "handler_only_keys": list(ORCHESTRATOR_HANDLER_ONLY_KEYS),
    "states": {
        OrchestratorStates.PROBLEM_ANALYSIS: {
            "description": "Initial analysis of the problem",
            "purpose": f"Analyze the '{ContextKeys.PROBLEM_STATEMENT}' to identify '{ContextKeys.PROBLEM_TYPE}', '{ContextKeys.PROBLEM_COMPONENTS}', and '{ContextKeys.CONSTRAINTS}'",
            "required_context_keys": [
                ContextKeys.PROBLEM_TYPE,
                ContextKeys.PROBLEM_COMPONENTS,
            ],
            "extraction_instructions": f"""
            Break down the problem systematically. For simple arithmetic (e.g., '1+1'), set problem_type='arithmetic' and components as operands/operator.

            Identify:
            - The type of problem (arithmetic, logical, creative, analytical, etc.)
            - Key components or elements involved
            - Any constraints or limitations
            - The expected outcome or goal

            {_IMPORTANT_NO_QUESTIONS} Work with the problem statement provided in the context.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": OrchestratorStates.STRATEGY_SELECTION,
                    "description": "Problem analyzed successfully",
                    "priority": 1,
                    "conditions": [
                        {
                            "description": "Problem type and components identified",
                            "requires_context_keys": [
                                ContextKeys.PROBLEM_TYPE,
                                ContextKeys.PROBLEM_COMPONENTS,
                            ],
                        }
                    ],
                }
            ],
        },
        OrchestratorStates.STRATEGY_SELECTION: {
            "description": "Select appropriate reasoning strategy",
            "purpose": f"Choose '{ContextKeys.REASONING_STRATEGY}' based on problem analysis. For arithmetic, choose 'simple_calculator'.",
            "required_context_keys": [
                ContextKeys.REASONING_STRATEGY,
                ContextKeys.STRATEGY_RATIONALE,
            ],
            "extraction_instructions": f"""
            Select the most appropriate reasoning strategy based on the problem analysis:
            - If problem_type is 'arithmetic', set reasoning_strategy to 'simple_calculator'
            - For logical problems, consider 'deductive' or 'analytical'
            - For creative problems, use 'creative' or 'hybrid'
            - For evaluation tasks, use 'critical'
            - For pattern recognition, use 'inductive'
            - For explanation tasks, use 'abductive'
            - For similarity-based problems, use 'analogical'
            - For complex multi-faceted problems, use 'hybrid'

            Provide clear rationale for your choice.

            {_IMPORTANT_NO_QUESTIONS} Make the strategy selection based on your problem analysis.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": OrchestratorStates.EXECUTE_REASONING,
                    "description": "Strategy selected",
                }
            ],
        },
        OrchestratorStates.EXECUTE_REASONING: {
            "description": "Execute selected reasoning strategy",
            "purpose": "Apply the chosen reasoning approach through specialized FSM execution",
            "extraction_instructions": f"""
            The appropriate reasoning FSM will be executed here by handlers based on the selected strategy.
            This state serves as a coordination point for specialized reasoning execution.

            {_IMPORTANT_NO_QUESTIONS} The execution will be handled automatically.
            """,
            "response_instructions": "Acknowledge the current state of reasoning. Do not ask questions or request additional input from the user.",
            "transitions": [
                {
                    "target_state": OrchestratorStates.SYNTHESIZE_SOLUTION,
                    "description": "Reasoning completed",
                }
            ],
        },
        OrchestratorStates.SYNTHESIZE_SOLUTION: {
            "description": "Synthesize solution from reasoning results",
            "purpose": f"Create '{ContextKeys.PROPOSED_SOLUTION}' and '{ContextKeys.KEY_INSIGHTS}' from reasoning results",
            "required_context_keys": [
                ContextKeys.PROPOSED_SOLUTION,
                ContextKeys.KEY_INSIGHTS,
            ],
            "extraction_instructions": f"""
            Synthesize a comprehensive solution from the reasoning results:

            - If '{ContextKeys.CALCULATION_RESULT}' exists (from simple calculator), use it as the proposed_solution
            - If analytical results exist, synthesize from integrated_analysis and conclusions
            - If creative results exist, use the best_creative_solution
            - If critical results exist, incorporate the critical_assessment
            - Always provide key_insights as a list of important findings
            - Ensure the solution directly addresses the original problem

            Create a clear, actionable solution with supporting insights.

            {_IMPORTANT_NO_QUESTIONS} Synthesize based on available reasoning outputs.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": OrchestratorStates.VALIDATE_REFINE,
                    "description": "Solution synthesized",
                }
            ],
        },
        OrchestratorStates.VALIDATE_REFINE: {
            "description": "Validate solution with retry limit protection",
            "purpose": f"Check '{ContextKeys.VALIDATION_RESULT}' and retry if needed (max {Defaults.MAX_RETRIES} times)",
            "extraction_instructions": f"""
            Validate the proposed solution:

            - Check if the solution adequately addresses the original problem
            - Evaluate the logical consistency and completeness
            - Assess confidence level (1-10 scale)
            - Identify any significant gaps or errors
            - Consider if retry is warranted (only for serious issues)

            Handlers set validation_result and manage retry_count to prevent infinite loops.

            {_IMPORTANT_NO_QUESTIONS} Validate based on the solution quality and completeness.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": OrchestratorStates.FINAL_ANSWER,
                    "description": "Solution valid or max retries reached",
                    "priority": 1,
                    "conditions": [
                        {
                            "description": "Valid solution or retry limit hit",
                            "logic": {
                                "or": [
                                    {
                                        "==": [
                                            {"var": ContextKeys.VALIDATION_RESULT},
                                            True,
                                        ]
                                    },
                                    {
                                        "==": [
                                            {"var": ContextKeys.MAX_RETRIES_REACHED},
                                            True,
                                        ]
                                    },
                                ]
                            },
                        }
                    ],
                },
                {
                    "target_state": OrchestratorStates.EXECUTE_REASONING,
                    "description": "Retry reasoning (if under limit)",
                    "priority": 2,
                    "conditions": [
                        {
                            "description": "Invalid and can retry",
                            "logic": {
                                "and": [
                                    {
                                        "==": [
                                            {"var": ContextKeys.VALIDATION_RESULT},
                                            False,
                                        ]
                                    },
                                    {
                                        "!=": [
                                            {"var": ContextKeys.MAX_RETRIES_REACHED},
                                            True,
                                        ]
                                    },
                                ]
                            },
                        }
                    ],
                },
            ],
        },
        OrchestratorStates.FINAL_ANSWER: {
            "description": "Present final answer with complete reasoning trace",
            "purpose": f"Set '{ContextKeys.FINAL_SOLUTION}' and final metadata",
            "required_context_keys": [
                ContextKeys.FINAL_SOLUTION,
                ContextKeys.REASONING_TRACE,
            ],
            "extraction_instructions": f"""
            Present the final solution with complete context:

            Set final_solution to:
            - proposed_solution if validation_result is True
            - proposed_solution with retry warning if max_retries_reached but solution exists
            - 'Unable to find valid solution after maximum attempts' if no valid solution

            Include:
            - Complete reasoning_trace showing the path taken
            - Summary of key insights and approach used

            {_IMPORTANT_NO_QUESTIONS} Present the final solution based on the reasoning process.
            """,
            "response_instructions": _PRESENT_FINAL,
            "transitions": [],
        },
    },
}
