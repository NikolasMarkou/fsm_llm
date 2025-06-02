"""
Python dictionary definitions for all reasoning FSMs.
Improved with loop prevention and standardized structure.
"""
from .constants import (
    OrchestratorStates,
    ClassifierStates,
    ContextKeys,
    Defaults
)

# Orchestrator FSM with retry limit
orchestrator_fsm = {
    "name": "reasoning_orchestrator",
    "description": "Orchestrates various reasoning strategies with retry limits.",
    "initial_state": OrchestratorStates.PROBLEM_ANALYSIS,
    "persona": "You are a reasoning guide helping to solve problems step by step. Be clear, logical, and thorough.",
    "states": {
        OrchestratorStates.PROBLEM_ANALYSIS: {
            "id": OrchestratorStates.PROBLEM_ANALYSIS,
            "description": "Initial analysis of the problem",
            "purpose": f"Analyze the '{ContextKeys.PROBLEM_STATEMENT}' to identify '{ContextKeys.PROBLEM_TYPE}', '{ContextKeys.PROBLEM_COMPONENTS}', and '{ContextKeys.CONSTRAINTS}'",
            "required_context_keys": [ContextKeys.PROBLEM_TYPE, ContextKeys.PROBLEM_COMPONENTS],
            "instructions": "Break down the problem systematically. For simple arithmetic (e.g., '1+1'), set problem_type='arithmetic' and components as operands/operator.",
            "transitions": [{
                "target_state": OrchestratorStates.STRATEGY_SELECTION,
                "description": "Problem analyzed successfully",
                "priority": 1,
                "conditions": [{
                    "description": "Problem type and components identified",
                    "requires_context_keys": [ContextKeys.PROBLEM_TYPE, ContextKeys.PROBLEM_COMPONENTS]
                }]
            }]
        },
        OrchestratorStates.STRATEGY_SELECTION: {
            "id": OrchestratorStates.STRATEGY_SELECTION,
            "description": "Select appropriate reasoning strategy",
            "purpose": f"Choose '{ContextKeys.REASONING_STRATEGY}' based on problem analysis. For arithmetic, choose 'direct computation'.",
            "required_context_keys": [ContextKeys.REASONING_STRATEGY, ContextKeys.STRATEGY_RATIONALE],
            "instructions": "If problem_type is 'arithmetic', set reasoning_strategy to 'direct computation'. Otherwise use classified type or choose from available strategies.",
            "transitions": [{
                "target_state": OrchestratorStates.EXECUTE_REASONING,
                "description": "Strategy selected"
            }]
        },
        OrchestratorStates.EXECUTE_REASONING: {
            "id": OrchestratorStates.EXECUTE_REASONING,
            "description": "Execute selected reasoning strategy",
            "purpose": "Apply the chosen reasoning approach",
            "instructions": "The appropriate reasoning FSM will be executed here by handlers.",
            "transitions": [{
                "target_state": OrchestratorStates.SYNTHESIZE_SOLUTION,
                "description": "Reasoning completed"
            }]
        },
        OrchestratorStates.SYNTHESIZE_SOLUTION: {
            "id": OrchestratorStates.SYNTHESIZE_SOLUTION,
            "description": "Synthesize solution from reasoning results",
            "purpose": f"Create '{ContextKeys.PROPOSED_SOLUTION}' and '{ContextKeys.KEY_INSIGHTS}' from reasoning results",
            "required_context_keys": [ContextKeys.PROPOSED_SOLUTION, ContextKeys.KEY_INSIGHTS],
            "instructions": f"""Based on reasoning results:
- If '{ContextKeys.CALCULATION_RESULT}' exists (from simple calculator), use it as proposed_solution
- Otherwise, synthesize from available reasoning outputs (integrated_analysis, conclusions, etc.)
- Always provide key_insights list""",
            "transitions": [{
                "target_state": OrchestratorStates.VALIDATE_REFINE,
                "description": "Solution synthesized"
            }]
        },
        OrchestratorStates.VALIDATE_REFINE: {
            "id": OrchestratorStates.VALIDATE_REFINE,
            "description": "Validate solution with retry limit",
            "purpose": f"Check '{ContextKeys.VALIDATION_RESULT}' and retry if needed (max {Defaults.MAX_RETRIES} times)",
            "required_context_keys": [ContextKeys.VALIDATION_RESULT, ContextKeys.CONFIDENCE_LEVEL],
            "instructions": f"Handlers will set validation_result. Check retry_count to prevent infinite loops.",
            "transitions": [
                {
                    "target_state": OrchestratorStates.FINAL_ANSWER,
                    "description": "Solution valid or max retries reached",
                    "priority": 1,
                    "conditions": [{
                        "description": "Valid solution or retry limit hit",
                        "logic": {
                            "or": [
                                {
                                    "or": [
                                        {"==": [{"var": ContextKeys.VALIDATION_RESULT}, True]},
                                        {"==": [{"var": ContextKeys.SOLUTION_VALID}, True]}
                                    ]
                                },
                                {"==": [{"var": ContextKeys.MAX_RETRIES_REACHED}, True]}
                            ]
                        }
                    }]
                },
                {
                    "target_state": OrchestratorStates.EXECUTE_REASONING,
                    "description": "Retry reasoning (if under limit)",
                    "priority": 2,
                    "conditions": [{
                        "description": "Invalid and can retry",
                        "logic": {
                            "and": [
                                {
                                    "and": [
                                        {"==": [{"var": ContextKeys.VALIDATION_RESULT}, False]},
                                        {"==": [{"var": ContextKeys.SOLUTION_VALID}, False]}
                                    ]
                                },
                                {"!=": [{"var": ContextKeys.MAX_RETRIES_REACHED}, True]}
                            ]
                        }
                    }]
                }
            ]
        },
        OrchestratorStates.FINAL_ANSWER: {
            "id": OrchestratorStates.FINAL_ANSWER,
            "description": "Present final answer",
            "purpose": f"Set '{ContextKeys.FINAL_SOLUTION}' and final metadata",
            "required_context_keys": [ContextKeys.FINAL_SOLUTION, ContextKeys.REASONING_TRACE, ContextKeys.SOLUTION_CONFIDENCE],
            "instructions": f"""Set final_solution to:
- proposed_solution if solution_valid is true
- proposed_solution with retry warning if max_retries_reached
- 'Unable to find valid solution' if neither

Copy solution_confidence and reasoning_trace from context.""",
            "transitions": []
        }
    }
}

# Classifier FSM
classifier_fsm = {
    "name": "problem_classifier",
    "description": "Classifies problems to determine reasoning strategy",
    "initial_state": ClassifierStates.ANALYZE_DOMAIN,
    "persona": "You are an expert problem analyst who identifies the best reasoning approach.",
    "states": {
        ClassifierStates.ANALYZE_DOMAIN: {
            "id": ClassifierStates.ANALYZE_DOMAIN,
            "description": "Identify problem domain",
            "purpose": f"Determine '{ContextKeys.PROBLEM_DOMAIN}' and '{ContextKeys.DOMAIN_INDICATORS}'",
            "required_context_keys": [ContextKeys.PROBLEM_DOMAIN, ContextKeys.DOMAIN_INDICATORS],
            "transitions": [{
                "target_state": ClassifierStates.ANALYZE_STRUCTURE,
                "description": "Domain identified"
            }]
        },
        ClassifierStates.ANALYZE_STRUCTURE: {
            "id": ClassifierStates.ANALYZE_STRUCTURE,
            "description": "Analyze problem structure",
            "purpose": f"Identify '{ContextKeys.PROBLEM_STRUCTURE}' and '{ContextKeys.STRUCTURAL_ELEMENTS}'",
            "required_context_keys": [ContextKeys.PROBLEM_STRUCTURE, ContextKeys.STRUCTURAL_ELEMENTS],
            "transitions": [{
                "target_state": ClassifierStates.IDENTIFY_REASONING_NEEDS,
                "description": "Structure analyzed"
            }]
        },
        ClassifierStates.IDENTIFY_REASONING_NEEDS: {
            "id": ClassifierStates.IDENTIFY_REASONING_NEEDS,
            "description": "Identify reasoning requirements",
            "purpose": f"Determine '{ContextKeys.REASONING_REQUIREMENTS}' and '{ContextKeys.KEY_CHALLENGES}'",
            "required_context_keys": [ContextKeys.REASONING_REQUIREMENTS, ContextKeys.KEY_CHALLENGES],
            "instructions": "For simple calculations, set reasoning_requirements='direct computation'",
            "transitions": [{
                "target_state": ClassifierStates.RECOMMEND_STRATEGY,
                "description": "Needs identified"
            }]
        },
        ClassifierStates.RECOMMEND_STRATEGY: {
            "id": ClassifierStates.RECOMMEND_STRATEGY,
            "description": "Recommend reasoning strategy",
            "purpose": f"Set '{ContextKeys.RECOMMENDED_REASONING_TYPE}', '{ContextKeys.STRATEGY_JUSTIFICATION}', and '{ContextKeys.ALTERNATIVE_APPROACHES}'",
            "required_context_keys": [
                ContextKeys.RECOMMENDED_REASONING_TYPE,
                ContextKeys.STRATEGY_JUSTIFICATION,
                ContextKeys.ALTERNATIVE_APPROACHES
            ],
            "instructions": "Choose from: analytical, deductive, inductive, creative, critical, hybrid, simple_calculator",
            "transitions": []
        }
    }
}

# Simple Calculator FSM
simple_calculator_fsm = {
    "name": "simple_calculator",
    "description": "Performs simple arithmetic calculations",
    "initial_state": "extract_elements",
    "persona": "You are a precise calculator.",
    "states": {
        "extract_elements": {
            "id": "extract_elements",
            "description": "Extract operands and operator",
            "purpose": f"Extract '{ContextKeys.OPERAND1}', '{ContextKeys.OPERAND2}', and '{ContextKeys.OPERATOR}'",
            "required_context_keys": [ContextKeys.OPERAND1, ContextKeys.OPERAND2, ContextKeys.OPERATOR],
            "instructions": "Extract numbers and operator (+, -, *, /) from problem_components",
            "transitions": [{
                "target_state": "perform_calculation",
                "description": "Elements extracted"
            }]
        },
        "perform_calculation": {
            "id": "perform_calculation",
            "description": "Calculate result",
            "purpose": f"Calculate and store in '{ContextKeys.CALCULATION_RESULT}'",
            "required_context_keys": [ContextKeys.CALCULATION_RESULT],
            "instructions": "Perform arithmetic and handle errors (e.g., division by zero)",
            "transitions": []
        }
    }
}

# Analytical FSM (simplified for brevity)
analytical_fsm = {
    "name": "analytical_reasoning",
    "description": "Analytical reasoning through decomposition",
    "initial_state": "decompose",
    "persona": "You are a methodical analytical thinker.",
    "states": {
        "decompose": {
            "id": "decompose",
            "description": "Break down the problem",
            "purpose": f"Identify '{ContextKeys.COMPONENTS}', '{ContextKeys.ATTRIBUTES}', and '{ContextKeys.RELATIONSHIPS}'",
            "required_context_keys": [ContextKeys.COMPONENTS, ContextKeys.ATTRIBUTES, ContextKeys.RELATIONSHIPS],
            "transitions": [{
                "target_state": "analyze_components",
                "description": "Decomposition complete"
            }]
        },
        "analyze_components": {
            "id": "analyze_components",
            "description": "Analyze each component",
            "purpose": f"Create '{ContextKeys.COMPONENT_ANALYSIS}' and identify '{ContextKeys.DATA_REQUIREMENTS}'",
            "required_context_keys": [ContextKeys.COMPONENT_ANALYSIS, ContextKeys.DATA_REQUIREMENTS],
            "transitions": [{
                "target_state": "identify_patterns",
                "description": "Analysis complete"
            }]
        },
        "identify_patterns": {
            "id": "identify_patterns",
            "description": "Find patterns and dependencies",
            "purpose": f"Identify '{ContextKeys.PATTERNS}', '{ContextKeys.CAUSAL_LINKS}', and '{ContextKeys.DEPENDENCIES}'",
            "required_context_keys": [ContextKeys.PATTERNS, ContextKeys.CAUSAL_LINKS, ContextKeys.DEPENDENCIES],
            "transitions": [{
                "target_state": "integrate_findings",
                "description": "Patterns identified"
            }]
        },
        "integrate_findings": {
            "id": "integrate_findings",
            "description": "Synthesize understanding",
            "purpose": f"Create '{ContextKeys.INTEGRATED_ANALYSIS}' and '{ContextKeys.KEY_INSIGHTS}'",
            "required_context_keys": [ContextKeys.INTEGRATED_ANALYSIS, ContextKeys.KEY_INSIGHTS],
            "transitions": []
        }
    }
}

# Other FSMs follow similar pattern...
# I'll include just the structure for brevity

deductive_fsm = {
    "name": "deductive_reasoning",
    "description": "Deductive reasoning from general to specific",
    "initial_state": "identify_premises",
    "persona": "You are a logical thinker.",
    "states": {
        "identify_premises": {
            "id": "identify_premises",
            "required_context_keys": [ContextKeys.PREMISES, ContextKeys.ASSUMPTIONS],
            "transitions": [{"target_state": "apply_logic"}]
        },
        "apply_logic": {
            "id": "apply_logic",
            "required_context_keys": [ContextKeys.LOGICAL_STEPS, ContextKeys.INTERMEDIATE_CONCLUSIONS],
            "transitions": [{"target_state": "derive_conclusion"}]
        },
        "derive_conclusion": {
            "id": "derive_conclusion",
            "required_context_keys": [ContextKeys.CONCLUSION, ContextKeys.LOGICAL_VALIDITY],
            "transitions": []
        }
    }
}

inductive_fsm = {
    "name": "inductive_reasoning",
    "description": "Inductive reasoning from specific to general",
    "initial_state": "gather_observations",
    "persona": "You are an empirical thinker.",
    "states": {
        "gather_observations": {
            "id": "gather_observations",
            "required_context_keys": [ContextKeys.OBSERVATIONS, ContextKeys.DATA_POINTS],
            "transitions": [{"target_state": "identify_commonalities"}]
        },
        "identify_commonalities": {
            "id": "identify_commonalities",
            "required_context_keys": [ContextKeys.COMMONALITIES, ContextKeys.TRENDS],
            "transitions": [{"target_state": "form_hypothesis"}]
        },
        "form_hypothesis": {
            "id": "form_hypothesis",
            "required_context_keys": [ContextKeys.HYPOTHESIS, ContextKeys.SUPPORTING_EVIDENCE],
            "transitions": [{"target_state": "test_generalization"}]
        },
        "test_generalization": {
            "id": "test_generalization",
            "required_context_keys": [ContextKeys.TEST_RESULTS, ContextKeys.COUNTER_EXAMPLES, ContextKeys.GENERALIZATION_STRENGTH],
            "transitions": []
        }
    }
}

creative_fsm = {
    "name": "creative_reasoning",
    "description": "Creative reasoning for novel solutions",
    "initial_state": "explore_perspectives",
    "persona": "You are an imaginative thinker.",
    "states": {
        "explore_perspectives": {
            "id": "explore_perspectives",
            "required_context_keys": [ContextKeys.PERSPECTIVES, ContextKeys.REFRAMINGS],
            "transitions": [{"target_state": "generate_ideas"}]
        },
        "generate_ideas": {
            "id": "generate_ideas",
            "required_context_keys": [ContextKeys.CREATIVE_IDEAS, ContextKeys.UNCONVENTIONAL_APPROACHES],
            "transitions": [{"target_state": "combine_concepts"}]
        },
        "combine_concepts": {
            "id": "combine_concepts",
            "required_context_keys": [ContextKeys.COMBINATIONS, ContextKeys.NOVEL_SOLUTIONS],
            "transitions": [{"target_state": "evaluate_novelty"}]
        },
        "evaluate_novelty": {
            "id": "evaluate_novelty",
            "required_context_keys": [ContextKeys.BEST_CREATIVE_SOLUTION, ContextKeys.INNOVATION_RATING],
            "transitions": []
        }
    }
}

critical_fsm = {
    "name": "critical_reasoning",
    "description": "Critical evaluation of arguments",
    "initial_state": "identify_claims",
    "persona": "You are a critical thinker.",
    "states": {
        "identify_claims": {
            "id": "identify_claims",
            "required_context_keys": [ContextKeys.CLAIMS, ContextKeys.ARGUMENTS],
            "transitions": [{"target_state": "examine_evidence"}]
        },
        "examine_evidence": {
            "id": "examine_evidence",
            "required_context_keys": [ContextKeys.EVIDENCE_QUALITY, ContextKeys.EVIDENCE_GAPS],
            "transitions": [{"target_state": "analyze_logic"}]
        },
        "analyze_logic": {
            "id": "analyze_logic",
            "required_context_keys": [ContextKeys.LOGICAL_ANALYSIS, ContextKeys.ASSUMPTIONS, ContextKeys.FALLACIES],
            "transitions": [{"target_state": "consider_alternatives"}]
        },
        "consider_alternatives": {
            "id": "consider_alternatives",
            "required_context_keys": [ContextKeys.ALTERNATIVE_EXPLANATIONS, ContextKeys.COUNTER_ARGUMENTS],
            "transitions": [{"target_state": "form_judgment"}]
        },
        "form_judgment": {
            "id": "form_judgment",
            "required_context_keys": [ContextKeys.CRITICAL_ASSESSMENT, ContextKeys.CONFIDENCE_RATING],
            "transitions": []
        }
    }
}

# Hybrid FSM with loop limit
hybrid_fsm = {
    "name": "hybrid_reasoning",
    "description": "Combines multiple reasoning approaches with loop prevention",
    "initial_state": "identify_components",
    "persona": "You are a master strategist.",
    "states": {
        "identify_components": {
            "id": "identify_components",
            "description": "Identify problem aspects",
            "purpose": f"Map '{ContextKeys.PROBLEM_ASPECTS}' to reasoning types in '{ContextKeys.REASONING_MAP}'",
            "required_context_keys": [ContextKeys.PROBLEM_ASPECTS, ContextKeys.REASONING_MAP],
            "transitions": [{
                "target_state": "apply_analytical",
                "description": "Components identified"
            }]
        },
        "apply_analytical": {
            "id": "apply_analytical",
            "required_context_keys": [ContextKeys.ANALYTICAL_BREAKDOWN, ContextKeys.COMPONENT_RELATIONSHIPS],
            "transitions": [{"target_state": "apply_logical"}]
        },
        "apply_logical": {
            "id": "apply_logical",
            "required_context_keys": [ContextKeys.LOGICAL_CONCLUSIONS, ContextKeys.REASONING_CHAIN],
            "transitions": [{"target_state": "apply_creative"}]
        },
        "apply_creative": {
            "id": "apply_creative",
            "required_context_keys": [ContextKeys.CREATIVE_INSIGHTS, ContextKeys.NOVEL_APPROACHES],
            "transitions": [{"target_state": "critical_evaluation"}]
        },
        "critical_evaluation": {
            "id": "critical_evaluation",
            "description": "Evaluate all findings",
            "purpose": f"Create '{ContextKeys.EVALUATION_RESULTS}' and check if refinement needed",
            "required_context_keys": [ContextKeys.EVALUATION_RESULTS],
            "instructions": "Only loop back if critical issues found AND hybrid_loop_count < 2",
            "transitions": [
                {
                    "target_state": "integrate_solution",
                    "description": "Ready to integrate",
                    "priority": 1,
                    "conditions": [{
                        "description": "No critical issues or loop limit reached",
                        "logic": {
                            "or": [
                                {"!=": [{"var": "needs_refinement"}, True]},
                                {">=": [{"var": "hybrid_loop_count"}, 2]}
                            ]
                        }
                    }]
                },
                {
                    "target_state": "identify_components",
                    "description": "Refinement needed (limited)",
                    "priority": 2,
                    "conditions": [{
                        "description": "Critical issues and can loop",
                        "logic": {
                            "and": [
                                {"==": [{"var": "needs_refinement"}, True]},
                                {"<": [{"var": "hybrid_loop_count"}, 2]}
                            ]
                        }
                    }]
                }
            ]
        },
        "integrate_solution": {
            "id": "integrate_solution",
            "required_context_keys": [ContextKeys.INTEGRATED_SOLUTION, ContextKeys.REASONING_SYNTHESIS_NOTES],
            "transitions": [{"target_state": "finalize_hybrid"}]
        },
        "finalize_hybrid": {
            "id": "finalize_hybrid",
            "required_context_keys": [ContextKeys.FINAL_HYBRID_SOLUTION, ContextKeys.REASONING_SYNTHESIS],
            "transitions": []
        }
    }
}

# Abductive and Analogical FSMs (structure only)
abductive_fsm = {
    "name": "abductive_reasoning",
    "description": "Find best explanation",
    "initial_state": "identify_observations",
    "persona": "You are a detective seeking explanations.",
    "states": {
        "identify_observations": {
            "id": "identify_observations",
            "required_context_keys": [ContextKeys.OBSERVATIONS, "surprising_elements"],
            "transitions": [{"target_state": "generate_hypotheses"}]
        },
        "generate_hypotheses": {
            "id": "generate_hypotheses",
            "required_context_keys": ["potential_hypotheses", "hypothesis_rationales"],
            "transitions": [{"target_state": "evaluate_hypotheses"}]
        },
        "evaluate_hypotheses": {
            "id": "evaluate_hypotheses",
            "required_context_keys": ["hypothesis_evaluations", "evaluation_criteria"],
            "transitions": [{"target_state": "select_best_explanation"}]
        },
        "select_best_explanation": {
            "id": "select_best_explanation",
            "required_context_keys": ["best_hypothesis", "selection_justification", "confidence_in_explanation", "next_steps_for_validation"],
            "transitions": []
        }
    }
}

analogical_fsm = {
    "name": "analogical_reasoning",
    "description": "Transfer insights via analogy",
    "initial_state": "define_target_problem",
    "persona": "You are an expert at finding connections.",
    "states": {
        "define_target_problem": {
            "id": "define_target_problem",
            "required_context_keys": ["target_problem_description", "key_features_of_target"],
            "transitions": [{"target_state": "find_source_analogs"}]
        },
        "find_source_analogs": {
            "id": "find_source_analogs",
            "required_context_keys": ["potential_analogs", "rationale_for_choice", "similarity_criteria_used"],
            "transitions": [{"target_state": "map_correspondences"}]
        },
        "map_correspondences": {
            "id": "map_correspondences",
            "required_context_keys": ["selected_analog", "analogical_mapping", "identified_similarities", "identified_differences"],
            "transitions": [{"target_state": "transfer_insights"}]
        },
        "transfer_insights": {
            "id": "transfer_insights",
            "required_context_keys": ["transferred_insights_or_solutions", "potential_inferences"],
            "transitions": [{"target_state": "evaluate_analogy_fit"}]
        },
        "evaluate_analogy_fit": {
            "id": "evaluate_analogy_fit",
            "required_context_keys": ["analogy_strengths", "analogy_weaknesses_or_limitations", "adapted_solution_or_understanding", "analogy_confidence_rating"],
            "transitions": []
        }
    }
}

# Dictionary for easy access
ALL_REASONING_FSMS = {
    "orchestrator": orchestrator_fsm,
    "classifier": classifier_fsm,
    "analytical": analytical_fsm,
    "deductive": deductive_fsm,
    "inductive": inductive_fsm,
    "creative": creative_fsm,
    "critical": critical_fsm,
    "hybrid": hybrid_fsm,
    "simple_calculator": simple_calculator_fsm,
    "abductive": abductive_fsm,
    "analogical": analogical_fsm,
}