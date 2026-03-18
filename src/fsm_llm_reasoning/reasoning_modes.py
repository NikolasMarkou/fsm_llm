"""
Complete FSM definitions for the reasoning engine.

This module contains all finite state machine definitions for various reasoning strategies,
including orchestration, classification, and specialized reasoning approaches. Each FSM
is designed to work autonomously without requiring user input during execution.

The FSMs are structured to work with the LLM-FSM framework and include:
- Orchestrator FSM: Main control flow with retry limits
- Classifier FSM: Problem classification and strategy recommendation
- Specialized reasoning FSMs: Analytical, deductive, inductive, creative, critical, hybrid, abductive, analogical
- Simple calculator FSM: Basic arithmetic operations

All FSMs use standardized context keys and include explicit instructions to prevent
user input requests during autonomous reasoning.

Author: Generated for reasoning engine
Python Version: 3.11+
Dependencies: llm-fsm framework, constants module
"""

from .constants import (
    OrchestratorStates,
    ClassifierStates,
    ContextKeys,
    Defaults
)


# ============================================================================
# ORCHESTRATOR FSM - Main control flow with retry management
# ============================================================================

orchestrator_fsm = {
    "name": "reasoning_orchestrator",
    "description": "Orchestrates various reasoning strategies with retry limits and loop prevention.",
    "initial_state": OrchestratorStates.PROBLEM_ANALYSIS,
    "persona": "You are a reasoning guide helping to solve problems step by step. Be clear, logical, and thorough.",
    "states": {
        OrchestratorStates.PROBLEM_ANALYSIS: {
            "id": OrchestratorStates.PROBLEM_ANALYSIS,
            "description": "Initial analysis of the problem",
            "purpose": f"Analyze the '{ContextKeys.PROBLEM_STATEMENT}' to identify '{ContextKeys.PROBLEM_TYPE}', '{ContextKeys.PROBLEM_COMPONENTS}', and '{ContextKeys.CONSTRAINTS}'",
            "required_context_keys": [ContextKeys.PROBLEM_TYPE, ContextKeys.PROBLEM_COMPONENTS],
            "instructions": f"""
            Break down the problem systematically. For simple arithmetic (e.g., '1+1'), set problem_type='arithmetic' and components as operands/operator.
            
            Identify:
            - The type of problem (arithmetic, logical, creative, analytical, etc.)
            - Key components or elements involved
            - Any constraints or limitations
            - The expected outcome or goal
            
            IMPORTANT: Do not ask questions or request additional input from the user. Work with the problem statement provided in the context.
            """,
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
            "instructions": """
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
            
            IMPORTANT: Do not ask questions or request additional input from the user. Make the strategy selection based on your problem analysis.
            """,
            "transitions": [{
                "target_state": OrchestratorStates.EXECUTE_REASONING,
                "description": "Strategy selected"
            }]
        },
        OrchestratorStates.EXECUTE_REASONING: {
            "id": OrchestratorStates.EXECUTE_REASONING,
            "description": "Execute selected reasoning strategy",
            "purpose": "Apply the chosen reasoning approach through specialized FSM execution",
            "instructions": """
            The appropriate reasoning FSM will be executed here by handlers based on the selected strategy.
            This state serves as a coordination point for specialized reasoning execution.
            
            IMPORTANT: Do not ask questions or request additional input from the user. The execution will be handled automatically.
            """,
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
            "instructions": f"""
            Synthesize a comprehensive solution from the reasoning results:
            
            - If '{ContextKeys.CALCULATION_RESULT}' exists (from simple calculator), use it as the proposed_solution
            - If analytical results exist, synthesize from integrated_analysis and conclusions
            - If creative results exist, use the best_creative_solution
            - If critical results exist, incorporate the critical_assessment
            - Always provide key_insights as a list of important findings
            - Ensure the solution directly addresses the original problem
            
            Create a clear, actionable solution with supporting insights.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Synthesize based on available reasoning outputs.
            """,
            "transitions": [{
                "target_state": OrchestratorStates.VALIDATE_REFINE,
                "description": "Solution synthesized"
            }]
        },
        OrchestratorStates.VALIDATE_REFINE: {
            "id": OrchestratorStates.VALIDATE_REFINE,
            "description": "Validate solution with retry limit protection",
            "purpose": f"Check '{ContextKeys.VALIDATION_RESULT}' and retry if needed (max {Defaults.MAX_RETRIES} times)",
            "required_context_keys": [ContextKeys.VALIDATION_RESULT, ContextKeys.CONFIDENCE_LEVEL],
            "instructions": f"""
            Validate the proposed solution:
            
            - Check if the solution adequately addresses the original problem
            - Evaluate the logical consistency and completeness
            - Assess confidence level (1-10 scale)
            - Identify any significant gaps or errors
            - Consider if retry is warranted (only for serious issues)
            
            Handlers will manage retry_count to prevent infinite loops. Set validation_result to True/False.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Validate based on the solution quality and completeness.
            """,
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
            "description": "Present final answer with complete reasoning trace",
            "purpose": f"Set '{ContextKeys.FINAL_SOLUTION}' and final metadata",
            "required_context_keys": [ContextKeys.FINAL_SOLUTION, ContextKeys.REASONING_TRACE, ContextKeys.SOLUTION_CONFIDENCE],
            "instructions": f"""
            Present the final solution with complete context:
            
            Set final_solution to:
            - proposed_solution if solution_valid is True
            - proposed_solution with retry warning if max_retries_reached but solution exists
            - 'Unable to find valid solution after maximum attempts' if no valid solution
            
            Include:
            - Copy solution_confidence from validation
            - Complete reasoning_trace showing the path taken
            - Summary of key insights and approach used
            
            IMPORTANT: Do not ask questions or request additional input from the user. Present the final solution based on the reasoning process.
            """,
            "transitions": []
        }
    }
}


# ============================================================================
# CLASSIFIER FSM - Problem classification and strategy recommendation
# ============================================================================

classifier_fsm = {
    "name": "problem_classifier",
    "description": "Classifies problems to determine the most appropriate reasoning strategy",
    "initial_state": ClassifierStates.ANALYZE_DOMAIN,
    "persona": "You are an expert problem analyst who identifies the best reasoning approach for any given problem.",
    "states": {
        ClassifierStates.ANALYZE_DOMAIN: {
            "id": ClassifierStates.ANALYZE_DOMAIN,
            "description": "Identify problem domain and context",
            "purpose": f"Determine '{ContextKeys.PROBLEM_DOMAIN}' and '{ContextKeys.DOMAIN_INDICATORS}'",
            "required_context_keys": [ContextKeys.PROBLEM_DOMAIN, ContextKeys.DOMAIN_INDICATORS],
            "instructions": """
            Analyze the problem domain:
            
            Identify the primary domain (mathematics, logic, creativity, analysis, evaluation, etc.) and specific indicators:
            - Mathematical: Contains numbers, operations, calculations, formulas
            - Logical: Involves premises, conclusions, if-then relationships
            - Creative: Requires novel solutions, brainstorming, innovation
            - Analytical: Needs breakdown, decomposition, systematic analysis
            - Critical: Involves evaluation, judgment, argument assessment
            - Empirical: Based on observations, patterns, data analysis
            - Explanatory: Seeks to explain phenomena or observations
            - Comparative: Involves analogies, similarities, pattern matching
            
            IMPORTANT: Do not ask questions or request additional input from the user. Analyze the domain based on the problem statement provided.
            """,
            "transitions": [{
                "target_state": ClassifierStates.ANALYZE_STRUCTURE,
                "description": "Domain identified"
            }]
        },
        ClassifierStates.ANALYZE_STRUCTURE: {
            "id": ClassifierStates.ANALYZE_STRUCTURE,
            "description": "Analyze problem structure and complexity",
            "purpose": f"Identify '{ContextKeys.PROBLEM_STRUCTURE}' and '{ContextKeys.STRUCTURAL_ELEMENTS}'",
            "required_context_keys": [ContextKeys.PROBLEM_STRUCTURE, ContextKeys.STRUCTURAL_ELEMENTS],
            "instructions": """
            Analyze the structural characteristics:
            
            Identify structure type and key elements:
            - Simple: Single-step, direct solution path
            - Sequential: Multi-step process with clear order
            - Hierarchical: Nested components with dependencies
            - Network: Multiple interconnected elements
            - Complex: Multi-faceted with various approaches needed
            
            Document structural elements like components, relationships, dependencies, constraints.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Analyze structure from the given problem.
            """,
            "transitions": [{
                "target_state": ClassifierStates.IDENTIFY_REASONING_NEEDS,
                "description": "Structure analyzed"
            }]
        },
        ClassifierStates.IDENTIFY_REASONING_NEEDS: {
            "id": ClassifierStates.IDENTIFY_REASONING_NEEDS,
            "description": "Identify specific reasoning requirements",
            "purpose": f"Determine '{ContextKeys.REASONING_REQUIREMENTS}' and '{ContextKeys.KEY_CHALLENGES}'",
            "required_context_keys": [ContextKeys.REASONING_REQUIREMENTS, ContextKeys.KEY_CHALLENGES],
            "instructions": """
            Identify what type of reasoning is needed:
            
            For simple calculations, set reasoning_requirements='direct computation'.
            For other problems, identify specific needs:
            - Decomposition and analysis
            - Logical deduction from premises
            - Pattern recognition and generalization
            - Creative solution generation
            - Critical evaluation of arguments
            - Best explanation finding
            - Analogical transfer of insights
            - Multi-approach integration
            
            Document key challenges that the reasoning approach must address.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Determine requirements based on your analysis.
            """,
            "transitions": [{
                "target_state": ClassifierStates.RECOMMEND_STRATEGY,
                "description": "Needs identified"
            }]
        },
        ClassifierStates.RECOMMEND_STRATEGY: {
            "id": ClassifierStates.RECOMMEND_STRATEGY,
            "description": "Recommend optimal reasoning strategy",
            "purpose": f"Set '{ContextKeys.RECOMMENDED_REASONING_TYPE}', '{ContextKeys.STRATEGY_JUSTIFICATION}', and '{ContextKeys.ALTERNATIVE_APPROACHES}'",
            "required_context_keys": [
                ContextKeys.RECOMMENDED_REASONING_TYPE,
                ContextKeys.STRATEGY_JUSTIFICATION,
                ContextKeys.ALTERNATIVE_APPROACHES
            ],
            "instructions": """
            Recommend the best reasoning strategy:
            
            Choose from available strategies:
            - simple_calculator: For basic arithmetic operations
            - analytical: For systematic breakdown and analysis
            - deductive: For logical reasoning from premises
            - inductive: For pattern recognition and generalization
            - creative: For novel solution generation
            - critical: For argument and evidence evaluation
            - abductive: For finding best explanations
            - analogical: For similarity-based problem solving
            - hybrid: For complex problems needing multiple approaches
            
            Provide clear justification and identify 1-2 alternative approaches that could also work.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Make recommendation based on your complete analysis.
            """,
            "transitions": []
        }
    }
}


# ============================================================================
# SIMPLE CALCULATOR FSM - Basic arithmetic operations
# ============================================================================

simple_calculator_fsm = {
    "name": "simple_calculator",
    "description": "Performs simple arithmetic calculations with error handling",
    "initial_state": "extract_elements",
    "persona": "You are a precise calculator that performs arithmetic operations accurately.",
    "states": {
        "extract_elements": {
            "id": "extract_elements",
            "description": "Extract operands and operator from problem",
            "purpose": f"Extract '{ContextKeys.OPERAND1}', '{ContextKeys.OPERAND2}', and '{ContextKeys.OPERATOR}'",
            "required_context_keys": [ContextKeys.OPERAND1, ContextKeys.OPERAND2, ContextKeys.OPERATOR],
            "instructions": """
            Extract the mathematical elements from the problem:
            
            - Identify the first number (operand1)
            - Identify the second number (operand2)  
            - Identify the operation (+, -, *, /, ^, etc.)
            - Handle decimal numbers and negative numbers
            - Extract from problem_components if available, otherwise parse from problem_statement
            
            For expressions like "2 + 3", set operand1=2, operand2=3, operator="+"
            
            IMPORTANT: Do not ask questions or request additional input from the user. Extract elements from the available problem information.
            """,
            "transitions": [{
                "target_state": "perform_calculation",
                "description": "Elements extracted successfully"
            }]
        },
        "perform_calculation": {
            "id": "perform_calculation",
            "description": "Calculate the arithmetic result",
            "purpose": f"Calculate and store result in '{ContextKeys.CALCULATION_RESULT}'",
            "required_context_keys": [ContextKeys.CALCULATION_RESULT],
            "instructions": """
            Perform the arithmetic calculation:
            
            - Execute the operation: operand1 operator operand2
            - Handle basic operations: +, -, *, /, ^(power)
            - Manage edge cases: division by zero, overflow, invalid operations
            - Store the numerical result in calculation_result
            - If error occurs, store error description in calculation_error
            
            Examples:
            - 2 + 3 = 5
            - 10 / 2 = 5
            - 5 * 4 = 20
            - 2^3 = 8
            
            IMPORTANT: Do not ask questions or request additional input from the user. Perform calculation with extracted elements.
            """,
            "transitions": []
        }
    }
}


# ============================================================================
# ANALYTICAL REASONING FSM - Systematic decomposition and analysis
# ============================================================================

analytical_fsm = {
    "name": "analytical_reasoning",
    "description": "Analytical reasoning through systematic decomposition and component analysis",
    "initial_state": "decompose",
    "persona": "You are a methodical analytical thinker who breaks down complex problems systematically.",
    "states": {
        "decompose": {
            "id": "decompose",
            "description": "Break down the problem into component parts",
            "purpose": f"Identify '{ContextKeys.COMPONENTS}', '{ContextKeys.ATTRIBUTES}', and '{ContextKeys.RELATIONSHIPS}'",
            "required_context_keys": [ContextKeys.COMPONENTS, ContextKeys.ATTRIBUTES, ContextKeys.RELATIONSHIPS],
            "instructions": """
            Systematically decompose the problem:
            
            - Break the problem into smaller, manageable components
            - Identify key attributes of each component
            - Map relationships and dependencies between components
            - Organize components hierarchically if applicable
            - Note any emergent properties from component interactions
            
            Focus on creating a clear structural understanding of the problem space.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Decompose based on the problem as presented.
            """,
            "transitions": [{
                "target_state": "analyze_components",
                "description": "Decomposition complete"
            }]
        },
        "analyze_components": {
            "id": "analyze_components",
            "description": "Analyze each component in detail",
            "purpose": f"Create '{ContextKeys.COMPONENT_ANALYSIS}' and identify '{ContextKeys.DATA_REQUIREMENTS}'",
            "required_context_keys": [ContextKeys.COMPONENT_ANALYSIS, ContextKeys.DATA_REQUIREMENTS],
            "instructions": """
            Conduct detailed analysis of each component:
            
            - Examine the function and role of each component
            - Identify the properties and characteristics
            - Determine how each component contributes to the whole
            - Assess the importance and priority of each component
            - Note what additional data might be needed for complete analysis
            
            Create comprehensive component analysis with clear insights.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Analyze components based on your decomposition.
            """,
            "transitions": [{
                "target_state": "identify_patterns",
                "description": "Component analysis complete"
            }]
        },
        "identify_patterns": {
            "id": "identify_patterns",
            "description": "Find patterns and dependencies between components",
            "purpose": f"Identify '{ContextKeys.PATTERNS}', '{ContextKeys.CAUSAL_LINKS}', and '{ContextKeys.DEPENDENCIES}'",
            "required_context_keys": [ContextKeys.PATTERNS, ContextKeys.CAUSAL_LINKS, ContextKeys.DEPENDENCIES],
            "instructions": """
            Identify patterns and relationships:
            
            - Look for recurring patterns across components
            - Establish causal relationships (cause-effect links)
            - Map dependencies (what depends on what)
            - Identify feedback loops or circular dependencies
            - Note any systematic behaviors or regularities
            
            Focus on understanding the dynamic interactions between components.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Identify patterns from your component analysis.
            """,
            "transitions": [{
                "target_state": "integrate_findings",
                "description": "Patterns identified"
            }]
        },
        "integrate_findings": {
            "id": "integrate_findings",
            "description": "Synthesize understanding from all analytical work",
            "purpose": f"Create '{ContextKeys.INTEGRATED_ANALYSIS}' and '{ContextKeys.KEY_INSIGHTS}'",
            "required_context_keys": [ContextKeys.INTEGRATED_ANALYSIS, ContextKeys.KEY_INSIGHTS],
            "instructions": """
            Integrate all analytical findings:
            
            - Synthesize component analysis, patterns, and relationships
            - Create a comprehensive understanding of the problem
            - Highlight the most important insights and discoveries
            - Identify implications and potential solutions
            - Summarize the analytical understanding clearly
            
            Provide integrated analysis that brings together all analytical work into coherent understanding.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Integrate based on your complete analytical process.
            """,
            "transitions": []
        }
    }
}


# ============================================================================
# DEDUCTIVE REASONING FSM - Logical reasoning from general to specific
# ============================================================================

deductive_fsm = {
    "name": "deductive_reasoning",
    "description": "Apply general principles and rules to reach specific conclusions through valid logical reasoning",
    "initial_state": "identify_premises",
    "persona": "You are a logical thinker who applies established principles and rules to reach certain conclusions through valid reasoning.",
    "states": {
        "identify_premises": {
            "id": "identify_premises",
            "description": "Identify the general rules, principles, and assumptions",
            "purpose": f"Establish '{ContextKeys.PREMISES}' and identify '{ContextKeys.ASSUMPTIONS}'",
            "required_context_keys": [ContextKeys.PREMISES, ContextKeys.ASSUMPTIONS],
            "instructions": """
            Identify the starting points for deductive reasoning:
            
            - What general rules, laws, or principles apply to this problem?
            - What facts or givens can we assume as true?
            - What established knowledge is relevant?
            - What premises does any argument rely on?
            - What assumptions are being made (both stated and unstated)?
            
            Be explicit about both obvious and hidden assumptions that underlie the reasoning.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Work with the premises and information already provided in the problem.
            """,
            "transitions": [{
                "target_state": "apply_logic",
                "description": "Premises and assumptions identified"
            }]
        },
        "apply_logic": {
            "id": "apply_logic",
            "description": "Apply logical rules to derive conclusions step by step",
            "purpose": f"Document '{ContextKeys.LOGICAL_STEPS}' and '{ContextKeys.INTERMEDIATE_CONCLUSIONS}'",
            "required_context_keys": [ContextKeys.LOGICAL_STEPS, ContextKeys.INTERMEDIATE_CONCLUSIONS],
            "instructions": """
            Apply logical reasoning systematically:
            
            - What follows logically from the established premises?
            - What intermediate conclusions can be drawn at each step?
            - What logical rules or forms are being applied (modus ponens, syllogism, etc.)?
            - How does each step follow necessarily from the previous ones?
            - What can be concluded with certainty at each stage?
            
            Show your logical work clearly, step by step, ensuring valid inferences.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Apply logical reasoning to the premises you've identified.
            """,
            "transitions": [{
                "target_state": "derive_conclusion",
                "description": "Logical steps applied"
            }]
        },
        "derive_conclusion": {
            "id": "derive_conclusion",
            "description": "Reach final conclusions and assess logical validity",
            "purpose": f"State final '{ContextKeys.CONCLUSION}' and assess '{ContextKeys.LOGICAL_VALIDITY}'",
            "required_context_keys": [ContextKeys.CONCLUSION, ContextKeys.LOGICAL_VALIDITY],
            "instructions": """
            Derive the final conclusion and validate the reasoning:
            
            - What specific conclusion follows from the complete logical chain?
            - Is the reasoning logically valid (do conclusions follow necessarily)?
            - Does the conclusion adequately address the original problem?
            - Are there any logical gaps, errors, or invalid inferences?
            - How certain can we be about the final conclusion?
            
            State your conclusion clearly and provide honest assessment of the logical validity.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Derive your conclusion from the logical steps you've taken.
            """,
            "transitions": []
        }
    }
}


# ============================================================================
# INDUCTIVE REASONING FSM - Reasoning from specific to general patterns
# ============================================================================

inductive_fsm = {
    "name": "inductive_reasoning",
    "description": "Reason from specific observations to discover general patterns and principles",
    "initial_state": "gather_observations",
    "persona": "You are an empirical thinker who discovers patterns by carefully examining specific examples and building general understanding from evidence.",
    "states": {
        "gather_observations": {
            "id": "gather_observations",
            "description": "Collect and organize specific observations and data points",
            "purpose": f"Identify '{ContextKeys.OBSERVATIONS}' and '{ContextKeys.DATA_POINTS}' relevant to the problem",
            "required_context_keys": [ContextKeys.OBSERVATIONS, ContextKeys.DATA_POINTS],
            "instructions": """
            Systematically gather specific, concrete observations:
            
            - What specific examples, cases, or instances are available?
            - What data points, measurements, or facts are relevant?
            - What have we observed in similar situations or contexts?
            - What specific behaviors, outcomes, or phenomena are documented?
            - What concrete evidence is available for analysis?
            
            Focus on collecting concrete, specific observations rather than general statements or theories.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Work with the observations and data available in the problem context.
            """,
            "transitions": [{
                "target_state": "identify_commonalities",
                "description": "Observations gathered"
            }]
        },
        "identify_commonalities": {
            "id": "identify_commonalities",
            "description": "Find patterns and commonalities across observations",
            "purpose": f"Identify '{ContextKeys.COMMONALITIES}' and '{ContextKeys.TRENDS}' in the data",
            "required_context_keys": [ContextKeys.COMMONALITIES, ContextKeys.TRENDS],
            "instructions": """
            Systematically look for patterns across your observations:
            
            - What do multiple cases, examples, or instances have in common?
            - What trends, regularities, or consistencies emerge from the data?
            - What relationships appear consistently across different observations?
            - What factors or variables seem to be associated or correlated?
            - What sequences, progressions, or developmental patterns do you notice?
            
            Identify both obvious and subtle patterns that might not be immediately apparent.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Find patterns in the observations you've systematically gathered.
            """,
            "transitions": [{
                "target_state": "form_hypothesis",
                "description": "Commonalities identified"
            }]
        },
        "form_hypothesis": {
            "id": "form_hypothesis",
            "description": "Form general hypothesis based on observed patterns",
            "purpose": f"Create '{ContextKeys.HYPOTHESIS}' supported by '{ContextKeys.SUPPORTING_EVIDENCE}'",
            "required_context_keys": [ContextKeys.HYPOTHESIS, ContextKeys.SUPPORTING_EVIDENCE],
            "instructions": """
            Form a well-grounded general hypothesis from the identified patterns:
            
            - What general rule, principle, or law might explain the observed patterns?
            - What predictions can you make about future or unobserved cases?
            - What broader principle or generalization seems to apply?
            - How would you clearly state this generalization?
            - What specific evidence best supports this hypothesis?
            
            Make your hypothesis specific enough to be testable while being general enough to be useful for prediction.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Form your hypothesis based on the patterns you've systematically identified.
            """,
            "transitions": [{
                "target_state": "test_generalization",
                "description": "Hypothesis formed"
            }]
        },
        "test_generalization": {
            "id": "test_generalization",
            "description": "Test the strength and limits of the generalization",
            "purpose": f"Evaluate with '{ContextKeys.TEST_RESULTS}', '{ContextKeys.COUNTER_EXAMPLES}', and '{ContextKeys.GENERALIZATION_STRENGTH}'",
            "required_context_keys": [ContextKeys.TEST_RESULTS, ContextKeys.COUNTER_EXAMPLES, ContextKeys.GENERALIZATION_STRENGTH],
            "instructions": """
            Rigorously test your generalization against available evidence:
            
            - How well does it predict or explain other cases not used in forming it?
            - What counter-examples, exceptions, or contradictory evidence exist?
            - Under what conditions does the generalization hold or fail?
            - How strong is the inductive support based on sample size and quality?
            - What would strengthen or weaken confidence in this generalization?
            
            Rate the strength of your generalization from 1-10 based on the quality and quantity of supporting evidence.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Test your generalization rigorously with all available information.
            """,
            "transitions": []
        }
    }
}


# ============================================================================
# CREATIVE REASONING FSM - Novel solution generation through creative thinking
# ============================================================================

creative_fsm = {
    "name": "creative_reasoning",
    "description": "Generate novel and innovative solutions through divergent and convergent creative thinking processes",
    "initial_state": "explore_perspectives",
    "persona": "You are an innovative creative thinker who generates novel solutions by seeing problems from fresh perspectives and making unexpected connections.",
    "states": {
        "explore_perspectives": {
            "id": "explore_perspectives",
            "description": "Explore the problem from multiple creative perspectives",
            "purpose": f"Generate '{ContextKeys.PERSPECTIVES}' and '{ContextKeys.REFRAMINGS}' of the problem",
            "required_context_keys": [ContextKeys.PERSPECTIVES, ContextKeys.REFRAMINGS],
            "instructions": """
            Systematically explore the problem through different creative lenses:
            
            - How would different people approach this (child, artist, engineer, scientist, entrepreneur)?
            - What if we flipped key assumptions or removed major constraints?
            - How is this problem similar to or different from challenges in completely different domains?
            - What would this look like from the opposite or inverse perspective?
            - What metaphors, analogies, or artistic representations reveal new angles?
            
            Generate multiple fresh ways of viewing and framing the problem to unlock creative potential.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Generate diverse perspectives based on the problem as stated.
            """,
            "transitions": [{
                "target_state": "generate_ideas",
                "description": "Multiple perspectives explored"
            }]
        },
        "generate_ideas": {
            "id": "generate_ideas",
            "description": "Brainstorm creative and unconventional ideas without judgment",
            "purpose": f"Create '{ContextKeys.CREATIVE_IDEAS}' and '{ContextKeys.UNCONVENTIONAL_APPROACHES}'",
            "required_context_keys": [ContextKeys.CREATIVE_IDEAS, ContextKeys.UNCONVENTIONAL_APPROACHES],
            "instructions": """
            Generate creative ideas through divergent thinking:
            
            - What wild, impossible, or seemingly silly ideas come to mind?
            - What if we completely removed key constraints or limitations?
            - How do completely different fields or domains solve similar challenges?
            - What would an ideal, unlimited-resource solution look like?
            - What approaches break conventional thinking or challenge standard methods?
            
            Focus on quantity and novelty over immediate practicality. Suspend judgment and let creativity flow freely.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Generate ideas freely based on your thorough perspective exploration.
            """,
            "transitions": [{
                "target_state": "combine_concepts",
                "description": "Ideas generated"
            }]
        },
        "combine_concepts": {
            "id": "combine_concepts",
            "description": "Combine and synthesize ideas in novel ways",
            "purpose": f"Create '{ContextKeys.COMBINATIONS}' and develop '{ContextKeys.NOVEL_SOLUTIONS}'",
            "required_context_keys": [ContextKeys.COMBINATIONS, ContextKeys.NOVEL_SOLUTIONS],
            "instructions": """
            Systematically combine ideas in unexpected and innovative ways:
            
            - What happens when we merge different approaches or solutions?
            - How can we combine the best elements of multiple ideas into something new?
            - What entirely new solutions emerge from mixing seemingly unrelated concepts?
            - How can we build bridges between ideas from different domains or perspectives?
            - What hybrid approaches might leverage multiple creative insights?
            
            Create truly novel solutions that integrate multiple creative elements in unexpected ways.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Combine the ideas you've generated in innovative ways.
            """,
            "transitions": [{
                "target_state": "evaluate_novelty",
                "description": "Concepts combined"
            }]
        },
        "evaluate_novelty": {
            "id": "evaluate_novelty",
            "description": "Evaluate creative solutions for novelty, feasibility, and impact",
            "purpose": f"Select '{ContextKeys.BEST_CREATIVE_SOLUTION}' and rate '{ContextKeys.INNOVATION_RATING}'",
            "required_context_keys": [ContextKeys.BEST_CREATIVE_SOLUTION, ContextKeys.INNOVATION_RATING],
            "instructions": """
            Critically evaluate your creative solutions using convergent thinking:
            
            - Which solutions are most genuinely novel and original?
            - Which best balance creativity with potential feasibility?
            - What makes each solution truly innovative or groundbreaking?
            - Which has the greatest potential for positive impact or effectiveness?
            - How would you rate the overall creativity and innovation level (1-10 scale)?
            
            Select the most promising creative solution and provide a detailed innovation assessment.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Evaluate and select based on the creative solutions you've systematically developed.
            """,
            "transitions": []
        }
    }
}


# ============================================================================
# CRITICAL REASONING FSM - Rigorous evaluation of arguments and evidence
# ============================================================================

critical_fsm = {
    "name": "critical_reasoning",
    "description": "Systematic critical evaluation of arguments, claims, evidence, and reasoning to distinguish sound from unsound conclusions",
    "initial_state": "identify_claims",
    "persona": "You are a rigorous critical thinker who carefully evaluates arguments, evidence, and reasoning to separate truth from error and strong arguments from weak ones.",
    "states": {
        "identify_claims": {
            "id": "identify_claims",
            "description": "Identify and categorize the main claims and arguments",
            "purpose": f"Extract '{ContextKeys.CLAIMS}' and '{ContextKeys.ARGUMENTS}' from the problem or text",
            "required_context_keys": [ContextKeys.CLAIMS, ContextKeys.ARGUMENTS],
            "instructions": """
            Systematically identify what is being claimed or argued:
            
            - What are the main conclusions, assertions, or claims being made?
            - What specific arguments are presented to support these claims?
            - What is the overall thesis, position, or central argument?
            - Are there implicit or unstated assumptions underlying the arguments?
            - How are factual claims distinguished from opinions, interpretations, or value judgments?
            
            Clearly separate main arguments from supporting points and distinguish between different types of claims.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Work with the claims and arguments present in the given material.
            """,
            "transitions": [{
                "target_state": "examine_evidence",
                "description": "Claims and arguments identified"
            }]
        },
        "examine_evidence": {
            "id": "examine_evidence",
            "description": "Critically examine the quality and sufficiency of supporting evidence",
            "purpose": f"Assess '{ContextKeys.EVIDENCE_QUALITY}' and identify '{ContextKeys.EVIDENCE_GAPS}'",
            "required_context_keys": [ContextKeys.EVIDENCE_QUALITY, ContextKeys.EVIDENCE_GAPS],
            "instructions": """
            Rigorously examine the evidence supporting the identified claims:
            
            - What evidence is actually provided? Is it relevant, sufficient, and appropriate?
            - How reliable, credible, and authoritative are the sources?
            - Is the evidence current, representative, and methodologically sound?
            - What important evidence is missing that would strengthen or weaken the argument?
            - Are there potential biases, conflicts of interest, or limitations in how evidence was collected or presented?
            
            Be specific about both strengths and critical weaknesses of the evidence base.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Evaluate the evidence that is already available in the material.
            """,
            "transitions": [{
                "target_state": "analyze_logic",
                "description": "Evidence examined"
            }]
        },
        "analyze_logic": {
            "id": "analyze_logic",
            "description": "Analyze logical structure and identify reasoning flaws",
            "purpose": f"Conduct '{ContextKeys.LOGICAL_ANALYSIS}', identify '{ContextKeys.ASSUMPTIONS}' and '{ContextKeys.FALLACIES}'",
            "required_context_keys": [ContextKeys.LOGICAL_ANALYSIS, ContextKeys.ASSUMPTIONS, ContextKeys.FALLACIES],
            "instructions": """
            Systematically analyze the logical structure and identify flaws:
            
            - Do the conclusions actually follow logically from the stated premises?
            - What key assumptions are being made (both stated and unstated)?
            - Are there identifiable logical fallacies (ad hominem, straw man, false dilemma, appeal to authority, etc.)?
            - Are there gaps in reasoning, unsupported logical leaps, or non sequiturs?
            - Is the reasoning internally consistent throughout the argument?
            
            Identify specific logical strengths and weaknesses with clear examples.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Analyze the logical structure of what has been provided.
            """,
            "transitions": [{
                "target_state": "consider_alternatives",
                "description": "Logic analyzed"
            }]
        },
        "consider_alternatives": {
            "id": "consider_alternatives",
            "description": "Consider alternative explanations and strong counter-arguments",
            "purpose": f"Identify '{ContextKeys.ALTERNATIVE_EXPLANATIONS}' and '{ContextKeys.COUNTER_ARGUMENTS}'",
            "required_context_keys": [ContextKeys.ALTERNATIVE_EXPLANATIONS, ContextKeys.COUNTER_ARGUMENTS],
            "instructions": """
            Systematically consider alternatives and opposing viewpoints:
            
            - What alternative explanations, interpretations, or conclusions are plausible?
            - What are the strongest possible counter-arguments to the main claims?
            - What would informed opponents or skeptics of this position argue?
            - Are there other reasonable ways to interpret the same evidence?
            - What additional considerations or factors might change the evaluation?
            
            Present fair and intellectually honest alternatives rather than weak straw man arguments.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Generate thoughtful alternatives based on your systematic analysis.
            """,
            "transitions": [{
                "target_state": "form_judgment",
                "description": "Alternatives considered"
            }]
        },
        "form_judgment": {
            "id": "form_judgment",
            "description": "Form comprehensive critical assessment with justified confidence level",
            "purpose": f"Provide '{ContextKeys.CRITICAL_ASSESSMENT}' and '{ContextKeys.CONFIDENCE_RATING}'",
            "required_context_keys": [ContextKeys.CRITICAL_ASSESSMENT, ContextKeys.CONFIDENCE_RATING],
            "instructions": """
            Form a well-reasoned overall critical judgment:
            
            - How strong and convincing are the arguments when all factors are considered?
            - What are the most significant strengths and critical weaknesses?
            - How much confidence should we reasonably have in the main claims?
            - What would most significantly strengthen or weaken these arguments?
            - What is your final, balanced assessment of the overall reasoning quality?
            
            Provide a fair, balanced evaluation with a confidence rating from 1-10 and clear justification.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Form your critical judgment based on your complete systematic analysis.
            """,
            "transitions": []
        }
    }
}


# ============================================================================
# ABDUCTIVE REASONING FSM - Finding best explanations for observations
# ============================================================================

abductive_fsm = {
    "name": "abductive_reasoning",
    "description": "Find the best explanation for puzzling observations through systematic inference to the best explanation",
    "initial_state": "identify_observations",
    "persona": "You are a detective and investigator who excels at finding the most plausible explanations for puzzling observations and unexplained phenomena.",
    "states": {
        "identify_observations": {
            "id": "identify_observations",
            "description": "Identify key observations that require explanation",
            "purpose": f"Catalog '{ContextKeys.OBSERVATIONS}' and identify 'surprising_elements' that require explanation",
            "required_context_keys": [ContextKeys.OBSERVATIONS, "surprising_elements"],
            "instructions": """
            Systematically identify what needs to be explained:
            
            - What specific facts, phenomena, or observations are we trying to explain?
            - What seems surprising, unexpected, anomalous, or puzzling?
            - What patterns, behaviors, or outcomes need to be accounted for?
            - What data points or evidence require explanation?
            - What aspects of the situation seem to call for further understanding?
            
            Focus on concrete, specific observations rather than interpretations or preliminary explanations.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Work with the observations already provided in the problem context.
            """,
            "transitions": [{
                "target_state": "generate_hypotheses",
                "description": "Key observations identified"
            }]
        },
        "generate_hypotheses": {
            "id": "generate_hypotheses",
            "description": "Generate multiple potential explanations",
            "purpose": f"Create 'potential_hypotheses' with 'hypothesis_rationales' for each explanation",
            "required_context_keys": ["potential_hypotheses", "hypothesis_rationales"],
            "instructions": """
            Generate multiple plausible explanations for the observations:
            
            - What could reasonably account for what we observe?
            - Consider different types of causes (direct, indirect, systemic, multiple contributing factors)
            - Think about both obvious and less obvious potential explanations
            - Include competing or alternative hypotheses that might explain the same phenomena
            - Consider explanations at different levels (individual, systemic, environmental, etc.)
            
            For each hypothesis, provide a clear rationale explaining why it could account for the observations.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Generate comprehensive hypotheses based on the observations you've identified.
            """,
            "transitions": [{
                "target_state": "evaluate_hypotheses",
                "description": "Hypotheses generated"
            }]
        },
        "evaluate_hypotheses": {
            "id": "evaluate_hypotheses",
            "description": "Systematically evaluate each hypothesis against standard criteria",
            "purpose": f"Create 'hypothesis_evaluations' using 'evaluation_criteria' for systematic assessment",
            "required_context_keys": ["hypothesis_evaluations", "evaluation_criteria"],
            "instructions": """
            Systematically evaluate each hypothesis using standard criteria for explanatory adequacy:
            
            - Explanatory scope: How comprehensively does it explain the observations?
            - Simplicity/parsimony: Is it unnecessarily complex or does it invoke minimal assumptions?
            - Plausibility: How likely is it given our background knowledge and experience?
            - Testability: Can it be verified, falsified, or further investigated?
            - Consistency: Does it fit coherently with other established knowledge?
            - Predictive power: Does it suggest new predictions or help anticipate future observations?
            
            Rate each hypothesis systematically on these dimensions with clear justifications.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Evaluate based on the hypotheses you've systematically generated.
            """,
            "transitions": [{
                "target_state": "select_best_explanation",
                "description": "Hypotheses evaluated"
            }]
        },
        "select_best_explanation": {
            "id": "select_best_explanation",
            "description": "Select most plausible explanation with clear justification",
            "purpose": f"Choose 'best_hypothesis' with 'selection_justification', 'confidence_in_explanation', and 'next_steps_for_validation'",
            "required_context_keys": ["best_hypothesis", "selection_justification", "confidence_in_explanation", "next_steps_for_validation"],
            "instructions": """
            Select the best explanation through careful comparative analysis:
            
            - Which hypothesis best balances all the evaluative criteria?
            - Why is this the most plausible and compelling explanation overall?
            - What is your confidence level in this explanation (1-10 scale) and why?
            - What would you need to do to further test, validate, or investigate this explanation?
            - What are the key strengths of this explanation and what limitations or uncertainties remain?
            
            Provide clear, detailed justification for your selection and acknowledge appropriate levels of uncertainty.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Make your selection based on your systematic evaluation process.
            """,
            "transitions": []
        }
    }
}


# ============================================================================
# ANALOGICAL REASONING FSM - Transfer insights through analogical thinking
# ============================================================================

analogical_fsm = {
    "name": "analogical_reasoning",
    "description": "Transfer insights and solutions via systematic analogical reasoning and pattern matching across domains",
    "initial_state": "define_target_problem",
    "persona": "You are an expert at finding meaningful connections and analogies. You help solve problems by identifying similar situations and transferring insights across domains.",
    "states": {
        "define_target_problem": {
            "id": "define_target_problem",
            "description": "Clearly define and characterize the target problem",
            "purpose": f"Analyze the problem to identify 'target_problem_description' and 'key_features_of_target'",
            "required_context_keys": ["target_problem_description", "key_features_of_target"],
            "instructions": """
            Systematically define and characterize the target problem:
            
            - What is the core challenge, question, or problem we're trying to solve?
            - What are the key features, constraints, context, and important characteristics?
            - What kind of solution, understanding, or outcome are we seeking?
            - What essential structural or functional characteristics should any good analogy match?
            - What are the most important aspects that an analogical source should share?
            
            Create a clear, comprehensive characterization of the target problem for analogical matching.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Work only with the information already provided in the problem statement and context.
            """,
            "transitions": [{
                "target_state": "find_source_analogs",
                "description": "Target problem clearly defined"
            }]
        },
        "find_source_analogs": {
            "id": "find_source_analogs",
            "description": "Identify potential analogous situations across various domains",
            "purpose": f"Find 'potential_analogs' with 'rationale_for_choice' and 'similarity_criteria_used'",
            "required_context_keys": ["potential_analogs", "rationale_for_choice", "similarity_criteria_used"],
            "instructions": """
            Systematically search for analogous situations across different domains:
            
            - What situations, problems, or systems share important structural similarities?
            - What comparable processes, mechanisms, or functional relationships exist in other fields?
            - What parallel challenges exist in different domains (nature, technology, history, etc.)?
            - What historical precedents, case studies, or examples show similar patterns?
            - What systems or situations exhibit comparable dynamics or relationships?
            
            Provide 2-4 potential analogs with clear rationale for why each might offer valuable insights.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Search for analogs based on your target problem characterization.
            """,
            "transitions": [{
                "target_state": "map_correspondences",
                "description": "Source analogs identified"
            }]
        },
        "map_correspondences": {
            "id": "map_correspondences",
            "description": "Create systematic mapping between source analog and target problem",
            "purpose": f"Select best analog and create detailed mapping with 'selected_analog', 'analogical_mapping', 'identified_similarities', 'identified_differences'",
            "required_context_keys": ["selected_analog", "analogical_mapping", "identified_similarities", "identified_differences"],
            "instructions": """
            Create systematic correspondences between the most promising analog and target:
            
            - Choose the analog with the strongest structural and functional similarities
            - Map specific elements from source to target (A corresponds to X, B relates to Y, etc.)
            - Identify the strongest similarities that support and validate the analogy
            - Note important differences or limitations that constrain the analogical inference
            - Create explicit, systematic mapping of relationships and correspondences
            
            Be precise about what maps to what and provide clear justification for the correspondences.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Create the mapping based on your analysis of potential analogs.
            """,
            "transitions": [{
                "target_state": "transfer_insights",
                "description": "Correspondences systematically mapped"
            }]
        },
        "transfer_insights": {
            "id": "transfer_insights",
            "description": "Transfer knowledge and solutions from analog to target domain",
            "purpose": f"Generate 'transferred_insights_or_solutions' and 'potential_inferences'",
            "required_context_keys": ["transferred_insights_or_solutions", "potential_inferences"],
            "instructions": """
            Systematically transfer insights using the analogical mapping:
            
            - What solutions, strategies, or approaches worked effectively in the source domain?
            - What principles, patterns, or mechanisms can be transferred to the target?
            - What new understanding or perspective does the analogy provide about the target problem?
            - What predictions, inferences, or hypotheses can we generate through analogical reasoning?
            - How do successful strategies in the source domain suggest approaches for the target?
            
            Be specific and explicit about how insights from the analog apply to and illuminate the target problem.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Generate insights based on the systematic analogical mapping you've created.
            """,
            "transitions": [{
                "target_state": "evaluate_analogy_fit",
                "description": "Insights transferred"
            }]
        },
        "evaluate_analogy_fit": {
            "id": "evaluate_analogy_fit",
            "description": "Critically evaluate the analogy's validity and practical utility",
            "purpose": f"Assess analogy with 'analogy_strengths', 'analogy_weaknesses_or_limitations', 'adapted_solution_or_understanding', 'analogy_confidence_rating'",
            "required_context_keys": ["analogy_strengths", "analogy_weaknesses_or_limitations", "adapted_solution_or_understanding", "analogy_confidence_rating"],
            "instructions": """
            Critically and systematically evaluate the analogical reasoning:
            
            - What are the strongest, most compelling aspects of this analogy?
            - Where does the analogy break down, mislead, or have significant limitations?
            - How should the transferred solution or insight be adapted for the target context?
            - What level of confidence is warranted in this analogical reasoning (1-10 scale)?
            - What would strengthen or weaken confidence in the analogical inference?
            
            Provide balanced evaluation with adapted solution and justified confidence rating.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Complete your evaluation with the information developed through your systematic analogical process.
            """,
            "transitions": []
        }
    }
}


# ============================================================================
# HYBRID REASONING FSM - Integrated multi-approach reasoning with loop prevention
# ============================================================================

hybrid_fsm = {
    "name": "hybrid_reasoning",
    "description": "Systematically combines multiple reasoning approaches for comprehensive problem solving with loop prevention mechanisms",
    "initial_state": "identify_components",
    "persona": "You are a master strategist who skillfully combines different reasoning approaches to tackle complex problems from multiple complementary angles.",
    "states": {
        "identify_components": {
            "id": "identify_components",
            "description": "Break problem into components requiring different reasoning approaches",
            "purpose": f"Map '{ContextKeys.PROBLEM_ASPECTS}' to reasoning types in '{ContextKeys.REASONING_MAP}'",
            "required_context_keys": [ContextKeys.PROBLEM_ASPECTS, ContextKeys.REASONING_MAP],
            "instructions": """
            Systematically analyze the problem to identify aspects requiring different reasoning approaches:
            
            - What parts need systematic analytical breakdown and decomposition?
            - What components require logical deduction from established principles?
            - Where might creative, innovative thinking provide valuable insights?
            - What aspects need critical evaluation of arguments or evidence?
            - What patterns might inductive reasoning help reveal from available data?
            
            Create a comprehensive map of problem aspects to appropriate reasoning approaches. Initialize hybrid_loop_count to 0 for loop management.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Analyze the problem as presented to identify reasoning needs.
            """,
            "transitions": [{
                "target_state": "apply_analytical",
                "description": "Components identified and mapped to reasoning approaches"
            }]
        },
        "apply_analytical": {
            "id": "apply_analytical",
            "description": "Apply systematic analytical reasoning to understand problem structure",
            "purpose": f"Create '{ContextKeys.ANALYTICAL_BREAKDOWN}' and '{ContextKeys.COMPONENT_RELATIONSHIPS}'",
            "required_context_keys": [ContextKeys.ANALYTICAL_BREAKDOWN, ContextKeys.COMPONENT_RELATIONSHIPS],
            "instructions": """
            Apply systematic analytical thinking to the problem:
            
            - Break down complex aspects into simpler, more manageable components
            - Identify relationships, dependencies, and interactions between components
            - Understand the underlying structure and organizational principles
            - Analyze how different elements contribute to the overall problem
            - Document systematic findings from the analytical decomposition
            
            Provide thorough analytical breakdown that will inform other reasoning approaches.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Proceed with systematic analytical breakdown based on your component identification.
            """,
            "transitions": [{
                "target_state": "apply_logical",
                "description": "Analytical reasoning systematically applied"
            }]
        },
        "apply_logical": {
            "id": "apply_logical",
            "description": "Apply logical reasoning to derive sound conclusions",
            "purpose": f"Establish '{ContextKeys.LOGICAL_CONCLUSIONS}' and '{ContextKeys.REASONING_CHAIN}'",
            "required_context_keys": [ContextKeys.LOGICAL_CONCLUSIONS, ContextKeys.REASONING_CHAIN],
            "instructions": """
            Apply systematic logical reasoning to the analytical findings:
            
            - What can be logically deduced from the analytical breakdown and identified relationships?
            - What logical steps follow necessarily from the evidence and established facts?
            - What conclusions can be drawn with high confidence based on logical inference?
            - How do the logical pieces fit together to form coherent understanding?
            - What clear chain of reasoning emerges from the logical analysis?
            
            Build a clear, valid chain of logical reasoning that builds on the analytical foundation.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Apply logical reasoning systematically to your analytical findings.
            """,
            "transitions": [{
                "target_state": "apply_creative",
                "description": "Logical reasoning systematically applied"
            }]
        },
        "apply_creative": {
            "id": "apply_creative",
            "description": "Apply creative thinking to generate novel approaches and insights",
            "purpose": f"Generate '{ContextKeys.CREATIVE_INSIGHTS}' and '{ContextKeys.NOVEL_APPROACHES}'",
            "required_context_keys": [ContextKeys.CREATIVE_INSIGHTS, ContextKeys.NOVEL_APPROACHES],
            "instructions": """
            Apply creative thinking to complement and enhance the analytical and logical work:
            
            - What new perspectives or insights emerge from viewing the analysis creatively?
            - How might innovative approaches complement or extend the logical conclusions?
            - What novel solutions become possible when we think beyond conventional boundaries?
            - What creative connections or unexpected relationships can enhance understanding?
            - How can creative thinking add value to the systematic analysis already completed?
            
            Generate creative insights that meaningfully enhance rather than contradict the systematic reasoning.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Generate creative insights building on your comprehensive analytical and logical work.
            """,
            "transitions": [{
                "target_state": "critical_evaluation",
                "description": "Creative reasoning systematically applied"
            }]
        },
        "critical_evaluation": {
            "id": "critical_evaluation",
            "description": "Critically evaluate all findings with systematic loop prevention",
            "purpose": f"Create '{ContextKeys.EVALUATION_RESULTS}' and determine if refinement needed (maximum 2 loops)",
            "required_context_keys": [ContextKeys.EVALUATION_RESULTS],
            "instructions": """
            Critically evaluate the integration of all reasoning approaches:
            
            - How effectively do the analytical, logical, and creative approaches complement each other?
            - Are there significant contradictions, gaps, or inconsistencies that need resolution?
            - What are the key strengths and potential weaknesses of this combined approach?
            - Are there critical flaws or missing elements that would justify returning for refinement?
            - How robust and comprehensive is the overall reasoning when considered together?
            
            Set needs_refinement=True only for serious, fundamental issues. Always increment hybrid_loop_count to prevent infinite loops.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Evaluate comprehensively based on your multi-faceted analysis.
            """,
            "transitions": [
                {
                    "target_state": "integrate_solution",
                    "description": "Ready to integrate (no critical issues or loop limit reached)",
                    "priority": 1,
                    "conditions": [{
                        "description": "No critical issues found or maximum loops reached",
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
                    "description": "Refinement needed (limited loops remaining)",
                    "priority": 2,
                    "conditions": [{
                        "description": "Critical issues found and loops available",
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
            "description": "Integrate insights from all reasoning approaches into comprehensive solution",
            "purpose": f"Create '{ContextKeys.INTEGRATED_SOLUTION}' with '{ContextKeys.REASONING_SYNTHESIS_NOTES}'",
            "required_context_keys": [ContextKeys.INTEGRATED_SOLUTION, ContextKeys.REASONING_SYNTHESIS_NOTES],
            "instructions": """
            Systematically integrate all reasoning approaches into a comprehensive, unified solution:
            
            - How do analytical insights, logical conclusions, and creative innovations combine synergistically?
            - What is the most complete and nuanced understanding of the problem that emerges?
            - How do different reasoning types reinforce, complement, or constructively challenge each other?
            - What is the best integrated approach to solving or addressing the problem?
            - How does the combination create understanding that is greater than the sum of individual parts?
            
            Synthesize rather than merely summarize—create genuine integration that leverages the strengths of each approach.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Integrate systematically based on your comprehensive multi-approach analysis.
            """,
            "transitions": [{
                "target_state": "finalize_hybrid",
                "description": "Solution comprehensively integrated"
            }]
        },
        "finalize_hybrid": {
            "id": "finalize_hybrid",
            "description": "Present final hybrid solution with complete reasoning synthesis",
            "purpose": f"Finalize '{ContextKeys.FINAL_HYBRID_SOLUTION}' and '{ContextKeys.REASONING_SYNTHESIS}'",
            "required_context_keys": [ContextKeys.FINAL_HYBRID_SOLUTION, ContextKeys.REASONING_SYNTHESIS],
            "instructions": """
            Present the final comprehensive hybrid solution with complete synthesis:
            
            - What is your final, most comprehensive and well-reasoned solution?
            - How did each reasoning type (analytical, logical, creative, critical) contribute uniquely?
            - What is the particular strength and advantage of this multi-faceted reasoning approach?
            - How confident are you in this hybrid solution and why?
            - What would be the most logical next steps for implementation, testing, or further development?
            
            Provide a complete synthesis that demonstrates and showcases the power of systematically combined reasoning approaches.
            
            IMPORTANT: Do not ask questions or request additional input from the user. Present your final integrated solution based on the complete hybrid reasoning process.
            """,
            "transitions": []
        }
    }
}


# ============================================================================
# FSM REGISTRY - Complete collection for easy access and management
# ============================================================================

ALL_REASONING_FSMS = {
    "orchestrator": orchestrator_fsm,
    "classifier": classifier_fsm,
    "simple_calculator": simple_calculator_fsm,
    "analytical": analytical_fsm,
    "deductive": deductive_fsm,
    "inductive": inductive_fsm,
    "creative": creative_fsm,
    "critical": critical_fsm,
    "abductive": abductive_fsm,
    "analogical": analogical_fsm,
    "hybrid": hybrid_fsm,
}


def get_fsm_by_name(fsm_name: str) -> dict:
    """
    Retrieve FSM definition by name.

    Args:
        fsm_name: Name of the FSM to retrieve

    Returns:
        FSM definition dictionary

    Raises:
        KeyError: If FSM name not found
    """
    if fsm_name not in ALL_REASONING_FSMS:
        available = list(ALL_REASONING_FSMS.keys())
        raise KeyError(f"FSM '{fsm_name}' not found. Available FSMs: {available}")

    return ALL_REASONING_FSMS[fsm_name]


def list_available_fsms() -> list[str]:
    """
    Get list of all available FSM names.

    Returns:
        List of FSM names
    """
    return list(ALL_REASONING_FSMS.keys())


def get_reasoning_fsms_only() -> dict:
    """
    Get only the specialized reasoning FSMs (excluding orchestrator and classifier).

    Returns:
        Dictionary of reasoning FSMs
    """
    return {
        name: fsm for name, fsm in ALL_REASONING_FSMS.items()
        if name not in ["orchestrator", "classifier"]
    }