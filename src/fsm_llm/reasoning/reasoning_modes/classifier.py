"""CLASSIFIER FSM - Problem classification and strategy recommendation.

Part of the reasoning FSM definitions (see `reasoning_modes/__init__.py`).
"""

from fsm_llm.reasoning.constants import (
    ClassifierStates,
    ContextKeys,
)

from ._phrases import _IMPORTANT_NO_QUESTIONS, _PRESENT_FINAL, _PRESENT_FINDINGS

classifier_fsm = {
    "name": "problem_classifier",
    "description": "Classifies problems to determine the most appropriate reasoning strategy",
    "initial_state": ClassifierStates.ANALYZE_DOMAIN,
    "persona": "You are an expert problem analyst who identifies the best reasoning approach for any given problem.",
    "states": {
        ClassifierStates.ANALYZE_DOMAIN: {
            "description": "Identify problem domain and context",
            "purpose": f"Determine '{ContextKeys.PROBLEM_DOMAIN}' and '{ContextKeys.DOMAIN_INDICATORS}'",
            "required_context_keys": [
                ContextKeys.PROBLEM_DOMAIN,
                ContextKeys.DOMAIN_INDICATORS,
            ],
            "extraction_instructions": f"""
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

            {_IMPORTANT_NO_QUESTIONS} Analyze the domain based on the problem statement provided.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": ClassifierStates.ANALYZE_STRUCTURE,
                    "description": "Domain identified",
                }
            ],
        },
        ClassifierStates.ANALYZE_STRUCTURE: {
            "description": "Analyze problem structure and complexity",
            "purpose": f"Identify '{ContextKeys.PROBLEM_STRUCTURE}' and '{ContextKeys.STRUCTURAL_ELEMENTS}'",
            "required_context_keys": [
                ContextKeys.PROBLEM_STRUCTURE,
                ContextKeys.STRUCTURAL_ELEMENTS,
            ],
            "extraction_instructions": f"""
            Analyze the structural characteristics:

            Identify structure type and key elements:
            - Simple: Single-step, direct solution path
            - Sequential: Multi-step process with clear order
            - Hierarchical: Nested components with dependencies
            - Network: Multiple interconnected elements
            - Complex: Multi-faceted with various approaches needed

            Document structural elements like components, relationships, dependencies, constraints.

            {_IMPORTANT_NO_QUESTIONS} Analyze structure from the given problem.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": ClassifierStates.IDENTIFY_REASONING_NEEDS,
                    "description": "Structure analyzed",
                }
            ],
        },
        ClassifierStates.IDENTIFY_REASONING_NEEDS: {
            "description": "Identify specific reasoning requirements",
            "purpose": f"Determine '{ContextKeys.REASONING_REQUIREMENTS}' and '{ContextKeys.KEY_CHALLENGES}'",
            "required_context_keys": [
                ContextKeys.REASONING_REQUIREMENTS,
                ContextKeys.KEY_CHALLENGES,
            ],
            "extraction_instructions": f"""
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

            {_IMPORTANT_NO_QUESTIONS} Determine requirements based on your analysis.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": ClassifierStates.RECOMMEND_STRATEGY,
                    "description": "Needs identified",
                }
            ],
        },
        ClassifierStates.RECOMMEND_STRATEGY: {
            "description": "Recommend optimal reasoning strategy",
            "purpose": f"Set '{ContextKeys.RECOMMENDED_REASONING_TYPE}', '{ContextKeys.STRATEGY_JUSTIFICATION}', and '{ContextKeys.ALTERNATIVE_APPROACHES}'",
            "required_context_keys": [
                ContextKeys.RECOMMENDED_REASONING_TYPE,
                ContextKeys.STRATEGY_JUSTIFICATION,
                ContextKeys.ALTERNATIVE_APPROACHES,
            ],
            "extraction_instructions": f"""
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

            {_IMPORTANT_NO_QUESTIONS} Make recommendation based on your complete analysis.
            """,
            "response_instructions": _PRESENT_FINAL,
            "transitions": [],
        },
    },
}
