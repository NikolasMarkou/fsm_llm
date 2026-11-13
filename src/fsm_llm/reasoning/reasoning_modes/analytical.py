"""ANALYTICAL REASONING FSM - Systematic decomposition and analysis.

Part of the reasoning FSM definitions (see `reasoning_modes/__init__.py`).
"""

from fsm_llm.reasoning.constants import (
    ContextKeys,
)

from ._phrases import _IMPORTANT_NO_QUESTIONS, _PRESENT_FINAL, _PRESENT_FINDINGS

analytical_fsm = {
    "name": "analytical_reasoning",
    "description": "Analytical reasoning through systematic decomposition and component analysis",
    "initial_state": "decompose",
    "persona": "You are a methodical analytical thinker who breaks down complex problems systematically.",
    "states": {
        "decompose": {
            "description": "Break down the problem into component parts",
            "purpose": f"Identify '{ContextKeys.COMPONENTS}', '{ContextKeys.ATTRIBUTES}', and '{ContextKeys.RELATIONSHIPS}'",
            "required_context_keys": [
                ContextKeys.COMPONENTS,
                ContextKeys.ATTRIBUTES,
                ContextKeys.RELATIONSHIPS,
            ],
            "extraction_instructions": f"""
            Systematically decompose the problem:

            - Break the problem into smaller, manageable components
            - Identify key attributes of each component
            - Map relationships and dependencies between components
            - Organize components hierarchically if applicable
            - Note any emergent properties from component interactions

            Focus on creating a clear structural understanding of the problem space.

            {_IMPORTANT_NO_QUESTIONS} Decompose based on the problem as presented.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "analyze_components",
                    "description": "Decomposition complete",
                }
            ],
        },
        "analyze_components": {
            "description": "Analyze each component in detail",
            "purpose": f"Create '{ContextKeys.COMPONENT_ANALYSIS}' and identify '{ContextKeys.DATA_REQUIREMENTS}'",
            "required_context_keys": [
                ContextKeys.COMPONENT_ANALYSIS,
                ContextKeys.DATA_REQUIREMENTS,
            ],
            "extraction_instructions": f"""
            Conduct detailed analysis of each component:

            - Examine the function and role of each component
            - Identify the properties and characteristics
            - Determine how each component contributes to the whole
            - Assess the importance and priority of each component
            - Note what additional data might be needed for complete analysis

            Create comprehensive component analysis with clear insights.

            {_IMPORTANT_NO_QUESTIONS} Analyze components based on your decomposition.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "identify_patterns",
                    "description": "Component analysis complete",
                }
            ],
        },
        "identify_patterns": {
            "description": "Find patterns and dependencies between components",
            "purpose": f"Identify '{ContextKeys.PATTERNS}', '{ContextKeys.CAUSAL_LINKS}', and '{ContextKeys.DEPENDENCIES}'",
            "required_context_keys": [
                ContextKeys.PATTERNS,
                ContextKeys.CAUSAL_LINKS,
                ContextKeys.DEPENDENCIES,
            ],
            "extraction_instructions": f"""
            Identify patterns and relationships:

            - Look for recurring patterns across components
            - Establish causal relationships (cause-effect links)
            - Map dependencies (what depends on what)
            - Identify feedback loops or circular dependencies
            - Note any systematic behaviors or regularities

            Focus on understanding the dynamic interactions between components.

            {_IMPORTANT_NO_QUESTIONS} Identify patterns from your component analysis.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "integrate_findings",
                    "description": "Patterns identified",
                }
            ],
        },
        "integrate_findings": {
            "description": "Synthesize understanding from all analytical work",
            "purpose": f"Create '{ContextKeys.INTEGRATED_ANALYSIS}' and '{ContextKeys.KEY_INSIGHTS}'",
            "required_context_keys": [
                ContextKeys.INTEGRATED_ANALYSIS,
                ContextKeys.KEY_INSIGHTS,
            ],
            "extraction_instructions": f"""
            Integrate all analytical findings:

            - Synthesize component analysis, patterns, and relationships
            - Create a comprehensive understanding of the problem
            - Highlight the most important insights and discoveries
            - Identify implications and potential solutions
            - Summarize the analytical understanding clearly

            Provide integrated analysis that brings together all analytical work into coherent understanding.

            {_IMPORTANT_NO_QUESTIONS} Integrate based on your complete analytical process.
            """,
            "response_instructions": _PRESENT_FINAL,
            "transitions": [],
        },
    },
}
