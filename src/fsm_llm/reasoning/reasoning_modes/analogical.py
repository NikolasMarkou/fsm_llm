"""ANALOGICAL REASONING FSM - Transfer insights through analogical thinking.

Part of the reasoning FSM definitions (see `reasoning_modes/__init__.py`).
"""

from fsm_llm.reasoning.constants import (
    ContextKeys,
)

from ._phrases import _IMPORTANT_NO_QUESTIONS, _PRESENT_FINAL, _PRESENT_FINDINGS

analogical_fsm = {
    "name": "analogical_reasoning",
    "description": "Transfer insights and solutions via systematic analogical reasoning and pattern matching across domains",
    "initial_state": "define_target_problem",
    "persona": "You are an expert at finding meaningful connections and analogies. You help solve problems by identifying similar situations and transferring insights across domains.",
    "states": {
        "define_target_problem": {
            "description": "Clearly define and characterize the target problem",
            "purpose": f"Analyze the problem to identify '{ContextKeys.TARGET_PROBLEM_DESCRIPTION}' and '{ContextKeys.KEY_FEATURES_OF_TARGET}'",
            "required_context_keys": [
                ContextKeys.TARGET_PROBLEM_DESCRIPTION,
                ContextKeys.KEY_FEATURES_OF_TARGET,
            ],
            "extraction_instructions": f"""
            Systematically define and characterize the target problem:

            - What is the core challenge, question, or problem we're trying to solve?
            - What are the key features, constraints, context, and important characteristics?
            - What kind of solution, understanding, or outcome are we seeking?
            - What essential structural or functional characteristics should any good analogy match?
            - What are the most important aspects that an analogical source should share?

            Create a clear, comprehensive characterization of the target problem for analogical matching.

            {_IMPORTANT_NO_QUESTIONS} Work only with the information already provided in the problem statement and context.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "find_source_analogs",
                    "description": "Target problem clearly defined",
                }
            ],
        },
        "find_source_analogs": {
            "description": "Identify potential analogous situations across various domains",
            "purpose": f"Find '{ContextKeys.POTENTIAL_ANALOGS}' with '{ContextKeys.RATIONALE_FOR_CHOICE}' and '{ContextKeys.SIMILARITY_CRITERIA_USED}'",
            "required_context_keys": [
                ContextKeys.POTENTIAL_ANALOGS,
                ContextKeys.RATIONALE_FOR_CHOICE,
                ContextKeys.SIMILARITY_CRITERIA_USED,
            ],
            "extraction_instructions": f"""
            Systematically search for analogous situations across different domains:

            - What situations, problems, or systems share important structural similarities?
            - What comparable processes, mechanisms, or functional relationships exist in other fields?
            - What parallel challenges exist in different domains (nature, technology, history, etc.)?
            - What historical precedents, case studies, or examples show similar patterns?
            - What systems or situations exhibit comparable dynamics or relationships?

            Provide 2-4 potential analogs with clear rationale for why each might offer valuable insights.

            {_IMPORTANT_NO_QUESTIONS} Search for analogs based on your target problem characterization.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "map_correspondences",
                    "description": "Source analogs identified",
                }
            ],
        },
        "map_correspondences": {
            "description": "Create systematic mapping between source analog and target problem",
            "purpose": f"Select best analog and create detailed mapping with '{ContextKeys.SELECTED_ANALOG}', '{ContextKeys.ANALOGICAL_MAPPING}', '{ContextKeys.IDENTIFIED_SIMILARITIES}', '{ContextKeys.IDENTIFIED_DIFFERENCES}'",
            "required_context_keys": [
                ContextKeys.SELECTED_ANALOG,
                ContextKeys.ANALOGICAL_MAPPING,
                ContextKeys.IDENTIFIED_SIMILARITIES,
                ContextKeys.IDENTIFIED_DIFFERENCES,
            ],
            "extraction_instructions": f"""
            Create systematic correspondences between the most promising analog and target:

            - Choose the analog with the strongest structural and functional similarities
            - Map specific elements from source to target (A corresponds to X, B relates to Y, etc.)
            - Identify the strongest similarities that support and validate the analogy
            - Note important differences or limitations that constrain the analogical inference
            - Create explicit, systematic mapping of relationships and correspondences

            Be precise about what maps to what and provide clear justification for the correspondences.

            {_IMPORTANT_NO_QUESTIONS} Create the mapping based on your analysis of potential analogs.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "transfer_insights",
                    "description": "Correspondences systematically mapped",
                }
            ],
        },
        "transfer_insights": {
            "description": "Transfer knowledge and solutions from analog to target domain",
            "purpose": f"Generate '{ContextKeys.TRANSFERRED_INSIGHTS_OR_SOLUTIONS}' and '{ContextKeys.POTENTIAL_INFERENCES}'",
            "required_context_keys": [
                ContextKeys.TRANSFERRED_INSIGHTS_OR_SOLUTIONS,
                ContextKeys.POTENTIAL_INFERENCES,
            ],
            "extraction_instructions": f"""
            Systematically transfer insights using the analogical mapping:

            - What solutions, strategies, or approaches worked effectively in the source domain?
            - What principles, patterns, or mechanisms can be transferred to the target?
            - What new understanding or perspective does the analogy provide about the target problem?
            - What predictions, inferences, or hypotheses can we generate through analogical reasoning?
            - How do successful strategies in the source domain suggest approaches for the target?

            Be specific and explicit about how insights from the analog apply to and illuminate the target problem.

            {_IMPORTANT_NO_QUESTIONS} Generate insights based on the systematic analogical mapping you've created.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "evaluate_analogy_fit",
                    "description": "Insights transferred",
                }
            ],
        },
        "evaluate_analogy_fit": {
            "description": "Critically evaluate the analogy's validity and practical utility",
            "purpose": f"Assess analogy with '{ContextKeys.ANALOGY_STRENGTHS}', '{ContextKeys.ANALOGY_WEAKNESSES_OR_LIMITATIONS}', '{ContextKeys.ADAPTED_SOLUTION_OR_UNDERSTANDING}', '{ContextKeys.ANALOGY_CONFIDENCE_RATING}'",
            "required_context_keys": [
                ContextKeys.ANALOGY_STRENGTHS,
                ContextKeys.ANALOGY_WEAKNESSES_OR_LIMITATIONS,
                ContextKeys.ADAPTED_SOLUTION_OR_UNDERSTANDING,
                ContextKeys.ANALOGY_CONFIDENCE_RATING,
            ],
            "extraction_instructions": f"""
            Critically and systematically evaluate the analogical reasoning:

            - What are the strongest, most compelling aspects of this analogy?
            - Where does the analogy break down, mislead, or have significant limitations?
            - How should the transferred solution or insight be adapted for the target context?
            - What level of confidence is warranted in this analogical reasoning (1-10 scale)?
            - What would strengthen or weaken confidence in the analogical inference?

            Provide balanced evaluation with adapted solution and justified confidence rating.

            {_IMPORTANT_NO_QUESTIONS} Complete your evaluation with the information developed through your systematic analogical process.
            """,
            "response_instructions": _PRESENT_FINAL,
            "transitions": [],
        },
    },
}
