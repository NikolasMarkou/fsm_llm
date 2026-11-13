"""CREATIVE REASONING FSM - Novel solution generation through creative thinking.

Part of the reasoning FSM definitions (see `reasoning_modes/__init__.py`).
"""

from fsm_llm.reasoning.constants import (
    ContextKeys,
)

from ._phrases import _IMPORTANT_NO_QUESTIONS, _PRESENT_FINAL, _PRESENT_FINDINGS

creative_fsm = {
    "name": "creative_reasoning",
    "description": "Generate novel and innovative solutions through divergent and convergent creative thinking processes",
    "initial_state": "explore_perspectives",
    "persona": "You are an innovative creative thinker who generates novel solutions by seeing problems from fresh perspectives and making unexpected connections.",
    "states": {
        "explore_perspectives": {
            "description": "Explore the problem from multiple creative perspectives",
            "purpose": f"Generate '{ContextKeys.PERSPECTIVES}' and '{ContextKeys.REFRAMINGS}' of the problem",
            "required_context_keys": [ContextKeys.PERSPECTIVES, ContextKeys.REFRAMINGS],
            "extraction_instructions": f"""
            Systematically explore the problem through different creative lenses:

            - How would different people approach this (child, artist, engineer, scientist, entrepreneur)?
            - What if we flipped key assumptions or removed major constraints?
            - How is this problem similar to or different from challenges in completely different domains?
            - What would this look like from the opposite or inverse perspective?
            - What metaphors, analogies, or artistic representations reveal new angles?

            Generate multiple fresh ways of viewing and framing the problem to unlock creative potential.

            {_IMPORTANT_NO_QUESTIONS} Generate diverse perspectives based on the problem as stated.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {
                    "target_state": "generate_ideas",
                    "description": "Multiple perspectives explored",
                }
            ],
        },
        "generate_ideas": {
            "description": "Brainstorm creative and unconventional ideas without judgment",
            "purpose": f"Create '{ContextKeys.CREATIVE_IDEAS}' and '{ContextKeys.UNCONVENTIONAL_APPROACHES}'",
            "required_context_keys": [
                ContextKeys.CREATIVE_IDEAS,
                ContextKeys.UNCONVENTIONAL_APPROACHES,
            ],
            "extraction_instructions": f"""
            Generate creative ideas through divergent thinking:

            - What wild, impossible, or seemingly silly ideas come to mind?
            - What if we completely removed key constraints or limitations?
            - How do completely different fields or domains solve similar challenges?
            - What would an ideal, unlimited-resource solution look like?
            - What approaches break conventional thinking or challenge standard methods?

            Focus on quantity and novelty over immediate practicality. Suspend judgment and let creativity flow freely.

            {_IMPORTANT_NO_QUESTIONS} Generate ideas freely based on your thorough perspective exploration.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {"target_state": "combine_concepts", "description": "Ideas generated"}
            ],
        },
        "combine_concepts": {
            "description": "Combine and synthesize ideas in novel ways",
            "purpose": f"Create '{ContextKeys.COMBINATIONS}' and develop '{ContextKeys.NOVEL_SOLUTIONS}'",
            "required_context_keys": [
                ContextKeys.COMBINATIONS,
                ContextKeys.NOVEL_SOLUTIONS,
            ],
            "extraction_instructions": f"""
            Systematically combine ideas in unexpected and innovative ways:

            - What happens when we merge different approaches or solutions?
            - How can we combine the best elements of multiple ideas into something new?
            - What entirely new solutions emerge from mixing seemingly unrelated concepts?
            - How can we build bridges between ideas from different domains or perspectives?
            - What hybrid approaches might leverage multiple creative insights?

            Create truly novel solutions that integrate multiple creative elements in unexpected ways.

            {_IMPORTANT_NO_QUESTIONS} Combine the ideas you've generated in innovative ways.
            """,
            "response_instructions": _PRESENT_FINDINGS,
            "transitions": [
                {"target_state": "evaluate_novelty", "description": "Concepts combined"}
            ],
        },
        "evaluate_novelty": {
            "description": "Evaluate creative solutions for novelty, feasibility, and impact",
            "purpose": f"Select '{ContextKeys.BEST_CREATIVE_SOLUTION}' and rate '{ContextKeys.INNOVATION_RATING}'",
            "required_context_keys": [
                ContextKeys.BEST_CREATIVE_SOLUTION,
                ContextKeys.INNOVATION_RATING,
            ],
            "extraction_instructions": f"""
            Critically evaluate your creative solutions using convergent thinking:

            - Which solutions are most genuinely novel and original?
            - Which best balance creativity with potential feasibility?
            - What makes each solution truly innovative or groundbreaking?
            - Which has the greatest potential for positive impact or effectiveness?
            - How would you rate the overall creativity and innovation level (1-10 scale)?

            Select the most promising creative solution and provide a detailed innovation assessment.

            {_IMPORTANT_NO_QUESTIONS} Evaluate and select based on the creative solutions you've systematically developed.
            """,
            "response_instructions": _PRESENT_FINAL,
            "transitions": [],
        },
    },
}
