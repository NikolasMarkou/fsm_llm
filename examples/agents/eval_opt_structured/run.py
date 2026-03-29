"""
Evaluator-Optimizer with Structured Output
============================================

Demonstrates EvaluatorOptimizer combined with Pydantic structured output
validation. The agent generates a structured JSON recipe, the evaluator
validates it against quality criteria, and the optimizer refines until
the schema is fully satisfied.

This pattern is useful for any task requiring both correctness (passes
evaluation) and structure (matches a schema).

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/eval_opt_structured/run.py
"""

import os

from pydantic import BaseModel, Field

from fsm_llm_agents import AgentConfig, EvaluationResult, EvaluatorOptimizerAgent


class RecipeOutput(BaseModel):
    """Structured recipe output schema."""

    name: str = Field(description="Recipe name")
    cuisine: str = Field(description="Cuisine type (e.g., Italian, Japanese)")
    prep_time_minutes: int = Field(description="Preparation time in minutes", ge=1)
    cook_time_minutes: int = Field(description="Cooking time in minutes", ge=0)
    servings: int = Field(description="Number of servings", ge=1)
    difficulty: str = Field(description="Easy, Medium, or Hard")
    ingredients: list[str] = Field(description="List of ingredients with quantities")
    steps: list[str] = Field(description="Step-by-step cooking instructions")
    nutritional_notes: str = Field(description="Brief nutritional information")


def evaluate_recipe(output: str, context: dict) -> EvaluationResult:
    """Evaluate recipe quality against comprehensive criteria."""
    feedback_parts = []
    checks = {}

    # Check basic structure (3 lines minimum)
    lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
    checks["has_content"] = len(lines) >= 5
    if not checks["has_content"]:
        feedback_parts.append("Output too short — need a complete recipe.")

    lower = output.lower()

    # Check for ingredients
    has_ingredients = any(
        kw in lower
        for kw in ["ingredient", "cup", "tablespoon", "teaspoon", "gram", "oz"]
    )
    checks["has_ingredients"] = has_ingredients
    if not has_ingredients:
        feedback_parts.append("Missing ingredient list with quantities.")

    # Check for cooking steps
    has_steps = any(
        kw in lower
        for kw in ["step", "heat", "cook", "mix", "stir", "bake", "simmer", "chop"]
    )
    checks["has_steps"] = has_steps
    if not has_steps:
        feedback_parts.append("Missing cooking instructions/steps.")

    # Check for timing info
    has_timing = any(
        kw in lower for kw in ["minute", "hour", "second", "prep", "cook time"]
    )
    checks["has_timing"] = has_timing
    if not has_timing:
        feedback_parts.append("Include preparation and cooking times.")

    # Check for serving info
    has_servings = any(kw in lower for kw in ["serving", "serves", "portion"])
    checks["has_servings"] = has_servings
    if not has_servings:
        feedback_parts.append("Specify number of servings.")

    # Check for JSON structure (since we want structured output)
    has_json = "{" in output and "}" in output
    checks["has_json_structure"] = has_json
    if not has_json:
        feedback_parts.append(
            "Output should be valid JSON matching the recipe schema with "
            "fields: name, cuisine, prep_time_minutes, cook_time_minutes, "
            "servings, difficulty, ingredients, steps, nutritional_notes."
        )

    passed = all(checks.values())
    score = sum(checks.values()) / len(checks) if checks else 0.0
    criteria_met = [name for name, ok in checks.items() if ok]

    feedback = "Excellent recipe!" if passed else " ".join(feedback_parts)

    return EvaluationResult(
        passed=passed,
        score=score,
        feedback=feedback,
        criteria_met=criteria_met,
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    config = AgentConfig(
        model=model,
        max_iterations=10,
        temperature=0.5,
        output_schema=RecipeOutput,
    )

    agent = EvaluatorOptimizerAgent(
        evaluation_fn=evaluate_recipe,
        config=config,
        max_refinements=2,
    )

    task = (
        "Create a detailed recipe for homemade pasta carbonara. "
        "Return your answer as JSON with these exact fields: "
        "name, cuisine, prep_time_minutes, cook_time_minutes, servings, "
        "difficulty (Easy/Medium/Hard), ingredients (list of strings with quantities), "
        "steps (list of instruction strings), and nutritional_notes. "
        "Be specific with quantities and timing."
    )

    print("=" * 60)
    print("Evaluator-Optimizer with Structured Output")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Output schema: {RecipeOutput.__name__}")
    print("Max refinements: 2")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nRaw answer:\n{result.answer[:500]}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")

        refinement_count = result.final_context.get("refinement_count", 0)
        eval_passed = result.final_context.get("evaluation_passed", False)
        eval_score = result.final_context.get("evaluation_score", 0)

        print(f"Refinements: {refinement_count}")
        print(f"Evaluation passed: {eval_passed}")
        print(f"Final score: {eval_score}")

        if result.structured_output:
            recipe = result.structured_output
            print("\nStructured Output (validated):")
            print(f"  Name: {recipe.name}")
            print(f"  Cuisine: {recipe.cuisine}")
            print(
                f"  Prep: {recipe.prep_time_minutes}min, Cook: {recipe.cook_time_minutes}min"
            )
            print(f"  Servings: {recipe.servings}")
            print(f"  Difficulty: {recipe.difficulty}")
            print(f"  Ingredients ({len(recipe.ingredients)}):")
            for ing in recipe.ingredients[:5]:
                print(f"    - {ing}")
            if len(recipe.ingredients) > 5:
                print(f"    ... and {len(recipe.ingredients) - 5} more")
            print(f"  Steps ({len(recipe.steps)}):")
            for i, step in enumerate(recipe.steps[:3], 1):
                print(f"    {i}. {step[:80]}")
            if len(recipe.steps) > 3:
                print(f"    ... and {len(recipe.steps) - 3} more steps")
        else:
            print("\nStructured output: None (validation failed)")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
