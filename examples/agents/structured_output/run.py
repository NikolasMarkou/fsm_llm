"""
Structured Output Example -- Pydantic-Validated Agent Output
============================================================

Demonstrates AgentConfig.output_schema to enforce that the agent
produces JSON matching a Pydantic model, parsed automatically
into AgentResult.structured_output.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/structured_output/run.py
"""

import os

from pydantic import BaseModel, Field

from fsm_llm_agents import AgentConfig, ReactAgent, ToolRegistry, tool


class MovieRecommendation(BaseModel):
    """Structured output schema for movie recommendations."""

    title: str = Field(description="Movie title")
    year: int = Field(description="Release year")
    genre: str = Field(description="Primary genre")
    reason: str = Field(description="Why this movie is recommended")
    rating: float = Field(description="Expected rating out of 10", ge=0, le=10)


@tool
def search_movies(query: str) -> str:
    """Search for movie information."""
    q = query.lower()
    movies = {
        "sci-fi": '{"title": "Blade Runner 2049", "year": 2017, "genre": "Sci-Fi", "reason": "Stunning visuals and deep themes about humanity", "rating": 8.5}',
        "action": '{"title": "Mad Max: Fury Road", "year": 2015, "genre": "Action", "reason": "Non-stop thrilling action with great world-building", "rating": 8.8}',
        "drama": '{"title": "Parasite", "year": 2019, "genre": "Drama", "reason": "Masterful social commentary with unexpected twists", "rating": 9.0}',
        "comedy": '{"title": "The Grand Budapest Hotel", "year": 2014, "genre": "Comedy", "reason": "Witty humor with beautiful cinematography", "rating": 8.1}',
        "horror": '{"title": "Get Out", "year": 2017, "genre": "Horror", "reason": "Intelligent social horror that redefines the genre", "rating": 8.2}',
    }
    for key, value in movies.items():
        if key in q:
            return value
    return '{"title": "Inception", "year": 2010, "genre": "Sci-Fi", "reason": "Mind-bending thriller with layered storytelling", "rating": 8.7}'


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    registry = ToolRegistry()
    registry.register(search_movies._tool_definition)

    config = AgentConfig(
        model=model,
        max_iterations=6,
        temperature=0.7,
        output_schema=MovieRecommendation,
    )
    agent = ReactAgent(tools=registry, config=config)

    print("=" * 60)
    print("Structured Output Agent -- Pydantic Validation")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Output schema: {MovieRecommendation.__name__}")
    print()

    task = "Recommend a great sci-fi movie. Return your answer as JSON with title, year, genre, reason, and rating fields."

    print(f"Task: {task}")
    print("-" * 40)

    try:
        result = agent.run(task)
        print(f"\nRaw answer: {result.answer}")
        print(f"Tools used: {result.tools_used}")
        print(f"Iterations: {result.iterations_used}")

        if result.structured_output:
            print("\nStructured output (validated):")
            print(f"  Title:  {result.structured_output.title}")
            print(f"  Year:   {result.structured_output.year}")
            print(f"  Genre:  {result.structured_output.genre}")
            print(f"  Reason: {result.structured_output.reason}")
            print(f"  Rating: {result.structured_output.rating}")
        else:
            print("\nStructured output: None (validation failed)")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
