"""
Agent-as-Tool Example -- Nested Agent Composition
==================================================

Demonstrates register_agent() to wrap one agent as a tool
callable by another agent, enabling hierarchical agent composition.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/agent_as_tool/run.py
"""

import os

from fsm_llm_agents import AgentConfig, ReactAgent, ToolRegistry, tool
from fsm_llm_agents.tools import register_agent


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    weather_data = {
        "london": "London: 12C, cloudy, 80% humidity, wind 15 km/h",
        "paris": "Paris: 15C, sunny, 60% humidity, wind 10 km/h",
        "tokyo": "Tokyo: 22C, partly cloudy, 70% humidity, wind 8 km/h",
        "new york": "New York: 18C, clear, 55% humidity, wind 12 km/h",
    }
    return weather_data.get(city.lower(), f"{city}: 20C, fair weather, no specific data")


@tool
def get_events(city: str) -> str:
    """Get upcoming events in a city."""
    events = {
        "london": "London events: West End shows, British Museum exhibitions, Premier League matches",
        "paris": "Paris events: Louvre special exhibit, Fashion Week preparations, Seine river cruises",
        "tokyo": "Tokyo events: Cherry blossom festival, Anime convention, Sumo tournament",
        "new york": "New York events: Broadway shows, Central Park concerts, Met Gala preparations",
    }
    return events.get(city.lower(), f"{city}: local festivals and cultural events")


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    # Create a specialist "research agent" with weather/events tools
    research_registry = ToolRegistry()
    research_registry.register(get_weather._tool_definition)
    research_registry.register(get_events._tool_definition)

    research_config = AgentConfig(model=model, max_iterations=6, temperature=0.7)
    research_agent = ReactAgent(tools=research_registry, config=research_config)

    # Create the main agent's registry and register the research agent as a tool
    main_registry = ToolRegistry()
    register_agent(
        main_registry,
        research_agent,
        name="city_research",
        description="Research a city -- gets weather, events, and local info. Pass the city name as the task.",
    )

    main_config = AgentConfig(model=model, max_iterations=8, temperature=0.7)
    main_agent = ReactAgent(tools=main_registry, config=main_config)

    task = "I'm planning a trip. Research Tokyo and tell me what the weather is like and what events are happening."

    print("=" * 60)
    print("Agent-as-Tool -- Nested Agent Composition")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Main agent tools: {', '.join(main_registry.tool_names)}")
    print(f"Research agent tools: {', '.join(research_registry.tool_names)}")
    print(f"Task: {task}\n")

    try:
        result = main_agent.run(task)
        print(f"\nAnswer: {result.answer}")
        print(f"Success: {result.success}")
        print(f"Tools used: {result.tools_used}")
        print(f"Iterations: {result.iterations_used}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
