"""
@tool Decorator Example -- Type-Hint Driven Tool Registration
=============================================================

Demonstrates the @tool decorator with automatic JSON schema
inference from type hints, including Annotated descriptions.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/tool_decorator/run.py
"""

import os
from typing import Annotated

from fsm_llm_agents import AgentConfig, ReactAgent, ToolRegistry, tool


@tool
def convert_temperature(
    value: Annotated[float, "Temperature value to convert"],
    from_unit: Annotated[str, "Source unit: celsius, fahrenheit, or kelvin"],
    to_unit: Annotated[str, "Target unit: celsius, fahrenheit, or kelvin"],
) -> str:
    """Convert temperature between Celsius, Fahrenheit, and Kelvin."""
    # Normalize to Celsius first
    if from_unit.lower().startswith("f"):
        celsius = (value - 32) * 5 / 9
    elif from_unit.lower().startswith("k"):
        celsius = value - 273.15
    else:
        celsius = value

    # Convert from Celsius to target
    if to_unit.lower().startswith("f"):
        result = celsius * 9 / 5 + 32
        unit_label = "F"
    elif to_unit.lower().startswith("k"):
        result = celsius + 273.15
        unit_label = "K"
    else:
        result = celsius
        unit_label = "C"

    return f"{value}{from_unit[0].upper()} = {result:.2f}{unit_label}"


@tool(description="Calculate the area of a geometric shape")
def calculate_area(
    shape: Annotated[str, "Shape: circle, rectangle, or triangle"],
    dimensions: Annotated[str, "Comma-separated dimensions (e.g., '5,3' for rectangle width,height)"],
) -> str:
    """Calculate area of basic geometric shapes."""
    import math

    dims = [float(d.strip()) for d in dimensions.split(",")]

    if shape.lower() == "circle":
        area = math.pi * dims[0] ** 2
        return f"Circle (radius={dims[0]}): area = {area:.2f} sq units"
    elif shape.lower() == "rectangle":
        area = dims[0] * (dims[1] if len(dims) > 1 else dims[0])
        return f"Rectangle ({dims[0]}x{dims[1] if len(dims) > 1 else dims[0]}): area = {area:.2f} sq units"
    elif shape.lower() == "triangle":
        if len(dims) >= 2:
            area = 0.5 * dims[0] * dims[1]
            return f"Triangle (base={dims[0]}, height={dims[1]}): area = {area:.2f} sq units"
    return f"Unknown shape: {shape}"


@tool
def unit_convert(
    value: float,
    from_unit: str,
    to_unit: str,
) -> str:
    """Convert between common units of measurement."""
    conversions = {
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
        ("m", "ft"): 3.28084,
        ("ft", "m"): 0.3048,
        ("l", "gal"): 0.264172,
        ("gal", "l"): 3.78541,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = value * conversions[key]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return f"Conversion not supported: {from_unit} -> {to_unit}"


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    # Register @tool-decorated functions
    registry = ToolRegistry()
    registry.register(convert_temperature._tool_definition)
    registry.register(calculate_area._tool_definition)
    registry.register(unit_convert._tool_definition)

    # Show inferred schemas
    print("=" * 60)
    print("@tool Decorator Example -- Auto-Schema Inference")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools registered: {', '.join(registry.tool_names)}")
    print()

    for t in registry.list_tools():
        print(f"  {t.name}: {t.description}")
        if t.parameter_schema:
            props = t.parameter_schema.get("properties", {})
            for pname, pdef in props.items():
                print(f"    - {pname}: {pdef.get('type', '?')} -- {pdef.get('description', '')}")
    print()

    config = AgentConfig(model=model, max_iterations=8, temperature=0.7)
    agent = ReactAgent(tools=registry, config=config)

    print("Type a question or 'quit' to exit.\n")

    while True:
        task = input("Task: ").strip()
        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not task:
            continue

        print(f"\nWorking on: {task}")
        print("-" * 40)
        try:
            result = agent.run(task)
            print(f"Answer: {result.answer}")
            print(f"Tools used: {result.tools_used}")
            print(f"Iterations: {result.iterations_used}")
        except Exception as e:
            print(f"Error: {e}")
        print()


if __name__ == "__main__":
    main()
